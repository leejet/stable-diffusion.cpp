#include "media_input.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <sstream>

#include "common/media_io.h"
#include "json.hpp"

bool load_images_from_dir(const std::string& dir,
                          std::vector<SDImageOwner>& images,
                          int max_image_num) {
    const std::filesystem::path dir_path = std::filesystem::u8path(dir);
    if (!std::filesystem::exists(dir_path) || !std::filesystem::is_directory(dir_path)) {
        LOG_ERROR("'%s' is not a valid directory\n", dir.c_str());
        return false;
    }

    std::vector<std::filesystem::directory_entry> entries;
    for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
        if (entry.is_regular_file()) {
            entries.push_back(entry);
        }
    }

    std::sort(entries.begin(), entries.end(),
              [](const std::filesystem::directory_entry& a, const std::filesystem::directory_entry& b) {
                  return a.path().filename().u8string() < b.path().filename().u8string();
              });

    for (const auto& entry : entries) {
        std::string path = entry.path().u8string();
        std::string ext  = entry.path().extension().u8string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".webp") {
            LOG_VERBOSE("load image %zu from '%s'", images.size(), path.c_str());
            int width             = 0;
            int height            = 0;
            int loaded_channel    = 0;
            uint8_t* image_buffer = load_image_from_file(path.c_str(), width, height, loaded_channel, 0, 0);
            if (image_buffer == nullptr) {
                LOG_ERROR("load image from '%s' failed", path.c_str());
                return false;
            }

            images.emplace_back(SDImage{(uint32_t)width,
                                        (uint32_t)height,
                                        (uint32_t)loaded_channel,
                                        image_buffer});

            if (max_image_num > 0 && static_cast<int>(images.size()) >= max_image_num) {
                break;
            }
        }
    }
    return true;
}

bool load_video_input(const std::string& path, int fps, size_t max_frames, bool audio, VideoInput& result) {
    std::ostringstream source;
    source << "fn main() { return video::load(" << nlohmann::json(path).dump()
           << ", fps_num = " << fps << ", max_frames = " << max_frames
           << ", audio = " << (audio ? "true" : "false") << ") }";
    std::unique_ptr<sd_dgl_ctx_t, decltype(&sd_dgl_ctx_free)> context(sd_dgl_ctx_create_once(), sd_dgl_ctx_free);
    if (!context) {
        LOG_ERROR("cannot prepare video input '%s': %s", path.c_str(), sd_dgl_error());
        return false;
    }
    sd_dgl_result_t* raw_result = nullptr;
    if (sd_dgl_run(context.get(), source.str().c_str(), nullptr, 0, &raw_result) != SD_STATUS_OK) {
        LOG_ERROR("cannot read video input '%s': %s", path.c_str(), sd_dgl_error());
        return false;
    }
    std::unique_ptr<sd_dgl_result_t, decltype(&sd_dgl_result_free)> loaded(raw_result, sd_dgl_result_free);
    const auto* root    = sd_dgl_result_root(loaded.get());
    const auto* sound   = sd_dgl_value_field(root, "audio");
    int64_t sample_rate = 0;
    if (sd_dgl_value_get_int(sd_dgl_value_field(sound, "sample_rate"), &sample_rate) != SD_STATUS_OK) {
        LOG_ERROR("video input has no audio sample rate: %s", sd_dgl_error());
        return false;
    }
    VideoInput input;
    input.frames.reset(sd_tensor_retain(sd_dgl_result_tensor(loaded.get(), "video.frames")));
    input.samples.reset(sd_tensor_retain(sd_dgl_result_tensor(loaded.get(), "audio.samples")));
    input.sample_rate = uint32_t(sample_rate);
    input.fps         = fps;
    if (!input.frames || !input.samples) {
        LOG_ERROR("video input is missing frame or audio tensors: %s", sd_dgl_error());
        return false;
    }
    result = std::move(input);
    return true;
}

bool load_reference_video_inputs(const std::vector<std::string>& paths, const std::vector<std::string>& audio_paths, std::vector<VideoInput>& result) {
    std::vector<VideoInput> inputs;
    inputs.reserve(paths.size());
    for (size_t i = 0; i < paths.size(); ++i) {
        VideoInput input;
        const bool external_audio = i < audio_paths.size();
        if (std::filesystem::is_directory(std::filesystem::u8path(paths[i]))) {
            std::vector<SDImageOwner> frames;
            if (!load_images_from_dir(paths[i], frames) || frames.empty()) {
                LOG_ERROR("load reference video frames from '%s' failed", paths[i].c_str());
                return false;
            }
            std::vector<SDImage> views;
            views.reserve(frames.size());
            for (const auto& frame : frames)
                views.push_back(frame.get());
            input.frames = frames_tensor(views.data(), views.size());
            if (!input.frames)
                return false;
        } else if (!load_video_input(paths[i], 24, 0, !external_audio, input)) {
            return false;
        }
        if (external_audio) {
            std::vector<float> samples;
            uint32_t rate = 0, channels = 0;
            if (!load_wav_from_file(audio_paths[i], samples, rate, channels)) {
                LOG_ERROR("load WAV audio from '%s' failed", audio_paths[i].c_str());
                return false;
            }
            const int64_t shape[] = {channels, int64_t(samples.size() / channels)};
            input.samples.reset(sd_tensor_create(SD_TENSOR_F32, 2, shape, samples.data(), samples.size() * sizeof(float)));
            input.sample_rate = rate;
            if (!input.samples)
                return false;
        }
        inputs.push_back(std::move(input));
    }
    result = std::move(inputs);
    return true;
}

bool set_video_inputs(sd_opt_t* options, const std::vector<VideoInput>& references, const sd_tensor_t* control) {
    std::vector<sd_video_t> views;
    views.reserve(references.size());
    for (const auto& video : references)
        views.push_back(video.view());
    if ((!views.empty() && sd_opt_set_videos(options, "ref_videos", views.data(), views.size()) != SD_STATUS_OK) ||
        (control && sd_opt_set_tensor(options, "control_frames", control) != SD_STATUS_OK)) {
        LOG_ERROR("cannot set video inputs: %s", sd_get_last_error());
        return false;
    }
    return true;
}
