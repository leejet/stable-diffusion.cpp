#include "api/generation_program.h"

#include <algorithm>
#include <cmath>

namespace sd::api {
    static std::string expression_list(const std::vector<std::string>& values) {
        std::string result = "[";
        for (size_t i = 0; i < values.size(); ++i)
            result += (i ? ", " : "") + values[i];
        return result + "]";
    }

    void prepare_h3_references(GenerationTemplate& request, const std::string& first, const std::string& last) {
        const auto images           = bound_tensors(request.options, "ref_images");
        const auto videos           = bound_videos(request.options, "ref_videos");
        const auto audios           = bound_audios(request.options, "ref_audios");
        const bool references       = !images.empty() || !videos.empty() || !audios.empty();
        const bool audio_references = !audios.empty() || std::any_of(videos.begin(), videos.end(), [](const sd_video_t& video) { return video.audio.data != nullptr; });
        if (audio_references && !request.flags.at("decode_audio").get<bool>())
            throw dgl::Error(SD_STATUS_LOAD_FAILED, "MiniMax-H3 audio references require audio VAE weights");
        if (references && (!first.empty() || !last.empty()))
            throw dgl::Error(SD_STATUS_INVALID_ARGUMENT, "MiniMax-H3 references and start/end keyframes cannot be combined");
        if (!references && first.empty() && last.empty())
            return;
        auto& prepare = request.fields["references"];
        std::vector<std::string> visuals, sounds, presentation, blocks, keys;
        const auto encode_visual = [&](const std::string& pixels, const nlohmann::json& options = nlohmann::json::object()) {
            const auto name = "h3_visual_" + std::to_string(visuals.size());
            nlohmann::json parameters{{"name", name}, {"pixels", pixels}, {"encode_audio", false}, {"resize_video", false}, {"resize_image", false}};
            parameters.update(options);
            prepare += request.render("video_reference", parameters);
            visuals.push_back(name);
        };
        const auto encode_sound = [&](const sd_audio_t& audio) {
            const auto name = "h3_sound_" + std::to_string(sounds.size());
            bind_tensor(request.program, name + "_samples", audio.data);
            const auto media = "{samples = " + name + "_samples, sample_rate = " + std::to_string(audio.sample_rate) + "}";
            prepare += request.render("video_reference", {{"encode_audio", true}, {"name", name}, {"sample_rate", audio.sample_rate}});
            sounds.push_back(name);
            presentation.push_back("{audio = " + media + "}");
        };
        if (!references) {
            std::vector<std::string> key_images;
            for (const auto& [pixels, index] : std::vector<std::pair<std::string, std::string>>{{first, "0"}, {last, "frames - 1"}}) {
                if (pixels.empty())
                    continue;
                encode_visual(pixels);
                key_images.push_back(pixels + "[:, :, 0]");
                keys.push_back(index);
            }
            request.fields["prompt_options"] += ", media = " + expression_list(key_images);
            request.fields["negative_prompt_options"] += ", media = " + expression_list(key_images);
        }
        for (size_t i = 0; i < images.size(); ++i) {
            const auto name = "h3_image_" + std::to_string(i);
            bind_tensor(request.program, name, images[i]);
            const int source_width = int(sd_tensor_dim(images[i], 1)), source_height = int(sd_tensor_dim(images[i], 2));
            const double scale = std::min(1.0, std::sqrt(double(request.values.at("width").get<int>()) * request.values.at("height").get<int>() / (double(source_width) * source_height)));
            const int width = std::max(32, int(std::round(source_width * scale / 32)) * 32), height = std::max(32, int(std::round(source_height * scale / 32)) * 32);
            blocks.push_back(literal({{"kind", "image"}, {"video", visuals.size()}}));
            encode_visual(name + "_pixels", {{"resize_image", true}, {"source", name}, {"width", width}, {"height", height}});
            presentation.push_back("{image = " + name + "_pixels}");
        }
        for (size_t i = 0; i < videos.size(); ++i) {
            const auto& video = videos[i];
            const auto name   = "h3_video_" + std::to_string(i);
            bind_tensor(request.program, name, video.data);
            const int count = int(sd_tensor_dim(video.data, 3));
            int frames      = std::min(request.values.at("frames").get<int>(), int(std::lround(double(count) * 24 / video.fps)));
            if (frames < 5)
                throw dgl::Error(SD_STATUS_INVALID_ARGUMENT, "MiniMax-H3 reference video needs at least 5 frames at 24 fps");
            frames = (frames - 5) / 17 * 17 + 5;
            std::vector<int> indices;
            for (int frame = 0; frame < frames; ++frame)
                indices.push_back(std::min(count - 1, int(std::floor(double(frame) * video.fps / 24))));
            const int source_width = int(sd_tensor_dim(video.data, 1)), source_height = int(sd_tensor_dim(video.data, 2));
            const double ratio = double(source_width) / source_height;
            double w = ratio >= 1 ? 768 * ratio : 768, h = ratio >= 1 ? 768 : 768 / ratio;
            if (w * h > 768.0 * 1344) {
                const auto scale = std::sqrt(768.0 * 1344 / (w * h));
                w *= scale;
                h *= scale;
            }
            int width = std::max(32, int(std::round(w / 32)) * 32), height = std::max(32, int(std::round(h / 32)) * 32);
            if (source_width * source_height < width * height) {
                width  = std::max(32, int(std::round(double(source_width) / 32)) * 32);
                height = std::max(32, int(std::round(double(source_height) / 32)) * 32);
            }
            auto block = nlohmann::json{{"kind", video.audio.data ? "video_audio" : "video"}, {"video", visuals.size()}};
            if (video.audio.data) {
                block["audio"] = sounds.size();
                encode_sound(video.audio);
            }
            blocks.push_back(literal(block));
            encode_visual(name + "_pixels", {{"resize_video", true}, {"source", name}, {"indices", literal(indices)}, {"width", width}, {"height", height}});
            presentation.push_back("{video = {frames = " + name + "_pixels, fps_num = 24, fps_den = 1}, stride = 12}");
        }
        for (const auto& audio : audios) {
            blocks.push_back(literal({{"kind", "audio"}, {"audio", sounds.size()}}));
            encode_sound(audio);
        }
        std::string options = ", ref_latents = " + expression_list(visuals);
        if (!sounds.empty())
            options += ", ref_audios = " + expression_list(sounds);
        if (!blocks.empty())
            options += ", reference_blocks = " + expression_list(blocks);
        if (!keys.empty())
            options += ", keyframe_indices = " + expression_list(keys);
        request.fields["predictor_options"] += options;
        request.fields["negative_predictor_options"] += options;
        if (!presentation.empty()) {
            request.fields["prompt_options"] += ", media = " + expression_list(presentation);
            request.fields["negative_prompt_options"] += ", media = " + expression_list(presentation);
        }
    }
}
