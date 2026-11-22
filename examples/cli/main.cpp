#include <stdio.h>
#include <string.h>
#include <time.h>
#include <algorithm>
#include <cctype>
#include <climits>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

// #include "preprocessing.hpp"
#include "stable-diffusion.h"

#include "common/common.h"
#include "common/media_io.h"
#include "common/resource_owners.hpp"
#include "dgl.h"
#include "media_input.h"

namespace fs = std::filesystem;

const char* previews_str[] = {
    "none",
    "proj",
    "tae",
    "vae",
};

std::regex format_specifier_regex("(?:[^%]|^)(?:%%)*(%\\d{0,3}d)");

struct SDCliParams {
    SDMode mode             = IMG_GEN;
    std::string output_path = "output.png";
    int output_begin_idx    = -1;
    int compression_quality = 90;
    std::string image_path;
    std::string metadata_format = "text";
    DGLModeParams dgl;

    sd_log_level_t log_level = SD_LOG_INFO;
    bool convert_name        = false;

    std::string preview_method = "none";
    int preview_interval       = 1;
    std::string preview_path   = "preview.png";
    int preview_fps            = 16;
    bool preview_noisy         = false;
    bool color                 = false;
    bool metadata_raw          = false;
    bool metadata_brief        = false;
    bool metadata_all          = false;

    std::string imatrix_out;
    std::vector<std::string> imatrix_in;

    bool normal_exit = false;

    ArgOptions get_options() {
        ArgOptions options;

        options.string_options = {
            {"-o",
             "--output",
             "path to write result image to. you can use printf-style %d format specifiers for image sequences (default: ./output.png) (eg. output_%03d.png). Single-file video outputs support .avi, .webm, and animated .webp",
             0,
             &output_path},
            {"",
             "--image",
             "path to the image or video to inspect (for metadata mode)",
             0,
             &image_path},
            {"",
             "--metadata-format",
             "metadata output format, one of [text, json] (default: text)",
             0,
             &metadata_format},
            {"",
             "--preview-path",
             "path to write preview image to (default: ./preview.png). For image generation, the filename can have %03d placeholder for sequential numbering. Multi-frame previews support .avi, .webm, and animated .webp",
             0,
             &preview_path},
            {"",
             "--imat-out",
             "compute the imatrix for this run and save it to the provided path",
             0,
             &imatrix_out},
        };

        options.int_options = {
            {"",
             "--preview-interval",
             "preview interval: in each sampling pass, positive N updates every Nth denoiser step and -N previews only completed logical step N; 0 previews the final completed step of the first pass (base-resolution or high-noise). Default: 1",
             &preview_interval},
            {"",
             "--output-begin-idx",
             "starting index for output image sequence, must be non-negative (default 0 if specified %d in output path, 1 otherwise)",
             &output_begin_idx},
            {"",
             "--compression-quality",
             "compression quality of video and JPEG / WebP images (90 by default)",
             &compression_quality},
        };

        options.bool_options = {
            {"",
             "--convert-name",
             "convert tensor name (for convert mode)",
             true, &convert_name},
            {"",
             "--color",
             "colors the logging tags according to level",
             true, &color},
            {"",
             "--preview-noisy",
             "enables previewing noisy inputs of the models rather than the denoised outputs",
             true, &preview_noisy},
            {"",
             "--metadata-raw",
             "include raw hex previews for unparsed metadata payloads",
             true, &metadata_raw},
            {"",
             "--metadata-brief",
             "truncate long metadata text values in text output",
             true, &metadata_brief},
            {"",
             "--metadata-all",
             "include structural/container entries such as IHDR, IDAT, and non-metadata JPEG segments",
             true, &metadata_all},

        };

        auto on_mode_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            const char* mode_c_str = argv[index];
            if (mode_c_str != nullptr) {
                int mode_found = -1;
                for (int i = 0; i < MODE_COUNT; i++) {
                    if (!strcmp(mode_c_str, modes_str[i])) {
                        mode_found = i;
                    }
                }
                if (mode_found == -1) {
                    LOG_ERROR("error: invalid mode %s, must be one of [%s]\n",
                              mode_c_str, SD_ALL_MODES_STR);
                    exit(1);
                }
                mode = (SDMode)mode_found;
            }
            return 1;
        };

        auto on_preview_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            preview_method = argv[index];
            return 1;
        };

        auto on_help_arg = [&](int argc, const char** argv, int index, bool& valid) {
            normal_exit = true;
            valid       = true;
            return -1;
        };

        auto on_imatrix_in_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            imatrix_in.push_back(argv_to_utf8(index, argv));
            return 1;
        };

        options.manual_options = {
            {"-M",
             "--mode",
             "run mode, one of [" SD_ALL_MODES_STR "], default: img_gen",
             on_mode_arg},
            {"",
             "--preview",
             std::string("preview method. must be one of the following [") + previews_str[0] + ", " + previews_str[1] + ", " + previews_str[2] + ", " + previews_str[3] + "] (default is " + previews_str[0] + ")",
             on_preview_arg},
            {"-h",
             "--help",
             "show this help message and exit",
             on_help_arg},
            {"",
             "--imat-in",
             "load an imatrix file for quantization or continued collection; can be specified multiple times",
             on_imatrix_in_arg},
        };

        add_log_options(options, log_level);
        return options;
    };

    bool resolve() {
        if (mode == CONVERT) {
            if (output_path == "output.png") {
                output_path = "output.gguf";
            }
        }
        return true;
    }

    bool validate() {
        if (mode == DGL) {
            return dgl.validate();
        }
        if (dgl.has_options()) {
            LOG_ERROR("DGL options require --mode dgl");
            return false;
        }
        if (mode != METADATA) {
            if (output_path.length() == 0) {
                LOG_ERROR("error: the following arguments are required: output_path");
                return false;
            }
        } else {
            if (image_path.empty()) {
                LOG_ERROR("error: metadata mode needs an image or video path (--image)");
                return false;
            }
            if (metadata_format != "text" && metadata_format != "json") {
                LOG_ERROR("error: invalid metadata format %s, must be one of [text, json]",
                          metadata_format.c_str());
                return false;
            }
        }
        return true;
    }

    bool resolve_and_validate() {
        if (!resolve()) {
            return false;
        }
        if (!validate()) {
            return false;
        }
        return true;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "SDCliParams {\n"
            << "  mode: " << modes_str[mode] << ",\n"
            << "  output_path: \"" << output_path << "\",\n"
            << "  image_path: \"" << image_path << "\",\n"
            << "  metadata_format: \"" << metadata_format << "\",\n"
            << "  log_level: " << log_level_name(log_level) << ",\n"
            << "  color: " << (color ? "true" : "false") << ",\n"
            << "  convert_name: " << (convert_name ? "true" : "false") << ",\n"
            << "  preview_method: " << preview_method << ",\n"
            << "  preview_interval: " << preview_interval << ",\n"
            << "  preview_path: \"" << preview_path << "\",\n"
            << "  preview_fps: " << preview_fps << ",\n"
            << "  preview_noisy: " << (preview_noisy ? "true" : "false") << ",\n"
            << "  imatrix_out: \"" << imatrix_out << "\",\n"
            << "  metadata_raw: " << (metadata_raw ? "true" : "false") << ",\n"
            << "  metadata_brief: " << (metadata_brief ? "true" : "false") << ",\n"
            << "  metadata_all: " << (metadata_all ? "true" : "false") << "\n"
            << "}";
        return oss.str();
    }
};

void print_usage(int argc, const char* argv[], const std::vector<ArgOptions>& options_list) {
    std::cout << version_string() << "\n";
    std::cout << "Usage: " << argv[0] << " [options]\n\n";
    std::cout << "CLI Options:\n";
    options_list[0].print();
    std::cout << "\nContext Options:\n";
    options_list[1].print();
    std::cout << "\nGeneration Options:\n";
    options_list[2].print();
    std::cout << "\nDGL Options:\n";
    options_list[3].print();
}

void parse_args(int argc, const char** argv, SDCliParams& cli_params, SDContextParams& ctx_params, SDGenerationParams& gen_params) {
    std::vector<ArgOptions> options_vec = {cli_params.get_options(), ctx_params.get_options(), gen_params.get_options(), cli_params.dgl.get_options()};

    if (!parse_options(argc, argv, options_vec)) {
        print_usage(argc, argv, options_vec);
        exit(cli_params.normal_exit ? 0 : 1);
    }

    log_level  = cli_params.log_level;
    log_color  = cli_params.color;
    log_stream = cli_params.mode == DGL ? stderr : nullptr;

    if (ctx_params.list_devices) {
        exit(ctx_params.print_devices() ? 0 : 1);
    }

    bool valid = cli_params.resolve_and_validate();
    if (valid && cli_params.mode != METADATA && cli_params.mode != DGL) {
        valid = ctx_params.resolve_and_validate(cli_params.mode) &&
                gen_params.resolve_and_validate(cli_params.mode,
                                                ctx_params.lora_model_dir,
                                                ctx_params.hires_upscalers_dir);
    }

    if (!valid) {
        print_usage(argc, argv, options_vec);
        exit(1);
    }
}

void sd_log_cb(enum sd_log_level_t level, const char* log, void* data) {
    SDCliParams* cli_params = (SDCliParams*)data;
    log_print(level, log, cli_params->log_level, cli_params->color, log_stream);
}

std::string format_frame_idx(std::string pattern, int frame_idx) {
    std::smatch match;
    std::string result = pattern;
    while (std::regex_search(result, match, format_specifier_regex)) {
        std::string specifier = match.str(1);
        char buffer[32];
        snprintf(buffer, sizeof(buffer), specifier.c_str(), frame_idx);
        result.replace(match.position(1), match.length(1), buffer);
    }

    // Then replace all '%%' with '%'
    size_t pos = 0;
    while ((pos = result.find("%%", pos)) != std::string::npos) {
        result.replace(pos, 2, "%");
        pos += 1;
    }
    return result;
}

int continuous_preview_counter = 0;

void step_callback(int step, int frame_count, const sd_tensor_t* const* tensors, bool is_noisy, void* data) {
    std::vector<SDImage> views;
    for (int i = 0; i < frame_count; ++i)
        views.push_back(tensor_image_view(tensors[i]));
    SDImage* image = views.data();
    (void)step;
    (void)is_noisy;
    SDCliParams* cli_params = (SDCliParams*)data;
    // is_noisy is set to true if the preview corresponds to noisy latents, false if it's denoised latents
    // unused in this app, it will either be always noisy or always denoised here
    if (frame_count == 1) {
        fs::path path = fs::u8path(cli_params->preview_path);
        if (encoded_image_format_from_path(path.u8string()) == EncodedImageFormat::UNKNOWN)
            path += ".png";
        if (std::regex_search(path.u8string(), format_specifier_regex))
            path = fs::u8path(format_frame_idx(path.u8string(), continuous_preview_counter++));
        if (!write_image_to_file(path.u8string(),
                                 image->data,
                                 image->width,
                                 image->height,
                                 image->channel,
                                 "",
                                 cli_params->compression_quality)) {
            LOG_ERROR("save preview image to '%s' failed", path.u8string().c_str());
        }
    } else {
        int fps = cli_params->preview_method == "proj" ? cli_params->preview_fps / 4 : cli_params->preview_fps;
        if (create_video_from_sd_images(cli_params->preview_path.c_str(), image, frame_count, fps, cli_params->compression_quality) != 0) {
            LOG_ERROR("save preview video to '%s' failed", cli_params->preview_path.c_str());
        }
    }
}

static fs::path get_video_audio_sidecar_path(const SDCliParams& cli_params) {
    fs::path out_path     = fs::u8path(cli_params.output_path);
    fs::path base_path    = out_path;
    fs::path ext          = out_path.has_extension() ? out_path.extension() : fs::path{};
    std::string ext_lower = ext.u8string();
    std::transform(ext_lower.begin(), ext_lower.end(), ext_lower.begin(), ::tolower);
    const EncodedImageFormat output_format = encoded_image_format_from_path(out_path.u8string());
    if (!ext.empty()) {
        if (output_format == EncodedImageFormat::JPEG ||
            output_format == EncodedImageFormat::PNG ||
            output_format == EncodedImageFormat::WEBP ||
            ext_lower == ".avi" ||
            ext_lower == ".webm") {
            base_path.replace_extension();
        }
    }
    base_path += ".wav";
    return base_path;
}

bool save_results(const SDCliParams& cli_params,
                  const SDContextParams& ctx_params,
                  const SDGenerationParams& gen_params,
                  SDImageVec& results,
                  int num_results,
                  const SDAudio* generated_audio,
                  const std::string& video_info) {
    if (results.empty() || num_results <= 0) {
        return false;
    }

    namespace fs      = std::filesystem;
    fs::path out_path = fs::u8path(cli_params.output_path);

    if (!out_path.parent_path().empty()) {
        std::error_code ec;
        fs::create_directories(out_path.parent_path(), ec);
        if (ec) {
            LOG_ERROR("failed to create directory '%s': %s",
                      out_path.parent_path().u8string().c_str(), ec.message().c_str());
            return false;
        }
    }

    fs::path base_path = out_path;
    fs::path ext       = out_path.has_extension() ? out_path.extension() : fs::path{};

    std::string ext_lower = ext.u8string();
    std::transform(ext_lower.begin(), ext_lower.end(), ext_lower.begin(), ::tolower);
    const EncodedImageFormat output_format = encoded_image_format_from_path(out_path.u8string());
    if (!ext.empty()) {
        if (output_format == EncodedImageFormat::JPEG ||
            output_format == EncodedImageFormat::PNG ||
            output_format == EncodedImageFormat::WEBP ||
            ext_lower == ".avi" ||
            ext_lower == ".webm") {
            base_path.replace_extension();
        }
    }

    int output_begin_idx = cli_params.output_begin_idx;
    if (output_begin_idx < 0) {
        output_begin_idx = 0;
    }

    auto write_image = [&](const fs::path& path, int idx) {
        const SDImage& img = results[idx];
        if (!img.data)
            return false;

        std::string params = gen_params.embed_image_metadata
                                 ? results.generation_info(idx)
                                 : "";
        const bool ok      = write_image_to_file(path.u8string(), img.data, img.width, img.height, img.channel, params, cli_params.compression_quality);
        LOG_INFO("save result image %d to '%s' (%s)", idx, path.u8string().c_str(), ok ? "success" : "failure");
        return ok;
    };

    auto write_audio_sidecar = [&](const fs::path& wav_path) {
        if (generated_audio == nullptr) {
            return;
        }
        if (write_wav_to_file(wav_path.u8string(),
                              generated_audio->data,
                              generated_audio->sample_count,
                              generated_audio->channels,
                              generated_audio->sample_rate)) {
            LOG_INFO("save result audio to '%s'", wav_path.u8string().c_str());
        } else {
            LOG_WARN("failed to save result audio to '%s'", wav_path.u8string().c_str());
        }
    };

    int sucessful_reults = 0;

    if (std::regex_search(cli_params.output_path, format_specifier_regex)) {
        if (output_format == EncodedImageFormat::UNKNOWN)
            ext = ".png";
        fs::path pattern = base_path;
        pattern += ext;

        for (int i = 0; i < num_results; ++i) {
            fs::path img_path = fs::u8path(format_frame_idx(pattern.u8string(), output_begin_idx + i));
            if (write_image(img_path, i)) {
                sucessful_reults++;
            }
        }
        LOG_INFO("%d/%d images saved", sucessful_reults, num_results);
        return sucessful_reults != 0;
    }

    if (cli_params.mode == VID_GEN && num_results > 1) {
        if (ext_lower != ".avi" && ext_lower != ".webp" && ext_lower != ".webm")
            ext = ".avi";
        std::string params = gen_params.embed_image_metadata
                                 ? video_info
                                 : "";

        fs::path video_path = base_path;
        video_path += ext;
        std::string final_ext_lower = ext.u8string();
        std::transform(final_ext_lower.begin(), final_ext_lower.end(), final_ext_lower.begin(), ::tolower);
        const bool mux_audio = generated_audio != nullptr && (final_ext_lower == ".avi" || final_ext_lower == ".webm");
        if (create_video_from_sd_images(video_path.u8string().c_str(), results.data(), num_results, gen_params.fps.value(), cli_params.compression_quality, mux_audio ? generated_audio : nullptr, params) == 0) {
            LOG_INFO("save result video to '%s'", video_path.u8string().c_str());
            if (generated_audio != nullptr && !mux_audio) {
                fs::path wav_path = video_path;
                wav_path.replace_extension(".wav");
                write_audio_sidecar(wav_path);
            }
            return true;
        } else {
            LOG_ERROR("Failed to save result video to '%s'", video_path.u8string().c_str());
            return false;
        }
    }

    if (output_format == EncodedImageFormat::UNKNOWN)
        ext = ".png";

    for (int i = 0; i < num_results; ++i) {
        fs::path img_path = base_path;
        if (num_results > 1) {
            img_path += "_" + std::to_string(output_begin_idx + i);
        }
        img_path += ext;
        if (write_image(img_path, i)) {
            sucessful_reults++;
        }
    }
    LOG_INFO("%d/%d images saved", sucessful_reults, num_results);
    if (generated_audio != nullptr) {
        write_audio_sidecar(get_video_audio_sidecar_path(cli_params));
    }
    return sucessful_reults != 0;
}

int main(int argc, const char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "--version") {
        std::cout << version_string() << "\n";
        return EXIT_SUCCESS;
    }

    SDCliParams cli_params;
    SDContextParams ctx_params;
    SDGenerationParams gen_params;
    gen_params.seed = 42;

    sd_set_log_callback(sd_log_cb, (void*)&cli_params);
    parse_args(argc, argv, cli_params, ctx_params, gen_params);

    if (cli_params.mode == DGL) {
        return cli_params.dgl.run(ctx_params, gen_params.embed_image_metadata);
    }

    if (cli_params.mode == METADATA) {
        uint32_t flags = 0;
        if (cli_params.metadata_format == "json")
            flags |= SD_METADATA_READ_JSON;
        if (cli_params.metadata_raw)
            flags |= SD_METADATA_READ_RAW;
        if (cli_params.metadata_brief)
            flags |= SD_METADATA_READ_BRIEF;
        if (cli_params.metadata_all)
            flags |= SD_METADATA_READ_STRUCTURAL;
        const auto* output = sd_read_metadata(cli_params.image_path.c_str(), flags);
        if (!output) {
            LOG_ERROR("%s", sd_get_last_error());
            return 1;
        }
        print_utf8(stdout, output);
        return 0;
    }

    if (!ctx_params.esrgan_path.empty() && cli_params.mode != UPSCALE) {
        LOG_ERROR("generation followed by automatic upscaling requires DGL; use -M dgl --script FILE");
        return 1;
    }

    if (cli_params.mode == VID_GEN || gen_params.video_frames > 4) {
        size_t last_dot_pos   = cli_params.preview_path.find_last_of(".");
        std::string base_path = cli_params.preview_path;
        std::string file_ext  = "";
        if (last_dot_pos != std::string::npos) {  // filename has extension
            base_path = cli_params.preview_path.substr(0, last_dot_pos);
            file_ext  = cli_params.preview_path.substr(last_dot_pos);
            std::transform(file_ext.begin(), file_ext.end(), file_ext.begin(), ::tolower);
        }
        if (file_ext == ".png") {
            cli_params.preview_path = base_path + ".avi";
        }
    }
    if (gen_params.fps)
        cli_params.preview_fps = *gen_params.fps;

    if (sd_set_preview_callback(step_callback,
                                cli_params.preview_method.c_str(),
                                cli_params.preview_interval,
                                !cli_params.preview_noisy,
                                cli_params.preview_noisy,
                                (void*)&cli_params) != SD_STATUS_OK) {
        LOG_ERROR("%s", sd_get_last_error());
        return 1;
    }

    LOG_VERBOSE("version: %s", version_string().c_str());
    LOG_VERBOSE("%s", sd_get_system_info());
    LOG_DEBUG("%s", cli_params.to_string().c_str());
    LOG_DEBUG("%s", ctx_params.to_string().c_str());
    LOG_DEBUG("%s", gen_params.to_string().c_str());

    if (!cli_params.imatrix_out.empty()) {
        if (fs::exists(fs::u8path(cli_params.imatrix_out)) &&
            std::find(cli_params.imatrix_in.begin(), cli_params.imatrix_in.end(), cli_params.imatrix_out) == cli_params.imatrix_in.end()) {
            LOG_WARN("imatrix file '%s' already exists and will be overwritten", cli_params.imatrix_out.c_str());
        }
        enable_imatrix_collection();
    }

    for (const auto& in_file : cli_params.imatrix_in) {
        LOG_INFO("loading imatrix from '%s'", in_file.c_str());
        if (!load_imatrix(in_file.c_str())) {
            LOG_WARN("failed to load imatrix from '%s'", in_file.c_str());
        }
    }

    if (cli_params.mode == CONVERT) {
        std::vector<sd_lora_t> loras;
        loras.reserve(gen_params.lora_map.size() + gen_params.high_noise_lora_map.size());
        for (const auto& [path, multiplier] : gen_params.lora_map) {
            loras.push_back({false, multiplier, path.c_str()});
        }
        for (const auto& [path, multiplier] : gen_params.high_noise_lora_map) {
            loras.push_back({true, multiplier, path.c_str()});
        }

        bool success = convert_with_components(ctx_params.model_path.c_str(),
                                               ctx_params.clip_l_path.c_str(),
                                               ctx_params.clip_g_path.c_str(),
                                               ctx_params.t5xxl_path.c_str(),
                                               ctx_params.diffusion_model_path.c_str(),
                                               ctx_params.vae_path.c_str(),
                                               cli_params.output_path.c_str(),
                                               ctx_params.wtype,
                                               ctx_params.tensor_type_rules.c_str(),
                                               cli_params.convert_name,
                                               ctx_params.n_threads.value_or(0),
                                               loras.empty() ? nullptr : loras.data(),
                                               (int)loras.size());
        if (!success) {
            LOG_ERROR("convert '%s'/'%s' to '%s' failed",
                      ctx_params.model_path.c_str(),
                      ctx_params.vae_path.c_str(),
                      cli_params.output_path.c_str());
            return 1;
        } else {
            LOG_INFO("convert '%s'/'%s' to '%s' success",
                     ctx_params.model_path.c_str(),
                     ctx_params.vae_path.c_str(),
                     cli_params.output_path.c_str());
            return 0;
        }
    }

    auto load_image_and_update_size = [&](const std::string& path,
                                          SDImageOwner& image,
                                          int expected_channel = 3) -> bool {
        if (!load_sd_image_from_file(image.put(), path.c_str(), 0, 0, expected_channel)) {
            LOG_ERROR("load image from '%s' failed", path.c_str());
            return false;
        }

        gen_params.set_width_and_height_if_unset(image.get().width, image.get().height);
        return true;
    };

    auto load_audio = [&](const std::string& path, SDAudioOwner& audio) -> bool {
        std::vector<float> samples;
        uint32_t sample_rate = 0;
        uint32_t channels    = 0;
        if (!load_wav_from_file(path, samples, sample_rate, channels)) {
            LOG_ERROR("load WAV audio from '%s' failed", path.c_str());
            return false;
        }
        audio.reset(std::move(samples), sample_rate, channels);
        return true;
    };

    if (gen_params.init_image_path.size() > 0) {
        const bool native_init = cli_params.mode == IMG_GEN;
        if (!load_image_and_update_size(gen_params.init_image_path, gen_params.init_image, native_init ? 0 : 3)) {
            return 1;
        }
    }

    if (gen_params.end_image_path.size() > 0) {
        if (!load_image_and_update_size(gen_params.end_image_path, gen_params.end_image)) {
            return 1;
        }
    }

    if (gen_params.ref_image_paths.size() > 0) {
        gen_params.ref_images.clear();
        for (auto& path : gen_params.ref_image_paths) {
            SDImageOwner ref_image({0, 0, 0, nullptr});
            if (!load_image_and_update_size(path, ref_image, 0)) {
                return 1;
            }
            gen_params.ref_images.push_back(std::move(ref_image));
        }
    }

    if (!gen_params.ref_video_paths.empty()) {
        gen_params.ref_videos.clear();
        gen_params.ref_video_audios.clear();
    }
    std::vector<VideoInput> reference_video_inputs;
    if (!load_reference_video_inputs(gen_params.ref_video_paths, gen_params.ref_video_audio_paths, reference_video_inputs))
        return 1;

    if (!gen_params.ref_audio_paths.empty()) {
        gen_params.ref_audios.clear();
        gen_params.ref_audios.resize(gen_params.ref_audio_paths.size());
        for (size_t i = 0; i < gen_params.ref_audio_paths.size(); ++i) {
            if (!load_audio(gen_params.ref_audio_paths[i], gen_params.ref_audios[i])) {
                return 1;
            }
        }
    }

    if (gen_params.mask_image_path.size() > 0) {
        if (!load_sd_image_from_file(gen_params.mask_image.put(),
                                     gen_params.mask_image_path.c_str(),
                                     0,
                                     0,
                                     1)) {
            LOG_ERROR("load image from '%s' failed", gen_params.mask_image_path.c_str());
            return 1;
        }
    }

    if (gen_params.control_image_path.size() > 0) {
        if (!load_sd_image_from_file(gen_params.control_image.put(),
                                     gen_params.control_image_path.c_str(),
                                     0,
                                     0)) {
            LOG_ERROR("load image from '%s' failed", gen_params.control_image_path.c_str());
            return 1;
        }
    }

    VideoInput control_video_input;
    if (!gen_params.control_video_path.empty()) {
        gen_params.control_frames.clear();
        if (fs::is_directory(fs::u8path(gen_params.control_video_path))) {
            if (!load_images_from_dir(gen_params.control_video_path,
                                      gen_params.control_frames,
                                      gen_params.video_frames.value_or(0)))
                return 1;
        } else if (!load_video_input(gen_params.control_video_path, gen_params.fps.value_or(16),
                                     gen_params.video_frames.value_or(0), false, control_video_input)) {
            return 1;
        }
    }

    SDOptPtr sd_ctx_params = ctx_params.to_context_opt();

    SDImageVec results;
    int num_results          = 0;
    SDAudio* generated_audio = nullptr;
    std::string video_info;

    if (cli_params.mode == UPSCALE) {
        num_results = 1;
        results.push_back(gen_params.init_image.release());
    } else {
        SDCtxPtr sd_ctx(create_sd_context(sd_ctx_params));

        if (sd_ctx == nullptr) {
            LOG_INFO("sd_ctx_create failed");
            return 1;
        }

        SDImageOwner prompt_image;
        sd_set_prompt_image_callback(
            sd_ctx.get(), [](const char* source, void* data, const sd_tensor_t** image_out) -> bool {
                auto& image = *static_cast<SDImageOwner*>(data);
                SDImage loaded{};
                if (!load_sd_image_from_file(&loaded, source, 0, 0, 3)) {
                    return false;
                }
                image.reset(loaded);
                *image_out = image.tensor();
                return true;
            },
            &prompt_image);

        if (cli_params.mode == IMG_GEN) {
            sd_result_t* generated = nullptr;
            generate_sd_image(sd_ctx.get(), gen_params.to_image_opt(), &generated);
            results.adopt(generated);
            num_results = results.count();
        } else if (cli_params.mode == VID_GEN) {
            SDOptPtr vid_gen_params = gen_params.to_video_opt();
            if (!vid_gen_params || !set_video_inputs(vid_gen_params.get(), reference_video_inputs, control_video_input.frames.get()))
                return 1;
            sd_result_t* generated = nullptr;
            if (generate_sd_video(sd_ctx.get(), vid_gen_params, &generated)) {
                gen_params.fps = cli_params.preview_fps = sd_result_get_fps(generated);
                generated_audio                         = retain_result_audio(generated);

                num_results = sd_result_get_image_count(generated);
                results.adopt_video(generated);
                video_info = results.generation_info(0);
            }
        }

        if (!results) {
            LOG_ERROR("generate failed: %s", sd_get_last_error());
            return 1;
        }
    }

    if (ctx_params.esrgan_path.size() > 0 && gen_params.upscale_repeats > 0) {
        UpscalerCtxPtr upscaler_ctx(create_sd_upscaler(ctx_params.esrgan_path.c_str(),
                                                       ctx_params.diffusion_conv_direct,
                                                       ctx_params.n_threads,
                                                       gen_params.upscale_tile_size,
                                                       ctx_params.effective_backend.c_str(),
                                                       ctx_params.effective_params_backend.c_str()));

        if (upscaler_ctx == nullptr) {
            LOG_ERROR("sd_upscaler_create failed");
        } else {
            for (int i = 0; i < num_results; i++) {
                if (results[i].data == nullptr) {
                    continue;
                }
                SDImageOwner current_image(results[i]);
                results[i]            = {0, 0, 0, nullptr};
                sd_result_t* upscaled = nullptr;
                if (!upscale_sd_image(upscaler_ctx.get(), current_image.get(), &upscaled, gen_params.upscale_repeats) || !upscaled) {
                    LOG_ERROR("upscale failed: %s", sd_get_last_error());
                    sd_result_free(upscaled);
                    return 1;
                }
                const std::string source = sd_result_get_generation_info(upscaled, 0);
                current_image.reset(take_result_image(upscaled, 0));
                results.set_generation_info(i, source);
                sd_result_free(upscaled);
                results[i] = current_image.release();  // Set the final upscaled image as the result
            }
        }
    }

    if (!video_info.empty() && results.count() > 0)
        video_info = results.generation_info(0);
    if (!save_results(cli_params, ctx_params, gen_params, results, num_results, generated_audio, video_info)) {
        free_sd_audio(generated_audio);
        return 1;
    }

    if (!cli_params.imatrix_out.empty()) {
        LOG_INFO("saving imatrix to '%s'", cli_params.imatrix_out.c_str());
        save_imatrix(cli_params.imatrix_out.c_str());
    }

    free_sd_audio(generated_audio);

    return 0;
}
