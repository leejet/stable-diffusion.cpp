#include "runtime.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <regex>
#include <sstream>

#include "common/common.h"
#include "common/log.h"
#include "model_catalog.h"

#include <map>
#include <set>

namespace fs = std::filesystem;

bool validate_server_img_gen_request(const ServerRuntime& runtime,
                                     const ImgGenJobRequest& request,
                                     std::string& error_message) {
    auto dimensions_valid = [&](double width, double height) {
        if (!std::isfinite(width) || !std::isfinite(height) ||
            width < 1 || height < 1 || width > k_max_generation_dimension || height > k_max_generation_dimension) {
            error_message = "generation width and height must be between 1 and " + std::to_string(k_max_generation_dimension);
            return false;
        }
        return true;
    };
    const auto& params = request.gen_params;
    const int width    = params.get_resolved_width();
    const int height   = params.get_resolved_height();
    if (!dimensions_valid(width, height)) {
        return false;
    }
    if (params.hires_enabled) {
        const double hires_width  = params.hires_width > 0 ? params.hires_width : params.hires_height > 0 ? params.hires_height
                                                                                                          : std::round(width * params.hires_scale);
        const double hires_height = params.hires_height > 0 ? params.hires_height : params.hires_width > 0 ? params.hires_width
                                                                                                           : std::round(height * params.hires_scale);
        if (!dimensions_valid(hires_width, hires_height)) {
            return false;
        }
    }
    if (!request.adetailer_enabled || runtime.default_gen_params->ad_model_path.empty()) {
        return true;
    }
    auto pass_valid = [&](const ServerADetailerPass& pass) {
        if (!dimensions_valid(pass.width, pass.height)) {
            return false;
        }
        // Match the native key normalization, but leave full option parsing to ADetailer.
        auto trim = [](const std::string& text) {
            const auto first = text.find_first_not_of(" \t\r\n");
            return first == std::string::npos ? std::string() : text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
        };
        std::stringstream stream(pass.extra_args);
        std::string item;
        while (std::getline(stream, item, ',')) {
            const auto equals = item.find('=');
            if (equals == std::string::npos) {
                continue;
            }
            auto key = trim(item.substr(0, equals));
            std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (key != "inpaint_width" && key != "inpaint_height" && key != "steps") {
                continue;
            }
            const auto value  = trim(item.substr(equals + 1));
            const int minimum = key == "steps" ? 0 : 1;
            const int maximum = key == "steps" ? k_max_adetailer_steps : k_max_generation_dimension;
            try {
                size_t used      = 0;
                const int parsed = std::stoi(value, &used);
                if (used == value.size() && parsed >= minimum && parsed <= maximum) {
                    continue;
                }
            } catch (const std::exception&) {
            }
            error_message = "ADetailer " + key + " must be between " + std::to_string(minimum) + " and " + std::to_string(maximum);
            return false;
        }
        return true;
    };
    if (request.adetailer_passes.empty()) {
        ServerADetailerPass pass;
        pass.extra_args = params.extra_ad_args;
        return pass_valid(pass);
    }
    return std::all_of(request.adetailer_passes.begin(), request.adetailer_passes.end(), pass_valid);
}

bool apply_server_adetailer(ServerRuntime& runtime,
                            const ImgGenJobRequest& request,
                            const sd_img_gen_params_t& params,
                            SDImageVec& results) {
    const auto& gen_params    = request.gen_params;
    const auto& ad_model_path = runtime.default_gen_params->ad_model_path;
    if (!request.adetailer_enabled || ad_model_path.empty()) {
        return true;
    }
    std::string error_message;
    if (!validate_server_img_gen_request(runtime, request, error_message)) {
        LOG_ERROR("%s", error_message.c_str());
        return false;
    }

    std::vector<ServerADetailerPass> passes = request.adetailer_passes;
    if (passes.empty()) {
        ServerADetailerPass pass;
        pass.prompt          = gen_params.ad_prompt;
        pass.negative_prompt = gen_params.ad_negative_prompt;
        pass.extra_args      = gen_params.extra_ad_args;
        passes.push_back(std::move(pass));
    }

    ADetailerCtxPtr ad_ctx(new_adetailer_ctx(ad_model_path.c_str(),
                                             runtime.ctx_params->n_threads,
                                             runtime.ctx_params->effective_backend.c_str(),
                                             runtime.ctx_params->effective_params_backend.c_str()));
    if (ad_ctx == nullptr) {
        LOG_ERROR("new_adetailer_ctx failed");
        return false;
    }

    for (const auto& pass : passes) {
        sd_adetailer_params_t ad_params{};
        ad_params.prompt          = pass.prompt.empty() ? nullptr : pass.prompt.c_str();
        ad_params.negative_prompt = pass.negative_prompt.empty() ? nullptr : pass.negative_prompt.c_str();
        ad_params.extra_ad_args   = pass.extra_args.c_str();
        for (int i = 0; i < results.count(); ++i) {
            if (results[i].data == nullptr) {
                continue;
            }
            sd_img_gen_params_t ad_generation_params = params;
            ad_generation_params.seed                = generation_seed(results.generation_info(i));
            ad_generation_params.width               = pass.width;
            ad_generation_params.height              = pass.height;
            ad_generation_params.strength            = pass.strength;
            if (pass.clip_skip > 0) {
                ad_generation_params.clip_skip = pass.clip_skip;
            }
            if (pass.automatic_scheduler) {
                ad_generation_params.sample_params.scheduler = SCHEDULER_COUNT;
            }

            sd_image_generation_result_t* detailed = nullptr;
            if (!adetail_image(ad_ctx.get(),
                               runtime.sd_ctx,
                               results[i],
                               &ad_params,
                               &ad_generation_params,
                               &detailed) ||
                detailed == nullptr || detailed->count <= 0 || detailed->items[0].image.data == nullptr) {
                free_sd_image_generation_result(detailed);
                LOG_ERROR("ADetailer failed for image %d", i + 1);
                return false;
            }
            results.append_generation_info(i, detailed->items[0].generation_info, runtime.ctx_params->lora_model_dir);
            free(results[i].data);
            results[i]               = detailed->items[0].image;
            detailed->items[0].image = {};
            free_sd_image_generation_result(detailed);
        }
    }
    return true;
}

static std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

static bool is_supported_model_ext(const fs::path& p) {
    auto ext = lower_ascii(p.extension().u8string());
    return ext == ".gguf" || ext == ".pt" || ext == ".pth" || ext == ".safetensors";
}

static const std::string k_base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

std::string base64_encode(const std::vector<uint8_t>& bytes) {
    std::string ret;
    int val  = 0;
    int valb = -6;
    for (uint8_t c : bytes) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            ret.push_back(k_base64_chars[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6) {
        ret.push_back(k_base64_chars[((val << 8) >> (valb + 8)) & 0x3F]);
    }
    while (ret.size() % 4) {
        ret.push_back('=');
    }
    return ret;
}

std::string normalize_output_format(std::string output_format) {
    std::transform(output_format.begin(), output_format.end(), output_format.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return output_format;
}

std::vector<std::string> supported_img_output_formats(bool allow_webp) {
    std::vector<std::string> formats = {"png", "jpeg"};
#ifdef SD_USE_WEBP
    if (allow_webp) {
        formats.push_back("webp");
    }
#else
    (void)allow_webp;
#endif
    return formats;
}

std::vector<std::string> supported_vid_output_formats() {
    std::vector<std::string> formats;
#ifdef SD_USE_WEBM
    formats.push_back("webm");
#endif
#ifdef SD_USE_WEBP
    formats.push_back("webp");
#endif
    formats.push_back("avi");
    return formats;
}

static std::string valid_vid_output_formats_message() {
    const std::vector<std::string> formats = supported_vid_output_formats();

    std::string message = "invalid output_format, must be one of [";
    for (size_t i = 0; i < formats.size(); ++i) {
        if (i > 0) {
            message += ", ";
        }
        message += formats[i];
    }
    message += "]";
    return message;
}

bool assign_output_options(ImgGenJobRequest& request,
                           std::string output_format,
                           int output_compression,
                           bool allow_webp,
                           std::string& error_message) {
    request.output_format      = normalize_output_format(std::move(output_format));
    request.output_compression = std::clamp(output_compression, 0, 100);

    const std::vector<std::string> valid_formats = supported_img_output_formats(allow_webp);
    const bool valid_format                      = std::find(valid_formats.begin(),
                                                             valid_formats.end(),
                                                             request.output_format) != valid_formats.end();
    if (!valid_format) {
        error_message = "invalid output_format, must be one of [";
        for (size_t i = 0; i < valid_formats.size(); ++i) {
            if (i > 0) {
                error_message += ", ";
            }
            error_message += valid_formats[i];
        }
        error_message += "]";
        return false;
    }

    return true;
}

bool assign_output_options(VidGenJobRequest& request,
                           std::string output_format,
                           int output_compression,
                           std::string& error_message) {
    request.output_format      = normalize_output_format(std::move(output_format));
    request.output_compression = std::clamp(output_compression, 0, 100);

    if (request.output_format == "avi") {
        return true;
    }

    if (request.output_format == "webm") {
#ifdef SD_USE_WEBM
        return true;
#else
        error_message = valid_vid_output_formats_message();
        return false;
#endif
    }

    if (request.output_format == "webp") {
#ifdef SD_USE_WEBP
        return true;
#else
        error_message = valid_vid_output_formats_message();
        return false;
#endif
    }

    error_message = valid_vid_output_formats_message();
    return false;
}

std::string video_mime_type(const std::string& output_format) {
    if (output_format == "webm") {
        return "video/webm";
    }
    if (output_format == "webp") {
        return "image/webp";
    }
    return "video/x-msvideo";
}

bool runtime_supports_generation_mode(const ServerRuntime& runtime, SDMode mode) {
    if (mode == VID_GEN) {
        return sd_ctx_supports_video_generation(runtime.sd_ctx);
    }
    if (mode == IMG_GEN) {
        return sd_ctx_supports_image_generation(runtime.sd_ctx);
    }
    return true;
}

std::string unsupported_generation_mode_error(SDMode mode) {
    if (mode == VID_GEN) {
        return "loaded model does not support vid_gen";
    }
    if (mode == IMG_GEN) {
        return "loaded model does not support img_gen";
    }
    return "loaded model does not support requested mode";
}

ArgOptions SDSvrParams::get_options() {
    ArgOptions options;

    options.string_options = {
        {"-l", "--listen-ip", "server listen ip (default: 127.0.0.1)", 0, &listen_ip},
        {"", "--serve-html-path", "path to HTML file to serve at root (optional)", 0, &serve_html_path},
    };

    options.int_options = {
        {"", "--listen-port", "server listen port (default: 1234)", &listen_port},
    };

    options.bool_options = {
        {"", "--color", "colors the logging tags according to level", true, &color},
    };

    auto on_help_arg = [&](int, const char**, int, bool& valid) {
        normal_exit = true;
        valid       = true;
        return -1;
    };

    options.manual_options = {
        {"-h", "--help", "show this help message and exit", on_help_arg},
        {"", "--model-dir", "model root; repeat to search multiple directories (default: models)",
         [this](int argc, const char** argv, int index) {
             if (++index >= argc || argv[index][0] == '\0')
                 return -1;
             model_dirs.push_back(argv_to_utf8(index, argv));
             return 1;
         }},
        {"", "--recipes", "recipe file (.json or .ini with one section per recipe); repeatable, takes precedence over recipes directories",
         [this](int argc, const char** argv, int index) {
             if (++index >= argc || argv[index][0] == '\0')
                 return -1;
             recipe_files.push_back(argv_to_utf8(index, argv));
             return 1;
         }},
    };
    add_log_options(options, log_level);
    return options;
}

bool SDSvrParams::validate() {
    if (listen_ip.empty()) {
        LOG_ERROR("error: the following arguments are required: listen_ip");
        return false;
    }

    if (listen_port < 0 || listen_port > 65535) {
        LOG_ERROR("error: listen_port should be in the range [0, 65535]");
        return false;
    }

    if (!serve_html_path.empty() && !fs::exists(fs::u8path(serve_html_path))) {
        LOG_ERROR("error: serve_html_path file does not exist: %s", serve_html_path.c_str());
        return false;
    }
    return true;
}

bool SDSvrParams::resolve_and_validate() {
    if (!validate()) {
        return false;
    }
    return true;
}

std::string SDSvrParams::to_string() const {
    std::ostringstream oss;
    oss << "SDSvrParams {\n"
        << "  log_level: " << log_level_name(log_level) << ",\n"
        << "  listen_ip: " << listen_ip << ",\n"
        << "  listen_port: \"" << listen_port << "\",\n"
        << "  serve_html_path: \"" << serve_html_path << "\",\n"
        << "}";
    return oss.str();
}

void refresh_lora_cache(ServerRuntime& rt) {
    std::vector<LoraEntry> new_cache;
    std::vector<ModelRoot> directories;
    if (!rt.ctx_params->lora_model_dir.empty()) {
        directories.push_back(make_model_root(fs::u8path(rt.ctx_params->lora_model_dir), true));
    } else {
        for (const auto& root : get_model_roots(*rt.svr_params)) {
            try {
                const auto paths = model_category_directories(root, "loras");
                for (size_t i = 0; i < paths.size(); ++i) {
                    if (!model_path_inside(root.path, fs::canonical(paths[i]))) {
                        LOG_WARN("LoRA directory escapes model directory: %s", paths[i].u8string().c_str());
                        continue;
                    }
                    directories.push_back(make_model_root(paths[i], root.primary && lower_ascii(paths[i].filename().u8string()) == "loras"));
                }
            } catch (const std::exception& e) {
                LOG_WARN("error while finding LoRA directories in %s: %s", root.label.c_str(), e.what());
            }
        }
    }
    std::set<std::string> seen;
    std::map<std::string, size_t> names;
    for (const auto& directory : directories) {
        try {
            if (!fs::exists(directory.path))
                continue;
            for (fs::recursive_directory_iterator it(directory.path), end; it != end; ++it) {
                std::error_code ec;
                const auto canonical = fs::canonical(it->path(), ec);
                if (ec || !model_path_inside(directory.path, canonical)) {
                    it.disable_recursion_pending();
                    LOG_WARN("skipping inaccessible LoRA or path outside LoRA directory: %s", it->path().u8string().c_str());
                    continue;
                }
                const auto ext = lower_ascii(it->path().extension().u8string());
                if (!it->is_regular_file() || (!is_supported_model_ext(it->path()) && ext != ".ckpt" && ext != ".sft") ||
                    !seen.insert(model_path_key(canonical)).second)
                    continue;
                const auto relative = it->path().lexically_relative(directory.path).generic_u8string();
                LoraEntry entry;
                entry.name         = it->path().stem().u8string();
                entry.path         = directory.primary ? relative : directory.id + "/" + relative;
                entry.fullpath     = it->path().u8string();
                entry.display_path = relative;
                entry.root         = directory.path;
                ++names[relative];
                new_cache.push_back(std::move(entry));
            }
        } catch (const std::exception& e) {
            LOG_WARN("error while scanning LoRA directory '%s': %s", directory.path.u8string().c_str(), e.what());
        }
    }
    for (auto& entry : new_cache) {
        if (names[entry.display_path] > 1)
            entry.display_path = fs::u8path(entry.fullpath).generic_u8string();
    }
    std::sort(new_cache.begin(), new_cache.end(), [](const LoraEntry& a, const LoraEntry& b) {
        return a.path < b.path;
    });
    std::lock_guard<std::mutex> lock(*rt.lora_mutex);
    *rt.lora_cache = std::move(new_cache);
}

std::string get_lora_full_path(ServerRuntime& rt, const std::string& path) {
    std::lock_guard<std::mutex> lock(*rt.lora_mutex);
    auto it = std::find_if(rt.lora_cache->begin(), rt.lora_cache->end(),
                           [&](const LoraEntry& entry) { return entry.path == path; });
    if (it == rt.lora_cache->end())
        return "";
    std::error_code ec;
    const auto canonical = fs::canonical(fs::u8path(it->fullpath), ec);
    if (ec || !model_path_inside(it->root, canonical) || !fs::is_regular_file(canonical, ec) || ec)
        return "";
    return canonical.u8string();
}

void refresh_upscaler_cache(ServerRuntime& rt) {
    std::vector<UpscalerEntry> new_cache;
    std::vector<UpscalerEntry> previous_cache;
    {
        std::lock_guard<std::mutex> lock(*rt.upscaler_mutex);
        previous_cache = *rt.upscaler_cache;
    }

    fs::path upscaler_dir = fs::u8path(rt.ctx_params->hires_upscalers_dir);
    std::error_code ec;
    if (fs::exists(upscaler_dir, ec) && !ec && fs::is_directory(upscaler_dir, ec) && !ec) {
        try {
            auto it = fs::directory_iterator(
                upscaler_dir,
                fs::directory_options::skip_permission_denied,
                ec);
            auto end = fs::directory_iterator();
            while (!ec && it != end) {
                std::error_code entry_ec;
                bool is_reg = it->is_regular_file(entry_ec);
                if (entry_ec || !is_reg) {
                    it.increment(ec);
                    continue;
                }
                const fs::path& p = it->path();
                if (!is_supported_model_ext(p)) {
                    it.increment(ec);
                    continue;
                }

                UpscalerEntry upscaler_entry;
                upscaler_entry.name = p.stem().u8string();
                std::error_code abs_ec;
                fs::path abs_path         = fs::absolute(p, abs_ec);
                upscaler_entry.fullpath   = (abs_ec ? p : abs_path.lexically_normal()).u8string();
                upscaler_entry.model_name = "ESRGAN_4x";
                upscaler_entry.path       = p.filename().u8string();

                std::error_code size_ec;
                upscaler_entry.file_size = it->file_size(size_ec);
                if (size_ec) {
                    it.increment(ec);
                    continue;
                }

                std::error_code time_ec;
                upscaler_entry.last_modified        = it->last_write_time(time_ec);
                auto previous                       = std::find_if(previous_cache.begin(), previous_cache.end(), [&](const UpscalerEntry& cached) {
                    return cached.fullpath == upscaler_entry.fullpath &&
                           cached.file_size == upscaler_entry.file_size &&
                           (!time_ec && cached.last_modified == upscaler_entry.last_modified);
                });
                upscaler_entry.image_upscale_factor = previous != previous_cache.end()
                                                          ? previous->image_upscale_factor
                                                          : get_upscaler_model_scale(upscaler_entry.fullpath.c_str());
                if (upscaler_entry.image_upscale_factor > 0) {
                    upscaler_entry.scale      = upscaler_entry.image_upscale_factor;
                    upscaler_entry.model_name = "ESRGAN_" + std::to_string(upscaler_entry.scale) + "x";
                }

                new_cache.push_back(std::move(upscaler_entry));
                it.increment(ec);
            }
        } catch (const std::exception& e) {
            LOG_WARN("error while scanning upscalers directory '%s': %s", upscaler_dir.u8string().c_str(), e.what());
            return;
        }
    }

    if (ec) {
        LOG_WARN("error while scanning upscalers directory '%s': %s", upscaler_dir.u8string().c_str(), ec.message().c_str());
        return;
    }

    std::sort(new_cache.begin(), new_cache.end(), [](const UpscalerEntry& a, const UpscalerEntry& b) {
        return a.name < b.name;
    });

    {
        std::lock_guard<std::mutex> lock(*rt.upscaler_mutex);
        *rt.upscaler_cache = std::move(new_cache);
    }
}

int64_t unix_timestamp_now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}
