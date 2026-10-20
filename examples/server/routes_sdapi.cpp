#include "routes.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include "common/common.h"
#include "common/media_io.h"
#include "common/resource_owners.hpp"
#include "model_catalog.h"

namespace fs = std::filesystem;

static std::string extract_and_remove_sd_cpp_extra_args(std::string& text) {
    std::regex re("<sd_cpp_extra_args>(.*?)</sd_cpp_extra_args>");
    std::smatch match;

    std::string extracted;
    if (std::regex_search(text, match, re)) {
        extracted = match[1].str();
        text      = std::regex_replace(text, re, "");
    }
    return extracted;
}

static fs::path resolve_display_model_path(const ServerRuntime& runtime) {
    const auto& ctx = *runtime.ctx_params;
    if (!ctx.model_path.empty()) {
        return fs::u8path(ctx.model_path);
    }
    if (!ctx.diffusion_model_path.empty()) {
        return fs::u8path(ctx.diffusion_model_path);
    }
    return {};
}

static std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

static enum sample_method_t get_sdapi_sample_method(std::string name) {
    enum sample_method_t result = str_to_sample_method(name.c_str());
    if (result != SAMPLE_METHOD_COUNT) {
        return result;
    }

    name = lower_ascii(name);
    static const std::unordered_map<std::string_view, sample_method_t> hardcoded{
        {"euler a", EULER_A_SAMPLE_METHOD},
        {"k_euler_a", EULER_A_SAMPLE_METHOD},
        {"euler", EULER_SAMPLE_METHOD},
        {"k_euler", EULER_SAMPLE_METHOD},
        {"heun", HEUN_SAMPLE_METHOD},
        {"k_heun", HEUN_SAMPLE_METHOD},
        {"dpm2", DPM2_SAMPLE_METHOD},
        {"k_dpm_2", DPM2_SAMPLE_METHOD},
        {"lcm", LCM_SAMPLE_METHOD},
        {"ddim", DDIM_TRAILING_SAMPLE_METHOD},
        {"dpm++ 2m", DPMPP2M_SAMPLE_METHOD},
        {"k_dpmpp_2m", DPMPP2M_SAMPLE_METHOD},
        {"dpm++ 2m sde", DPMPP2M_SDE_SAMPLE_METHOD},
        {"k_dpmpp_2m_sde", DPMPP2M_SDE_SAMPLE_METHOD},
        {"dpm++ 2m sde gpu", DPMPP2M_SDE_BT_SAMPLE_METHOD},
        {"k_dpmpp_2m_sde_gpu", DPMPP2M_SDE_BT_SAMPLE_METHOD},
        {"res multistep", RES_MULTISTEP_SAMPLE_METHOD},
        {"k_res_multistep", RES_MULTISTEP_SAMPLE_METHOD},
        {"res 2s", RES_2S_SAMPLE_METHOD},
        {"k_res_2s", RES_2S_SAMPLE_METHOD},
        {"euler_cfg_pp", EULER_CFG_PP_SAMPLE_METHOD},
        {"k_euler_cfg_pp", EULER_CFG_PP_SAMPLE_METHOD},
        {"euler_a_cfg_pp", EULER_CFG_PP_SAMPLE_METHOD},
        {"k_euler_a_cfg_pp", EULER_CFG_PP_SAMPLE_METHOD},
    };
    auto it = hardcoded.find(name);
    return it != hardcoded.end() ? it->second : SAMPLE_METHOD_COUNT;
}

static bool parse_sdapi_adetailer(const json& body,
                                  ServerRuntime& runtime,
                                  bool img2img,
                                  ImgGenJobRequest& request,
                                  std::string& error_message) {
    request.adetailer_enabled = false;
    request.adetailer_passes.clear();
    if (!body.contains("alwayson_scripts")) {
        return true;
    }
    try {
        const auto& scripts = body.at("alwayson_scripts");
        if (!scripts.is_object()) {
            throw std::invalid_argument("alwayson_scripts must be an object");
        }
        if (!scripts.contains("ADetailer")) {
            return true;
        }
        const auto& args = scripts.at("ADetailer").at("args");
        if (!args.is_array()) {
            throw std::invalid_argument("ADetailer.args must be an array");
        }
        if (args.empty() || (args[0].is_boolean() && !args[0].get<bool>())) {
            return true;
        }
        static const std::unordered_set<std::string> keys = {
            "ad_model", "ad_model_classes", "ad_tab_enable", "ad_prompt", "ad_negative_prompt",
            "ad_confidence", "ad_mask_filter_method", "ad_mask_k", "ad_mask_min_ratio", "ad_mask_max_ratio",
            "ad_dilate_erode", "ad_x_offset", "ad_y_offset", "ad_mask_merge_invert", "ad_mask_blur",
            "ad_denoising_strength", "ad_inpaint_only_masked", "ad_inpaint_only_masked_padding",
            "ad_use_inpaint_width_height", "ad_inpaint_width", "ad_inpaint_height", "ad_use_steps", "ad_steps",
            "ad_use_cfg_scale", "ad_cfg_scale", "ad_use_sampler", "ad_sampler", "ad_scheduler",
            "ad_use_checkpoint", "ad_checkpoint", "ad_use_vae", "ad_vae", "ad_use_noise_multiplier",
            "ad_noise_multiplier", "ad_use_clip_skip", "ad_clip_skip", "ad_restore_face", "ad_controlnet_model",
            "ad_controlnet_module", "ad_controlnet_weight", "ad_controlnet_guidance_start", "ad_controlnet_guidance_end", "is_api"};
        for (const auto& config : args) {
            if (!config.is_object()) {
                continue;
            }
            auto boolean = [&](const char* key, bool fallback) {
                if (!config.contains(key))
                    return fallback;
                if (!config.at(key).is_boolean())
                    throw std::invalid_argument(std::string(key) + " must be boolean");
                return config.at(key).get<bool>();
            };
            auto text = [&](const char* key, const char* fallback) {
                if (!config.contains(key))
                    return std::string(fallback);
                if (!config.at(key).is_string())
                    throw std::invalid_argument(std::string(key) + " must be a string");
                return config.at(key).get<std::string>();
            };
            auto number = [&](const char* key, double fallback, double minimum, double maximum) {
                double value = fallback;
                if (config.contains(key)) {
                    if (!config.at(key).is_number())
                        throw std::invalid_argument(std::string(key) + " must be numeric");
                    value = config.at(key).get<double>();
                }
                if (!std::isfinite(value) || value < minimum || value > maximum)
                    throw std::invalid_argument(std::string(key) + " is out of range");
                return value;
            };
            auto integer = [&](const char* key, int fallback, int minimum, int maximum = std::numeric_limits<int>::max()) {
                if (config.contains(key) && !config.at(key).is_number_integer())
                    throw std::invalid_argument(std::string(key) + " must be an integer");
                return static_cast<int>(number(key, fallback, minimum, maximum));
            };
            const std::string model = text("ad_model", "None");
            if (model == "None" || !boolean("ad_tab_enable", true)) {
                continue;
            }
            for (const auto& item : config.items()) {
                if (keys.count(item.key()) == 0)
                    throw std::invalid_argument("unsupported ADetailer setting: " + item.key());
            }
            const auto& configured = runtime.default_gen_params->ad_model_path;
            if (configured.empty())
                throw std::invalid_argument("ADetailer requires a server-configured --ad-model");
            const auto path = fs::u8path(configured);
            if (model.find_first_of("/\\") != std::string::npos ||
                (lower_ascii(model) != lower_ascii(path.filename().u8string()) && lower_ascii(model) != lower_ascii(path.stem().u8string() + ".pt"))) {
                throw std::invalid_argument("ADetailer model must match the server-configured detector");
            }
            if (!text("ad_model_classes", "").empty())
                throw std::invalid_argument("ad_model_classes is not supported");
            if (boolean("ad_use_checkpoint", false)) {
                const auto checkpoint = config.contains("ad_checkpoint") && config.at("ad_checkpoint").is_null() ? "" : text("ad_checkpoint", "Use same checkpoint");
                if (!checkpoint.empty() && checkpoint != "None" && checkpoint != "Use same checkpoint")
                    throw std::invalid_argument("ADetailer checkpoint switching is not supported");
            }
            if (boolean("ad_use_vae", false)) {
                const auto vae = config.contains("ad_vae") && config.at("ad_vae").is_null() ? "" : text("ad_vae", "Use same VAE");
                if (!vae.empty() && vae != "None" && vae != "Use same VAE")
                    throw std::invalid_argument("ADetailer VAE switching is not supported");
            }
            if (boolean("ad_use_noise_multiplier", false) && number("ad_noise_multiplier", 1, 0.5, 1.5) != 1)
                throw std::invalid_argument("ADetailer noise multiplier is not supported");
            if (boolean("ad_restore_face", false))
                throw std::invalid_argument("ADetailer face restoration is not supported");
            if (text("ad_controlnet_model", "None") != "None")
                throw std::invalid_argument("ADetailer ControlNet is not supported");

            ServerADetailerPass pass;
            pass.prompt          = text("ad_prompt", "");
            pass.negative_prompt = text("ad_negative_prompt", "");
            pass.width           = request.gen_params.get_resolved_width();
            pass.height          = request.gen_params.get_resolved_height();
            pass.strength        = static_cast<float>(number("ad_denoising_strength", 0.4, 0, 1));
            auto append          = [&](const char* key, const std::string& value) {
                if (!pass.extra_args.empty())
                    pass.extra_args += ",";
                pass.extra_args += std::string(key) + "=" + value;
            };
            auto append_number = [&](const char* target, const char* key, double fallback, double minimum, double maximum) {
                append(target, json(number(key, fallback, minimum, maximum)).dump());
            };
            auto append_integer = [&](const char* target, const char* key, int fallback, int minimum, int maximum = std::numeric_limits<int>::max()) {
                append(target, std::to_string(integer(key, fallback, minimum, maximum)));
            };
            append_number("confidence", "ad_confidence", 0.3, 0, 1);
            append_integer("mask_k_largest", "ad_mask_k", 0, 0);
            const auto filter = text("ad_mask_filter_method", "Area");
            if (filter != "Area" && filter != "Confidence")
                throw std::invalid_argument("invalid ad_mask_filter_method");
            append("mask_filter_method", lower_ascii(filter));
            const auto min_ratio = number("ad_mask_min_ratio", 0, 0, 1);
            const auto max_ratio = number("ad_mask_max_ratio", 1, 0, 1);
            if (min_ratio > max_ratio)
                throw std::invalid_argument("ad_mask_min_ratio exceeds ad_mask_max_ratio");
            append("mask_min_ratio", json(min_ratio).dump());
            append("mask_max_ratio", json(max_ratio).dump());
            // Bound pixel operations before converting them to the native mask parameters.
            const int mask_limit = std::clamp(std::max(pass.width, pass.height), 32, 16384);
            append_integer("dilate_erode", "ad_dilate_erode", 4, -mask_limit, mask_limit);
            append_integer("x_offset", "ad_x_offset", 0, -mask_limit, mask_limit);
            append_integer("y_offset", "ad_y_offset", 0, -mask_limit, mask_limit);
            append_integer("mask_blur", "ad_mask_blur", 4, 0, mask_limit);
            append_integer("inpaint_padding", "ad_inpaint_only_masked_padding", 32, 0, mask_limit);
            append("inpaint_only_masked", boolean("ad_inpaint_only_masked", true) ? "true" : "false");
            const auto merge = text("ad_mask_merge_invert", "None");
            if (merge != "None" && merge != "Merge" && merge != "Merge and Invert")
                throw std::invalid_argument("invalid ad_mask_merge_invert");
            append("mask_mode", merge == "None" ? "none" : merge == "Merge" ? "merge"
                                                                            : "merge_invert");
            if (boolean("ad_use_inpaint_width_height", false)) {
                pass.width  = integer("ad_inpaint_width", 512, 1);
                pass.height = integer("ad_inpaint_height", 512, 1);
            }
            if (boolean("ad_use_steps", false))
                append_integer("steps", "ad_steps", 28, 1);
            if (boolean("ad_use_cfg_scale", false))
                append_number("cfg_scale", "ad_cfg_scale", 7, 0, std::numeric_limits<float>::max());
            if (boolean("ad_use_clip_skip", false)) {
                pass.clip_skip = integer("ad_clip_skip", 1, 1);
                if (pass.clip_skip > 12)
                    throw std::invalid_argument("ad_clip_skip is out of range");
            }
            if (boolean("ad_use_sampler", false)) {
                std::string sampler = text("ad_sampler", "DPM++ 2M Karras");
                if (sampler != "Use same sampler") {
                    const std::string suffix = " karras";
                    const bool legacy_karras = sampler.size() > suffix.size() && lower_ascii(sampler.substr(sampler.size() - suffix.size())) == suffix;
                    if (legacy_karras)
                        sampler.resize(sampler.size() - suffix.size());
                    const auto method = get_sdapi_sample_method(sampler);
                    if (method == SAMPLE_METHOD_COUNT)
                        throw std::invalid_argument("unsupported ADetailer sampler");
                    append("sample_method", sd_sample_method_name(method));
                    if (legacy_karras && text("ad_scheduler", "Use same scheduler") == "Use same scheduler")
                        append("scheduler", sd_scheduler_name(KARRAS_SCHEDULER));
                }
                const auto scheduler = text("ad_scheduler", "Use same scheduler");
                if (scheduler == "Automatic") {
                    pass.automatic_scheduler = true;
                } else if (scheduler != "Use same scheduler") {
                    const auto value = str_to_scheduler(scheduler.c_str());
                    if (value == SCHEDULER_COUNT)
                        throw std::invalid_argument("unsupported ADetailer scheduler");
                    append("scheduler", sd_scheduler_name(value));
                }
            }
            request.adetailer_passes.push_back(std::move(pass));
        }
        if (!request.adetailer_passes.empty() && img2img && args.size() > 1 && args[1].is_boolean() && args[1].get<bool>()) {
            throw std::invalid_argument("ADetailer skip img2img is not supported");
        }
        request.adetailer_enabled = !request.adetailer_passes.empty();
        return true;
    } catch (const std::exception& error) {
        error_message = std::string("invalid ADetailer args: ") + error.what();
        return false;
    }
}

static bool build_sdapi_img_gen_request(const json& j,
                                        ServerRuntime& runtime,
                                        bool img2img,
                                        ImgGenJobRequest& request,
                                        std::string& error_message) {
    std::string prompt          = j.value("prompt", "");
    std::string negative_prompt = j.value("negative_prompt", "");
    int width                   = j.value("width", 512);
    int height                  = j.value("height", 512);
    int steps                   = j.value("steps", runtime.default_gen_params->sample_params.sample_steps);
    float cfg_scale             = j.value("cfg_scale", runtime.default_gen_params->sample_params.guidance.txt_cfg);
    int64_t seed                = j.value("seed", -1);
    int batch_size              = j.value("batch_size", 1);
    int clip_skip               = j.value("clip_skip", -1);
    std::string sampler_name    = j.value("sampler_name", "");
    std::string scheduler_name  = j.value("scheduler", "");

    if (width <= 0 || height <= 0) {
        error_message = "width and height must be positive";
        return false;
    }

    if (prompt.empty()) {
        error_message = "prompt required";
        return false;
    }

    request.gen_params = *runtime.default_gen_params;

    request.gen_params.prompt                         = prompt;
    request.gen_params.negative_prompt                = negative_prompt;
    request.gen_params.seed                           = seed;
    request.gen_params.sample_params.sample_steps     = steps;
    request.gen_params.batch_count                    = batch_size;
    request.gen_params.sample_params.guidance.txt_cfg = cfg_scale;
    const bool recipe_dimensions                      = !img2img && model_status(runtime)["recipe"].is_string();
    request.gen_params.width                          = j.value("width", recipe_dimensions ? runtime.default_gen_params->width : -1);
    request.gen_params.height                         = j.value("height", recipe_dimensions ? runtime.default_gen_params->height : -1);

    if (!img2img && j.value("enable_hr", false)) {
        request.gen_params.hires_enabled = true;
        request.gen_params.hires_scale   = j.value("hr_scale", request.gen_params.hires_scale);
        request.gen_params.hires_width   = j.value("hr_resize_x", request.gen_params.hires_width);
        request.gen_params.hires_height  = j.value("hr_resize_y", request.gen_params.hires_height);
        request.gen_params.hires_steps   = j.value("hr_steps", request.gen_params.hires_steps);
        request.gen_params.hires_denoising_strength =
            j.value("denoising_strength", request.gen_params.hires_denoising_strength);

        request.gen_params.hires_upscaler = j.value("hr_upscaler", request.gen_params.hires_upscaler);
    }

    std::string sd_cpp_extra_args_str = extract_and_remove_sd_cpp_extra_args(request.gen_params.prompt);
    if (!sd_cpp_extra_args_str.empty() && !request.gen_params.from_json_str(sd_cpp_extra_args_str)) {
        error_message = "invalid sd_cpp_extra_args";
        return false;
    }

    if (clip_skip > 0) {
        request.gen_params.clip_skip = clip_skip;
    }

    enum sample_method_t sample_method = get_sdapi_sample_method(sampler_name);
    if (sample_method != SAMPLE_METHOD_COUNT) {
        request.gen_params.sample_params.sample_method = sample_method;
    }

    enum scheduler_t scheduler = str_to_scheduler(scheduler_name.c_str());
    if (scheduler != SCHEDULER_COUNT) {
        request.gen_params.sample_params.scheduler = scheduler;
    }

    if (j.contains("lora") && j["lora"].is_array()) {
        refresh_lora_cache(runtime);
        request.gen_params.lora_map.clear();
        request.gen_params.high_noise_lora_map.clear();

        for (const auto& item : j["lora"]) {
            if (!item.is_object()) {
                continue;
            }

            std::string path   = item.value("path", "");
            float multiplier   = item.value("multiplier", 1.0f);
            bool is_high_noise = item.value("is_high_noise", false);

            if (path.empty()) {
                error_message = "lora.path required";
                return false;
            }

            std::string fullpath = get_lora_full_path(runtime, path);
            if (fullpath.empty()) {
                error_message = "invalid lora path: " + path;
                return false;
            }

            if (is_high_noise) {
                request.gen_params.high_noise_lora_map[fullpath] += multiplier;
            } else {
                request.gen_params.lora_map[fullpath] += multiplier;
            }
        }
    }

    if (j.contains("ref_image_args")) {
        if (!j["ref_image_args"].is_string()) {
            error_message = "ref_image_args must be a string";
            return false;
        }
        request.gen_params.ref_image_args = j["ref_image_args"].get<std::string>();
    }

    if (j.contains("image_preprocess") && !request.gen_params.parse_image_preprocess_json(j["image_preprocess"].dump())) {
        error_message = "invalid image_preprocess";
        return false;
    }

    if (img2img) {
        if (j.contains("init_images") && j["init_images"].is_array() && !j["init_images"].empty()) {
            if (decode_base64_image(j["init_images"][0].get<std::string>(),
                                    0,
                                    0,
                                    0,
                                    request.gen_params.init_image)) {
                const sd_image_t& image = request.gen_params.init_image.get();
                request.gen_params.set_width_and_height_if_unset(image.width, image.height);
            }
        }

        if (j.contains("mask") && j["mask"].is_string()) {
            if (decode_base64_image(j["mask"].get<std::string>(),
                                    1,
                                    0,
                                    0,
                                    request.gen_params.mask_image)) {
                const sd_image_t& image = request.gen_params.mask_image.get();
                request.gen_params.set_width_and_height_if_unset(image.width, image.height);
            }
            sd_image_t& mask_image      = request.gen_params.mask_image.get();
            bool inpainting_mask_invert = j.value("inpainting_mask_invert", 0) != 0;
            if (inpainting_mask_invert && mask_image.data != nullptr) {
                for (uint32_t i = 0; i < mask_image.width * mask_image.height; ++i) {
                    mask_image.data[i] = 255 - mask_image.data[i];
                }
            }
        } else {
            request.gen_params.mask_image.reset({0, 0, 1, nullptr});
        }

        float denoising_strength = j.value("denoising_strength", -1.f);
        if (denoising_strength >= 0.f) {
            request.gen_params.strength = std::min(denoising_strength, 1.0f);
        }
    }

    if (j.contains("extra_images") && j["extra_images"].is_array()) {
        for (const auto& extra_image : j["extra_images"]) {
            if (!extra_image.is_string()) {
                continue;
            }
            SDImageOwner image_owner;
            if (decode_base64_image(extra_image.get<std::string>(),
                                    0,
                                    0, 0,
                                    image_owner)) {
                const sd_image_t& image = image_owner.get();
                request.gen_params.set_width_and_height_if_unset(image.width, image.height);
                request.gen_params.ref_images.push_back(std::move(image_owner));
            }
        }
    }

    // Intentionally disable prompt-embedded LoRA tag parsing for server APIs.
    if (!request.gen_params.resolve_and_validate(IMG_GEN, "", runtime.ctx_params->hires_upscalers_dir, true)) {
        error_message = "invalid params";
        return false;
    }

    return parse_sdapi_adetailer(j, runtime, img2img, request, error_message);
}

static nlohmann::json prepare_info_field(const SDContextParams& ctx_params,
                                         const SDGenerationParams& gen_params,
                                         bool img2img) {
    nlohmann::json jsoninfo = nlohmann::json::object();
    jsoninfo["prompt"]      = gen_params.prompt;
    if (!gen_params.negative_prompt.empty()) {
        jsoninfo["negative_prompt"] = gen_params.negative_prompt;
    }
    jsoninfo["seed"]         = gen_params.seed;
    jsoninfo["cfg_scale"]    = gen_params.sample_params.guidance.txt_cfg;
    jsoninfo["width"]        = gen_params.get_resolved_width();
    jsoninfo["height"]       = gen_params.get_resolved_height();
    jsoninfo["steps"]        = gen_params.sample_params.sample_steps;
    jsoninfo["sampler_name"] = sd_sample_method_name(gen_params.sample_params.sample_method);
    if (gen_params.clip_skip != -1) {
        jsoninfo["clip_skip"] = gen_params.clip_skip;
    }
    if (gen_params.sample_params.scheduler != scheduler_t::SCHEDULER_COUNT) {
        jsoninfo["extra_generation_params"]                  = nlohmann::json::object();
        jsoninfo["extra_generation_params"]["Schedule type"] = sd_scheduler_name(gen_params.sample_params.scheduler);
    }
    if (img2img) {
        jsoninfo["denoising_strength"] = gen_params.strength;
    }
    // not clear what should happen if we have both model and diffusion_model
    if (!ctx_params.diffusion_model_path.empty()) {
        jsoninfo["sd_model_name"] = sd_basename(ctx_params.diffusion_model_path);
    } else if (!ctx_params.model_path.empty()) {
        jsoninfo["sd_model_name"] = sd_basename(ctx_params.model_path);
    }
    if (!ctx_params.vae_path.empty()) {
        jsoninfo["sd_vae_name"] = sd_basename(ctx_params.vae_path);
    }
    jsoninfo["version"] = "stable-diffusion.cpp";

    jsoninfo["infotexts"]            = nlohmann::json::array();
    jsoninfo["all_prompts"]          = nlohmann::json::array();
    jsoninfo["all_negative_prompts"] = nlohmann::json::array();
    jsoninfo["all_seeds"]            = nlohmann::json::array();
    return jsoninfo;
}

struct SDAPIPreviewState {
    std::mutex mutex;
    std::string b64;
    std::string id_task;
    std::string job_type;
    int step          = 0;
    int total         = 0;
    bool active       = false;
    int64_t job_count = 0;
    int job_no        = 0;
};

static SDAPIPreviewState g_preview;

static void preview_callback(int step, int frame_count, sd_image_t* frames, bool is_noisy, void* data) {
    (void)is_noisy;
    (void)data;
    if (frame_count <= 0 || frames == nullptr || frames[0].data == nullptr) {
        return;
    }

    const sd_image_t& frame      = frames[0];
    const sd_preview_info_t info = sd_get_preview_info();

    auto image_bytes = encode_image_to_vector(EncodedImageFormat::JPEG,
                                              frame.data,
                                              frame.width,
                                              frame.height,
                                              frame.channel,
                                              "",
                                              80);
    if (image_bytes.empty()) {
        return;
    }

    std::string b64 = base64_encode(image_bytes);
    std::lock_guard<std::mutex> lock(g_preview.mutex);
    g_preview.b64    = std::move(b64);
    g_preview.step   = static_cast<int>(std::min<int64_t>(step < 0 ? -static_cast<int64_t>(step) : step,
                                                        std::max(0, info.total_steps)));
    g_preview.total  = std::max(0, info.total_steps);
    g_preview.job_no = std::max(0, info.sample_pass - 1);
}

struct SDAPIPreviewGuard {
    SDAPIPreviewGuard(const std::string& id_task, const sd_img_gen_params_t& params, const std::string& job_type, bool detailing = false) {
        std::lock_guard<std::mutex> lock(g_preview.mutex);
        g_preview.id_task  = id_task;
        g_preview.total    = 0;
        g_preview.job_type = job_type;
        g_preview.b64.clear();
        g_preview.step   = 0;
        g_preview.active = true;
        // Each batch item has a base pass and, when enabled, a hires pass.
        g_preview.job_count = static_cast<int64_t>(std::max(1, params.batch_count)) * (params.hires.enabled ? 2 : 1) + (detailing ? 1 : 0);
        g_preview.job_no    = 0;
        sd_set_preview_callback(preview_callback, PREVIEW_PROJ, 1, true, false, nullptr);
    }

    void begin_detailing() {
        sd_set_preview_callback(nullptr, PREVIEW_NONE, 1, false, false, nullptr);
        std::lock_guard<std::mutex> lock(g_preview.mutex);
        g_preview.job_type = "adetailer";
        g_preview.job_no   = static_cast<int>(g_preview.job_count - 1);
        g_preview.step     = 0;
        g_preview.total    = 0;
    }

    ~SDAPIPreviewGuard() {
        sd_set_preview_callback(nullptr, PREVIEW_NONE, 1, false, false, nullptr);
        std::lock_guard<std::mutex> lock(g_preview.mutex);
        g_preview.active = false;
        g_preview.b64.clear();
    }

    SDAPIPreviewGuard(const SDAPIPreviewGuard&)            = delete;
    SDAPIPreviewGuard& operator=(const SDAPIPreviewGuard&) = delete;
};

void register_sdapi_endpoints(httplib::Server& svr, ServerRuntime& rt) {
    ServerRuntime* runtime = &rt;
    RuntimeRoutes routes(svr, rt);

    auto sdapi_any2img = [runtime](const httplib::Request& req, httplib::Response& res, bool img2img) {
        try {
            if (req.body.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"empty body"})", "application/json");
                return;
            }
            if (!runtime_supports_generation_mode(*runtime, IMG_GEN)) {
                res.status = 400;
                res.set_content(json({{"error", unsupported_generation_mode_error(IMG_GEN)}}).dump(), "application/json");
                return;
            }

            std::string id_task = "sdapi";
            json j              = json::parse(req.body);
            id_task             = j.value("id_task", id_task);
            ImgGenJobRequest request;
            std::string error_message;
            if (!build_sdapi_img_gen_request(j, *runtime, img2img, request, error_message)) {
                res.status = 400;
                res.set_content(json({{"error", error_message}}).dump(), "application/json");
                return;
            }

            LOG_DEBUG("%s\n", request.gen_params.to_string().c_str());

            sd_img_gen_params_t img_gen_params = request.to_sd_img_gen_params_t();
            SDImageVec results;
            int num_results = 0;

            {
                std::lock_guard<std::mutex> lock(*runtime->sd_ctx_mutex);
                const bool detailing = request.adetailer_enabled && !runtime->default_gen_params->ad_model_path.empty();
                SDAPIPreviewGuard preview_guard(id_task, img_gen_params, img2img ? "img2img" : "txt2img", detailing);
                sd_image_t* raw_results = nullptr;
                if (!generate_image(runtime->sd_ctx, &img_gen_params, &raw_results, &num_results)) {
                    raw_results = nullptr;
                    num_results = 0;
                }
                results.adopt(raw_results, num_results);
                if (detailing && !results.empty())
                    preview_guard.begin_detailing();
                if (!results.empty() && !apply_server_adetailer(*runtime, request, img_gen_params, results)) {
                    res.status = 500;
                    res.set_content(R"({"error":"ADetailer failed"})", "application/json");
                    return;
                }
            }

            if (results.empty()) {
                res.status = 500;
                res.set_content(R"({"error":"generate_image returned no results"})", "application/json");
                return;
            }

            json out;
            out["images"]     = json::array();
            out["parameters"] = j;
            json jsoninfo     = prepare_info_field(*runtime->ctx_params, request.gen_params, img2img);

            int images_per_batch = request.gen_params.batch_count > 0 ? std::max(1, num_results / request.gen_params.batch_count) : 1;
            for (int i = 0; i < num_results; ++i) {
                if (results[i].data == nullptr) {
                    continue;
                }

                bool embed_meta = request.gen_params.embed_image_metadata;

                std::string params = get_image_params(*runtime->ctx_params,
                                                      request.gen_params,
                                                      request.gen_params.seed + i / images_per_batch);

                auto image_bytes = encode_image_to_vector(EncodedImageFormat::PNG,
                                                          results[i].data,
                                                          results[i].width,
                                                          results[i].height,
                                                          results[i].channel,
                                                          embed_meta ? params : "");

                if (image_bytes.empty()) {
                    LOG_ERROR("write image to mem failed");
                    continue;
                }

                out["images"].push_back(base64_encode(image_bytes));

                jsoninfo["infotexts"][i]            = params;
                jsoninfo["all_seeds"][i]            = request.gen_params.seed + i;
                jsoninfo["all_prompts"][i]          = request.gen_params.prompt;
                jsoninfo["all_negative_prompts"][i] = request.gen_params.negative_prompt;
            }

            // not a mistake: it is supposed to be a string in json format
            out["info"] = jsoninfo.dump();

            res.set_content(out.dump(), "application/json");
            res.status = 200;

        } catch (const std::exception& e) {
            res.status = 500;
            json err;
            err["error"]   = "server_error";
            err["message"] = e.what();
            res.set_content(err.dump(), "application/json");
        }
    };

    routes.post("/sdapi/v1/txt2img", [sdapi_any2img](const httplib::Request& req, httplib::Response& res) {
        sdapi_any2img(req, res, false);
    });

    routes.post("/sdapi/v1/img2img", [sdapi_any2img](const httplib::Request& req, httplib::Response& res) {
        sdapi_any2img(req, res, true);
    });

    routes.get("/sdapi/v1/loras", [runtime](const httplib::Request&, httplib::Response& res) {
        refresh_lora_cache(*runtime);

        json result = json::array();
        {
            std::lock_guard<std::mutex> lock(*runtime->lora_mutex);
            for (const auto& e : *runtime->lora_cache) {
                json item;
                item["name"]         = e.name;
                item["path"]         = e.path;
                item["display_path"] = e.display_path;
                result.push_back(item);
            }
        }

        res.set_content(result.dump(), "application/json");
    });

    routes.get("/sdapi/v1/upscalers", [runtime](const httplib::Request&, httplib::Response& res) {
        refresh_upscaler_cache(*runtime);

        auto make_builtin = [](const char* name) {
            json item;
            item["name"]       = name;
            item["model_name"] = nullptr;
            item["model_path"] = nullptr;
            item["model_url"]  = nullptr;
            item["scale"]      = 4;
            return item;
        };

        json result = json::array();
        result.push_back(make_builtin("None"));
        result.push_back(make_builtin("Lanczos"));
        result.push_back(make_builtin("Nearest"));

        {
            std::lock_guard<std::mutex> lock(*runtime->upscaler_mutex);
            for (const auto& e : *runtime->upscaler_cache) {
                json item;
                item["name"]       = e.name;
                item["model_name"] = e.model_name;
                item["model_path"] = e.fullpath;
                item["model_url"]  = nullptr;
                item["scale"]      = e.scale;
                result.push_back(item);
            }
        }

        res.set_content(result.dump(), "application/json");
    });

    routes.get("/sdapi/v1/latent-upscale-modes", [](const httplib::Request&, httplib::Response& res) {
        json result = json::array({
            {{"name", "Latent"}},
            {{"name", "Latent (nearest)"}},
            {{"name", "Latent (nearest-exact)"}},
            {{"name", "Latent (antialiased)"}},
            {{"name", "Latent (bicubic)"}},
            {{"name", "Latent (bicubic antialiased)"}},
        });
        res.set_content(result.dump(), "application/json");
    });

    routes.get("/sdapi/v1/samplers", [runtime](const httplib::Request&, httplib::Response& res) {
        std::vector<std::string> sampler_names;
        sampler_names.push_back("default");
        for (int i = 0; i < SAMPLE_METHOD_COUNT; i++) {
            sampler_names.push_back(sd_sample_method_name((sample_method_t)i));
        }
        json r = json::array();
        for (auto name : sampler_names) {
            json entry;
            entry["name"]    = name;
            entry["aliases"] = json::array({name});
            entry["options"] = json::object();
            r.push_back(entry);
        }
        res.set_content(r.dump(), "application/json");
    });

    routes.get("/sdapi/v1/schedulers", [runtime](const httplib::Request&, httplib::Response& res) {
        std::vector<std::string> scheduler_names;
        scheduler_names.push_back("default");
        for (int i = 0; i < SCHEDULER_COUNT; i++) {
            scheduler_names.push_back(sd_scheduler_name((scheduler_t)i));
            if (i == DISCRETE_SCHEDULER) {
                scheduler_names.push_back("normal");
            }
        }
        json r = json::array();
        for (auto name : scheduler_names) {
            json entry;
            entry["name"]  = name;
            entry["label"] = name;
            r.push_back(entry);
        }
        res.set_content(r.dump(), "application/json");
    });

    routes.get("/sdapi/v1/sd-models", [runtime](const httplib::Request&, httplib::Response& res) {
        fs::path model_path = resolve_display_model_path(*runtime);
        json entry;
        entry["title"]      = model_path.stem().u8string();
        entry["model_name"] = model_path.stem().u8string();
        entry["filename"]   = model_path.filename().u8string();
        entry["hash"]       = "8888888888";
        entry["sha256"]     = "8888888888888888888888888888888888888888888888888888888888888888";
        entry["config"]     = nullptr;
        json r              = json::array();
        if (runtime->sd_ctx) {
            r.push_back(entry);
        }
        res.set_content(r.dump(), "application/json");
    });

    routes.get("/sdapi/v1/options", [runtime](const httplib::Request&, httplib::Response& res) {
        fs::path model_path = resolve_display_model_path(*runtime);
        json r;
        r["samples_format"]      = "png";
        r["sd_model_checkpoint"] = model_path.stem().u8string();
        res.set_content(r.dump(), "application/json");
    });

    routes.get("/sdapi/v1/progress", [](const httplib::Request& req, httplib::Response& res) {
        bool skip_current_image = false;
        bool has_id_param       = req.has_param("id_task");
        std::string req_id_task = has_id_param ? req.get_param_value("id_task") : "";

        if (req.has_param("skip_current_image")) {
            std::string skip_str = req.get_param_value("skip_current_image");
            skip_current_image   = (skip_str == "true" || skip_str == "1");
        }

        json r;
        r["eta_relative"] = 0.0f;
        json state;
        state["skipped"]       = false;
        state["interrupted"]   = false;
        state["job_timestamp"] = "0";
        r["textinfo"]          = nullptr;

        {
            std::lock_guard<std::mutex> lock(g_preview.mutex);

            bool is_matching_task = g_preview.active && (!has_id_param || req_id_task == g_preview.id_task);

            int step             = is_matching_task ? g_preview.step : 0;
            int steps            = is_matching_task ? g_preview.total : 0;
            std::string b64      = is_matching_task ? g_preview.b64 : "";
            std::string job_type = is_matching_task ? g_preview.job_type : "";
            int64_t job_count    = is_matching_task ? g_preview.job_count : 0;
            int job_no           = is_matching_task ? g_preview.job_no : 0;

            float batch_progress = (steps > 0) ? (static_cast<float>(step) / static_cast<float>(steps)) : 0.0f;
            if (batch_progress > 1.0f)
                batch_progress = 1.0f;

            float overall_progress = job_count > 0
                                         ? (static_cast<float>(job_no) + batch_progress) / static_cast<float>(job_count)
                                         : 0.0f;
            if (overall_progress > 1.0f)
                overall_progress = 1.0f;

            r["progress"]           = overall_progress;
            state["job"]            = job_type;
            state["sampling_step"]  = step;
            state["sampling_steps"] = steps;
            state["job_count"]      = job_count;
            state["job_no"]         = job_no;
            r["state"]              = state;

            if (!skip_current_image && !b64.empty()) {
                r["current_image"] = b64;
            } else {
                r["current_image"] = nullptr;
            }
        }

        res.set_content(r.dump(), "application/json");
    });
}
