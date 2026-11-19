#include "routes.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>

#include "async_jobs.h"
#include "common/common.h"
#include "common/media_io.h"
#include "common/resource_owners.hpp"
#include "model_catalog.h"

namespace fs = std::filesystem;

static constexpr uint32_t k_max_upscale_dimension = 8192;

static bool valid_upscale_dimensions(const SDImage& image, int factor, int repeats) {
    if (image.width == 0 || image.height == 0 || factor < 1 || repeats < 1 || repeats > 4) {
        return false;
    }
    uint32_t width  = image.width;
    uint32_t height = image.height;
    for (int i = 0; i < repeats; ++i) {
        if (width > k_max_upscale_dimension / factor || height > k_max_upscale_dimension / factor) {
            return false;
        }
        width *= factor;
        height *= factor;
    }
    return true;
}

static bool parse_cache_mode(const std::string& value, std::string& mode) {
    const std::string canonical = value == "cache-dit" ? "cache_dit" : value;
    SDOptPtr opt(sd_img_gen_opt_create());
    if (!opt || sd_opt_set_string(opt.get(), "cache.mode", canonical.c_str()) != SD_STATUS_OK)
        return false;
    mode = canonical;
    return true;
}

static json finite_number_or_null(const std::optional<float>& value) {
    return value && std::isfinite(*value) ? json(*value) : json(nullptr);
}

static json make_vae_tiling_json(const SDTilingParams& params) {
    return {
        {"enabled", params.enabled},
        {"temporal_tiling", params.temporal_tiling},
        {"tile_size_w", params.tile_size_w},
        {"tile_size_h", params.tile_size_h},
        {"target_overlap", params.target_overlap},
        {"rel_size_w", params.rel_size_w},
        {"rel_size_h", params.rel_size_h},
        {"extra_tiling_args", params.extra_tiling_args ? params.extra_tiling_args : ""},
    };
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

static json make_sample_params_json(const SDSampleParams& sample_params, const std::optional<std::vector<int>>& skip_layers, const std::optional<std::vector<float>>& custom_sigmas = {}) {
    const auto& guidance = sample_params.guidance;
    return {
        {"scheduler", sample_params.scheduler},
        {"sample_method", sample_params.sample_method},
        {"sample_steps", sample_params.sample_steps},
        {"custom_sigmas", custom_sigmas},
        {"eta", finite_number_or_null(sample_params.eta)},
        {"shifted_timestep", sample_params.shifted_timestep},
        {"flow_shift", finite_number_or_null(sample_params.flow_shift)},
        {"guidance",
         {
             {"txt_cfg", guidance.txt_cfg},
             {"img_cfg", finite_number_or_null(guidance.img_cfg)},
             {"distilled_guidance", guidance.distilled_guidance},
             {"slg",
              {
                  {"layers", skip_layers},
                  {"layer_start", guidance.slg.layer_start},
                  {"layer_end", guidance.slg.layer_end},
                  {"scale", guidance.slg.scale},
              }},
         }},
    };
}

static json make_hires_json(const SDGenerationParams& defaults) {
    return {
        {"enabled", defaults.hires_enabled},
        {"upscaler", defaults.hires_upscaler},
        {"scale", defaults.hires_scale},
        {"target_width", defaults.hires_width},
        {"target_height", defaults.hires_height},
        {"steps", defaults.hires_steps},
        {"denoising_strength", defaults.hires_denoising_strength},
        {"custom_sigmas", defaults.hires_custom_sigmas},
        {"upscale_tile_size", defaults.hires_upscale_tile_size},
    };
}

static json make_img_gen_defaults_json(const SDGenerationParams& defaults, const std::string& output_format) {
    return {
        {"prompt", defaults.prompt},
        {"negative_prompt", defaults.negative_prompt},
        {"clip_skip", defaults.clip_skip},
        {"width", defaults.width},
        {"height", defaults.height},
        {"strength", defaults.strength},
        {"seed", defaults.seed},
        {"batch_count", defaults.batch_count},
        {"qwen_image_layers", defaults.qwen_image_layers},
        {"ref_image_args", defaults.ref_image_args},
        {"image_preprocess", defaults.image_preprocess},
        {"increase_ref_index", defaults.increase_ref_index},
        {"control_strength", defaults.control_strength},
        {"control_type", defaults.control_type},
        {"ip_adapter_strength", defaults.ip_adapter_strength},
        {"sample_params", make_sample_params_json(defaults.sample_params, defaults.skip_layers, defaults.custom_sigmas)},
        {"hires", make_hires_json(defaults)},
        {"vae_tiling_params", make_vae_tiling_json(defaults.vae_tiling_params)},
        {"cache_mode", defaults.cache_mode},
        {"cache_option", defaults.cache_option},
        {"scm_mask", defaults.scm_mask},
        {"scm_policy_dynamic", defaults.scm_policy_dynamic},
        {"output_format", output_format},
        {"output_compression", 100},
    };
}

static json make_vid_gen_defaults_json(const SDGenerationParams& defaults, const std::string& output_format) {
    return {
        {"prompt", defaults.prompt},
        {"negative_prompt", defaults.negative_prompt},
        {"clip_skip", defaults.clip_skip},
        {"width", defaults.width},
        {"height", defaults.height},
        {"strength", defaults.strength},
        {"seed", defaults.seed},
        {"video_frames", defaults.video_frames},
        {"image_preprocess", defaults.image_preprocess},
        {"ref_image_args", defaults.ref_image_args},
        {"fps", defaults.fps},
        {"moe_boundary", defaults.moe_boundary},
        {"vace_strength", defaults.vace_strength},
        {"sample_params", make_sample_params_json(defaults.sample_params, defaults.skip_layers, defaults.custom_sigmas)},
        {"high_noise_sample_params", make_sample_params_json(defaults.high_noise_sample_params, defaults.high_noise_skip_layers)},
        {"hires", make_hires_json(defaults)},
        {"vae_tiling_params", make_vae_tiling_json(defaults.vae_tiling_params)},
        {"cache_mode", defaults.cache_mode},
        {"cache_option", defaults.cache_option},
        {"scm_mask", defaults.scm_mask},
        {"scm_policy_dynamic", defaults.scm_policy_dynamic},
        {"output_format", output_format},
        {"output_compression", 100},
    };
}

static json make_img_gen_features_json() {
    return {
        {"init_image", true},
        {"mask_image", true},
        {"control_image", true},
        {"ip_adapter_image", true},
        {"ref_images", true},
        {"lora", true},
        {"vae_tiling", true},
        {"hires", true},
        {"cache", true},
        {"cancel_queued", true},
        {"cancel_generating", true},
        {"preview", true},
    };
}

/** Builds the default feature set advertised for vid_gen requests. */
static json make_vid_gen_features_json() {
    return {
        {"init_image", true},
        {"end_image", true},
        {"control_frames", true},
        {"ref_videos", true},
        {"ref_audios", true},
        {"ref_video_audios", true},
        {"high_noise_sample_params", true},
        {"lora", true},
        {"vae_tiling", true},
        {"cache", true},
        {"cancel_queued", true},
        {"cancel_generating", true},
        {"preview", true},
    };
}

static json make_preview_modes_json() {
    json modes = json::array();
    for (const char* name : example_options::preview_modes) {
        modes.push_back(name);
    }
    return modes;
}

/** Builds the /capabilities payload: model, defaults, limits and feature maps. */
static json make_capabilities_json(ServerRuntime& runtime) {
    refresh_lora_cache(runtime);
    refresh_upscaler_cache(runtime);

    AsyncJobManager& manager  = *runtime.async_job_manager;
    const auto& defaults      = *runtime.default_gen_params;
    const fs::path model_path = resolve_display_model_path(runtime);
    const bool supports_img   = runtime_supports_generation_mode(runtime, IMG_GEN);
    const bool supports_vid   = runtime_supports_generation_mode(runtime, VID_GEN);
    json samplers             = json::array();
    json schedulers           = json::array();
    json image_output_formats = supported_img_output_formats();
    json video_output_formats = supported_vid_output_formats();
    json available_loras      = json::array();
    json available_upscalers  = json::array();
    json supported_modes      = json::array();

    for (const char* name : example_options::sample_methods) {
        samplers.push_back(name);
    }

    for (const std::string name : example_options::schedulers) {
        schedulers.push_back(name);
        if (name == "discrete") {
            schedulers.push_back("normal");
        }
    }

    {
        std::lock_guard<std::mutex> lock(*runtime.lora_mutex);
        for (const auto& entry : *runtime.lora_cache) {
            available_loras.push_back({
                {"name", entry.name},
                {"path", entry.path},
                {"display_path", entry.display_path},
            });
        }
    }

    available_upscalers.push_back({
        {"name", "None"},
        {"model", false},
        {"image_upscale", false},
    });
    available_upscalers.push_back({
        {"name", "Lanczos"},
        {"model", false},
        {"image_upscale", false},
    });
    available_upscalers.push_back({
        {"name", "Nearest"},
        {"model", false},
        {"image_upscale", false},
    });
    available_upscalers.push_back({
        {"name", "Latent"},
        {"model", false},
        {"image_upscale", false},
    });
    available_upscalers.push_back({
        {"name", "Latent (nearest)"},
        {"model", false},
        {"image_upscale", false},
    });
    available_upscalers.push_back({
        {"name", "Latent (nearest-exact)"},
        {"model", false},
        {"image_upscale", false},
    });
    available_upscalers.push_back({
        {"name", "Latent (antialiased)"},
        {"model", false},
        {"image_upscale", false},
    });
    available_upscalers.push_back({
        {"name", "Latent (bicubic)"},
        {"model", false},
        {"image_upscale", false},
    });
    available_upscalers.push_back({
        {"name", "Latent (bicubic antialiased)"},
        {"model", false},
        {"image_upscale", false},
    });
    bool have_upscaler_models = false;
    {
        std::lock_guard<std::mutex> lock(*runtime.upscaler_mutex);
        for (const auto& entry : *runtime.upscaler_cache) {
            available_upscalers.push_back({
                {"name", entry.name},
                {"model", true},
                {"image_upscale", entry.image_upscale_factor > 0},
            });
            have_upscaler_models = have_upscaler_models || entry.image_upscale_factor > 0;
        }
    }

    if (supports_img) {
        supported_modes.push_back("img_gen");
    }
    if (supports_vid) {
        supported_modes.push_back("vid_gen");
    }

    std::string default_img_output_format = "png";
    std::string default_vid_output_format = "avi";
    if (!image_output_formats.empty()) {
        default_img_output_format = image_output_formats[0].get<std::string>();
    }
    if (!video_output_formats.empty()) {
        default_vid_output_format = video_output_formats[0].get<std::string>();
    }

    const json generation_features = model_generation_features(runtime);
    auto recipe_features           = [&](json features, bool video) {
        features["init_image"] = generation_features.at(video ? "init_image" : "img2img");
        for (const char* key : {"mask_image", "ref_images", "end_image", "control_frames", "ref_videos", "ref_audios", "ref_video_audios", "lora", "vae_tiling", "cache", "preview"}) {
            if (features.contains(key) || std::string(key) == "ref_images" ||
                (video && (std::string(key) == "ref_videos" || std::string(key) == "ref_audios" ||
                           std::string(key) == "ref_video_audios")))
                features[key] = generation_features.at(key);
        }
        if (video)
            features["high_noise_sample_params"] = generation_features.at("high_noise");
        else {
            features["mask_image"]       = generation_features.at("mask_image").get<bool>() && generation_features.at("img2img").get<bool>();
            features["control_image"]    = generation_features.at("controlnet").get<bool>() || generation_features.at("control_image").get<bool>();
            features["ip_adapter_image"] = generation_features.at("ip_adapter");
        }
        return features;
    };
    json defaults_by_mode       = json::object();
    json output_formats_by_mode = json::object();
    json features_by_mode       = json::object();
    if (supports_img) {
        defaults_by_mode["img_gen"]       = make_img_gen_defaults_json(defaults, default_img_output_format);
        output_formats_by_mode["img_gen"] = image_output_formats;
        features_by_mode["img_gen"]       = recipe_features(make_img_gen_features_json(), false);
    }
    if (supports_vid) {
        defaults_by_mode["vid_gen"]       = make_vid_gen_defaults_json(defaults, default_vid_output_format);
        output_formats_by_mode["vid_gen"] = video_output_formats;
        features_by_mode["vid_gen"]       = recipe_features(make_vid_gen_features_json(), true);
    }

    json top_level_defaults       = json::object();
    json top_level_output_formats = json::array();
    json top_level_features       = {
        {"cancel_queued", true},
        {"cancel_generating", true},
    };
    std::string current_mode = "";
    if (supports_img) {
        current_mode             = "img_gen";
        top_level_defaults       = defaults_by_mode["img_gen"];
        top_level_output_formats = output_formats_by_mode["img_gen"];
        top_level_features       = features_by_mode["img_gen"];
    } else if (supports_vid) {
        current_mode             = "vid_gen";
        top_level_defaults       = defaults_by_mode["vid_gen"];
        top_level_output_formats = output_formats_by_mode["vid_gen"];
        top_level_features       = features_by_mode["vid_gen"];
    }

    json result;
    result["model"] = {
        {"name", model_path.filename().u8string()},
        {"stem", model_path.stem().u8string()},
        {"path", model_path.u8string()},
    };
    result["generation_features"] = generation_features;
    result["runtime"]             = model_status(runtime);
    result["current_mode"]        = current_mode;
    result["supported_modes"]     = supported_modes;
    result["defaults"]            = top_level_defaults;
    result["defaults_by_mode"]    = defaults_by_mode;
    result["limits"]              = {
        {"min_width", 64},
        {"max_width", k_max_generation_dimension},
        {"min_height", 64},
        {"max_height", k_max_generation_dimension},
        {"max_batch_count", 8},
        {"max_queue_size", manager.max_pending_jobs},
        {"max_upscale_width", k_max_upscale_dimension},
        {"max_upscale_height", k_max_upscale_dimension},
    };
    result["samplers"]               = samplers;
    result["schedulers"]             = schedulers;
    result["output_formats"]         = top_level_output_formats;
    result["output_formats_by_mode"] = output_formats_by_mode;
    result["features"]               = top_level_features;
    result["features_by_mode"]       = features_by_mode;
    result["loras"]                  = available_loras;
    result["upscalers"]              = available_upscalers;
    result["upscale"]                = have_upscaler_models;
    result["preview_modes"]          = make_preview_modes_json();
    return result;
}

static bool parse_img_gen_request(const json& body,
                                  ServerRuntime& runtime,
                                  ImgGenJobRequest& request,
                                  std::string& error_message) {
    request.gen_params = *runtime.default_gen_params;

    refresh_lora_cache(runtime);
    if (!request.gen_params.from_json_str(body.dump(), [&](const std::string& path) {
            return get_lora_full_path(runtime, path);
        })) {
        error_message = "invalid generation parameters";
        return false;
    }

    std::string output_format = body.value("output_format", "png");
    int output_compression    = body.value("output_compression", 100);
    if (!assign_output_options(request, output_format, output_compression, true, error_message)) {
        return false;
    }
    request.preview_mode     = body.value("preview", std::string("none"));
    request.preview_interval = body.value("preview_interval", 1);
    // Intentionally disable prompt-embedded LoRA tag parsing for server APIs.
    if (!request.gen_params.resolve_and_validate(IMG_GEN, "", runtime.ctx_params->hires_upscalers_dir, true)) {
        error_message = "invalid generation parameters";
        return false;
    }
    return validate_server_img_gen_request(runtime, request, error_message);
}

static bool parse_vid_gen_request(const json& body,
                                  ServerRuntime& runtime,
                                  VidGenJobRequest& request,
                                  std::string& error_message) {
    request.gen_params = *runtime.default_gen_params;

    refresh_lora_cache(runtime);
    if (!request.gen_params.from_json_str(body.dump(), [&](const std::string& path) {
            return get_lora_full_path(runtime, path);
        })) {
        error_message = "invalid generation parameters";
        return false;
    }

    auto& params                   = request.gen_params;
    auto merge_shared_sample_input = [&](const char* key, auto member) {
        const auto base = body.find("sample_params");
        const auto high = body.find("high_noise_sample_params");
        if (base != body.end() && base->is_object() && base->contains(key)) {
            if (base->at(key).is_null())
                (params.high_noise_sample_params.*member).reset();
        } else if (high != body.end() && high->is_object() && high->contains(key)) {
            params.sample_params.*member = params.high_noise_sample_params.*member;
        }
    };
    merge_shared_sample_input("scheduler", &SDSampleParams::scheduler);
    merge_shared_sample_input("flow_shift", &SDSampleParams::flow_shift);
    merge_shared_sample_input("shifted_timestep", &SDSampleParams::shifted_timestep);

    std::string output_format = body.value("output_format", "webm");
    int output_compression    = body.value("output_compression", 100);
    if (!assign_output_options(request, output_format, output_compression, error_message)) {
        return false;
    }
    request.preview_mode     = body.value("preview", std::string("none"));
    request.preview_interval = body.value("preview_interval", 1);
    // Intentionally disable prompt-embedded LoRA tag parsing for server APIs.
    if (!request.gen_params.resolve_and_validate(VID_GEN, "", runtime.ctx_params->hires_upscalers_dir, true)) {
        error_message = "invalid generation parameters";
        return false;
    }
    return true;
}

void register_sdcpp_api_endpoints(httplib::Server& svr, ServerRuntime& rt) {
    ServerRuntime* runtime = &rt;
    RuntimeRoutes routes(svr, rt);

    routes.get("/sdcpp/v1/capabilities", [runtime](const httplib::Request&, httplib::Response& res) {
        res.status = 200;
        res.set_content(make_capabilities_json(*runtime).dump(), "application/json");
    });

    routes.post("/sdcpp/v1/upscale", [runtime](const httplib::Request& req, httplib::Response& res) {
        try {
            if (req.body.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"empty body"})", "application/json");
                return;
            }
            json body = json::parse(req.body);
            if (!body.is_object()) {
                res.status = 400;
                res.set_content(R"({"error":"body must be an object"})", "application/json");
                return;
            }
            for (const char* key : {"repeats", "tile_size", "output_compression"}) {
                if (!body.contains(key)) {
                    continue;
                }
                const auto& value = body[key];
                const bool valid  = value.is_number_unsigned()
                                        ? value.get<uint64_t>() <= static_cast<uint64_t>(std::numeric_limits<int>::max())
                                        : value.is_number_integer() && value.get<int64_t>() >= std::numeric_limits<int>::min() &&
                                              value.get<int64_t>() <= std::numeric_limits<int>::max();
                if (!valid) {
                    res.status = 400;
                    res.set_content(json({{"error", std::string(key) + " must be a 32-bit integer"}}).dump(), "application/json");
                    return;
                }
            }
            ImgGenJobRequest output_options;
            std::string error_message;
            if (!assign_output_options(output_options,
                                       body.value("output_format", std::string("png")),
                                       body.value("output_compression", 100),
                                       true,
                                       error_message)) {
                res.status = 400;
                res.set_content(json({{"error", error_message}}).dump(), "application/json");
                return;
            }
            const auto tile_size = body.value("tile_size", runtime->default_gen_params->upscale_tile_size);
            if (tile_size && *tile_size < 32) {
                res.status = 400;
                res.set_content(json({{"error", "tile_size must be at least 32"}}).dump(), "application/json");
                return;
            }
            const int repeats        = std::clamp(body.value("repeats", 1), 1, 4);
            const std::string wanted = body.value("upscaler", std::string());

            const std::string encoded = body.value("image", std::string());
            if (encoded.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"image is required"})", "application/json");
                return;
            }
            SDImageOwner input;
            if (!decode_base64_image(encoded, 3, 0, 0, input) || input.get().data == nullptr) {
                res.status = 400;
                res.set_content(R"({"error":"image could not be read"})", "application/json");
                return;
            }

            refresh_upscaler_cache(*runtime);
            int model_scale = 0;
            std::string model_path;
            std::string used_name;
            {
                std::lock_guard<std::mutex> lock(*runtime->upscaler_mutex);
                for (const auto& entry : *runtime->upscaler_cache) {
                    if (entry.image_upscale_factor > 0 && (wanted.empty() || entry.name == wanted)) {
                        model_path  = entry.fullpath;
                        used_name   = entry.name;
                        model_scale = entry.image_upscale_factor;
                        break;
                    }
                }
            }
            if (model_path.empty()) {
                res.status = 400;
                res.set_content(json({{"error", wanted.empty()
                                                    ? std::string("no RGB ESRGAN upscaler models are available; "
                                                                  "start the server with --hires-upscalers-dir")
                                                    : "no compatible image upscaler called " + wanted}})
                                    .dump(),
                                "application/json");
                return;
            }

            if (!valid_upscale_dimensions(input.get(), model_scale, repeats)) {
                res.status = 400;
                res.set_content(R"({"error":"upscaled dimensions must not exceed 8192 x 8192"})", "application/json");
                return;
            }

            // One GPU: an upscale must not run while a generation is using it.
            std::lock_guard<std::mutex> ctx_lock(*runtime->sd_ctx_mutex);
            UpscalerCtxPtr upscaler_ctx(create_sd_upscaler(model_path.c_str(),
                                                           runtime->ctx_params->diffusion_conv_direct,
                                                           runtime->ctx_params->n_threads,
                                                           tile_size,
                                                           runtime->ctx_params->backend ? runtime->ctx_params->backend->c_str() : nullptr,
                                                           runtime->ctx_params->params_backend ? runtime->ctx_params->params_backend->c_str() : nullptr));
            if (upscaler_ctx == nullptr) {
                res.status = 500;
                res.set_content(R"({"error":"the upscaler model could not be loaded"})", "application/json");
                return;
            }
            const int factor = sd_upscaler_get_scale(upscaler_ctx.get());
            // The model file may have changed since its metadata was cached.
            if (!valid_upscale_dimensions(input.get(), factor, repeats)) {
                res.status = 400;
                res.set_content(R"({"error":"upscaled dimensions must not exceed 8192 x 8192"})", "application/json");
                return;
            }

            SDImageOwner current(input.release());
            std::string generation_info;
            for (int i = 0; i < repeats; ++i) {
                sd_result_t* generated = nullptr;
                if (!upscale_sd_image(upscaler_ctx.get(), current.get(), &generated) ||
                    !generated || sd_result_get_image_count(generated) <= 0 || sd_tensor_data(sd_result_get_image(generated, 0)) == nullptr) {
                    sd_result_free(generated);
                    res.status = 500;
                    res.set_content(R"({"error":"upscale failed"})", "application/json");
                    return;
                }
                generation_info  = merge_generation_info(generation_info, sd_result_get_generation_info(generated, 0));
                SDImage produced = take_result_image(generated, 0);

                sd_result_free(generated);
                current.reset(produced);
            }

            const std::string& format = output_options.output_format;
            const int compression     = output_options.output_compression;
            const SDImage result      = current.get();
            auto image_bytes          = encode_image_to_vector(format == "jpeg"   ? EncodedImageFormat::JPEG
                                                               : format == "webp" ? EncodedImageFormat::WEBP
                                                                                  : EncodedImageFormat::PNG,
                                                               result.data,
                                                               result.width,
                                                               result.height,
                                                               result.channel,
                                                               generation_info,
                                                               compression);
            if (image_bytes.empty()) {
                res.status = 500;
                res.set_content(R"({"error":"the result could not be encoded"})", "application/json");
                return;
            }

            json out;
            out["generation_info"] = json::parse(generation_info);
            out["upscaler"]        = used_name;
            out["scale"]           = factor;
            out["repeats"]         = repeats;
            out["width"]           = result.width;
            out["height"]          = result.height;
            out["output_format"]   = format;
            json images            = json::array();
            images.push_back({{"index", 0}, {"b64_json", base64_encode(image_bytes)}});
            out["images"] = std::move(images);
            res.set_content(out.dump(), "application/json");
            res.status = 200;
        } catch (const json::exception& e) {
            res.status = 400;
            res.set_content(json({{"error", "invalid request"}, {"message", e.what()}}).dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(json({{"error", std::string("server_error: ") + e.what()}}).dump(), "application/json");
        }
    });

    routes.post("/sdcpp/v1/img_gen", [runtime](const httplib::Request& req, httplib::Response& res) {
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

            json body = json::parse(req.body);
            ImgGenJobRequest request;
            std::string error_message;
            if (!parse_img_gen_request(body, *runtime, request, error_message)) {
                res.status = 400;
                res.set_content(json({{"error", error_message}}).dump(), "application/json");
                return;
            }

            AsyncJobManager& manager                = *runtime->async_job_manager;
            std::shared_ptr<AsyncGenerationJob> job = std::make_shared<AsyncGenerationJob>();
            job->kind                               = AsyncJobKind::ImgGen;
            job->status                             = AsyncJobStatus::Queued;
            job->created_at                         = unix_timestamp_now_ms();
            job->img_gen                            = std::move(request);

            {
                std::lock_guard<std::mutex> lock(manager.mutex);
                purge_expired_jobs(manager);
                if (count_pending_jobs(manager) >= manager.max_pending_jobs) {
                    res.status = 429;
                    res.set_content(R"({"error":"job queue is full"})", "application/json");
                    return;
                }
                job->id               = make_async_job_id(manager);
                manager.jobs[job->id] = job;
                manager.queue.push_back(job->id);
            }

            manager.cv.notify_one();

            json out;
            out["id"]       = job->id;
            out["kind"]     = async_job_kind_name(job->kind);
            out["status"]   = async_job_status_name(job->status);
            out["created"]  = job->created_at;
            out["poll_url"] = "/sdcpp/v1/jobs/" + job->id;

            res.status = 202;
            res.set_content(out.dump(), "application/json");
        } catch (const json::parse_error& e) {
            res.status = 400;
            res.set_content(json({{"error", "invalid json"}, {"message", e.what()}}).dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(json({{"error", "server_error"}, {"message", e.what()}}).dump(), "application/json");
        }
    });

    routes.post("/sdcpp/v1/vid_gen", [runtime](const httplib::Request& req, httplib::Response& res) {
        try {
            if (req.body.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"empty body"})", "application/json");
                return;
            }
            if (!runtime_supports_generation_mode(*runtime, VID_GEN)) {
                res.status = 400;
                res.set_content(json({{"error", unsupported_generation_mode_error(VID_GEN)}}).dump(), "application/json");
                return;
            }

            json body = json::parse(req.body);
            VidGenJobRequest request;
            std::string error_message;
            if (!parse_vid_gen_request(body, *runtime, request, error_message)) {
                res.status = 400;
                res.set_content(json({{"error", error_message}}).dump(), "application/json");
                return;
            }

            AsyncJobManager& manager                = *runtime->async_job_manager;
            std::shared_ptr<AsyncGenerationJob> job = std::make_shared<AsyncGenerationJob>();
            job->kind                               = AsyncJobKind::VidGen;
            job->status                             = AsyncJobStatus::Queued;
            job->created_at                         = unix_timestamp_now_ms();
            job->vid_gen                            = std::move(request);

            {
                std::lock_guard<std::mutex> lock(manager.mutex);
                purge_expired_jobs(manager);
                if (count_pending_jobs(manager) >= manager.max_pending_jobs) {
                    res.status = 429;
                    res.set_content(R"({"error":"job queue is full"})", "application/json");
                    return;
                }
                job->id               = make_async_job_id(manager);
                manager.jobs[job->id] = job;
                manager.queue.push_back(job->id);
            }

            manager.cv.notify_one();

            json out;
            out["id"]       = job->id;
            out["kind"]     = async_job_kind_name(job->kind);
            out["status"]   = async_job_status_name(job->status);
            out["created"]  = job->created_at;
            out["poll_url"] = "/sdcpp/v1/jobs/" + job->id;

            res.status = 202;
            res.set_content(out.dump(), "application/json");
        } catch (const json::parse_error& e) {
            res.status = 400;
            res.set_content(json({{"error", "invalid json"}, {"message", e.what()}}).dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(json({{"error", "server_error"}, {"message", e.what()}}).dump(), "application/json");
        }
    });

    routes.get(R"(/sdcpp/v1/jobs/([A-Za-z0-9_\-]+))", [runtime](const httplib::Request& req, httplib::Response& res) {
        AsyncJobManager& manager = *runtime->async_job_manager;
        std::lock_guard<std::mutex> lock(manager.mutex);
        purge_expired_jobs(manager);

        std::string job_id = req.matches[1];
        auto it            = manager.jobs.find(job_id);
        if (it == manager.jobs.end()) {
            if (manager.expired_jobs.find(job_id) != manager.expired_jobs.end()) {
                res.status = 410;
                res.set_content(R"({"error":"job expired"})", "application/json");
            } else {
                res.status = 404;
                res.set_content(R"({"error":"job not found"})", "application/json");
            }
            return;
        }

        res.status = 200;
        res.set_content(make_async_job_json(manager, *it->second).dump(), "application/json");
    });

    routes.post(R"(/sdcpp/v1/jobs/([A-Za-z0-9_\-]+)/cancel)", [runtime](const httplib::Request& req, httplib::Response& res) {
        AsyncJobManager& manager = *runtime->async_job_manager;
        std::lock_guard<std::mutex> lock(manager.mutex);
        purge_expired_jobs(manager);

        std::string job_id = req.matches[1];
        auto it            = manager.jobs.find(job_id);
        if (it == manager.jobs.end()) {
            if (manager.expired_jobs.find(job_id) != manager.expired_jobs.end()) {
                res.status = 410;
                res.set_content(R"({"error":"job expired"})", "application/json");
            } else {
                res.status = 404;
                res.set_content(R"({"error":"job not found"})", "application/json");
            }
            return;
        }

        auto& job = *it->second;
        if (job.status == AsyncJobStatus::Queued) {
            if (!cancel_queued_job(manager, job)) {
                res.status = 409;
                res.set_content(R"({"error":"job queue state changed before cancellation"})", "application/json");
                return;
            }
            res.status = 200;
            res.set_content(make_async_job_json(manager, job).dump(), "application/json");
            return;
        }

        if (job.status == AsyncJobStatus::Generating) {
            job.cancel_requested.store(true);
            res.status = 200;
            res.set_content(make_async_job_json(manager, job).dump(), "application/json");
            return;
        }

        res.status = 200;
        res.set_content(make_async_job_json(manager, job).dump(), "application/json");
    });
}
