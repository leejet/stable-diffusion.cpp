#include "routes.h"

#include <cmath>
#include <filesystem>
#include <utility>

#include "async_jobs.h"
#include "common/common.h"
#include "common/resource_owners.hpp"
#include "model_catalog.h"

namespace fs = std::filesystem;

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

static json make_sample_params_json(const SDSampleParams& sample_params, const std::optional<std::vector<int>>& skip_layers, const std::optional<std::vector<float>>& custom_sigmas) {
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

static json make_vid_gen_defaults_json(const SDGenerationParams& defaults, const std::string& output_format, bool high_noise) {
    json result = {
        {"prompt", defaults.prompt},
        {"negative_prompt", defaults.negative_prompt},
        {"clip_skip", defaults.clip_skip},
        {"width", defaults.width},
        {"height", defaults.height},
        {"strength", defaults.strength},
        {"seed", defaults.seed},
        {"video_frames", defaults.video_frames},
        {"fps", defaults.fps},
        {"vace_strength", defaults.vace_strength},
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
    if (high_noise) {
        result["moe_boundary"]             = defaults.moe_boundary;
        result["high_noise_sample_params"] = {
            {"scheduler", defaults.high_noise_sample_params.scheduler},
            {"sample_steps", defaults.high_noise_sample_params.sample_steps},
            {"flow_shift", finite_number_or_null(defaults.high_noise_sample_params.flow_shift)},
        };
    }
    return result;
}

static json make_img_gen_features_json() {
    return {
        {"init_image", true},
        {"mask_image", true},
        {"control_image", true},
        {"ip_adapter_image", false},
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
static json make_vid_gen_features_json(bool high_noise) {
    return {
        {"init_image", true},
        {"end_image", true},
        {"control_frames", true},
        {"ref_videos", true},
        {"ref_audios", true},
        {"ref_video_audios", true},
        {"high_noise_sample_params", high_noise},
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
    });
    available_upscalers.push_back({
        {"name", "Lanczos"},
        {"model", false},
    });
    available_upscalers.push_back({
        {"name", "Nearest"},
        {"model", false},
    });
    available_upscalers.push_back({
        {"name", "Latent"},
        {"model", false},
    });
    available_upscalers.push_back({
        {"name", "Latent (nearest)"},
        {"model", false},
    });
    available_upscalers.push_back({
        {"name", "Latent (nearest-exact)"},
        {"model", false},
    });
    available_upscalers.push_back({
        {"name", "Latent (antialiased)"},
        {"model", false},
    });
    available_upscalers.push_back({
        {"name", "Latent (bicubic)"},
        {"model", false},
    });
    available_upscalers.push_back({
        {"name", "Latent (bicubic antialiased)"},
        {"model", false},
    });
    {
        std::lock_guard<std::mutex> lock(*runtime.upscaler_mutex);
        for (const auto& entry : *runtime.upscaler_cache) {
            available_upscalers.push_back({
                {"name", entry.name},
                {"model", true},
            });
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
        if (!video) {
            features["mask_image"]       = generation_features.at("mask_image").get<bool>() && generation_features.at("img2img").get<bool>();
            features["control_image"]    = generation_features.at("controlnet").get<bool>() || generation_features.at("control_image").get<bool>();
            features["ip_adapter_image"] = generation_features.at("ip_adapter");
        }
        return features;
    };
    json defaults_by_mode       = json::object();
    json output_formats_by_mode = {{"img_gen", image_output_formats}, {"vid_gen", video_output_formats}};
    json features_by_mode       = json::object();
    if (supports_img) {
        defaults_by_mode["img_gen"] = make_img_gen_defaults_json(defaults, default_img_output_format);
        features_by_mode["img_gen"] = recipe_features(make_img_gen_features_json(), false);
    }
    if (supports_vid) {
        const bool high_noise       = generation_features.at("high_noise").get<bool>();
        defaults_by_mode["vid_gen"] = make_vid_gen_defaults_json(defaults, default_vid_output_format, high_noise);
        features_by_mode["vid_gen"] = recipe_features(make_vid_gen_features_json(high_noise), true);
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
    result["generation_features"]        = generation_features;
    result["generation_model_selection"] = true;
    result["media_view"]                 = runtime.svr_params->media_view;
    if (runtime.svr_params->media_view && !runtime.svr_params->output_dir.empty()) {
        const auto& output  = runtime.models->workspace_file_access.output;
        auto paths          = json::array({output.generic_u8string()});
        const auto relative = output.lexically_relative(fs::weakly_canonical(fs::current_path()));
        if (!relative.empty())
            paths.push_back(relative.generic_u8string());
        result["media_output"] = {{"paths", std::move(paths)}};
#ifdef _WIN32
        result["media_output"]["case_insensitive"] = true;
#endif
    }
    result["runtime"]          = model_status(runtime);
    result["current_mode"]     = current_mode;
    result["supported_modes"]  = supported_modes;
    result["defaults"]         = top_level_defaults;
    result["defaults_by_mode"] = defaults_by_mode;
    result["limits"]           = {
        {"min_width", 64},
        {"max_width", k_max_generation_dimension},
        {"min_height", 64},
        {"max_height", k_max_generation_dimension},
        {"max_batch_count", 8},
        {"max_queue_size", manager.max_pending_jobs},
    };
    result["samplers"]               = samplers;
    result["schedulers"]             = schedulers;
    result["output_formats"]         = top_level_output_formats;
    result["output_formats_by_mode"] = output_formats_by_mode;
    result["features"]               = top_level_features;
    result["features_by_mode"]       = features_by_mode;
    result["loras"]                  = available_loras;
    result["upscalers"]              = available_upscalers;
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

    std::string output_format = body.value("output_format", supported_vid_output_formats().front());
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

bool parse_native_job_request(const json& body, ServerRuntime& runtime, AsyncGenerationJob& job, std::string& error_message) {
    return job.kind == AsyncJobKind::VidGen ? parse_vid_gen_request(body, runtime, job.vid_gen, error_message)
                                            : parse_img_gen_request(body, runtime, job.img_gen, error_message);
}

static void submit_generation_job(ServerRuntime& runtime, const httplib::Request& req, httplib::Response& res, SDMode mode) {
    try {
        if (req.body.empty()) {
            res.status = 400;
            res.set_content(R"({"error":"empty body"})", "application/json");
            return;
        }
        auto body = json::parse(req.body);
        if (!body.is_object()) {
            res.status = 400;
            res.set_content(R"({"error":"generation request must be an object"})", "application/json");
            return;
        }

        auto job                      = std::make_shared<AsyncGenerationJob>();
        job->kind                     = mode == VID_GEN ? AsyncJobKind::VidGen : AsyncJobKind::ImgGen;
        ServerRuntime request_runtime = runtime;
        std::shared_lock<std::shared_mutex> model_lock(runtime.models->model_mutex, std::defer_lock);
        if (auto model = body.find("model"); model != body.end()) {
            try {
                job->model = std::make_unique<ServerModelSelection>(resolve_model_selection(runtime, *model, mode));
            } catch (const std::exception& error) {
                res.status = 400;
                res.set_content(json({{"error", error.what()}, {"code", "invalid_model_config"}}).dump(), "application/json");
                return;
            }
            body.erase(model);
            request_runtime.ctx_params         = &job->model->context;
            request_runtime.default_gen_params = &job->model->defaults;
        } else {
            model_lock.lock();
            if (!has_current_model(runtime)) {
                res.status = 503;
                res.set_content(R"({"error":"No current model is configured","code":"model_not_loaded"})", "application/json");
                return;
            }
            if (!runtime_supports_generation_mode(runtime, mode)) {
                res.status = 400;
                res.set_content(json({{"error", unsupported_generation_mode_error(mode)}}).dump(), "application/json");
                return;
            }
        }

        std::string error_message;
        if (!parse_native_job_request(body, request_runtime, *job, error_message)) {
            res.status = 400;
            res.set_content(json({{"error", error_message}}).dump(), "application/json");
            return;
        }
        if (model_lock.owns_lock()) {
            job->model_revision = runtime.models->revision;
            job->request_body   = std::move(body);
            model_lock.unlock();
        }

        auto& manager = *runtime.async_job_manager;
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
        out["status"]   = async_job_status_name(AsyncJobStatus::Queued);
        out["created"]  = job->created_at;
        out["poll_url"] = "/sdcpp/v1/jobs/" + job->id;
        res.status      = 202;
        res.set_content(out.dump(), "application/json");
    } catch (const json::parse_error& error) {
        res.status = 400;
        res.set_content(json({{"error", "invalid json"}, {"message", error.what()}}).dump(), "application/json");
    } catch (const std::exception& error) {
        res.status = 500;
        res.set_content(json({{"error", "server_error"}, {"message", error.what()}}).dump(), "application/json");
    }
}

void register_sdcpp_api_endpoints(httplib::Server& svr, ServerRuntime& rt) {
    ServerRuntime* runtime = &rt;
    RuntimeRoutes routes(svr, rt);

    routes.get("/sdcpp/v1/capabilities", [runtime](const httplib::Request&, httplib::Response& res) {
        res.status = 200;
        res.set_content(make_capabilities_json(*runtime).dump(), "application/json");
    });

    routes.post("/sdcpp/v1/img_gen", [runtime](const httplib::Request& req, httplib::Response& res) {
        submit_generation_job(*runtime, req, res, IMG_GEN);
    });

    routes.post("/sdcpp/v1/vid_gen", [runtime](const httplib::Request& req, httplib::Response& res) {
        submit_generation_job(*runtime, req, res, VID_GEN);
    });

    routes.get(R"(/sdcpp/v1/jobs/([A-Za-z0-9_\-]+))", [runtime](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Cache-Control", "no-store");
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
        res.set_header("Cache-Control", "no-store");
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
        } else if (job.status == AsyncJobStatus::Generating) {
            job.cancel_requested.store(true);
        }

        res.status = 200;
        res.set_content(make_async_job_json(manager, job).dump(), "application/json");
    });
}
