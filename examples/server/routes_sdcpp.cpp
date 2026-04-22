#include "routes.h"

#include <algorithm>
#include <cmath>
#include <filesystem>

#include "async_jobs.h"
#include "common/common.h"

namespace fs = std::filesystem;

static bool parse_cache_mode(const std::string& mode_str, sd_cache_mode_t& mode_out) {
    if (mode_str == "disabled") {
        mode_out = SD_CACHE_DISABLED;
        return true;
    }
    if (mode_str == "easycache") {
        mode_out = SD_CACHE_EASYCACHE;
        return true;
    }
    if (mode_str == "ucache") {
        mode_out = SD_CACHE_UCACHE;
        return true;
    }
    if (mode_str == "dbcache") {
        mode_out = SD_CACHE_DBCACHE;
        return true;
    }
    if (mode_str == "taylorseer") {
        mode_out = SD_CACHE_TAYLORSEER;
        return true;
    }
    if (mode_str == "cache-dit") {
        mode_out = SD_CACHE_CACHE_DIT;
        return true;
    }
    if (mode_str == "spectrum") {
        mode_out = SD_CACHE_SPECTRUM;
        return true;
    }
    return false;
}

static json finite_number_or_null(float value) {
    return std::isfinite(value) ? json(value) : json(nullptr);
}

static const char* capability_scheduler_name(enum scheduler_t scheduler) {
    return scheduler < SCHEDULER_COUNT ? sd_scheduler_name(scheduler) : "default";
}

static const char* capability_sample_method_name(enum sample_method_t sample_method) {
    return sample_method < SAMPLE_METHOD_COUNT ? sd_sample_method_name(sample_method) : "default";
}

static json make_vae_tiling_json(const sd_tiling_params_t& params) {
    return {
        {"enabled", params.enabled},
        {"tile_size_x", params.tile_size_x},
        {"tile_size_y", params.tile_size_y},
        {"target_overlap", params.target_overlap},
        {"rel_size_x", params.rel_size_x},
        {"rel_size_y", params.rel_size_y},
    };
}

static fs::path resolve_display_model_path(const ServerRuntime& runtime) {
    const auto& ctx = *runtime.ctx_params;
    if (!ctx.model_path.empty()) {
        return fs::path(ctx.model_path);
    }
    if (!ctx.diffusion_model_path.empty()) {
        return fs::path(ctx.diffusion_model_path);
    }
    return {};
}

static json make_sample_params_json(const sd_sample_params_t& sample_params, const std::vector<int>& skip_layers) {
    const auto& guidance = sample_params.guidance;
    return {
        {"scheduler", capability_scheduler_name(sample_params.scheduler)},
        {"sample_method", capability_sample_method_name(sample_params.sample_method)},
        {"sample_steps", sample_params.sample_steps},
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

static json make_img_gen_defaults_json(const SDGenerationParams& defaults, const std::string& output_format) {
    return {
        {"prompt", defaults.prompt},
        {"negative_prompt", defaults.negative_prompt},
        {"clip_skip", defaults.clip_skip},
        {"width", defaults.width > 0 ? defaults.width : 512},
        {"height", defaults.height > 0 ? defaults.height : 512},
        {"strength", defaults.strength},
        {"seed", defaults.seed},
        {"batch_count", defaults.batch_count},
        {"auto_resize_ref_image", defaults.auto_resize_ref_image},
        {"increase_ref_index", defaults.increase_ref_index},
        {"control_strength", defaults.control_strength},
        {"sample_params", make_sample_params_json(defaults.sample_params, defaults.skip_layers)},
        {"hires",
         {
             {"enabled", defaults.hires_enabled},
             {"upscaler", defaults.hires_upscaler},
             {"scale", defaults.hires_scale},
             {"target_width", defaults.hires_width},
             {"target_height", defaults.hires_height},
             {"steps", defaults.hires_steps},
             {"denoising_strength", defaults.hires_denoising_strength},
             {"upscale_tile_size", defaults.hires_upscale_tile_size},
         }},
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
        {"width", defaults.width > 0 ? defaults.width : 512},
        {"height", defaults.height > 0 ? defaults.height : 512},
        {"strength", defaults.strength},
        {"seed", defaults.seed},
        {"video_frames", defaults.video_frames},
        {"fps", defaults.fps},
        {"moe_boundary", defaults.moe_boundary},
        {"vace_strength", defaults.vace_strength},
        {"sample_params", make_sample_params_json(defaults.sample_params, defaults.skip_layers)},
        {"high_noise_sample_params", make_sample_params_json(defaults.high_noise_sample_params, defaults.high_noise_skip_layers)},
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
        {"ref_images", true},
        {"lora", true},
        {"vae_tiling", true},
        {"hires", true},
        {"cache", true},
        {"cancel_queued", true},
        {"cancel_generating", false},
    };
}

static json make_vid_gen_features_json() {
    return {
        {"init_image", true},
        {"end_image", true},
        {"control_frames", true},
        {"high_noise_sample_params", true},
        {"lora", true},
        {"vae_tiling", true},
        {"cache", true},
        {"cancel_queued", true},
        {"cancel_generating", false},
    };
}

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

    for (int i = 0; i < SAMPLE_METHOD_COUNT; ++i) {
        samplers.push_back(sd_sample_method_name((sample_method_t)i));
    }

    for (int i = 0; i < SCHEDULER_COUNT; ++i) {
        schedulers.push_back(sd_scheduler_name((scheduler_t)i));
    }

    {
        std::lock_guard<std::mutex> lock(*runtime.lora_mutex);
        for (const auto& entry : *runtime.lora_cache) {
            available_loras.push_back({
                {"name", entry.name},
                {"path", entry.path},
            });
        }
    }

    available_upscalers.push_back({
        {"name", "None"},
    });
    available_upscalers.push_back({
        {"name", "Latent (nearest)"},
    });
    {
        std::lock_guard<std::mutex> lock(*runtime.upscaler_mutex);
        for (const auto& entry : *runtime.upscaler_cache) {
            available_upscalers.push_back({
                {"name", entry.name},
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

    json defaults_by_mode       = json::object();
    json output_formats_by_mode = json::object();
    json features_by_mode       = json::object();
    if (supports_img) {
        defaults_by_mode["img_gen"]       = make_img_gen_defaults_json(defaults, default_img_output_format);
        output_formats_by_mode["img_gen"] = image_output_formats;
        features_by_mode["img_gen"]       = make_img_gen_features_json();
    }
    if (supports_vid) {
        defaults_by_mode["vid_gen"]       = make_vid_gen_defaults_json(defaults, default_vid_output_format);
        output_formats_by_mode["vid_gen"] = video_output_formats;
        features_by_mode["vid_gen"]       = make_vid_gen_features_json();
    }

    json top_level_defaults       = json::object();
    json top_level_output_formats = json::array();
    json top_level_features       = {
              {"cancel_queued", true},
              {"cancel_generating", false},
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
    result["current_mode"]     = current_mode;
    result["supported_modes"]  = supported_modes;
    result["defaults"]         = top_level_defaults;
    result["defaults_by_mode"] = defaults_by_mode;
    result["limits"]           = {
                  {"min_width", 64},
                  {"max_width", 4096},
                  {"min_height", 64},
                  {"max_height", 4096},
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
    // Intentionally disable prompt-embedded LoRA tag parsing for server APIs.
    if (!request.gen_params.resolve_and_validate(IMG_GEN, "", runtime.ctx_params->hires_upscalers_dir, true)) {
        error_message = "invalid generation parameters";
        return false;
    }
    return true;
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

    std::string output_format = body.value("output_format", "webm");
    int output_compression    = body.value("output_compression", 100);
    if (!assign_output_options(request, output_format, output_compression, error_message)) {
        return false;
    }
    // Intentionally disable prompt-embedded LoRA tag parsing for server APIs.
    if (!request.gen_params.resolve_and_validate(VID_GEN, "", runtime.ctx_params->hires_upscalers_dir, true)) {
        error_message = "invalid generation parameters";
        return false;
    }
    return true;
}

void register_sdcpp_api_endpoints(httplib::Server& svr, ServerRuntime& rt) {
    ServerRuntime* runtime = &rt;

    svr.Get("/sdcpp/v1/capabilities", [runtime](const httplib::Request&, httplib::Response& res) {
        res.status = 200;
        res.set_content(make_capabilities_json(*runtime).dump(), "application/json");
    });

    svr.Post("/sdcpp/v1/img_gen", [runtime](const httplib::Request& req, httplib::Response& res) {
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
            job->created_at                         = unix_timestamp_now();
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

    svr.Post("/sdcpp/v1/vid_gen", [runtime](const httplib::Request& req, httplib::Response& res) {
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
            job->created_at                         = unix_timestamp_now();
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

    svr.Get(R"(/sdcpp/v1/jobs/([A-Za-z0-9_\-]+))", [runtime](const httplib::Request& req, httplib::Response& res) {
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

    svr.Post(R"(/sdcpp/v1/jobs/([A-Za-z0-9_\-]+)/cancel)", [runtime](const httplib::Request& req, httplib::Response& res) {
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
            res.status = 409;
            res.set_content(R"({"error":"job is currently generating and cannot be interrupted yet"})", "application/json");
            return;
        }

        res.status = 200;
        res.set_content(make_async_job_json(manager, job).dump(), "application/json");
    });
}
