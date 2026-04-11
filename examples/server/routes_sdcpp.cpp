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

static json make_capabilities_json(ServerRuntime& runtime) {
    refresh_lora_cache(runtime);

    AsyncJobManager& manager  = *runtime.async_job_manager;
    const auto& defaults      = *runtime.default_gen_params;
    const auto& sample_params = defaults.sample_params;
    const auto& guidance      = sample_params.guidance;
    const fs::path model_path = resolve_display_model_path(runtime);
    json samplers             = json::array();
    json schedulers           = json::array();
    json output_formats       = json::array({"png", "jpeg"});
    json available_loras      = json::array();

    for (int i = 0; i < SAMPLE_METHOD_COUNT; ++i) {
        samplers.push_back(sd_sample_method_name((sample_method_t)i));
    }

    for (int i = 0; i < SCHEDULER_COUNT; ++i) {
        schedulers.push_back(sd_scheduler_name((scheduler_t)i));
    }

#ifdef SD_USE_WEBP
    output_formats.push_back("webp");
#endif

    {
        std::lock_guard<std::mutex> lock(*runtime.lora_mutex);
        for (const auto& entry : *runtime.lora_cache) {
            available_loras.push_back({
                {"name", entry.name},
                {"path", entry.path},
            });
        }
    }

    json result;
    result["model"] = {
        {"name", model_path.filename().u8string()},
        {"stem", model_path.stem().u8string()},
        {"path", model_path.u8string()},
    };
    result["defaults"] = {
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
        {"sample_params",
         {
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
                       {"layers", defaults.skip_layers},
                       {"layer_start", guidance.slg.layer_start},
                       {"layer_end", guidance.slg.layer_end},
                       {"scale", guidance.slg.scale},
                   }},
              }},
         }},
        {"vae_tiling_params", make_vae_tiling_json(defaults.vae_tiling_params)},
        {"cache_mode", defaults.cache_mode},
        {"cache_option", defaults.cache_option},
        {"scm_mask", defaults.scm_mask},
        {"scm_policy_dynamic", defaults.scm_policy_dynamic},
        {"output_format", "png"},
        {"output_compression", 100},
    };
    result["limits"] = {
        {"min_width", 64},
        {"max_width", 4096},
        {"min_height", 64},
        {"max_height", 4096},
        {"max_batch_count", 8},
        {"max_queue_size", manager.max_pending_jobs},
    };
    result["samplers"]       = samplers;
    result["schedulers"]     = schedulers;
    result["output_formats"] = output_formats;
    result["features"]       = {
              {"init_image", true},
              {"mask_image", true},
              {"control_image", true},
              {"ref_images", true},
              {"lora", true},
              {"vae_tiling", true},
              {"cache", true},
              {"cancel_queued", true},
              {"cancel_generating", false},
    };
    result["loras"] = available_loras;
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
    if (!request.gen_params.resolve_and_validate(IMG_GEN, "", true)) {
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

    svr.Post("/sdcpp/v1/vid_gen", [](const httplib::Request&, httplib::Response& res) {
        res.status = 501;
        res.set_content(R"({"error":"vid_gen is reserved and not implemented yet"})", "application/json");
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
