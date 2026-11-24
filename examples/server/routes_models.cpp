#include "routes.h"

#include "async_jobs.h"
#include "model_catalog.h"

class SynchronousContextJob {
    AsyncJobManager& manager_;
    const SDSvrParams& params_;

public:
    explicit SynchronousContextJob(const ServerRuntime& runtime)
        : manager_(*runtime.async_job_manager), params_(*runtime.svr_params) {
        std::lock_guard<std::mutex> lock(manager_.mutex);
        ++manager_.synchronous_jobs;
    }

    SynchronousContextJob(const SynchronousContextJob&)            = delete;
    SynchronousContextJob& operator=(const SynchronousContextJob&) = delete;

    ~SynchronousContextJob() {
        {
            std::lock_guard<std::mutex> lock(manager_.mutex);
            --manager_.synchronous_jobs;
            manager_.record_context_job_completion(params_);
        }
        manager_.cv.notify_one();
    }
};

static void model_error(httplib::Response& res, int status, const std::string& code, const std::string& message) {
    res.status = status;
    res.set_content(safe_json_dump(json({{"error", message}, {"code", code}})), "application/json");
}

httplib::Server::Handler RuntimeRoutes::protect(httplib::Server::Handler handler) {
    ServerRuntime* runtime = &runtime_;
    return [runtime, handler](const httplib::Request& req, httplib::Response& res) {
        const bool generation = req.path == "/sdapi/v1/txt2img" || req.path == "/sdapi/v1/img2img" ||
                                req.path == "/v1/images/generations" || req.path == "/v1/images/edits";
        if (generation) {
            SynchronousContextJob job(*runtime);
            std::lock_guard<std::mutex> lock(*runtime->sd_ctx_mutex);
            if (!runtime->keeper->has_model()) {
                model_error(res, 503, "model_not_loaded", "No model is loaded");
                return;
            }
            handler(req, res);
        } else {
            handler(req, res);
        }
    };
}

void RuntimeRoutes::get(const std::string& pattern, httplib::Server::Handler handler) {
    if (pattern.find("/sdcpp/v1/jobs/") != std::string::npos) {
        server_.Get(pattern, handler);
        return;
    }
    ServerModels* models = runtime_.models;
    server_.Get(pattern, [models, handler](const httplib::Request& req, httplib::Response& res) {
        std::shared_lock<std::shared_mutex> lock(models->model_mutex);
        handler(req, res);
    });
}

void RuntimeRoutes::post(const std::string& pattern, httplib::Server::Handler handler) {
    server_.Post(pattern, pattern.find("/sdcpp/v1/jobs/") != std::string::npos ? handler : protect(handler));
}

// Loads the model of a recipe ahead of the requests that use it, or unloads the current one. Generation jobs
// pick their model themselves; these routes are for programs that want the model ready, or the memory back.
static void change_model(ServerRuntime& runtime, const httplib::Request& req, httplib::Response& res, bool unload) {
    auto& models = *runtime.models;
    {
        std::lock_guard<std::mutex> jobs_lock(runtime.async_job_manager->mutex);
        if (count_pending_jobs(*runtime.async_job_manager) != 0) {
            model_error(res, 409, "model_busy", "Finish or cancel pending generation jobs before switching models");
            return;
        }
    }
    std::unique_lock<std::mutex> context_lock(*runtime.sd_ctx_mutex, std::try_to_lock);
    if (!context_lock.owns_lock()) {
        model_error(res, 409, "model_busy", "Model is in use or another model operation is running");
        return;
    }
    if (unload) {
        std::unique_lock<std::shared_mutex> lock(models.model_mutex);
        runtime.keeper->unload();
        *runtime.ctx_params         = models.base_context;
        *runtime.default_gen_params = models.base_defaults;
        models.current_recipe       = nullptr;
        models.current_files        = json::object();
        ++models.revision;
    } else {
        ServerModelSelection selection;
        try {
            selection = resolve_model_selection(runtime, json::parse(req.body), IMG_GEN);
        } catch (const std::exception& e) {
            model_error(res, 400, "invalid_model_config", e.what());
            return;
        }
        std::string error_message;
        if (!switch_current_model(runtime, std::move(selection), error_message)) {
            model_error(res, 422, "model_load_failed", error_message);
            return;
        }
    }
    std::shared_lock<std::shared_mutex> lock(models.model_mutex);
    res.set_content(safe_json_dump(model_status(runtime)), "application/json");
}

void register_model_endpoints(httplib::Server& svr, ServerRuntime& runtime) {
    svr.Get("/sdcpp/v1/recipes", [&runtime](const httplib::Request&, httplib::Response& res) {
        auto& models = *runtime.models;
        std::lock_guard<std::mutex> lock(models.catalog_mutex);
        json items = json::array();
        for (const auto& entry : models.recipes)
            items.push_back(entry.second);
        res.set_content(json({{"recipes", items}, {"errors", models.catalog_errors}}).dump(), "application/json");
    });
    svr.Get("/sdcpp/v1/model-files", [&runtime](const httplib::Request&, httplib::Response& res) {
        auto& models = *runtime.models;
        std::lock_guard<std::mutex> lock(models.catalog_mutex);
        json items = json::array();
        for (const auto& entry : models.files) {
            items.push_back({{"id", entry.first}, {"name", entry.second.name}, {"category", entry.second.category}, {"root", entry.second.root.id}, {"relative_path", entry.second.relative_path}});
        }
        res.set_content(json({{"files", items}, {"errors", models.catalog_errors}}).dump(), "application/json");
    });
    svr.Post("/sdcpp/v1/models/refresh", [&runtime](const httplib::Request&, httplib::Response& res) {
        refresh_model_catalog(runtime);
        std::lock_guard<std::mutex> lock(runtime.models->catalog_mutex);
        res.set_content(json({{"errors", runtime.models->catalog_errors}}).dump(), "application/json");
    });
    svr.Get("/sdcpp/v1/models/status", [&runtime](const httplib::Request&, httplib::Response& res) {
        std::shared_lock<std::shared_mutex> lock(runtime.models->model_mutex);
        res.set_content(safe_json_dump(model_status(runtime)), "application/json");
    });
    svr.Post("/sdcpp/v1/models/load", [&runtime](const httplib::Request& req, httplib::Response& res) {
        change_model(runtime, req, res, false);
    });
    svr.Post("/sdcpp/v1/models/unload", [&runtime](const httplib::Request& req, httplib::Response& res) {
        change_model(runtime, req, res, true);
    });
}
