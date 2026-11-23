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
    res.set_content(json({{"error", message}, {"code", code}}).dump(), "application/json");
}

httplib::Server::Handler RuntimeRoutes::protect(httplib::Server::Handler handler) {
    ServerRuntime* runtime = &runtime_;
    return [runtime, handler](const httplib::Request& req, httplib::Response& res) {
        const bool generation = req.path == "/sdapi/v1/txt2img" || req.path == "/sdapi/v1/img2img" ||
                                req.path == "/v1/images/generations" || req.path == "/v1/images/edits";
        if (generation) {
            SynchronousContextJob job(*runtime);
            std::lock_guard<std::mutex> lock(*runtime->sd_ctx_mutex);
            if (!has_current_model(*runtime)) {
                model_error(res, 503, "model_not_loaded", "No current model is configured");
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
    server_.Post(pattern, protect(handler));
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
        res.set_content(model_status(runtime).dump(), "application/json");
    });
}
