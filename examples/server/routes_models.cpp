#include "routes.h"

#include "async_jobs.h"
#include "model_catalog.h"

static void model_error(httplib::Response& res, int status, const std::string& code, const std::string& message) {
    res.status = status;
    res.set_content(json({{"error", message}, {"code", code}}).dump(), "application/json");
}

httplib::Server::Handler RuntimeRoutes::protect(httplib::Server::Handler handler) {
    ServerRuntime* runtime = &runtime_;
    return [runtime, handler](const httplib::Request& req, httplib::Response& res) {
        std::shared_lock<std::shared_mutex> lock(runtime->models->runtime_mutex, std::try_to_lock);
        if (!lock.owns_lock()) {
            model_error(res, 503, "model_loading", "Model is loading or unloading");
            return;
        }
        const bool generation = req.path == "/sdcpp/v1/img_gen" || req.path == "/sdcpp/v1/vid_gen" ||
                                req.path == "/sdapi/v1/txt2img" || req.path == "/sdapi/v1/img2img" ||
                                req.path == "/v1/images/generations" || req.path == "/v1/images/edits";
        if (generation && !runtime->sd_ctx) {
            model_error(res, 503, "model_not_loaded", "No model is loaded");
            return;
        }
        handler(req, res);
    };
}

void RuntimeRoutes::get(const std::string& pattern, httplib::Server::Handler handler) {
    // Job polling and cancellation only access the job manager, even during a model transition.
    server_.Get(pattern, pattern.find("/sdcpp/v1/jobs/") != std::string::npos ? handler : protect(handler));
}

void RuntimeRoutes::post(const std::string& pattern, httplib::Server::Handler handler) {
    server_.Post(pattern, pattern.find("/sdcpp/v1/jobs/") != std::string::npos ? handler : protect(handler));
}

static void set_model_status(ServerRuntime& runtime, const char* state, const json& recipe, const json& selection, const json& error = nullptr) {
    std::lock_guard<std::mutex> lock(runtime.models->status_mutex);
    auto& models            = *runtime.models;
    const uint64_t revision = models.status.value("revision", uint64_t(0)) + 1;
    json elapsed            = nullptr;
    if (!recipe.is_null()) {
        const auto now = std::chrono::steady_clock::now();
        if (std::string(state) == "loading") {
            models.load_started_at = now;
            elapsed                = 0.0;
        } else {
            elapsed = std::chrono::duration<double>(now - models.load_started_at).count();
        }
    }
    models.status = {{"state", state}, {"recipe", recipe}, {"models", selection}, {"error", error}, {"revision", revision}, {"load_elapsed_seconds", elapsed}};
}

static void change_model(ServerRuntime& runtime, const httplib::Request& req, httplib::Response& res, bool unload) {
    auto& models = *runtime.models;
    std::unique_lock<std::shared_mutex> lock(models.runtime_mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        model_error(res, 409, "model_busy", "Model is in use or another model operation is running");
        return;
    }
    {
        // Admission holds runtime_mutex through enqueue. Jobs stay pending until
        // generation and metadata encoding finish, so no worker can retain this context here.
        std::lock_guard<std::mutex> jobs_lock(runtime.async_job_manager->mutex);
        if (count_pending_jobs(*runtime.async_job_manager) != 0) {
            model_error(res, 409, "model_busy", "Finish or cancel pending generation jobs before switching models");
            return;
        }
    }

    json body;
    json loaded_recipe;
    SDContextParams context;
    SDGenerationParams defaults;
    if (!unload) {
        try {
            body = json::parse(req.body);
            prepare_model_load(runtime, body, context, defaults, &loaded_recipe);
        } catch (const std::exception& e) {
            model_error(res, 400, "invalid_model_config", e.what());
            return;
        }
    }

    const json recipe    = unload ? json(nullptr) : body.at("recipe");
    const json selection = unload ? json::object() : body.at("models");
    set_model_status(runtime, "loading", recipe, selection);
    try {
        runtime.sd_ctx = nullptr;
        models.context.reset();
        models.active_recipe        = nullptr;
        *runtime.ctx_params         = unload ? models.base_context : std::move(context);
        *runtime.default_gen_params = unload ? models.base_defaults : std::move(defaults);
        if (!unload) {
            auto params = runtime.ctx_params->to_sd_ctx_params_t();
            models.context.reset(new_sd_ctx(&params));
            if (!models.context) {
                throw std::runtime_error("Model initialization failed; check the server log for component compatibility and memory errors");
            }
            runtime.sd_ctx       = models.context.get();
            models.active_recipe = std::move(loaded_recipe);
        }
        set_model_status(runtime, unload ? "empty" : "ready", recipe, selection);
        res.set_content(model_status(runtime).dump(), "application/json");
    } catch (const std::exception& e) {
        runtime.sd_ctx = nullptr;
        models.context.reset();
        models.active_recipe        = nullptr;
        *runtime.ctx_params         = models.base_context;
        *runtime.default_gen_params = models.base_defaults;
        set_model_status(runtime, "error", recipe, selection, e.what());
        model_error(res, 422, "model_load_failed", e.what());
    }
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
        res.set_content(model_status(runtime).dump(), "application/json");
    });
    svr.Post("/sdcpp/v1/models/load", [&runtime](const httplib::Request& req, httplib::Response& res) {
        change_model(runtime, req, res, false);
    });
    svr.Post("/sdcpp/v1/models/unload", [&runtime](const httplib::Request& req, httplib::Response& res) {
        change_model(runtime, req, res, true);
    });
}
