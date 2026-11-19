#include "routes.h"

#include <limits>
#include <stdexcept>

#include "async_jobs.h"
#include "common/log.h"
#include "model_catalog.h"

static int64_t steady_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void mark_model_activity(ServerRuntime& runtime) {
    runtime.models->last_activity_ms.store(steady_time_ms());
}

struct ModelActivityGuard {
    ServerRuntime& runtime;

    explicit ModelActivityGuard(ServerRuntime& runtime)
        : runtime(runtime) {
        mark_model_activity(runtime);
    }

    ~ModelActivityGuard() {
        mark_model_activity(runtime);
    }
};

void sleep_idle_model(ServerRuntime& runtime) {
    auto& models = *runtime.models;
    if (models.sleep_idle_seconds.load() <= 0)
        return;
    std::unique_lock<std::mutex> admission(models.admission_mutex, std::try_to_lock);
    if (!admission.owns_lock())
        return;
    std::unique_lock<std::shared_mutex> lock(models.runtime_mutex, std::try_to_lock);
    if (!lock.owns_lock() || !runtime.sd_ctx || models.sleeping)
        return;
    const int timeout = models.sleep_idle_seconds.load();
    if (timeout <= 0)
        return;
    {
        std::lock_guard<std::mutex> jobs_lock(runtime.async_job_manager->mutex);
        if (runtime.async_job_manager->stop || count_pending_jobs(*runtime.async_job_manager) != 0 ||
            steady_time_ms() - models.last_activity_ms.load() < int64_t(timeout) * 1000)
            return;
    }
    {
        std::lock_guard<std::mutex> status_lock(models.status_mutex);
        models.status["state"]                = "loading";
        models.status["load_elapsed_seconds"] = nullptr;
        models.status["revision"]             = models.status.value("revision", uint64_t(0)) + 1;
    }
    LOG_INFO("Unloading model after %d idle seconds", timeout);
    models.sleeping_supports_image = runtime_supports_generation_mode(runtime, IMG_GEN);
    models.sleeping_supports_video = runtime_supports_generation_mode(runtime, VID_GEN);
    runtime.sd_ctx                 = nullptr;
    models.context.reset();
    models.sleeping = true;
    {
        std::lock_guard<std::mutex> status_lock(models.status_mutex);
        models.status["state"]    = "sleeping";
        models.status["revision"] = models.status.value("revision", uint64_t(0)) + 1;
    }
}

// Caller owns runtime_mutex exclusively. The resolved parameters, defaults and
// recipe are retained during sleep, including command-line-only configurations.
static bool wake_model(ServerRuntime& runtime, std::string& error) {
    auto& models = *runtime.models;
    if (!models.sleeping)
        return true;
    {
        std::lock_guard<std::mutex> status_lock(models.status_mutex);
        models.load_started_at                = std::chrono::steady_clock::now();
        models.status["state"]                = "loading";
        models.status["error"]                = nullptr;
        models.status["load_elapsed_seconds"] = 0.0;
        models.status["revision"]             = models.status.value("revision", uint64_t(0)) + 1;
    }
    LOG_INFO("Reloading model for generation request");
    try {
        auto params = runtime.ctx_params->to_context_opt();
        models.context.reset(create_sd_context(params));
        if (!models.context)
            throw std::runtime_error("Model initialization failed; check the server log for component compatibility and memory errors");
        runtime.sd_ctx = models.context.get();
    } catch (const std::exception& e) {
        error          = e.what();
        runtime.sd_ctx = nullptr;
        models.context.reset();
    }
    models.sleeping = false;
    mark_model_activity(runtime);
    {
        std::lock_guard<std::mutex> status_lock(models.status_mutex);
        models.status["state"]                = error.empty() ? "ready" : "error";
        models.status["error"]                = error.empty() ? json(nullptr) : json(error);
        models.status["load_elapsed_seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - models.load_started_at).count();
        models.status["revision"]             = models.status.value("revision", uint64_t(0)) + 1;
    }
    return error.empty();
}

static void model_error(httplib::Response& res, int status, const std::string& code, const std::string& message) {
    res.status = status;
    res.set_content(json({{"error", message}, {"code", code}}).dump(), "application/json");
}

httplib::Server::Handler RuntimeRoutes::protect(httplib::Server::Handler handler) {
    ServerRuntime* runtime = &runtime_;
    return [runtime, handler](const httplib::Request& req, httplib::Response& res) {
        const bool generation = req.path == "/sdcpp/v1/img_gen" || req.path == "/sdcpp/v1/vid_gen" ||
                                req.path == "/sdapi/v1/txt2img" || req.path == "/sdapi/v1/img2img" ||
                                req.path == "/v1/images/generations" || req.path == "/v1/images/edits";
        auto& models = *runtime->models;
        std::unique_lock<std::mutex> admission(models.admission_mutex, std::defer_lock);
        if (generation)
            admission.lock();
        std::shared_lock<std::shared_mutex> lock(runtime->models->runtime_mutex, std::try_to_lock);
        if (!lock.owns_lock()) {
            model_error(res, 503, "model_loading", "Model is loading or unloading");
            return;
        }
        if (generation && models.sleeping) {
            lock.unlock();
            std::string error;
            {
                std::unique_lock<std::shared_mutex> wake_lock(models.runtime_mutex);
                if (!wake_model(*runtime, error)) {
                    model_error(res, 422, "model_load_failed", error);
                    return;
                }
            }
            // admission_mutex prevents another idle unload in the gap between
            // the exclusive wake lock and this request's shared lifetime lock.
            lock.lock();
        }
        if (admission.owns_lock())
            admission.unlock();
        if (generation && !runtime->sd_ctx) {
            model_error(res, 503, "model_not_loaded", "No model is loaded");
            return;
        }
        if (generation) {
            ModelActivityGuard activity(*runtime);
            handler(req, res);
        } else {
            handler(req, res);
        }
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
    if (recipe == "@command-line") {
        models.status["recipe"]         = nullptr;
        models.status["origin"]         = "command_line";
        models.status["runtime_recipe"] = models.command_line_recipe;
    }
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

    // Idempotent load: if this exact recipe and resolved selection are already
    // active, report the current status instead of tearing down the context.
    // Comparing full recipe objects (not just the id) also reloads when the
    // catalog was refreshed with modified recipe metadata for the same id.
    // Comparing resolved component paths catches the same file IDs resolving
    // to different files (e.g. a model directory symlink retargeted inside
    // the configured root since the catalog was last scanned).
    if (!unload) {
        json current              = model_status(runtime);
        const json current_recipe = current.value("origin", "") == "command_line" ? json("@command-line") : current.value("recipe", json(nullptr));
        if (current.value("state", "") == "ready" && current_recipe == body.at("recipe") &&
            current.value("models", json::object()) == body.at("models") && models.active_recipe == loaded_recipe &&
            component_paths_match(*runtime.ctx_params, context)) {
            current["already_loaded"] = true;
            mark_model_activity(runtime);
            res.set_content(current.dump(), "application/json");
            return;
        }
    }

    const json recipe    = unload ? json(nullptr) : body.at("recipe");
    const json selection = unload ? json::object() : body.value("models", json::object());
    models.sleeping      = false;
    set_model_status(runtime, "loading", recipe, selection);
    try {
        runtime.sd_ctx = nullptr;
        models.context.reset();
        models.active_recipe        = nullptr;
        *runtime.ctx_params         = unload ? models.base_context : std::move(context);
        *runtime.default_gen_params = unload ? models.base_defaults : std::move(defaults);
        if (!unload) {
            auto params = runtime.ctx_params->to_context_opt();
            models.context.reset(create_sd_context(params));
            if (!models.context) {
                throw std::runtime_error("Model initialization failed; check the server log for component compatibility and memory errors");
            }
            runtime.sd_ctx       = models.context.get();
            models.active_recipe = std::move(loaded_recipe);
        }
        set_model_status(runtime, unload ? "empty" : "ready", recipe, selection);
        mark_model_activity(runtime);
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
    svr.Post("/sdcpp/v1/models/settings", [&runtime](const httplib::Request& req, httplib::Response& res) {
        try {
            const auto body = json::parse(req.body);
            if (!body.is_object() || body.size() != 1 || !body.contains("sleep_idle_seconds") ||
                !body["sleep_idle_seconds"].is_number_integer() || body["sleep_idle_seconds"] < 0 ||
                body["sleep_idle_seconds"] > std::numeric_limits<int>::max())
                throw std::runtime_error("sleep_idle_seconds must be an integer from 0 to 2147483647");
            std::shared_lock<std::shared_mutex> lock(runtime.models->runtime_mutex);
            runtime.models->sleep_idle_seconds = body["sleep_idle_seconds"].get<int>();
            mark_model_activity(runtime);
            res.set_content(model_status(runtime).dump(), "application/json");
        } catch (const std::exception& e) {
            model_error(res, 400, "invalid_model_settings", e.what());
        }
    });
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
