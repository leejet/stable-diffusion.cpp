#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "httplib.h"

#include "async_jobs.h"
#include "common/common.h"
#include "common/profile.h"
#include "common/resource_owners.hpp"
#include "model_catalog.h"
#include "recovery.h"
#include "routes.h"
#include "runtime.h"

#ifdef HAVE_INDEX_HTML
#include "frontend/dist/gen_index_html.h"
#endif

static void print_usage(const char* argv0, const std::vector<ArgOptions>& options_list) {
    std::cout << version_string() << "\n";
    std::cout << "Usage: " << argv0 << " [options]\n\n";
    std::cout << "Svr Options:\n";
    options_list[0].print();
    std::cout << "\nContext Options:\n";
    options_list[1].print();
    std::cout << "\nDefault Generation Options:\n";
    options_list[2].print();
}

static sd_profile::Level parse_args(int argc,
                                    const char** argv,
                                    SDSvrParams& svr_params,
                                    SDContextParams& ctx_params,
                                    SDGenerationParams& default_gen_params) {
    std::vector<ArgOptions> options_vec = {
        svr_params.get_options(),
        ctx_params.get_options(),
        default_gen_params.get_options(),
    };

    if (!parse_options(argc, argv, options_vec)) {
        print_usage(argv[0], options_vec);
        exit(svr_params.normal_exit ? 0 : 1);
    }

    log_level = svr_params.log_level;
    log_color = svr_params.color;

    if (ctx_params.list_devices) {
        exit(ctx_params.print_devices() ? 0 : 1);
    }

    const bool random_seed_requested = default_gen_params.seed < 0;

    sd_profile::Level profile_level = sd_profile::Level::MAX;
    if (!svr_params.resolve_and_validate() ||
        !ctx_params.resolve_and_validate(IMG_GEN, /*require_model=*/false) ||
        !default_gen_params.resolve_and_validate(IMG_GEN,
                                                 ctx_params.lora_model_dir,
                                                 ctx_params.hires_upscalers_dir) ||
        !sd_profile::apply_from_options(ctx_params, default_gen_params, profile_level)) {
        print_usage(argv[0], options_vec);
        exit(1);
    }

    if (random_seed_requested) {
        default_gen_params.seed = -1;
    }
    return profile_level;
}

void sd_log_cb(enum sd_log_level_t level, const char* log, void* data) {
    SDSvrParams* svr_params = (SDSvrParams*)data;
    log_print(level, log, svr_params->log_level, svr_params->color);
}

static bool equals_constant_time(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) {
        return false;
    }
    unsigned char diff = 0;
    for (size_t i = 0; i < a.size(); i++) {
        diff |= static_cast<unsigned char>(a[i] ^ b[i]);
    }
    return diff == 0;
}

static bool request_authorized(const httplib::Request& req, const std::string& api_key) {
    const std::string bearer = req.get_header_value("Authorization");
    if (bearer.rfind("Bearer ", 0) == 0 && equals_constant_time(bearer.substr(7), api_key)) {
        return true;
    }
    return equals_constant_time(req.get_header_value("X-API-Key"), api_key);
}

static bool origin_allowed(const std::string& allowed_csv, const std::string& origin) {
    if (allowed_csv.empty()) {
        return true;
    }
    size_t start = 0;
    while (start <= allowed_csv.size()) {
        size_t end      = allowed_csv.find(',', start);
        size_t len      = (end == std::string::npos ? allowed_csv.size() : end) - start;
        std::string one = allowed_csv.substr(start, len);
        one.erase(0, one.find_first_not_of(" \t"));
        one.erase(one.find_last_not_of(" \t") + 1);
        if (!one.empty() && one == origin) {
            return true;
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return false;
}

int main(int argc, const char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--version") {
        std::cout << version_string() << "\n";
        return EXIT_SUCCESS;
    }
    SDSvrParams svr_params;
    SDContextParams ctx_params;
    // Preserve the CLI default while disabling cwd scans for sd-server.
    ctx_params.lora_model_dir.clear();
    SDGenerationParams default_gen_params;

    sd_set_log_callback(sd_log_cb, (void*)&svr_params);
    const sd_profile::Level profile_level = parse_args(argc, argv, svr_params, ctx_params, default_gen_params);

    LOG_VERBOSE("version: %s", version_string().c_str());
    LOG_VERBOSE("%s", sd_get_system_info());
    LOG_DEBUG("%s", svr_params.to_string().c_str());
    LOG_DEBUG("%s", ctx_params.to_string().c_str());
    LOG_DEBUG("%s", default_gen_params.to_string().c_str());

    const bool model_requested = !ctx_params.model_path.empty() || !ctx_params.diffusion_model_path.empty();
    ServerModels models;

    ContextKeeper::Settings keeper_settings;
    keeper_settings.degradation      = !svr_params.disable_degradation;
    keeper_settings.cooldown_seconds = svr_params.degradation_cooldown;
    ContextKeeper keeper(ctx_params, profile_level, keeper_settings);

    if (model_requested) {
        const auto load_started = std::chrono::steady_clock::now();
        GenerationFailure failure;
        if (!keeper.start(failure)) {
            LOG_ERROR("the model could not be loaded: '%s' (%s: %s); check the model file and the logs above, then restart the server",
                      !ctx_params.model_path.empty() ? ctx_params.model_path.c_str()
                                                     : ctx_params.diffusion_model_path.c_str(),
                      sd_error_code_name(failure.code),
                      failure.message.c_str());
            return 1;
        }
        models.status["load_elapsed_seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - load_started).count();
        if (svr_params.disable_degradation) {
            LOG_INFO("a request that runs out of memory is answered with an error (--disable-degradation)");
        } else if (!keeper.can_step_down()) {
            LOG_INFO("a request that runs out of memory is answered with an error: no profile is lighter than %s",
                     keeper.describe_ladder().c_str());
        } else {
            LOG_INFO("a request that runs out of memory makes the server step down: %s", keeper.describe_ladder().c_str());
        }
        const std::string fallback_devices = keeper.describe_devices();
        if (!fallback_devices.empty()) {
            LOG_INFO("a device that fails is replaced by the next one before the cpu takes over: %s", fallback_devices.c_str());
        }
        const nlohmann::json health = keeper.health_json();
        LOG_INFO("the model runs on device '%s' with profile %s",
                 health["device"].get<std::string>().c_str(),
                 health["profile"].get<std::string>().c_str());
        if (!health["cpu_modules"].empty()) {
            LOG_WARN("the cpu does the work of: %s (the GPUs do the rest)", health["cpu_modules"].dump().c_str());
        }
    } else {
        LOG_WARN(
            "starting without a model: device and upscale endpoints work, "
            "generation endpoints will report that no model is loaded "
            "(load one from the model catalog, or restart with -m/--diffusion-model)");
    }

    std::vector<LoraEntry> lora_cache;
    std::mutex lora_mutex;
    std::vector<UpscalerEntry> upscaler_cache;
    std::mutex upscaler_mutex;
    AsyncJobManager async_job_manager;
    ServerRuntime runtime = {
        &keeper,
        &svr_params,
        &ctx_params,
        &default_gen_params,
        &lora_cache,
        &lora_mutex,
        &upscaler_cache,
        &upscaler_mutex,
        &async_job_manager,
        &models,
    };

    try {
        initialize_server_models(runtime);
    } catch (const std::exception& e) {
        LOG_ERROR("model catalog initialization failed: %s", e.what());
        return 1;
    }

    std::thread async_worker(async_job_worker, std::ref(runtime));

    httplib::Server svr;

    svr.set_pre_routing_handler([&svr_params](const httplib::Request& req, httplib::Response& res) {
        std::string origin = req.get_header_value("Origin");
        if (origin.empty()) {
            origin = "*";
        }
        if (origin_allowed(svr_params.cors_origins, origin)) {
            res.set_header("Access-Control-Allow-Origin", origin);
            res.set_header("Access-Control-Allow-Credentials", "true");
            res.set_header("Access-Control-Allow-Methods", "*");
            res.set_header("Access-Control-Allow-Headers", "*");
        }

        if (req.method == "OPTIONS") {
            res.status = 204;
            return httplib::Server::HandlerResponse::Handled;
        }
        if (!svr_params.api_key.empty() && !request_authorized(req, svr_params.api_key)) {
            res.status = 401;
            res.set_content(R"({"error":"unauthorized: missing or invalid API key"})", "application/json");
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    std::string index_html;
#ifdef HAVE_INDEX_HTML
    index_html.assign(reinterpret_cast<const char*>(index_html_bytes), index_html_size);
#else
    index_html = "Stable Diffusion Server is running";
#endif
    register_index_endpoints(svr, svr_params, index_html);
    register_openai_api_endpoints(svr, runtime);
    register_sdapi_endpoints(svr, runtime);
    register_sdcpp_api_endpoints(svr, runtime);
    register_model_endpoints(svr, runtime);

    LOG_INFO("listening on: http://%s:%d\n", svr_params.listen_ip.c_str(), svr_params.listen_port);
    int exit_code = 0;
    try {
        if (!svr.listen(svr_params.listen_ip, svr_params.listen_port)) {
            LOG_ERROR("the server could not listen on %s:%d; check that the port is free and the address is valid",
                      svr_params.listen_ip.c_str(),
                      svr_params.listen_port);
            exit_code = 1;
        }
    } catch (const std::exception& error) {
        // The worker threads of the HTTP server could not be created, typically for lack of memory.
        LOG_ERROR("the HTTP server could not run: %s", error.what());
        exit_code = 1;
    }

    {
        std::lock_guard<std::mutex> lock(async_job_manager.mutex);
        async_job_manager.stop = true;
    }
    async_job_manager.cv.notify_all();
    async_worker.join();
    return exit_code;
}
