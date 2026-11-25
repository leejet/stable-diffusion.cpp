#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "httplib.h"

#include "async_jobs.h"
#include "common/common.h"
#include "common/resource_owners.hpp"
#include "model_catalog.h"
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

static void parse_args(int argc,
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

    if (!svr_params.resolve_and_validate() ||
        !ctx_params.resolve_and_validate(ctx_params.model_path.empty() && ctx_params.diffusion_model_path.empty() ? METADATA : IMG_GEN) ||
        !default_gen_params.resolve_and_validate(IMG_GEN,
                                                 ctx_params.lora_model_dir,
                                                 ctx_params.hires_upscalers_dir)) {
        print_usage(argv[0], options_vec);
        exit(1);
    }
}

// api_key_matches compares in constant time, so the answer's timing
// doesn't tell how much of a key was right.
static bool api_key_matches(const std::vector<std::string>& keys, std::string_view key) {
    bool found = false;
    for (const auto& k : keys) {
        unsigned char diff = k.size() != key.size();
        for (size_t i = 0; i < k.size(); i++) {
            diff |= static_cast<unsigned char>(k[i] ^ (i < key.size() ? key[i] : 0));
        }
        found |= diff == 0;
    }
    return found;
}

static bool needs_api_key(const httplib::Request& req) {
    return req.method != "OPTIONS" && !(req.method == "GET" && req.path == "/");
}

static bool has_valid_api_key(const httplib::Request& req, const std::vector<std::string>& keys) {
    const auto key = req.get_header_value("X-Api-Key");
    if (!key.empty()) {
        return api_key_matches(keys, key);
    }

    constexpr std::string_view scheme = "bearer";
    const auto auth                   = req.get_header_value("Authorization");
    const auto separator              = auth.find(' ');
    if (separator != scheme.size() ||
        !std::equal(scheme.begin(), scheme.end(), auth.begin(), [](char expected, unsigned char actual) {
            return expected == std::tolower(actual);
        })) {
        return false;
    }
    const auto key_start = auth.find_first_not_of(' ', separator);
    return key_start != std::string::npos && api_key_matches(keys, std::string_view(auth).substr(key_start));
}

class SDHttpServer final : public httplib::Server {
public:
    void close_after_response() {
        close_after_response_ = true;
    }

    bool will_close_after_response() const {
        return close_after_response_;
    }

private:
    // cpp-httplib writes Connection: close for errors but does not stop its
    // keep-alive loop when a handler rejects a request before reading the body.
    bool process_and_close_socket(socket_t socket) override {
        std::string remote_addr;
        std::string local_addr;
        int remote_port = 0;
        int local_port  = 0;
        httplib::detail::get_remote_ip_and_port(socket, remote_addr, remote_port);
        httplib::detail::get_local_ip_and_port(socket, local_addr, local_port);
        const bool result = httplib::detail::process_server_socket(
            svr_sock_, socket, keep_alive_max_count_, keep_alive_timeout_sec_,
            read_timeout_sec_, read_timeout_usec_, write_timeout_sec_, write_timeout_usec_,
            [&](httplib::Stream& stream, bool close_connection, bool& connection_closed) {
                close_after_response_ = false;
                const bool written    = process_request(stream, remote_addr, remote_port, local_addr,
                                                        local_port, close_connection, connection_closed, nullptr);
                connection_closed |= close_after_response_;
                return written;
            });
        httplib::detail::shutdown_socket(socket);
        httplib::detail::close_socket(socket);
        return result;
    }

    static inline thread_local bool close_after_response_ = false;
};

static void configure_server_access(SDHttpServer& svr, const SDSvrParams& params) {
    using HandlerResponse = httplib::Server::HandlerResponse;

    svr.set_pre_routing_handler([&svr, &params](const httplib::Request& req, httplib::Response& res) {
        if (req.method == "OPTIONS") {
            if (req.has_header("Transfer-Encoding") || req.get_header_value_u64("Content-Length") > 0) {
                svr.close_after_response();
            }
            res.status = 204;
            return HandlerResponse::Handled;
        }
        if (!params.api_keys.empty() && needs_api_key(req) && !has_valid_api_key(req, params.api_keys)) {
            res.status = 401;
            res.set_header("WWW-Authenticate", "Bearer realm=\"sd-server\"");
            res.set_content(R"({"error":{"code":401,"message":"Invalid API Key","type":"authentication_error"}})",
                            "application/json");
            svr.close_after_response();
            return HandlerResponse::Handled;
        }
        return HandlerResponse::Unhandled;
    });

    svr.set_post_routing_handler([&svr, &params](const httplib::Request& req, httplib::Response& res) {
        if (svr.will_close_after_response()) {
            res.headers.erase("Connection");
            res.headers.erase("Keep-Alive");
            res.set_header("Connection", "close");
        }
        if (!params.api_keys.empty() && needs_api_key(req)) {
            res.headers.erase("Cache-Control");
            res.set_header("Cache-Control", "no-store");
        }

        const bool preflight = req.method == "OPTIONS";
        res.set_header("Vary", preflight ? "Origin, Access-Control-Request-Headers" : "Origin");

        const auto& allowed = params.cors_origins;
        const auto origin   = req.get_header_value("Origin");
        if (allowed.empty() || std::find(allowed.begin(), allowed.end(), origin) != allowed.end()) {
            res.set_header("Access-Control-Allow-Origin", origin.empty() ? "*" : origin);
            res.set_header("Access-Control-Allow-Credentials", "true");
            res.set_header("Access-Control-Allow-Methods", "*");
            const auto headers = preflight ? req.get_header_value("Access-Control-Request-Headers") : std::string{};
            res.set_header("Access-Control-Allow-Headers", headers.empty() ? "*" : headers);
        }
    });
}

void sd_log_cb(enum sd_log_level_t level, const char* log, void* data) {
    SDSvrParams* svr_params = (SDSvrParams*)data;
    log_print(level, log, svr_params->log_level, svr_params->color);
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
    parse_args(argc, argv, svr_params, ctx_params, default_gen_params);

    LOG_VERBOSE("version: %s", version_string().c_str());
    LOG_VERBOSE("%s", sd_get_system_info());
    LOG_DEBUG("%s", svr_params.to_string().c_str());
    LOG_DEBUG("%s", ctx_params.to_string().c_str());
    LOG_DEBUG("%s", default_gen_params.to_string().c_str());

    if (!ctx_params.register_rpc_devices()) {
        return 1;
    }
    ctx_params.rpc_servers.reset();

    SDOptPtr sd_ctx_params = ctx_params.to_context_opt();
    ServerModels models;
    models.context.reset(create_sd_context(sd_ctx_params));
    if (!models.context) {
        LOG_ERROR("sd_ctx_create failed: %s", sd_get_last_error());
        return 1;
    }
    try {
        configure_server_file_access(models, svr_params, ctx_params);
    } catch (const std::exception& error) {
        LOG_ERROR("Context setup failed: %s", error.what());
        return 1;
    }

    std::mutex sd_ctx_mutex;

    std::vector<LoraEntry> lora_cache;
    std::mutex lora_mutex;
    std::vector<UpscalerEntry> upscaler_cache;
    std::mutex upscaler_mutex;
    AsyncJobManager async_job_manager;
    async_job_manager.max_pending_jobs =
        static_cast<size_t>(svr_params.max_pending_jobs);
    async_job_manager.completed_ttl_seconds =
        static_cast<int64_t>(svr_params.completed_ttl_seconds);
    ServerRuntime runtime = {
        models.context.get(),
        &sd_ctx_mutex,
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

    SDHttpServer svr;
    configure_server_access(svr, svr_params);

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
    register_dgl_endpoints(svr, runtime);
    register_model_endpoints(svr, runtime);

    LOG_INFO("listening on: http://%s:%d\n", svr_params.listen_ip.c_str(), svr_params.listen_port);
    svr.listen(svr_params.listen_ip, svr_params.listen_port);

    {
        std::lock_guard<std::mutex> lock(async_job_manager.mutex);
        async_job_manager.stop = true;
    }
    async_job_manager.cv.notify_all();
    async_worker.join();
    return 0;
}
