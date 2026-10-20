#pragma once

#include <string>

#include "httplib.h"
#include "runtime.h"

void register_index_endpoints(httplib::Server& svr, const SDSvrParams& svr_params, const std::string& index_html);
void register_openai_api_endpoints(httplib::Server& svr, ServerRuntime& rt);
void register_sdapi_endpoints(httplib::Server& svr, ServerRuntime& rt);
void register_sdcpp_api_endpoints(httplib::Server& svr, ServerRuntime& rt);
void register_model_endpoints(httplib::Server& svr, ServerRuntime& rt);

class RuntimeRoutes {
    httplib::Server& server_;
    ServerRuntime& runtime_;

    httplib::Server::Handler protect(httplib::Server::Handler handler);

public:
    RuntimeRoutes(httplib::Server& server, ServerRuntime& runtime)
        : server_(server), runtime_(runtime) {}

    void get(const std::string& pattern, httplib::Server::Handler handler);
    void post(const std::string& pattern, httplib::Server::Handler handler);
};
