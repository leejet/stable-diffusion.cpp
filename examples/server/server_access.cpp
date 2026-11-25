#include "server_access.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string_view>

#include "runtime.h"

static bool is_registered_option(const std::vector<ArgOptions>& groups, std::string_view name) {
    const auto contains = [name](const auto& options) {
        return std::any_of(options.begin(), options.end(), [name](const auto& option) {
            return name == option.short_name || name == option.long_name;
        });
    };
    return std::any_of(groups.begin(), groups.end(), [&](const ArgOptions& options) {
        return contains(options.string_options) || contains(options.int_options) ||
               contains(options.float_options) || contains(options.bool_options) ||
               contains(options.manual_options);
    });
}

static bool append_option_list(const char* name, const char* value, const char* entry_name, std::vector<std::string>& values) {
    std::istringstream input(value);
    const auto previous_count = values.size();
    for (std::string entry; std::getline(input, entry, ',');) {
        if (!entry.empty()) {
            values.push_back(entry);
        }
    }
    if (values.size() == previous_count) {
        LOG_ERROR("error: %s needs at least one %s", name, entry_name);
        return false;
    }
    return true;
}

void SDSvrParams::add_access_options(ArgOptions& options, const std::vector<ArgOptions>& groups) {
    const auto add_option = [&](const char* name, const char* description, auto apply) {
        const auto parse_value = [&groups, apply](int argc, const char** argv, int index, bool& error_reported) {
            const int value_index = index + 1;
            if (value_index >= argc || !argv[value_index][0] || is_registered_option(groups, argv[value_index])) {
                LOG_ERROR("error: missing value for argument \"%s\"", argv[index]);
                error_reported = true;
                return -1;
            }
            return apply(value_index, argv) ? 1 : -1;
        };
        options.manual_options.push_back({"", name, description, parse_value});
    };
    add_option("--api-key", "API key clients must send (Authorization: Bearer KEY or X-Api-Key); comma-separated for several",
               [this](int index, const char** argv) {
                   return append_option_list(argv[index - 1], argv[index], "key", api_keys);
               });
    add_option("--api-key-file", "file with API keys, one per line (keeps them off the command line)",
               [this](int index, const char** argv) {
                   api_key_file = argv_to_utf8(index, argv);
                   return true;
               });
    add_option("--cors-origins", "additional web origins allowed to make cross-origin requests, comma-separated (default: none)",
               [this](int index, const char** argv) {
                   return append_option_list(argv[index - 1], argv[index], "origin", cors_origins);
               });
}

bool SDSvrParams::load_api_keys() {
    if (api_key_file.empty()) {
        return true;
    }
    std::ifstream file(std::filesystem::u8path(api_key_file));
    if (!file) {
        LOG_ERROR("error: can't read api_key_file: %s", api_key_file.c_str());
        return false;
    }
    const auto previous_count = api_keys.size();
    for (std::string line; std::getline(file, line);) {
        const auto last = line.find_last_not_of(" \t\r\n");
        if (last != std::string::npos) {
            const auto first = line.find_first_not_of(" \t");
            api_keys.push_back(line.substr(first, last - first + 1));
        }
    }
    if (api_keys.size() == previous_count) {
        LOG_ERROR("error: api_key_file has no keys: %s", api_key_file.c_str());
        return false;
    }
    return true;
}

static bool is_web_ui_request(const httplib::Request& req) {
    return req.method == "GET" && req.path == "/";
}

static bool requires_api_key(const SDSvrParams& params, const httplib::Request& req) {
    return !params.api_keys.empty() && req.method != "OPTIONS" && !is_web_ui_request(req);
}

static bool api_key_matches(const std::vector<std::string>& keys, std::string_view key) {
    bool found = false;
    // Compare every byte of every configured key without a prefix-dependent early return.
    for (const auto& configured_key : keys) {
        unsigned char diff = configured_key.size() != key.size();
        for (size_t i = 0; i < configured_key.size(); i++) {
            diff |= static_cast<unsigned char>(configured_key[i] ^ (i < key.size() ? key[i] : 0));
        }
        found |= diff == 0;
    }
    return found;
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

static bool is_allowed_origin(const std::vector<std::string>& allowed, std::string_view origin) {
    return !origin.empty() && std::find(allowed.begin(), allowed.end(), origin) != allowed.end();
}

static void set_cors_headers(const httplib::Request& req, httplib::Response& res, const std::vector<std::string>& allowed) {
    const bool preflight = req.method == "OPTIONS";
    std::string vary     = "Origin, Sec-Fetch-Site";
    if (preflight)
        vary += ", Access-Control-Request-Headers";
    else if (is_web_ui_request(req))
        vary += ", Sec-Fetch-Mode, Sec-Fetch-Dest";
    res.set_header("Vary", vary);

    const auto origin = req.get_header_value("Origin");
    if (!is_allowed_origin(allowed, origin)) {
        return;
    }
    res.set_header("Access-Control-Allow-Origin", origin);
    res.set_header("Access-Control-Allow-Credentials", "true");
    res.set_header("Access-Control-Allow-Methods", "GET, HEAD, POST, PUT, PATCH, DELETE, OPTIONS");
    const auto headers = preflight ? req.get_header_value("Access-Control-Request-Headers") : std::string{};
    res.set_header("Access-Control-Allow-Headers", headers.empty() ? "*" : headers);
}

static bool is_server_host(std::string_view host, const std::string& listen_ip) {
    if (host.empty())
        return false;
    if (host.back() != ']') {
        const auto port_separator = host.rfind(':');
        if (port_separator != std::string_view::npos) {
            const auto port = host.substr(port_separator + 1);
            if (port.empty() || port.find_first_not_of("0123456789") != std::string_view::npos)
                return false;
            host = host.substr(0, port_separator);
        }
    }
    if (host == "localhost" || host == listen_ip)
        return true;
    if (host.size() > 2 && host.front() == '[' && host.back() == ']') {
        const std::string ipv6(host.substr(1, host.size() - 2));
        in6_addr address{};
        return inet_pton(AF_INET6, ipv6.c_str(), &address) == 1;
    }
    const std::string ipv4(host);
    in_addr address{};
    return inet_pton(AF_INET, ipv4.c_str(), &address) == 1;
}

static bool is_allowed_request_origin(const httplib::Request& req, const SDSvrParams& params) {
    if (req.get_header_value_count("Host") != 1 ||
        req.get_header_value_count("Origin") > 1 || req.get_header_value_count("Sec-Fetch-Site") > 1)
        return false;

    // Same-origin browser headers alone do not distinguish a DNS-rebound hostname.
    const auto host         = req.get_header_value("Host");
    const auto http_origin  = "http://" + host;
    const auto https_origin = "https://" + host;
    if (!is_server_host(host, params.listen_ip) && !is_allowed_origin(params.cors_origins, http_origin) &&
        !is_allowed_origin(params.cors_origins, https_origin))
        return false;

    if (is_web_ui_request(req) && req.get_header_value_count("Sec-Fetch-Mode") == 1 &&
        req.get_header_value_count("Sec-Fetch-Dest") == 1 && req.get_header_value("Sec-Fetch-Mode") == "navigate" &&
        req.get_header_value("Sec-Fetch-Dest") == "document")
        return true;

    const auto origin = req.get_header_value("Origin");
    if (is_allowed_origin(params.cors_origins, origin))
        return true;
    const auto site = req.get_header_value("Sec-Fetch-Site");
    return (site.empty() || site == "same-origin" || site == "none") &&
           (origin.empty() || origin == http_origin);
}

SDHttpServer::SDHttpServer(const SDSvrParams& params) {
    set_pre_routing_handler([&params](const httplib::Request& req, httplib::Response& res) {
        if (!is_allowed_request_origin(req, params)) {
            res.status = 403;
            res.set_content(R"({"error":"Request origin is not allowed"})", "application/json");
            close_after_response_ = true;
            return HandlerResponse::Handled;
        }
        if (req.method == "OPTIONS") {
            close_after_response_ = req.has_header("Transfer-Encoding") || req.get_header_value_u64("Content-Length") > 0;
            res.status            = 204;
            return HandlerResponse::Handled;
        }
        if (requires_api_key(params, req) && !has_valid_api_key(req, params.api_keys)) {
            res.status = 401;
            res.set_header("WWW-Authenticate", "Bearer realm=\"sd-server\"");
            res.set_content(R"({"error":{"code":401,"message":"Invalid API Key","type":"authentication_error"}})",
                            "application/json");
            close_after_response_ = true;
            return HandlerResponse::Handled;
        }
        return HandlerResponse::Unhandled;
    });

    set_post_routing_handler([&params](const httplib::Request& req, httplib::Response& res) {
        if (close_after_response_) {
            res.headers.erase("Connection");
            res.headers.erase("Keep-Alive");
            res.set_header("Connection", "close");
        }
        if (requires_api_key(params, req)) {
            res.headers.erase("Cache-Control");
            res.set_header("Cache-Control", "no-store");
        }
        res.set_header("Cross-Origin-Resource-Policy", "same-origin");
        set_cors_headers(req, res, params.cors_origins);
    });
}

thread_local bool SDHttpServer::close_after_response_ = false;

// cpp-httplib can send Connection: close without stopping its keep-alive loop.
// Stop the loop after early responses so unread request bytes are discarded.
bool SDHttpServer::process_and_close_socket(socket_t socket) {
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
