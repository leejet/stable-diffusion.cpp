#include "runtime.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <regex>
#include <sstream>

#include "common/common.h"
#include "common/log.h"

namespace fs = std::filesystem;

static const std::string k_base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

std::string base64_encode(const std::vector<uint8_t>& bytes) {
    std::string ret;
    int val  = 0;
    int valb = -6;
    for (uint8_t c : bytes) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            ret.push_back(k_base64_chars[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6) {
        ret.push_back(k_base64_chars[((val << 8) >> (valb + 8)) & 0x3F]);
    }
    while (ret.size() % 4) {
        ret.push_back('=');
    }
    return ret;
}

std::string normalize_output_format(std::string output_format) {
    std::transform(output_format.begin(), output_format.end(), output_format.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return output_format;
}

bool assign_output_options(ImgGenJobRequest& request,
                           std::string output_format,
                           int output_compression,
                           bool allow_webp,
                           std::string& error_message) {
    request.output_format      = normalize_output_format(std::move(output_format));
    request.output_compression = std::clamp(output_compression, 0, 100);

    const bool valid_format = request.output_format == "png" ||
                              request.output_format == "jpeg" ||
                              (allow_webp && request.output_format == "webp");
    if (!valid_format) {
        error_message = allow_webp
                            ? "invalid output_format, must be one of [png, jpeg, webp]"
                            : "invalid output_format, must be one of [png, jpeg]";
        return false;
    }

    return true;
}

ArgOptions SDSvrParams::get_options() {
    ArgOptions options;

    options.string_options = {
        {"-l", "--listen-ip", "server listen ip (default: 127.0.0.1)", &listen_ip},
        {"", "--serve-html-path", "path to HTML file to serve at root (optional)", &serve_html_path},
    };

    options.int_options = {
        {"", "--listen-port", "server listen port (default: 1234)", &listen_port},
    };

    options.bool_options = {
        {"-v", "--verbose", "print extra info", true, &verbose},
        {"", "--color", "colors the logging tags according to level", true, &color},
    };

    auto on_help_arg = [&](int, const char**, int) {
        normal_exit = true;
        return -1;
    };

    options.manual_options = {
        {"-h", "--help", "show this help message and exit", on_help_arg},
    };
    return options;
}

bool SDSvrParams::validate() {
    if (listen_ip.empty()) {
        LOG_ERROR("error: the following arguments are required: listen_ip");
        return false;
    }

    if (listen_port < 0 || listen_port > 65535) {
        LOG_ERROR("error: listen_port should be in the range [0, 65535]");
        return false;
    }

    if (!serve_html_path.empty() && !fs::exists(serve_html_path)) {
        LOG_ERROR("error: serve_html_path file does not exist: %s", serve_html_path.c_str());
        return false;
    }
    return true;
}

bool SDSvrParams::resolve_and_validate() {
    if (!validate()) {
        return false;
    }
    return true;
}

std::string SDSvrParams::to_string() const {
    std::ostringstream oss;
    oss << "SDSvrParams {\n"
        << "  listen_ip: " << listen_ip << ",\n"
        << "  listen_port: \"" << listen_port << "\",\n"
        << "  serve_html_path: \"" << serve_html_path << "\",\n"
        << "}";
    return oss.str();
}

void refresh_lora_cache(ServerRuntime& rt) {
    std::vector<LoraEntry> new_cache;

    fs::path lora_dir = rt.ctx_params->lora_model_dir;
    if (fs::exists(lora_dir) && fs::is_directory(lora_dir)) {
        auto is_lora_ext = [](const fs::path& p) {
            auto ext = p.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            return ext == ".gguf" || ext == ".pt" || ext == ".pth" || ext == ".safetensors";
        };

        for (auto& entry : fs::recursive_directory_iterator(lora_dir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            const fs::path& p = entry.path();
            if (!is_lora_ext(p)) {
                continue;
            }

            LoraEntry lora_entry;
            lora_entry.name     = p.stem().u8string();
            lora_entry.fullpath = p.u8string();
            std::string rel     = p.lexically_relative(lora_dir).u8string();
            std::replace(rel.begin(), rel.end(), '\\', '/');
            lora_entry.path = rel;

            new_cache.push_back(std::move(lora_entry));
        }
    }

    std::sort(new_cache.begin(), new_cache.end(), [](const LoraEntry& a, const LoraEntry& b) {
        return a.path < b.path;
    });

    {
        std::lock_guard<std::mutex> lock(*rt.lora_mutex);
        *rt.lora_cache = std::move(new_cache);
    }
}

std::string get_lora_full_path(ServerRuntime& rt, const std::string& path) {
    std::lock_guard<std::mutex> lock(*rt.lora_mutex);
    auto it = std::find_if(rt.lora_cache->begin(), rt.lora_cache->end(),
                           [&](const LoraEntry& entry) { return entry.path == path; });
    return it != rt.lora_cache->end() ? it->fullpath : "";
}

int64_t unix_timestamp_now() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}
