// main.cpp
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

#include "httplib.h"
#include "stable-diffusion.h"

#include "common/common.hpp"

namespace fs = std::filesystem;

// ----------------------- helpers -----------------------
static const std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

std::string base64_encode(const std::vector<uint8_t>& bytes) {
    std::string ret;
    int val = 0, valb = -6;
    for (uint8_t c : bytes) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            ret.push_back(base64_chars[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6)
        ret.push_back(base64_chars[((val << 8) >> (valb + 8)) & 0x3F]);
    while (ret.size() % 4)
        ret.push_back('=');
    return ret;
}

std::string iso_timestamp_now() {
    using namespace std::chrono;
    auto now      = system_clock::now();
    std::time_t t = system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _MSC_VER
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

struct SDSvrParams {
    std::string listen_ip = "127.0.0.1";
    int listen_port       = 1234;
    bool normal_exit      = false;
    bool verbose          = false;
    bool color            = false;

    ArgOptions get_options() {
        ArgOptions options;

        options.string_options = {
            {"-l",
             "--listen-ip",
             "server listen ip (default: 127.0.0.1)",
             &listen_ip}};

        options.int_options = {
            {"",
             "--listen-port",
             "server listen ip (default: 1234)",
             &listen_port},
        };

        options.bool_options = {
            {"-v",
             "--verbose",
             "print extra info",
             true, &verbose},
            {"",
             "--color",
             "colors the logging tags according to level",
             true, &color},
        };

        auto on_help_arg = [&](int argc, const char** argv, int index) {
            normal_exit = true;
            return -1;
        };

        options.manual_options = {
            {"-h",
             "--help",
             "show this help message and exit",
             on_help_arg},
        };
        return options;
    };

    bool process_and_check() {
        if (listen_ip.empty()) {
            fprintf(stderr, "error: the following arguments are required: listen_ip\n");
            return false;
        }

        if (listen_port < 0 || listen_port > 65535) {
            fprintf(stderr, "error: listen_port should be in the range [0, 65535]\n");
            return false;
        }
        return true;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "SDSvrParams {\n"
            << "  listen_ip: " << listen_ip << ",\n"
            << "  listen_port: \"" << listen_port << "\",\n"
            << "}";
        return oss.str();
    }
};

void print_usage(int argc, const char* argv[], const std::vector<ArgOptions>& options_list) {
    std::cout << "Usage: " << argv[0] << " [options]\n\n";
    std::cout << "Svr Options:\n";
    options_list[0].print();
    std::cout << "\nContext Options:\n";
    options_list[1].print();
    std::cout << "\nDefault Generation Options:\n";
    options_list[2].print();
}

void parse_args(int argc, const char** argv, SDSvrParams& svr_params, SDContextParams& ctx_params, SDGenerationParams& default_gen_params) {
    std::vector<ArgOptions> options_vec = {svr_params.get_options(), ctx_params.get_options(), default_gen_params.get_options()};

    if (!parse_options(argc, argv, options_vec)) {
        print_usage(argc, argv, options_vec);
        exit(svr_params.normal_exit ? 0 : 1);
    }

    if (!svr_params.process_and_check() || !ctx_params.process_and_check(IMG_GEN) || !default_gen_params.process_and_check(IMG_GEN)) {
        print_usage(argc, argv, options_vec);
        exit(1);
    }
}

std::string extract_and_remove_sd_cpp_extra_args(std::string& text) {
    std::regex re("<sd_cpp_extra_args>(.*?)</sd_cpp_extra_args>");
    std::smatch match;

    std::string extracted;
    if (std::regex_search(text, match, re)) {
        extracted = match[1].str();
        text      = std::regex_replace(text, re, "");
    }
    return extracted;
}

enum class ImageFormat { JPEG,
                         PNG };

std::vector<uint8_t> write_image_to_vector(
    ImageFormat format,
    const uint8_t* image,
    int width,
    int height,
    int channels,
    int quality = 90) {
    std::vector<uint8_t> buffer;

    auto write_func = [&buffer](void* context, void* data, int size) {
        uint8_t* src = reinterpret_cast<uint8_t*>(data);
        buffer.insert(buffer.end(), src, src + size);
    };

    struct ContextWrapper {
        decltype(write_func)& func;
    } ctx{write_func};

    auto c_func = [](void* context, void* data, int size) {
        auto* wrapper = reinterpret_cast<ContextWrapper*>(context);
        wrapper->func(context, data, size);
    };

    int result = 0;
    switch (format) {
        case ImageFormat::JPEG:
            result = stbi_write_jpg_to_func(c_func, &ctx, width, height, channels, image, quality);
            break;
        case ImageFormat::PNG:
            result = stbi_write_png_to_func(c_func, &ctx, width, height, channels, image, width * channels);
            break;
        default:
            throw std::runtime_error("invalid image format");
    }

    if (!result) {
        throw std::runtime_error("write imgage to mem failed");
    }

    return buffer;
}

/* Enables Printing the log level tag in color using ANSI escape codes */
void sd_log_cb(enum sd_log_level_t level, const char* log, void* data) {
    SDSvrParams* svr_params = (SDSvrParams*)data;
    int tag_color;
    const char* level_str;
    FILE* out_stream = (level == SD_LOG_ERROR) ? stderr : stdout;

    if (!log || (!svr_params->verbose && level <= SD_LOG_DEBUG)) {
        return;
    }

    switch (level) {
        case SD_LOG_DEBUG:
            tag_color = 37;
            level_str = "DEBUG";
            break;
        case SD_LOG_INFO:
            tag_color = 34;
            level_str = "INFO";
            break;
        case SD_LOG_WARN:
            tag_color = 35;
            level_str = "WARN";
            break;
        case SD_LOG_ERROR:
            tag_color = 31;
            level_str = "ERROR";
            break;
        default: /* Potential future-proofing */
            tag_color = 33;
            level_str = "?????";
            break;
    }

    if (svr_params->color == true) {
        fprintf(out_stream, "\033[%d;1m[%-5s]\033[0m ", tag_color, level_str);
    } else {
        fprintf(out_stream, "[%-5s] ", level_str);
    }
    fputs(log, out_stream);
    fflush(out_stream);
}

int main(int argc, const char** argv) {
    SDSvrParams svr_params;
    SDContextParams ctx_params;
    SDGenerationParams default_gen_params;
    parse_args(argc, argv, svr_params, ctx_params, default_gen_params);

    sd_set_log_callback(sd_log_cb, (void*)&svr_params);

    if (svr_params.verbose) {
        printf("%s", sd_get_system_info());
        printf("%s\n", svr_params.to_string().c_str());
        printf("%s\n", ctx_params.to_string().c_str());
        printf("%s\n", default_gen_params.to_string().c_str());
    }

    sd_ctx_params_t sd_ctx_params = ctx_params.to_sd_ctx_params_t(false, false, false);
    sd_ctx_t* sd_ctx              = new_sd_ctx(&sd_ctx_params);

    if (sd_ctx == nullptr) {
        printf("new_sd_ctx_t failed\n");
        return 1;
    }

    std::mutex sd_ctx_mutex;

    httplib::Server svr;
    
    svr.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
        std::string origin = req.get_header_value("Origin");
        if (origin.empty()) {
            origin = "*";
        }
        res.set_header("Access-Control-Allow-Origin", origin);
        res.set_header("Access-Control-Allow-Credentials", "true");
        res.set_header("Access-Control-Allow-Methods", "*");
        res.set_header("Access-Control-Allow-Headers", "*");
        
        if (req.method == "OPTIONS") {
            res.status = 204;
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    // health
    svr.Get("/", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"ok":true,"service":"sd-cpp-http"})", "application/json");
    });

    // models endpoint (minimal)
    svr.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        json r;
        r["data"] = json::array();
        r["data"].push_back({{"id", "sd-cpp-local"}, {"object", "model"}, {"owned_by", "local"}});
        res.set_content(r.dump(), "application/json");
    });

    // core endpoint: /v1/images/generations
    svr.Post("/v1/images/generations", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            if (req.body.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"empty body"})", "application/json");
                return;
            }

            json j                    = json::parse(req.body);
            std::string prompt        = j.value("prompt", "");
            int n                     = std::max(1, j.value("n", 1));
            std::string size          = j.value("size", "");
            std::string output_format = j.value("output_format", "png");
            int output_compression    = j.value("output_compression", 100);
            int width                 = 512;
            int height                = 512;
            if (!size.empty()) {
                auto pos = size.find('x');
                if (pos != std::string::npos) {
                    try {
                        width  = std::stoi(size.substr(0, pos));
                        height = std::stoi(size.substr(pos + 1));
                    } catch (...) {
                    }
                }
            }

            if (prompt.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"prompt required"})", "application/json");
                return;
            }

            std::string sd_cpp_extra_args_str = extract_and_remove_sd_cpp_extra_args(prompt);

            if (output_format != "png" && output_format != "jpeg") {
                res.status = 400;
                res.set_content(R"({"error":"invalid output_format, must be one of [png, jpeg]"})", "application/json");
                return;
            }
            if (n <= 0)
                n = 1;
            if (n > 8)
                n = 8;  // safety
            if (output_compression > 100) {
                output_compression = 100;
            }
            if (output_compression < 0) {
                output_compression = 0;
            }

            json out;
            out["created"]       = iso_timestamp_now();
            out["data"]          = json::array();
            out["output_format"] = output_format;

            SDGenerationParams gen_params = default_gen_params;
            gen_params.prompt                    = prompt;
            gen_params.width                     = width;
            gen_params.height                    = height;
            gen_params.batch_count               = n;

            if (!sd_cpp_extra_args_str.empty() && !gen_params.from_json_str(sd_cpp_extra_args_str)) {
                res.status = 400;
                res.set_content(R"({"error":"invalid sd_cpp_extra_args"})", "application/json");
                return;
            }

            if (!gen_params.process_and_check(IMG_GEN)) {
                res.status = 400;
                res.set_content(R"({"error":"invalid params"})", "application/json");
                return;
            }

            if (svr_params.verbose) {
                printf("%s\n", gen_params.to_string().c_str());
            }

            sd_image_t init_image    = {(uint32_t)gen_params.width, (uint32_t)gen_params.height, 3, nullptr};
            sd_image_t control_image = {(uint32_t)gen_params.width, (uint32_t)gen_params.height, 3, nullptr};
            sd_image_t mask_image    = {(uint32_t)gen_params.width, (uint32_t)gen_params.height, 1, nullptr};
            std::vector<sd_image_t> pmid_images;

            sd_img_gen_params_t img_gen_params = {
                gen_params.prompt.c_str(),
                gen_params.negative_prompt.c_str(),
                gen_params.clip_skip,
                init_image,
                nullptr,
                0,
                gen_params.auto_resize_ref_image,
                gen_params.increase_ref_index,
                mask_image,
                gen_params.width,
                gen_params.height,
                gen_params.sample_params,
                gen_params.strength,
                gen_params.seed,
                gen_params.batch_count,
                control_image,
                gen_params.control_strength,
                {
                    pmid_images.data(),
                    (int)pmid_images.size(),
                    gen_params.pm_id_embed_path.c_str(),
                    gen_params.pm_style_strength,
                },  // pm_params
                ctx_params.vae_tiling_params,
                gen_params.easycache_params,
            };

            sd_image_t* results = nullptr;
            int num_results     = 0;

            {
                std::lock_guard<std::mutex> lock(sd_ctx_mutex);
                results     = generate_image(sd_ctx, &img_gen_params);
                num_results = gen_params.batch_count;
            }

            for (int i = 0; i < num_results; i++) {
                if (results[i].data == nullptr) {
                    continue;
                }
                auto image_bytes = write_image_to_vector(output_format == "jpeg" ? ImageFormat::JPEG : ImageFormat::PNG,
                                                         results[i].data,
                                                         results[i].width,
                                                         results[i].height,
                                                         results[i].channel,
                                                         output_compression);
                if (image_bytes.empty()) {
                    printf("write image to mem failed\n");
                    continue;
                }

                // base64 encode
                std::string b64 = base64_encode(image_bytes);
                json item;
                item["b64_json"] = b64;
                out["data"].push_back(item);
            }

            res.set_content(out.dump(), "application/json");
            res.status = 200;

        } catch (const std::exception& e) {
            res.status = 500;
            json err;
            err["error"]   = "server_error";
            err["message"] = e.what();
            res.set_content(err.dump(), "application/json");
        }
    });

    printf("listening on: %s:%d\n", svr_params.listen_ip.c_str(), svr_params.listen_port);
    svr.listen(svr_params.listen_ip, svr_params.listen_port);

    // cleanup
    free_sd_ctx(sd_ctx);
    return 0;
}
