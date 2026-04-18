#pragma once

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include <json.hpp>
#include "common/common.h"
#include "common/resource_owners.hpp"
#include "stable-diffusion.h"

using json = nlohmann::json;

struct ArgOptions;
struct SDContextParams;
struct AsyncJobManager;

struct SDSvrParams {
    std::string listen_ip = "127.0.0.1";
    int listen_port       = 1234;
    std::string serve_html_path;
    bool normal_exit = false;
    bool verbose     = false;
    bool color       = false;

    ArgOptions get_options();
    bool validate();
    bool resolve_and_validate();
    std::string to_string() const;
};

struct LoraEntry {
    std::string name;
    std::string path;
    std::string fullpath;
};

struct ServerRuntime {
    sd_ctx_t* sd_ctx;
    std::mutex* sd_ctx_mutex;
    const SDSvrParams* svr_params;
    const SDContextParams* ctx_params;
    const SDGenerationParams* default_gen_params;
    std::vector<LoraEntry>* lora_cache;
    std::mutex* lora_mutex;
    AsyncJobManager* async_job_manager;
};

struct ImgGenJobRequest {
    SDGenerationParams gen_params;
    std::string output_format = "png";
    int output_compression    = 100;

    sd_img_gen_params_t to_sd_img_gen_params_t() {
        return gen_params.to_sd_img_gen_params_t();
    }
};

struct VidGenJobRequest {
    SDGenerationParams gen_params;
    std::string output_format = "webm";
    int output_compression    = 100;

    sd_vid_gen_params_t to_sd_vid_gen_params_t() {
        return gen_params.to_sd_vid_gen_params_t();
    }
};

std::string base64_encode(const std::vector<uint8_t>& bytes);
std::string normalize_output_format(std::string output_format);
std::vector<std::string> supported_img_output_formats(bool allow_webp = true);
std::vector<std::string> supported_vid_output_formats();
bool assign_output_options(ImgGenJobRequest& request,
                           std::string output_format,
                           int output_compression,
                           bool allow_webp,
                           std::string& error_message);
bool assign_output_options(VidGenJobRequest& request,
                           std::string output_format,
                           int output_compression,
                           std::string& error_message);
std::string video_mime_type(const std::string& output_format);
bool runtime_supports_generation_mode(const ServerRuntime& runtime, SDMode mode);
std::string unsupported_generation_mode_error(SDMode mode);
void refresh_lora_cache(ServerRuntime& rt);
std::string get_lora_full_path(ServerRuntime& rt, const std::string& path);
int64_t unix_timestamp_now();
