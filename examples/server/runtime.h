#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
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
struct ServerModels;

struct SDSvrParams {
    std::string listen_ip = "127.0.0.1";
    int listen_port       = 1234;
    std::string serve_html_path;
    std::vector<std::string> model_dirs;
    bool normal_exit         = false;
    sd_log_level_t log_level = SD_LOG_INFO;
    bool color               = false;

    ArgOptions get_options();
    bool validate();
    bool resolve_and_validate();
    std::string to_string() const;
};

struct LoraEntry {
    std::string name;
    std::string path;
    std::string fullpath;
    std::string display_path;
    std::filesystem::path root;
};

struct UpscalerEntry {
    std::string name;
    std::string path;
    std::string fullpath;
    std::string model_name;
    int scale                = 4;
    int image_upscale_factor = 0;
    uintmax_t file_size      = 0;
    std::filesystem::file_time_type last_modified;
};

struct ServerRuntime {
    sd_ctx_t* sd_ctx;
    std::mutex* sd_ctx_mutex;
    const SDSvrParams* svr_params;
    SDContextParams* ctx_params;
    SDGenerationParams* default_gen_params;
    std::vector<LoraEntry>* lora_cache;
    std::mutex* lora_mutex;
    std::vector<UpscalerEntry>* upscaler_cache;
    std::mutex* upscaler_mutex;
    AsyncJobManager* async_job_manager;
    ServerModels* models;
};

struct ServerADetailerPass {
    std::string prompt;
    std::string negative_prompt;
    std::string extra_args;
    int width                = 512;
    int height               = 512;
    float strength           = 0.4f;
    int clip_skip            = -1;
    bool automatic_scheduler = false;
};

struct ImgGenJobRequest {
    SDGenerationParams gen_params;
    std::string output_format = "png";
    int output_compression    = 100;
    std::string preview_mode  = "none";
    int preview_interval      = 1;
    bool adetailer_enabled    = true;
    std::vector<ServerADetailerPass> adetailer_passes;

    sd_img_gen_params_t to_sd_img_gen_params_t() {
        return gen_params.to_sd_img_gen_params_t();
    }
};

struct VidGenJobRequest {
    SDGenerationParams gen_params;
    std::string output_format = "webm";
    int output_compression    = 100;
    std::string preview_mode  = "none";
    int preview_interval      = 1;

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
void refresh_upscaler_cache(ServerRuntime& rt);
int64_t unix_timestamp_now_ms();
bool apply_server_adetailer(ServerRuntime& runtime,
                            const ImgGenJobRequest& request,
                            const sd_img_gen_params_t& params,
                            SDImageVec& results);
