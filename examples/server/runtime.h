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
    std::string listen_ip     = "127.0.0.1";
    int listen_port           = 1234;
    int max_pending_jobs      = 64;
    int completed_ttl_seconds = 600;
    int weight_staging_ttl    = -1;
    int weight_residency_ttl  = -1;
    std::string serve_html_path;
    std::string dgl_dir = "dgl";
    bool media_view = true;
    std::vector<std::string> model_dirs;
    std::string output_dir;
    std::vector<std::string> api_keys;
    std::string api_key_file;
    std::vector<std::string> cors_origins;
    bool normal_exit         = false;
    sd_log_level_t log_level = SD_LOG_INFO;
    bool color               = false;

    ArgOptions get_options();
    void add_access_options(ArgOptions& options, const std::vector<ArgOptions>& groups);
    bool validate();
    bool resolve_and_validate();
    std::string to_string() const;

private:
    bool load_api_keys();
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
    std::optional<int> width;
    std::optional<int> height;
    std::optional<float> strength;
    std::optional<int> clip_skip;
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

    SDOptPtr to_image_opt() { return gen_params.to_image_opt(); }
};

class PromptImageCallbackGuard {
    sd_ctx_t* ctx_;
    const SDGenerationParams& params_;
    SDTensorPtr image_tensor_;

    static bool load_image(const char* source, void* data, const sd_tensor_t** image_out);

public:
    PromptImageCallbackGuard(sd_ctx_t* ctx, const SDGenerationParams& params);
    ~PromptImageCallbackGuard();
    PromptImageCallbackGuard(const PromptImageCallbackGuard&)            = delete;
    PromptImageCallbackGuard& operator=(const PromptImageCallbackGuard&) = delete;
};

constexpr int k_max_generation_dimension = 4096;
constexpr int k_max_adetailer_steps      = 100;

bool validate_server_img_gen_request(const ServerRuntime& runtime,
                                     const ImgGenJobRequest& request,
                                     std::string& error_message);

struct VidGenJobRequest {
    SDGenerationParams gen_params;
    std::string output_format = "webm";
    int output_compression    = 100;
    std::string preview_mode  = "none";
    int preview_interval      = 1;

    SDOptPtr to_video_opt() { return gen_params.to_video_opt(); }
};

std::string base64_encode(const std::vector<uint8_t>& bytes);
std::string base64_encode(const void* bytes, size_t size);
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
bool has_current_model(const ServerRuntime& runtime);
void configure_server_file_access(ServerModels& models, const SDSvrParams& params, const SDContextParams& config);
bool generate_server_images(sd_ctx_t* ctx, ImgGenJobRequest& request, SDImageVec& results, std::string& error_message);
std::string unsupported_generation_mode_error(SDMode mode);
void refresh_lora_cache(ServerRuntime& rt);
std::string get_lora_full_path(ServerRuntime& rt, const std::string& path);
void refresh_upscaler_cache(ServerRuntime& rt);
int64_t unix_timestamp_now_ms();
