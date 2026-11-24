#ifndef SD_OPTIONS_H
#define SD_OPTIONS_H

#include "api.h"
#include "config.h"
#include "json.hpp"

#include <memory>
#include <string>
#include <vector>

namespace sd::api {
    enum Operation {
        Context  = 1,
        Image    = 2,
        Video    = 4,
        Upscaler = 8,
        Upscale  = 16
    };

    sd_status_t resolve(const sd_opt_t* params, Operation operation, OptConfig& config);
    bool is_set(const sd_opt_t* params, const char* key);
    bool has_prefix(const sd_opt_t* params, const char* prefix);
    std::shared_ptr<sd_opt_t> copy_options(const sd_opt_t* params);
    std::vector<std::string> explicit_keys(const sd_opt_t* params);
    const sd_tensor_t* bound_tensor(const sd_opt_t* params, const char* key);
    std::vector<const sd_tensor_t*> bound_tensors(const sd_opt_t* params, const char* key);
    std::vector<sd_audio_t> bound_audios(const sd_opt_t* params, const char* key);
    std::vector<sd_video_t> bound_videos(const sd_opt_t* params, const char* key);
    nlohmann::json scalar_values(const sd_opt_t* params);
}

#endif  // SD_OPTIONS_H
