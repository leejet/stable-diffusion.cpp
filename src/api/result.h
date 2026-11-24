#ifndef SD_RESULT_H
#define SD_RESULT_H

#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "tensor.h"

#include "stable-diffusion.h"

namespace sd::dgl {
    class Value;
}

namespace sd::api {
    void append_program_image(sd_result_t& result, const dgl::Value& value, const std::string& source);
    void set_program_video(sd_result_t& result, const dgl::Value& value, const std::string& source);
}

struct sd_result {
    sd_result_kind_t kind = SD_RESULT_INVALID;
    std::vector<sd::api::TensorRef> images;
    std::vector<std::string> generation_infos;
    sd::api::TensorRef audio_tensor;
    sd_audio_t audio{};
    int fps = 0;
    mutable std::mutex video_mutex;
    mutable sd::api::TensorRef video_tensor;
    mutable sd_video_t video{};
};

#endif  // SD_RESULT_H
