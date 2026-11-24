#ifndef __SD_CORE_IMAGE_GEOMETRY_H__
#define __SD_CORE_IMAGE_GEOMETRY_H__

#include "core/tensor.hpp"

namespace sd {
    std::pair<int, int> select_image_size_bucket(int source_width, int source_height, int area, int factor, int fallback_width, int fallback_height);
    Tensor<float> resize_and_center_crop(const Tensor<float>& image, int width, int height, ops::InterpolateMode filter = ops::InterpolateMode::Lanczos);
    int64_t align_image_dimension(int64_t size, int64_t multiple);
    int64_t align_video_frame_count(int64_t frames, int64_t multiple, int64_t minimum = 1);
    // An empty result means no adjustment is needed; borrow the original tensor.
    Tensor<float> fit_image_tensor(const Tensor<float>& image, int64_t multiple, const char* operation);
    Tensor<float> pad_video_tensor(const Tensor<float>& video, int64_t frames);
}

#endif
