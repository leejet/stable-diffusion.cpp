#ifndef __SD_CORE_IMAGE_TENSOR_H__
#define __SD_CORE_IMAGE_TENSOR_H__

#include "core/tensor.hpp"

namespace sd {
    Tensor<float> ensure_image_tensor_channels(const Tensor<float>& image, int channels);
}

#endif
