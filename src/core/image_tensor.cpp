#include "core/image_tensor.h"
#include "ggml.h"

namespace sd {
    sd::Tensor<float> ensure_image_tensor_channels(const sd::Tensor<float>& image, int channels) {
        if (image.empty()) {
            return image;
        }
        GGML_ASSERT(image.dim() == 4 || image.dim() == 5);
        const size_t channel_axis = image.dim() == 5 ? 3 : 2;
        int64_t current_channels  = image.shape()[channel_axis];
        if (current_channels == channels) {
            return image;
        }
        sd::Tensor<float> expanded;
        const auto* source = &image;
        if (current_channels == 1 && (channels == 3 || channels == 4)) {
            auto rg  = sd::ops::concat(image, image, channel_axis);
            expanded = sd::ops::concat(rg, image, channel_axis);
            if (channels == 3) {
                return expanded;
            }
            source           = &expanded;
            current_channels = 3;
        }
        if (channels == 4 && current_channels == 3) {
            auto alpha_shape          = image.shape();
            alpha_shape[channel_axis] = 1;
            return sd::ops::concat(*source, sd::full<float>(alpha_shape, 1.f), channel_axis);
        }
        if (channels == 3 && current_channels >= 3) {
            return sd::ops::slice(image, channel_axis, 0, 3);
        }
        GGML_ABORT("cannot convert image tensor from %lld to %d channels",
                   (long long)current_channels,
                   channels);
    }

}
