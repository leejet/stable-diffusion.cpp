#ifndef __SD_DGL_BINDINGS_ARGUMENTS_H__
#define __SD_DGL_BINDINGS_ARGUMENTS_H__

#include "dgl/bindings/conversion.h"
#include "dgl/models/fwd.h"

namespace sd::dgl {
    namespace ops {
        struct SamplingOptions;
    }
    using models::AudioEncoder;
    using models::AudioVAE;
    using models::ControlNet;
    using models::DiT;
    using models::IPAdapter;
    using models::Model;
    using models::PhotoMaker;
    using models::TextEncoder;
    using models::VAE;
    using models::VisionEncoder;

    template <typename T>
    const sd::Tensor<T>& read_tensor(const Value& value) {
        const auto* tensor = value.get_if<sd_tensor_t>();
        if (!tensor || tensor->type != tensor_dtype<T>() ||
            tensor->bytes == 0 || tensor->shape().empty())
            throw std::runtime_error("unexpected tensor type, scalar or empty tensor");
        return tensor->as<T>();
    }

    std::vector<sd::Tensor<float>> float_tensor_list(const Value& value);
    const sd::Tensor<float>& float_tensor(const Values& args, const std::string& name);
    const sd::Tensor<float>& sigma_tensor(const Values& args, const char* key);
    ops::SamplingOptions sampling_options(const Values& args, const Context& context);
    const sd::Tensor<float>& pixels(const Value& value, sd::Tensor<float>& decoded);
    sd::Tensor<float> pixels(const Value& value);
    Value image_to_u8(const sd::Tensor<float>& tensor);
    void validate_float_shape(const std::vector<int64_t>& shape);
}

#endif  // __SD_DGL_BINDINGS_ARGUMENTS_H__
