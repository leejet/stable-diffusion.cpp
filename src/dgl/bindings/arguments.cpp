#include "dgl/bindings/arguments.h"

#include "dgl/bindings/values.h"
#include "dgl/context.h"
#include "dgl/ops/media.h"
#include "dgl/ops/preview.h"
#include "dgl/ops/sampling.h"

namespace sd::dgl {
    const sd::Tensor<float>& float_tensor(const Values& args, const std::string& name) {
        return read_tensor<float>(args.at(name));
    }

    std::vector<sd::Tensor<float>> float_tensor_list(const Value& value) {
        if (value.type().kind != Kind::List) {
            throw std::runtime_error("expected a tensor list");
        }
        std::vector<sd::Tensor<float>> tensors;
        for (const auto& item : value.items) {
            tensors.push_back(read_tensor<float>(item));
        }
        return tensors;
    }

    void validate_float_shape(const std::vector<int64_t>& shape) {
        size_t remaining = tensor_byte_limit / sizeof(float);
        if (shape.empty() || shape.size() > 8) {
            throw std::runtime_error("invalid tensor rank");
        }
        for (auto dim : shape) {
            if (dim <= 0 || static_cast<uint64_t>(dim) > remaining) {
                throw std::runtime_error("invalid tensor dimension or tensor exceeds 1 GiB");
            }
            remaining /= static_cast<size_t>(dim);
        }
    }

    const sd::Tensor<float>& pixels(const Value& value, sd::Tensor<float>& decoded) {
        if (value.type().kind != Kind::Tensor) {
            throw std::runtime_error("expected an image or pixel tensor");
        }
        const auto& tensor = *value.as<sd_tensor_t>();
        if (tensor.type == SD_TENSOR_F32) {
            const auto& s           = tensor.shape();
            const auto channel_axis = s.size() == 5 ? 3 : 2;
            if ((s.size() != 4 && s.size() != 5) || s[channel_axis] < 1 || s[channel_axis] > 4 || s.back() != 1) {
                throw std::runtime_error("expected a float pixel tensor with width, height, optional time, channels and batch");
            }
            return read_tensor<float>(value);
        }
        decoded = ops::image_to_f32(tensor);
        return decoded;
    }

    sd::Tensor<float> pixels(const Value& value) {
        sd::Tensor<float> decoded;
        const auto& input = pixels(value, decoded);
        if (&input == &decoded)
            return decoded;
        return input;
    }

    Value image_to_u8(const sd::Tensor<float>& tensor) {
        return Value::object(ops::image_to_u8(tensor));
    }

    const sd::Tensor<float>& sigma_tensor(const Values& args, const char* key) {
        const auto tensor = object<sd_tensor_t>(args, key);
        if (tensor->type != SD_TENSOR_F32 || tensor->shape().size() != 1)
            throw std::runtime_error(std::string(key) + " must be a rank-one f32 tensor");
        if (!tensor->bytes)
            throw std::runtime_error(std::string(key) + " must be nonempty");
        return tensor->as<float>();
    }

    ops::SamplingOptions sampling_options(const Values& args, const Context& context) {
        ops::SamplingOptions options;
        options.batched_cfg = context.batched_cfg;
        options.method      = get_string(args, "method");
        options.eta         = float(get_number(args, "eta"));
        if (args.count("denoise_mask")) {
            if (args.count("base"))
                options.base = &float_tensor(args, "base");
            options.denoise_mask = &float_tensor(args, "denoise_mask");
        }
        options.mask_input = get_bool(args, "mask_input");
        options.extra_args = get_string(args, "extra_args");
        if (args.count("finish"))
            options.finish = get_bool(args, "finish");
        if (args.count("cache"))
            options.cache = cache_options(args.at("cache").fields);
        if (args.count("observer"))
            options.observer = object<ops::SamplingObserver>(args, "observer");
        else {
            static const auto observer = std::make_shared<ops::PreviewObserver>();
            options.observer           = observer;
        }
        return options;
    }
}  // namespace sd::dgl
