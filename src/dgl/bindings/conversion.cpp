#include "dgl/bindings/conversion.h"

namespace sd::dgl {
    const sd_tensor_t& i64_vector(const Value& value) {
        const auto& tensor = *value.as<sd_tensor_t>();
        if (tensor.type != SD_TENSOR_I64 || tensor.shape().size() != 1)
            throw std::runtime_error("expected a rank-one i64 tensor");
        return tensor;
    }

    std::vector<int64_t> read_int_vector(const Value& value) {
        if (value.type().kind == Kind::Tensor) {
            const auto& tensor = i64_vector(value);
            if (!tensor.bytes)
                return {};
            const auto* data = static_cast<const int64_t*>(tensor.data());
            return {data, data + tensor.shape()[0]};
        }
        std::vector<int64_t> result;
        for (const auto& item : value.items)
            result.push_back(item.scalar.get<int64_t>());
        return result;
    }

    Value shape_value(const std::vector<int64_t>& shape) {
        return Value::object(make_tensor(SD_TENSOR_I64, std::vector<int64_t>{int64_t(shape.size())}, shape.data(), shape.size() * sizeof(int64_t)));
    }

    Value tensor_value(sd::Tensor<float> tensor) {
        return Value::object(make_tensor(std::move(tensor)));
    }

    sd::Image image_view(const Value& value) {
        if (value.type().kind != Kind::Tensor)
            throw std::runtime_error("expected an image tensor");
        return image_view(*value.as<sd_tensor_t>());
    }

}  // namespace sd::dgl
