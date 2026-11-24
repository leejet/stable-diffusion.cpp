#ifndef __SD_DGL_OPS_TENSOR_H__
#define __SD_DGL_OPS_TENSOR_H__

#include <optional>

class RNG;
#include <variant>
#include "dgl/tensor.h"

namespace sd::dgl::ops {
    using TensorNumber = std::variant<int64_t, double>;
    struct TensorSlice {
        std::optional<int64_t> start;
        std::optional<int64_t> end;
    };
    using TensorIndex       = std::variant<int64_t, TensorSlice>;
    using TensorIndexResult = std::variant<int64_t, double, std::shared_ptr<sd_tensor_t>>;

    TensorIndexResult index_tensor(const std::shared_ptr<sd_tensor_t>& input, const std::vector<TensorIndex>& indices);
    template <class F>
    auto with_float_dtype(sd_tensor_type_t dtype, F function) {
        if (dtype == SD_TENSOR_F32)
            return function(float{});
        if (dtype == SD_TENSOR_F64)
            return function(double{});
        throw std::runtime_error("expected f32 or f64 tensors");
    }
    template <typename To, typename From>
    To cast_number(From value) {
        if constexpr (std::is_integral_v<To>) {
            if constexpr (std::is_floating_point_v<From>) {
                value            = std::trunc(value);
                const auto limit = std::ldexp(From(1), std::numeric_limits<To>::digits);
                const auto lower = std::is_signed_v<To> ? -limit : From(0);
                if (!(value >= lower && value < limit))
                    throw std::runtime_error("value is outside the target dtype range");
            } else if (int64_t(value) < int64_t(std::numeric_limits<To>::lowest()) ||
                       int64_t(value) > int64_t(std::numeric_limits<To>::max())) {
                throw std::runtime_error("value is outside the target dtype range");
            }
        } else if constexpr (sizeof(From) > sizeof(To) && std::is_floating_point_v<From>) {
            if (std::isfinite(value) && std::abs(value) > std::numeric_limits<To>::max())
                throw std::runtime_error("value is outside the target dtype range");
        }
        return static_cast<To>(value);
    }

    size_t element_size(sd_tensor_type_t type);
    size_t element_count(const std::vector<int64_t>& shape, sd_tensor_type_t type);
    size_t tensor_axis(const sd_tensor_t& tensor, int64_t axis);
    uint8_t rounded_byte(double value);
    void copy_tensor_box(const sd_tensor_t& input, sd_tensor_t& output, const std::vector<int64_t>& shape, const std::vector<int64_t>& source_offset, const std::vector<int64_t>& target_offset);
    std::shared_ptr<sd_tensor_t> crop_tensor(const sd_tensor_t& input, const std::vector<int64_t>& axes, const std::vector<int64_t>& start, const std::vector<int64_t>& size);
    std::shared_ptr<sd_tensor_t> interpolate_tensor(const sd_tensor_t& input, const std::vector<int64_t>& size, const std::vector<int64_t>& axes, const std::string& mode, bool antialias);
    std::shared_ptr<sd_tensor_t> pool_tensor(const sd_tensor_t& input, const std::vector<int64_t>& axes, const std::vector<int64_t>& kernel, const std::vector<int64_t>& stride, const std::vector<int64_t>& before, const std::vector<int64_t>& after, std::optional<TensorNumber> pad_value, bool maximum);
    std::shared_ptr<sd_tensor_t> blur_tensor(const std::shared_ptr<sd_tensor_t>& input, const std::vector<int64_t>& axes, int64_t radius, double sigma, const std::string& precision);
    template <typename T>
    T scalar_number(const TensorNumber& value) {
        return std::visit([](auto item) { return cast_number<T>(item); }, value);
    }

    inline std::shared_ptr<sd_tensor_t> tensor_result(sd_tensor_type_t dtype, std::vector<int64_t> shape, const void* data, size_t bytes) {
        return make_tensor(dtype, std::move(shape), data, bytes);
    }
    template <typename T>
    std::shared_ptr<sd_tensor_t> tensor_result(sd::Tensor<T> tensor, bool scalar = false) {
        return make_tensor(std::move(tensor), scalar);
    }
    template <typename T>
    std::shared_ptr<sd_tensor_t> tensor_result(std::vector<int64_t> shape, std::vector<T> values) {
        if (values.empty())
            return tensor_result(tensor_dtype<T>(), std::move(shape), nullptr, 0);
        const bool scalar = shape.empty();
        if (scalar)
            shape.push_back(1);
        return tensor_result(sd::Tensor<T>(std::move(shape), std::move(values)), scalar);
    }
    std::shared_ptr<sd_tensor_t> filled_tensor(std::vector<int64_t> shape, sd_tensor_type_t dtype, const TensorNumber& value);

    std::shared_ptr<sd_tensor_t> unsqueeze_tensor(const std::shared_ptr<sd_tensor_t>& input, int64_t axis);
    std::shared_ptr<sd_tensor_t> pad_tensor(const std::shared_ptr<sd_tensor_t>& input, const std::vector<int64_t>& pad, const TensorNumber& value);
    std::shared_ptr<sd_tensor_t> cast_tensor(const std::shared_ptr<sd_tensor_t>& input, sd_tensor_type_t dtype);
    std::shared_ptr<sd_tensor_t> translate_tensor(const std::shared_ptr<sd_tensor_t>& input, const std::vector<int64_t>& axes, const std::vector<int64_t>& offset, const TensorNumber& value);
    std::shared_ptr<sd_tensor_t> arange_tensor(int64_t start, int64_t end, int64_t step);
    std::shared_ptr<sd_tensor_t> index_select_tensor(const std::shared_ptr<sd_tensor_t>& input, int64_t axis_arg, const std::vector<int64_t>& indices);
    std::shared_ptr<sd_tensor_t> replace_slice(const std::shared_ptr<sd_tensor_t>& input, const std::shared_ptr<sd_tensor_t>& value, int64_t axis_arg, int64_t start);
    std::shared_ptr<sd_tensor_t> concat_tensors(const std::vector<std::shared_ptr<sd_tensor_t>>& items, int64_t axis_arg);
    std::shared_ptr<sd_tensor_t> grayscale_tensor(const std::shared_ptr<sd_tensor_t>& input, int64_t axis_arg);
    std::shared_ptr<sd_tensor_t> threshold_tensor(const std::shared_ptr<sd_tensor_t>& input, const TensorNumber& threshold, const TensorNumber& low_value, const TensorNumber& high_value);
    std::shared_ptr<sd_tensor_t> maximum_tensor(const std::shared_ptr<sd_tensor_t>& a, const std::shared_ptr<sd_tensor_t>& b);
    std::shared_ptr<sd_tensor_t> permute(const std::shared_ptr<sd_tensor_t>& input, const std::vector<int64_t>& axes);
    std::shared_ptr<sd_tensor_t> broadcast_to(const std::shared_ptr<sd_tensor_t>& input, std::vector<int64_t> dims);
    std::shared_ptr<sd_tensor_t> index_fill(const std::shared_ptr<sd_tensor_t>& input, int64_t axis_arg, const std::vector<int64_t>& indices, const TensorNumber& fill_value);
    std::shared_ptr<sd_tensor_t> randn_like(const std::shared_ptr<sd_tensor_t>& like, const std::shared_ptr<RNG>& rng);
    int64_t count_nonzero(const std::shared_ptr<sd_tensor_t>& input);
    std::shared_ptr<sd_tensor_t> lerp_bytes(const sd_tensor_t& a, const sd_tensor_t& b, const sd_tensor_t* weight, double scalar);

}

#endif
