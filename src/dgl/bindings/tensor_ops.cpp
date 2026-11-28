#include "dgl/bindings/tensor_ops.h"

#include "dgl/bindings/conversion.h"

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "core/rng.hpp"
#include "core/tensor.hpp"
#include "dgl/indexing.h"
#include "dgl/tensor_io.h"

namespace sd::dgl {
    using ops::cast_number;
    using ops::copy_tensor_box;
    using ops::element_count;
    using ops::element_size;
    using ops::rounded_byte;
    using ops::tensor_axis;
    using ops::with_float_dtype;

    static const std::map<std::string, sd_tensor_type_t> dtypes{
        {"u8", SD_TENSOR_U8},
        {"i32", SD_TENSOR_I32},
        {"i64", SD_TENSOR_I64},
        {"f32", SD_TENSOR_F32},
        {"f64", SD_TENSOR_F64},
    };

    static std::vector<int64_t> int_list(const Values& args, const char* name) {
        return read_int_vector(args.at(name));
    }

    static Value tensor_result(sd_tensor_type_t dtype, std::vector<int64_t> shape, const void* data, size_t bytes) {
        return Value::object(make_tensor(dtype, std::move(shape), data, bytes));
    }

    template <typename T>
    static Value tensor_result(sd::Tensor<T> tensor, bool scalar = false) {
        return Value::object(make_tensor(std::move(tensor), scalar));
    }

    template <typename T>
    static Value tensor_result(std::vector<int64_t> shape, std::vector<T> values) {
        if (values.empty())
            return tensor_result(tensor_dtype<T>(), std::move(shape), nullptr, 0);
        const bool scalar = shape.empty();
        if (scalar)
            shape.push_back(1);
        return tensor_result(sd::Tensor<T>(std::move(shape), std::move(values)), scalar);
    }

    static bool little_endian() {
        const uint16_t endian = 1;
        return *reinterpret_cast<const uint8_t*>(&endian) != 0;
    }

    static void swap_endian(uint8_t* bytes, size_t size, size_t stride) {
        for (size_t i = 0; i < size; i += stride)
            std::reverse(bytes + i, bytes + i + stride);
    }

    template <typename T>
    static T scalar_number(const Value& value) {
        if (value.type().kind == Kind::Integer)
            return cast_number<T>(value.scalar.get<int64_t>());
        if (value.type().kind == Kind::Number)
            return cast_number<T>(value.scalar.get<double>());
        throw std::runtime_error("expected a number or tensor");
    }

    static ops::TensorNumber number_value(const Value& value) {
        if (value.type().kind == Kind::Integer)
            return value.scalar.get<int64_t>();
        if (value.type().kind == Kind::Number)
            return value.scalar.get<double>();
        throw std::runtime_error("expected a scalar number");
    }

    static Value filled_tensor(std::vector<int64_t> shape, sd_tensor_type_t dtype, const Value& value) {
        return Value::object(ops::filled_tensor(std::move(shape), dtype, number_value(value)));
    }

    template <typename T>
    static const sd::Tensor<T>& numeric_operand(const Value& value, sd::Tensor<T>& scalar) {
        if (const auto* tensor = value.get_if<sd_tensor_t>())
            return tensor->as<T>();
        scalar = sd::Tensor<T>({1}, {scalar_number<T>(value)});
        return scalar;
    }

    template <class F>
    static Value numeric_operation(std::initializer_list<const Value*> operands, F compute) {
        sd_tensor_type_t dtype = 0;
        std::vector<int64_t> shape;
        for (const auto* operand : operands) {
            if (operand->type().kind == Kind::Tensor) {
                const auto& tensor = *operand->as<sd_tensor_t>();
                if (dtype && dtype != tensor.type)
                    throw std::runtime_error("numeric tensor operands must have matching dtypes");
                dtype = tensor.type;
                shape = sd::tensor_broadcast_shape(shape, tensor.shape());
            } else if (operand->type().kind != Kind::Number && operand->type().kind != Kind::Integer) {
                throw std::runtime_error("expected a number or tensor");
            }
        }
        if (!dtype)
            dtype = SD_TENSOR_F32;
        const auto count = element_count(shape, dtype);
        return with_float_dtype(dtype, [&](auto tag) {
            if (!count)
                return tensor_result(dtype, std::move(shape), nullptr, 0);
            auto output = compute(tag);
            return tensor_result(std::move(output), shape.empty());
        });
    }

    static const sd_tensor_t* tensor_operand(const Value& value) {
        return value.get_if<sd_tensor_t>();
    }

    static std::vector<int64_t> spatial_axes(const Values& args) {
        return args.count("axes") ? int_list(args, "axes") : std::vector<int64_t>{0, 1};
    }

    static Value pool_tensor(const Values& args, bool maximum) {
        std::optional<ops::TensorNumber> pad;
        if (args.count("pad_value")) {
            const auto& value = args.at("pad_value");
            if (value.type().kind == Kind::Integer)
                pad = value.scalar.get<int64_t>();
            else
                pad = value.scalar.get<double>();
        }
        auto before = args.count("padding_before") ? int_list(args, "padding_before") : std::vector<int64_t>{0, 0};
        return Value::object(ops::pool_tensor(*object<sd_tensor_t>(args, "input"), spatial_axes(args), int_list(args, "kernel"),
                                              args.count("stride") ? int_list(args, "stride") : std::vector<int64_t>{1, 1}, before,
                                              args.count("padding_after") ? int_list(args, "padding_after") : before, pad, maximum));
    }

    Value blur_tensor(const Values& args, RunState&) {
        return Value::object(ops::blur_tensor(object<sd_tensor_t>(args, "input"), spatial_axes(args), get_int(args, "radius"), get_number(args, "sigma"), get_string(args, "precision")));
    }

    Value execute_load_tensor(Values& args, RunState&) {
        return Value::object(load_npy(get_string(args, "path"), tensor_byte_limit));
    }

    Value execute_save_tensor(Values& args, RunState&) {
        save_npy_atomic(*object<sd_tensor_t>(args, "tensor"), get_string(args, "path"));
        return Value::void_value();
    }

    Value execute_tensor_literal(Values& args, RunState&) {
        const auto type = dtypes.at(get_string(args, "dtype"));
        auto dims       = int_list(args, "shape");
        element_count(dims, type);
        auto result = make_tensor(type, std::move(dims));
        auto* data  = static_cast<uint8_t*>(result->mutable_data());
        decode_base64(get_string(args, "data"), data, result->bytes);
        if (!little_endian())
            swap_endian(data, result->bytes, element_size(type));
        return Value::object(std::move(result));
    }

    Value execute_tensor_shape(Values& args, RunState&) {
        return shape_value(object<sd_tensor_t>(args, "input")->shape());
    }

    Value execute_tensor_index(Values& args, RunState&) {
        return index_value(args.at("input"), args.at("indices"));
    }
    Value execute_tensor_vector(Values& args, RunState&) {
        const auto type   = dtypes.at(get_string(args, "dtype"));
        const auto& items = args.at("values").items;
        std::vector<int64_t> dims{int64_t(items.size())};
        element_count(dims, type);
        return with_dtype(type, [&](auto tag) {
            using T = decltype(tag);
            std::vector<T> data;
            data.reserve(items.size());
            for (const auto& item : items)
                data.push_back(scalar_number<T>(item));
            return tensor_result(std::move(dims), std::move(data));
        });
    }

    Value execute_tensor_to_list(Values& args, RunState&) {
        const auto input = object<sd_tensor_t>(args, "input");
        if (input->shape().size() != 1)
            throw std::runtime_error("tensor::to_list requires a rank-one tensor");
        const auto count = input->shape()[0];
        if (count > 10000)
            throw std::runtime_error("tensor::to_list supports at most 10000 elements");
        return with_dtype(input->type, [&](auto tag) {
            using T             = decltype(tag);
            constexpr auto kind = std::is_integral_v<T> ? Kind::Integer : Kind::Number;
            const auto element  = get_string(args, "element");
            if (element != "auto" && element != (kind == Kind::Integer ? "int" : "number"))
                throw Error(SD_STATUS_TYPE_MISMATCH, "tensor::to_list element does not match input dtype; use tensor::cast to convert it");
            std::vector<Value> items;
            items.reserve(size_t(count));
            const auto* data = input->as<T>().data();
            for (int64_t i = 0; i < count; ++i) {
                if constexpr (std::is_integral_v<T>)
                    items.push_back(Value::literal(int64_t(data[i])));
                else
                    items.push_back(Value::literal(double(data[i])));
            }
            return Value::list(std::move(items), kind);
        });
    }

    Value execute_tensor_reshape(Values& args, RunState&) {
        const auto input = object<sd_tensor_t>(args, "input");
        auto dims        = int_list(args, "shape");
        if (element_count(dims, input->type) * element_size(input->type) != input->bytes)
            throw std::runtime_error("reshape must preserve element count");
        if (dims == input->shape())
            return args.at("input");
        return tensor_result(input->type, std::move(dims), input->data(), input->bytes);
    }

    Value execute_tensor_unsqueeze(Values& args, RunState&) {
        return Value::object(ops::unsqueeze_tensor(object<sd_tensor_t>(args, "input"), get_int(args, "axis")));
    }

    Value execute_tensor_pad(Values& args, RunState&) {
        return Value::object(ops::pad_tensor(object<sd_tensor_t>(args, "input"), int_list(args, "pad"), number_value(args.at("value"))));
    }

    Value execute_tensor_crop(Values& args, RunState&) {
        return Value::object(crop_tensor(*object<sd_tensor_t>(args, "input"), int_list(args, "axes"),
                                         int_list(args, "start"), int_list(args, "size")));
    }

    Value execute_tensor_numel(Values& args, RunState&) {
        const auto input = object<sd_tensor_t>(args, "input");
        return Value::literal(int64_t(input->bytes / element_size(input->type)));
    }

    Value execute_tensor_zeros(Values& args, RunState&) {
        return filled_tensor(int_list(args, "shape"), dtypes.at(get_string(args, "dtype")), Value::literal(0));
    }

    Value execute_tensor_full(Values& args, RunState&) {
        return filled_tensor(int_list(args, "shape"), dtypes.at(get_string(args, "dtype")), args.at("value"));
    }

    Value execute_tensor_zeros_like(Values& args, RunState&) {
        const auto like = object<sd_tensor_t>(args, "like");
        return filled_tensor(like->shape(), like->type, Value::literal(0));
    }

    Value execute_tensor_lerp(Values& args, RunState&) {
        const auto& a     = args.at("a");
        const auto& b     = args.at("b");
        const auto& w     = args.at("weight");
        const auto* left  = tensor_operand(a);
        const auto* right = tensor_operand(b);
        if ((left && left->type == SD_TENSOR_U8) || (right && right->type == SD_TENSOR_U8)) {
            if (!left || !right)
                throw std::runtime_error("u8 lerp requires two u8 endpoints");
            return Value::object(ops::lerp_bytes(*left, *right, tensor_operand(w), tensor_operand(w) ? 0 : scalar_number<double>(w)));
        }
        return numeric_operation({&a, &b, &w}, [&](auto tag) {
            using T = decltype(tag);
            sd::Tensor<T> scalar_a, scalar_b, scalar_weight;
            const auto& weight = numeric_operand(w, scalar_weight);
            return numeric_operand(a, scalar_a) * (T(1) - weight) + numeric_operand(b, scalar_b) * weight;
        });
    }

    Value execute_tensor_affine(Values& args, RunState&) {
        const auto& input = args.at("input");
        return numeric_operation({&input}, [&](auto tag) {
            using T = decltype(tag);
            sd::Tensor<T> literal;
            return numeric_operand(input, literal) * scalar_number<T>(args.at("scale")) + scalar_number<T>(args.at("offset"));
        });
    }

    Value execute_tensor_clamp(Values& args, RunState&) {
        const auto& input = args.at("input");
        if (get_number(args, "min") > get_number(args, "max"))
            throw std::runtime_error("clamp min exceeds max");
        return numeric_operation({&input}, [&](auto tag) {
            using T = decltype(tag);
            sd::Tensor<T> literal;
            return sd::ops::clamp(numeric_operand(input, literal), scalar_number<T>(args.at("min")), scalar_number<T>(args.at("max")));
        });
    }

    Value execute_tensor_max_pool2d(Values& args, RunState&) {
        return pool_tensor(args, true);
    }

    Value execute_tensor_min_pool2d(Values& args, RunState&) {
        return pool_tensor(args, false);
    }

    Value execute_tensor_interpolate(Values& args, RunState&) {
        const auto axes = args.count("axes") ? int_list(args, "axes") : std::vector<int64_t>{0, 1};
        return Value::object(interpolate_tensor(*object<sd_tensor_t>(args, "input"), int_list(args, "size"),
                                                axes, get_string(args, "mode"), get_bool(args, "antialias")));
    }

    template <typename F>
    static Value binary_tensor(const Values& args, F compute) {
        const auto& a = args.at("a");
        const auto& b = args.at("b");
        return numeric_operation({&a, &b}, [&](auto tag) {
            using T = decltype(tag);
            sd::Tensor<T> scalar_a, scalar_b;
            return compute(numeric_operand(a, scalar_a), numeric_operand(b, scalar_b));
        });
    }

    Value execute_tensor_add(Values& args, RunState&) {
        return binary_tensor(args, [](const auto& a, const auto& b) { return a + b; });
    }

    Value execute_tensor_mul(Values& args, RunState&) {
        return binary_tensor(args, [](const auto& a, const auto& b) { return a * b; });
    }

    Value execute_tensor_div(Values& args, RunState&) {
        return binary_tensor(args, [](const auto& a, const auto& b) { return a / b; });
    }

    Value cast_tensor(const Values& args, RunState&) {
        return Value::object(ops::cast_tensor(object<sd_tensor_t>(args, "input"), dtypes.at(get_string(args, "dtype"))));
    }
    Value translate_tensor(const Values& args, RunState&) {
        return Value::object(ops::translate_tensor(object<sd_tensor_t>(args, "input"), int_list(args, "axes"), int_list(args, "offset"), number_value(args.at("value"))));
    }
    Value arange_tensor(const Values& args, RunState&) {
        return Value::object(ops::arange_tensor(get_int(args, "start"), get_int(args, "end"), get_int(args, "step")));
    }
    Value index_select_tensor(const Values& args, RunState&) {
        return Value::object(ops::index_select_tensor(object<sd_tensor_t>(args, "input"), get_int(args, "axis"), int_list(args, "indices")));
    }
    Value replace_slice(const Values& args, RunState&) {
        return Value::object(ops::replace_slice(object<sd_tensor_t>(args, "input"), object<sd_tensor_t>(args, "value"), get_int(args, "axis"), get_int(args, "start")));
    }
    Value concat_tensors(const Values& args, RunState&) {
        std::vector<std::shared_ptr<sd_tensor_t>> items;
        for (const auto& item : args.at("items").items)
            items.push_back(item.as<sd_tensor_t>());
        return Value::object(ops::concat_tensors(items, get_int(args, "axis")));
    }
    Value grayscale_tensor(const Values& args, RunState&) {
        return Value::object(ops::grayscale_tensor(object<sd_tensor_t>(args, "input"), get_int(args, "axis")));
    }
    Value threshold_tensor(const Values& args, RunState&) {
        return Value::object(ops::threshold_tensor(object<sd_tensor_t>(args, "input"), number_value(args.at("threshold")), number_value(args.at("low")), number_value(args.at("high"))));
    }
    Value maximum_tensor(const Values& args, RunState&) {
        return Value::object(ops::maximum_tensor(object<sd_tensor_t>(args, "a"), object<sd_tensor_t>(args, "b")));
    }
    Value execute_tensor_permute(Values& args, RunState&) {
        return Value::object(ops::permute(object<sd_tensor_t>(args, "input"), int_list(args, "axes")));
    }
    Value execute_tensor_broadcast_to(Values& args, RunState&) {
        return Value::object(ops::broadcast_to(object<sd_tensor_t>(args, "input"), int_list(args, "shape")));
    }
    Value execute_tensor_index_fill(Values& args, RunState&) {
        return Value::object(ops::index_fill(object<sd_tensor_t>(args, "input"), get_int(args, "axis"), int_list(args, "indices"), number_value(args.at("value"))));
    }
    Value execute_tensor_randn_like(Values& args, RunState&) {
        return Value::object(ops::randn_like(object<sd_tensor_t>(args, "like"), object<RNG>(args, "rng")));
    }
    Value execute_tensor_count_nonzero(Values& args, RunState&) {
        return Value::literal(ops::count_nonzero(object<sd_tensor_t>(args, "input")));
    }
}  // namespace sd::dgl
