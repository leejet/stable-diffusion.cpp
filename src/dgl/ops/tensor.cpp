#include "dgl/ops/tensor.h"

#include <map>

#include "core/rng.hpp"

namespace sd::dgl::ops {
    size_t element_size(sd_tensor_type_t dtype) {
        return with_dtype(dtype, [](auto value) { return sizeof(value); });
    }

    size_t element_count(const std::vector<int64_t>& shape, sd_tensor_type_t dtype) {
        if (shape.size() > 8 || std::any_of(shape.begin(), shape.end(), [](int64_t dim) { return dim < 0; }))
            throw std::runtime_error("invalid tensor shape");
        if (std::find(shape.begin(), shape.end(), 0) != shape.end())
            return 0;
        const auto limit = tensor_byte_limit / element_size(dtype);
        size_t count     = 1;
        for (auto dim : shape) {
            if (uint64_t(dim) > limit / count)
                throw std::runtime_error("tensor exceeds 1 GiB");
            count *= size_t(dim);
        }
        return count;
    }

    size_t tensor_axis(const sd_tensor_t& tensor, int64_t axis) {
        if (axis < 0 || size_t(axis) >= tensor.shape().size())
            throw std::runtime_error("axis falls outside the tensor rank");
        return size_t(axis);
    }

    static void copy_tensor_box(const sd_tensor_t& input, sd_tensor_t& output, const std::vector<int64_t>& shape, const std::vector<int64_t>& source_offset, const std::vector<int64_t>& target_offset, const std::vector<int64_t>& out_stride) {
        if (std::find(shape.begin(), shape.end(), 0) != shape.end())
            return;
        const auto in_stride = sd::tensor_compute_strides(input.shape());
        const auto bytes     = element_size(input.type);
        const auto row_bytes = size_t(shape[0]) * bytes;
        int64_t source_start = 0, target_start = 0, rows = 1;
        for (size_t axis = 0; axis < shape.size(); ++axis) {
            source_start += source_offset[axis] * in_stride[axis];
            target_start += target_offset[axis] * out_stride[axis];
            if (axis)
                rows *= shape[axis];
        }
        const auto* from = static_cast<const uint8_t*>(input.data()) + source_start * bytes;
        auto* to         = static_cast<uint8_t*>(output.mutable_data()) + target_start * bytes;
        parallel_for(0, rows, std::max<int64_t>(1, 16384 / row_bytes), [&](int64_t begin, int64_t end) {
            for (int64_t row = begin; row < end; ++row) {
                int64_t index = row, source = 0, target = 0;
                for (size_t axis = 1; axis < shape.size(); ++axis) {
                    const auto coord = index % shape[axis];
                    source += coord * in_stride[axis];
                    target += coord * out_stride[axis];
                    index /= shape[axis];
                }
                std::memcpy(to + target * bytes, from + source * bytes, row_bytes);
            }
        });
    }

    void copy_tensor_box(const sd_tensor_t& input, sd_tensor_t& output, const std::vector<int64_t>& shape, const std::vector<int64_t>& source_offset, const std::vector<int64_t>& target_offset) {
        if (std::find(shape.begin(), shape.end(), 0) != shape.end())
            return;
        copy_tensor_box(input, output, shape, source_offset, target_offset, sd::tensor_compute_strides(output.shape()));
    }

    TensorIndexResult index_tensor(const std::shared_ptr<sd_tensor_t>& input, const std::vector<TensorIndex>& indices) {
        const auto& shape = input->shape();
        if (indices.empty() || indices.size() > shape.size())
            throw std::runtime_error("tensor indexing requires one or more axes within the tensor rank");
        std::vector<int64_t> region = shape, offset(shape.size(), 0), output_shape;
        for (size_t axis = 0; axis < shape.size(); ++axis) {
            if (axis >= indices.size()) {
                output_shape.push_back(shape[axis]);
                continue;
            }
            if (const auto* index = std::get_if<int64_t>(&indices[axis])) {
                const auto position = *index < 0 ? shape[axis] + *index : *index;
                if (position < 0 || position >= shape[axis])
                    throw std::runtime_error("tensor index out of range on axis " + std::to_string(axis));
                offset[axis] = position;
                region[axis] = 1;
            } else {
                const auto& slice    = std::get<TensorSlice>(indices[axis]);
                const auto normalize = [&](int64_t value) {
                    return value < 0 ? std::max<int64_t>(0, shape[axis] + value) : std::min(value, shape[axis]);
                };
                const auto start = slice.start ? normalize(*slice.start) : 0;
                const auto end   = slice.end ? normalize(*slice.end) : shape[axis];
                offset[axis]     = start;
                region[axis]     = std::max<int64_t>(0, end - start);
                output_shape.push_back(region[axis]);
            }
        }
        if (output_shape.empty()) {
            const auto strides = sd::tensor_compute_strides(shape);
            int64_t position   = 0;
            for (size_t axis = 0; axis < shape.size(); ++axis)
                position += offset[axis] * strides[axis];
            return with_dtype(input->type, [&](auto tag) -> TensorIndexResult {
                using T          = decltype(tag);
                const auto value = static_cast<const T*>(input->data())[position];
                if constexpr (std::is_integral_v<T>)
                    return int64_t(value);
                else
                    return double(value);
            });
        }
        if (output_shape == shape && region == shape)
            return input;
        auto output = make_tensor(input->type, std::move(output_shape));
        if (!output->bytes)
            return output;
        // Dropping singleton integer axes preserves the region's contiguous byte order.
        copy_tensor_box(*input, *output, region, offset, std::vector<int64_t>(shape.size(), 0), sd::tensor_compute_strides(region));
        return output;
    }

    std::shared_ptr<sd_tensor_t> crop_tensor(const sd_tensor_t& input, const std::vector<int64_t>& axes, const std::vector<int64_t>& start, const std::vector<int64_t>& size) {
        if (axes.empty() || axes.size() != start.size() || axes.size() != size.size())
            throw std::runtime_error("crop requires matching axes, start and size lists");
        auto shape = input.shape();
        std::vector<int64_t> offset(shape.size(), 0);
        std::vector<bool> used(shape.size(), false);
        for (size_t i = 0; i < axes.size(); ++i) {
            const auto axis = tensor_axis(input, axes[i]);
            if (used[axis] || start[i] < 0 || size[i] < 0 || start[i] > shape[axis] || size[i] > shape[axis] - start[i])
                throw std::runtime_error("crop axes must be distinct and the crop must fit inside the tensor");
            used[axis]   = true;
            shape[axis]  = size[i];
            offset[axis] = start[i];
        }
        auto result = make_tensor(input.type, shape);
        copy_tensor_box(input, *result, shape, offset, std::vector<int64_t>(shape.size(), 0));
        return result;
    }

    uint8_t rounded_byte(double value) {
        if (!(value > 0.0))
            return 0;
        if (value >= 255.0)
            return 255;
        return uint8_t(std::lround(value));
    }

    template <class T, class F>
    static sd::Tensor<T> apply_spatial(const sd::Tensor<T>& tensor, size_t x, size_t y, F filter) {
        if (x == 0 && y == 1)
            return filter(tensor);
        std::vector<size_t> order{x, y}, inverse(size_t(tensor.dim()));
        for (size_t axis = 0; axis < inverse.size(); ++axis)
            if (axis != x && axis != y)
                order.push_back(axis);
        for (size_t i = 0; i < order.size(); ++i)
            inverse[order[i]] = i;
        return filter(tensor.permute(order)).permute(inverse);
    }

    static std::pair<size_t, size_t> spatial_axes(const sd_tensor_t& input, const std::vector<int64_t>& axes) {
        if (axes.size() != 2 || axes[0] == axes[1] || !input.bytes)
            throw std::runtime_error("expected a nonempty tensor and two distinct axes");
        return {tensor_axis(input, axes[0]), tensor_axis(input, axes[1])};
    }

    std::shared_ptr<sd_tensor_t> pool_tensor(const sd_tensor_t& source, const std::vector<int64_t>& axes, const std::vector<int64_t>& kernel, const std::vector<int64_t>& stride, const std::vector<int64_t>& before, const std::vector<int64_t>& after, std::optional<TensorNumber> pad_value, bool maximum) {
        const auto* input  = &source;
        const auto spatial = spatial_axes(source, axes);
        const auto x = spatial.first, y = spatial.second;
        if (kernel.size() != 2 || stride.size() != 2 || before.size() != 2 || after.size() != 2)
            throw std::runtime_error("pooling kernel, stride and padding need two entries");
        auto output_size = [&](size_t i, int64_t length) {
            if (kernel[i] <= 0 || stride[i] <= 0 || before[i] < 0 || after[i] < 0 ||
                before[i] > INT64_MAX - length || after[i] > INT64_MAX - length - before[i])
                throw std::runtime_error("invalid pooling kernel, stride or padding");
            const auto padded = length + before[i] + after[i];
            if (kernel[i] > padded)
                throw std::runtime_error("pooling kernel exceeds padded size");
            return (padded - kernel[i]) / stride[i] + 1;
        };
        const auto ow = output_size(0, input->shape()[x]), oh = output_size(1, input->shape()[y]);
        auto shape = input->shape();
        shape[x]   = ow;
        shape[y]   = oh;
        element_count(shape, input->type);
        return with_dtype(input->type, [&](auto tag) {
            using T        = decltype(tag);
            const auto pad = bool(pad_value) ? std::optional<T>{std::visit([](auto value) { return cast_number<T>(value); }, *pad_value)} : std::nullopt;
            auto filtered  = apply_spatial(input->as<T>(), x, y, [&](const sd::Tensor<T>& tensor) {
                auto dims     = tensor.shape();
                const auto iw = dims[0], ih = dims[1];
                dims[0] = ow;
                dims[1] = oh;
                sd::Tensor<T> output(std::move(dims));
                const auto combine = [&](T a, T b) { return maximum ? std::max(a, b) : std::min(a, b); };
                const auto neutral = maximum ? std::numeric_limits<T>::lowest() : std::numeric_limits<T>::max();
                parallel_for(0, output.numel() / ow, std::max<int64_t>(1, 16384 / ow), [&](int64_t begin, int64_t end) {
                    for (int64_t row = begin; row < end; ++row) {
                        const auto* plane = tensor.data() + row / oh * iw * ih;
                        const auto y0 = (row % oh) * stride[1] - before[1], y1 = y0 + kernel[1];
                        for (int64_t col = 0; col < ow; ++col) {
                            const auto x0 = col * stride[0] - before[0], x1 = x0 + kernel[0];
                            const auto l = std::max<int64_t>(0, x0), r = std::min(iw, x1);
                            const auto t = std::max<int64_t>(0, y0), b = std::min(ih, y1);
                            const bool padded = pad && (x0 < 0 || x1 > iw || y0 < 0 || y1 > ih);
                            T value           = padded ? *pad : neutral;
                            for (int64_t sy = t; sy < b; ++sy)
                                for (int64_t sx = l; sx < r; ++sx)
                                    value = combine(value, plane[sy * iw + sx]);
                            output.data()[row * ow + col] = padded || (l < r && t < b) ? value : T(0);
                        }
                    }
                });
                return output;
            });
            return make_tensor(std::move(filtered));
        });
    }

    template <class T, class Compute>
    static sd::Tensor<T> gaussian_blur(const sd::Tensor<T>& input, int64_t radius, Compute sigma) {
        std::vector<Compute> weights(size_t(2 * radius + 1));
        Compute total = 0;
        for (int64_t i = -radius; i <= radius; ++i) {
            const auto distance         = Compute(i);
            const auto weight           = i == 0 ? Compute(1) : std::exp(-(distance * distance) / (Compute(2) * sigma * sigma));
            weights[size_t(i + radius)] = weight;
            total += weight;
        }
        for (auto& weight : weights)
            weight /= total;
        const auto width = input.shape()[0], height = input.shape()[1];
        // Byte kernels round their horizontal pass to float before the vertical pass.
        using Intermediate = std::conditional_t<std::is_same_v<T, uint8_t>, float, Compute>;
        std::vector<Intermediate> horizontal(size_t(input.numel()));
        sd::Tensor<T> output(input.shape());
        auto filter = [&](const auto* source, auto* target, bool along_x) {
            using Output = std::remove_pointer_t<decltype(target)>;
            parallel_for(0, input.numel() / width, std::max<int64_t>(1, 16384 / width), [&](int64_t begin, int64_t end) {
                for (int64_t row = begin; row < end; ++row) {
                    const auto y     = row % height;
                    const auto plane = row / height * width * height;
                    for (int64_t x = 0; x < width; ++x) {
                        Compute sum = 0;
                        for (int64_t k = -radius; k <= radius; ++k) {
                            const auto sx = along_x ? std::clamp(x + k, int64_t(0), width - 1) : x;
                            const auto sy = along_x ? y : std::clamp(y + k, int64_t(0), height - 1);
                            sum += Compute(source[plane + sy * width + sx]) * weights[size_t(k + radius)];
                        }
                        if constexpr (std::is_same_v<Output, uint8_t> && std::is_same_v<Compute, float>)
                            target[row * width + x] = uint8_t(std::clamp(sum, Compute(0), Compute(255)) + 0.5f);
                        else if constexpr (std::is_same_v<Output, uint8_t>)
                            target[row * width + x] = rounded_byte(sum);
                        else
                            target[row * width + x] = static_cast<Output>(sum);
                    }
                }
            });
        };
        filter(input.data(), horizontal.data(), true);
        filter(horizontal.data(), output.data(), false);
        return output;
    }

    std::shared_ptr<sd_tensor_t> blur_tensor(const std::shared_ptr<sd_tensor_t>& input, const std::vector<int64_t>& axes, int64_t radius, double sigma, const std::string& precision) {
        const auto spatial = spatial_axes(*input, axes);
        const auto x = spatial.first, y = spatial.second;
        if (input->type != SD_TENSOR_U8 && input->type != SD_TENSOR_F32 && input->type != SD_TENSOR_F64)
            throw std::runtime_error("gaussian blur requires u8, f32 or f64");
        if (!radius)
            return input;
        if (sigma <= 0)
            throw std::runtime_error("nonzero Gaussian radius requires positive sigma");
        auto blur = [&](auto tag) {
            using T               = decltype(tag);
            const bool use_double = precision == "f64" || (precision == "auto" && input->type == SD_TENSOR_F64);
            auto result           = apply_spatial(input->as<T>(), x, y, [&](const sd::Tensor<T>& tensor) {
                return use_double ? gaussian_blur(tensor, radius, sigma) : gaussian_blur(tensor, radius, cast_number<float>(sigma));
            });
            return make_tensor(std::move(result));
        };
        if (input->type == SD_TENSOR_U8)
            return blur(uint8_t{});
        return with_float_dtype(input->type, blur);
    }

    static sd::Tensor<uint8_t> interpolate_bytes(const sd::Tensor<uint8_t>& input, std::vector<int64_t> shape, sd::ops::InterpolateMode mode, bool antialias) {
        using Mode               = sd::ops::InterpolateMode;
        const bool nearest_exact = mode == Mode::NearestExact;
        if (mode == Mode::Nearest || (nearest_exact && antialias))
            return sd::ops::interpolate(input, std::move(shape), mode, false, antialias);
        if ((!nearest_exact && mode != Mode::Bilinear) || antialias) {
            const sd::Tensor<double> values(input.shape(), std::vector<double>(input.values().begin(), input.values().end()));
            auto resized = sd::ops::interpolate(values, std::move(shape), mode, false, antialias);
            sd::Tensor<uint8_t> output(resized.shape());
            std::transform(resized.values().begin(), resized.values().end(), output.values().begin(), rounded_byte);
            return output;
        }

        // Keep half-pixel sampling and rounding for byte images.
        const auto iw = input.shape()[0], ih = input.shape()[1];
        const auto ow = shape[0], oh = shape[1];
        sd::Tensor<uint8_t> output(std::move(shape));
        parallel_for(0, output.numel() / ow, std::max<int64_t>(1, 16384 / ow), [&](int64_t begin, int64_t end) {
            for (int64_t row = begin; row < end; ++row) {
                const auto* plane = input.data() + row / oh * iw * ih;
                if (nearest_exact) {
                    const auto y = (2 * (row % oh) + 1) * ih / (2 * oh);
                    for (int64_t x = 0; x < ow; ++x)
                        output.data()[row * ow + x] = plane[y * iw + (2 * x + 1) * iw / (2 * ow)];
                    continue;
                }
                const auto sy   = std::max(0.0, (row % oh + 0.5) * ih / oh - 0.5);
                const auto y0   = std::min(ih - 1, int64_t(sy));
                const auto y1   = std::min(ih - 1, y0 + 1);
                const double fy = sy - y0;
                for (int64_t x = 0; x < ow; ++x) {
                    const auto sx               = std::max(0.0, (x + 0.5) * iw / ow - 0.5);
                    const auto x0               = std::min(iw - 1, int64_t(sx));
                    const auto x1               = std::min(iw - 1, x0 + 1);
                    const double fx             = sx - x0;
                    const double value          = (1 - fy) * ((1 - fx) * plane[y0 * iw + x0] + fx * plane[y0 * iw + x1]) +
                                                  fy * ((1 - fx) * plane[y1 * iw + x0] + fx * plane[y1 * iw + x1]);
                    output.data()[row * ow + x] = rounded_byte(value);
                }
            }
        });
        return output;
    }

    std::shared_ptr<sd_tensor_t> interpolate_tensor(const sd_tensor_t& input, const std::vector<int64_t>& size, const std::vector<int64_t>& axes, const std::string& name, bool antialias) {
        if (size.size() != 2 || size[0] < 1 || size[1] < 1 || axes.size() != 2 || axes[0] == axes[1] || !input.bytes)
            throw std::runtime_error("interpolation requires a nonempty tensor, two positive sizes and two distinct axes");
        const auto x = tensor_axis(input, axes[0]), y = tensor_axis(input, axes[1]);
        auto shape = input.shape();
        shape[x]   = size[0];
        shape[y]   = size[1];
        element_count(shape, input.type);
        using Mode = sd::ops::InterpolateMode;
        static const std::map<std::string, Mode> modes{
            {"nearest", Mode::Nearest},
            {"nearest_exact", Mode::NearestExact},
            {"nearest_max", Mode::NearestMax},
            {"bilinear", Mode::Bilinear},
            {"bicubic", Mode::Bicubic},
            {"lanczos", Mode::Lanczos},
        };
        const auto mode = modes.at(name);
        auto resize     = [&](auto tag) {
            using T       = decltype(tag);
            auto resample = [&](const sd::Tensor<T>& source, std::vector<int64_t> target) {
                if constexpr (std::is_same_v<T, uint8_t>)
                    return interpolate_bytes(source, std::move(target), mode, antialias);
                else
                    return sd::ops::interpolate(source, std::move(target), mode, false, antialias);
            };
            auto result = apply_spatial(input.as<T>(), x, y, [&](const sd::Tensor<T>& source) {
                auto target = source.shape();
                target[0]   = size[0];
                target[1]   = size[1];
                return resample(source, std::move(target));
            });
            return make_tensor(std::move(result));
        };
        if (input.type == SD_TENSOR_U8)
            return resize(uint8_t{});
        return with_float_dtype(input.type, resize);
    }
    std::shared_ptr<sd_tensor_t> filled_tensor(std::vector<int64_t> shape, sd_tensor_type_t dtype, const TensorNumber& value) {
        const auto count = element_count(shape, dtype);
        return with_dtype(dtype, [&](auto tag) {
            using T = decltype(tag);
            std::vector<T> data(count, scalar_number<T>(value));
            return tensor_result(std::move(shape), std::move(data));
        });
    }
    std::shared_ptr<sd_tensor_t> unsqueeze_tensor(const std::shared_ptr<sd_tensor_t>& input, int64_t axis) {
        auto shape      = input->shape();
        const auto rank = int64_t(shape.size());
        if (axis < 0)
            axis += rank + 1;
        if (axis < 0 || axis > rank)
            throw std::runtime_error("unsqueeze axis must be in [-rank-1, rank]");
        if (rank == 8)
            throw std::runtime_error("unsqueeze would exceed the maximum tensor rank of 8");
        shape.insert(shape.begin() + axis, 1);
        return make_tensor(input->type, std::move(shape), input->data(), input->bytes);
    }

    std::shared_ptr<sd_tensor_t> pad_tensor(const std::shared_ptr<sd_tensor_t>& input, const std::vector<int64_t>& pad, const TensorNumber& value) {
        if (pad.size() % 2 || pad.size() / 2 > input->shape().size())
            throw std::runtime_error("pad requires an even number of values and at most one pair per tensor axis");
        auto shape = input->shape();
        std::vector<int64_t> offset(shape.size(), 0);
        bool changed = false;
        for (size_t axis = 0; axis < pad.size() / 2; ++axis) {
            const auto before = pad[2 * axis], after = pad[2 * axis + 1];
            if (before < 0 || after < 0)
                throw std::runtime_error("padding lengths must be nonnegative");
            if (before > INT64_MAX - shape[axis] || after > INT64_MAX - shape[axis] - before)
                throw std::runtime_error("padded tensor shape exceeds int64 range");
            shape[axis] += before + after;
            offset[axis] = before;
            changed      = changed || before != 0 || after != 0;
        }
        if (!changed) {
            with_dtype(input->type, [&](auto tag) { scalar_number<decltype(tag)>(value); });
            return input;
        }
        auto result = filled_tensor(std::move(shape), input->type, value);
        copy_tensor_box(*input, *result, input->shape(), std::vector<int64_t>(offset.size(), 0), offset);
        return result;
    }

    std::shared_ptr<sd_tensor_t> cast_tensor(const std::shared_ptr<sd_tensor_t>& input, sd_tensor_type_t dtype) {
        if (input->type == dtype)
            return input;
        const auto count = element_count(input->shape(), dtype);
        if (!count)
            return tensor_result(dtype, input->shape(), nullptr, 0);
        return with_dtype(input->type, [&](auto source) {
            using From       = decltype(source);
            const auto* data = static_cast<const From*>(input->data());
            return with_dtype(dtype, [&](auto target) {
                using To = decltype(target);
                std::vector<To> output(count);
                std::transform(data, data + count, output.begin(), cast_number<To, From>);
                return tensor_result(input->shape(), std::move(output));
            });
        });
    }
    std::shared_ptr<sd_tensor_t> translate_tensor(const std::shared_ptr<sd_tensor_t>& input, const std::vector<int64_t>& axes, const std::vector<int64_t>& offset, const TensorNumber& value) {
        if (axes.empty() || axes.size() != offset.size())
            throw std::runtime_error("translate requires matching axes and offsets");
        auto shape = input->shape();
        std::vector<int64_t> source(shape.size(), 0), target(shape.size(), 0);
        std::vector<bool> used(shape.size(), false);
        for (size_t i = 0; i < axes.size(); ++i) {
            const auto axis = tensor_axis(*input, axes[i]);
            if (used[axis])
                throw std::runtime_error("translate axes must be distinct");
            used[axis]       = true;
            const auto delta = std::clamp(offset[i], -shape[axis], shape[axis]);
            source[axis]     = std::max<int64_t>(0, -delta);
            target[axis]     = std::max<int64_t>(0, delta);
            shape[axis] -= source[axis] + target[axis];
        }
        auto result = filled_tensor(input->shape(), input->type, value);
        copy_tensor_box(*input, *result, shape, source, target);
        return result;
    }
    std::shared_ptr<sd_tensor_t> arange_tensor(int64_t start, int64_t end, int64_t step) {
        if (!step)
            throw std::runtime_error("arange step cannot be zero");
        const bool ascending = step > 0;
        const auto distance  = ascending && start < end    ? uint64_t(end) - uint64_t(start)
                               : !ascending && start > end ? uint64_t(start) - uint64_t(end)
                                                           : uint64_t(0);
        const auto stride    = ascending ? uint64_t(step) : uint64_t(0) - uint64_t(step);
        const auto count     = distance ? 1 + (distance - 1) / stride : 0;
        if (count > tensor_byte_limit / sizeof(int64_t))
            throw std::runtime_error("arange tensor exceeds 1 GiB");
        std::vector<int64_t> values(size_t(count), start);
        for (size_t i = 1; i < values.size(); ++i)
            values[i] = values[i - 1] + step;
        return make_tensor(SD_TENSOR_I64, std::vector<int64_t>{int64_t(values.size())}, values.data(), values.size() * sizeof(int64_t));
    }
    std::shared_ptr<sd_tensor_t> index_select_tensor(const std::shared_ptr<sd_tensor_t>& input, int64_t axis_arg, const std::vector<int64_t>& indices) {
        const auto axis      = tensor_axis(*input, axis_arg);
        const auto axis_size = input->shape()[axis];
        for (auto index : indices)
            if (index < 0 || index >= axis_size)
                throw std::runtime_error("index_select index falls outside the tensor axis");
        auto shape  = input->shape();
        shape[axis] = int64_t(indices.size());
        auto output = make_tensor(input->type, std::move(shape));
        if (output->bytes) {
            const auto inner_bytes = size_t(sd::tensor_compute_strides(input->shape())[axis]) * element_size(input->type);
            const auto* source     = static_cast<const uint8_t*>(input->data());
            auto* target           = static_cast<uint8_t*>(output->mutable_data());
            const auto rows        = output->bytes / inner_bytes;
            parallel_for(0, rows, std::max<int64_t>(1, 16384 / inner_bytes), [&](int64_t begin, int64_t end) {
                for (int64_t row = begin; row < end; ++row) {
                    const auto offset = row / indices.size() * axis_size + indices[row % indices.size()];
                    std::memcpy(target + row * inner_bytes, source + offset * inner_bytes, inner_bytes);
                }
            });
        }
        return output;
    }
    std::shared_ptr<sd_tensor_t> replace_slice(const std::shared_ptr<sd_tensor_t>& input, const std::shared_ptr<sd_tensor_t>& value, int64_t axis_arg, int64_t start) {
        const auto axis = tensor_axis(*input, axis_arg);
        if (input->type != value->type || input->shape().size() != value->shape().size())
            throw std::runtime_error("replacement must have the same dtype and rank");
        if (start > input->shape()[axis] || value->shape()[axis] > input->shape()[axis] - start)
            throw std::runtime_error("replacement slice falls outside the tensor");
        auto expected  = input->shape();
        expected[axis] = value->shape()[axis];
        if (expected != value->shape())
            throw std::runtime_error("replacement shape differs outside the slice axis");
        if (!value->bytes)
            return input;
        return with_dtype(input->type, [&](auto tag) {
            using T     = decltype(tag);
            auto result = input->as<T>();
            sd::ops::slice_assign(&result, axis, start, start + value->shape()[axis], value->as<T>());
            return tensor_result(std::move(result));
        });
    }
    std::shared_ptr<sd_tensor_t> concat_tensors(const std::vector<std::shared_ptr<sd_tensor_t>>& items, int64_t axis_arg) {
        if (items.empty())
            throw std::runtime_error("concat requires at least one tensor");
        const auto& first = *items.front();
        const auto axis   = tensor_axis(first, axis_arg);
        auto shape        = first.shape();
        shape[axis]       = 0;
        for (const auto& item : items) {
            const auto& input = *item;
            auto expected     = first.shape();
            if (input.type != first.type || input.shape().size() != expected.size())
                throw std::runtime_error("concat requires matching dtypes and ranks");
            expected[axis] = input.shape()[axis];
            if (expected != input.shape())
                throw std::runtime_error("tensor shapes differ outside the concat axis");
            if (input.shape()[axis] > INT64_MAX - shape[axis])
                throw std::runtime_error("concatenated axis exceeds int64 range");
            shape[axis] += input.shape()[axis];
        }
        if (!element_count(shape, first.type))
            return tensor_result(first.type, std::move(shape), nullptr, 0);
        return with_dtype(first.type, [&](auto tag) {
            using T = decltype(tag);
            sd::Tensor<T> result(shape);
            int64_t start = 0;
            for (const auto& item : items) {
                const auto& input = *item;
                const auto end    = start + input.shape()[axis];
                if (input.bytes)
                    sd::ops::slice_assign(&result, axis, start, end, input.as<T>());
                start = end;
            }
            return tensor_result(std::move(result));
        });
    }
    std::shared_ptr<sd_tensor_t> grayscale_tensor(const std::shared_ptr<sd_tensor_t>& input, int64_t axis_arg) {
        const auto axis     = tensor_axis(*input, axis_arg);
        const auto channels = input->shape()[axis];
        if (input->type != SD_TENSOR_U8 && input->type != SD_TENSOR_F32 && input->type != SD_TENSOR_F64)
            throw std::runtime_error("grayscale requires u8, f32 or f64");
        if (channels == 1)
            return input;
        if (channels != 3 && channels != 4)
            throw std::runtime_error("grayscale requires one, three or four channels");
        auto shape  = input->shape();
        shape[axis] = 1;
        auto result = make_tensor(input->type, std::move(shape));
        if (!result->bytes)
            return result;
        const auto stride = sd::tensor_compute_strides(input->shape())[axis];
        auto convert      = [&](auto tag) {
            using T         = decltype(tag);
            const auto* src = static_cast<const T*>(input->data());
            auto* dst       = static_cast<T*>(result->mutable_data());
            sd::tensor_for_each(result->bytes / sizeof(T), [&](int64_t i) {
                const auto offset = i / stride * stride * channels + i % stride;
                const auto red = src[offset], green = src[offset + stride], blue = src[offset + 2 * stride];
                if constexpr (std::is_same_v<T, uint8_t>)
                    dst[i] = uint8_t((77 * red + 150 * green + 29 * blue + 128) / 256);
                else
                    dst[i] = (T(77) * red + T(150) * green + T(29) * blue) / T(256);
            });
            return result;
        };
        if (input->type == SD_TENSOR_U8)
            return convert(uint8_t{});
        return with_float_dtype(input->type, convert);
    }
    std::shared_ptr<sd_tensor_t> threshold_tensor(const std::shared_ptr<sd_tensor_t>& input, const TensorNumber& threshold, const TensorNumber& low_value, const TensorNumber& high_value) {
        auto result = make_tensor(input->type, input->shape());
        return with_dtype(input->type, [&](auto tag) {
            using T        = decltype(tag);
            const auto low = scalar_number<T>(low_value), high = scalar_number<T>(high_value);
            const auto* src   = static_cast<const T*>(input->data());
            auto* dst         = static_cast<T*>(result->mutable_data());
            const auto cutoff = std::visit([](auto value) { return double(value); }, threshold);
            if constexpr (std::is_integral_v<T>) {
                if (std::holds_alternative<int64_t>(threshold)) {
                    const auto integer = std::get<int64_t>(threshold);
                    sd::tensor_for_each(result->bytes / sizeof(T), [&](int64_t i) {
                        dst[i] = int64_t(src[i]) >= integer ? high : low;
                    });
                    return result;
                }
            }
            sd::tensor_for_each(result->bytes / sizeof(T), [&](int64_t i) {
                dst[i] = double(src[i]) >= cutoff ? high : low;
            });
            return result;
        });
    }
    std::shared_ptr<sd_tensor_t> maximum_tensor(const std::shared_ptr<sd_tensor_t>& a, const std::shared_ptr<sd_tensor_t>& b) {
        if (a->type != b->type)
            throw std::runtime_error("maximum requires matching tensor dtypes");
        auto result = make_tensor(a->type, sd::tensor_broadcast_shape(a->shape(), b->shape()));
        if (!result->bytes)
            return result;
        const auto a_shape   = a->shape().empty() ? std::vector<int64_t>{1} : a->shape();
        const auto b_shape   = b->shape().empty() ? std::vector<int64_t>{1} : b->shape();
        const auto out_shape = result->shape().empty() ? std::vector<int64_t>{1} : result->shape();
        return with_dtype(a->type, [&](auto tag) {
            using T           = decltype(tag);
            const auto* left  = static_cast<const T*>(a->data());
            const auto* right = static_cast<const T*>(b->data());
            auto* dst         = static_cast<T*>(result->mutable_data());
            sd::tensor_for_each_broadcast_offset(out_shape, a_shape, sd::tensor_compute_strides(a_shape),
                                                 b_shape, sd::tensor_compute_strides(b_shape), [&](int64_t i, int64_t ai, int64_t bi) {
                                                     dst[i] = std::max(left[ai], right[bi]);
                                                 });
            return result;
        });
    }
    std::shared_ptr<sd_tensor_t> permute(const std::shared_ptr<sd_tensor_t>& input, const std::vector<int64_t>& axes) {
        if (axes.size() != input->shape().size())
            throw std::runtime_error("permute requires one axis per tensor dimension");
        auto shape = input->shape();
        std::vector<bool> used(axes.size());
        std::vector<size_t> order(axes.size());
        bool identical = true;
        for (size_t i = 0; i < axes.size(); ++i) {
            const auto axis = tensor_axis(*input, axes[i]);
            if (used[axis])
                throw std::runtime_error("permute axes must be unique");
            used[axis] = true;
            order[i]   = axis;
            identical  = identical && axis == i;
            shape[i]   = input->shape()[axis];
        }
        if (identical)
            return input;
        if (!input->bytes)
            return tensor_result(input->type, std::move(shape), nullptr, 0);
        return with_dtype(input->type, [&](auto tag) {
            using T = decltype(tag);
            return tensor_result(input->as<T>().permute(order));
        });
    }
    std::shared_ptr<sd_tensor_t> broadcast_to(const std::shared_ptr<sd_tensor_t>& input, std::vector<int64_t> dims) {
        element_count(dims, input->type);
        if (sd::tensor_broadcast_shape(input->shape(), dims) != dims)
            throw std::runtime_error("input cannot broadcast to the target shape");
        if (dims == input->shape())
            return input;
        auto result = make_tensor(input->type, std::move(dims));
        if (result->bytes) {
            const auto source_shape = input->shape().empty() ? std::vector<int64_t>{1} : input->shape();
            const auto strides      = sd::tensor_compute_strides(source_shape);
            const auto stride       = element_size(input->type);
            const auto* from        = static_cast<const uint8_t*>(input->data());
            auto* to                = static_cast<uint8_t*>(result->mutable_data());
            sd::tensor_for_each_broadcast_offset(result->shape(), source_shape, strides, source_shape, strides,
                                                 [&](int64_t index, int64_t source, int64_t) {
                                                     std::memcpy(to + index * stride, from + source * stride, stride);
                                                 });
        }
        return result;
    }
    std::shared_ptr<sd_tensor_t> index_fill(const std::shared_ptr<sd_tensor_t>& input, int64_t axis_arg, const std::vector<int64_t>& indices, const TensorNumber& fill_value) {
        const auto axis = tensor_axis(*input, axis_arg);
        for (auto index : indices)
            if (index < 0 || index >= input->shape()[axis])
                throw std::runtime_error("fill index falls outside the tensor");
        if (indices.empty() || !input->bytes)
            return input;
        return with_dtype(input->type, [&](auto tag) {
            using T          = decltype(tag);
            auto result      = input->as<T>();
            const auto value = scalar_number<T>(fill_value);
            for (auto index : indices)
                sd::ops::fill_slice(&result, axis, index, index + 1, value);
            return tensor_result(std::move(result));
        });
    }
    std::shared_ptr<sd_tensor_t> randn_like(const std::shared_ptr<sd_tensor_t>& like, const std::shared_ptr<RNG>& rng) {
        return with_float_dtype(like->type, [&](auto tag) {
            using T     = decltype(tag);
            auto values = rng->randn(uint32_t(like->bytes / sizeof(T)));
            if constexpr (std::is_same_v<T, float>) {
                return tensor_result(like->shape(), std::move(values));
            } else {
                std::vector<T> output(values.begin(), values.end());
                return tensor_result(like->shape(), std::move(output));
            }
        });
    }
    int64_t count_nonzero(const std::shared_ptr<sd_tensor_t>& input) {
        if (!input->bytes)
            return 0;
        return with_dtype(input->type, [&](auto zero) {
            const auto* data = static_cast<const decltype(zero)*>(input->data());
            return int64_t(std::count_if(data, data + input->bytes / sizeof(zero),
                                         [zero](auto value) { return value != zero; }));
        });
    }
    std::shared_ptr<sd_tensor_t> lerp_bytes(const sd_tensor_t& a, const sd_tensor_t& b, const sd_tensor_t* w, double scalar) {
        if (a.type != SD_TENSOR_U8 || b.type != SD_TENSOR_U8)
            throw std::runtime_error("u8 lerp requires two u8 endpoints");
        auto shape = sd::tensor_broadcast_shape(a.shape(), b.shape());
        if (w) {
            if (w->type != SD_TENSOR_F32 && w->type != SD_TENSOR_F64)
                throw std::runtime_error("u8 lerp requires a floating-point weight tensor or scalar");
            shape = sd::tensor_broadcast_shape(shape, w->shape());
        }
        auto result = make_tensor(SD_TENSOR_U8, std::move(shape));
        if (!result->bytes)
            return result;
        const auto a_shape   = a.shape().empty() ? std::vector<int64_t>{1} : a.shape();
        const auto b_shape   = b.shape().empty() ? std::vector<int64_t>{1} : b.shape();
        const auto out_shape = result->shape().empty() ? std::vector<int64_t>{1} : result->shape();
        const auto w_strides = w ? sd::tensor_compute_strides(w->shape()) : std::vector<int64_t>{};
        const auto* a_data   = static_cast<const uint8_t*>(a.data());
        const auto* b_data   = static_cast<const uint8_t*>(b.data());
        auto* out            = static_cast<uint8_t*>(result->mutable_data());
        sd::tensor_for_each_broadcast_offset(out_shape, a_shape, sd::tensor_compute_strides(a_shape),
                                             b_shape, sd::tensor_compute_strides(b_shape), [&](int64_t index, int64_t ai, int64_t bi) {
                                                 double blend = scalar;
                                                 if (w) {
                                                     int64_t wi = 0, coord = index;
                                                     for (size_t axis = 0; axis < w->shape().size(); ++axis) {
                                                         if (w->shape()[axis] > 1)
                                                             wi += coord % out_shape[axis] * w_strides[axis];
                                                         coord /= out_shape[axis];
                                                     }
                                                     blend = w->type == SD_TENSOR_F32 ? static_cast<const float*>(w->data())[wi]
                                                                                      : static_cast<const double*>(w->data())[wi];
                                                 }
                                                 out[index] = rounded_byte(a_data[ai] * (1.0 - blend) + b_data[bi] * blend);
                                             });
        return result;
    }
}
