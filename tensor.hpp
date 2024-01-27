#ifndef __TENSOR_HPP__
#define __TENSOR_HPP__

#include <iostream>
#include <sstream>
#include <vector>

#include "ggml/ggml.h"

template <typename T>
class Tensor {
private:
    std::vector<int> shape_;
    std::vector<T> data_;

public:
    Tensor(const std::vector<int>& shape)
        : shape_(shape) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        data_.resize(size, T());
    }

    Tensor(const std::vector<int>& tensor_shape, const std::vector<T>& tensor_data)
        : shape_(tensor_shape), data_(tensor_data) {
        assert(tensor_data.size() == compute_numel(tensor_shape) && "Data size must match the product of shape dimensions.");
    }

    std::vector<int> sizes() const {
        return shape_;
    }

    int numel() const {
        return data_.size();
    }

    size_t nbytes() const {
        return data_.size() * sizeof(T);
    }

    T item(const std::vector<int>& indices) const {
        int index = indices_to_flat_index(indices);
        if (index < 0 || index >= numel()) {
            GGML_ASSERT(index >= 0 && index < numel() && "Index out of bounds.");
            return T();
        }
        return data_[index];
    }

    T mean() const {
        assert(numel() > 0 && "Cannot compute mean of an empty Tensor.");

        T sum = std::accumulate(data_.begin(), data_.end(), T());
        return sum / static_cast<T>(numel());
    }

    void set_item(const std::vector<int>& indices, T value) {
        int index = indices_to_flat_index(indices);
        if (index < 0 || index >= numel()) {
            GGML_ASSERT(index >= 0 && index < numel() && "Index out of bounds.");
            return;
        }
        data_[index] = value;
    }

    std::string to_string() const {
        std::stringstream ss;
        ss << "Tensor Shape: [";
        for (int i = 0; i < shape_.size(); ++i) {
            ss << shape_[i];
            if (i < shape_.size() - 1) {
                ss << ", ";
            }
        }
        ss << "]\n";

        ss << "Tensor Size: " << numel() << "\n";

        ss << "Data: [";
        for (int i = 0; i < numel(); ++i) {
            ss << data_[i];
            if (i < numel() - 1) {
                ss << ", ";
            }
        }
        ss << "]\n";

        return ss.str();
    }

    void fill(T value) {
        for (int i = 0; i < numel(); ++i) {
            data_[i] = value;
        }
    }

    static Tensor zeros(const std::vector<int>& shape) {
        Tensor zero_tensor(shape);
        zero_tensor.fill(0);
        return zero_tensor;
    }

    static Tensor ones(const std::vector<int>& shape) {
        Tensor zero_tensor(shape);
        zero_tensor.fill(1);
        return zero_tensor;
    }

    static Tensor concat(const Tensor& tensor1, const Tensor& tensor2, int dim) {
        assert(tensor1.shape_.size() == tensor2.sizes().size() && "Tensor dimensions must match.");
        assert(dim >= 0 && dim < tensor1.shape_.size() && "Invalid concatenation dimension.");

        std::vector<int> result_shape = tensor1.shape_;
        result_shape[dim] += tensor2.sizes()[dim];

        Tensor result(result_shape);

        for (int i = 0; i < result.numel(); ++i) {
            std::vector<int> indices = flat_index_to_indices(i, result_shape);
            if (indices[dim] < tensor1.sizes()[dim]) {
                result.set_item(indices, tensor1.item(indices));
            } else {
                indices[dim] -= tensor1.sizes()[dim];
                result.set_item(indices, tensor2.item(indices));
            }
        }

        return result;
    }

private:
    int indices_to_flat_index(const std::vector<int>& indices) const {
        if (indices.size() != shape_.size()) {
            GGML_ASSERT(indices.size() == shape_.size() && "Invalid number of indices.");
            return -1;
        }

        int index  = 0;
        int stride = 1;
        for (int i = shape_.size() - 1; i >= 0; --i) {
            if (indices[i] < 0 || indices[i] >= shape_[i]) {
                GGML_ASSERT(indices[i] >= 0 && indices[i] < shape_[i] && "Index out of bounds for dimension.");
                return -1;
            }
            index += indices[i] * stride;
            stride *= shape_[i];
        }

        return index;
    }

    static std::vector<int> flat_index_to_indices(int flat_index, const std::vector<int>& shape) {
        std::vector<int> indices(shape.size(), 0);

        int remainder = flat_index;
        for (int i = shape.size() - 1; i >= 0; --i) {
            indices[i] = remainder % shape[i];
            remainder /= shape[i];
        }

        return indices;
    }

    static int compute_numel(const std::vector<int>& shape) {
        int numel = 1;
        for (int dim : shape) {
            numel *= dim;
        }
        return numel;
    }
};

#endif  // __TENSOR_HPP__