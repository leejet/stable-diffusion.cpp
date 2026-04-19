#include "safetensors_io.h"

#include <cstdint>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include "binary_io.h"
#include "json.hpp"

static constexpr size_t ST_HEADER_SIZE_LEN = 8;

static void set_error(std::string* error, const std::string& message) {
    if (error != nullptr) {
        *error = message;
    }
}

bool is_safetensors_file(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // get file size
    file.seekg(0, file.end);
    size_t file_size_ = file.tellg();
    file.seekg(0, file.beg);

    // read header size
    if (file_size_ <= ST_HEADER_SIZE_LEN) {
        return false;
    }

    uint8_t header_size_buf[ST_HEADER_SIZE_LEN];
    file.read((char*)header_size_buf, ST_HEADER_SIZE_LEN);
    if (!file) {
        return false;
    }

    size_t header_size_ = model_io::read_u64(header_size_buf);
    if (header_size_ >= file_size_ || header_size_ <= 2) {
        return false;
    }

    // read header
    std::vector<char> header_buf;
    header_buf.resize(header_size_ + 1);
    header_buf[header_size_] = '\0';
    file.read(header_buf.data(), header_size_);
    if (!file) {
        return false;
    }
    try {
        nlohmann::json header_ = nlohmann::json::parse(header_buf.data());
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

static ggml_type str_to_ggml_type(const std::string& dtype) {
    ggml_type ttype = GGML_TYPE_COUNT;
    if (dtype == "F16") {
        ttype = GGML_TYPE_F16;
    } else if (dtype == "BF16") {
        ttype = GGML_TYPE_BF16;
    } else if (dtype == "F32") {
        ttype = GGML_TYPE_F32;
    } else if (dtype == "F64") {
        ttype = GGML_TYPE_F32;
    } else if (dtype == "F8_E4M3") {
        ttype = GGML_TYPE_F16;
    } else if (dtype == "F8_E5M2") {
        ttype = GGML_TYPE_F16;
    } else if (dtype == "I32") {
        ttype = GGML_TYPE_I32;
    } else if (dtype == "I64") {
        ttype = GGML_TYPE_I32;
    }
    return ttype;
}

// https://huggingface.co/docs/safetensors/index
bool read_safetensors_file(const std::string& file_path,
                           std::vector<TensorStorage>& tensor_storages,
                           std::string* error) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        set_error(error, "failed to open '" + file_path + "'");
        return false;
    }

    // get file size
    file.seekg(0, file.end);
    size_t file_size_ = file.tellg();
    file.seekg(0, file.beg);

    // read header size
    if (file_size_ <= ST_HEADER_SIZE_LEN) {
        set_error(error, "invalid safetensor file '" + file_path + "'");
        return false;
    }

    uint8_t header_size_buf[ST_HEADER_SIZE_LEN];
    file.read((char*)header_size_buf, ST_HEADER_SIZE_LEN);
    if (!file) {
        set_error(error, "read safetensors header size failed: '" + file_path + "'");
        return false;
    }

    size_t header_size_ = model_io::read_u64(header_size_buf);
    if (header_size_ >= file_size_) {
        set_error(error, "invalid safetensor file '" + file_path + "'");
        return false;
    }

    // read header
    std::vector<char> header_buf;
    header_buf.resize(header_size_ + 1);
    header_buf[header_size_] = '\0';
    file.read(header_buf.data(), header_size_);
    if (!file) {
        set_error(error, "read safetensors header failed: '" + file_path + "'");
        return false;
    }

    nlohmann::json header_;
    try {
        header_ = nlohmann::json::parse(header_buf.data());
    } catch (const std::exception&) {
        set_error(error, "parsing safetensors header failed: '" + file_path + "'");
        return false;
    }

    tensor_storages.clear();
    for (auto& item : header_.items()) {
        std::string name           = item.key();
        nlohmann::json tensor_info = item.value();
        // LOG_DEBUG("%s %s\n", name.c_str(), tensor_info.dump().c_str());

        if (name == "__metadata__") {
            continue;
        }

        std::string dtype    = tensor_info["dtype"];
        nlohmann::json shape = tensor_info["shape"];

        if (dtype == "U8") {
            continue;
        }

        size_t begin = tensor_info["data_offsets"][0].get<size_t>();
        size_t end   = tensor_info["data_offsets"][1].get<size_t>();

        ggml_type type = str_to_ggml_type(dtype);
        if (type == GGML_TYPE_COUNT) {
            set_error(error, "unsupported dtype '" + dtype + "' (tensor '" + name + "')");
            return false;
        }

        if (shape.size() > SD_MAX_DIMS) {
            set_error(error, "invalid tensor '" + name + "'");
            return false;
        }

        int n_dims              = (int)shape.size();
        int64_t ne[SD_MAX_DIMS] = {1, 1, 1, 1, 1};
        for (int i = 0; i < n_dims; i++) {
            ne[i] = shape[i].get<int64_t>();
        }

        if (n_dims == 5) {
            n_dims = 4;
            ne[0]  = ne[0] * ne[1];
            ne[1]  = ne[2];
            ne[2]  = ne[3];
            ne[3]  = ne[4];
        }

        // ggml_n_dims returns 1 for scalars
        if (n_dims == 0) {
            n_dims = 1;
        }

        TensorStorage tensor_storage(name, type, ne, n_dims, 0, ST_HEADER_SIZE_LEN + header_size_ + begin);
        tensor_storage.reverse_ne();

        size_t tensor_data_size = end - begin;

        bool tensor_size_ok;
        if (dtype == "F8_E4M3") {
            tensor_storage.is_f8_e4m3 = true;
            // f8 -> f16
            tensor_size_ok = (tensor_storage.nbytes() == tensor_data_size * 2);
        } else if (dtype == "F8_E5M2") {
            tensor_storage.is_f8_e5m2 = true;
            // f8 -> f16
            tensor_size_ok = (tensor_storage.nbytes() == tensor_data_size * 2);
        } else if (dtype == "F64") {
            tensor_storage.is_f64 = true;
            // f64 -> f32
            tensor_size_ok = (tensor_storage.nbytes() * 2 == tensor_data_size);
        } else if (dtype == "I64") {
            tensor_storage.is_i64 = true;
            // i64 -> i32
            tensor_size_ok = (tensor_storage.nbytes() * 2 == tensor_data_size);
        } else {
            tensor_size_ok = (tensor_storage.nbytes() == tensor_data_size);
        }
        if (!tensor_size_ok) {
            set_error(error, "size mismatch for tensor '" + name + "' (" + dtype + ")");
            return false;
        }

        tensor_storages.push_back(tensor_storage);

        // LOG_DEBUG("%s %s", tensor_storage.to_string().c_str(), dtype.c_str());
    }

    return true;
}
