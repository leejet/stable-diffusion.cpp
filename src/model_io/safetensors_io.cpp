#include "safetensors_io.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include "binary_io.h"
#include "json.hpp"
#include "util.h"

static constexpr size_t ST_HEADER_SIZE_LEN = 8;

static void set_error(std::string* error, const std::string& message) {
    if (error != nullptr) {
        *error = message;
    }
}

#ifdef _WIN32
#define sd_file_seek _fseeki64
#else
#define sd_file_seek fseeko
#endif

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

static ggml_type safetensors_dtype_to_ggml_type(const std::string& dtype) {
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

        ggml_type type = safetensors_dtype_to_ggml_type(dtype);
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

static bool ggml_type_to_safetensors_dtype(ggml_type type, std::string* dtype) {
    switch (type) {
        case GGML_TYPE_F16:
            *dtype = "F16";
            return true;
        case GGML_TYPE_BF16:
            *dtype = "BF16";
            return true;
        case GGML_TYPE_F32:
            *dtype = "F32";
            return true;
        case GGML_TYPE_I32:
            *dtype = "I32";
            return true;
        default:
            return false;
    }
}

bool write_safetensors_file(const std::string& file_path,
                            const std::vector<TensorWriteInfo>& tensors,
                            std::string* error) {
    nlohmann::ordered_json header = nlohmann::ordered_json::object();

    uint64_t data_offset = 0;
    for (const TensorWriteInfo& write_tensor : tensors) {
        ggml_tensor* tensor = write_tensor.tensor;
        if (tensor == nullptr) {
            set_error(error, "null tensor cannot be written to safetensors");
            return false;
        }

        const std::string name = ggml_get_name(tensor);
        std::string dtype;
        if (!ggml_type_to_safetensors_dtype(tensor->type, &dtype)) {
            set_error(error,
                      "unsupported safetensors dtype '" + std::string(ggml_type_name(tensor->type)) +
                          "' for tensor '" + name + "'");
            return false;
        }

        const uint64_t tensor_nbytes = ggml_nbytes(tensor);

        nlohmann::ordered_json json_tensor_info = nlohmann::ordered_json::object();
        json_tensor_info["dtype"]               = dtype;

        nlohmann::ordered_json shape = nlohmann::ordered_json::array();
        for (int i = 0; i < write_tensor.n_dims; ++i) {
            shape.push_back(write_tensor.ne[write_tensor.n_dims - 1 - i]);
        }
        json_tensor_info["shape"] = shape;

        nlohmann::ordered_json data_offsets = nlohmann::ordered_json::array();
        data_offsets.push_back(data_offset);
        data_offsets.push_back(data_offset + tensor_nbytes);
        json_tensor_info["data_offsets"] = data_offsets;

        header[name] = json_tensor_info;
        data_offset += tensor_nbytes;
    }

    const std::string header_str = header.dump();

    std::ofstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        set_error(error, "failed to open '" + file_path + "' for writing");
        return false;
    }

    LOG_INFO("trying to save tensors to %s", file_path.c_str());
    model_io::write_u64(file, header_str.size());
    file.write(header_str.data(), header_str.size());
    if (!file) {
        set_error(error, "failed to write safetensors header to '" + file_path + "'");
        return false;
    }

    for (const TensorWriteInfo& write_tensor : tensors) {
        ggml_tensor* tensor        = write_tensor.tensor;
        const std::string name     = ggml_get_name(tensor);
        const size_t tensor_nbytes = ggml_nbytes(tensor);
        file.write((const char*)tensor->data, tensor_nbytes);
        if (!file) {
            set_error(error,
                      "failed to write tensor '" + name + "' to '" + file_path + "'");
            return false;
        }
    }

    return true;
}

SafetensorsStreamingWriter::~SafetensorsStreamingWriter() {
    close();
}

bool SafetensorsStreamingWriter::open(const std::string& file_path,
                                      const std::vector<TensorWritePlan>& tensors,
                                      int n_writers,
                                      std::string* error) {
    close();
    file_path_ = file_path;
    tensors_   = tensors;

    nlohmann::ordered_json header = nlohmann::ordered_json::object();
    uint64_t data_offset          = 0;
    tensor_offsets_.resize(tensors.size());
    for (size_t i = 0; i < tensors.size(); i++) {
        const TensorWritePlan& plan = tensors[i];
        std::string dtype;
        if (!ggml_type_to_safetensors_dtype(plan.type, &dtype)) {
            set_error(error,
                      "unsupported safetensors dtype '" + std::string(ggml_type_name(plan.type)) +
                          "' for tensor '" + plan.name + "'");
            close();
            return false;
        }

        nlohmann::ordered_json json_tensor_info = nlohmann::ordered_json::object();
        json_tensor_info["dtype"]               = dtype;

        nlohmann::ordered_json shape = nlohmann::ordered_json::array();
        for (int j = 0; j < plan.n_dims; ++j) {
            shape.push_back(plan.ne[plan.n_dims - 1 - j]);
        }
        json_tensor_info["shape"] = shape;

        nlohmann::ordered_json data_offsets = nlohmann::ordered_json::array();
        data_offsets.push_back(data_offset);
        data_offsets.push_back(data_offset + plan.nbytes());
        json_tensor_info["data_offsets"] = data_offsets;

        header[plan.name]    = json_tensor_info;
        tensor_offsets_[i]   = data_offset;
        data_offset         += plan.nbytes();
    }

    const std::string header_str = header.dump();
    data_start_                  = ST_HEADER_SIZE_LEN + header_str.size();

    LOG_INFO("trying to save tensors to %s", file_path.c_str());
    FILE* file = fopen(file_path.c_str(), "wb+");
    if (file == nullptr) {
        set_error(error, "failed to open '" + file_path + "' for writing");
        close();
        return false;
    }
    files_.push_back(file);

    uint8_t header_size[ST_HEADER_SIZE_LEN];
    for (int i = 0; i < static_cast<int>(ST_HEADER_SIZE_LEN); ++i) {
        header_size[i] = static_cast<uint8_t>((header_str.size() >> (8 * i)) & 0xFF);
    }
    if (fwrite(header_size, 1, sizeof(header_size), file) != sizeof(header_size) ||
        fwrite(header_str.data(), 1, header_str.size(), file) != header_str.size()) {
        set_error(error, "failed to write safetensors header to '" + file_path + "'");
        close();
        return false;
    }

    const uint64_t file_size = data_start_ + data_offset;
    if (file_size > 0 && sd_file_seek(file, static_cast<int64_t>(file_size - 1), SEEK_SET) != 0) {
        set_error(error, "failed to preallocate safetensors file '" + file_path + "'");
        close();
        return false;
    }
    if (file_size > 0 && fputc(0, file) == EOF) {
        set_error(error, "failed to preallocate safetensors file '" + file_path + "'");
        close();
        return false;
    }
    fflush(file);

    n_writers = std::max(1, n_writers);
    for (int i = 1; i < n_writers; i++) {
        FILE* writer_file = fopen(file_path.c_str(), "rb+");
        if (writer_file == nullptr) {
            set_error(error, "failed to open output file handle for '" + file_path + "'");
            close();
            return false;
        }
        files_.push_back(writer_file);
    }
    return true;
}

bool SafetensorsStreamingWriter::write_tensor(size_t tensor_index,
                                              const uint8_t* data,
                                              size_t size,
                                              int writer_index,
                                              std::string* error) {
    if (tensor_index >= tensors_.size() || tensor_index >= tensor_offsets_.size()) {
        set_error(error, "invalid safetensors tensor index");
        return false;
    }
    if (writer_index < 0 || writer_index >= static_cast<int>(files_.size())) {
        set_error(error, "invalid safetensors writer index");
        return false;
    }
    const TensorWritePlan& plan = tensors_[tensor_index];
    if (size != plan.nbytes()) {
        set_error(error, "size mismatch while writing tensor '" + plan.name + "'");
        return false;
    }
    FILE* file = files_[writer_index];
    if (sd_file_seek(file, static_cast<int64_t>(data_start_ + tensor_offsets_[tensor_index]), SEEK_SET) != 0) {
        set_error(error, "failed to seek output for tensor '" + plan.name + "'");
        return false;
    }
    if (size > 0 && fwrite(data, 1, size, file) != size) {
        set_error(error, "failed to write tensor '" + plan.name + "' to '" + file_path_ + "'");
        return false;
    }
    return true;
}

void SafetensorsStreamingWriter::close() {
    for (FILE* file : files_) {
        if (file != nullptr) {
            fclose(file);
        }
    }
    files_.clear();
    tensor_offsets_.clear();
    tensors_.clear();
    data_start_ = 0;
}
