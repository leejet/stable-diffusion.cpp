#include "gguf_io.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "gguf.h"
#include "gguf_reader_ext.h"
#include "util.h"

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

bool is_gguf_file(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    char magic[4];

    file.read(magic, sizeof(magic));
    if (!file) {
        return false;
    }
    for (uint32_t i = 0; i < sizeof(magic); i++) {
        if (magic[i] != GGUF_MAGIC[i]) {
            return false;
        }
    }

    return true;
}

bool read_gguf_file(const std::string& file_path,
                    std::vector<TensorStorage>& tensor_storages,
                    std::string* error) {
    tensor_storages.clear();

    gguf_context* ctx_gguf_ = nullptr;
    ggml_context* ctx_meta_ = nullptr;

    ctx_gguf_ = gguf_init_from_file(file_path.c_str(), {true, &ctx_meta_});
    if (!ctx_gguf_) {
        GGUFReader gguf_reader;
        if (!gguf_reader.load(file_path)) {
            set_error(error, "failed to open '" + file_path + "' with GGUFReader");
            return false;
        }

        size_t data_offset = gguf_reader.data_offset();
        for (const auto& gguf_tensor_info : gguf_reader.tensors()) {
            TensorStorage tensor_storage(
                gguf_tensor_info.name,
                gguf_tensor_info.type,
                gguf_tensor_info.shape.data(),
                static_cast<int>(gguf_tensor_info.shape.size()),
                0,
                data_offset + gguf_tensor_info.offset);

            tensor_storages.push_back(tensor_storage);
        }

        return true;
    }

    int n_tensors = static_cast<int>(gguf_get_n_tensors(ctx_gguf_));

    size_t data_offset = gguf_get_data_offset(ctx_gguf_);
    for (int i = 0; i < n_tensors; i++) {
        std::string name   = gguf_get_tensor_name(ctx_gguf_, i);
        ggml_tensor* dummy = ggml_get_tensor(ctx_meta_, name.c_str());
        size_t offset      = data_offset + gguf_get_tensor_offset(ctx_gguf_, i);

        TensorStorage tensor_storage(name, dummy->type, dummy->ne, ggml_n_dims(dummy), 0, offset);

        if (ggml_nbytes(dummy) != tensor_storage.nbytes()) {
            gguf_free(ctx_gguf_);
            ggml_free(ctx_meta_);
            set_error(error, "size mismatch for tensor '" + name + "'");
            return false;
        }

        tensor_storages.push_back(tensor_storage);
    }

    gguf_free(ctx_gguf_);
    ggml_free(ctx_meta_);

    return true;
}

bool write_gguf_file(const std::string& file_path,
                     const std::vector<TensorWriteInfo>& tensors,
                     std::string* error) {
    gguf_context* gguf_ctx = gguf_init_empty();
    if (gguf_ctx == nullptr) {
        set_error(error, "gguf_init_empty failed");
        return false;
    }

    for (const TensorWriteInfo& write_tensor : tensors) {
        ggml_tensor* tensor = write_tensor.tensor;
        if (tensor == nullptr) {
            set_error(error, "null tensor cannot be written to GGUF");
            gguf_free(gguf_ctx);
            return false;
        }
        gguf_add_tensor(gguf_ctx, tensor);
    }

    LOG_INFO("trying to save tensors to %s", file_path.c_str());
    bool success = gguf_write_to_file(gguf_ctx, file_path.c_str(), false);
    if (!success) {
        set_error(error, "failed to write GGUF file '" + file_path + "'");
    }
    gguf_free(gguf_ctx);
    return success;
}

GGUFStreamingWriter::~GGUFStreamingWriter() {
    close();
}

bool GGUFStreamingWriter::open(const std::string& file_path,
                               const std::vector<TensorWritePlan>& tensors,
                               int n_writers,
                               std::string* error) {
    close();
    file_path_ = file_path;
    tensors_   = tensors;

    size_t meta_mem = 1 * 1024 * 1024 + tensors.size() * ggml_tensor_overhead();
    meta_ctx_       = ggml_init({meta_mem, nullptr, true});
    if (meta_ctx_ == nullptr) {
        set_error(error, "ggml_init failed for GGUF metadata");
        return false;
    }

    gguf_ctx_ = gguf_init_empty();
    if (gguf_ctx_ == nullptr) {
        set_error(error, "gguf_init_empty failed");
        close();
        return false;
    }

    for (const TensorWritePlan& plan : tensors) {
        ggml_tensor* tensor = ggml_new_tensor(meta_ctx_, plan.type, plan.n_dims, plan.ne);
        if (tensor == nullptr) {
            set_error(error, "ggml_new_tensor failed for tensor '" + plan.name + "'");
            close();
            return false;
        }
        ggml_set_name(tensor, plan.name.c_str());
        gguf_add_tensor(gguf_ctx_, tensor);
    }

    LOG_INFO("trying to save tensors to %s", file_path.c_str());
    FILE* file = fopen(file_path.c_str(), "wb+");
    if (file == nullptr) {
        set_error(error, "failed to open output file '" + file_path + "'");
        close();
        return false;
    }
    files_.push_back(file);

    if (!gguf_write_to_file_ptr(gguf_ctx_, file, true)) {
        set_error(error, "failed to write GGUF metadata to '" + file_path + "'");
        close();
        return false;
    }

    const uint64_t data_start = gguf_get_meta_size(gguf_ctx_);
    tensor_offsets_.resize(tensors.size());
    uint64_t file_size = data_start;
    for (size_t i = 0; i < tensors.size(); i++) {
        tensor_offsets_[i] = data_start + gguf_get_tensor_offset(gguf_ctx_, static_cast<int64_t>(i));
        file_size          = std::max(file_size, tensor_offsets_[i] + tensors[i].nbytes());
    }

    if (file_size > 0 && sd_file_seek(file, static_cast<int64_t>(file_size - 1), SEEK_SET) != 0) {
        set_error(error, "failed to preallocate GGUF file '" + file_path + "'");
        close();
        return false;
    }
    if (file_size > 0 && fputc(0, file) == EOF) {
        set_error(error, "failed to preallocate GGUF file '" + file_path + "'");
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

bool GGUFStreamingWriter::write_tensor(size_t tensor_index,
                                       const uint8_t* data,
                                       size_t size,
                                       int writer_index,
                                       std::string* error) {
    if (tensor_index >= tensors_.size() || tensor_index >= tensor_offsets_.size()) {
        set_error(error, "invalid GGUF tensor index");
        return false;
    }
    if (writer_index < 0 || writer_index >= static_cast<int>(files_.size())) {
        set_error(error, "invalid GGUF writer index");
        return false;
    }
    const TensorWritePlan& plan = tensors_[tensor_index];
    if (size != plan.nbytes()) {
        set_error(error, "size mismatch while writing tensor '" + plan.name + "'");
        return false;
    }
    FILE* file = files_[writer_index];
    if (sd_file_seek(file, static_cast<int64_t>(tensor_offsets_[tensor_index]), SEEK_SET) != 0) {
        set_error(error, "failed to seek output for tensor '" + plan.name + "'");
        return false;
    }
    if (size > 0 && fwrite(data, 1, size, file) != size) {
        set_error(error, "failed to write tensor '" + plan.name + "'");
        return false;
    }
    return true;
}

void GGUFStreamingWriter::close() {
    for (FILE* file : files_) {
        if (file != nullptr) {
            fclose(file);
        }
    }
    files_.clear();
    tensor_offsets_.clear();
    tensors_.clear();
    if (gguf_ctx_ != nullptr) {
        gguf_free(gguf_ctx_);
        gguf_ctx_ = nullptr;
    }
    if (meta_ctx_ != nullptr) {
        ggml_free(meta_ctx_);
        meta_ctx_ = nullptr;
    }
}
