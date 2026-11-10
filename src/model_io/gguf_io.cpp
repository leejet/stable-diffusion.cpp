#include "gguf_io.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <ostream>
#include <string>
#include <vector>

#include "core/util.h"
#include "gguf.h"
#include "gguf_reader_ext.h"

static void set_error(std::string* error, const std::string& message) {
    if (error != nullptr) {
        *error = message;
    }
}

// ComfyUI-GGUF stores some tensors flattened and keeps their PyTorch shape
// under this key.
static bool apply_comfy_orig_shape(gguf_context* ctx, TensorStorage& tensor_storage) {
    int64_t key = gguf_find_key(ctx, ("comfy.gguf.orig_shape." + tensor_storage.name).c_str());
    if (key < 0) {
        return true;
    }
    if (gguf_get_kv_type(ctx, key) != GGUF_TYPE_ARRAY || gguf_get_arr_type(ctx, key) != GGUF_TYPE_INT32) {
        return false;
    }
    size_t n_dims = gguf_get_arr_n(ctx, key);
    if (n_dims == 0 || n_dims > SD_MAX_DIMS) {
        return false;
    }
    const auto* dims = static_cast<const int32_t*>(gguf_get_arr_data(ctx, key));
    std::vector<int64_t> shape(dims, dims + n_dims);
    if (shape.size() == 5) {
        shape = {shape[0] * shape[1], shape[2], shape[3], shape[4]};
    }
    TensorStorage reshaped(tensor_storage.name,
                           tensor_storage.type,
                           shape.data(),
                           static_cast<int>(shape.size()),
                           tensor_storage.file_index,
                           tensor_storage.offset);
    reshaped.reverse_ne();
    if (reshaped.nelements() != tensor_storage.nelements() ||
        reshaped.ne[0] % ggml_blck_size(reshaped.type) != 0) {
        return false;
    }
    tensor_storage = reshaped;
    return true;
}

bool is_gguf_file(const std::string& file_path) {
    std::ifstream file(std::filesystem::u8path(file_path), std::ios::binary);
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

    GGUFReader gguf_reader;
    bool probe_ok = gguf_reader.load(file_path);

    if (!probe_ok || !gguf_reader.has_tensors_beyond_ggml_limits()) {
        ctx_gguf_ = gguf_init_from_file(file_path.c_str(), {true, &ctx_meta_});
    }
    if (!ctx_gguf_) {
        if (!probe_ok && !gguf_reader.load(file_path)) {
            set_error(error, "failed to open '" + file_path + "' with GGUFReader");
            return false;
        }

        size_t data_offset = gguf_reader.data_offset();
        for (const auto& gguf_tensor_info : gguf_reader.tensors()) {
#ifdef SD_USE_UPSTREAM_GGML
            if (static_cast<int>(gguf_tensor_info.type) == SD_TYPE_F8_E4M3 ||
                static_cast<int>(gguf_tensor_info.type) == SD_TYPE_F8_E5M2) {
                set_error(error, "FP8 is not supported by this ggml build (tensor '" + gguf_tensor_info.name + "')");
                return false;
            }
#endif
            if (static_cast<unsigned>(gguf_tensor_info.type) >= GGML_TYPE_COUNT ||
                ggml_get_type_traits(gguf_tensor_info.type)->type_size == 0) {
                set_error(error, "unsupported GGUF tensor type (tensor '" + gguf_tensor_info.name + "')");
                return false;
            }
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
        if (!apply_comfy_orig_shape(ctx_gguf_, tensor_storage)) {
            gguf_free(ctx_gguf_);
            ggml_free(ctx_meta_);
            set_error(error, "invalid original shape for tensor '" + name + "'");
            return false;
        }

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

bool GGUFStreamingWriter::write_metadata(const std::string& file_path,
                                         const std::vector<TensorWritePlan>& tensors,
                                         std::string* error) {
    close();
    tensors_   = tensors;
    file_size_ = 0;

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
    FILE* file = ggml_fopen(file_path.c_str(), "wb+");
    if (file == nullptr) {
        set_error(error, "failed to open output file '" + file_path + "'");
        close();
        return false;
    }

    // ggml exposes GGUF metadata writing through FILE* only. Keep FILE usage
    // isolated here; tensor data is written through std::fstream by the shared
    // streaming pipeline.
    if (!gguf_write_to_file_ptr(gguf_ctx_, file, true)) {
        fclose(file);
        set_error(error, "failed to write GGUF metadata to '" + file_path + "'");
        close();
        return false;
    }
    fclose(file);

    const uint64_t data_start = gguf_get_meta_size(gguf_ctx_);
    tensor_offsets_.resize(tensors.size());
    file_size_ = data_start;
    for (size_t i = 0; i < tensors.size(); i++) {
        tensor_offsets_[i] = data_start + gguf_get_tensor_offset(gguf_ctx_, static_cast<int64_t>(i));
        file_size_         = std::max(file_size_, tensor_offsets_[i] + tensors[i].nbytes());
    }
    return true;
}

bool GGUFStreamingWriter::write_tensor(std::ostream& output,
                                       size_t tensor_index,
                                       const uint8_t* data,
                                       size_t size,
                                       std::string* error) const {
    if (tensor_index >= tensors_.size() || tensor_index >= tensor_offsets_.size()) {
        set_error(error, "invalid GGUF tensor index");
        return false;
    }
    const TensorWritePlan& plan = tensors_[tensor_index];
    if (size != plan.nbytes()) {
        set_error(error, "size mismatch while writing tensor '" + plan.name + "'");
        return false;
    }
    output.seekp(static_cast<std::streamoff>(tensor_offsets_[tensor_index]), std::ios::beg);
    if (!output) {
        set_error(error, "failed to seek output for tensor '" + plan.name + "'");
        return false;
    }
    if (size > 0) {
        output.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    }
    if (!output) {
        set_error(error, "failed to write tensor '" + plan.name + "'");
        return false;
    }
    return true;
}

uint64_t GGUFStreamingWriter::file_size() const {
    return file_size_;
}

void GGUFStreamingWriter::close() {
    tensor_offsets_.clear();
    tensors_.clear();
    file_size_ = 0;
    if (gguf_ctx_ != nullptr) {
        gguf_free(gguf_ctx_);
        gguf_ctx_ = nullptr;
    }
    if (meta_ctx_ != nullptr) {
        ggml_free(meta_ctx_);
        meta_ctx_ = nullptr;
    }
}
