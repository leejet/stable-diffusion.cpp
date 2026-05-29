#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include "model.h"
#include "model_io/binary_io.h"
#include "model_io/gguf_io.h"
#include "model_io/safetensors_io.h"
#include "json.hpp"
#include "util.h"

#include "ggml-cpu.h"
#include "gguf.h"

#ifdef _WIN32
#define sd_file_seek _fseeki64
#else
#define sd_file_seek fseeko
#endif

struct TensorExportInfo {
    TensorStorage storage;
    ggml_type type;
};

struct TensorExportJob {
    TensorExportInfo info;
    std::vector<uint8_t> data;
    std::string error;
    bool success = false;
};

static size_t export_tensor_nbytes(const TensorExportInfo& info) {
    TensorStorage output_storage = info.storage;
    output_storage.type          = info.type;
    return static_cast<size_t>(output_storage.nbytes());
}

static ggml_type get_export_tensor_type(ModelLoader& model_loader,
                                        const TensorStorage& tensor_storage,
                                        ggml_type type,
                                        const TensorTypeRules& tensor_type_rules) {
    const std::string& name = tensor_storage.name;
    ggml_type tensor_type   = tensor_storage.type;
    ggml_type dst_type      = type;

    for (const auto& tensor_type_rule : tensor_type_rules) {
        std::regex pattern(tensor_type_rule.first);
        if (std::regex_search(name, pattern)) {
            dst_type = tensor_type_rule.second;
            break;
        }
    }

    if (model_loader.tensor_should_be_converted(tensor_storage, dst_type)) {
        tensor_type = dst_type;
    }

    return tensor_type;
}

static bool collect_tensors_for_export(ModelLoader& model_loader,
                                       ggml_type type,
                                       const TensorTypeRules& tensor_type_rules,
                                       std::vector<TensorExportInfo>& tensors) {
    std::mutex tensor_mutex;
    auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool {
        TensorExportInfo info;
        info.storage = tensor_storage;
        info.type    = get_export_tensor_type(model_loader, tensor_storage, type, tensor_type_rules);

        std::lock_guard<std::mutex> lock(tensor_mutex);
        tensors.push_back(std::move(info));
        *dst_tensor = nullptr;
        return true;
    };

    bool success = model_loader.load_tensors(on_new_tensor_cb);
    LOG_INFO("collected %zu tensors for export", tensors.size());
    return success;
}

static bool load_tensor_for_export(ModelLoader& model_loader, TensorExportJob& job) {
    size_t mem_size = 1 * 1024 * 1024;
    mem_size += ggml_tensor_overhead();
    TensorStorage output_storage = job.info.storage;
    output_storage.type          = job.info.type;
    mem_size += static_cast<size_t>(output_storage.nbytes());

    ggml_context* ggml_ctx = ggml_init({mem_size, nullptr, false});
    if (ggml_ctx == nullptr) {
        job.error = "ggml_init failed for tensor '" + job.info.storage.name + "'";
        return false;
    }

    ggml_tensor* tensor = ggml_new_tensor(ggml_ctx, job.info.type, job.info.storage.n_dims, job.info.storage.ne);
    if (tensor == nullptr) {
        ggml_free(ggml_ctx);
        job.error = "ggml_new_tensor failed for tensor '" + job.info.storage.name + "'";
        return false;
    }
    ggml_set_name(tensor, job.info.storage.name.c_str());

    const size_t tensor_nbytes = ggml_nbytes(tensor);
    if (tensor_nbytes > 0 && !model_loader.load_tensor(job.info.storage, tensor)) {
        ggml_free(ggml_ctx);
        job.error = "failed to load tensor '" + job.info.storage.name + "'";
        return false;
    }

    job.data.resize(tensor_nbytes);
    if (tensor_nbytes > 0) {
        memcpy(job.data.data(), tensor->data, tensor_nbytes);
    }
    ggml_free(ggml_ctx);
    return true;
}

template <typename WriteTensorFn>
static bool stream_tensor_data(ModelLoader& model_loader,
                               const std::vector<TensorExportInfo>& tensors,
                               WriteTensorFn write_tensor,
                               std::string* error) {
    const int n_threads = std::max(1, sd_get_num_physical_cores());
    LOG_INFO("streaming convert with %d threads", n_threads);

    int64_t start_time      = ggml_time_ms();
    uint64_t bytes_written  = 0;
    size_t tensors_written  = 0;
    const size_t max_batch_bytes = 1024ull * 1024ull * 1024ull;

    for (size_t batch_start = 0; batch_start < tensors.size();) {
        size_t batch_size  = 0;
        size_t batch_bytes = 0;
        while (batch_start + batch_size < tensors.size() &&
               batch_size < static_cast<size_t>(n_threads)) {
            size_t tensor_bytes = export_tensor_nbytes(tensors[batch_start + batch_size]);
            if (batch_size > 0 && batch_bytes + tensor_bytes > max_batch_bytes) {
                break;
            }
            batch_bytes += tensor_bytes;
            batch_size++;
        }

        std::vector<TensorExportJob> jobs(batch_size);
        std::vector<std::thread> workers;
        workers.reserve(batch_size);

        for (size_t i = 0; i < batch_size; i++) {
            jobs[i].info = tensors[batch_start + i];
            workers.emplace_back([&model_loader, &jobs, i]() {
                try {
                    jobs[i].success = load_tensor_for_export(model_loader, jobs[i]);
                } catch (const std::exception& e) {
                    jobs[i].error   = e.what();
                    jobs[i].success = false;
                }
            });
        }
        for (auto& worker : workers) {
            worker.join();
        }

        for (size_t i = 0; i < batch_size; i++) {
            if (!jobs[i].success) {
                if (error != nullptr) {
                    *error = jobs[i].error.empty() ? "streaming conversion failed" : jobs[i].error;
                }
                return false;
            }
            if (!write_tensor(batch_start + i, jobs[i].data, error)) {
                return false;
            }
            bytes_written += jobs[i].data.size();
            tensors_written++;
            float elapsed_seconds = (ggml_time_ms() - start_time) / 1000.0f;
            pretty_bytes_progress(static_cast<int>(tensors_written),
                                  static_cast<int>(tensors.size()),
                                  bytes_written,
                                  elapsed_seconds);
        }
        batch_start += batch_size;
    }
    printf("\n");
    LOG_INFO("streaming conversion completed, taking %.2fs", (ggml_time_ms() - start_time) / 1000.f);
    return true;
}

static bool write_gguf_file_streaming(ModelLoader& model_loader,
                                      const char* output_path,
                                      const std::vector<TensorExportInfo>& tensors,
                                      std::string* error) {
    size_t meta_mem = 1 * 1024 * 1024 + tensors.size() * ggml_tensor_overhead();
    ggml_context* meta_ctx = ggml_init({meta_mem, nullptr, true});
    if (meta_ctx == nullptr) {
        if (error != nullptr) {
            *error = "ggml_init failed for GGUF metadata";
        }
        return false;
    }

    gguf_context* gguf_ctx = gguf_init_empty();
    if (gguf_ctx == nullptr) {
        ggml_free(meta_ctx);
        if (error != nullptr) {
            *error = "gguf_init_empty failed";
        }
        return false;
    }

    for (const TensorExportInfo& info : tensors) {
        ggml_tensor* tensor = ggml_new_tensor(meta_ctx, info.type, info.storage.n_dims, info.storage.ne);
        if (tensor == nullptr) {
            gguf_free(gguf_ctx);
            ggml_free(meta_ctx);
            if (error != nullptr) {
                *error = "ggml_new_tensor failed for tensor '" + info.storage.name + "'";
            }
            return false;
        }
        ggml_set_name(tensor, info.storage.name.c_str());
        gguf_add_tensor(gguf_ctx, tensor);
    }

    LOG_INFO("trying to save tensors to %s", output_path);
    FILE* file = fopen(output_path, "wb");
    if (file == nullptr) {
        gguf_free(gguf_ctx);
        ggml_free(meta_ctx);
        if (error != nullptr) {
            *error = "failed to open output file '" + std::string(output_path) + "'";
        }
        return false;
    }
    if (!gguf_write_to_file_ptr(gguf_ctx, file, true)) {
        fclose(file);
        gguf_free(gguf_ctx);
        ggml_free(meta_ctx);
        if (error != nullptr) {
            *error = "failed to write GGUF metadata to '" + std::string(output_path) + "'";
        }
        return false;
    }

    const size_t data_start = gguf_get_meta_size(gguf_ctx);
    auto write_tensor       = [&](size_t tensor_index, const std::vector<uint8_t>& data, std::string* write_error) -> bool {
        const TensorExportInfo& info = tensors[tensor_index];
        const size_t offset          = data_start + gguf_get_tensor_offset(gguf_ctx, static_cast<int64_t>(tensor_index));
        if (sd_file_seek(file, static_cast<int64_t>(offset), SEEK_SET) != 0) {
            if (write_error != nullptr) {
                *write_error = "failed to seek output for tensor '" + info.storage.name + "'";
            }
            return false;
        }
        if (!data.empty() && fwrite(data.data(), 1, data.size(), file) != data.size()) {
            if (write_error != nullptr) {
                *write_error = "failed to write tensor '" + info.storage.name + "'";
            }
            return false;
        }
        return true;
    };

    bool success = stream_tensor_data(model_loader, tensors, write_tensor, error);
    fclose(file);
    gguf_free(gguf_ctx);
    ggml_free(meta_ctx);
    return success;
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

static bool write_safetensors_file_streaming(ModelLoader& model_loader,
                                             const char* output_path,
                                             const std::vector<TensorExportInfo>& tensors,
                                             std::string* error) {
    nlohmann::ordered_json header = nlohmann::ordered_json::object();

    uint64_t data_offset = 0;
    for (const TensorExportInfo& info : tensors) {
        std::string dtype;
        if (!ggml_type_to_safetensors_dtype(info.type, &dtype)) {
            if (error != nullptr) {
                *error = "unsupported safetensors dtype '" + std::string(ggml_type_name(info.type)) +
                         "' for tensor '" + info.storage.name + "'";
            }
            return false;
        }

        TensorStorage output_storage = info.storage;
        output_storage.type          = info.type;
        const uint64_t tensor_nbytes = static_cast<uint64_t>(output_storage.nbytes());

        nlohmann::ordered_json json_tensor_info = nlohmann::ordered_json::object();
        json_tensor_info["dtype"]               = dtype;

        nlohmann::ordered_json shape = nlohmann::ordered_json::array();
        for (int i = 0; i < info.storage.n_dims; ++i) {
            shape.push_back(info.storage.ne[info.storage.n_dims - 1 - i]);
        }
        json_tensor_info["shape"] = shape;

        nlohmann::ordered_json data_offsets = nlohmann::ordered_json::array();
        data_offsets.push_back(data_offset);
        data_offsets.push_back(data_offset + tensor_nbytes);
        json_tensor_info["data_offsets"] = data_offsets;

        header[info.storage.name] = json_tensor_info;
        data_offset += tensor_nbytes;
    }

    const std::string header_str = header.dump();
    std::ofstream file(output_path, std::ios::binary);
    if (!file.is_open()) {
        if (error != nullptr) {
            *error = "failed to open '" + std::string(output_path) + "' for writing";
        }
        return false;
    }

    LOG_INFO("trying to save tensors to %s", output_path);
    model_io::write_u64(file, header_str.size());
    file.write(header_str.data(), header_str.size());
    if (!file) {
        if (error != nullptr) {
            *error = "failed to write safetensors header to '" + std::string(output_path) + "'";
        }
        return false;
    }

    auto write_tensor = [&](size_t tensor_index, const std::vector<uint8_t>& data, std::string* write_error) -> bool {
        const TensorExportInfo& info = tensors[tensor_index];
        if (!data.empty()) {
            file.write(reinterpret_cast<const char*>(data.data()), data.size());
        }
        if (!file) {
            if (write_error != nullptr) {
                *write_error = "failed to write tensor '" + info.storage.name + "' to '" + std::string(output_path) + "'";
            }
            return false;
        }
        return true;
    };

    return stream_tensor_data(model_loader, tensors, write_tensor, error);
}

bool convert(const char* input_path,
             const char* vae_path,
             const char* output_path,
             sd_type_t output_type,
             const char* tensor_type_rules,
             bool convert_name) {
    ModelLoader model_loader;

    if (!model_loader.init_from_file(input_path)) {
        LOG_ERROR("init model loader from file failed: '%s'", input_path);
        return false;
    }

    if (vae_path != nullptr && strlen(vae_path) > 0) {
        if (!model_loader.init_from_file(vae_path, "vae.")) {
            LOG_ERROR("init model loader from file failed: '%s'", vae_path);
            return false;
        }
    }
    if (convert_name) {
        model_loader.convert_tensors_name();
    }

    ggml_type type             = (ggml_type)output_type;
    bool output_is_safetensors = ends_with(output_path, ".safetensors");
    TensorTypeRules type_rules = parse_tensor_type_rules(tensor_type_rules);

    std::vector<TensorExportInfo> tensors;
    bool success = collect_tensors_for_export(model_loader, type, type_rules, tensors);
    std::string error;
    if (success) {
        if (output_is_safetensors) {
            success = write_safetensors_file_streaming(model_loader, output_path, tensors, &error);
        } else {
            success = write_gguf_file_streaming(model_loader, output_path, tensors, &error);
        }
    }

    if (!success && !error.empty()) {
        LOG_ERROR("%s", error.c_str());
    }

    return success;
}
