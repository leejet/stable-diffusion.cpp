#include <algorithm>
#include <cstring>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include "model.h"
#include "model_io/gguf_io.h"
#include "model_io/safetensors_io.h"
#include "util.h"

#include "ggml-cpu.h"

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

static TensorWritePlan tensor_write_plan_from_export_info(const TensorExportInfo& info) {
    TensorWritePlan plan;
    plan.name   = info.storage.name;
    plan.type   = info.type;
    plan.n_dims = info.storage.n_dims;
    for (int i = 0; i < SD_MAX_DIMS; i++) {
        plan.ne[i] = info.storage.ne[i];
    }
    return plan;
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
                                       int n_threads,
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

    bool success = model_loader.load_tensors(on_new_tensor_cb, n_threads);
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

template <typename Writer>
static bool stream_tensor_data(ModelLoader& model_loader,
                               const std::vector<TensorExportInfo>& tensors,
                               Writer& writer,
                               int n_threads,
                               std::string* error) {
    n_threads = n_threads > 0 ? n_threads : sd_get_num_physical_cores();
    n_threads = std::max(1, n_threads);
    LOG_INFO("streaming convert with %d threads", n_threads);

    int64_t start_time       = ggml_time_ms();
    uint64_t bytes_written   = 0;
    size_t tensors_written   = 0;
    size_t next_tensor_index = 0;
    bool failed              = false;
    std::string failure;

    const size_t memory_budget = 1024ull * 1024ull * 1024ull;
    size_t reserved_bytes      = 0;

    std::mutex work_mutex;
    std::mutex progress_mutex;
    std::condition_variable memory_cv;
    std::vector<std::thread> workers;
    workers.reserve(n_threads);

    auto reserve_memory = [&](size_t bytes) -> bool {
        std::unique_lock<std::mutex> lock(work_mutex);
        memory_cv.wait(lock, [&]() {
            return failed || reserved_bytes == 0 || reserved_bytes + bytes <= memory_budget;
        });
        if (failed) {
            return false;
        }
        reserved_bytes += bytes;
        return true;
    };

    auto release_memory = [&](size_t bytes) {
        {
            std::lock_guard<std::mutex> lock(work_mutex);
            reserved_bytes -= std::min(reserved_bytes, bytes);
        }
        memory_cv.notify_all();
    };

    auto fail = [&](const std::string& message) {
        {
            std::lock_guard<std::mutex> lock(work_mutex);
            if (!failed) {
                failed  = true;
                failure = message;
            }
        }
        memory_cv.notify_all();
    };

    for (int worker_index = 0; worker_index < n_threads; worker_index++) {
        workers.emplace_back([&, worker_index]() {
            while (true) {
                size_t tensor_index = 0;
                {
                    std::lock_guard<std::mutex> lock(work_mutex);
                    if (failed || next_tensor_index >= tensors.size()) {
                        return;
                    }
                    tensor_index = next_tensor_index++;
                }

                const size_t tensor_bytes = export_tensor_nbytes(tensors[tensor_index]);
                if (!reserve_memory(tensor_bytes)) {
                    return;
                }

                TensorExportJob job;
                job.info = tensors[tensor_index];
                try {
                    job.success = load_tensor_for_export(model_loader, job);
                } catch (const std::exception& e) {
                    job.error   = e.what();
                    job.success = false;
                }

                if (!job.success) {
                    release_memory(tensor_bytes);
                    fail(job.error.empty() ? "streaming conversion failed" : job.error);
                    return;
                }

                std::string write_error;
                if (!writer.write_tensor(tensor_index,
                                         job.data.empty() ? nullptr : job.data.data(),
                                         job.data.size(),
                                         worker_index,
                                         &write_error)) {
                    release_memory(tensor_bytes);
                    fail(write_error.empty() ? "streaming conversion write failed" : write_error);
                    return;
                }

                {
                    std::lock_guard<std::mutex> lock(progress_mutex);
                    bytes_written += job.data.size();
                    tensors_written++;
                    float elapsed_seconds = (ggml_time_ms() - start_time) / 1000.0f;
                    pretty_bytes_progress(static_cast<int>(tensors_written),
                                          static_cast<int>(tensors.size()),
                                          bytes_written,
                                          elapsed_seconds);
                }
                release_memory(tensor_bytes);
            }
        });
    }

    for (auto& worker : workers) {
        worker.join();
    }
    printf("\n");
    if (failed) {
        if (error != nullptr) {
            *error = failure;
        }
        return false;
    }
    LOG_INFO("streaming conversion completed, taking %.2fs", (ggml_time_ms() - start_time) / 1000.f);
    return true;
}

static bool write_gguf_file_streaming(ModelLoader& model_loader,
                                      const char* output_path,
                                      const std::vector<TensorExportInfo>& tensors,
                                      int n_threads,
                                      std::string* error) {
    std::vector<TensorWritePlan> plans;
    plans.reserve(tensors.size());
    for (const TensorExportInfo& info : tensors) {
        plans.push_back(tensor_write_plan_from_export_info(info));
    }
    GGUFStreamingWriter writer;
    if (!writer.open(output_path, plans, n_threads, error)) {
        return false;
    }
    return stream_tensor_data(model_loader, tensors, writer, n_threads, error);
}

static bool write_safetensors_file_streaming(ModelLoader& model_loader,
                                             const char* output_path,
                                             const std::vector<TensorExportInfo>& tensors,
                                             int n_threads,
                                             std::string* error) {
    std::vector<TensorWritePlan> plans;
    plans.reserve(tensors.size());
    for (const TensorExportInfo& info : tensors) {
        plans.push_back(tensor_write_plan_from_export_info(info));
    }
    SafetensorsStreamingWriter writer;
    if (!writer.open(output_path, plans, n_threads, error)) {
        return false;
    }
    return stream_tensor_data(model_loader, tensors, writer, n_threads, error);
}

bool convert_with_threads(const char* input_path,
                          const char* vae_path,
                          const char* output_path,
                          sd_type_t output_type,
                          const char* tensor_type_rules,
                          bool convert_name,
                          int n_threads) {
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
    bool success = collect_tensors_for_export(model_loader, type, type_rules, n_threads, tensors);
    std::string error;
    if (success) {
        if (output_is_safetensors) {
            success = write_safetensors_file_streaming(model_loader, output_path, tensors, n_threads, &error);
        } else {
            success = write_gguf_file_streaming(model_loader, output_path, tensors, n_threads, &error);
        }
    }

    if (!success && !error.empty()) {
        LOG_ERROR("%s", error.c_str());
    }

    return success;
}

bool convert(const char* input_path,
             const char* vae_path,
             const char* output_path,
             sd_type_t output_type,
             const char* tensor_type_rules,
             bool convert_name) {
    return convert_with_threads(input_path,
                                vae_path,
                                output_path,
                                output_type,
                                tensor_type_rules,
                                convert_name,
                                0);
}
