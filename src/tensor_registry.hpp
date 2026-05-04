#ifndef __TENSOR_REGISTRY_HPP__
#define __TENSOR_REGISTRY_HPP__

#include <algorithm>
#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include "util.h"

namespace LayerStreaming {

struct TensorInfo {
    ggml_tensor* gpu_tensor = nullptr;
    ggml_tensor* cpu_tensor = nullptr;
    size_t size_bytes       = 0;
    bool on_gpu             = false;
    int layer_index         = -1;
    std::string layer_name;
    uint64_t last_access    = 0;
};

struct LayerInfo {
    std::string name;
    int index                       = -1;
    std::vector<std::string> tensor_names;
    size_t total_size_bytes         = 0;
    bool on_gpu                     = false;
    ggml_backend_buffer_t gpu_buffer = nullptr;
};

// Tracks in-flight async transfers
struct AsyncLoadState {
    struct CopyInfo {
        std::string name;
        ggml_tensor* cpu_tensor;
        ggml_tensor* gpu_tensor;
    };

    ggml_context* temp_ctx = nullptr;
    ggml_backend_buffer_t gpu_buffer = nullptr;
    std::vector<CopyInfo> copy_list;
    int64_t start_time = 0;
};

class TensorRegistry {
public:
    TensorRegistry(ggml_backend_t gpu_backend, ggml_backend_t cpu_backend)
        : gpu_backend_(gpu_backend), cpu_backend_(cpu_backend) {}

    ~TensorRegistry() {
        clear();
    }

    void register_tensor(const std::string& name,
                         ggml_tensor* cpu_tensor,
                         const std::string& layer_name,
                         int layer_index) {
        TensorInfo info;
        info.cpu_tensor  = cpu_tensor;
        info.gpu_tensor  = nullptr;
        info.size_bytes  = ggml_nbytes(cpu_tensor);
        info.on_gpu      = false;
        info.layer_index = layer_index;
        info.layer_name  = layer_name;
        info.last_access = 0;

        tensors_[name] = info;

        if (layers_.find(layer_name) == layers_.end()) {
            LayerInfo layer_info;
            layer_info.name            = layer_name;
            layer_info.index           = layer_index;
            layer_info.total_size_bytes = 0;
            layer_info.on_gpu          = false;
            layer_info.gpu_buffer      = nullptr;
            layers_[layer_name]        = layer_info;
        }
        layers_[layer_name].tensor_names.push_back(name);
        layers_[layer_name].total_size_bytes += info.size_bytes;
    }

    // Only works if tensor names are set with ggml_set_name()
    void register_from_context(ggml_context* ctx,
                               const std::string& prefix,
                               std::function<std::pair<std::string, int>(const std::string&)> layer_pattern_fn) {
        for (ggml_tensor* t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            std::string name = ggml_get_name(t);
            auto [layer_name, layer_index] = layer_pattern_fn(name);
            register_tensor(name, t, layer_name, layer_index);
        }
    }

    // Preferred method: tensor names are properly preserved in the map keys
    void register_from_map(const std::map<std::string, ggml_tensor*>& tensors,
                           std::function<std::pair<std::string, int>(const std::string&)> layer_pattern_fn) {
        for (const auto& [name, tensor] : tensors) {
            auto [layer_name, layer_index] = layer_pattern_fn(name);
            register_tensor(name, tensor, layer_name, layer_index);
        }
    }

    bool move_layer_to_gpu(const std::string& layer_name) {
        auto it = layers_.find(layer_name);
        if (it == layers_.end()) {
            LOG_ERROR("layer '%s' not found", layer_name.c_str());
            return false;
        }

        LayerInfo& layer = it->second;
        if (layer.on_gpu) {
            return true;
        }

        int64_t t0 = ggml_time_ms();

        size_t ctx_size = layer.tensor_names.size() * ggml_tensor_overhead() + 1024;
        struct ggml_init_params ctx_params = {
            ctx_size,
            nullptr,
            true,
        };
        ggml_context* temp_ctx = ggml_init(ctx_params);
        if (temp_ctx == nullptr) {
            LOG_ERROR("failed to create temp context for layer '%s'", layer_name.c_str());
            return false;
        }

        // Can't rely on ggml_get_name() because GGMLBlock doesn't call ggml_set_name()
        struct CopyInfo {
            std::string name;
            ggml_tensor* cpu_tensor;
            ggml_tensor* gpu_tensor;
        };
        std::vector<CopyInfo> copy_list;

        for (const auto& tensor_name : layer.tensor_names) {
            TensorInfo& info = tensors_[tensor_name];
            if (info.on_gpu) {
                continue;
            }

            ggml_tensor* gpu_tensor = ggml_dup_tensor(temp_ctx, info.cpu_tensor);
            ggml_set_name(gpu_tensor, tensor_name.c_str());
            copy_list.push_back({tensor_name, info.cpu_tensor, gpu_tensor});
        }

        if (copy_list.empty()) {
            ggml_free(temp_ctx);
            layer.on_gpu = true;
            return true;
        }

        layer.gpu_buffer = ggml_backend_alloc_ctx_tensors(temp_ctx, gpu_backend_);
        if (layer.gpu_buffer == nullptr) {
            LOG_ERROR("failed to allocate GPU buffer for layer '%s'", layer_name.c_str());
            ggml_free(temp_ctx);
            return false;
        }

        for (auto& item : copy_list) {
            ggml_backend_tensor_copy(item.cpu_tensor, item.gpu_tensor);
        }
        ggml_backend_synchronize(gpu_backend_);

        for (auto& item : copy_list) {
            TensorInfo& info = tensors_[item.name];
            info.gpu_tensor  = item.gpu_tensor;
            info.on_gpu      = true;
            info.last_access = access_counter_++;

            // Swap pointers so the original tensor now points to GPU memory
            std::swap(item.cpu_tensor->buffer, item.gpu_tensor->buffer);
            std::swap(item.cpu_tensor->data, item.gpu_tensor->data);
            std::swap(item.cpu_tensor->extra, item.gpu_tensor->extra);
        }

        layer.on_gpu = true;
        current_gpu_usage_ += layer.total_size_bytes;
        layer_contexts_[layer_name] = temp_ctx;

        return true;
    }

    void move_layer_to_cpu(const std::string& layer_name) {
        auto it = layers_.find(layer_name);
        if (it == layers_.end()) {
            return;
        }

        LayerInfo& layer = it->second;
        if (!layer.on_gpu) {
            return;
        }

        for (const auto& tensor_name : layer.tensor_names) {
            TensorInfo& info = tensors_[tensor_name];
            if (!info.on_gpu || info.gpu_tensor == nullptr) {
                continue;
            }

            std::swap(info.cpu_tensor->buffer, info.gpu_tensor->buffer);
            std::swap(info.cpu_tensor->data, info.gpu_tensor->data);
            std::swap(info.cpu_tensor->extra, info.gpu_tensor->extra);

            info.gpu_tensor = nullptr;
            info.on_gpu     = false;
        }

        if (layer.gpu_buffer != nullptr) {
            ggml_backend_buffer_free(layer.gpu_buffer);
            layer.gpu_buffer = nullptr;
        }

        auto ctx_it = layer_contexts_.find(layer_name);
        if (ctx_it != layer_contexts_.end()) {
            ggml_free(ctx_it->second);
            layer_contexts_.erase(ctx_it);
        }

        current_gpu_usage_ -= layer.total_size_bytes;
        layer.on_gpu = false;
    }

    bool is_layer_on_gpu(const std::string& layer_name) const {
        auto it = layers_.find(layer_name);
        if (it == layers_.end()) {
            return false;
        }
        return it->second.on_gpu;
    }

    size_t get_layer_size(const std::string& layer_name) const {
        auto it = layers_.find(layer_name);
        if (it == layers_.end()) {
            return 0;
        }
        return it->second.total_size_bytes;
    }

    size_t get_gpu_usage() const {
        return current_gpu_usage_;
    }

    std::vector<std::string> get_layer_names_sorted() const {
        std::vector<std::pair<int, std::string>> indexed_layers;
        for (const auto& [name, info] : layers_) {
            indexed_layers.push_back({info.index, name});
        }
        std::sort(indexed_layers.begin(), indexed_layers.end());

        std::vector<std::string> result;
        for (const auto& [idx, name] : indexed_layers) {
            result.push_back(name);
        }
        return result;
    }

    std::vector<std::string> get_layers_on_gpu() const {
        std::vector<std::string> result;
        for (const auto& [name, info] : layers_) {
            if (info.on_gpu) {
                result.push_back(name);
            }
        }
        return result;
    }

    size_t get_layer_count() const {
        return layers_.size();
    }

    // Initiates transfer without waiting; call complete_async_layer_load() to finalize
    bool start_async_layer_load(const std::string& layer_name,
                                ggml_backend_t gpu_backend,
                                ggml_backend_t cpu_backend) {
        auto it = layers_.find(layer_name);
        if (it == layers_.end()) {
            LOG_ERROR("layer '%s' not found for async load", layer_name.c_str());
            return false;
        }

        LayerInfo& layer = it->second;
        if (layer.on_gpu) {
            return true;
        }

        if (async_loading_layers_.find(layer_name) != async_loading_layers_.end()) {
            return true;
        }

        int64_t t0 = ggml_time_ms();

        size_t ctx_size = layer.tensor_names.size() * ggml_tensor_overhead() + 1024;
        struct ggml_init_params ctx_params = {
            ctx_size,
            nullptr,
            true,
        };
        ggml_context* temp_ctx = ggml_init(ctx_params);
        if (temp_ctx == nullptr) {
            LOG_ERROR("failed to create temp context for async load of layer '%s'", layer_name.c_str());
            return false;
        }

        std::vector<AsyncLoadState::CopyInfo> copy_list;

        for (const auto& tensor_name : layer.tensor_names) {
            TensorInfo& info = tensors_[tensor_name];
            if (info.on_gpu) {
                continue;
            }

            ggml_tensor* gpu_tensor = ggml_dup_tensor(temp_ctx, info.cpu_tensor);
            ggml_set_name(gpu_tensor, tensor_name.c_str());
            copy_list.push_back({tensor_name, info.cpu_tensor, gpu_tensor});
        }

        if (copy_list.empty()) {
            ggml_free(temp_ctx);
            layer.on_gpu = true;
            return true;
        }

        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(temp_ctx, gpu_backend);
        if (buffer == nullptr) {
            LOG_ERROR("failed to allocate GPU buffer for async load of layer '%s'", layer_name.c_str());
            ggml_free(temp_ctx);
            return false;
        }

        // May fall back to sync for CPU->CUDA
        for (auto& item : copy_list) {
            ggml_backend_tensor_copy_async(cpu_backend, gpu_backend, item.cpu_tensor, item.gpu_tensor);
        }

        AsyncLoadState state;
        state.temp_ctx = temp_ctx;
        state.gpu_buffer = buffer;
        state.copy_list = std::move(copy_list);
        state.start_time = t0;

        async_loading_layers_[layer_name] = std::move(state);

        return true;
    }

    // Waits for pending async transfers and finalizes the layer state
    bool complete_async_layer_load(const std::string& layer_name,
                                   ggml_backend_t gpu_backend) {
        auto async_it = async_loading_layers_.find(layer_name);
        if (async_it == async_loading_layers_.end()) {
            // Not in async loading - check if already on GPU
            auto layer_it = layers_.find(layer_name);
            if (layer_it != layers_.end() && layer_it->second.on_gpu) {
                return true;
            }
            return false;
        }

        AsyncLoadState& state = async_it->second;
        auto layer_it = layers_.find(layer_name);
        if (layer_it == layers_.end()) {
            ggml_backend_buffer_free(state.gpu_buffer);
            ggml_free(state.temp_ctx);
            async_loading_layers_.erase(async_it);
            return false;
        }

        LayerInfo& layer = layer_it->second;

        ggml_backend_synchronize(gpu_backend);

        for (auto& item : state.copy_list) {
            TensorInfo& info = tensors_[item.name];
            info.gpu_tensor = item.gpu_tensor;
            info.on_gpu = true;
            info.last_access = access_counter_++;

            std::swap(item.cpu_tensor->buffer, item.gpu_tensor->buffer);
            std::swap(item.cpu_tensor->data, item.gpu_tensor->data);
            std::swap(item.cpu_tensor->extra, item.gpu_tensor->extra);
        }

        layer.on_gpu = true;
        layer.gpu_buffer = state.gpu_buffer;
        current_gpu_usage_ += layer.total_size_bytes;
        layer_contexts_[layer_name] = state.temp_ctx;

        async_loading_layers_.erase(async_it);
        return true;
    }

    bool is_layer_async_loading(const std::string& layer_name) const {
        return async_loading_layers_.find(layer_name) != async_loading_layers_.end();
    }

    void clear() {
        for (auto& [name, state] : async_loading_layers_) {
            if (state.gpu_buffer) {
                ggml_backend_buffer_free(state.gpu_buffer);
            }
            if (state.temp_ctx) {
                ggml_free(state.temp_ctx);
            }
        }
        async_loading_layers_.clear();

        for (auto& [name, layer] : layers_) {
            if (layer.on_gpu) {
                move_layer_to_cpu(name);
            }
        }

        for (auto& [name, ctx] : layer_contexts_) {
            ggml_free(ctx);
        }

        tensors_.clear();
        layers_.clear();
        layer_contexts_.clear();
        current_gpu_usage_ = 0;
    }

private:
    ggml_backend_t gpu_backend_;
    ggml_backend_t cpu_backend_;

    std::unordered_map<std::string, TensorInfo> tensors_;
    std::unordered_map<std::string, LayerInfo> layers_;
    std::unordered_map<std::string, ggml_context*> layer_contexts_;
    std::unordered_map<std::string, AsyncLoadState> async_loading_layers_;

    size_t current_gpu_usage_ = 0;
    uint64_t access_counter_  = 0;
};

// Extract Flux layer info: double_blocks.N, single_blocks.N, or _global
inline std::pair<std::string, int> flux_layer_pattern(const std::string& tensor_name) {
    size_t db_pos = tensor_name.find("double_blocks.");
    if (db_pos != std::string::npos) {
        size_t num_start = db_pos + 14;  // Length of "double_blocks."
        size_t num_end = tensor_name.find('.', num_start);
        if (num_end == std::string::npos) {
            num_end = tensor_name.length();
        }
        std::string num_str = tensor_name.substr(num_start, num_end - num_start);
        int block_idx = std::stoi(num_str);
        return {"double_blocks." + num_str, block_idx};
    }

    size_t sb_pos = tensor_name.find("single_blocks.");
    if (sb_pos != std::string::npos) {
        size_t num_start = sb_pos + 14;  // Length of "single_blocks."
        size_t num_end = tensor_name.find('.', num_start);
        if (num_end == std::string::npos) {
            num_end = tensor_name.length();
        }
        std::string num_str = tensor_name.substr(num_start, num_end - num_start);
        int block_idx = std::stoi(num_str);
        // Offset past 19 double_blocks
        return {"single_blocks." + num_str, 19 + block_idx};
    }

    return {"_global", -1};
}

// Extract UNet layer info: input_blocks.N, middle_block, output_blocks.N, or _global
inline std::pair<std::string, int> unet_layer_pattern(const std::string& tensor_name) {
    size_t ib_pos = tensor_name.find("input_blocks.");
    if (ib_pos != std::string::npos) {
        size_t num_start = ib_pos + 13;  // Length of "input_blocks."
        size_t num_end = tensor_name.find('.', num_start);
        if (num_end == std::string::npos) {
            num_end = tensor_name.length();
        }
        std::string num_str = tensor_name.substr(num_start, num_end - num_start);
        int block_idx = std::stoi(num_str);
        return {"input_blocks." + num_str, block_idx};
    }

    if (tensor_name.find("middle_block") != std::string::npos) {
        return {"middle_block", 100};
    }

    size_t ob_pos = tensor_name.find("output_blocks.");
    if (ob_pos != std::string::npos) {
        size_t num_start = ob_pos + 14;  // Length of "output_blocks."
        size_t num_end = tensor_name.find('.', num_start);
        if (num_end == std::string::npos) {
            num_end = tensor_name.length();
        }
        std::string num_str = tensor_name.substr(num_start, num_end - num_start);
        int block_idx = std::stoi(num_str);
        return {"output_blocks." + num_str, 200 + block_idx};
    }

    return {"_global", -1};
}

// Extract MMDiT layer info: joint_blocks.N, or _global
inline std::pair<std::string, int> mmdit_layer_pattern(const std::string& tensor_name) {
    size_t jb_pos = tensor_name.find("joint_blocks.");
    if (jb_pos != std::string::npos) {
        size_t num_start = jb_pos + 13;  // Length of "joint_blocks."
        size_t num_end = tensor_name.find('.', num_start);
        if (num_end == std::string::npos) {
            num_end = tensor_name.length();
        }
        std::string num_str = tensor_name.substr(num_start, num_end - num_start);
        int block_idx = std::stoi(num_str);
        return {"joint_blocks." + num_str, block_idx};
    }

    return {"_global", -1};
}

// Extract WAN layer info: blocks.N, vace_blocks.N, or _global
inline std::pair<std::string, int> wan_layer_pattern(const std::string& tensor_name) {
    size_t b_pos = tensor_name.find("blocks.");
    // Exclude "vace_blocks" matches
    if (b_pos != std::string::npos && (b_pos == 0 || tensor_name[b_pos - 1] != '_')) {
        size_t num_start = b_pos + 7;  // Length of "blocks."
        size_t num_end = tensor_name.find('.', num_start);
        if (num_end == std::string::npos) {
            num_end = tensor_name.length();
        }
        std::string num_str = tensor_name.substr(num_start, num_end - num_start);
        int block_idx = std::stoi(num_str);
        return {"blocks." + num_str, block_idx};
    }

    size_t vb_pos = tensor_name.find("vace_blocks.");
    if (vb_pos != std::string::npos) {
        size_t num_start = vb_pos + 12;  // Length of "vace_blocks."
        size_t num_end = tensor_name.find('.', num_start);
        if (num_end == std::string::npos) {
            num_end = tensor_name.length();
        }
        std::string num_str = tensor_name.substr(num_start, num_end - num_start);
        int block_idx = std::stoi(num_str);
        return {"vace_blocks." + num_str, 100 + block_idx};
    }

    return {"_global", -1};
}

// Extract QwenImage layer info: transformer_blocks.N, or _global
inline std::pair<std::string, int> qwen_image_layer_pattern(const std::string& tensor_name) {
    size_t tb_pos = tensor_name.find("transformer_blocks.");
    if (tb_pos != std::string::npos) {
        size_t num_start = tb_pos + 19;  // Length of "transformer_blocks."
        size_t num_end = tensor_name.find('.', num_start);
        if (num_end == std::string::npos) {
            num_end = tensor_name.length();
        }
        std::string num_str = tensor_name.substr(num_start, num_end - num_start);
        int block_idx = std::stoi(num_str);
        return {"transformer_blocks." + num_str, block_idx};
    }

    return {"_global", -1};
}

// Extract ZImage layer info: context_refiner.N, noise_refiner.N, layers.N, or _global
inline std::pair<std::string, int> zimage_layer_pattern(const std::string& tensor_name) {
    size_t cr_pos = tensor_name.find("context_refiner.");
    if (cr_pos != std::string::npos) {
        size_t num_start = cr_pos + 16;  // Length of "context_refiner."
        size_t num_end = tensor_name.find('.', num_start);
        if (num_end == std::string::npos) {
            num_end = tensor_name.length();
        }
        std::string num_str = tensor_name.substr(num_start, num_end - num_start);
        int block_idx = std::stoi(num_str);
        return {"context_refiner." + num_str, block_idx};
    }

    size_t nr_pos = tensor_name.find("noise_refiner.");
    if (nr_pos != std::string::npos) {
        size_t num_start = nr_pos + 14;  // Length of "noise_refiner."
        size_t num_end = tensor_name.find('.', num_start);
        if (num_end == std::string::npos) {
            num_end = tensor_name.length();
        }
        std::string num_str = tensor_name.substr(num_start, num_end - num_start);
        int block_idx = std::stoi(num_str);
        return {"noise_refiner." + num_str, 10 + block_idx};
    }

    size_t l_pos = tensor_name.find("layers.");
    if (l_pos != std::string::npos) {
        size_t num_start = l_pos + 7;  // Length of "layers."
        size_t num_end = tensor_name.find('.', num_start);
        if (num_end == std::string::npos) {
            num_end = tensor_name.length();
        }
        std::string num_str = tensor_name.substr(num_start, num_end - num_start);
        int block_idx = std::stoi(num_str);
        return {"layers." + num_str, 100 + block_idx};
    }

    return {"_global", -1};
}

// Extract Anima layer info: blocks.N (from net.blocks.N), or _global
inline std::pair<std::string, int> anima_layer_pattern(const std::string& tensor_name) {
    size_t nb_pos = tensor_name.find("net.blocks.");
    if (nb_pos != std::string::npos) {
        size_t num_start = nb_pos + 11;  // Length of "net.blocks."
        size_t num_end = tensor_name.find('.', num_start);
        if (num_end == std::string::npos) {
            num_end = tensor_name.length();
        }
        std::string num_str = tensor_name.substr(num_start, num_end - num_start);
        int block_idx = std::stoi(num_str);
        return {"blocks." + num_str, block_idx};
    }

    return {"_global", -1};
}

}  // namespace LayerStreaming

#endif  // __TENSOR_REGISTRY_HPP__
