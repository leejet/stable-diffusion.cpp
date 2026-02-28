#ifndef __TENSOR_REGISTRY_HPP__
#define __TENSOR_REGISTRY_HPP__

#include <algorithm>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include "util.h"

/**
 * TensorRegistry - Tracks individual tensor locations for granular offloading
 *
 * This component enables layer-by-layer GPU memory management by:
 * 1. Mapping tensor names to their GPU/CPU locations
 * 2. Grouping tensors by layer for batch operations
 * 3. Tracking memory usage per layer
 * 4. Supporting efficient tensor movement between backends
 */

namespace LayerStreaming {

// Information about a single tensor's location and metadata
struct TensorInfo {
    ggml_tensor* gpu_tensor = nullptr;   // Tensor in GPU memory (or nullptr if on CPU)
    ggml_tensor* cpu_tensor = nullptr;   // Tensor in CPU memory (always present as source)
    size_t size_bytes       = 0;         // Size in bytes (cached for performance)
    bool on_gpu             = false;     // Current location
    int layer_index         = -1;        // Which layer this belongs to (-1 = shared/global)
    std::string layer_name;              // Full layer name (e.g., "double_blocks.5")
    uint64_t last_access    = 0;         // For LRU eviction tracking
};

// Information about a layer (group of tensors)
struct LayerInfo {
    std::string name;                           // Layer name (e.g., "double_blocks.5")
    int index                       = -1;       // Layer index for ordering
    std::vector<std::string> tensor_names;      // Tensor names belonging to this layer
    size_t total_size_bytes         = 0;        // Total size of all tensors in this layer
    bool on_gpu                     = false;    // Whether all tensors are on GPU
    ggml_backend_buffer_t gpu_buffer = nullptr; // GPU buffer for this layer's tensors
};

/**
 * TensorRegistry tracks tensor locations and supports layer-wise operations
 */
class TensorRegistry {
public:
    TensorRegistry(ggml_backend_t gpu_backend, ggml_backend_t cpu_backend)
        : gpu_backend_(gpu_backend), cpu_backend_(cpu_backend) {}

    ~TensorRegistry() {
        clear();
    }

    /**
     * Register a tensor with the registry
     * @param name Fully qualified tensor name (e.g., "model.double_blocks.5.img_attn.qkv.weight")
     * @param cpu_tensor The tensor in CPU memory
     * @param layer_name The layer this tensor belongs to (e.g., "double_blocks.5")
     * @param layer_index The numeric index of the layer
     */
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

        // Update layer info
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

    /**
     * Register all tensors from a GGML context, auto-detecting layer names from tensor names
     * @param ctx The GGML context containing tensors
     * @param prefix Prefix to strip from tensor names for layer detection
     * @param layer_pattern_fn Function to extract layer name and index from tensor name
     */
    void register_from_context(ggml_context* ctx,
                               const std::string& prefix,
                               std::function<std::pair<std::string, int>(const std::string&)> layer_pattern_fn) {
        for (ggml_tensor* t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            std::string name = ggml_get_name(t);
            auto [layer_name, layer_index] = layer_pattern_fn(name);
            register_tensor(name, t, layer_name, layer_index);
        }
    }

    /**
     * Move a specific layer's tensors to GPU
     * @param layer_name The layer to move
     * @return true if successful
     */
    bool move_layer_to_gpu(const std::string& layer_name) {
        auto it = layers_.find(layer_name);
        if (it == layers_.end()) {
            LOG_ERROR("TensorRegistry: layer '%s' not found", layer_name.c_str());
            return false;
        }

        LayerInfo& layer = it->second;
        if (layer.on_gpu) {
            return true;  // Already on GPU
        }

        int64_t t0 = ggml_time_ms();

        // Create a temporary context for GPU tensor allocation
        size_t ctx_size = layer.tensor_names.size() * ggml_tensor_overhead() + 1024;
        struct ggml_init_params ctx_params = {
            ctx_size,
            nullptr,
            true  // no_alloc
        };
        ggml_context* temp_ctx = ggml_init(ctx_params);
        if (temp_ctx == nullptr) {
            LOG_ERROR("TensorRegistry: failed to create temp context for layer '%s'", layer_name.c_str());
            return false;
        }

        // Create GPU tensor copies
        std::vector<std::pair<ggml_tensor*, ggml_tensor*>> copy_pairs;
        for (const auto& tensor_name : layer.tensor_names) {
            TensorInfo& info = tensors_[tensor_name];
            if (info.on_gpu) {
                continue;  // Already on GPU
            }

            ggml_tensor* gpu_tensor = ggml_dup_tensor(temp_ctx, info.cpu_tensor);
            ggml_set_name(gpu_tensor, tensor_name.c_str());
            copy_pairs.push_back({info.cpu_tensor, gpu_tensor});
        }

        if (copy_pairs.empty()) {
            ggml_free(temp_ctx);
            layer.on_gpu = true;
            return true;
        }

        // Allocate GPU buffer for these tensors
        layer.gpu_buffer = ggml_backend_alloc_ctx_tensors(temp_ctx, gpu_backend_);
        if (layer.gpu_buffer == nullptr) {
            LOG_ERROR("TensorRegistry: failed to allocate GPU buffer for layer '%s'", layer_name.c_str());
            ggml_free(temp_ctx);
            return false;
        }

        // Copy data from CPU to GPU
        for (auto& [cpu_t, gpu_t] : copy_pairs) {
            ggml_backend_tensor_copy(cpu_t, gpu_t);
        }
        ggml_backend_synchronize(gpu_backend_);

        // Update tensor info and swap buffer pointers
        for (auto& [cpu_t, gpu_t] : copy_pairs) {
            std::string name = ggml_get_name(cpu_t);
            TensorInfo& info = tensors_[name];
            info.gpu_tensor  = gpu_t;
            info.on_gpu      = true;
            info.last_access = access_counter_++;

            // Swap the buffer pointers so the original tensor now points to GPU memory
            std::swap(cpu_t->buffer, gpu_t->buffer);
            std::swap(cpu_t->data, gpu_t->data);
            std::swap(cpu_t->extra, gpu_t->extra);
        }

        layer.on_gpu = true;
        current_gpu_usage_ += layer.total_size_bytes;

        // Store the temp context for later cleanup
        layer_contexts_[layer_name] = temp_ctx;

        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("TensorRegistry: moved layer '%s' to GPU (%.2f MB) in %.2fs",
                  layer_name.c_str(),
                  layer.total_size_bytes / (1024.0 * 1024.0),
                  (t1 - t0) / 1000.0);

        return true;
    }

    /**
     * Move a specific layer's tensors to CPU (offload from GPU)
     * @param layer_name The layer to move
     */
    void move_layer_to_cpu(const std::string& layer_name) {
        auto it = layers_.find(layer_name);
        if (it == layers_.end()) {
            return;
        }

        LayerInfo& layer = it->second;
        if (!layer.on_gpu) {
            return;  // Already on CPU
        }

        int64_t t0 = ggml_time_ms();

        // Restore original CPU buffer pointers
        for (const auto& tensor_name : layer.tensor_names) {
            TensorInfo& info = tensors_[tensor_name];
            if (!info.on_gpu || info.gpu_tensor == nullptr) {
                continue;
            }

            // Swap back to CPU buffer
            std::swap(info.cpu_tensor->buffer, info.gpu_tensor->buffer);
            std::swap(info.cpu_tensor->data, info.gpu_tensor->data);
            std::swap(info.cpu_tensor->extra, info.gpu_tensor->extra);

            info.gpu_tensor = nullptr;
            info.on_gpu     = false;
        }

        // Free GPU buffer
        if (layer.gpu_buffer != nullptr) {
            ggml_backend_buffer_free(layer.gpu_buffer);
            layer.gpu_buffer = nullptr;
        }

        // Free temp context
        auto ctx_it = layer_contexts_.find(layer_name);
        if (ctx_it != layer_contexts_.end()) {
            ggml_free(ctx_it->second);
            layer_contexts_.erase(ctx_it);
        }

        current_gpu_usage_ -= layer.total_size_bytes;
        layer.on_gpu = false;

        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("TensorRegistry: moved layer '%s' to CPU (%.2f MB) in %.2fs",
                  layer_name.c_str(),
                  layer.total_size_bytes / (1024.0 * 1024.0),
                  (t1 - t0) / 1000.0);
    }

    /**
     * Check if a layer is currently on GPU
     */
    bool is_layer_on_gpu(const std::string& layer_name) const {
        auto it = layers_.find(layer_name);
        if (it == layers_.end()) {
            return false;
        }
        return it->second.on_gpu;
    }

    /**
     * Get the size of a layer in bytes
     */
    size_t get_layer_size(const std::string& layer_name) const {
        auto it = layers_.find(layer_name);
        if (it == layers_.end()) {
            return 0;
        }
        return it->second.total_size_bytes;
    }

    /**
     * Get current GPU memory usage by tracked tensors
     */
    size_t get_gpu_usage() const {
        return current_gpu_usage_;
    }

    /**
     * Get list of all layer names in order
     */
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

    /**
     * Get list of layers currently on GPU (for eviction decisions)
     */
    std::vector<std::string> get_layers_on_gpu() const {
        std::vector<std::string> result;
        for (const auto& [name, info] : layers_) {
            if (info.on_gpu) {
                result.push_back(name);
            }
        }
        return result;
    }

    /**
     * Get total number of layers
     */
    size_t get_layer_count() const {
        return layers_.size();
    }

    /**
     * Clear all registrations and free GPU resources
     */
    void clear() {
        // Move all layers to CPU first
        for (auto& [name, layer] : layers_) {
            if (layer.on_gpu) {
                move_layer_to_cpu(name);
            }
        }

        // Free any remaining contexts
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

    size_t current_gpu_usage_ = 0;
    uint64_t access_counter_  = 0;
};

/**
 * Helper function to extract Flux layer information from tensor name
 * Returns (layer_name, layer_index) or ("_global", -1) for non-layer tensors
 */
inline std::pair<std::string, int> flux_layer_pattern(const std::string& tensor_name) {
    // Look for double_blocks.N or single_blocks.N pattern
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
        // Offset single_blocks to come after double_blocks (19 double blocks)
        return {"single_blocks." + num_str, 19 + block_idx};
    }

    // Non-layer tensor (global, like img_in, txt_in, final_layer)
    return {"_global", -1};
}

/**
 * Helper function to extract UNet layer information from tensor name
 * Returns (layer_name, layer_index) or ("_global", -1) for non-layer tensors
 */
inline std::pair<std::string, int> unet_layer_pattern(const std::string& tensor_name) {
    // Look for input_blocks.N, middle_block, output_blocks.N patterns
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
        return {"middle_block", 100};  // Use high index to come after input_blocks
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
        return {"output_blocks." + num_str, 200 + block_idx};  // After middle_block
    }

    // Non-layer tensor (global)
    return {"_global", -1};
}

}  // namespace LayerStreaming

#endif  // __TENSOR_REGISTRY_HPP__
