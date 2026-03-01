#ifndef __LAYER_STREAMING_HPP__
#define __LAYER_STREAMING_HPP__

#include <algorithm>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include "memory_budget.hpp"
#include "tensor_registry.hpp"
#include "util.h"

/**
 * LayerExecutionEngine - Orchestrates layer-by-layer model execution
 *
 * This component enables executing models one layer at a time, managing:
 * 1. Per-layer graph building and execution
 * 2. Intermediate tensor storage between layers
 * 3. Async prefetching of upcoming layer weights
 * 4. Automatic offloading of completed layers
 */

namespace LayerStreaming {

// Forward declaration
class LayerExecutionEngine;

/**
 * Represents a single layer that can be executed independently
 */
struct LayerSubgraph {
    std::string name;                    // Layer name (e.g., "double_blocks.5")
    int index;                           // Execution order index
    size_t estimated_compute_size = 0;   // Estimated compute buffer size

    // Function to build and execute this layer's subgraph
    // Takes input tensors and returns output tensors
    using ExecuteFn = std::function<std::vector<ggml_tensor*>(
        ggml_context* ctx,
        ggml_backend_t backend,
        const std::vector<ggml_tensor*>& inputs)>;

    ExecuteFn execute_fn;
};

/**
 * Configuration for layer streaming
 */
struct StreamingConfig {
    bool enabled              = false;   // Whether streaming is enabled
    int prefetch_layers       = 1;       // How many layers ahead to prefetch
    int keep_layers_behind    = 0;       // How many layers to keep after execution (for skip connections)
    size_t min_free_vram      = 512 * 1024 * 1024;  // Minimum VRAM to keep free (512 MB)
    bool async_prefetch       = true;    // Use async memory transfers when available
    bool log_operations       = true;    // Log streaming operations
};

/**
 * Manages intermediate tensors between layer executions
 */
class IntermediateTensorManager {
public:
    IntermediateTensorManager(ggml_backend_t gpu_backend)
        : gpu_backend_(gpu_backend) {}

    ~IntermediateTensorManager() {
        clear();
    }

    /**
     * Store an intermediate tensor (copies data to managed buffer)
     * @param name Identifier for this tensor
     * @param tensor The tensor to store
     * @return Pointer to the stored tensor (valid until clear() or overwrite)
     */
    ggml_tensor* store(const std::string& name, ggml_tensor* tensor) {
        // Create context for this tensor if needed
        if (contexts_.find(name) != contexts_.end()) {
            // Reuse existing - free old buffer first
            if (buffers_.find(name) != buffers_.end()) {
                ggml_backend_buffer_free(buffers_[name]);
            }
            ggml_free(contexts_[name]);
        }

        size_t ctx_size = ggml_tensor_overhead() + 1024;
        struct ggml_init_params params = {
            ctx_size,
            nullptr,
            true  // no_alloc
        };
        ggml_context* ctx = ggml_init(params);
        if (ctx == nullptr) {
            LOG_ERROR("IntermediateTensorManager: failed to create context for '%s'", name.c_str());
            return nullptr;
        }

        // Create tensor copy
        ggml_tensor* stored = ggml_dup_tensor(ctx, tensor);
        ggml_set_name(stored, name.c_str());

        // Allocate buffer and copy data
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, gpu_backend_);
        if (buffer == nullptr) {
            LOG_ERROR("IntermediateTensorManager: failed to allocate buffer for '%s'", name.c_str());
            ggml_free(ctx);
            return nullptr;
        }

        ggml_backend_tensor_copy(tensor, stored);
        ggml_backend_synchronize(gpu_backend_);

        contexts_[name] = ctx;
        buffers_[name]  = buffer;
        tensors_[name]  = stored;

        return stored;
    }

    /**
     * Retrieve a stored tensor
     */
    ggml_tensor* get(const std::string& name) {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) {
            return nullptr;
        }
        return it->second;
    }

    /**
     * Check if a tensor is stored
     */
    bool has(const std::string& name) const {
        return tensors_.find(name) != tensors_.end();
    }

    /**
     * Remove a specific tensor
     */
    void remove(const std::string& name) {
        auto buf_it = buffers_.find(name);
        if (buf_it != buffers_.end()) {
            ggml_backend_buffer_free(buf_it->second);
            buffers_.erase(buf_it);
        }

        auto ctx_it = contexts_.find(name);
        if (ctx_it != contexts_.end()) {
            ggml_free(ctx_it->second);
            contexts_.erase(ctx_it);
        }

        tensors_.erase(name);
    }

    /**
     * Clear all stored tensors
     */
    void clear() {
        for (auto& [name, buffer] : buffers_) {
            ggml_backend_buffer_free(buffer);
        }
        for (auto& [name, ctx] : contexts_) {
            ggml_free(ctx);
        }
        tensors_.clear();
        buffers_.clear();
        contexts_.clear();
    }

    /**
     * Get total memory used by stored tensors
     */
    size_t get_memory_usage() const {
        size_t total = 0;
        for (const auto& [name, buffer] : buffers_) {
            total += ggml_backend_buffer_get_size(buffer);
        }
        return total;
    }

private:
    ggml_backend_t gpu_backend_;
    std::unordered_map<std::string, ggml_context*> contexts_;
    std::unordered_map<std::string, ggml_backend_buffer_t> buffers_;
    std::unordered_map<std::string, ggml_tensor*> tensors_;
};

/**
 * LayerExecutionEngine - Main orchestrator for layer streaming
 */
class LayerExecutionEngine {
public:
    LayerExecutionEngine(ggml_backend_t gpu_backend,
                         ggml_backend_t cpu_backend)
        : gpu_backend_(gpu_backend),
          cpu_backend_(cpu_backend),
          registry_(gpu_backend, cpu_backend),
          budget_(registry_, gpu_backend),
          intermediates_(gpu_backend) {}

    /**
     * Set streaming configuration
     */
    void set_config(const StreamingConfig& config) {
        config_ = config;
    }

    /**
     * Get current configuration
     */
    const StreamingConfig& get_config() const {
        return config_;
    }

    /**
     * Get the tensor registry for registration
     */
    TensorRegistry& get_registry() {
        return registry_;
    }

    /**
     * Get the memory budget manager
     */
    MemoryBudgetManager& get_budget() {
        return budget_;
    }

    /**
     * Register layers from a model's parameter context
     * @param params_ctx The GGML context containing model parameters
     * @param layer_pattern_fn Function to extract layer info from tensor names
     * @deprecated Use register_model_layers_from_map() instead - context tensors often lack proper names
     */
    void register_model_layers(ggml_context* params_ctx,
                               std::function<std::pair<std::string, int>(const std::string&)> layer_pattern_fn) {
        registry_.register_from_context(params_ctx, "", layer_pattern_fn);
        log_registered_layers();
    }

    /**
     * Register layers from a model's tensor map (preferred method)
     * Uses GGMLBlock::get_param_tensors() which preserves proper tensor names
     * @param tensors Map of tensor name to tensor pointer
     * @param layer_pattern_fn Function to extract layer info from tensor names
     */
    void register_model_layers_from_map(const std::map<std::string, ggml_tensor*>& tensors,
                                        std::function<std::pair<std::string, int>(const std::string&)> layer_pattern_fn) {
        registry_.register_from_map(tensors, layer_pattern_fn);
        log_registered_layers();
    }

private:
    void log_registered_layers() {
        if (config_.log_operations) {
            auto layers = registry_.get_layer_names_sorted();
            LOG_INFO("LayerExecutionEngine: registered %zu layers", layers.size());
            for (const auto& layer : layers) {
                LOG_DEBUG("  - %s: %.2f MB",
                          layer.c_str(),
                          registry_.get_layer_size(layer) / (1024.0 * 1024.0));
            }
        }
    }

public:

    /**
     * Execute a sequence of layers with streaming
     * @param layers The layers to execute in order
     * @param initial_inputs Initial input tensors
     * @param output_ctx Context for output tensor allocation
     * @return Final output tensors
     */
    std::vector<ggml_tensor*> execute_streaming(
        const std::vector<LayerSubgraph>& layers,
        const std::vector<ggml_tensor*>& initial_inputs,
        ggml_context* output_ctx) {

        if (!config_.enabled || layers.empty()) {
            LOG_WARN("LayerExecutionEngine: streaming disabled or no layers");
            return {};
        }

        int64_t total_start = ggml_time_ms();
        std::vector<ggml_tensor*> current_inputs = initial_inputs;

        for (size_t i = 0; i < layers.size(); i++) {
            const auto& layer = layers[i];
            int64_t layer_start = ggml_time_ms();

            // Step 1: Ensure this layer's weights are on GPU
            if (!ensure_layer_loaded(layer.name, static_cast<int>(i))) {
                LOG_ERROR("LayerExecutionEngine: failed to load layer '%s'", layer.name.c_str());
                return {};
            }

            // Step 2: Start prefetching next layer(s) asynchronously
            if (config_.async_prefetch) {
                for (int j = 1; j <= config_.prefetch_layers && i + j < layers.size(); j++) {
                    prefetch_layer(layers[i + j].name);
                }
            }

            // Step 3: Build and execute this layer's subgraph
            ggml_context* layer_ctx = create_layer_context(layer);
            if (layer_ctx == nullptr) {
                LOG_ERROR("LayerExecutionEngine: failed to create context for layer '%s'", layer.name.c_str());
                return {};
            }

            std::vector<ggml_tensor*> outputs = layer.execute_fn(layer_ctx, gpu_backend_, current_inputs);

            // Step 4: Store outputs as intermediates for next layer
            for (size_t j = 0; j < outputs.size(); j++) {
                std::string name = "intermediate_" + std::to_string(i) + "_" + std::to_string(j);
                ggml_tensor* stored = intermediates_.store(name, outputs[j]);
                if (stored != nullptr) {
                    outputs[j] = stored;
                }
            }

            // Step 5: Offload completed layer if needed
            if (should_offload_layer(layer.name, static_cast<int>(i), layers)) {
                registry_.move_layer_to_cpu(layer.name);
            }

            // Step 6: Clean up layer context
            ggml_free(layer_ctx);

            current_inputs = outputs;

            if (config_.log_operations) {
                int64_t layer_end = ggml_time_ms();
                LOG_DEBUG("LayerExecutionEngine: executed layer '%s' in %.2fs",
                          layer.name.c_str(),
                          (layer_end - layer_start) / 1000.0);
            }
        }

        int64_t total_end = ggml_time_ms();
        if (config_.log_operations) {
            LOG_INFO("LayerExecutionEngine: executed %zu layers in %.2fs",
                     layers.size(),
                     (total_end - total_start) / 1000.0);
        }

        return current_inputs;
    }

    /**
     * Clear all state (call between generations)
     */
    void clear() {
        intermediates_.clear();
        // Don't clear registry - model weights persist
    }

    /**
     * Reset for a new model (clears everything including registry)
     */
    void reset() {
        intermediates_.clear();
        registry_.clear();
    }

private:
    /**
     * Ensure a layer's weights are loaded to GPU
     */
    bool ensure_layer_loaded(const std::string& layer_name, int current_idx) {
        if (registry_.is_layer_on_gpu(layer_name)) {
            return true;
        }

        // Use budget manager to ensure space and load
        if (!budget_.ensure_vram_for_layer(layer_name, current_idx)) {
            LOG_ERROR("LayerExecutionEngine: cannot ensure VRAM for layer '%s'", layer_name.c_str());
            return false;
        }

        return registry_.move_layer_to_gpu(layer_name);
    }

    /**
     * Start prefetching a layer asynchronously
     * Note: True async requires CUDA streams, this is a placeholder for now
     */
    void prefetch_layer(const std::string& layer_name) {
        // TODO: Implement async prefetch using ggml_backend_tensor_copy_async
        // For now, this is a no-op - the layer will be loaded synchronously when needed
        // In a full implementation:
        // 1. Use a separate CUDA stream for memory transfers
        // 2. Queue the transfer asynchronously
        // 3. Track pending transfers
    }

    /**
     * Decide if a layer should be offloaded after execution
     */
    bool should_offload_layer(const std::string& layer_name,
                              int layer_idx,
                              const std::vector<LayerSubgraph>& layers) {
        // Don't offload global/shared layers
        if (layer_name == "_global") {
            return false;
        }

        // Don't offload if we have plenty of VRAM
        size_t free_vram = budget_.get_available_vram();
        if (free_vram > config_.min_free_vram * 2) {
            return false;
        }

        // Check if we need this layer's skip connections (UNet)
        if (config_.keep_layers_behind > 0) {
            // For UNet, input_blocks are needed by output_blocks
            // This would need more sophisticated logic
            return false;
        }

        // Offload if we're running low on VRAM
        return free_vram < config_.min_free_vram;
    }

    /**
     * Create a GGML context for a layer's computation
     */
    ggml_context* create_layer_context(const LayerSubgraph& layer) {
        // Estimate context size based on layer complexity
        // This is a rough estimate - actual size depends on the layer
        size_t ctx_size = 1024 * 1024;  // 1 MB base
        if (layer.estimated_compute_size > 0) {
            ctx_size = layer.estimated_compute_size;
        }

        struct ggml_init_params params = {
            ctx_size,
            nullptr,
            true  // no_alloc - we'll use gallocr for proper allocation
        };

        return ggml_init(params);
    }

    ggml_backend_t gpu_backend_;
    ggml_backend_t cpu_backend_;

    TensorRegistry registry_;
    MemoryBudgetManager budget_;
    IntermediateTensorManager intermediates_;

    StreamingConfig config_;
};

/**
 * Helper to build layer subgraphs for Flux model
 * @param depth Number of double_blocks
 * @param depth_single Number of single_blocks
 * @param skip_layers Layers to skip (for caching)
 * @return Vector of LayerSubgraph definitions
 */
inline std::vector<LayerSubgraph> build_flux_layer_subgraphs(
    int depth,
    int depth_single,
    const std::vector<int>& skip_layers = {}) {

    std::vector<LayerSubgraph> layers;

    // Double blocks
    for (int i = 0; i < depth; i++) {
        if (std::find(skip_layers.begin(), skip_layers.end(), i) != skip_layers.end()) {
            continue;
        }

        LayerSubgraph layer;
        layer.name  = "double_blocks." + std::to_string(i);
        layer.index = i;
        // execute_fn will be set by the model when it sets up streaming
        layers.push_back(layer);
    }

    // Single blocks
    for (int i = 0; i < depth_single; i++) {
        if (std::find(skip_layers.begin(), skip_layers.end(), i + depth) != skip_layers.end()) {
            continue;
        }

        LayerSubgraph layer;
        layer.name  = "single_blocks." + std::to_string(i);
        layer.index = depth + i;
        layers.push_back(layer);
    }

    return layers;
}

/**
 * Helper to build layer subgraphs for UNet model
 * Uses coarse stages for UNet due to skip connections
 */
inline std::vector<LayerSubgraph> build_unet_layer_subgraphs(
    int num_input_blocks,
    int num_output_blocks) {

    std::vector<LayerSubgraph> layers;

    // For UNet, we use coarse stages instead of per-layer
    // Stage 1: All input blocks
    LayerSubgraph input_stage;
    input_stage.name  = "input_blocks";
    input_stage.index = 0;
    layers.push_back(input_stage);

    // Stage 2: Middle block
    LayerSubgraph middle_stage;
    middle_stage.name  = "middle_block";
    middle_stage.index = 1;
    layers.push_back(middle_stage);

    // Stage 3: All output blocks
    LayerSubgraph output_stage;
    output_stage.name  = "output_blocks";
    output_stage.index = 2;
    layers.push_back(output_stage);

    return layers;
}

}  // namespace LayerStreaming

#endif  // __LAYER_STREAMING_HPP__
