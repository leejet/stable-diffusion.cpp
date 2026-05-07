#ifndef __LAYER_STREAMING_HPP__
#define __LAYER_STREAMING_HPP__

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include "memory_budget.hpp"
#include "tensor_registry.hpp"
#include "util.h"

namespace LayerStreaming {

class LayerExecutionEngine;

struct LayerSubgraph {
    std::string name;
    int index;
    size_t estimated_compute_size = 0;

    using ExecuteFn = std::function<std::vector<ggml_tensor*>(
        ggml_context* ctx,
        ggml_backend_t backend,
        const std::vector<ggml_tensor*>& inputs)>;

    ExecuteFn execute_fn;
};

struct StreamingConfig {
    bool enabled              = false;
    int prefetch_layers       = 1;
    int keep_layers_behind    = 0;
    size_t min_free_vram      = 512 * 1024 * 1024;
    bool async_prefetch       = true;
    bool log_operations       = false;
};

class IntermediateTensorManager {
public:
    IntermediateTensorManager(ggml_backend_t gpu_backend)
        : gpu_backend_(gpu_backend) {}

    ~IntermediateTensorManager() {
        clear();
    }

    ggml_tensor* store(const std::string& name, ggml_tensor* tensor) {
        if (contexts_.find(name) != contexts_.end()) {
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
            LOG_ERROR("failed to create context for '%s'", name.c_str());
            return nullptr;
        }

        ggml_tensor* stored = ggml_dup_tensor(ctx, tensor);
        ggml_set_name(stored, name.c_str());

        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, gpu_backend_);
        if (buffer == nullptr) {
            LOG_ERROR("failed to allocate buffer for '%s'", name.c_str());
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

    ggml_tensor* get(const std::string& name) {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) {
            return nullptr;
        }
        return it->second;
    }

    bool has(const std::string& name) const {
        return tensors_.find(name) != tensors_.end();
    }

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

class LayerExecutionEngine {
public:
    LayerExecutionEngine(ggml_backend_t gpu_backend,
                         ggml_backend_t cpu_backend)
        : gpu_backend_(gpu_backend),
          cpu_backend_(cpu_backend),
          registry_(gpu_backend, cpu_backend),
          budget_(registry_, gpu_backend),
          intermediates_(gpu_backend) {}

    void set_config(const StreamingConfig& config) {
        config_ = config;
    }

    const StreamingConfig& get_config() const {
        return config_;
    }

    TensorRegistry& get_registry() {
        return registry_;
    }

    MemoryBudgetManager& get_budget() {
        return budget_;
    }

    // Prefer register_model_layers_from_map() - context tensors often lack proper names
    void register_model_layers(ggml_context* params_ctx,
                               std::function<std::pair<std::string, int>(const std::string&)> layer_pattern_fn) {
        registry_.register_from_context(params_ctx, "", layer_pattern_fn);
        log_registered_layers();
    }

    void register_model_layers_from_map(const std::map<std::string, ggml_tensor*>& tensors,
                                        std::function<std::pair<std::string, int>(const std::string&)> layer_pattern_fn) {
        registry_.register_from_map(tensors, layer_pattern_fn);
        log_registered_layers();
    }

private:
    void log_registered_layers() {
        if (config_.log_operations) {
            auto layers = registry_.get_layer_names_sorted();
            LOG_INFO("registered %zu layers", layers.size());
            for (const auto& layer : layers) {
                LOG_DEBUG("  - %s: %.2f MB",
                          layer.c_str(),
                          registry_.get_layer_size(layer) / (1024.0 * 1024.0));
            }
        }
    }

public:

    std::vector<ggml_tensor*> execute_streaming(
        const std::vector<LayerSubgraph>& layers,
        const std::vector<ggml_tensor*>& initial_inputs,
        ggml_context* output_ctx) {

        if (!config_.enabled || layers.empty()) {
            LOG_WARN("streaming disabled or no layers");
            return {};
        }

        int64_t total_start = ggml_time_ms();
        std::vector<ggml_tensor*> current_inputs = initial_inputs;

        for (size_t i = 0; i < layers.size(); i++) {
            const auto& layer = layers[i];
            int64_t layer_start = ggml_time_ms();

            if (!ensure_layer_loaded(layer.name, static_cast<int>(i))) {
                LOG_ERROR("failed to load layer '%s'", layer.name.c_str());
                return {};
            }

            if (config_.async_prefetch) {
                for (int j = 1; j <= config_.prefetch_layers && i + j < layers.size(); j++) {
                    prefetch_layer(layers[i + j].name);
                }
            }

            ggml_context* layer_ctx = create_layer_context(layer);
            if (layer_ctx == nullptr) {
                LOG_ERROR("failed to create context for layer '%s'", layer.name.c_str());
                return {};
            }

            std::vector<ggml_tensor*> outputs = layer.execute_fn(layer_ctx, gpu_backend_, current_inputs);

            for (size_t j = 0; j < outputs.size(); j++) {
                std::string name = "intermediate_" + std::to_string(i) + "_" + std::to_string(j);
                ggml_tensor* stored = intermediates_.store(name, outputs[j]);
                if (stored != nullptr) {
                    outputs[j] = stored;
                }
            }

            if (should_offload_layer(layer.name, static_cast<int>(i), layers)) {
                registry_.move_layer_to_cpu(layer.name);
            }

            ggml_free(layer_ctx);

            current_inputs = outputs;

            if (config_.log_operations) {
                int64_t layer_end = ggml_time_ms();
                LOG_DEBUG("executed layer '%s' in %.2fs",
                          layer.name.c_str(),
                          (layer_end - layer_start) / 1000.0);
            }
        }

        int64_t total_end = ggml_time_ms();
        if (config_.log_operations) {
            LOG_INFO("executed %zu layers in %.2fs",
                     layers.size(),
                     (total_end - total_start) / 1000.0);
        }

        return current_inputs;
    }

    void clear() {
        intermediates_.clear();
    }

    // Clears everything including registry (for new model)
    void reset() {
        intermediates_.clear();
        registry_.clear();
    }

    void prefetch_layer(const std::string& layer_name) {
        if (!config_.async_prefetch) {
            return;
        }

        if (registry_.is_layer_on_gpu(layer_name)) {
            return;
        }

        if (pending_prefetches_.find(layer_name) != pending_prefetches_.end()) {
            return;
        }

        if (registry_.start_async_layer_load(layer_name, gpu_backend_, cpu_backend_)) {
            pending_prefetches_.insert(layer_name);
            if (config_.log_operations) {
                LOG_DEBUG("started async prefetch for '%s'", layer_name.c_str());
            }
        }
    }

    void wait_for_prefetch(const std::string& layer_name) {
        auto it = pending_prefetches_.find(layer_name);
        if (it == pending_prefetches_.end()) {
            return;
        }

        if (registry_.complete_async_layer_load(layer_name, gpu_backend_)) {
            pending_prefetches_.erase(it);
            if (config_.log_operations) {
                LOG_DEBUG("completed async prefetch for '%s'", layer_name.c_str());
            }
        }
    }

    void wait_for_all_prefetches() {
        for (const auto& layer_name : pending_prefetches_) {
            registry_.complete_async_layer_load(layer_name, gpu_backend_);
        }
        pending_prefetches_.clear();
    }

    bool is_prefetch_pending(const std::string& layer_name) const {
        return pending_prefetches_.find(layer_name) != pending_prefetches_.end();
    }

    // Decides how many blocks to keep permanently resident on GPU for a
    // section of the model (e.g. all "layers.N" or all "double_blocks.N").
    // Static partition follows ComfyUI's partially_load() — for the cyclic
    // sequential access pattern of diffusion sampling, caching a fixed
    // prefix is simpler and faster than dynamic eviction. Caller is
    // responsible for storing the result and only computing it once per
    // section so that consecutive calls inside the same generation see a
    // consistent VRAM budget.
    //
    // sample_block_name should be a real block in the section (e.g.
    // "layers.0") so per-block size can be measured. compute_buffer_reserve
    // should be set per-runner to the peak compute buffer observed during
    // a single block forward pass.
    int compute_resident_block_count(const std::string& sample_block_name,
                                     int num_blocks,
                                     size_t compute_buffer_reserve = 768ULL * 1024 * 1024) {
        if (num_blocks <= 0) {
            return 0;
        }

        size_t per_block = registry_.get_layer_size(sample_block_name);
        if (per_block == 0) {
            return 0;
        }

        // Headroom: prefetch window in flight + the active block + the
        // upcoming compute buffer + a hard safety margin. Without this
        // slack the next prefetch's cudaMalloc can fail mid-loop.
        int prefetch_count = std::max(1, config_.prefetch_layers);
        size_t prefetch_reserve = static_cast<size_t>(prefetch_count + 1) * per_block;
        size_t safety = std::max<size_t>(config_.min_free_vram, 512ULL * 1024 * 1024);
        size_t reserved = prefetch_reserve + safety + compute_buffer_reserve;

        size_t free_vram = budget_.get_free_vram();
        if (free_vram <= reserved) {
            return 0;
        }
        size_t available = free_vram - reserved;
        int max_resident = static_cast<int>(available / per_block);
        return std::min(num_blocks, max_resident);
    }

    // Prime the prefetch pipeline by kicking off transfers for the first
    // prefetch_layers blocks starting at start_idx. Call once before the
    // streaming loop. name_for(i) -> the registry key for block i.
    void prime_prefetch(const std::function<std::string(int)>& name_for,
                        int start_idx, int num_blocks) {
        int n = config_.prefetch_layers > 0 ? config_.prefetch_layers : 1;
        for (int j = 0; j < n && (start_idx + j) < num_blocks; j++) {
            prefetch_layer(name_for(start_idx + j));
        }
    }

    // After moving block current_idx to GPU, kick off prefetch of the slot
    // (current_idx + prefetch_layers) so the window stays full.
    void advance_prefetch(const std::function<std::string(int)>& name_for,
                          int current_idx, int num_blocks) {
        int n = config_.prefetch_layers > 0 ? config_.prefetch_layers : 1;
        int target = current_idx + n;
        if (target < num_blocks) {
            prefetch_layer(name_for(target));
        }
    }

private:
    bool ensure_layer_loaded(const std::string& layer_name, int current_idx) {
        if (registry_.is_layer_on_gpu(layer_name)) {
            return true;
        }

        if (!budget_.ensure_vram_for_layer(layer_name, current_idx)) {
            LOG_ERROR("cannot ensure VRAM for layer '%s'", layer_name.c_str());
            return false;
        }

        return registry_.move_layer_to_gpu(layer_name);
    }

    bool should_offload_layer(const std::string& layer_name,
                              int layer_idx,
                              const std::vector<LayerSubgraph>& layers) {
        if (layer_name == "_global") {
            return false;
        }

        size_t free_vram = budget_.get_available_vram();
        if (free_vram > config_.min_free_vram * 2) {
            return false;
        }

        // UNet skip connections need more sophisticated logic
        if (config_.keep_layers_behind > 0) {
            return false;
        }

        return free_vram < config_.min_free_vram;
    }

    ggml_context* create_layer_context(const LayerSubgraph& layer) {
        size_t ctx_size = 1024 * 1024;
        if (layer.estimated_compute_size > 0) {
            ctx_size = layer.estimated_compute_size;
        }

        struct ggml_init_params params = {
            ctx_size,
            nullptr,
            true  // no_alloc
        };

        return ggml_init(params);
    }

    ggml_backend_t gpu_backend_;
    ggml_backend_t cpu_backend_;

    TensorRegistry registry_;
    MemoryBudgetManager budget_;
    IntermediateTensorManager intermediates_;

    StreamingConfig config_;

    std::set<std::string> pending_prefetches_;
};

inline std::vector<LayerSubgraph> build_flux_layer_subgraphs(
    int depth,
    int depth_single,
    const std::vector<int>& skip_layers = {}) {

    std::vector<LayerSubgraph> layers;

    for (int i = 0; i < depth; i++) {
        if (std::find(skip_layers.begin(), skip_layers.end(), i) != skip_layers.end()) {
            continue;
        }

        LayerSubgraph layer;
        layer.name  = "double_blocks." + std::to_string(i);
        layer.index = i;
        layers.push_back(layer);
    }

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

// UNet uses coarse stages due to skip connections
inline std::vector<LayerSubgraph> build_unet_layer_subgraphs(
    int num_input_blocks,
    int num_output_blocks) {

    std::vector<LayerSubgraph> layers;

    LayerSubgraph input_stage;
    input_stage.name  = "input_blocks";
    input_stage.index = 0;
    layers.push_back(input_stage);

    LayerSubgraph middle_stage;
    middle_stage.name  = "middle_block";
    middle_stage.index = 1;
    layers.push_back(middle_stage);

    LayerSubgraph output_stage;
    output_stage.name  = "output_blocks";
    output_stage.index = 2;
    layers.push_back(output_stage);

    return layers;
}

}  // namespace LayerStreaming

#endif  // __LAYER_STREAMING_HPP__
