#ifndef __MEMORY_BUDGET_HPP__
#define __MEMORY_BUDGET_HPP__

#include <algorithm>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"

#include "tensor_registry.hpp"
#include "util.h"

#ifdef SD_USE_CUDA
#include "ggml-cuda.h"
#endif

/**
 * MemoryBudgetManager - Manages GPU memory budget for layer streaming
 *
 * This component:
 * 1. Tracks total and free GPU memory
 * 2. Decides which layers to evict when memory is needed
 * 3. Estimates memory requirements for upcoming operations
 * 4. Implements eviction policies (e.g., distance-based, LRU)
 */

namespace LayerStreaming {

// Eviction policy types
enum class EvictionPolicy {
    LAYER_DISTANCE,  // Evict layers farthest from current execution point
    LRU,             // Evict least recently used layers
    LARGEST_FIRST,   // Evict largest layers first
};

/**
 * MemoryBudgetManager decides when and what to offload
 */
class MemoryBudgetManager {
public:
    MemoryBudgetManager(TensorRegistry& registry,
                        ggml_backend_t gpu_backend,
                        size_t safety_margin_bytes = 512 * 1024 * 1024)  // 512 MB default safety margin
        : registry_(registry),
          gpu_backend_(gpu_backend),
          safety_margin_(safety_margin_bytes) {
        // Query total VRAM
        query_device_memory();
    }

    /**
     * Set the eviction policy
     */
    void set_eviction_policy(EvictionPolicy policy) {
        eviction_policy_ = policy;
    }

    /**
     * Set safety margin (memory to keep free)
     */
    void set_safety_margin(size_t bytes) {
        safety_margin_ = bytes;
    }

    /**
     * Query current device memory status
     */
    void query_device_memory() {
#ifdef SD_USE_CUDA
        // Get CUDA device memory
        ggml_backend_cuda_get_device_memory(0, &free_vram_, &total_vram_);
#else
        // For non-CUDA backends, use conservative estimates
        // This could be extended for other backends (Vulkan, Metal, etc.)
        total_vram_ = 8ULL * 1024 * 1024 * 1024;  // Assume 8 GB
        free_vram_  = total_vram_ / 2;            // Assume half free
#endif
        LOG_DEBUG("MemoryBudgetManager: total VRAM = %.2f GB, free = %.2f GB",
                  total_vram_ / (1024.0 * 1024.0 * 1024.0),
                  free_vram_ / (1024.0 * 1024.0 * 1024.0));
    }

    /**
     * Get current free VRAM (refreshed)
     */
    size_t get_free_vram() {
        query_device_memory();
        return free_vram_;
    }

    /**
     * Get total VRAM
     */
    size_t get_total_vram() const {
        return total_vram_;
    }

    /**
     * Get available VRAM (accounting for safety margin)
     */
    size_t get_available_vram() {
        size_t free = get_free_vram();
        if (free <= safety_margin_) {
            return 0;
        }
        return free - safety_margin_;
    }

    /**
     * Check if we have enough VRAM for a given requirement
     */
    bool has_enough_vram(size_t required_bytes) {
        return get_available_vram() >= required_bytes;
    }

    /**
     * Ensure VRAM is available for a specific layer
     * Will evict other layers if necessary
     * @param layer_name The layer we want to load
     * @param current_layer_idx Current execution position (for distance-based eviction)
     * @return true if VRAM is now available
     */
    bool ensure_vram_for_layer(const std::string& layer_name, int current_layer_idx = -1) {
        if (registry_.is_layer_on_gpu(layer_name)) {
            return true;  // Already on GPU
        }

        size_t layer_size = registry_.get_layer_size(layer_name);
        if (layer_size == 0) {
            LOG_ERROR("MemoryBudgetManager: layer '%s' not found", layer_name.c_str());
            return false;
        }

        // Check if we already have enough space
        if (has_enough_vram(layer_size)) {
            return true;
        }

        // Need to evict some layers
        size_t needed = layer_size - get_available_vram();
        return evict_layers_for_space(needed, layer_name, current_layer_idx);
    }

    /**
     * Estimate compute buffer size for a graph
     * This performs a dry-run allocation to get exact requirements
     */
    size_t estimate_compute_buffer_size(ggml_cgraph* graph) {
        if (graph == nullptr) {
            return 0;
        }

        ggml_gallocr_t temp_allocr = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(gpu_backend_));

        if (!ggml_gallocr_reserve(temp_allocr, graph)) {
            ggml_gallocr_free(temp_allocr);
            return 0;
        }

        size_t compute_size = ggml_gallocr_get_buffer_size(temp_allocr, 0);
        ggml_gallocr_free(temp_allocr);

        return compute_size;
    }

    /**
     * Check if a layer should be offloaded after execution
     * @param layer_name The layer to check
     * @param next_layer_name The next layer to be executed
     * @param keep_layers_ahead How many layers ahead to keep in GPU
     * @return true if layer should be offloaded
     */
    bool should_offload_layer(const std::string& layer_name,
                              const std::string& next_layer_name,
                              int keep_layers_ahead = 1) {
        // If we have plenty of VRAM, don't offload
        size_t next_layer_size = registry_.get_layer_size(next_layer_name);
        if (has_enough_vram(next_layer_size * (keep_layers_ahead + 1))) {
            return false;
        }

        // If we're running low on VRAM, offload completed layers
        return true;
    }

    /**
     * Get suggested layers to keep on GPU based on current position
     * @param current_layer_idx Current execution position
     * @param layers_ahead How many layers ahead to keep
     * @param layers_behind How many layers behind to keep (for skip connections)
     */
    std::vector<std::string> get_suggested_gpu_layers(int current_layer_idx,
                                                       int layers_ahead = 1,
                                                       int layers_behind = 0) {
        auto all_layers = registry_.get_layer_names_sorted();
        std::vector<std::string> result;

        for (const auto& name : all_layers) {
            // Always keep global layers
            if (name == "_global") {
                result.push_back(name);
                continue;
            }

            // Get layer index from registry
            size_t layer_size = registry_.get_layer_size(name);
            // For now, use a simple range check
            // In a full implementation, we'd track layer indices properly
            result.push_back(name);  // Simplified - would filter by index in production
        }

        return result;
    }

private:
    /**
     * Evict layers to free up space
     * @param bytes_needed How many bytes we need to free
     * @param protected_layer Layer that should NOT be evicted
     * @param current_layer_idx Current execution position (for distance-based eviction)
     * @return true if we freed enough space
     */
    bool evict_layers_for_space(size_t bytes_needed,
                                const std::string& protected_layer,
                                int current_layer_idx) {
        auto layers_on_gpu = registry_.get_layers_on_gpu();
        if (layers_on_gpu.empty()) {
            LOG_ERROR("MemoryBudgetManager: no layers to evict but need %.2f MB",
                      bytes_needed / (1024.0 * 1024.0));
            return false;
        }

        // Remove protected layer from candidates
        layers_on_gpu.erase(
            std::remove(layers_on_gpu.begin(), layers_on_gpu.end(), protected_layer),
            layers_on_gpu.end());

        // Also protect _global layer (shared tensors)
        layers_on_gpu.erase(
            std::remove(layers_on_gpu.begin(), layers_on_gpu.end(), "_global"),
            layers_on_gpu.end());

        if (layers_on_gpu.empty()) {
            LOG_ERROR("MemoryBudgetManager: no evictable layers available");
            return false;
        }

        // Sort candidates by eviction policy
        std::vector<std::pair<std::string, int>> scored_layers;
        for (const auto& layer : layers_on_gpu) {
            int score = compute_eviction_score(layer, current_layer_idx);
            scored_layers.push_back({layer, score});
        }

        // Sort by score (higher score = more likely to evict)
        std::sort(scored_layers.begin(), scored_layers.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        // Evict layers until we have enough space
        size_t freed = 0;
        for (const auto& [layer, score] : scored_layers) {
            size_t layer_size = registry_.get_layer_size(layer);
            registry_.move_layer_to_cpu(layer);
            freed += layer_size;

            LOG_DEBUG("MemoryBudgetManager: evicted layer '%s' (%.2f MB), total freed: %.2f MB",
                      layer.c_str(),
                      layer_size / (1024.0 * 1024.0),
                      freed / (1024.0 * 1024.0));

            if (freed >= bytes_needed) {
                return true;
            }
        }

        LOG_WARN("MemoryBudgetManager: only freed %.2f MB, needed %.2f MB",
                 freed / (1024.0 * 1024.0),
                 bytes_needed / (1024.0 * 1024.0));
        return freed >= bytes_needed;
    }

    /**
     * Compute eviction score for a layer (higher = more likely to evict)
     */
    int compute_eviction_score(const std::string& layer, int current_layer_idx) {
        switch (eviction_policy_) {
            case EvictionPolicy::LAYER_DISTANCE: {
                // Extract layer index from name and compute distance from current position
                // Layers farther from current position get higher scores
                int layer_idx = extract_layer_index(layer);
                if (layer_idx < 0 || current_layer_idx < 0) {
                    return 0;  // Can't compute distance
                }
                return std::abs(layer_idx - current_layer_idx);
            }

            case EvictionPolicy::LARGEST_FIRST: {
                // Larger layers get higher scores
                return static_cast<int>(registry_.get_layer_size(layer) / (1024 * 1024));
            }

            case EvictionPolicy::LRU:
            default:
                // For LRU, we'd need access tracking in TensorRegistry
                // For now, fall back to size-based
                return static_cast<int>(registry_.get_layer_size(layer) / (1024 * 1024));
        }
    }

    /**
     * Extract numeric layer index from layer name
     */
    int extract_layer_index(const std::string& layer_name) {
        // Handle "double_blocks.N" pattern
        size_t db_pos = layer_name.find("double_blocks.");
        if (db_pos != std::string::npos) {
            size_t num_start = db_pos + 14;
            try {
                return std::stoi(layer_name.substr(num_start));
            } catch (...) {
                return -1;
            }
        }

        // Handle "single_blocks.N" pattern
        size_t sb_pos = layer_name.find("single_blocks.");
        if (sb_pos != std::string::npos) {
            size_t num_start = sb_pos + 14;
            try {
                return 19 + std::stoi(layer_name.substr(num_start));  // Offset by double_blocks count
            } catch (...) {
                return -1;
            }
        }

        // Handle "input_blocks.N" pattern
        size_t ib_pos = layer_name.find("input_blocks.");
        if (ib_pos != std::string::npos) {
            size_t num_start = ib_pos + 13;
            try {
                return std::stoi(layer_name.substr(num_start));
            } catch (...) {
                return -1;
            }
        }

        // Handle "output_blocks.N" pattern
        size_t ob_pos = layer_name.find("output_blocks.");
        if (ob_pos != std::string::npos) {
            size_t num_start = ob_pos + 14;
            try {
                return 200 + std::stoi(layer_name.substr(num_start));  // High offset
            } catch (...) {
                return -1;
            }
        }

        // Handle "middle_block"
        if (layer_name.find("middle_block") != std::string::npos) {
            return 100;  // Between input and output blocks
        }

        return -1;  // Unknown layer type
    }

    TensorRegistry& registry_;
    ggml_backend_t gpu_backend_;

    size_t total_vram_    = 0;
    size_t free_vram_     = 0;
    size_t safety_margin_ = 512 * 1024 * 1024;  // 512 MB default

    EvictionPolicy eviction_policy_ = EvictionPolicy::LAYER_DISTANCE;
};

}  // namespace LayerStreaming

#endif  // __MEMORY_BUDGET_HPP__
