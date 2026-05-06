#ifndef __MEMORY_BUDGET_HPP__
#define __MEMORY_BUDGET_HPP__

#include <algorithm>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"

#include "tensor_registry.hpp"
#include "util.h"

namespace LayerStreaming {

enum class EvictionPolicy {
    LAYER_DISTANCE,
    LRU,
    LARGEST_FIRST,
};

class MemoryBudgetManager {
public:
    MemoryBudgetManager(TensorRegistry& registry,
                        ggml_backend_t gpu_backend,
                        size_t safety_margin_bytes = 512 * 1024 * 1024)
        : registry_(registry),
          gpu_backend_(gpu_backend),
          safety_margin_(safety_margin_bytes) {
        query_device_memory();
    }

    void set_eviction_policy(EvictionPolicy policy) {
        eviction_policy_ = policy;
    }

    void set_safety_margin(size_t bytes) {
        safety_margin_ = bytes;
    }

    void query_device_memory() {
        // Use runtime backend device API (works for CUDA, Vulkan, Metal, etc.).
        // The previous SD_USE_CUDA gate broke after PR #1448 removed compile-time
        // backend selection, leaving every build on the 8 GB / 4 GB fallback.
        ggml_backend_dev_t dev = gpu_backend_ ? ggml_backend_get_device(gpu_backend_) : nullptr;
        if (dev != nullptr) {
            ggml_backend_dev_memory(dev, &free_vram_, &total_vram_);
        } else {
            total_vram_ = 8ULL * 1024 * 1024 * 1024;
            free_vram_  = total_vram_ / 2;
        }
        LOG_DEBUG("total VRAM = %.2f GB, free = %.2f GB",
                  total_vram_ / (1024.0 * 1024.0 * 1024.0),
                  free_vram_ / (1024.0 * 1024.0 * 1024.0));
    }

    size_t get_free_vram() {
        query_device_memory();
        return free_vram_;
    }

    size_t get_total_vram() const {
        return total_vram_;
    }

    size_t get_available_vram() {
        size_t free = get_free_vram();
        if (free <= safety_margin_) {
            return 0;
        }
        return free - safety_margin_;
    }

    bool has_enough_vram(size_t required_bytes) {
        return get_available_vram() >= required_bytes;
    }

    // Evicts other layers if necessary to make room
    bool ensure_vram_for_layer(const std::string& layer_name, int current_layer_idx = -1) {
        if (registry_.is_layer_on_gpu(layer_name)) {
            return true;
        }

        size_t layer_size = registry_.get_layer_size(layer_name);
        if (layer_size == 0) {
            LOG_ERROR("layer '%s' not found", layer_name.c_str());
            return false;
        }

        if (has_enough_vram(layer_size)) {
            return true;
        }

        size_t needed = layer_size - get_available_vram();
        return evict_layers_for_space(needed, layer_name, current_layer_idx);
    }

    // Dry-run allocation to get exact buffer requirements
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

    bool should_offload_layer(const std::string& layer_name,
                              const std::string& next_layer_name,
                              int keep_layers_ahead = 1) {
        size_t next_layer_size = registry_.get_layer_size(next_layer_name);
        if (has_enough_vram(next_layer_size * (keep_layers_ahead + 1))) {
            return false;
        }
        return true;
    }

    std::vector<std::string> get_suggested_gpu_layers(int current_layer_idx,
                                                       int layers_ahead = 1,
                                                       int layers_behind = 0) {
        auto all_layers = registry_.get_layer_names_sorted();
        std::vector<std::string> result;

        for (const auto& name : all_layers) {
            if (name == "_global") {
                result.push_back(name);
                continue;
            }

            // TODO: filter by index range once layer index tracking is implemented
            result.push_back(name);
        }

        return result;
    }

private:
    bool evict_layers_for_space(size_t bytes_needed,
                                const std::string& protected_layer,
                                int current_layer_idx) {
        auto layers_on_gpu = registry_.get_layers_on_gpu();
        if (layers_on_gpu.empty()) {
            LOG_ERROR("no layers to evict but need %.2f MB",
                      bytes_needed / (1024.0 * 1024.0));
            return false;
        }

        layers_on_gpu.erase(
            std::remove(layers_on_gpu.begin(), layers_on_gpu.end(), protected_layer),
            layers_on_gpu.end());

        // _global contains shared tensors, never evict
        layers_on_gpu.erase(
            std::remove(layers_on_gpu.begin(), layers_on_gpu.end(), "_global"),
            layers_on_gpu.end());

        if (layers_on_gpu.empty()) {
            LOG_ERROR("no evictable layers available");
            return false;
        }

        std::vector<std::pair<std::string, int>> scored_layers;
        for (const auto& layer : layers_on_gpu) {
            int score = compute_eviction_score(layer, current_layer_idx);
            scored_layers.push_back({layer, score});
        }

        std::sort(scored_layers.begin(), scored_layers.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        size_t freed = 0;
        for (const auto& [layer, score] : scored_layers) {
            size_t layer_size = registry_.get_layer_size(layer);
            registry_.move_layer_to_cpu(layer);
            freed += layer_size;

            LOG_DEBUG("evicted layer '%s' (%.2f MB), total freed: %.2f MB",
                      layer.c_str(),
                      layer_size / (1024.0 * 1024.0),
                      freed / (1024.0 * 1024.0));

            if (freed >= bytes_needed) {
                return true;
            }
        }

        LOG_WARN("only freed %.2f MB, needed %.2f MB",
                 freed / (1024.0 * 1024.0),
                 bytes_needed / (1024.0 * 1024.0));
        return freed >= bytes_needed;
    }

    // Higher score = more likely to evict
    int compute_eviction_score(const std::string& layer, int current_layer_idx) {
        switch (eviction_policy_) {
            case EvictionPolicy::LAYER_DISTANCE: {
                int layer_idx = extract_layer_index(layer);
                if (layer_idx < 0 || current_layer_idx < 0) {
                    return 0;
                }
                return std::abs(layer_idx - current_layer_idx);
            }

            case EvictionPolicy::LARGEST_FIRST: {
                return static_cast<int>(registry_.get_layer_size(layer) / (1024 * 1024));
            }

            case EvictionPolicy::LRU:
            default:
                // TODO: LRU needs access tracking in TensorRegistry, falling back to size-based
                return static_cast<int>(registry_.get_layer_size(layer) / (1024 * 1024));
        }
    }

    int extract_layer_index(const std::string& layer_name) {
        size_t db_pos = layer_name.find("double_blocks.");
        if (db_pos != std::string::npos) {
            size_t num_start = db_pos + 14;
            try {
                return std::stoi(layer_name.substr(num_start));
            } catch (...) {
                return -1;
            }
        }

        size_t sb_pos = layer_name.find("single_blocks.");
        if (sb_pos != std::string::npos) {
            size_t num_start = sb_pos + 14;
            try {
                return 19 + std::stoi(layer_name.substr(num_start));  // offset past double_blocks
            } catch (...) {
                return -1;
            }
        }

        size_t ib_pos = layer_name.find("input_blocks.");
        if (ib_pos != std::string::npos) {
            size_t num_start = ib_pos + 13;
            try {
                return std::stoi(layer_name.substr(num_start));
            } catch (...) {
                return -1;
            }
        }

        size_t ob_pos = layer_name.find("output_blocks.");
        if (ob_pos != std::string::npos) {
            size_t num_start = ob_pos + 14;
            try {
                return 200 + std::stoi(layer_name.substr(num_start));
            } catch (...) {
                return -1;
            }
        }

        if (layer_name.find("middle_block") != std::string::npos) {
            return 100;
        }

        return -1;
    }

    TensorRegistry& registry_;
    ggml_backend_t gpu_backend_;

    size_t total_vram_    = 0;
    size_t free_vram_     = 0;
    size_t safety_margin_ = 512 * 1024 * 1024;

    EvictionPolicy eviction_policy_ = EvictionPolicy::LAYER_DISTANCE;
};

}  // namespace LayerStreaming

#endif  // __MEMORY_BUDGET_HPP__
