#ifndef __WEIGHT_MANAGER_H__
#define __WEIGHT_MANAGER_H__

#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

#include "ggml-backend.h"

struct ggml_tensor;

enum class ParamPrefetchResult : uint8_t {
    SUCCESS = 0,
    ALLOCATION_FAILURE,
    FAILURE,
};

struct StreamingPoolAllocation {
    size_t slot_count = 0;
    size_t slot_bytes = 0;

    explicit operator bool() const {
        return slot_count > 0 && slot_bytes > 0;
    }
};

struct RunnerWeightManager {
    virtual ~RunnerWeightManager()                                                          = default;
    virtual bool assign_compute_backend(const std::vector<ggml_tensor*>& tensors,
                                        ggml_backend_t compute_backend)                     = 0;
    virtual bool prepare_params(const std::vector<ggml_tensor*>& tensors)                   = 0;
    virtual size_t release_compute_backend_params(const std::vector<ggml_tensor*>& tensors) = 0;
    virtual void release_params_backend_params(const std::vector<ggml_tensor*>& tensors)    = 0;

    virtual ParamPrefetchResult enqueue_param_prefetch(
        uintptr_t owner_id,
        uint64_t segment_id,
        const std::vector<ggml_tensor*>& tensors,
        bool require_streaming_pool)                                               = 0;
    virtual bool activate_param_prefetch(uintptr_t owner_id,
                                         uint64_t segment_id,
                                         const std::vector<ggml_tensor*>& tensors) = 0;
    virtual void clear_param_prefetches(uintptr_t owner_id)                        = 0;
    virtual StreamingPoolAllocation configure_streaming_pool(uintptr_t owner_id,
                                                             ggml_backend_t compute_backend,
                                                             size_t slot_count,
                                                             size_t slot_bytes) {
        (void)owner_id;
        (void)compute_backend;
        (void)slot_count;
        (void)slot_bytes;
        return {};
    }
    virtual bool release_streaming_pool(uintptr_t owner_id) {
        (void)owner_id;
        return true;
    }
    virtual size_t streaming_allocation_bytes(
        uintptr_t owner_id,
        ggml_backend_t compute_backend,
        const std::unordered_set<const ggml_tensor*>& resident_tensors) const = 0;
};

#endif  // __WEIGHT_MANAGER_H__
