#ifndef __SD_RUNTIME_SAMPLE_CACHE_H__
#define __SD_RUNTIME_SAMPLE_CACHE_H__

#include "config.h"

#include <vector>

#include "core/tensor.hpp"
#include "core/util.h"
#include "model.h"
#include "runtime/cache_dit.hpp"
#include "runtime/denoiser.hpp"
#include "runtime/easycache.hpp"
#include "runtime/spectrum.hpp"
#include "runtime/ucache.hpp"

namespace sd_sample {

    enum class SampleCacheMode {
        NONE,
        EASYCACHE,
        UCACHE,
        CACHEDIT,
    };

    struct ConditionCacheRuntime {
        SampleCacheMode mode = SampleCacheMode::NONE;

        EasyCacheState easycache;
        UCacheState ucache;
        CacheDitConditionState cachedit;
        bool easycache_enabled() const;
        bool ucache_enabled() const;
        bool cachedit_enabled() const;
        bool enabled() const { return mode != SampleCacheMode::NONE; }
        void retain_conditions(const std::vector<const void*>& conditions);
        void invalidate_condition(const void* condition);
    };

    struct SampleCacheRuntime : ConditionCacheRuntime {
        SpectrumState spectrum;
        bool spectrum_enabled = false;
    };

    struct SampleStepCacheDispatcher {
        ConditionCacheRuntime& runtime;
        int step;
        float sigma;
        int step_index;

        SampleStepCacheDispatcher(ConditionCacheRuntime& runtime, int step, float sigma);

        bool before_condition(const void* condition, const sd::Tensor<float>& input, sd::Tensor<float>* output);
        void after_condition(const void* condition, const sd::Tensor<float>& input, const sd::Tensor<float>& output);
        void invalidate_condition(const void* condition);
        bool is_step_skipped() const;
    };

    ConditionCacheRuntime init_condition_cache_runtime(SDVersion version, const sd::CacheConfig* params, Denoiser* denoiser, const std::vector<float>& sigmas);
    SpectrumConfig spectrum_config(const sd::CacheConfig& params);
    void log_condition_cache_summary(const ConditionCacheRuntime& runtime, size_t total_steps);

    SampleCacheRuntime init_sample_cache_runtime(SDVersion version,
                                                 const sd::CacheConfig* cache_params,
                                                 Denoiser* denoiser,
                                                 const std::vector<float>& sigmas);

    void log_sample_cache_summary(const SampleCacheRuntime& runtime, size_t total_steps);

}  // namespace sd_sample

#endif  // __SD_RUNTIME_SAMPLE_CACHE_H__
