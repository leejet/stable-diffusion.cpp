#include "dgl/ops/spectrum_cache.h"

#include "config.h"
#include "runtime/sample-cache.h"

namespace sd::dgl::ops {
    SpectrumCache::SpectrumCache(const CacheConfig* config, size_t steps) {
        enabled_ = config && config->mode == SD_CACHE_SPECTRUM;
        if (enabled_) {
            for (auto& phase : phases_)
                phase.init(sd_sample::spectrum_config(*config), steps);
            LOG_INFO("Spectrum enabled - warmup: %d logical steps, stop: %.0f%%; separate predictor/corrector histories",
                     config->spectrum_warmup_steps, config->spectrum_stop_percent * 100.f);
        }
    }

    Tensor<float> SpectrumCache::predict(const PredictorCacheKey& key, const std::vector<int64_t>& shape, const models::DiffusionPredictionStep& step) {
        if (!enabled_)
            return {};
        ++evaluations_;
        if (!(key_ == key) || shape_ != shape) {
            for (auto& phase : phases_)
                phase.reset_history();
            key_   = key;
            shape_ = shape;
        }
        // A Heun predictor and corrector observe different inputs at the same logical step.
        auto& phase     = phases_[step.solver_index < 0 ? 0 : 1];
        const int index = step.index - 1;
        if (!phase.should_predict(index))
            return {};
        Tensor<float> output(shape);
        phase.predict(&output, index);
        return output;
    }

    void SpectrumCache::record(const Tensor<float>& result, const models::DiffusionPredictionStep& step) {
        if (enabled_)
            phases_[step.solver_index < 0 ? 0 : 1].update(result, step.index - 1);
    }

    void SpectrumCache::log_summary() const {
        if (enabled_)
            LOG_INFO("Spectrum reused %d/%zu predictor evaluations",
                     phases_[0].total_steps_skipped + phases_[1].total_steps_skipped, evaluations_);
    }
}
