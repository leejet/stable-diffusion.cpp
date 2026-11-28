#ifndef __SD_DGL_OPS_SPECTRUM_CACHE_H__
#define __SD_DGL_OPS_SPECTRUM_CACHE_H__

#include <array>
#include "dgl/models/diffusion_prediction.h"
#include "runtime/spectrum.hpp"

namespace sd {
    struct CacheConfig;
}
namespace sd::dgl::ops {
    struct Predictor;
    struct ModelPredictor;

    struct PredictorCacheKey {
        std::vector<const Predictor*> switches;
        std::vector<std::pair<const Predictor*, float>> guidance;
        std::vector<std::pair<const Predictor*, float>> slg;
        std::vector<const ModelPredictor*> models;
        std::vector<bool> conditioning;

        bool operator==(const PredictorCacheKey& other) const {
            return switches == other.switches && guidance == other.guidance && slg == other.slg && models == other.models && conditioning == other.conditioning;
        }
    };

    class SpectrumCache {
        bool enabled_ = false;
        std::array<SpectrumState, 2> phases_;
        PredictorCacheKey key_;
        std::vector<int64_t> shape_;
        size_t evaluations_ = 0;

    public:
        SpectrumCache(const CacheConfig* config, size_t steps);
        Tensor<float> predict(const PredictorCacheKey& key, const std::vector<int64_t>& shape, const models::DiffusionPredictionStep& step);
        void record(const Tensor<float>& result, const models::DiffusionPredictionStep& step);
        void log_summary() const;
    };
}
#endif  // __SD_DGL_OPS_SPECTRUM_CACHE_H__
