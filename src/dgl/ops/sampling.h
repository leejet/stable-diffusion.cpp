#ifndef __SD_DGL_OPS_SAMPLING_H__
#define __SD_DGL_OPS_SAMPLING_H__

#include "dgl/models/fwd.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "config.h"
#include "core/tensor.hpp"
#include "runtime/guidance.h"

class RNG;

namespace sd::dgl {
    class RunEvents;

    namespace ops {
        struct Predictor;
        class PredictionExecution;
        class SamplingObserver;

        struct SamplingOptions {
            std::string method;
            float eta                             = 1.f;
            const sd::Tensor<float>* base         = nullptr;
            const sd::Tensor<float>* denoise_mask = nullptr;
            bool mask_input                       = false;
            std::string extra_args;
            std::optional<bool> finish;
            std::optional<CacheConfig> cache;
            std::shared_ptr<SamplingObserver> observer;
            bool batched_cfg = true;
        };

        using SamplePrediction = std::function<sd::guidance::GuiderOutput(const Tensor<float>&, float, int)>;

        struct SamplingSolver {
            std::function<Tensor<float>(const Tensor<float>&, const SamplePrediction&)> execute;
            std::function<void(models::DiffusionPredictionStep&)> prepare_step;
        };

        const std::vector<float>& sampling_schedule(const sd::Tensor<float>& tensor, bool allow_repeated = false);
        sd::Tensor<float> sample(const sd::Tensor<float>& input, const std::vector<float>& sigmas, const Predictor& predictor, const std::shared_ptr<RNG>& rng, const SamplingOptions& options, RunEvents& events);
        Tensor<float> run_sample(const Tensor<float>& input, const std::vector<float>& sigmas, PredictionExecution& prediction, const SamplingSolver& solver, const SamplingOptions& options, RunEvents& events);
    }
}

#endif  // __SD_DGL_OPS_SAMPLING_H__
