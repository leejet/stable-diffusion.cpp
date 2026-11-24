#ifndef __SD_DGL_OPS_SAMPLING_OBSERVER_H__
#define __SD_DGL_OPS_SAMPLING_OBSERVER_H__

#include "dgl/models/fwd.h"

#include <memory>

#include "core/tensor.hpp"
#include "runtime/guidance.h"

namespace sd::dgl {
    class RunEvents;

    namespace ops {

        struct SamplingRun {
            models::DiT& dit;
            RunEvents& events;
            int total_steps;
            bool terminal_zero;
            const models::DenoisingParameterization& parameterization;
        };

        struct SamplingStep {
            int index;
            int evaluation;
            bool completes_step;
            float sigma;
            const models::DiffusionPredictionStep& prediction;
        };

        class SamplingObserverSession {
        public:
            virtual ~SamplingObserverSession() = default;
            virtual void before_prediction(const SamplingStep&, const Tensor<float>&) {}
            virtual void after_prediction(const SamplingStep&, const Tensor<float>&, const sd::guidance::GuiderOutput&) {}
            virtual void step_completed(const SamplingStep&) {}
            virtual void end(const Tensor<float>&) {}
        };

        class SamplingObserver {
        public:
            virtual ~SamplingObserver()                                                          = default;
            virtual std::unique_ptr<SamplingObserverSession> start(const SamplingRun& run) const = 0;
        };

        class SamplingObservation {
            SamplingRun run_;
            std::unique_ptr<SamplingObserverSession> session_;
            int evaluations_ = 0;
            int completed_   = 0;

        public:
            SamplingObservation(SamplingRun run, const std::shared_ptr<SamplingObserver>& observer);
            SamplingStep step(float sigma, const models::DiffusionPredictionStep& prediction);
            void before_prediction(const SamplingStep& step, const Tensor<float>& x);
            void after_prediction(const SamplingStep& step, const Tensor<float>& x, const sd::guidance::GuiderOutput& prediction);
            void end(const Tensor<float>& output);
        };
    }
}  // namespace sd::dgl

#endif  // __SD_DGL_OPS_SAMPLING_OBSERVER_H__
