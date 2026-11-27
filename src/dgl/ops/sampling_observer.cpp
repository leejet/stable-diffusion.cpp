#include "dgl/ops/sampling_observer.h"

#include "dgl/models/diffusion_prediction.h"
#include "dgl/models/fwd.h"
#include "dgl/run_events.h"

namespace sd::dgl::ops {
    SamplingObservation::SamplingObservation(SamplingRun run, const std::shared_ptr<SamplingObserver>& observer)
        : run_(run) {
        run_.events.begin_sample(run_.total_steps);
        if (observer)
            session_ = observer->start(run_);
        run_.events.check_cancelled();
    }

    SamplingStep SamplingObservation::step(float sigma, const models::DiffusionPredictionStep& prediction) {
        return {prediction.index, ++evaluations_, prediction.completes_step && prediction.index > completed_, sigma, prediction};
    }

    void SamplingObservation::before_prediction(const SamplingStep& step, const Tensor<float>& x) {
        run_.events.check_cancelled();
        if (session_)
            session_->before_prediction(step, x);
        run_.events.check_cancelled();
    }

    void SamplingObservation::after_prediction(const SamplingStep& step, const Tensor<float>& x, const sd::guidance::GuiderOutput& prediction) {
        run_.events.check_cancelled();
        if (session_)
            session_->after_prediction(step, x, prediction);
        run_.events.check_cancelled();
        if (step.completes_step) {
            completed_ = step.index;
            if (session_)
                session_->step_completed(step);
            run_.events.sample_step(completed_, run_.total_steps);
        }
        run_.events.check_cancelled();
    }

    void SamplingObservation::end(const Tensor<float>& output) {
        run_.events.check_cancelled();
        if (session_)
            session_->end(output);
        run_.events.check_cancelled();
    }
}  // namespace sd::dgl::ops
