#include "dgl/ops/sampling.h"

#include <cmath>

#include "core/util.h"
#include "dgl/models/dit.h"
#include "dgl/models/fwd.h"
#include "dgl/ops/predictor.h"
#include "dgl/ops/sampling_observer.h"
#include "dgl/run_events.h"
#include "runtime/denoiser.hpp"

namespace sd::dgl::ops {
    const std::vector<float>& sampling_schedule(const sd::Tensor<float>& tensor, bool allow_repeated) {
        const auto& sigmas = tensor.values();
        if (!allow_repeated && sigmas.size() == 1 && std::isfinite(sigmas.front()) && sigmas.front() >= 0.f)
            return sigmas;
        if (sigmas.size() < 2 || sigmas.front() <= 0.f)
            throw std::runtime_error("sampling requires at least two sigmas and a positive first sigma");
        for (size_t i = 0; i < sigmas.size(); ++i) {
            if (!std::isfinite(sigmas[i]) || sigmas[i] < 0.f ||
                (i && (sigmas[i] > sigmas[i - 1] || (!allow_repeated && sigmas[i] == sigmas[i - 1]))))
                throw std::runtime_error(allow_repeated ? "sigmas must be finite, nonnegative and nonincreasing"
                                                        : "sigmas must be finite, nonnegative and strictly decreasing");
        }
        return sigmas;
    }

    static void validate_mask(const Tensor<float>& input, const SamplingOptions& options) {
        if (!options.denoise_mask)
            return;
        if (!options.base)
            throw std::runtime_error("denoise_mask requires a clean base tensor");
        const auto& mask = *options.denoise_mask;
        if (options.base->shape() != input.shape() || tensor_broadcast_shape(mask.shape(), input.shape()) != input.shape())
            throw std::runtime_error("mask or base shape differs from the sampling tensor");
        if (!std::all_of(mask.values().begin(), mask.values().end(), [](float value) { return value >= 0.f && value <= 1.f; }))
            throw std::runtime_error("mask values must be between zero and one");
    }

    Tensor<float> sample(const Tensor<float>& input, const std::vector<float>& sigmas, const Predictor& predictor, const std::shared_ptr<RNG>& rng, const SamplingOptions& options, RunEvents& events) {
        if (sigmas.size() == 1)
            return input;
        const auto method = str_to_sample_method(options.method.c_str());
        if (method == SAMPLE_METHOD_COUNT)
            throw std::runtime_error("unknown sampler '" + options.method + "'");
        const bool need_unconditional = method == EULER_CFG_PP_SAMPLE_METHOD || method == EULER_A_CFG_PP_SAMPLE_METHOD;
        PredictionExecution execution(predictor, sigmas, {options.cache ? &*options.cache : nullptr, need_unconditional, false, options.batched_cfg});
        const auto& parameterization = execution.parameterization();
        SamplingSolver solver;
        solver.execute = [&](const Tensor<float>& start, const SamplePrediction& evaluate) {
            return sample_k_diffusion(method, evaluate, start, sigmas, rng, options.eta, parameterization.is_flow(), options.extra_args.c_str());
        };
        return run_sample(input, sigmas, execution, solver, options, events);
    }

    Tensor<float> run_sample(const Tensor<float>& input, const std::vector<float>& sigmas, PredictionExecution& prediction_run, const SamplingSolver& solver, const SamplingOptions& options, RunEvents& events) {
        validate_mask(input, options);
        const int64_t sampling_start = ggml_time_ms();
        LOG_INFO("sampling started: method=%s, steps=%zu", options.method.c_str(), sigmas.size() - 1);
        auto& model                  = prediction_run.dit();
        const auto& parameterization = prediction_run.parameterization();
        sd::ParallelScope tensor_scope(model.tensor_executor.get());
        const Tensor<float> empty;
        const auto& mask         = options.denoise_mask ? *options.denoise_mask : empty;
        const auto& base         = options.denoise_mask ? *options.base : empty;
        const bool terminal_zero = sigmas.back() == 0.f;
        SamplingObservation observation({model, events, static_cast<int>(sigmas.size() - 1), terminal_zero, parameterization}, options.observer);
        const auto evaluate = [&](const Tensor<float>& x, float sigma, int index) {
            events.check_cancelled();
            models::DiffusionPredictionStep step{base.empty() ? x : base, mask};
            step.index          = std::abs(index);
            step.schedule_size  = sigmas.size();
            step.solver_index   = index;
            step.completes_step = index > 0 || (terminal_zero && index < 0 && step.index == static_cast<int>(sigmas.size() - 1));
            step.mask_input     = options.mask_input;
            const auto target   = static_cast<size_t>(step.index);
            if (target < sigmas.size())
                step.target_sigma = sigmas[target];
            if (index + 1 >= 0 && static_cast<size_t>(index + 1) < sigmas.size())
                step.lookahead_sigma = sigmas[static_cast<size_t>(index + 1)];
            if (solver.prepare_step)
                solver.prepare_step(step);

            const auto observed_step = observation.step(sigma, step);
            observation.before_prediction(observed_step, x);
            auto prediction = prediction_run.evaluate(x, sigma, step);
            step.blend_mask(prediction.pred);
            observation.after_prediction(observed_step, x, prediction);
            return prediction;
        };
        auto output = solver.execute(input, evaluate);
        if (output.empty())
            throw std::runtime_error("sampling failed");
        prediction_run.log_cache_summary();
        if (options.finish.value_or(false) && !terminal_zero)
            output = parameterization.finish(sigmas.back(), output);
        observation.end(output);
        LOG_INFO("sampling completed, taking %.2fs", (ggml_time_ms() - sampling_start) / 1000.0);
        return output;
    }

}  // namespace sd::dgl::ops
