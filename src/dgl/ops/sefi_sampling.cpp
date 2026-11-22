#include "dgl/ops/sefi_sampling.h"

#include "dgl/models/diffusion_prediction.h"
#include "dgl/models/dit.h"
#include "dgl/models/fwd.h"
#include "dgl/ops/predictor.h"
#include "runtime/denoiser.hpp"

namespace sd::dgl::ops {
    Tensor<float> sefi_sample(const Tensor<float>& input, const Tensor<float>& sigmas, const Tensor<float>& semantic_sigmas, const Predictor& predictor, const SamplingOptions& options, RunEvents& events) {
        if (options.method != "euler")
            throw std::runtime_error("sefi::sample requires the Euler method");
        const auto& texture  = sampling_schedule(sigmas, true);
        const auto& semantic = sampling_schedule(semantic_sigmas, true);
        if (texture.size() != semantic.size())
            throw std::runtime_error("SeFi sigma lists differ in length");
        if (input.dim() != 4 || input.shape()[2] != SefiFlowDenoiser::kTotalChannels)
            throw std::runtime_error("sefi::sample requires a rank-four latent with 144 channels on axis 2");
        PredictionExecution execution(predictor, texture, {options.cache ? &*options.cache : nullptr, false, true, options.batched_cfg});
        if (execution.parameterization().prediction() != SEFI_FLOW_PRED)
            throw std::runtime_error("sefi::sample requires a dual-time SeFi predictor");

        SefiFlowDenoiser schedule;
        schedule.tex_sigmas = texture;
        schedule.sem_sigmas = semantic;
        SamplingSolver solver;
        solver.prepare_step = [&](models::DiffusionPredictionStep& step) {
            step.semantic_sigma = semantic.at(static_cast<size_t>(step.index - 1));
        };
        solver.execute = [&](const Tensor<float>& start, const SamplePrediction& evaluate) {
            return sample_sefi_euler(&schedule, evaluate, start);
        };
        return run_sample(input, texture, execution, solver, options, events);
    }
}  // namespace sd::dgl::ops
