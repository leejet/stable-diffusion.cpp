#include "dgl/bindings/sefi.h"

#include "dgl/bindings/arguments.h"
#include "dgl/bindings/values.h"

#include "dgl/ops/predictor.h"
#include "dgl/ops/sefi_sampling.h"
#include "runtime/denoiser.hpp"

namespace sd::dgl {
    Value sefi_sigmas(Values& args, RunState&) {
        SefiFlowDenoiser denoiser;
        denoiser.timestep_shift_alpha = float(get_number(args, "alpha"));
        denoiser.delta_t              = float(get_number(args, "delta"));
        auto sigmas                   = denoiser.get_sigmas(int(get_int(args, "steps")), 0, DISCRETE_SCHEDULER, VERSION_SEFI_IMAGE);
        return Value::dict({{"sigmas", tensor_value(sd::Tensor<float>::from_vector(std::move(sigmas)))},
                            {"semantic_sigmas", tensor_value(sd::Tensor<float>::from_vector(std::move(denoiser.sem_sigmas)))}});
    }

    Value sample_sefi_tensor(Values& args, RunState& state) {
        const auto predictor = object<ops::Predictor>(args, "predict");
        const auto options   = sampling_options(args, state.context);
        return tensor_value(ops::sefi_sample(float_tensor(args, "start"), sigma_tensor(args, "sigmas"),
                                             sigma_tensor(args, "semantic_sigmas"), *predictor, options, state.events));
    }
}  // namespace sd::dgl
