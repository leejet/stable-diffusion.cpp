#ifndef __SD_DGL_OPS_SEFI_SAMPLING_H__
#define __SD_DGL_OPS_SEFI_SAMPLING_H__

#include "dgl/models/fwd.h"

#include "dgl/ops/sampling.h"

namespace sd::dgl::ops {
    Tensor<float> sefi_sample(const Tensor<float>& input, const Tensor<float>& sigmas, const Tensor<float>& semantic_sigmas, const Predictor& predictor, const SamplingOptions& options, RunEvents& events);
}

#endif  // __SD_DGL_OPS_SEFI_SAMPLING_H__
