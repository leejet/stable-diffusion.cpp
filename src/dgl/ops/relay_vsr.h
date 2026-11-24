#ifndef __SD_DGL_OPS_RELAY_VSR_H__
#define __SD_DGL_OPS_RELAY_VSR_H__

#include "dgl/models/fwd.h"

#include "core/tensor.hpp"
#include "model_build_context.h"
#include "model_component.h"
#include "model_manager.h"

namespace sd::dgl::ops {
    std::vector<int64_t> relay_vsr_latent_shape(const DiffusionModelRunner& model, int64_t width, int64_t height, int64_t frames);
    Tensor<float> relay_vsr_positive_embedding(const DiffusionModelRunner& model, ModelManager::ResourceHandle& weights);
    Tensor<uint8_t> relay_vsr_decode(::VAE& model, int threads, const Tensor<float>& latent, const Tensor<uint8_t>& frames, const Tensor<int64_t>& indices, float detail);
}  // namespace sd::dgl::ops

#endif  // __SD_DGL_OPS_RELAY_VSR_H__
