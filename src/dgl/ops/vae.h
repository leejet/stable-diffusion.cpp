#ifndef __SD_DGL_OPS_VAE_H__
#define __SD_DGL_OPS_VAE_H__

#include <memory>
#include <optional>

#include "config.h"
#include "core/tensor.hpp"
#include "dgl/models/fwd.h"

class RNG;

namespace sd::dgl::ops {
    struct VAEOptions {
        bool circular_x = false;
        bool circular_y = false;
        std::optional<sd::TilingConfig> tiling;
    };

    Tensor<float> encode_vae(models::VAE& model, const Tensor<float>& pixels, const std::shared_ptr<RNG>& rng, bool posterior_mode, const VAEOptions& options);
    Tensor<float> decode_vae(models::VAE& model, const Tensor<float>& latent, const VAEOptions& options);
}

#endif  // __SD_DGL_OPS_VAE_H__
