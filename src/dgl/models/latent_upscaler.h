#ifndef __SD_DGL_MODELS_LATENT_UPSCALER_H__
#define __SD_DGL_MODELS_LATENT_UPSCALER_H__

#include "core/tensor.hpp"
#include "dgl/models/model_runtime.h"

namespace H3VUpsampler {
    struct H3LatentUpscalerRunner;
}
namespace LTXVUpsampler {
    struct LatentUpsamplerRunner;
}

namespace sd::dgl::models {
    class LatentUpscaler final : public Model {
        ModelRuntime runtime_;
        std::shared_ptr<LTXVUpsampler::LatentUpsamplerRunner> ltx_;
        std::shared_ptr<H3VUpsampler::H3LatentUpscalerRunner> h3_;

        bool refresh_weights() override { return runtime_.refresh_weights(); }
        bool attach_loras() override { return runtime_.attach_loras(); }
        bool build_variant(const Model& source, ModelLoader loader) override;
        bool build_runners();

    public:
        LatentUpscaler(std::shared_ptr<ModelManager> manager, std::shared_ptr<SDBackendPool> backends);
        bool init(ModelLoadParams params);
        Tensor<float> upscale(const Tensor<float>& latent, int64_t width, int64_t height);
    };
}

#endif  // __SD_DGL_MODELS_LATENT_UPSCALER_H__
