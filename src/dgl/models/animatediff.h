#ifndef __SD_DGL_MODELS_ANIMATEDIFF_H__
#define __SD_DGL_MODELS_ANIMATEDIFF_H__

#include "dgl/models/model_runtime.h"
#include "model/diffusion/model.hpp"

namespace AnimateDiff {
    class MotionRunner;
}

namespace sd::dgl::models {
    class DiT;

    class AnimateDiff final : public Model, public UNetBlockProcessor {
        ModelRuntime runtime_;
        std::shared_ptr<::AnimateDiff::MotionRunner> runner_;

        bool refresh_weights() override { return runtime_.refresh_weights(); }
        bool attach_loras() override { return runtime_.attach_loras(); }
        bool build_variant(const Model& source, ModelLoader loader) override;
        bool build_runners();

    public:
        AnimateDiff(std::shared_ptr<ModelManager> manager, std::shared_ptr<SDBackendPool> backends);
        bool init(ModelLoadParams params);
        void validate_target(const DiT& target) const;
        void validate_latent(const Tensor<float>& latent) const;
        ggml_tensor* apply(GGMLRunnerContext& ctx, UNetBlockStage stage, int level, int block, ggml_tensor* activation, int frames) const override;
        size_t extra_graph_size() const override;
    };
}

#endif  // __SD_DGL_MODELS_ANIMATEDIFF_H__
