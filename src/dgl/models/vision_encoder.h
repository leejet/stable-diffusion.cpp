#ifndef __SD_DGL_MODELS_VISION_ENCODER_H__
#define __SD_DGL_MODELS_VISION_ENCODER_H__

#include "core/tensor.hpp"
#include "dgl/models/fwd.h"
#include "dgl/models/model_runtime.h"

namespace sd::dgl::models {
    class VisionEncoder final : public Model {
        ModelRuntime runtime_;

        bool refresh_weights() override { return runtime_.refresh_weights(); }
        bool attach_loras() override { return runtime_.attach_loras(); }
        bool build_variant(const Model& source, ModelLoader loader) override {
            return runtime_.build_variant(static_cast<const VisionEncoder&>(source).runtime_, std::move(loader));
        }

        std::shared_ptr<FrozenCLIPVisionEmbedder> clip_vision;
        bool build_runners();

    public:
        SDVersion version() const { return runtime_.version; }
        bool init(ModelLoadParams params);
        VisionEncoder(std::shared_ptr<ModelManager> manager, std::shared_ptr<SDBackendPool> backends);
        Tensor<float> get_clip_vision_output(const Tensor<float>& image, bool return_pooled, int clip_skip, bool zero_out_masked, bool zero_input);
    };
}

#endif  // __SD_DGL_MODELS_VISION_ENCODER_H__
