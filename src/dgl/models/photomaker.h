#ifndef __SD_DGL_MODELS_PHOTOMAKER_H__
#define __SD_DGL_MODELS_PHOTOMAKER_H__

#include "core/tensor.hpp"
#include "dgl/models/fwd.h"
#include "dgl/models/model_runtime.h"

struct PhotoMakerIDEncoder;

namespace sd::dgl::models {
    class PhotoMaker final : public Model {
        ModelRuntime runtime_;
        std::shared_ptr<PhotoMakerIDEncoder> runner_;

        bool refresh_weights() override { return runtime_.refresh_weights(); }
        bool attach_loras() override { return runtime_.attach_loras(); }
        bool build_variant(const Model& source, ModelLoader loader) override {
            return runtime_.build_variant(static_cast<const PhotoMaker&>(source).runtime_, std::move(loader));
        }

        bool build_runners();

    public:
        SDVersion version() const { return runtime_.version; }
        PhotoMaker(std::shared_ptr<ModelManager> manager, std::shared_ptr<SDBackendPool> backends);
        bool init(ModelLoadParams params);
        Tensor<float> project_images(const std::vector<Tensor<float>>& images, const Tensor<float>* embedding);
        Tensor<float> fuse_prompt(const Tensor<float>& prompt, const Tensor<float>& tokens, const Tensor<uint8_t>& mask);
    };
}

#endif  // __SD_DGL_MODELS_PHOTOMAKER_H__
