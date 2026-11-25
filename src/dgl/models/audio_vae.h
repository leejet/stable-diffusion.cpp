#ifndef __SD_DGL_MODELS_AUDIO_VAE_H__
#define __SD_DGL_MODELS_AUDIO_VAE_H__

#include "core/tensor.hpp"
#include "dgl/models/fwd.h"
#include "dgl/models/model_runtime.h"

namespace sd::dgl::models {
    class AudioVAE final : public Model {
        ModelRuntime runtime_;

        bool refresh_weights() override { return runtime_.refresh_weights(); }
        bool attach_loras() override { return runtime_.attach_loras(); }
        bool build_variant(const Model& source, ModelLoader loader) override {
            return runtime_.build_variant(static_cast<const AudioVAE&>(source).runtime_, std::move(loader));
        }

        bool build_runners();

    public:
        SDVersion version() const { return runtime_.version; }
        bool init(ModelLoadParams params);
        std::shared_ptr<AudioVAERunner> audio_vae_model;

        AudioVAE(std::shared_ptr<ModelManager> manager, std::shared_ptr<SDBackendPool> backends);
        std::vector<int64_t> latent_shape(int frames, int fps) const;
        Tensor<float> decode_ltx_audio_latent(const Tensor<float>& latent);
    };
}

#endif  // __SD_DGL_MODELS_AUDIO_VAE_H__
