#ifndef __SD_DGL_MODELS_AUDIO_ENCODER_H__
#define __SD_DGL_MODELS_AUDIO_ENCODER_H__

#include "core/media.h"
#include "core/tensor.hpp"
#include "dgl/models/fwd.h"
#include "dgl/models/model_runtime.h"

namespace sd::dgl::models {
    class AudioEncoder final : public Model {
        ModelRuntime runtime_;

        bool refresh_weights() override { return runtime_.refresh_weights(); }
        bool attach_loras() override { return runtime_.attach_loras(); }
        bool build_variant(const Model& source, ModelLoader loader) override {
            return runtime_.build_variant(static_cast<const AudioEncoder&>(source).runtime_, std::move(loader));
        }

        std::shared_ptr<Wav2Vec2::Wav2Vec2ModelRunner> audio_encoder;
        bool build_runners();

    public:
        SDVersion version() const { return runtime_.version; }
        bool init(ModelLoadParams params);
        AudioEncoder(std::shared_ptr<ModelManager> manager, std::shared_ptr<SDBackendPool> backends);
        Tensor<float> get_audio_embedding(const sd::Audio& audio);
    };
}

#endif  // __SD_DGL_MODELS_AUDIO_ENCODER_H__
