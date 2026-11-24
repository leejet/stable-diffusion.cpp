#ifndef __SD_DGL_MODELS_TEXT_ENCODER_H__
#define __SD_DGL_MODELS_TEXT_ENCODER_H__

#include "core/tensor.hpp"
#include "dgl/models/fwd.h"
#include "dgl/models/model_runtime.h"
#include "tokenizers/tokenizer_config.h"

class ConditioningCache;
struct ConditionerParams;
struct SDCondition;

namespace sd::dgl::models {
    struct TextEncoderLoadParams {
        ModelLoadParams model;
        TokenizerConfig tokenizer;
        std::string vision_path;
        std::string projection_path;
        std::string connectors_path;
        std::string t5_path;
        std::string clip_l_path;
        std::string clip_g_path;
    };

    class TextEncoder final : public Model {
        ModelRuntime runtime_;
        TokenizerConfig tokenizer_;

        bool refresh_weights() override { return runtime_.refresh_weights(); }
        bool attach_loras() override { return runtime_.attach_loras(); }
        bool build_variant(const Model& source, ModelLoader loader) override {
            const auto& base = static_cast<const TextEncoder&>(source);
            tokenizer_       = base.tokenizer_;
            return runtime_.build_variant(base.runtime_, std::move(loader));
        }

        std::unique_ptr<ConditioningCache> conditioning_cache_;

        bool build_runners();
        void invalidate_cached_state() override;

    public:
        SDVersion version() const { return runtime_.version; }
        std::shared_ptr<Conditioner> cond_stage_model;

        TextEncoder(std::shared_ptr<ModelManager> manager, std::shared_ptr<SDBackendPool> backends);
        ~TextEncoder() override;
        bool init(TextEncoderLoadParams params);
        void set_conditioning_cache_size(size_t capacity);
        bool supports_prompt_images() const;
        size_t max_reference_images() const;
        bool supports_multimodal_prompt() const;
        bool supports_textual_inversion() const;
        SDCondition get_learned_condition(const ConditionerParams& params);
        PhotoMakerPrompt encode_photomaker_prompt(ConditionerParams params, const Tensor<float>& tokens);
    };
}

#endif  // __SD_DGL_MODELS_TEXT_ENCODER_H__
