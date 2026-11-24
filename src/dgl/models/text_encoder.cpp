#include "dgl/models/text_encoder.h"

#include <algorithm>

#include "conditioning/conditioner.hpp"
#include "conditioning/conditioning_cache.h"
#include "conditioning/photomaker_prompt.h"
#include "core/util.h"
#include "dgl/models/component_builders.h"
#include "dgl/models/component_sources.h"
#include "dgl/models/photomaker_prompt.h"

namespace sd::dgl::models {
    TextEncoder::TextEncoder(std::shared_ptr<ModelManager> manager, std::shared_ptr<SDBackendPool> backends)
        : Model(std::move(manager)),
          runtime_(*this, std::move(backends), ModelComponent::Conditioner, SDBackendModule::TE, [this] { return build_runners(); }),
          conditioning_cache_(std::make_unique<ConditioningCache>()) {}

    TextEncoder::~TextEncoder() = default;

    bool TextEncoder::init(TextEncoderLoadParams params) {
        tokenizer_ = std::move(params.tokenizer);
        WeightSources sources{
            {params.connectors_path, "embeddings_connectors."},
            {params.vision_path, "text_encoders.llm.visual."},
            {params.projection_path, ""},
            {params.t5_path, "text_encoders.t5xxl.transformer."},
            {params.clip_l_path, "clip_l."},
            {params.clip_g_path, "clip_g."},
        };
        return runtime_.init(std::move(params.model),
                             [&sources](ModelLoader& loader, const ModelLoadParams& model, SDVersion& family, FamilyProbeCache& family_probes) {
                                 return init_component_loader(loader, model, ModelComponent::Conditioner, family, sources, &family_probes);
                             });
    }

    bool TextEncoder::build_runners() {
        auto context              = runtime_.model_build_context();
        context.build_conditioner = true;
        context.build_diffusion   = false;
        context.build_auxiliary   = false;
        sd::model_builders::CoreRunners runners;
        if (!component_builders::build_family_runners(context, runners, tokenizer_) || !runners.conditioner) {
            return false;
        }
        cond_stage_model = std::move(runners.conditioner);
        cond_stage_model->set_flash_attention_enabled(runtime_.flash_attention_for());
        return runtime_.register_runner_params(runtime_.component, cond_stage_model);
    }

    void TextEncoder::invalidate_cached_state() {
        conditioning_cache_->clear();
    }

    void TextEncoder::set_conditioning_cache_size(size_t capacity) {
        conditioning_cache_->set_capacity(capacity);
    }

    SDCondition TextEncoder::get_learned_condition(const ConditionerParams& params) {
        if (params.raw && params.system_prompt)
            throw std::runtime_error("raw prompt encoding cannot be combined with system");
        if (params.raw && !cond_stage_model->supports_prompt_formatting())
            throw std::runtime_error("this text encoder does not accept raw prompt encoding");
        if (params.system_prompt && !cond_stage_model->supports_prompt_formatting())
            throw std::runtime_error("this text encoder does not accept a system prompt");
        auto resolved      = params;
        resolved.clip_skip = cond_stage_model->resolve_clip_skip(params.clip_skip);
        // Wan attends all UMT5 slots, so padding must contain zero embeddings.
        if (sd_version_is_wan(version()) || sd_version_is_lingbot_video(version()))
            resolved.zero_out_masked = true;
        return conditioning_cache_->get(*cond_stage_model, n_threads, resolved);
    }

    bool TextEncoder::supports_textual_inversion() const {
        return dynamic_cast<FrozenCLIPEmbedderWithCustomWords*>(cond_stage_model.get()) != nullptr;
    }

    bool TextEncoder::supports_prompt_images() const {
        return cond_stage_model->supports_prompt_images();
    }

    size_t TextEncoder::max_reference_images() const {
        return cond_stage_model->max_reference_images();
    }

    bool TextEncoder::supports_multimodal_prompt() const {
        return cond_stage_model->supports_multimodal_prompt();
    }

    PhotoMakerPrompt TextEncoder::encode_photomaker_prompt(ConditionerParams params, const Tensor<float>& tokens) {
        if (tokens.empty() || tokens.dim() != 2 || tokens.shape()[0] != 2048 || tokens.shape()[1] > 75)
            throw std::runtime_error("PhotoMaker tokens must have shape [2048, slots] with up to 75 slots");
        Model::ExecutionScope scope(*this);
        scope.require_ready();
        auto* clip = dynamic_cast<FrozenCLIPEmbedderWithCustomWords*>(cond_stage_model.get());
        if (!sd_version_is_sdxl(version()) || !clip)
            throw std::runtime_error("PhotoMaker requires an SDXL CLIP text encoder");
        params.clip_skip      = clip->resolve_clip_skip(params.clip_skip);
        auto [encoded, slots] = sd::photomaker::encode_prompt(*clip, n_threads, params, "img", int(tokens.shape()[1]));
        if (encoded.empty() || std::count(slots.begin(), slots.end(), true) != tokens.shape()[1])
            throw std::runtime_error("PhotoMaker prompt requires an 'img' trigger after a class word and enough slots for all identity tokens");
        std::string plain_text;
        if (!sd::photomaker::remove_trigger(*clip, params.text, "img", plain_text))
            throw std::runtime_error("cannot remove the PhotoMaker prompt trigger");
        params.text = std::move(plain_text);
        auto plain  = get_learned_condition(params);
        if (plain.empty())
            throw std::runtime_error("PhotoMaker plain prompt encoding failed");
        auto mask = sd::Tensor<uint8_t>::from_vector(std::vector<uint8_t>(slots.begin(), slots.end()));
        return {std::move(plain), std::move(encoded), std::move(mask)};
    }
}
