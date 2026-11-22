#include "dgl/model_value.h"

#include <type_traits>

#include "context.h"
#include "dgl/models/audio_encoder.h"
#include "dgl/models/audio_vae.h"
#include "dgl/models/control.h"
#include "dgl/models/dit.h"
#include "dgl/models/ip_adapter.h"
#include "dgl/models/photomaker.h"
#include "dgl/models/pulid.h"
#include "dgl/models/text_encoder.h"
#include "dgl/models/vae.h"
#include "dgl/models/vision_encoder.h"
#include "models/animatediff.h"
#include "models/detail_detector.h"
#include "models/diffusion_model.h"
#include "models/latent_upscaler.h"
#include "models/upscaler.h"

namespace sd::dgl {
    template <class F>
    static decltype(auto) visit_model(const Value& value, F&& visit) {
        switch (value.type().kind) {
            case Kind::Denoiser:
                return visit(value.as<models::DiffusionModel>()->dit);
            case Kind::TextEncoder:
                return visit(value.as<models::TextEncoder>());
            case Kind::VAE:
                return visit(value.as<models::VAE>());
            case Kind::ControlNet:
                return visit(value.as<models::ControlNet>());
            case Kind::VisionEncoder:
                return visit(value.as<models::VisionEncoder>());
            case Kind::AudioEncoder:
                return visit(value.as<models::AudioEncoder>());
            case Kind::AudioVAE:
                return visit(value.as<models::AudioVAE>());
            case Kind::IPAdapter:
                return visit(value.as<models::IPAdapter>());
            case Kind::PhotoMaker:
                return visit(value.as<models::PhotoMaker>());
            case Kind::PuLID:
                return visit(value.as<models::PuLID>());
            case Kind::AnimateDiff:
                return visit(value.as<models::AnimateDiff>());
            case Kind::LatentUpscaler:
                return visit(value.as<models::LatentUpscaler>());
            case Kind::Upscaler:
                return visit(value.as<models::ImageUpscaler>());
            case Kind::Detector:
                return visit(value.as<models::DetailDetector>());
            default:
                throw Error(SD_STATUS_TYPE_MISMATCH, "expected a model, got " + type_name(value.type()));
        }
    }

    models::Model& model_from_value(const Value& value) {
        return visit_model(value, [](const auto& model) -> models::Model& { return *model; });
    }

    void set_conditioning_cache_size(const Value& value, size_t capacity) {
        if (value.type().kind == Kind::TextEncoder)
            value.as<models::TextEncoder>()->set_conditioning_cache_size(capacity);
    }

    Value build_model_variant(Context& context, const Value& base, const std::shared_ptr<const models::LoraConfig>& loras) {
        return visit_model(base, [&](const auto& model) {
            using T      = typename std::decay_t<decltype(model)>::element_type;
            auto variant = context.make_model<T>(base.model_source->key);
            if (!variant->init_variant(*model, loras))
                throw std::runtime_error("cannot build LoRA variant of " + type_name(base.type()));
            if constexpr (std::is_same_v<T, models::DiT>)
                return Value::object(std::make_shared<models::DiffusionModel>(models::DiffusionModel{std::move(variant)}));
            else
                return Value::object(std::move(variant));
        });
    }

    Value with_animatediff(const Value& base, const std::shared_ptr<models::AnimateDiff>& motion) {
        const auto base_model = base.as<models::DiffusionModel>();
        motion->validate_target(*base_model->dit);
        auto model         = std::make_shared<models::DiffusionModel>(models::DiffusionModel{
            base_model->dit, motion, base_model->source ? base_model->source : base_model});
        auto value         = Value::object(std::move(model));
        value.model_source = base.model_source;
        value.loras        = base.loras;
        return value;
    }

    Value bind_model_variant(const Value& base, Value variant) {
        const auto* base_model = base.get_if<models::DiffusionModel>();
        return base_model && base_model->motion ? with_animatediff(variant, base_model->motion) : std::move(variant);
    }
}
