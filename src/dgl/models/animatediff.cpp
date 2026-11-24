#include "dgl/models/animatediff.h"

#include <climits>

#include "core/util.h"
#include "dgl/models/component_sources.h"
#include "dgl/models/dit.h"
#include "model/diffusion/unet.hpp"

namespace sd::dgl::models {
    using MotionRunner = ::AnimateDiff::MotionRunner;

    AnimateDiff::AnimateDiff(std::shared_ptr<ModelManager> manager, std::shared_ptr<SDBackendPool> backends)
        : Model(std::move(manager)),
          runtime_(*this, std::move(backends), ModelComponent::AnimateDiff, SDBackendModule::DIFFUSION, [this] { return build_runners(); }) {}

    bool AnimateDiff::init(ModelLoadParams params) {
        return runtime_.init(std::move(params), load_component_catalog<ModelComponent::AnimateDiff>);
    }

    bool AnimateDiff::build_variant(const Model& source, ModelLoader loader) {
        return runtime_.build_variant(static_cast<const AnimateDiff&>(source).runtime_, std::move(loader));
    }

    bool AnimateDiff::build_runners() {
        if (runtime_.version != VERSION_SD1)
            throw std::runtime_error("AnimateDiff requires family = \"sd1\"");

        const auto& weights      = model_weights->loader().get_tensor_storage_map();
        const std::string prefix = std::string(MotionRunner::prefix) + ".";
        const auto first_pe      = weights.find(prefix + "down_blocks.0.motion_modules.0.temporal_transformer.transformer_blocks.0.attention_blocks.0.pos_encoder.pe");
        if (first_pe == weights.end() || first_pe->second.ne[1] <= 0 || first_pe->second.ne[1] > INT_MAX)
            throw std::runtime_error("AnimateDiff checkpoint has no valid positional encoding");

        ::AnimateDiff::MotionModuleConfig config;
        config.max_frames       = int(first_pe->second.ne[1]);
        config.enable_mid_block = weights.count(prefix + "mid_block.motion_modules.0.temporal_transformer.proj_in.weight") != 0;
        for (const auto& [name, tensor] : weights) {
            if (starts_with(name, prefix) && ends_with(name, ".pos_encoder.pe") &&
                (tensor.ne[1] != config.max_frames || tensor.ne[2] != 1))
                throw std::runtime_error("inconsistent AnimateDiff positional encoding at '" + name + "'");
        }

        runner_ = std::make_shared<MotionRunner>(runtime_.backend_for(), weights, config, model_weights);
        runner_->set_scale_overrides(runtime_.params.linear_scale,
                                     runtime_.params.attn_scale);
        runner_->set_sage_attention_enabled(runtime_.params.sage_attn);
        return runtime_.register_runner_params(runtime_.component, runner_);
    }

    void AnimateDiff::validate_target(const DiT& target) const {
        const auto* unet = dynamic_cast<const UNetModelRunner*>(target.diffusion_model.get());
        if (!unet || !sd_version_is_sd1(target.version()) || unet->config.tiny_unet ||
            unet->config.model_channels != 320 || unet->config.num_res_blocks != 2 ||
            unet->config.channel_mult != std::vector<int>{1, 2, 4, 4})
            throw std::runtime_error("AnimateDiff requires a compatible SD 1.5 UNet");
        if (runner_->compute_backend() != unet->compute_backend())
            throw std::runtime_error("AnimateDiff and the UNet require the same compute backend");
    }

    void AnimateDiff::validate_latent(const Tensor<float>& latent) const {
        if (latent.empty() || latent.dim() != 4)
            throw std::runtime_error("AnimateDiff expects a latent with shape [width, height, channels, frames]");
        if (latent.shape()[3] > runner_->max_frames())
            throw std::runtime_error("AnimateDiff checkpoint supports up to " + std::to_string(runner_->max_frames()) + " frames");
    }

    ggml_tensor* AnimateDiff::apply(GGMLRunnerContext& ctx, UNetBlockStage stage, int level, int block, ggml_tensor* activation, int frames) const {
        const auto block_name = stage == UNetBlockStage::Middle ? std::string("mid_block")
                                                                : (stage == UNetBlockStage::Down ? "down_blocks." : "up_blocks.") + std::to_string(level);
        return runner_->forward(ctx, block_name + ".motion_modules." + std::to_string(block), activation, frames);
    }

    size_t AnimateDiff::extra_graph_size() const {
        return runner_->extra_graph_size();
    }
}
