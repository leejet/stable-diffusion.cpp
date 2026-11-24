#include "dgl/models/control.h"

#include "conditioning/conditioner.hpp"
#include "dgl/models/component_builders.h"
#include "dgl/models/component_sources.h"
#include "dgl/models/dit.h"
#include "model/diffusion/control.hpp"
#include "model/diffusion/z_image.hpp"

namespace sd::dgl::models {
    ControlNet::ControlNet(std::shared_ptr<ModelManager> manager, std::shared_ptr<SDBackendPool> backends)
        : Model(std::move(manager)),
          runtime_(*this, std::move(backends), ModelComponent::ControlNet, SDBackendModule::CONTROL_NET, [this] { return build_runners(); }) {}

    bool ControlNet::init(ModelLoadParams params) {
        return runtime_.init(std::move(params), load_component_catalog<ModelComponent::ControlNet>);
    }

    bool ControlNet::build_variant(const Model& source, ModelLoader loader) {
        const auto& base = static_cast<const ControlNet&>(source);
        return runtime_.build_variant(base.runtime_, std::move(loader));
    }

    bool ControlNet::build_runners() {
        const auto& weights = model_weights->loader().get_tensor_storage_map();
        if (sd_version_is_z_image(version())) {
            z_image_control_ = std::make_shared<ZImage::ControlNetRunner>(runtime_.backend_for(), weights, model_weights);
            z_image_control_->set_scale_overrides(runtime_.params.linear_scale,
                                                  runtime_.params.attn_scale);
            runtime_.configure_runner(*z_image_control_);
            return runtime_.register_runner_params(runtime_.component, z_image_control_);
        }
        if (!sd_version_is_unet(version()) || ZImage::ControlNet::has_weights(weights))
            throw std::runtime_error("ControlNet weights do not match the requested family");
        if (!component_builders::build_control_net_runner(runtime_.model_build_context(), control_net) || !control_net) {
            return false;
        }
        runtime_.configure_runner(*control_net, true);
        return runtime_.register_runner_params(runtime_.component, control_net);
    }

    void ControlNet::validate_hint(const Tensor<float>& hint, sd_control_type_t type) const {
        if (type == SD_CONTROL_TYPE_COUNT)
            throw std::runtime_error("invalid ControlNet type");
        const auto channels = z_image_control_ ? ZImage::CONTROL_IN_CHANNELS : 3;
        if (hint.empty() || hint.dim() != 4 || hint.shape()[2] != channels)
            throw std::runtime_error(z_image_control_ ? "Z-Image ControlNet expects a 33-channel latent"
                                                      : "ControlNet expects an RGB float tensor in [width, height, channels, batch] order");
        if (z_image_control_) {
            if (type != SD_CONTROL_TYPE_NONE)
                throw std::runtime_error("Z-Image ControlNet uses the type encoded in its hint");
        } else if (!control_net->control_index(type)) {
            throw std::runtime_error("invalid type for this ControlNet");
        }
    }

    void ControlNet::validate_target(const DiT& target) const {
        if (target.version() != version())
            throw std::runtime_error("ControlNet family differs from the diffusion model");
        if (z_image_control_) {
            const auto* model = dynamic_cast<const ZImage::ZImageRunner*>(target.diffusion_model.get());
            if (!model || !z_image_control_->model.compatible(model->config))
                throw std::runtime_error("Z-Image ControlNet architecture differs from the diffusion model");
            if (z_image_control_->compute_backend() != model->compute_backend())
                throw std::runtime_error("Z-Image ControlNet and DiT require the same compute backend");
        }
    }

    void ControlNet::apply_circular_axes(bool x, bool y) {
        circular_x = x;
        circular_y = y;
        if (x || y)
            LOG_DEBUG("Using circular padding for convolutions (x=%s, y=%s)", x ? "true" : "false", y ? "true" : "false");
        if (control_net)
            control_net->set_circular_axes(x, y);
    }

    std::vector<Tensor<float>> ControlNet::compute_sample_controls(const ControlBinding& binding,
                                                                   const Tensor<float>& input,
                                                                   const Tensor<float>& timesteps,
                                                                   const SDCondition& condition) {
        const auto& shape = binding.hint->shape();
        if (input.dim() != 4 || shape[0] != input.shape()[0] * 8 || shape[1] != input.shape()[1] * 8 ||
            (shape[3] != 1 && shape[3] != input.shape()[3]))
            throw std::runtime_error("ControlNet hint size or batch does not match the diffusion latent");
        const auto cache_key = "controlnet.guided_hint." + std::to_string(reinterpret_cast<uintptr_t>(binding.hint.get())) +
                               "." + std::to_string(circular_x) + "." + std::to_string(circular_y);
        auto control_result  = control_net->compute(n_threads,
                                                    input,
                                                    *binding.hint,
                                                    timesteps,
                                                    condition.c_crossattn,
                                                    condition.c_vector,
                                                    cache_key,
                                                    binding.type);
        if (!control_result || control_result->empty())
            throw std::runtime_error("ControlNet did not return residuals");
        return std::move(*control_result);
    }

    ZImageControlInput ControlNet::z_image_input(const ControlBinding& binding, const Tensor<float>& input) const {
        const auto& shape = binding.hint->shape();
        if (input.dim() != 4 || shape[0] != input.shape()[0] || shape[1] != input.shape()[1] || shape[3] != input.shape()[3])
            throw std::runtime_error("Z-Image ControlNet hint size or batch does not match the diffusion latent");
        return {z_image_control_.get(), binding.hint.get(), binding.strength};
    }

}  // namespace sd::dgl::models
