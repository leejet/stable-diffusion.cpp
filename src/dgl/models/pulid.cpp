#include "dgl/models/pulid.h"

#include "dgl/models/component_sources.h"
#include "dgl/models/dit.h"
#include "model/adapter/pulid.hpp"
#include "model/diffusion/flux.hpp"

namespace sd::dgl::models {
    PuLID::PuLID(std::shared_ptr<ModelManager> manager, std::shared_ptr<SDBackendPool> backends)
        : Model(std::move(manager)),
          runtime_(*this, std::move(backends), ModelComponent::PuLID, SDBackendModule::DIFFUSION, [this] { return build_runners(); }) {}

    bool PuLID::init(ModelLoadParams params) {
        return runtime_.init(std::move(params), load_component_catalog<ModelComponent::PuLID>);
    }

    bool PuLID::build_variant(const Model& source, ModelLoader loader) {
        return runtime_.build_variant(static_cast<const PuLID&>(source).runtime_, std::move(loader));
    }

    bool PuLID::build_runners() {
        if (runtime_.version != VERSION_FLUX)
            throw std::runtime_error("PuLID requires family = \"flux\"");
        runner_ = std::make_shared<PuLIDRunner>(runtime_.backend_for(), model_weights->loader().get_tensor_storage_map(), model_weights);
        runner_->set_scale_overrides(runtime_.params.linear_scale,
                                     runtime_.params.attn_scale);
        return runtime_.register_runner_params(runtime_.component, runner_);
    }

    void PuLID::validate_target(const DiT& target) const {
        const auto* flux = dynamic_cast<const Flux::FluxRunner*>(target.diffusion_model.get());
        if (target.version() != VERSION_FLUX || !flux ||
            flux->config.depth != PuLIDRunner::double_blocks || flux->config.depth_single_blocks != PuLIDRunner::single_blocks ||
            flux->config.hidden_size != PuLIDPerceiverAttentionCA::DEFAULT_DIM)
            throw std::runtime_error("PuLID requires a matching FLUX.1 denoiser");
        if (runner_->compute_backend() != flux->compute_backend())
            throw std::runtime_error("PuLID and the denoiser require the same compute backend");
    }

    void PuLID::validate_embedding(const Tensor<float>& embedding) const {
        if (embedding.empty() || embedding.dim() != 2 || embedding.shape()[0] != PuLIDPerceiverAttentionCA::DEFAULT_KV_DIM)
            throw std::runtime_error("PuLID identity must have shape [2048, tokens]");
    }

    std::vector<Tensor<float>> PuLID::compute_kv(const Tensor<float>& embedding) {
        return runner_->compute_kv(n_threads, embedding);
    }

    size_t PuLID::extra_graph_size() const {
        return runner_->extra_graph_size();
    }

    ggml_tensor* PuLID::attend(GGMLRunnerContext& ctx, size_t layer, const Tensor<float>& kv, ggml_tensor* image) const {
        return runner_->attend(ctx, layer, kv, image);
    }

}
