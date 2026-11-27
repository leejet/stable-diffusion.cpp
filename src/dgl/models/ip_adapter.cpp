#include "dgl/models/ip_adapter.h"

#include <algorithm>

#include "core/util.h"
#include "dgl/models/component_sources.h"
#include "dgl/models/dit.h"
#include "model/adapter/ip_adapter.hpp"
#include "model/diffusion/model.hpp"

namespace sd::dgl::models {
    IPAdapter::IPAdapter(std::shared_ptr<ModelManager> manager, std::shared_ptr<SDBackendPool> backends)
        : Model(std::move(manager)),
          runtime_(*this, std::move(backends), ModelComponent::IPAdapter, SDBackendModule::DIFFUSION, [this] { return build_runners(); }) {}

    bool IPAdapter::init(ModelLoadParams params) {
        return runtime_.init(std::move(params), load_component_catalog<ModelComponent::IPAdapter>);
    }

    bool IPAdapter::build_runners() {
        ip_adapter = std::make_shared<::IPAdapter::IPAdapterRunner>(runtime_.backend_for(), model_weights->loader().get_tensor_storage_map(), "ip_adapter", model_weights, true);
        ip_adapter->set_flash_attention_enabled(false);
        return runtime_.register_runner_params(runtime_.component, ip_adapter);
    }

    Tensor<float> IPAdapter::project_image(const Tensor<float>& embedding) {
        const auto width    = ip_adapter->is_plus ? ip_adapter->resampler.embed_dim : ip_adapter->image_proj.clip_dim;
        const auto min_rank = ip_adapter->is_plus ? 2 : 1;
        if (embedding.empty() || embedding.dim() < min_rank || embedding.dim() > min_rank + 1 ||
            embedding.shape()[0] != width) {
            throw std::runtime_error(ip_adapter->is_plus ? "IP-Adapter Plus requires CLIP Vision hidden states"
                                                         : "IP-Adapter Classic requires pooled CLIP Vision embeddings");
        }
        return ip_adapter->compute(n_threads, embedding);
    }

    void IPAdapter::validate_tokens(const Tensor<float>& tokens) const {
        if (tokens.empty() || tokens.dim() < 2 || tokens.dim() > 3 ||
            tokens.shape()[0] != ip_adapter->token_width()) {
            throw std::runtime_error("IP-Adapter tokens must have shape [projection width, tokens, optional batch]");
        }
    }

    void IPAdapter::validate_target(const DiT& target) const {
        if (!sd_version_is_unet(target.version()))
            throw std::runtime_error("IP-Adapter requires a UNet predictor");
        if (!(sd_version_is_sd1(version()) && sd_version_is_sd1(target.version())) &&
            !(sd_version_is_sdxl(version()) && sd_version_is_sdxl(target.version()))) {
            throw std::runtime_error("IP-Adapter and UNet must use the same SD 1.x or SDXL family");
        }
        std::map<std::string, ggml_tensor*> weights;
        target.diffusion_model->get_param_tensors(weights);
        for (const auto& [layer, projection] : ip_adapter->attention) {
            const auto query = weights.find(layer + ".to_q.weight");
            if (query == weights.end() || query->second->ne[1] != projection.width)
                throw std::runtime_error("IP-Adapter is incompatible with UNet attention layer '" + layer + "'");
        }
    }
}
