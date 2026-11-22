#include "dgl/models/relay_vsr.h"

#include "dgl/models/vae.h"
#include "json.hpp"
#include "model/diffusion/relay_vsr.hpp"
#include "model/upscaler/flash_vsr.hpp"
#include "model_loader.h"

namespace sd::dgl::models {
    std::shared_ptr<::VAE> build_relay_vsr_vae(const model_builders::Context& ctx, const ModelLoader& loader) {
        const auto& metadata = loader.get_metadata();
        const auto config    = metadata.find("config");
        if (config == metadata.end())
            throw std::runtime_error("RelayVSR Flash weights require upscale_factor metadata");
        const auto scale = nlohmann::json::parse(config->second).at("upscale_factor").get<int>();
        return std::make_shared<RelayVSR::FlashVSRRunner>(ctx.backends.runtime_backend(SDBackendModule::VAE),
                                                          ctx.tensor_storage_map, scale, ctx.weight_manager);
    }

}
