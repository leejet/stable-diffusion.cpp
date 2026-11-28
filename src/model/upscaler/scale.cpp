#include "model/upscaler/scale.hpp"

#include "model/upscaler/esrgan.hpp"
#include "model/upscaler/kandinsky_imagesr.hpp"
#include "model/upscaler/mosr.hpp"
#include "model_loader.h"

namespace sd::upscaling {
    Description describe(const ModelLoader& loader) {
        if (loader.get_metadata().count("kandinsky_imagesr_config"))
            return {Architecture::KandinskyImageSR, 4};
        const auto& tensors = loader.get_tensor_storage_map();
        if (tensors.count("gblocks.0.weight")) {
            const auto config = MoSRConfig::detect_from_weights(tensors);
            return config.is_valid() ? Description{Architecture::MoSR, config.scale} : Description{};
        }
        const auto first = tensors.find("conv_first.weight");
        const auto last  = tensors.find("conv_last.weight");
        if (first == tensors.end() || last == tensors.end() || !tensors.count("body.0.rdb1.conv1.weight") ||
            first->second.n_dims != 4 || last->second.n_dims != 4 || first->second.ne[2] != 3 || last->second.ne[3] != 3)
            return {};
        return {Architecture::ESRGAN, ESRGANConfig::detect_from_weights(tensors).scale};
    }

    bool load_weights(ModelLoader& loader, const std::string& path) {
        if (!loader.init_from_file(path))
            return false;
        loader.convert_tensors_name(loader.get_metadata().count("kandinsky_imagesr_config") ? VERSION_FLUX : VERSION_ESRGAN);
        return true;
    }

    int probe_scale(const std::string& path) {
        ModelLoader loader;
        if (path.empty() || !load_weights(loader, path))
            return 0;
        const auto description = describe(loader);
        if (description.architecture == Architecture::KandinskyImageSR && !KandinskyImageSR::probe_bundle(loader))
            return 0;
        return description.scale;
    }
}
