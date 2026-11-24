#include "model/upscaler/scale.hpp"

#include "model/upscaler/esrgan.hpp"
#include "model/upscaler/mosr.hpp"
#include "model_loader.h"

namespace sd::upscaling {
    Description describe(const String2TensorStorage& tensors) {
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

    int probe_scale(const std::string& path) {
        ModelLoader loader;
        if (path.empty() || !loader.init_from_file_and_convert_name(path, "", VERSION_ESRGAN))
            return 0;
        return describe(loader.get_tensor_storage_map()).scale;
    }
}
