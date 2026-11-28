#ifndef __SD_MODEL_UPSCALER_SCALE_HPP__
#define __SD_MODEL_UPSCALER_SCALE_HPP__

#include <string>

#include "model.h"

class ModelLoader;

namespace sd::upscaling {
    enum class Architecture { Unknown,
                              ESRGAN,
                              MoSR,
                              KandinskyImageSR };
    struct Description {
        Architecture architecture = Architecture::Unknown;
        int scale                 = 0;
    };
    Description describe(const String2TensorStorage& tensors);
    bool load_weights(ModelLoader& loader, const std::string& path);
    int probe_scale(const std::string& path);
}

#endif  // __SD_MODEL_UPSCALER_SCALE_HPP__
