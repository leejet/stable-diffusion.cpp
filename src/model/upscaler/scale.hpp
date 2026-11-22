#ifndef __SD_MODEL_UPSCALER_SCALE_HPP__
#define __SD_MODEL_UPSCALER_SCALE_HPP__

#include <string>

#include "model.h"

namespace sd::upscaling {
    enum class Architecture { Unknown,
                              ESRGAN,
                              MoSR };
    struct Description {
        Architecture architecture = Architecture::Unknown;
        int scale                 = 0;
    };
    Description describe(const String2TensorStorage& tensors);
    int probe_scale(const std::string& path);
}

#endif  // __SD_MODEL_UPSCALER_SCALE_HPP__
