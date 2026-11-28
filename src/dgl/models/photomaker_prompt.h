#ifndef __SD_DGL_MODELS_PHOTOMAKER_PROMPT_H__
#define __SD_DGL_MODELS_PHOTOMAKER_PROMPT_H__

#include "conditioning/conditioner.hpp"

namespace sd::dgl::models {
    struct PhotoMakerPrompt {
        SDCondition plain;
        SDCondition condition;
        sd::Tensor<uint8_t> mask;
    };
}

#endif  // __SD_DGL_MODELS_PHOTOMAKER_PROMPT_H__
