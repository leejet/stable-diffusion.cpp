#ifndef __SD_DGL_MODELS_MODEL_LOAD_PARAMS_H__
#define __SD_DGL_MODELS_MODEL_LOAD_PARAMS_H__

#include <string>

#include "model.h"
#include "stable-diffusion.h"

namespace sd::dgl::models {

    enum class OperatorMode { Auto,
                              On,
                              Off };

    struct ModelLoadParams {
        std::string path;
        SDVersion family = VERSION_COUNT;
        std::string family_from;
        std::string backend;
        std::string params_backend;
        std::string split_mode;
        sd_type_t wtype = SD_TYPE_COUNT;
        std::string tensor_type_rules;
        std::string model_args;
        OperatorMode flash_attn_mode  = OperatorMode::Auto;
        OperatorMode conv_direct_mode = OperatorMode::Auto;
        bool sage_attn                = false;
        float conv_scale              = 0.f;
        float linear_scale            = 0.f;
        float attn_scale              = 0.f;
    };

}  // namespace sd::dgl::models

#endif  // __SD_DGL_MODELS_MODEL_LOAD_PARAMS_H__
