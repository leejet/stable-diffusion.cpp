#ifndef __SD_CONDITIONING_TEXTUAL_INVERSION_H__
#define __SD_CONDITIONING_TEXTUAL_INVERSION_H__

#include "core/tensor.hpp"
#include "model_loader.h"

namespace sd {
    bool read_textual_inversion(ModelLoader& loader, ggml_context* ctx, int64_t hidden_size, int64_t hidden_size2, ggml_type type, ggml_type type2, ggml_tensor*& first, ggml_tensor*& second);

    struct TextualInversion {
        ModelLoader::Dependencies sources;
        ModelLoader::Dependencies dependencies;
        Tensor<float> first;
        Tensor<float> second;
        std::map<std::string, std::pair<int, int>> token_ranges;

        void load(int64_t hidden_size, int64_t hidden_size2);
    };
}

#endif  // __SD_CONDITIONING_TEXTUAL_INVERSION_H__
