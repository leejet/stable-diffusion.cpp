#ifndef __SD_DGL_OPS_PROMPT_H__
#define __SD_DGL_OPS_PROMPT_H__

#include <optional>
#include <variant>

#include "conditioning/prompt_input.h"
#include "dgl/models/fwd.h"
#include "dgl/ops/audio.h"

struct MiniMaxH3PresentationItem;
struct SDCondition;

namespace sd::dgl::ops {
    struct PromptMedia {
        std::variant<Tensor<float>, std::shared_ptr<const sd_tensor_t>, AudioReadResult> media;
        int64_t stride  = 1;
        int64_t fps_num = 24;
        int64_t fps_den = 1;
    };

    struct PromptEncodingOptions {
        std::string text;
        int width     = 512;
        int height    = 512;
        int clip_skip = -1;
        bool raw      = false;
        std::optional<std::string> system;
        std::optional<std::vector<PromptMedia>> media;
        // An empty tensor means the image has no host binding and should be read from its path.
        sd::PromptImageLoader image_bindings;
    };

    MiniMaxH3PresentationItem prepare_prompt_media(PromptMedia input);
    SDCondition encode_prompt(models::TextEncoder& model, PromptEncodingOptions options);
}

#endif  // __SD_DGL_OPS_PROMPT_H__
