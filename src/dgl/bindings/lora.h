#ifndef __SD_DGL_BINDINGS_LORA_H__
#define __SD_DGL_BINDINGS_LORA_H__

#include "dgl/value.h"

namespace sd::dgl {
    struct RunState;
    struct PreparedResource;

    PreparedResource load_lora(Values& args);
    Value apply_lora(Values& args, RunState& run);
}  // namespace sd::dgl

#endif  // __SD_DGL_BINDINGS_LORA_H__
