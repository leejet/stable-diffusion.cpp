#ifndef __SD_DGL_BINDINGS_MODEL_H__
#define __SD_DGL_BINDINGS_MODEL_H__

#include "dgl/value.h"

namespace sd::dgl {
    struct RunState;
    struct PreparedResource;

    PreparedResource execute_load_upscaler(Values& args);
    Value execute_upscale(Values& args, RunState&);
    PreparedResource execute_load_detector(Values& args);
}  // namespace sd::dgl

#endif  // __SD_DGL_BINDINGS_MODEL_H__
