#ifndef __SD_DGL_BINDINGS_POSTPROCESS_H__
#define __SD_DGL_BINDINGS_POSTPROCESS_H__

#include "dgl/value.h"

namespace sd::dgl {
    struct RunState;

    Value execute_canny(Values& args, RunState&);
}  // namespace sd::dgl

#endif  // __SD_DGL_BINDINGS_POSTPROCESS_H__
