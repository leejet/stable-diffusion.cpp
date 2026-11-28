#ifndef __SD_DGL_BINDINGS_SEEDVR2_H__
#define __SD_DGL_BINDINGS_SEEDVR2_H__

#include "dgl/value.h"

namespace sd::dgl {
    struct RunState;

    Value seedvr2_positive_embedding(Values& args, RunState&);
}  // namespace sd::dgl

#endif  // __SD_DGL_BINDINGS_SEEDVR2_H__
