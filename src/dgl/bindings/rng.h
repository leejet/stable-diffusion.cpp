#ifndef __SD_DGL_BINDINGS_RNG_H__
#define __SD_DGL_BINDINGS_RNG_H__

#include "dgl/value.h"

namespace sd::dgl {
    struct RunState;

    Value execute_make_rng(Values& args, RunState&);
}  // namespace sd::dgl

#endif  // __SD_DGL_BINDINGS_RNG_H__
