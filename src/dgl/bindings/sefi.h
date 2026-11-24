#ifndef __SD_DGL_BINDINGS_SEFI_H__
#define __SD_DGL_BINDINGS_SEFI_H__

#include "dgl/value.h"

namespace sd::dgl {
    struct RunState;

    Value sefi_sigmas(Values& args, RunState&);
    Value sample_sefi_tensor(Values& args, RunState& state);
}  // namespace sd::dgl

#endif  // __SD_DGL_BINDINGS_SEFI_H__
