#ifndef __SD_DGL_BINDINGS_RELAY_VSR_H__
#define __SD_DGL_BINDINGS_RELAY_VSR_H__

#include "dgl/value.h"

namespace sd::dgl {
    struct RunState;

    Value positive_embedding(Values& args, RunState&);
    Value relay_latent_shape(Values& args, RunState&);
    Value relay_decode(Values& args, RunState&);
}  // namespace sd::dgl

#endif  // __SD_DGL_BINDINGS_RELAY_VSR_H__
