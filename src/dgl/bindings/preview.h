#ifndef __SD_DGL_BINDINGS_PREVIEW_H__
#define __SD_DGL_BINDINGS_PREVIEW_H__

#include "dgl/value.h"

namespace sd::dgl {
    struct RunState;

    Value make_preview(Values& args, RunState& state);
}  // namespace sd::dgl

#endif  // __SD_DGL_BINDINGS_PREVIEW_H__
