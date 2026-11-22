#ifndef __SD_DGL_BINDINGS_DETAILER_H__
#define __SD_DGL_BINDINGS_DETAILER_H__

#include "dgl/value.h"

namespace sd::dgl {
    struct RunState;
    Value plan_mask_crops(Values& args, RunState&);
}

#endif  // __SD_DGL_BINDINGS_DETAILER_H__
