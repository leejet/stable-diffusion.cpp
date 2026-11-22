#ifndef __SD_DGL_BINDINGS_REGIONS_H__
#define __SD_DGL_BINDINGS_REGIONS_H__

#include "dgl/registry.h"

namespace sd::dgl {
    struct RunState;

    OutputSignature regions_signature();
    Value regions_from_boxes(Values& args, RunState&);
    Value regions_value(int64_t width, int64_t height, std::shared_ptr<sd_tensor_t> boxes);
    Value crop_regions(Values& args, RunState&);
    Value paste_regions(Values& args, RunState&);
}

#endif  // __SD_DGL_BINDINGS_REGIONS_H__
