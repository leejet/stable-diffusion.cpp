#ifndef __SD_DGL_BINDINGS_DETECTIONS_H__
#define __SD_DGL_BINDINGS_DETECTIONS_H__

#include "dgl/registry.h"

namespace sd::dgl {
    struct RunState;

    OutputSignature detections_signature();
    Value detect_objects(Values& args, RunState&);
    Value filter_detections(Values& args, RunState&);
    Value detection_box_masks(Values& args, RunState&);
    Value detections_to_regions(Values& args, RunState&);
}

#endif  // __SD_DGL_BINDINGS_DETECTIONS_H__
