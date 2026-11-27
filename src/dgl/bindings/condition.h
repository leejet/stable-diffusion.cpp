#ifndef __SD_DGL_BINDINGS_CONDITION_H__
#define __SD_DGL_BINDINGS_CONDITION_H__

#include "dgl/registry.h"

struct SDCondition;

namespace sd::dgl {
    OutputSignature condition_signature();
    Value zero_out_condition(Values& args, RunState& state);
    Value condition_value(SDCondition condition);
    SDCondition read_condition(const Values& fields, const Values& overrides = {});
    void validate_reference_blocks(const SDCondition& condition, size_t video_count);
}

#endif  // __SD_DGL_BINDINGS_CONDITION_H__
