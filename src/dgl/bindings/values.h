#ifndef __SD_DGL_BINDINGS_VALUES_H__
#define __SD_DGL_BINDINGS_VALUES_H__

#include "config.h"
#include "dgl/value.h"

namespace sd::dgl {
    struct RunState;

    sd::CacheConfig cache_options(const Values& settings);
    sd::TilingConfig tiling_options(const Values& settings);

    Value execute_list_replace(Values& args, RunState&);
    Value execute_math_product(Values& args, RunState&);
    Value execute_math_ceil_div(Values& args, RunState&);
    Value execute_math_min(Values& args, RunState&);
    Value execute_math_add(Values& args, RunState&);
    Value execute_math_sub(Values& args, RunState&);
    Value execute_math_mul(Values& args, RunState&);
    Value execute_math_div(Values& args, RunState&);
    Value execute_configuration(Values& args, RunState&);
    Value execute_math_floor(Values& args, RunState&);
    Value execute_math_round(Values& args, RunState&);
    Value execute_math_ceil(Values& args, RunState&);
}  // namespace sd::dgl

#endif  // __SD_DGL_BINDINGS_VALUES_H__
