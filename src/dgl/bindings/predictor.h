#ifndef __SD_DGL_BINDINGS_PREDICTOR_H__
#define __SD_DGL_BINDINGS_PREDICTOR_H__

#include "dgl/value.h"

namespace sd::dgl {
    struct RunState;

    Value apply_controlnet(Values& args, RunState&);
    Value apply_ip_adapter(Values& args, RunState&);
    Value apply_pulid(Values& args, RunState&);
    Value apply_animatediff(Values& args, RunState&);
    Value make_predictor(Values& args, RunState&);
    Value make_cfg(Values& args, RunState&);
    Value make_apg(Values& args, RunState&);
    Value make_slg(Values& args, RunState&);
    Value switch_predictor(Values& args, RunState&);
}  // namespace sd::dgl

#endif  // __SD_DGL_BINDINGS_PREDICTOR_H__
