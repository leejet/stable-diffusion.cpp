#ifndef __SD_DGL_OPS_CONTROL_H__
#define __SD_DGL_OPS_CONTROL_H__

#include "dgl/models/fwd.h"

#include "dgl/models/control.h"

namespace sd::dgl::ops {
    models::UNetControls compute_unet_controls(const std::vector<const models::ControlBinding*>& bindings, const Tensor<float>& input, const Tensor<float>& timesteps, const SDCondition& condition);
}

#endif  // __SD_DGL_OPS_CONTROL_H__
