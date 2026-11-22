#ifndef __SD_DGL_OPS_SEEDVR2_H__
#define __SD_DGL_OPS_SEEDVR2_H__

#include "dgl/models/fwd.h"

#include "core/tensor.hpp"

namespace sd::dgl::ops {
    Tensor<float> seedvr2_positive_embedding(models::DiT& dit);
}

#endif  // __SD_DGL_OPS_SEEDVR2_H__
