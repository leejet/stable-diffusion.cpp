#ifndef __SD_DGL_BINDINGS_TENSOR_OPS_H__
#define __SD_DGL_BINDINGS_TENSOR_OPS_H__

#include "dgl/value.h"

namespace sd::dgl {
    struct RunState;

    Value blur_tensor(const Values& args, RunState&);
    Value execute_load_tensor(Values& args, RunState&);
    Value execute_save_tensor(Values& args, RunState&);
    Value execute_tensor_literal(Values& args, RunState&);
    Value execute_tensor_shape(Values& args, RunState&);
    Value execute_tensor_index(Values& args, RunState&);
    Value execute_tensor_vector(Values& args, RunState&);
    Value execute_tensor_to_list(Values& args, RunState&);
    Value execute_tensor_reshape(Values& args, RunState&);
    Value execute_tensor_unsqueeze(Values& args, RunState&);
    Value execute_tensor_pad(Values& args, RunState&);
    Value execute_tensor_crop(Values& args, RunState&);
    Value execute_tensor_numel(Values& args, RunState&);
    Value execute_tensor_zeros(Values& args, RunState&);
    Value execute_tensor_full(Values& args, RunState&);
    Value execute_tensor_zeros_like(Values& args, RunState&);
    Value execute_tensor_lerp(Values& args, RunState&);
    Value execute_tensor_affine(Values& args, RunState&);
    Value execute_tensor_clamp(Values& args, RunState&);
    Value execute_tensor_max_pool2d(Values& args, RunState&);
    Value execute_tensor_min_pool2d(Values& args, RunState&);
    Value execute_tensor_interpolate(Values& args, RunState&);
    Value execute_tensor_add(Values& args, RunState&);
    Value execute_tensor_mul(Values& args, RunState&);
    Value execute_tensor_div(Values& args, RunState&);
    Value cast_tensor(const Values& args, RunState&);
    Value translate_tensor(const Values& args, RunState&);
    Value arange_tensor(const Values& args, RunState&);
    Value index_select_tensor(const Values& args, RunState&);
    Value replace_slice(const Values& args, RunState&);
    Value concat_tensors(const Values& args, RunState&);
    Value grayscale_tensor(const Values& args, RunState&);
    Value threshold_tensor(const Values& args, RunState&);
    Value maximum_tensor(const Values& args, RunState&);
    Value execute_tensor_permute(Values& args, RunState&);
    Value execute_tensor_broadcast_to(Values& args, RunState&);
    Value execute_tensor_index_fill(Values& args, RunState&);
    Value execute_tensor_randn_like(Values& args, RunState&);
    Value execute_tensor_count_nonzero(Values& args, RunState&);
}  // namespace sd::dgl

#endif  // __SD_DGL_BINDINGS_TENSOR_OPS_H__
