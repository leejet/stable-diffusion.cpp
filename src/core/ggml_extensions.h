#ifndef __SD_CORE_GGML_EXTENSIONS_H__
#define __SD_CORE_GGML_EXTENSIONS_H__

#include "ggml-backend.h"

void sd_ggml_register_extensions(ggml_backend_t backend);

#ifndef SD_USE_UPSTREAM_GGML
#include "ggml-ext.h"

ggml_ext_op_t sd_ggml_rope_op();
bool sd_ggml_rope_describe(const ggml_tensor* node);
// Apply precomputed rotary matrices, with shapes in ne[] order:
// a: F32 [head_dim, heads, tokens, batch], pe: F32 [2, 2, rot_dim/2, tokens].
// pe stores [[cos, -sin], [sin, cos]] for each pair and token, shared across heads/batches.
// Returns contiguous F32 [head_dim, tokens, heads*batch]. Input strides are preserved.
// Interleaved pairs are (2*i, 2*i+1); otherwise pairs are (i, i+rot_dim/2).
// Channels from rot_dim to head_dim are copied unchanged.
ggml_tensor* sd_ggml_rope_apply(ggml_context* ctx, ggml_backend_t backend, ggml_tensor* x, ggml_tensor* pe, bool interleaved);

struct SDI8Operation {
    const ggml_tensor* weight;
    const ggml_tensor* input;
    const ggml_tensor* input_scale;
    const ggml_tensor* weight_scale;
    const ggml_tensor* bias;
    int group_size;
    bool packed;
};

ggml_ext_op_t sd_ggml_i8_op();
ggml_ext_op_t sd_ggml_convrot_op();
bool sd_ggml_i8_describe(const ggml_tensor* node, SDI8Operation* operation);
bool sd_ggml_convrot_describe(const ggml_tensor* node, int* group_size);
ggml_tensor* sd_ggml_mul_mat_i8(ggml_context* ctx, ggml_tensor* weight, ggml_tensor* input, ggml_tensor* input_scale, ggml_tensor* weight_scale = nullptr, ggml_tensor* bias = nullptr);
ggml_tensor* sd_ggml_linear_i8(ggml_context* ctx, ggml_tensor* weight, ggml_tensor* input, ggml_tensor* weight_scale, ggml_tensor* bias, int group_size);
ggml_tensor* sd_ggml_quantize_i8_convrot(ggml_context* ctx, ggml_tensor* input, int group_size);
#endif

#endif
