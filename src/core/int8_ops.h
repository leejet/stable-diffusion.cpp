#ifndef __SD_CORE_INT8_OPS_H__
#define __SD_CORE_INT8_OPS_H__

#include "ggml-backend.h"

#ifndef SD_USE_UPSTREAM_GGML
#include "ggml-ext.h"
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
void sd_ggml_register_cpu_i8(ggml_backend_dev_t device);
#endif

#endif
