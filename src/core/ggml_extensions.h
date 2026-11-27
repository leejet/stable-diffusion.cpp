#ifndef __SD_CORE_GGML_EXTENSIONS_H__
#define __SD_CORE_GGML_EXTENSIONS_H__

#include "ggml-backend.h"

void sd_ggml_register_extensions(ggml_backend_t backend);
void sd_ggml_rotate_convrot(float* values, int64_t width, int group);

#ifndef SD_USE_UPSTREAM_GGML
#include "ggml-ext.h"

ggml_ext_op_t sd_ggml_modulate_op();

bool sd_ggml_modulate_describe(const ggml_tensor* node);

size_t sd_ggml_modulate_workspace_size(const ggml_tensor* node);

ggml_ext_op_t sd_ggml_rope_op();

bool sd_ggml_rope_describe(const ggml_tensor* node);

struct SDI8Attributes {
    int64_t shape[3];
    int32_t group_size;
    int32_t flags;
};
enum SDI8Flags { SD_I8_PACKED       = 1,
                 SD_I8_INPUT_SCALE  = 2,
                 SD_I8_WEIGHT_SCALE = 4,
                 SD_I8_BIAS         = 8,
                 SD_I8_WEIGHT_W4    = 16 };

struct SDI8Operation {
    const ggml_tensor* weight;
    const ggml_tensor* input;
    const ggml_tensor* input_scale;
    const ggml_tensor* weight_scale;
    const ggml_tensor* bias;
    int group_size;
    bool packed;
    const ggml_tensor* codebook;
    const ggml_tensor* relative_scale;

    int64_t k() const { return input->ne[0]; }
    size_t decoded_weight_bytes() const { return codebook ? size_t(k()) * weight->ne[1] : 0; }
};

bool sd_ggml_valid_convrot_group(int group);

ggml_ext_op_t sd_ggml_i8_op();
ggml_ext_op_t sd_ggml_requantize_q8_op();

ggml_ext_op_t sd_ggml_convrot_op();

bool sd_ggml_i8_describe(const ggml_tensor* node, SDI8Operation* operation);

bool sd_ggml_convrot_describe(const ggml_tensor* node, int* group_size);

enum SDI8LoRAMergeFlags : uint32_t {
    SD_I8_LORA_CHANGED   = 1,
    SD_I8_LORA_NONFINITE = 2,
};

struct SDI8LoRARowStats {
    float maximum;
    uint32_t changed;
};

struct SDI8LoRAMerge {
    const ggml_tensor* weight;
    const ggml_tensor* scale;
    const ggml_tensor* diff;
    int32_t group;

    bool scalar_scale() const { return ggml_nelements(scale) == 1; }
    size_t scale_offset() const { return sizeof(uint32_t); }
    size_t weight_offset() const { return scale_offset() + ggml_nbytes(scale); }
    size_t workspace_size() const { return ggml_nbytes(diff) + ggml_nrows(weight) * sizeof(SDI8LoRARowStats); }
};

ggml_ext_op_t sd_ggml_i8_lora_merge_op();
bool sd_ggml_i8_lora_merge_describe(const ggml_tensor* node, SDI8LoRAMerge* op);

struct SDW4Attributes {
    int64_t shape[3];
    int32_t has_bias;
};

struct SDW4Operation {
    const ggml_tensor* weight;
    const ggml_tensor* input;
    const ggml_tensor* weight_scale;
    const ggml_tensor* bias;
};

ggml_ext_op_t sd_ggml_w4_op();

bool sd_ggml_w4_describe(const ggml_tensor* node, SDW4Operation* operation);

enum ggml_sage_attn_mode {
    GGML_SAGE_ATTN_AUTO = 0,
    GGML_SAGE_ATTN_2,
    GGML_SAGE_ATTN_2_PLUS_PLUS,
};

struct SDSageAttentionParams {
    float scale;
    ggml_sage_attn_mode mode;
};

ggml_ext_op_t sd_ggml_sage_attn_op();

bool sd_ggml_sage_attn_describe(const ggml_tensor* node, SDSageAttentionParams* params);

struct SDSolAttentionParams {
    float scale;
    float tau;
};

struct SDWindowAttentionParams {
    int32_t width, height, heads, planes;
    int16_t window_height[4], window_width[4];
    int32_t preserve_boundary;
};

bool sd_ggml_window_attn_infer(ggml_tensor* const* inputs,
                               int count,
                               const void* attributes,
                               ggml_ext_tensor_desc* out);

ggml_ext_op_t sd_ggml_window_attn_op();

bool sd_ggml_window_attn_describe(const ggml_tensor* node, SDWindowAttentionParams* params);

ggml_ext_op_t sd_ggml_sol_attn_op();

bool sd_ggml_sol_attn_describe(const ggml_tensor* node, SDSolAttentionParams* params);

#endif

#endif
