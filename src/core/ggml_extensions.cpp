#include "core/ggml_extensions.h"

#include <cmath>

void sd_ggml_rotate_convrot(float* values, int64_t width, int group) {
    if (group <= 1)
        return;
    const float scale = 1.f / std::sqrt(static_cast<float>(group));
    for (int64_t i = 0; i < width; ++i)
        values[i] *= scale;
    for (int stride = 1; stride < group; stride *= 4) {
        for (int64_t base = 0; base < width; base += 4 * stride) {
            for (int j = 0; j < stride; ++j) {
                const int64_t i = base + j;
                const float a = values[i], b = values[i + stride], c = values[i + 2 * stride], d = values[i + 3 * stride];
                values[i]              = a + b + c - d;
                values[i + stride]     = a + b - c + d;
                values[i + 2 * stride] = a - b + c + d;
                values[i + 3 * stride] = -a + b + c + d;
            }
        }
    }
}

#ifndef SD_USE_UPSTREAM_GGML
#include "core/cpu/registration.h"
#include "ggml-ext.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <unordered_set>
#include <utility>

#ifdef SD_BACKEND_EXTENSIONS
#include "sd_backend_extensions.h"
#endif

static bool modulate_infer(ggml_tensor* const* inputs, int count, const void*, ggml_ext_tensor_desc* out) {
    if (count != 2 && count != 3) {
        return false;
    }
    const auto* x = inputs[0];
    for (int i = 0; i < count; ++i) {
        const auto* t = inputs[i];
        if (!t || t->type != GGML_TYPE_F32) {
            return false;
        }
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (t->ne[d] <= 0 || x->ne[d] % t->ne[d] != 0 || t->nb[d] % sizeof(float)) {
                return false;
            }
        }
    }
    out->type = GGML_TYPE_F32;
    std::memcpy(out->ne, x->ne, sizeof(out->ne));
    return true;
}

ggml_ext_op_t sd_ggml_modulate_op() {
    static const auto op = [] {
        const ggml_ext_op_desc desc = {GGML_EXT_ABI_VERSION, "sd.modulate.v1", 0, modulate_infer};
        auto result                 = ggml_ext_register_op(&desc);
        GGML_ASSERT(result);
        return result;
    }();
    return op;
}

bool sd_ggml_modulate_describe(const ggml_tensor* node) {
    if (!node || ggml_ext_op_get(node) != sd_ggml_modulate_op() || !ggml_is_contiguous(node) || node->src[3]) {
        return false;
    }
    ggml_ext_tensor_desc out;
    return modulate_infer(node->src, node->src[2] ? 3 : 2, nullptr, &out) &&
           node->type == out.type && std::memcmp(node->ne, out.ne, sizeof(out.ne)) == 0;
}

size_t sd_ggml_modulate_workspace_size(const ggml_tensor* node) {
    // Owning custom outputs cannot reuse live input storage. Segment outputs may
    // already be allocated, so reserve scratch only if their ranges can overlap.
    if (!node->data && !node->view_src) {
        return 0;
    }
    if (node->view_src || !node->buffer) {
        return ggml_nbytes(node);
    }
    for (const auto* input : node->src) {
        if (!input) {
            continue;
        }
        const size_t bytes = ggml_nbytes(input);
        size_t offset      = 0;
        while (!input->data && input->view_src) {
            offset += input->view_offs;
            input = input->view_src;
        }
        if (!input->data) {
            continue;
        }
        if (!input->buffer) {
            return ggml_nbytes(node);
        }
        if (node->buffer != input->buffer) {
            continue;
        }
        const auto dst = reinterpret_cast<uintptr_t>(node->data);
        const auto src = reinterpret_cast<uintptr_t>(input->data) + offset;
        if (dst < src + bytes && src < dst + ggml_nbytes(node)) {
            return ggml_nbytes(node);
        }
    }
    return 0;
}

static bool rope_infer(ggml_tensor* const* inputs, int count, const void*, ggml_ext_tensor_desc* out) {
    if (count != 2) {
        return false;
    }
    const auto* x  = inputs[0];
    const auto* pe = inputs[1];
    if (x->type != GGML_TYPE_F32 || pe->type != GGML_TYPE_F32 || x->ne[0] % 2 != 0 ||
        pe->ne[0] != 2 || pe->ne[1] != 2 || pe->ne[2] <= 0 || pe->ne[2] > x->ne[0] / 2 ||
        pe->ne[3] != x->ne[2]) {
        return false;
    }
    out->type  = GGML_TYPE_F32;
    out->ne[0] = x->ne[0];
    out->ne[1] = x->ne[2];
    out->ne[2] = x->ne[1] * x->ne[3];
    out->ne[3] = 1;
    return true;
}

ggml_ext_op_t sd_ggml_rope_op() {
    static const ggml_ext_op_t op = [] {
        const ggml_ext_op_desc desc = {GGML_EXT_ABI_VERSION, "sd.rope_apply.v1", sizeof(int32_t), rope_infer};
        auto result                 = ggml_ext_register_op(&desc);
        GGML_ASSERT(result);
        return result;
    }();
    return op;
}

bool sd_ggml_rope_describe(const ggml_tensor* node) {
    if (node->type != GGML_TYPE_F32 || !ggml_is_contiguous(node) || !node->src[0] || !node->src[1]) {
        return false;
    }
    ggml_tensor* inputs[] = {node->src[0], node->src[1]};
    ggml_ext_tensor_desc output;
    return rope_infer(inputs, 2, nullptr, &output) &&
           std::memcmp(output.ne, node->ne, sizeof(output.ne)) == 0;
}

bool sd_ggml_valid_convrot_group(int group) {
    return group == 1 || group == 4 || group == 16 || group == 64 || group == 256;
}

static constexpr int64_t i8_size_limit = int64_t(std::min(uint64_t(INT64_MAX), uint64_t(SIZE_MAX)));

static bool packed_height(int64_t k, int64_t rows, int64_t* height) {
    if (k <= 0 || rows <= 0 || rows > i8_size_limit / 4)
        return false;
    const int64_t padded = (rows + 3) / 4 * 4;
    const int64_t scales = (rows * 4 - 1) / k + 1;
    if (padded > i8_size_limit / k || scales > i8_size_limit / k - padded)
        return false;
    *height = padded + scales;
    return true;
}

static bool scale_matches(const ggml_tensor* scale, int64_t count) {
    return !scale || (scale->type == GGML_TYPE_F32 && ggml_is_contiguous(scale) && ggml_nelements(scale) == count);
}

static bool i8_shape(const SDI8Operation& op, const int64_t* shape) {
    const auto* w = op.weight;
    const auto* x = op.input;
    if (!w || !x || w->type != GGML_TYPE_I8 || w->ne[2] != 1 || w->ne[3] != 1 ||
        !ggml_is_contiguous(w) || !ggml_is_contiguous(x) ||
        (op.group_size != 0 && (!sd_ggml_valid_convrot_group(op.group_size) || op.k() % op.group_size)))
        return false;
    int64_t rows = 1;
    for (int i = 0; i < 3; ++i) {
        if (shape[i] <= 0 || shape[i] > (INT64_MAX - 3) / rows)
            return false;
        rows *= shape[i];
    }
    const int64_t k = op.k(), n = w->ne[1];
    int64_t height;
    if (n <= 0 || n > (i8_size_limit - 4) / 4 || k <= 0 || k > i8_size_limit - 4 - n * 4 ||
        rows > (i8_size_limit - 3) / (k + 4 + n * 4) || !packed_height(k, rows, &height))
        return false;
    if (op.codebook) {
        const auto* scales = op.relative_scale;
        if (k % 16 || w->ne[0] != k / 2 || k > i8_size_limit / n || !op.weight_scale ||
            !scale_matches(op.codebook, 16) || scales->type != GGML_TYPE_F8_E4M3 ||
            !ggml_is_contiguous(scales) || ggml_n_dims(scales) > 2 ||
            scales->ne[0] != k / 16 || scales->ne[1] != n)
            return false;
    } else if (w->ne[0] != k) {
        return false;
    }
    if (op.packed) {
        if (x->type != GGML_TYPE_I8 || !sd_ggml_valid_convrot_group(op.group_size) || op.input_scale ||
            x->ne[1] != height || x->ne[2] != 1 || x->ne[3] != 1)
            return false;
    } else {
        if (std::memcmp(shape, x->ne + 1, 3 * sizeof(int64_t)))
            return false;
        if (x->type == GGML_TYPE_I8) {
            if (!op.input_scale || op.group_size != 0)
                return false;
        } else if (x->type != GGML_TYPE_F32 || op.input_scale)
            return false;
    }
    return scale_matches(op.input_scale, rows) && scale_matches(op.weight_scale, w->ne[1]) && scale_matches(op.bias, w->ne[1]);
}

static constexpr int i8_flags = SD_I8_PACKED | SD_I8_INPUT_SCALE | SD_I8_WEIGHT_SCALE | SD_I8_BIAS | SD_I8_WEIGHT_W4;

static bool i8_operands(ggml_tensor* const* inputs, int count, const SDI8Attributes& attr, SDI8Operation* op) {
    if (count < 2 || (attr.flags & ~i8_flags))
        return false;
    *op       = {inputs[0], inputs[1], nullptr, nullptr, nullptr, attr.group_size, (attr.flags & SD_I8_PACKED) != 0, nullptr, nullptr};
    int index = 2;
    for (auto entry : {std::make_pair(SD_I8_INPUT_SCALE, &op->input_scale), std::make_pair(SD_I8_WEIGHT_SCALE, &op->weight_scale), std::make_pair(SD_I8_BIAS, &op->bias)}) {
        if (attr.flags & entry.first) {
            if (index >= count || !inputs[index])
                return false;
            *entry.second = inputs[index++];
        }
    }
    if (attr.flags & SD_I8_WEIGHT_W4) {
        if (count - index != 2 || !inputs[index] || !inputs[index + 1])
            return false;
        op->codebook       = inputs[index++];
        op->relative_scale = inputs[index++];
    }
    return index == count && i8_shape(*op, attr.shape);
}

static bool i8_infer(ggml_tensor* const* inputs, int count, const void* attributes, ggml_ext_tensor_desc* output) {
    SDI8Attributes attr;
    std::memcpy(&attr, attributes, sizeof(attr));
    SDI8Operation op;
    if (!i8_operands(inputs, count, attr, &op))
        return false;
    output->type  = GGML_TYPE_F32;
    output->ne[0] = op.weight->ne[1];
    std::memcpy(output->ne + 1, attr.shape, sizeof(attr.shape));
    return true;
}

static bool requantize_q8_infer(ggml_tensor* const* inputs, int count, const void*, ggml_ext_tensor_desc* output) {
    if (count != 2 || !inputs[0] || !inputs[1] || inputs[0]->type != GGML_TYPE_Q8_0 || inputs[1]->type != GGML_TYPE_I8)
        return false;
    const auto* weight = inputs[0];
    const int64_t k = weight->ne[0], n = weight->ne[1];
    constexpr int64_t scale_bytes = sizeof(float);
    int group;
    if (!sd_ggml_convrot_describe(inputs[1], &group) || group != 256 || k % group ||
        weight->ne[2] != 1 || weight->ne[3] != 1 || !ggml_is_contiguous(weight) ||
        k <= 0 || k > i8_size_limit - scale_bytes || n <= 0 || inputs[1]->ne[0] != k || n > i8_size_limit / (k + scale_bytes))
        return false;
    output->type  = GGML_TYPE_I8;
    output->ne[0] = (k + scale_bytes) * n;
    output->ne[1] = output->ne[2] = output->ne[3] = 1;
    return true;
}

static bool convrot_infer(ggml_tensor* const* inputs, int count, const void* attributes, ggml_ext_tensor_desc* output) {
    int32_t group;
    std::memcpy(&group, attributes, sizeof(group));
    if (count != 1 || inputs[0]->type != GGML_TYPE_F32 || !ggml_is_contiguous(inputs[0]) || !sd_ggml_valid_convrot_group(group) || inputs[0]->ne[0] <= 0 || inputs[0]->ne[0] % group)
        return false;
    const int64_t k = inputs[0]->ne[0], rows = ggml_nrows(inputs[0]);
    int64_t height;
    if (!packed_height(k, rows, &height))
        return false;
    output->type  = GGML_TYPE_I8;
    output->ne[0] = k;
    output->ne[1] = height;
    output->ne[2] = output->ne[3] = 1;
    return true;
}

ggml_ext_op_t sd_ggml_i8_op() {
    static const auto op = [] { ggml_ext_op_desc d = {GGML_EXT_ABI_VERSION, "sd.linear_i8.v1", sizeof(SDI8Attributes), i8_infer}; auto result = ggml_ext_register_op(&d); GGML_ASSERT(result); return result; }();
    return op;
}

ggml_ext_op_t sd_ggml_requantize_q8_op() {
    static const auto op = [] {
        const ggml_ext_op_desc desc = {GGML_EXT_ABI_VERSION, "sd.requantize_q8.v1", 0, requantize_q8_infer};
        const auto result           = ggml_ext_register_op(&desc);
        GGML_ASSERT(result);
        return result;
    }();
    return op;
}

ggml_ext_op_t sd_ggml_convrot_op() {
    static const auto op = [] { ggml_ext_op_desc d = {GGML_EXT_ABI_VERSION, "sd.quantize_i8_convrot.v1", sizeof(int32_t), convrot_infer}; auto result = ggml_ext_register_op(&d); GGML_ASSERT(result); return result; }();
    return op;
}

bool sd_ggml_i8_describe(const ggml_tensor* node, SDI8Operation* op) {
    if (!node || node->type != GGML_TYPE_F32 || !ggml_is_contiguous(node) || ggml_ext_op_get(node) != sd_ggml_i8_op())
        return false;
    SDI8Attributes attr;
    std::memcpy(&attr, ggml_ext_op_attributes(node), sizeof(attr));
    if (std::memcmp(attr.shape, node->ne + 1, sizeof(attr.shape)))
        return false;
    int count = 0;
    while (count < GGML_MAX_SRC && node->src[count])
        ++count;
    return i8_operands(node->src, count, attr, op) && node->ne[0] == op->weight->ne[1];
}

bool sd_ggml_convrot_describe(const ggml_tensor* node, int* group) {
    if (!node || !node->src[0] || !ggml_is_contiguous(node))
        return false;
    if (ggml_ext_op_get(node) == sd_ggml_convrot_op())
        std::memcpy(group, ggml_ext_op_attributes(node), sizeof(int32_t));
    else
        return false;
    ggml_ext_tensor_desc output;
    return convrot_infer(node->src, 1, group, &output) && output.type == node->type && !std::memcmp(output.ne, node->ne, sizeof(output.ne));
}

static bool i8_lora_merge_infer(ggml_tensor* const* inputs, int count, const void* attributes, ggml_ext_tensor_desc* output) {
    if (count != 3)
        return false;
    int32_t group;
    std::memcpy(&group, attributes, sizeof(group));
    const auto* weight = inputs[0];
    const auto* scale  = inputs[1];
    const auto* diff   = inputs[2];
    const int64_t k = weight->ne[0], rows = weight->ne[1];
    if (weight->type != GGML_TYPE_I8 || weight->ne[2] != 1 || weight->ne[3] != 1 ||
        k <= 0 || rows <= 0 || k > (i8_size_limit / rows - int64_t(sizeof(SDI8LoRARowStats))) / int64_t(sizeof(float)) ||
        diff->type != GGML_TYPE_F32 || !ggml_are_same_shape(weight, diff) ||
        scale->type != GGML_TYPE_F32 || (ggml_nelements(scale) != 1 && ggml_nelements(scale) != rows) ||
        !ggml_is_contiguous(weight) || !ggml_is_contiguous(diff) || !ggml_is_contiguous(scale) ||
        (group != 0 && (!sd_ggml_valid_convrot_group(group) || k % group != 0)))
        return false;

    const SDI8LoRAMerge op = {weight, scale, diff, group};
    const size_t bytes     = op.weight_offset() + ggml_nbytes(weight);
    output->type           = GGML_TYPE_I8;
    output->ne[0]          = GGML_PAD(bytes, sizeof(uint32_t));
    output->ne[1] = output->ne[2] = output->ne[3] = 1;
    return true;
}

ggml_ext_op_t sd_ggml_i8_lora_merge_op() {
    static const auto op = [] {
        const ggml_ext_op_desc desc = {GGML_EXT_ABI_VERSION, "sd.i8_lora_merge.v1", sizeof(int32_t), i8_lora_merge_infer};
        auto result                 = ggml_ext_register_op(&desc);
        GGML_ASSERT(result);
        return result;
    }();
    return op;
}

bool sd_ggml_i8_lora_merge_describe(const ggml_tensor* node, SDI8LoRAMerge* op) {
    if (!node || ggml_ext_op_get(node) != sd_ggml_i8_lora_merge_op())
        return false;
    const void* attributes = ggml_ext_op_attributes(node);
    ggml_ext_tensor_desc output;
    if (!i8_lora_merge_infer(node->src, 3, attributes, &output) || output.type != node->type ||
        std::memcmp(output.ne, node->ne, sizeof(output.ne)) != 0 || !ggml_is_contiguous(node))
        return false;
    *op = {node->src[0], node->src[1], node->src[2], 0};
    std::memcpy(&op->group, attributes, sizeof(op->group));
    return true;
}

static bool w4_infer(ggml_tensor* const* inputs, int count, const void* attributes, ggml_ext_tensor_desc* out) {
    if (!attributes || count < 3)
        return false;
    SDW4Attributes attr;
    std::memcpy(&attr, attributes, sizeof(attr));
    if ((attr.has_bias != 0 && attr.has_bias != 1) || count != 3 + attr.has_bias)
        return false;
    for (int i = 0; i < count; ++i)
        if (!inputs[i] || !ggml_is_contiguous(inputs[i]))
            return false;
    const auto* w = inputs[0];
    const auto* x = inputs[1];
    if (w->type != GGML_TYPE_I8 || w->ne[0] <= 0 || w->ne[0] > i8_size_limit / 2 ||
        w->ne[1] <= 0 || w->ne[2] != 1 || w->ne[3] != 1)
        return false;
    const int64_t k = 2 * w->ne[0], n = w->ne[1];
    int64_t rows = 1;
    for (auto dim : attr.shape) {
        if (dim <= 0 || dim > i8_size_limit / rows)
            return false;
        rows *= dim;
    }
    int64_t height;
    if (!packed_height(k, rows, &height) || n > i8_size_limit / 4 / rows ||
        x->type != GGML_TYPE_I8 || x->ne[0] != k || x->ne[1] != height || x->ne[2] != 1 || x->ne[3] != 1 ||
        !scale_matches(inputs[2], n) ||
        (attr.has_bias && !scale_matches(inputs[count - 1], n)))
        return false;
    out->type  = GGML_TYPE_F32;
    out->ne[0] = n;
    std::memcpy(out->ne + 1, attr.shape, sizeof(attr.shape));
    return true;
}

ggml_ext_op_t sd_ggml_w4_op() {
    static const auto op = [] {
        const ggml_ext_op_desc desc = {GGML_EXT_ABI_VERSION, "sd.linear_w4_convrot.v1", sizeof(SDW4Attributes), w4_infer};
        auto result                 = ggml_ext_register_op(&desc);
        GGML_ASSERT(result);
        return result;
    }();
    return op;
}

bool sd_ggml_w4_describe(const ggml_tensor* node, SDW4Operation* op) {
    if (!node || ggml_ext_op_get(node) != sd_ggml_w4_op() || !ggml_is_contiguous(node))
        return false;
    SDW4Attributes attr;
    std::memcpy(&attr, ggml_ext_op_attributes(node), sizeof(attr));
    const int count = 3 + (attr.has_bias != 0);
    ggml_ext_tensor_desc out;
    if (!w4_infer(node->src, count, &attr, &out) || node->type != out.type ||
        std::memcmp(node->ne, out.ne, sizeof(out.ne)))
        return false;
    *op = {node->src[0], node->src[1], node->src[2], attr.has_bias ? node->src[3] : nullptr};
    return true;
}

static bool sage_attn_infer(ggml_tensor* const* inputs, int count, const void* attributes, ggml_ext_tensor_desc* out) {
    if (count != 3 || !attributes)
        return false;
    SDSageAttentionParams params;
    std::memcpy(&params, attributes, sizeof(params));
    if (!std::isfinite(params.scale) || params.scale <= 0.f || params.mode < GGML_SAGE_ATTN_AUTO || params.mode > GGML_SAGE_ATTN_2_PLUS_PLUS)
        return false;
    for (int i = 0; i < count; ++i) {
        const auto* t = inputs[i];
        if (!t || !ggml_is_contiguous(t) || t->type != (i == 2 ? GGML_TYPE_F16 : GGML_TYPE_F32))
            return false;
        int64_t elements = 1;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (t->ne[d] <= 0 || t->ne[d] > INT64_MAX / int64_t(sizeof(float)) / elements)
                return false;
            elements *= t->ne[d];
        }
    }
    const auto* q = inputs[0];
    const auto* k = inputs[1];
    if (q->ne[0] != k->ne[0] || !ggml_are_same_shape(k, inputs[2]) || q->ne[2] % k->ne[2] != 0 || q->ne[3] != k->ne[3])
        return false;
    *out = {GGML_TYPE_F32, {q->ne[0], q->ne[2], q->ne[1], q->ne[3]}};
    return true;
}

ggml_ext_op_t sd_ggml_sage_attn_op() {
    static const auto op = [] {
        const ggml_ext_op_desc desc = {GGML_EXT_ABI_VERSION, "sd.sage_attn.v1", sizeof(SDSageAttentionParams), sage_attn_infer};
        auto result                 = ggml_ext_register_op(&desc);
        GGML_ASSERT(result);
        return result;
    }();
    return op;
}

bool sd_ggml_sage_attn_describe(const ggml_tensor* node, SDSageAttentionParams* params) {
    if (!node || ggml_ext_op_get(node) != sd_ggml_sage_attn_op() || node->src[3] || !ggml_is_contiguous(node))
        return false;
    ggml_ext_tensor_desc output;
    const auto* attributes = ggml_ext_op_attributes(node);
    if (!sage_attn_infer(node->src, 3, attributes, &output) || node->type != output.type || std::memcmp(node->ne, output.ne, sizeof(output.ne)))
        return false;
    std::memcpy(params, attributes, sizeof(*params));
    return true;
}

static bool sol_attn_infer(ggml_tensor* const* inputs, int count, const void* attributes, ggml_ext_tensor_desc* out) {
    if (count != 3 || !attributes)
        return false;
    SDSolAttentionParams params;
    std::memcpy(&params, attributes, sizeof(params));
    if (!std::isfinite(params.scale) || params.scale <= 0.f || !std::isfinite(params.tau))
        return false;
    const auto* q = inputs[0];
    if (!q || q->ne[0] != 128)
        return false;
    for (int i = 0; i < count; ++i) {
        const auto* t = inputs[i];
        if (!t || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || !ggml_are_same_shape(q, t))
            return false;
    }
    int64_t elements = 1;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (q->ne[d] <= 0 || q->ne[d] > INT64_MAX / int64_t(sizeof(float)) / elements)
            return false;
        elements *= q->ne[d];
    }
    *out = {GGML_TYPE_F32, {q->ne[0], q->ne[2], q->ne[1], q->ne[3]}};
    return true;
}

ggml_ext_op_t sd_ggml_sol_attn_op() {
    static const auto op = [] {
        const ggml_ext_op_desc desc = {GGML_EXT_ABI_VERSION, "sd.sol_attn.v1", sizeof(SDSolAttentionParams), sol_attn_infer};
        auto result                 = ggml_ext_register_op(&desc);
        GGML_ASSERT(result);
        return result;
    }();
    return op;
}

bool sd_ggml_sol_attn_describe(const ggml_tensor* node, SDSolAttentionParams* params) {
    if (!node || ggml_ext_op_get(node) != sd_ggml_sol_attn_op() || node->src[3] || !ggml_is_contiguous(node))
        return false;
    ggml_ext_tensor_desc output;
    const auto* attributes = ggml_ext_op_attributes(node);
    if (!sol_attn_infer(node->src, 3, attributes, &output) || node->type != output.type || std::memcmp(node->ne, output.ne, sizeof(output.ne)))
        return false;
    std::memcpy(params, attributes, sizeof(*params));
    return true;
}

bool sd_ggml_window_attn_infer(ggml_tensor* const* inputs,
                               int count,
                               const void* attributes,
                               ggml_ext_tensor_desc* out) {
    if (count != 3 || !attributes)
        return false;
    SDWindowAttentionParams p;
    std::memcpy(&p, attributes, sizeof(p));
    if (p.width <= 0 || p.height <= 0 || int64_t(p.width) * p.height > INT32_MAX / 4 ||
        p.heads <= 0 || p.planes < 1 || p.planes > 4 || (p.preserve_boundary != 0 && p.preserve_boundary != 1))
        return false;
    for (int i = 0; i < p.planes; ++i)
        if (p.window_height[i] <= 0 || p.window_width[i] <= 0 ||
            (!p.preserve_boundary && (p.window_height[i] % 2 || p.window_width[i] % 2)))
            return false;
    for (int i = 0; i < 3; ++i) {
        const auto* t = inputs[i];
        if (!t || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || t->ne[2] != 1 || t->ne[3] != 1 ||
            t->ne[1] != int64_t(p.width) * p.height * (i ? p.planes : 1))
            return false;
    }
    const auto* q = inputs[0];
    const auto* k = inputs[1];
    if (q->ne[0] % p.heads || (q->ne[0] / p.heads != 64 && q->ne[0] / p.heads != 128) ||
        k->ne[0] <= 0 || k->ne[0] > q->ne[0] || q->ne[0] % k->ne[0] ||
        k->ne[0] % (q->ne[0] / p.heads) || !ggml_are_same_shape(k, inputs[2]))
        return false;
    *out = {GGML_TYPE_F32, {q->ne[0], q->ne[1], 1, 1}};
    return true;
}

ggml_ext_op_t sd_ggml_window_attn_op() {
    static const auto op = [] {
        const ggml_ext_op_desc desc = {GGML_EXT_ABI_VERSION, "sd.window_attn.v1", sizeof(SDWindowAttentionParams), sd_ggml_window_attn_infer};
        auto result                 = ggml_ext_register_op(&desc);
        GGML_ASSERT(result);
        return result;
    }();
    return op;
}

bool sd_ggml_window_attn_describe(const ggml_tensor* node, SDWindowAttentionParams* params) {
    if (!node || ggml_ext_op_get(node) != sd_ggml_window_attn_op() || node->src[3] || !ggml_is_contiguous(node))
        return false;
    const auto* attributes = ggml_ext_op_attributes(node);
    ggml_ext_tensor_desc out;
    if (!sd_ggml_window_attn_infer(node->src, 3, attributes, &out) || node->type != out.type || std::memcmp(node->ne, out.ne, sizeof(out.ne)))
        return false;
    std::memcpy(params, attributes, sizeof(*params));
    return true;
}

void sd_ggml_register_extensions(ggml_backend_t backend) {
    if (!backend) {
        return;
    }
    auto device    = ggml_backend_get_device(backend);
    auto reg       = ggml_backend_dev_backend_reg(device);
    const bool cpu = ggml_backend_reg_get_proc_address(reg, "ggml_ext_cpu_parallel_v1") != nullptr;
    static std::mutex mutex;
    static std::unordered_set<ggml_backend_dev_t> registered;
    std::lock_guard<std::mutex> lock(mutex);
    if (!registered.insert(device).second) {
        return;
    }
    if (!cpu) {
#ifdef SD_BACKEND_EXTENSIONS
        for (auto register_backend : sd_backend_extensions)
            register_backend(device);
#endif
        return;
    }
    sd_cpu::register_extensions(device);
}

#else
void sd_ggml_register_extensions(ggml_backend_t) {}
#endif
