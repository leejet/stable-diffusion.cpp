#include "core/ggml_extensions.h"

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

struct I8Attributes {
    int64_t shape[3];
    int32_t group_size;
    int32_t flags;
};
enum I8Flags { I8_PACKED       = 1,
               I8_INPUT_SCALE  = 2,
               I8_WEIGHT_SCALE = 4,
               I8_BIAS         = 8 };

static bool valid_group(int group) {
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
        !ggml_is_contiguous(w) || !ggml_is_contiguous(x) || x->ne[0] != w->ne[0] ||
        (op.group_size != 0 && (!valid_group(op.group_size) || w->ne[0] % op.group_size)))
        return false;
    int64_t rows = 1;
    for (int i = 0; i < 3; ++i) {
        if (shape[i] <= 0 || shape[i] > (INT64_MAX - 3) / rows)
            return false;
        rows *= shape[i];
    }
    const int64_t k = w->ne[0], n = w->ne[1];
    int64_t height;
    if (n <= 0 || n > (i8_size_limit - 4) / 4 || k <= 0 || k > i8_size_limit - 4 - n * 4 ||
        rows > (i8_size_limit - 3) / (k + 4 + n * 4) || !packed_height(k, rows, &height))
        return false;
    if (op.packed) {
        if (x->type != GGML_TYPE_I8 || !valid_group(op.group_size) || op.input_scale ||
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
static bool i8_infer(ggml_tensor* const* inputs, int count, const void* attributes, ggml_ext_tensor_desc* output) {
    I8Attributes attr;
    std::memcpy(&attr, attributes, sizeof(attr));
    if (count < 2 || (attr.flags & ~15))
        return false;
    SDI8Operation op = {inputs[0], inputs[1], nullptr, nullptr, nullptr, attr.group_size, (attr.flags & I8_PACKED) != 0};
    int index        = 2;
    for (auto entry : {std::make_pair(I8_INPUT_SCALE, &op.input_scale), std::make_pair(I8_WEIGHT_SCALE, &op.weight_scale), std::make_pair(I8_BIAS, &op.bias)}) {
        if (attr.flags & entry.first) {
            if (index >= count)
                return false;
            *entry.second = inputs[index++];
        }
    }
    if (index != count || !i8_shape(op, attr.shape))
        return false;
    output->type  = GGML_TYPE_F32;
    output->ne[0] = op.weight->ne[1];
    std::memcpy(output->ne + 1, attr.shape, sizeof(attr.shape));
    return true;
}
static bool convrot_infer(ggml_tensor* const* inputs, int count, const void* attributes, ggml_ext_tensor_desc* output) {
    int32_t group;
    std::memcpy(&group, attributes, sizeof(group));
    if (count != 1 || inputs[0]->type != GGML_TYPE_F32 || !ggml_is_contiguous(inputs[0]) || !valid_group(group) || inputs[0]->ne[0] <= 0 || inputs[0]->ne[0] % group)
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
    static const auto op = [] { ggml_ext_op_desc d = {GGML_EXT_ABI_VERSION, "sd.linear_i8.v1", sizeof(I8Attributes), i8_infer}; auto result = ggml_ext_register_op(&d); GGML_ASSERT(result); return result; }();
    return op;
}
ggml_ext_op_t sd_ggml_convrot_op() {
    static const auto op = [] { ggml_ext_op_desc d = {GGML_EXT_ABI_VERSION, "sd.quantize_i8_convrot.v1", sizeof(int32_t), convrot_infer}; auto result = ggml_ext_register_op(&d); GGML_ASSERT(result); return result; }();
    return op;
}
bool sd_ggml_i8_describe(const ggml_tensor* node, SDI8Operation* op) {
    if (!node || node->type != GGML_TYPE_F32 || !ggml_is_contiguous(node))
        return false;
    *op = {node->src[0], node->src[1], nullptr, nullptr, nullptr, 0, false};
    if (node->op == GGML_OP_MUL_MAT) {
        op->weight_scale = node->src[2];
        op->bias         = node->src[3];
        op->group_size   = node->op_params[2];
        op->packed       = node->op_params[4] > 0;
        if (op->packed)
            op->group_size = node->op_params[4];
    } else if (ggml_ext_op_get(node) == sd_ggml_i8_op()) {
        I8Attributes attr;
        std::memcpy(&attr, ggml_ext_op_attributes(node), sizeof(attr));
        if (attr.flags & ~15 || std::memcmp(attr.shape, node->ne + 1, sizeof(attr.shape)))
            return false;
        op->group_size = attr.group_size;
        op->packed     = (attr.flags & I8_PACKED) != 0;
        int index      = 2;
        if (attr.flags & I8_INPUT_SCALE)
            op->input_scale = node->src[index++];
        if (attr.flags & I8_WEIGHT_SCALE)
            op->weight_scale = node->src[index++];
        if (attr.flags & I8_BIAS)
            op->bias = node->src[index++];
        if (((attr.flags & I8_INPUT_SCALE) && !op->input_scale) || ((attr.flags & I8_WEIGHT_SCALE) && !op->weight_scale) || ((attr.flags & I8_BIAS) && !op->bias))
            return false;
    } else
        return false;
    return i8_shape(*op, node->ne + 1) && node->ne[0] == op->weight->ne[1];
}
bool sd_ggml_convrot_describe(const ggml_tensor* node, int* group) {
    if (!node || !node->src[0] || !ggml_is_contiguous(node))
        return false;
    if (node->op == GGML_OP_QUANTIZE_I8_CONVROT)
        *group = node->op_params[0];
    else if (ggml_ext_op_get(node) == sd_ggml_convrot_op())
        std::memcpy(group, ggml_ext_op_attributes(node), sizeof(int32_t));
    else
        return false;
    ggml_ext_tensor_desc output;
    return convrot_infer(node->src, 1, group, &output) && output.type == node->type && !std::memcmp(output.ne, node->ne, sizeof(output.ne));
}
static ggml_tensor* i8_create(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x, ggml_tensor* xs, ggml_tensor* ws, ggml_tensor* bias, int group, const ggml_tensor* logical) {
    I8Attributes attr      = {{logical->ne[1], logical->ne[2], logical->ne[3]}, group, logical != x ? I8_PACKED : 0};
    ggml_tensor* inputs[5] = {w, x};
    int count              = 2;
    if (xs) {
        attr.flags |= I8_INPUT_SCALE;
        inputs[count++] = xs;
    }
    if (ws) {
        attr.flags |= I8_WEIGHT_SCALE;
        inputs[count++] = ws;
    }
    if (bias) {
        attr.flags |= I8_BIAS;
        inputs[count++] = bias;
    }
    return ggml_ext_op_create(ctx, sd_ggml_i8_op(), inputs, count, &attr);
}
ggml_tensor* sd_ggml_mul_mat_i8(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x, ggml_tensor* xs, ggml_tensor* ws, ggml_tensor* bias) {
    if (!w || !x || !xs || x->type != GGML_TYPE_I8)
        return nullptr;
    return i8_create(ctx, w, x, xs, ws, bias, 0, x);
}
ggml_tensor* sd_ggml_linear_i8(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x, ggml_tensor* ws, ggml_tensor* bias, int group) {
    if (!w || !x)
        return nullptr;
    const auto* logical = x;
    if (x->type == GGML_TYPE_I8) {
        int packed_group;
        if (!sd_ggml_convrot_describe(x, &packed_group) || packed_group != group)
            return nullptr;
        logical = x->src[0];
    }
    return i8_create(ctx, w, x, nullptr, ws, bias, group, logical);
}
ggml_tensor* sd_ggml_quantize_i8_convrot(ggml_context* ctx, ggml_tensor* x, int group) {
    int32_t attr = group;
    return ggml_ext_op_create(ctx, sd_ggml_convrot_op(), &x, 1, &attr);
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

ggml_tensor* sd_ggml_rope_apply(ggml_context* ctx, ggml_backend_t backend, ggml_tensor* x, ggml_tensor* pe, bool interleaved) {
    sd_ggml_register_extensions(backend);
    ggml_tensor* inputs[] = {x, pe};
    int32_t attributes    = interleaved ? 1 : 0;
    return ggml_ext_op_create(ctx, sd_ggml_rope_op(), inputs, 2, &attributes);
}

#else
void sd_ggml_register_extensions(ggml_backend_t) {}
#endif
