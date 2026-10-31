#include "core/int8_ops.h"

#ifndef SD_USE_UPSTREAM_GGML
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

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

static void transform(float* values, int group) {
    const float scale = 1.f / std::sqrt(float(group));
    for (int i = 0; i < group; ++i)
        values[i] *= scale;
    for (int stride = 1; stride < group; stride *= 4)
        for (int base = 0; base < group; base += 4 * stride)
            for (int j = 0; j < stride; ++j) {
                const int i   = base + j;
                const float a = values[i], b = values[i + stride], c = values[i + 2 * stride], d = values[i + 3 * stride];
                values[i]              = a + b + c - d;
                values[i + stride]     = a + b - c + d;
                values[i + 2 * stride] = a - b + c + d;
                values[i + 3 * stride] = -a + b + c + d;
            }
}
struct I8Execution {
    SDI8Operation op;
    int64_t rows;
    int8_t* quant;
    float* scales;
    float* output;
};
static void quantize(ggml_tensor*, int ith, int nth, void* data) {
    auto& e           = *static_cast<I8Execution*>(data);
    const int64_t k   = e.op.input->ne[0];
    const int group   = e.op.group_size;
    const auto* input = static_cast<const float*>(e.op.input->data);
    for (int64_t row = ith; row < e.rows; row += nth) {
        const float* src = input + row * k;
        int8_t* dst      = e.quant + row * k;
        float amax       = 0.f, values[256];
        auto visit       = [&](auto fn) { if (!group) { for (int64_t i = 0; i < k; ++i) fn(i, src[i]); }
            else for (int64_t base = 0; base < k; base += group) { std::memcpy(values, src + base, group * sizeof(float)); transform(values, group); for (int j = 0; j < group; ++j) fn(base + j, values[j]); } };
        visit([&](int64_t, float value) { amax = std::max(amax, std::fabs(value)); });
        const float scale = amax / 127.f, inverse = scale == 0.f ? 0.f : 1.f / scale;
        e.scales[row] = scale;
        if (scale == 0.f)
            std::memset(dst, 0, k);
        else
            visit([&](int64_t i, float value) { dst[i] = static_cast<int8_t>(std::max(-127L, std::min(127L, std::lrintf(value * inverse)))); });
    }
}
static void multiply(ggml_tensor*, int ith, int nth, void* data) {
    auto& e         = *static_cast<I8Execution*>(data);
    const auto& op  = e.op;
    const int64_t k = op.weight->ne[0], n = op.weight->ne[1];
    const auto* w    = static_cast<const int8_t*>(op.weight->data);
    const auto* ws   = op.weight_scale ? static_cast<const float*>(op.weight_scale->data) : nullptr;
    const auto* bias = op.bias ? static_cast<const float*>(op.bias->data) : nullptr;
    for (int64_t index = ith; index < e.rows * n; index += nth) {
        const int64_t row = index / n, column = index % n;
        uint32_t sum = 0;
        for (int64_t i = 0; i < k; ++i)
            sum += uint32_t(int32_t(w[column * k + i]) * int32_t(e.quant[row * k + i]));
        int32_t signed_sum;
        std::memcpy(&signed_sum, &sum, sizeof(sum));
        float value = float(signed_sum) * e.scales[row];
        if (ws)
            value *= ws[column];
        if (bias)
            value += bias[column];
        e.output[index] = value;
    }
}
static bool overlaps(const ggml_tensor* dst, const ggml_tensor* src) {
    if (!src)
        return false;
    const auto a = reinterpret_cast<uintptr_t>(dst->data), b = reinterpret_cast<uintptr_t>(src->data);
    return a <= b ? b - a < ggml_nbytes(dst) : a - b < ggml_nbytes(src);
}
static size_t quant_workspace(const ggml_tensor* node, const SDI8Operation& op) {
    return op.input->type == GGML_TYPE_F32 ? (size_t(ggml_nelements(op.input)) + 3) / 4 * 4 + size_t(ggml_nrows(node)) * 4 : 0;
}
static bool i8_supports(ggml_backend_dev_t, const ggml_tensor* node, void*) {
    SDI8Operation op;
    return sd_ggml_i8_describe(node, &op);
}
static bool convrot_supports(ggml_backend_dev_t, const ggml_tensor* node, void*) {
    int group;
    return sd_ggml_convrot_describe(node, &group);
}
static ggml_status i8_prepare(ggml_backend_t, ggml_tensor* const* nodes, int, void**, size_t* bytes, void*) {
    SDI8Operation op;
    if (!sd_ggml_i8_describe(nodes[0], &op))
        return GGML_STATUS_FAILED;
    *bytes = quant_workspace(nodes[0], op) + ggml_nbytes(nodes[0]);
    return GGML_STATUS_SUCCESS;
}
static ggml_status convrot_prepare(ggml_backend_t, ggml_tensor* const* nodes, int, void**, size_t* bytes, void*) {
    *bytes = ggml_nbytes(nodes[0]);
    return GGML_STATUS_SUCCESS;
}
static void destroy(void*, void*) {}
static ggml_ext_cpu_parallel_fn parallel(ggml_backend_t backend) {
    return reinterpret_cast<ggml_ext_cpu_parallel_fn>(ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(ggml_backend_get_device(backend)), "ggml_ext_cpu_parallel_v1"));
}
static ggml_status i8_execute(const ggml_ext_execution_context* context, ggml_tensor* const* nodes, int, void*, void*) {
    auto* dst     = nodes[0];
    I8Execution e = {};
    if (!sd_ggml_i8_describe(dst, &e.op))
        return GGML_STATUS_FAILED;
    e.rows   = ggml_nrows(dst);
    auto run = parallel(context->backend);
    if (!run)
        return GGML_STATUS_FAILED;
    char* scratch           = static_cast<char*>(ggml_backend_buffer_get_base(context->workspace));
    const size_t quant_size = quant_workspace(dst, e.op);
    if (e.op.input->type == GGML_TYPE_F32) {
        e.quant     = reinterpret_cast<int8_t*>(scratch);
        e.scales    = reinterpret_cast<float*>(scratch + (size_t(ggml_nelements(e.op.input)) + 3) / 4 * 4);
        auto status = run(context->backend, dst, quantize, &e);
        if (status != GGML_STATUS_SUCCESS)
            return status;
    } else {
        e.quant  = static_cast<int8_t*>(e.op.input->data);
        e.scales = e.op.packed ? reinterpret_cast<float*>(e.quant + e.op.input->ne[0] * ((e.rows + 3) / 4 * 4)) : static_cast<float*>(e.op.input_scale->data);
    }
    bool alias = false;
    for (auto* input : dst->src)
        alias |= overlaps(dst, input);
    e.output    = alias ? reinterpret_cast<float*>(scratch + quant_size) : static_cast<float*>(dst->data);
    auto status = run(context->backend, dst, multiply, &e);
    if (status == GGML_STATUS_SUCCESS && alias)
        std::memcpy(dst->data, e.output, ggml_nbytes(dst));
    return status;
}
static ggml_status convrot_execute(const ggml_ext_execution_context* context, ggml_tensor* const* nodes, int, void*, void*) {
    auto* dst     = nodes[0];
    I8Execution e = {};
    if (!sd_ggml_convrot_describe(dst, &e.op.group_size))
        return GGML_STATUS_FAILED;
    auto run = parallel(context->backend);
    if (!run)
        return GGML_STATUS_FAILED;
    e.op.input      = dst->src[0];
    e.rows          = ggml_nrows(e.op.input);
    bool alias      = overlaps(dst, e.op.input);
    e.quant         = static_cast<int8_t*>(alias ? ggml_backend_buffer_get_base(context->workspace) : dst->data);
    const int64_t k = dst->ne[0], padded = (e.rows + 3) / 4 * 4;
    e.scales    = reinterpret_cast<float*>(e.quant + k * padded);
    auto status = run(context->backend, dst, quantize, &e);
    if (status == GGML_STATUS_SUCCESS) {
        std::memset(e.quant + k * e.rows, 0, k * (padded - e.rows));
        if (alias)
            std::memcpy(dst->data, e.quant, ggml_nbytes(dst));
    }
    return status;
}
void sd_ggml_register_cpu_i8(ggml_backend_dev_t device) {
    ggml_ext_kernel_desc d = {};
    d.abi_version          = GGML_EXT_ABI_VERSION;
    d.priority             = 100;
    d.destroy              = destroy;
    d.name                 = "sd.linear_i8.cpu.v1";
    d.op                   = GGML_OP_CUSTOM;
    d.custom_op            = sd_ggml_i8_op();
    d.supports             = i8_supports;
    d.prepare              = i8_prepare;
    d.execute              = i8_execute;
    GGML_ASSERT(ggml_ext_register_kernel(device, &d));
    d.name      = "sd.linear_i8.cpu.override.v1";
    d.op        = GGML_OP_MUL_MAT;
    d.custom_op = nullptr;
    GGML_ASSERT(ggml_ext_register_kernel(device, &d));
    d.name      = "sd.quantize_i8_convrot.cpu.v1";
    d.op        = GGML_OP_CUSTOM;
    d.custom_op = sd_ggml_convrot_op();
    d.supports  = convrot_supports;
    d.prepare   = convrot_prepare;
    d.execute   = convrot_execute;
    GGML_ASSERT(ggml_ext_register_kernel(device, &d));
    d.name      = "sd.quantize_i8_convrot.cpu.override.v1";
    d.op        = GGML_OP_QUANTIZE_I8_CONVROT;
    d.custom_op = nullptr;
    GGML_ASSERT(ggml_ext_register_kernel(device, &d));
}
#endif
