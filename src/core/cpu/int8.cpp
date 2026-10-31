#include "core/cpu/registration.h"
#include "core/ggml_extensions.h"

#ifndef SD_USE_UPSTREAM_GGML
#include <algorithm>
#include <cmath>
#include <cstring>

namespace sd_cpu {

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
    void register_i8(ggml_backend_dev_t device) {
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

}
#endif
