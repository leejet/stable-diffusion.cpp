#include "core/cpu/registration.h"
#include "core/ggml_extensions.h"

#ifndef SD_USE_UPSTREAM_GGML
#include <cstring>

namespace sd_cpu {
    struct W4Execution {
        SDW4Operation op;
        float* output;
        int64_t rows;
    };

    static void multiply_w4(ggml_tensor*, int ith, int nth, void* data) {
        auto& e         = *static_cast<W4Execution*>(data);
        const auto& op  = e.op;
        const int64_t k = op.input->ne[0], n = op.weight->ne[1];
        const auto* w       = static_cast<const uint8_t*>(op.weight->data);
        const auto* x       = static_cast<const int8_t*>(op.input->data);
        const auto* xs      = reinterpret_cast<const float*>(x + k * ((e.rows + 3) / 4 * 4));
        const auto* ws      = static_cast<const float*>(op.weight_scale->data);
        const auto* bias    = op.bias ? static_cast<const float*>(op.bias->data) : nullptr;
        const auto* sc      = op.s_channel ? static_cast<const float*>(op.s_channel->data) : nullptr;
        const auto* sr      = op.s_rel ? static_cast<const uint8_t*>(op.s_rel->data) : nullptr;
        const auto to_float = ggml_get_type_traits(GGML_TYPE_F8_E4M3)->to_float;
        for (int64_t index = ith; index < n * e.rows; index += nth) {
            const int64_t row = index / n, col = index % n;
            const auto* packed = w + col * (k / 2);
            float value;
            if (op.kind == SDW4Kind::W4A4) {
                uint32_t sum = 0;
                for (int64_t i = 0; i < k; ++i) {
                    const int nibble = (packed[i / 2] >> (4 * (i % 2))) & 15;
                    sum += uint32_t(((nibble ^ 8) - 8) * int32_t(x[row * k + i]));
                }
                int32_t signed_sum;
                std::memcpy(&signed_sum, &sum, sizeof(sum));
                value = float(signed_sum) * xs[row] * ws[col];
            } else {
                float sum = 0.f;
                for (int64_t g = 0; g < k / 16; ++g) {
                    float relative;
                    to_float(sr + col * (k / 16) + g, &relative, 1);
                    const float scale = sc[col] * relative;
                    for (int j = 0; j < 16; ++j) {
                        const int64_t i = g * 16 + j;
                        const int code  = (packed[i / 2] >> (4 * (i % 2))) & 15;
                        sum += (ws[code] * scale) * float(x[row * k + i]);
                    }
                }
                value = sum * xs[row];
            }
            e.output[index] = value + (bias ? bias[col] : 0.f);
        }
    }

    static bool w4_supports(ggml_backend_dev_t, const ggml_tensor* node, void*) {
        SDW4Operation op;
        return sd_ggml_w4_describe(node, &op);
    }
    static ggml_status w4_prepare(ggml_backend_t, ggml_tensor* const* nodes, int, void**, size_t* bytes, void*) {
        *bytes = ggml_nbytes(nodes[0]);
        return GGML_STATUS_SUCCESS;
    }
    static ggml_status w4_execute(const ggml_ext_execution_context* ctx, ggml_tensor* const* nodes, int, void*, void*) {
        auto* dst     = nodes[0];
        W4Execution e = {};
        if (!sd_ggml_w4_describe(dst, &e.op))
            return GGML_STATUS_FAILED;
        auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(ctx->backend));
        auto run = reinterpret_cast<ggml_ext_cpu_parallel_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_ext_cpu_parallel_v1"));
        if (!run)
            return GGML_STATUS_FAILED;
        e.rows      = ggml_nrows(dst);
        e.output    = static_cast<float*>(ggml_backend_buffer_get_base(ctx->workspace));
        auto status = run(ctx->backend, dst, multiply_w4, &e);
        if (status == GGML_STATUS_SUCCESS)
            std::memcpy(dst->data, e.output, ggml_nbytes(dst));
        return status;
    }
    void register_w4(ggml_backend_dev_t device) {
        ggml_ext_kernel_desc d = {};
        d.abi_version          = GGML_EXT_ABI_VERSION;
        d.name                 = "sd.linear_w4_convrot.cpu.v1";
        d.op                   = GGML_OP_CUSTOM;
        d.custom_op            = sd_ggml_w4_op();
        d.priority             = 100;
        d.supports             = w4_supports;
        d.prepare              = w4_prepare;
        d.execute              = w4_execute;
        d.destroy              = [](void*, void*) {};
        GGML_ASSERT(ggml_ext_register_kernel(device, &d));
    }
}
#endif
