#include "core/cpu/registration.h"
#include "core/ggml_extensions.h"

#ifndef SD_USE_UPSTREAM_GGML
#include <cstring>

namespace sd_cpu {
    static bool modulate_supports(ggml_backend_dev_t, const ggml_tensor* node, void*) {
        return sd_ggml_modulate_describe(node);
    }

    static float modulate_value(const ggml_tensor* t, const int64_t* index) {
        size_t offset = 0;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            offset += (index[d] % t->ne[d]) * t->nb[d];
        }
        return *reinterpret_cast<const float*>(static_cast<const char*>(t->data) + offset);
    }

    static void modulate_compute(ggml_tensor* dst, int ith, int nth, void*) {
        auto* output       = static_cast<float*>(dst->data);
        const int64_t rows = ggml_nrows(dst);
        for (int64_t row = ith; row < rows; row += nth) {
            int64_t index[] = {0, row % dst->ne[1], (row / dst->ne[1]) % dst->ne[2], row / (dst->ne[1] * dst->ne[2])};
            for (int64_t col = 0; col < dst->ne[0]; ++col) {
                index[0]      = col;
                const float x = modulate_value(dst->src[0], index);
                // Match separate ggml MUL/ADD rounding even with contraction enabled.
                volatile float product         = x * modulate_value(dst->src[1], index);
                volatile float scaled          = x + product;
                output[row * dst->ne[0] + col] = dst->src[2] ? scaled + modulate_value(dst->src[2], index) : scaled;
            }
        }
    }

    static ggml_status modulate_prepare(ggml_backend_t, ggml_tensor* const* nodes, int, void**, size_t* bytes, void*) {
        *bytes = sd_ggml_modulate_workspace_size(nodes[0]);
        return GGML_STATUS_SUCCESS;
    }

    static void modulate_destroy(void*, void*) {}

    static ggml_status modulate_execute(const ggml_ext_execution_context* context, ggml_tensor* const* nodes, int, void*, void*) {
        auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(context->backend));
        auto run = reinterpret_cast<ggml_ext_cpu_parallel_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_ext_cpu_parallel_v1"));
        GGML_ASSERT(run);
        auto* dst          = nodes[0];
        const auto address = reinterpret_cast<uintptr_t>(dst->data);
        bool alias         = false;
        for (const auto* input : dst->src) {
            if (input) {
                const auto src = reinterpret_cast<uintptr_t>(input->data);
                alias |= address < src + ggml_nbytes(input) && src < address + ggml_nbytes(dst);
            }
        }
        if (!alias) {
            return run(context->backend, dst, modulate_compute, nullptr);
        }
        ggml_tensor temporary = *dst;
        temporary.data        = ggml_backend_buffer_get_base(context->workspace);
        const auto status     = run(context->backend, &temporary, modulate_compute, nullptr);
        if (status == GGML_STATUS_SUCCESS) {
            std::memcpy(dst->data, temporary.data, ggml_nbytes(dst));
        }
        return status;
    }

    void register_modulate(ggml_backend_dev_t device) {
        ggml_ext_kernel_desc desc = {};
        desc.abi_version          = GGML_EXT_ABI_VERSION;
        desc.name                 = "sd.modulate.cpu.v1";
        desc.op                   = GGML_OP_CUSTOM;
        desc.custom_op            = sd_ggml_modulate_op();
        desc.priority             = 100;
        desc.supports             = modulate_supports;
        desc.prepare              = modulate_prepare;
        desc.destroy              = modulate_destroy;
        desc.execute              = modulate_execute;
        GGML_ASSERT(ggml_ext_register_kernel(device, &desc));
    }
}
#endif
