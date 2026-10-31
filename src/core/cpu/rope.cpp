#include "core/cpu/registration.h"
#include "core/ggml_extensions.h"

#ifndef SD_USE_UPSTREAM_GGML
#include <cmath>
#include <cstring>

namespace sd_cpu {

    static bool rope_supports(ggml_backend_dev_t, const ggml_tensor* node, void*) {
        return sd_ggml_rope_describe(node);
    }

    struct RopeExecution {
        const ggml_tensor* rope;
        const ggml_tensor* norm;
        const ggml_tensor* weight;
    };

    static void rope_compute(ggml_tensor* dst, int ith, int nth, void* user_data) {
        const auto& execution = *static_cast<const RopeExecution*>(user_data);
        const auto* rope      = execution.rope;
        const auto* x         = execution.norm ? execution.norm->src[0] : rope->src[0];
        const auto* pe        = rope->src[1];
        int32_t interleaved;
        std::memcpy(&interleaved, ggml_ext_op_attributes(rope), sizeof(interleaved));
        float eps = 0.0f;
        if (execution.norm) {
            std::memcpy(&eps, execution.norm->op_params, sizeof(eps));
        }
        const int64_t half_dim = pe->ne[2];
        const int64_t rows     = x->ne[1] * x->ne[2] * x->ne[3];
        for (int64_t row = ith; row < rows; row += nth) {
            const int64_t token = row % x->ne[2];
            const int64_t head  = (row / x->ne[2]) % x->ne[1];
            const int64_t batch = row / (x->ne[2] * x->ne[1]);
            const char* input   = static_cast<const char*>(x->data) + head * x->nb[1] + token * x->nb[2] + batch * x->nb[3];
            const char* pos     = static_cast<const char*>(pe->data) + token * pe->nb[3];
            float* output       = static_cast<float*>(dst->data) + row * x->ne[0];
            float scale         = 1.0f;
            if (execution.norm) {
                double sum = 0.0;
                for (int64_t col = 0; col < x->ne[0]; ++col) {
                    const float v = *reinterpret_cast<const float*>(input + col * x->nb[0]);
                    sum += static_cast<double>(v * v);
                }
                const float mean = static_cast<float>(sum / x->ne[0]);
                scale            = 1.0f / std::sqrt(mean + eps);
            }
            auto value = [&](int64_t col) {
                float v = *reinterpret_cast<const float*>(input + col * x->nb[0]);
                if (execution.norm) {
                    const auto* weight = static_cast<const char*>(execution.weight->data);
                    v                  = (v * scale) * *reinterpret_cast<const float*>(weight + col * execution.weight->nb[0]);
                }
                return v;
            };
            for (int64_t j = 0; j < half_dim; ++j) {
                const int64_t i0 = interleaved ? 2 * j : j;
                const int64_t i1 = interleaved ? 2 * j + 1 : j + half_dim;
                const float a    = value(i0);
                const float b    = value(i1);
                const char* m    = pos + j * pe->nb[2];
                const float c    = *reinterpret_cast<const float*>(m);
                const float s    = *reinterpret_cast<const float*>(m + pe->nb[1]);
                if (interleaved) {
                    const float neg_s = *reinterpret_cast<const float*>(m + pe->nb[0]);
                    const float c1    = *reinterpret_cast<const float*>(m + pe->nb[0] + pe->nb[1]);
                    output[i0]        = a * c + b * neg_s;
                    output[i1]        = a * s + b * c1;
                } else {
                    output[i0] = a * c - b * s;
                    output[i1] = a * s + b * c;
                }
            }
            for (int64_t col = 2 * half_dim; col < x->ne[0]; ++col) {
                output[col] = value(col);
            }
        }
    }

    static int rope_fusion_match(ggml_backend_dev_t, const ggml_cgraph* graph, int first, void*) {
        auto* view = const_cast<ggml_cgraph*>(graph);
        if (first + 2 >= ggml_graph_n_nodes(view)) {
            return 0;
        }
        const auto* norm = ggml_graph_node(view, first);
        const auto* mul  = ggml_graph_node(view, first + 1);
        const auto* rope = ggml_graph_node(view, first + 2);
        if (norm->op != GGML_OP_RMS_NORM || norm->type != GGML_TYPE_F32 || !norm->src[0] ||
            norm->src[0]->type != GGML_TYPE_F32 || norm->src[0]->nb[0] != sizeof(float) ||
            mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32 || rope->src[0] != mul ||
            ggml_ext_op_get(rope) != sd_ggml_rope_op() ||
            !sd_ggml_rope_describe(rope) || !ggml_are_same_shape(norm, mul)) {
            return 0;
        }
        const auto* weight = mul->src[0] == norm ? mul->src[1] : (mul->src[1] == norm ? mul->src[0] : nullptr);
        if (!weight || weight->type != GGML_TYPE_F32 || weight->ne[0] != norm->ne[0] ||
            weight->ne[1] != 1 || weight->ne[2] != 1 || weight->ne[3] != 1) {
            return 0;
        }
        return 3;
    }

    static ggml_status rope_execute(const ggml_ext_execution_context* context, ggml_tensor* const* nodes, int count, void*, void*) {
        auto device = ggml_backend_get_device(context->backend);
        auto reg    = ggml_backend_dev_backend_reg(device);
        auto run    = reinterpret_cast<ggml_ext_cpu_parallel_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_ext_cpu_parallel_v1"));
        GGML_ASSERT(run);
        RopeExecution execution = {nodes[count - 1], nullptr, nullptr};
        if (count == 3) {
            execution.norm   = nodes[0];
            execution.weight = nodes[1]->src[0] == nodes[0] ? nodes[1]->src[1] : nodes[1]->src[0];
        }
        return run(context->backend, nodes[count - 1], rope_compute, &execution);
    }

    void register_rope(ggml_backend_dev_t device) {
        ggml_ext_kernel_desc desc = {};
        desc.abi_version          = GGML_EXT_ABI_VERSION;
        desc.name                 = "sd.rope_apply.cpu.v1";
        desc.op                   = GGML_OP_CUSTOM;
        desc.custom_op            = sd_ggml_rope_op();
        desc.priority             = 100;
        desc.supports             = rope_supports;
        desc.execute              = rope_execute;
        GGML_ASSERT(ggml_ext_register_kernel(device, &desc));

        desc.name      = "sd.rms_norm_mul_rope.cpu.v1";
        desc.op        = GGML_OP_RMS_NORM;
        desc.custom_op = nullptr;
        desc.priority  = 200;
        desc.supports  = nullptr;
        desc.match     = rope_fusion_match;
        GGML_ASSERT(ggml_ext_register_kernel(device, &desc));
    }

}
#endif
