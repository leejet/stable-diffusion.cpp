#include <climits>
#include <cstdlib>
#include <cstring>
#include "bf16.h"
#include "core/ggml_extensions.h"
#include "rms_affine.h"

namespace SD_CUDA_NAMESPACE {
    namespace sd_rms_affine {
        static bool disabled() {
            static const bool value = [] {
                const char* fusion = std::getenv("GGML_CUDA_DISABLE_FUSION");
                return fusion && std::atoi(fusion);
            }();
            return value;
        }

        static bool f32(const ggml_tensor* t) {
            return t && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t);
        }

        static int modulate_match(ggml_backend_dev_t, const ggml_cgraph* graph, int first, void*) {
            auto* g = const_cast<ggml_cgraph*>(graph);
            if (disabled() || first + 2 >= ggml_graph_n_nodes(g))
                return 0;
            const auto* norm     = ggml_graph_node(g, first);
            const auto* affine   = ggml_graph_node(g, first + 1);
            const auto* modulate = ggml_graph_node(g, first + 2);
            if (norm->op != GGML_OP_RMS_NORM || affine->op != GGML_OP_MUL || affine->src[0] != norm ||
                !sd_ggml_modulate_describe(modulate) || modulate->src[0] != affine)
                return 0;
            const auto* x = norm->src[0];
            if (!f32(x) || !f32(norm) || !f32(affine) || x->ne[0] < 32 || x->ne[0] > 8192 ||
                ggml_nrows(x) > INT_MAX || !ggml_are_same_shape(x, norm) || !ggml_are_same_shape(x, affine))
                return 0;
            for (const auto* parameter : {affine->src[1], modulate->src[1], modulate->src[2]}) {
                if (parameter && (!f32(parameter) || parameter == norm || parameter == affine ||
                                  parameter->ne[0] != x->ne[0] || ggml_nelements(parameter) != x->ne[0]))
                    return 0;
            }
            return affine->src[1] ? 3 : 0;
        }

        static int match(ggml_backend_dev_t, const ggml_cgraph* graph, int first, void*) {
            auto* g = const_cast<ggml_cgraph*>(graph);
            if (disabled() || first + 3 >= ggml_graph_n_nodes(g))
                return 0;
            const auto* norm   = ggml_graph_node(g, first);
            const auto* affine = ggml_graph_node(g, first + 1);
            const auto* mul    = ggml_graph_node(g, first + 2);
            const auto* add    = ggml_graph_node(g, first + 3);
            if (norm->op != GGML_OP_RMS_NORM || affine->op != GGML_OP_MUL || affine->src[0] != norm ||
                mul->op != GGML_OP_MUL || mul->src[0] != affine || add->op != GGML_OP_ADD ||
                (add->src[0] != mul && add->src[1] != mul))
                return 0;
            const auto* x        = norm->src[0];
            const auto* residual = add->src[0] == mul ? add->src[1] : add->src[0];
            return f32(x) && f32(affine->src[1]) && f32(mul->src[1]) && f32(residual) && f32(add) &&
                           x->ne[0] >= 32 && x->ne[0] <= 8192 && ggml_nrows(x) <= INT_MAX &&
                           ggml_nelements(affine->src[1]) == x->ne[0] && ggml_nelements(mul->src[1]) == x->ne[0] &&
                           affine->src[1]->ne[0] == x->ne[0] && mul->src[1]->ne[0] == x->ne[0] &&
                           residual != mul && ggml_are_same_shape(residual, x) && ggml_are_same_shape(add, x)
                       ? 4
                       : 0;
        }

        static void optimize(ggml_backend_dev_t, ggml_cgraph* graph, void*) {
            if (disabled())
                return;
            for (int i = 2; i < ggml_graph_n_nodes(graph); ++i) {
                const auto* mul = ggml_graph_node(graph, i);
                if (!sd_ggml_modulate_describe(mul)) {
                    if (i + 1 >= ggml_graph_n_nodes(graph))
                        continue;
                    const auto* add = ggml_graph_node(graph, i + 1);
                    if (mul->op != GGML_OP_MUL || add->op != GGML_OP_ADD || (add->src[0] != mul && add->src[1] != mul))
                        continue;
                }
                const auto* affine = mul->src[0];
                if (!affine || affine->op != GGML_OP_MUL || !affine->src[0] || affine->src[0]->op != GGML_OP_RMS_NORM)
                    continue;
                for (int j = i - 2; j >= 0 && j >= i - 32; --j) {
                    if (ggml_graph_node(graph, j) == affine->src[0] && ggml_graph_node(graph, j + 1) == affine) {
                        if (j + 2 < i)
                            ggml_ext_graph_delay(graph, j, 2, i);
                        break;
                    }
                }
            }
        }

        static __device__ __forceinline__ float warp_sum(float value) {
            for (int offset = 16; offset > 0; offset >>= 1)
                value = __fadd_rn(value, cuda_shuffle_xor(value, offset, 32));
            return value;
        }

        enum class Kind { Residual,
                          SelfResidual,
                          Modulate };

        struct Operation {
            Kind kind = Kind::Residual;
            const ggml_tensor* input;
            const ggml_tensor* weight;
            const ggml_tensor* gate;
            const ggml_tensor* residual = nullptr;
            const ggml_tensor* shift    = nullptr;
            ggml_tensor* output;
            float eps;
        };

        static Operation describe(ggml_tensor* const* nodes, int count) {
            Operation op;
            op.input  = nodes[0]->src[0];
            op.weight = nodes[1]->src[1];
            op.gate   = nodes[2]->src[1];
            op.output = nodes[count - 1];
            std::memcpy(&op.eps, nodes[0]->op_params, sizeof(op.eps));
            if (count == 3) {
                op.kind  = Kind::Modulate;
                op.shift = op.output->src[2];
            } else {
                op.residual = op.output->src[0] == nodes[2] ? op.output->src[1] : op.output->src[0];
                if (op.residual == nodes[1]) {
                    op.kind     = Kind::SelfResidual;
                    op.residual = nullptr;
                }
            }
            return op;
        }

        template <int Block, int Groups, Kind Op, typename T>
        static __global__ void kernel(const float* x, const float* weight, const float* gate, const float* residual, const float* shift, T* dst, int width, float eps) {
            const int col0     = threadIdx.x;
            const int64_t base = int64_t(blockIdx.x) * width;
            float values[Groups][8];
            __shared__ float sums[32];
            const int lane = col0 % 32;
#pragma unroll
            for (int group = 0; group < Groups; ++group) {
                float sum = 0.f;
                for (int j = 0, col = col0 + group * Block; col < width; ++j, col += Block * Groups) {
                    values[group][j] = x[base + col];
                    sum              = __fmaf_rn(values[group][j], values[group][j], sum);
                }
                sum = warp_sum(sum);
                if (lane == 0)
                    sums[(col0 + group * Block) / 32] = sum;
            }
            // Groups preserve the native reduction tree when using fewer physical threads.
            __syncthreads();
            const float sum   = warp_sum(lane < Block * Groups / 32 ? sums[lane] : 0.f);
            const float scale = rsqrtf(sum / width + eps);
#pragma unroll
            for (int group = 0; group < Groups; ++group) {
                for (int j = 0, col = col0 + group * Block; col < width; ++j, col += Block * Groups) {
                    float value = __fmul_rn(__fmul_rn(scale, values[group][j]), weight[col]);
                    if constexpr (Op == Kind::Modulate) {
                        value = __fadd_rn(value, __fmul_rn(value, gate[col]));
                        if (shift)
                            value = __fadd_rn(value, shift[col]);
                    } else {
                        const float r = Op == Kind::SelfResidual ? value : residual[base + col];
                        value         = __fadd_rn(__fmul_rn(value, gate[col]), r);
                    }
                    dst[base + col] = T(value);
                }
            }
        }

        static bool overlaps(const ggml_tensor* a, const ggml_tensor* b) {
            if (!b)
                return false;
            const auto x = reinterpret_cast<uintptr_t>(a->data), y = reinterpret_cast<uintptr_t>(b->data);
            return x <= y ? y - x < ggml_nbytes(a) : x - y < ggml_nbytes(b);
        }

        static ggml_status prepare(ggml_backend_t, ggml_tensor* const* nodes, int count, void**, size_t* workspace, void*) {
            *workspace = ggml_nbytes(nodes[count - 1]);
            return GGML_STATUS_SUCCESS;
        }
        static void destroy(void*, void*) {}

        template <typename T, Kind Op>
        static CUDAError launch(CUDAStream stream, const Operation& op, T* output) {
            auto compute = kernel<256, 1, Op, T>;
            int threads  = 256;
            if (op.input->ne[0] >= 1024) {
                if constexpr (Op == Kind::Modulate) {
                    compute = kernel<512, 2, Op, T>;
                    threads = 512;
                } else {
                    compute = kernel<1024, 1, Op, T>;
                    threads = 1024;
                }
            }
            compute<<<unsigned(ggml_nrows(op.input)), threads, 0, stream>>>(
                static_cast<const float*>(op.input->data), static_cast<const float*>(op.weight->data),
                static_cast<const float*>(op.gate->data), op.residual ? static_cast<const float*>(op.residual->data) : nullptr,
                op.shift ? static_cast<const float*>(op.shift->data) : nullptr, output, int(op.input->ne[0]), op.eps);
            return cuda_last_error();
        }

        static ggml_status execute(const ggml_ext_execution_context* execution, ggml_tensor* const* nodes, int count, void*, void*) {
            ggml_ext_gpu_context context;
            if (!cuda_context(execution->backend, &context))
                return GGML_STATUS_FAILED;
            const auto op     = describe(nodes, count);
            auto* dst         = op.output;
            const bool alias  = (overlaps(dst, op.input) && dst->data != op.input->data) ||
                                (overlaps(dst, op.residual) && dst->data != op.residual->data) ||
                                overlaps(dst, op.weight) || overlaps(dst, op.gate) || overlaps(dst, op.shift);
            auto* output      = static_cast<float*>(alias ? ggml_backend_buffer_get_base(execution->workspace) : dst->data);
            const auto stream = static_cast<CUDAStream>(context.stream);
            CUDAError error   = cuda_invalid_value;
            switch (op.kind) {
                case Kind::Residual:
                    error = launch<float, Kind::Residual>(stream, op, output);
                    break;
                case Kind::SelfResidual:
                    error = launch<float, Kind::SelfResidual>(stream, op, output);
                    break;
                case Kind::Modulate:
                    error = launch<float, Kind::Modulate>(stream, op, output);
                    break;
            }
            if (error == cuda_success && alias)
                error = cuda_copy_async(dst->data, output, ggml_nbytes(dst), stream);
            return error == cuda_success ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
        }
    }

    int rms_affine_match(const ggml_cgraph* graph, int first) {
        return sd_rms_affine::match(nullptr, graph, first, nullptr);
    }

    ggml_status rms_affine_pack(CUDAStream stream, ggml_tensor* const* nodes, void* output) {
        const auto op    = sd_rms_affine::describe(nodes, 4);
        auto* dst        = static_cast<CUDABFloat16*>(output);
        const auto error = op.kind == sd_rms_affine::Kind::SelfResidual
                               ? sd_rms_affine::launch<CUDABFloat16, sd_rms_affine::Kind::SelfResidual>(stream, op, dst)
                               : sd_rms_affine::launch<CUDABFloat16, sd_rms_affine::Kind::Residual>(stream, op, dst);
        return error == cuda_success ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
    }

    void register_rms_affine(ggml_backend_dev_t device) {
        ggml_ext_kernel_desc desc = {};
        desc.abi_version          = GGML_EXT_ABI_VERSION;
        desc.name                 = "sd.rms_affine_residual." SD_CUDA_NAME ".v1";
        desc.op                   = GGML_OP_RMS_NORM;
        desc.priority             = 300;
        desc.match                = sd_rms_affine::match;
        desc.optimize             = sd_rms_affine::optimize;
        desc.prepare              = sd_rms_affine::prepare;
        desc.destroy              = sd_rms_affine::destroy;
        desc.execute              = sd_rms_affine::execute;
        GGML_ASSERT(ggml_ext_register_kernel_ex(device, &desc, GGML_EXT_KERNEL_CAPTURE_SAFE | GGML_EXT_KERNEL_REPLAY_REUSABLE));
        desc.name = "sd.rms_affine_modulate." SD_CUDA_NAME ".v1";
        // Prefer the existing fusion that also produces ConvRot input when available.
        desc.priority = 200;
        desc.match    = sd_rms_affine::modulate_match;
        desc.optimize = nullptr;
        GGML_ASSERT(ggml_ext_register_kernel_ex(device, &desc, GGML_EXT_KERNEL_CAPTURE_SAFE | GGML_EXT_KERNEL_REPLAY_REUSABLE));
    }
}
