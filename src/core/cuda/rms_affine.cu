#include <climits>
#include <cstdlib>
#include <cstring>
#include "bf16.h"
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
            for (int i = 2; i + 1 < ggml_graph_n_nodes(graph); ++i) {
                const auto* mul = ggml_graph_node(graph, i);
                const auto* add = ggml_graph_node(graph, i + 1);
                if (mul->op != GGML_OP_MUL || add->op != GGML_OP_ADD || (add->src[0] != mul && add->src[1] != mul))
                    continue;
                const auto* affine = mul->src[0];
                if (affine->op != GGML_OP_MUL || affine->src[0]->op != GGML_OP_RMS_NORM)
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

        template <int Block, bool Modulate, typename T = float>
        static __global__ void kernel(const float* x, const float* weight, const float* gate, const float* residual, T* dst, int width, float eps) {
            const int col0     = threadIdx.x;
            const int64_t base = int64_t(blockIdx.x) * width;
            float values[8], sum = 0.f;
            for (int j = 0, col = col0; col < width; ++j, col += Block) {
                values[j] = x[base + col];
                sum       = __fmaf_rn(values[j], values[j], sum);
            }
            sum = warp_sum(sum);
            __shared__ float sums[32];
            const int lane = col0 % 32;
            if (lane == 0)
                sums[col0 / 32] = sum;
            __syncthreads();
            sum               = warp_sum(lane < Block / 32 ? sums[lane] : 0.f);
            const float scale = rsqrtf(sum / width + eps);
            for (int j = 0, col = col0; col < width; ++j, col += Block) {
                const float value = __fmul_rn(__fmul_rn(scale, values[j]), weight[col]);
                const float r     = Modulate ? value : residual[base + col];
                dst[base + col]   = T(__fadd_rn(__fmul_rn(value, gate[col]), r));
            }
        }

        static bool overlaps(const ggml_tensor* a, const ggml_tensor* b) {
            const auto x = reinterpret_cast<uintptr_t>(a->data), y = reinterpret_cast<uintptr_t>(b->data);
            return x <= y ? y - x < ggml_nbytes(a) : x - y < ggml_nbytes(b);
        }

        static ggml_status prepare(ggml_backend_t, ggml_tensor* const* nodes, int, void**, size_t* workspace, void*) {
            *workspace = ggml_nbytes(nodes[3]);
            return GGML_STATUS_SUCCESS;
        }
        static void destroy(void*, void*) {}

        static ggml_status execute(const ggml_ext_execution_context* execution, ggml_tensor* const* nodes, int, void*, void*) {
            ggml_ext_gpu_context context;
            if (!cuda_context(execution->backend, &context))
                return GGML_STATUS_FAILED;
            const auto* x         = nodes[0]->src[0];
            const auto* w         = nodes[1]->src[1];
            const auto* gate      = nodes[2]->src[1];
            auto* dst             = nodes[3];
            const auto* residual  = dst->src[0] == nodes[2] ? dst->src[1] : dst->src[0];
            const bool modulation = residual == nodes[1];
            const bool alias      = (overlaps(dst, x) && dst->data != x->data) || overlaps(dst, w) || overlaps(dst, gate) ||
                               (!modulation && overlaps(dst, residual) && dst->data != residual->data);
            auto* output = static_cast<float*>(alias ? ggml_backend_buffer_get_base(execution->workspace) : dst->data);
            float eps;
            std::memcpy(&eps, nodes[0]->op_params, sizeof(eps));
            auto launch       = x->ne[0] < 1024 ? (modulation ? kernel<256, true> : kernel<256, false>)
                                                : (modulation ? kernel<1024, true> : kernel<1024, false>);
            const auto stream = static_cast<CUDAStream>(context.stream);
            const int threads = x->ne[0] < 1024 ? 256 : 1024;
            launch<<<unsigned(ggml_nrows(x)), threads, 0, stream>>>(
                static_cast<const float*>(x->data), static_cast<const float*>(w->data), static_cast<const float*>(gate->data),
                modulation ? nullptr : static_cast<const float*>(residual->data), output, int(x->ne[0]), eps);
            auto error = cuda_last_error();
            if (error == cuda_success && alias)
                error = cuda_copy_async(dst->data, output, ggml_nbytes(dst), stream);
            return error == cuda_success ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
        }
    }

    int rms_affine_match(const ggml_cgraph* graph, int first) {
        return sd_rms_affine::match(nullptr, graph, first, nullptr);
    }

    ggml_status rms_affine_pack(CUDAStream stream, ggml_tensor* const* nodes, void* output) {
        const auto* x         = nodes[0]->src[0];
        const auto* residual  = nodes[3]->src[0] == nodes[2] ? nodes[3]->src[1] : nodes[3]->src[0];
        const bool modulation = residual == nodes[1];
        auto launch           = x->ne[0] < 1024 ? (modulation ? sd_rms_affine::kernel<256, true, CUDABFloat16> : sd_rms_affine::kernel<256, false, CUDABFloat16>)
                                                : (modulation ? sd_rms_affine::kernel<1024, true, CUDABFloat16> : sd_rms_affine::kernel<1024, false, CUDABFloat16>);
        float eps;
        std::memcpy(&eps, nodes[0]->op_params, sizeof(eps));
        const int threads = x->ne[0] < 1024 ? 256 : 1024;
        launch<<<unsigned(ggml_nrows(x)), threads, 0, stream>>>(
            static_cast<const float*>(x->data), static_cast<const float*>(nodes[1]->src[1]->data),
            static_cast<const float*>(nodes[2]->src[1]->data), modulation ? nullptr : static_cast<const float*>(residual->data),
            static_cast<CUDABFloat16*>(output), int(x->ne[0]), eps);
        return cuda_last_error() == cuda_success ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
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
    }
}
