#include "core/cuda/rope.h"
#include "core/ggml_extensions.h"

#include <climits>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace SD_CUDA_NAMESPACE {
    static __device__ __forceinline__ float sd_warp_sum(float v) {
        for (int mask = 16; mask > 0; mask >>= 1)
            v = __fadd_rn(v, cuda_shuffle_xor(v, mask));
        return v;
    }
    template <int D, bool interleaved, typename T>
    static __global__ void rms_norm_mul_rope_apply_fused(
        const float* x,
        const float* weight,
        const float* pe,
        T* dst,
        int64_t heads,
        int64_t tokens,
        int64_t rows,
        int64_t nb1,
        int64_t nb2,
        int64_t nb3,
        int rot_half,
        float eps,
        float output_scale) {
        const int lane    = threadIdx.x;
        const int64_t row = int64_t(blockIdx.x) * blockDim.y + threadIdx.y;
        if (row >= rows) {
            return;
        }
        const int64_t head  = row % heads;
        const int64_t token = (row / heads) % tokens;
        const int64_t batch = row / (heads * tokens);
        const float* x_row  = x + head * nb1 + token * nb2 + batch * nb3;
        float v[D / 32], sums[8] = {};
#pragma unroll
        for (int j = 0; j < D / 32; ++j) {
            v[j]    = x_row[j * 32 + lane];
            sums[j] = sd_warp_sum(__fmul_rn(v[j], v[j]));
        }
        // Preserve the 256-thread RMSNorm kernel's reduction tree, including inactive warps.
        const float sum = __fadd_rn(
            __fadd_rn(__fadd_rn(sums[0], sums[4]), __fadd_rn(sums[2], sums[6])),
            __fadd_rn(__fadd_rn(sums[1], sums[5]), __fadd_rn(sums[3], sums[7])));
        const float scale = rsqrtf(sum / float(D) + eps);
        T* out            = dst + ((batch * heads + head) * tokens + token) * D;
#pragma unroll
        for (int j = 0; j < D / 32; ++j) {
            const int col     = j * 32 + lane;
            const float value = __fmul_rn(__fmul_rn(scale, v[j]), weight[col]);
            float rotated     = value;
            if constexpr (interleaved) {
                const float other = cuda_shuffle_xor(value, 1, 32);
                if (col < 2 * rot_half) {
                    const float* m      = pe + 4 * (token * rot_half + col / 2);
                    const float4 matrix = make_float4(m[0], m[1], m[2], m[3]);
                    rotated             = lane % 2 == 0
                                              ? __fadd_rn(__fmul_rn(value, matrix.x), __fmul_rn(other, matrix.y))
                                              : __fadd_rn(__fmul_rn(other, matrix.z), __fmul_rn(value, matrix.w));
                }
            } else {
                if (col < 2 * rot_half) {
                    const int partner   = col < rot_half ? col + rot_half : col - rot_half;
                    const float other   = __fmul_rn(__fmul_rn(scale, x_row[partner]), weight[partner]);
                    const float* m      = pe + 4 * (token * rot_half + col % rot_half);
                    const float4 matrix = make_float4(m[0], m[1], m[2], m[3]);
                    rotated             = col < rot_half
                                              ? __fsub_rn(__fmul_rn(value, matrix.x), __fmul_rn(other, matrix.z))
                                              : __fadd_rn(__fmul_rn(other, matrix.z), __fmul_rn(value, matrix.x));
                }
            }
            if constexpr (std::is_same<T, half>::value) {
                out[col] = __float2half_rn(__fmul_rn(rotated, output_scale));
            } else {
                out[col] = rotated;
            }
        }
    }

    struct RopeApplyStrides {
        int64_t x[4];
        int64_t pe[4];
    };

    template <bool interleaved>
    static __global__ void rope_apply_f32(
        const float* x,
        const float* pe,
        float* dst,
        int64_t half_dim,
        int64_t rot_half,
        int64_t heads,
        int64_t tokens,
        int64_t pairs,
        RopeApplyStrides strides) {
        const int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= pairs) {
            return;
        }
        const int64_t j     = index % half_dim;
        const int64_t row   = index / half_dim;
        const int64_t token = row % tokens;
        const int64_t head  = (row / tokens) % heads;
        const int64_t batch = row / (tokens * heads);
        const int64_t i0    = interleaved || j >= rot_half ? 2 * j : j;
        const int64_t i1    = interleaved || j >= rot_half ? 2 * j + 1 : j + rot_half;
        const float* x_row  = x + head * strides.x[1] + token * strides.x[2] + batch * strides.x[3];
        const float a       = x_row[i0 * strides.x[0]];
        const float b       = x_row[i1 * strides.x[0]];
        float* out          = dst + row * half_dim * 2;
        if (j >= rot_half) {
            out[i0] = a;
            out[i1] = b;
            return;
        }
        const float* matrix = pe + j * strides.pe[2] + token * strides.pe[3];
        const float c       = matrix[0];
        const float s       = matrix[strides.pe[1]];
        // Match the separate mul/add rounding of the unfused graph.
        if (interleaved) {
            out[i0] = __fadd_rn(__fmul_rn(a, c), __fmul_rn(b, matrix[strides.pe[0]]));
            out[i1] = __fadd_rn(__fmul_rn(a, s), __fmul_rn(b, matrix[strides.pe[0] + strides.pe[1]]));
        } else {
            out[i0] = __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s));
            out[i1] = __fadd_rn(__fmul_rn(a, s), __fmul_rn(b, c));
        }
    }

    static bool rope_interleaved(const ggml_tensor* rope) {
        int32_t value;
        std::memcpy(&value, ggml_ext_op_attributes(rope), sizeof(value));
        return value != 0;
    }

    static float param_f32(const ggml_tensor* node, int index) {
        float value;
        std::memcpy(&value, node->op_params + index, sizeof(value));
        return value;
    }

    static bool rope_is_node(const ggml_tensor* node) {
        return node && ggml_ext_op_get(node) == sd_ggml_rope_op();
    }

    static bool rope_supports(ggml_backend_dev_t, const ggml_tensor* rope, void*) {
        const auto* x  = rope->src[0];
        const auto* pe = rope->src[1];
        if (!x || !pe || x->type != GGML_TYPE_F32 || pe->type != GGML_TYPE_F32 || rope->type != GGML_TYPE_F32 ||
            x->ne[0] % 2 || pe->ne[0] != 2 || pe->ne[1] != 2 || pe->ne[2] <= 0 || pe->ne[2] > x->ne[0] / 2 ||
            pe->ne[3] != x->ne[2] || !ggml_is_contiguous(rope) || rope->ne[0] != x->ne[0] ||
            rope->ne[1] != x->ne[2] || rope->ne[2] != x->ne[1] * x->ne[3] || rope->ne[3] != 1 ||
            (ggml_nelements(x) / 2 + 255) / 256 > INT_MAX) {
            return false;
        }
        for (int i = 0; i < 4; ++i) {
            if (x->nb[i] % sizeof(float) || pe->nb[i] % sizeof(float))
                return false;
        }
        return true;
    }

    bool rope_fusion_supports(const ggml_tensor* norm, const ggml_tensor* mul, const ggml_tensor* rope) {
        if (norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || !rope_is_node(rope) ||
            mul->src[0] != norm || rope->src[0] != mul || !rope_supports(nullptr, rope, nullptr))
            return false;
        const auto* x      = norm->src[0];
        const auto* weight = mul->src[1];
        return x && weight && x->ne[0] >= 32 && x->ne[0] <= 256 && x->ne[0] % 32 == 0 &&
               x->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_F32 && norm->type == GGML_TYPE_F32 &&
               ggml_are_same_shape(norm, mul) && ggml_are_same_shape(x, norm) &&
               weight->ne[0] == x->ne[0] && ggml_nelements(weight) == x->ne[0] &&
               x->nb[0] == sizeof(float) && x->nb[1] % sizeof(float) == 0 && x->nb[2] % sizeof(float) == 0 &&
               x->nb[3] % sizeof(float) == 0 && ggml_is_contiguous(weight) && ggml_is_contiguous(rope->src[1]) &&
               (ggml_nrows(x) + 3) / 4 <= INT_MAX;
    }

    template <typename T, bool Interleaved>
    static CUDAError rms_rope_launch(CUDAStream stream, ggml_tensor* const* nodes, T* output, float scale) {
        const auto* x      = nodes[0]->src[0];
        const auto* pe     = nodes[2]->src[1];
        const int64_t rows = ggml_nrows(x);
#define SD_ROPE(D)                                                                                                               \
    case D:                                                                                                                      \
        rms_norm_mul_rope_apply_fused<D, Interleaved, T><<<static_cast<unsigned int>((rows + 3) / 4), dim3(32, 4), 0, stream>>>( \
            static_cast<const float*>(x->data), static_cast<const float*>(nodes[1]->src[1]->data),                               \
            static_cast<const float*>(pe->data), output, x->ne[1], x->ne[2], rows,                                               \
            x->nb[1] / 4, x->nb[2] / 4, x->nb[3] / 4, int(pe->ne[2]), param_f32(nodes[0], 0), scale);                            \
        break
        switch (x->ne[0]) {
            SD_ROPE(32);
            SD_ROPE(64);
            SD_ROPE(96);
            SD_ROPE(128);
            SD_ROPE(160);
            SD_ROPE(192);
            SD_ROPE(224);
            SD_ROPE(256);
            default:
                return cuda_invalid_value;
        }
#undef SD_ROPE
        return cuda_last_error();
    }

    CUDAError rms_rope_launch(CUDAStream stream, ggml_tensor* const* nodes, void* output, bool fp16, float scale) {
        if (fp16) {
            return rope_interleaved(nodes[2]) ? rms_rope_launch<half, true>(stream, nodes, static_cast<half*>(output), scale)
                                              : rms_rope_launch<half, false>(stream, nodes, static_cast<half*>(output), scale);
        }
        return rope_interleaved(nodes[2]) ? rms_rope_launch<float, true>(stream, nodes, static_cast<float*>(output), scale)
                                          : rms_rope_launch<float, false>(stream, nodes, static_cast<float*>(output), scale);
    }

    static int rope_match(ggml_backend_dev_t, const ggml_cgraph* graph, int first, void*) {
        static const bool disabled = [] {const char* value=std::getenv("GGML_CUDA_DISABLE_FUSION"); return value && std::atoi(value)!=0; }();
        if (disabled)
            return 0;
        auto* g = const_cast<ggml_cgraph*>(graph);
        if (first + 2 >= ggml_graph_n_nodes(g) || !rope_fusion_supports(
                                                      ggml_graph_node(g, first), ggml_graph_node(g, first + 1), ggml_graph_node(g, first + 2)))
            return 0;
        if (first + 4 < ggml_graph_n_nodes(g)) {
            const auto* rope  = ggml_graph_node(g, first + 2);
            const auto* scale = ggml_graph_node(g, first + 3);
            const auto* copy  = ggml_graph_node(g, first + 4);
            if (scale->op == GGML_OP_SCALE && copy->op == GGML_OP_CPY && scale->src[0] == rope && copy->src[0] == scale &&
                scale->type == GGML_TYPE_F32 && copy->type == GGML_TYPE_F16 && param_f32(scale, 1) == 0.f &&
                ggml_are_same_shape(rope, scale) && ggml_are_same_shape(rope, copy) && ggml_is_contiguous(copy))
                return 5;
        }
        return 3;
    }

    static void rope_optimize(ggml_backend_dev_t, ggml_cgraph* graph, void*) {
        static const bool disabled = [] {
            const char* value = std::getenv("GGML_CUDA_DISABLE_FUSION");
            return value && std::atoi(value) != 0;
        }();
        if (disabled)
            return;
        const int total = ggml_graph_n_nodes(graph);
        for (int i = 2; i + 1 < total; ++i) {
            const auto* mul  = ggml_graph_node(graph, i);
            const auto* rope = ggml_graph_node(graph, i + 1);
            if (mul->op != GGML_OP_MUL || !rope_is_node(rope) || rope->src[0] != mul ||
                !mul->src[0] || mul->src[0]->op != GGML_OP_RMS_NORM)
                continue;
            const auto* norm = mul->src[0];
            for (int j = i - 1; j >= 0 && j >= i - 64; --j) {
                if (ggml_graph_node(graph, j) != norm)
                    continue;
                if (j + 1 < i) {
                    if (j >= 2) {
                        const auto* mat  = ggml_graph_node(graph, j - 2);
                        const auto* view = ggml_graph_node(graph, j - 1);
                        if (ggml_ext_op_get(mat) == sd_ggml_i8_op() &&
                            view->op == GGML_OP_RESHAPE && view->src[0] == mat && norm->src[0] == view &&
                            ggml_ext_graph_delay(graph, j - 2, 3, i))
                            break;
                    }
                    ggml_ext_graph_delay(graph, j, 1, i);
                }
                break;
            }
        }
    }

    static ggml_status rope_prepare(ggml_backend_t, ggml_tensor* const* nodes, int count, void**, size_t* workspace, void*) {
        *workspace = ggml_nbytes(nodes[count - 1]);
        return GGML_STATUS_SUCCESS;
    }

    static void rope_destroy(void*, void*) {}

    static bool overlaps(const ggml_tensor* a, const ggml_tensor* b) {
        const auto x = reinterpret_cast<uintptr_t>(a->data), y = reinterpret_cast<uintptr_t>(b->data);
        return x <= y ? y - x < ggml_nbytes(a) : x - y < ggml_nbytes(b);
    }

    static ggml_status rope_execute(const ggml_ext_execution_context* execution, ggml_tensor* const* nodes, int count, void*, void*) {
        ggml_ext_gpu_context context;
        if (!cuda_context(execution->backend, &context))
            return GGML_STATUS_FAILED;
        auto stream      = static_cast<CUDAStream>(context.stream);
        const auto* rope = nodes[count == 1 ? 0 : 2];
        const auto* x    = count == 1 ? rope->src[0] : nodes[0]->src[0];
        auto* dst        = nodes[count - 1];
        const bool alias = overlaps(dst, x) || overlaps(dst, rope->src[1]) || (count > 1 && overlaps(dst, nodes[1]->src[1]));
        void* output     = alias ? ggml_backend_buffer_get_base(execution->workspace) : dst->data;
        CUDAError error;
        if (count > 1) {
            error = rms_rope_launch(stream, nodes, output, count == 5, count == 5 ? param_f32(nodes[3], 0) : 1.f);
        } else {
            RopeApplyStrides strides;
            const auto* pe = rope->src[1];
            for (int i = 0; i < 4; ++i) {
                strides.x[i]  = x->nb[i] / 4;
                strides.pe[i] = pe->nb[i] / 4;
            }
            const int64_t pairs = ggml_nelements(x) / 2;
            auto kernel         = rope_interleaved(rope) ? rope_apply_f32<true> : rope_apply_f32<false>;
            kernel<<<static_cast<unsigned int>((pairs + 255) / 256), 256, 0, stream>>>(static_cast<const float*>(x->data), static_cast<const float*>(pe->data),
                                                                                       static_cast<float*>(output), x->ne[0] / 2, pe->ne[2], x->ne[1], x->ne[2], pairs, strides);
            error = cuda_last_error();
        }
        if (error == cuda_success && alias)
            error = cuda_copy_async(dst->data, output, ggml_nbytes(dst), stream);
        return error == cuda_success ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
    }

    void register_rope(ggml_backend_dev_t device) {
        ggml_ext_kernel_desc desc = {};
        desc.abi_version          = GGML_EXT_ABI_VERSION;
        desc.name                 = "sd.rope_apply." SD_CUDA_NAME ".v1";
        desc.op                   = GGML_OP_CUSTOM;
        desc.custom_op            = sd_ggml_rope_op();
        desc.priority             = 100;
        desc.supports             = rope_supports;
        desc.prepare              = rope_prepare;
        desc.destroy              = rope_destroy;
        desc.execute              = rope_execute;
        GGML_ASSERT(ggml_ext_register_kernel_ex(device, &desc, GGML_EXT_KERNEL_CAPTURE_SAFE | GGML_EXT_KERNEL_REPLAY_REUSABLE));
        desc.name      = "sd.rms_norm_mul_rope." SD_CUDA_NAME ".v1";
        desc.op        = GGML_OP_RMS_NORM;
        desc.custom_op = nullptr;
        desc.priority  = 250;
        desc.supports  = nullptr;
        desc.match     = rope_match;
        desc.optimize  = rope_optimize;
        GGML_ASSERT(ggml_ext_register_kernel_ex(device, &desc, GGML_EXT_KERNEL_CAPTURE_SAFE | GGML_EXT_KERNEL_REPLAY_REUSABLE));
    }

}
