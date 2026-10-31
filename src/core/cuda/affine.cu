#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <climits>
#include <cstdlib>
#include <cstring>
#include "ggml-ext.h"

namespace sd_affine_cuda {
    static bool disabled() {
        static const bool value = [] { const char* v = std::getenv("GGML_CUDA_DISABLE_FUSION"); return v && std::atoi(v) != 0; }();
        return value;
    }
    static bool aligned(const ggml_tensor* tensor, size_t alignment) {
        size_t offset = 0;
        while (!tensor->data && tensor->view_src) {
            offset += tensor->view_offs;
            tensor = tensor->view_src;
        }
        return (reinterpret_cast<uintptr_t>(tensor->data) + offset) % alignment == 0;
    }
    static float param(const ggml_tensor* tensor, int index) {
        float value;
        std::memcpy(&value, tensor->op_params + index, sizeof(value));
        return value;
    }
    static bool contiguous_f32(const ggml_tensor* tensor) {
        return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) && aligned(tensor, 16);
    }
    static bool bounded(const ggml_tensor* tensor) {
        return ggml_nelements(tensor) > 0 && ggml_nelements(tensor) <= int64_t(INT_MAX) * 1024;
    }
    static int match(ggml_backend_dev_t, const ggml_cgraph* graph, int first, void*) {
        auto* g = const_cast<ggml_cgraph*>(graph);
        if (disabled() || first + 1 >= ggml_graph_n_nodes(g))
            return 0;
        const auto* node = ggml_graph_node(g, first);
        const auto* next = ggml_graph_node(g, first + 1);
        if (node->op == GGML_OP_MUL) {
            if (next->op != GGML_OP_ADD || (next->src[0] != node && next->src[1] != node))
                return 0;
            const auto* x        = node->src[0];
            const auto* gate     = node->src[1];
            const auto* residual = next->src[0] == node ? next->src[1] : next->src[0];
            return residual != node && contiguous_f32(x) && contiguous_f32(gate) && contiguous_f32(residual) && contiguous_f32(next) &&
                           node->type == GGML_TYPE_F32 && ggml_are_same_shape(node, next) && ggml_are_same_shape(x, next) &&
                           ggml_are_same_shape(residual, next) && gate->ne[0] == x->ne[0] && ggml_nelements(gate) == x->ne[0] &&
                           x->ne[0] % 4 == 0 && bounded(next)
                       ? 2
                       : 0;
        }
        if (node->op == GGML_OP_CONCAT) {
            const bool permuted = next->op == GGML_OP_PERMUTE;
            const int count     = permuted ? 3 : 2;
            if (first + count > ggml_graph_n_nodes(g) || node->type != GGML_TYPE_F32 || node->ne[0] % 4 ||
                node->op_params[0] != (permuted ? 2 : 1) || !ggml_is_contiguous(node) ||
                !contiguous_f32(node->src[0]) || !contiguous_f32(node->src[1]))
                return 0;
            if (permuted && (next->src[0] != node || next->ne[0] != node->ne[0] || next->ne[1] != node->ne[2] ||
                             next->ne[2] != node->ne[1] || next->ne[3] != node->ne[3] || next->nb[0] != node->nb[0] ||
                             next->nb[1] != node->nb[2] || next->nb[2] != node->nb[1] || next->nb[3] != node->nb[3]))
                return 0;
            const auto* dst = ggml_graph_node(g, first + count - 1);
            return dst->op == GGML_OP_CPY && dst->src[0] == (permuted ? next : node) && dst->type == GGML_TYPE_F16 &&
                           ggml_is_contiguous(dst) && ggml_are_same_shape(dst, dst->src[0]) && aligned(dst, 4) && bounded(dst)
                       ? count
                       : 0;
        }
        if (node->op == GGML_OP_CONT && first + 3 < ggml_graph_n_nodes(g)) {
            const auto* src     = node->src[0];
            const auto* permute = ggml_graph_node(g, first + 2);
            const auto* dst     = ggml_graph_node(g, first + 3);
            return ggml_are_same_shape(src, node) && src->type == GGML_TYPE_F32 && src->nb[0] == 4 && src->ne[0] % 4 == 0 &&
                           aligned(src, 16) && (src->nb[1] | src->nb[2] | src->nb[3]) % 16 == 0 &&
                           next->op == GGML_OP_SCALE && next->src[0] == node && next->type == GGML_TYPE_F32 && param(next, 1) == 0.f &&
                           permute->op == GGML_OP_PERMUTE && permute->src[0] == next && permute->op_params[0] == 0 &&
                           permute->op_params[1] == 2 && permute->op_params[2] == 1 && permute->op_params[3] == 3 &&
                           dst->op == GGML_OP_CPY && dst->src[0] == permute && dst->type == GGML_TYPE_F16 &&
                           ggml_are_same_shape(dst, permute) && ggml_is_contiguous(dst) && aligned(dst, 4) && bounded(dst)
                       ? 4
                       : 0;
        }
        return 0;
    }
    static __global__ void gated_residual(const float4* x, const float4* gate, const float4* residual, float4* dst, int64_t n4, int64_t count4) {
        const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= count4)
            return;
        const float4 a = x[i], g = gate[i % n4], r = residual[i];
        // The graph performs two FP32 operations; contracting them into FMA changes rounding.
        dst[i] = make_float4(__fadd_rn(__fmul_rn(a.x, g.x), r.x), __fadd_rn(__fmul_rn(a.y, g.y), r.y),
                             __fadd_rn(__fmul_rn(a.z, g.z), r.z), __fadd_rn(__fmul_rn(a.w, g.w), r.w));
    }
    template <bool Permuted>
    static __global__ void concat_cast(const float4* prefix, const float4* current, half2* dst, int64_t columns4, int64_t prefix_tokens, int64_t tokens, int64_t heads, int64_t count4) {
        const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= count4)
            return;
        const int64_t col = i % columns4, row = i / columns4;
        const int64_t token = row % tokens, head = row / tokens % heads, batch = row / (tokens * heads);
        const bool from_prefix   = token < prefix_tokens;
        const int64_t src_tokens = from_prefix ? prefix_tokens : tokens - prefix_tokens;
        const int64_t src_token  = from_prefix ? token : token - prefix_tokens;
        const int64_t src_row    = Permuted ? (batch * src_tokens + src_token) * heads + head
                                            : (batch * heads + head) * src_tokens + src_token;
        const float4 value       = (from_prefix ? prefix : current)[src_row * columns4 + col];
        dst[2 * i]               = __floats2half2_rn(value.x, value.y);
        dst[2 * i + 1]           = __floats2half2_rn(value.z, value.w);
    }
    static __global__ void scale_permute(const float* src, half2* dst, int64_t dim, int64_t tokens, int64_t heads, int64_t count, int64_t nb1, int64_t nb2, int64_t nb3, float scale) {
        const int64_t index = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
        if (index >= count)
            return;
        const int64_t col = index % dim, row = index / dim;
        const int64_t token = row % tokens, head = row / tokens % heads, batch = row / (tokens * heads);
        const float4 value = *reinterpret_cast<const float4*>(src + col + head * nb1 + token * nb2 + batch * nb3);
        dst[index / 2]     = __floats2half2_rn(__fmul_rn(value.x, scale), __fmul_rn(value.y, scale));
        dst[index / 2 + 1] = __floats2half2_rn(__fmul_rn(value.z, scale), __fmul_rn(value.w, scale));
    }
    static bool overlaps(const ggml_tensor* a, const ggml_tensor* b) {
        const auto x = reinterpret_cast<uintptr_t>(a->data), y = reinterpret_cast<uintptr_t>(b->data);
        return x <= y ? y - x < ggml_nbytes(a) : x - y < ggml_nbytes(b);
    }
    static ggml_status prepare(ggml_backend_t, ggml_tensor* const* nodes, int count, void**, size_t* workspace, void*) {
        *workspace = ggml_nbytes(nodes[count - 1]);
        return GGML_STATUS_SUCCESS;
    }
    static void destroy(void*, void*) {}
    static ggml_status execute(const ggml_ext_execution_context* execution, ggml_tensor* const* nodes, int count, void*, void*) {
        auto reg   = ggml_backend_dev_backend_reg(ggml_backend_get_device(execution->backend));
        auto query = reinterpret_cast<ggml_ext_cuda_context_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_ext_cuda_context_v1"));
        ggml_ext_cuda_context context;
        if (!query || !query(execution->backend, &context))
            return GGML_STATUS_FAILED;
        auto stream          = static_cast<cudaStream_t>(context.stream);
        const auto* node     = nodes[0];
        const auto* a        = node->src[0];
        const auto* b        = node->op == GGML_OP_CONT ? nullptr : node->src[1];
        auto* dst            = nodes[count - 1];
        const auto* residual = node->op == GGML_OP_MUL ? (dst->src[0] == node ? dst->src[1] : dst->src[0]) : nullptr;
        const bool gate      = residual != nullptr;
        const bool alias     = (overlaps(dst, a) && !(gate && dst->data == a->data)) || (b && overlaps(dst, b)) ||
                           (residual && overlaps(dst, residual) && dst->data != residual->data);
        void* output          = alias ? ggml_backend_buffer_get_base(execution->workspace) : dst->data;
        const int64_t count4  = ggml_nelements(dst) / 4;
        const unsigned blocks = unsigned((count4 + 255) / 256);
        if (gate) {
            gated_residual<<<blocks, 256, 0, stream>>>(static_cast<const float4*>(a->data), static_cast<const float4*>(b->data),
                                                       static_cast<const float4*>(residual->data), static_cast<float4*>(output), a->ne[0] / 4, count4);
        } else if (node->op == GGML_OP_CONCAT) {
            const bool permuted = count == 3;
            auto kernel         = permuted ? concat_cast<true> : concat_cast<false>;
            kernel<<<blocks, 256, 0, stream>>>(static_cast<const float4*>(a->data), static_cast<const float4*>(b->data),
                                               static_cast<half2*>(output), node->ne[0] / 4, a->ne[permuted ? 2 : 1], node->ne[permuted ? 2 : 1],
                                               node->ne[permuted ? 1 : 2], count4);
        } else {
            scale_permute<<<blocks, 256, 0, stream>>>(static_cast<const float*>(a->data), static_cast<half2*>(output),
                                                      a->ne[0], a->ne[2], a->ne[1], ggml_nelements(dst), a->nb[1] / 4, a->nb[2] / 4, a->nb[3] / 4, param(nodes[1], 0));
        }
        auto error = cudaGetLastError();
        if (error == cudaSuccess && alias)
            error = cudaMemcpyAsync(dst->data, output, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, stream);
        return error == cudaSuccess ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
    }
}

void sd_ggml_register_cuda_affine(ggml_backend_dev_t device) {
    ggml_ext_kernel_desc desc = {};
    desc.abi_version          = GGML_EXT_ABI_VERSION;
    desc.priority             = 200;
    desc.match                = sd_affine_cuda::match;
    desc.prepare              = sd_affine_cuda::prepare;
    desc.destroy              = sd_affine_cuda::destroy;
    desc.execute              = sd_affine_cuda::execute;
    desc.name                 = "sd.gated_residual.cuda.v1";
    desc.op                   = GGML_OP_MUL;
    GGML_ASSERT(ggml_ext_register_kernel_ex(device, &desc, GGML_EXT_KERNEL_CAPTURE_SAFE | GGML_EXT_KERNEL_REPLAY_REUSABLE));
    desc.name = "sd.concat_cast.cuda.v1";
    desc.op   = GGML_OP_CONCAT;
    GGML_ASSERT(ggml_ext_register_kernel_ex(device, &desc, GGML_EXT_KERNEL_CAPTURE_SAFE | GGML_EXT_KERNEL_REPLAY_REUSABLE));
    desc.name = "sd.cont_scale_permute_cast.cuda.v1";
    desc.op   = GGML_OP_CONT;
    GGML_ASSERT(ggml_ext_register_kernel_ex(device, &desc, GGML_EXT_KERNEL_CAPTURE_SAFE | GGML_EXT_KERNEL_REPLAY_REUSABLE));
}
