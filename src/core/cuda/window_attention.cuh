/*
 * Adapted from FlashAttention v2.8.3:
 * https://github.com/Dao-AILab/flash-attention/tree/v2.8.3/csrc/flash_attn/src
 * Sources: kernel_traits.h, utils.h, softmax.h, flash_fwd_kernel.h.
 * Copyright (c) 2023-2024, Tri Dao.
 *
 * BSD 3-Clause License
 *
 * Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef __SD_CORE_CUDA_WINDOW_ATTENTION_CUH__
#define __SD_CORE_CUDA_WINDOW_ATTENTION_CUH__

namespace sd_fattn_cute {
    struct WindowRegion {
        int left, top, width, height;
    };

    static __device__ WindowRegion window_region(const SDWindowAttentionParams& p, int plane, int x, int y) {
        int w = p.window_width[plane], h = p.window_height[plane];
        if (p.preserve_boundary) {
            w = min(w, p.width);
            h = min(h, p.height);
            return {min(max(x - w / 2, 0), p.width - w), min(max(y - h / 2, 0), p.height - h), w, h};
        }
        int left = max(x - w / 2, 0), top = max(y - h / 2, 0);
        return {left, top, min(p.width, x + w / 2) - left, min(p.height, y + h / 2) - top};
    }

    template <typename Dst, typename Id, typename Index>
    __device__ __forceinline__ void window_copy(const cutlass::half_t* source, Dst& dst, const Id& id, int dim, Index index) {
#pragma unroll
        for (int m = 0; m < size<1>(dst); ++m) {
            const int row = index(get<0>(id(0, m, 0)));
#pragma unroll
            for (int k = 0; k < size<2>(dst); ++k) {
                if (row >= 0) {
                    const auto* src        = source + int64_t(row) * dim + get<1>(id(0, m, k));
                    const unsigned address = unsigned(__cvta_generic_to_shared(&dst(0, m, k)));
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                                 :
                                 : "r"(address), "l"(src));
                } else {
                    cute::clear(dst(_, m, k));
                }
            }
        }
    }

    template <int D>
    __global__ __launch_bounds__(128) void window_kernel(const cutlass::half_t* q, const cutlass::half_t* k, const cutlass::half_t* v, float* output, SDWindowAttentionParams p, int kv_heads) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        constexpr int BM = 64, BN = 64, TW = 8;
        using T       = Traits<BM, BN, D>;
        using Element = typename T::Element;
        extern __shared__ char storage[];
        const int tid = threadIdx.x, head = blockIdx.z;
        const int qx = blockIdx.x * TW, qy = blockIdx.y * (BM / TW);
        const int tokens = p.width * p.height;
        q += int64_t(head) * tokens * D;
        const int64_t kv_offset = int64_t(head / (p.heads / kv_heads)) * tokens * p.planes * D;
        k += kv_offset;
        v += kv_offset;
        auto sQ      = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(storage)), typename T::QLayout{});
        auto sK      = make_tensor(sQ.data() + size(sQ), typename T::KVLayout{});
        auto sV      = make_tensor(sK.data() + size(sK), typename T::KVLayout{});
        auto sVt     = make_tensor(sV.data(), typename T::VtLayout{});
        auto sVplain = make_tensor(sV.data().get(), get_nonswizzle_portion(typename T::VtLayout{}));
        typename T::GlobalCopy global_copy;
        auto thread_copy = global_copy.get_thread_slice(tid);
        auto tQsQ        = thread_copy.partition_D(sQ);
        auto tKsK        = thread_copy.partition_D(sK);
        auto tVsV        = thread_copy.partition_D(sV);
        auto q_id        = thread_copy.partition_S(make_identity_tensor(Shape<Int<BM>, Int<D>>{}));
        auto kv_id       = thread_copy.partition_S(make_identity_tensor(Shape<Int<BN>, Int<D>>{}));
        auto query_index = [&](int row) {
            const int x = qx + row % TW, y = qy + row / TW;
            return x < p.width && y < p.height ? y * p.width + x : -1;
        };
        window_copy(q, tQsQ, q_id, D, query_index);
        cute::cp_async_fence();

        typename T::Mma mma;
        auto thread_mma = mma.get_thread_slice(tid);
        auto rQ         = thread_mma.partition_fragment_A(sQ);
        auto rK         = thread_mma.partition_fragment_B(sK);
        auto rV         = thread_mma.partition_fragment_B(sVplain);
        auto acc_o      = partition_fragment_C(mma, Shape<Int<BM>, Int<D>>{});
        auto copyQ      = make_tiled_copy_A(typename T::CopyAtom{}, mma);
        auto copyK      = make_tiled_copy_B(typename T::CopyAtom{}, mma);
        auto copyV      = make_tiled_copy_B(typename T::CopyAtomT{}, mma);
        auto threadQ    = copyQ.get_thread_slice(tid);
        auto threadK    = copyK.get_thread_slice(tid);
        auto threadV    = copyV.get_thread_slice(tid);
        auto tSsQ       = threadQ.partition_S(sQ);
        auto tSsK       = threadK.partition_S(sK);
        auto tOsV       = threadV.partition_S(sVt);
        auto score_id   = thread_mma.partition_C(make_identity_tensor(Shape<Int<BM>, Int<BN>>{}));
        clear(acc_o);
        Softmax<2 * size<1>(acc_o), true> softmax;
        const float scale = rsqrtf(float(D));
        for (int plane = 0; plane < p.planes; ++plane) {
            const auto first = window_region(p, plane, qx, qy);
            const auto last  = window_region(p, plane, min(qx + TW - 1, p.width - 1), min(qy + BM / TW - 1, p.height - 1));
            const int kw     = last.left + last.width - first.left;
            const int kh     = last.top + last.height - first.top;
            for (int start = 0; start < kw * kh; start += BN) {
                auto key_index = [&](int row) {
                    row += start;
                    return row < kw * kh ? plane * tokens + (first.top + row / kw) * p.width + first.left + row % kw : -1;
                };
                window_copy(k, tKsK, kv_id, D, key_index);
                window_copy(v, tVsV, kv_id, D, key_index);
                cute::cp_async_fence();
                cute::cp_async_wait<0>();
                __syncthreads();
                auto acc_s = partition_fragment_C(mma, Shape<Int<BM>, Int<BN>>{});
                clear(acc_s);
                gemm(acc_s, rQ, rK, tSsQ, tSsK, mma, copyQ, copyK, threadQ, threadK);
#pragma unroll
                for (int i = 0; i < size(acc_s); ++i) {
                    const int qr = get<0>(score_id(i)), kr = start + get<1>(score_id(i));
                    const int x = qx + qr % TW, y = qy + qr / TW;
                    const auto region = window_region(p, plane, x, y);
                    const int kx = first.left + kr % kw, ky = first.top + kr / kw;
                    const bool valid = x < p.width && y < p.height && kr < kw * kh &&
                                       kx >= region.left && kx < region.left + region.width &&
                                       ky >= region.top && ky < region.top + region.height;
                    acc_s(i)         = valid ? acc_s(i) * scale : -INFINITY;
                }
                if (plane == 0 && start == 0)
                    softmax.template update<true>(acc_s, acc_o);
                else
                    softmax.template update<false>(acc_s, acc_o);
                auto rP   = to_half(acc_s);
                auto tOrP = make_tensor(rP.data(), convert_layout_acc_Aregs<typename T::Mma>(rP.layout()));
                gemm_rs(acc_o, tOrP, rV, tOsV, mma, copyV, threadV);
                __syncthreads();
            }
        }
        softmax.normalize(acc_o);
        auto output_id = thread_mma.partition_C(make_identity_tensor(Shape<Int<BM>, Int<D>>{}));
#pragma unroll
        for (int i = 0; i < size(acc_o); ++i) {
            const int token = query_index(get<0>(output_id(i)));
            if (token >= 0)
                output[(int64_t(token) * p.heads + head) * D + get<1>(output_id(i))] = acc_o(i);
        }
#else
        // This translation unit also targets older GPUs; runtime dispatch excludes this kernel there.
        asm volatile("trap;");
#endif
    }
}

static bool window_attention_supports(ggml_backend_dev_t device, const ggml_tensor* node, void*) {
    static const bool disabled = [] {
        const char* value = std::getenv("GGML_CUDA_DISABLE_WINDOW_ATTENTION");
        return value && std::atoi(value) != 0;
    }();
    SDWindowAttentionParams params;
    const auto* info = cuda_device(device);
    return !disabled && info && info->properties.major >= 8 && sd_ggml_window_attn_describe(node, &params);
}

static ggml_status window_attention_prepare(ggml_backend_t, ggml_tensor* const* nodes, int, void**, size_t* workspace, void*) {
    *workspace = 0;
    for (int i = 0; i < 3; ++i)
        *workspace += (size_t(ggml_nelements(nodes[0]->src[i])) * sizeof(half) + 255) / 256 * 256;
    return GGML_STATUS_SUCCESS;
}

static void window_attention_destroy(void*, void*) {}

template <int D>
static cudaError_t window_attention_launch(const ggml_ext_cuda_context& context, const half* const* inputs, float* output, SDWindowAttentionParams params, int kv_heads) {
    auto kernel                = sd_fattn_cute::window_kernel<D>;
    constexpr int shared_bytes = sd_fattn_cute::Traits<64, 64, D>::SharedBytes;
    static std::mutex mutex;
    static std::unordered_map<int, bool> initialized;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized[context.device]) {
            auto error = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
            if (error != cudaSuccess)
                return error;
            initialized[context.device] = true;
        }
    }
    kernel<<<dim3((params.width + 7) / 8, (params.height + 7) / 8, params.heads), 128, shared_bytes, static_cast<cudaStream_t>(context.stream)>>>(
        reinterpret_cast<const cutlass::half_t*>(inputs[0]), reinterpret_cast<const cutlass::half_t*>(inputs[1]),
        reinterpret_cast<const cutlass::half_t*>(inputs[2]), output, params, kv_heads);
    return cudaGetLastError();
}

static ggml_status window_attention_execute(const ggml_ext_execution_context* execution, ggml_tensor* const* nodes, int, void*, void*) {
    auto device = ggml_backend_get_device(execution->backend);
    auto reg    = ggml_backend_dev_backend_reg(device);
    auto query  = reinterpret_cast<ggml_ext_cuda_context_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_ext_cuda_context_v1"));
    ggml_ext_cuda_context context;
    if (!query || !query(execution->backend, &context))
        return GGML_STATUS_FAILED;
    SDWindowAttentionParams params;
    auto* node = nodes[0];
    if (!sd_ggml_window_attn_describe(node, &params))
        return GGML_STATUS_FAILED;
    auto stream   = static_cast<cudaStream_t>(context.stream);
    auto* base    = static_cast<char*>(ggml_backend_buffer_get_base(execution->workspace));
    const int dim = int(node->ne[0] / params.heads);
    const half* inputs[3];
    for (int i = 0; i < 3; ++i) {
        const auto* src     = node->src[i];
        auto* packed        = reinterpret_cast<half*>(base);
        const int64_t heads = src->ne[0] / dim;
        copy_to_half<float><<<(ggml_nelements(src) + 255) / 256, 256, 0, stream>>>(static_cast<const char*>(src->data), packed,
                                                                                   ggml_nelements(src), dim, src->ne[1], heads, src->nb[1], dim * sizeof(float), ggml_nbytes(src), 1.f);
        if (cudaGetLastError() != cudaSuccess)
            return GGML_STATUS_FAILED;
        inputs[i] = packed;
        base += (size_t(ggml_nelements(src)) * sizeof(half) + 255) / 256 * 256;
    }
    const int kv_heads = int(node->src[1]->ne[0] / dim);
    auto* output       = static_cast<float*>(node->data);
    const auto error   = dim == 64 ? window_attention_launch<64>(context, inputs, output, params, kv_heads)
                                   : window_attention_launch<128>(context, inputs, output, params, kv_heads);
    return error == cudaSuccess ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
}

void sd_ggml_register_cuda_window_attention(ggml_backend_dev_t device) {
    const auto* info = cuda_device(device);
    if (!info || info->properties.major < 8)
        return;
    int previous;
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(info->ordinal) != cudaSuccess)
        return;
    cudaFuncAttributes attributes64{}, attributes128{};
    const auto error64  = cudaFuncGetAttributes(&attributes64, sd_fattn_cute::window_kernel<64>);
    const auto error128 = cudaFuncGetAttributes(&attributes128, sd_fattn_cute::window_kernel<128>);
    const auto restored = cudaSetDevice(previous);
    // JIT can produce an SM80+ binary from older PTX containing only the trap.
    if (restored != cudaSuccess || error64 != cudaSuccess || error128 != cudaSuccess ||
        std::min(attributes64.binaryVersion, attributes64.ptxVersion) < 80 ||
        std::min(attributes128.binaryVersion, attributes128.ptxVersion) < 80)
        return;
    ggml_ext_kernel_desc desc = {};
    desc.abi_version          = GGML_EXT_ABI_VERSION;
    desc.name                 = "sd.window_attention.cuda.v1";
    desc.op                   = GGML_OP_CUSTOM;
    desc.custom_op            = sd_ggml_window_attn_op();
    desc.priority             = 100;
    desc.supports             = window_attention_supports;
    desc.prepare              = window_attention_prepare;
    desc.destroy              = window_attention_destroy;
    desc.execute              = window_attention_execute;
    GGML_ASSERT(ggml_ext_register_kernel_ex(device, &desc, GGML_EXT_KERNEL_CAPTURE_SAFE | GGML_EXT_KERNEL_REPLAY_REUSABLE));
}

#endif
