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
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/numeric_conversion.h>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <cute/tensor.hpp>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include "ggml-ext.h"

static __device__ __forceinline__ float sd_warp_sum(float v) {
    for (int mask = 16; mask > 0; mask >>= 1)
        v = __fadd_rn(v, __shfl_xor_sync(0xffffffff, v, mask));
    return v;
}
static __device__ __forceinline__ float sd_warp_max(float v) {
    for (int mask = 16; mask > 0; mask >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, mask));
    return v;
}
namespace sd_fattn_cute {
    using namespace cute;
    template <bool A_in_regs = false, bool B_in_regs = false, typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3, typename Tensor4, typename TiledMma, typename TiledCopyA, typename TiledCopyB, typename ThrCopyA, typename ThrCopyB>
    __forceinline__ __device__ void gemm(Tensor0& acc, Tensor1& tCrA, Tensor2& tCrB, Tensor3 const& tCsA, Tensor4 const& tCsB, TiledMma tiled_mma, TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B, ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
        CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
        CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
        CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
        Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
        CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
        Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
        CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
        if (!A_in_regs) {
            cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
        }
        if (!B_in_regs) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
        }
#pragma unroll
        for (int i = 0; i < size<2>(tCrA); ++i) {
            if (i < size<2>(tCrA) - 1) {
                if (!A_in_regs) {
                    cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
                }
                if (!B_in_regs) {
                    cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
                }
            }
            cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
        }
    }

    template <typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3, typename TiledMma, typename TiledCopy, typename ThrCopy>
    __forceinline__ __device__ void gemm_rs(Tensor0& acc, Tensor1& tCrA, Tensor2& tCrB, Tensor3 const& tCsB, TiledMma tiled_mma, TiledCopy smem_tiled_copy_B, ThrCopy smem_thr_copy_B) {
        CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
        CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
        CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
        Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
        CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
        cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
#pragma unroll
        for (int i = 0; i < size<2>(tCrA); ++i) {
            if (i < size<2>(tCrA) - 1) {
                cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
            }
            cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
        }
    }

    template <typename Layout>
    __forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
        static_assert(decltype(size<0>(acc_layout))::value == 4);
        static_assert(decltype(rank(acc_layout))::value == 3);
        auto l = logical_divide(acc_layout, Shape<_2>{});
        return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
    };

    template <typename MMA_traits, typename Layout>
    __forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
        using X = Underscore;
        static_assert(decltype(size<0>(acc_layout))::value == 4);
        static_assert(decltype(rank(acc_layout))::value == 3);
        constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
        static_assert(mma_shape_K == 8 || mma_shape_K == 16);
        if constexpr (mma_shape_K == 8) {
            return acc_layout;
        } else {
            auto l = logical_divide(acc_layout, Shape<X, X, _2>{});
            return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
        }
    };

    template <typename TensorT>
    __device__ __forceinline__ auto to_half(const TensorT& src) {
        constexpr int count = decltype(size(src))::value;
        auto dst            = make_tensor<cutlass::half_t>(src.layout());
        cutlass::NumericArrayConverter<cutlass::half_t, float, count> convert;
        *reinterpret_cast<cutlass::Array<cutlass::half_t, count>*>(dst.data()) =
            convert(*reinterpret_cast<const cutlass::Array<float, count>*>(src.data()));
        return dst;
    }

    template <bool Full, bool Clear, bool FullCols = true, typename Copy, typename Src, typename Dst, typename Id>
    __device__ __forceinline__ void copy_rows(Copy copier, const Src& src, Dst& dst, const Id& id, int rows, int cols = 0) {
#pragma unroll
        for (int m = 0; m < size<1>(src); ++m) {
#pragma unroll
            for (int k = 0; k < size<2>(src); ++k) {
                if ((Full || get<0>(id(0, m, k)) < rows) && (FullCols || get<1>(id(0, m, k)) < cols)) {
                    cute::copy(copier, src(_, m, k), dst(_, m, k));
                } else if constexpr (Clear) {
                    cute::clear(dst(_, m, k));
                }
            }
        }
    }

    template <int Rows>
    struct Softmax {
        decltype(make_tensor<float>(Shape<Int<Rows>>{})) maximum, sum;
        template <bool First, typename Scores, typename Output>
        __device__ __forceinline__ void update(Scores& acc_s, Output& acc_o) {
            auto scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
            auto output = make_tensor(acc_o.data(), convert_layout_acc_rowcol(acc_o.layout()));
#pragma unroll
            for (int m = 0; m < Rows; ++m) {
                const float old_max = First ? -INFINITY : maximum(m);
                float next_max      = First ? scores(m, 0) : fmaxf(old_max, scores(m, 0));
#pragma unroll
                for (int n = 1; n < size<1>(scores); ++n)
                    next_max = fmaxf(next_max, scores(m, n));
                next_max      = fmaxf(next_max, __shfl_xor_sync(0xffffffff, next_max, 2));
                next_max      = fmaxf(next_max, __shfl_xor_sync(0xffffffff, next_max, 1));
                maximum(m)    = next_max;
                float row_sum = 0.f;
                if constexpr (!First) {
                    row_sum = sum(m);
                    if (old_max != next_max) {
                        const float factor = exp2f((old_max - next_max) * 1.4426950408889634f);
                        row_sum *= factor;
#pragma unroll
                        for (int n = 0; n < size<1>(output); ++n)
                            output(m, n) *= factor;
                    }
                }
#pragma unroll
                for (int n = 0; n < size<1>(scores); ++n) {
                    const float value = exp2f(scores(m, n) * 1.4426950408889634f - next_max * 1.4426950408889634f);
                    scores(m, n)      = value;
                    row_sum += value;
                }
                // Defer the four-lane sum reduction until the final output normalization.
                sum(m) = row_sum;
            }
        }
        template <typename Output>
        __device__ __forceinline__ void normalize(Output& acc_o) {
            auto output = make_tensor(acc_o.data(), convert_layout_acc_rowcol(acc_o.layout()));
#pragma unroll
            for (int m = 0; m < Rows; ++m) {
                float total = sum(m);
                total += __shfl_xor_sync(0xffffffff, total, 2);
                total += __shfl_xor_sync(0xffffffff, total, 1);
                const float inverse = 1.f / total;
#pragma unroll
                for (int n = 0; n < size<1>(output); ++n)
                    output(m, n) *= inverse;
            }
        }
    };

    template <int BM, int BN, int D = 128, int Warps = 4>
    struct Traits {
        using Element                    = cutlass::half_t;
        static constexpr int Threads     = Warps * 32;
        using Mma                        = TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                             Layout<Shape<Int<Warps>, _1, _1>>,
                             Tile<Int<16 * Warps>, _16, _16>>;
        static constexpr int SmemWidth   = D % 64 == 0 ? 64 : D % 32 == 0 ? 32
                                                                          : 16;
        static constexpr int SwizzleBits = SmemWidth == 64 ? 3 : SmemWidth == 32 ? 2
                                                                                 : 1;
        using SmemAtom                   = decltype(composition(Swizzle<SwizzleBits, 3, 3>{},
                                                                Layout<Shape<_8, Int<SmemWidth>>, Stride<Int<SmemWidth>, _1>>{}));
        using QLayout                    = decltype(tile_to_shape(SmemAtom{}, Shape<Int<BM>, Int<D>>{}));
        using KVLayout                   = decltype(tile_to_shape(SmemAtom{}, Shape<Int<BN>, Int<D>>{}));
        using VtLayout                   = decltype(composition(KVLayout{}, make_layout(Shape<Int<D>, Int<BN>>{}, GenRowMajor{})));
        using CopyAtom                   = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
        using CopyAtomT                  = Copy_Atom<SM75_U16x8_LDSM_T, Element>;
        using GlobalLayout               = Layout<Shape<Int<Threads / (SmemWidth / 8)>, Int<SmemWidth / 8>>, Stride<Int<SmemWidth / 8>, _1>>;
        using GlobalCopy                 = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>{},
                                                                    GlobalLayout{},
                                                                    Layout<Shape<_1, _8>>{}));
        using OutputCopy                 = decltype(make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                                                                    GlobalLayout{},
                                                                    Layout<Shape<_1, _8>>{}));
        static constexpr int SharedBytes = (BM + 2 * BN) * D * sizeof(Element);
    };

    struct Params {
        const cutlass::half_t *q, *k, *v;
        cutlass::half_t* o;
        int nq, nk, heads, kv_heads, dim;
    };

    template <int BM, int BN, int D = 128, bool FullDim = true, int Warps = 4>
    __global__ __launch_bounds__(Warps * 32) void kernel(Params p) {
        using T       = Traits<BM, BN, D, Warps>;
        using Element = typename T::Element;
        extern __shared__ char storage[];
        const int tid = threadIdx.x, m_block = blockIdx.x, batch = blockIdx.y, head = blockIdx.z;
        const int64_t q_offset  = (int64_t(batch) * p.heads + head) * p.nq * p.dim;
        const int64_t kv_offset = (int64_t(batch) * p.kv_heads + head / (p.heads / p.kv_heads)) * p.nk * p.dim;
        auto mQ                 = make_tensor(make_gmem_ptr(p.q + q_offset), make_shape(p.nq, p.dim), make_stride(p.dim, _1{}));
        auto mK                 = make_tensor(make_gmem_ptr(p.k + kv_offset), make_shape(p.nk, p.dim), make_stride(p.dim, _1{}));
        auto mV                 = make_tensor(make_gmem_ptr(p.v + kv_offset), make_shape(p.nk, p.dim), make_stride(p.dim, _1{}));
        auto gQ                 = local_tile(mQ, Shape<Int<BM>, Int<D>>{}, make_coord(m_block, 0));
        auto gK                 = local_tile(mK, Shape<Int<BN>, Int<D>>{}, make_coord(_, 0));
        auto gV                 = local_tile(mV, Shape<Int<BN>, Int<D>>{}, make_coord(_, 0));
        auto sQ                 = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(storage)), typename T::QLayout{});
        auto sK                 = make_tensor(sQ.data() + size(sQ), typename T::KVLayout{});
        auto sV                 = make_tensor(sK.data() + size(sK), typename T::KVLayout{});
        auto sVt                = make_tensor(sV.data(), typename T::VtLayout{});
        auto sVplain            = make_tensor(sV.data().get(), get_nonswizzle_portion(typename T::VtLayout{}));
        typename T::GlobalCopy global_copy;
        auto thread_copy = global_copy.get_thread_slice(tid);
        auto tQgQ        = thread_copy.partition_S(gQ);
        auto tQsQ        = thread_copy.partition_D(sQ);
        auto tKgK        = thread_copy.partition_S(gK);
        auto tKsK        = thread_copy.partition_D(sK);
        auto tVgV        = thread_copy.partition_S(gV);
        auto tVsV        = thread_copy.partition_D(sV);
        auto tQcQ        = thread_copy.partition_S(make_identity_tensor(Shape<Int<BM>, Int<D>>{}));
        auto tKVcKV      = thread_copy.partition_S(make_identity_tensor(Shape<Int<BN>, Int<D>>{}));
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
        // Process the partial K/V tile first; subsequent tiles need no score masking.
        int block = (p.nk - 1) / BN;
        copy_rows<false, true, FullDim>(global_copy, tQgQ, tQsQ, tQcQ, p.nq - m_block * BM, p.dim);
        auto last_k = tKgK(_, _, _, block);
        copy_rows<false, true, FullDim>(global_copy, last_k, tKsK, tKVcKV, p.nk - block * BN, p.dim);
        cute::cp_async_fence();
        clear(acc_o);
        Softmax<2 * size<1>(acc_o)> softmax;
        auto step = [&](auto first_flag, int n) {
            constexpr bool First = decltype(first_flag)::value;
            auto acc_s           = partition_fragment_C(mma, Shape<Int<BM>, Int<BN>>{});
            clear(acc_s);
            cute::cp_async_wait<0>();
            __syncthreads();
            auto v_tile = tVgV(_, _, _, n);
            copy_rows<!First, First, FullDim>(global_copy, v_tile, tVsV, tKVcKV, p.nk - n * BN, p.dim);
            cute::cp_async_fence();
            gemm(acc_s, rQ, rK, tSsQ, tSsK, mma, copyQ, copyK, threadQ, threadK);
            if constexpr (First) {
#pragma unroll
                for (int i = 0; i < size(acc_s); ++i) {
                    if (get<1>(score_id(i)) >= p.nk - n * BN)
                        acc_s(i) = -INFINITY;
                }
            }
            cute::cp_async_wait<0>();
            __syncthreads();
            // Do not commit an empty async group: the next iteration waits for this K tile.
            if (n > 0) {
                auto next_k = tKgK(_, _, _, n - 1);
                copy_rows<true, false, FullDim>(global_copy, next_k, tKsK, tKVcKV, 0, p.dim);
                cute::cp_async_fence();
            }
            softmax.template update<First>(acc_s, acc_o);
            auto rP   = to_half(acc_s);
            auto tOrP = make_tensor(rP.data(), convert_layout_acc_Aregs<typename T::Mma>(rP.layout()));
            gemm_rs(acc_o, tOrP, rV, tOsV, mma, copyV, threadV);
        };
        step(std::true_type{}, block);
        for (--block; block >= 0; --block)
            step(std::false_type{}, block);
        softmax.normalize(acc_o);
        auto rO      = to_half(acc_o);
        auto sO      = make_tensor(sQ.data(), typename T::QLayout{});
        auto copyO   = make_tiled_copy_C(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{}, mma);
        auto threadO = copyO.get_thread_slice(tid);
        auto tOrO    = threadO.retile_S(rO);
        auto tOsO    = threadO.partition_D(sO);
        cute::copy(copyO, tOrO, tOsO);
        auto mO = make_tensor(make_gmem_ptr(p.o + int64_t(batch) * p.nq * p.heads * p.dim + int64_t(head) * p.dim),
                              make_shape(p.nq, p.dim), make_stride(p.heads * p.dim, _1{}));
        auto gO = local_tile(mO, Shape<Int<BM>, Int<D>>{}, make_coord(m_block, 0));
        typename T::OutputCopy output_copy;
        auto global_threadO = output_copy.get_thread_slice(tid);
        auto from_sO        = global_threadO.partition_S(sO);
        auto to_gO          = global_threadO.partition_D(gO);
        auto output_id      = global_threadO.partition_D(make_identity_tensor(Shape<Int<BM>, Int<D>>{}));
        __syncthreads();
        auto output_regs = make_tensor<Element>(shape(to_gO));
        cute::copy(output_copy, from_sO, output_regs);
        copy_rows<false, false, FullDim>(output_copy, output_regs, to_gO, output_id, p.nq - m_block * BM, p.dim);
    }
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
            const float other = __shfl_xor_sync(0xffffffff, value, 1, 32);
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

static __device__ __forceinline__ float convrot_h4_cuda(float a, float b, float c, float d, int index) {
    a = index == 3 ? -a : a;
    b = index == 2 ? -b : b;
    c = index == 1 ? -c : c;
    d = index == 0 ? -d : d;
    return __fadd_rn(__fadd_rn(__fadd_rn(a, b), c), d);
}

static __global__ void quantize_attention(const half* src, int8_t* dst, float* scales, int64_t k, int64_t rows, float input_scale, float input_bias) {
    const int64_t row = blockIdx.x;
    const int tid     = threadIdx.x;
    if (row >= rows)
        return;
    extern __shared__ float values[];
    __shared__ float maxima[32];
    const half* src_row = src + row * k;
    int8_t* dst_row     = dst + row * k;
    const int lane_id   = tid % 32;
    {
        const int lane           = lane_id;
        constexpr int group_size = 256;
        for (int64_t group = tid / 32; group < k / group_size; group += blockDim.x / 32) {
            float v[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float value = __fmaf_rn(__half2float(src_row[group * group_size + j * 32 + lane]), input_scale, input_bias);
                v[j]              = value * (1.0f / 16.0f);
            }
            // One logical warp owns H256, keeping every butterfly in registers.
#pragma unroll
            for (int stride = 1; stride <= 4; stride *= 4) {
                const int base = lane & ~(3 * stride);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float a = __shfl_sync(0xffffffff, v[j], base, 32);
                    const float b = __shfl_sync(0xffffffff, v[j], base + stride, 32);
                    const float c = __shfl_sync(0xffffffff, v[j], base + 2 * stride, 32);
                    const float d = __shfl_sync(0xffffffff, v[j], base + 3 * stride, 32);
                    v[j]          = convrot_h4_cuda(a, b, c, d, (lane / stride) % 4);
                }
            }
#pragma unroll
            for (int j = 0; j < 8; j += 2) {
                const float a = __shfl_sync(0xffffffff, v[j], lane % 16, 32);
                const float b = __shfl_sync(0xffffffff, v[j], lane % 16 + 16, 32);
                const float c = __shfl_sync(0xffffffff, v[j + 1], lane % 16, 32);
                const float d = __shfl_sync(0xffffffff, v[j + 1], lane % 16 + 16, 32);
                v[j]          = convrot_h4_cuda(a, b, c, d, lane / 16);
                v[j + 1]      = convrot_h4_cuda(a, b, c, d, 2 + lane / 16);
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const float a = v[j], b = v[j + 2], c = v[j + 4], d = v[j + 6];
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    values[group * group_size + (j + 2 * t) * 32 + lane] = convrot_h4_cuda(a, b, c, d, t);
                }
            }
        }
        __syncthreads();
    }
    float local_max = 0.0f;
    for (int64_t i = tid; i < k; i += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(values[i]));
    }
    local_max      = sd_warp_max(local_max);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) {
        maxima[warp] = local_max;
    }
    __syncthreads();

    if (warp == 0) {
        local_max = lane < blockDim.x / 32 ? maxima[lane] : 0.0f;
        local_max = sd_warp_max(local_max);
        if (lane == 0) {
            maxima[0] = local_max / 127.0f;
            if (scales != nullptr) {
                scales[row] = maxima[0];
            } else {
                *((float*)(dst_row + k)) = maxima[0];
            }
        }
    }
    __syncthreads();

    const float row_scale = maxima[0];
    const float inv_scale = row_scale == 0.0f ? 0.0f : 1.0f / row_scale;
    for (int64_t i = tid; i < k; i += blockDim.x) {
        int value  = row_scale == 0.0f ? 0 : __float2int_rn(values[i] * inv_scale);
        value      = max(-127, min(127, value));
        dst_row[i] = (int8_t)value;
    }
}

template <typename T>
static __global__ void copy_to_half(const char* src, half* dst, int64_t count, int64_t dim, int64_t rows, int64_t heads, size_t nb1, size_t nb2, size_t nb3, float scale) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const int64_t row = i / dim;
        const auto* p     = reinterpret_cast<const T*>(src + (row % rows) * nb1 +
                                                   ((row / rows) % heads) * nb2 + (row / (rows * heads)) * nb3);
        dst[i]            = __float2half_rn(float(p[i % dim]) * scale);
    }
}

static __global__ void copy_from_half(const half* src, float* dst, int64_t count, float scale, float bias, bool scaled) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count)
        dst[i] = scaled ? __fmaf_rn(__half2float(src[i]), scale, bias) : __half2float(src[i]);
}

static float param_f32(const ggml_tensor* node, int index) {
    float value;
    std::memcpy(&value, reinterpret_cast<const char*>(node->op_params) + index * sizeof(value), sizeof(value));
    return value;
}

static int padded_dim(int64_t dim) {
    return dim == 80 || dim == 96 || dim == 112 ? int(dim) : int((dim + 63) / 64 * 64);
}

struct CUDADevice {
    int ordinal;
    cudaDeviceProp properties;
};

static const CUDADevice* cuda_device(ggml_backend_dev_t device) {
    static std::mutex mutex;
    static std::unordered_map<ggml_backend_dev_t, CUDADevice> devices;
    std::lock_guard<std::mutex> lock(mutex);
    auto found = devices.find(device);
    if (found != devices.end())
        return &found->second;
    auto reg   = ggml_backend_dev_backend_reg(device);
    auto query = reinterpret_cast<ggml_ext_cuda_device_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_ext_cuda_device_v1"));
    if (!query)
        return nullptr;
    CUDADevice info = {};
    info.ordinal    = query(device);
    if (info.ordinal < 0 || cudaGetDeviceProperties(&info.properties, info.ordinal) != cudaSuccess)
        return nullptr;
    return &devices.emplace(device, info).first->second;
}

static bool attention_supports(ggml_backend_dev_t device, const ggml_tensor* node, void*) {
    static const bool disabled = std::getenv("GGML_CUDA_DISABLE_CUTE_FATTN") || std::getenv("GGML_CUDA_DISABLE_CUTLASS_FATTN");
    const auto* info           = cuda_device(device);
    const auto* q              = node->src[0];
    const auto* k              = node->src[1];
    const auto* v              = node->src[2];
    if (disabled || !info || info->properties.major < 8 || !q || !k || !v ||
        q->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32 ||
        q->ne[0] < 32 || q->ne[0] > 256 || q->ne[0] % 8 ||
        size_t(192 * padded_dim(q->ne[0]) * sizeof(half)) > info->properties.sharedMemPerBlockOptin ||
        q->ne[1] < 1 || k->ne[1] < 1 || k->ne[2] < 1 ||
        node->src[3] || node->src[4] || param_f32(node, 1) != 0.0f || param_f32(node, 2) != 0.0f ||
        q->ne[2] % k->ne[2] || q->ne[3] != k->ne[3] || !ggml_are_same_shape(k, v) ||
        q->nb[0] != sizeof(float) || !ggml_is_contiguous(node) ||
        (k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_F32) ||
        (v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_F32) ||
        k->nb[0] != ggml_type_size(k->type) || v->nb[0] != ggml_type_size(v->type))
        return false;
    return k->ne[0] == q->ne[0] && q->ne[1] <= INT_MAX / q->ne[0] && k->ne[1] <= INT_MAX / q->ne[0] &&
           q->ne[2] <= 65535 && q->ne[3] <= 65535 && ggml_nelements(q) <= INT_MAX && ggml_nelements(k) <= INT_MAX;
}

static bool attention_preferred(ggml_backend_dev_t, const ggml_tensor* node, void*) {
    const auto* q = node->src[0];
    const auto* k = node->src[1];
    if (q->ne[0] == 40 || q->ne[0] == 72)
        return q->ne[1] >= 512 && k->ne[1] >= 512;
    if (q->ne[0] == 80 || q->ne[0] == 96 || q->ne[0] == 112)
        return false;
    const int minimum = q->ne[0] == 128 ? 2048 : 4096;
    return q->ne[1] >= minimum && k->ne[1] >= minimum;
}

static bool attention_selected(ggml_backend_dev_t device, const ggml_tensor* node) {
    return node->op == GGML_OP_FLASH_ATTN_EXT && attention_supports(device, node, nullptr) &&
           (attention_preferred(device, node, nullptr) || !ggml_ext_dev_supports_native_op(device, node));
}

static bool fusion_enabled() {
    static const bool disabled = [] {
        const char* value = std::getenv("GGML_CUDA_DISABLE_FUSION");
        return value && std::atoi(value) != 0;
    }();
    return !disabled;
}

static bool query_fusion(const ggml_tensor* norm, const ggml_tensor* mul, const ggml_tensor* rope) {
    if (norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || rope->op != GGML_OP_ROPE_APPLY ||
        mul->src[0] != norm || rope->src[0] != mul)
        return false;
    const auto* x      = norm->src[0];
    const auto* weight = mul->src[1];
    const auto* pe     = rope->src[1];
    return x && weight && pe && x->ne[0] >= 32 && x->ne[0] <= 256 && x->ne[0] % 32 == 0 &&
           x->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_F32 && pe->type == GGML_TYPE_F32 &&
           norm->type == GGML_TYPE_F32 && mul->type == GGML_TYPE_F32 && rope->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(norm, mul) && weight->ne[0] == x->ne[0] && ggml_nelements(weight) == x->ne[0] &&
           pe->ne[0] == 2 && pe->ne[1] == 2 && pe->ne[2] > 0 && pe->ne[2] <= x->ne[0] / 2 && pe->ne[3] == x->ne[2] &&
           x->nb[0] == sizeof(float) && ggml_is_contiguous(weight) && ggml_is_contiguous(pe) && ggml_is_contiguous(rope);
}

static const ggml_tensor* graph_node(const ggml_cgraph* graph, int index) {
    return ggml_graph_node(const_cast<ggml_cgraph*>(graph), index);
}

static int output_fusion(ggml_backend_dev_t device, const ggml_cgraph* graph, int first) {
    const auto* attn     = graph_node(graph, first);
    const auto* previous = attn;
    const int total      = ggml_graph_n_nodes(const_cast<ggml_cgraph*>(graph));
    bool scaled          = false;
    for (int count = 1; count < 8 && first + count < total; ++count) {
        const auto* node = graph_node(graph, first + count);
        if (node->src[0] != previous)
            break;
        if (node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE) {
            if (node->view_offs || !ggml_is_contiguous(node) || ggml_nelements(node) != ggml_nelements(attn))
                break;
        } else if (node->op == GGML_OP_SCALE && !scaled) {
            if (!ggml_is_contiguous(node) || !ggml_are_same_shape(previous, node))
                break;
            scaled = true;
        } else if (node->op == GGML_OP_QUANTIZE_I8_CONVROT) {
            if (node->op_params[0] == 256 && previous->ne[0] % 256 == 0 &&
                size_t(previous->ne[0]) * sizeof(float) + 128 <= cuda_device(device)->properties.sharedMemPerBlockOptin)
                return count + 1;
            break;
        } else
            break;
        previous = node;
    }
    if (first + 1 < total) {
        const auto* next = graph_node(graph, first + 1);
        if (next->op == GGML_OP_SCALE && next->src[0] == attn && ggml_is_contiguous(next) && ggml_are_same_shape(attn, next))
            return 2;
    }
    return 1;
}

static int attention_match(ggml_backend_dev_t device, const ggml_cgraph* graph, int first, void*) {
    if (!fusion_enabled())
        return 0;
    int offset = 0;
    if (graph_node(graph, first)->op == GGML_OP_RMS_NORM) {
        if (first + 3 >= ggml_graph_n_nodes(const_cast<ggml_cgraph*>(graph)) ||
            !query_fusion(graph_node(graph, first), graph_node(graph, first + 1), graph_node(graph, first + 2)) ||
            graph_node(graph, first + 3)->src[0] != graph_node(graph, first + 2))
            return 0;
        offset = 3;
    }
    if (!attention_selected(device, graph_node(graph, first + offset)))
        return 0;
    const int count = offset + output_fusion(device, graph, first + offset);
    return count > 1 ? count : 0;
}

static void attention_optimize(ggml_backend_dev_t device, ggml_cgraph* graph, void*) {
    if (!fusion_enabled())
        return;
    for (int i = 3; i < ggml_graph_n_nodes(const_cast<ggml_cgraph*>(graph)); ++i) {
        const auto* attn = graph_node(graph, i);
        if (!attention_selected(device, attn))
            continue;
        for (int j = i - 3; j >= 0 && j >= i - 32; --j) {
            if (attn->src[0] == graph_node(graph, j + 2) &&
                query_fusion(graph_node(graph, j), graph_node(graph, j + 1), graph_node(graph, j + 2))) {
                ggml_ext_graph_delay(graph, j, 3, i);
                break;
            }
        }
    }
}

struct AttentionPlan {
    int index;
    size_t offsets[4];
};

static ggml_status attention_prepare(ggml_backend_t, ggml_tensor* const* nodes, int count, void** state, size_t* workspace_size, void*) {
    auto* plan  = new AttentionPlan{};
    *state      = plan;
    plan->index = nodes[0]->op == GGML_OP_RMS_NORM ? 3 : 0;
    GGML_ASSERT(plan->index < count);
    const auto* attn = nodes[plan->index];
    size_t size      = 0;
    for (int i = 0; i < 4; ++i) {
        plan->offsets[i]   = size;
        const size_t bytes = size_t(ggml_nelements(i < 3 ? attn->src[i] : attn)) * sizeof(half);
        size += (bytes + 255) / 256 * 256;
    }
    *workspace_size = size;
    return GGML_STATUS_SUCCESS;
}

template <int D, bool FullDim>
static cudaError_t attention_launch(const ggml_ext_cuda_context& context, const sd_fattn_cute::Params& p, int64_t batches) {
    auto kernel                = sd_fattn_cute::kernel<64, 64, D, FullDim>;
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
    kernel<<<dim3((p.nq + 63) / 64, batches, p.heads), 128, shared_bytes, static_cast<cudaStream_t>(context.stream)>>>(p);
    return cudaGetLastError();
}

template <int D>
static cudaError_t attention_launch(const ggml_ext_cuda_context& context, const sd_fattn_cute::Params& p, int64_t batches) {
    return p.dim == D ? attention_launch<D, true>(context, p, batches) : attention_launch<D, false>(context, p, batches);
}

template <bool Interleaved>
static void query_launch(cudaStream_t stream, ggml_tensor* const* nodes, half* output, float scale) {
    const auto* x      = nodes[0]->src[0];
    const auto* pe     = nodes[2]->src[1];
    const int64_t rows = ggml_nrows(x);
#define SD_QUERY(D)                                                                                                                     \
    case D:                                                                                                                             \
        rms_norm_mul_rope_apply_fused<D, Interleaved, half><<<(rows + 3) / 4, dim3(32, 4), 0, stream>>>(                                \
            static_cast<const float*>(x->data), static_cast<const float*>(nodes[1]->src[1]->data), static_cast<const float*>(pe->data), \
            output, x->ne[1], x->ne[2], rows, x->nb[1] / 4, x->nb[2] / 4, x->nb[3] / 4, int(pe->ne[2]), param_f32(nodes[0], 0), scale); \
        break
    switch (x->ne[0]) {
        SD_QUERY(32);
        SD_QUERY(64);
        SD_QUERY(96);
        SD_QUERY(128);
        SD_QUERY(160);
        SD_QUERY(192);
        SD_QUERY(224);
        SD_QUERY(256);
    }
#undef SD_QUERY
}

static ggml_status attention_execute(const ggml_ext_execution_context* execution, ggml_tensor* const* nodes, int count, void* state, void*) {
    const auto& plan = *static_cast<AttentionPlan*>(state);
    const auto* attn = nodes[plan.index];
    auto device      = ggml_backend_get_device(execution->backend);
    auto reg         = ggml_backend_dev_backend_reg(device);
    auto query       = reinterpret_cast<ggml_ext_cuda_context_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_ext_cuda_context_v1"));
    ggml_ext_cuda_context context;
    if (!query || !query(execution->backend, &context))
        return GGML_STATUS_FAILED;
    auto stream = static_cast<cudaStream_t>(context.stream);
    auto* base  = static_cast<char*>(ggml_backend_buffer_get_base(execution->workspace));
    half* work[4];
    for (int i = 0; i < 4; ++i)
        work[i] = reinterpret_cast<half*>(base + plan.offsets[i]);
    const half* inputs[3] = {work[0], work[1], work[2]};
    for (int i = 0; i < 3; ++i) {
        const auto* src   = attn->src[i];
        const float scale = i == 0 ? param_f32(attn, 0) : 1.0f;
        if (i == 0 && plan.index == 3) {
            if (nodes[2]->op_params[0])
                query_launch<true>(stream, nodes, work[0], scale);
            else
                query_launch<false>(stream, nodes, work[0], scale);
        } else if (i != 0 && src->type == GGML_TYPE_F16 && ggml_is_contiguous(src) && uintptr_t(src->data) % 16 == 0) {
            inputs[i] = static_cast<const half*>(src->data);
        } else {
            auto kernel = src->type == GGML_TYPE_F16 ? copy_to_half<half> : copy_to_half<float>;
            kernel<<<(ggml_nelements(src) + 255) / 256, 256, 0, stream>>>(static_cast<const char*>(src->data), work[i],
                                                                          ggml_nelements(src), src->ne[0], src->ne[1], src->ne[2], src->nb[1], src->nb[2], src->nb[3], scale);
        }
        if (cudaGetLastError() != cudaSuccess)
            return GGML_STATUS_FAILED;
    }
    const auto* q           = attn->src[0];
    const auto* k           = attn->src[1];
    sd_fattn_cute::Params p = {reinterpret_cast<const cutlass::half_t*>(inputs[0]),
                               reinterpret_cast<const cutlass::half_t*>(inputs[1]), reinterpret_cast<const cutlass::half_t*>(inputs[2]),
                               reinterpret_cast<cutlass::half_t*>(work[3]), int(q->ne[1]), int(k->ne[1]), int(q->ne[2]), int(k->ne[2]), int(q->ne[0])};
    cudaError_t error       = cudaErrorInvalidValue;
    switch (padded_dim(p.dim)) {
        case 64:
            error = attention_launch<64>(context, p, q->ne[3]);
            break;
        case 80:
            error = attention_launch<80, true>(context, p, q->ne[3]);
            break;
        case 96:
            error = attention_launch<96, true>(context, p, q->ne[3]);
            break;
        case 112:
            error = attention_launch<112, true>(context, p, q->ne[3]);
            break;
        case 128:
            error = attention_launch<128>(context, p, q->ne[3]);
            break;
        case 192:
            error = attention_launch<192>(context, p, q->ne[3]);
            break;
        case 256:
            error = attention_launch<256>(context, p, q->ne[3]);
            break;
    }
    if (error != cudaSuccess)
        return GGML_STATUS_FAILED;
    float scale = 1.0f, bias = 0.0f;
    for (int i = plan.index + 1; i < count; ++i) {
        if (nodes[i]->op == GGML_OP_SCALE) {
            scale = param_f32(nodes[i], 0);
            bias  = param_f32(nodes[i], 1);
        }
    }
    auto* dst = nodes[count - 1];
    if (dst->op == GGML_OP_QUANTIZE_I8_CONVROT) {
        const int64_t width = dst->src[0]->ne[0], rows = ggml_nrows(dst->src[0]), padded_rows = (rows + 3) / 4 * 4;
        auto* packed = static_cast<int8_t*>(dst->data);
        if (rows != padded_rows && cudaMemsetAsync(packed + width * rows, 0, width * (padded_rows - rows), stream) != cudaSuccess)
            return GGML_STATUS_FAILED;
        if (cudaFuncSetAttribute(quantize_attention, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 int(cuda_device(device)->properties.sharedMemPerBlockOptin - 128)) != cudaSuccess)
            return GGML_STATUS_FAILED;
        quantize_attention<<<rows, 256, width * sizeof(float), stream>>>(work[3], packed,
                                                                         reinterpret_cast<float*>(packed + width * padded_rows), width, rows, scale, bias);
    } else {
        copy_from_half<<<(ggml_nelements(attn) + 255) / 256, 256, 0, stream>>>(work[3], static_cast<float*>(dst->data), ggml_nelements(attn), scale, bias, dst->op == GGML_OP_SCALE);
    }
    return cudaGetLastError() == cudaSuccess ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
}

static void attention_destroy(void* state, void*) {
    delete static_cast<AttentionPlan*>(state);
}

void sd_ggml_register_cuda_attention(ggml_backend_dev_t device) {
    if (!cuda_device(device))
        return;
    ggml_ext_kernel_desc desc = {};
    desc.abi_version          = GGML_EXT_ABI_VERSION;
    desc.name                 = "sd.cute_attention.cuda.v1";
    desc.op                   = GGML_OP_FLASH_ATTN_EXT;
    desc.priority             = 200;
    desc.supports             = attention_supports;
    desc.preferred            = attention_preferred;
    desc.match                = attention_match;
    desc.optimize             = attention_optimize;
    desc.prepare              = attention_prepare;
    desc.execute              = attention_execute;
    desc.destroy              = attention_destroy;
    GGML_ASSERT(ggml_ext_register_kernel(device, &desc));
    desc.name      = "sd.rms_rope_cute_attention.cuda.v1";
    desc.op        = GGML_OP_RMS_NORM;
    desc.priority  = 300;
    desc.supports  = nullptr;
    desc.preferred = nullptr;
    desc.optimize  = nullptr;
    GGML_ASSERT(ggml_ext_register_kernel(device, &desc));
}
