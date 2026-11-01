/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SD_CORE_CUDA_SAGEATTENTION_ATTN_UTILS_CUH
#define SD_CORE_CUDA_SAGEATTENTION_ATTN_UTILS_CUH
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>

#include "cp_async.cuh"
#include "math.cuh"
#include "mma.cuh"
#include "numeric_conversion.cuh"
#include "permuted_smem.cuh"

namespace sageattention {

#define WARP_SIZE 32

#define S_FP8_OFFSET 8.807f
#define S_FP8_OFFSET_EXP 6680.8477f
#define S_FP8_OFFSET_EXP_INV 0.0022326917f

#define SAGE_DIV_CEIL(M, N) (((M) + (N)-1) / (N))

    enum class MaskMode {
        kNone   = 0,
        kCausal = 1,
    };

    enum class DataType {
        kHalf,
        kInt8,
        kInt4,
        kE4M3,
        kE5M2,
    };

    enum class QuantGranularity {
        kPerTensor = 0,
        kPerBlock  = 1,
        kPerWarp   = 2,
        kPerThread = 3,
    };

    enum class ComputeUnit {
        kTensorCore,
        kCudaCore,
    };

    __device__ __forceinline__ uint32_t get_warp_id() {
        return threadIdx.y;
    }

    __device__ __forceinline__ uint32_t get_lane_id() {
        return threadIdx.x;
    }

    template <uint32_t num_warps_q, uint32_t num_warps_k>
    __device__ __forceinline__ uint32_t get_warp_idx_q() {
        return get_warp_id() / num_warps_k;
    }

    template <uint32_t num_warps_q, uint32_t num_warps_k>
    __device__ __forceinline__ uint32_t get_warp_idx_k() {
        return get_warp_id() % num_warps_k;
    }

    template <uint32_t global_to_shared_line_lanes, uint32_t global_to_shared_copy_lines_per_warp_per_iter, uint32_t smem_iters_row, uint32_t smem_iters_col, SwizzleMode swizzle_mode, uint32_t stride, uint32_t CTA, typename T>
    __device__ __forceinline__ void load_global_to_share(T** lane_ptr, uint32_t& smem_offset, const uint32_t& gmem_stride, const smem_t<swizzle_mode, stride>& smem) {
        static_assert(global_to_shared_copy_lines_per_warp_per_iter * global_to_shared_line_lanes == WARP_SIZE);
        static_assert(std::is_same<T, half>::value || std::is_same<T, int8_t>::value);

        constexpr uint32_t pack_size = std::is_same<T, half>::value ? 8 : 16;

#pragma unroll
        for (uint32_t i = 0; i < smem_iters_col; i++) {
#pragma unroll
            for (uint32_t j = 0; j < smem_iters_row; j++) {
                smem.load_128b_async(smem_offset, *lane_ptr);
                *lane_ptr += (global_to_shared_line_lanes * pack_size);
                smem_offset = smem.advance_offset_by_column<global_to_shared_line_lanes>(smem_offset);
            }

            smem_offset = smem.advance_offset_by_row<global_to_shared_copy_lines_per_warp_per_iter>(smem_offset - (smem_iters_row * global_to_shared_line_lanes));
            *lane_ptr += ((global_to_shared_copy_lines_per_warp_per_iter * gmem_stride) - (smem_iters_row * global_to_shared_line_lanes * pack_size));
        }
        smem_offset -= (smem_iters_col * global_to_shared_copy_lines_per_warp_per_iter * stride);
        *lane_ptr += (CTA - smem_iters_col * global_to_shared_copy_lines_per_warp_per_iter) * gmem_stride;
    }

    // with predicate
    template <uint32_t global_to_shared_line_lanes, uint32_t global_to_shared_copy_lines_per_warp_per_iter, uint32_t smem_iters_row, uint32_t smem_iters_col, SwizzleMode swizzle_mode, uint32_t stride, uint32_t CTA, typename T>
    __device__ __forceinline__ void load_global_to_share(T** lane_ptr, uint32_t& smem_offset, const uint32_t& gmem_stride, const smem_t<swizzle_mode, stride>& smem, uint32_t base_idx, uint32_t max_len) {
        static_assert(global_to_shared_copy_lines_per_warp_per_iter * global_to_shared_line_lanes == WARP_SIZE);
        static_assert(std::is_same<T, half>::value || std::is_same<T, int8_t>::value);

        constexpr uint32_t pack_size = std::is_same<T, half>::value ? 8 : 16;

#pragma unroll
        for (uint32_t i = 0; i < smem_iters_col; i++) {
#pragma unroll
            for (uint32_t j = 0; j < smem_iters_row; j++) {
                // Masked probabilities still propagate NaNs from stale V lanes through 0 * NaN.
                smem.load_128b_async<std::is_same<T, half>::value ? cp_async::SharedMemFillMode::kFillZero : cp_async::SharedMemFillMode::kNoFill>(smem_offset, *lane_ptr, base_idx < max_len);
                *lane_ptr += (global_to_shared_line_lanes * pack_size);
                smem_offset = smem.advance_offset_by_column<global_to_shared_line_lanes>(smem_offset);
            }

            smem_offset = smem.advance_offset_by_row<global_to_shared_copy_lines_per_warp_per_iter>(smem_offset - (smem_iters_row * global_to_shared_line_lanes));
            *lane_ptr += ((global_to_shared_copy_lines_per_warp_per_iter * gmem_stride) - (smem_iters_row * global_to_shared_line_lanes * pack_size));
            base_idx += global_to_shared_copy_lines_per_warp_per_iter;
        }
        smem_offset -= (smem_iters_col * global_to_shared_copy_lines_per_warp_per_iter * stride);
        *lane_ptr += (CTA - smem_iters_col * global_to_shared_copy_lines_per_warp_per_iter) * gmem_stride;
    }

    template <uint32_t global_to_shared_line_lanes, uint32_t global_to_shared_copy_lines_per_warp_per_iter, uint32_t smem_iters_row, uint32_t smem_iters_col, SwizzleMode swizzle_mode, uint32_t stride, uint32_t CTA>
    __device__ __forceinline__ void load_fp8_V_global_to_share(int8_t** lane_ptr, uint32_t& smem_offset, const uint32_t& gmem_stride, const smem_t<swizzle_mode, stride>& smem) {
        static_assert(global_to_shared_copy_lines_per_warp_per_iter * global_to_shared_line_lanes == WARP_SIZE);
        constexpr uint32_t pack_size_fp8 = 16;

#pragma unroll
        for (uint32_t i = 0; i < smem_iters_col; i++) {
#pragma unroll
            for (uint32_t j = 0; j < smem_iters_row; j++) {
                smem.load_128b_async(smem_offset, *lane_ptr);
                *lane_ptr += (global_to_shared_line_lanes * pack_size_fp8);
                smem_offset = smem.advance_offset_by_column<global_to_shared_line_lanes>(smem_offset);
            }

            smem_offset = smem.advance_offset_by_row<global_to_shared_copy_lines_per_warp_per_iter>(smem_offset - (smem_iters_row * global_to_shared_line_lanes));
            *lane_ptr += ((global_to_shared_copy_lines_per_warp_per_iter * gmem_stride) - (smem_iters_row * global_to_shared_line_lanes * pack_size_fp8));
        }
        smem_offset -= (smem_iters_col * global_to_shared_copy_lines_per_warp_per_iter * stride);
        // for QK: *lane_ptr += (CTA - smem_iters_col * global_to_shared_copy_lines_per_warp_per_iter) * gmem_stride;
        *lane_ptr += CTA;  // ! prevent underflow
        *lane_ptr -= (smem_iters_col * global_to_shared_copy_lines_per_warp_per_iter) * gmem_stride;
    }

    template <uint32_t num_warps_q, uint32_t num_warps_k, uint32_t num_tiles_q, uint32_t num_tiles_k, uint32_t num_tiles_qk_inner, SwizzleMode swizzle_mode, uint32_t stride, DataType DTypeQK>
    __device__ __forceinline__ void compute_int_qk(const smem_t<swizzle_mode, stride>& smem_Q, const smem_t<swizzle_mode, stride>& smem_K, int32_t RS[][num_tiles_k][8], uint32_t& offset_Q, uint32_t& offset_K) {
        static_assert(DTypeQK == DataType::kInt8 || DTypeQK == DataType::kInt4);

        uint32_t RQ[num_tiles_q][4];
        uint32_t RK[4];

        // the first iteration, mma mode is kInit
#pragma unroll
        for (uint32_t iter = 0; iter < 1; iter++) {
            // load RQ
#pragma unroll
            for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
                smem_Q.ldmatrix_m8n8x4(offset_Q, RQ[fq]);
                offset_Q = smem_Q.advance_offset_by_row<16>(offset_Q);
            }
            // ! using permutation invariance
            offset_Q = smem_Q.advance_offset_by_column<2>(offset_Q - (num_tiles_q * 16 * stride), iter);

#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
                // load RK
                smem_K.ldmatrix_m8n8x4(offset_K, RK);
                offset_K = smem_K.advance_offset_by_row<16>(offset_K);

                // mma
#pragma unroll
                for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
                    if constexpr (DTypeQK == DataType::kInt8) {
                        mma::mma_sync_m16n16k32_row_col_s8s8s32<mma::MMAMode::kInit>(RS[fq][fk], RQ[fq], RK);
                    } else if constexpr (DTypeQK == DataType::kInt4) {
                        mma::mma_sync_m16n16k64_row_col_s4s4s32<mma::MMAMode::kInit>(RS[fq][fk], RQ[fq], RK);
                    }
                }
            }
            offset_K = smem_K.advance_offset_by_column<2>(offset_K - (num_tiles_k * 16 * stride), iter);
        }

        // following iteration, mma mode is kInplace
#pragma unroll
        for (uint32_t iter = 1; iter < num_tiles_qk_inner; iter++) {
            // load RQ
#pragma unroll
            for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
                smem_Q.ldmatrix_m8n8x4(offset_Q, RQ[fq]);
                offset_Q = smem_Q.advance_offset_by_row<16>(offset_Q);
            }
            offset_Q = smem_Q.advance_offset_by_column<2>(offset_Q - (num_tiles_q * 16 * stride), iter);

#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
                // load RK
                smem_K.ldmatrix_m8n8x4(offset_K, RK);
                offset_K = smem_K.advance_offset_by_row<16>(offset_K);

                // mma
#pragma unroll
                for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
                    if constexpr (DTypeQK == DataType::kInt8) {
                        mma::mma_sync_m16n16k32_row_col_s8s8s32<mma::MMAMode::kInplaceUpdate>(RS[fq][fk], RQ[fq], RK);
                    } else if constexpr (DTypeQK == DataType::kInt4) {
                        mma::mma_sync_m16n16k64_row_col_s4s4s32<mma::MMAMode::kInplaceUpdate>(RS[fq][fk], RQ[fq], RK);
                    }
                }
            }
            offset_K = smem_K.advance_offset_by_column<2>(offset_K - (num_tiles_k * 16 * stride), iter);
        }

        offset_Q -= (2 * num_tiles_qk_inner);
        offset_K -= (2 * num_tiles_qk_inner);
    }

    // for case when num_tiles_qk_inner = 1
    template <uint32_t num_warps_q, uint32_t num_warps_k, uint32_t num_tiles_q, uint32_t num_tiles_k, uint32_t num_tiles_qk_inner, SwizzleMode swizzle_mode, uint32_t stride, DataType DTypeQK>
    __device__ __forceinline__ void compute_int_qk(const smem_t<swizzle_mode, stride>& smem_K, int32_t RS[][num_tiles_k][8], uint32_t RQ[][4], uint32_t offset_K) {
        static_assert(DTypeQK == DataType::kInt8 || DTypeQK == DataType::kInt4);
        static_assert(num_tiles_qk_inner == 1);

        uint32_t RK[4];

        // mma mode is kInit
#pragma unroll
        for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
            // load RK
            smem_K.ldmatrix_m8n8x4(offset_K, RK);
            offset_K = smem_K.advance_offset_by_row<16>(offset_K);

            // mma
#pragma unroll
            for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
                if constexpr (DTypeQK == DataType::kInt8) {
                    mma::mma_sync_m16n16k32_row_col_s8s8s32<mma::MMAMode::kInit>(RS[fq][fk], RQ[fq], RK);
                } else if constexpr (DTypeQK == DataType::kInt4) {
                    mma::mma_sync_m16n16k64_row_col_s4s4s32<mma::MMAMode::kInit>(RS[fq][fk], RQ[fq], RK);
                }
            }
        }
    }

    template <uint32_t num_tiles_q, uint32_t num_tiles_k, typename DTypeQKAccum>
    __device__ __forceinline__ void apply_causal_mask(const uint32_t& Q_idx_lane_base, const uint32_t& K_idx_lane_base, DTypeQKAccum RS[][num_tiles_k][8]) {
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
                for (uint32_t k = 0; k < 8; k++) {
                    const uint32_t q_idx       = Q_idx_lane_base + fq * 16 + 8 * ((k % 4) / 2);
                    const uint32_t kv_idx      = K_idx_lane_base + fk * 16 + 8 * (k / 4) + k % 2;
                    const bool out_of_boundary = (kv_idx > q_idx);

                    if constexpr (std::is_same<DTypeQKAccum, float>::value) {
                        RS[fq][fk][k] = (out_of_boundary ? -5000000.0f : RS[fq][fk][k]);
                    } else if constexpr (std::is_same<DTypeQKAccum, half>::value) {
                        RS[fq][fk][k] = (out_of_boundary ? __float2half_rn(-50000.0f) : RS[fq][fk][k]);
                    }
                }
            }
        }
    }

    template <uint32_t num_tiles_q, uint32_t num_tiles_k, typename DTypeQKAccum>
    __device__ __forceinline__ void apply_out_of_bound_mask(const uint32_t& K_idx_lane_base, DTypeQKAccum RS[][num_tiles_k][8], const uint32_t& kv_len) {
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
                for (uint32_t k = 0; k < 8; k++) {
                    const uint32_t kv_idx      = K_idx_lane_base + fk * 16 + 8 * (k / 4) + k % 2;
                    const bool out_of_boundary = (kv_idx >= kv_len);

                    if constexpr (std::is_same<DTypeQKAccum, float>::value) {
                        RS[fq][fk][k] = (out_of_boundary ? -5000000.0f : RS[fq][fk][k]);
                    } else if constexpr (std::is_same<DTypeQKAccum, half>::value) {
                        RS[fq][fk][k] = (out_of_boundary ? __float2half_rn(-50000.0f) : RS[fq][fk][k]);
                    }
                }
            }
        }
    }

    // for DTypeQKAccum float
    template <uint32_t num_tiles_q, uint32_t num_tiles_k, uint32_t num_tiles_v, bool use_half_o_scale, bool exp_offset, bool fuse_scale = false, typename DTypeSVAccum>
    __device__ __forceinline__ void update_mdo(float RS[][num_tiles_k][8], DTypeSVAccum RO[][num_tiles_v][8], float m[][2], float d[][2], const float& sm_scale) {
        static_assert(std::is_same<DTypeSVAccum, half>::value || (!use_half_o_scale));
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t k = 0; k < 2; k++) {
                // assign the smallest value possible
                float m_prev = m[fq][k];
                float m_temp = -5000000.0f;
#pragma unroll
                for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
                    float m_local = max(max(RS[fq][fk][k * 2 + 0], RS[fq][fk][k * 2 + 1]),
                                        max(RS[fq][fk][k * 2 + 4], RS[fq][fk][k * 2 + 5]));
                    m_temp        = max(m_temp, m_local);
                }

                if constexpr (!fuse_scale) {
                    if constexpr (exp_offset) {
                        m_temp = fmaf(m_temp, sm_scale, -S_FP8_OFFSET);
                    } else {
                        m_temp *= sm_scale;
                    }
                } else if constexpr (exp_offset) {
                    m_temp += (-S_FP8_OFFSET);
                }

                // exchange element with the 4 threads in the row
                m_temp = max(m_temp, __shfl_xor_sync(0xffffffff, m_temp, 0x1));  // 0 exchange with 1, 2 exchange with 3
                m_temp = max(m_temp, __shfl_xor_sync(0xffffffff, m_temp, 0x2));  // 0 exchange with 2, 1 exchange with 3

                m[fq][k] = max(m[fq][k], m_temp);

                float o_scale = math::ptx_exp2(m_prev - m[fq][k]);

                // update denominator
                d[fq][k] *= o_scale;

                half2 o_scale2;
                if constexpr (use_half_o_scale) {
                    o_scale2 = __floats2half2_rn(o_scale, o_scale);
                }

                // update RO
#pragma unroll
                for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
                    if constexpr (std::is_same<DTypeSVAccum, float>::value) {
                        RO[fq][fv][k * 2 + 0] *= o_scale;
                        RO[fq][fv][k * 2 + 1] *= o_scale;
                        RO[fq][fv][k * 2 + 4] *= o_scale;
                        RO[fq][fv][k * 2 + 5] *= o_scale;
                    } else if constexpr (std::is_same<DTypeSVAccum, half>::value) {
                        if constexpr (use_half_o_scale) {
                            ((half2*)RO[fq][fv])[k]     = __hmul2(((half2*)RO[fq][fv])[k], o_scale2);
                            ((half2*)RO[fq][fv])[k + 2] = __hmul2(((half2*)RO[fq][fv])[k + 2], o_scale2);
                        } else {
                            RO[fq][fv][k * 2 + 0] = __float2half_rn(__half2float(RO[fq][fv][k * 2 + 0]) * o_scale);
                            RO[fq][fv][k * 2 + 1] = __float2half_rn(__half2float(RO[fq][fv][k * 2 + 1]) * o_scale);
                            RO[fq][fv][k * 2 + 4] = __float2half_rn(__half2float(RO[fq][fv][k * 2 + 4]) * o_scale);
                            RO[fq][fv][k * 2 + 5] = __float2half_rn(__half2float(RO[fq][fv][k * 2 + 5]) * o_scale);
                        }
                    }
                }

                // raise RS to exponent
                float negative_m = -m[fq][k];
#pragma unroll
                for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
                    if constexpr (fuse_scale) {
                        RS[fq][fk][k * 2 + 0] = math::ptx_exp2(RS[fq][fk][k * 2 + 0] + negative_m);
                        RS[fq][fk][k * 2 + 1] = math::ptx_exp2(RS[fq][fk][k * 2 + 1] + negative_m);
                        RS[fq][fk][k * 2 + 4] = math::ptx_exp2(RS[fq][fk][k * 2 + 4] + negative_m);
                        RS[fq][fk][k * 2 + 5] = math::ptx_exp2(RS[fq][fk][k * 2 + 5] + negative_m);
                    } else {
                        RS[fq][fk][k * 2 + 0] = math::ptx_exp2(fmaf(RS[fq][fk][k * 2 + 0], sm_scale, negative_m));
                        RS[fq][fk][k * 2 + 1] = math::ptx_exp2(fmaf(RS[fq][fk][k * 2 + 1], sm_scale, negative_m));
                        RS[fq][fk][k * 2 + 4] = math::ptx_exp2(fmaf(RS[fq][fk][k * 2 + 4], sm_scale, negative_m));
                        RS[fq][fk][k * 2 + 5] = math::ptx_exp2(fmaf(RS[fq][fk][k * 2 + 5], sm_scale, negative_m));
                    }
                }
            }
        }
    }

    template <uint32_t num_tiles_q, uint32_t num_tiles_k, typename T>
    __device__ __forceinline__ void RS_32_to_16(T RS[][num_tiles_k][8], uint32_t RS_16[][num_tiles_k][4]) {
        static_assert(sizeof(T) == 4);
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
                ((half2*)RS_16[fq][fk])[0] = __float22half2_rn(((float2*)RS[fq][fk])[0]);
                ((half2*)RS_16[fq][fk])[1] = __float22half2_rn(((float2*)RS[fq][fk])[1]);
                ((half2*)RS_16[fq][fk])[2] = __float22half2_rn(((float2*)RS[fq][fk])[2]);
                ((half2*)RS_16[fq][fk])[3] = __float22half2_rn(((float2*)RS[fq][fk])[3]);
            }
        }
    }

    template <uint32_t num_tiles_q, uint32_t num_tiles_k>
    __device__ __forceinline__ void RS_32_to_8(float RS[][num_tiles_k][8], uint32_t RS_8[][num_tiles_k / 2][4]) {
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++) {
                floatx4_to_e4m3x4(RS_8[fq][fk], RS[fq][fk * 2 + 0], RS[fq][fk * 2 + 0] + 4);
                floatx4_to_e4m3x4(RS_8[fq][fk] + 1, RS[fq][fk * 2 + 0] + 2, RS[fq][fk * 2 + 0] + 6);
                floatx4_to_e4m3x4(RS_8[fq][fk] + 2, RS[fq][fk * 2 + 1], RS[fq][fk * 2 + 1] + 4);
                floatx4_to_e4m3x4(RS_8[fq][fk] + 3, RS[fq][fk * 2 + 1] + 2, RS[fq][fk * 2 + 1] + 6);
            }
        }
    }

    template <uint32_t num_tiles_q, uint32_t num_tiles_k>
    __device__ __forceinline__ void RS_16_to_8(uint32_t RS[][num_tiles_k][4], uint32_t RS_8[][num_tiles_k / 2][4]) {
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++) {
                halfx4_to_e4m3x4(RS_8[fq][fk], RS[fq][fk * 2 + 0], RS[fq][fk * 2 + 0] + 2);
                halfx4_to_e4m3x4(RS_8[fq][fk] + 1, RS[fq][fk * 2 + 0] + 1, RS[fq][fk * 2 + 0] + 3);
                halfx4_to_e4m3x4(RS_8[fq][fk] + 2, RS[fq][fk * 2 + 1], RS[fq][fk * 2 + 1] + 2);
                halfx4_to_e4m3x4(RS_8[fq][fk] + 3, RS[fq][fk * 2 + 1] + 1, RS[fq][fk * 2 + 1] + 3);
            }
        }
    }

    template <uint32_t num_tiles_q, uint32_t num_tiles_k>
    __device__ __forceinline__ void RS_8_to_16(uint32_t RS_8[][num_tiles_k / 2][4], uint32_t RS[][num_tiles_k][4]) {
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++) {
                e4m3x4_to_halfx4(RS[fq][fk * 2 + 0], RS[fq][fk * 2 + 0] + 2, RS_8[fq][fk]);
                e4m3x4_to_halfx4(RS[fq][fk * 2 + 0] + 1, RS[fq][fk * 2 + 0] + 3, RS_8[fq][fk] + 1);
                e4m3x4_to_halfx4(RS[fq][fk * 2 + 1], RS[fq][fk * 2 + 1] + 2, RS_8[fq][fk] + 2);
                e4m3x4_to_halfx4(RS[fq][fk * 2 + 1] + 1, RS[fq][fk * 2 + 1] + 3, RS_8[fq][fk] + 3);
            }
        }
    }

    template <uint32_t num_tiles_q, uint32_t num_tiles_k, ComputeUnit compute_unit = ComputeUnit::kTensorCore, typename T>
    __device__ __forceinline__ void accumulate_d(T RS[][num_tiles_k][(compute_unit == ComputeUnit::kTensorCore) ? 4 : 8], float d[][2]) {
        // for compute unit cuda core, RS is float
        // for compute unit tensor core, RS is packed half
        static_assert((std::is_same<T, float>::value && compute_unit == ComputeUnit::kCudaCore) ||
                      (std::is_same<T, uint32_t>::value && compute_unit == ComputeUnit::kTensorCore));

#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
                if constexpr (compute_unit == ComputeUnit::kTensorCore) {
                    // full accumulate with tensor core
                    mma::rowsum_f16f16f32(d[fq], (uint32_t*)(RS[fq][fk]));
                } else if constexpr (compute_unit == ComputeUnit::kCudaCore) {
                    // partial accumulate with cuda core
                    d[fq][0] += RS[fq][fk][0] + RS[fq][fk][1] + RS[fq][fk][4] + RS[fq][fk][5];
                    d[fq][1] += RS[fq][fk][2] + RS[fq][fk][3] + RS[fq][fk][6] + RS[fq][fk][7];
                }
            }
        }
    }

    template <uint32_t num_tiles_q, uint32_t num_tiles_k>
    __device__ __forceinline__ void accumulate_d_f8(uint32_t RS[][num_tiles_k / 2][4], float d[][2]) {
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++) {
                mma::rowsum_f8f8f32(d[fq], RS[fq][fk]);
            }
        }
    }

    template <uint32_t num_warps_q, uint32_t num_warps_k, uint32_t num_tiles_q, uint32_t num_tiles_k, uint32_t num_tiles_v, SwizzleMode swizzle_mode, uint32_t stride, typename DTypeSVAccum>
    __device__ __forceinline__ void compute_fp16_sv(const smem_t<swizzle_mode, stride>& smem_V, uint32_t RS_f16[][num_tiles_k][4], DTypeSVAccum RO[][num_tiles_v][8], float d[][2]) {
        uint32_t smem_V_row_base = get_warp_idx_k<num_warps_q, num_warps_k>() * (num_tiles_k * 16) + get_lane_id() % 16;
        uint32_t smem_V_col_base = get_lane_id() / 16;
#pragma unroll
        for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
            for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
                // load RV
                uint32_t RV[4];
                uint32_t offset_V = (smem_V).get_permuted_offset(smem_V_row_base + fk * 16, smem_V_col_base + fv * 2);
                smem_V.ldmatrix_m8n8x4_trans(offset_V, RV);
#pragma unroll
                for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
                    if constexpr (std::is_same<DTypeSVAccum, float>::value) {
                        mma::mma_sync_m16n16k16_row_col_f16f16f32(RO[fq][fv], RS_f16[fq][fk], RV);
                    } else if constexpr (std::is_same<DTypeSVAccum, half>::value) {
                        mma::mma_sync_m16n16k16_row_col_f16f16f16((uint32_t*)RO[fq][fv], RS_f16[fq][fk], RV);
                    }
                }
            }
        }
    }

    template <uint32_t num_warps_q, uint32_t num_warps_k, uint32_t num_tiles_q, uint32_t num_tiles_k, uint32_t num_tiles_v, SwizzleMode swizzle_mode, uint32_t stride, uint32_t RS_width = 4, typename T, typename DTypeSVAccum>
    __device__ __forceinline__ void compute_fp16_sv_permuted(const smem_t<swizzle_mode, stride>& smem_V, T RS_f16[][num_tiles_k][RS_width], DTypeSVAccum RO[][num_tiles_v][8], float d[][2], uint32_t& offset_V) {
        static_assert(sizeof(T) == 4);

        // ! be sure you know what you are doing
#pragma unroll
        for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
            for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
                // load RV
                uint32_t RV[4];
                smem_V.ldmatrix_m8n8x4_trans(offset_V, RV);
#pragma unroll
                for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
                    if constexpr (std::is_same<DTypeSVAccum, float>::value) {
                        mma::mma_sync_m16n16k16_row_col_f16f16f32(RO[fq][fv], (uint32_t*)(RS_f16[fq][fk]), RV);
                    } else if constexpr (std::is_same<DTypeSVAccum, half>::value) {
                        mma::mma_sync_m16n16k16_row_col_f16f16f16((uint32_t*)RO[fq][fv], (uint32_t*)(RS_f16[fq][fk]), RV);
                    }
                }

                offset_V = smem_V.advance_offset_by_column<2>(offset_V, fv);
            }
            offset_V = smem_V.advance_offset_by_row<16>(offset_V - (2 * num_tiles_v));
        }

        // make offset_V their original value
        offset_V -= (16 * num_tiles_k * stride);
    }

    template <uint32_t num_warps_q, uint32_t num_warps_k, uint32_t num_tiles_q, uint32_t num_tiles_k, uint32_t num_tiles_v, SwizzleMode swizzle_mode, uint32_t stride, uint32_t RS_width = 4, typename T, typename DTypeSVAccum>
    __device__ __forceinline__ void compute_fp16_sv_permuted_inst_buf(const smem_t<swizzle_mode, stride>& smem_V, T RS_f16[][num_tiles_k][RS_width], DTypeSVAccum RO[][num_tiles_v][8], float d[][2], uint32_t& offset_V) {
        static_assert(sizeof(T) == 4);
        static_assert(std::is_same<DTypeSVAccum, float>::value);

        uint32_t RO_inst_buf[num_tiles_q][num_tiles_v][4];

        // ! be sure you know what you are doing
#pragma unroll
        for (uint32_t fk = 0; fk < 1; fk++) {
#pragma unroll
            for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
                // load RV
                uint32_t RV[4];
                smem_V.ldmatrix_m8n8x4_trans(offset_V, RV);
#pragma unroll
                for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
                    {
                        mma::mma_sync_m16n16k16_row_col_f16f16f16<mma::MMAMode::kInit>((uint32_t*)RO_inst_buf[fq][fv], (uint32_t*)(RS_f16[fq][fk]), RV);
                    }
                }

                offset_V = smem_V.advance_offset_by_column<2>(offset_V, fv);
            }
            offset_V = smem_V.advance_offset_by_row<16>(offset_V - (2 * num_tiles_v));
        }

#pragma unroll
        for (uint32_t fk = 1; fk < num_tiles_k; fk++) {
#pragma unroll
            for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
                // load RV
                uint32_t RV[4];
                smem_V.ldmatrix_m8n8x4_trans(offset_V, RV);
#pragma unroll
                for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
                    {
                        mma::mma_sync_m16n16k16_row_col_f16f16f16<mma::MMAMode::kInplaceUpdate>((uint32_t*)RO_inst_buf[fq][fv], (uint32_t*)(RS_f16[fq][fk]), RV);
                    }
                }

                offset_V = smem_V.advance_offset_by_column<2>(offset_V, fv);
            }
            offset_V = smem_V.advance_offset_by_row<16>(offset_V - (2 * num_tiles_v));
        }

        // accumulate into RO
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
                RO[fq][fv][0] += __half2float(((half2*)RO_inst_buf[fq][fv])[0].x);
                RO[fq][fv][1] += __half2float(((half2*)RO_inst_buf[fq][fv])[0].y);
                RO[fq][fv][2] += __half2float(((half2*)RO_inst_buf[fq][fv])[1].x);
                RO[fq][fv][3] += __half2float(((half2*)RO_inst_buf[fq][fv])[1].y);
                RO[fq][fv][4] += __half2float(((half2*)RO_inst_buf[fq][fv])[2].x);
                RO[fq][fv][5] += __half2float(((half2*)RO_inst_buf[fq][fv])[2].y);
                RO[fq][fv][6] += __half2float(((half2*)RO_inst_buf[fq][fv])[3].x);
                RO[fq][fv][7] += __half2float(((half2*)RO_inst_buf[fq][fv])[3].y);
            }
        }

        // make offset_V their original value
        offset_V -= (16 * num_tiles_k * stride);
    }

    template <uint32_t num_tiles_q, uint32_t num_tiles_v,
              ComputeUnit compute_unit = ComputeUnit::kTensorCore,  // compute unit for accumulate_d
              typename DTypeQKAccum,
              typename DTypeSVAccum>
    __device__ __forceinline__ void normalize_d(DTypeSVAccum RO[][num_tiles_v][8], DTypeQKAccum m[][2], float d[][2]) {
        if constexpr (compute_unit == ComputeUnit::kCudaCore) {
            // accumulate_d performs partial accumulation with cuda core
            // aggregate d
#pragma unroll
            for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
                for (uint32_t k = 0; k < 2; k++) {
                    d[fq][k] += __shfl_xor_sync(0xffffffff, d[fq][k], 0x1);  // sum 0 and 1, 2 and 3
                    d[fq][k] += __shfl_xor_sync(0xffffffff, d[fq][k], 0x2);  // sum 0 and 2, 1 and 3
                }
            }
        }

        // divide O by d
        float d_rcp[num_tiles_q][2];
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t k = 0; k < 2; k++) {
                // TODO: check m to prevent nan
                d_rcp[fq][k] = math::ptx_rcp(d[fq][k]);
            }
        }

#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
#pragma unroll
                for (uint32_t k = 0; k < 8; k++) {
                    if constexpr (std::is_same<DTypeSVAccum, float>::value) {
                        RO[fq][fv][k] *= d_rcp[fq][(k % 4) / 2];
                    } else if constexpr (std::is_same<DTypeSVAccum, half>::value) {
                        RO[fq][fv][k] = __float2half_rn(__half2float(RO[fq][fv][k]) * d_rcp[fq][(k % 4) / 2]);
                    }
                }
            }
        }
    }

    template <uint32_t num_warps_q, uint32_t num_warps_k, uint32_t num_tiles_q, uint32_t num_tiles_k, uint32_t num_tiles_v, SwizzleMode swizzle_mode, uint32_t stride, typename DTypeSVAccum>
    __device__ __forceinline__ void compute_fp8_sv(const smem_t<swizzle_mode, stride>& smem_V, uint32_t RS_f8[][num_tiles_k / 2][4], DTypeSVAccum RO[][num_tiles_v][8], float d[][2]) {
        uint32_t smem_V_row_base = get_lane_id() % 8 + (get_lane_id() / 16) * 8;
        // uint32_t smem_V_col_base = get_warp_idx_k<num_warps_q, num_warps_k>() * ((16 * num_tiles_k) / 16) + (get_lane_id() / 8) % 2;
        uint32_t smem_V_col_base = (get_lane_id() / 8) % 2;
#pragma unroll
        for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++) {
            uint32_t offset_V = smem_V.get_permuted_offset(smem_V_row_base, smem_V_col_base + fk * 2);
#pragma unroll
            for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
                // load RV
                uint32_t RV[4];
                // uint32_t offset_V = (smem_V).get_permuted_offset(smem_V_row_base + fv * 16, smem_V_col_base + fk * 2);
                smem_V.ldmatrix_m8n8x4(offset_V, RV);
#pragma unroll
                for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
                    if constexpr (std::is_same<DTypeSVAccum, float>::value) {
                        mma::mma_sync_m16n16k32_row_col_f8f8f32(RO[fq][fv], RS_f8[fq][fk], RV);
                    } else if constexpr (std::is_same<DTypeSVAccum, half>::value) {
                        // ! Not Implemented
                    }
                }
                offset_V = smem_V.advance_offset_by_row<16>(offset_V);
            }
        }
    }

    template <uint32_t num_warps_q, uint32_t num_warps_k, uint32_t num_tiles_q, uint32_t num_tiles_k, uint32_t num_tiles_v, SwizzleMode swizzle_mode, uint32_t stride, typename DTypeSVAccum>
    __device__ __forceinline__ void compute_fp8_sv_inst_buf(const smem_t<swizzle_mode, stride>& smem_V, uint32_t RS_f8[][num_tiles_k / 2][4], DTypeSVAccum RO[][num_tiles_v][8], float d[][2]) {
        uint32_t smem_V_row_base = get_lane_id() % 8 + (get_lane_id() / 16) * 8;
        // uint32_t smem_V_col_base = get_warp_idx_k<num_warps_q, num_warps_k>() * ((16 * num_tiles_k) / 16) + (get_lane_id() / 8) % 2;
        uint32_t smem_V_col_base = (get_lane_id() / 8) % 2;

        float RO_inst_buf[num_tiles_q][num_tiles_v][8];

#pragma unroll
        for (uint32_t fk = 0; fk < 1; fk++) {
            uint32_t offset_V = smem_V.get_permuted_offset(smem_V_row_base, smem_V_col_base + fk * 2);
#pragma unroll
            for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
                // load RV
                uint32_t RV[4];
                // uint32_t offset_V = (smem_V).get_permuted_offset(smem_V_row_base + fv * 16, smem_V_col_base + fk * 2);
                smem_V.ldmatrix_m8n8x4(offset_V, RV);
#pragma unroll
                for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
                    if constexpr (std::is_same<DTypeSVAccum, float>::value) {
                        mma::mma_sync_m16n16k32_row_col_f8f8f32<mma::MMAMode::kInit>(RO_inst_buf[fq][fv], RS_f8[fq][fk], RV);
                    } else if constexpr (std::is_same<DTypeSVAccum, half>::value) {
                        // ! Not Implemented
                    }
                }
                offset_V = smem_V.advance_offset_by_row<16>(offset_V);
            }
        }

#pragma unroll
        for (uint32_t fk = 1; fk < num_tiles_k / 2; fk++) {
            uint32_t offset_V = smem_V.get_permuted_offset(smem_V_row_base, smem_V_col_base + fk * 2);
#pragma unroll
            for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
                // load RV
                uint32_t RV[4];
                // uint32_t offset_V = (smem_V).get_permuted_offset(smem_V_row_base + fv * 16, smem_V_col_base + fk * 2);
                smem_V.ldmatrix_m8n8x4(offset_V, RV);
#pragma unroll
                for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
                    if constexpr (std::is_same<DTypeSVAccum, float>::value) {
                        mma::mma_sync_m16n16k32_row_col_f8f8f32<mma::MMAMode::kInplaceUpdate>(RO_inst_buf[fq][fv], RS_f8[fq][fk], RV);
                    } else if constexpr (std::is_same<DTypeSVAccum, half>::value) {
                        // ! Not Implemented
                    }
                }
                offset_V = smem_V.advance_offset_by_row<16>(offset_V);
            }
        }

#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
                RO[fq][fv][0] += RO_inst_buf[fq][fv][0];
                RO[fq][fv][1] += RO_inst_buf[fq][fv][1];
                RO[fq][fv][2] += RO_inst_buf[fq][fv][2];
                RO[fq][fv][3] += RO_inst_buf[fq][fv][3];
                RO[fq][fv][4] += RO_inst_buf[fq][fv][4];
                RO[fq][fv][5] += RO_inst_buf[fq][fv][5];
                RO[fq][fv][6] += RO_inst_buf[fq][fv][6];
                RO[fq][fv][7] += RO_inst_buf[fq][fv][7];
            }
        }
    }

    template <uint32_t num_warps_q, uint32_t num_warps_k, uint32_t num_tiles_q, uint32_t num_tiles_k, uint32_t num_tiles_v, SwizzleMode swizzle_mode, uint32_t stride, typename DTypeSVAccum>
    __device__ __forceinline__ void compute_fp8_sv_inst_buf_fp16_accu(const smem_t<swizzle_mode, stride>& smem_V, uint32_t RS_f8[][num_tiles_k / 2][4], DTypeSVAccum RO[][num_tiles_v][8], float d[][2]) {
        uint32_t smem_V_row_base = get_lane_id() % 8 + (get_lane_id() / 16) * 8;
        // uint32_t smem_V_col_base = get_warp_idx_k<num_warps_q, num_warps_k>() * ((16 * num_tiles_k) / 16) + (get_lane_id() / 8) % 2;
        uint32_t smem_V_col_base = (get_lane_id() / 8) % 2;

        uint32_t RO_int32[num_tiles_q][num_tiles_v][4];

#pragma unroll
        for (uint32_t fk = 0; fk < 1; fk++) {
            uint32_t offset_V = smem_V.get_permuted_offset(smem_V_row_base, smem_V_col_base + fk * 2);
#pragma unroll
            for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
                // load RV
                uint32_t RV[4];
                // uint32_t offset_V = (smem_V).get_permuted_offset(smem_V_row_base + fv * 16, smem_V_col_base + fk * 2);
                smem_V.ldmatrix_m8n8x4(offset_V, RV);
#pragma unroll
                for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
                    if constexpr (std::is_same<DTypeSVAccum, float>::value) {
                        // mma::mma_sync_m16n16k32_row_col_f8f8f32<mma::MMAMode::kInit>(RO_inst_buf[fq][fv], RS_f8[fq][fk], RV);
                        mma::mma_sync_m16n16k32_row_col_f8f8f16<mma::MMAMode::kInit>(RO_int32[fq][fv], RS_f8[fq][fk], RV);
                    } else if constexpr (std::is_same<DTypeSVAccum, half>::value) {
                        // ! Not Implemented
                    }
                }
                offset_V = smem_V.advance_offset_by_row<16>(offset_V);
            }
        }

#pragma unroll
        for (uint32_t fk = 1; fk < num_tiles_k / 2; fk++) {
            uint32_t offset_V = smem_V.get_permuted_offset(smem_V_row_base, smem_V_col_base + fk * 2);
#pragma unroll
            for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
                // load RV
                uint32_t RV[4];
                // uint32_t offset_V = (smem_V).get_permuted_offset(smem_V_row_base + fv * 16, smem_V_col_base + fk * 2);
                smem_V.ldmatrix_m8n8x4(offset_V, RV);
#pragma unroll
                for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
                    if constexpr (std::is_same<DTypeSVAccum, float>::value) {
                        // mma::mma_sync_m16n16k32_row_col_f8f8f32<mma::MMAMode::kInplaceUpdate>(RO_inst_buf[fq][fv], RS_f8[fq][fk], RV);
                        mma::mma_sync_m16n16k32_row_col_f8f8f16<mma::MMAMode::kInplaceUpdate>(RO_int32[fq][fv], RS_f8[fq][fk], RV);
                    } else if constexpr (std::is_same<DTypeSVAccum, half>::value) {
                        // ! Not Implemented
                    }
                }
                offset_V = smem_V.advance_offset_by_row<16>(offset_V);
            }
        }
        float RO_tmp_float[2];
#pragma unroll
        for (int i = 0; i < num_tiles_q; i++) {
#pragma unroll
            for (int j = 0; j < num_tiles_v; j++) {
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    unpack_half2_from_uint32_to_float(RO_tmp_float, RO_int32[i][j][k]);
                    RO[i][j][k * 2 + 0] += RO_tmp_float[0];
                    RO[i][j][k * 2 + 1] += RO_tmp_float[1];
                }
            }
        }

        // #pragma unroll
        //   for (uint32_t fq = 0; fq < num_tiles_q; fq++)
        //   {
        // #pragma unroll
        //     for (uint32_t fv = 0; fv < num_tiles_v; fv++)
        //     {
        //       RO[fq][fv][0] += RO_inst_buf[fq][fv][0];
        //       RO[fq][fv][1] += RO_inst_buf[fq][fv][1];
        //       RO[fq][fv][2] += RO_inst_buf[fq][fv][2];
        //       RO[fq][fv][3] += RO_inst_buf[fq][fv][3];
        //       RO[fq][fv][4] += RO_inst_buf[fq][fv][4];
        //       RO[fq][fv][5] += RO_inst_buf[fq][fv][5];
        //       RO[fq][fv][6] += RO_inst_buf[fq][fv][6];
        //       RO[fq][fv][7] += RO_inst_buf[fq][fv][7];
        //     }
        //   }
    }

}  // namespace sageattention

#endif
