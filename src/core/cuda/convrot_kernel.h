#ifndef __SD_CORE_CUDA_CONVROT_KERNEL_H__
#define __SD_CORE_CUDA_CONVROT_KERNEL_H__
#include <cstdint>
#include "runtime.h"
namespace SD_CUDA_NAMESPACE {
    namespace sd_convrot {
        static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                x += cuda_shuffle_xor(x, offset, 32);
            return x;
        }
        static __device__ __forceinline__ float2 warp_reduce_sum(float2 x) {
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                x.x += cuda_shuffle_xor(x.x, offset, 32);
                x.y += cuda_shuffle_xor(x.y, offset, 32);
            }
            return x;
        }
        template <int Width>
        static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
            for (int offset = Width / 2; offset > 0; offset >>= 1)
                x = fmaxf(x, cuda_shuffle_xor(x, offset, Width));
            return x;
        }
        static __device__ float2 block_sum(float2 x, float2* shared) {
            x = warp_reduce_sum(x);
            if (threadIdx.x % 32 == 0)
                shared[threadIdx.x / 32] = x;
            __syncthreads();
            return warp_reduce_sum(shared[threadIdx.x % 32]);
        }
        static __device__ __forceinline__ float convrot_h4_cuda(float a, float b, float c, float d, int index) {
            a = index == 3 ? -a : a;
            b = index == 2 ? -b : b;
            c = index == 1 ? -c : c;
            d = index == 0 ? -d : d;
            return __fadd_rn(__fadd_rn(__fadd_rn(a, b), c), d);
        }

        template <bool swiglu, bool dequantize = false, bool normalize = false, bool scaled = false, bool warp_rotation = false, typename T = float, bool sigmoid_gate = false, bool rms_modulation = false>
        __global__ void quantize(
            const T* src,
            int8_t* dst,
            float* scales,
            int64_t k,
            int64_t rows,
            int64_t dst_stride,
            const float* up,
            int64_t src_stride,
            int64_t up_stride,
            const float* activation_scales = nullptr,
            const float* weight_scales     = nullptr,
            const float* modulation        = nullptr,
            float eps                      = 0.0f,
            float input_scale              = 1.0f,
            float input_bias               = 0.0f,
            const float* rms_weight        = nullptr,
            const float* rms_shift         = nullptr) {
            constexpr int group_size = 256;

            const int64_t row = blockIdx.x;
            const int tid     = threadIdx.x;
            if (row >= rows) {
                return;
            }

            extern __shared__ float values[];
            __shared__ float maxima[32];
            const T* src_row = src + row * src_stride;
            int8_t* dst_row  = dst + row * dst_stride;

            [[maybe_unused]] float rms_scale = 1.0f;
            if constexpr (rms_modulation) {
                __shared__ float sums[32];
                // Preserve the unfused RMSNorm's 1024-thread reduction order with fewer physical warps.
                for (int virtual_tid = tid; virtual_tid < 1024; virtual_tid += blockDim.x) {
                    float partial = 0.0f;
                    for (int64_t i = virtual_tid; i < k; i += 1024) {
                        const float value = src_row[i];
                        partial += value * value;
                    }
                    partial = warp_reduce_sum(partial);
                    if (tid % 32 == 0) {
                        sums[virtual_tid / 32] = partial;
                    }
                }
                __syncthreads();
                const float sum = warp_reduce_sum(sums[tid % 32]);
                rms_scale       = rsqrtf(sum / k + eps);
            }

            if constexpr (swiglu || warp_rotation) {
                const int lane = tid % 32;
                for (int64_t group = tid / 32; group < k / group_size; group += blockDim.x / 32) {
                    float v[8];
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int64_t i = group * group_size + j * 32 + lane;
                        if constexpr (!swiglu) {
                            float value = src_row[i];
                            if constexpr (rms_modulation) {
                                value = __fmul_rn(__fmul_rn(value, rms_scale), rms_weight[i]);
                                value = __fadd_rn(__fadd_rn(value, __fmul_rn(value, modulation[i])), rms_shift[i]);
                            }
                            if constexpr (scaled) {
                                value = __fmaf_rn(value, input_scale, input_bias);
                            }
                            v[j] = value * (1.0f / 16.0f);
                        } else {
                            float gate_value, up_value;
                            if constexpr (dequantize) {
                                gate_value = (float)((const int32_t*)src_row)[i] * activation_scales[row];
                                up_value   = (float)((const int32_t*)up)[row * up_stride + i] * activation_scales[row];
                                gate_value *= weight_scales[i];
                                up_value *= weight_scales[k + i];
                            } else {
                                gate_value = src_row[i];
                                up_value   = up[row * up_stride + i];
                            }
                            const float activated = sigmoid_gate ? 1.0f / (1.0f + expf(-gate_value)) : (gate_value / (1.0f + expf(-gate_value)));
                            v[j]                  = (activated * up_value) * (1.0f / 16.0f);
                        }
                    }
                    // One logical warp owns H256, keeping every butterfly in registers.
#pragma unroll
                    for (int stride = 1; stride <= 4; stride *= 4) {
                        const int base = lane & ~(3 * stride);
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            const float a = cuda_shuffle(v[j], base, 32);
                            const float b = cuda_shuffle(v[j], base + stride, 32);
                            const float c = cuda_shuffle(v[j], base + 2 * stride, 32);
                            const float d = cuda_shuffle(v[j], base + 3 * stride, 32);
                            v[j]          = convrot_h4_cuda(a, b, c, d, (lane / stride) % 4);
                        }
                    }
#pragma unroll
                    for (int j = 0; j < 8; j += 2) {
                        const float a = cuda_shuffle(v[j], lane % 16, 32);
                        const float b = cuda_shuffle(v[j], lane % 16 + 16, 32);
                        const float c = cuda_shuffle(v[j + 1], lane % 16, 32);
                        const float d = cuda_shuffle(v[j + 1], lane % 16 + 16, 32);
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
            } else {
                constexpr int group_threads   = group_size / 4;
                constexpr int groups_per_wave = 1024 / group_threads;
                if constexpr (normalize) {
                    __shared__ float2 sums[32];
                    float2 mean_var = make_float2(0.0f, 0.0f);
                    for (int64_t i = tid; i < k; i += 1024) {
                        const float value = src_row[i];
                        mean_var.x += value;
                        mean_var.y += value * value;
                    }
                    mean_var             = block_sum(mean_var, sums);
                    const float mean     = mean_var.x / k;
                    const float variance = mean_var.y / k - mean * mean;
                    const float inv_std  = rsqrtf(variance + eps);
                    for (int64_t i = tid; i < k; i += 1024) {
                        const float normalized = __fmul_rn(__fsub_rn(src_row[i], mean), inv_std);
                        values[i]              = __fmul_rn(__fmul_rn(normalized, modulation[i]), 1.0f / 16.0f);
                    }
                } else {
                    for (int64_t i = tid; i < k; i += blockDim.x) {
                        const float value = scaled ? __fmaf_rn(src_row[i], input_scale, input_bias) : src_row[i];
                        values[i]         = value * (1.0f / 16.0f);
                    }
                }
                __syncthreads();

                const int group_lane = tid % group_threads;
                const int wave_group = tid / group_threads;
                const int64_t groups = k / group_size;
                for (int64_t group_base = 0; group_base < groups; group_base += groups_per_wave) {
                    const int64_t group = group_base + wave_group;
#pragma unroll
                    for (int stride = 1; stride < group_size; stride *= 4) {
                        if (group < groups) {
                            float* group_values = values + group * group_size;
                            const int base      = (group_lane / stride) * 4 * stride + group_lane % stride;
                            const int i0        = base;
                            const int i1        = i0 + stride;
                            const int i2        = i1 + stride;
                            const int i3        = i2 + stride;
                            const float a       = group_values[i0];
                            const float b       = group_values[i1];
                            const float c       = group_values[i2];
                            const float d       = group_values[i3];
                            group_values[i0]    = a + b + c - d;
                            group_values[i1]    = a + b - c + d;
                            group_values[i2]    = a - b + c + d;
                            group_values[i3]    = -a + b + c + d;
                        }
                        __syncthreads();
                    }
                }
            }

            float local_max = 0.0f;
            for (int64_t i = tid; i < k; i += blockDim.x) {
                local_max = fmaxf(local_max, fabsf(values[i]));
            }
            local_max      = warp_reduce_max<32>(local_max);
            const int lane = tid & 31;
            const int warp = tid >> 5;
            if (lane == 0) {
                maxima[warp] = local_max;
            }
            __syncthreads();

            if (warp == 0) {
                local_max = lane < blockDim.x / 32 ? maxima[lane] : 0.0f;
                local_max = warp_reduce_max<32>(local_max);
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

    }
}
#endif
