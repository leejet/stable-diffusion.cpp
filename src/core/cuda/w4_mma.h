#ifndef __SD_CORE_CUDA_W4_MMA_H__
#define __SD_CORE_CUDA_W4_MMA_H__

#include <cuda_runtime.h>
#include <cutlass/arch/mma_sm80.h>
#include <cstdint>

namespace sd_w4_mma {
    static __device__ __forceinline__ float fp8_e4m3(uint8_t v) {
        const int exponent = (v >> 3) & 15, mantissa = v & 7;
        const float result = exponent == 15 && mantissa == 7 ? __int_as_float(0x7fc00000) : exponent == 0 ? ldexpf(float(mantissa), -9)
                                                                                                          : ldexpf(1.f + float(mantissa) / 8.f, exponent - 7);
        return v & 128 ? -result : result;
    }

    template <int M, int N, int K>
    static __global__ void __launch_bounds__(256) w4a8(
        const uint8_t* __restrict__ w,
        const int8_t* __restrict__ x,
        const float* __restrict__ xs,
        const float* __restrict__ codebook,
        const float* __restrict__ channel,
        const uint8_t* __restrict__ relative,
        const float* __restrict__ bias,
        float* __restrict__ dst,
        int rows,
        int n,
        int k) {
        const int row_base = blockIdx.y * M, col_base = blockIdx.x * N;
        const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
        __shared__ int cb[16];
        __shared__ float cbscale;
        if (tid == 0) {
            float amax = 0.f;
#pragma unroll
            for (int i = 0; i < 16; ++i)
                amax = fmaxf(amax, fabsf(codebook[i]));
            cbscale = amax / 127.f;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int value = cbscale == 0.f ? 0 : __float2int_rn(codebook[i] / cbscale);
                cb[i]           = max(-127, min(127, value));
            }
        }
        __syncthreads();
#if __CUDA_ARCH__ >= 800
        constexpr int stride = K / 4 + 4;
        __shared__ __align__(16) uint32_t a[M][stride], b[N][stride];
        __shared__ uint32_t lut[256];
        __shared__ float scales[K / 16][N];
        lut[tid] = (cb[tid & 15] & 255) | ((cb[tid >> 4] & 255) << 8);
        __syncthreads();
        constexpr int warp_cols = N / 32;
        constexpr int row_tiles = M / (16 * (8 / warp_cols));
        const int wr = (warp / warp_cols) * 16 * row_tiles, wc = (warp % warp_cols) * 32;
        float acc[row_tiles][4][4] = {};
        for (int base = 0; base < k; base += K) {
            for (int i = tid; i < M * K / 16; i += 256) {
                const int r = i / (K / 16), t = (i % (K / 16)) * 16;
                const int4 value                       = row_base + r < rows && base + t < k ? *reinterpret_cast<const int4*>(x + int64_t(row_base + r) * k + base + t) : make_int4(0, 0, 0, 0);
                *reinterpret_cast<int4*>(&a[r][t / 4]) = value;
            }
            for (int i = tid; i < N * K / 8; i += 256) {
                const int c = i / (K / 8), t = i % (K / 8);
                const uint32_t value = col_base + c < n && base + t * 8 < k ? *reinterpret_cast<const uint32_t*>(w + int64_t(col_base + c) * (k / 2) + base / 2 + t * 4) : 0;
                b[c][2 * t]          = lut[value & 255] | (lut[(value >> 8) & 255] << 16);
                b[c][2 * t + 1]      = lut[(value >> 16) & 255] | (lut[value >> 24] << 16);
            }
            for (int i = tid; i < N * K / 16; i += 256) {
                const int c = i % N, g = i / N;
                scales[g][c] = col_base + c < n && base + g * 16 < k ? cbscale * (channel[col_base + c] * fp8_e4m3(relative[int64_t(col_base + c) * (k / 16) + base / 16 + g])) : 0.f;
            }
            __syncthreads();
#pragma unroll
            for (int g = 0; g < K / 16; ++g) {
#pragma unroll
                for (int r = 0; r < row_tiles; ++r) {
                    const int ar = wr + r * 16 + lane / 4, ac = g * 4 + lane % 4;
                    const uint32_t a0 = a[ar][ac], a1 = a[ar + 8][ac];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const uint32_t b0 = b[wc + j * 8 + lane / 4][ac];
                        using Mma         = cutlass::arch::Mma<cutlass::gemm::GemmShape<16, 8, 16>, 32,
                                                               int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
                                                               int, cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAddSaturate>;
                        Mma::FragmentA af;
                        Mma::FragmentB bf;
                        Mma::FragmentC d;
                        reinterpret_cast<uint32_t*>(&af)[0] = a0;
                        reinterpret_cast<uint32_t*>(&af)[1] = a1;
                        reinterpret_cast<uint32_t*>(&bf)[0] = b0;
                        d.clear();
                        // Each K=16 MMA has its own scale; combining groups before
                        // dequantization would change the W4A8 representation.
                        Mma{}(d, af, bf, d);
                        const float s0 = scales[g][wc + j * 8 + (lane % 4) * 2], s1 = scales[g][wc + j * 8 + (lane % 4) * 2 + 1];
#pragma unroll
                        for (int v = 0; v < 4; ++v)
                            acc[r][j][v] = fmaf(float(d[v]), v % 2 ? s1 : s0, acc[r][j][v]);
                    }
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int r = 0; r < row_tiles; ++r) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    const int row = row_base + wr + r * 16 + lane / 4 + (v / 2) * 8;
                    const int col = col_base + wc + j * 8 + (lane % 4) * 2 + v % 2;
                    if (row < rows && col < n)
                        dst[int64_t(row) * n + col] = __fmul_rn(acc[r][j][v], xs[row]) + (bias ? bias[col] : 0.f);
                }
            }
        }
#else
        for (int i = tid; i < M * N; i += 256) {
            const int row = row_base + i / N, col = col_base + i % N;
            if (row < rows && col < n) {
                float sum = 0.f;
                for (int g = 0; g < k / 16; ++g) {
                    int d = 0;
                    for (int j = 0; j < 16; ++j) {
                        const int t = g * 16 + j;
                        d += cb[(w[int64_t(col) * (k / 2) + t / 2] >> (4 * (t % 2))) & 15] * int(x[int64_t(row) * k + t]);
                    }
                    sum = fmaf(float(d), cbscale * (channel[col] * fp8_e4m3(relative[int64_t(col) * (k / 16) + g])), sum);
                }
                dst[int64_t(row) * n + col] = __fmul_rn(sum, xs[row]) + (bias ? bias[col] : 0.f);
            }
        }
#endif
    }

}
#endif
