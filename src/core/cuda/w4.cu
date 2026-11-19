#include <climits>
#include "core/ggml_extensions.h"
#include "runtime.h"
#ifdef SD_CUDA_W4_CUTLASS
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include "i8_gemm.h"
#include "w4_mma.h"
#endif

namespace SD_CUDA_NAMESPACE {
    namespace sd_w4_cuda {
        static __device__ __forceinline__ int shuffle_sum(int value, int lane, int width) {
#if defined(SD_CUDA_VENDOR_HIP)
            // ROCm LLVM 24 can crash lowering __shfl's wave_shuffle on RDNA2.
            // These 8/16-lane groups stay within one ds_bpermute half-wave.
            const int index = (lane & (width - 1)) + (__lane_id() & ~(width - 1));
            return __builtin_amdgcn_ds_bpermute(index * 4, value);
#else
            return cuda_shuffle(value, lane, width);
#endif
        }

        // Signed byte dots preserve the INT4 grid after the XOR-8 bias correction.
        static __device__ __forceinline__ int dot4(int a, int b, int c) {
#if defined(SD_CUDA_VENDOR_HIP) && (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1200__) || defined(__gfx1201__))
            return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#elif defined(SD_CUDA_VENDOR_HIP) && (defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__) || defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx942__))
            return __builtin_amdgcn_sdot4(a, b, c, false);
#elif defined(SD_CUDA_VENDOR_MUSA) || (!defined(SD_CUDA_VENDOR_HIP) && __CUDA_ARCH__ >= 610)
            return __dp4a(a, b, c);
#else
            const auto* av = reinterpret_cast<const signed char*>(&a);
            const auto* bv = reinterpret_cast<const signed char*>(&b);
            for (int i = 0; i < 4; ++i)
                c += int(av[i]) * int(bv[i]);
            return c;
#endif
        }
        static __device__ __forceinline__ float fp8_e4m3(uint8_t v) {
            const int exponent = (v >> 3) & 15, mantissa = v & 7;
            float result = exponent == 15 && mantissa == 7 ? __int_as_float(0x7fc00000) : exponent == 0 ? ldexpf(float(mantissa), -9)
                                                                                                        : ldexpf(1.f + float(mantissa) / 8.f, exponent - 7);
            return v & 128 ? -result : result;
        }
        // A CTA shares decoded weights across 32 activation rows. Each double-buffered
        // stage covers 64 K values; W4A8 restores the codebook scale per 16-value group.

        static __device__ __forceinline__ uint32_t perm_bytes(uint32_t lo, uint32_t hi, uint32_t sel) {
#if defined(SD_CUDA_VENDOR_HIP)
            return __builtin_amdgcn_perm(hi, lo, sel);
#elif !defined(SD_CUDA_VENDOR_MUSA)
            return __byte_perm(lo, hi, sel);
#else
            uint32_t r = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t b    = (sel >> (4 * i)) & 0xF;
                const uint32_t byte = (b < 4 ? lo : hi) >> (8 * (b & 3)) & 0xFF;
                r |= byte << (8 * i);
            }
            return r;
#endif
        }

#if defined(SD_CUDA_VENDOR_HIP)
#define PERM_SEL_EVEN 0x06040200u
#define PERM_SEL_ODD 0x07050301u
#else
#define PERM_SEL_EVEN 0x6420u
#define PERM_SEL_ODD 0x7531u
#endif

        static __device__ __forceinline__ void interleave_act_pair(
            const int lo,
            const int hi,
            int* __restrict__ dst) {
            dst[0] = (int)perm_bytes((uint32_t)lo, (uint32_t)hi, PERM_SEL_EVEN);
            dst[1] = (int)perm_bytes((uint32_t)lo, (uint32_t)hi, PERM_SEL_ODD);
        }

#define MUL_MAT_W4_BLOCK 16
#define MUL_MAT_W4_ROWS (2 * MUL_MAT_W4_BLOCK)
#define MUL_MAT_W4_GPS 4
#define MUL_MAT_W4_SLOTS (4 * MUL_MAT_W4_GPS)
#define MUL_MAT_W4_OUT (MUL_MAT_W4_BLOCK * 4)

        static __global__ void __launch_bounds__(MUL_MAT_W4_BLOCK* MUL_MAT_W4_BLOCK) mul_mat_w4a4_convrot_dp4a(
            const char* __restrict__ w,
            const char* __restrict__ x,
            int* __restrict__ y,
            const uint32_t n,
            const uint32_t rows,
            const uint32_t packed_k) {
            const uint32_t o_base = blockIdx.x * MUL_MAT_W4_OUT;
            const uint32_t r0     = blockIdx.y * MUL_MAT_W4_ROWS + threadIdx.y;
            const uint32_t r1     = r0 + MUL_MAT_W4_BLOCK;
            const uint32_t lx     = threadIdx.x;
            const uint32_t ly     = threadIdx.y;
            const uint32_t idx    = ly * MUL_MAT_W4_BLOCK + lx;
            const uint32_t dc     = idx & (MUL_MAT_W4_OUT - 1);
            const uint32_t dw     = idx >> 6;

            __shared__ int w_dec[2][MUL_MAT_W4_SLOTS * MUL_MAT_W4_OUT];
            __shared__ int x_tile[2][MUL_MAT_W4_ROWS * MUL_MAT_W4_SLOTS];

            const uint32_t w_words = packed_k / 4;
            const uint32_t x_words = (packed_k * 2) / 4;
            const uint32_t groups  = packed_k / 8;
            const int* wi          = (const int*)w;
            const int* xi          = (const int*)x;

            const uint32_t fl = lx & 7;
            const uint32_t fr = (lx >> 3) ? r1 : r0;
            const uint32_t tr = (lx >> 3) * MUL_MAT_W4_BLOCK + ly;
            int my_sum        = 0;

            {
                const uint32_t ocol = o_base + dc;
                int2 wv             = make_int2(0, 0);
                if (ocol < n && dw * 2 < w_words) {
                    wv = ((const int2*)wi)[(size_t)ocol * (w_words / 2) + dw];
                }
                w_dec[0][((dw * 2 + 0) * 2 + 0) * MUL_MAT_W4_OUT + dc] = (wv.x & 0x0F0F0F0F) ^ 0x08080808;
                w_dec[0][((dw * 2 + 0) * 2 + 1) * MUL_MAT_W4_OUT + dc] = ((wv.x >> 4) & 0x0F0F0F0F) ^ 0x08080808;
                w_dec[0][((dw * 2 + 1) * 2 + 0) * MUL_MAT_W4_OUT + dc] = (wv.y & 0x0F0F0F0F) ^ 0x08080808;
                w_dec[0][((dw * 2 + 1) * 2 + 1) * MUL_MAT_W4_OUT + dc] = ((wv.y >> 4) & 0x0F0F0F0F) ^ 0x08080808;

                const uint32_t gi = fl >> 1;
                const uint32_t xb = gi * 4 + (fl & 1) * 2;
                const int a       = (fr < rows && xb < x_words) ? (int)xi[(size_t)fr * x_words + xb] : 0;
                const int b       = (fr < rows && xb + 1 < x_words) ? (int)xi[(size_t)fr * x_words + xb + 1] : 0;
                interleave_act_pair(a, b, &x_tile[0][tr * MUL_MAT_W4_SLOTS + (fl >> 1) * 4 + (fl & 1) * 2]);
                my_sum += dot4(a, 0x01010101, 0) + dot4(b, 0x01010101, 0);
            }
            __syncthreads();

            int acc0a = 0, acc0b = 0;
            int acc1a = 0, acc1b = 0;
            int acc2a = 0, acc2b = 0;
            int acc3a = 0, acc3b = 0;
            int acc4a = 0, acc4b = 0;
            int acc5a = 0, acc5b = 0;
            int acc6a = 0, acc6b = 0;
            int acc7a = 0, acc7b = 0;

            for (uint32_t gs = 0; gs < groups; gs += MUL_MAT_W4_GPS) {
                const uint32_t buf = (gs >> 2) & 1;

                if (gs + MUL_MAT_W4_GPS < groups) {
                    const uint32_t nbuf = buf ^ 1;
                    const uint32_t ngs  = gs + MUL_MAT_W4_GPS;
                    {
                        const uint32_t ocol = o_base + dc;
                        const uint32_t widx = ngs * 2 + dw * 2;
                        int2 wv             = make_int2(0, 0);
                        if (ocol < n && widx < w_words) {
                            wv = ((const int2*)wi)[(size_t)ocol * (w_words / 2) + ngs + dw];
                        }
                        w_dec[nbuf][((dw * 2 + 0) * 2 + 0) * MUL_MAT_W4_OUT + dc] = (wv.x & 0x0F0F0F0F) ^ 0x08080808;
                        w_dec[nbuf][((dw * 2 + 0) * 2 + 1) * MUL_MAT_W4_OUT + dc] = ((wv.x >> 4) & 0x0F0F0F0F) ^ 0x08080808;
                        w_dec[nbuf][((dw * 2 + 1) * 2 + 0) * MUL_MAT_W4_OUT + dc] = (wv.y & 0x0F0F0F0F) ^ 0x08080808;
                        w_dec[nbuf][((dw * 2 + 1) * 2 + 1) * MUL_MAT_W4_OUT + dc] = ((wv.y >> 4) & 0x0F0F0F0F) ^ 0x08080808;
                        const uint32_t gi                                         = ngs + (fl >> 1);
                        const uint32_t xb                                         = gi * 4 + (fl & 1) * 2;
                        const int a                                               = (fr < rows && xb < x_words) ? (int)xi[(size_t)fr * x_words + xb] : 0;
                        const int b                                               = (fr < rows && xb + 1 < x_words) ? (int)xi[(size_t)fr * x_words + xb + 1] : 0;
                        interleave_act_pair(a, b, &x_tile[nbuf][tr * MUL_MAT_W4_SLOTS + (fl >> 1) * 4 + (fl & 1) * 2]);
                        my_sum += dot4(a, 0x01010101, 0) + dot4(b, 0x01010101, 0);
                    }
                }
#pragma unroll
                for (uint32_t gl = 0; gl < MUL_MAT_W4_GPS; ++gl) {
                    const int u0[4] = {
                        x_tile[buf][ly * MUL_MAT_W4_SLOTS + gl * 4 + 0],
                        x_tile[buf][ly * MUL_MAT_W4_SLOTS + gl * 4 + 1],
                        x_tile[buf][ly * MUL_MAT_W4_SLOTS + gl * 4 + 2],
                        x_tile[buf][ly * MUL_MAT_W4_SLOTS + gl * 4 + 3],
                    };
                    const int u1[4] = {
                        x_tile[buf][(MUL_MAT_W4_BLOCK + ly) * MUL_MAT_W4_SLOTS + gl * 4 + 0],
                        x_tile[buf][(MUL_MAT_W4_BLOCK + ly) * MUL_MAT_W4_SLOTS + gl * 4 + 1],
                        x_tile[buf][(MUL_MAT_W4_BLOCK + ly) * MUL_MAT_W4_SLOTS + gl * 4 + 2],
                        x_tile[buf][(MUL_MAT_W4_BLOCK + ly) * MUL_MAT_W4_SLOTS + gl * 4 + 3],
                    };
#pragma unroll
                    for (uint32_t jp = 0; jp < 2; ++jp) {
                        const int ue0     = jp == 0 ? u0[0] : u0[2];
                        const int uo0     = jp == 0 ? u0[1] : u0[3];
                        const int ue1     = jp == 0 ? u1[0] : u1[2];
                        const int uo1     = jp == 0 ? u1[1] : u1[3];
                        const uint32_t we = (gl * 4 + jp * 2) * MUL_MAT_W4_OUT;
                        const uint32_t wo = we + MUL_MAT_W4_OUT;

                        acc0a = dot4(w_dec[buf][we + lx], ue0, acc0a);
                        acc1a = dot4(w_dec[buf][we + 16 + lx], ue0, acc1a);
                        acc2a = dot4(w_dec[buf][we + 32 + lx], ue0, acc2a);
                        acc3a = dot4(w_dec[buf][we + 48 + lx], ue0, acc3a);
                        acc0b = dot4(w_dec[buf][wo + lx], uo0, acc0b);
                        acc1b = dot4(w_dec[buf][wo + 16 + lx], uo0, acc1b);
                        acc2b = dot4(w_dec[buf][wo + 32 + lx], uo0, acc2b);
                        acc3b = dot4(w_dec[buf][wo + 48 + lx], uo0, acc3b);
                        acc4a = dot4(w_dec[buf][we + lx], ue1, acc4a);
                        acc5a = dot4(w_dec[buf][we + 16 + lx], ue1, acc5a);
                        acc6a = dot4(w_dec[buf][we + 32 + lx], ue1, acc6a);
                        acc7a = dot4(w_dec[buf][we + 48 + lx], ue1, acc7a);
                        acc4b = dot4(w_dec[buf][wo + lx], uo1, acc4b);
                        acc5b = dot4(w_dec[buf][wo + 16 + lx], uo1, acc5b);
                        acc6b = dot4(w_dec[buf][wo + 32 + lx], uo1, acc6b);
                        acc7b = dot4(w_dec[buf][wo + 48 + lx], uo1, acc7b);
                    }
                }
                __syncthreads();
            }

            int v = my_sum;
            v += shuffle_sum(v, (int(threadIdx.x) % 8 + 4) % 8, 8);
            v += shuffle_sum(v, (int(threadIdx.x) % 8 + 2) % 8, 8);
            v += shuffle_sum(v, (int(threadIdx.x) % 8 + 1) % 8, 8);
            const int sum0 = shuffle_sum(v, 0, 16);
            const int sum1 = shuffle_sum(v, 8, 16);

            const int accv[2][4] = {
                {acc0a + acc0b, acc1a + acc1b, acc2a + acc2b, acc3a + acc3b},
                {acc4a + acc4b, acc5a + acc5b, acc6a + acc6b, acc7a + acc7b},
            };
            const int sumv[2] = {sum0, sum1};
#pragma unroll
            for (uint32_t t = 0; t < 2; ++t) {
                const uint32_t rt = t == 0 ? r0 : r1;
                if (rt < rows) {
#pragma unroll
                    for (uint32_t j = 0; j < 4; ++j) {
                        const uint32_t oj = o_base + j * 16 + lx;
                        if (oj < n) {
                            y[(size_t)rt * n + oj] = accv[t][j] - 8 * sumv[t];
                        }
                    }
                }
            }
        }

#define MUL_MAT_W4A8_BLOCK 16
#define MUL_MAT_W4A8_ROWS (2 * MUL_MAT_W4A8_BLOCK)
#define MUL_MAT_W4A8_GPS 4
#define MUL_MAT_W4A8_SLOTS (4 * MUL_MAT_W4A8_GPS)
#define MUL_MAT_W4A8_OUT (MUL_MAT_W4A8_BLOCK * 4)

#if defined(SD_CUDA_VENDOR_HIP) && defined(__HIP_DEVICE_COMPILE__)
// __has_builtin checks the active device target's features, not just whether
// the compiler recognizes the name. Prefer the non-replicated fragment layout.
#if __has_builtin(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w64_gfx12)
#define SD_W4_WMMA_GFX12
#elif __has_builtin(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w64)
#define SD_W4_WMMA_GFX11
#endif
#endif

// Keep each intrinsic paired with its fragment layout, including on newer GPUs.
#if defined(SD_W4_WMMA_GFX11)
        typedef int v4si32_w4a8 __attribute__((ext_vector_type(4)));

        static __device__ __forceinline__ void mul_mat_w4a8_convrot_wmma(
            const char* __restrict__ w,
            const char* __restrict__ x,
            const float* __restrict__ codebook,
            const float* __restrict__ s_channel,
            const uint8_t* __restrict__ s_rel,
            const float* __restrict__ act_scales,
            const float* __restrict__ bias,
            float* __restrict__ y,
            const uint32_t n,
            const uint32_t rows,
            const uint32_t k,
            const uint32_t r_base) {
            const uint32_t o_base = blockIdx.x * MUL_MAT_W4A8_OUT;
            const uint32_t idx    = threadIdx.y * MUL_MAT_W4A8_BLOCK + threadIdx.x;
            const uint32_t wv     = idx >> 6;
            const uint32_t lane   = idx & 63;
            const uint32_t j      = lane & 15;
            const uint32_t q      = lane >> 4;
            const uint32_t ocol   = o_base + wv * 16 + j;
            const uint32_t ra     = r_base + j;

            __shared__ __align__(4) int8_t cb8[16];
            __shared__ float s_cb;
            __shared__ int lut[256];
            __shared__ float s_lut[256];

            const uint32_t w_words = k / 8;
            const uint32_t x_words = k / 4;
            const uint32_t groups  = k / 16;
            const int* wi          = (const int*)w;
            const int* xi          = (const int*)x;

            if (idx == 0) {
                float amax = 0.0f;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    amax = fmaxf(amax, fabsf(codebook[i]));
                }
                s_cb = amax / 127.0f;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int v = s_cb == 0.f ? 0 : lrintf(codebook[i] / s_cb);
                    cb8[i]      = (int8_t)(v < -127 ? -127 : (v > 127 ? 127 : v));
                }
#pragma unroll
                for (int b = 0; b < 256; ++b) {
                    lut[b] = (cb8[b & 0xF] & 0xFF) | ((cb8[b >> 4] & 0xFF) << 8);
                }
#pragma unroll
                for (int b = 0; b < 256; ++b) {
                    s_lut[b] = fp8_e4m3(uint8_t(b));
                }
            }
            __syncthreads();

            const bool o_valid = ocol < n;
            const float s_chan = o_valid ? s_channel[ocol] : 0.0f;
            const float scb    = s_cb;

            float acc[4]           = {0.0f, 0.0f, 0.0f, 0.0f};
            const v4si32_w4a8 zero = {};

            int w0c = 0, w1c = 0, xc0 = 0, xc1 = 0, xc2 = 0, xc3 = 0, sc = 0;
            int w0n = 0, w1n = 0, xn0 = 0, xn1 = 0, xn2 = 0, xn3 = 0, sn = 0;
            {
                const int* xr = xi + (size_t)ra * x_words;
                xc0           = ra < rows ? xr[0] : 0;
                xc1           = ra < rows ? xr[1] : 0;
                xc2           = ra < rows ? xr[2] : 0;
                xc3           = ra < rows ? xr[3] : 0;
                if (o_valid) {
                    const int* wp = wi + (size_t)ocol * w_words;
                    w0c           = wp[0];
                    w1c           = wp[1];
                    sc            = s_rel[(size_t)ocol * groups];
                }
            }
            if (groups > 1) {
                const int* xr = xi + (size_t)ra * x_words + 4;
                xn0           = ra < rows ? xr[0] : 0;
                xn1           = ra < rows ? xr[1] : 0;
                xn2           = ra < rows ? xr[2] : 0;
                xn3           = ra < rows ? xr[3] : 0;
                if (o_valid) {
                    const int* wp = wi + (size_t)ocol * w_words + 2;
                    w0n           = wp[0];
                    w1n           = wp[1];
                    sn            = s_rel[(size_t)ocol * groups + 1];
                }
            }

            for (uint32_t g = 0; g < groups; ++g) {
                v4si32_w4a8 bfrag = {};
                float s           = 0.0f;
                if (o_valid) {
                    bfrag = {
                        (lut[w0c & 0xFF] & 0xFFFF) | ((lut[(w0c >> 8) & 0xFF] & 0xFFFF) << 16),
                        (lut[(w0c >> 16) & 0xFF] & 0xFFFF) | ((lut[(w0c >> 24) & 0xFF] & 0xFFFF) << 16),
                        (lut[w1c & 0xFF] & 0xFFFF) | ((lut[(w1c >> 8) & 0xFF] & 0xFFFF) << 16),
                        (lut[(w1c >> 16) & 0xFF] & 0xFFFF) | ((lut[(w1c >> 24) & 0xFF] & 0xFFFF) << 16),
                    };

                    s = scb * (s_chan * s_lut[sc & 0xFF]);
                }
                const v4si32_w4a8 afrag = {xc0, xc1, xc2, xc3};
                const v4si32_w4a8 dfrag =
                    __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64(true, afrag, true, bfrag, zero, false);

                w0c = w0n;
                w1c = w1n;
                sc  = sn;
                xc0 = xn0;
                xc1 = xn1;
                xc2 = xn2;
                xc3 = xn3;
                if (g + 2 < groups) {
                    const int* xr = xi + (size_t)ra * x_words + (g + 2) * 4;
                    xn0           = ra < rows ? xr[0] : 0;
                    xn1           = ra < rows ? xr[1] : 0;
                    xn2           = ra < rows ? xr[2] : 0;
                    xn3           = ra < rows ? xr[3] : 0;
                    if (o_valid) {
                        const int* wp = wi + (size_t)ocol * w_words + (g + 2) * 2;
                        w0n           = wp[0];
                        w1n           = wp[1];
                        sn            = s_rel[(size_t)ocol * groups + g + 2];
                    }
                }

                acc[0] = fmaf((float)dfrag.x, s, acc[0]);
                acc[1] = fmaf((float)dfrag.y, s, acc[1]);
                acc[2] = fmaf((float)dfrag.z, s, acc[2]);
                acc[3] = fmaf((float)dfrag.w, s, acc[3]);
            }

            if (o_valid) {
#pragma unroll
                for (uint32_t d = 0; d < 4; ++d) {
                    const uint32_t r = r_base + 4 * d + q;
                    if (r < rows) {
                        float value = acc[d] * act_scales[r];
                        if (bias != nullptr) {
                            value += bias[ocol];
                        }
                        y[(size_t)r * n + ocol] = value;
                    }
                }
            }
        }
#endif

#if defined(SD_W4_WMMA_GFX12)
        typedef int v4si32_w4a8_g12 __attribute__((ext_vector_type(4)));

        static __device__ __forceinline__ void mul_mat_w4a8_convrot_wmma_gfx12(
            const char* __restrict__ w,
            const char* __restrict__ x,
            const float* __restrict__ codebook,
            const float* __restrict__ s_channel,
            const uint8_t* __restrict__ s_rel,
            const float* __restrict__ act_scales,
            const float* __restrict__ bias,
            float* __restrict__ y,
            const uint32_t n,
            const uint32_t rows,
            const uint32_t k,
            const uint32_t r_base) {
            const uint32_t o_base = blockIdx.x * MUL_MAT_W4A8_OUT;
            const uint32_t idx    = threadIdx.y * MUL_MAT_W4A8_BLOCK + threadIdx.x;
            const uint32_t wv     = idx >> 6;
            const uint32_t lane   = idx & 63;
            const uint32_t j      = lane & 15;
            const uint32_t h      = lane >> 4;
            const uint32_t kh     = ((h & 1) << 1) | (h >> 1);
            const uint32_t ocol   = o_base + wv * 16 + j;
            const uint32_t ra     = r_base + j;

            __shared__ __align__(4) int8_t cb8[16];
            __shared__ float s_cb;
            __shared__ int lut[256];
            __shared__ float s_lut[256];

            const uint32_t w_words = k / 8;
            const uint32_t x_words = k / 4;
            const uint32_t groups  = k / 16;
            const int* wi          = (const int*)w;
            const int* xi          = (const int*)x;

            if (idx == 0) {
                float amax = 0.0f;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    amax = fmaxf(amax, fabsf(codebook[i]));
                }
                s_cb = amax / 127.0f;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int v = s_cb == 0.f ? 0 : lrintf(codebook[i] / s_cb);
                    cb8[i]      = (int8_t)(v < -127 ? -127 : (v > 127 ? 127 : v));
                }
#pragma unroll
                for (int b = 0; b < 256; ++b) {
                    lut[b] = (cb8[b & 0xF] & 0xFF) | ((cb8[b >> 4] & 0xFF) << 8);
                }
#pragma unroll
                for (int b = 0; b < 256; ++b) {
                    s_lut[b] = fp8_e4m3(uint8_t(b));
                }
            }
            __syncthreads();

            const bool o_valid = ocol < n;
            const float s_chan = o_valid ? s_channel[ocol] : 0.0f;
            const float scb    = s_cb;

            float acc[4]               = {0.0f, 0.0f, 0.0f, 0.0f};
            const v4si32_w4a8_g12 zero = {};

            int w0c = 0, w1c = 0, xc = 0, sc = 0;
            int w0n = 0, w1n = 0, xn = 0, sn = 0;
            {
                const int* xr = xi + (size_t)ra * x_words;
                xc            = ra < rows ? xr[kh] : 0;
                if (o_valid) {
                    const int* wp = wi + (size_t)ocol * w_words;
                    w0c           = wp[0];
                    w1c           = wp[1];
                    sc            = s_rel[(size_t)ocol * groups];
                }
            }
            if (groups > 1) {
                const int* xr = xi + (size_t)ra * x_words;
                xn            = ra < rows ? xr[4 + kh] : 0;
                if (o_valid) {
                    const int* wp = wi + (size_t)ocol * w_words + 2;
                    w0n           = wp[0];
                    w1n           = wp[1];
                    sn            = s_rel[(size_t)ocol * groups + 1];
                }
            }

            for (uint32_t g = 0; g < groups; ++g) {
                int bfrag = 0;
                float s   = 0.0f;
                if (o_valid) {
                    const int bwd = (kh & 2) ? w1c : w0c;
                    const int bsh = (kh & 1) * 16;
                    bfrag         = (lut[(bwd >> bsh) & 0xFF] & 0xFFFF) |
                                    ((lut[(bwd >> (bsh + 8)) & 0xFF] & 0xFFFF) << 16);

                    s = scb * (s_chan * s_lut[sc & 0xFF]);
                }
                const v4si32_w4a8_g12 dfrag =
                    __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64_gfx12(true, xc, true, bfrag, zero, false);

                w0c = w0n;
                w1c = w1n;
                sc  = sn;
                xc  = xn;
                if (g + 2 < groups) {
                    const int* xr = xi + (size_t)ra * x_words;
                    xn            = ra < rows ? xr[(g + 2) * 4 + kh] : 0;
                    if (o_valid) {
                        const int* wp = wi + (size_t)ocol * w_words + (g + 2) * 2;
                        w0n           = wp[0];
                        w1n           = wp[1];
                        sn            = s_rel[(size_t)ocol * groups + g + 2];
                    }
                }

                acc[0] = fmaf((float)dfrag.x, s, acc[0]);
                acc[1] = fmaf((float)dfrag.y, s, acc[1]);
                acc[2] = fmaf((float)dfrag.z, s, acc[2]);
                acc[3] = fmaf((float)dfrag.w, s, acc[3]);
            }

            if (o_valid) {
#pragma unroll
                for (uint32_t d = 0; d < 4; ++d) {
                    const uint32_t r = r_base + 4 * kh + d;
                    if (r < rows) {
                        float value = acc[d] * act_scales[r];
                        if (bias != nullptr) {
                            value += bias[ocol];
                        }
                        y[(size_t)r * n + ocol] = value;
                    }
                }
            }
        }
#endif

        static __global__ void __launch_bounds__(MUL_MAT_W4A8_BLOCK* MUL_MAT_W4A8_BLOCK) mul_mat_w4a8_convrot(
            const char* __restrict__ w,
            const char* __restrict__ x,
            const float* __restrict__ codebook,
            const float* __restrict__ s_channel,
            const uint8_t* __restrict__ s_rel,
            const float* __restrict__ act_scales,
            const float* __restrict__ bias,
            float* __restrict__ y,
            const uint32_t n,
            const uint32_t rows,
            const uint32_t k) {
#if defined(SD_W4_WMMA_GFX11) || defined(SD_W4_WMMA_GFX12)
            // Retain the 32-row launch grid shared with DP4A. Each WMMA wave
            // computes a 16-row tile; the barrier protects the reused LUTs.
            for (uint32_t tile = 0; tile < 2; ++tile) {
#if defined(SD_W4_WMMA_GFX12)
                mul_mat_w4a8_convrot_wmma_gfx12(w, x, codebook, s_channel, s_rel, act_scales, bias, y, n, rows, k,
                                                blockIdx.y * MUL_MAT_W4A8_ROWS + tile * MUL_MAT_W4A8_BLOCK);
#else
                mul_mat_w4a8_convrot_wmma(w, x, codebook, s_channel, s_rel, act_scales, bias, y, n, rows, k,
                                          blockIdx.y * MUL_MAT_W4A8_ROWS + tile * MUL_MAT_W4A8_BLOCK);
#endif
                __syncthreads();
            }
#else
            const uint32_t o_base = blockIdx.x * MUL_MAT_W4A8_OUT;
            const uint32_t r0     = blockIdx.y * MUL_MAT_W4A8_ROWS + threadIdx.y;
            const uint32_t r1     = r0 + MUL_MAT_W4A8_BLOCK;
            const uint32_t lx     = threadIdx.x;
            const uint32_t ly     = threadIdx.y;
            const uint32_t idx    = ly * MUL_MAT_W4A8_BLOCK + lx;
            const uint32_t dc     = idx & (MUL_MAT_W4A8_OUT - 1);
            const uint32_t dw     = idx >> 6;

            __shared__ int w_dec[2][MUL_MAT_W4A8_SLOTS * MUL_MAT_W4A8_OUT];
            __shared__ int x_tile[2][MUL_MAT_W4A8_ROWS * MUL_MAT_W4A8_SLOTS];
            __shared__ float s_dec[2][MUL_MAT_W4A8_GPS * MUL_MAT_W4A8_OUT];
            __shared__ __align__(4) int8_t cb8[16];
            __shared__ float s_cb;
            __shared__ int lut[256];

            const uint32_t w_words = k / 8;
            const uint32_t x_words = k / 4;
            const uint32_t groups  = k / 16;
            const int* wi          = (const int*)w;
            const int* xi          = (const int*)x;

            if (ly == 0 && lx == 0) {
                float amax = 0.0f;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    amax = fmaxf(amax, fabsf(codebook[i]));
                }

                s_cb = amax / 127.0f;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int v = s_cb == 0.f ? 0 : __float2int_rn(codebook[i] / s_cb);
                    cb8[i]      = (int8_t)(v < -127 ? -127 : (v > 127 ? 127 : v));
                }
            }
            __syncthreads();

            lut[idx] = (cb8[idx & 0xF] & 0xFF) | ((cb8[idx >> 4] & 0xFF) << 8);
            __syncthreads();

#pragma unroll
            for (uint32_t j = 0; j < 2; ++j) {
                const uint32_t pw                              = dw * 2 + j;
                const uint32_t ocol                            = o_base + dc;
                const int wv                                   = (ocol < n && pw < w_words) ? wi[(size_t)ocol * w_words + pw] : 0;
                w_dec[0][(pw * 2 + 0) * MUL_MAT_W4A8_OUT + dc] = (lut[wv & 0xFF] & 0xFFFF) | ((lut[(wv >> 8) & 0xFF] & 0xFFFF) << 16);
                w_dec[0][(pw * 2 + 1) * MUL_MAT_W4A8_OUT + dc] = (lut[(wv >> 16) & 0xFF] & 0xFFFF) | ((lut[(wv >> 24) & 0xFF] & 0xFFFF) << 16);
            }
            {
                const uint32_t ocol = o_base + dc;
                float s             = 0.0f;
                if (ocol < n && dw < groups) {
                    s = s_channel[ocol] * fp8_e4m3(s_rel[(size_t)ocol * groups + dw]);
                }
                s_dec[0][dw * MUL_MAT_W4A8_OUT + dc] = s_cb * s;
            }
            if (lx < 16) {
                const uint32_t aidx = lx;
                x_tile[0][ly * MUL_MAT_W4A8_SLOTS + lx] =
                    (r0 < rows && aidx < x_words) ? xi[(size_t)r0 * x_words + aidx] : 0;
                x_tile[0][(MUL_MAT_W4A8_BLOCK + ly) * MUL_MAT_W4A8_SLOTS + lx] =
                    (r1 < rows && aidx < x_words) ? xi[(size_t)r1 * x_words + aidx] : 0;
            }
            __syncthreads();

            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            float acc4 = 0.0f;
            float acc5 = 0.0f;
            float acc6 = 0.0f;
            float acc7 = 0.0f;

            for (uint32_t gs = 0; gs < groups; gs += MUL_MAT_W4A8_GPS) {
                const uint32_t buf = (gs >> 2) & 1;

                if (gs + MUL_MAT_W4A8_GPS < groups) {
                    const uint32_t nbuf = buf ^ 1;
                    const uint32_t ngs  = gs + MUL_MAT_W4A8_GPS;
#pragma unroll
                    for (uint32_t j = 0; j < 2; ++j) {
                        const uint32_t pw                                 = dw * 2 + j;
                        const uint32_t ocol                               = o_base + dc;
                        const uint32_t widx                               = ngs * 2 + pw;
                        const int wv                                      = (ocol < n && widx < w_words) ? wi[(size_t)ocol * w_words + widx] : 0;
                        w_dec[nbuf][(pw * 2 + 0) * MUL_MAT_W4A8_OUT + dc] = (lut[wv & 0xFF] & 0xFFFF) | ((lut[(wv >> 8) & 0xFF] & 0xFFFF) << 16);
                        w_dec[nbuf][(pw * 2 + 1) * MUL_MAT_W4A8_OUT + dc] = (lut[(wv >> 16) & 0xFF] & 0xFFFF) | ((lut[(wv >> 24) & 0xFF] & 0xFFFF) << 16);
                    }
                    {
                        const uint32_t ocol = o_base + dc;
                        float s             = 0.0f;
                        if (ocol < n && ngs + dw < groups) {
                            s = s_channel[ocol] * fp8_e4m3(s_rel[(size_t)ocol * groups + ngs + dw]);
                        }
                        s_dec[nbuf][dw * MUL_MAT_W4A8_OUT + dc] = s_cb * s;
                    }
                    if (lx < 16) {
                        const uint32_t aidx = ngs * 4 + lx;
                        x_tile[nbuf][ly * MUL_MAT_W4A8_SLOTS + lx] =
                            (r0 < rows && aidx < x_words) ? xi[(size_t)r0 * x_words + aidx] : 0;
                        x_tile[nbuf][(MUL_MAT_W4A8_BLOCK + ly) * MUL_MAT_W4A8_SLOTS + lx] =
                            (r1 < rows && aidx < x_words) ? xi[(size_t)r1 * x_words + aidx] : 0;
                    }
                }

#pragma unroll
                for (uint32_t gl = 0; gl < MUL_MAT_W4A8_GPS; ++gl) {
                    const int u0[4] = {
                        x_tile[buf][ly * MUL_MAT_W4A8_SLOTS + gl * 4 + 0],
                        x_tile[buf][ly * MUL_MAT_W4A8_SLOTS + gl * 4 + 1],
                        x_tile[buf][ly * MUL_MAT_W4A8_SLOTS + gl * 4 + 2],
                        x_tile[buf][ly * MUL_MAT_W4A8_SLOTS + gl * 4 + 3],
                    };
                    const int u1[4] = {
                        x_tile[buf][(MUL_MAT_W4A8_BLOCK + ly) * MUL_MAT_W4A8_SLOTS + gl * 4 + 0],
                        x_tile[buf][(MUL_MAT_W4A8_BLOCK + ly) * MUL_MAT_W4A8_SLOTS + gl * 4 + 1],
                        x_tile[buf][(MUL_MAT_W4A8_BLOCK + ly) * MUL_MAT_W4A8_SLOTS + gl * 4 + 2],
                        x_tile[buf][(MUL_MAT_W4A8_BLOCK + ly) * MUL_MAT_W4A8_SLOTS + gl * 4 + 3],
                    };
                    int accg0 = dot4(w_dec[buf][(gl * 4 + 0) * MUL_MAT_W4A8_OUT + lx], u0[0], 0);
                    accg0     = dot4(w_dec[buf][(gl * 4 + 1) * MUL_MAT_W4A8_OUT + lx], u0[1], accg0);
                    accg0     = dot4(w_dec[buf][(gl * 4 + 2) * MUL_MAT_W4A8_OUT + lx], u0[2], accg0);
                    accg0     = dot4(w_dec[buf][(gl * 4 + 3) * MUL_MAT_W4A8_OUT + lx], u0[3], accg0);
                    int accg1 = dot4(w_dec[buf][(gl * 4 + 0) * MUL_MAT_W4A8_OUT + 16 + lx], u0[0], 0);
                    accg1     = dot4(w_dec[buf][(gl * 4 + 1) * MUL_MAT_W4A8_OUT + 16 + lx], u0[1], accg1);
                    accg1     = dot4(w_dec[buf][(gl * 4 + 2) * MUL_MAT_W4A8_OUT + 16 + lx], u0[2], accg1);
                    accg1     = dot4(w_dec[buf][(gl * 4 + 3) * MUL_MAT_W4A8_OUT + 16 + lx], u0[3], accg1);
                    int accg2 = dot4(w_dec[buf][(gl * 4 + 0) * MUL_MAT_W4A8_OUT + 32 + lx], u0[0], 0);
                    accg2     = dot4(w_dec[buf][(gl * 4 + 1) * MUL_MAT_W4A8_OUT + 32 + lx], u0[1], accg2);
                    accg2     = dot4(w_dec[buf][(gl * 4 + 2) * MUL_MAT_W4A8_OUT + 32 + lx], u0[2], accg2);
                    accg2     = dot4(w_dec[buf][(gl * 4 + 3) * MUL_MAT_W4A8_OUT + 32 + lx], u0[3], accg2);
                    int accg3 = dot4(w_dec[buf][(gl * 4 + 0) * MUL_MAT_W4A8_OUT + 48 + lx], u0[0], 0);
                    accg3     = dot4(w_dec[buf][(gl * 4 + 1) * MUL_MAT_W4A8_OUT + 48 + lx], u0[1], accg3);
                    accg3     = dot4(w_dec[buf][(gl * 4 + 2) * MUL_MAT_W4A8_OUT + 48 + lx], u0[2], accg3);
                    accg3     = dot4(w_dec[buf][(gl * 4 + 3) * MUL_MAT_W4A8_OUT + 48 + lx], u0[3], accg3);
                    int accg4 = dot4(w_dec[buf][(gl * 4 + 0) * MUL_MAT_W4A8_OUT + lx], u1[0], 0);
                    accg4     = dot4(w_dec[buf][(gl * 4 + 1) * MUL_MAT_W4A8_OUT + lx], u1[1], accg4);
                    accg4     = dot4(w_dec[buf][(gl * 4 + 2) * MUL_MAT_W4A8_OUT + lx], u1[2], accg4);
                    accg4     = dot4(w_dec[buf][(gl * 4 + 3) * MUL_MAT_W4A8_OUT + lx], u1[3], accg4);
                    int accg5 = dot4(w_dec[buf][(gl * 4 + 0) * MUL_MAT_W4A8_OUT + 16 + lx], u1[0], 0);
                    accg5     = dot4(w_dec[buf][(gl * 4 + 1) * MUL_MAT_W4A8_OUT + 16 + lx], u1[1], accg5);
                    accg5     = dot4(w_dec[buf][(gl * 4 + 2) * MUL_MAT_W4A8_OUT + 16 + lx], u1[2], accg5);
                    accg5     = dot4(w_dec[buf][(gl * 4 + 3) * MUL_MAT_W4A8_OUT + 16 + lx], u1[3], accg5);
                    int accg6 = dot4(w_dec[buf][(gl * 4 + 0) * MUL_MAT_W4A8_OUT + 32 + lx], u1[0], 0);
                    accg6     = dot4(w_dec[buf][(gl * 4 + 1) * MUL_MAT_W4A8_OUT + 32 + lx], u1[1], accg6);
                    accg6     = dot4(w_dec[buf][(gl * 4 + 2) * MUL_MAT_W4A8_OUT + 32 + lx], u1[2], accg6);
                    accg6     = dot4(w_dec[buf][(gl * 4 + 3) * MUL_MAT_W4A8_OUT + 32 + lx], u1[3], accg6);
                    int accg7 = dot4(w_dec[buf][(gl * 4 + 0) * MUL_MAT_W4A8_OUT + 48 + lx], u1[0], 0);
                    accg7     = dot4(w_dec[buf][(gl * 4 + 1) * MUL_MAT_W4A8_OUT + 48 + lx], u1[1], accg7);
                    accg7     = dot4(w_dec[buf][(gl * 4 + 2) * MUL_MAT_W4A8_OUT + 48 + lx], u1[2], accg7);
                    accg7     = dot4(w_dec[buf][(gl * 4 + 3) * MUL_MAT_W4A8_OUT + 48 + lx], u1[3], accg7);
                    acc0      = fmaf((float)accg0, s_dec[buf][gl * MUL_MAT_W4A8_OUT + lx], acc0);
                    acc1      = fmaf((float)accg1, s_dec[buf][gl * MUL_MAT_W4A8_OUT + 16 + lx], acc1);
                    acc2      = fmaf((float)accg2, s_dec[buf][gl * MUL_MAT_W4A8_OUT + 32 + lx], acc2);
                    acc3      = fmaf((float)accg3, s_dec[buf][gl * MUL_MAT_W4A8_OUT + 48 + lx], acc3);
                    acc4      = fmaf((float)accg4, s_dec[buf][gl * MUL_MAT_W4A8_OUT + lx], acc4);
                    acc5      = fmaf((float)accg5, s_dec[buf][gl * MUL_MAT_W4A8_OUT + 16 + lx], acc5);
                    acc6      = fmaf((float)accg6, s_dec[buf][gl * MUL_MAT_W4A8_OUT + 32 + lx], acc6);
                    acc7      = fmaf((float)accg7, s_dec[buf][gl * MUL_MAT_W4A8_OUT + 48 + lx], acc7);
                }
                __syncthreads();
            }

            const float accv[8] = {acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7};
#pragma unroll
            for (uint32_t t = 0; t < 2; ++t) {
                const uint32_t rt = t == 0 ? r0 : r1;
                if (rt < rows) {
#pragma unroll
                    for (uint32_t j = 0; j < 4; ++j) {
                        const uint32_t oj = o_base + j * 16 + lx;
                        if (oj < n) {
                            float value = accv[t * 4 + j] * act_scales[rt];
                            if (bias != nullptr) {
                                value += bias[oj];
                            }
                            y[(size_t)rt * n + oj] = value;
                        }
                    }
                }
            }
#endif
        }

        static __global__ void dequantize(const int32_t* accum, const float* xs, const float* ws, const float* bias, float* dst, int64_t n, int64_t total) {
            const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < total) {
                float value = float(accum[i]) * xs[i / n] * ws[i % n];
                dst[i]      = value + (bias ? bias[i % n] : 0.f);
            }
        }
        static bool aligned(const ggml_tensor* t, size_t alignment) {
            if (!t)
                return true;
            size_t offset = 0;
            while (!t->data && t->view_src) {
                offset += t->view_offs;
                t = t->view_src;
            }
            return (reinterpret_cast<uintptr_t>(t->data) + offset) % alignment == 0;
        }
        static bool supports(ggml_backend_dev_t device, const ggml_tensor* node, void*) {
            SDW4Operation op;
            if (!sd_ggml_w4_describe(node, &op) || cuda_device_info(device).device < 0)
                return false;
            const int64_t k = op.input->ne[0], n = node->ne[0], rows = ggml_nrows(node);
            if (k % 64 || k > INT_MAX || n > INT_MAX - 63 || rows > 65535 * 32 ||
                ggml_nelements(node) > int64_t(INT_MAX) * 256 || !aligned(op.weight, 8) || !aligned(node, 4))
                return false;
            for (auto* t : node->src)
                if (!aligned(t, 4))
                    return false;
            return true;
        }
        struct Plan {
            bool mma           = false;
            size_t gemm_offset = 0;
#ifdef SD_CUDA_W4_CUTLASS
            SDI8CudaGemmPlan* gemm = nullptr;
            ~Plan() { sd_cuda_i8_gemm_free(gemm); }
#endif
        };
        static bool use_mma(ggml_backend_t backend, const SDW4Operation& op) {
#ifdef SD_CUDA_W4_CUTLASS
            if (!aligned(op.input, 16) || !aligned(op.weight, 16))
                return false;
            static const bool disabled = [] { const char* v = std::getenv("GGML_CUDA_W4_FORCE_DP4A"); return v && std::atoi(v); }();
            if (disabled)
                return false;
            static std::mutex mutex;
            static std::unordered_map<int, bool> devices;
            const auto device = cuda_device_info(ggml_backend_get_device(backend)).device;
            std::lock_guard<std::mutex> lock(mutex);
            auto found = devices.find(device);
            if (found != devices.end())
                return found->second;
            cudaDeviceProp prop  = {};
            const bool available = cudaGetDeviceProperties(&prop, device) == cudaSuccess && prop.major >= 8;
            devices.emplace(device, available);
            return available;
#else
            return false;
#endif
        }
        static ggml_status prepare(ggml_backend_t backend, ggml_tensor* const* nodes, int, void** state, size_t* bytes, void*) {
            SDW4Operation op;
            if (!sd_ggml_w4_describe(nodes[0], &op))
                return GGML_STATUS_FAILED;
            const auto* dst = nodes[0];
            bool may_alias  = dst->data || dst->view_src;
            if (dst->data && dst->buffer && !dst->view_src) {
                may_alias = false;
                for (const auto* input : dst->src) {
                    if (!input)
                        continue;
                    size_t offset     = 0;
                    const size_t size = ggml_nbytes(input);
                    while (!input->data && input->view_src) {
                        offset += input->view_offs;
                        input = input->view_src;
                    }
                    if (!input->data)
                        continue;
                    const auto a = reinterpret_cast<uintptr_t>(dst->data), b = reinterpret_cast<uintptr_t>(input->data) + offset;
                    may_alias |= !input->buffer || (input->buffer == dst->buffer && a < b + size && b < a + ggml_nbytes(dst));
                }
            }
            auto* plan        = new Plan;
            plan->mma         = use_mma(backend, op);
            *state            = plan;
            size_t gemm_bytes = 0;
#ifdef SD_CUDA_W4_CUTLASS
            if (plan->mma && op.kind == SDW4Kind::W4A4) {
                if (!op.bias && aligned(dst, 16) && aligned(op.weight_scale, 16))
                    plan->gemm = sd_cuda_i8_gemm_prepare(ggml_backend_get_device(backend), ggml_nrows(dst), dst->ne[0], op.input->ne[0], SDI8GemmActivation::None, &gemm_bytes, true);
                plan->mma = plan->gemm != nullptr;
                if (plan->gemm && gemm_bytes == 0)
                    gemm_bytes = 256;
            }
#endif
            const size_t tensor_bytes = ggml_nbytes(dst) * ((op.kind == SDW4Kind::W4A4 && !plan->mma ? 1 : 0) + (may_alias ? 1 : 0));
            plan->gemm_offset         = (tensor_bytes + 255) / 256 * 256;
            *bytes                    = plan->gemm_offset + gemm_bytes;
            return GGML_STATUS_SUCCESS;
        }
        static bool overlaps(const ggml_tensor* dst, const ggml_tensor* src) {
            if (!src)
                return false;
            const auto a = reinterpret_cast<uintptr_t>(dst->data), b = reinterpret_cast<uintptr_t>(src->data);
            return a <= b ? b - a < ggml_nbytes(dst) : a - b < ggml_nbytes(src);
        }
        static ggml_status execute(const ggml_ext_execution_context* e, ggml_tensor* const* nodes, int, void* state, void*) {
            auto* dst = nodes[0];
            SDW4Operation op;
            ggml_ext_gpu_context context;
            if (!sd_ggml_w4_describe(dst, &op) || !cuda_context(e->backend, &context))
                return GGML_STATUS_FAILED;
            const auto stream = static_cast<CUDAStream>(context.stream);
            const int64_t n = dst->ne[0], rows = ggml_nrows(dst), k = op.input->ne[0];
            auto* scratch = e->workspace ? static_cast<float*>(ggml_backend_buffer_get_base(e->workspace)) : nullptr;
            bool alias    = false;
            for (auto* input : dst->src)
                alias |= overlaps(dst, input);
            auto* output     = alias ? scratch : static_cast<float*>(dst->data);
            const auto* w    = static_cast<const char*>(op.weight->data);
            const auto* x    = static_cast<const char*>(op.input->data);
            const auto* xs   = reinterpret_cast<const float*>(x + k * ((rows + 3) / 4 * 4));
            const auto* ws   = static_cast<const float*>(op.weight_scale->data);
            const auto* bias = op.bias ? static_cast<const float*>(op.bias->data) : nullptr;
            const dim3 grid(unsigned((n + 63) / 64), unsigned((rows + 31) / 32)), block(16, 16);
#ifdef SD_CUDA_W4_CUTLASS
            auto* plan = static_cast<Plan*>(state);
            if (plan->mma) {
                if (plan->gemm) {
                    auto* workspace = scratch ? reinterpret_cast<char*>(scratch) + plan->gemm_offset : nullptr;
                    if (sd_cuda_i8_gemm_execute(plan->gemm, reinterpret_cast<const int8_t*>(x), reinterpret_cast<const int8_t*>(w), xs, ws, output, workspace, stream) != GGML_STATUS_SUCCESS)
                        return GGML_STATUS_FAILED;
                } else {
                    const dim3 tiles(unsigned((n + 63) / 64), unsigned((rows + 127) / 128));
                    sd_w4_mma::w4a8<128, 64, 128><<<tiles, 256, 0, stream>>>(reinterpret_cast<const uint8_t*>(w), reinterpret_cast<const int8_t*>(x), xs, ws,
                                                                             static_cast<const float*>(op.s_channel->data), static_cast<const uint8_t*>(op.s_rel->data), bias, output, int(rows), int(n), int(k));
                }
            } else
#endif
                if (op.kind == SDW4Kind::W4A4) {
                auto* accum = reinterpret_cast<int32_t*>(scratch + (alias ? ggml_nelements(dst) : 0));
                mul_mat_w4a4_convrot_dp4a<<<grid, block, 0, stream>>>(w, x, accum, uint32_t(n), uint32_t(rows), uint32_t(k / 2));
                if (cuda_last_error() != cuda_success)
                    return GGML_STATUS_FAILED;
                dequantize<<<unsigned((n * rows + 255) / 256), 256, 0, stream>>>(accum, xs, ws, bias, output, n, n * rows);
            } else {
                mul_mat_w4a8_convrot<<<grid, block, 0, stream>>>(w, x, ws,
                                                                 static_cast<const float*>(op.s_channel->data), static_cast<const uint8_t*>(op.s_rel->data),
                                                                 xs, bias, output, uint32_t(n), uint32_t(rows), uint32_t(k));
            }
            auto error = cuda_last_error();
            if (error == cuda_success && alias)
                error = cuda_copy_async(dst->data, output, ggml_nbytes(dst), stream);
            return error == cuda_success ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
        }
    }
    void register_w4(ggml_backend_dev_t device) {
        ggml_ext_kernel_desc d = {};
        d.abi_version          = GGML_EXT_ABI_VERSION;
        d.name                 = "sd.linear_w4_convrot." SD_CUDA_NAME ".v1";
        d.op                   = GGML_OP_CUSTOM;
        d.custom_op            = sd_ggml_w4_op();
        d.priority             = 100;
        d.supports             = sd_w4_cuda::supports;
        d.prepare              = sd_w4_cuda::prepare;
        d.execute              = sd_w4_cuda::execute;
        d.destroy              = [](void* state, void*) { delete static_cast<sd_w4_cuda::Plan*>(state); };
        GGML_ASSERT(ggml_ext_register_kernel_ex(device, &d, GGML_EXT_KERNEL_CAPTURE_SAFE | GGML_EXT_KERNEL_REPLAY_REUSABLE));
    }
}
