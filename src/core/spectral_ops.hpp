#ifndef __SD_CORE_SPECTRAL_OPS_HPP__
#define __SD_CORE_SPECTRAL_OPS_HPP__

#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/tensor.hpp"

// CPU-side FFT / DCT / DWT primitives for use by spectral samplers (SPEED etc.).
// Operates on sd::Tensor<float> laid out ggml-style [W, H, C, N] (W innermost).
// The 2D transforms run over the trailing two dims (W and H); leading dims are
// iterated as independent batches.
//
// Radix-2 only for FFT; DCT-II built on top via the mirror trick.  Sizes must
// be powers of 2 — the SPEED transitions the paper uses (0.25/0.5/1.0 fractions
// of a power-of-two latent) always land on power-of-two spatial dims.

namespace sd {
namespace spectral {

using cplx = std::complex<float>;

inline bool is_pow2(int64_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// In-place radix-2 Cooley-Tukey FFT of length N (must be a power of 2).
// sign = -1: forward transform;  sign = +1: inverse transform (unscaled).
inline void fft_1d(cplx* a, int64_t n, int sign) {
    if (!is_pow2(n)) {
        throw std::runtime_error("spectral::fft_1d: length " + std::to_string(n) + " is not a power of 2");
    }
    for (int64_t i = 1, j = 0; i < n; ++i) {
        int64_t bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }
    for (int64_t len = 2; len <= n; len <<= 1) {
        float ang = sign * 2.0f * static_cast<float>(M_PI) / static_cast<float>(len);
        cplx wlen(std::cos(ang), std::sin(ang));
        for (int64_t i = 0; i < n; i += len) {
            cplx w(1.0f, 0.0f);
            for (int64_t k = 0; k < len / 2; ++k) {
                cplx u = a[i + k];
                cplx v = a[i + k + len / 2] * w;
                a[i + k]           = u + v;
                a[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

// 2D FFT over the trailing two dims [..., H, W] of a real 4D tensor in ggml
// layout [W, H, C, N].  Returns a complex-valued grid of the same [W, H]
// dimensions for each (C, N).  Ortho normalization (divides by sqrt(H*W)).
inline std::vector<cplx> fft_2d(const sd::Tensor<float>& x) {
    const auto& s = x.shape();
    if (s.size() != 4) {
        throw std::runtime_error("spectral::fft_2d: expected 4D tensor [W,H,C,N]");
    }
    int64_t W  = s[0];
    int64_t H  = s[1];
    int64_t C  = s[2];
    int64_t N  = s[3];
    int64_t HW = H * W;
    if (!is_pow2(H) || !is_pow2(W)) {
        throw std::runtime_error("spectral::fft_2d: spatial dims must be powers of 2, got W=" +
                                 std::to_string(W) + " H=" + std::to_string(H));
    }
    std::vector<cplx> out(static_cast<size_t>(N * C * H * W));
    float norm      = 1.0f / std::sqrt(static_cast<float>(HW));
    const float* xp = x.data();
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            cplx* plane = out.data() + (n * C + c) * HW;
            for (int64_t i = 0; i < HW; ++i) {
                plane[i] = cplx(xp[(n * C + c) * HW + i], 0.0f);
            }
            std::vector<cplx> row(W);
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    row[w] = plane[h * W + w];
                }
                fft_1d(row.data(), W, -1);
                for (int64_t w = 0; w < W; ++w) {
                    plane[h * W + w] = row[w];
                }
            }
            std::vector<cplx> col(H);
            for (int64_t w = 0; w < W; ++w) {
                for (int64_t h = 0; h < H; ++h) {
                    col[h] = plane[h * W + w];
                }
                fft_1d(col.data(), H, -1);
                for (int64_t h = 0; h < H; ++h) {
                    plane[h * W + w] = col[h];
                }
            }
            for (int64_t i = 0; i < HW; ++i) {
                plane[i] *= norm;
            }
        }
    }
    return out;
}

// Inverse of fft_2d: takes a complex [N, C, H, W] grid and returns the real
// tensor of the given shape.  Ortho normalization.
inline sd::Tensor<float> ifft_2d(const std::vector<cplx>& in, const std::vector<int64_t>& shape) {
    if (shape.size() != 4) {
        throw std::runtime_error("spectral::ifft_2d: expected 4D output shape");
    }
    int64_t W  = shape[0];
    int64_t H  = shape[1];
    int64_t C  = shape[2];
    int64_t N  = shape[3];
    int64_t HW = H * W;
    if (!is_pow2(H) || !is_pow2(W)) {
        throw std::runtime_error("spectral::ifft_2d: spatial dims must be powers of 2");
    }
    if (static_cast<int64_t>(in.size()) != N * C * HW) {
        throw std::runtime_error("spectral::ifft_2d: input buffer size does not match shape");
    }
    sd::Tensor<float> out(shape);
    float norm = 1.0f / std::sqrt(static_cast<float>(HW));
    float* op  = out.data();
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            std::vector<cplx> plane(HW);
            for (int64_t i = 0; i < HW; ++i) {
                plane[i] = in[(n * C + c) * HW + i];
            }
            std::vector<cplx> row(W);
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    row[w] = plane[h * W + w];
                }
                fft_1d(row.data(), W, +1);
                for (int64_t w = 0; w < W; ++w) {
                    plane[h * W + w] = row[w] / static_cast<float>(W);
                }
            }
            std::vector<cplx> col(H);
            for (int64_t w = 0; w < W; ++w) {
                for (int64_t h = 0; h < H; ++h) {
                    col[h] = plane[h * W + w];
                }
                fft_1d(col.data(), H, +1);
                for (int64_t h = 0; h < H; ++h) {
                    plane[h * W + w] = col[h] / static_cast<float>(H);
                }
            }
            for (int64_t i = 0; i < HW; ++i) {
                op[(n * C + c) * HW + i] = plane[i].real() / norm;
            }
        }
    }
    return out;
}

// 1D orthonormal DCT-II by direct summation (O(N^2)).
// X[k] = f_k * sum_n x[n] * cos(pi * (2n+1) * k / (2N)),
// where f_0 = sqrt(1/N) and f_k = sqrt(2/N) for k > 0.
inline void dct1d_direct(const float* in, float* out_row, int64_t N) {
    float scale0    = 1.0f / std::sqrt(static_cast<float>(N));
    float scale     = std::sqrt(2.0f / static_cast<float>(N));
    float inv_2N    = 1.0f / static_cast<float>(2 * N);
    for (int64_t k = 0; k < N; ++k) {
        float sum = 0.0f;
        for (int64_t n = 0; n < N; ++n) {
            sum += in[n] * std::cos(static_cast<float>(M_PI) * (2 * n + 1) * k * inv_2N);
        }
        out_row[k] = sum * ((k == 0) ? scale0 : scale);
    }
}

// 1D orthonormal DCT-III (inverse of DCT-II by direct summation).
inline void idct1d_direct(const float* in, float* out_row, int64_t N) {
    float scale0    = 1.0f / std::sqrt(static_cast<float>(N));
    float scale     = std::sqrt(2.0f / static_cast<float>(N));
    float inv_2N    = 1.0f / static_cast<float>(2 * N);
    for (int64_t n = 0; n < N; ++n) {
        float sum = in[0] * scale0;
        for (int64_t k = 1; k < N; ++k) {
            sum += in[k] * scale * std::cos(static_cast<float>(M_PI) * (2 * n + 1) * k * inv_2N);
        }
        out_row[n] = sum;
    }
}

// 2D orthonormal DCT-II.  Row-column decomposition using dct1d_direct.
inline sd::Tensor<float> dct2_2d(const sd::Tensor<float>& x) {
    const auto& s = x.shape();
    if (s.size() != 4) {
        throw std::runtime_error("spectral::dct2_2d: expected 4D tensor");
    }
    int64_t W  = s[0];
    int64_t H  = s[1];
    int64_t C  = s[2];
    int64_t N  = s[3];
    int64_t HW = H * W;
    sd::Tensor<float> out(s);
    const float* xp = x.data();
    float* op       = out.data();
    std::vector<float> col_in(H), col_out(H);
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            float* plane = op + (n * C + c) * HW;
            for (int64_t h = 0; h < H; ++h) {
                dct1d_direct(xp + (n * C + c) * HW + h * W, plane + h * W, W);
            }
            for (int64_t w = 0; w < W; ++w) {
                for (int64_t h = 0; h < H; ++h) {
                    col_in[h] = plane[h * W + w];
                }
                dct1d_direct(col_in.data(), col_out.data(), H);
                for (int64_t h = 0; h < H; ++h) {
                    plane[h * W + w] = col_out[h];
                }
            }
        }
    }
    return out;
}

// 2D orthonormal inverse DCT (DCT-III).
inline sd::Tensor<float> idct2_2d(const sd::Tensor<float>& c_in) {
    const auto& s = c_in.shape();
    if (s.size() != 4) {
        throw std::runtime_error("spectral::idct2_2d: expected 4D tensor");
    }
    int64_t W  = s[0];
    int64_t H  = s[1];
    int64_t C  = s[2];
    int64_t N  = s[3];
    int64_t HW = H * W;
    sd::Tensor<float> out(s);
    const float* cp = c_in.data();
    float* op       = out.data();
    std::vector<float> col_in(H), col_out(H);
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t ch = 0; ch < C; ++ch) {
            float* plane = op + (n * C + ch) * HW;
            for (int64_t h = 0; h < H; ++h) {
                idct1d_direct(cp + (n * C + ch) * HW + h * W, plane + h * W, W);
            }
            for (int64_t w = 0; w < W; ++w) {
                for (int64_t h = 0; h < H; ++h) {
                    col_in[h] = plane[h * W + w];
                }
                idct1d_direct(col_in.data(), col_out.data(), H);
                for (int64_t h = 0; h < H; ++h) {
                    plane[h * W + w] = col_out[h];
                }
            }
        }
    }
    return out;
}

}  // namespace spectral
}  // namespace sd

#endif  // __SD_CORE_SPECTRAL_OPS_HPP__
