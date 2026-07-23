#ifndef __SD_RUNTIME_SPEED_SAMPLER_HPP__
#define __SD_RUNTIME_SPEED_SAMPLER_HPP__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "core/rng.hpp"
#include "core/spectral_ops.hpp"
#include "core/tensor.hpp"

// SPEED — Spectral Progressive Diffusion sampler.
// Ref: Xiao et al. arXiv:2605.18736; https://github.com/howardhx/speed;
// ComfyUI-SPEED (https://github.com/ruwwww/ComfyUI-SPEED).
//
// Runs the denoising trajectory of a flow model at progressively higher
// spatial resolutions, saving the majority of the DiT compute during the
// low-frequency-dominated early steps. Between resolution levels, the latent
// is expanded via a spectral basis (DCT or FFT) — high-frequency bands are
// filled with fresh Gaussian noise scaled by the current sigma. The
// flow-matching time at each transition is rescaled by kappa (paper Eq. 5-6)
// so the base sampler operates on a consistent schedule after resize.
//
// Kept lean in this initial implementation: image latents only, DCT + FFT
// transforms, base sampler is deterministic euler, delta-optimal transitions
// only. DWT / manual sigmas / video latents can be added incrementally.

namespace sd {
namespace speed {

enum class Transform {
    DCT,
    FFT,
};

struct Config {
    std::vector<float> scales = {0.5f, 1.0f};  // strictly increasing, ends at 1.0
    float delta               = 0.01f;         // noise-dominated tolerance (paper Eq. 9)
    float spectrum_A          = 203.615097f;   // Flux dev preset
    float spectrum_beta       = 1.915461f;
    Transform transform       = Transform::DCT;
    uint64_t seed             = 0;             // for the spectral noise padding
    // If non-empty, overrides the delta-optimal schedule: use these sigma
    // thresholds directly, one per transition (length == scales.size() - 1).
    // Values must be strictly decreasing and each in (0, 1).
    std::vector<float> manual_sigmas;
};

// Power-spectrum presets fitted per VAE in the reference impl's configs.yaml.
struct Preset {
    float A;
    float beta;
};
inline Preset preset_flux()  { return {203.615097f, 1.915461f}; }
inline Preset preset_wan21() { return {219.484718f, 2.422687f}; }

// Auto-select a preset based on which VAE the model uses. All Flux-family
// image models (Flux, Flux2, SeFi, Z-Image, Qwen Image, ...) share the Flux
// VAE calibration in the reference impl; only WAN gets a distinct preset.
inline Preset preset_for(SDVersion version) {
    if (sd_version_is_wan(version)) {
        return preset_wan21();
    }
    return preset_flux();
}

inline Config default_config_for(SDVersion version) {
    Config cfg;
    Preset p          = preset_for(version);
    cfg.spectrum_A    = p.A;
    cfg.spectrum_beta = p.beta;
    return cfg;
}

// Power-law radial spectrum P(omega) = A * |omega|^(-beta) (Eq. 8).
inline float power_spectrum(float omega, float A, float beta) {
    return A * std::pow(std::abs(omega), -beta);
}

// Activation time for one radial frequency (Eq. 9).
inline float activation_time(float P_omega, float delta) {
    return 1.0f / (1.0f + std::sqrt(delta / (P_omega * (1.0f + P_omega - delta))));
}

// Rescale factor at a transition (paper Eq. 5).
inline float kappa(float t, float r) {
    return r / (1.0f + (r - 1.0f) * t);
}

// Aligned flow-matching time after spectral noise expansion (Eq. 6).
inline float align_timestep(float t, float r) {
    return t * kappa(t, r);
}

// Delta-optimal transition times between consecutive scales (Eq. 10).
inline std::vector<float> delta_optimal_transitions(const std::vector<float>& scales,
                                                    float delta,
                                                    float A,
                                                    float beta,
                                                    int64_t H,
                                                    int64_t W) {
    float omega_max = 0.5f * static_cast<float>(std::min(H, W));
    std::vector<float> out;
    out.reserve(scales.size() > 0 ? scales.size() - 1 : 0);
    for (size_t i = 0; i + 1 < scales.size(); ++i) {
        float omega_i = scales[i] * omega_max;
        out.push_back(activation_time(power_spectrum(omega_i, A, beta), delta));
    }
    return out;
}

struct Transition {
    int step_idx;
    float s_from;
    float s_to;
};

// Resolve which sampler step index each threshold in `t_stars` first crosses.
inline std::vector<Transition> resolve_transitions(const std::vector<float>& sigmas,
                                                   const std::vector<float>& scales,
                                                   const std::vector<float>& t_stars) {
    std::vector<Transition> out;
    if (scales.size() < 2) {
        return out;
    }
    int n_steps = static_cast<int>(sigmas.size()) - 1;
    for (size_t i = 0; i + 1 < scales.size(); ++i) {
        float thr    = t_stars[i];
        int step_idx = n_steps;
        for (int j = 0; j < n_steps; ++j) {
            if (sigmas[j] <= thr) {
                step_idx = j;
                break;
            }
        }
        if (step_idx >= n_steps) {
            break;
        }
        out.push_back({step_idx, scales[i], scales[i + 1]});
    }
    return out;
}

// Round a scale fraction to an even integer size so the DiT patchify stays valid.
inline int64_t scaled_dim(float s, int64_t full) {
    int64_t v = static_cast<int64_t>(std::llround(static_cast<double>(s) * static_cast<double>(full)));
    if (v < 2) {
        v = 2;
    }
    if ((v & 1) != 0) {
        v++;  // keep even so patch_size=2 grids align
    }
    return v;
}

// DCT-truncation downscale of a 4D [W, H, C, N] tensor to (H_lo, W_lo).
// This is the "initial coarse latent" step of SPEED.
inline sd::Tensor<float> dct_downscale(const sd::Tensor<float>& x, int64_t H_lo, int64_t W_lo) {
    const auto& s = x.shape();
    if (s.size() != 4) {
        throw std::runtime_error("speed::dct_downscale: expected 4D tensor");
    }
    int64_t W_hi = s[0];
    int64_t H_hi = s[1];
    int64_t C    = s[2];
    int64_t N    = s[3];
    if (H_lo >= H_hi && W_lo >= W_hi) {
        return x;
    }
    sd::Tensor<float> coeffs = sd::spectral::dct2_2d(x);
    sd::Tensor<float> trunc({W_lo, H_lo, C, N});
    const float* cp = coeffs.data();
    float* tp       = trunc.data();
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t h = 0; h < H_lo; ++h) {
                for (int64_t w = 0; w < W_lo; ++w) {
                    tp[((n * C + c) * H_lo + h) * W_lo + w] =
                        cp[((n * C + c) * H_hi + h) * W_hi + w];
                }
            }
        }
    }
    return sd::spectral::idct2_2d(trunc);
}

// Spectral noise expansion (DCT variant): promote a low-res latent to a
// higher-res one by placing its DCT coefficients in the low-frequency block of
// a larger coefficient plane and filling the rest with scaled Gaussian noise.
inline sd::Tensor<float> dct_expand(const sd::Tensor<float>& x,
                                    int64_t H_tgt,
                                    int64_t W_tgt,
                                    float t_noise,
                                    uint64_t seed) {
    const auto& s = x.shape();
    if (s.size() != 4) {
        throw std::runtime_error("speed::dct_expand: expected 4D tensor");
    }
    int64_t W_src = s[0];
    int64_t H_src = s[1];
    int64_t C     = s[2];
    int64_t N     = s[3];
    if (H_tgt < H_src || W_tgt < W_src) {
        throw std::runtime_error("speed::dct_expand: target smaller than source");
    }
    sd::Tensor<float> src_coeffs = sd::spectral::dct2_2d(x);
    sd::Tensor<float> big({W_tgt, H_tgt, C, N});
    auto rng = std::make_shared<STDDefaultRNG>();
    rng->manual_seed(seed);
    auto noise_flat = rng->randn(static_cast<uint32_t>(big.numel()));
    std::memcpy(big.data(), noise_flat.data(), noise_flat.size() * sizeof(float));
    for (int64_t i = 0; i < big.numel(); ++i) {
        big.data()[i] *= t_noise;
    }
    // Overwrite the low-freq block with the source coefficients.
    const float* sp = src_coeffs.data();
    float* bp       = big.data();
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t h = 0; h < H_src; ++h) {
                for (int64_t w = 0; w < W_src; ++w) {
                    bp[((n * C + c) * H_tgt + h) * W_tgt + w] =
                        sp[((n * C + c) * H_src + h) * W_src + w];
                }
            }
        }
    }
    return sd::spectral::idct2_2d(big);
}

// FFT variant of the noise expansion: shifted-spectrum padding with the source
// spectrum centered in the higher-frequency plane. Matches the reference
// _fft_expand_np. Currently unused by the default DCT transform path but kept
// for completeness so we can switch via config.
inline sd::Tensor<float> fft_expand(const sd::Tensor<float>& x,
                                    int64_t H_tgt,
                                    int64_t W_tgt,
                                    float t_noise,
                                    uint64_t seed) {
    const auto& s = x.shape();
    int64_t W_src = s[0];
    int64_t H_src = s[1];
    int64_t C     = s[2];
    int64_t N     = s[3];
    auto src_fft  = sd::spectral::fft_2d(x);
    std::vector<sd::spectral::cplx> big(static_cast<size_t>(N * C * H_tgt * W_tgt));
    auto rng = std::make_shared<STDDefaultRNG>();
    rng->manual_seed(seed);
    auto noise_r = rng->randn(static_cast<uint32_t>(N * C * H_tgt * W_tgt));
    auto noise_i = rng->randn(static_cast<uint32_t>(N * C * H_tgt * W_tgt));
    float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
    for (size_t i = 0; i < big.size(); ++i) {
        big[i] = sd::spectral::cplx(t_noise * noise_r[i] * inv_sqrt2,
                                    t_noise * noise_i[i] * inv_sqrt2);
    }
    int64_t pad_h = (H_tgt - H_src) / 2;
    int64_t pad_w = (W_tgt - W_src) / 2;
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t h = 0; h < H_src; ++h) {
                for (int64_t w = 0; w < W_src; ++w) {
                    big[((n * C + c) * H_tgt + pad_h + h) * W_tgt + pad_w + w] =
                        src_fft[((n * C + c) * H_src + h) * W_src + w];
                }
            }
        }
    }
    return sd::spectral::ifft_2d(big, {W_tgt, H_tgt, C, N});
}

// Signature matches the other flow samplers: (model, x, sigmas, ...).
// The extra_sample_args pointer is parsed for speed_* overrides.
template <typename ModelCb>
inline sd::Tensor<float> sample_speed_flow(ModelCb model,
                                           sd::Tensor<float> x,
                                           std::vector<float> sigmas,
                                           const Config& cfg) {
    const auto& s = x.shape();
    if (s.size() != 4) {
        throw std::runtime_error("speed::sample_speed_flow: expected 4D latent [W,H,C,N]");
    }
    int64_t W_full = s[0];
    int64_t H_full = s[1];
    if (cfg.scales.size() < 2) {
        // Nothing to do; run plain euler over the full schedule.
        int steps = static_cast<int>(sigmas.size()) - 1;
        for (int i = 0; i < steps; ++i) {
            auto denoised_opt = model(x, sigmas[i], i + 1);
            if (denoised_opt.pred.empty()) {
                return {};
            }
            sd::Tensor<float> denoised = std::move(denoised_opt.pred);
            if (sigmas[i + 1] == 0.f) {
                x = denoised;
            } else {
                float r = sigmas[i + 1] / sigmas[i];
                x       = r * x + (1.0f - r) * denoised;
            }
        }
        return x;
    }

    std::vector<float> t_stars;
    if (!cfg.manual_sigmas.empty()) {
        if (cfg.manual_sigmas.size() != cfg.scales.size() - 1) {
            throw std::runtime_error("speed::sample_speed_flow: speed_manual_sigmas length must be scales.size() - 1");
        }
        t_stars = cfg.manual_sigmas;
    } else {
        t_stars = delta_optimal_transitions(cfg.scales, cfg.delta, cfg.spectrum_A,
                                            cfg.spectrum_beta, H_full, W_full);
    }
    auto transitions = resolve_transitions(sigmas, cfg.scales, t_stars);
    // Only downscale if there's at least one transition that will bring the
    // latent back up; otherwise the loop would return a low-res tensor to the
    // VAE and crash. If no transitions fire, run at full resolution.
    if (!transitions.empty() && cfg.scales.front() < 1.0f) {
        int64_t H_lo = scaled_dim(cfg.scales.front(), H_full);
        int64_t W_lo = scaled_dim(cfg.scales.front(), W_full);
        x            = dct_downscale(x, H_lo, W_lo);
    }
    int total_steps = static_cast<int>(sigmas.size()) - 1;
    int seg_start   = 0;
    for (size_t seg = 0; seg <= transitions.size(); ++seg) {
        int seg_end = (seg < transitions.size()) ? transitions[seg].step_idx : total_steps;
        for (int i = seg_start; i < seg_end; ++i) {
            auto denoised_opt = model(x, sigmas[i], i + 1);
            if (denoised_opt.pred.empty()) {
                return {};
            }
            sd::Tensor<float> denoised = std::move(denoised_opt.pred);
            if (sigmas[i + 1] == 0.f) {
                x = denoised;
            } else {
                float r = sigmas[i + 1] / sigmas[i];
                x       = r * x + (1.0f - r) * denoised;
            }
        }
        if (seg == transitions.size()) {
            break;
        }
        const Transition& tr = transitions[seg];
        int64_t H_tgt        = scaled_dim(tr.s_to, H_full);
        int64_t W_tgt        = scaled_dim(tr.s_to, W_full);
        float r              = tr.s_to / tr.s_from;
        float t_at           = sigmas[tr.step_idx];
        uint64_t seed        = cfg.seed + static_cast<uint64_t>(seg + 1) * 10000ULL;
        if (cfg.transform == Transform::FFT) {
            x = fft_expand(x, H_tgt, W_tgt, t_at, seed);
        } else {
            x = dct_expand(x, H_tgt, W_tgt, t_at, seed);
        }
        float k = kappa(t_at, r);
        for (int64_t i = 0; i < x.numel(); ++i) {
            x.data()[i] *= k;
        }
        // Patch the transition sigma so the base solver continues on the aligned schedule.
        sigmas[tr.step_idx] = align_timestep(t_at, r);
        seg_start           = tr.step_idx;
    }
    return x;
}

// Parse "speed_scales=0.5,1.0", "speed_delta=0.01", "speed_transform=dct|fft",
// "speed_A=...", "speed_beta=...", "speed_seed=..." out of extra_sample_args.
// Missing keys inherit from the caller-supplied defaults.
inline void parse_config_from_args(Config& cfg,
                                   const std::vector<std::pair<std::string, std::string>>& kv) {
    for (const auto& [key, value] : kv) {
        if (key == "speed_scales") {
            // Accept ':' as the scales separator so the value survives the
            // comma-splitting parse_key_value_args does at the top level.
            // e.g. --extra-sample-args speed_scales=0.25:0.5:1.0
            std::vector<float> parsed;
            std::string cur;
            for (char ch : value) {
                if (ch == ':' || ch == '/' || ch == '|') {
                    if (!cur.empty()) {
                        parsed.push_back(std::stof(cur));
                        cur.clear();
                    }
                } else if (ch != ' ') {
                    cur.push_back(ch);
                }
            }
            if (!cur.empty()) {
                parsed.push_back(std::stof(cur));
            }
            if (!parsed.empty()) {
                cfg.scales = std::move(parsed);
            }
        } else if (key == "speed_levels") {
            // Convenience shortcut: speed_levels=2 -> {0.5, 1.0}; =3 -> {0.25, 0.5, 1.0}; etc.
            // scales[i] = 2^(i - (n-1)) for i in [0, n-1].
            int n = std::stoi(value);
            if (n >= 2) {
                std::vector<float> parsed;
                parsed.reserve(static_cast<size_t>(n));
                for (int i = 0; i < n; ++i) {
                    parsed.push_back(std::ldexp(1.0f, i - (n - 1)));
                }
                cfg.scales = std::move(parsed);
            }
        } else if (key == "speed_manual_sigmas") {
            std::vector<float> parsed;
            std::string cur;
            for (char ch : value) {
                if (ch == ':' || ch == '/' || ch == '|') {
                    if (!cur.empty()) {
                        parsed.push_back(std::stof(cur));
                        cur.clear();
                    }
                } else if (ch != ' ') {
                    cur.push_back(ch);
                }
            }
            if (!cur.empty()) {
                parsed.push_back(std::stof(cur));
            }
            cfg.manual_sigmas = std::move(parsed);
        } else if (key == "speed_preset") {
            if (value == "wan21" || value == "wan" || value == "wan2.1") {
                Preset p         = preset_wan21();
                cfg.spectrum_A    = p.A;
                cfg.spectrum_beta = p.beta;
            } else {
                Preset p         = preset_flux();
                cfg.spectrum_A    = p.A;
                cfg.spectrum_beta = p.beta;
            }
        } else if (key == "speed_delta") {
            cfg.delta = std::stof(value);
        } else if (key == "speed_A") {
            cfg.spectrum_A = std::stof(value);
        } else if (key == "speed_beta") {
            cfg.spectrum_beta = std::stof(value);
        } else if (key == "speed_seed") {
            cfg.seed = static_cast<uint64_t>(std::stoull(value));
        } else if (key == "speed_transform") {
            if (value == "fft") {
                cfg.transform = Transform::FFT;
            } else {
                cfg.transform = Transform::DCT;
            }
        }
    }
}

}  // namespace speed
}  // namespace sd

#endif  // __SD_RUNTIME_SPEED_SAMPLER_HPP__
