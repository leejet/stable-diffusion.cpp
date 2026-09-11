#ifndef __SD_RUNTIME_LANPAINT_HPP__
#define __SD_RUNTIME_LANPAINT_HPP__

/*
 * LanPaint: Langevin-dynamics inpainting, following the ComfyUI LanPaint
 * algorithm (LanPaint/src/LanPaint/lanpaint.py + nodes.py).
 *
 * The underlying model evaluation is abstracted behind `lanpaint_eval_t`
 * (the unblended (x0, x0_BIG) pair), bundled with its Denoiser in
 * `LanPaintInnerModel`. This header includes `runtime/denoiser.hpp`;
 * `denoiser.hpp` must not include this header.
 *
 * Mask convention (comfy `latent_mask = 1 - denoise_mask`):
 *   `latent_mask` is a float tensor, 1 = KEEP the original image, 0 = EDIT.
 * sd.cpp's `denoise_mask` is a latent-space tensor of shape {W,H,1,1}
 * (images) / {W,H,T,1,1} (video) against latents {W,H,C,1} / {W,H,T,C,1},
 * so the caller passes `latent_mask = 1 - denoise_mask` and broadcasting
 * is automatic.
 *
 * Sigma convention: `sigma` is the native denoiser sigma -- flow time t in
 * [0,1] for the DiscreteFlowDenoiser family, the VE sigma (e.g.
 * ~0.03..14.6 for SD1.5) for the CompVisDenoiser family. `compute_times()`
 * derives the (VE_Sigma, abt, Flow_t) triple from it.
 */

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include "core/rng.hpp"
#include "core/tensor.hpp"
#include "runtime/denoiser.hpp"
#include "runtime/guidance.h"

// Parameters (defaults from the ComfyUI node, v2.1.0).
struct LanPaintParams {
    int n_steps    = 5;              // NSteps: inner Langevin steps per outer step
    float friction = 15.f;           // Friction: unused by the overdamped scheme
                                     // implemented here
    float lambda        = 5.f;       // Lambda: strength of the keep-region score_y
    float beta          = 1.f;       // Beta: time-scale ratio of the y branch
    float step_size     = 0.2f;      // StepSize
    int early_stop      = 1;         // EarlyStop: drop the inner loop in the last N outer steps
    float min_step_frac = 1.f;       // MinStepFrac: pin step size below this, ramp n_eff down
    float cfg_big       = INFINITY;  // BIG-CFG scale; sentinel INFINITY == "Image First" (cfg_BIG == cfg).
                                     // Consumed by the evaluation, not by the cycle.
};

// Unified time triple (comfy: `current_times = (VE_Sigma, abt, Flow_t)`).
struct LanPaintModelTimes {
    float ve_sigma;  // variance-exploding sigma
    float abt;       // the "remaining signal" fraction in [0,1]
    float flow_t;    // flow time t
};

// Result of one diffusion forward: the CFG denoised x0 and its BIG-CFG
// variant. Both must be in NATIVE latent coordinates and UNBLENDED (the
// LanPaint cycle does the mask blending itself).
struct LanPaintEval {
    sd::Tensor<float> x0;      // pred (full CFG, unblended)
    sd::Tensor<float> x0_big;  // BIG-CFG denoised; may equal x0
};

// The underlying network evaluation. `x` is in native latent coordinates,
// `sigma` in native units (flow t for flow models, VE sigma otherwise) --
// i.e. exactly the sigma the outer sampler passes at this step. Must be
// pure (no side effects): the caller owns `x`.
using lanpaint_eval_t = std::function<LanPaintEval(const sd::Tensor<float>& x, float sigma)>;

// The model LanPaint paints with: the unblended network eval plus the
// Denoiser it belongs to (noise-scaling formulas, flow/VE mode) -- comfy's
// `inner_model` + `inner_model.inner_model.model_sampling`. `denoiser` is a
// borrowed, non-owning pointer (non-const because noise_scaling() is a
// non-const virtual) that must outlive the LanPaint engine and its
// callback.
struct LanPaintInnerModel {
    lanpaint_eval_t eval;
    Denoiser* denoiser;
    bool is_flow;
};

// Builds a LanPaintInnerModel from a runtime denoiser: derives the flow/VE
// mode from the denoiser type and refuses the denoisers whose latent
// scaling / time conventions are unsupported (MiniT2I: noise*2 start,
// reversed time; SeFi: dual timesteps). Returns nullopt when refused.
inline std::optional<LanPaintInnerModel> make_lanpaint_inner_model(lanpaint_eval_t eval,
                                                                   const std::shared_ptr<Denoiser>& denoiser) {
    if (!denoiser) {
        LOG_ERROR("LanPaint requires a denoiser");
        return std::nullopt;
    }
    if (std::dynamic_pointer_cast<MiniT2IFlowDenoiser>(denoiser) || std::dynamic_pointer_cast<SefiFlowDenoiser>(denoiser)) {
        LOG_ERROR("LanPaint does not support this denoiser's latent scaling / time conventions");
        return std::nullopt;
    }
    return LanPaintInnerModel{std::move(eval),
                              denoiser.get(),
                              (bool)std::dynamic_pointer_cast<DiscreteFlowDenoiser>(denoiser)};
}

// One LanPaint engine. Holds the per-sample context (the inner model, the
// replace-step noise, the target latent, the keep mask and the RNG); `run()`
// executes one full outer-step cycle on the sampler's latent in place and
// returns the blended denoised x0 (comfy's `KSamplerX0Inpaint.__call__` +
// `LanPaint.LanPaint` for one step). `make_callback()` wraps `run()` into
// the callback shape the sampler kernels call (outer index == |step|-1; `x`
// by non-const reference so the evolved latent is written back to the
// sampler's local state, comfy's `input_x.copy_(x)`).
struct LanPaint {
    LanPaint(const LanPaintParams& params,
             LanPaintInnerModel inner_model,
             const std::shared_ptr<RNG>& rng,
             const sd::Tensor<float>& noise,
             const sd::Tensor<float>& latent_image,
             const sd::Tensor<float>& latent_mask)
        : params_(params),
          inner_model_(std::move(inner_model)),
          rng_(rng),
          noise_(noise),
          latent_image_(latent_image),
          latent_mask_(latent_mask) {
    }

    static LanPaintModelTimes compute_times(bool is_flow, float sigma) {
        LanPaintModelTimes t;
        if (is_flow) {
            t.flow_t   = sigma;
            float f    = t.flow_t;
            t.abt      = (1.f - f) * (1.f - f) / ((1.f - f) * (1.f - f) + f * f);
            t.ve_sigma = f / (1.f - f);
        } else {
            t.ve_sigma = sigma;
            t.abt      = 1.f / (1.f + sigma * sigma);
            t.flow_t   = std::sqrt(1.f - t.abt) / (std::sqrt(1.f - t.abt) + std::sqrt(t.abt));
        }
        return t;
    }

    // One full LanPaint cycle (comfy `KSamplerX0Inpaint.__call__` +
    // `LanPaint.LanPaint`) for a single outer step.
    //
    // In:  `x` = the sampler's latent in NATIVE coordinates.
    // Out: `x` is overwritten with the post-Langevin native latent (comfy's
    //      in-place `input_x.copy_(x)`), and the returned tensor is the
    //      blended denoised x0 (`out`).
    //
    // `sigma` is the native sigma of this outer step (flow t or VE sigma);
    // `sigmas` is the full schedule (for total_steps); `outer_step` is the
    // 0-based outer index (== |step|-1 in the sd.cpp kernel convention).
    sd::Tensor<float> run(sd::Tensor<float>& x, float sigma, const std::vector<float>& sigmas, int outer_step) const {
        if (latent_mask_.empty()) {
            // No mask: LanPaint is a no-op -- return the plain denoised x0.
            LanPaintEval ev = inner_model_.eval(x, sigma);
            return ev.x0.empty() ? sd::Tensor<float>() : std::move(ev.x0);
        }

        const LanPaintModelTimes ct = compute_times(inner_model_.is_flow, sigma);
        const float abt             = ct.abt;

        // 1. Step size: StepSize * clamp(1 - abt, min = MinStepFrac).
        const float step_size = params_.step_size * std::max(1.f - abt, params_.min_step_frac);

        // 2. Effective inner-step count (comfy `min_step_frac_effective_steps`).
        const int total_steps = static_cast<int>(sigmas.size()) - 1;
        int n_eff             = params_.n_steps;
        if (total_steps - outer_step <= params_.early_stop) {
            n_eff = 0;
        } else if (params_.min_step_frac > 0.f && (1.f - abt) < params_.min_step_frac && params_.n_steps > 0.f) {
            n_eff = std::max(0, static_cast<int>(std::lround((float)params_.n_steps * (1.f - abt) / params_.min_step_frac)));
        }

        // 3. Replace step: the keep region is re-noised around the current
        //    noise level, at every outer step (even when n_eff == 0):
        //      x <- x*(1-m) + model_sampling.noise_scaling(sigma, noise, y)*m
        //    with the noise scaling delegated to the Denoiser.
        const bool is_flow = inner_model_.is_flow;
        x                  = x * (1.f - latent_mask_) +
            inner_model_.denoiser->noise_scaling(sigma, noise_, latent_image_) * latent_mask_;

        // 4. To variance-preserving form.
        sd::Tensor<float> x_t;
        if (is_flow) {
            const float s = std::sqrt(abt) + std::sqrt(1.f - abt);
            x_t           = x * s;
        } else {
            x_t = x / std::sqrt(1.f + ct.ve_sigma * ct.ve_sigma);
        }

        // 5. Langevin inner loop. Scalar coefficients (sigma is a scalar,
        //    hence so are abt and step_size):
        //      sigma_x = abt^0 = 1 ; sigma_y = beta*abt^0 = beta
        //      dtx = 2*step_size*sigma_x ; dty = 2*step_size*sigma_y
        //      (prepare_step_size returns dtx/2, dty/2)
        //      dt   = (dtx/2)*(1-m) + (dty/2)*m
        //      A_x  = 1/(1-abt) ; A_y = (1+lambda)/(1-abt)
        //      A    = A_x*(1-m) + A_y*m
        //      D^2  = 2
        const float one_minus_abt = 1.f - abt;
        const float A_x           = 1.f / one_minus_abt;
        const float A_y           = (1.f + params_.lambda) / one_minus_abt;
        const float half_dtx      = step_size;                 // step_size * sigma_x
        const float half_dty      = step_size * params_.beta;  // step_size * sigma_y
        const float D2            = 2.f;

        sd::Tensor<float> dt = half_dtx * (1.f - latent_mask_) + half_dty * latent_mask_;
        sd::Tensor<float> A  = A_x * (1.f - latent_mask_) + A_y * latent_mask_;

        // The inner loop runs only when the effective step count is positive
        // (comfy: KSamplerX0Inpaint sets n_eff = 0 for EarlyStop / the
        // MinStepFrac ramp, making `range(n_eff)` empty) and the time step is
        // positive (comfy: `if mean(dtx) <= 0: return` in langevin_dynamics;
        // dtx == step_size here).
        const bool inner_ok = (n_eff > 0) && (step_size > 0.f);

        // Coef_C(x_t): x0 = x_t + score(x_t);
        //              C  = (sqrt(abt)*x0 - x_t)/(1-abt) + A*x_t
        // The score (comfy `score_model`):
        //   deconvert x_t -> native x, eval model -> (x0, x0_big)
        //   score_x = -(x_t - x0)
        //   score_y = -(1+lambda)*(x_t - y) + lambda*(x_t - x0_big)
        //   score   = score_x*(1-m) + score_y*m
        // Only returns C since x0 is not used nor updated by further pipeline
        auto score_and_C = [&](const sd::Tensor<float>& xt) -> sd::Tensor<float> {
            sd::Tensor<float> x_native;
            if (is_flow) {
                const float s = std::sqrt(abt) + std::sqrt(1.f - abt);
                x_native      = xt / s;
            } else {
                x_native = xt * std::sqrt(1.f + ct.ve_sigma * ct.ve_sigma);
            }
            LanPaintEval ev = inner_model_.eval(x_native, sigma);
            if (ev.x0.empty()) {
                return sd::Tensor<float>();
            }
            sd::Tensor<float> score_x = -(xt - ev.x0);
            sd::Tensor<float> score_y = -(1.f + params_.lambda) * (xt - latent_image_) + params_.lambda * (xt - ev.x0_big);
            sd::Tensor<float> score   = score_x * (1.f - latent_mask_) + score_y * latent_mask_;
            sd::Tensor<float> x0      = xt + score;
            return (std::sqrt(abt) * x0 - xt) / one_minus_abt + A * xt;
        };

        // comfy `run_overdamped`
        sd::Tensor<float> C;
        bool have_C = false;
        if (inner_ok) {
            const sd::Tensor<float> dt_half = dt * 0.5f;
            for (int i = 0; i < n_eff; ++i) {
                if (!have_C) {
                    C = score_and_C(x_t);
                    if (C.empty()) {
                        return sd::Tensor<float>();
                    }
                    x_t    = advance_overdamped(x_t, dt, A, C, D2, rng_);
                    have_C = true;
                } else {
                    x_t                     = advance_overdamped(x_t, dt_half, A, C, D2, rng_);
                    sd::Tensor<float> C_new = score_and_C(x_t);
                    if (C_new.empty()) {
                        return sd::Tensor<float>();
                    }
                    x_t = x_t + (C_new - C) * dt;
                    x_t = advance_overdamped(x_t, dt_half, A, C_new, D2, rng_);
                    C   = C_new;
                }
            }
        }

        // 6. Back to native.
        if (is_flow) {
            const float s = std::sqrt(abt) + std::sqrt(1.f - abt);
            x             = x_t / s;
        } else {
            x = x_t * std::sqrt(1.f + ct.ve_sigma * ct.ve_sigma);
        }

        // 7. Final model eval at (x, sigma) -> out (blended).
        //    comfy: out = out*(1-m) + latent_image*m
        LanPaintEval out_ev = inner_model_.eval(x, sigma);
        if (out_ev.x0.empty()) {
            return sd::Tensor<float>();
        }
        sd::Tensor<float> out = out_ev.x0 * (1.f - latent_mask_) + latent_image_ * latent_mask_;
        return out;
    }

    // Wraps `run()` into the callback shape the sampler kernels call. The
    // returned function must not outlive the engine, its inner model, or the
    // referenced tensors.
    std::function<sd::guidance::GuiderOutput(sd::Tensor<float>&, float, int)>
    make_callback(const std::vector<float>& sigmas) const {
        return [this, sigmas](sd::Tensor<float>& x, float sigma, int step) -> sd::guidance::GuiderOutput {
            const int outer_step = std::abs(step) - 1;
            sd::guidance::GuiderOutput result;
            result.pred = run(x, sigma, sigmas, outer_step);
            return result;
        };
    }

private:
    // Overdamped (Gamma -> infinity) Langevin substep (comfy
    // `advance_time_overdamped`):
    //   dx = -A x dt + C dt + D dW_t   (C treated as constant over the substep)
    //   k  = (1 - exp(-A dt)) / A      (-> dt as A -> 0)
    //   k2 = (1 - exp(-2 A dt)) / (2 A)
    //   x  = exp(-A dt) x + k C + sqrt(D^2 k2) eps
    // `A` and `C` are latent-shaped masked tensors; `dtv` is the (also
    // masked, per-pixel) time step `dt` or `dt/2`.
    static sd::Tensor<float> advance_overdamped(const sd::Tensor<float>& xin,
                                                const sd::Tensor<float>& dtv,
                                                const sd::Tensor<float>& A,
                                                const sd::Tensor<float>& C,
                                                float D2,
                                                const std::shared_ptr<RNG>& rng) {
        const float eps = 1e-8f;
        // A > 0 over the valid domain (A = A_x*(1-m)+A_y*m with A_x, A_y >
        // 0); the clamp only guards a divide-by-zero if A were exactly 0.
        sd::Tensor<float> A_safe  = sd::ops::clamp(A, eps, std::numeric_limits<float>::max());
        sd::Tensor<float> A_dt    = A * dtv;
        sd::Tensor<float> exp_neg = sd::ops::exp(-A_dt);
        sd::Tensor<float> k       = (1.f - exp_neg) / A_safe;
        sd::Tensor<float> k2      = (1.f - sd::ops::exp(-2.f * A_dt)) / (2.f * A_safe);

        sd::Tensor<float> mean     = exp_neg * xin + k * C;
        sd::Tensor<float> var      = D2 * k2;
        sd::Tensor<float> eps_draw = sd::Tensor<float>::randn_like(xin, rng);
        return mean + eps_draw * sd::ops::sqrt(sd::ops::clamp(var, 0.f, std::numeric_limits<float>::max()));
    }

    LanPaintParams params_;
    LanPaintInnerModel inner_model_;
    std::shared_ptr<RNG> rng_;
    const sd::Tensor<float>& noise_;
    const sd::Tensor<float>& latent_image_;
    const sd::Tensor<float>& latent_mask_;
};

#endif  // __SD_RUNTIME_LANPAINT_HPP__
