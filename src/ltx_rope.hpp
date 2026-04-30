#ifndef __LTX_ROPE_HPP__
#define __LTX_ROPE_HPP__

#include <cmath>
#include <vector>
#include "ggml_extend.hpp"

namespace LTXRope {
    // Generate a log-spaced frequency grid from 1 to theta, scaled by pi/2.
    // Returns num_freqs = inner_dim / (2 * n_pos_dims) values.
    //
    // Python reference: generate_freq_grid_pytorch in ltx_core/model/transformer/rope.py.
    // We mirror the fp32 linspace path byte-exactly: torch.linspace(0., 1., N, fp32)
    // produces indices computed as `i * (1/(N-1))` in fp32 (start + step*i), so we
    // replicate that order of operations rather than `(double)i / (N-1)` which
    // differs by ~1 ULP at the tail. That 1-ULP freq drift becomes ~3-5 ULPs in
    // the freq value and ~5e-2 cos/sin error once the angle hits 1e5 radians at
    // T=8. `pow(theta, v)` is then computed in fp32 (std::powf) to match.
    __STATIC_INLINE__ std::vector<float> generate_freq_grid(float theta,
                                                            int n_pos_dims,
                                                            int inner_dim) {
        int n_elem    = 2 * n_pos_dims;
        int num_freqs = inner_dim / n_elem;
        std::vector<float> indices(num_freqs);
        // Compute in fp64 then cast. For the video DiT (3D RoPE, max_pos normalizes
        // to [0, 1]) fp32 would be fine, but the connector's 1D RoPE uses max_pos=[1]
        // so raw integer positions feed into the angle → arguments reach ~2e5 radians
        // at T=8. At that scale, fp32 libm `exp(t*log(theta))` drifts ~1 ULP in
        // the freq value, cascading to ~5e-2 cos/sin diffs vs the numpy-fp64 reference
        // used by the connector dumper (`double_precision_rope=True`). fp64 pow matches
        // numpy closely enough to land connector parity at ~2e-3 max_abs.
        constexpr double pi_half = 1.57079632679489661923;
        double theta_d           = static_cast<double>(theta);
        for (int i = 0; i < num_freqs; ++i) {
            double t   = num_freqs == 1 ? 0.0 : static_cast<double>(i) / (num_freqs - 1);
            indices[i] = static_cast<float>(std::pow(theta_d, t) * pi_half);
        }
        return indices;
    }

    // Build a 3D indices grid for a video latent of shape (F, H, W).
    //
    // Mirrors the real LTX-2 pipeline: VideoLatentTools.create_initial_state ->
    // get_patch_grid_bounds -> get_pixel_coords (ltx_core/components/patchifiers.py and
    // ltx_core/tools.py). Per-axis behaviour:
    //   latent_coords[axis] = [f, f+1]           (integer latent indices per patch)
    //   pixel_coords[axis]  = latent_coords * scale_factors[axis]
    //   if causal_fix:      pixel_coords[0] = clamp(pixel_coords[0] + 1 - scale_factors[0], 0, +)
    //   positions[0]       /= fps                 (temporal axis only)
    //   if use_middle_indices_grid: pos = midpoint(start, end); else pos = start
    //
    // Defaults ({1,1,1}, causal_fix=false, fps=1) preserve the parity-test flow, which
    // feeds the Python model the simplified (f, h, w) positions directly. Real inference
    // MUST pass scale_factors={8, 32, 32} and causal_fix=true (the LTX-2 VAE scale).
    //
    // Returns a 3×(F*H*W) matrix with layout [axis][token_idx].
    __STATIC_INLINE__ std::vector<std::vector<float>> gen_video_positions(int F,
                                                                          int H,
                                                                          int W,
                                                                          bool use_middle_indices_grid = true,
                                                                          float fps                    = 1.0f,
                                                                          const std::vector<int>& scale_factors = {1, 1, 1},
                                                                          bool causal_fix              = false) {
        GGML_ASSERT(fps > 0.0f);
        GGML_ASSERT(scale_factors.size() == 3);
        int total = F * H * W;
        std::vector<std::vector<float>> pos(3, std::vector<float>(total, 0.f));
        const float s0 = static_cast<float>(scale_factors[0]);
        const float s1 = static_cast<float>(scale_factors[1]);
        const float s2 = static_cast<float>(scale_factors[2]);
        for (int f = 0; f < F; ++f) {
            float t_s = static_cast<float>(f)     * s0;
            float t_e = static_cast<float>(f + 1) * s0;
            if (causal_fix) {
                const float shift = 1.f - s0;
                t_s = std::max(0.f, t_s + shift);
                t_e = std::max(0.f, t_e + shift);
            }
            t_s /= fps;
            t_e /= fps;
            for (int h = 0; h < H; ++h) {
                float h_s = static_cast<float>(h)     * s1;
                float h_e = static_cast<float>(h + 1) * s1;
                for (int w = 0; w < W; ++w) {
                    float w_s = static_cast<float>(w)     * s2;
                    float w_e = static_cast<float>(w + 1) * s2;
                    int idx   = (f * H + h) * W + w;
                    if (use_middle_indices_grid) {
                        pos[0][idx] = (t_s + t_e) * 0.5f;
                        pos[1][idx] = (h_s + h_e) * 0.5f;
                        pos[2][idx] = (w_s + w_e) * 0.5f;
                    } else {
                        pos[0][idx] = t_s;
                        pos[1][idx] = h_s;
                        pos[2][idx] = w_s;
                    }
                }
            }
        }
        return pos;
    }

    // Precompute interleaved cos/sin freqs for LTX-2 RoPE.
    // positions[axis][token]: fractional-ready float positions, size n_pos_dims * T.
    // max_pos: normalisation per axis, e.g. {20, 2048, 2048}.
    // Returns a packed [2, T, inner_dim] vector: slice [0] = cos, slice [1] = sin.
    __STATIC_INLINE__ std::vector<float> precompute_freqs_cis_interleaved(const std::vector<std::vector<float>>& positions,
                                                                          int inner_dim,
                                                                          float theta                     = 10000.f,
                                                                          const std::vector<int>& max_pos = {20, 2048, 2048}) {
        int n_pos_dims = static_cast<int>(positions.size());
        GGML_ASSERT(n_pos_dims > 0);
        GGML_ASSERT(static_cast<int>(max_pos.size()) == n_pos_dims);
        int T = static_cast<int>(positions[0].size());

        int n_elem    = 2 * n_pos_dims;
        int num_freqs = inner_dim / n_elem;
        int pad_size  = inner_dim - (num_freqs * n_pos_dims * 2);

        std::vector<float> freq_grid = generate_freq_grid(theta, n_pos_dims, inner_dim);  // [num_freqs]

        std::vector<float> pe(2 * T * inner_dim, 0.f);
        // Slice 0 (cos) starts at offset 0, slice 1 (sin) starts at T * inner_dim.
        size_t cos_off = 0;
        size_t sin_off = static_cast<size_t>(T) * inner_dim;

        // Initialise the pad region: cos = 1.0, sin = 0.0.
        for (int t = 0; t < T; ++t) {
            for (int i = 0; i < pad_size; ++i) {
                pe[cos_off + static_cast<size_t>(t) * inner_dim + i] = 1.f;
            }
        }

        for (int t = 0; t < T; ++t) {
            std::vector<float> frac_pos(n_pos_dims);
            for (int d = 0; d < n_pos_dims; ++d) {
                frac_pos[d] = positions[d][t] / static_cast<float>(max_pos[d]);
            }
            // Freq layout after flatten is [f * n_pos_dims + d], so pair index p = f*n_pos_dims + d.
            // After repeat_interleave(2), each pair p corresponds to slots (2p, 2p+1) in the [pad_size:] region.
            //
            // Note: compute cos/sin in double precision then cast to float. At high frequencies
            // (theta^1 * pi/2 ≈ 15708) times (2*t - 1), the angle reaches hundreds of thousands of
            // radians — fp32 argument reduction in std::cosf/sinf loses enough precision to drift
            // ~5e-2 from PyTorch's tensor-level cos/sin. Python's torch.cos does the reduction
            // against a more precise modulus internally (matching fp64 behavior closely enough).
            for (int f = 0; f < num_freqs; ++f) {
                for (int d = 0; d < n_pos_dims; ++d) {
                    double angle = static_cast<double>(freq_grid[f]) *
                                   (static_cast<double>(frac_pos[d]) * 2.0 - 1.0);
                    float c     = static_cast<float>(std::cos(angle));
                    float s     = static_cast<float>(std::sin(angle));
                    int pair_i  = f * n_pos_dims + d;
                    int slot0   = pad_size + 2 * pair_i;
                    int slot1   = pad_size + 2 * pair_i + 1;
                    pe[cos_off + static_cast<size_t>(t) * inner_dim + slot0] = c;
                    pe[cos_off + static_cast<size_t>(t) * inner_dim + slot1] = c;
                    pe[sin_off + static_cast<size_t>(t) * inner_dim + slot0] = s;
                    pe[sin_off + static_cast<size_t>(t) * inner_dim + slot1] = s;
                }
            }
        }
        return pe;
    }

    // Apply LTX-2 interleaved rotary embedding to x.
    // x: [inner_dim, T, B]  (ggml ne order; logical shape [B, T, inner_dim])
    // cos, sin: [inner_dim, T, 1]  (broadcast across batch)
    // Returns x rotated, same shape as x.
    __STATIC_INLINE__ ggml_tensor* apply_rotary_emb_interleaved(ggml_context* ctx,
                                                                ggml_tensor* x,
                                                                ggml_tensor* cos_freq,
                                                                ggml_tensor* sin_freq) {
        int64_t inner_dim = x->ne[0];
        int64_t T         = x->ne[1];
        int64_t B         = x->ne[2];
        GGML_ASSERT(inner_dim % 2 == 0);

        // Reshape to pairs: [2, inner_dim/2, T, B].
        auto x_pairs = ggml_reshape_4d(ctx, x, 2, inner_dim / 2, T, B);

        // Views: x_even (offset 0) and x_odd (offset nb[0]) each shape [1, inner_dim/2, T, B].
        auto x_even = ggml_view_4d(ctx, x_pairs, 1, inner_dim / 2, T, B,
                                   x_pairs->nb[1], x_pairs->nb[2], x_pairs->nb[3], 0);
        auto x_odd  = ggml_view_4d(ctx, x_pairs, 1, inner_dim / 2, T, B,
                                   x_pairs->nb[1], x_pairs->nb[2], x_pairs->nb[3], x_pairs->nb[0]);
        x_even      = ggml_cont(ctx, x_even);
        x_odd       = ggml_cont(ctx, x_odd);

        // Rotated pair (−x_odd, x_even) → concat along dim 0 → [2, inner_dim/2, T, B].
        auto neg_x_odd = ggml_scale(ctx, x_odd, -1.f);
        auto rotated   = ggml_concat(ctx, neg_x_odd, x_even, 0);
        rotated        = ggml_reshape_3d(ctx, rotated, inner_dim, T, B);

        // out = x * cos + rotated * sin
        auto out = ggml_add(ctx, ggml_mul(ctx, x, cos_freq), ggml_mul(ctx, rotated, sin_freq));
        return out;
    }

    // Precompute SPLIT cos/sin freqs for LTX-2.3 DiT. Python reference:
    // `precompute_freqs_cis(..., rope_type=LTXRopeType.SPLIT)`.
    //   - Unlike the interleaved variant, freqs are NOT repeat_interleaved; each of
    //     the inner_dim/2 frequencies is broadcast once across the corresponding
    //     position in the first AND second halves of head_dim.
    //   - cos/sin are reshaped to per-head: shape [B, T, H, head_dim/2].
    //   - We pack both into a single buffer of ne [head_dim/2, num_heads, T, 2]
    //     (slice 0 = cos, slice 1 = sin), matching the interleaved helper's
    //     single-buffer convention. split_pe_split() below slices that back.
    //
    // freqs flattened length is num_freqs * n_pos_dims; when it's less than
    // inner_dim/2, the leading (pad_size) slots are filled cos=1, sin=0, matching
    // Python's `split_freqs_cis`.
    __STATIC_INLINE__ std::vector<float> precompute_freqs_cis_split(const std::vector<std::vector<float>>& positions,
                                                                    int inner_dim,
                                                                    int num_heads,
                                                                    float theta                     = 10000.f,
                                                                    const std::vector<int>& max_pos = {20, 2048, 2048}) {
        int n_pos_dims = static_cast<int>(positions.size());
        GGML_ASSERT(n_pos_dims > 0);
        GGML_ASSERT(static_cast<int>(max_pos.size()) == n_pos_dims);
        GGML_ASSERT(inner_dim % (2 * num_heads) == 0);
        int T         = static_cast<int>(positions[0].size());
        int half_dim  = inner_dim / 2;   // per-token freq count
        int head_dim2 = half_dim / num_heads;  // per-head freq count

        int n_elem      = 2 * n_pos_dims;
        int num_freqs   = inner_dim / n_elem;
        int current     = num_freqs * n_pos_dims;  // pre-pad flat freq count
        int pad_size    = half_dim - current;
        GGML_ASSERT(pad_size >= 0);

        std::vector<float> freq_grid = generate_freq_grid(theta, n_pos_dims, inner_dim);

        // Output layout (ne): [head_dim/2, num_heads, T, 2].  Flat index:
        //   (slice=cos/sin)*T*num_heads*head_dim2 + t*num_heads*head_dim2 + h*head_dim2 + k
        std::vector<float> pe(2 * T * num_heads * head_dim2, 0.f);
        size_t cos_off = 0;
        size_t sin_off = static_cast<size_t>(T) * num_heads * head_dim2;

        // Pad region (first `pad_size` columns of the per-token freq vector): cos=1, sin=0.
        // Per-head reshape means pad_size slots at the start of the head-major flat
        // vector. Since cos/sin for a token are stored as [h=0 head_dim2, h=1 head_dim2, …],
        // the pad falls in the first pad_size consecutive positions across the head groups.
        for (int t = 0; t < T; ++t) {
            for (int p = 0; p < pad_size; ++p) {
                int h  = p / head_dim2;
                int k  = p % head_dim2;
                size_t dst = static_cast<size_t>(t) * num_heads * head_dim2 + h * head_dim2 + k;
                pe[cos_off + dst] = 1.f;
                pe[sin_off + dst] = 0.f;
            }
        }

        constexpr double pi_half = 1.57079632679489661923;
        (void)pi_half;
        for (int t = 0; t < T; ++t) {
            std::vector<float> frac_pos(n_pos_dims);
            for (int d = 0; d < n_pos_dims; ++d) {
                frac_pos[d] = positions[d][t] / static_cast<float>(max_pos[d]);
            }
            // Non-pad slots start at column `pad_size` in the flat per-token freq vector.
            // Python layout: freqs = (indices * (fractional*2-1)).transpose(-1,-2).flatten(2).
            // With indices shape [num_freqs] and fractional [n_pos_dims], after broadcast
            // and transpose the order is [f * n_pos_dims + d]. Slot index in the padded
            // per-token vector = pad_size + f*n_pos_dims + d.
            for (int f = 0; f < num_freqs; ++f) {
                for (int d = 0; d < n_pos_dims; ++d) {
                    double angle = static_cast<double>(freq_grid[f]) *
                                   (static_cast<double>(frac_pos[d]) * 2.0 - 1.0);
                    float c      = static_cast<float>(std::cos(angle));
                    float s      = static_cast<float>(std::sin(angle));
                    int flat_slot = pad_size + f * n_pos_dims + d;
                    int h  = flat_slot / head_dim2;
                    int k  = flat_slot % head_dim2;
                    size_t dst = static_cast<size_t>(t) * num_heads * head_dim2 + h * head_dim2 + k;
                    pe[cos_off + dst] = c;
                    pe[sin_off + dst] = s;
                }
            }
        }
        return pe;
    }

    // Split-half rotary embedding. Python: apply_split_rotary_emb.
    //   first_half  = x[..., 0:head_dim/2]
    //   second_half = x[..., head_dim/2:head_dim]
    //   out = concat(first*cos - second*sin, second*cos + first*sin, dim=last)
    // Operates per-head. x ne=[inner_dim, T, B]; pe tensors (cos/sin) ne=[head_dim/2, num_heads, T, 1].
    __STATIC_INLINE__ ggml_tensor* apply_rotary_emb_split(ggml_context* ctx,
                                                          ggml_tensor* x,
                                                          ggml_tensor* cos_freq,
                                                          ggml_tensor* sin_freq,
                                                          int num_heads) {
        int64_t inner_dim = x->ne[0];
        int64_t T         = x->ne[1];
        int64_t B         = x->ne[2];
        GGML_ASSERT(inner_dim % (2 * num_heads) == 0);
        int64_t head_dim  = inner_dim / num_heads;
        int64_t half      = head_dim / 2;

        // Reshape x [inner_dim, T, B] → [head_dim, num_heads, T, B], then split halves.
        auto x4 = ggml_reshape_4d(ctx, x, head_dim, num_heads, T, B);

        // first_half view: offset 0, shape [half, num_heads, T, B].
        auto first  = ggml_view_4d(ctx, x4, half, num_heads, T, B,
                                   x4->nb[1], x4->nb[2], x4->nb[3], 0);
        // second_half view: offset = half * sizeof(el).
        auto second = ggml_view_4d(ctx, x4, half, num_heads, T, B,
                                   x4->nb[1], x4->nb[2], x4->nb[3], half * x4->nb[0]);
        first  = ggml_cont(ctx, first);
        second = ggml_cont(ctx, second);

        // cos/sin ne [half, num_heads, T, 1] broadcast on B axis with first/second [half, num_heads, T, B].
        auto first_out  = ggml_sub(ctx, ggml_mul(ctx, first, cos_freq),
                                        ggml_mul(ctx, second, sin_freq));
        auto second_out = ggml_add(ctx, ggml_mul(ctx, second, cos_freq),
                                        ggml_mul(ctx, first,  sin_freq));

        // Re-concat along dim 0 (head_dim) → [head_dim, num_heads, T, B].
        auto joined = ggml_concat(ctx, first_out, second_out, 0);
        joined      = ggml_reshape_3d(ctx, joined, inner_dim, T, B);
        return joined;
    }

    // Slice a packed split pe buffer of ne [half, num_heads, T, 2] into cos (slice 0)
    // and sin (slice 1) views, each ne=[half, num_heads, T, 1].
    __STATIC_INLINE__ std::pair<ggml_tensor*, ggml_tensor*> split_pe_split(ggml_context* ctx, ggml_tensor* pe) {
        int64_t half      = pe->ne[0];
        int64_t num_heads = pe->ne[1];
        int64_t T         = pe->ne[2];
        auto cos_freq     = ggml_view_4d(ctx, pe, half, num_heads, T, 1,
                                         pe->nb[1], pe->nb[2], pe->nb[3], 0);
        auto sin_freq     = ggml_view_4d(ctx, pe, half, num_heads, T, 1,
                                         pe->nb[1], pe->nb[2], pe->nb[3], pe->nb[3]);
        return {cos_freq, sin_freq};
    }

    // Convenience: split a packed [2, T, inner_dim] pe tensor (slice 0 = cos, slice 1 = sin)
    // into two views usable as cos/sin operands.
    __STATIC_INLINE__ std::pair<ggml_tensor*, ggml_tensor*> split_pe(ggml_context* ctx, ggml_tensor* pe) {
        // pe: [inner_dim, T, 2] in ggml ne order.
        int64_t inner_dim = pe->ne[0];
        int64_t T         = pe->ne[1];
        auto cos_freq     = ggml_view_3d(ctx, pe, inner_dim, T, 1, pe->nb[1], pe->nb[2], 0);
        auto sin_freq     = ggml_view_3d(ctx, pe, inner_dim, T, 1, pe->nb[1], pe->nb[2], pe->nb[2]);
        return {cos_freq, sin_freq};
    }
};  // namespace LTXRope

#endif  // __LTX_ROPE_HPP__
