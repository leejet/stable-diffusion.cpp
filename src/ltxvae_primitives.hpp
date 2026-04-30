#ifndef __LTXVAE_PRIMITIVES_HPP__
#define __LTXVAE_PRIMITIVES_HPP__

#include "ggml.h"

// Space-to-depth / depth-to-space helpers for the LTX-2 VAE.
//
// The VAE's `SpaceToDepthDownsample` and `DepthToSpaceUpsample` blocks compress
// or expand one or more of the (T, H, W) axes into/out of the channel axis. In
// einops notation (with B=1 elided):
//
//   rearrange(x, "c (t p1) (h p2) (w p3) -> (c p1 p2 p3) t h w", ...)   # space-to-depth
//   rearrange(x, "(c p1 p2 p3) t h w -> c (t p1) (h p2) (w p3)", ...)   # depth-to-space
//
// The einops grouping "(c p1 p2 p3)" puts p3 innermost (fastest-varying) within
// the merged channel axis, so c_new = c*p1*p2*p3 + i1*p2*p3 + i2*p3 + i3.
//
// GGML caps tensors at 4-D, which prevents a single reshape from representing the
// natural 5-D/6-D intermediate. We achieve the same result by folding the three
// strided axes ONE AT A TIME, composing three 4-D rearranges. The fold order
// matters: because each single-axis fold puts the just-folded factor innermost
// within the merged channel axis, folding in the order T→H→W produces p3 as the
// innermost factor in the final output — matching einops "(c p1 p2 p3)".
//
// Convention: all tensors use GGML ne=[W, H, T, C] (B=1 collapsed). A "factor"
// of 1 is a no-op; single-axis folds require the target axis to be divisible
// by factor.
//
// The primitives are verified byte-exact against PyTorch einops in the
// standalone test sd-s2d-primitives-test.

namespace LTXVAE {

// ---------- SpaceToDepth ----------

inline ggml_tensor* space_to_depth_axisW(ggml_context* ctx, ggml_tensor* x, int factor) {
    int64_t W = x->ne[0], H = x->ne[1], T = x->ne[2], C = x->ne[3];
    GGML_ASSERT(W % factor == 0);
    int64_t W_out = W / factor;
    // Split innermost axis into [factor (innermost), W_out]. Merge H,T to stay 4D.
    auto y = ggml_reshape_4d(ctx, x, factor, W_out, H * T, C);
    // Move "factor" from axis 0 to axis 2 (adjacent to C).
    // ggml_permute(a, p0, p1, p2, p3) says "old axis i goes to new position p_i".
    // Here old→new: 0→2, 1→0, 2→1, 3→3.
    y = ggml_cont(ctx, ggml_permute(ctx, y, 2, 0, 1, 3));  // ne=[W_out, H*T, factor, C]
    // Merge (factor, C) with factor innermost of the new channel axis, matching einops (c p3).
    y = ggml_reshape_4d(ctx, y, W_out, H, T, C * factor);
    return y;
}

inline ggml_tensor* space_to_depth_axisH(ggml_context* ctx, ggml_tensor* x, int factor) {
    int64_t W = x->ne[0], H = x->ne[1], T = x->ne[2], C = x->ne[3];
    GGML_ASSERT(H % factor == 0);
    int64_t H_out = H / factor;
    auto y = ggml_reshape_4d(ctx, x, W, factor, H_out * T, C);
    y = ggml_cont(ctx, ggml_permute(ctx, y, 0, 2, 1, 3));  // ne=[W, H*T, factor, C]
    y = ggml_reshape_4d(ctx, y, W, H_out, T, C * factor);
    return y;
}

inline ggml_tensor* space_to_depth_axisT(ggml_context* ctx, ggml_tensor* x, int factor) {
    int64_t W = x->ne[0], H = x->ne[1], T = x->ne[2], C = x->ne[3];
    GGML_ASSERT(T % factor == 0);
    int64_t T_out = T / factor;
    auto y = ggml_reshape_4d(ctx, x, W * H, factor, T_out, C);
    y = ggml_cont(ctx, ggml_permute(ctx, y, 0, 2, 1, 3));  // ne=[W*H, T, factor, C]
    y = ggml_reshape_4d(ctx, y, W, H, T_out, C * factor);
    return y;
}

// Compose: fold T first (so p1 ends up outer), then H (p2), then W (p3 innermost)
// — matching einops "(c p1 p2 p3)" channel ordering.
inline ggml_tensor* space_to_depth(ggml_context* ctx, ggml_tensor* x,
                                   int p1, int p2, int p3) {
    if (p1 > 1) x = space_to_depth_axisT(ctx, x, p1);
    if (p2 > 1) x = space_to_depth_axisH(ctx, x, p2);
    if (p3 > 1) x = space_to_depth_axisW(ctx, x, p3);
    return x;
}

// ---------- DepthToSpace (inverse) ----------
//
// Each single-axis depth-to-space splits the last axis (C_in = C_out * factor)
// with factor innermost, moves factor to the strided spatial axis, then merges.
// To invert space_to_depth's T→H→W fold order, we unfold in reverse: W→H→T.

inline ggml_tensor* depth_to_space_axisW(ggml_context* ctx, ggml_tensor* x, int factor) {
    int64_t W = x->ne[0], H = x->ne[1], T = x->ne[2], C = x->ne[3];
    GGML_ASSERT(C % factor == 0);
    int64_t C_out = C / factor;
    // Split last axis into [factor (innermost), C_out].
    auto y = ggml_reshape_4d(ctx, x, W, H * T, factor, C_out);
    // Inverse of the S2D-axisW permute (2,0,1,3). Inverse of that map is (1,2,0,3):
    //   old 0→new 1, old 1→new 2, old 2→new 0, old 3→new 3.
    y = ggml_cont(ctx, ggml_permute(ctx, y, 1, 2, 0, 3));  // ne=[factor, W, H*T, C_out]
    y = ggml_reshape_4d(ctx, y, W * factor, H, T, C_out);
    return y;
}

inline ggml_tensor* depth_to_space_axisH(ggml_context* ctx, ggml_tensor* x, int factor) {
    int64_t W = x->ne[0], H = x->ne[1], T = x->ne[2], C = x->ne[3];
    GGML_ASSERT(C % factor == 0);
    int64_t C_out = C / factor;
    auto y = ggml_reshape_4d(ctx, x, W, H * T, factor, C_out);
    // Inverse of S2D-axisH's (0,2,1,3) is itself (0,2,1,3).
    y = ggml_cont(ctx, ggml_permute(ctx, y, 0, 2, 1, 3));  // ne=[W, factor, H*T, C_out]
    y = ggml_reshape_4d(ctx, y, W, H * factor, T, C_out);
    return y;
}

inline ggml_tensor* depth_to_space_axisT(ggml_context* ctx, ggml_tensor* x, int factor) {
    int64_t W = x->ne[0], H = x->ne[1], T = x->ne[2], C = x->ne[3];
    GGML_ASSERT(C % factor == 0);
    int64_t C_out = C / factor;
    auto y = ggml_reshape_4d(ctx, x, W * H, T, factor, C_out);
    y = ggml_cont(ctx, ggml_permute(ctx, y, 0, 2, 1, 3));  // ne=[W*H, factor, T, C_out]
    y = ggml_reshape_4d(ctx, y, W, H, T * factor, C_out);
    return y;
}

// Inverse of space_to_depth: unfold in reverse order (W first, then H, then T)
// because S2D folded T first, H, W.
inline ggml_tensor* depth_to_space(ggml_context* ctx, ggml_tensor* x,
                                   int p1, int p2, int p3) {
    if (p3 > 1) x = depth_to_space_axisW(ctx, x, p3);
    if (p2 > 1) x = depth_to_space_axisH(ctx, x, p2);
    if (p1 > 1) x = depth_to_space_axisT(ctx, x, p1);
    return x;
}

// ---------- patchify / unpatchify ----------
//
// The VAE's patchify op uses a DIFFERENT channel ordering from the Downsample/Upsample
// blocks: einops `"b c (f p) (h q) (w r) -> b (c p r q) f h w"` — innermost within the
// merged channel axis is q (H-patch), NOT p3/W as elsewhere. To match, we fold in the
// order T (p), W (r), H (q) — last fold ends up innermost.

inline ggml_tensor* patchify(ggml_context* ctx, ggml_tensor* x, int pt, int ph, int pw) {
    if (pt > 1) x = space_to_depth_axisT(ctx, x, pt);
    if (pw > 1) x = space_to_depth_axisW(ctx, x, pw);
    if (ph > 1) x = space_to_depth_axisH(ctx, x, ph);
    return x;
}

inline ggml_tensor* unpatchify(ggml_context* ctx, ggml_tensor* x, int pt, int ph, int pw) {
    if (ph > 1) x = depth_to_space_axisH(ctx, x, ph);
    if (pw > 1) x = depth_to_space_axisW(ctx, x, pw);
    if (pt > 1) x = depth_to_space_axisT(ctx, x, pt);
    return x;
}

// ---------- PixelNorm ----------
//
// Python (ltx_core.model.common.normalization.PixelNorm, dim=1):
//   y = x / sqrt(mean(x^2, dim=1, keepdim=True) + eps)
// PyTorch dim=1 is the channel axis. In our GGML layout ne=[W, H, T, C] that's
// ne[3] (outermost). ggml_rms_norm normalizes along ne[0] (innermost), so we
// permute C to innermost, rms-normalize, then permute back.
//
// This has NO learnable parameters — the Python PixelNorm is parameter-free.

inline ggml_tensor* pixel_norm(ggml_context* ctx, ggml_tensor* x, float eps) {
    int64_t W = x->ne[0], H = x->ne[1], T = x->ne[2], C = x->ne[3];
    // Move C to innermost. old→new: 0→1 (W to pos 1), 1→2 (H to 2), 2→3 (T to 3), 3→0 (C to 0).
    auto y = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 3, 0));  // ne=[C, W, H, T]
    y      = ggml_rms_norm(ctx, y, eps);                         // normalize along ne[0]=C
    // Permute back: C to outermost. old→new: 0→3, 1→0, 2→1, 3→2.
    y = ggml_cont(ctx, ggml_permute(ctx, y, 3, 0, 1, 2));        // ne=[W, H, T, C]
    (void)W; (void)H; (void)T; (void)C;
    return y;
}

// ---------- PerChannelStatistics ----------
//
// Python: buffers `mean-of-means` [C] and `std-of-means` [C].
//   normalize(x)    = (x - mean) / std
//   un_normalize(x) = x * std + mean
// In GGML with ne=[W, H, T, C] and a 1D buffer of shape [C] (ne=[C, 1, 1, 1]),
// we broadcast over W*H*T by using the asymmetric-broadcast ggml_add/mul:
// ggml_mul(a, b) requires a->ne[i] % b->ne[i] == 0, so we pass x as `a` and the
// [C] buffer reshaped to ne=[1, 1, 1, C] as `b` — same outermost-axis shape.

inline ggml_tensor* pcs_normalize(ggml_context* ctx, ggml_tensor* x,
                                  ggml_tensor* mean_of_means,
                                  ggml_tensor* std_of_means) {
    int64_t C = x->ne[3];
    // Reshape both buffers to ne=[1, 1, 1, C] so they broadcast along W/H/T.
    auto mu    = ggml_reshape_4d(ctx, mean_of_means, 1, 1, 1, C);
    auto sigma = ggml_reshape_4d(ctx, std_of_means,  1, 1, 1, C);
    // (x - mu) / sigma = (x - mu) * (1/sigma). Compute the reciprocal by dividing.
    // ggml doesn't have a direct div by tensor; emulate with ggml_div if available,
    // else compute inv_sigma on the host. Since sigma is a loaded buffer (constant
    // at inference), the cheapest is to do: x_shifted = x - mu; x_norm = x_shifted / sigma.
    auto x_shifted = ggml_sub(ctx, x, mu);
    auto x_norm    = ggml_div(ctx, x_shifted, sigma);
    return x_norm;
}

inline ggml_tensor* pcs_unnormalize(ggml_context* ctx, ggml_tensor* x,
                                    ggml_tensor* mean_of_means,
                                    ggml_tensor* std_of_means) {
    int64_t C = x->ne[3];
    auto mu    = ggml_reshape_4d(ctx, mean_of_means, 1, 1, 1, C);
    auto sigma = ggml_reshape_4d(ctx, std_of_means,  1, 1, 1, C);
    auto y = ggml_mul(ctx, x, sigma);
    y      = ggml_add(ctx, y, mu);
    return y;
}

}  // namespace LTXVAE

#endif
