#ifndef __LTXVAE_HPP__
#define __LTXVAE_HPP__

#include "common_block.hpp"
#include "ltxv.hpp"              // CausalConv3d
#include "ltxvae_primitives.hpp" // space/depth, pixel_norm, pcs_*
#include "vae.hpp"               // VAE base class

// LTX-2 video VAE. Companion to src/ltxvae_primitives.hpp (pure ggml ops) —
// this file adds the parameterized composition blocks (ResnetBlock3D,
// UNetMidBlock3D, SpaceToDepthDownsample, DepthToSpaceUpsample) and the
// VideoEncoder / VideoDecoder top-levels.
//
// Tensor convention throughout: B=1 collapsed; ggml ne=[W, H, T, C].
// Weight naming mirrors the Python reference verbatim — see
// `/tmp/vae_ref/tensor_names.txt` for the canonical prefix layout.

namespace LTXVAE {

    // ---------- TimestepEmbedder ----------
    //
    // PixArtAlphaCombinedTimestepSizeEmbeddings with size_emb_dim=0. Python
    // structure: `.timestep_embedder.linear_{1,2}`. Sinusoidal projection into
    // TIME_PROJ_DIM (256) fed to a two-Linear MLP with SiLU between.

    struct TimestepEmbedder : public GGMLBlock {
    protected:
        int embedding_dim = 0;
        static constexpr int TIME_PROJ_DIM = 256;

    public:
        TimestepEmbedder() = default;
        TimestepEmbedder(int embedding_dim) : embedding_dim(embedding_dim) {
            blocks["timestep_embedder.linear_1"] = std::make_shared<Linear>(TIME_PROJ_DIM, embedding_dim, true);
            blocks["timestep_embedder.linear_2"] = std::make_shared<Linear>(embedding_dim, embedding_dim, true);
        }

        // timestep: ne=[B]. Returns ne=[embedding_dim, B].
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* timestep) {
            auto l1 = std::dynamic_pointer_cast<Linear>(blocks["timestep_embedder.linear_1"]);
            auto l2 = std::dynamic_pointer_cast<Linear>(blocks["timestep_embedder.linear_2"]);
            auto proj = ggml_ext_timestep_embedding(ctx->ggml_ctx, timestep, TIME_PROJ_DIM, 10000, 1.0f);
            auto h    = l1->forward(ctx, proj);
            h         = ggml_silu_inplace(ctx->ggml_ctx, h);
            return l2->forward(ctx, h);
        }
    };

    // ---------- ResnetBlock3D ----------
    //
    // Python forward (when timestep_conditioning=True):
    //   h = norm1(x)  [PixelNorm]
    //   ada  = scale_shift_table + time_embed.reshape(B, 4, in_channels, 1,1,1)
    //   shift1, scale1, shift2, scale2 = ada.unbind(dim=1)
    //   h = h * (1 + scale1) + shift1
    //   h = silu(h); h = conv1(h)
    //   h = norm2(h); h = h * (1 + scale2) + shift2
    //   h = silu(h); h = conv2(h)
    //   return input + h
    //
    // When in_channels != out_channels, the skip path goes through
    // norm3 = GroupNorm(num_groups=1, ...) + conv_shortcut (1×1×1 Conv3d).
    // Our parity config keeps in==out, so we hard-disable that path until
    // we land a use case that needs it.
    //
    // inject_noise is not yet supported (would require a seeded randn in ggml).

    struct ResnetBlock3D : public GGMLBlock {
    protected:
        int in_channels = 0;
        int out_channels = 0;
        bool timestep_conditioning = false;
        bool has_shortcut = false;
        float eps = 1e-6f;

        void init_params(ggml_context* ctx, const String2TensorStorage&, const std::string /*prefix*/ = "") override {
            if (timestep_conditioning) {
                // Python ne: [4, in_channels] → GGML ne [in_channels, 4].
                params["scale_shift_table"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_channels, 4);
            }
        }

    public:
        ResnetBlock3D() = default;
        ResnetBlock3D(int in_ch, int out_ch, bool timestep_cond, float eps_ = 1e-8f,
                      LTXV::SpatialPadding pad = LTXV::SpatialPadding::ZEROS)
            : in_channels(in_ch),
              out_channels(out_ch),
              timestep_conditioning(timestep_cond),
              has_shortcut(in_ch != out_ch),
              eps(eps_) {
            blocks["conv1"] = std::make_shared<LTXV::CausalConv3d>(
                in_ch, out_ch, 3, std::tuple<int,int,int>{1,1,1}, 1, true, pad);
            blocks["conv2"] = std::make_shared<LTXV::CausalConv3d>(
                out_ch, out_ch, 3, std::tuple<int,int,int>{1,1,1}, 1, true, pad);
            if (has_shortcut) {
                GGML_ABORT("ResnetBlock3D with in != out not yet implemented (norm3 + conv_shortcut)");
            }
        }

        // x: ne=[W, H, T, C_in]. time_embed (optional): ne=[4*in_channels, B=1].
        // `causal` propagates to the inner CausalConv3d.forward calls.
        // If traces is non-null, pushes intermediates in order:
        //   0 post_norm1, 1 shift1, 2 scale1, 3 post_adaln1, 4 post_conv1,
        //   5 post_norm2, 6 shift2, 7 scale2, 8 post_adaln2, 9 post_conv2, 10 final.
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* time_embed = nullptr,
                             std::vector<ggml_tensor*>* traces = nullptr, bool causal = true) {
            auto conv1 = std::dynamic_pointer_cast<LTXV::CausalConv3d>(blocks["conv1"]);
            auto conv2 = std::dynamic_pointer_cast<LTXV::CausalConv3d>(blocks["conv2"]);

            auto input = x;
            auto h = pixel_norm(ctx->ggml_ctx, x, eps);
            if (traces) traces->push_back(ggml_cont(ctx->ggml_ctx, h));

            ggml_tensor *shift1 = nullptr, *scale1 = nullptr;
            ggml_tensor *shift2 = nullptr, *scale2 = nullptr;
            if (timestep_conditioning) {
                GGML_ASSERT(time_embed != nullptr);
                auto sst = params["scale_shift_table"];                       // ne [in_channels, 4]
                // time_embed has ne [4*in_channels, B=1]. Reshape to ne [in_channels, 4] (implicit B=1).
                auto te  = ggml_reshape_2d(ctx->ggml_ctx, time_embed, in_channels, 4);
                auto ada = ggml_add(ctx->ggml_ctx, te, sst);                   // [in_channels, 4]

                shift1 = ggml_ext_slice(ctx->ggml_ctx, ada, 1, 0, 1);
                scale1 = ggml_ext_slice(ctx->ggml_ctx, ada, 1, 1, 2);
                shift2 = ggml_ext_slice(ctx->ggml_ctx, ada, 1, 2, 3);
                scale2 = ggml_ext_slice(ctx->ggml_ctx, ada, 1, 3, 4);
                if (traces) {
                    traces->push_back(ggml_cont(ctx->ggml_ctx, shift1));
                    traces->push_back(ggml_cont(ctx->ggml_ctx, scale1));
                }
                // Reshape happens below; the apply also happens below.
                // Reshape each [in_channels, 1] → [1, 1, 1, in_channels] so they broadcast
                // over (W, H, T) when added/multiplied with h [W, H, T, in_channels].
                shift1 = ggml_reshape_4d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, shift1), 1, 1, 1, in_channels);
                scale1 = ggml_reshape_4d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, scale1), 1, 1, 1, in_channels);
                shift2 = ggml_reshape_4d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, shift2), 1, 1, 1, in_channels);
                scale2 = ggml_reshape_4d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, scale2), 1, 1, 1, in_channels);

                auto h_scaled = ggml_mul(ctx->ggml_ctx, h, scale1);
                h = ggml_add(ctx->ggml_ctx, h, h_scaled);
                h = ggml_add(ctx->ggml_ctx, h, shift1);
                if (traces) traces->push_back(ggml_cont(ctx->ggml_ctx, h));
            }

            h = ggml_silu(ctx->ggml_ctx, h);
            h = conv1->forward(ctx, h, causal);
            if (traces) traces->push_back(ggml_cont(ctx->ggml_ctx, h));

            h = pixel_norm(ctx->ggml_ctx, h, eps);
            if (traces) traces->push_back(ggml_cont(ctx->ggml_ctx, h));

            if (timestep_conditioning) {
                auto h_scaled = ggml_mul(ctx->ggml_ctx, h, scale2);
                h = ggml_add(ctx->ggml_ctx, h, h_scaled);
                h = ggml_add(ctx->ggml_ctx, h, shift2);
            }

            h = ggml_silu(ctx->ggml_ctx, h);
            h = conv2->forward(ctx, h, causal);

            // in_channels == out_channels so skip is Identity (the `has_shortcut` path aborts above).
            return ggml_add(ctx->ggml_ctx, h, input);
        }
    };

    // ---------- UNetMidBlock3D ----------

    struct UNetMidBlock3D : public GGMLBlock {
    protected:
        int in_channels = 0;
        int num_layers = 0;
        bool timestep_conditioning = false;

    public:
        UNetMidBlock3D() = default;
        UNetMidBlock3D(int in_ch, int num_layers, bool timestep_cond,
                       LTXV::SpatialPadding pad = LTXV::SpatialPadding::ZEROS)
            : in_channels(in_ch), num_layers(num_layers), timestep_conditioning(timestep_cond) {
            for (int i = 0; i < num_layers; i++) {
                blocks["res_blocks." + std::to_string(i)] = std::make_shared<ResnetBlock3D>(in_ch, in_ch, timestep_cond, 1e-8f, pad);
            }
            if (timestep_cond) {
                blocks["time_embedder"] = std::make_shared<TimestepEmbedder>(in_ch * 4);
            }
        }

        // timestep: ne=[B=1] if conditioning enabled, else null.
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* timestep = nullptr,
                             std::vector<ggml_tensor*>* traces = nullptr, bool causal = true) {
            ggml_tensor* time_embed = nullptr;
            if (timestep_conditioning) {
                GGML_ASSERT(timestep != nullptr);
                auto te = std::dynamic_pointer_cast<TimestepEmbedder>(blocks["time_embedder"]);
                time_embed = te->forward(ctx, timestep);  // ne=[4*in_channels, 1]
                if (traces) traces->push_back(ggml_cont(ctx->ggml_ctx, time_embed));
            }
            for (int i = 0; i < num_layers; i++) {
                auto res = std::dynamic_pointer_cast<ResnetBlock3D>(blocks["res_blocks." + std::to_string(i)]);
                x = res->forward(ctx, x, time_embed, traces, causal);
            }
            return x;
        }
    };

    // ---------- SpaceToDepthDownsample (encoder) ----------
    //
    // Python forward:
    //   if stride[0]==2: x = cat([x[:,:,:1], x], dim=2)  # duplicate first frame
    //   x_in = rearrange(x, "b c (d p1)(h p2)(w p3) -> b (c p1 p2 p3) d h w", ...)
    //   x_in = rearrange(x_in, "b (c g) d h w -> b c g d h w", g=group_size).mean(dim=2)
    //   x = self.conv(x, causal); x = rearrange(x, ...s2d...); return x + x_in

    struct SpaceToDepthDownsample : public GGMLBlock {
    protected:
        int in_channels = 0;
        int out_channels = 0;
        int p1 = 1, p2 = 1, p3 = 1;
        int group_size = 1;

        // Helper: collapse group-size consecutive channels via mean along axis 2
        // (after reshaping [W,H,T,C_exp] → [W*H, T, g, C_new]).
        ggml_tensor* group_mean_channel(ggml_context* ctx, ggml_tensor* x, int g) const {
            if (g == 1) return x;
            int64_t W = x->ne[0], H = x->ne[1], T = x->ne[2], Cexp = x->ne[3];
            GGML_ASSERT(Cexp % g == 0);
            int64_t C_new = Cexp / g;
            // Reshape: merge W,H; split Cexp into [g, C_new] (g innermost-of-that-group,
            // matching einops "(c g)" with g innermost).
            auto y = ggml_reshape_4d(ctx, x, W * H, T, g, C_new);
            // Move g to innermost (axis 0) for ggml_mean.
            y = ggml_cont(ctx, ggml_permute(ctx, y, 1, 2, 0, 3));  // ne=[g, W*H, T, C_new]
            y = ggml_mean(ctx, y);                                  // ne=[1, W*H, T, C_new]
            // Permute back & reshape to [W, H, T, C_new].
            y = ggml_cont(ctx, ggml_permute(ctx, y, 3, 0, 1, 2));  // ne=[W*H, T, C_new, 1]
            y = ggml_reshape_4d(ctx, y, W, H, T, C_new);
            return y;
        }

    public:
        SpaceToDepthDownsample() = default;
        SpaceToDepthDownsample(int in_ch, int out_ch, std::tuple<int,int,int> stride)
            : in_channels(in_ch), out_channels(out_ch),
              p1(std::get<0>(stride)), p2(std::get<1>(stride)), p3(std::get<2>(stride)) {
            int prod = p1 * p2 * p3;
            GGML_ASSERT((in_ch * prod) % out_ch == 0);
            group_size = in_ch * prod / out_ch;
            GGML_ASSERT(out_ch % prod == 0);
            blocks["conv"] = std::make_shared<LTXV::CausalConv3d>(in_ch, out_ch / prod, 3);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, bool causal = true) {
            auto conv = std::dynamic_pointer_cast<LTXV::CausalConv3d>(blocks["conv"]);

            // Duplicate first frame if temporal stride is 2.
            if (p1 == 2) {
                auto first = ggml_view_4d(ctx->ggml_ctx, x,
                                          x->ne[0], x->ne[1], 1, x->ne[3],
                                          x->nb[1], x->nb[2], x->nb[3], 0);
                first = ggml_cont(ctx->ggml_ctx, first);
                x = ggml_concat(ctx->ggml_ctx, first, x, 2);  // prepend along T
            }

            // Skip: s2d → group-mean.
            auto x_in = space_to_depth(ctx->ggml_ctx, x, p1, p2, p3);
            x_in = group_mean_channel(ctx->ggml_ctx, x_in, group_size);

            // Main: conv (preserves T because of causal padding, stride=1), then s2d.
            auto y = conv->forward(ctx, x, causal);
            y = space_to_depth(ctx->ggml_ctx, y, p1, p2, p3);

            return ggml_add(ctx->ggml_ctx, y, x_in);
        }
    };

    // ---------- DepthToSpaceUpsample (decoder) ----------
    //
    // For the parity test we only need residual=False (compress_time, compress_space).
    // `compress_all` blocks with residual=True have a repeat-based skip path; we'll
    // add that when a decoder config needs it.

    struct DepthToSpaceUpsample : public GGMLBlock {
    protected:
        int in_channels = 0;
        int p1 = 1, p2 = 1, p3 = 1;
        int reduction_factor = 1;

    public:
        DepthToSpaceUpsample() = default;
        DepthToSpaceUpsample(int in_ch, std::tuple<int,int,int> stride, int reduction_factor = 1,
                             LTXV::SpatialPadding pad = LTXV::SpatialPadding::ZEROS)
            : in_channels(in_ch),
              p1(std::get<0>(stride)), p2(std::get<1>(stride)), p3(std::get<2>(stride)),
              reduction_factor(reduction_factor) {
            int prod = p1 * p2 * p3;
            int conv_out = prod * in_ch / reduction_factor;
            blocks["conv"] = std::make_shared<LTXV::CausalConv3d>(
                in_ch, conv_out, 3, std::tuple<int,int,int>{1,1,1}, 1, true, pad);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, bool causal = true) {
            auto conv = std::dynamic_pointer_cast<LTXV::CausalConv3d>(blocks["conv"]);
            x = conv->forward(ctx, x, causal);
            x = depth_to_space(ctx->ggml_ctx, x, p1, p2, p3);
            if (p1 == 2) {
                // Drop first frame along T to match Python x[:, :, 1:, ...].
                int64_t W = x->ne[0], H = x->ne[1], T = x->ne[2], C = x->ne[3];
                auto sliced = ggml_view_4d(ctx->ggml_ctx, x,
                                           W, H, T - 1, C,
                                           x->nb[1], x->nb[2], x->nb[3], x->nb[2]);  // skip frame 0
                x = ggml_cont(ctx->ggml_ctx, sliced);
            }
            return x;
        }
    };

    // ---------- PerChannelStatistics wrapper ----------
    //
    // Python uses register_buffer("std-of-means", ...) and ("mean-of-means", ...) —
    // dashed names which don't appear elsewhere in this codebase. We register them
    // as tensors via init_params and carry the dashed names verbatim so loader
    // name matching finds them.

    struct PerChannelStatisticsBlock : public GGMLBlock {
    protected:
        int latent_channels = 0;

        void init_params(ggml_context* ctx, const String2TensorStorage&, const std::string /*prefix*/ = "") override {
            params["std-of-means"]  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, latent_channels);
            params["mean-of-means"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, latent_channels);
        }

    public:
        PerChannelStatisticsBlock() = default;
        explicit PerChannelStatisticsBlock(int latent_channels) : latent_channels(latent_channels) {}

        ggml_tensor* normalize(GGMLRunnerContext* ctx, ggml_tensor* x) {
            return pcs_normalize(ctx->ggml_ctx, x, params["mean-of-means"], params["std-of-means"]);
        }
        ggml_tensor* un_normalize(GGMLRunnerContext* ctx, ggml_tensor* x) {
            return pcs_unnormalize(ctx->ggml_ctx, x, params["mean-of-means"], params["std-of-means"]);
        }
    };

    // ---------- VideoEncoder ----------
    //
    // The encoder config is a list of (block_name, block_config) tuples — we keep
    // that shape in C++ via an EncoderBlockSpec. Only `res_x`, `compress_space_res`,
    // `compress_time_res`, `compress_all_res` are handled here; more variants can
    // be added as their use-cases land. `norm_layer` is pixel_norm only (group_norm
    // would require new primitives). `latent_log_var` is UNIFORM only.

    enum class EncoderBlockKind {
        RES_X,
        COMPRESS_SPACE_RES,   // stride=(1,2,2)
        COMPRESS_TIME_RES,    // stride=(2,1,1)
        COMPRESS_ALL_RES,     // stride=(2,2,2)
    };

    struct EncoderBlockSpec {
        EncoderBlockKind kind;
        int num_layers = 1;  // used for RES_X
        int multiplier = 2;  // used for compress_*_res
    };

    struct VideoEncoder : public GGMLBlock {
    protected:
        int in_channels = 3;
        int latent_channels = 128;
        int patch_size = 4;
        std::vector<EncoderBlockSpec> encoder_blocks;
        float eps = 1e-6f;

    public:
        VideoEncoder() = default;
        VideoEncoder(int in_ch, int latent_ch, int patch,
                     const std::vector<EncoderBlockSpec>& enc_blocks)
            : in_channels(in_ch), latent_channels(latent_ch), patch_size(patch),
              encoder_blocks(enc_blocks) {
            int feature_ch = latent_ch;
            int cur_in     = in_ch * patch * patch;   // after patchify

            blocks["conv_in"] = std::make_shared<LTXV::CausalConv3d>(cur_in, feature_ch, 3);

            int cur_c = feature_ch;
            for (size_t i = 0; i < encoder_blocks.size(); ++i) {
                const auto& b = encoder_blocks[i];
                std::string key = "down_blocks." + std::to_string(i);
                switch (b.kind) {
                    case EncoderBlockKind::RES_X:
                        blocks[key] = std::make_shared<UNetMidBlock3D>(cur_c, b.num_layers, /*timestep_cond=*/false);
                        break;
                    case EncoderBlockKind::COMPRESS_SPACE_RES:
                        blocks[key] = std::make_shared<SpaceToDepthDownsample>(cur_c, cur_c * b.multiplier, std::tuple<int,int,int>{1,2,2});
                        cur_c *= b.multiplier;
                        break;
                    case EncoderBlockKind::COMPRESS_TIME_RES:
                        blocks[key] = std::make_shared<SpaceToDepthDownsample>(cur_c, cur_c * b.multiplier, std::tuple<int,int,int>{2,1,1});
                        cur_c *= b.multiplier;
                        break;
                    case EncoderBlockKind::COMPRESS_ALL_RES:
                        blocks[key] = std::make_shared<SpaceToDepthDownsample>(cur_c, cur_c * b.multiplier, std::tuple<int,int,int>{2,2,2});
                        cur_c *= b.multiplier;
                        break;
                }
            }

            // UNIFORM log-var: conv_out gets one extra channel for the shared logvar.
            int conv_out_ch = latent_ch + 1;
            blocks["conv_out"] = std::make_shared<LTXV::CausalConv3d>(cur_c, conv_out_ch, 3);
            blocks["per_channel_statistics"] = std::make_shared<PerChannelStatisticsBlock>(latent_ch);
        }

        // sample: ne=[W, H, T, C=3] (B=1). Returns normalized latent ne=[W', H', T', latent_ch].
        // If trace_outputs is non-null, intermediates are pushed in this order:
        //   0: post_patchify, 1: post_conv_in, 2..K-1: per down_block output,
        //   K: post_norm, K+1: post_conv_out, K+2: means_preNorm, K+3: latent.
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* sample,
                             std::vector<ggml_tensor*>* trace_outputs = nullptr) {
            // patchify (distinct channel ordering from the SpaceToDepthDownsample blocks;
            // see `patchify` comment in ltxvae_primitives.hpp).
            auto x = patchify(ctx->ggml_ctx, sample, 1, patch_size, patch_size);
            if (trace_outputs) trace_outputs->push_back(ggml_cont(ctx->ggml_ctx, x));

            auto conv_in = std::dynamic_pointer_cast<LTXV::CausalConv3d>(blocks["conv_in"]);
            x = conv_in->forward(ctx, x);
            if (trace_outputs) trace_outputs->push_back(ggml_cont(ctx->ggml_ctx, x));

            for (size_t i = 0; i < encoder_blocks.size(); ++i) {
                std::string key = "down_blocks." + std::to_string(i);
                switch (encoder_blocks[i].kind) {
                    case EncoderBlockKind::RES_X: {
                        auto b = std::dynamic_pointer_cast<UNetMidBlock3D>(blocks[key]);
                        x = b->forward(ctx, x, nullptr);
                        break;
                    }
                    default: {
                        auto b = std::dynamic_pointer_cast<SpaceToDepthDownsample>(blocks[key]);
                        x = b->forward(ctx, x);
                        break;
                    }
                }
                if (trace_outputs) trace_outputs->push_back(ggml_cont(ctx->ggml_ctx, x));
            }

            x = pixel_norm(ctx->ggml_ctx, x, eps);
            if (trace_outputs) trace_outputs->push_back(ggml_cont(ctx->ggml_ctx, x));
            x = ggml_silu(ctx->ggml_ctx, x);

            auto conv_out = std::dynamic_pointer_cast<LTXV::CausalConv3d>(blocks["conv_out"]);
            x = conv_out->forward(ctx, x);  // ne=[W', H', T', latent_ch+1]
            if (trace_outputs) trace_outputs->push_back(ggml_cont(ctx->ggml_ctx, x));

            // UNIFORM log_var handling: means = x[:, :-1], we skip logvar entirely (it would be
            // expanded then discarded after the chunk(2) split). Take the first latent_ch channels.
            auto means = ggml_view_4d(ctx->ggml_ctx, x,
                                      x->ne[0], x->ne[1], x->ne[2], latent_channels,
                                      x->nb[1], x->nb[2], x->nb[3], 0);
            means = ggml_cont(ctx->ggml_ctx, means);
            if (trace_outputs) trace_outputs->push_back(means);

            auto pcs = std::dynamic_pointer_cast<PerChannelStatisticsBlock>(blocks["per_channel_statistics"]);
            auto latent = pcs->normalize(ctx, means);
            if (trace_outputs) trace_outputs->push_back(ggml_cont(ctx->ggml_ctx, latent));
            return latent;
        }
    };

    // ---------- VideoDecoder ----------

    enum class DecoderBlockKind {
        RES_X,
        COMPRESS_SPACE,  // stride=(1,2,2), residual=False
        COMPRESS_TIME,   // stride=(2,1,1), residual=False
        COMPRESS_ALL,    // stride=(2,2,2), residual configurable (default False here)
    };

    struct DecoderBlockSpec {
        DecoderBlockKind kind;
        int num_layers = 1;  // RES_X
        int multiplier = 1;  // channel reduction factor for compress_*
        // Per-res_block timestep conditioning. Defaults to true (older
        // configs assumed all RES_X blocks were timestep-conditioned), but
        // the real LTX-2 22B VAE only conditions the inner res_blocks; the
        // outer ones lack scale_shift_table + time_embedder weights.
        bool timestep_cond = true;
    };

    struct VideoDecoder : public GGMLBlock {
    protected:
        int latent_channels = 128;
        int out_channels = 3;
        int patch_size = 4;
        int base_channels = 128;
        bool timestep_conditioning = true;
        std::vector<DecoderBlockSpec> decoder_blocks;  // stored in ENCODER-side order; forward reverses
        float eps = 1e-6f;
        int feature_channels = 0;
        // Decoder uses `reflect` spatial padding by default per the Python reference
        // (VideoDecoderConfigurator.from_config default). All CausalConv3d instances we
        // construct below are handed this padding mode.
        static constexpr LTXV::SpatialPadding PAD = LTXV::SpatialPadding::REFLECT;
        // Python configurator defaults: `causal_decoder=False`. All our CausalConv3d.forward
        // calls within the decoder should therefore use causal=False. (Encoder uses True.)
        static constexpr bool DECODER_CAUSAL = false;

        void init_params(ggml_context* ctx, const String2TensorStorage&, const std::string /*prefix*/ = "") override {
            if (timestep_conditioning) {
                // Python: last_scale_shift_table = Parameter(torch.empty(2, feature_channels)).
                params["last_scale_shift_table"]  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, feature_channels, 2);
                // timestep_scale_multiplier: scalar.
                params["timestep_scale_multiplier"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
            }
        }

    public:
        VideoDecoder() = default;
        VideoDecoder(int latent_ch, int out_ch, int patch, int base_ch, bool timestep_cond,
                     const std::vector<DecoderBlockSpec>& dec_blocks)
            : latent_channels(latent_ch), out_channels(out_ch), patch_size(patch),
              base_channels(base_ch), timestep_conditioning(timestep_cond),
              decoder_blocks(dec_blocks) {
            // Decoder's feature_channels = base_channels * 8 per LTX-2 default (3 upsample blocks × 2).
            feature_channels = base_ch * 8;

            blocks["conv_in"] = std::make_shared<LTXV::CausalConv3d>(
                latent_ch, feature_channels, 3, std::tuple<int,int,int>{1,1,1}, 1, true, PAD);

            // Decoder config is stored in encoder-side order; construct up_blocks in REVERSED order
            // (matching the Python `list(reversed(decoder_blocks))`).
            int cur_c = feature_channels;
            for (size_t i = 0; i < decoder_blocks.size(); ++i) {
                const auto& b = decoder_blocks[decoder_blocks.size() - 1 - i];
                std::string key = "up_blocks." + std::to_string(i);
                switch (b.kind) {
                    case DecoderBlockKind::RES_X:
                        // Per-block timestep conditioning is independent from the top-level
                        // last-step conditioning. The 22B VAE only ships scale_shift_table +
                        // time_embedder weights for the inner up_blocks (those reachable from
                        // dec_specs entries with timestep_cond=true).
                        blocks[key] = std::make_shared<UNetMidBlock3D>(cur_c, b.num_layers,
                                                                      timestep_conditioning && b.timestep_cond,
                                                                      PAD);
                        break;
                    case DecoderBlockKind::COMPRESS_SPACE:
                        blocks[key] = std::make_shared<DepthToSpaceUpsample>(cur_c, std::tuple<int,int,int>{1,2,2}, b.multiplier, PAD);
                        cur_c = cur_c / b.multiplier;
                        break;
                    case DecoderBlockKind::COMPRESS_TIME:
                        blocks[key] = std::make_shared<DepthToSpaceUpsample>(cur_c, std::tuple<int,int,int>{2,1,1}, b.multiplier, PAD);
                        cur_c = cur_c / b.multiplier;
                        break;
                    case DecoderBlockKind::COMPRESS_ALL:
                        blocks[key] = std::make_shared<DepthToSpaceUpsample>(cur_c, std::tuple<int,int,int>{2,2,2}, b.multiplier, PAD);
                        cur_c = cur_c / b.multiplier;
                        break;
                }
            }

            int final_out_ch = out_ch * patch * patch;
            blocks["conv_out"] = std::make_shared<LTXV::CausalConv3d>(
                cur_c, final_out_ch, 3, std::tuple<int,int,int>{1,1,1}, 1, true, PAD);

            if (timestep_conditioning) {
                blocks["last_time_embedder"] = std::make_shared<TimestepEmbedder>(feature_channels * 2);
            }
            blocks["per_channel_statistics"] = std::make_shared<PerChannelStatisticsBlock>(latent_ch);
        }

        // Trace stage order (for parity debugging):
        //   0 post_unnorm, 1 post_conv_in, 2..K-1 per up_block output,
        //   K post_pixel_norm (pre-ada), K+1 post_ada, K+2 post_conv_out, K+3 video_out.
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* latent, ggml_tensor* timestep = nullptr,
                             std::vector<ggml_tensor*>* trace_outputs = nullptr) {
            auto pcs = std::dynamic_pointer_cast<PerChannelStatisticsBlock>(blocks["per_channel_statistics"]);
            auto x = pcs->un_normalize(ctx, latent);
            if (trace_outputs) trace_outputs->push_back(ggml_cont(ctx->ggml_ctx, x));

            auto conv_in = std::dynamic_pointer_cast<LTXV::CausalConv3d>(blocks["conv_in"]);
            // Earlier dump_vae.py used default causal=True for conv_in, but the real Python
            // decoder.forward uses self.causal which is False — the dumper is now aligned.
            x = conv_in->forward(ctx, x, DECODER_CAUSAL);
            if (trace_outputs) trace_outputs->push_back(ggml_cont(ctx->ggml_ctx, x));

            for (size_t i = 0; i < decoder_blocks.size(); ++i) {
                const auto& b = decoder_blocks[decoder_blocks.size() - 1 - i];
                std::string key = "up_blocks." + std::to_string(i);
                if (b.kind == DecoderBlockKind::RES_X) {
                    auto blk = std::dynamic_pointer_cast<UNetMidBlock3D>(blocks[key]);
                    x = blk->forward(ctx, x, timestep_conditioning ? timestep : nullptr, trace_outputs, DECODER_CAUSAL);
                } else {
                    auto blk = std::dynamic_pointer_cast<DepthToSpaceUpsample>(blocks[key]);
                    x = blk->forward(ctx, x, DECODER_CAUSAL);
                }
                if (trace_outputs) trace_outputs->push_back(ggml_cont(ctx->ggml_ctx, x));
            }

            // Final norm + AdaLN + SiLU + conv_out.
            x = pixel_norm(ctx->ggml_ctx, x, eps);
            if (trace_outputs) trace_outputs->push_back(ggml_cont(ctx->ggml_ctx, x));

            if (timestep_conditioning) {
                GGML_ASSERT(timestep != nullptr);
                auto te = std::dynamic_pointer_cast<TimestepEmbedder>(blocks["last_time_embedder"]);
                // Python multiplies the timestep by timestep_scale_multiplier BEFORE the embed.
                auto tsm = params["timestep_scale_multiplier"];  // scalar [1]
                auto t_scaled = ggml_mul(ctx->ggml_ctx, timestep, tsm);
                auto time_embed = te->forward(ctx, t_scaled);   // ne=[2*feature_channels, 1]

                auto sst = params["last_scale_shift_table"];     // ne=[feature_channels, 2]
                auto te2 = ggml_reshape_2d(ctx->ggml_ctx, time_embed, feature_channels, 2);
                auto ada = ggml_add(ctx->ggml_ctx, te2, sst);    // ne=[feature_channels, 2]

                auto shift = ggml_ext_slice(ctx->ggml_ctx, ada, 1, 0, 1);
                auto scale = ggml_ext_slice(ctx->ggml_ctx, ada, 1, 1, 2);
                shift = ggml_reshape_4d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, shift), 1, 1, 1, feature_channels);
                scale = ggml_reshape_4d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, scale), 1, 1, 1, feature_channels);

                auto x_scaled = ggml_mul(ctx->ggml_ctx, x, scale);
                x = ggml_add(ctx->ggml_ctx, x, x_scaled);
                x = ggml_add(ctx->ggml_ctx, x, shift);
            }
            if (trace_outputs) trace_outputs->push_back(ggml_cont(ctx->ggml_ctx, x));

            x = ggml_silu(ctx->ggml_ctx, x);

            auto conv_out = std::dynamic_pointer_cast<LTXV::CausalConv3d>(blocks["conv_out"]);
            x = conv_out->forward(ctx, x, DECODER_CAUSAL);
            if (trace_outputs) trace_outputs->push_back(ggml_cont(ctx->ggml_ctx, x));

            x = unpatchify(ctx->ggml_ctx, x, 1, patch_size, patch_size);
            if (trace_outputs) trace_outputs->push_back(ggml_cont(ctx->ggml_ctx, x));
            return x;
        }
    };

    // ---------- GGMLRunner wrappers ----------

    struct VAEEncoderRunner : public GGMLRunner {
        VideoEncoder encoder;
        int in_channels;
        int latent_channels;
        int patch_size;

        VAEEncoderRunner(ggml_backend_t backend,
                         bool offload_params_to_cpu,
                         const String2TensorStorage& tensor_storage_map,
                         const std::string& prefix,
                         int in_ch,
                         int latent_ch,
                         int patch,
                         const std::vector<EncoderBlockSpec>& specs)
            : GGMLRunner(backend, offload_params_to_cpu),
              encoder(in_ch, latent_ch, patch, specs),
              in_channels(in_ch), latent_channels(latent_ch), patch_size(patch) {
            encoder.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override { return "ltx2_vae_encoder"; }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) {
            encoder.get_param_tensors(tensors, prefix);
        }

        // stage_index==-1 returns the final latent; >=0 returns the matching trace.
        // Full forward is always built so buffer allocation covers every declared input.
        sd::Tensor<float> compute(int n_threads, const sd::Tensor<float>& video_tensor,
                                  int stage_index = -1) {
            auto get_g = [&]() -> ggml_cgraph* {
                ggml_cgraph* gf = ggml_new_graph(compute_ctx);
                ggml_tensor* x  = make_input(video_tensor);
                auto runner_ctx = get_context();
                std::vector<ggml_tensor*> traces;
                ggml_tensor* final_out = encoder.forward(&runner_ctx, x, &traces);
                ggml_build_forward_expand(gf, final_out);
                if (stage_index >= 0 && stage_index < (int)traces.size()) {
                    ggml_build_forward_expand(gf, traces[stage_index]);
                }
                return gf;
            };
            return take_or_empty(GGMLRunner::compute<float>(get_g, n_threads, true));
        }
    };

    struct VAEDecoderRunner : public GGMLRunner {
        VideoDecoder decoder;

        VAEDecoderRunner(ggml_backend_t backend,
                         bool offload_params_to_cpu,
                         const String2TensorStorage& tensor_storage_map,
                         const std::string& prefix,
                         int latent_ch,
                         int out_ch,
                         int patch,
                         int base_ch,
                         bool timestep_cond,
                         const std::vector<DecoderBlockSpec>& specs)
            : GGMLRunner(backend, offload_params_to_cpu),
              decoder(latent_ch, out_ch, patch, base_ch, timestep_cond, specs) {
            decoder.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override { return "ltx2_vae_decoder"; }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) {
            decoder.get_param_tensors(tensors, prefix);
        }

        // stage_index==-1 returns the final video output; >=0 returns the matching trace.
        // We always build the FULL forward graph so every declared input has a backend
        // buffer; when stage_index is set we just re-expand that trace last so it becomes
        // the final-result tensor that GGMLRunner::compute extracts.
        sd::Tensor<float> compute(int n_threads,
                                  const sd::Tensor<float>& latent_tensor,
                                  const sd::Tensor<float>& timestep_tensor,
                                  int stage_index = -1) {
            auto get_g = [&]() -> ggml_cgraph* {
                ggml_cgraph* gf = ggml_new_graph(compute_ctx);
                ggml_tensor* z  = make_input(latent_tensor);
                ggml_tensor* t  = timestep_tensor.empty() ? nullptr : make_input(timestep_tensor);
                auto runner_ctx = get_context();
                std::vector<ggml_tensor*> traces;
                ggml_tensor* final_out = decoder.forward(&runner_ctx, z, t, &traces);
                ggml_build_forward_expand(gf, final_out);
                if (stage_index >= 0 && stage_index < (int)traces.size()) {
                    ggml_build_forward_expand(gf, traces[stage_index]);
                }
                return gf;
            };
            return take_or_empty(GGMLRunner::compute<float>(get_g, n_threads, true));
        }
    };

    // ---------- Combined VAE runner ----------
    //
    // Plumbs both VideoEncoder and VideoDecoder into the shared VAE interface so
    // create_vae() in stable-diffusion.cpp can treat LTX-2 like any other VAE.
    //
    // Prefix convention matches the real LTX-2 checkpoint: `vae.encoder.*`,
    // `vae.decoder.*`, `vae.per_channel_statistics.*`. Since our VideoEncoder and
    // VideoDecoder each register a PerChannelStatisticsBlock under their own
    // sub-prefix, we need the state dict to have nested PCS copies (which our
    // parity dumper provides). Real LTX-2 checkpoints only ship the top-level
    // `vae.per_channel_statistics.*` — see FUTURE note below.

    struct LTX2VAERunner : public VAE {
        VideoEncoder encoder;
        VideoDecoder decoder;
        float decode_timestep = 0.05f;  // Python default.
        bool uses_timestep_conditioning = true;

        LTX2VAERunner(ggml_backend_t backend,
                      bool offload_params_to_cpu,
                      const String2TensorStorage& tensor_storage_map,
                      const std::string& prefix,
                      SDVersion version_,
                      int in_ch                                 = 3,
                      int latent_ch                             = 128,
                      int patch                                 = 4,
                      int decoder_base_ch                       = 128,
                      bool timestep_cond                        = true,
                      std::vector<EncoderBlockSpec> enc_specs   = {},
                      std::vector<DecoderBlockSpec> dec_specs   = {})
            : VAE(version_, backend, offload_params_to_cpu),
              encoder(in_ch, latent_ch, patch, enc_specs.empty() ? default_enc_specs() : enc_specs),
              decoder(latent_ch, in_ch, patch, decoder_base_ch, timestep_cond,
                      dec_specs.empty() ? default_dec_specs() : dec_specs),
              uses_timestep_conditioning(timestep_cond) {
            // LTX-2 callers pass already-[-1,1] RGB to encode (e.g. preprocessing.hpp
            // hands us raw pixel values mapped to [-1, 1]), so the encoder must NOT
            // re-scale [0,1]→[-1,1]. The decoder, however, still produces [-1,1]
            // outputs that downstream tensor_to_sd_image expects mapped to [0, 1] —
            // so we leave scale_output at its default `true` (added separately to
            // the base VAE for asymmetric paths like this one).
            scale_input = false;
            encoder.init(params_ctx, tensor_storage_map, prefix + ".encoder");
            decoder.init(params_ctx, tensor_storage_map, prefix + ".decoder");
        }

        // Production default: 1× compress_space_res, 1× compress_time_res, 2× compress_all_res,
        // per the LTXV paper "Standard LTX Video configuration" docstring.
        static std::vector<EncoderBlockSpec> default_enc_specs() {
            return {
                {EncoderBlockKind::COMPRESS_SPACE_RES, 1, 2},
                {EncoderBlockKind::COMPRESS_TIME_RES,  1, 2},
                {EncoderBlockKind::COMPRESS_ALL_RES,   1, 2},
                {EncoderBlockKind::COMPRESS_ALL_RES,   1, 2},
            };
        }
        static std::vector<DecoderBlockSpec> default_dec_specs() {
            // Stored in encoder-side order; VideoDecoder reverses.
            return {
                {DecoderBlockKind::COMPRESS_SPACE, 1, 1},
                {DecoderBlockKind::COMPRESS_TIME,  1, 1},
                {DecoderBlockKind::COMPRESS_ALL,   1, 1},
                {DecoderBlockKind::COMPRESS_ALL,   1, 1},
            };
        }

        // Real 22B LTX-2 video VAE spec, reverse-engineered from the checkpoint's
        // weight shapes (encoder ch progression: 128 → 256 → 512 → 1024 → 1024):
        //   idx kind                    cur_c after
        //    0  RES_X(4 layers)         128
        //    1  COMPRESS_SPACE_RES(m=2) 128 → 256
        //    2  RES_X(6 layers)         256
        //    3  COMPRESS_TIME_RES(m=2)  256 → 512
        //    4  RES_X(4 layers)         512
        //    5  COMPRESS_ALL_RES(m=2)   512 → 1024
        //    6  RES_X(2 layers)         1024
        //    7  COMPRESS_ALL_RES(m=1)   1024 → 1024 (spatial/temporal compress only)
        //    8  RES_X(2 layers)         1024
        // Final conv_out: 1024 → 129 (128 latent + 1 logvar).
        // Decoder mirrors in encoder-side order; VideoDecoder reverses at construct.
        static std::vector<EncoderBlockSpec> ltx2_22b_enc_specs() {
            return {
                {EncoderBlockKind::RES_X,              4, 1},
                {EncoderBlockKind::COMPRESS_SPACE_RES, 1, 2},
                {EncoderBlockKind::RES_X,              6, 1},
                {EncoderBlockKind::COMPRESS_TIME_RES,  1, 2},
                {EncoderBlockKind::RES_X,              4, 1},
                {EncoderBlockKind::COMPRESS_ALL_RES,   1, 2},
                {EncoderBlockKind::RES_X,              2, 1},
                {EncoderBlockKind::COMPRESS_ALL_RES,   1, 1},
                {EncoderBlockKind::RES_X,              2, 1},
            };
        }
        static std::vector<DecoderBlockSpec> ltx2_22b_dec_specs() {
            // Encoder-side order; VideoDecoder iterates in reverse at construct.
            // Reverse iteration maps decoder_blocks[i] → up_blocks.[N-1-i], so the
            // last entry here becomes up_blocks.0 (innermost, 1024 channels).
            //
            // Decoder channel progression (verified against real weight shapes):
            //    up_blocks.0 RES_X(2)               @ 1024
            //    up_blocks.1 COMPRESS_ALL(m=2)      1024 → 512 (conv:4096, d2s/8)
            //    up_blocks.2 RES_X(2)               @ 512
            //    up_blocks.3 COMPRESS_ALL(m=1)      512 → 512  (conv:4096, d2s/8)
            //    up_blocks.4 RES_X(4)               @ 512
            //    up_blocks.5 COMPRESS_TIME(m=2)     512 → 256  (conv:512,  d2s/2)
            //    up_blocks.6 RES_X(6)               @ 256
            //    up_blocks.7 COMPRESS_SPACE(m=2)    256 → 128  (conv:512,  d2s/4)
            //    up_blocks.8 RES_X(4)               @ 128
            // Decoder's compress multipliers are NOT a mirror of the encoder's
            // — the model is architecturally asymmetric (different res counts, different
            // compress kinds at each level). Enc vs dec must each be traced separately.
            // Per-block timestep_cond per the actual 22B checkpoint contents:
            // NO up_block has per-res_block scale_shift_table + time_embedder
            // weights — the LTX-2 22B VAE only conditions at the LAST step
            // (last_scale_shift_table + last_time_embedder, gated by the
            // top-level timestep_conditioning flag).
            return {
                {DecoderBlockKind::RES_X,          4, 1, /*timestep_cond=*/false},
                {DecoderBlockKind::COMPRESS_SPACE, 1, 2},
                {DecoderBlockKind::RES_X,          6, 1, /*timestep_cond=*/false},
                {DecoderBlockKind::COMPRESS_TIME,  1, 2},
                {DecoderBlockKind::RES_X,          4, 1, /*timestep_cond=*/false},
                {DecoderBlockKind::COMPRESS_ALL,   1, 1},
                {DecoderBlockKind::RES_X,          2, 1, /*timestep_cond=*/false},
                {DecoderBlockKind::COMPRESS_ALL,   1, 2},
                {DecoderBlockKind::RES_X,          2, 1, /*timestep_cond=*/false},
            };
        }

        std::string get_desc() override { return "ltx2_vae"; }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) override {
            encoder.get_param_tensors(tensors, prefix + ".encoder");
            decoder.get_param_tensors(tensors, prefix + ".decoder");
        }

        int get_encoder_output_channels(int /*input_channels*/) override {
            return 128;  // latent_channels
        }

        sd::Tensor<float> vae_output_to_latents(const sd::Tensor<float>& vae_output,
                                                std::shared_ptr<RNG> /*rng*/) override {
            return vae_output;
        }
        sd::Tensor<float> diffusion_to_vae_latents(const sd::Tensor<float>& latents) override {
            return latents;
        }
        sd::Tensor<float> vae_to_diffusion_latents(const sd::Tensor<float>& latents) override {
            return latents;
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& z_tensor, bool decode_graph) {
            // 10240 fit the 4-block parity test. The 22B VAE has 9 encoder + 9
            // decoder blocks with up to 6 res_blocks each, plus per-channel stats
            // and conv_in/out. Bumped for safety.
            ggml_cgraph* gf = new_graph_custom(65536);
            ggml_tensor* z  = make_input(z_tensor);
            auto runner_ctx = get_context();
            ggml_tensor* out;
            if (decode_graph) {
                ggml_tensor* t = nullptr;
                if (uses_timestep_conditioning) {
                    // Build a scalar timestep tensor inline (no external input needed).
                    t = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_F32, 1);
                    ggml_set_name(t, "ltx2_vae_decode_timestep");
                    decode_timestep_backing.resize(1);
                    decode_timestep_backing[0] = decode_timestep;
                    set_backend_tensor_data(t, decode_timestep_backing.data());
                }
                out = decoder.forward(&runner_ctx, z, t);
            } else {
                out = encoder.forward(&runner_ctx, z);
            }
            ggml_build_forward_expand(gf, out);
            return gf;
        }

        sd::Tensor<float> _compute(const int n_threads,
                                   const sd::Tensor<float>& z,
                                   bool decode_graph) override {
            auto get_g = [&]() -> ggml_cgraph* { return build_graph(z, decode_graph); };
            auto out = take_or_empty(GGMLRunner::compute<float>(get_g, n_threads, true));
            // Decoder output is [W, H, T, C]; decode_video_outputs + tensor_to_sd_image
            // expect 5D [W, H, T, C, B] to pick the video-shaped index path. Add the
            // trailing batch axis so the conversion uses the (iw, ih, frame, ic, 0)
            // accessor (the 4D path assumes [W, H, C, F] which is the wrong layout).
            if (decode_graph && !out.empty() && out.shape().size() == 4) {
                auto s = out.shape();
                out.reshape_({s[0], s[1], s[2], s[3], 1});
            }
            return out;
        }

    private:
        std::vector<float> decode_timestep_backing;
    };

}  // namespace LTXVAE

#endif  // __LTXVAE_HPP__
