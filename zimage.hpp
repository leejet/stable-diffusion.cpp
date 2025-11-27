#ifndef __ZIMAGE_HPP__
#define __ZIMAGE_HPP__

#include "ggml_extend.hpp"
#include "rope.hpp"

namespace ZImage {
    constexpr int ZIMAGE_GRAPH_SIZE = 40960;
    constexpr int ADALN_EMBED_DIM   = 256;
    constexpr int SEQ_MULTI_OF      = 32;

    struct TimestepEmbedder : public UnaryBlock {
    protected:
        int64_t out_size;
        int64_t frequency_embedding_size;

    public:
        TimestepEmbedder(int64_t out_size, int64_t mid_size = 1024, int64_t frequency_embedding_size = 256)
            : out_size(out_size), frequency_embedding_size(frequency_embedding_size) {
            blocks["mlp.0"] = std::shared_ptr<GGMLBlock>(new Linear(frequency_embedding_size, mid_size, true));
            blocks["mlp.2"] = std::shared_ptr<GGMLBlock>(new Linear(mid_size, out_size, true));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* t) override {
            auto mlp_0 = std::dynamic_pointer_cast<Linear>(blocks["mlp.0"]);
            auto mlp_2 = std::dynamic_pointer_cast<Linear>(blocks["mlp.2"]);

            auto t_freq = ggml_ext_timestep_embedding(ctx->ggml_ctx, t, (int)frequency_embedding_size, 10000, 1000.f);
            auto t_emb  = mlp_0->forward(ctx, t_freq);
            t_emb       = ggml_silu_inplace(ctx->ggml_ctx, t_emb);
            t_emb       = mlp_2->forward(ctx, t_emb);
            return t_emb;
        }
    };

    struct ZImageFeedForward : public UnaryBlock {
    protected:
        int64_t dim;
        int64_t hidden_dim;

    public:
        ZImageFeedForward(int64_t dim)
            : dim(dim) {
            hidden_dim   = (int64_t)(dim / 3 * 8);
            blocks["w1"] = std::shared_ptr<GGMLBlock>(new Linear(dim, hidden_dim, false));
            blocks["w2"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_dim, dim, false));
            blocks["w3"] = std::shared_ptr<GGMLBlock>(new Linear(dim, hidden_dim, false));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) override {
            auto w1 = std::dynamic_pointer_cast<Linear>(blocks["w1"]);
            auto w2 = std::dynamic_pointer_cast<Linear>(blocks["w2"]);
            auto w3 = std::dynamic_pointer_cast<Linear>(blocks["w3"]);

            auto h1 = w1->forward(ctx, x);
            h1      = ggml_silu_inplace(ctx->ggml_ctx, h1);
            auto h3 = w3->forward(ctx, x);
            h1      = ggml_mul_inplace(ctx->ggml_ctx, h1, h3);
            return w2->forward(ctx, h1);
        }
    };

    struct ZImageSelfAttention : public GGMLBlock {
    protected:
        int64_t dim;
        int64_t num_heads;
        int64_t head_dim;
        bool qk_norm;

    public:
        ZImageSelfAttention(int64_t dim, int64_t num_heads, bool qk_norm = true)
            : dim(dim), num_heads(num_heads), qk_norm(qk_norm) {
            head_dim       = dim / num_heads;
            blocks["qkv"]  = std::shared_ptr<GGMLBlock>(new Linear(dim, 3 * dim, false));
            blocks["out"]  = std::shared_ptr<GGMLBlock>(new Linear(dim, dim, false));
            if (qk_norm) {
                blocks["q_norm"] = std::shared_ptr<GGMLBlock>(new RMSNorm(head_dim, 1e-5f));
                blocks["k_norm"] = std::shared_ptr<GGMLBlock>(new RMSNorm(head_dim, 1e-5f));
            }
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* pe,
                                    struct ggml_tensor* attn_mask = nullptr) {
            auto qkv_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv"]);
            auto out      = std::dynamic_pointer_cast<Linear>(blocks["out"]);

            int64_t n_token = x->ne[1];
            int64_t N       = x->ne[2];

            auto qkv = qkv_proj->forward(ctx, x);
            size_t elem_size = ggml_element_size(qkv);
            auto q_view = ggml_view_3d(ctx->ggml_ctx, qkv, dim, n_token, N, qkv->nb[1], qkv->nb[2], 0 * dim * elem_size);
            auto k_view = ggml_view_3d(ctx->ggml_ctx, qkv, dim, n_token, N, qkv->nb[1], qkv->nb[2], 1 * dim * elem_size);
            auto v_view = ggml_view_3d(ctx->ggml_ctx, qkv, dim, n_token, N, qkv->nb[1], qkv->nb[2], 2 * dim * elem_size);
            q_view = ggml_cont(ctx->ggml_ctx, q_view);
            k_view = ggml_cont(ctx->ggml_ctx, k_view);
            v_view = ggml_cont(ctx->ggml_ctx, v_view);

            auto q = ggml_reshape_4d(ctx->ggml_ctx, q_view, head_dim, num_heads, n_token, N);
            auto k = ggml_reshape_4d(ctx->ggml_ctx, k_view, head_dim, num_heads, n_token, N);
            auto v = ggml_reshape_4d(ctx->ggml_ctx, v_view, head_dim, num_heads, n_token, N);

            if (qk_norm) {
                auto norm_q = std::dynamic_pointer_cast<RMSNorm>(blocks["q_norm"]);
                auto norm_k = std::dynamic_pointer_cast<RMSNorm>(blocks["k_norm"]);
                q           = norm_q->forward(ctx, q);
                k           = norm_k->forward(ctx, k);
            }

            if (pe != nullptr) {
                // Try with rope_interleaved=false for Z-Image
                x = Rope::attention(ctx, q, k, v, pe, attn_mask, 1.0f, false);
            } else {
                q = ggml_reshape_3d(ctx->ggml_ctx, q, dim, n_token, N);
                k = ggml_reshape_3d(ctx->ggml_ctx, k, dim, n_token, N);
                v = ggml_reshape_3d(ctx->ggml_ctx, v, dim, n_token, N);
                x = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, num_heads, attn_mask, false, false, ctx->flash_attn_enabled, 1.0f);
            }
            x = out->forward(ctx, x);
            return x;
        }
    };

    struct ZImageTransformerBlock : public GGMLBlock {
    protected:
        int64_t dim;
        bool modulation;

    public:
        ZImageTransformerBlock(int64_t dim,
                               int64_t n_heads,
                               float norm_eps   = 1e-5f,
                               bool qk_norm     = true,
                               bool modulation  = true)
            : dim(dim), modulation(modulation) {
            blocks["attention"]       = std::shared_ptr<GGMLBlock>(new ZImageSelfAttention(dim, n_heads, qk_norm));
            blocks["feed_forward"]    = std::shared_ptr<GGMLBlock>(new ZImageFeedForward(dim));
            blocks["attention_norm1"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim, norm_eps));
            blocks["ffn_norm1"]       = std::shared_ptr<GGMLBlock>(new RMSNorm(dim, norm_eps));
            blocks["attention_norm2"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim, norm_eps));
            blocks["ffn_norm2"]       = std::shared_ptr<GGMLBlock>(new RMSNorm(dim, norm_eps));
            if (modulation) {
                int64_t adaln_in             = dim < ADALN_EMBED_DIM ? dim : ADALN_EMBED_DIM;
                blocks["adaLN_modulation.0"] = std::shared_ptr<GGMLBlock>(new Linear(adaln_in, 4 * dim, true));
            }
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* attn_mask,
                                    struct ggml_tensor* freqs_cis,
                                    struct ggml_tensor* adaln_input = nullptr) {
            auto attention       = std::dynamic_pointer_cast<ZImageSelfAttention>(blocks["attention"]);
            auto feed_forward    = std::dynamic_pointer_cast<ZImageFeedForward>(blocks["feed_forward"]);
            auto attention_norm1 = std::dynamic_pointer_cast<RMSNorm>(blocks["attention_norm1"]);
            auto ffn_norm1       = std::dynamic_pointer_cast<RMSNorm>(blocks["ffn_norm1"]);
            auto attention_norm2 = std::dynamic_pointer_cast<RMSNorm>(blocks["attention_norm2"]);
            auto ffn_norm2       = std::dynamic_pointer_cast<RMSNorm>(blocks["ffn_norm2"]);

            if (modulation && adaln_input != nullptr) {
                LOG_DEBUG("Block modulation: adaln_input=[%lld, %lld]", adaln_input->ne[0], adaln_input->ne[1]);
                auto adaLN = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.0"]);
                auto mod   = adaLN->forward(ctx, adaln_input);
                LOG_DEBUG("Block modulation: mod=[%lld, %lld]", mod->ne[0], mod->ne[1]);
                int64_t B  = mod->ne[1];
                size_t elem_size = ggml_element_size(mod);
                int64_t stride_B = dim * 4 * elem_size;

                auto scale_msa = ggml_view_2d(ctx->ggml_ctx, mod, dim, B, stride_B, 0 * dim * elem_size);
                auto gate_msa  = ggml_view_2d(ctx->ggml_ctx, mod, dim, B, stride_B, 1 * dim * elem_size);
                auto scale_mlp = ggml_view_2d(ctx->ggml_ctx, mod, dim, B, stride_B, 2 * dim * elem_size);
                auto gate_mlp  = ggml_view_2d(ctx->ggml_ctx, mod, dim, B, stride_B, 3 * dim * elem_size);
                LOG_DEBUG("Block modulation: views created");

                scale_msa = ggml_reshape_3d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, scale_msa), dim, 1, B);
                gate_msa  = ggml_tanh(ctx->ggml_ctx, ggml_reshape_3d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, gate_msa), dim, 1, B));
                scale_mlp = ggml_reshape_3d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, scale_mlp), dim, 1, B);
                gate_mlp  = ggml_tanh(ctx->ggml_ctx, ggml_reshape_3d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, gate_mlp), dim, 1, B));
                LOG_DEBUG("Block modulation: gates/scales reshaped");

                auto normed   = attention_norm1->forward(ctx, x);
                LOG_DEBUG("Block: after attention_norm1");
                normed        = ggml_add(ctx->ggml_ctx, normed, ggml_mul(ctx->ggml_ctx, normed, scale_msa));
                LOG_DEBUG("Block: calling attention");
                auto attn_out = attention->forward(ctx, normed, freqs_cis, attn_mask);
                attn_out      = attention_norm2->forward(ctx, attn_out);
                attn_out      = ggml_mul_inplace(ctx->ggml_ctx, attn_out, gate_msa);
                x             = ggml_add_inplace(ctx->ggml_ctx, x, attn_out);

                normed       = ffn_norm1->forward(ctx, x);
                normed       = ggml_add(ctx->ggml_ctx, normed, ggml_mul(ctx->ggml_ctx, normed, scale_mlp));
                auto ffn_out = feed_forward->forward(ctx, normed);
                ffn_out      = ffn_norm2->forward(ctx, ffn_out);
                ffn_out      = ggml_mul_inplace(ctx->ggml_ctx, ffn_out, gate_mlp);
                x            = ggml_add_inplace(ctx->ggml_ctx, x, ffn_out);
            } else {
                auto normed   = attention_norm1->forward(ctx, x);
                auto attn_out = attention->forward(ctx, normed, freqs_cis, attn_mask);
                attn_out      = attention_norm2->forward(ctx, attn_out);
                x             = ggml_add_inplace(ctx->ggml_ctx, x, attn_out);

                normed       = ffn_norm1->forward(ctx, x);
                auto ffn_out = feed_forward->forward(ctx, normed);
                ffn_out      = ffn_norm2->forward(ctx, ffn_out);
                x            = ggml_add_inplace(ctx->ggml_ctx, x, ffn_out);
            }
            return x;
        }
    };

    struct FinalLayer : public GGMLBlock {
    protected:
        int64_t hidden_size;

    public:
        FinalLayer(int64_t hidden_size, int64_t out_channels)
            : hidden_size(hidden_size) {
            blocks["norm_final"]         = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-6f, false));
            blocks["linear"]             = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, out_channels, true));
            int64_t adaln_in             = hidden_size < ADALN_EMBED_DIM ? hidden_size : ADALN_EMBED_DIM;
            blocks["adaLN_modulation.1"] = std::shared_ptr<GGMLBlock>(new Linear(adaln_in, hidden_size, true));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* c) {
            auto norm_final = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_final"]);
            auto linear     = std::dynamic_pointer_cast<Linear>(blocks["linear"]);
            auto adaLN      = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.1"]);

            auto scale = ggml_silu(ctx->ggml_ctx, c);
            scale      = adaLN->forward(ctx, scale);
            scale      = ggml_reshape_3d(ctx->ggml_ctx, scale, scale->ne[0], 1, scale->ne[1]);

            x = norm_final->forward(ctx, x);
            x = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, x, scale));
            x = linear->forward(ctx, x);
            return x;
        }
    };

    struct ZImageTransformer2DModel : public GGMLBlock {
    protected:
        int64_t in_channels;
        int64_t dim;
        int64_t n_layers;
        int64_t n_refiner_layers;
        int64_t n_heads;
        int64_t patch_size;
        float t_scale;
        int theta;
        std::vector<int64_t> axes_dims;
        std::vector<int64_t> axes_lens;

    public:
        ZImageTransformer2DModel(int64_t in_channels      = 16,
                                 int64_t dim              = 3840,
                                 int64_t n_layers         = 30,
                                 int64_t n_refiner_layers = 2,
                                 int64_t n_heads          = 30,
                                 float norm_eps           = 1e-5f,
                                 bool qk_norm             = true,
                                 int64_t cap_feat_dim     = 2560,
                                 int64_t patch_size       = 2,
                                 float t_scale            = 1000.f,
                                 int theta_               = 256,
                                 std::vector<int64_t> axes_dims = {32, 48, 48},
                                 std::vector<int64_t> axes_lens = {1536, 512, 512})
            : in_channels(in_channels), dim(dim), n_layers(n_layers), n_refiner_layers(n_refiner_layers),
              n_heads(n_heads), patch_size(patch_size), t_scale(t_scale), theta(theta_),
              axes_dims(axes_dims), axes_lens(axes_lens) {
            blocks["x_embedder"] = std::shared_ptr<GGMLBlock>(
                new Linear(patch_size * patch_size * in_channels, dim, true));
            blocks["final_layer"] = std::shared_ptr<GGMLBlock>(
                new FinalLayer(dim, patch_size * patch_size * in_channels));

            for (int i = 0; i < n_refiner_layers; i++) {
                blocks["noise_refiner." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(
                    new ZImageTransformerBlock(dim, n_heads, norm_eps, qk_norm, true));
            }

            for (int i = 0; i < n_refiner_layers; i++) {
                blocks["context_refiner." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(
                    new ZImageTransformerBlock(dim, n_heads, norm_eps, qk_norm, false));
            }

            int64_t adaln_size   = dim < ADALN_EMBED_DIM ? dim : ADALN_EMBED_DIM;
            blocks["t_embedder"] = std::shared_ptr<GGMLBlock>(new TimestepEmbedder(adaln_size, 1024, 256));

            blocks["cap_embedder.0"] = std::shared_ptr<GGMLBlock>(new RMSNorm(cap_feat_dim, norm_eps));
            blocks["cap_embedder.1"] = std::shared_ptr<GGMLBlock>(new Linear(cap_feat_dim, dim, true));

            for (int i = 0; i < n_layers; i++) {
                blocks["layers." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(
                    new ZImageTransformerBlock(dim, n_heads, norm_eps, qk_norm, true));
            }
        }


        int64_t get_dim() const { return dim; }
        int64_t get_patch_size() const { return patch_size; }
        int64_t get_in_channels() const { return in_channels; }
        float get_t_scale() const { return t_scale; }
        int get_theta() const { return theta; }
        const std::vector<int64_t>& get_axes_dims() const { return axes_dims; }
        const std::vector<int64_t>& get_axes_lens() const { return axes_lens; }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* t,
                                    struct ggml_tensor* cap_feats,
                                    struct ggml_tensor* x_freqs_cis,
                                    struct ggml_tensor* cap_freqs_cis,
                                    struct ggml_tensor* unified_freqs_cis,
                                    int64_t x_seq_len,
                                    int64_t cap_seq_len) {
            LOG_DEBUG("ZImage forward: x=[%lld, %lld, %lld], cap=[%lld, %lld, %lld]",
                      x->ne[0], x->ne[1], x->ne[2],
                      cap_feats->ne[0], cap_feats->ne[1], cap_feats->ne[2]);

            auto x_embedder   = std::dynamic_pointer_cast<Linear>(blocks["x_embedder"]);
            auto cap_norm     = std::dynamic_pointer_cast<RMSNorm>(blocks["cap_embedder.0"]);
            auto cap_proj     = std::dynamic_pointer_cast<Linear>(blocks["cap_embedder.1"]);
            auto final_layer  = std::dynamic_pointer_cast<FinalLayer>(blocks["final_layer"]);
            auto t_embedder   = std::dynamic_pointer_cast<TimestepEmbedder>(blocks["t_embedder"]);

            LOG_DEBUG("ZImage forward: calling x_embedder");
            x = x_embedder->forward(ctx, x);
            LOG_DEBUG("ZImage forward: after x_embedder x=[%lld, %lld, %lld]", x->ne[0], x->ne[1], x->ne[2]);

            LOG_DEBUG("ZImage forward: calling t_embedder");
            auto c = t_embedder->forward(ctx, t);
            LOG_DEBUG("ZImage forward: after t_embedder c=[%lld, %lld]", c->ne[0], c->ne[1]);

            LOG_DEBUG("ZImage forward: calling cap_norm");
            cap_feats = cap_norm->forward(ctx, cap_feats);
            LOG_DEBUG("ZImage forward: after cap_norm");
            cap_feats = cap_proj->forward(ctx, cap_feats);
            LOG_DEBUG("ZImage forward: after cap_proj cap=[%lld, %lld, %lld]", cap_feats->ne[0], cap_feats->ne[1], cap_feats->ne[2]);

            for (int i = 0; i < n_refiner_layers; i++) {
                LOG_DEBUG("ZImage forward: noise_refiner.%d", i);
                auto block = std::dynamic_pointer_cast<ZImageTransformerBlock>(
                    blocks["noise_refiner." + std::to_string(i)]);
                x = block->forward(ctx, x, nullptr, x_freqs_cis, c);
            }
            LOG_DEBUG("ZImage forward: after noise_refiners x=[%lld, %lld, %lld]", x->ne[0], x->ne[1], x->ne[2]);

            for (int i = 0; i < n_refiner_layers; i++) {
                LOG_DEBUG("ZImage forward: context_refiner.%d", i);
                auto block = std::dynamic_pointer_cast<ZImageTransformerBlock>(
                    blocks["context_refiner." + std::to_string(i)]);
                cap_feats = block->forward(ctx, cap_feats, nullptr, cap_freqs_cis, nullptr);
            }
            LOG_DEBUG("ZImage forward: after context_refiners cap=[%lld, %lld, %lld]", cap_feats->ne[0], cap_feats->ne[1], cap_feats->ne[2]);

            // ComfyUI order: caption first, then image
            // Note: ggml_concat(a, b, dim) concatenates a first, then b
            x = ggml_concat(ctx->ggml_ctx, cap_feats, x, 1);
            int64_t total_seq_len = x->ne[1];
            LOG_DEBUG("ZImage forward: after concat total_seq_len=%lld (cap=%lld + img=%lld)",
                      total_seq_len, cap_seq_len, x_seq_len);

            for (int i = 0; i < n_layers; i++) {
                auto block = std::dynamic_pointer_cast<ZImageTransformerBlock>(
                    blocks["layers." + std::to_string(i)]);
                x = block->forward(ctx, x, nullptr, unified_freqs_cis, c);
            }

            // Extract image tokens (they come after caption tokens)
            // Current shape: [dim, total_seq, batch]
            // Image tokens are at positions cap_seq_len to (cap_seq_len + x_seq_len - 1)
            // Use view with byte offset to skip caption tokens
            size_t elem_size = ggml_element_size(x);
            size_t offset_bytes = cap_seq_len * x->ne[0] * elem_size;  // Skip cap_seq_len tokens
            x = ggml_view_3d(ctx->ggml_ctx, x, x->ne[0], x_seq_len, x->ne[2],
                             x->nb[1], x->nb[2], offset_bytes);
            x = ggml_cont(ctx->ggml_ctx, x);

            x = final_layer->forward(ctx, x, c);

            return x;
        }
    };

    __STATIC_INLINE__ std::vector<float> gen_zimage_pe(
        const std::vector<std::vector<int>>& pos_ids,
        int theta,
        const std::vector<int>& axes_dims) {
        std::vector<std::vector<float>> ids(pos_ids.size(), std::vector<float>(3));
        for (size_t i = 0; i < pos_ids.size(); i++) {
            ids[i][0] = (float)pos_ids[i][0];
            ids[i][1] = (float)pos_ids[i][1];
            ids[i][2] = (float)pos_ids[i][2];
        }
        return Rope::embed_nd(ids, 1, theta, axes_dims);
    }

    // Generate image position IDs for Z-Image
    // In diffusers: axis_0 = offset (for relative position after caption), axis_1 = row, axis_2 = col
    __STATIC_INLINE__ std::vector<std::vector<float>> gen_zimage_img_ids(int h, int w, int patch_size, int bs, int axis0_offset = 0) {
        int h_len = h / patch_size;
        int w_len = w / patch_size;
        std::vector<std::vector<float>> ids(bs * h_len * w_len, std::vector<float>(3));
        for (int b = 0; b < bs; b++) {
            for (int i = 0; i < h_len; i++) {
                for (int j = 0; j < w_len; j++) {
                    int idx = b * h_len * w_len + i * w_len + j;
                    ids[idx][0] = (float)axis0_offset;  // axis_0: offset/frame index
                    ids[idx][1] = (float)i;             // axis_1: row (height)
                    ids[idx][2] = (float)j;             // axis_2: col (width)
                }
            }
        }
        return ids;
    }

    // Generate caption position IDs for Z-Image
    // In diffusers: axis_0 = sequential position (1, 2, 3, ...), axis_1 = 0, axis_2 = 0
    __STATIC_INLINE__ std::vector<std::vector<float>> gen_zimage_cap_ids(int cap_seq_len, int bs, int axis0_start = 1) {
        std::vector<std::vector<float>> ids(bs * cap_seq_len, std::vector<float>(3));
        for (int b = 0; b < bs; b++) {
            for (int i = 0; i < cap_seq_len; i++) {
                int idx = b * cap_seq_len + i;
                ids[idx][0] = (float)(axis0_start + i);  // axis_0: sequential position starting from axis0_start
                ids[idx][1] = 0.0f;                       // axis_1: 0 for text
                ids[idx][2] = 0.0f;                       // axis_2: 0 for text
            }
        }
        return ids;
    }

    // Generate unified PE for the main transformer layers
    // Sequence order: [caption_tokens, image_tokens] (ComfyUI order)
    // Caption axis_0 = 1, 2, 3, ..., cap_seq_len
    // Image axis_0 = cap_seq_len + 1 (constant for all image tokens)
    __STATIC_INLINE__ std::vector<float> gen_zimage_unified_pe(int h, int w, int patch_size, int cap_seq_len, int bs,
                                                                int theta, const std::vector<int>& axes_dims) {
        // Caption positions: axis_0 = 1, 2, 3, ..., axis_1 = 0, axis_2 = 0
        auto cap_ids = gen_zimage_cap_ids(cap_seq_len, bs, 1);
        // Image positions: axis_0 = cap_seq_len + 1, axis_1 = row, axis_2 = col
        auto img_ids = gen_zimage_img_ids(h, w, patch_size, bs, cap_seq_len + 1);
        // Concatenate: [caption, image]
        auto ids = Rope::concat_ids(cap_ids, img_ids, bs);
        return Rope::embed_nd(ids, bs, theta, axes_dims);
    }

    struct ZImageRunner : public GGMLRunner {
        ZImageTransformer2DModel model;

        std::vector<float> unified_pe_vec;
        std::vector<float> timestep_vec;

        ZImageRunner(ggml_backend_t backend,
                     bool offload_params_to_cpu,
                     const String2TensorStorage& tensor_storage_map,
                     const std::string& prefix = "model.diffusion_model.")
            : GGMLRunner(backend, offload_params_to_cpu) {
            model = ZImageTransformer2DModel();
            model.init(params_ctx, tensor_storage_map, prefix);

            std::map<std::string, struct ggml_tensor*> model_tensors;
            model.get_param_tensors(model_tensors, prefix);
            LOG_DEBUG("ZImageRunner: model has %zu param tensors", model_tensors.size());
            for (const auto& [name, tensor] : model_tensors) {
                if (name.find("x_embedder") != std::string::npos) {
                    LOG_DEBUG("  %s: [%lld, %lld] buffer=%p", name.c_str(),
                              tensor->ne[0], tensor->ne[1], (void*)tensor->buffer);
                }
            }
        }

        std::string get_desc() override {
            return "Z-Image";
        }

        void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string& prefix = "model.diffusion_model.") {
            model.get_param_tensors(tensors, prefix);
        }

        struct ggml_cgraph* build_graph(struct ggml_tensor* x,
                                        struct ggml_tensor* timestep,
                                        struct ggml_tensor* cap_feats,
                                        int height,
                                        int width) {
            struct ggml_cgraph* gf = new_graph_custom(ZIMAGE_GRAPH_SIZE);

            LOG_DEBUG("ZImage build_graph input shapes:");
            LOG_DEBUG("  x: [%lld, %lld, %lld, %lld]", x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
            LOG_DEBUG("  cap_feats: [%lld, %lld, %lld, %lld]", cap_feats->ne[0], cap_feats->ne[1], cap_feats->ne[2], cap_feats->ne[3]);

            x         = to_backend(x);
            timestep  = to_backend(timestep);
            cap_feats = to_backend(cap_feats);

            int64_t C          = x->ne[2];
            int64_t H          = x->ne[1];
            int64_t W          = x->ne[0];
            int64_t B          = x->ne[3];
            int64_t patch_size = model.get_patch_size();

            int64_t H_patches = H / patch_size;
            int64_t W_patches = W / patch_size;
            int64_t x_seq_len = H_patches * W_patches;
            int64_t cap_seq_len = cap_feats->ne[1];

            LOG_DEBUG("ZImage patchify: H=%lld, W=%lld, C=%lld, patch_size=%lld", H, W, C, patch_size);

            auto x_patchified = ggml_reshape_4d(compute_ctx, x,
                                                patch_size, W_patches,
                                                patch_size, H_patches * C * B);
            x_patchified      = ggml_cont(compute_ctx, ggml_ext_torch_permute(compute_ctx, x_patchified, 0, 2, 1, 3));
            x_patchified      = ggml_reshape_3d(compute_ctx, x_patchified,
                                                patch_size * patch_size * C,
                                                H_patches * W_patches,
                                                B);

            LOG_DEBUG("ZImage x_patchified: [%lld, %lld, %lld]", x_patchified->ne[0], x_patchified->ne[1], x_patchified->ne[2]);

            cap_feats = ggml_reshape_3d(compute_ctx, cap_feats, cap_feats->ne[0], cap_feats->ne[1], B);

            // Enable RoPE with non-interleaved mode
            bool use_rope = true;

            struct ggml_tensor* x_freqs_cis = nullptr;
            struct ggml_tensor* cap_freqs_cis = nullptr;
            struct ggml_tensor* unified_freqs_cis = nullptr;

            if (use_rope) {
                LOG_DEBUG("ZImage: generating PE...");
                auto axes_dims = model.get_axes_dims();
                int theta = model.get_theta();
                std::vector<int> axes_dims_int(axes_dims.begin(), axes_dims.end());
                int emb_dim = 0;
                for (int d : axes_dims_int) emb_dim += d / 2;
                LOG_DEBUG("ZImage: emb_dim=%d, theta=%d, cap_seq_len=%lld", emb_dim, theta, cap_seq_len);

                // Following diffusers reference:
                // Image positions: axis_0 = cap_seq_len + 1, axis_1 = row, axis_2 = col
                // Caption positions: axis_0 = 1, 2, 3, ..., axis_1 = 0, axis_2 = 0
                int img_axis0_offset = (int)(cap_seq_len + 1);

                LOG_DEBUG("ZImage: generating img_ids with axis0_offset=%d...", img_axis0_offset);
                auto img_ids = gen_zimage_img_ids(H, W, patch_size, B, img_axis0_offset);
                auto img_pe_vec = Rope::embed_nd(img_ids, B, theta, axes_dims_int);

                x_freqs_cis = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, emb_dim, x_seq_len);
                set_backend_tensor_data(x_freqs_cis, img_pe_vec.data());

                LOG_DEBUG("ZImage: generating cap_pe with axis0_start=1...");
                auto cap_ids = gen_zimage_cap_ids(cap_seq_len, B, 1);  // axis_0 = 1, 2, 3, ...
                auto cap_pe_vec = Rope::embed_nd(cap_ids, B, theta, axes_dims_int);
                cap_freqs_cis = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, emb_dim, cap_seq_len);
                set_backend_tensor_data(cap_freqs_cis, cap_pe_vec.data());

                LOG_DEBUG("ZImage: generating unified_pe (caption first, then image)...");
                // ComfyUI order: caption first, then image
                auto unified_ids = Rope::concat_ids(cap_ids, img_ids, B);
                auto unified_pe_vec = Rope::embed_nd(unified_ids, B, theta, axes_dims_int);
                int64_t unified_seq_len = cap_seq_len + x_seq_len;
                unified_freqs_cis = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, emb_dim, unified_seq_len);
                set_backend_tensor_data(unified_freqs_cis, unified_pe_vec.data());

                LOG_DEBUG("ZImage PE shapes: x=[2,2,%d,%lld], cap=[2,2,%d,%lld], unified=[2,2,%d,%lld]",
                          emb_dim, x_seq_len, emb_dim, cap_seq_len, emb_dim, (int64_t)(x_seq_len + cap_seq_len));

                // Debug: print PE values at different positions and dims
                int pe_stride = emb_dim * 4;
                LOG_DEBUG("ZImage img_ids[0]: %.1f %.1f %.1f", img_ids[0][0], img_ids[0][1], img_ids[0][2]);
                LOG_DEBUG("ZImage img_ids[1]: %.1f %.1f %.1f", img_ids[1][0], img_ids[1][1], img_ids[1][2]);
                LOG_DEBUG("ZImage img_ids[32]: %.1f %.1f %.1f", img_ids[32][0], img_ids[32][1], img_ids[32][2]);
                LOG_DEBUG("ZImage cap_ids[0]: %.1f %.1f %.1f", cap_ids[0][0], cap_ids[0][1], cap_ids[0][2]);
                LOG_DEBUG("ZImage cap_ids[1]: %.1f %.1f %.1f", cap_ids[1][0], cap_ids[1][1], cap_ids[1][2]);
            }

            auto runner_ctx = get_context();
            struct ggml_tensor* out = model.forward(&runner_ctx, x_patchified, timestep, cap_feats,
                                                    x_freqs_cis, cap_freqs_cis, unified_freqs_cis,
                                                    x_seq_len, cap_seq_len);

            LOG_DEBUG("ZImage forward returned: [%lld, %lld, %lld]", out->ne[0], out->ne[1], out->ne[2]);

            out = ggml_reshape_4d(compute_ctx, out, patch_size, patch_size, C, H_patches * W_patches * B);
            out = ggml_cont(compute_ctx, ggml_ext_torch_permute(compute_ctx, out, 0, 2, 1, 3));
            out = ggml_reshape_4d(compute_ctx, out, patch_size * W_patches, patch_size * H_patches, C, B);

            LOG_DEBUG("ZImage unpatchify: [%lld, %lld, %lld, %lld]", out->ne[0], out->ne[1], out->ne[2], out->ne[3]);

            ggml_build_forward_expand(gf, out);
            LOG_DEBUG("ZImage build_graph completed");

            return gf;
        }

        void compute(const int n_threads,
                     struct ggml_tensor* x,
                     struct ggml_tensor* timestep,
                     struct ggml_tensor* cap_feats,
                     int height,
                     int width,
                     ggml_tensor** output,
                     ggml_context* output_ctx = nullptr) {
            auto get_graph = [&]() -> struct ggml_cgraph* {
                return build_graph(x, timestep, cap_feats, height, width);
            };
            GGMLRunner::compute(get_graph, n_threads, true, output, output_ctx);
        }
    };

};

#endif
