#ifndef __PULID_HPP__
#define __PULID_HPP__

#include "ggml_extend.hpp"

/**
 * PuLID-Flux identity injection for stable-diffusion.cpp.
 *
 * Mirrors the PerceiverAttentionCA module from
 * https://github.com/ToTheBeginning/PuLID/blob/main/pulid/encoders_transformer.py
 *
 * Each instance is a cross-attention layer where:
 *   Q comes from image tokens             (dim = 3072 = Flux hidden_size)
 *   K, V come from a precomputed ID embedding (kv_dim = 2048, num_tokens = 32)
 *
 * 14 instances are inserted into the Flux denoise loop at fixed intervals:
 *   - Every 2nd of the 19 double_blocks  (10 hook points)
 *   - Every 4th of the 38 single_blocks  (10 hook points... but the v0.9.1
 *     reference uses 4 single hooks, for 14 total)
 *
 * Weight key prefix in pulid_flux_v0.9.1.safetensors:
 *   pulid_ca.<i>.norm1.{weight,bias}
 *   pulid_ca.<i>.norm2.{weight,bias}
 *   pulid_ca.<i>.to_q.weight
 *   pulid_ca.<i>.to_kv.weight
 *   pulid_ca.<i>.to_out.weight
 *
 * Pure-ggml implementation: all ops have Vulkan / CUDA / Metal kernels in
 * the upstream ggml backends, so this works cross-vendor by construction.
 */
class PuLIDPerceiverAttentionCA : public GGMLBlock {
public:
    static constexpr int64_t DEFAULT_DIM     = 3072;  // Flux hidden size
    static constexpr int64_t DEFAULT_DIM_HEAD = 128;
    static constexpr int64_t DEFAULT_HEADS   = 16;
    static constexpr int64_t DEFAULT_KV_DIM  = 2048;  // PuLID ID-embedding dim

protected:
    int64_t dim;
    int64_t dim_head;
    int64_t heads;
    int64_t kv_dim;
    int64_t inner_dim;  // dim_head * heads = 2048

public:
    PuLIDPerceiverAttentionCA(int64_t dim       = DEFAULT_DIM,
                              int64_t dim_head  = DEFAULT_DIM_HEAD,
                              int64_t heads     = DEFAULT_HEADS,
                              int64_t kv_dim    = DEFAULT_KV_DIM)
        : dim(dim),
          dim_head(dim_head),
          heads(heads),
          kv_dim(kv_dim),
          inner_dim(dim_head * heads) {
        // Note the PyTorch reference's surprising signature:
        // norm1 operates on x (the id_embedding side, kv_dim wide)
        // norm2 operates on latents (the image tokens, dim wide)
        // to_q  consumes latents (dim -> inner_dim)
        // to_kv consumes x       (kv_dim -> 2*inner_dim)
        // to_out projects        (inner_dim -> dim)
        blocks["norm1"]  = std::shared_ptr<GGMLBlock>(new LayerNorm(kv_dim));
        blocks["norm2"]  = std::shared_ptr<GGMLBlock>(new LayerNorm(dim));
        blocks["to_q"]   = std::shared_ptr<GGMLBlock>(new Linear(dim,    inner_dim,     /*bias=*/false));
        blocks["to_kv"]  = std::shared_ptr<GGMLBlock>(new Linear(kv_dim, inner_dim * 2, /*bias=*/false));
        blocks["to_out"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, dim,        /*bias=*/false));
    }

    /**
     * Compute: residual_to_image = PerceiverAttentionCA(id_embedding, image_tokens)
     *
     * Inputs:
     *   id_embedding  [N, n_id_tokens=32, kv_dim=2048]
     *   image_tokens  [N, n_img_tokens,  dim=3072]
     *
     * Returns:
     *   [N, n_img_tokens, dim=3072]  -- to be added to image_tokens by the caller,
     *                                  scaled by id_weight.
     */
    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor*       id_embedding,
                         ggml_tensor*       image_tokens) {
        auto norm1  = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
        auto norm2  = std::dynamic_pointer_cast<LayerNorm>(blocks["norm2"]);
        auto to_q   = std::dynamic_pointer_cast<Linear>(blocks["to_q"]);
        auto to_kv  = std::dynamic_pointer_cast<Linear>(blocks["to_kv"]);
        auto to_out = std::dynamic_pointer_cast<Linear>(blocks["to_out"]);

        // Normalize each input on its own dim. The PyTorch reference normalizes
        // x (id_embedding) and `latents` (image_tokens) separately, then uses
        // latents for Q and x for K/V -- mind the unusual cross-attention shape.
        ggml_tensor* x_normed   = norm1->forward(ctx, id_embedding);    // [N, 32, 2048]
        ggml_tensor* lat_normed = norm2->forward(ctx, image_tokens);    // [N, T_img, 3072]

        // Projections. to_q : 3072 -> 2048 ; to_kv : 2048 -> 4096 (k concat v).
        ggml_tensor* q  = to_q->forward(ctx, lat_normed);   // [N, T_img, 2048]
        ggml_tensor* kv = to_kv->forward(ctx, x_normed);    // [N, 32,    4096]

        // Split KV into K (first inner_dim of last axis) and V (second
        // inner_dim). ggml_view_3d gives strided views without copying;
        // ggml_cont materializes them so ggml_ext_attention_ext sees
        // contiguous tensors.
        ggml_tensor* k = ggml_view_3d(ctx->ggml_ctx, kv,
                                       inner_dim, kv->ne[1], kv->ne[2],
                                       kv->nb[1], kv->nb[2],
                                       /*offset=*/0);                              // [N, 32, 2048]
        ggml_tensor* v = ggml_view_3d(ctx->ggml_ctx, kv,
                                       inner_dim, kv->ne[1], kv->ne[2],
                                       kv->nb[1], kv->nb[2],
                                       /*offset=*/inner_dim * ggml_element_size(kv)); // [N, 32, 2048]
        k = ggml_cont(ctx->ggml_ctx, k);
        v = ggml_cont(ctx->ggml_ctx, v);

        // Standard multi-head attention. ggml_ext_attention_ext expects
        // [N, n_token, embed_dim] and reshapes into heads internally.
        // n_head = heads (=16), per-head dim = inner_dim / heads (=128).
        ggml_tensor* attn_out = ggml_ext_attention_ext(
            ctx->ggml_ctx, ctx->backend,
            q, k, v,
            heads,
            /*mask=*/nullptr,
            /*diag_mask_inf=*/false);  // [N, T_img, inner_dim=2048]

        // Project back to image-token width (3072).
        ggml_tensor* out = to_out->forward(ctx, attn_out);  // [N, T_img, 3072]
        return out;
    }
};

#endif  // __PULID_HPP__
