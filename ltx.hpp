#ifndef __LTX_HPP__
#define __LTX_HPP__

#include "ggml_extend.hpp"
#include "model.h"

#define LTX_GRAPH_SIZE 10240
namespace Ltx {

    struct Mlp : public GGMLBlock {
    public:
        Mlp(int64_t in_features,
            int64_t hidden_features = -1,
            int64_t out_features    = -1,
            bool bias               = true) {
            // act_layer is always lambda: nn.GELU(approximate="tanh")
            // norm_layer is always None
            // use_conv is always False
            if (hidden_features == -1) {
                hidden_features = in_features;
            }
            if (out_features == -1) {
                out_features = in_features;
            }
            blocks["net.0.proj"] = std::shared_ptr<GGMLBlock>(new Linear(in_features, hidden_features, bias));
            blocks["net.2"]      = std::shared_ptr<GGMLBlock>(new Linear(hidden_features, out_features, bias));
        }

        struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
            // x: [N, n_token, in_features]
            auto fc1 = std::dynamic_pointer_cast<Linear>(blocks["net.0.proj"]);
            auto fc2 = std::dynamic_pointer_cast<Linear>(blocks["net.2"]);

            x = fc1->forward(ctx, x);
            x = ggml_gelu_inplace(ctx, x);
            x = fc2->forward(ctx, x);
            return x;
        }
    };

    struct EmbedProjection : public GGMLBlock {
        // Embeds scalar timesteps into vector representations.
    public:
        EmbedProjection(int64_t hidden_size,
                        int64_t embedding_size = 256) {
            blocks["linear_1"] = std::shared_ptr<GGMLBlock>(new Linear(embedding_size, hidden_size, true, true));
            blocks["linear_2"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size, true, true));
        }

        struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* t) {
            // t: [N, ]
            // return: [N, hidden_size]
            auto mlp_0 = std::dynamic_pointer_cast<Linear>(blocks["linear_1"]);
            auto mlp_2 = std::dynamic_pointer_cast<Linear>(blocks["linear_2"]);

            auto t_emb = mlp_0->forward(ctx, t);
            t_emb      = ggml_silu_inplace(ctx, t_emb);
            t_emb      = mlp_2->forward(ctx, t_emb);
            return t_emb;
        }
    };

    struct AdaLnSingleEmbedder : public GGMLBlock {
    protected:
        int64_t frequency_embedding_size;

    public:
        AdaLnSingleEmbedder(int64_t hidden_size, int64_t frequency_embedding_size = 256)
            : frequency_embedding_size(frequency_embedding_size) {
            blocks["timestep_embedder"] = std::shared_ptr<GGMLBlock>(new EmbedProjection(hidden_size, frequency_embedding_size));
        }
        struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* t) {
            auto t_embedder = std::dynamic_pointer_cast<EmbedProjection>(blocks["timestep_embedder"]);
            auto t_freq     = ggml_nn_timestep_embedding(ctx, t, frequency_embedding_size);  // [N, frequency_embedding_size]

            return t_embedder->forward(ctx, t_freq);
        }
    };

    struct AdaLnSingle : public GGMLBlock {
        // Embeds scalar timesteps into vector representations.
    public:
        AdaLnSingle(int64_t hidden_size, int64_t frequency_embedding_size = 256, int64_t num_scales_shifts = 6) {
            blocks["emb"]    = std::shared_ptr<GGMLBlock>(new AdaLnSingleEmbedder(hidden_size, frequency_embedding_size));
            blocks["linear"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size * num_scales_shifts, true, true));
        }

        struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* t) {
            auto embedder = std::dynamic_pointer_cast<AdaLnSingleEmbedder>(blocks["emb"]);
            auto linear   = std::dynamic_pointer_cast<Linear>(blocks["linear"]);

            auto embeds = embedder->forward(ctx, t);
            embeds      = ggml_silu_inplace(ctx, embeds);
            return linear->forward(ctx, embeds);
        }
    };

    class RMSNorm : public UnaryBlock {
    protected:
        int64_t hidden_size;
        float eps;

        void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, std::string prefix = "") {
            enum ggml_type wtype = GGML_TYPE_F32;  //(tensor_types.find(prefix + "weight") != tensor_types.end()) ? tensor_types[prefix + "weight"] : GGML_TYPE_F32;
            params["weight"]     = ggml_new_tensor_1d(ctx, wtype, hidden_size);
        }

    public:
        RMSNorm(int64_t hidden_size,
                float eps = 1e-06f)
            : hidden_size(hidden_size),
              eps(eps) {}

        struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
            struct ggml_tensor* w = params["weight"];
            x                     = ggml_rms_norm(ctx, x, eps);
            x                     = ggml_mul(ctx, x, w);
            return x;
        }
    };

    class Attention : public GGMLBlock {
    public:
        int64_t num_heads;
        std::string qk_norm;

    public:
        Attention(int64_t dim,
                  int64_t num_heads   = 8,
                  std::string qk_norm = "",
                  bool qkv_bias       = false)
            : num_heads(num_heads), qk_norm(qk_norm) {

            blocks["to_q"]   = std::shared_ptr<GGMLBlock>(new Linear(dim, dim, qkv_bias));
            blocks["to_k"]   = std::shared_ptr<GGMLBlock>(new Linear(dim, dim, qkv_bias));
            blocks["to_v"]   = std::shared_ptr<GGMLBlock>(new Linear(dim, dim, qkv_bias));
            blocks["to_out.0"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim, qkv_bias));

            blocks["k_norm"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim, 1.0e-6));
            blocks["q_norm"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim, 1.0e-6));
        }

        std::vector<struct ggml_tensor*> pre_attention(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* y) {
            auto q_proj = std::dynamic_pointer_cast<Linear>(blocks["to_q"]);
            auto k_proj = std::dynamic_pointer_cast<Linear>(blocks["to_k"]);
            auto v_proj = std::dynamic_pointer_cast<Linear>(blocks["to_v"]);

            auto q = q_proj->forward(ctx, x);
            auto k = k_proj->forward(ctx, y);
            auto v = v_proj->forward(ctx, y);

            {
                auto q_norm = std::dynamic_pointer_cast<UnaryBlock>(blocks["q_norm"]);
                auto k_norm = std::dynamic_pointer_cast<UnaryBlock>(blocks["k_norm"]);
                q           = q_norm->forward(ctx, q);
                k           = k_norm->forward(ctx, k);
            }

            q = ggml_reshape_3d(ctx, q, q->ne[0] * q->ne[1], q->ne[2], q->ne[3]);  // [N, n_token, n_head*d_head]
            k = ggml_reshape_3d(ctx, k, k->ne[0] * k->ne[1], k->ne[2], k->ne[3]);  // [N, n_token, n_head*d_head]

            return {q, k, v};
        }

        struct ggml_tensor* post_attention(struct ggml_context* ctx, struct ggml_tensor* x) {
            auto out_proj = std::dynamic_pointer_cast<Linear>(blocks["to_out.0"]);

            x = out_proj->forward(ctx, x);  // [N, n_token, dim]
            return x;
        }

        // x: [N, n_token, dim]
        struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* y) {
            auto qkv = pre_attention(ctx, x, y);
            x        = ggml_nn_attention_ext(ctx, qkv[0], qkv[1], qkv[2], num_heads);  // [N, n_token, dim]
            x        = post_attention(ctx, x);                                         // [N, n_token, dim]
            return x;
        }
    };

    __STATIC_INLINE__ struct ggml_tensor* modulate(struct ggml_context* ctx,
                                                   struct ggml_tensor* x,
                                                   struct ggml_tensor* shift,
                                                   struct ggml_tensor* scale) {
        // x: [N, L, C]
        // scale: [N, C]
        // shift: [N, C]
        scale = ggml_reshape_3d(ctx, scale, scale->ne[0], 1, scale->ne[1]);  // [N, 1, C]
        shift = ggml_reshape_3d(ctx, shift, shift->ne[0], 1, shift->ne[1]);  // [N, 1, C]
        x     = ggml_add(ctx, x, ggml_mul(ctx, x, scale));
        x     = ggml_add(ctx, x, shift);
        return x;
    }

    struct TransformerBlock : public GGMLBlock {
    public:
        int64_t hidden_size;

    public:
        void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, std::string prefix = "") {
            enum ggml_type wtype        = (tensor_types.find(prefix + "scale_shift_table") != tensor_types.end()) ? tensor_types[prefix + "scale_shift_table"] : GGML_TYPE_F32;
            params["scale_shift_table"] = ggml_new_tensor_2d(ctx, wtype, hidden_size, 6);
            ;
        }
        TransformerBlock(int64_t hidden_size,
                         int64_t num_heads,
                         float mlp_ratio     = 4.0,
                         std::string qk_norm = "",
                         bool qkv_bias       = false)
            : hidden_size(hidden_size) {
            blocks["attn1"] = std::shared_ptr<GGMLBlock>(new Attention(hidden_size, num_heads, qk_norm, qkv_bias));
            blocks["attn2"] = std::shared_ptr<GGMLBlock>(new Attention(hidden_size, num_heads, qk_norm, qkv_bias));

            blocks["ff"] = std::shared_ptr<GGMLBlock>(new Mlp(hidden_size, hidden_size * mlp_ratio, hidden_size, true));
        }

        struct ggml_tensor* forward(struct ggml_context* ctx,
                                    struct ggml_tensor* context,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* c,
                                    struct ggml_tensor* shift_scale) {
            struct ggml_tensor* ss_table = params["scale_shift_table"];  // [hidden_size, 6]

            auto ss = ggml_add(ctx, shift_scale, ss_table);

            int64_t offset = ss->nb[0] * ss->ne[0];
            // TODO: Is that the right order?
            // assuming [scale0, scale2, shift0, shift2, scale1, scale3] from Pixart alpha paper

            auto scale_0 = ggml_view_1d(ctx, ss, ss->ne[0], offset * 0);
            auto scale_2 = ggml_view_1d(ctx, ss, ss->ne[0], offset * 1);

            auto shift_0 = ggml_view_1d(ctx, ss, ss->ne[0], offset * 2);
            auto shift_2 = ggml_view_1d(ctx, ss, ss->ne[0], offset * 3);

            auto scale_1 = ggml_view_1d(ctx, ss, ss->ne[0], offset * 4);
            auto scale_3 = ggml_view_1d(ctx, ss, ss->ne[0], offset * 5);

            auto attn1 = std::dynamic_pointer_cast<Attention>(blocks["attn1"]);
            auto attn2 = std::dynamic_pointer_cast<Attention>(blocks["attn2"]);

            auto ff = std::dynamic_pointer_cast<Mlp>(blocks["ff"]);

            x = ggml_add(ctx, x, ggml_mul(ctx, x, scale_0));
            x = ggml_add(ctx, x, shift_0);

            x = attn1->forward(ctx, x, x);
            x = ggml_add(ctx, x, ggml_mul(ctx, x, scale_1));

            x = attn2->forward(ctx, x, c);
            x = ggml_add(ctx, x, ggml_mul(ctx, x, scale_2));
            x = ggml_add(ctx, x, shift_2);

            x = ff->forward(ctx, x);
            x = ggml_add(ctx, x, ggml_mul(ctx, x, scale_3));

            return x;
        }
    };

    struct LTXv : public GGMLBlock {
        // TODO: This seems to be closely related to Pixart Alpha models
        // Support both here?
    protected:
        int64_t input_size         = -1;
        int64_t patch_size         = 2;
        int64_t in_channels        = 128;
        int64_t depth              = 24;
        float mlp_ratio            = 4.0f;
        int64_t adm_in_channels    = 2048;
        int64_t out_channels       = 128;
        int64_t pos_embed_max_size = 192;
        int64_t num_patchs         = 36864;  // 192 * 192
        int64_t context_size       = 4096;
        int64_t hidden_size;

        void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, std::string prefix = "") {
            enum ggml_type wtype        = GGML_TYPE_F32;                                   //(tensor_types.find(prefix + "pos_embed") != tensor_types.end()) ? tensor_types[prefix + "pos_embed"] : GGML_TYPE_F32;
            params["scale_shift_table"] = ggml_new_tensor_2d(ctx, wtype, hidden_size, 2);  // scales and shifts for last layer
        }

    public:
        LTXv(std::map<std::string, enum ggml_type>& tensor_types) {
            // read tensors from tensor_types
            for (auto pair : tensor_types) {
                std::string tensor_name = pair.first;
                if (tensor_name.find("model.diffusion_model.") == std::string::npos)
                    continue;
                size_t jb = tensor_name.find("transformer_blocks.");
                if (jb != std::string::npos) {
                    tensor_name     = tensor_name.substr(jb);  // remove prefix
                    int block_depth = atoi(tensor_name.substr(19, tensor_name.find(".", 19)).c_str());
                    if (block_depth + 1 > depth) {
                        depth = block_depth + 1;
                    }
                }
            }

            LOG_INFO("Transformer layers: %d", depth);

            int64_t default_out_channels = in_channels;
            hidden_size                  = 2048;
            int64_t num_heads            = depth;

            blocks["patchify_proj"] = std::shared_ptr<GGMLBlock>(new Linear(in_channels, hidden_size));

            blocks["adaln_single"] = std::shared_ptr<GGMLBlock>(new AdaLnSingle(hidden_size));

            blocks["caption_projection"] = std::shared_ptr<GGMLBlock>(new EmbedProjection(hidden_size, context_size));

            for (int i = 0; i < depth; i++) {
                blocks["transformer_blocks." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new TransformerBlock(hidden_size,
                                                                                                                    num_heads,
                                                                                                                    mlp_ratio,
                                                                                                                    "rms",
                                                                                                                    true));
            }

            // params["scale_shift_table"] (in init_params())
            blocks["proj_out"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, out_channels));
            LOG_INFO("Loaded");
        }

        struct ggml_tensor* forward_core(struct ggml_context* ctx,
                                         struct ggml_tensor* x,
                                         struct ggml_tensor* c_mod,
                                         struct ggml_tensor* shift_scale,
                                         struct ggml_tensor* context,
                                         std::vector<int> skip_layers = std::vector<int>()) {
            auto final_layer_proj       = std::dynamic_pointer_cast<Linear>(blocks["proj_out"]);
            auto final_layer_modulation = params["scale_shift_table"];  // [hidden_size, 2]

            // TODO: figure out last layer modulation
            // we need to slice shift_scale : [hidden_size, 6] => [hidden_size, 2] (which columns to keep?)
            // then add to final_layer_modulation

            // auto scale = ggml_view_1d(ctx, final_layer_modulation, final_layer_modulation->ne[0], offset * 0);
            // auto shift = ggml_view_1d(ctx, final_layer_modulation, final_layer_modulation->ne[0], offset * 1);

            for (int i = 0; i < depth; i++) {
                // skip iteration if i is in skip_layers
                if (skip_layers.size() > 0 && std::find(skip_layers.begin(), skip_layers.end(), i) != skip_layers.end()) {
                    continue;
                }

                auto block = std::dynamic_pointer_cast<TransformerBlock>(blocks["transformer_blocks." + std::to_string(i)]);

                x = block->forward(ctx, context, x, c_mod, shift_scale);
            }

            x = final_layer_proj->forward(ctx, x);  // (N, T, patch_size ** 2 * out_channels)

            // TODO: before or after final proj? (probably after)
            // x = ggml_add(ctx, x, ggml_mul(ctx, x, scale));
            // x = ggml_add(ctx, x, shift);

            return x;
        }

        struct ggml_tensor* forward(struct ggml_context* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* t,
                                    struct ggml_tensor* y        = NULL,
                                    struct ggml_tensor* context  = NULL,
                                    std::vector<int> skip_layers = std::vector<int>()) {
            auto x_embedder = std::dynamic_pointer_cast<Linear>(blocks["patchify_proj"]);
            auto t_embedder = std::dynamic_pointer_cast<AdaLnSingle>(blocks["adaln_single"]);

            int64_t w = x->ne[0];
            int64_t h = x->ne[1];

            auto hidden_states = x_embedder->forward(ctx, x);  // [N, H*W, hidden_size]

            auto shift_scales = t_embedder->forward(ctx, t);                         // [hidden_size * 6]
            shift_scales      = ggml_reshape_2d(ctx, shift_scales, hidden_size, 6);  // [hidden_size, 6]

            auto c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            if (y != NULL && adm_in_channels != -1) {
                auto y_embedder = std::dynamic_pointer_cast<EmbedProjection>(blocks["caption_projection"]);

                y = y_embedder->forward(ctx, y);  // [N, hidden_size]
                c = ggml_add(ctx, c, y);
            }

            // if (context != NULL) {
            //     auto context_embedder = std::dynamic_pointer_cast<Linear>(blocks["context_embedder"]);

            //     context = context_embedder->forward(ctx, context);  // [N, L, D] aka [N, L, 1536]
            // }

            x = forward_core(ctx, x, c, shift_scales, context, skip_layers);  // (N, H*W, patch_size ** 2 * out_channels)

            // x = unpatchify(ctx, x, h, w);  // [N, C, H, W]

            return x;
        }
    };
    struct LTXRunner : public GGMLRunner {
        LTXv ltx;

        static std::map<std::string, enum ggml_type> empty_tensor_types;

        LTXRunner(ggml_backend_t backend,
                  std::map<std::string, enum ggml_type>& tensor_types = empty_tensor_types,
                  const std::string prefix                            = "")
            : GGMLRunner(backend), ltx(tensor_types) {
            ltx.init(params_ctx, tensor_types, prefix);
        }

        std::string get_desc() {
            return "ltx";
        }

        void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
            ltx.get_param_tensors(tensors, prefix);
        }

        struct ggml_cgraph* build_graph(struct ggml_tensor* x,
                                        struct ggml_tensor* timesteps,
                                        struct ggml_tensor* context,
                                        struct ggml_tensor* y,
                                        std::vector<int> skip_layers = std::vector<int>()) {
            struct ggml_cgraph* gf = ggml_new_graph_custom(compute_ctx, LTX_GRAPH_SIZE, false);

            x         = to_backend(x);
            context   = to_backend(context);
            y         = to_backend(y);
            timesteps = to_backend(timesteps);

            struct ggml_tensor* out = ltx.forward(compute_ctx,
                                                  x,
                                                  timesteps,
                                                  y,
                                                  context,
                                                  skip_layers);

            ggml_build_forward_expand(gf, out);

            return gf;
        }

        void compute(int n_threads,
                     struct ggml_tensor* x,
                     struct ggml_tensor* timesteps,
                     struct ggml_tensor* context,
                     struct ggml_tensor* y,
                     struct ggml_tensor** output     = NULL,
                     struct ggml_context* output_ctx = NULL,
                     std::vector<int> skip_layers    = std::vector<int>()) {
            // x: [N, in_channels, h, w]
            // timesteps: [N, ]
            // context: [N, max_position, hidden_size]([N, 154, 4096]) or [1, max_position, hidden_size]
            // y: [N, adm_in_channels] or [1, adm_in_channels]
            auto get_graph = [&]() -> struct ggml_cgraph* {
                return build_graph(x, timesteps, context, y, skip_layers);
            };

            GGMLRunner::compute(get_graph, n_threads, false, output, output_ctx);
        }

        void test() {
            struct ggml_init_params params;
            params.mem_size   = static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
            params.mem_buffer = NULL;
            params.no_alloc   = false;

            struct ggml_context* work_ctx = ggml_init(params);
            GGML_ASSERT(work_ctx != NULL);

            {
                // cpu f16: pass
                // cpu f32: pass
                // cuda f16: pass
                // cuda f32: pass
                auto x = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 128, 128, 16, 1);
                std::vector<float> timesteps_vec(1, 999.f);
                auto timesteps = vector_to_ggml_tensor(work_ctx, timesteps_vec);
                ggml_set_f32(x, 0.01f);
                // print_ggml_tensor(x);

                auto context = ggml_new_tensor_3d(work_ctx, GGML_TYPE_F32, 4096, 154, 1);
                ggml_set_f32(context, 0.01f);
                // print_ggml_tensor(context);

                auto y = ggml_new_tensor_2d(work_ctx, GGML_TYPE_F32, 2048, 1);
                ggml_set_f32(y, 0.01f);
                // print_ggml_tensor(y);

                struct ggml_tensor* out = NULL;

                int t0 = ggml_time_ms();
                compute(8, x, timesteps, context, y, &out, work_ctx);
                int t1 = ggml_time_ms();

                print_ggml_tensor(out);
                LOG_DEBUG("ltx test done in %dms", t1 - t0);
            }
        }

        static void load_from_file_and_test(const std::string& file_path) {
            // ggml_backend_t backend    = ggml_backend_cuda_init(0);
            ggml_backend_t backend         = ggml_backend_cpu_init();
            ggml_type model_data_type      = GGML_TYPE_F16;
            std::shared_ptr<LTXRunner> ltx = std::shared_ptr<LTXRunner>(new LTXRunner(backend));
            {
                LOG_INFO("loading from '%s'", file_path.c_str());

                ltx->alloc_params_buffer();
                std::map<std::string, ggml_tensor*> tensors;
                ltx->get_param_tensors(tensors, "model.diffusion_model");

                ModelLoader model_loader;
                if (!model_loader.init_from_file(file_path)) {
                    LOG_ERROR("init model loader from file failed: '%s'", file_path.c_str());
                    return;
                }

                bool success = model_loader.load_tensors(tensors, backend);

                if (!success) {
                    LOG_ERROR("load tensors from model loader failed");
                    return;
                }

                LOG_INFO("ltx model loaded");
            }
            ltx->test();
        }
    };
}  // namespace Flux

#endif  // __LTX_HPP__