#ifndef __QWEN3_HPP__
#define __QWEN3_HPP__

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "ggml_extend.hpp"
#include "json.hpp"

namespace Qwen3 {
    constexpr int QWEN3_GRAPH_SIZE = 10240;

    struct Qwen3MLP : public GGMLBlock {
    public:
        Qwen3MLP(int64_t hidden_size, int64_t intermediate_size) {
            blocks["gate_proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, intermediate_size, false));
            blocks["up_proj"]   = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, intermediate_size, false));
            blocks["down_proj"] = std::shared_ptr<GGMLBlock>(new Linear(intermediate_size, hidden_size, false));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
            auto gate_proj = std::dynamic_pointer_cast<Linear>(blocks["gate_proj"]);
            auto up_proj   = std::dynamic_pointer_cast<Linear>(blocks["up_proj"]);
            auto down_proj = std::dynamic_pointer_cast<Linear>(blocks["down_proj"]);

            auto h = gate_proj->forward(ctx, x);
            h      = ggml_silu_inplace(ctx->ggml_ctx, h);
            h      = ggml_mul_inplace(ctx->ggml_ctx, h, up_proj->forward(ctx, x));
            h      = down_proj->forward(ctx, h);
            return h;
        }
    };

    struct Qwen3Attention : public GGMLBlock {
    protected:
        int64_t head_dim;
        int64_t num_heads;
        int64_t num_kv_heads;
        float rope_theta;

    public:
        Qwen3Attention(int64_t hidden_size,
                       int64_t num_heads,
                       int64_t num_kv_heads,
                       int64_t head_dim = 128,
                       float rope_theta = 1000000.f)
            : head_dim(head_dim), num_heads(num_heads), num_kv_heads(num_kv_heads), rope_theta(rope_theta) {
            blocks["q_proj"]   = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, num_heads * head_dim, false));
            blocks["k_proj"]   = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, num_kv_heads * head_dim, false));
            blocks["v_proj"]   = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, num_kv_heads * head_dim, false));
            blocks["o_proj"]   = std::shared_ptr<GGMLBlock>(new Linear(num_heads * head_dim, hidden_size, false));
            blocks["q_norm"]   = std::shared_ptr<GGMLBlock>(new RMSNorm(head_dim, 1e-6f));
            blocks["k_norm"]   = std::shared_ptr<GGMLBlock>(new RMSNorm(head_dim, 1e-6f));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* input_pos) {
            int64_t n_token = x->ne[1];
            int64_t N       = x->ne[2];
            auto q_proj     = std::dynamic_pointer_cast<Linear>(blocks["q_proj"]);
            auto k_proj     = std::dynamic_pointer_cast<Linear>(blocks["k_proj"]);
            auto v_proj     = std::dynamic_pointer_cast<Linear>(blocks["v_proj"]);
            auto out_proj   = std::dynamic_pointer_cast<Linear>(blocks["o_proj"]);
            auto q_norm     = std::dynamic_pointer_cast<RMSNorm>(blocks["q_norm"]);
            auto k_norm     = std::dynamic_pointer_cast<RMSNorm>(blocks["k_norm"]);

            auto q = q_proj->forward(ctx, x);
            auto k = k_proj->forward(ctx, x);
            auto v = v_proj->forward(ctx, x);

            q = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim, num_heads, n_token, N);
            k = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim, num_kv_heads, n_token, N);
            v = ggml_reshape_4d(ctx->ggml_ctx, v, head_dim, num_kv_heads, n_token, N);

            q = q_norm->forward(ctx, q);
            k = k_norm->forward(ctx, k);

            q = ggml_rope_ext(ctx->ggml_ctx, q, input_pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 128000, rope_theta, 1.f, 0.f, 1.f, 32.f, 1.f);
            k = ggml_rope_ext(ctx->ggml_ctx, k, input_pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 128000, rope_theta, 1.f, 0.f, 1.f, 32.f, 1.f);

            q = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, q, 0, 2, 1, 3));
            q = ggml_reshape_3d(ctx->ggml_ctx, q, q->ne[0], q->ne[1], q->ne[2] * q->ne[3]);

            k = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, k, 0, 2, 1, 3));
            k = ggml_reshape_3d(ctx->ggml_ctx, k, k->ne[0], k->ne[1], k->ne[2] * k->ne[3]);

            x = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, num_heads, nullptr, true, true, false);

            x = out_proj->forward(ctx, x);
            return x;
        }
    };

    struct Qwen3Block : public GGMLBlock {
    public:
        Qwen3Block(int64_t hidden_size,
                   int64_t intermediate_size,
                   int64_t num_heads,
                   int64_t num_kv_heads,
                   int64_t head_dim  = 128,
                   float rope_theta  = 1000000.f,
                   float eps         = 1e-6f) {
            blocks["self_attn"]                = std::shared_ptr<GGMLBlock>(new Qwen3Attention(hidden_size, num_heads, num_kv_heads, head_dim, rope_theta));
            blocks["mlp"]                      = std::shared_ptr<GGMLBlock>(new Qwen3MLP(hidden_size, intermediate_size));
            blocks["input_layernorm"]          = std::shared_ptr<GGMLBlock>(new RMSNorm(hidden_size, eps));
            blocks["post_attention_layernorm"] = std::shared_ptr<GGMLBlock>(new RMSNorm(hidden_size, eps));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* input_pos) {
            auto self_attn                = std::dynamic_pointer_cast<Qwen3Attention>(blocks["self_attn"]);
            auto mlp                      = std::dynamic_pointer_cast<Qwen3MLP>(blocks["mlp"]);
            auto input_layernorm          = std::dynamic_pointer_cast<RMSNorm>(blocks["input_layernorm"]);
            auto post_attention_layernorm = std::dynamic_pointer_cast<RMSNorm>(blocks["post_attention_layernorm"]);

            auto residual = x;
            x             = input_layernorm->forward(ctx, x);
            x             = self_attn->forward(ctx, x, input_pos);
            x             = ggml_add_inplace(ctx->ggml_ctx, x, residual);

            residual = x;
            x        = post_attention_layernorm->forward(ctx, x);
            x        = mlp->forward(ctx, x);
            x        = ggml_add_inplace(ctx->ggml_ctx, x, residual);

            return x;
        }
    };

    struct Qwen3TextModel : public GGMLBlock {
    protected:
        int64_t num_layers;

    public:
        Qwen3TextModel(int64_t num_layers,
                       int64_t vocab_size,
                       int64_t hidden_size,
                       int64_t intermediate_size,
                       int64_t num_heads,
                       int64_t num_kv_heads,
                       int64_t head_dim  = 128,
                       float rope_theta  = 1000000.f,
                       float eps         = 1e-6f)
            : num_layers(num_layers) {
            blocks["embed_tokens"] = std::shared_ptr<GGMLBlock>(new Embedding(vocab_size, hidden_size));
            for (int i = 0; i < num_layers; i++) {
                blocks["layers." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new Qwen3Block(hidden_size,
                                                                                                   intermediate_size,
                                                                                                   num_heads,
                                                                                                   num_kv_heads,
                                                                                                   head_dim,
                                                                                                   rope_theta,
                                                                                                   eps));
            }
            blocks["norm"] = std::shared_ptr<GGMLBlock>(new RMSNorm(hidden_size, eps));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* input_ids,
                                    struct ggml_tensor* input_pos) {
            auto embed_tokens = std::dynamic_pointer_cast<Embedding>(blocks["embed_tokens"]);
            auto norm         = std::dynamic_pointer_cast<RMSNorm>(blocks["norm"]);

            auto x = embed_tokens->forward(ctx, input_ids);

            for (int i = 0; i < num_layers; i++) {
                auto block = std::dynamic_pointer_cast<Qwen3Block>(blocks["layers." + std::to_string(i)]);
                x          = block->forward(ctx, x, input_pos);
            }

            x = norm->forward(ctx, x);
            return x;
        }
    };

    struct Qwen3Params {
        int64_t num_layers        = 36;
        int64_t hidden_size       = 2560;
        int64_t intermediate_size = 9728;
        int64_t num_heads         = 32;
        int64_t num_kv_heads      = 8;
        int64_t head_dim          = 128;
        int64_t vocab_size        = 151936;
        float rope_theta          = 1000000.f;
        float rms_norm_eps        = 1e-06f;
    };

    struct Qwen3 : public GGMLBlock {
        Qwen3Params params;

    public:
        Qwen3() {}
        Qwen3(Qwen3Params params)
            : params(params) {
            blocks["model"] = std::shared_ptr<GGMLBlock>(new Qwen3TextModel(params.num_layers,
                                                                             params.vocab_size,
                                                                             params.hidden_size,
                                                                             params.intermediate_size,
                                                                             params.num_heads,
                                                                             params.num_kv_heads,
                                                                             params.head_dim,
                                                                             params.rope_theta,
                                                                             params.rms_norm_eps));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* input_ids,
                                    struct ggml_tensor* input_pos) {
            auto model = std::dynamic_pointer_cast<Qwen3TextModel>(blocks["model"]);
            auto x     = model->forward(ctx, input_ids, input_pos);
            return x;
        }
    };

    struct Qwen3Runner : public GGMLRunner {
        Qwen3Params params;
        Qwen3 model;
        std::vector<int> input_pos_vec;

        Qwen3Runner(ggml_backend_t backend,
                    bool offload_params_to_cpu,
                    const String2TensorStorage& tensor_storage_map,
                    const std::string prefix)
            : GGMLRunner(backend, offload_params_to_cpu) {
            model = Qwen3(params);
            model.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "qwen3";
        }

        void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
            model.get_param_tensors(tensors, prefix);
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* input_ids,
                                    struct ggml_tensor* input_pos) {
            auto hidden_states = model.forward(ctx, input_ids, input_pos);
            return hidden_states;
        }

        struct ggml_cgraph* build_graph(struct ggml_tensor* input_ids) {
            struct ggml_cgraph* gf = ggml_new_graph(compute_ctx);

            input_ids = to_backend(input_ids);

            int64_t n_tokens = input_ids->ne[0];
            input_pos_vec.resize(n_tokens);
            for (int i = 0; i < n_tokens; ++i) {
                input_pos_vec[i] = i;
            }

            auto input_pos = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, n_tokens);
            set_backend_tensor_data(input_pos, input_pos_vec.data());

            auto runner_ctx = get_context();

            struct ggml_tensor* hidden_states = forward(&runner_ctx, input_ids, input_pos);

            ggml_build_forward_expand(gf, hidden_states);

            return gf;
        }

        void compute(const int n_threads,
                     struct ggml_tensor* input_ids,
                     ggml_tensor** output,
                     ggml_context* output_ctx = nullptr) {
            auto get_graph = [&]() -> struct ggml_cgraph* {
                return build_graph(input_ids);
            };
            GGMLRunner::compute(get_graph, n_threads, true, output, output_ctx);
        }
    };

};

#endif
