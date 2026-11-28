#ifndef __PULID_HPP__
#define __PULID_HPP__

#include "core/ggml_extend.h"
#include "core/ggml_runner.h"
#include "model/common/block.hpp"
#include "model/common/ggml_block.hpp"

class PuLIDPerceiverAttentionCA : public GGMLBlock {
public:
    static constexpr int64_t DEFAULT_DIM      = 3072;  // Flux hidden size
    static constexpr int64_t DEFAULT_DIM_HEAD = 128;
    static constexpr int64_t DEFAULT_HEADS    = 16;
    static constexpr int64_t DEFAULT_KV_DIM   = 2048;  // PuLID ID-embedding dim

protected:
    int64_t dim;
    int64_t dim_head;
    int64_t heads;
    int64_t kv_dim;
    int64_t inner_dim;

public:
    PuLIDPerceiverAttentionCA(int64_t dim      = DEFAULT_DIM,
                              int64_t dim_head = DEFAULT_DIM_HEAD,
                              int64_t heads    = DEFAULT_HEADS,
                              int64_t kv_dim   = DEFAULT_KV_DIM)
        : dim(dim),
          dim_head(dim_head),
          heads(heads),
          kv_dim(kv_dim),
          inner_dim(dim_head * heads) {
        blocks["norm1"]  = std::shared_ptr<GGMLBlock>(new LayerNorm(kv_dim));
        blocks["norm2"]  = std::shared_ptr<GGMLBlock>(new LayerNorm(dim));
        blocks["to_q"]   = std::shared_ptr<GGMLBlock>(new Linear(dim, inner_dim, /*bias=*/false));
        blocks["to_kv"]  = std::shared_ptr<GGMLBlock>(new Linear(kv_dim, inner_dim * 2, /*bias=*/false));
        blocks["to_out"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, dim, /*bias=*/false));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* id_embedding,
                         ggml_tensor* image_tokens) {
        return attend(ctx, project_kv(ctx, id_embedding), image_tokens);
    }

    ggml_tensor* project_kv(GGMLRunnerContext* ctx, ggml_tensor* id_embedding) {
        auto norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
        auto to_kv = std::dynamic_pointer_cast<Linear>(blocks["to_kv"]);
        return to_kv->forward(ctx, norm1->forward(ctx, id_embedding));
    }

    ggml_tensor* attend(GGMLRunnerContext* ctx, ggml_tensor* kv, ggml_tensor* image_tokens) {
        auto norm2              = std::dynamic_pointer_cast<LayerNorm>(blocks["norm2"]);
        auto to_q               = std::dynamic_pointer_cast<Linear>(blocks["to_q"]);
        auto to_out             = std::dynamic_pointer_cast<Linear>(blocks["to_out"]);
        ggml_tensor* lat_normed = norm2->forward(ctx, image_tokens);

        ggml_tensor* q = to_q->forward(ctx, lat_normed);  // [N, T_img, 2048]

        ggml_tensor* k = ggml_view_3d(ctx->ggml_ctx, kv,
                                      inner_dim, kv->ne[1], kv->ne[2],
                                      kv->nb[1], kv->nb[2],
                                      /*offset=*/0);
        ggml_tensor* v = ggml_view_3d(ctx->ggml_ctx, kv,
                                      inner_dim, kv->ne[1], kv->ne[2],
                                      kv->nb[1], kv->nb[2],
                                      /*offset=*/inner_dim * ggml_element_size(kv));
        k              = ggml_cont(ctx->ggml_ctx, k);
        v              = ggml_cont(ctx->ggml_ctx, v);

        ggml_tensor* attn_out = ggml_ext_attention_ext(ctx,
                                                       q, k, v,
                                                       heads,
                                                       /*mask=*/nullptr,
                                                       /*diag_mask_inf=*/false);

        ggml_tensor* out = to_out->forward(ctx, attn_out);
        return out;
    }
};

class PuLIDRunner : public GGMLRunner {
    std::vector<PuLIDPerceiverAttentionCA> layers_;

    static std::string layer_name(size_t index) {
        return "model.diffusion_model.pulid_ca." + std::to_string(index);
    }

public:
    static constexpr int double_blocks   = 19;
    static constexpr int single_blocks   = 38;
    static constexpr int double_interval = 2;
    static constexpr int single_interval = 4;
    static constexpr int double_layers   = (double_blocks + double_interval - 1) / double_interval;
    static constexpr int layer_count     = double_layers + (single_blocks + single_interval - 1) / single_interval;

    PuLIDRunner(ggml_backend_t backend, const String2TensorStorage& weights, std::shared_ptr<RunnerWeightManager> manager)
        : GGMLRunner(backend, std::move(manager)), layers_(layer_count) {
        for (size_t index = 0; index < layers_.size(); ++index) {
            const auto name = layer_name(index);
            if (!weights.count(name + ".to_kv.weight"))
                throw std::runtime_error("missing PuLID weights for '" + name + "'");
            layers_[index].init(params_ctx, weights, name);
        }
        if (weights.count(layer_name(layers_.size()) + ".to_kv.weight"))
            throw std::runtime_error("unsupported PuLID attention layer count");
    }

    std::string get_desc() override { return "pulid"; }
    ggml_backend_t compute_backend() const { return runtime_backend; }
    size_t extra_graph_size() const {
        return layer_count * 32 + (weight_adapter ? weight_adapter->get_extra_graph_size() : 0);
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
        for (size_t index = 0; index < layers_.size(); ++index)
            layers_[index].get_param_tensors(tensors, layer_name(index));
    }

    std::vector<sd::Tensor<float>> compute_kv(int n_threads, const sd::Tensor<float>& embedding) {
        std::vector<ggml_tensor*> outputs(layers_.size());
        auto get_graph = [&]() {
            auto* graph    = new_graph_custom(1024);
            auto ctx       = get_context();
            auto* identity = make_input(embedding);
            for (size_t i = 0; i < layers_.size(); ++i) {
                outputs[i] = layers_[i].project_kv(&ctx, identity);
                ggml_set_output(outputs[i]);
                ggml_build_forward_expand(graph, outputs[i]);
            }
            return graph;
        };
        std::vector<sd::Tensor<float>> kv;
        auto read_outputs = [&]() {
            kv.reserve(layers_.size());
            for (const auto* output : outputs) {
                auto tensor = sd::make_sd_tensor_from_ggml<float>(output);
                if (tensor.empty())
                    return false;
                kv.push_back(std::move(tensor));
            }
            return true;
        };
        if (!GGMLRunner::compute(get_graph, n_threads, true, true, read_outputs).has_value())
            return {};
        return kv;
    }

    ggml_tensor* attend(GGMLRunnerContext& ctx, size_t layer, const sd::Tensor<float>& kv, ggml_tensor* image) {
        auto branch           = ctx;
        branch.weight_adapter = weight_adapter;
        branch.linear_scale   = linear_scale;
        branch.attn_scale     = attn_scale;
        auto* input           = sd::make_ggml_tensor(ctx.ggml_ctx, kv, false);
        ctx.set_backend_tensor_data(input, kv.data());
        return layers_[layer].attend(&branch, input, image);
    }
};

#endif  // __PULID_HPP__
