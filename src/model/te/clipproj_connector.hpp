#ifndef __SD_MODEL_TE_CLIPPROJ_CONNECTOR_HPP__
#define __SD_MODEL_TE_CLIPPROJ_CONNECTOR_HPP__

#include "core/ggml_runner.h"
#include "model/common/ggml_block.hpp"

struct MLP : public UnaryBlock {
    MLP(int64_t input_dim, int64_t hidden_dim, int64_t output_dim) {
        blocks["0"] = std::shared_ptr<GGMLBlock>(new Linear(input_dim, hidden_dim));
        blocks["2"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_dim, output_dim));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        auto layer0 = std::dynamic_pointer_cast<Linear>(blocks["0"]);
        auto layer2 = std::dynamic_pointer_cast<Linear>(blocks["2"]);

        x = layer0->forward(ctx, x);
        x = ggml_ext_gelu(ctx, x, true);
        x = layer2->forward(ctx, x);
        return x;
    }
};

struct ClipProjConnector : public UnaryBlock {
    int64_t input_dim  = 2560;
    int64_t output_dim = 5120;
    bool has_linear    = true;
    bool has_mlp       = false;
    bool has_sink      = true;

    int64_t mlp_hidden = 0;

    void config_from_weights(const String2TensorStorage& tensor_storage_map = {}) {
        static constexpr const char* prefix = "text_encoders.connector";

        has_linear = false;
        has_mlp    = false;
        has_sink   = false;
        for (const auto& [name, tensor_storage] : tensor_storage_map) {
            if (!starts_with(name, prefix)) {
                continue;
            }
            if (name.find("W") != std::string::npos) {
                has_linear = true;
            }
            if (name.find("mlp.0.bias") != std::string::npos) {
                has_mlp    = true;
                mlp_hidden = tensor_storage.ne[0];
            }
            // optional in the original code, but always present in the model weights...
            if (name.find("sink_out") != std::string::npos) {
                has_sink = true;
            }
        }
    }

    ClipProjConnector(int64_t input_dim = 2560, int64_t output_dim = 5120, const String2TensorStorage& tensor_storage_map = {}) : input_dim(input_dim), output_dim(output_dim) {
        config_from_weights(tensor_storage_map);
        if (has_mlp) {
            if (mlp_hidden <= 0) {
                throw std::runtime_error("Cannot find hidden size for ClipProj mlp");
            }
            blocks["mlp"] = std::shared_ptr<GGMLBlock>(new MLP(input_dim, mlp_hidden, output_dim));
        }
    }

    void init_params(ggml_context* ctx,
                     const String2TensorStorage& tensor_storage_map = {},
                     const std::string prefix                       = "") override {
        if (has_linear) {
            params["W"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, output_dim, input_dim);
        }
        for (const auto* name : {"mean_in", "std_in"}) {
            params[name] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, input_dim);
        }
        for (const auto* name : {"mean_out", "std_out"}) {
            params[name] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, output_dim);
        }
        if (has_sink) {
            params["sink_out"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, output_dim);
        }
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        auto x_std = ggml_div(ctx->ggml_ctx, ggml_sub(ctx->ggml_ctx, x, params["mean_in"]), params["std_in"]);
        if (has_linear) {
            auto projW = params["W"];
            projW      = ggml_cont(ctx->ggml_ctx, ggml_transpose(ctx->ggml_ctx, projW));
            x          = ggml_mul_mat(ctx->ggml_ctx, projW, x_std);
        }
        if (has_mlp) {
            auto mlp   = std::dynamic_pointer_cast<MLP>(blocks["mlp"]);
            auto x_mlp = mlp->forward(ctx, x_std);
            x          = has_linear ? ggml_add_inplace(ctx->ggml_ctx, x, x_mlp) : x_mlp;
        }
        x = ggml_add(ctx->ggml_ctx, ggml_mul(ctx->ggml_ctx, x, params["std_out"]), params["mean_out"]);
        if (has_sink) {
            auto sink = ggml_repeat(ctx->ggml_ctx, params["sink_out"], ggml_ext_slice(ctx, x, 1, 0, 1));
            if (x->ne[1] == 1) {
                return sink;
            }
            return ggml_concat(ctx->ggml_ctx, sink, ggml_ext_slice(ctx, x, 1, 1, x->ne[1]), 1);
        } else {
            return x;
        }
    }
};

struct ClipProjConnectorRunner : public GGMLRunner {
    ClipProjConnector model;
    static constexpr const char* prefix = "text_encoders.connector";

    ClipProjConnectorRunner(ggml_backend_t backend,
                            int64_t input_dim,
                            int64_t output_dim,
                            const String2TensorStorage& tensor_storage_map,
                            std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : GGMLRunner(backend, weight_manager), model(input_dim, output_dim, tensor_storage_map) {
        model.init(params_ctx, tensor_storage_map, prefix);
    }

    std::string get_desc() override {
        return "clipproj_text_connector";
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
        model.get_param_tensors(tensors, prefix);
    }

    sd::Tensor<float> compute(int n_threads, const sd::Tensor<float>& hidden_states) {
        auto get_graph = [&]() {
            auto gf         = ggml_new_graph(compute_ctx);
            auto x          = make_input(hidden_states);
            auto runner_ctx = get_context();
            ggml_build_forward_expand(gf, model.forward(&runner_ctx, x));
            return gf;
        };
        return take_or_empty(GGMLRunner::compute(get_graph, n_threads, false));
    }
};

#endif  // __SD_MODEL_TE_CLIPPROJ_CONNECTOR_HPP__
