#ifndef __LORA_HPP__
#define __LORA_HPP__

#include "ggml_extend.hpp"

#define LORA_GRAPH_SIZE 10240

struct LoraModel : public GGMLModule {
    float multiplier = 1.0f;
    std::map<std::string, struct ggml_tensor*> lora_tensors;
    std::string file_path;
    ModelLoader model_loader;
    bool load_failed = false;

    LoraModel(const std::string file_path = "")
        : file_path(file_path) {
        name = "lora";
        if (!model_loader.init_from_file(file_path)) {
            load_failed = true;
        }
    }

    size_t get_num_tensors() {
        return LORA_GRAPH_SIZE;
    }

    size_t calculate_mem_size() {
        return model_loader.cal_mem_size(NULL);
    }

    bool load_from_file(ggml_backend_t backend) {
        if (!alloc_params_buffer(backend)) {
            return false;
        }
        LOG_INFO("loading LoRA from '%s'", file_path.c_str());

        if (load_failed) {
            LOG_ERROR("init lora model loader from file failed: '%s'", file_path.c_str());
            return false;
        }

        ggml_allocr* alloc = ggml_allocr_new_from_buffer(params_buffer);

        auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool {
            const std::string& name = tensor_storage.name;

            struct ggml_tensor* real = ggml_new_tensor(params_ctx, tensor_storage.type, tensor_storage.n_dims, tensor_storage.ne);
            ggml_allocr_alloc(alloc, real);

            *dst_tensor = real;

            lora_tensors[name] = real;
            return true;
        };

        model_loader.load_tensors(on_new_tensor_cb, backend);

        LOG_DEBUG("finished loaded lora");
        ggml_allocr_free(alloc);
        return true;
    }

    struct ggml_cgraph* build_graph(std::map<std::string, struct ggml_tensor*> model_tensors) {
        // make a graph to compute all lora, expected lora and models tensors are in the same backend
        // since we are using ggml-alloc, this buffer only needs enough space to hold the ggml_tensor and ggml_cgraph structs, but not the tensor data
        static size_t buf_size = ggml_tensor_overhead() * LORA_GRAPH_SIZE + ggml_graph_overhead();
        static std::vector<uint8_t> buf(buf_size);

        struct ggml_init_params params = {
            /*.mem_size   =*/buf_size,
            /*.mem_buffer =*/buf.data(),
            /*.no_alloc   =*/true,  // the tensors will be allocated later by ggml_allocr_alloc_graph()
        };
        // LOG_DEBUG("mem_size %u ", params.mem_size);

        struct ggml_context* ctx0 = ggml_init(params);
        struct ggml_cgraph* gf    = ggml_new_graph_custom(ctx0, LORA_GRAPH_SIZE, false);

        std::set<std::string> applied_lora_tensors;
        for (auto it : model_tensors) {
            std::string k_tensor       = it.first;
            struct ggml_tensor* weight = model_tensors[it.first];

            size_t k_pos = k_tensor.find(".weight");
            if (k_pos == std::string::npos) {
                continue;
            }
            k_tensor = k_tensor.substr(0, k_pos);
            replace_all_chars(k_tensor, '.', '_');
            std::string lora_up_name   = "lora." + k_tensor + ".lora_up.weight";
            std::string lora_down_name = "lora." + k_tensor + ".lora_down.weight";
            std::string alpha_name     = "lora." + k_tensor + ".alpha";
            std::string scale_name     = "lora." + k_tensor + ".scale";

            ggml_tensor* lora_up   = NULL;
            ggml_tensor* lora_down = NULL;

            if (lora_tensors.find(lora_up_name) != lora_tensors.end()) {
                lora_up = lora_tensors[lora_up_name];
            }

            if (lora_tensors.find(lora_down_name) != lora_tensors.end()) {
                lora_down = lora_tensors[lora_down_name];
            }

            if (lora_up == NULL || lora_down == NULL) {
                continue;
            }

            applied_lora_tensors.insert(lora_up_name);
            applied_lora_tensors.insert(lora_down_name);
            applied_lora_tensors.insert(alpha_name);
            applied_lora_tensors.insert(scale_name);

            // calc_cale
            int64_t dim       = lora_down->ne[ggml_n_dims(lora_down) - 1];
            float scale_value = 1.0f;
            if (lora_tensors.find(scale_name) != lora_tensors.end()) {
                scale_value = ggml_backend_tensor_get_f32(lora_tensors[scale_name]);
            } else if (lora_tensors.find(alpha_name) != lora_tensors.end()) {
                float alpha = ggml_backend_tensor_get_f32(lora_tensors[alpha_name]);
                scale_value = alpha / dim;
            }
            scale_value *= multiplier;

            // flat lora tensors to multiply it
            int64_t lora_up_rows   = lora_up->ne[ggml_n_dims(lora_up) - 1];
            lora_up                = ggml_reshape_2d(ctx0, lora_up, ggml_nelements(lora_up) / lora_up_rows, lora_up_rows);
            int64_t lora_down_rows = lora_down->ne[ggml_n_dims(lora_down) - 1];
            lora_down              = ggml_reshape_2d(ctx0, lora_down, ggml_nelements(lora_down) / lora_down_rows, lora_down_rows);

            // ggml_mul_mat requires tensor b transposed
            lora_down                  = ggml_cont(ctx0, ggml_transpose(ctx0, lora_down));
            struct ggml_tensor* updown = ggml_mul_mat(ctx0, lora_up, lora_down);
            updown                     = ggml_cont(ctx0, ggml_transpose(ctx0, updown));
            updown                     = ggml_reshape(ctx0, updown, weight);
            GGML_ASSERT(ggml_nelements(updown) == ggml_nelements(weight));
            updown = ggml_scale_inplace(ctx0, updown, scale_value);
            ggml_tensor* final_weight;
            // if (weight->type != GGML_TYPE_F32 && weight->type != GGML_TYPE_F16) {
            //     final_weight = ggml_new_tensor(ctx0, GGML_TYPE_F32, weight->n_dims, weight->ne);
            //     final_weight = ggml_cpy_inplace(ctx0, weight, final_weight);
            //     final_weight = ggml_add_inplace(ctx0, final_weight, updown);
            //     final_weight = ggml_cpy_inplace(ctx0, final_weight, weight);
            // } else {
            //     final_weight = ggml_add_inplace(ctx0, weight, updown);
            // }
            final_weight = ggml_add_inplace(ctx0, weight, updown);  // apply directly
            ggml_build_forward_expand(gf, final_weight);
        }

        for (auto& kv : lora_tensors) {
            if (applied_lora_tensors.find(kv.first) == applied_lora_tensors.end()) {
                LOG_WARN("unused lora tensor %s", kv.first.c_str());
            }
        }

        return gf;
    }

    void alloc_compute_buffer(std::map<std::string, struct ggml_tensor*> model_tensors) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(model_tensors);
        };
        GGMLModule::alloc_compute_buffer(get_graph);
    }

    void apply(std::map<std::string, struct ggml_tensor*> model_tensors, int n_threads) {
        alloc_compute_buffer(model_tensors);

        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(model_tensors);
        };
        GGMLModule::compute(get_graph, n_threads);
    }
};

#endif  // __LORA_HPP__