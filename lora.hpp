#ifndef __LORA_HPP__
#define __LORA_HPP__

#include "ggml_extend.hpp"

#define LORA_GRAPH_SIZE 10240

struct LoraModel : public GGMLRunner {
    static enum lora_t {
        REGULAR      = 0,
        DIFFUSERS    = 1,
        DIFFUSERS_2  = 2,
        DIFFUSERS_3  = 3,
        TRANSFORMERS = 4,
        LORA_TYPE_COUNT
    };

    const std::string lora_ups[LORA_TYPE_COUNT] = {
        ".lora_up",
        "_lora.up",
        ".lora_B",
        ".lora.up",
        ".lora_linear_layer.up",
    };

    const std::string lora_downs[LORA_TYPE_COUNT] = {
        ".lora_down",
        "_lora.down",
        ".lora_A",
        ".lora.down",
        ".lora_linear_layer.down",
    };

    const std::string lora_pre[LORA_TYPE_COUNT] = {
        "lora.",
        "",
        "",
        "",
        "",
    };

    const std::string* type_fingerprints = lora_ups;

    float multiplier = 1.0f;
    std::map<std::string, struct ggml_tensor*> lora_tensors;
    std::string file_path;
    ModelLoader model_loader;
    bool load_failed                = false;
    bool applied                    = false;
    std::vector<int> zero_index_vec = {0};
    ggml_tensor* zero_index         = NULL;
    enum lora_t type                = REGULAR;

    LoraModel(ggml_backend_t backend,
              const std::string& file_path = "",
              const std::string prefix     = "")
        : file_path(file_path), GGMLRunner(backend) {
        if (!model_loader.init_from_file(file_path, prefix)) {
            load_failed = true;
        }
    }

    std::string get_desc() {
        return "lora";
    }

    bool load_from_file(bool filter_tensor = false) {
        LOG_INFO("loading LoRA from '%s'", file_path.c_str());

        if (load_failed) {
            LOG_ERROR("init lora model loader from file failed: '%s'", file_path.c_str());
            return false;
        }

        bool dry_run          = true;
        auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool {
            const std::string& name = tensor_storage.name;

            if (filter_tensor && !contains(name, "lora")) {
                // LOG_INFO("skipping LoRA tesnor '%s'", name.c_str());
                return true;
            }
            // LOG_INFO("%s", name.c_str());
            for (int i = 0; i < LORA_TYPE_COUNT; i++) {
                if (name.find(type_fingerprints[i]) != std::string::npos) {
                    type = (lora_t)i;
                    break;
                }
            }
            // if (name.find(".transformer_blocks.0") != std::string::npos) {
            //     LOG_INFO("%s", name.c_str());
            // }

            if (dry_run) {
                struct ggml_tensor* real = ggml_new_tensor(params_ctx,
                                                           tensor_storage.type,
                                                           tensor_storage.n_dims,
                                                           tensor_storage.ne);
                lora_tensors[name]       = real;
            } else {
                auto real   = lora_tensors[name];
                *dst_tensor = real;
            }

            return true;
        };

        model_loader.load_tensors(on_new_tensor_cb, backend);
        alloc_params_buffer();
        // exit(0);
        dry_run = false;
        model_loader.load_tensors(on_new_tensor_cb, backend);

        LOG_DEBUG("finished loaded lora");
        return true;
    }

    ggml_tensor* to_f32(ggml_context* ctx, ggml_tensor* a) {
        auto out = ggml_reshape_1d(ctx, a, ggml_nelements(a));
        out      = ggml_get_rows(ctx, out, zero_index);
        out      = ggml_reshape(ctx, out, a);
        return out;
    }

    std::vector<std::string> to_lora_keys(std::string blk_name, SDVersion version) {
        std::vector<std::string> keys;
        size_t k_pos = blk_name.find(".weight");
        if (k_pos == std::string::npos) {
            return keys;
        }
        blk_name = blk_name.substr(0, k_pos);
        if (type == REGULAR) {
            keys.push_back(blk_name);
            // blk_name = blk_name.substr(sizeof("diffusion_model."));
            replace_all_chars(blk_name, '.', '_');
            keys.push_back(blk_name);
            return keys;
        } else if (type == DIFFUSERS || type == DIFFUSERS_2 || DIFFUSERS_3) {
            // if (sd_version_is_Flux(version)) {
            if (blk_name.find("model.diffusion_model") != std::string::npos) {
                blk_name.replace(blk_name.find("model.diffusion_model"), sizeof("model.diffusion_model") - 1, "transformer");
            }
            if (blk_name.find(".single_blocks") != std::string::npos) {
                blk_name.replace(blk_name.find(".single_blocks"), sizeof(".single_blocks") - 1, ".single_transformer_blocks");
            }
            if (blk_name.find(".double_blocks") != std::string::npos) {
                blk_name.replace(blk_name.find(".double_blocks"), sizeof(".double_blocks") - 1, ".transformer_blocks");
            }
            keys.push_back(blk_name);
            // }
        }
        // LOG_DEBUG("k_tensor %s", k_tensor.c_str());
        return keys;
    }

    struct ggml_cgraph* build_lora_graph(std::map<std::string, struct ggml_tensor*> model_tensors, SDVersion version) {
        struct ggml_cgraph* gf = ggml_new_graph_custom(compute_ctx, LORA_GRAPH_SIZE, false);

        zero_index = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, 1);
        set_backend_tensor_data(zero_index, zero_index_vec.data());
        ggml_build_forward_expand(gf, zero_index);

        std::set<std::string> applied_lora_tensors;
        for (auto it : model_tensors) {
            std::string k_tensor       = it.first;
            struct ggml_tensor* weight = model_tensors[it.first];

            // LOG_INFO("%s", k_tensor.c_str());
            std::vector<std::string> keys = to_lora_keys(k_tensor, version);
            if (keys.size() == 0)
                continue;
            for (auto& key : keys) {
                ggml_tensor* lora_up   = NULL;
                ggml_tensor* lora_down = NULL;

                std::string alpha_name     = "";
                std::string scale_name     = "";
                std::string lora_down_name = "";
                std::string lora_up_name   = "";
                // LOG_DEBUG("k_tensor %s", k_tensor.c_str());
                if (sd_version_is_flux(version)) {
                    size_t linear1    = key.find("linear1");
                    size_t linear2    = key.find("linear2");
                    size_t modulation = key.find("modulation.lin");

                    size_t txt_attn_qkv = key.find("txt_attn.qkv");
                    size_t img_attn_qkv = key.find("img_attn.qkv");

                    size_t txt_attn_proj = key.find("txt_attn.proj");
                    size_t img_attn_proj = key.find("img_attn.proj");

                    size_t txt_mlp_0 = key.find("txt_mlp.0");
                    size_t txt_mlp_2 = key.find("txt_mlp.2");
                    size_t img_mlp_0 = key.find("img_mlp.0");
                    size_t img_mlp_2 = key.find("img_mlp.2");

                    size_t txt_mod_lin = key.find("txt_mod.lin");
                    size_t img_mod_lin = key.find("img_mod.lin");

                    if (linear1 != std::string::npos) {
                        linear1--;
                        auto split_q_d_name = lora_pre[type] + key.substr(0, linear1) + ".attn.to_q" + lora_downs[type] + ".weight";
                        if (lora_tensors.find(split_q_d_name) != lora_tensors.end()) {
                            // print_ggml_tensor(it.second, true);  //[3072, 21504, 1, 1]
                            // find qkv and mlp up parts in LoRA model
                            auto split_k_d_name = lora_pre[type] + key.substr(0, linear1) + ".attn.to_k" + lora_downs[type] + ".weight";
                            auto split_v_d_name = lora_pre[type] + key.substr(0, linear1) + ".attn.to_v" + lora_downs[type] + ".weight";

                            auto split_q_u_name = lora_pre[type] + key.substr(0, linear1) + ".attn.to_q" + lora_ups[type] + ".weight";
                            auto split_k_u_name = lora_pre[type] + key.substr(0, linear1) + ".attn.to_k" + lora_ups[type] + ".weight";
                            auto split_v_u_name = lora_pre[type] + key.substr(0, linear1) + ".attn.to_v" + lora_ups[type] + ".weight";

                            auto split_m_d_name = lora_pre[type] + key.substr(0, linear1) + ".proj_mlp" + lora_downs[type] + ".weight";
                            auto split_m_u_name = lora_pre[type] + key.substr(0, linear1) + ".proj_mlp" + lora_ups[type] + ".weight";

                            ggml_tensor* lora_q_down = NULL;
                            ggml_tensor* lora_q_up   = NULL;
                            ggml_tensor* lora_k_down = NULL;
                            ggml_tensor* lora_k_up   = NULL;
                            ggml_tensor* lora_v_down = NULL;
                            ggml_tensor* lora_v_up   = NULL;

                            ggml_tensor* lora_m_down = NULL;
                            ggml_tensor* lora_m_up   = NULL;

                            lora_q_up = to_f32(compute_ctx, lora_tensors[split_q_u_name]);

                            if (lora_tensors.find(split_q_d_name) != lora_tensors.end()) {
                                lora_q_down = to_f32(compute_ctx, lora_tensors[split_q_d_name]);
                            }

                            if (lora_tensors.find(split_q_u_name) != lora_tensors.end()) {
                                lora_q_up = to_f32(compute_ctx, lora_tensors[split_q_u_name]);
                            }

                            if (lora_tensors.find(split_k_d_name) != lora_tensors.end()) {
                                lora_k_down = to_f32(compute_ctx, lora_tensors[split_k_d_name]);
                            }

                            if (lora_tensors.find(split_k_u_name) != lora_tensors.end()) {
                                lora_k_up = to_f32(compute_ctx, lora_tensors[split_k_u_name]);
                            }

                            if (lora_tensors.find(split_v_d_name) != lora_tensors.end()) {
                                lora_v_down = to_f32(compute_ctx, lora_tensors[split_v_d_name]);
                            }

                            if (lora_tensors.find(split_v_u_name) != lora_tensors.end()) {
                                lora_v_up = to_f32(compute_ctx, lora_tensors[split_v_u_name]);
                            }

                            if (lora_tensors.find(split_m_d_name) != lora_tensors.end()) {
                                lora_m_down = to_f32(compute_ctx, lora_tensors[split_m_d_name]);
                            }

                            if (lora_tensors.find(split_m_u_name) != lora_tensors.end()) {
                                lora_m_up = to_f32(compute_ctx, lora_tensors[split_m_u_name]);
                            }

                            // print_ggml_tensor(lora_q_down, true);  //[3072, R, 1, 1]
                            // print_ggml_tensor(lora_k_down, true);  //[3072, R, 1, 1]
                            // print_ggml_tensor(lora_v_down, true);  //[3072, R, 1, 1]
                            // print_ggml_tensor(lora_m_down, true);  //[3072, R, 1, 1]
                            // print_ggml_tensor(lora_q_up, true);  //[R, 3072, 1, 1]
                            // print_ggml_tensor(lora_k_up, true);  //[R, 3072, 1, 1]
                            // print_ggml_tensor(lora_v_up, true);  //[R, 3072, 1, 1]
                            // print_ggml_tensor(lora_m_up, true);  //[R, 12288, 1, 1]

                            // these need to be stitched together this way:
                            //                                 |q_up,0   ,0   ,0   |
                            //                                 |0   ,k_up,0   ,0   |
                            //                                 |0   ,0   ,v_up,0   |
                            //                                 |0   ,0   ,0   ,m_up|
                            // (q_down,k_down,v_down,m_down) . (q   ,k   ,v   ,m)

                            // up_concat will be [21504, R*4, 1, 1]
                            // down_concat will be [R*4, 3072, 1, 1]

                            ggml_tensor* lora_down_concat = ggml_concat(compute_ctx, ggml_concat(compute_ctx, lora_q_down, lora_k_down, 1), ggml_concat(compute_ctx, lora_v_down, lora_m_down, 1), 1);
                            // print_ggml_tensor(lora_down_concat, true);  //[3072, R*4, 1, 1]

                            // this also means that if rank is bigger than 672, it is less memory efficient to do it this way (should be fine)
                            // print_ggml_tensor(lora_q_up, true);  //[3072, R, 1, 1]
                            ggml_tensor* z     = ggml_dup_tensor(compute_ctx, lora_q_up);
                            ggml_tensor* mlp_z = ggml_dup_tensor(compute_ctx, lora_m_up);
                            ggml_scale(compute_ctx, z, 0);
                            ggml_scale(compute_ctx, mlp_z, 0);
                            ggml_tensor* zz = ggml_concat(compute_ctx, z, z, 1);

                            ggml_tensor* q_up = ggml_concat(compute_ctx, ggml_concat(compute_ctx, lora_q_up, zz, 1), mlp_z, 1);
                            ggml_tensor* k_up = ggml_concat(compute_ctx, ggml_concat(compute_ctx, z, lora_k_up, 1), ggml_concat(compute_ctx, z, mlp_z, 1), 1);
                            ggml_tensor* v_up = ggml_concat(compute_ctx, ggml_concat(compute_ctx, zz, lora_v_up, 1), mlp_z, 1);
                            ggml_tensor* m_up = ggml_concat(compute_ctx, ggml_concat(compute_ctx, zz, z, 1), lora_m_up, 1);
                            // print_ggml_tensor(q_up, true);  //[R, 21504, 1, 1]
                            // print_ggml_tensor(k_up, true);  //[R, 21504, 1, 1]
                            // print_ggml_tensor(v_up, true);  //[R, 21504, 1, 1]
                            // print_ggml_tensor(m_up, true);  //[R, 21504, 1, 1]

                            ggml_tensor* lora_up_concat = ggml_concat(compute_ctx, ggml_concat(compute_ctx, q_up, k_up, 0), ggml_concat(compute_ctx, v_up, m_up, 0), 0);
                            // print_ggml_tensor(lora_up_concat, true);  //[R*4, 21504, 1, 1]

                            lora_down = ggml_cont(compute_ctx, lora_down_concat);
                            lora_up   = ggml_cont(compute_ctx, lora_up_concat);

                            // lora_down_name = lora_pre[type] + key + lora_downs[type] + ".weight";
                            // lora_up_name   = lora_pre[type] + key + lora_ups[type] + ".weight";

                            // lora_tensors[lora_down_name] = lora_down;
                            // lora_tensors[lora_up_name]   = lora_up;

                            // Would be nice to be able to clean up lora_tensors, but it breaks because this is called twice :/
                            applied_lora_tensors.insert(split_q_u_name);
                            applied_lora_tensors.insert(split_k_u_name);
                            applied_lora_tensors.insert(split_v_u_name);
                            applied_lora_tensors.insert(split_m_u_name);

                            applied_lora_tensors.insert(split_q_d_name);
                            applied_lora_tensors.insert(split_k_d_name);
                            applied_lora_tensors.insert(split_v_d_name);
                            applied_lora_tensors.insert(split_m_d_name);
                        }
                    } else if (linear2 != std::string::npos) {
                        linear2--;
                        lora_down_name = lora_pre[type] + key.substr(0, linear2) + ".proj_out" + lora_downs[type] + ".weight";
                        if (lora_tensors.find(lora_down_name) != lora_tensors.end()) {
                            lora_up_name = lora_pre[type] + key.substr(0, linear2) + ".proj_out" + lora_ups[type] + ".weight";
                            if (lora_tensors.find(lora_up_name) != lora_tensors.end()) {
                                lora_up = lora_tensors[lora_up_name];
                            }

                            if (lora_tensors.find(lora_down_name) != lora_tensors.end()) {
                                lora_down = lora_tensors[lora_down_name];
                            }

                            applied_lora_tensors.insert(lora_down_name);
                            applied_lora_tensors.insert(lora_up_name);
                        }
                    } else if (modulation != std::string::npos) {
                        modulation--;
                        lora_down_name = lora_pre[type] + key.substr(0, modulation) + ".norm.linear" + lora_downs[type] + ".weight";
                        if (lora_tensors.find(lora_down_name) != lora_tensors.end()) {
                            lora_up_name = lora_pre[type] + key.substr(0, modulation) + ".norm.linear" + lora_ups[type] + ".weight";
                            if (lora_tensors.find(lora_up_name) != lora_tensors.end()) {
                                lora_up = lora_tensors[lora_up_name];
                            }

                            if (lora_tensors.find(lora_down_name) != lora_tensors.end()) {
                                lora_down = lora_tensors[lora_down_name];
                            }

                            applied_lora_tensors.insert(lora_down_name);
                            applied_lora_tensors.insert(lora_up_name);
                        }
                    }
                    // Double blocks
                    else if (txt_attn_qkv != std::string::npos || img_attn_qkv != std::string::npos) {
                        size_t match       = txt_attn_qkv;
                        std::string prefix = ".attn.add_";
                        std::string suffix = "_proj";
                        if (img_attn_qkv != std::string::npos) {
                            match  = img_attn_qkv;
                            prefix = ".attn.to_";
                            suffix = "";
                        }
                        match--;

                        auto split_q_d_name = lora_pre[type] + key.substr(0, match) + prefix + "q" + suffix + lora_downs[type] + ".weight";
                        if (lora_tensors.find(split_q_d_name) != lora_tensors.end()) {
                            // print_ggml_tensor(it.second, true);  //[3072, 21504, 1, 1]
                            // find qkv and mlp up parts in LoRA model
                            auto split_k_d_name = lora_pre[type] + key.substr(0, match) + prefix + "k" + suffix + lora_downs[type] + ".weight";
                            auto split_v_d_name = lora_pre[type] + key.substr(0, match) + prefix + "v" + suffix + lora_downs[type] + ".weight";

                            auto split_q_u_name = lora_pre[type] + key.substr(0, match) + prefix + "q" + suffix + lora_ups[type] + ".weight";
                            auto split_k_u_name = lora_pre[type] + key.substr(0, match) + prefix + "k" + suffix + lora_ups[type] + ".weight";
                            auto split_v_u_name = lora_pre[type] + key.substr(0, match) + prefix + "v" + suffix + lora_ups[type] + ".weight";

                            ggml_tensor* lora_q_down = NULL;
                            ggml_tensor* lora_q_up   = NULL;
                            ggml_tensor* lora_k_down = NULL;
                            ggml_tensor* lora_k_up   = NULL;
                            ggml_tensor* lora_v_down = NULL;
                            ggml_tensor* lora_v_up   = NULL;

                            lora_q_down = to_f32(compute_ctx, lora_tensors[split_q_d_name]);

                            if (lora_tensors.find(split_q_u_name) != lora_tensors.end()) {
                                lora_q_up = to_f32(compute_ctx, lora_tensors[split_q_u_name]);
                            }

                            if (lora_tensors.find(split_k_d_name) != lora_tensors.end()) {
                                lora_k_down = to_f32(compute_ctx, lora_tensors[split_k_d_name]);
                            }

                            if (lora_tensors.find(split_k_u_name) != lora_tensors.end()) {
                                lora_k_up = to_f32(compute_ctx, lora_tensors[split_k_u_name]);
                            }

                            if (lora_tensors.find(split_v_d_name) != lora_tensors.end()) {
                                lora_v_down = to_f32(compute_ctx, lora_tensors[split_v_d_name]);
                            }

                            if (lora_tensors.find(split_v_u_name) != lora_tensors.end()) {
                                lora_v_up = to_f32(compute_ctx, lora_tensors[split_v_u_name]);
                            }

                            // print_ggml_tensor(lora_q_down, true);  //[3072, R, 1, 1]
                            // print_ggml_tensor(lora_k_down, true);  //[3072, R, 1, 1]
                            // print_ggml_tensor(lora_v_down, true);  //[3072, R, 1, 1]
                            // print_ggml_tensor(lora_q_up, true);    //[R, 3072, 1, 1]
                            // print_ggml_tensor(lora_k_up, true);    //[R, 3072, 1, 1]
                            // print_ggml_tensor(lora_v_up, true);    //[R, 3072, 1, 1]

                            // these need to be stitched together this way:
                            //                          |q_up,0   ,0   |
                            //                          |0   ,k_up,0   |
                            //                          |0   ,0   ,v_up|
                            // (q_down,k_down,v_down) . (q   ,k   ,v)

                            // up_concat will be [9216, R*3, 1, 1]
                            // down_concat will be [R*3, 3072, 1, 1]
                            ggml_tensor* lora_down_concat = ggml_concat(compute_ctx, ggml_concat(compute_ctx, lora_q_down, lora_k_down, 1), lora_v_down, 1);

                            ggml_tensor* z = ggml_dup_tensor(compute_ctx, lora_q_up);
                            ggml_scale(compute_ctx, z, 0);
                            ggml_tensor* zz = ggml_concat(compute_ctx, z, z, 1);

                            ggml_tensor* q_up = ggml_concat(compute_ctx, lora_q_up, zz, 1);
                            ggml_tensor* k_up = ggml_concat(compute_ctx, ggml_concat(compute_ctx, z, lora_k_up, 1), z, 1);
                            ggml_tensor* v_up = ggml_concat(compute_ctx, zz, lora_v_up, 1);
                            // print_ggml_tensor(q_up, true);  //[R, 9216, 1, 1]
                            // print_ggml_tensor(k_up, true);  //[R, 9216, 1, 1]
                            // print_ggml_tensor(v_up, true);  //[R, 9216, 1, 1]
                            ggml_tensor* lora_up_concat = ggml_concat(compute_ctx, ggml_concat(compute_ctx, q_up, k_up, 0), v_up, 0);
                            // print_ggml_tensor(lora_up_concat, true);  //[R*3, 9216, 1, 1]

                            lora_down = ggml_cont(compute_ctx, lora_down_concat);
                            lora_up   = ggml_cont(compute_ctx, lora_up_concat);

                            // lora_down_name = lora_pre[type] + key + lora_downs[type] + ".weight";
                            // lora_up_name   = lora_pre[type] + key + lora_ups[type] + ".weight";

                            // lora_tensors[lora_down_name] = lora_down;
                            // lora_tensors[lora_up_name]   = lora_up;

                            // Would be nice to be able to clean up lora_tensors, but it breaks because this is called twice :/
                            applied_lora_tensors.insert(split_q_u_name);
                            applied_lora_tensors.insert(split_k_u_name);
                            applied_lora_tensors.insert(split_v_u_name);

                            applied_lora_tensors.insert(split_q_d_name);
                            applied_lora_tensors.insert(split_k_d_name);
                            applied_lora_tensors.insert(split_v_d_name);
                        }
                    } else if (txt_attn_proj != std::string::npos || img_attn_proj != std::string::npos) {
                        size_t match         = txt_attn_proj;
                        std::string new_name = ".attn.to_add_out";
                        if (img_attn_proj != std::string::npos) {
                            match    = img_attn_proj;
                            new_name = ".attn.to_out.0";
                        }
                        match--;

                        lora_down_name = lora_pre[type] + key.substr(0, match) + new_name + lora_downs[type] + ".weight";
                        if (lora_tensors.find(lora_down_name) != lora_tensors.end()) {
                            lora_up_name = lora_pre[type] + key.substr(0, match) + new_name + lora_ups[type] + ".weight";
                            if (lora_tensors.find(lora_up_name) != lora_tensors.end()) {
                                lora_up = lora_tensors[lora_up_name];
                            }

                            if (lora_tensors.find(lora_down_name) != lora_tensors.end()) {
                                lora_down = lora_tensors[lora_down_name];
                            }

                            applied_lora_tensors.insert(lora_down_name);
                            applied_lora_tensors.insert(lora_up_name);
                        }
                    } else if (txt_mlp_0 != std::string::npos || txt_mlp_2 != std::string::npos || img_mlp_0 != std::string::npos || img_mlp_2 != std::string::npos) {
                        bool has_two       = txt_mlp_2 != std::string::npos || img_mlp_2 != std::string::npos;
                        std::string prefix = ".ff_context.net.";
                        std::string suffix = "0.proj";
                        if (img_mlp_0 != std::string::npos || img_mlp_2 != std::string::npos) {
                            prefix = ".ff.net.";
                        }
                        if (has_two) {
                            suffix = "2";
                        }
                        size_t match = txt_mlp_0;
                        if (txt_mlp_2 != std::string::npos) {
                            match = txt_mlp_2;
                        } else if (img_mlp_0 != std::string::npos) {
                            match = img_mlp_0;
                        } else if (img_mlp_2 != std::string::npos) {
                            match = img_mlp_2;
                        }
                        match--;
                        lora_down_name = lora_pre[type] + key.substr(0, match) + prefix + suffix + lora_downs[type] + ".weight";
                        if (lora_tensors.find(lora_down_name) != lora_tensors.end()) {
                            lora_up_name = lora_pre[type] + key.substr(0, match) + prefix + suffix + lora_ups[type] + ".weight";
                            if (lora_tensors.find(lora_up_name) != lora_tensors.end()) {
                                lora_up = lora_tensors[lora_up_name];
                            }

                            if (lora_tensors.find(lora_down_name) != lora_tensors.end()) {
                                lora_down = lora_tensors[lora_down_name];
                            }

                            applied_lora_tensors.insert(lora_down_name);
                            applied_lora_tensors.insert(lora_up_name);
                        }
                    } else if (txt_mod_lin != std::string::npos || img_mod_lin != std::string::npos) {
                        size_t match         = txt_mod_lin;
                        std::string new_name = ".norm1_context.linear";
                        if (img_mod_lin != std::string::npos) {
                            match    = img_mod_lin;
                            new_name = ".norm1.linear";
                        }
                        match--;

                        lora_down_name = lora_pre[type] + key.substr(0, match) + new_name + lora_downs[type] + ".weight";
                        if (lora_tensors.find(lora_down_name) != lora_tensors.end()) {
                            lora_up_name = lora_pre[type] + key.substr(0, match) + new_name + lora_ups[type] + ".weight";
                            if (lora_tensors.find(lora_up_name) != lora_tensors.end()) {
                                lora_up = lora_tensors[lora_up_name];
                            }

                            if (lora_tensors.find(lora_down_name) != lora_tensors.end()) {
                                lora_down = lora_tensors[lora_down_name];
                            }

                            applied_lora_tensors.insert(lora_down_name);
                            applied_lora_tensors.insert(lora_up_name);
                        }
                    }
                }

                if (lora_up == NULL || lora_down == NULL) {
                    lora_up_name = lora_pre[type] + key + lora_ups[type] + ".weight";
                    if (lora_tensors.find(lora_up_name) == lora_tensors.end()) {
                        if (key == "model_diffusion_model_output_blocks_2_2_conv") {
                            // fix for some sdxl lora, like lcm-lora-xl
                            key          = "model_diffusion_model_output_blocks_2_1_conv";
                            lora_up_name = lora_pre[type] + key + lora_ups[type] + ".weight";
                        }
                    }

                    lora_down_name = lora_pre[type] + key + lora_downs[type] + ".weight";
                    alpha_name     = lora_pre[type] + key + ".alpha";
                    scale_name     = lora_pre[type] + key + ".scale";

                    if (lora_tensors.find(lora_up_name) != lora_tensors.end()) {
                        lora_up = lora_tensors[lora_up_name];
                    }

                    if (lora_tensors.find(lora_down_name) != lora_tensors.end()) {
                        lora_down = lora_tensors[lora_down_name];
                    }
                    applied_lora_tensors.insert(lora_up_name);
                    applied_lora_tensors.insert(lora_down_name);
                    applied_lora_tensors.insert(alpha_name);
                    applied_lora_tensors.insert(scale_name);
                }

                if (lora_up == NULL || lora_down == NULL) {
                    continue;
                }
                // calc_scale
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
                lora_up                = ggml_reshape_2d(compute_ctx, lora_up, ggml_nelements(lora_up) / lora_up_rows, lora_up_rows);
                int64_t lora_down_rows = lora_down->ne[ggml_n_dims(lora_down) - 1];
                lora_down              = ggml_reshape_2d(compute_ctx, lora_down, ggml_nelements(lora_down) / lora_down_rows, lora_down_rows);

                // ggml_mul_mat requires tensor b transposed
                lora_down                  = ggml_cont(compute_ctx, ggml_transpose(compute_ctx, lora_down));
                struct ggml_tensor* updown = ggml_mul_mat(compute_ctx, lora_up, lora_down);
                updown                     = ggml_cont(compute_ctx, ggml_transpose(compute_ctx, updown));
                updown                     = ggml_reshape(compute_ctx, updown, weight);
                GGML_ASSERT(ggml_nelements(updown) == ggml_nelements(weight));
                updown = ggml_scale_inplace(compute_ctx, updown, scale_value);
                ggml_tensor* final_weight;
                if (weight->type != GGML_TYPE_F32 && weight->type != GGML_TYPE_F16) {
                    // final_weight = ggml_new_tensor(compute_ctx, GGML_TYPE_F32, ggml_n_dims(weight), weight->ne);
                    // final_weight = ggml_cpy(compute_ctx, weight, final_weight);
                    final_weight = to_f32(compute_ctx, weight);
                    final_weight = ggml_add_inplace(compute_ctx, final_weight, updown);
                    final_weight = ggml_cpy(compute_ctx, final_weight, weight);
                } else {
                    final_weight = ggml_add_inplace(compute_ctx, weight, updown);
                }
                // final_weight = ggml_add_inplace(compute_ctx, weight, updown);  // apply directly
                ggml_build_forward_expand(gf, final_weight);
            }
        }
        size_t total_lora_tensors_count   = 0;
        size_t applied_lora_tensors_count = 0;

        for (auto& kv : lora_tensors) {
            total_lora_tensors_count++;
            if (applied_lora_tensors.find(kv.first) == applied_lora_tensors.end()) {
                LOG_WARN("unused lora tensor %s", kv.first.c_str());
            } else {
                applied_lora_tensors_count++;
            }
        }
        /* Don't worry if this message shows up twice in the logs per LoRA,
         * this function is called once to calculate the required buffer size
         * and then again to actually generate a graph to be used */
        if (applied_lora_tensors_count != total_lora_tensors_count) {
            LOG_WARN("Only (%lu / %lu) LoRA tensors have been applied",
                     applied_lora_tensors_count, total_lora_tensors_count);
        } else {
            LOG_DEBUG("(%lu / %lu) LoRA tensors applied successfully",
                      applied_lora_tensors_count, total_lora_tensors_count);
        }

        return gf;
    }

    void apply(std::map<std::string, struct ggml_tensor*> model_tensors, SDVersion version, int n_threads) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_lora_graph(model_tensors, version);
        };
        GGMLRunner::compute(get_graph, n_threads, true);
    }
};

#endif  // __LORA_HPP__
