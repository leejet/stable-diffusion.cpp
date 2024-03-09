#ifndef __PMI_HPP__
#define __PMI_HPP__

#include "ggml_extend.hpp"

#include "clip.hpp"

struct FuseBlock : public GGMLBlock {
    // network hparams
    int in_dim;
    int out_dim;
    int hidden_dim;
    bool use_residue;

public:
    FuseBlock(int i_d, int o_d, int h_d, bool use_residue = true)
        : in_dim(i_d), out_dim(o_d), hidden_dim(h_d), use_residue(use_residue) {
        blocks["fc1"]       = std::shared_ptr<GGMLBlock>(new Linear(in_dim, hidden_dim, true));
        blocks["fc2"]       = std::shared_ptr<GGMLBlock>(new Linear(hidden_dim, out_dim, true));
        blocks["layernorm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(in_dim));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, channels, h, w]

        auto fc1        = std::dynamic_pointer_cast<Linear>(blocks["fc1"]);
        auto fc2        = std::dynamic_pointer_cast<Linear>(blocks["fc2"]);
        auto layer_norm = std::dynamic_pointer_cast<LayerNorm>(blocks["layernorm"]);

        struct ggml_tensor* r = x;
        // x = ggml_nn_layer_norm(ctx, x, ln_w, ln_b);
        x = layer_norm->forward(ctx, x);
        // x = ggml_add(ctx, ggml_mul_mat(ctx, fc1_w, x),  fc1_b);
        x = fc1->forward(ctx, x);
        x = ggml_gelu_inplace(ctx, x);
        x = fc2->forward(ctx, x);
        // x = ggml_add(ctx, ggml_mul_mat(ctx, fc2_w, x),  fc2_b);
        if (use_residue)
            x = ggml_add(ctx, x, r);
        return x;
    }
};

struct FuseModule : public GGMLBlock {
    // network hparams
    int embed_dim;

public:
    FuseModule(int imb_d)
        : embed_dim(imb_d) {
        blocks["mlp1"]       = std::shared_ptr<GGMLBlock>(new FuseBlock(imb_d * 2, imb_d, imb_d, false));
        blocks["mlp2"]       = std::shared_ptr<GGMLBlock>(new FuseBlock(imb_d, imb_d, imb_d, true));
        blocks["layer_norm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(embed_dim));
    }

    struct ggml_tensor* fuse_fn(struct ggml_context* ctx,
                                struct ggml_tensor* prompt_embeds,
                                struct ggml_tensor* id_embeds) {
        auto mlp1       = std::dynamic_pointer_cast<FuseBlock>(blocks["mlp1"]);
        auto mlp2       = std::dynamic_pointer_cast<FuseBlock>(blocks["mlp2"]);
        auto layer_norm = std::dynamic_pointer_cast<LayerNorm>(blocks["layer_norm"]);

        auto prompt_embeds0 = ggml_cont(ctx, ggml_permute(ctx, prompt_embeds, 2, 0, 1, 3));
        auto id_embeds0     = ggml_cont(ctx, ggml_permute(ctx, id_embeds, 2, 0, 1, 3));
        // concat is along dim 2
        auto stacked_id_embeds = ggml_concat(ctx, prompt_embeds0, id_embeds0);
        stacked_id_embeds      = ggml_cont(ctx, ggml_permute(ctx, stacked_id_embeds, 1, 2, 0, 3));

        // stacked_id_embeds = mlp1.forward(ctx, stacked_id_embeds);
        // stacked_id_embeds = ggml_add(ctx, stacked_id_embeds, prompt_embeds);
        // stacked_id_embeds = mlp2.forward(ctx, stacked_id_embeds);
        // stacked_id_embeds = ggml_nn_layer_norm(ctx, stacked_id_embeds, ln_w, ln_b);

        stacked_id_embeds = mlp1->forward(ctx, stacked_id_embeds);
        stacked_id_embeds = ggml_add(ctx, stacked_id_embeds, prompt_embeds);
        stacked_id_embeds = mlp2->forward(ctx, stacked_id_embeds);
        stacked_id_embeds = layer_norm->forward(ctx, stacked_id_embeds);

        return stacked_id_embeds;
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* prompt_embeds,
                                struct ggml_tensor* id_embeds,
                                struct ggml_tensor* class_tokens_mask,
                                struct ggml_tensor* class_tokens_mask_pos,
                                struct ggml_tensor* left,
                                struct ggml_tensor* right) {
        // x: [N, channels, h, w]

        struct ggml_tensor* valid_id_embeds = id_embeds;
        // # slice out the image token embeddings
        // print_ggml_tensor(class_tokens_mask_pos, false);
        ggml_set_name(class_tokens_mask_pos, "class_tokens_mask_pos");
        ggml_set_name(prompt_embeds, "prompt_embeds");
        // print_ggml_tensor(valid_id_embeds, true, "valid_id_embeds");
        // print_ggml_tensor(class_tokens_mask_pos, true, "class_tokens_mask_pos");
        struct ggml_tensor* image_token_embeds = ggml_get_rows(ctx, prompt_embeds, class_tokens_mask_pos);
        ggml_set_name(image_token_embeds, "image_token_embeds");
        struct ggml_tensor* stacked_id_embeds = fuse_fn(ctx, image_token_embeds, valid_id_embeds);

        stacked_id_embeds = ggml_cont(ctx, ggml_permute(ctx, stacked_id_embeds, 0, 2, 1, 3));
        if (left && right) {
            stacked_id_embeds = ggml_concat(ctx, left, stacked_id_embeds);
            stacked_id_embeds = ggml_concat(ctx, stacked_id_embeds, right);
        } else if (left) {
            stacked_id_embeds = ggml_concat(ctx, left, stacked_id_embeds);
        } else if (right) {
            stacked_id_embeds = ggml_concat(ctx, stacked_id_embeds, right);
        }
        stacked_id_embeds                         = ggml_cont(ctx, ggml_permute(ctx, stacked_id_embeds, 0, 2, 1, 3));
        class_tokens_mask                         = ggml_cont(ctx, ggml_transpose(ctx, class_tokens_mask));
        class_tokens_mask                         = ggml_repeat(ctx, class_tokens_mask, prompt_embeds);
        prompt_embeds                             = ggml_mul(ctx, prompt_embeds, class_tokens_mask);
        struct ggml_tensor* updated_prompt_embeds = ggml_add(ctx, prompt_embeds, stacked_id_embeds);
        ggml_set_name(updated_prompt_embeds, "updated_prompt_embeds");
        return updated_prompt_embeds;
    }
};

struct PhotoMakerIDEncoderBlock : public CLIPVisionModelProjection {
    PhotoMakerIDEncoderBlock()
        : CLIPVisionModelProjection(OPENAI_CLIP_VIT_L_14) {
        blocks["visual_projection_2"] = std::shared_ptr<GGMLBlock>(new Linear(1024, 1280, false));
        blocks["fuse_module"]         = std::shared_ptr<GGMLBlock>(new FuseModule(2048));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* id_pixel_values,
                                struct ggml_tensor* prompt_embeds,
                                struct ggml_tensor* class_tokens_mask,
                                struct ggml_tensor* class_tokens_mask_pos,
                                struct ggml_tensor* left,
                                struct ggml_tensor* right) {
        // x: [N, channels, h, w]
        auto vision_model        = std::dynamic_pointer_cast<CLIPVisionModel>(blocks["vision_model"]);
        auto visual_projection   = std::dynamic_pointer_cast<CLIPProjection>(blocks["visual_projection"]);
        auto visual_projection_2 = std::dynamic_pointer_cast<Linear>(blocks["visual_projection_2"]);
        auto fuse_module         = std::dynamic_pointer_cast<FuseModule>(blocks["fuse_module"]);

        struct ggml_tensor* shared_id_embeds = vision_model->forward(ctx, id_pixel_values);          // [N, hidden_size]
        struct ggml_tensor* id_embeds        = visual_projection->forward(ctx, shared_id_embeds);    // [N, proj_dim(768)]
        struct ggml_tensor* id_embeds_2      = visual_projection_2->forward(ctx, shared_id_embeds);  // [N, 1280]

        id_embeds   = ggml_cont(ctx, ggml_permute(ctx, id_embeds, 2, 0, 1, 3));
        id_embeds_2 = ggml_cont(ctx, ggml_permute(ctx, id_embeds_2, 2, 0, 1, 3));

        id_embeds = ggml_concat(ctx, id_embeds, id_embeds_2);  // [batch_size, seq_length, 1, 2048] check whether concat at dim 2 is right
        id_embeds = ggml_cont(ctx, ggml_permute(ctx, id_embeds, 1, 2, 0, 3));

        struct ggml_tensor* updated_prompt_embeds = fuse_module->forward(ctx,
                                                                         prompt_embeds,
                                                                         id_embeds,
                                                                         class_tokens_mask,
                                                                         class_tokens_mask_pos,
                                                                         left, right);
        return updated_prompt_embeds;
    }
};

struct PhotoMakerIDEncoder : public GGMLModule {
public:
    SDVersion version = VERSION_XL;
    PhotoMakerIDEncoderBlock id_encoder;
    float style_strength;

    std::vector<float> ctm;
    std::vector<ggml_fp16_t> ctmf16;
    std::vector<int> ctmpos;

    std::vector<ggml_fp16_t> zeros_left_16;
    std::vector<float> zeros_left;
    std::vector<ggml_fp16_t> zeros_right_16;
    std::vector<float> zeros_right;

public:
    PhotoMakerIDEncoder(ggml_backend_t backend, ggml_type wtype, SDVersion version = VERSION_XL, float sty = 20.f)
        : GGMLModule(backend, wtype),
          version(version),
          style_strength(sty) {
        id_encoder.init(params_ctx, wtype);
    }

    std::string get_desc() {
        return "pmid";
    }

    size_t get_params_mem_size() {
        size_t params_mem_size = id_encoder.get_params_mem_size();
        return params_mem_size;
    }

    size_t get_params_num() {
        size_t params_num = id_encoder.get_params_num();
        return params_num;
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        id_encoder.get_param_tensors(tensors, prefix);
    }

    struct ggml_cgraph* build_graph(  // struct ggml_allocr* allocr,
        struct ggml_tensor* id_pixel_values,
        struct ggml_tensor* prompt_embeds,
        std::vector<bool>& class_tokens_mask) {
        ctm.clear();
        ctmf16.clear();
        ctmpos.clear();
        zeros_left.clear();
        zeros_left_16.clear();
        zeros_right.clear();
        zeros_right_16.clear();

        ggml_context* ctx0 = compute_ctx;

        struct ggml_cgraph* gf = ggml_new_graph(compute_ctx);

        int64_t hidden_size = prompt_embeds->ne[0];
        int64_t seq_length  = prompt_embeds->ne[1];
        ggml_type type      = GGML_TYPE_F32;

        struct ggml_tensor* class_tokens_mask_d = ggml_new_tensor_1d(ctx0, type, class_tokens_mask.size());

        struct ggml_tensor* id_pixel_values_d = to_backend(id_pixel_values);
        struct ggml_tensor* prompt_embeds_d   = to_backend(prompt_embeds);

        struct ggml_tensor* left  = NULL;
        struct ggml_tensor* right = NULL;
        for (int i = 0; i < class_tokens_mask.size(); i++) {
            if (class_tokens_mask[i]) {
                ctm.push_back(0.f);                        // here use 0.f instead of 1.f to make a scale mask
                ctmf16.push_back(ggml_fp32_to_fp16(0.f));  // here use 0.f instead of 1.f to make a scale mask
                ctmpos.push_back(i);
            } else {
                ctm.push_back(1.f);                        // here use 1.f instead of 0.f to make a scale mask
                ctmf16.push_back(ggml_fp32_to_fp16(1.f));  // here use 0.f instead of 1.f to make a scale mask
            }
        }
        if (ctmpos[0] > 0) {
            left = ggml_new_tensor_3d(ctx0, type, hidden_size, 1, ctmpos[0]);
        }
        if (ctmpos[ctmpos.size() - 1] < seq_length - 1) {
            right = ggml_new_tensor_3d(ctx0, type,
                                       hidden_size, 1, seq_length - ctmpos[ctmpos.size() - 1] - 1);
        }
        struct ggml_tensor* class_tokens_mask_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ctmpos.size());

        {
            if (type == GGML_TYPE_F16)
                set_backend_tensor_data(class_tokens_mask_d, ctmf16.data());
            else
                set_backend_tensor_data(class_tokens_mask_d, ctm.data());
            set_backend_tensor_data(class_tokens_mask_pos, ctmpos.data());
            if (left) {
                if (type == GGML_TYPE_F16) {
                    for (int i = 0; i < ggml_nelements(left); ++i)
                        zeros_left_16.push_back(ggml_fp32_to_fp16(0.f));
                    set_backend_tensor_data(left, zeros_left_16.data());
                } else {
                    for (int i = 0; i < ggml_nelements(left); ++i)
                        zeros_left.push_back(0.f);
                    set_backend_tensor_data(left, zeros_left.data());
                }
            }
            if (right) {
                if (type == GGML_TYPE_F16) {
                    for (int i = 0; i < ggml_nelements(right); ++i)
                        zeros_right_16.push_back(ggml_fp32_to_fp16(0.f));
                    set_backend_tensor_data(right, zeros_right_16.data());
                } else {
                    for (int i = 0; i < ggml_nelements(right); ++i)
                        zeros_right.push_back(0.f);
                    set_backend_tensor_data(right, zeros_right.data());
                }
            }
        }
        struct ggml_tensor* updated_prompt_embeds = id_encoder.forward(ctx0,
                                                                       id_pixel_values_d,
                                                                       prompt_embeds_d,
                                                                       class_tokens_mask_d,
                                                                       class_tokens_mask_pos,
                                                                       left, right);
        ggml_build_forward_expand(gf, updated_prompt_embeds);

        return gf;
    }

    void compute(const int n_threads,
                 struct ggml_tensor* id_pixel_values,
                 struct ggml_tensor* prompt_embeds,
                 std::vector<bool>& class_tokens_mask,
                 struct ggml_tensor** updated_prompt_embeds,
                 ggml_context* output_ctx) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            // return build_graph(compute_allocr, id_pixel_values, prompt_embeds, class_tokens_mask);
            return build_graph(id_pixel_values, prompt_embeds, class_tokens_mask);
        };

        // GGMLModule::compute(get_graph, n_threads, updated_prompt_embeds);
        GGMLModule::compute(get_graph, n_threads, true, updated_prompt_embeds, output_ctx);
    }
};

#define PM_LORA_GRAPH_SIZE 10240

struct PhotoMakerLoraModel : public GGMLModule {
    float multiplier = 1.f;
    SDVersion version;
    std::map<std::string, struct ggml_tensor*> lora_tensors;
    std::vector<std::string> lora_tensors_to_be_ignored;
    std::string file_path;
    ModelLoader model_loader;
    bool load_failed = false;

    // PhotoMakerLoraModel()
    // {}

public:
    PhotoMakerLoraModel(ggml_backend_t backend, ggml_type wtype, SDVersion version = VERSION_XL, const std::string file_path = "")
        : GGMLModule(backend, wtype), version(version), file_path(file_path) {
        // name = "photomaker lora";
        if (!model_loader.init_from_file(file_path, "pmid.")) {
            load_failed = true;
        }
    }

    std::string get_desc() {
        return "lora_pmid";
    }

    size_t get_params_num() {
        return PM_LORA_GRAPH_SIZE;
    }

    size_t get_params_mem_size() {
        return model_loader.get_params_mem_size(NULL);
    }

    bool load_from_file(ggml_backend_t backend) {
        LOG_INFO("loading LoRA from '%s'", file_path.c_str());

        if (load_failed) {
            LOG_ERROR("init lora model loader from file failed: '%s'", file_path.c_str());
            return false;
        }

        bool dry_run          = true;
        auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool {
            std::string name = tensor_storage.name;
            // LOG_INFO("loading LoRA tesnor '%s'", name.c_str());
            if (!starts_with(name, "pmid.unet")) {
                // LOG_INFO("skipping LoRA tesnor '%s'", name.c_str());
                return true;
            }

            lora_tensors_to_be_ignored.push_back(name);
            size_t k_pos = name.find(".processor");
            if (k_pos != std::string::npos)
                name.replace(k_pos, strlen(".processor"), "");

            if (dry_run) {
                // LOG_INFO("loading LoRA tesnor '%s'", name.c_str());
                struct ggml_tensor* real = ggml_new_tensor(params_ctx,
                                                           tensor_storage.type,
                                                           tensor_storage.n_dims,
                                                           tensor_storage.ne);
                lora_tensors[name]       = real;
            } else {
                auto real   = lora_tensors[name];
                *dst_tensor = real;
            }

            // if(starts_with(name, "pmid.unet.down_blocks.2.attentions.1.transformer_blocks.9.attn2"))
            //     print_ggml_tensor(real, true, name.c_str());
            // lora_tensors[name] = real;
            return true;
        };

        model_loader.load_tensors(on_new_tensor_cb, backend);
        alloc_params_buffer();

        dry_run = false;
        model_loader.load_tensors(on_new_tensor_cb, backend);

        LOG_DEBUG("finished loaded lora");
        return true;
    }

    std::pair<int, int> find_ij0(int n) {
        int i, j;
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 2; j++) {
                if ((i * 3 + j + 1) == n)
                    return {i, j};
            }
        }
        return {-1, -1};
    }

    std::pair<int, int> find_ij(int n) {
        int i, j;
        for (i = 0; i < 2; i++) {
            for (j = 0; j < 3; j++) {
                if ((i * 3 + j) == n)
                    return {i, j};
            }
        }
        return {-1, -1};
    }

    struct ggml_cgraph* build_graph(std::map<std::string, struct ggml_tensor*> model_tensors) {
        // make a graph to compute all lora, expected lora and models tensors are in the same backend
        // since we are using ggml-alloc, this buffer only needs enough space to hold the ggml_tensor and ggml_cgraph structs, but not the tensor data
        static size_t buf_size = ggml_tensor_overhead() * PM_LORA_GRAPH_SIZE + ggml_graph_overhead();
        static std::vector<uint8_t> buf(buf_size);

        struct ggml_init_params params = {
            /*.mem_size   =*/buf_size,
            /*.mem_buffer =*/buf.data(),
            /*.no_alloc   =*/true,  // the tensors will be allocated later by ggml_allocr_alloc_graph()
        };
        // LOG_DEBUG("mem_size %u ", params.mem_size);

        struct ggml_context* ctx0 = ggml_init(params);
        struct ggml_cgraph* gf    = ggml_new_graph_custom(ctx0, PM_LORA_GRAPH_SIZE, false);

        std::set<std::string> applied_lora_tensors;
        for (auto it : model_tensors) {
            std::string k_tensor       = it.first;
            struct ggml_tensor* weight = model_tensors[it.first];
            std::string full_name      = k_tensor;
            // size_t k_pos = k_tensor.find(".weight");
            size_t k_pos = k_tensor.find(".attn1");
            if (k_pos == std::string::npos) {
                k_pos = k_tensor.find(".attn2");
                if (k_pos == std::string::npos) {
                    continue;
                }
            }
            if (ends_with(k_tensor, "bias"))
                continue;
            int block_kind = -1;
            int block_id   = -1;
            if ((k_pos = k_tensor.find("input_blocks")) != std::string::npos) {
                block_id   = atoi(k_tensor.substr(k_pos + strlen("input_blocks") + 1).c_str());
                block_kind = 0;  // input ->  down block

            } else if ((k_pos = k_tensor.find("output_blocks")) != std::string::npos) {
                block_id   = atoi(k_tensor.substr(k_pos + strlen("output_blocks") + 1).c_str());
                block_kind = 1;  // output ->  up block
            } else {
                k_pos      = k_tensor.find("transformer_blocks");
                block_id   = atoi(k_tensor.substr(k_pos - 4, 1).c_str());
                block_kind = 2;  // middle block
            }

            std::string lora_up_name;
            std::string lora_down_name;
            std::string prefix = "pmid.unet";
            if (block_kind == 0) {
                prefix   = prefix + ".down_blocks";
                k_pos    = k_tensor.find(".weight");
                k_tensor = k_tensor.substr(0, k_pos);
                k_pos    = k_tensor.find("transformer_blocks");
                k_tensor = k_tensor.substr(k_pos);
                if (ends_with(k_tensor, "0")) {
                    k_tensor = k_tensor.substr(0, k_tensor.length() - 2);
                }
                std::pair<int, int> ij = find_ij0(block_id);
                if (ij.first == -1)
                    continue;
                prefix         = prefix + "." + std::to_string(ij.first) + ".attentions." + std::to_string(ij.second) + ".";
                lora_up_name   = prefix + k_tensor + "_lora.up.weight";
                lora_down_name = prefix + k_tensor + "_lora.down.weight";
            } else if (block_kind == 1) {
                prefix   = prefix + ".up_blocks";
                k_pos    = k_tensor.find(".weight");
                k_tensor = k_tensor.substr(0, k_pos);
                k_pos    = k_tensor.find("transformer_blocks");
                k_tensor = k_tensor.substr(k_pos);
                if (ends_with(k_tensor, "0")) {
                    k_tensor = k_tensor.substr(0, k_tensor.length() - 2);
                }
                std::pair<int, int> ij = find_ij(block_id);
                if (ij.first == -1)
                    continue;
                prefix         = prefix + "." + std::to_string(ij.first) + ".attentions." + std::to_string(ij.second) + ".";
                lora_up_name   = prefix + k_tensor + "_lora.up.weight";
                lora_down_name = prefix + k_tensor + "_lora.down.weight";
            } else {
                prefix   = prefix + ".mid_block" + ".attentions.0.";
                k_pos    = k_tensor.find(".weight");
                k_tensor = k_tensor.substr(0, k_pos);
                k_pos    = k_tensor.find("transformer_blocks");
                k_tensor = k_tensor.substr(k_pos);
                if (ends_with(k_tensor, "0")) {
                    k_tensor = k_tensor.substr(0, k_tensor.length() - 2);
                }
                lora_up_name   = prefix + k_tensor + "_lora.up.weight";
                lora_down_name = prefix + k_tensor + "_lora.down.weight";
            }
            // LOG_INFO("unet transformer tensor: %s ", full_name.c_str());
            // LOG_INFO("corresponding up tensor: %s ", lora_up_name.c_str());
            // LOG_INFO("corresponding dn tensor: %s ", lora_down_name.c_str());

            ggml_tensor* lora_up   = NULL;
            ggml_tensor* lora_down = NULL;

            if (lora_tensors.find(lora_up_name) != lora_tensors.end()) {
                lora_up = lora_tensors[lora_up_name];
            }

            if (lora_tensors.find(lora_down_name) != lora_tensors.end()) {
                lora_down = lora_tensors[lora_down_name];
            }

            if (lora_up == NULL || lora_down == NULL) {
                LOG_WARN("can not find: %s and %s  k,id = (%d, %d)", lora_down_name.c_str(), lora_up_name.c_str(), block_kind, block_id);
                continue;
            }

            applied_lora_tensors.insert(lora_up_name);
            applied_lora_tensors.insert(lora_down_name);

            // same as in lora.hpp
            lora_down                  = ggml_cont(ctx0, ggml_transpose(ctx0, lora_down));
            struct ggml_tensor* updown = ggml_mul_mat(ctx0, lora_up, lora_down);
            updown                     = ggml_cont(ctx0, ggml_transpose(ctx0, updown));
            updown                     = ggml_reshape(ctx0, updown, weight);
            GGML_ASSERT(ggml_nelements(updown) == ggml_nelements(weight));
            updown = ggml_scale_inplace(ctx0, updown, multiplier);
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
        // GGMLModule::compute(get_graph, n_threads);
        GGMLModule::compute(get_graph, n_threads, true);
    }
};

#endif  // __PMI_HPP__
