#ifndef __CONTROL_HPP__
#define __CONTROL_HPP__

#include "common.hpp"
#include "ggml_extend.hpp"
#include "model.h"

#define CONTROL_NET_GRAPH_SIZE 1536

/*
    =================================== ControlNet ===================================
    Reference: https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/cldm/cldm.py

*/

struct CNHintBlock {
    int hint_channels    = 3;
    int model_channels   = 320;  // SD 1.5
    int feat_channels[4] = {16, 32, 96, 256};
    int num_blocks       = 3;
    ggml_tensor* conv_first_w;  // [feat_channels[0], hint_channels, 3, 3]
    ggml_tensor* conv_first_b;  // [feat_channels[0]]

    struct hint_block {
        ggml_tensor* conv_0_w;  // [feat_channels[idx], feat_channels[idx], 3, 3]
        ggml_tensor* conv_0_b;  // [feat_channels[idx]]

        ggml_tensor* conv_1_w;  // [feat_channels[idx + 1], feat_channels[idx], 3, 3]
        ggml_tensor* conv_1_b;  // [feat_channels[idx + 1]]
    };

    hint_block blocks[3];
    ggml_tensor* conv_final_w;  // [model_channels, feat_channels[3], 3, 3]
    ggml_tensor* conv_final_b;  // [model_channels]

    size_t calculate_mem_size() {
        size_t mem_size = feat_channels[0] * hint_channels * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_first_w
        mem_size += feat_channels[0] * ggml_type_size(GGML_TYPE_F32);                                // conv_first_b
        for (int i = 0; i < num_blocks; i++) {
            mem_size += feat_channels[i] * feat_channels[i] * 3 * 3 * ggml_type_size(GGML_TYPE_F16);      // conv_0_w
            mem_size += feat_channels[i] * ggml_type_size(GGML_TYPE_F32);                                 // conv_0_b
            mem_size += feat_channels[i + 1] * feat_channels[i] * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_1_w
            mem_size += feat_channels[i + 1] * ggml_type_size(GGML_TYPE_F32);                             // conv_1_b
        }
        mem_size += model_channels * feat_channels[3] * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_final_w
        mem_size += model_channels * ggml_type_size(GGML_TYPE_F32);                             // conv_final_b
        return mem_size;
    }

    void init_params(struct ggml_context* ctx) {
        conv_first_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, hint_channels, feat_channels[0]);
        conv_first_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, feat_channels[0]);

        for (int i = 0; i < num_blocks; i++) {
            blocks[i].conv_0_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, feat_channels[i], feat_channels[i]);
            blocks[i].conv_0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, feat_channels[i]);
            blocks[i].conv_1_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, feat_channels[i], feat_channels[i + 1]);
            blocks[i].conv_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, feat_channels[i + 1]);
        }

        conv_final_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, feat_channels[3], model_channels);
        conv_final_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model_channels);
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        tensors[prefix + "input_hint_block.0.weight"] = conv_first_w;
        tensors[prefix + "input_hint_block.0.bias"]   = conv_first_b;
        int index                                     = 2;
        for (int i = 0; i < num_blocks; i++) {
            tensors[prefix + "input_hint_block." + std::to_string(index) + ".weight"] = blocks[i].conv_0_w;
            tensors[prefix + "input_hint_block." + std::to_string(index) + ".bias"]   = blocks[i].conv_0_b;
            index += 2;
            tensors[prefix + "input_hint_block." + std::to_string(index) + ".weight"] = blocks[i].conv_1_w;
            tensors[prefix + "input_hint_block." + std::to_string(index) + ".bias"]   = blocks[i].conv_1_b;
            index += 2;
        }
        tensors[prefix + "input_hint_block.14.weight"] = conv_final_w;
        tensors[prefix + "input_hint_block.14.bias"]   = conv_final_b;
    }

    struct ggml_tensor* forward(ggml_context* ctx, struct ggml_tensor* x) {
        auto h = ggml_nn_conv_2d(ctx, x, conv_first_w, conv_first_b, 1, 1, 1, 1);
        h      = ggml_silu_inplace(ctx, h);

        auto body_h = h;
        for (int i = 0; i < num_blocks; i++) {
            // operations.conv_nd(dims, 16, 16, 3, padding=1)
            body_h = ggml_nn_conv_2d(ctx, body_h, blocks[i].conv_0_w, blocks[i].conv_0_b, 1, 1, 1, 1);
            body_h = ggml_silu_inplace(ctx, body_h);
            // operations.conv_nd(dims, 16, 32, 3, padding=1, stride=2)
            body_h = ggml_nn_conv_2d(ctx, body_h, blocks[i].conv_1_w, blocks[i].conv_1_b, 2, 2, 1, 1);
            body_h = ggml_silu_inplace(ctx, body_h);
        }

        h = ggml_nn_conv_2d(ctx, body_h, conv_final_w, conv_final_b, 1, 1, 1, 1);
        h = ggml_silu_inplace(ctx, h);
        return h;
    }
};

struct CNZeroConv {
    int channels;
    ggml_tensor* conv_w;  // [channels, channels, 1, 1]
    ggml_tensor* conv_b;  // [channels]

    void init_params(struct ggml_context* ctx) {
        conv_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, channels, channels);
        conv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
    }
};

struct ControlNet : public GGMLModule {
    int in_channels                        = 4;
    int model_channels                     = 320;
    int out_channels                       = 4;
    int num_res_blocks                     = 2;
    std::vector<int> attention_resolutions = {4, 2, 1};
    std::vector<int> channel_mult          = {1, 2, 4, 4};
    std::vector<int> transformer_depth     = {1, 1, 1, 1};
    int time_embed_dim                     = 1280;  // model_channels*4
    int num_heads                          = 8;
    int num_head_channels                  = -1;  // channels // num_heads
    int context_dim                        = 768;
    int middle_out_channel;
    CNHintBlock input_hint_block;
    CNZeroConv zero_convs[12];
    int num_zero_convs = 1;

    // network params
    struct ggml_tensor* time_embed_0_w;  // [time_embed_dim, model_channels]
    struct ggml_tensor* time_embed_0_b;  // [time_embed_dim, ]
    // time_embed_1 is nn.SILU()
    struct ggml_tensor* time_embed_2_w;  // [time_embed_dim, time_embed_dim]
    struct ggml_tensor* time_embed_2_b;  // [time_embed_dim, ]

    struct ggml_tensor* input_block_0_w;  // [model_channels, in_channels, 3, 3]
    struct ggml_tensor* input_block_0_b;  // [model_channels, ]

    // input_blocks
    ResBlock input_res_blocks[4][2];
    SpatialTransformer input_transformers[3][2];
    DownSample input_down_samples[3];

    // middle_block
    ResBlock middle_block_0;
    SpatialTransformer middle_block_1;
    ResBlock middle_block_2;

    struct ggml_tensor* middle_block_out_w;       // [middle_out_channel, middle_out_channel, 1, 1]
    struct ggml_tensor* middle_block_out_b;       // [middle_out_channel, ]
    ggml_backend_buffer_t control_buffer = NULL;  // keep control output tensors in backend memory
    ggml_context* control_ctx            = NULL;
    std::vector<struct ggml_tensor*> controls;  // (12 input block outputs, 1 middle block output) SD 1.5

    ControlNet() {
        name = "controlnet";
        // input_blocks
        std::vector<int> input_block_chans;
        input_block_chans.push_back(model_channels);
        int ch                 = model_channels;
        zero_convs[0].channels = model_channels;
        int ds                 = 1;

        int len_mults = channel_mult.size();
        for (int i = 0; i < len_mults; i++) {
            int mult = channel_mult[i];
            for (int j = 0; j < num_res_blocks; j++) {
                input_res_blocks[i][j].channels     = ch;
                input_res_blocks[i][j].emb_channels = time_embed_dim;
                input_res_blocks[i][j].out_channels = mult * model_channels;

                ch = mult * model_channels;
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    int n_head = num_heads;
                    int d_head = ch / num_heads;
                    if (num_head_channels != -1) {
                        d_head = num_head_channels;
                        n_head = ch / d_head;
                    }
                    input_transformers[i][j]             = SpatialTransformer(transformer_depth[i]);
                    input_transformers[i][j].in_channels = ch;
                    input_transformers[i][j].n_head      = n_head;
                    input_transformers[i][j].d_head      = d_head;
                    input_transformers[i][j].context_dim = context_dim;
                }
                input_block_chans.push_back(ch);

                zero_convs[num_zero_convs].channels = ch;
                num_zero_convs++;
            }
            if (i != len_mults - 1) {
                input_down_samples[i].channels     = ch;
                input_down_samples[i].out_channels = ch;
                input_block_chans.push_back(ch);

                zero_convs[num_zero_convs].channels = ch;
                num_zero_convs++;
                ds *= 2;
            }
        }
        GGML_ASSERT(num_zero_convs == 12);

        // middle blocks
        middle_block_0.channels     = ch;
        middle_block_0.emb_channels = time_embed_dim;
        middle_block_0.out_channels = ch;

        int n_head = num_heads;
        int d_head = ch / num_heads;
        if (num_head_channels != -1) {
            d_head = num_head_channels;
            n_head = ch / d_head;
        }
        middle_block_1             = SpatialTransformer(transformer_depth[transformer_depth.size() - 1]);
        middle_block_1.in_channels = ch;
        middle_block_1.n_head      = n_head;
        middle_block_1.d_head      = d_head;
        middle_block_1.context_dim = context_dim;

        middle_block_2.channels     = ch;
        middle_block_2.emb_channels = time_embed_dim;
        middle_block_2.out_channels = ch;
        middle_out_channel          = ch;
    }

    size_t calculate_mem_size() {
        size_t mem_size = 0;
        mem_size += input_hint_block.calculate_mem_size();
        mem_size += ggml_row_size(wtype, time_embed_dim * model_channels);  // time_embed_0_w
        mem_size += ggml_row_size(GGML_TYPE_F32, time_embed_dim);           // time_embed_0_b
        mem_size += ggml_row_size(wtype, time_embed_dim * time_embed_dim);  // time_embed_2_w
        mem_size += ggml_row_size(GGML_TYPE_F32, time_embed_dim);           // time_embed_2_b

        mem_size += ggml_row_size(GGML_TYPE_F16, model_channels * in_channels * 3 * 3);  // input_block_0_w
        mem_size += ggml_row_size(GGML_TYPE_F32, model_channels);                        // input_block_0_b

        // input_blocks
        int ds        = 1;
        int len_mults = channel_mult.size();
        for (int i = 0; i < len_mults; i++) {
            for (int j = 0; j < num_res_blocks; j++) {
                mem_size += input_res_blocks[i][j].calculate_mem_size(wtype);
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    mem_size += input_transformers[i][j].calculate_mem_size(wtype);
                }
            }
            if (i != len_mults - 1) {
                ds *= 2;
                mem_size += input_down_samples[i].calculate_mem_size(wtype);
            }
        }

        for (int i = 0; i < num_zero_convs; i++) {
            mem_size += ggml_row_size(GGML_TYPE_F16, zero_convs[i].channels * zero_convs[i].channels);
            mem_size += ggml_row_size(GGML_TYPE_F32, zero_convs[i].channels);
        }

        // middle_block
        mem_size += middle_block_0.calculate_mem_size(wtype);
        mem_size += middle_block_1.calculate_mem_size(wtype);
        mem_size += middle_block_2.calculate_mem_size(wtype);

        mem_size += ggml_row_size(GGML_TYPE_F16, middle_out_channel * middle_out_channel);  // middle_block_out_w
        mem_size += ggml_row_size(GGML_TYPE_F32, middle_out_channel);                       // middle_block_out_b

        return mem_size;
    }

    size_t get_num_tensors() {
        // in
        size_t num_tensors = 6;

        num_tensors += num_zero_convs * 2;

        // input blocks
        int ds        = 1;
        int len_mults = channel_mult.size();
        for (int i = 0; i < len_mults; i++) {
            for (int j = 0; j < num_res_blocks; j++) {
                num_tensors += 12;
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    num_tensors += input_transformers[i][j].get_num_tensors();
                }
            }
            if (i != len_mults - 1) {
                ds *= 2;
                num_tensors += 2;
            }
        }

        // middle blocks
        num_tensors += 13 * 2;
        num_tensors += middle_block_1.get_num_tensors();
        return num_tensors;
    }

    void init_params() {
        ggml_allocr* alloc = ggml_allocr_new_from_buffer(params_buffer);

        input_hint_block.init_params(params_ctx);

        time_embed_0_w = ggml_new_tensor_2d(params_ctx, wtype, model_channels, time_embed_dim);
        time_embed_0_b = ggml_new_tensor_1d(params_ctx, GGML_TYPE_F32, time_embed_dim);
        time_embed_2_w = ggml_new_tensor_2d(params_ctx, wtype, time_embed_dim, time_embed_dim);
        time_embed_2_b = ggml_new_tensor_1d(params_ctx, GGML_TYPE_F32, time_embed_dim);

        // input_blocks
        input_block_0_w = ggml_new_tensor_4d(params_ctx, GGML_TYPE_F16, 3, 3, in_channels, model_channels);
        input_block_0_b = ggml_new_tensor_1d(params_ctx, GGML_TYPE_F32, model_channels);

        int ds        = 1;
        int len_mults = channel_mult.size();
        for (int i = 0; i < len_mults; i++) {
            for (int j = 0; j < num_res_blocks; j++) {
                input_res_blocks[i][j].init_params(params_ctx, wtype);
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    input_transformers[i][j].init_params(params_ctx, alloc, wtype);
                }
            }
            if (i != len_mults - 1) {
                input_down_samples[i].init_params(params_ctx, wtype);
                ds *= 2;
            }
        }

        for (int i = 0; i < num_zero_convs; i++) {
            zero_convs[i].init_params(params_ctx);
        }

        // middle_blocks
        middle_block_0.init_params(params_ctx, wtype);
        middle_block_1.init_params(params_ctx, alloc, wtype);
        middle_block_2.init_params(params_ctx, wtype);

        // middle_block_out
        middle_block_out_w = ggml_new_tensor_4d(params_ctx, GGML_TYPE_F16, 1, 1, middle_out_channel, middle_out_channel);
        middle_block_out_b = ggml_new_tensor_1d(params_ctx, GGML_TYPE_F32, middle_out_channel);

        // alloc all tensors linked to this context
        for (struct ggml_tensor* t = ggml_get_first_tensor(params_ctx); t != NULL; t = ggml_get_next_tensor(params_ctx, t)) {
            if (t->data == NULL) {
                ggml_allocr_alloc(alloc, t);
            }
        }

        ggml_allocr_free(alloc);
    }

    bool load_from_file(const std::string& file_path, ggml_backend_t backend_, ggml_type wtype_) {
        LOG_INFO("loading control net from '%s'", file_path.c_str());

        std::map<std::string, ggml_tensor*> control_tensors;

        ModelLoader model_loader;
        if (!model_loader.init_from_file(file_path)) {
            LOG_ERROR("init control net model loader from file failed: '%s'", file_path.c_str());
            return false;
        }

        if (!alloc_params_buffer(backend_, wtype_)) {
            return false;
        }

        // prepare memory for the weights
        {
            init_params();
            map_by_name(control_tensors, "");
        }

        std::set<std::string> tensor_names_in_file;

        auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool {
            const std::string& name = tensor_storage.name;
            tensor_names_in_file.insert(name);

            struct ggml_tensor* real;
            if (control_tensors.find(name) != control_tensors.end()) {
                real = control_tensors[name];
            } else {
                LOG_ERROR("unknown tensor '%s' in model file", name.data());
                return true;
            }

            if (
                real->ne[0] != tensor_storage.ne[0] ||
                real->ne[1] != tensor_storage.ne[1] ||
                real->ne[2] != tensor_storage.ne[2] ||
                real->ne[3] != tensor_storage.ne[3]) {
                LOG_ERROR(
                    "tensor '%s' has wrong shape in model file: "
                    "got [%d, %d, %d, %d], expected [%d, %d, %d, %d]",
                    name.c_str(),
                    (int)tensor_storage.ne[0], (int)tensor_storage.ne[1], (int)tensor_storage.ne[2], (int)tensor_storage.ne[3],
                    (int)real->ne[0], (int)real->ne[1], (int)real->ne[2], (int)real->ne[3]);
                return false;
            }

            *dst_tensor = real;

            return true;
        };

        bool success = model_loader.load_tensors(on_new_tensor_cb, backend);

        bool some_tensor_not_init = false;

        for (auto pair : control_tensors) {
            if (tensor_names_in_file.find(pair.first) == tensor_names_in_file.end()) {
                LOG_ERROR("tensor '%s' not in model file", pair.first.c_str());
                some_tensor_not_init = true;
            }
        }

        if (some_tensor_not_init) {
            return false;
        }

        LOG_INFO("control net model loaded");
        return success;
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        input_hint_block.map_by_name(tensors, "");
        tensors[prefix + "time_embed.0.weight"] = time_embed_0_w;
        tensors[prefix + "time_embed.0.bias"]   = time_embed_0_b;
        tensors[prefix + "time_embed.2.weight"] = time_embed_2_w;
        tensors[prefix + "time_embed.2.bias"]   = time_embed_2_b;

        // input_blocks
        tensors[prefix + "input_blocks.0.0.weight"] = input_block_0_w;
        tensors[prefix + "input_blocks.0.0.bias"]   = input_block_0_b;

        int len_mults       = channel_mult.size();
        int input_block_idx = 0;
        int ds              = 1;
        for (int i = 0; i < len_mults; i++) {
            for (int j = 0; j < num_res_blocks; j++) {
                input_block_idx += 1;
                input_res_blocks[i][j].map_by_name(tensors, prefix + "input_blocks." + std::to_string(input_block_idx) + ".0.");
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    input_transformers[i][j].map_by_name(tensors, prefix + "input_blocks." + std::to_string(input_block_idx) + ".1.");
                }
            }
            if (i != len_mults - 1) {
                input_block_idx += 1;
                input_down_samples[i].map_by_name(tensors, prefix + "input_blocks." + std::to_string(input_block_idx) + ".0.");
                ds *= 2;
            }
        }

        for (int i = 0; i < num_zero_convs; i++) {
            tensors[prefix + "zero_convs." + std::to_string(i) + ".0.weight"] = zero_convs[i].conv_w;
            tensors[prefix + "zero_convs." + std::to_string(i) + ".0.bias"]   = zero_convs[i].conv_b;
        }

        // middle_blocks
        middle_block_0.map_by_name(tensors, prefix + "middle_block.0.");
        middle_block_1.map_by_name(tensors, prefix + "middle_block.1.");
        middle_block_2.map_by_name(tensors, prefix + "middle_block.2.");

        tensors[prefix + "middle_block_out.0.weight"] = middle_block_out_w;
        tensors[prefix + "middle_block_out.0.bias"]   = middle_block_out_b;
    }

    struct ggml_cgraph* build_graph_hint(struct ggml_tensor* hint) {
        // since we are using ggml-alloc, this buffer only needs enough space to hold the ggml_tensor and ggml_cgraph structs, but not the tensor data
        static size_t buf_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
        static std::vector<uint8_t> buf(buf_size);

        struct ggml_init_params params = {
            /*.mem_size   =*/buf_size,
            /*.mem_buffer =*/buf.data(),
            /*.no_alloc   =*/true,  // the tensors will be allocated later by ggml_allocr_alloc_graph()
        };

        struct ggml_context* ctx0 = ggml_init(params);
        struct ggml_cgraph* gf    = ggml_new_graph(ctx0);
        // temporal tensors for transfer tensors from cpu to gpu if needed
        struct ggml_tensor* hint_t = NULL;
        // it's performing a compute, check if backend isn't cpu
        if (!ggml_backend_is_cpu(backend)) {
            // pass input tensors to gpu memory
            hint_t = ggml_dup_tensor(ctx0, hint);
            ggml_allocr_alloc(compute_allocr, hint_t);
            // pass data to device backend
            if (!ggml_allocr_is_measure(compute_allocr)) {
                ggml_backend_tensor_set(hint_t, hint->data, 0, ggml_nbytes(hint));
            }
        } else {
            // if it's cpu backend just pass the same tensors
            hint_t = hint;
        }
        struct ggml_tensor* out = input_hint_block.forward(ctx0, hint_t);
        ggml_build_forward_expand(gf, out);
        ggml_free(ctx0);
        return gf;
    }

    void process_hint(struct ggml_tensor* output, int n_threads, struct ggml_tensor* hint) {
        // compute buffer size
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph_hint(hint);
        };
        GGMLModule::alloc_compute_buffer(get_graph);
        // perform computation
        GGMLModule::compute(get_graph, n_threads, output);
        GGMLModule::free_compute_buffer();
    }

    void forward(struct ggml_cgraph* gf,
                 struct ggml_context* ctx0,
                 struct ggml_tensor* x,
                 struct ggml_tensor* hint,
                 struct ggml_tensor* timesteps,
                 struct ggml_tensor* context,
                 struct ggml_tensor* t_emb = NULL) {
        // x: [N, in_channels, h, w]
        // timesteps: [N, ]
        // t_emb: [N, model_channels]
        // context: [N, max_position, hidden_size]([N, 77, 768])
        if (t_emb == NULL && timesteps != NULL) {
            t_emb = new_timestep_embedding(ctx0, compute_allocr, timesteps, model_channels);  // [N, model_channels]
        }

        // time_embed = nn.Sequential
        auto emb = ggml_nn_linear(ctx0, t_emb, time_embed_0_w, time_embed_0_b);
        emb      = ggml_silu_inplace(ctx0, emb);
        emb      = ggml_nn_linear(ctx0, emb, time_embed_2_w, time_embed_2_b);  // [N, time_embed_dim]

        // input_blocks
        int zero_conv_offset = 0;

        // input block 0
        struct ggml_tensor* h = ggml_nn_conv_2d(ctx0, x, input_block_0_w, input_block_0_b, 1, 1, 1, 1);  // [N, model_channels, h, w]
        h                     = ggml_add(ctx0, h, hint);

        auto h_c = ggml_nn_conv_2d(ctx0, h, zero_convs[zero_conv_offset].conv_w, zero_convs[zero_conv_offset].conv_b);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, h_c, controls[zero_conv_offset]));
        zero_conv_offset++;

        // input block 1-11
        int len_mults = channel_mult.size();
        int ds        = 1;
        for (int i = 0; i < len_mults; i++) {
            int mult = channel_mult[i];
            for (int j = 0; j < num_res_blocks; j++) {
                h = input_res_blocks[i][j].forward(ctx0, h, emb);  // [N, mult*model_channels, h, w]
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    h = input_transformers[i][j].forward(ctx0, h, context);  // [N, mult*model_channels, h, w]
                }
                h_c = ggml_nn_conv_2d(ctx0, h, zero_convs[zero_conv_offset].conv_w, zero_convs[zero_conv_offset].conv_b);
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, h_c, controls[zero_conv_offset]));
                zero_conv_offset++;
            }
            if (i != len_mults - 1) {
                ds *= 2;
                h   = input_down_samples[i].forward(ctx0, h);  // [N, mult*model_channels, h/(2^(i+1)), w/(2^(i+1))]
                h_c = ggml_nn_conv_2d(ctx0, h, zero_convs[zero_conv_offset].conv_w, zero_convs[zero_conv_offset].conv_b);
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, h_c, controls[zero_conv_offset]));
                zero_conv_offset++;
            }
        }
        // [N, 4*model_channels, h/8, w/8]

        // middle_block
        h = middle_block_0.forward(ctx0, h, emb);      // [N, 4*model_channels, h/8, w/8]
        h = middle_block_1.forward(ctx0, h, context);  // [N, 4*model_channels, h/8, w/8]
        h = middle_block_2.forward(ctx0, h, emb);      // [N, 4*model_channels, h/8, w/8]

        h_c = ggml_nn_conv_2d(ctx0, h, middle_block_out_w, middle_block_out_b);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, h_c, controls[zero_conv_offset]));
    }

    struct ggml_cgraph* build_graph(struct ggml_tensor* x,
                                    struct ggml_tensor* hint,
                                    struct ggml_tensor* timesteps,
                                    struct ggml_tensor* context,
                                    struct ggml_tensor* t_emb = NULL) {
        // since we are using ggml-alloc, this buffer only needs enough space to hold the ggml_tensor and ggml_cgraph structs, but not the tensor data
        static size_t buf_size = ggml_tensor_overhead() * CONTROL_NET_GRAPH_SIZE + ggml_graph_overhead();
        static std::vector<uint8_t> buf(buf_size);

        struct ggml_init_params params = {
            /*.mem_size   =*/buf_size,
            /*.mem_buffer =*/buf.data(),
            /*.no_alloc   =*/true,  // the tensors will be allocated later by ggml_allocr_alloc_graph()
        };
        // LOG_DEBUG("mem_size %u ", params.mem_size);

        struct ggml_context* ctx0 = ggml_init(params);

        struct ggml_cgraph* gf = ggml_new_graph_custom(ctx0, CONTROL_NET_GRAPH_SIZE, false);

        // temporal tensors for transfer tensors from cpu to gpu if needed
        struct ggml_tensor* x_t         = NULL;
        struct ggml_tensor* hint_t      = NULL;
        struct ggml_tensor* timesteps_t = NULL;
        struct ggml_tensor* context_t   = NULL;
        struct ggml_tensor* t_emb_t     = NULL;

        // it's performing a compute, check if backend isn't cpu
        if (!ggml_backend_is_cpu(backend)) {
            // pass input tensors to gpu memory
            x_t       = ggml_dup_tensor(ctx0, x);
            context_t = ggml_dup_tensor(ctx0, context);
            hint_t    = ggml_dup_tensor(ctx0, hint);
            ggml_allocr_alloc(compute_allocr, x_t);
            if (timesteps != NULL) {
                timesteps_t = ggml_dup_tensor(ctx0, timesteps);
                ggml_allocr_alloc(compute_allocr, timesteps_t);
            }
            ggml_allocr_alloc(compute_allocr, context_t);
            ggml_allocr_alloc(compute_allocr, hint_t);
            if (t_emb != NULL) {
                t_emb_t = ggml_dup_tensor(ctx0, t_emb);
                ggml_allocr_alloc(compute_allocr, t_emb_t);
            }
            // pass data to device backend
            if (!ggml_allocr_is_measure(compute_allocr)) {
                ggml_backend_tensor_set(x_t, x->data, 0, ggml_nbytes(x));
                ggml_backend_tensor_set(context_t, context->data, 0, ggml_nbytes(context));
                ggml_backend_tensor_set(hint_t, hint->data, 0, ggml_nbytes(hint));
                if (timesteps_t != NULL) {
                    ggml_backend_tensor_set(timesteps_t, timesteps->data, 0, ggml_nbytes(timesteps));
                }
                if (t_emb_t != NULL) {
                    ggml_backend_tensor_set(t_emb_t, t_emb->data, 0, ggml_nbytes(t_emb));
                }
            }
        } else {
            // if it's cpu backend just pass the same tensors
            x_t         = x;
            timesteps_t = timesteps;
            context_t   = context;
            t_emb_t     = t_emb;
            hint_t      = hint;
        }

        forward(gf, ctx0, x_t, hint_t, timesteps_t, context_t, t_emb_t);

        ggml_free(ctx0);

        return gf;
    }

    void alloc_compute_buffer(struct ggml_tensor* x,
                              struct ggml_tensor* hint,
                              struct ggml_tensor* context,
                              struct ggml_tensor* t_emb = NULL) {
        {
            struct ggml_init_params params;
            params.mem_size            = static_cast<size_t>(14 * ggml_tensor_overhead()) + 256;
            params.mem_buffer          = NULL;
            params.no_alloc            = true;
            control_ctx                = ggml_init(params);
            size_t control_buffer_size = 0;
            int w = x->ne[0], h = x->ne[1], steps = 0;
            for (int i = 0; i < (num_zero_convs + 1); i++) {
                bool last = i == num_zero_convs;
                int c     = last ? middle_out_channel : zero_convs[i].channels;
                if (!last && steps == 3) {
                    w /= 2;
                    h /= 2;
                    steps = 0;
                }
                controls.push_back(ggml_new_tensor_4d(control_ctx, GGML_TYPE_F32, w, h, c, 1));
                control_buffer_size += ggml_nbytes(controls[i]);
                steps++;
            }
            control_buffer = ggml_backend_alloc_ctx_tensors(control_ctx, backend);
        }
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(x, hint, NULL, context, t_emb);
        };
        GGMLModule::alloc_compute_buffer(get_graph);
    }

    void compute(int n_threads,
                 struct ggml_tensor* x,
                 struct ggml_tensor* hint,
                 struct ggml_tensor* context,
                 struct ggml_tensor* t_emb = NULL) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(x, hint, NULL, context, t_emb);
        };
        GGMLModule::compute(get_graph, n_threads, NULL);
    }

    void free_compute_buffer() {
        GGMLModule::free_compute_buffer();
        ggml_free(control_ctx);
        ggml_backend_buffer_free(control_buffer);
        control_buffer = NULL;
    }
};

#endif  // __CONTROL_HPP__