#ifndef __TAE_HPP__
#define __TAE_HPP__

#include "ggml_extend.hpp"

#include "model.h"

/*
    ===================================    TinyAutoEncoder  ===================================
    References:
    https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoder_tiny.py
    https://github.com/madebyollin/taesd/blob/main/taesd.py

*/
struct TAEBlock {
    int in_channels;
    int out_channels;

    // conv
    ggml_tensor* conv_0_w;  // [in_channels, out_channels, 3, 3]
    ggml_tensor* conv_0_b;  // [in_channels]
    ggml_tensor* conv_1_w;  // [out_channels, out_channels, 3, 3]
    ggml_tensor* conv_1_b;  // [out_channels]
    ggml_tensor* conv_2_w;  // [out_channels, out_channels, 3, 3]
    ggml_tensor* conv_2_b;  // [out_channels]

    // skip
    ggml_tensor* conv_skip_w;  // [in_channels, out_channels, 1, 1]

    size_t calculate_mem_size() {
        size_t mem_size = in_channels * out_channels * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_0_w
        mem_size += in_channels * ggml_type_size(GGML_TYPE_F32);                               // conv_0_b
        mem_size += out_channels * out_channels * 3 * 3 * ggml_type_size(GGML_TYPE_F16);       // conv_1_w
        mem_size += out_channels * ggml_type_size(GGML_TYPE_F32);                              // conv_1_b
        mem_size += out_channels * out_channels * 3 * 3 * ggml_type_size(GGML_TYPE_F16);       // conv_1_w
        mem_size += out_channels * ggml_type_size(GGML_TYPE_F32);                              // conv_1_b
        mem_size += out_channels * out_channels * 3 * 3 * ggml_type_size(GGML_TYPE_F16);       // conv_2_w
        mem_size += out_channels * ggml_type_size(GGML_TYPE_F32);                              // conv_2_b

        if (in_channels != out_channels) {
            mem_size += in_channels * out_channels * ggml_type_size(GGML_TYPE_F16);  // conv_skip_w
        }
        return mem_size;
    }

    int get_num_tensors() {
        return 6 + (in_channels != out_channels ? 1 : 0);
    }

    void init_params(ggml_context* ctx) {
        conv_0_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, out_channels, in_channels);
        conv_0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);

        conv_1_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, out_channels, out_channels);
        conv_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);

        conv_2_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, out_channels, out_channels);
        conv_2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);

        if (in_channels != out_channels) {
            conv_skip_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, out_channels, in_channels);
        }
    }

    void map_by_name(std::map<std::string, ggml_tensor*>& tensors, std::string prefix) {
        tensors[prefix + "conv.0.weight"] = conv_0_w;
        tensors[prefix + "conv.0.bias"]   = conv_0_b;

        tensors[prefix + "conv.2.weight"] = conv_1_w;
        tensors[prefix + "conv.2.bias"]   = conv_1_b;

        tensors[prefix + "conv.4.weight"] = conv_2_w;
        tensors[prefix + "conv.4.bias"]   = conv_2_b;

        if (in_channels != out_channels) {
            tensors[prefix + "skip.weight"] = conv_skip_w;
        }
    }

    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* x) {
        // conv(n_in, n_out)
        ggml_tensor* h;
        h = ggml_nn_conv_2d(ctx, x, conv_0_w, conv_0_b, 1, 1, 1, 1);
        h = ggml_relu_inplace(ctx, h);
        h = ggml_nn_conv_2d(ctx, h, conv_1_w, conv_1_b, 1, 1, 1, 1);
        h = ggml_relu_inplace(ctx, h);
        h = ggml_nn_conv_2d(ctx, h, conv_2_w, conv_2_b, 1, 1, 1, 1);

        // skip connection
        if (in_channels != out_channels) {
            // skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
            x = ggml_nn_conv_2d(ctx, x, conv_skip_w, NULL, 1, 1, 1, 1);
        }

        h = ggml_add(ctx, h, x);
        h = ggml_relu_inplace(ctx, h);
        return h;
    }
};

struct TinyEncoder {
    int in_channels = 3;
    int z_channels  = 4;
    int channels    = 64;
    int num_blocks  = 3;

    // input
    ggml_tensor* conv_input_w;  // [channels, in_channels, 3, 3]
    ggml_tensor* conv_input_b;  // [channels]
    TAEBlock initial_block;

    ggml_tensor* conv_1_w;  // [channels, channels, 3, 3]
    TAEBlock input_blocks[3];

    // middle
    ggml_tensor* conv_2_w;  // [channels, channels, 3, 3]
    TAEBlock middle_blocks[3];

    // output
    ggml_tensor* conv_3_w;  // [channels, channels, 3, 3]
    TAEBlock output_blocks[3];

    // final
    ggml_tensor* conv_final_w;  // [z_channels, channels, 3, 3]
    ggml_tensor* conv_final_b;  // [z_channels]

    TinyEncoder() {
        for (int i = 0; i < num_blocks; i++) {
            input_blocks[i].in_channels  = channels;
            input_blocks[i].out_channels = channels;

            middle_blocks[i].in_channels  = channels;
            middle_blocks[i].out_channels = channels;

            output_blocks[i].in_channels  = channels;
            output_blocks[i].out_channels = channels;
        }

        initial_block.in_channels  = channels;
        initial_block.out_channels = channels;
    }

    size_t calculate_mem_size() {
        size_t mem_size = channels * in_channels * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_input_w
        mem_size += channels * ggml_type_size(GGML_TYPE_F32);                              // conv_input_b

        mem_size += initial_block.calculate_mem_size();

        mem_size += channels * channels * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_1_w
        mem_size += channels * channels * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_2_w
        mem_size += channels * channels * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_3_w

        for (int i = 0; i < num_blocks; i++) {
            mem_size += input_blocks[i].calculate_mem_size();
            mem_size += middle_blocks[i].calculate_mem_size();
            mem_size += output_blocks[i].calculate_mem_size();
        }
        mem_size += z_channels * channels * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_input_w
        mem_size += z_channels * ggml_type_size(GGML_TYPE_F32);                     // conv_input_b
        return mem_size;
    }

    int get_num_tensors() {
        int num_tensors = 7;
        for (int i = 0; i < num_blocks; i++) {
            num_tensors += input_blocks[i].get_num_tensors();
            num_tensors += middle_blocks[i].get_num_tensors();
            num_tensors += output_blocks[i].get_num_tensors();
        }
        num_tensors += initial_block.get_num_tensors();
        return num_tensors;
    }

    void init_params(ggml_context* ctx) {
        conv_input_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, in_channels, channels);
        conv_input_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);

        initial_block.init_params(ctx);

        conv_1_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, channels, channels);
        conv_2_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, channels, channels);
        conv_3_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, channels, channels);

        conv_final_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, channels, z_channels);
        conv_final_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, z_channels);

        for (int i = 0; i < num_blocks; i++) {
            input_blocks[i].init_params(ctx);
            middle_blocks[i].init_params(ctx);
            output_blocks[i].init_params(ctx);
        }
    }

    void map_by_name(std::map<std::string, ggml_tensor*>& tensors, std::string prefix) {
        tensors[prefix + "0.weight"] = conv_input_w;
        tensors[prefix + "0.bias"]   = conv_input_b;

        initial_block.map_by_name(tensors, prefix + "1.");

        tensors[prefix + "2.weight"] = conv_1_w;
        for (int i = 0; i < num_blocks; i++) {
            input_blocks[i].map_by_name(tensors, prefix + std::to_string(i + 3) + ".");
        }

        tensors[prefix + "6.weight"] = conv_2_w;
        for (int i = 0; i < num_blocks; i++) {
            middle_blocks[i].map_by_name(tensors, prefix + std::to_string(i + 7) + ".");
        }

        tensors[prefix + "10.weight"] = conv_3_w;
        for (int i = 0; i < num_blocks; i++) {
            output_blocks[i].map_by_name(tensors, prefix + std::to_string(i + 11) + ".");
        }

        tensors[prefix + "14.weight"] = conv_final_w;
        tensors[prefix + "14.bias"]   = conv_final_b;
    }

    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* x) {
        // conv(3, 64)
        auto z = ggml_nn_conv_2d(ctx, x, conv_input_w, conv_input_b, 1, 1, 1, 1);

        // Block(64, 64)
        z = initial_block.forward(ctx, z);

        // conv(64, 64, stride=2, bias=False)
        z = ggml_nn_conv_2d(ctx, z, conv_1_w, NULL, 2, 2, 1, 1);

        // Block(64, 64), Block(64, 64), Block(64, 64)
        for (int i = 0; i < num_blocks; i++) {
            z = input_blocks[i].forward(ctx, z);
        }

        // conv(64, 64, stride=2, bias=False)
        z = ggml_nn_conv_2d(ctx, z, conv_2_w, NULL, 2, 2, 1, 1);

        // Block(64, 64), Block(64, 64), Block(64, 64)
        for (int i = 0; i < num_blocks; i++) {
            z = middle_blocks[i].forward(ctx, z);
        }

        // conv(64, 64, stride=2, bias=False)
        z = ggml_nn_conv_2d(ctx, z, conv_3_w, NULL, 2, 2, 1, 1);

        // Block(64, 64), Block(64, 64), Block(64, 64)
        for (int i = 0; i < num_blocks; i++) {
            z = output_blocks[i].forward(ctx, z);
        }

        // conv(64, 4)
        z = ggml_nn_conv_2d(ctx, z, conv_final_w, conv_final_b, 1, 1, 1, 1);
        return z;
    }
};

struct TinyDecoder {
    int z_channels      = 4;
    int channels        = 64;
    int output_channels = 3;
    int num_blocks      = 3;

    // input
    ggml_tensor* conv_input_w;  // [channels, z_channels, 3, 3]
    ggml_tensor* conv_input_b;  // [channels]
    TAEBlock input_blocks[3];
    ggml_tensor* conv_1_w;  // [channels, channels, 3, 3]

    // middle
    TAEBlock middle_blocks[3];
    ggml_tensor* conv_2_w;  // [channels, channels, 3, 3]

    // output
    TAEBlock output_blocks[3];
    ggml_tensor* conv_3_w;  // [channels, channels, 3, 3]

    // final
    TAEBlock final_block;
    ggml_tensor* conv_final_w;  // [output_channels, channels, 3, 3]
    ggml_tensor* conv_final_b;  // [output_channels]

    TinyDecoder() {
        for (int i = 0; i < num_blocks; i++) {
            input_blocks[i].in_channels  = channels;
            input_blocks[i].out_channels = channels;

            middle_blocks[i].in_channels  = channels;
            middle_blocks[i].out_channels = channels;

            output_blocks[i].in_channels  = channels;
            output_blocks[i].out_channels = channels;
        }

        final_block.in_channels  = channels;
        final_block.out_channels = channels;
    }

    size_t calculate_mem_size() {
        size_t mem_size = channels * z_channels * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_input_w
        mem_size += channels * ggml_type_size(GGML_TYPE_F32);                             // conv_input_b

        for (int i = 0; i < num_blocks; i++) {
            mem_size += input_blocks[i].calculate_mem_size();
        }
        mem_size += channels * channels * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_1_w

        for (int i = 0; i < num_blocks; i++) {
            mem_size += middle_blocks[i].calculate_mem_size();
        }
        mem_size += channels * channels * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_2_w

        for (int i = 0; i < num_blocks; i++) {
            mem_size += output_blocks[i].calculate_mem_size();
        }
        mem_size += channels * channels * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_3_w

        mem_size += final_block.calculate_mem_size();
        mem_size += output_channels * channels * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_input_w
        mem_size += output_channels * ggml_type_size(GGML_TYPE_F32);                     // conv_input_b
        return mem_size;
    }

    int get_num_tensors() {
        int num_tensors = 9;
        for (int i = 0; i < num_blocks; i++) {
            num_tensors += input_blocks[i].get_num_tensors();
            num_tensors += middle_blocks[i].get_num_tensors();
            num_tensors += output_blocks[i].get_num_tensors();
        }
        num_tensors += final_block.get_num_tensors();
        return num_tensors;
    }

    void init_params(ggml_allocr* alloc, ggml_context* ctx) {
        conv_input_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, z_channels, channels);
        conv_input_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);

        conv_1_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, channels, channels);
        conv_2_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, channels, channels);
        conv_3_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, channels, channels);

        conv_final_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, channels, output_channels);
        conv_final_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, output_channels);

        for (int i = 0; i < num_blocks; i++) {
            input_blocks[i].init_params(ctx);
            middle_blocks[i].init_params(ctx);
            output_blocks[i].init_params(ctx);
        }

        final_block.init_params(ctx);
    }

    void map_by_name(std::map<std::string, ggml_tensor*>& tensors, std::string prefix) {
        tensors[prefix + "0.weight"] = conv_input_w;
        tensors[prefix + "0.bias"]   = conv_input_b;

        for (int i = 0; i < num_blocks; i++) {
            input_blocks[i].map_by_name(tensors, prefix + std::to_string(i + 2) + ".");
        }

        tensors[prefix + "6.weight"] = conv_1_w;
        for (int i = 0; i < num_blocks; i++) {
            middle_blocks[i].map_by_name(tensors, prefix + std::to_string(i + 7) + ".");
        }

        tensors[prefix + "11.weight"] = conv_2_w;
        for (int i = 0; i < num_blocks; i++) {
            output_blocks[i].map_by_name(tensors, prefix + std::to_string(i + 12) + ".");
        }

        tensors[prefix + "16.weight"] = conv_3_w;

        final_block.map_by_name(tensors, prefix + "17.");

        tensors[prefix + "18.weight"] = conv_final_w;
        tensors[prefix + "18.bias"]   = conv_final_b;
    }

    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* z) {
        // torch.tanh(x / 3) * 3
        auto h = ggml_scale(ctx, z, 1.0f / 3.0f);
        h      = ggml_tanh_inplace(ctx, h);
        h      = ggml_scale(ctx, h, 3.0f);

        // conv(4, 64)
        h = ggml_nn_conv_2d(ctx, h, conv_input_w, conv_input_b, 1, 1, 1, 1);

        // nn.ReLU()
        h = ggml_relu_inplace(ctx, h);

        // Block(64, 64), Block(64, 64), Block(64, 64)
        for (int i = 0; i < num_blocks; i++) {
            h = input_blocks[i].forward(ctx, h);
        }

        // nn.Upsample(scale_factor=2)
        h = ggml_upscale(ctx, h, 2);

        // conv(64, 64, bias=False)
        h = ggml_nn_conv_2d(ctx, h, conv_1_w, NULL, 1, 1, 1, 1);

        // Block(64, 64), Block(64, 64), Block(64, 64)
        for (int i = 0; i < num_blocks; i++) {
            h = middle_blocks[i].forward(ctx, h);
        }

        // nn.Upsample(scale_factor=2)
        h = ggml_upscale(ctx, h, 2);

        // conv(64, 64, bias=False)
        h = ggml_nn_conv_2d(ctx, h, conv_2_w, NULL, 1, 1, 1, 1);

        // Block(64, 64), Block(64, 64), Block(64, 64)
        for (int i = 0; i < num_blocks; i++) {
            h = output_blocks[i].forward(ctx, h);
        }

        // nn.Upsample(scale_factor=2)
        h = ggml_upscale(ctx, h, 2);

        // conv(64, 64, bias=False)
        h = ggml_nn_conv_2d(ctx, h, conv_3_w, NULL, 1, 1, 1, 1);

        // Block(64, 64)
        h = final_block.forward(ctx, h);

        // conv(64, 3)
        h = ggml_nn_conv_2d(ctx, h, conv_final_w, conv_final_b, 1, 1, 1, 1);
        return h;
    }
};

struct TinyAutoEncoder : public GGMLModule {
    TinyEncoder encoder;
    TinyDecoder decoder;
    bool decode_only = false;

    TinyAutoEncoder(bool decoder_only_ = true)
        : decode_only(decoder_only_) {
        name = "tae";
    }

    size_t calculate_mem_size() {
        size_t mem_size = decoder.calculate_mem_size();
        if (!decode_only) {
            mem_size += encoder.calculate_mem_size();
        }
        mem_size += 1024;  // padding
        return mem_size;
    }

    size_t get_num_tensors() {
        size_t num_tensors = decoder.get_num_tensors();
        if (!decode_only) {
            num_tensors += encoder.get_num_tensors();
        }
        return num_tensors;
    }

    void init_params() {
        ggml_allocr* alloc = ggml_allocr_new_from_buffer(params_buffer);
        decoder.init_params(alloc, params_ctx);
        if (!decode_only) {
            encoder.init_params(params_ctx);
        }

        // alloc all tensors linked to this context
        for (struct ggml_tensor* t = ggml_get_first_tensor(params_ctx); t != NULL; t = ggml_get_next_tensor(params_ctx, t)) {
            if (t->data == NULL) {
                ggml_allocr_alloc(alloc, t);
            }
        }
        ggml_allocr_free(alloc);
    }

    void map_by_name(std::map<std::string, ggml_tensor*>& tensors) {
        decoder.map_by_name(tensors, "decoder.layers.");
        encoder.map_by_name(tensors, "encoder.layers.");
    }

    bool load_from_file(const std::string& file_path, ggml_backend_t backend) {
        LOG_INFO("loading taesd from '%s'", file_path.c_str());

        if (!alloc_params_buffer(backend)) {
            return false;
        }

        std::map<std::string, ggml_tensor*> taesd_tensors;

        // prepare memory for the weights
        {
            init_params();
            map_by_name(taesd_tensors);
        }

        std::map<std::string, struct ggml_tensor*> tensors_need_to_load;
        std::set<std::string> ignore_tensors;
        for (auto& pair : taesd_tensors) {
            const std::string& name = pair.first;

            if (decode_only && starts_with(name, "encoder")) {
                ignore_tensors.insert(name);
                continue;
            }

            tensors_need_to_load.insert(pair);
        }

        ModelLoader model_loader;
        if (!model_loader.init_from_file(file_path)) {
            LOG_ERROR("init taesd model loader from file failed: '%s'", file_path.c_str());
            return false;
        }

        bool success = model_loader.load_tensors(tensors_need_to_load, backend, ignore_tensors);

        if (!success) {
            LOG_ERROR("load tae tensors from model loader failed");
            return false;
        }

        LOG_INFO("taesd model loaded");
        return success;
    }

    struct ggml_cgraph* build_graph(struct ggml_tensor* z, bool decode_graph) {
        // since we are using ggml-alloc, this buffer only needs enough space to hold the ggml_tensor and ggml_cgraph structs, but not the tensor data
        static size_t buf_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
        static std::vector<uint8_t> buf(buf_size);

        struct ggml_init_params params = {
            /*.mem_size   =*/buf_size,
            /*.mem_buffer =*/buf.data(),
            /*.no_alloc   =*/true,  // the tensors will be allocated later by ggml_allocr_alloc_graph()
        };
        // LOG_DEBUG("mem_size %u ", params.mem_size);

        struct ggml_context* ctx0 = ggml_init(params);

        struct ggml_cgraph* gf = ggml_new_graph(ctx0);

        struct ggml_tensor* z_ = NULL;

        // it's performing a compute, check if backend isn't cpu
        if (!ggml_backend_is_cpu(backend)) {
            // pass input tensors to gpu memory
            z_ = ggml_dup_tensor(ctx0, z);
            ggml_allocr_alloc(compute_allocr, z_);

            // pass data to device backend
            if (!ggml_allocr_is_measure(compute_allocr)) {
                ggml_backend_tensor_set(z_, z->data, 0, ggml_nbytes(z));
            }
        } else {
            z_ = z;
        }

        struct ggml_tensor* out = decode_graph ? decoder.forward(ctx0, z_) : encoder.forward(ctx0, z_);

        ggml_build_forward_expand(gf, out);
        ggml_free(ctx0);

        return gf;
    }

    void alloc_compute_buffer(struct ggml_tensor* x, bool decode) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(x, decode);
        };
        GGMLModule::alloc_compute_buffer(get_graph);
    }

    void compute(struct ggml_tensor* work_result, int n_threads, struct ggml_tensor* z, bool decode_graph) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(z, decode_graph);
        };
        GGMLModule::compute(get_graph, n_threads, work_result);
    }
};

#endif  // __TAE_HPP__