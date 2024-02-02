#ifndef __ESRGAN_HPP__
#define __ESRGAN_HPP__

#include "ggml_extend.hpp"
#include "model.h"

/*
    ===================================    ESRGAN  ===================================
    References:
    https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan.py
    https://github.com/XPixelGroup/BasicSR/blob/v1.4.2/basicsr/archs/rrdbnet_arch.py

*/

struct ResidualDenseBlock {
    int num_features;
    int num_grow_ch;
    ggml_tensor* conv1_w;  // [num_grow_ch, num_features, 3, 3]
    ggml_tensor* conv1_b;  // [num_grow_ch]

    ggml_tensor* conv2_w;  // [num_grow_ch, num_features + num_grow_ch, 3, 3]
    ggml_tensor* conv2_b;  // [num_grow_ch]

    ggml_tensor* conv3_w;  // [num_grow_ch, num_features + 2 * num_grow_ch, 3, 3]
    ggml_tensor* conv3_b;  // [num_grow_ch]

    ggml_tensor* conv4_w;  // [num_grow_ch, num_features + 3 * num_grow_ch, 3, 3]
    ggml_tensor* conv4_b;  // [num_grow_ch]

    ggml_tensor* conv5_w;  // [num_features, num_features + 4 * num_grow_ch, 3, 3]
    ggml_tensor* conv5_b;  // [num_features]

    ResidualDenseBlock() {}

    ResidualDenseBlock(int num_feat, int n_grow_ch) {
        num_features = num_feat;
        num_grow_ch  = n_grow_ch;
    }

    size_t calculate_mem_size() {
        size_t mem_size = num_features * num_grow_ch * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv1_w
        mem_size += num_grow_ch * ggml_type_size(GGML_TYPE_F32);                               // conv1_b

        mem_size += (num_features + num_grow_ch) * num_grow_ch * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv2_w
        mem_size += num_grow_ch * ggml_type_size(GGML_TYPE_F32);                                         // conv2_b

        mem_size += (num_features + 2 * num_grow_ch) * num_grow_ch * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv3_w
        mem_size += num_grow_ch * ggml_type_size(GGML_TYPE_F32);                                             // conv3_w

        mem_size += (num_features + 3 * num_grow_ch) * num_grow_ch * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv4_w
        mem_size += num_grow_ch * ggml_type_size(GGML_TYPE_F32);                                             // conv4_w

        mem_size += (num_features + 4 * num_grow_ch) * num_features * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv5_w
        mem_size += num_features * ggml_type_size(GGML_TYPE_F32);                                             // conv5_w

        return mem_size;
    }

    int get_num_tensors() {
        int num_tensors = 10;
        return num_tensors;
    }

    void init_params(ggml_context* ctx) {
        conv1_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, num_features, num_grow_ch);
        conv1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_grow_ch);
        conv2_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, num_features + num_grow_ch, num_grow_ch);
        conv2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_grow_ch);
        conv3_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, num_features + 2 * num_grow_ch, num_grow_ch);
        conv3_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_grow_ch);
        conv4_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, num_features + 3 * num_grow_ch, num_grow_ch);
        conv4_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_grow_ch);
        conv5_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, num_features + 4 * num_grow_ch, num_features);
        conv5_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_features);
    }

    void map_by_name(std::map<std::string, ggml_tensor*>& tensors, std::string prefix) {
        tensors[prefix + "conv1.weight"] = conv1_w;
        tensors[prefix + "conv1.bias"]   = conv1_b;

        tensors[prefix + "conv2.weight"] = conv2_w;
        tensors[prefix + "conv2.bias"]   = conv2_b;

        tensors[prefix + "conv3.weight"] = conv3_w;
        tensors[prefix + "conv3.bias"]   = conv3_b;

        tensors[prefix + "conv4.weight"] = conv4_w;
        tensors[prefix + "conv4.bias"]   = conv4_b;

        tensors[prefix + "conv5.weight"] = conv5_w;
        tensors[prefix + "conv5.bias"]   = conv5_b;
    }

    ggml_tensor* forward(ggml_context* ctx, float out_scale, ggml_tensor* x /* feat */) {
        // x1 = self.lrelu(self.conv1(x))
        ggml_tensor* x1 = ggml_nn_conv_2d(ctx, x, conv1_w, conv1_b, 1, 1, 1, 1);
        x1              = ggml_leaky_relu(ctx, x1, 0.2f, true);

        // x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        ggml_tensor* x_cat = ggml_concat(ctx, x, x1);
        ggml_tensor* x2    = ggml_nn_conv_2d(ctx, x_cat, conv2_w, conv2_b, 1, 1, 1, 1);
        x2                 = ggml_leaky_relu(ctx, x2, 0.2f, true);

        // x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x_cat           = ggml_concat(ctx, x_cat, x2);
        ggml_tensor* x3 = ggml_nn_conv_2d(ctx, x_cat, conv3_w, conv3_b, 1, 1, 1, 1);
        x3              = ggml_leaky_relu(ctx, x3, 0.2f, true);

        // x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x_cat           = ggml_concat(ctx, x_cat, x3);
        ggml_tensor* x4 = ggml_nn_conv_2d(ctx, x_cat, conv4_w, conv4_b, 1, 1, 1, 1);
        x4              = ggml_leaky_relu(ctx, x4, 0.2f, true);

        // self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x_cat           = ggml_concat(ctx, x_cat, x4);
        ggml_tensor* x5 = ggml_nn_conv_2d(ctx, x_cat, conv5_w, conv5_b, 1, 1, 1, 1);

        // return x5 * 0.2 + x
        x5 = ggml_add(ctx, ggml_scale(ctx, x5, out_scale), x);
        return x5;
    }
};

struct EsrganBlock {
    ResidualDenseBlock rd_blocks[3];
    int num_residual_blocks = 3;

    EsrganBlock() {}

    EsrganBlock(int num_feat, int num_grow_ch) {
        for (int i = 0; i < num_residual_blocks; i++) {
            rd_blocks[i] = ResidualDenseBlock(num_feat, num_grow_ch);
        }
    }

    int get_num_tensors() {
        int num_tensors = 0;
        for (int i = 0; i < num_residual_blocks; i++) {
            num_tensors += rd_blocks[i].get_num_tensors();
        }
        return num_tensors;
    }

    size_t calculate_mem_size() {
        size_t mem_size = 0;
        for (int i = 0; i < num_residual_blocks; i++) {
            mem_size += rd_blocks[i].calculate_mem_size();
        }
        return mem_size;
    }

    void init_params(ggml_context* ctx) {
        for (int i = 0; i < num_residual_blocks; i++) {
            rd_blocks[i].init_params(ctx);
        }
    }

    void map_by_name(std::map<std::string, ggml_tensor*>& tensors, std::string prefix) {
        for (int i = 0; i < num_residual_blocks; i++) {
            rd_blocks[i].map_by_name(tensors, prefix + "rdb" + std::to_string(i + 1) + ".");
        }
    }

    ggml_tensor* forward(ggml_context* ctx, float out_scale, ggml_tensor* x) {
        ggml_tensor* out = x;
        for (int i = 0; i < num_residual_blocks; i++) {
            // out = self.rdb...(x)
            out = rd_blocks[i].forward(ctx, out_scale, out);
        }
        // return out * 0.2 + x
        out = ggml_add(ctx, ggml_scale(ctx, out, out_scale), x);
        return out;
    }
};

struct ESRGAN : public GGMLModule {
    int scale        = 4;  // default RealESRGAN_x4plus_anime_6B
    int num_blocks   = 6;  // default RealESRGAN_x4plus_anime_6B
    int in_channels  = 3;
    int out_channels = 3;
    int num_features = 64;   // default RealESRGAN_x4plus_anime_6B
    int num_grow_ch  = 32;   // default RealESRGAN_x4plus_anime_6B
    int tile_size    = 128;  // avoid cuda OOM for 4gb VRAM

    ggml_tensor* conv_first_w;  // [num_features, in_channels, 3, 3]
    ggml_tensor* conv_first_b;  // [num_features]

    EsrganBlock body_blocks[6];
    ggml_tensor* conv_body_w;  // [num_features, num_features, 3, 3]
    ggml_tensor* conv_body_b;  // [num_features]

    // upsample
    ggml_tensor* conv_up1_w;  // [num_features, num_features, 3, 3]
    ggml_tensor* conv_up1_b;  // [num_features]
    ggml_tensor* conv_up2_w;  // [num_features, num_features, 3, 3]
    ggml_tensor* conv_up2_b;  // [num_features]

    ggml_tensor* conv_hr_w;    // [num_features, num_features, 3, 3]
    ggml_tensor* conv_hr_b;    // [num_features]
    ggml_tensor* conv_last_w;  // [out_channels, num_features, 3, 3]
    ggml_tensor* conv_last_b;  // [out_channels]

    bool decode_only = false;

    ESRGAN() {
        name = "esrgan";
        for (int i = 0; i < num_blocks; i++) {
            body_blocks[i] = EsrganBlock(num_features, num_grow_ch);
        }
    }

    size_t calculate_mem_size() {
        size_t mem_size = num_features * in_channels * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_first_w
        mem_size += num_features * ggml_type_size(GGML_TYPE_F32);                              // conv_first_b

        for (int i = 0; i < num_blocks; i++) {
            mem_size += body_blocks[i].calculate_mem_size();
        }

        mem_size += num_features * num_features * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_body_w
        mem_size += num_features * ggml_type_size(GGML_TYPE_F32);                         // conv_body_w

        // upsample
        mem_size += num_features * num_features * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_up1_w
        mem_size += num_features * ggml_type_size(GGML_TYPE_F32);                         // conv_up1_b

        mem_size += num_features * num_features * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_up2_w
        mem_size += num_features * ggml_type_size(GGML_TYPE_F32);                         // conv_up2_b

        mem_size += num_features * num_features * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_hr_w
        mem_size += num_features * ggml_type_size(GGML_TYPE_F32);                         // conv_hr_b

        mem_size += out_channels * num_features * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_last_w
        mem_size += out_channels * ggml_type_size(GGML_TYPE_F32);                         // conv_last_b
        return mem_size;
    }

    size_t get_num_tensors() {
        size_t num_tensors = 12;
        for (int i = 0; i < num_blocks; i++) {
            num_tensors += body_blocks[i].get_num_tensors();
        }
        return num_tensors;
    }

    void init_params() {
        ggml_allocr* alloc = ggml_allocr_new_from_buffer(params_buffer);
        conv_first_w       = ggml_new_tensor_4d(params_ctx, GGML_TYPE_F16, 3, 3, in_channels, num_features);
        conv_first_b       = ggml_new_tensor_1d(params_ctx, GGML_TYPE_F32, num_features);
        conv_body_w        = ggml_new_tensor_4d(params_ctx, GGML_TYPE_F16, 3, 3, num_features, num_features);
        conv_body_b        = ggml_new_tensor_1d(params_ctx, GGML_TYPE_F32, num_features);
        conv_up1_w         = ggml_new_tensor_4d(params_ctx, GGML_TYPE_F16, 3, 3, num_features, num_features);
        conv_up1_b         = ggml_new_tensor_1d(params_ctx, GGML_TYPE_F32, num_features);
        conv_up2_w         = ggml_new_tensor_4d(params_ctx, GGML_TYPE_F16, 3, 3, num_features, num_features);
        conv_up2_b         = ggml_new_tensor_1d(params_ctx, GGML_TYPE_F32, num_features);
        conv_hr_w          = ggml_new_tensor_4d(params_ctx, GGML_TYPE_F16, 3, 3, num_features, num_features);
        conv_hr_b          = ggml_new_tensor_1d(params_ctx, GGML_TYPE_F32, num_features);
        conv_last_w        = ggml_new_tensor_4d(params_ctx, GGML_TYPE_F16, 3, 3, num_features, out_channels);
        conv_last_b        = ggml_new_tensor_1d(params_ctx, GGML_TYPE_F32, out_channels);

        for (int i = 0; i < num_blocks; i++) {
            body_blocks[i].init_params(params_ctx);
        }

        // alloc all tensors linked to this context
        for (struct ggml_tensor* t = ggml_get_first_tensor(params_ctx); t != NULL; t = ggml_get_next_tensor(params_ctx, t)) {
            if (t->data == NULL) {
                ggml_allocr_alloc(alloc, t);
            }
        }
        ggml_allocr_free(alloc);
    }

    bool load_from_file(const std::string& file_path, ggml_backend_t backend) {
        LOG_INFO("loading esrgan from '%s'", file_path.c_str());

        if (!alloc_params_buffer(backend)) {
            return false;
        }

        std::map<std::string, ggml_tensor*> esrgan_tensors;

        // prepare memory for the weights
        {
            init_params();
            map_by_name(esrgan_tensors);
        }

        ModelLoader model_loader;
        if (!model_loader.init_from_file(file_path)) {
            LOG_ERROR("init esrgan model loader from file failed: '%s'", file_path.c_str());
            return false;
        }

        bool success = model_loader.load_tensors(esrgan_tensors, backend);

        if (!success) {
            LOG_ERROR("load esrgan tensors from model loader failed");
            return false;
        }

        LOG_INFO("esrgan model loaded");
        return success;
    }

    void map_by_name(std::map<std::string, ggml_tensor*>& tensors) {
        tensors["conv_first.weight"] = conv_first_w;
        tensors["conv_first.bias"]   = conv_first_b;

        for (int i = 0; i < num_blocks; i++) {
            body_blocks[i].map_by_name(tensors, "body." + std::to_string(i) + ".");
        }

        tensors["conv_body.weight"] = conv_body_w;
        tensors["conv_body.bias"]   = conv_body_b;

        tensors["conv_up1.weight"] = conv_up1_w;
        tensors["conv_up1.bias"]   = conv_up1_b;
        tensors["conv_up2.weight"] = conv_up2_w;
        tensors["conv_up2.bias"]   = conv_up2_b;
        tensors["conv_hr.weight"]  = conv_hr_w;
        tensors["conv_hr.bias"]    = conv_hr_b;

        tensors["conv_last.weight"] = conv_last_w;
        tensors["conv_last.bias"]   = conv_last_b;
    }

    ggml_tensor* forward(ggml_context* ctx0, float out_scale, ggml_tensor* x /* feat */) {
        // feat = self.conv_first(feat)
        auto h = ggml_nn_conv_2d(ctx0, x, conv_first_w, conv_first_b, 1, 1, 1, 1);

        auto body_h = h;
        // self.body(feat)
        for (int i = 0; i < num_blocks; i++) {
            body_h = body_blocks[i].forward(ctx0, out_scale, body_h);
        }

        // body_feat = self.conv_body(self.body(feat))
        body_h = ggml_nn_conv_2d(ctx0, body_h, conv_body_w, conv_body_b, 1, 1, 1, 1);

        // feat = feat + body_feat
        h = ggml_add(ctx0, h, body_h);

        // upsample
        // feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        h = ggml_upscale(ctx0, h, 2);
        h = ggml_nn_conv_2d(ctx0, h, conv_up1_w, conv_up1_b, 1, 1, 1, 1);
        h = ggml_leaky_relu(ctx0, h, 0.2f, true);

        // feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        h = ggml_upscale(ctx0, h, 2);
        h = ggml_nn_conv_2d(ctx0, h, conv_up2_w, conv_up2_b, 1, 1, 1, 1);
        h = ggml_leaky_relu(ctx0, h, 0.2f, true);

        // out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        h = ggml_nn_conv_2d(ctx0, h, conv_hr_w, conv_hr_b, 1, 1, 1, 1);
        h = ggml_leaky_relu(ctx0, h, 0.2f, true);

        h = ggml_nn_conv_2d(ctx0, h, conv_last_w, conv_last_b, 1, 1, 1, 1);
        return h;
    }

    struct ggml_cgraph* build_graph(struct ggml_tensor* x) {
        // since we are using ggml-alloc, this buffer only needs enough space to hold the ggml_tensor and ggml_cgraph structs, but not the tensor data
        static size_t buf_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
        static std::vector<uint8_t> buf(buf_size);

        struct ggml_init_params params = {
            /*.mem_size   =*/buf_size,
            /*.mem_buffer =*/buf.data(),
            /*.no_alloc   =*/true,  // the tensors will be allocated later by ggml_allocr_alloc_graph()
        };

        struct ggml_context* ctx0 = ggml_init(params);

        struct ggml_cgraph* gf = ggml_new_graph(ctx0);

        struct ggml_tensor* x_ = NULL;
        float out_scale        = 0.2f;

        // it's performing a compute, check if backend isn't cpu
        if (!ggml_backend_is_cpu(backend)) {
            // pass input tensors to gpu memory
            x_ = ggml_dup_tensor(ctx0, x);
            ggml_allocr_alloc(compute_allocr, x_);

            // pass data to device backend
            if (!ggml_allocr_is_measure(compute_allocr)) {
                ggml_backend_tensor_set(x_, x->data, 0, ggml_nbytes(x));
            }
        } else {
            x_ = x;
        }

        struct ggml_tensor* out = forward(ctx0, out_scale, x);

        ggml_build_forward_expand(gf, out);
        ggml_free(ctx0);

        return gf;
    }

    void alloc_compute_buffer(struct ggml_tensor* x) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(x);
        };
        GGMLModule::alloc_compute_buffer(get_graph);
    }

    void compute(struct ggml_tensor* work_result, const int n_threads, struct ggml_tensor* x) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(x);
        };
        GGMLModule::compute(get_graph, n_threads, work_result);
    }
};

#endif  // __ESRGAN_HPP__