#ifndef __VAE_HPP__
#define __VAE_HPP__

#include "common.hpp"
#include "ggml_extend.hpp"

/*================================================== AutoEncoderKL ===================================================*/

#define VAE_GRAPH_SIZE 10240

struct ResnetBlock {
    // network hparams
    int in_channels;
    int out_channels;

    // network params
    struct ggml_tensor* norm1_w;  // [in_channels, ]
    struct ggml_tensor* norm1_b;  // [in_channels, ]

    struct ggml_tensor* conv1_w;  // [out_channels, in_channels, 3, 3]
    struct ggml_tensor* conv1_b;  // [out_channels, ]

    struct ggml_tensor* norm2_w;  // [out_channels, ]
    struct ggml_tensor* norm2_b;  // [out_channels, ]

    struct ggml_tensor* conv2_w;  // [out_channels, out_channels, 3, 3]
    struct ggml_tensor* conv2_b;  // [out_channels, ]

    // nin_shortcut, only if out_channels != in_channels
    struct ggml_tensor* nin_shortcut_w;  // [out_channels, in_channels, 1, 1]
    struct ggml_tensor* nin_shortcut_b;  // [out_channels, ]

    size_t calculate_mem_size(ggml_type wtype) {
        double mem_size = 0;
        mem_size += 2 * ggml_row_size(GGML_TYPE_F32, in_channels);                      // norm1_w/b
        mem_size += ggml_row_size(GGML_TYPE_F16, out_channels * in_channels * 3 * 3);   // conv1_w
        mem_size += 4 * ggml_row_size(GGML_TYPE_F32, out_channels);                     // conv1_b/norm2_w/norm2_b/conv2_b
        mem_size += ggml_row_size(GGML_TYPE_F16, out_channels * out_channels * 3 * 3);  // conv2_w

        if (out_channels != in_channels) {
            mem_size += ggml_row_size(GGML_TYPE_F16, out_channels * in_channels * 1 * 1);  // nin_shortcut_w
            mem_size += ggml_row_size(GGML_TYPE_F32, out_channels);                        // nin_shortcut_b
        }
        return static_cast<size_t>(mem_size);
    }

    void init_params(struct ggml_context* ctx, ggml_type wtype) {
        norm1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
        norm1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
        conv1_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, in_channels, out_channels);
        conv1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);

        norm2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
        norm2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
        conv2_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, out_channels, out_channels);
        conv2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);

        if (out_channels != in_channels) {
            nin_shortcut_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, in_channels, out_channels);
            nin_shortcut_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
        }
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        tensors[prefix + "norm1.weight"] = norm1_w;
        tensors[prefix + "norm1.bias"]   = norm1_b;
        tensors[prefix + "conv1.weight"] = conv1_w;
        tensors[prefix + "conv1.bias"]   = conv1_b;

        tensors[prefix + "norm2.weight"] = norm2_w;
        tensors[prefix + "norm2.bias"]   = norm2_b;
        tensors[prefix + "conv2.weight"] = conv2_w;
        tensors[prefix + "conv2.bias"]   = conv2_b;

        if (out_channels != in_channels) {
            tensors[prefix + "nin_shortcut.weight"] = nin_shortcut_w;
            tensors[prefix + "nin_shortcut.bias"]   = nin_shortcut_b;
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* z) {
        // z: [N, in_channels, h, w]

        auto h = ggml_nn_group_norm(ctx, z, norm1_w, norm1_b);
        h      = ggml_silu_inplace(ctx, h);
        h      = ggml_nn_conv_2d(ctx, h, conv1_w, conv1_b, 1, 1, 1, 1);  // [N, out_channels, h, w]
        h      = ggml_nn_group_norm(ctx, h, norm2_w, norm2_b);
        h      = ggml_silu_inplace(ctx, h);
        // dropout, skip for inference
        h = ggml_nn_conv_2d(ctx, h, conv2_w, conv2_b, 1, 1, 1, 1);  // [N, out_channels, h, w]

        // skip connection
        if (out_channels != in_channels) {
            z = ggml_nn_conv_2d(ctx, z, nin_shortcut_w, nin_shortcut_b);  // [N, out_channels, h, w]
        }

        h = ggml_add(ctx, h, z);
        return h;  // [N, out_channels, h, w]
    }
};

struct AttnBlock {
    int in_channels;  // mult * model_channels

    // group norm
    struct ggml_tensor* norm_w;  // [in_channels,]
    struct ggml_tensor* norm_b;  // [in_channels,]

    // q/k/v
    struct ggml_tensor* q_w;  // [in_channels, in_channels, 1, 1]
    struct ggml_tensor* q_b;  // [in_channels,]
    struct ggml_tensor* k_w;  // [in_channels, in_channels, 1, 1]
    struct ggml_tensor* k_b;  // [in_channels,]
    struct ggml_tensor* v_w;  // [in_channels, in_channels, 1, 1]
    struct ggml_tensor* v_b;  // [in_channels,]

    // proj_out
    struct ggml_tensor* proj_out_w;  // [in_channels, in_channels, 1, 1]
    struct ggml_tensor* proj_out_b;  // [in_channels,]

    size_t calculate_mem_size(ggml_type wtype) {
        double mem_size = 0;
        mem_size += 6 * ggml_row_size(GGML_TYPE_F32, in_channels);                        // norm_w/norm_b/q_b/k_v/v_b/proj_out_b
        mem_size += 4 * ggml_row_size(GGML_TYPE_F16, in_channels * in_channels * 1 * 1);  // q_w/k_w/v_w/proj_out_w                                            // object overhead
        return static_cast<size_t>(mem_size);
    }

    void init_params(struct ggml_context* ctx, ggml_allocr* alloc, ggml_type wtype) {
        norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
        norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);

        q_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, in_channels, in_channels);
        q_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
        k_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, in_channels, in_channels);
        k_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
        v_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, in_channels, in_channels);
        v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);

        proj_out_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, in_channels, in_channels);
        proj_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        tensors[prefix + "norm.weight"]     = norm_w;
        tensors[prefix + "norm.bias"]       = norm_b;
        tensors[prefix + "q.weight"]        = q_w;
        tensors[prefix + "q.bias"]          = q_b;
        tensors[prefix + "k.weight"]        = k_w;
        tensors[prefix + "k.bias"]          = k_b;
        tensors[prefix + "v.weight"]        = v_w;
        tensors[prefix + "v.bias"]          = v_b;
        tensors[prefix + "proj_out.weight"] = proj_out_w;
        tensors[prefix + "proj_out.bias"]   = proj_out_b;
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, in_channels, h, w]

        auto h_ = ggml_nn_group_norm(ctx, x, norm_w, norm_b);

        const int64_t n = h_->ne[3];
        const int64_t c = h_->ne[2];
        const int64_t h = h_->ne[1];
        const int64_t w = h_->ne[0];

        auto q = ggml_nn_conv_2d(ctx, h_, q_w, q_b);  // [N, in_channels, h, w]
        auto k = ggml_nn_conv_2d(ctx, h_, k_w, k_b);  // [N, in_channels, h, w]
        auto v = ggml_nn_conv_2d(ctx, h_, v_w, v_b);  // [N, in_channels, h, w]

        q = ggml_cont(ctx, ggml_permute(ctx, q, 1, 2, 0, 3));  // [N, h, w, in_channels]
        q = ggml_reshape_3d(ctx, q, c, h * w, n);              // [N, h * w, in_channels]

        k = ggml_cont(ctx, ggml_permute(ctx, k, 1, 2, 0, 3));  // [N, h, w, in_channels]
        k = ggml_reshape_3d(ctx, k, c, h * w, n);              // [N, h * w, in_channels]

        auto w_ = ggml_mul_mat(ctx, k, q);  // [N, h * w, h * w]
        w_      = ggml_scale_inplace(ctx, w_, 1.0f / sqrt((float)in_channels));
        w_      = ggml_soft_max_inplace(ctx, w_);

        v  = ggml_reshape_3d(ctx, v, h * w, c, n);               // [N, in_channels, h * w]
        h_ = ggml_mul_mat(ctx, v, w_);                           // [N, h * w, in_channels]
        h_ = ggml_cont(ctx, ggml_permute(ctx, h_, 1, 0, 2, 3));  // [N, in_channels, h * w]
        h_ = ggml_reshape_4d(ctx, h_, w, h, c, n);               // [N, in_channels, h, w]

        // proj_out
        h_ = ggml_nn_conv_2d(ctx, h_, proj_out_w, proj_out_b);  // [N, in_channels, h, w]

        h_ = ggml_add(ctx, h_, x);
        return h_;
    }
};

// ldm.modules.diffusionmodules.model.Encoder
struct Encoder {
    int embed_dim      = 4;
    int ch             = 128;
    int z_channels     = 4;
    int in_channels    = 3;
    int num_res_blocks = 2;
    int ch_mult[4]     = {1, 2, 4, 4};

    struct ggml_tensor* conv_in_w;  // [ch, in_channels, 3, 3]
    struct ggml_tensor* conv_in_b;  // [ch, ]

    ResnetBlock down_blocks[4][2];
    DownSample down_samples[3];

    struct
    {
        ResnetBlock block_1;
        AttnBlock attn_1;
        ResnetBlock block_2;
    } mid;

    // block_in = ch * ch_mult[len_mults - 1]
    struct ggml_tensor* norm_out_w;  // [block_in, ]
    struct ggml_tensor* norm_out_b;  // [block_in, ]

    struct ggml_tensor* conv_out_w;  // [embed_dim*2, block_in, 3, 3]
    struct ggml_tensor* conv_out_b;  // [embed_dim*2, ]

    Encoder() {
        int len_mults = sizeof(ch_mult) / sizeof(int);

        int block_in = 1;
        for (int i = 0; i < len_mults; i++) {
            if (i == 0) {
                block_in = ch;
            } else {
                block_in = ch * ch_mult[i - 1];
            }
            int block_out = ch * ch_mult[i];
            for (int j = 0; j < num_res_blocks; j++) {
                down_blocks[i][j].in_channels  = block_in;
                down_blocks[i][j].out_channels = block_out;
                block_in                       = block_out;
            }
            if (i != len_mults - 1) {
                down_samples[i].channels       = block_in;
                down_samples[i].out_channels   = block_in;
                down_samples[i].vae_downsample = true;
            }
        }

        mid.block_1.in_channels  = block_in;
        mid.block_1.out_channels = block_in;
        mid.attn_1.in_channels   = block_in;
        mid.block_2.in_channels  = block_in;
        mid.block_2.out_channels = block_in;
    }

    size_t get_num_tensors() {
        int num_tensors = 6;

        // mid
        num_tensors += 10 * 3;

        int len_mults = sizeof(ch_mult) / sizeof(int);
        for (int i = len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                num_tensors += 10;
            }

            if (i != 0) {
                num_tensors += 2;
            }
        }
        return num_tensors;
    }

    size_t calculate_mem_size(ggml_type wtype) {
        size_t mem_size = 0;
        int len_mults   = sizeof(ch_mult) / sizeof(int);
        int block_in    = ch * ch_mult[len_mults - 1];

        mem_size += ggml_row_size(GGML_TYPE_F16, ch * in_channels * 3 * 3);  // conv_in_w
        mem_size += ggml_row_size(GGML_TYPE_F32, ch);                        // conv_in_b

        mem_size += 2 * ggml_row_size(GGML_TYPE_F32, block_in);  // norm_out_w/b

        mem_size += ggml_row_size(GGML_TYPE_F16, z_channels * 2 * block_in * 3 * 3);  // conv_out_w
        mem_size += ggml_row_size(GGML_TYPE_F32, z_channels * 2);                     // conv_out_b

        mem_size += mid.block_1.calculate_mem_size(wtype);
        mem_size += mid.attn_1.calculate_mem_size(wtype);
        mem_size += mid.block_2.calculate_mem_size(wtype);

        for (int i = len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                mem_size += down_blocks[i][j].calculate_mem_size(wtype);
            }
            if (i != 0) {
                mem_size += down_samples[i - 1].calculate_mem_size(wtype);
            }
        }

        return mem_size;
    }

    void init_params(struct ggml_context* ctx, ggml_allocr* alloc, ggml_type wtype) {
        int len_mults = sizeof(ch_mult) / sizeof(int);
        int block_in  = ch * ch_mult[len_mults - 1];

        conv_in_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, in_channels, ch);
        conv_in_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ch);

        norm_out_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, block_in);
        norm_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, block_in);

        conv_out_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, block_in, z_channels * 2);
        conv_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, z_channels * 2);

        mid.block_1.init_params(ctx, wtype);
        mid.attn_1.init_params(ctx, alloc, wtype);
        mid.block_2.init_params(ctx, wtype);

        for (int i = 0; i < len_mults; i++) {
            for (int j = 0; j < num_res_blocks; j++) {
                down_blocks[i][j].init_params(ctx, wtype);
            }
            if (i != len_mults - 1) {
                down_samples[i].init_params(ctx, wtype);
            }
        }
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        tensors[prefix + "norm_out.weight"] = norm_out_w;
        tensors[prefix + "norm_out.bias"]   = norm_out_b;
        tensors[prefix + "conv_in.weight"]  = conv_in_w;
        tensors[prefix + "conv_in.bias"]    = conv_in_b;
        tensors[prefix + "conv_out.weight"] = conv_out_w;
        tensors[prefix + "conv_out.bias"]   = conv_out_b;

        mid.block_1.map_by_name(tensors, prefix + "mid.block_1.");
        mid.attn_1.map_by_name(tensors, prefix + "mid.attn_1.");
        mid.block_2.map_by_name(tensors, prefix + "mid.block_2.");

        int len_mults = sizeof(ch_mult) / sizeof(int);
        for (int i = 0; i < len_mults; i++) {
            for (int j = 0; j < num_res_blocks; j++) {
                down_blocks[i][j].map_by_name(tensors, prefix + "down." + std::to_string(i) + ".block." + std::to_string(j) + ".");
            }
            if (i != len_mults - 1) {
                down_samples[i].map_by_name(tensors, prefix + "down." + std::to_string(i) + ".downsample.");
            }
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, in_channels, h, w]

        // conv_in
        auto h = ggml_nn_conv_2d(ctx, x, conv_in_w, conv_in_b, 1, 1, 1, 1);  // [N, ch, h, w]
        ggml_set_name(h, "b-start");
        int len_mults = sizeof(ch_mult) / sizeof(int);
        for (int i = 0; i < len_mults; i++) {
            for (int j = 0; j < num_res_blocks; j++) {
                h = down_blocks[i][j].forward(ctx, h);
            }
            if (i != len_mults - 1) {
                h = down_samples[i].forward(ctx, h);
            }
        }

        h = mid.block_1.forward(ctx, h);
        h = mid.attn_1.forward(ctx, h);
        h = mid.block_2.forward(ctx, h);  // [N, block_in, h, w]

        h = ggml_nn_group_norm(ctx, h, norm_out_w, norm_out_b);
        h = ggml_silu_inplace(ctx, h);

        // conv_out
        h = ggml_nn_conv_2d(ctx, h, conv_out_w, conv_out_b, 1, 1, 1, 1);  // [N, z_channels*2, h, w]

        return h;
    }
};

// ldm.modules.diffusionmodules.model.Decoder
struct Decoder {
    int embed_dim      = 4;
    int ch             = 128;
    int z_channels     = 4;
    int out_ch         = 3;
    int num_res_blocks = 2;
    int ch_mult[4]     = {1, 2, 4, 4};

    // block_in = ch *  ch_mult[-1], 512
    struct ggml_tensor* conv_in_w;  // [block_in, z_channels, 3, 3]
    struct ggml_tensor* conv_in_b;  // [block_in, ]

    struct
    {
        ResnetBlock block_1;
        AttnBlock attn_1;
        ResnetBlock block_2;
    } mid;

    ResnetBlock up_blocks[4][3];
    UpSample up_samples[3];

    struct ggml_tensor* norm_out_w;  // [ch *  ch_mult[0], ]
    struct ggml_tensor* norm_out_b;  // [ch *  ch_mult[0], ]

    struct ggml_tensor* conv_out_w;  // [out_ch, ch *  ch_mult[0], 3, 3]
    struct ggml_tensor* conv_out_b;  // [out_ch, ]

    Decoder() {
        int len_mults = sizeof(ch_mult) / sizeof(int);
        int block_in  = ch * ch_mult[len_mults - 1];

        mid.block_1.in_channels  = block_in;
        mid.block_1.out_channels = block_in;
        mid.attn_1.in_channels   = block_in;
        mid.block_2.in_channels  = block_in;
        mid.block_2.out_channels = block_in;

        for (int i = len_mults - 1; i >= 0; i--) {
            int mult      = ch_mult[i];
            int block_out = ch * mult;
            for (int j = 0; j < num_res_blocks + 1; j++) {
                up_blocks[i][j].in_channels  = block_in;
                up_blocks[i][j].out_channels = block_out;
                block_in                     = block_out;
            }
            if (i != 0) {
                up_samples[i - 1].channels     = block_in;
                up_samples[i - 1].out_channels = block_in;
            }
        }
    }

    size_t calculate_mem_size(ggml_type wtype) {
        double mem_size = 0;
        int len_mults   = sizeof(ch_mult) / sizeof(int);
        int block_in    = ch * ch_mult[len_mults - 1];

        mem_size += ggml_row_size(GGML_TYPE_F16, block_in * z_channels * 3 * 3);  // conv_in_w
        mem_size += ggml_row_size(GGML_TYPE_F32, block_in);                       // conv_in_b

        mem_size += 2 * ggml_row_size(GGML_TYPE_F32, (ch * ch_mult[0]));  // norm_out_w/b

        mem_size += ggml_row_size(GGML_TYPE_F16, (ch * ch_mult[0]) * out_ch * 3 * 3);  // conv_out_w
        mem_size += ggml_row_size(GGML_TYPE_F32, out_ch);                              // conv_out_b

        mem_size += mid.block_1.calculate_mem_size(wtype);
        mem_size += mid.attn_1.calculate_mem_size(wtype);
        mem_size += mid.block_2.calculate_mem_size(wtype);

        for (int i = len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                mem_size += up_blocks[i][j].calculate_mem_size(wtype);
            }
            if (i != 0) {
                mem_size += up_samples[i - 1].calculate_mem_size(wtype);
            }
        }

        return static_cast<size_t>(mem_size);
    }

    size_t get_num_tensors() {
        int num_tensors = 8;

        // mid
        num_tensors += 10 * 3;

        int len_mults = sizeof(ch_mult) / sizeof(int);
        for (int i = len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                num_tensors += 10;
            }

            if (i != 0) {
                num_tensors += 2;
            }
        }
        return num_tensors;
    }

    void init_params(struct ggml_context* ctx, ggml_allocr* alloc, ggml_type wtype) {
        int len_mults = sizeof(ch_mult) / sizeof(int);
        int block_in  = ch * ch_mult[len_mults - 1];

        norm_out_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ch * ch_mult[0]);
        norm_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ch * ch_mult[0]);

        conv_in_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, z_channels, block_in);
        conv_in_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, block_in);

        conv_out_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, ch * ch_mult[0], out_ch);
        conv_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_ch);

        mid.block_1.init_params(ctx, wtype);
        mid.attn_1.init_params(ctx, alloc, wtype);
        mid.block_2.init_params(ctx, wtype);

        for (int i = len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                up_blocks[i][j].init_params(ctx, wtype);
            }

            if (i != 0) {
                up_samples[i - 1].init_params(ctx, wtype);
            }
        }
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        tensors[prefix + "norm_out.weight"] = norm_out_w;
        tensors[prefix + "norm_out.bias"]   = norm_out_b;
        tensors[prefix + "conv_in.weight"]  = conv_in_w;
        tensors[prefix + "conv_in.bias"]    = conv_in_b;
        tensors[prefix + "conv_out.weight"] = conv_out_w;
        tensors[prefix + "conv_out.bias"]   = conv_out_b;

        mid.block_1.map_by_name(tensors, prefix + "mid.block_1.");
        mid.attn_1.map_by_name(tensors, prefix + "mid.attn_1.");
        mid.block_2.map_by_name(tensors, prefix + "mid.block_2.");

        int len_mults = sizeof(ch_mult) / sizeof(int);
        for (int i = len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                up_blocks[i][j].map_by_name(tensors, prefix + "up." + std::to_string(i) + ".block." + std::to_string(j) + ".");
            }
            if (i != 0) {
                up_samples[i - 1].map_by_name(tensors, prefix + "up." + std::to_string(i) + ".upsample.");
            }
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* z) {
        // z: [N, z_channels, h, w]
        // conv_in
        auto h = ggml_nn_conv_2d(ctx, z, conv_in_w, conv_in_b, 1, 1, 1, 1);  // [N, block_in, h, w]

        h = mid.block_1.forward(ctx, h);
        h = mid.attn_1.forward(ctx, h);
        h = mid.block_2.forward(ctx, h);  // [N, block_in, h, w]

        int len_mults = sizeof(ch_mult) / sizeof(int);
        for (int i = len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                h = up_blocks[i][j].forward(ctx, h);
            }
            if (i != 0) {
                h = up_samples[i - 1].forward(ctx, h);
            }
        }

        // group norm 32
        h = ggml_nn_group_norm(ctx, h, norm_out_w, norm_out_b);
        h = ggml_silu_inplace(ctx, h);

        // conv_out
        h = ggml_nn_conv_2d(ctx, h, conv_out_w, conv_out_b, 1, 1, 1, 1);  // [N, out_ch, h, w]
        return h;
    }
};

// ldm.models.autoencoder.AutoencoderKL
struct AutoEncoderKL : public GGMLModule {
    bool decode_only = true;
    int embed_dim    = 4;
    struct {
        int z_channels     = 4;
        int resolution     = 256;
        int in_channels    = 3;
        int out_ch         = 3;
        int ch             = 128;
        int ch_mult[4]     = {1, 2, 4, 4};
        int num_res_blocks = 2;
    } dd_config;

    struct ggml_tensor* quant_conv_w;  // [2*embed_dim, 2*z_channels, 1, 1]
    struct ggml_tensor* quant_conv_b;  // [2*embed_dim, ]

    struct ggml_tensor* post_quant_conv_w;  // [z_channels, embed_dim, 1, 1]
    struct ggml_tensor* post_quant_conv_b;  // [z_channels, ]

    Encoder encoder;
    Decoder decoder;

    AutoEncoderKL(bool decode_only = false)
        : decode_only(decode_only) {
        name = "vae";
        assert(sizeof(dd_config.ch_mult) == sizeof(encoder.ch_mult));
        assert(sizeof(dd_config.ch_mult) == sizeof(decoder.ch_mult));

        encoder.embed_dim      = embed_dim;
        decoder.embed_dim      = embed_dim;
        encoder.ch             = dd_config.ch;
        decoder.ch             = dd_config.ch;
        encoder.z_channels     = dd_config.z_channels;
        decoder.z_channels     = dd_config.z_channels;
        encoder.in_channels    = dd_config.in_channels;
        decoder.out_ch         = dd_config.out_ch;
        encoder.num_res_blocks = dd_config.num_res_blocks;

        int len_mults = sizeof(dd_config.ch_mult) / sizeof(int);
        for (int i = 0; i < len_mults; i++) {
            encoder.ch_mult[i] = dd_config.ch_mult[i];
            decoder.ch_mult[i] = dd_config.ch_mult[i];
        }
    }

    size_t calculate_mem_size() {
        size_t mem_size = 0;

        if (!decode_only) {
            mem_size += ggml_row_size(GGML_TYPE_F16, 2 * embed_dim * 2 * dd_config.z_channels * 1 * 1);  // quant_conv_w
            mem_size += ggml_row_size(GGML_TYPE_F32, 2 * embed_dim);                                     // quant_conv_b
            mem_size += encoder.calculate_mem_size(wtype);
        }

        mem_size += ggml_row_size(GGML_TYPE_F16, dd_config.z_channels * embed_dim * 1 * 1);  // post_quant_conv_w
        mem_size += ggml_row_size(GGML_TYPE_F32, dd_config.z_channels);                      // post_quant_conv_b

        mem_size += decoder.calculate_mem_size(wtype);
        return mem_size;
    }

    size_t get_num_tensors() {
        size_t num_tensors = decoder.get_num_tensors();
        if (!decode_only) {
            num_tensors += 2;
            num_tensors += encoder.get_num_tensors();
        }
        return num_tensors;
    }

    void init_params() {
        ggml_allocr* alloc = ggml_allocr_new_from_buffer(params_buffer);

        if (!decode_only) {
            quant_conv_w = ggml_new_tensor_4d(params_ctx, GGML_TYPE_F16, 1, 1, 2 * dd_config.z_channels, 2 * embed_dim);
            quant_conv_b = ggml_new_tensor_1d(params_ctx, GGML_TYPE_F32, 2 * embed_dim);
            encoder.init_params(params_ctx, alloc, wtype);
        }

        post_quant_conv_w = ggml_new_tensor_4d(params_ctx, GGML_TYPE_F16, 1, 1, embed_dim, dd_config.z_channels);
        post_quant_conv_b = ggml_new_tensor_1d(params_ctx, GGML_TYPE_F32, dd_config.z_channels);
        decoder.init_params(params_ctx, alloc, wtype);

        // alloc all tensors linked to this context
        for (struct ggml_tensor* t = ggml_get_first_tensor(params_ctx); t != NULL; t = ggml_get_next_tensor(params_ctx, t)) {
            if (t->data == NULL) {
                ggml_allocr_alloc(alloc, t);
            }
        }
        ggml_allocr_free(alloc);
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        tensors[prefix + "quant_conv.weight"] = quant_conv_w;
        tensors[prefix + "quant_conv.bias"]   = quant_conv_b;
        encoder.map_by_name(tensors, prefix + "encoder.");

        tensors[prefix + "post_quant_conv.weight"] = post_quant_conv_w;
        tensors[prefix + "post_quant_conv.bias"]   = post_quant_conv_b;
        decoder.map_by_name(tensors, prefix + "decoder.");
    }

    struct ggml_tensor* decode(struct ggml_context* ctx0, struct ggml_tensor* z) {
        // z: [N, z_channels, h, w]
        // post_quant_conv
        auto h = ggml_nn_conv_2d(ctx0, z, post_quant_conv_w, post_quant_conv_b);  // [N, z_channels, h, w]
        ggml_set_name(h, "bench-start");
        h = decoder.forward(ctx0, h);
        ggml_set_name(h, "bench-end");
        return h;
    }

    struct ggml_tensor* encode(struct ggml_context* ctx0, struct ggml_tensor* x) {
        // x: [N, in_channels, h, w]
        auto h = encoder.forward(ctx0, x);  // [N, 2*z_channels, h/8, w/8]
        // quant_conv
        h = ggml_nn_conv_2d(ctx0, h, quant_conv_w, quant_conv_b);  // [N, 2*embed_dim, h/8, w/8]
        ggml_set_name(h, "b-end");
        return h;
    }

    struct ggml_cgraph* build_graph(struct ggml_tensor* z, bool decode_graph) {
        // since we are using ggml-alloc, this buffer only needs enough space to hold the ggml_tensor and ggml_cgraph structs, but not the tensor data
        static size_t buf_size = ggml_tensor_overhead() * VAE_GRAPH_SIZE + ggml_graph_overhead();
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

        struct ggml_tensor* out = decode_graph ? decode(ctx0, z_) : encode(ctx0, z_);

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

    void compute(struct ggml_tensor* work_result, const int n_threads, struct ggml_tensor* z, bool decode_graph) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(z, decode_graph);
        };

        GGMLModule::compute(get_graph, n_threads, work_result);
    }
};

#endif