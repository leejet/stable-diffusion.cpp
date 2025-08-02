#ifndef __WAN_HPP__
#define __WAN_HPP__

#include <map>

#include "common.hpp"
#include "ggml_extend.hpp"

namespace WAN {

    constexpr int CACHE_T = 2;

    class CausalConv3d : public GGMLBlock {
    protected:
        int64_t in_channels;
        int64_t out_channels;
        std::tuple<int, int, int> kernel_size;
        std::tuple<int, int, int> stride;
        std::tuple<int, int, int> padding;
        std::tuple<int, int, int> dilation;
        bool bias;

        void init_params(struct ggml_context* ctx, const String2GGMLType& tensor_types = {}, const std::string prefix = "") {
            params["weight"]     = ggml_new_tensor_4d(ctx,
                                                      GGML_TYPE_F16,
                                                      std::get<2>(kernel_size),
                                                      std::get<1>(kernel_size),
                                                      std::get<0>(kernel_size),
                                                      in_channels * out_channels);
            if (bias) {
                params["bias"]       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
            }
        }

    public:
        CausalConv3d(int64_t in_channels,
                     int64_t out_channels,
                     std::tuple<int, int, int> kernel_size,
                     std::tuple<int, int, int> stride   = {1, 1, 1},
                     std::tuple<int, int, int> padding  = {0, 0, 0},
                     std::tuple<int, int, int> dilation = {1, 1, 1},
                     bool bias                          = true)
            : in_channels(in_channels),
              out_channels(out_channels),
              kernel_size(kernel_size),
              stride(stride),
              padding(padding),
              dilation(dilation),
              bias(bias) {}

        struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* cache_x = NULL) {
            // x: [N*IC, ID, IH, IW]
            // result: x: [N*OC, ID, IH, IW]
            struct ggml_tensor* w = params["weight"];
            struct ggml_tensor* b = NULL;
            if (bias) {
                b = params["bias"];
            }

            int lp0 = std::get<2>(padding);
            int rp0 = std::get<2>(padding);
            int lp1 = std::get<1>(padding);
            int rp1 = std::get<1>(padding);
            int lp2 = 2 * std::get<0>(padding);
            int rp2 = 0;

            if (cache_x != NULL && std::get<0>(padding) > 0) {
                x = ggml_concat(ctx, cache_x, x, 2);
                lp2 -= (int)cache_x->ne[2];
            }

            x = ggml_pad_ext(ctx, x, lp0, rp0, lp1, rp1, lp2, rp2, 0, 0);
            return ggml_nn_conv_3d(ctx, x, w, b, in_channels,
                                   std::get<2>(stride), std::get<1>(stride), std::get<0>(stride),
                                   0, 0, 0,
                                   std::get<2>(dilation), std::get<1>(dilation), std::get<0>(dilation));
        }
    };

    class RMS_norm : public UnaryBlock {
    protected:
        int64_t dim;

        void init_params(struct ggml_context* ctx, const String2GGMLType& tensor_types = {}, const std::string prefix = "") {
            ggml_type wtype = GGML_TYPE_F32;
            params["gamma"] = ggml_new_tensor_1d(ctx, wtype, dim);
        }

    public:
        RMS_norm(int64_t dim)
            : dim(dim) {}

        struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
            // x: [N*IC, ID, IH, IW], IC == dim
            // assert N == 1

            struct ggml_tensor* w = params["gamma"];
            auto h = ggml_cont(ctx, ggml_torch_permute(ctx, x, 3, 0, 1, 2)); // [ID, IH, IW, N*IC]
            h = ggml_rms_norm(ctx, h, 1e-12);
            h = ggml_mul(ctx, h, w);
            h = ggml_cont(ctx, ggml_torch_permute(ctx, h, 1, 2, 3, 0));

            return h;
        }
    };

    class Resample : public GGMLBlock {
    protected:
        int64_t dim;
        std::string mode;

    public:
        Resample(int64_t dim, const std::string& mode)
            : dim(dim), mode(mode) {
            if (mode == "upsample2d") {
                blocks["resample.1"] = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim / 2, {3, 3}, {1, 1}, {1, 1}));
            } else if (mode == "upsample3d") {
                blocks["resample.1"] = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim / 2, {3, 3}, {1, 1}, {1, 1}));
                blocks["time_conv"]  = std::shared_ptr<GGMLBlock>(new CausalConv3d(dim, dim * 2, {3, 1, 1}, {1, 1, 1}, {1, 0, 0}));
            } else if (mode == "downsample2d") {
                blocks["resample.1"] = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim, {3, 3}, {2, 2}));
            } else if (mode == "downsample3d") {
                blocks["resample.1"] = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim, {3, 3}, {2, 2}));
                blocks["time_conv"]  = std::shared_ptr<GGMLBlock>(new CausalConv3d(dim, dim, {3, 1, 1}, {2, 1, 1}, {0, 0, 0}));
            } else if (mode == "none") {
                // nn.Identity()
            } else {
                GGML_ASSERT(false && "invalid mode");
            }
        }

        struct ggml_tensor* forward(struct ggml_context* ctx,
                                    struct ggml_tensor* x,
                                    int64_t b,
                                    std::vector<struct ggml_tensor*>& feat_cache,
                                    int& feat_idx) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            int64_t c = x->ne[3] / b;
            int64_t t = x->ne[2];
            int64_t h = x->ne[1];
            int64_t w = x->ne[0];

            struct ggml_tensor* Rep = (struct ggml_tensor*)1;

            if (mode == "upsample3d") {
                if (feat_cache.size() > 0) {
                    int idx = feat_idx;
                    if (feat_cache[idx] == NULL) {
                        feat_cache[idx] = Rep;  // Rep
                        feat_idx += 1;
                    } else {
                        auto time_conv = std::dynamic_pointer_cast<CausalConv3d>(blocks["time_conv"]);

                        auto cache_x = ggml_slice(ctx, x, 2, -CACHE_T, x->ne[2]);
                        if (cache_x->ne[2] < 2 && feat_cache[idx] != NULL && feat_cache[idx] != Rep) {
                            // cache last frame of last two chunk
                            cache_x = ggml_concat(ctx,
                                                  ggml_slice(ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                                  cache_x,
                                                  2);
                        }
                        if (cache_x->ne[1] < 2 && feat_cache[idx] != NULL && feat_cache[idx] == Rep) {
                            cache_x = ggml_pad_ext(ctx, cache_x, 0, 0, 1, 1, (int)cache_x->ne[2], 0, 0, 0);
                            // aka cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device),cache_x],dim=2)
                        }
                        if (feat_cache[idx] == Rep) {
                            x = time_conv->forward(ctx, x);
                        } else {
                            x = time_conv->forward(ctx, x, feat_cache[idx]);
                        }
                        feat_cache[idx] = cache_x;
                        feat_idx += 1;
                        x = ggml_reshape_4d(ctx, x, w * h, t, c, 2);                 // (2, c, t, h*w)
                        x = ggml_cont(ctx, ggml_torch_permute(ctx, x, 0, 3, 1, 2));  // (c, t, 2, h*w)
                        x = ggml_reshape_4d(ctx, x, w, h, 2 * t, c);                 // (c, t*2, h, w)
                    }
                }
            }

            t = x->ne[2];
            if (mode != "none") {
                auto resample_1 = std::dynamic_pointer_cast<Conv2d>(blocks["resample.1"]);

                x = ggml_cont(ctx, ggml_torch_permute(ctx, x, 0, 1, 3, 2));  // (t, c, h, w)
                if (mode == "upsample2d") {
                    x = ggml_upscale(ctx, x, 2, GGML_SCALE_MODE_NEAREST);
                } else if (mode == "upsample3d") {
                    x = ggml_upscale(ctx, x, 2, GGML_SCALE_MODE_NEAREST);
                } else if (mode == "downsample2d") {
                    x = ggml_pad(ctx, x, 1, 1, 0, 0);
                } else if (mode == "downsample3d") {
                    x = ggml_pad(ctx, x, 1, 1, 0, 0);
                }
                x = resample_1->forward(ctx, x);
                x = ggml_cont(ctx, ggml_torch_permute(ctx, x, 0, 1, 3, 2));  // (c, t, h, w)
            }

            if (mode == "downsample3d") {
                if (feat_cache.size() > 0) {
                    int idx = feat_idx;
                    if (feat_cache[idx] == NULL) {
                        feat_cache[idx] = x;
                        feat_idx += 1;
                    } else {
                        auto time_conv = std::dynamic_pointer_cast<CausalConv3d>(blocks["time_conv"]);

                        auto cache_x    = ggml_slice(ctx, x, 2, -1, x->ne[2]);
                        x               = ggml_concat(ctx,
                                                      ggml_slice(ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                                      x,
                                                      2);
                        x               = time_conv->forward(ctx, x);
                        feat_cache[idx] = cache_x;
                        feat_idx += 1;
                    }
                }
            }

            return x;
        }
    };

    class ResidualBlock : public GGMLBlock {
    protected:
        int64_t in_dim;
        int64_t out_dim;

    public:
        ResidualBlock(int64_t in_dim, int64_t out_dim)
            : in_dim(in_dim), out_dim(out_dim) {
            blocks["residual.0"] = std::shared_ptr<GGMLBlock>(new RMS_norm(in_dim));
            // residual.1 is nn.SiLU()
            blocks["residual.2"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(in_dim, out_dim, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
            blocks["residual.3"] = std::shared_ptr<GGMLBlock>(new RMS_norm(out_dim));
            // residual.4 is nn.SiLU()
            // residual.5 is nn.Dropout()
            blocks["residual.6"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(out_dim, out_dim, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
            if (in_dim != out_dim) {
                blocks["shortcut"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(in_dim, out_dim, {1, 1, 1}));
            }
        }

        struct ggml_tensor* forward(struct ggml_context* ctx,
                                    struct ggml_tensor* x,
                                    int64_t b,
                                    std::vector<struct ggml_tensor*>& feat_cache,
                                    int& feat_idx) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            struct ggml_tensor* h = x;
            if (in_dim != out_dim) {
                auto shortcut = std::dynamic_pointer_cast<CausalConv3d>(blocks["shortcut"]);

                h = shortcut->forward(ctx, x);
            }

            for (int i = 0; i < 7; i++) {
                if (i == 0 || i == 3) {  // RMS_norm
                    auto layer = std::dynamic_pointer_cast<RMS_norm>(blocks["residual." + std::to_string(i)]);
                    x = layer->forward(ctx, x);
                } else if (i == 2 || i == 6) {  // CausalConv3d
                    auto layer = std::dynamic_pointer_cast<CausalConv3d>(blocks["residual." + std::to_string(i)]);

                    if (feat_cache.size() > 0) {
                        int idx      = feat_idx;
                        auto cache_x = ggml_slice(ctx, x, 2, -CACHE_T, x->ne[2]);
                        if (cache_x->ne[2] < 2 && feat_cache[idx] != NULL) {
                            // cache last frame of last two chunk
                            cache_x = ggml_concat(ctx,
                                                  ggml_slice(ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                                  cache_x,
                                                  2);
                        }

                        x               = layer->forward(ctx, x, feat_cache[idx]);
                        feat_cache[idx] = cache_x;
                        feat_idx += 1;
                    }
                } else if (i == 1 || i == 4) {
                    x = ggml_silu(ctx, x);
                } else {  // i == 5
                    // nn.Dropout(), ignore
                }
            }

            x = ggml_add(ctx, x, h);
            return x;
        }
    };

    class AttentionBlock : public GGMLBlock {
    protected:
        int64_t dim;

    public:
        AttentionBlock(int64_t dim)
            : dim(dim) {
            blocks["norm"]   = std::shared_ptr<GGMLBlock>(new RMS_norm(dim));
            blocks["to_qkv"] = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim * 3, {1, 1}));
            blocks["proj"]   = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim, {1, 1}));
        }

        struct ggml_tensor* forward(struct ggml_context* ctx,
                                    struct ggml_tensor* x,
                                    int64_t b) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            auto norm   = std::dynamic_pointer_cast<RMS_norm>(blocks["norm"]);
            auto to_qkv = std::dynamic_pointer_cast<Conv2d>(blocks["to_qkv"]);
            auto proj   = std::dynamic_pointer_cast<Conv2d>(blocks["proj"]);

            auto identity = x;

            x            = norm->forward(ctx, x);

            x = ggml_cont(ctx, ggml_torch_permute(ctx, x, 0, 1, 3, 2));  // (t, c, h, w)

            const int64_t n = x->ne[3];
            const int64_t c = x->ne[2];
            const int64_t h = x->ne[1];
            const int64_t w = x->ne[0];

            auto qkv     = to_qkv->forward(ctx, x);
            auto qkv_vec = split_image_qkv(ctx, qkv);

            auto q = qkv_vec[0];
            q      = ggml_cont(ctx, ggml_torch_permute(ctx, q, 2, 0, 1, 3));  // [t, h, w, c]
            q      = ggml_reshape_3d(ctx, q, c, h * w, n);                    // [t, h * w, c]

            auto k = qkv_vec[1];
            k      = ggml_cont(ctx, ggml_torch_permute(ctx, k, 2, 0, 1, 3));  // [t, h, w, c]
            k      = ggml_reshape_3d(ctx, k, c, h * w, n);                    // [t, h * w, c]

            auto v = qkv_vec[2];
            v      = ggml_reshape_3d(ctx, v, h * w, c, n);  // [t, c, h * w]

            x = ggml_nn_attention(ctx, q, k, v, false);  // [t, h * w, c]

            x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));  // [t, c, h * w]
            x = ggml_reshape_4d(ctx, x, w, h, c, n);               // [t, c, h, w]

            x = proj->forward(ctx, x);

            x = ggml_cont(ctx, ggml_torch_permute(ctx, x, 0, 1, 3, 2));  // (c, t, h, w)

            x = ggml_add(ctx, x, identity);
            return x;
        }
    };

    class Encoder3d : public GGMLBlock {
    protected:
        int64_t dim;
        int64_t z_dim;
        std::vector<int> dim_mult;
        int num_res_blocks;
        std::vector<bool> temperal_downsample;

    public:
        Encoder3d(int64_t dim                           = 128,
                  int64_t z_dim                         = 4,
                  std::vector<int> dim_mult             = {1, 2, 4, 4},
                  int num_res_blocks                    = 2,
                  std::vector<bool> temperal_downsample = {false, true, true})
            : dim(dim), z_dim(z_dim), dim_mult(dim_mult), num_res_blocks(num_res_blocks), temperal_downsample(temperal_downsample) {
            // attn_scales is always []
            std::vector<int64_t> dims = {dim};
            for (int u : dim_mult) {
                dims.push_back(dim * u);
            }

            blocks["conv1"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(3, dims[0], {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));

            int index = 0;
            int64_t in_dim;
            int64_t out_dim;
            for (int i = 0; i < dims.size() - 1; i++) {
                in_dim  = dims[i];
                out_dim = dims[i + 1];
                for (int j = 0; j < num_res_blocks; j++) {
                    auto block                                       = std::shared_ptr<GGMLBlock>(new ResidualBlock(in_dim, out_dim));
                    blocks["downsamples." + std::to_string(index++)] = block;
                    in_dim                                           = out_dim;
                }

                if (i != dim_mult.size() - 1) {
                    std::string mode                                 = temperal_downsample[i] ? "downsample3d" : "downsample2d";
                    auto block                                       = std::shared_ptr<GGMLBlock>(new Resample(out_dim, mode));
                    blocks["downsamples." + std::to_string(index++)] = block;
                }
            }

            blocks["middle.0"] = std::shared_ptr<GGMLBlock>(new ResidualBlock(out_dim, out_dim));
            blocks["middle.1"] = std::shared_ptr<GGMLBlock>(new AttentionBlock(out_dim));
            blocks["middle.2"] = std::shared_ptr<GGMLBlock>(new ResidualBlock(out_dim, out_dim));

            blocks["head.0"] = std::shared_ptr<GGMLBlock>(new RMS_norm(out_dim));
            // head.1 is nn.SiLU()
            blocks["head.2"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(out_dim, z_dim, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
        }

        struct ggml_tensor* forward(struct ggml_context* ctx,
                                    struct ggml_tensor* x,
                                    int64_t b,
                                    std::vector<struct ggml_tensor*>& feat_cache,
                                    int& feat_idx) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            auto conv1    = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv1"]);
            auto middle_0 = std::dynamic_pointer_cast<ResidualBlock>(blocks["middle.0"]);
            auto middle_1 = std::dynamic_pointer_cast<AttentionBlock>(blocks["middle.1"]);
            auto middle_2 = std::dynamic_pointer_cast<ResidualBlock>(blocks["middle.2"]);
            auto head_0   = std::dynamic_pointer_cast<RMS_norm>(blocks["head.0"]);
            auto head_2   = std::dynamic_pointer_cast<CausalConv3d>(blocks["head.2"]);

            // conv1
            if (feat_cache.size() > 0) {
                int idx      = feat_idx;
                auto cache_x = ggml_slice(ctx, x, 2, -CACHE_T, x->ne[2]);
                if (cache_x->ne[2] < 2 && feat_cache[idx] != NULL) {
                    // cache last frame of last two chunk
                    cache_x = ggml_concat(ctx,
                                          ggml_slice(ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                          cache_x,
                                          2);
                }

                x               = conv1->forward(ctx, x, feat_cache[idx]);
                feat_cache[idx] = cache_x;
                feat_idx += 1;
            } else {
                x = conv1->forward(ctx, x);
            }

            // downsamples
            std::vector<int64_t> dims = {dim};
            for (int u : dim_mult) {
                dims.push_back(dim * u);
            }
            int index = 0;
            for (int i = 0; i < dims.size() - 1; i++) {
                for (int j = 0; j < num_res_blocks; j++) {
                    auto layer = std::dynamic_pointer_cast<ResidualBlock>(blocks["downsamples." + std::to_string(index++)]);

                    x = layer->forward(ctx, x, b, feat_cache, feat_idx);
                }

                if (i != dim_mult.size() - 1) {
                    auto layer = std::dynamic_pointer_cast<Resample>(blocks["downsamples." + std::to_string(index++)]);

                    x = layer->forward(ctx, x, b, feat_cache, feat_idx);
                }
            }

            // middle
            x = middle_0->forward(ctx, x, b, feat_cache, feat_idx);
            x = middle_1->forward(ctx, x, b);
            x = middle_2->forward(ctx, x, b, feat_cache, feat_idx);

            // head
            x = head_0->forward(ctx, x);
            x = ggml_silu(ctx, x);
            if (feat_cache.size() > 0) {
                int idx      = feat_idx;
                auto cache_x = ggml_slice(ctx, x, 2, -CACHE_T, x->ne[2]);
                if (cache_x->ne[2] < 2 && feat_cache[idx] != NULL) {
                    // cache last frame of last two chunk
                    cache_x = ggml_concat(ctx,
                                          ggml_slice(ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                          cache_x,
                                          2);
                }

                x               = head_2->forward(ctx, x, feat_cache[idx]);
                feat_cache[idx] = cache_x;
                feat_idx += 1;
            } else {
                x = head_2->forward(ctx, x);
            }

            return x;
        }
    };

    class Decoder3d : public GGMLBlock {
    protected:
        int64_t dim;
        int64_t z_dim;
        std::vector<int> dim_mult;
        int num_res_blocks;
        std::vector<bool> temperal_upsample;

    public:
        Decoder3d(int64_t dim                         = 128,
                  int64_t z_dim                       = 4,
                  std::vector<int> dim_mult           = {1, 2, 4, 4},
                  int num_res_blocks                  = 2,
                  std::vector<bool> temperal_upsample = {true, true, false})
            : dim(dim), z_dim(z_dim), dim_mult(dim_mult), num_res_blocks(num_res_blocks), temperal_upsample(temperal_upsample) {
            // attn_scales is always []
            std::vector<int64_t> dims = {dim_mult[dim_mult.size() - 1] * dim};
            for (int i = static_cast<int>(dim_mult.size()) - 1; i >= 0; i--) {
                dims.push_back(dim * dim_mult[i]);
            }

            // init block
            blocks["conv1"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(z_dim, dims[0], {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));

            // middle blocks
            blocks["middle.0"] = std::shared_ptr<GGMLBlock>(new ResidualBlock(dims[0], dims[0]));
            blocks["middle.1"] = std::shared_ptr<GGMLBlock>(new AttentionBlock(dims[0]));
            blocks["middle.2"] = std::shared_ptr<GGMLBlock>(new ResidualBlock(dims[0], dims[0]));

            // upsample blocks
            int index = 0;
            int64_t in_dim;
            int64_t out_dim;
            for (int i = 0; i < dims.size() - 1; i++) {
                in_dim  = dims[i];
                out_dim = dims[i + 1];
                LOG_DEBUG("in_dim %u out_dim %u", in_dim, out_dim);
                if (i == 1 || i == 2 || i == 3) {
                    in_dim = in_dim / 2;
                }
                for (int j = 0; j < num_res_blocks + 1; j++) {
                    auto block                                     = std::shared_ptr<GGMLBlock>(new ResidualBlock(in_dim, out_dim));
                    blocks["upsamples." + std::to_string(index++)] = block;
                    in_dim                                         = out_dim;
                }

                if (i != dim_mult.size() - 1) {
                    std::string mode                               = temperal_upsample[i] ? "upsample3d" : "upsample2d";
                    auto block                                     = std::shared_ptr<GGMLBlock>(new Resample(out_dim, mode));
                    blocks["upsamples." + std::to_string(index++)] = block;
                }
            }

            // output blocks
            blocks["head.0"] = std::shared_ptr<GGMLBlock>(new RMS_norm(out_dim));
            // head.1 is nn.SiLU()
            blocks["head.2"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(out_dim, 3, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
        }

        struct ggml_tensor* forward(struct ggml_context* ctx,
                                    struct ggml_tensor* x,
                                    int64_t b,
                                    std::vector<struct ggml_tensor*>& feat_cache,
                                    int& feat_idx) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            auto conv1    = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv1"]);
            auto middle_0 = std::dynamic_pointer_cast<ResidualBlock>(blocks["middle.0"]);
            auto middle_1 = std::dynamic_pointer_cast<AttentionBlock>(blocks["middle.1"]);
            auto middle_2 = std::dynamic_pointer_cast<ResidualBlock>(blocks["middle.2"]);
            auto head_0   = std::dynamic_pointer_cast<RMS_norm>(blocks["head.0"]);
            auto head_2   = std::dynamic_pointer_cast<CausalConv3d>(blocks["head.2"]);

            // conv1
            if (feat_cache.size() > 0) {
                int idx      = feat_idx;
                auto cache_x = ggml_slice(ctx, x, 2, -CACHE_T, x->ne[2]);
                if (cache_x->ne[2] < 2 && feat_cache[idx] != NULL) {
                    // cache last frame of last two chunk
                    cache_x = ggml_concat(ctx,
                                          ggml_slice(ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                          cache_x,
                                          2);
                }

                x               = conv1->forward(ctx, x, feat_cache[idx]);
                feat_cache[idx] = cache_x;
                feat_idx += 1;
            } else {
                x = conv1->forward(ctx, x);
            }

            // middle
            x = middle_0->forward(ctx, x, b, feat_cache, feat_idx);
            x = middle_1->forward(ctx, x, b);
            x = middle_2->forward(ctx, x, b, feat_cache, feat_idx);

            // upsamples
            std::vector<int64_t> dims = {dim_mult[dim_mult.size() - 1] * dim};
            for (int i = static_cast<int>(dim_mult.size()) - 1; i >= 0; i--) {
                dims.push_back(dim * dim_mult[i]);
            }
            int index = 0;
            for (int i = 0; i < dims.size() - 1; i++) {
                for (int j = 0; j < num_res_blocks + 1; j++) {
                    auto layer = std::dynamic_pointer_cast<ResidualBlock>(blocks["upsamples." + std::to_string(index++)]);

                    x = layer->forward(ctx, x, b, feat_cache, feat_idx);
                }

                if (i != dim_mult.size() - 1) {
                    auto layer = std::dynamic_pointer_cast<Resample>(blocks["upsamples." + std::to_string(index++)]);

                    x = layer->forward(ctx, x, b, feat_cache, feat_idx);
                }
            }

            // head
            x = head_0->forward(ctx, x);
            x = ggml_silu(ctx, x);
            if (feat_cache.size() > 0) {
                int idx      = feat_idx;
                auto cache_x = ggml_slice(ctx, x, 2, -CACHE_T, x->ne[2]);
                if (cache_x->ne[2] < 2 && feat_cache[idx] != NULL) {
                    // cache last frame of last two chunk
                    cache_x = ggml_concat(ctx,
                                          ggml_slice(ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                          cache_x,
                                          2);
                }

                x               = head_2->forward(ctx, x, feat_cache[idx]);
                feat_cache[idx] = cache_x;
                feat_idx += 1;
            } else {
                x = head_2->forward(ctx, x);
            }

            return x;
        }
    };

    class WanVAE : public GGMLBlock {
    protected:
        bool decode_only                      = true;
        int64_t dim                           = 96;
        int64_t z_dim                         = 16;
        std::vector<int> dim_mult             = {1, 2, 4, 4};
        int num_res_blocks                    = 2;
        std::vector<bool> temperal_upsample   = {true, true, false};
        std::vector<bool> temperal_downsample = {false, true, true};

        int _conv_num = 33;
        int _conv_idx = 0;
        std::vector<struct ggml_tensor*> _feat_map;
        int _enc_conv_num = 28;
        int _enc_conv_idx = 0;
        std::vector<struct ggml_tensor*> _enc_feat_map;

        void clear_cache() {
            _conv_idx     = 0;
            _feat_map     = std::vector<struct ggml_tensor*>(_conv_num, NULL);
            _enc_conv_idx = 0;
            _enc_feat_map = std::vector<struct ggml_tensor*>(_enc_conv_num, NULL);
        }

    public:
        WanVAE(bool decode_only = true)
            : decode_only(decode_only) {
            // attn_scales is always []
            if (!decode_only) {
                blocks["encoder"] = std::shared_ptr<GGMLBlock>(new Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks, temperal_downsample));
                blocks["conv1"]   = std::shared_ptr<GGMLBlock>(new CausalConv3d(z_dim * 2, z_dim * 2, {1, 1, 1}));
            }
            blocks["decoder"] = std::shared_ptr<GGMLBlock>(new Decoder3d(dim, z_dim, dim_mult, num_res_blocks, temperal_upsample));
            blocks["conv2"]   = std::shared_ptr<GGMLBlock>(new CausalConv3d(z_dim, z_dim, {1, 1, 1}));
        }

        struct ggml_tensor* encode(struct ggml_context* ctx,
                                   struct ggml_tensor* x,
                                   int64_t b = 1) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            GGML_ASSERT(decode_only == false);

            clear_cache();

            auto encoder = std::dynamic_pointer_cast<Encoder3d>(blocks["encoder"]);
            auto conv1   = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv1"]);

            int64_t t     = x->ne[2];
            int64_t iter_ = 1 + (t - 1) / 4;
            struct ggml_tensor* out;
            for (int i = 0; i < iter_; i++) {
                _enc_conv_idx = 0;
                if (i == 0) {
                    auto in = ggml_slice(ctx, x, 2, 0, 1);  // [b*c, 1, h, w]
                    out     = encoder->forward(ctx, in, b, _enc_feat_map, _enc_conv_idx);
                } else {
                    auto in   = ggml_slice(ctx, x, 2, 1 + 4 * (i - 1), 1 + 4 * i);  // [b*c, 4, h, w]
                    auto out_ = encoder->forward(ctx, in, b, _enc_feat_map, _enc_conv_idx);
                    out       = ggml_concat(ctx, out, out_, 2);
                }
            }
            out     = conv1->forward(ctx, out);
            auto mu = ggml_chunk(ctx, out, 2, 3)[0];
            clear_cache();
            return mu;
        }

        struct ggml_tensor* decode(struct ggml_context* ctx,
                                   struct ggml_tensor* z,
                                   int64_t b = 1) {
            // z: [b*c, t, h, w]
            GGML_ASSERT(b == 1);

            clear_cache();

            auto decoder = std::dynamic_pointer_cast<Decoder3d>(blocks["decoder"]);
            auto conv2   = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv2"]);

            int64_t iter_ = z->ne[2];
            auto x        = conv2->forward(ctx, z);
            struct ggml_tensor* out;
            for (int64_t i = 0; i < iter_; i++) {
                _conv_idx = 0;
                if (i == 0) {
                    auto in = ggml_slice(ctx, x, 2, i, i + 1);  // [b*c, 1, h, w]
                    out     = decoder->forward(ctx, in, b, _feat_map, _conv_idx);
                } else {
                    auto in   = ggml_slice(ctx, x, 2, i, i + 1);  // [b*c, 1, h, w]
                    auto out_ = decoder->forward(ctx, in, b, _feat_map, _conv_idx);
                    out       = ggml_concat(ctx, out, out_, 2);
                }
            }
            clear_cache();
            return out;
        }
    };

    struct WanVAERunner : public GGMLRunner {
        bool decode_only = true;
        WanVAE ae;

        WanVAERunner(ggml_backend_t backend,
                     const String2GGMLType& tensor_types = {},
                     const std::string prefix            = "",
                     bool decode_only                    = false)
            : decode_only(decode_only), ae(decode_only), GGMLRunner(backend) {
            ae.init(params_ctx, tensor_types, prefix);
        }

        std::string get_desc() {
            return "wan_vae";
        }

        void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
            ae.get_param_tensors(tensors, prefix);
        }

        struct ggml_cgraph* build_graph(struct ggml_tensor* z, bool decode_graph) {
            struct ggml_cgraph* gf = ggml_new_graph_custom(compute_ctx, 20480, false);

            z = to_backend(z);

            struct ggml_tensor* out = decode_graph ? ae.decode(compute_ctx, z) : ae.encode(compute_ctx, z);

            ggml_build_forward_expand(gf, out);

            return gf;
        }

        void compute(const int n_threads,
                     struct ggml_tensor* z,
                     bool decode_graph,
                     struct ggml_tensor** output,
                     struct ggml_context* output_ctx = NULL) {
            auto get_graph = [&]() -> struct ggml_cgraph* {
                return build_graph(z, decode_graph);
            };
            // ggml_set_f32(z, 0.5f);
            // print_ggml_tensor(z);
            GGMLRunner::compute(get_graph, n_threads, true, output, output_ctx);
        }

        void test() {
            struct ggml_init_params params;
            params.mem_size   = static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
            params.mem_buffer = NULL;
            params.no_alloc   = false;

            struct ggml_context* work_ctx = ggml_init(params);
            GGML_ASSERT(work_ctx != NULL);

            if (true) {
                // cpu f32, pass
                // cpu f16, pass
                // cuda f16, pass
                // cuda f32, pass
                auto z = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 8, 8, 1, 16);
                z = load_tensor_from_file(work_ctx, "wan_vae_z.bin");
                // ggml_set_f32(z, 0.5f);
                print_ggml_tensor(z);
                struct ggml_tensor* out = NULL;

                int64_t t0 = ggml_time_ms();
                compute(8, z, true, &out, work_ctx);
                int64_t t1 = ggml_time_ms();

                print_ggml_tensor(out);
                LOG_DEBUG("decode test done in %ldms", t1 - t0);
            }
        };

        static void load_from_file_and_test(const std::string& file_path) {
            ggml_backend_t backend    = ggml_backend_cuda_init(0);
            // ggml_backend_t backend            = ggml_backend_cpu_init();
            ggml_type model_data_type         = GGML_TYPE_F32;
            std::shared_ptr<WanVAERunner> vae = std::shared_ptr<WanVAERunner>(new WanVAERunner(backend));
            {
                LOG_INFO("loading from '%s'", file_path.c_str());

                vae->alloc_params_buffer();
                std::map<std::string, ggml_tensor*> tensors;
                vae->get_param_tensors(tensors, "first_stage_model");

                ModelLoader model_loader;
                if (!model_loader.init_from_file(file_path, "vae.")) {
                    LOG_ERROR("init model loader from file failed: '%s'", file_path.c_str());
                    return;
                }

                bool success = model_loader.load_tensors(tensors, backend);

                if (!success) {
                    LOG_ERROR("load tensors from model loader failed");
                    return;
                }

                LOG_INFO("vae model loaded");
            }
            vae->test();
        }
    };

};

#endif