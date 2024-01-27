#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include "ggml_extend.hpp"

class DownSampleBlock : public GGMLBlock {
protected:
    int channels;
    int out_channels;
    bool vae_downsample;

public:
    DownSampleBlock(int channels,
                    int out_channels,
                    bool vae_downsample = false)
        : channels(channels),
          out_channels(out_channels),
          vae_downsample(vae_downsample) {
        if (vae_downsample) {
            blocks["conv"] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, out_channels, {3, 3}, {2, 2}, {0, 0}));
        } else {
            blocks["op"] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, out_channels, {3, 3}, {2, 2}, {1, 1}));
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, channels, h, w]
        if (vae_downsample) {
            auto conv = std::dynamic_pointer_cast<Conv2d>(blocks["conv"]);

            x = ggml_pad(ctx, x, 1, 1, 0, 0);
            x = conv->forward(ctx, x);
        } else {
            auto conv = std::dynamic_pointer_cast<Conv2d>(blocks["op"]);

            x = conv->forward(ctx, x);
        }
        return x;  // [N, out_channels, h/2, w/2]
    }
};

class UpSampleBlock : public GGMLBlock {
protected:
    int channels;
    int out_channels;

public:
    UpSampleBlock(int channels,
                  int out_channels)
        : channels(channels),
          out_channels(out_channels) {
        blocks["conv"] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, out_channels, {3, 3}, {1, 1}, {1, 1}));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, channels, h, w]
        auto conv = std::dynamic_pointer_cast<Conv2d>(blocks["conv"]);

        x = ggml_upscale(ctx, x, 2);  // [N, channels, h*2, w*2]
        x = conv->forward(ctx, x);    // [N, out_channels, h*2, w*2]
        return x;
    }
};

class ResBlock : public GGMLBlock {
protected:
    // network hparams
    int64_t channels;      // model_channels * (1, 1, 1, 2, 2, 4, 4, 4)
    int64_t emb_channels;  // time_embed_dim
    int64_t out_channels;  // mult * model_channels
    std::pair<int, int> kernel_size;
    int dims;
    bool skip_t_emb;
    bool exchange_temb_dims;

    std::shared_ptr<GGMLBlock> conv_nd(int dims,
                                       int64_t in_channels,
                                       int64_t out_channels,
                                       std::pair<int, int> kernel_size,
                                       std::pair<int, int> padding) {
        GGML_ASSERT(dims == 2 || dims == 3);
        if (dims == 3) {
            return std::shared_ptr<GGMLBlock>(new Conv3dnx1x1(in_channels, out_channels, kernel_size.first, 1, padding.first));
        } else {
            return std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, out_channels, kernel_size, {1, 1}, padding));
        }
    }

public:
    ResBlock(int64_t channels,
             int64_t emb_channels,
             int64_t out_channels,
             std::pair<int, int> kernel_size = {3, 3},
             int dims                        = 2,
             bool exchange_temb_dims         = false,
             bool skip_t_emb                 = false)
        : channels(channels),
          emb_channels(emb_channels),
          out_channels(out_channels),
          kernel_size(kernel_size),
          dims(dims),
          skip_t_emb(skip_t_emb),
          exchange_temb_dims(exchange_temb_dims) {
        std::pair<int, int> padding = {kernel_size.first / 2, kernel_size.second / 2};
        blocks["in_layers.0"]       = std::shared_ptr<GGMLBlock>(new GroupNorm32(channels));
        // in_layer_1 is nn.SILU()
        blocks["in_layers.2"] = conv_nd(dims, channels, out_channels, kernel_size, padding);

        if (!skip_t_emb) {
            // emb_layer_0 is nn.SILU()
            blocks["emb_layers.1"] = std::shared_ptr<GGMLBlock>(new Linear(emb_channels, out_channels));
        }

        blocks["out_layers.0"] = std::shared_ptr<GGMLBlock>(new GroupNorm32(out_channels));
        // out_layer_1 is nn.SILU()
        // out_layer_2 is nn.Dropout(), skip for inference
        blocks["out_layers.3"] = conv_nd(dims, out_channels, out_channels, kernel_size, padding);

        if (out_channels != channels) {
            blocks["skip_connection"] = conv_nd(dims, channels, out_channels, {1, 1}, {0, 0});
        }
    }

    virtual struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* emb = NULL) {
        // For dims==3, we reduce dimension from 5d to 4d by merging h and w, in order not to change ggml
        // [N, c, t, h, w] => [N, c, t, h * w]
        // x: [N, channels, h, w] if dims == 2 else [N, channels, t, h, w]
        // emb: [N, emb_channels] if dims == 2 else [N, t, emb_channels]
        auto in_layers_0  = std::dynamic_pointer_cast<GroupNorm32>(blocks["in_layers.0"]);
        auto in_layers_2  = std::dynamic_pointer_cast<UnaryBlock>(blocks["in_layers.2"]);
        auto out_layers_0 = std::dynamic_pointer_cast<GroupNorm32>(blocks["out_layers.0"]);
        auto out_layers_3 = std::dynamic_pointer_cast<UnaryBlock>(blocks["out_layers.3"]);

        if (emb == NULL) {
            GGML_ASSERT(skip_t_emb);
        }

        // in_layers
        auto h = in_layers_0->forward(ctx, x);
        h      = ggml_silu_inplace(ctx, h);
        h      = in_layers_2->forward(ctx, h);  // [N, out_channels, h, w] if dims == 2 else [N, out_channels, t, h, w]

        // emb_layers
        if (!skip_t_emb) {
            auto emb_layer_1 = std::dynamic_pointer_cast<Linear>(blocks["emb_layers.1"]);

            auto emb_out = ggml_silu(ctx, emb);
            emb_out      = emb_layer_1->forward(ctx, emb_out);  // [N, out_channels] if dims == 2 else [N, t, out_channels]

            if (dims == 2) {
                emb_out = ggml_reshape_4d(ctx, emb_out, 1, 1, emb_out->ne[0], emb_out->ne[1]);  // [N, out_channels, 1, 1]
            } else {
                emb_out = ggml_reshape_4d(ctx, emb_out, 1, emb_out->ne[0], emb_out->ne[1], emb_out->ne[2]);  // [N, t, out_channels, 1]
                if (exchange_temb_dims) {
                    // emb_out = rearrange(emb_out, "b t c ... -> b c t ...")
                    emb_out = ggml_cont(ctx, ggml_permute(ctx, emb_out, 0, 2, 1, 3));  // [N, out_channels, t, 1]
                }
            }

            h = ggml_add(ctx, h, emb_out);  // [N, out_channels, h, w] if dims == 2 else [N, out_channels, t, h, w]
        }

        // out_layers
        h = out_layers_0->forward(ctx, h);
        h = ggml_silu_inplace(ctx, h);
        // dropout, skip for inference
        h = out_layers_3->forward(ctx, h);

        // skip connection
        if (out_channels != channels) {
            auto skip_connection = std::dynamic_pointer_cast<UnaryBlock>(blocks["skip_connection"]);
            x                    = skip_connection->forward(ctx, x);  // [N, out_channels, h, w] if dims == 2 else [N, out_channels, t, h, w]
        }

        h = ggml_add(ctx, h, x);
        return h;  // [N, out_channels, h, w] if dims == 2 else [N, out_channels, t, h, w]
    }
};

class AlphaBlender : public GGMLBlock {
protected:
    void init_params(struct ggml_context* ctx, ggml_type wtype) {
        params["mix_factor"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    }

    float get_alpha() {
        // image_only_indicator is always tensor([0.]) and since mix_factor.shape is [1,]
        // so learned_with_images is same as learned
        float alpha = ggml_backend_tensor_get_f32(params["mix_factor"]);
        return sigmoid(alpha);
    }

public:
    AlphaBlender() {
        // merge_strategy is always learned_with_images
        // for inference, we don't need to set alpha
        // since mix_factor.shape is [1,], we don't need rearrange using  rearrange_pattern
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* x_spatial,
                                struct ggml_tensor* x_temporal) {
        // image_only_indicator is always tensor([0.])
        float alpha = get_alpha();
        auto x      = ggml_add(ctx,
                               ggml_scale(ctx, x_spatial, alpha),
                               ggml_scale(ctx, x_temporal, 1.0f - alpha));
        return x;
    }
};

class VideoResBlock : public ResBlock {
public:
    VideoResBlock(int channels,
                  int emb_channels,
                  int out_channels,
                  std::pair<int, int> kernel_size = {3, 3},
                  int64_t video_kernel_size       = 3,
                  int dims                        = 2)  // always 2
        : ResBlock(channels, emb_channels, out_channels, kernel_size, dims) {
        blocks["time_stack"] = std::shared_ptr<GGMLBlock>(new ResBlock(out_channels, emb_channels, out_channels, kernel_size, 3, true));
        blocks["time_mixer"] = std::shared_ptr<GGMLBlock>(new AlphaBlender());
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* x,
                                struct ggml_tensor* emb,
                                int num_video_frames) {
        // x: [N, channels, h, w] aka [b*t, channels, h, w]
        // emb: [N, emb_channels] aka [b*t, emb_channels]
        // image_only_indicator is always tensor([0.])
        auto time_stack = std::dynamic_pointer_cast<ResBlock>(blocks["time_stack"]);
        auto time_mixer = std::dynamic_pointer_cast<AlphaBlender>(blocks["time_mixer"]);

        x = ResBlock::forward(ctx, x, emb);

        int64_t T = num_video_frames;
        int64_t B = x->ne[3] / T;
        int64_t C = x->ne[2];
        int64_t H = x->ne[1];
        int64_t W = x->ne[0];

        x          = ggml_reshape_4d(ctx, x, W * H, C, T, B);           // (b t) c h w -> b t c (h w)
        x          = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));  // b t c (h w) -> b c t (h w)
        auto x_mix = x;

        emb = ggml_reshape_4d(ctx, emb, emb->ne[0], T, B, emb->ne[3]);  // (b t) ... -> b t ...

        x = time_stack->forward(ctx, x, emb);  // b t c (h w)

        x = time_mixer->forward(ctx, x_mix, x);  // b t c (h w)

        x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));  // b c t (h w) -> b t c (h w)
        x = ggml_reshape_4d(ctx, x, W, H, C, T * B);           // b t c (h w) -> (b t) c h w

        return x;
    }
};

#endif  // __COMMON_HPP__