#ifndef __UNET_HPP__
#define __UNET_HPP__

#include <array>
#include <cstring>

#include "common_block.hpp"
#include "layer_streaming.hpp"
#include "layer_streaming_executor.hpp"
#include "model.h"

/*==================================================== UnetModel =====================================================*/

#define UNET_GRAPH_SIZE 102400

class SpatialVideoTransformer : public SpatialTransformer {
protected:
    int64_t time_depth;
    int max_time_embed_period;

public:
    SpatialVideoTransformer(int64_t in_channels,
                            int64_t n_head,
                            int64_t d_head,
                            int64_t depth,
                            int64_t context_dim,
                            bool use_linear,
                            int64_t time_depth        = 1,
                            int max_time_embed_period = 10000)
        : SpatialTransformer(in_channels, n_head, d_head, depth, context_dim, use_linear),
          max_time_embed_period(max_time_embed_period) {
        // We will convert unet transformer linear to conv2d 1x1 when loading the weights, so use_linear is always False
        // use_spatial_context is always True
        // merge_strategy is always learned_with_images
        // merge_factor is loaded from weights
        // time_context_dim is always None
        // ff_in is always True
        // disable_self_attn is always False
        // disable_temporal_crossattention is always False

        int64_t inner_dim = n_head * d_head;

        GGML_ASSERT(depth == time_depth);
        GGML_ASSERT(in_channels == inner_dim);

        int64_t time_mix_d_head    = d_head;
        int64_t n_time_mix_heads   = n_head;
        int64_t time_mix_inner_dim = time_mix_d_head * n_time_mix_heads;  // equal to inner_dim
        int64_t time_context_dim   = context_dim;

        for (int i = 0; i < time_depth; i++) {
            std::string name = "time_stack." + std::to_string(i);
            blocks[name]     = std::shared_ptr<GGMLBlock>(new BasicTransformerBlock(inner_dim,
                                                                                    n_time_mix_heads,
                                                                                    time_mix_d_head,
                                                                                    time_context_dim,
                                                                                    true));
        }

        int64_t time_embed_dim     = in_channels * 4;
        blocks["time_pos_embed.0"] = std::shared_ptr<GGMLBlock>(new Linear(in_channels, time_embed_dim));
        // time_pos_embed.1 is nn.SiLU()
        blocks["time_pos_embed.2"] = std::shared_ptr<GGMLBlock>(new Linear(time_embed_dim, in_channels));

        blocks["time_mixer"] = std::shared_ptr<GGMLBlock>(new AlphaBlender());
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* x,
                         ggml_tensor* context,
                         int timesteps) {
        // x: [N, in_channels, h, w] aka [b*t, in_channels, h, w], t == timesteps
        // context: [N, max_position(aka n_context), hidden_size(aka context_dim)] aka [b*t, n_context, context_dim], t == timesteps
        // t_emb: [N, in_channels] aka [b*t, in_channels]
        // timesteps is num_frames
        // time_context is always None
        // image_only_indicator is always tensor([0.])
        // transformer_options is not used
        // GGML_ASSERT(ggml_n_dims(context) == 3);

        auto norm             = std::dynamic_pointer_cast<GroupNorm32>(blocks["norm"]);
        auto proj_in          = std::dynamic_pointer_cast<Conv2d>(blocks["proj_in"]);
        auto proj_out         = std::dynamic_pointer_cast<Conv2d>(blocks["proj_out"]);
        auto time_pos_embed_0 = std::dynamic_pointer_cast<Linear>(blocks["time_pos_embed.0"]);
        auto time_pos_embed_2 = std::dynamic_pointer_cast<Linear>(blocks["time_pos_embed.2"]);
        auto time_mixer       = std::dynamic_pointer_cast<AlphaBlender>(blocks["time_mixer"]);

        auto x_in         = x;
        int64_t n         = x->ne[3];
        int64_t h         = x->ne[1];
        int64_t w         = x->ne[0];
        int64_t inner_dim = n_head * d_head;

        GGML_ASSERT(n == timesteps);  // We compute cond and uncond separately, so batch_size==1

        auto time_context    = context;  // [b*t, n_context, context_dim]
        auto spatial_context = context;
        // time_context_first_timestep = time_context[::timesteps]
        auto time_context_first_timestep = ggml_view_3d(ctx->ggml_ctx,
                                                        time_context,
                                                        time_context->ne[0],
                                                        time_context->ne[1],
                                                        1,
                                                        time_context->nb[1],
                                                        time_context->nb[2],
                                                        0);  // [b, n_context, context_dim]
        time_context                     = ggml_new_tensor_3d(ctx->ggml_ctx, GGML_TYPE_F32,
                                                              time_context_first_timestep->ne[0],
                                                              time_context_first_timestep->ne[1],
                                                              time_context_first_timestep->ne[2] * h * w);
        time_context                     = ggml_repeat(ctx->ggml_ctx, time_context_first_timestep, time_context);  // [b*h*w, n_context, context_dim]

        x = norm->forward(ctx, x);
        x = proj_in->forward(ctx, x);  // [N, inner_dim, h, w]

        x = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 2, 0, 3));  // [N, h, w, inner_dim]
        x = ggml_reshape_3d(ctx->ggml_ctx, x, inner_dim, w * h, n);                // [N, h * w, inner_dim]

        auto num_frames = ggml_arange(ctx->ggml_ctx, 0.f, static_cast<float>(timesteps), 1.f);
        // since b is 1, no need to do repeat
        auto t_emb = ggml_ext_timestep_embedding(ctx->ggml_ctx, num_frames, static_cast<int>(in_channels), max_time_embed_period);  // [N, in_channels]

        auto emb = time_pos_embed_0->forward(ctx, t_emb);
        emb      = ggml_silu_inplace(ctx->ggml_ctx, emb);
        emb      = time_pos_embed_2->forward(ctx, emb);                             // [N, in_channels]
        emb      = ggml_reshape_3d(ctx->ggml_ctx, emb, emb->ne[0], 1, emb->ne[1]);  // [N, 1, in_channels]

        for (int i = 0; i < depth; i++) {
            std::string transformer_name = "transformer_blocks." + std::to_string(i);
            std::string time_stack_name  = "time_stack." + std::to_string(i);

            auto block     = std::dynamic_pointer_cast<BasicTransformerBlock>(blocks[transformer_name]);
            auto mix_block = std::dynamic_pointer_cast<BasicTransformerBlock>(blocks[time_stack_name]);

            x = block->forward(ctx, x, spatial_context);  // [N, h * w, inner_dim]

            // in_channels == inner_dim
            auto x_mix = x;
            x_mix      = ggml_add(ctx->ggml_ctx, x_mix, emb);  // [N, h * w, inner_dim]

            int64_t N = x_mix->ne[2];
            int64_t T = timesteps;
            int64_t B = N / T;
            int64_t S = x_mix->ne[1];
            int64_t C = x_mix->ne[0];

            x_mix = ggml_reshape_4d(ctx->ggml_ctx, x_mix, C, S, T, B);                         // (b t) s c -> b t s c
            x_mix = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x_mix, 0, 2, 1, 3));  // b t s c -> b s t c
            x_mix = ggml_reshape_3d(ctx->ggml_ctx, x_mix, C, T, S * B);                        // b s t c -> (b s) t c

            x_mix = mix_block->forward(ctx, x_mix, time_context);  // [B * h * w, T, inner_dim]

            x_mix = ggml_reshape_4d(ctx->ggml_ctx, x_mix, C, T, S, B);                         // (b s) t c -> b s t c
            x_mix = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x_mix, 0, 2, 1, 3));  // b s t c -> b t s c
            x_mix = ggml_reshape_3d(ctx->ggml_ctx, x_mix, C, S, T * B);                        // b t s c -> (b t) s c

            x = time_mixer->forward(ctx, x, x_mix);  // [N, h * w, inner_dim]
        }

        x = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));  // [N, inner_dim, h * w]
        x = ggml_reshape_4d(ctx->ggml_ctx, x, w, h, inner_dim, n);                 // [N, inner_dim, h, w]

        // proj_out
        x = proj_out->forward(ctx, x);  // [N, in_channels, h, w]

        x = ggml_add(ctx->ggml_ctx, x, x_in);
        return x;
    }
};

// ldm.modules.diffusionmodules.openaimodel.UNetModel
class UnetModelBlock : public GGMLBlock {
protected:
    SDVersion version = VERSION_SD1;
    // network hparams
    int in_channels                        = 4;
    int out_channels                       = 4;
    int num_res_blocks                     = 2;
    std::vector<int> attention_resolutions = {4, 2, 1};
    std::vector<int> channel_mult          = {1, 2, 4, 4};
    std::vector<int> transformer_depth     = {1, 1, 1, 1};
    int time_embed_dim                     = 1280;  // model_channels*4
    int num_heads                          = 8;
    int num_head_channels                  = -1;   // channels // num_heads
    int context_dim                        = 768;  // 1024 for VERSION_SD2, 2048 for VERSION_SDXL
    bool use_linear_projection             = false;
    bool tiny_unet                         = false;

public:
    int model_channels  = 320;
    int adm_in_channels = 2816;  // only for VERSION_SDXL/SVD

    UnetModelBlock(SDVersion version = VERSION_SD1, const String2TensorStorage& tensor_storage_map = {})
        : version(version) {
        if (sd_version_is_sd2(version)) {
            context_dim           = 1024;
            num_head_channels     = 64;
            num_heads             = -1;
            use_linear_projection = true;
        } else if (sd_version_is_sdxl(version)) {
            context_dim           = 2048;
            attention_resolutions = {4, 2};
            channel_mult          = {1, 2, 4};
            transformer_depth     = {1, 2, 10};
            num_head_channels     = 64;
            num_heads             = -1;
            use_linear_projection = true;
            if (version == VERSION_SDXL_VEGA) {
                transformer_depth = {1, 1, 2};
            }
        } else if (version == VERSION_SVD) {
            in_channels           = 8;
            out_channels          = 4;
            context_dim           = 1024;
            adm_in_channels       = 768;
            num_head_channels     = 64;
            num_heads             = -1;
            use_linear_projection = true;
        }
        if (sd_version_is_inpaint(version)) {
            in_channels = 9;
        } else if (sd_version_is_unet_edit(version)) {
            in_channels = 8;
        }
        if (version == VERSION_SD1_TINY_UNET || version == VERSION_SD2_TINY_UNET || version == VERSION_SDXS_512_DS || version == VERSION_SDXS_09) {
            num_res_blocks = 1;
            channel_mult   = {1, 2, 4};
            tiny_unet      = true;
            if (version == VERSION_SDXS_512_DS) {
                attention_resolutions = {4, 2};  // here just like SDXL
            }
        }

        // dims is always 2
        // use_temporal_attention is always True for SVD

        blocks["time_embed.0"] = std::shared_ptr<GGMLBlock>(new Linear(model_channels, time_embed_dim));
        // time_embed_1 is nn.SiLU()
        blocks["time_embed.2"] = std::shared_ptr<GGMLBlock>(new Linear(time_embed_dim, time_embed_dim));

        if (sd_version_is_sdxl(version) || version == VERSION_SVD) {
            blocks["label_emb.0.0"] = std::shared_ptr<GGMLBlock>(new Linear(adm_in_channels, time_embed_dim));
            // label_emb_1 is nn.SiLU()
            blocks["label_emb.0.2"] = std::shared_ptr<GGMLBlock>(new Linear(time_embed_dim, time_embed_dim));
        }

        // input_blocks
        blocks["input_blocks.0.0"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, model_channels, {3, 3}, {1, 1}, {1, 1}));

        std::vector<int> input_block_chans;
        input_block_chans.push_back(model_channels);
        int ch              = model_channels;
        int input_block_idx = 0;
        int ds              = 1;

        auto get_resblock = [&](int64_t channels, int64_t emb_channels, int64_t out_channels) -> ResBlock* {
            if (version == VERSION_SVD) {
                return new VideoResBlock(channels, emb_channels, out_channels);
            } else {
                return new ResBlock(channels, emb_channels, out_channels);
            }
        };

        auto get_attention_layer = [&](int64_t in_channels,
                                       int64_t n_head,
                                       int64_t d_head,
                                       int64_t depth,
                                       int64_t context_dim) -> SpatialTransformer* {
            if (version == VERSION_SVD) {
                return new SpatialVideoTransformer(in_channels, n_head, d_head, depth, context_dim, use_linear_projection);
            } else {
                if (version == VERSION_SDXS_09 && n_head == 5) {
                    n_head = 1;    // to carry a special case of sdxs_09 into CrossAttentionLayer,
                    d_head = 320;  // works as long the product remains equal (5*64 == 1*320)
                }
                return new SpatialTransformer(in_channels, n_head, d_head, depth, context_dim, use_linear_projection);
            }
        };

        size_t len_mults = channel_mult.size();
        for (int i = 0; i < len_mults; i++) {
            int mult = channel_mult[i];
            for (int j = 0; j < num_res_blocks; j++) {
                input_block_idx += 1;
                std::string name = "input_blocks." + std::to_string(input_block_idx) + ".0";
                blocks[name]     = std::shared_ptr<GGMLBlock>(get_resblock(ch, time_embed_dim, mult * model_channels));

                ch = mult * model_channels;
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    int n_head = num_heads;
                    int d_head = ch / num_heads;
                    if (num_head_channels != -1) {
                        d_head = num_head_channels;
                        n_head = ch / d_head;
                    }
                    std::string name = "input_blocks." + std::to_string(input_block_idx) + ".1";
                    int td           = transformer_depth[i];
                    if (version == VERSION_SDXL_SSD1B) {
                        if (i == 2) {
                            td = 4;
                        }
                    }
                    blocks[name] = std::shared_ptr<GGMLBlock>(get_attention_layer(ch,
                                                                                  n_head,
                                                                                  d_head,
                                                                                  td,
                                                                                  context_dim));
                }
                input_block_chans.push_back(ch);
                if (tiny_unet) {
                    input_block_idx++;
                }
            }
            if (i != len_mults - 1) {
                input_block_idx += 1;
                std::string name = "input_blocks." + std::to_string(input_block_idx) + ".0";
                blocks[name]     = std::shared_ptr<GGMLBlock>(new DownSampleBlock(ch, ch));

                input_block_chans.push_back(ch);
                ds *= 2;
            }
        }

        // middle blocks
        int n_head = num_heads;
        int d_head = ch / num_heads;
        if (num_head_channels != -1) {
            d_head = num_head_channels;
            n_head = ch / d_head;
        }
        if (!tiny_unet) {
            blocks["middle_block.0"] = std::shared_ptr<GGMLBlock>(get_resblock(ch, time_embed_dim, ch));
            if (version != VERSION_SDXL_SSD1B && version != VERSION_SDXL_VEGA) {
                blocks["middle_block.1"] = std::shared_ptr<GGMLBlock>(get_attention_layer(ch,
                                                                                          n_head,
                                                                                          d_head,
                                                                                          transformer_depth[transformer_depth.size() - 1],
                                                                                          context_dim));
                blocks["middle_block.2"] = std::shared_ptr<GGMLBlock>(get_resblock(ch, time_embed_dim, ch));
            }
        }
        // output_blocks
        int output_block_idx = 0;
        for (int i = (int)len_mults - 1; i >= 0; i--) {
            int mult = channel_mult[i];
            for (int j = 0; j < num_res_blocks + 1; j++) {
                int ich = input_block_chans.back();
                input_block_chans.pop_back();

                std::string name = "output_blocks." + std::to_string(output_block_idx) + ".0";
                blocks[name]     = std::shared_ptr<GGMLBlock>(get_resblock(ch + ich, time_embed_dim, mult * model_channels));

                ch                = mult * model_channels;
                int up_sample_idx = 1;
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    int n_head = num_heads;
                    int d_head = ch / num_heads;
                    if (num_head_channels != -1) {
                        d_head = num_head_channels;
                        n_head = ch / d_head;
                    }
                    std::string name = "output_blocks." + std::to_string(output_block_idx) + ".1";
                    int td           = transformer_depth[i];
                    if (version == VERSION_SDXL_SSD1B) {
                        if (i == 2 && (j == 0 || j == 1)) {
                            td = 4;
                        }
                        if (i == 1 && (j == 1 || j == 2)) {
                            td = 1;
                        }
                    }
                    blocks[name] = std::shared_ptr<GGMLBlock>(get_attention_layer(ch, n_head, d_head, td, context_dim));

                    up_sample_idx++;
                }

                if (i > 0 && j == num_res_blocks) {
                    if (tiny_unet) {
                        output_block_idx++;
                        if (output_block_idx == 2) {
                            up_sample_idx = 1;
                        }
                    }
                    std::string name = "output_blocks." + std::to_string(output_block_idx) + "." + std::to_string(up_sample_idx);
                    blocks[name]     = std::shared_ptr<GGMLBlock>(new UpSampleBlock(ch, ch));

                    ds /= 2;
                }

                output_block_idx += 1;
            }
        }

        // out
        blocks["out.0"] = std::shared_ptr<GGMLBlock>(new GroupNorm32(ch));  // ch == model_channels
        // out_1 is nn.SiLU()
        blocks["out.2"] = std::shared_ptr<GGMLBlock>(new Conv2d(model_channels, out_channels, {3, 3}, {1, 1}, {1, 1}));
    }

    ggml_tensor* resblock_forward(std::string name,
                                  GGMLRunnerContext* ctx,
                                  ggml_tensor* x,
                                  ggml_tensor* emb,
                                  int num_video_frames) {
        if (version == VERSION_SVD) {
            auto block = std::dynamic_pointer_cast<VideoResBlock>(blocks[name]);

            return block->forward(ctx, x, emb, num_video_frames);
        } else {
            auto block = std::dynamic_pointer_cast<ResBlock>(blocks[name]);

            return block->forward(ctx, x, emb);
        }
    }

    ggml_tensor* attention_layer_forward(std::string name,
                                         GGMLRunnerContext* ctx,
                                         ggml_tensor* x,
                                         ggml_tensor* context,
                                         int timesteps) {
        if (version == VERSION_SVD) {
            auto block = std::dynamic_pointer_cast<SpatialVideoTransformer>(blocks[name]);

            return block->forward(ctx, x, context, timesteps);
        } else {
            auto block = std::dynamic_pointer_cast<SpatialTransformer>(blocks[name]);

            return block->forward(ctx, x, context);
        }
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* x,
                         ggml_tensor* timesteps,
                         ggml_tensor* context,
                         ggml_tensor* c_concat              = nullptr,
                         ggml_tensor* y                     = nullptr,
                         int num_video_frames               = -1,
                         std::vector<ggml_tensor*> controls = {},
                         float control_strength             = 0.f) {
        // x: [N, in_channels, h, w] or [N, in_channels/2, h, w]
        // timesteps: [N,]
        // context: [N, max_position, hidden_size] or [1, max_position, hidden_size]. for example, [N, 77, 768]
        // c_concat: [N, in_channels, h, w] or [1, in_channels, h, w]
        // y: [N, adm_in_channels] or [1, adm_in_channels]
        // return: [N, out_channels, h, w]
        if (context != nullptr) {
            if (context->ne[2] != x->ne[3]) {
                context = ggml_repeat(ctx->ggml_ctx, context, ggml_new_tensor_3d(ctx->ggml_ctx, GGML_TYPE_F32, context->ne[0], context->ne[1], x->ne[3]));
            }
        }

        if (c_concat != nullptr) {
            if (c_concat->ne[3] != x->ne[3]) {
                c_concat = ggml_repeat(ctx->ggml_ctx, c_concat, x);
            }
            x = ggml_concat(ctx->ggml_ctx, x, c_concat, 2);
        }

        if (y != nullptr) {
            if (y->ne[1] != x->ne[3]) {
                y = ggml_repeat(ctx->ggml_ctx, y, ggml_new_tensor_2d(ctx->ggml_ctx, GGML_TYPE_F32, y->ne[0], x->ne[3]));
            }
        }

        auto time_embed_0     = std::dynamic_pointer_cast<Linear>(blocks["time_embed.0"]);
        auto time_embed_2     = std::dynamic_pointer_cast<Linear>(blocks["time_embed.2"]);
        auto input_blocks_0_0 = std::dynamic_pointer_cast<Conv2d>(blocks["input_blocks.0.0"]);

        auto out_0 = std::dynamic_pointer_cast<GroupNorm32>(blocks["out.0"]);
        auto out_2 = std::dynamic_pointer_cast<Conv2d>(blocks["out.2"]);

        auto t_emb = ggml_ext_timestep_embedding(ctx->ggml_ctx, timesteps, model_channels);  // [N, model_channels]

        auto emb = time_embed_0->forward(ctx, t_emb);
        emb      = ggml_silu_inplace(ctx->ggml_ctx, emb);
        emb      = time_embed_2->forward(ctx, emb);  // [N, time_embed_dim]

        // SDXL/SVD
        if (y != nullptr) {
            auto label_embed_0 = std::dynamic_pointer_cast<Linear>(blocks["label_emb.0.0"]);
            auto label_embed_2 = std::dynamic_pointer_cast<Linear>(blocks["label_emb.0.2"]);

            auto label_emb = label_embed_0->forward(ctx, y);
            label_emb      = ggml_silu_inplace(ctx->ggml_ctx, label_emb);
            label_emb      = label_embed_2->forward(ctx, label_emb);  // [N, time_embed_dim]

            emb = ggml_add(ctx->ggml_ctx, emb, label_emb);  // [N, time_embed_dim]
        }
        // sd::ggml_graph_cut::mark_graph_cut(emb, "unet.prelude", "emb");

        // input_blocks
        std::vector<ggml_tensor*> hs;

        // input block 0
        auto h = input_blocks_0_0->forward(ctx, x);
        sd::ggml_graph_cut::mark_graph_cut(h, "unet.input_blocks.0", "h");

        ggml_set_name(h, "bench-start");
        hs.push_back(h);
        // input block 1-11
        size_t len_mults    = channel_mult.size();
        int input_block_idx = 0;
        int ds              = 1;
        for (int i = 0; i < len_mults; i++) {
            int mult = channel_mult[i];
            for (int j = 0; j < num_res_blocks; j++) {
                input_block_idx += 1;
                std::string name = "input_blocks." + std::to_string(input_block_idx) + ".0";
                h                = resblock_forward(name, ctx, h, emb, num_video_frames);  // [N, mult*model_channels, h, w]
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    std::string name = "input_blocks." + std::to_string(input_block_idx) + ".1";
                    h                = attention_layer_forward(name, ctx, h, context, num_video_frames);  // [N, mult*model_channels, h, w]
                }
                sd::ggml_graph_cut::mark_graph_cut(h, "unet.input_blocks." + std::to_string(input_block_idx), "h");
                hs.push_back(h);
            }
            if (tiny_unet) {
                input_block_idx++;
            }
            if (i != len_mults - 1) {
                ds *= 2;
                input_block_idx += 1;

                std::string name = "input_blocks." + std::to_string(input_block_idx) + ".0";
                auto block       = std::dynamic_pointer_cast<DownSampleBlock>(blocks[name]);

                h = block->forward(ctx, h);  // [N, mult*model_channels, h/(2^(i+1)), w/(2^(i+1))]
                // sd::ggml_graph_cut::mark_graph_cut(h, "unet.input_blocks." + std::to_string(input_block_idx), "h");
                hs.push_back(h);
            }
        }
        // [N, 4*model_channels, h/8, w/8]

        // middle_block
        if (!tiny_unet) {
            h = resblock_forward("middle_block.0", ctx, h, emb, num_video_frames);  // [N, 4*model_channels, h/8, w/8]
            if (version != VERSION_SDXL_SSD1B && version != VERSION_SDXL_VEGA) {
                h = attention_layer_forward("middle_block.1", ctx, h, context, num_video_frames);  // [N, 4*model_channels, h/8, w/8]
                h = resblock_forward("middle_block.2", ctx, h, emb, num_video_frames);             // [N, 4*model_channels, h/8, w/8]
            }
        }
        sd::ggml_graph_cut::mark_graph_cut(h, "unet.middle_block", "h");
        if (controls.size() > 0) {
            auto cs = ggml_ext_scale(ctx->ggml_ctx, controls[controls.size() - 1], control_strength, true);
            h       = ggml_add(ctx->ggml_ctx, h, cs);  // middle control
        }
        int control_offset = static_cast<int>(controls.size() - 2);

        // output_blocks
        int output_block_idx = 0;
        for (int i = (int)len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                auto h_skip = hs.back();
                hs.pop_back();

                if (controls.size() > 0) {
                    auto cs = ggml_ext_scale(ctx->ggml_ctx, controls[control_offset], control_strength, true);
                    h_skip  = ggml_add(ctx->ggml_ctx, h_skip, cs);  // control net condition
                    control_offset--;
                }

                h = ggml_concat(ctx->ggml_ctx, h, h_skip, 2);

                std::string name = "output_blocks." + std::to_string(output_block_idx) + ".0";

                h = resblock_forward(name, ctx, h, emb, num_video_frames);

                int up_sample_idx = 1;
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    std::string name = "output_blocks." + std::to_string(output_block_idx) + ".1";

                    h = attention_layer_forward(name, ctx, h, context, num_video_frames);

                    up_sample_idx++;
                }

                if (i > 0 && j == num_res_blocks) {
                    if (tiny_unet) {
                        output_block_idx++;
                        if (output_block_idx == 2) {
                            up_sample_idx = 1;
                        }
                    }
                    std::string name = "output_blocks." + std::to_string(output_block_idx) + "." + std::to_string(up_sample_idx);
                    auto block       = std::dynamic_pointer_cast<UpSampleBlock>(blocks[name]);

                    h = block->forward(ctx, h);

                    ds /= 2;
                }

                output_block_idx += 1;
                sd::ggml_graph_cut::mark_graph_cut(h, "unet.output_blocks." + std::to_string(output_block_idx - 1), "h");
            }
        }

        // out
        h = out_0->forward(ctx, h);
        h = ggml_silu_inplace(ctx->ggml_ctx, h);
        h = out_2->forward(ctx, h);
        ggml_set_name(h, "bench-end");
        return h;  // [N, out_channels, h, w]
    }

    ggml_tensor* forward_embedding_stage(GGMLRunnerContext* ctx,
                                          struct ggml_tensor* timesteps,
                                          struct ggml_tensor* label) {
        auto time_embed_0 = std::dynamic_pointer_cast<Linear>(blocks["time_embed.0"]);
        auto time_embed_2 = std::dynamic_pointer_cast<Linear>(blocks["time_embed.2"]);

        auto emb = ggml_ext_timestep_embedding(ctx->ggml_ctx, timesteps, model_channels);
        emb      = time_embed_0->forward(ctx, emb);
        emb      = ggml_silu_inplace(ctx->ggml_ctx, emb);
        emb      = time_embed_2->forward(ctx, emb);

        if (label != nullptr && adm_in_channels != -1) {
            auto label_embed_0 = std::dynamic_pointer_cast<Linear>(blocks["label_emb.0.0"]);
            auto label_embed_2 = std::dynamic_pointer_cast<Linear>(blocks["label_emb.0.2"]);

            auto label_emb = label_embed_0->forward(ctx, label);
            label_emb      = ggml_silu_inplace(ctx->ggml_ctx, label_emb);
            label_emb      = label_embed_2->forward(ctx, label_emb);

            emb = ggml_add(ctx->ggml_ctx, emb, label_emb);
        }

        return emb;
    }

    ggml_tensor* forward_initial_conv(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
        auto input_blocks_0_0 = std::dynamic_pointer_cast<Conv2d>(blocks["input_blocks.0.0"]);
        return input_blocks_0_0->forward(ctx, x);
    }

    ggml_tensor* forward_input_block(GGMLRunnerContext* ctx,
                                      int block_idx,
                                      struct ggml_tensor* h,
                                      struct ggml_tensor* emb,
                                      struct ggml_tensor* context,
                                      int num_video_frames) {
        // input_blocks.X.0 is either a ResBlock or a DownSampleBlock —
        // SDXL/SD1.x put the per-stage downsample at indices 3 and 6. The
        // non-streaming forward() differentiates these inline; the streaming
        // path does the same here.
        std::string slot0_name = "input_blocks." + std::to_string(block_idx) + ".0";
        auto slot0_it = blocks.find(slot0_name);
        if (slot0_it != blocks.end()) {
            if (auto downsample = std::dynamic_pointer_cast<DownSampleBlock>(slot0_it->second)) {
                h = downsample->forward(ctx, h);
            } else {
                h = resblock_forward(slot0_name, ctx, h, emb, num_video_frames);
            }
        }

        // input_blocks.X.1 is a SpatialTransformer when attention applies at this resolution.
        std::string attn_name = "input_blocks." + std::to_string(block_idx) + ".1";
        auto attn_block = blocks.find(attn_name);
        if (attn_block != blocks.end()) {
            h = attention_layer_forward(attn_name, ctx, h, context, num_video_frames);
        }

        return h;
    }

    ggml_tensor* forward_middle_block(GGMLRunnerContext* ctx,
                                       struct ggml_tensor* h,
                                       struct ggml_tensor* emb,
                                       struct ggml_tensor* context,
                                       int num_video_frames) {
        h = resblock_forward("middle_block.0", ctx, h, emb, num_video_frames);
        if (version == VERSION_SD1 || version == VERSION_SD2 || version == VERSION_SVD) {
            h = attention_layer_forward("middle_block.1", ctx, h, context, num_video_frames);
            h = resblock_forward("middle_block.2", ctx, h, emb, num_video_frames);
        }
        return h;
    }

    ggml_tensor* forward_output_block(GGMLRunnerContext* ctx,
                                       int block_idx,
                                       struct ggml_tensor* h,
                                       struct ggml_tensor* skip,
                                       struct ggml_tensor* emb,
                                       struct ggml_tensor* context,
                                       int num_video_frames) {
        h = ggml_concat(ctx->ggml_ctx, h, skip, 2);

        std::string res_name = "output_blocks." + std::to_string(block_idx) + ".0";
        h = resblock_forward(res_name, ctx, h, emb, num_video_frames);

        // output_blocks.X.1/.2 may be SpatialTransformer (attention), UpSampleBlock,
        // or both: when the resolution has attention, slot .1 = transformer and
        // slot .2 = upsample; without attention, slot .1 = upsample. Dispatch
        // by actual block type so SD1.x's deepest output block (no attention)
        // doesn't end up casting an UpSampleBlock to a SpatialTransformer.
        for (int i = 1; i <= 2; i++) {
            std::string slot_name = "output_blocks." + std::to_string(block_idx) + "." + std::to_string(i);
            auto slot_it = blocks.find(slot_name);
            if (slot_it == blocks.end()) {
                continue;
            }
            if (auto upsample = std::dynamic_pointer_cast<UpSampleBlock>(slot_it->second)) {
                h = upsample->forward(ctx, h);
            } else {
                h = attention_layer_forward(slot_name, ctx, h, context, num_video_frames);
            }
        }

        return h;
    }

    ggml_tensor* forward_output_stage(GGMLRunnerContext* ctx, struct ggml_tensor* h) {
        auto out_0 = std::dynamic_pointer_cast<GroupNorm32>(blocks["out.0"]);
        auto out_2 = std::dynamic_pointer_cast<Conv2d>(blocks["out.2"]);

        h = out_0->forward(ctx, h);
        h = ggml_silu_inplace(ctx->ggml_ctx, h);
        h = out_2->forward(ctx, h);

        return h;
    }

    // Walk the blocks map to find the largest "input_blocks.N.0" index that
    // actually exists, then return N+1 so callers can iterate [0, count).
    // SDXL ends at 8 (9 total), SD1/SD2 at 11 (12 total), tiny_unet has gaps
    // — the streaming loop treats missing indices as "skip" via blocks.find().
    int get_num_input_blocks() const {
        return count_blocks_with_prefix("input_blocks.");
    }
    int get_num_output_blocks() const {
        return count_blocks_with_prefix("output_blocks.");
    }

private:
    int count_blocks_with_prefix(const std::string& prefix) const {
        int max_idx = -1;
        for (const auto& kv : blocks) {
            const std::string& name = kv.first;
            if (name.compare(0, prefix.size(), prefix) != 0) {
                continue;
            }
            // name looks like "input_blocks.N.M"; extract N
            size_t i_start = prefix.size();
            size_t i_end = name.find('.', i_start);
            if (i_end == std::string::npos) {
                continue;
            }
            try {
                int idx = std::stoi(name.substr(i_start, i_end - i_start));
                if (idx > max_idx) max_idx = idx;
            } catch (...) {
                continue;
            }
        }
        return max_idx + 1;
    }

public:
};

struct UNetModelRunner : public GGMLRunner {
    UnetModelBlock unet;

    // ---- Streaming state captured across executor stages ----------------
    // Stage 1 produces the timestep+label embedding. The per-layer factory
    // dispatches by phase (initial conv / input block / middle / output)
    // and writes back into the persistent host buffers.
    ggml_tensor* stage1_emb_out_   = nullptr;
    ggml_tensor* layer_h_out_      = nullptr;

    int64_t      h_ne_[4]          = {0, 0, 0, 0};
    int64_t      emb_ne_[4]        = {0, 0, 0, 0};

    // Persistent host buffers backing per-block ggml_backend_tensor_get /
    // set_backend_tensor_data round trips. Pinned via ensure_pinned_act_buffers
    // when possible; fallback to pageable std::vector otherwise.
    float*       persistent_h_     = nullptr;
    float*       persistent_emb_   = nullptr;
    size_t       persistent_h_count_   = 0;
    size_t       persistent_emb_count_ = 0;
    std::vector<float> persistent_h_fallback_;
    std::vector<float> persistent_emb_fallback_;

    // Skip outputs from each input block — sized to num_input_blocks at
    // the top of compute_streaming_true. Each slot has its own shape because
    // downsamples change H/W; track ne[4] per slot. Decoder blocks read by
    // reversed index.
    std::vector<std::vector<float>>      skip_outputs_fallback_;
    std::vector<std::array<int64_t, 4>>  skip_outputs_ne_;

    UNetModelRunner(ggml_backend_t backend,
                    ggml_backend_t params_backend,
                    const String2TensorStorage& tensor_storage_map,
                    const std::string prefix,
                    SDVersion version = VERSION_SD1)
        : GGMLRunner(backend, params_backend), unet(version, tensor_storage_map) {
        unet.init(params_ctx, tensor_storage_map, prefix);
    }

    std::string get_desc() override {
        return "unet";
    }

    void enable_layer_streaming(const LayerStreaming::StreamingConfig& config = {}) {
        std::map<std::string, ggml_tensor*> tensor_map;
        unet.get_param_tensors(tensor_map, "model.diffusion_model");
        init_streaming(config, tensor_map, LayerStreaming::unet_layer_pattern);
        LOG_INFO("%s layer streaming enabled (coarse-stage mode)", get_desc().c_str());
    }

    bool compute_streaming(int n_threads,
                           ggml_tensor* x,
                           ggml_tensor* timesteps,
                           ggml_tensor* context,
                           ggml_tensor* c_concat,
                           ggml_tensor* y,
                           int num_video_frames                      = -1,
                           std::vector<ggml_tensor*> controls = {},
                           float control_strength                    = 0.f,
                           ggml_tensor** output               = nullptr,
                           ggml_context* output_ctx           = nullptr) {
        if (!is_streaming_enabled()) {
            LOG_WARN("%s streaming not enabled, falling back to regular compute", get_desc().c_str());
            return compute(n_threads, x, timesteps, context, c_concat, y,
                           num_video_frames, controls, control_strength, output, output_ctx);
        }

        int64_t t0 = ggml_time_ms();
        auto analysis = analyze_vram_budget();

        if (analysis.fits_in_vram) {
            LOG_INFO("%s model fits in VRAM, using coarse-stage streaming", get_desc().c_str());
            load_all_layers_coarse();
            bool result = compute(n_threads, x, timesteps, context, c_concat, y,
                                  num_video_frames, controls, control_strength, output, output_ctx,
                                  /*skip_param_offload=*/true);
            int64_t t1 = ggml_time_ms();
            LOG_INFO("%s coarse-stage streaming completed in %.2fs", get_desc().c_str(), (t1 - t0) / 1000.0);
            free_compute_buffer();
            return result;
        }

        LOG_INFO("%s remaining %.2f GB exceeds available %.2f GB, using per-layer streaming",
                 get_desc().c_str(),
                 analysis.remaining_to_load / (1024.0 * 1024.0 * 1024.0),
                 analysis.available_vram / (1024.0 * 1024.0 * 1024.0));

        return compute_streaming_true(n_threads, x, timesteps, context, c_concat, y,
                                      num_video_frames, controls, control_strength, output, output_ctx);
    }

    bool compute_streaming_true(int n_threads,
                                ggml_tensor* x,
                                ggml_tensor* timesteps,
                                ggml_tensor* context,
                                ggml_tensor* c_concat             = nullptr,
                                ggml_tensor* y                    = nullptr,
                                int num_video_frames              = -1,
                                std::vector<ggml_tensor*> controls = {},
                                float control_strength            = 0.f,
                                ggml_tensor** output              = nullptr,
                                ggml_context* output_ctx          = nullptr) {
        (void)controls;
        (void)control_strength;

        const int num_input_blocks  = unet.get_num_input_blocks();
        const int num_output_blocks = unet.get_num_output_blocks();
        const int num_layers        = num_input_blocks + 1 + num_output_blocks;

        LOG_INFO("%s: streaming dispatch (%d input + 1 middle + %d output blocks)",
                 get_desc().c_str(), num_input_blocks, num_output_blocks);

        // Pre-size per-input-block skip-output tracking. Shapes vary per
        // slot (downsamples change H/W), so each holds its own ne[4] and
        // pageable buffer. Buffers are reused across sampling steps without
        // shrinking — sized once on the first step.
        if ((int)skip_outputs_fallback_.size() != num_input_blocks) {
            skip_outputs_fallback_.assign(num_input_blocks, {});
        }
        if ((int)skip_outputs_ne_.size() != num_input_blocks) {
            skip_outputs_ne_.assign(num_input_blocks, {0, 0, 0, 0});
        }

        // ---- Stage 1: timestep + label embedding ------------------------
        // _global has been loaded by the executor before Stage 1 runs;
        // time_embed.* / label_emb.* live in _global. The initial conv
        // (input_blocks.0.0) is handled by the per-layer factory at
        // layer_idx == 0 so it can be loaded/evicted with the rest.
        LayerStreaming::Stage input_stage;
        input_stage.build_graph = [this, timesteps, y]() -> ggml_cgraph* {
            ggml_cgraph* gf = new_graph_custom(UNET_GRAPH_SIZE / 8);
            auto runner_ctx = get_context();

            ggml_tensor* timesteps_b = to_backend(timesteps);
            ggml_tensor* y_b         = y ? to_backend(y) : nullptr;

            stage1_emb_out_ = unet.forward_embedding_stage(&runner_ctx, timesteps_b, y_b);
            ggml_build_forward_expand(gf, stage1_emb_out_);
            return gf;
        };
        input_stage.post_compute = [this]() {
            const size_t emb_n = ggml_nelements(stage1_emb_out_);
            persistent_emb_count_ = emb_n;

            std::vector<float*> ptrs;
            if (ensure_pinned_act_buffers({emb_n * sizeof(float)}, ptrs)) {
                persistent_emb_ = ptrs[0];
            } else {
                persistent_emb_fallback_.resize(emb_n);
                persistent_emb_ = persistent_emb_fallback_.data();
            }
            ggml_backend_tensor_get(stage1_emb_out_, persistent_emb_, 0, emb_n * sizeof(float));
            for (int i = 0; i < 4; ++i) emb_ne_[i] = stage1_emb_out_->ne[i];
        };

        // ---- Stage 2: phase-aware per-layer factory ---------------------
        // num_layers = num_input + 1 + num_output. Dispatch:
        //   layer_idx == 0                          → initial conv (input_blocks.0.0)
        //   1 .. num_input_blocks - 1               → forward_input_block(layer_idx)
        //   == num_input_blocks                     → forward_middle_block
        //   > num_input_blocks                      → forward_output_block(output_idx)
        // forward_input_block / forward_output_block already encode the
        // DownSample/UpSample type-dispatch fixes from commit dbd4a35.
        auto layer_factory =
            [this, x, c_concat, context, num_video_frames, num_input_blocks]
            (int layer_idx, ggml_tensor* prev_gpu_output) -> LayerStreaming::Stage {
            (void)prev_gpu_output;  // No chunk-K dispatch for UNet today.
            LayerStreaming::Stage s;

            s.build_graph = [this, layer_idx, x, c_concat, context, num_video_frames, num_input_blocks]() -> ggml_cgraph* {
                ggml_cgraph* gf = new_graph_custom(UNET_GRAPH_SIZE / 8);
                auto runner_ctx = get_context();

                if (layer_idx == 0) {
                    // Initial conv: pulls from CPU input x (+ optional c_concat)
                    // and produces the first 'h' that doubles as skip_outputs_[0].
                    ggml_tensor* x_b = to_backend(x);
                    if (c_concat != nullptr) {
                        ggml_tensor* c_b = to_backend(c_concat);
                        x_b = ggml_concat(compute_ctx, x_b, c_b, 2);
                    }
                    layer_h_out_ = unet.forward_initial_conv(&runner_ctx, x_b);
                    ggml_build_forward_expand(gf, layer_h_out_);
                    return gf;
                }

                // Rebind hidden + embedding from host buffers for every
                // streamed layer ≥ 1. emb is invariant after Stage 1.
                ggml_tensor* h_in = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32,
                                                       h_ne_[0], h_ne_[1], h_ne_[2], h_ne_[3]);
                ggml_tensor* emb_in = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32,
                                                         emb_ne_[0], emb_ne_[1], emb_ne_[2], emb_ne_[3]);
                ggml_tensor* context_b = context ? to_backend(context) : nullptr;

                h_in   = to_backend(h_in);
                emb_in = to_backend(emb_in);

                set_backend_tensor_data(h_in, persistent_h_);
                set_backend_tensor_data(emb_in, persistent_emb_);

                if (layer_idx < num_input_blocks) {
                    // INPUT phase. forward_input_block dispatches by actual
                    // block type (DownSampleBlock vs ResBlock) at slot .0
                    // and SpatialTransformer at slot .1 — the dbd4a35 fix.
                    layer_h_out_ = unet.forward_input_block(&runner_ctx, layer_idx,
                                                             h_in, emb_in,
                                                             context_b, num_video_frames);
                } else if (layer_idx == num_input_blocks) {
                    // MIDDLE phase.
                    layer_h_out_ = unet.forward_middle_block(&runner_ctx,
                                                              h_in, emb_in,
                                                              context_b, num_video_frames);
                } else {
                    // OUTPUT phase. Read the matching skip output from host
                    // (reverse-indexed); forward_output_block already encodes
                    // the UpSampleBlock vs SpatialTransformer dispatch fix
                    // from dbd4a35.
                    int output_idx = layer_idx - num_input_blocks - 1;
                    int skip_idx   = num_input_blocks - 1 - output_idx;

                    ggml_tensor* skip_in = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32,
                                                              skip_outputs_ne_[skip_idx][0],
                                                              skip_outputs_ne_[skip_idx][1],
                                                              skip_outputs_ne_[skip_idx][2],
                                                              skip_outputs_ne_[skip_idx][3]);
                    skip_in = to_backend(skip_in);
                    set_backend_tensor_data(skip_in, skip_outputs_fallback_[skip_idx].data());

                    layer_h_out_ = unet.forward_output_block(&runner_ctx, output_idx,
                                                              h_in, skip_in, emb_in,
                                                              context_b, num_video_frames);
                }

                ggml_build_forward_expand(gf, layer_h_out_);
                return gf;
            };

            s.post_compute = [this, layer_idx, num_input_blocks]() {
                const size_t bytes  = ggml_nbytes(layer_h_out_);
                const size_t nelems = ggml_nelements(layer_h_out_);

                if (persistent_h_count_ < nelems || persistent_h_ == nullptr) {
                    std::vector<float*> ptrs;
                    if (ensure_pinned_act_buffers({nelems * sizeof(float)}, ptrs)) {
                        persistent_h_ = ptrs[0];
                    } else {
                        persistent_h_fallback_.resize(nelems);
                        persistent_h_ = persistent_h_fallback_.data();
                    }
                    persistent_h_count_ = nelems;
                }

                ggml_backend_tensor_get(layer_h_out_, persistent_h_, 0, bytes);
                for (int i = 0; i < 4; ++i) h_ne_[i] = layer_h_out_->ne[i];

                // If this was an input block (including initial conv at
                // layer 0), persist the result as a skip connection for the
                // matching decoder block to read later.
                if (layer_idx < num_input_blocks) {
                    if (skip_outputs_fallback_[layer_idx].size() < nelems) {
                        skip_outputs_fallback_[layer_idx].resize(nelems);
                    }
                    std::memcpy(skip_outputs_fallback_[layer_idx].data(),
                                persistent_h_,
                                bytes);
                    for (int i = 0; i < 4; ++i) {
                        skip_outputs_ne_[layer_idx][i] = layer_h_out_->ne[i];
                    }
                }
            };

            return s;
        };

        // ---- Stage 3: out_norm + silu + out_conv ------------------------
        LayerStreaming::Stage output_stage;
        output_stage.build_graph = [this]() -> ggml_cgraph* {
            ggml_cgraph* gf = new_graph_custom(UNET_GRAPH_SIZE / 8);
            auto runner_ctx = get_context();

            ggml_tensor* h_in = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32,
                                                    h_ne_[0], h_ne_[1], h_ne_[2], h_ne_[3]);
            h_in = to_backend(h_in);
            set_backend_tensor_data(h_in, persistent_h_);

            auto final_out = unet.forward_output_stage(&runner_ctx, h_in);
            ggml_build_forward_expand(gf, final_out);
            return gf;
        };

        return LayerStreaming::run_streaming(
            this, n_threads, streaming_engine_->get_config(),
            input_stage, layer_factory, output_stage,
            num_layers,
            [num_input_blocks](int i) -> std::string {
                if (i < num_input_blocks) return "input_blocks." + std::to_string(i);
                if (i == num_input_blocks) return std::string("middle_block");
                return "output_blocks." + std::to_string(i - num_input_blocks - 1);
            },
            /*start_layer_idx=*/0,
            output, output_ctx);
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
        unet.get_param_tensors(tensors, prefix);
    }

    ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                             const sd::Tensor<float>& timesteps_tensor,
                             const sd::Tensor<float>& context_tensor               = {},
                             const sd::Tensor<float>& c_concat_tensor              = {},
                             const sd::Tensor<float>& y_tensor                     = {},
                             int num_video_frames                                  = -1,
                             const std::vector<sd::Tensor<float>>& controls_tensor = {},
                             float control_strength                                = 0.f) {
        ggml_cgraph* gf = new_graph_custom(UNET_GRAPH_SIZE);

        ggml_tensor* x         = make_input(x_tensor);
        ggml_tensor* timesteps = make_input(timesteps_tensor);
        ggml_tensor* context   = make_optional_input(context_tensor);
        ggml_tensor* c_concat  = make_optional_input(c_concat_tensor);
        ggml_tensor* y         = make_optional_input(y_tensor);
        std::vector<ggml_tensor*> controls;
        controls.reserve(controls_tensor.size());
        for (const auto& control_tensor : controls_tensor) {
            controls.push_back(make_input(control_tensor));
        }

        if (num_video_frames == -1) {
            num_video_frames = static_cast<int>(x->ne[3]);
        }

        auto runner_ctx = get_context();

        ggml_tensor* out = unet.forward(&runner_ctx,
                                        x,
                                        timesteps,
                                        context,
                                        c_concat,
                                        y,
                                        num_video_frames,
                                        controls,
                                        control_strength);

        ggml_build_forward_expand(gf, out);

        return gf;
    }

    // Legacy overload used by streaming code paths (takes raw ggml_tensor pointers)
    ggml_cgraph* build_graph(ggml_tensor* x,
                             ggml_tensor* timesteps,
                             ggml_tensor* context,
                             ggml_tensor* c_concat              = nullptr,
                             ggml_tensor* y                     = nullptr,
                             int num_video_frames               = -1,
                             std::vector<ggml_tensor*> controls = {},
                             float control_strength             = 0.f) {
        ggml_cgraph* gf = new_graph_custom(UNET_GRAPH_SIZE);

        if (num_video_frames == -1) {
            num_video_frames = static_cast<int>(x->ne[3]);
        }

        x         = to_backend(x);
        context   = to_backend(context);
        y         = to_backend(y);
        timesteps = to_backend(timesteps);
        c_concat  = to_backend(c_concat);

        for (size_t i = 0; i < controls.size(); i++) {
            controls[i] = to_backend(controls[i]);
        }

        auto runner_ctx = get_context();

        ggml_tensor* out = unet.forward(&runner_ctx,
                                        x,
                                        timesteps,
                                        context,
                                        c_concat,
                                        y,
                                        num_video_frames,
                                        controls,
                                        control_strength);

        ggml_build_forward_expand(gf, out);

        return gf;
    }

    // Legacy overload used by streaming code paths (takes raw ggml_tensor pointers)
    bool compute(int n_threads,
                 ggml_tensor* x,
                 ggml_tensor* timesteps,
                 ggml_tensor* context,
                 ggml_tensor* c_concat,
                 ggml_tensor* y,
                 int num_video_frames               = -1,
                 std::vector<ggml_tensor*> controls = {},
                 float control_strength             = 0.f,
                 ggml_tensor** output               = nullptr,
                 ggml_context* output_ctx           = nullptr,
                 bool skip_param_offload            = false) {
        auto get_graph = [&]() -> ggml_cgraph* {
            return build_graph(x, timesteps, context, c_concat, y, num_video_frames, controls, control_strength);
        };

        return GGMLRunner::compute(get_graph, n_threads, false, output, output_ctx, skip_param_offload);
    }

    // Upstream public API (takes sd::Tensor)
    sd::Tensor<float> compute(int n_threads,
                              const sd::Tensor<float>& x,
                              const sd::Tensor<float>& timesteps,
                              const sd::Tensor<float>& context               = {},
                              const sd::Tensor<float>& c_concat              = {},
                              const sd::Tensor<float>& y                     = {},
                              int num_video_frames                           = -1,
                              const std::vector<sd::Tensor<float>>& controls = {},
                              float control_strength                         = 0.f) {
        // x: [N, in_channels, h, w]
        // timesteps: [N, ]
        // context: [N, max_position, hidden_size]([N, 77, 768]) or [1, max_position, hidden_size]
        // c_concat: [N, in_channels, h, w] or [1, in_channels, h, w]
        // y: [N, adm_in_channels] or [1, adm_in_channels]
        auto get_graph = [&]() -> ggml_cgraph* {
            return build_graph(x, timesteps, context, c_concat, y, num_video_frames, controls, control_strength);
        };

        return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false), x.dim());
    }

    void test() {
        ggml_init_params params;
        params.mem_size   = static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
        params.mem_buffer = nullptr;
        params.no_alloc   = false;

        ggml_context* ctx = ggml_init(params);
        GGML_ASSERT(ctx != nullptr);

        {
            // CPU, num_video_frames = 1, x{num_video_frames, 8, 8, 8}: Pass
            // CUDA, num_video_frames = 1, x{num_video_frames, 8, 8, 8}: Pass
            // CPU, num_video_frames = 3, x{num_video_frames, 8, 8, 8}: Wrong result
            // CUDA, num_video_frames = 3, x{num_video_frames, 8, 8, 8}: nan
            int num_video_frames = 3;

            sd::Tensor<float> x({8, 8, 8, num_video_frames});
            std::vector<float> timesteps_vec(num_video_frames, 999.f);
            auto timesteps = sd::Tensor<float>::from_vector(timesteps_vec);
            x.fill_(0.5f);
            // print_ggml_tensor(x);

            sd::Tensor<float> context({1024, 1, num_video_frames});
            context.fill_(0.5f);
            // print_ggml_tensor(context);

            sd::Tensor<float> y({768, num_video_frames});
            y.fill_(0.5f);
            // print_ggml_tensor(y);

            sd::Tensor<float> out;

            int64_t t0   = ggml_time_ms();
            auto out_opt = compute(8,
                                   x,
                                   timesteps,
                                   context,
                                   {},
                                   y,
                                   num_video_frames,
                                   {},
                                   0.f);
            int64_t t1   = ggml_time_ms();

            GGML_ASSERT(!out_opt.empty());
            out = std::move(out_opt);
            print_sd_tensor(out);
            LOG_DEBUG("unet test done in %lldms", t1 - t0);
        }
    }
};

#endif  // __UNET_HPP__
