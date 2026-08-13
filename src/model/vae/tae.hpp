#ifndef __SD_MODEL_VAE_TAE_HPP__
#define __SD_MODEL_VAE_TAE_HPP__

#include "core/ggml_extend.hpp"
#include "model.h"

/*
    ===================================    TinyAutoEncoder  ===================================
    References:
    https://github.com/huggingface/diffusers/blob/main/src/diffusers/model/autoencoders/vae.py
    https://github.com/madebyollin/taesd/blob/main/taesd.py

*/

class TAEBlock : public UnaryBlock {
protected:
    int n_in;
    int n_out;
    bool use_midblock_gn;

public:
    TAEBlock(int n_in, int n_out, bool use_midblock_gn = false)
        : n_in(n_in), n_out(n_out), use_midblock_gn(use_midblock_gn) {
        blocks["conv.0"] = std::shared_ptr<GGMLBlock>(new Conv2d(n_in, n_out, {3, 3}, {1, 1}, {1, 1}));
        blocks["conv.2"] = std::shared_ptr<GGMLBlock>(new Conv2d(n_out, n_out, {3, 3}, {1, 1}, {1, 1}));
        blocks["conv.4"] = std::shared_ptr<GGMLBlock>(new Conv2d(n_out, n_out, {3, 3}, {1, 1}, {1, 1}));
        if (n_in != n_out) {
            blocks["skip"] = std::shared_ptr<GGMLBlock>(new Conv2d(n_in, n_out, {1, 1}, {1, 1}, {1, 1}, {1, 1}, false));
        }
        if (use_midblock_gn) {
            int n_gn         = n_in * 4;
            blocks["pool.0"] = std::shared_ptr<GGMLBlock>(new Conv2d(n_in, n_gn, {1, 1}, {1, 1}, {0, 0}, {1, 1}, false));
            blocks["pool.1"] = std::shared_ptr<GGMLBlock>(new GroupNorm(4, n_gn));
            // pool.2 is ReLU, handled in forward
            blocks["pool.3"] = std::shared_ptr<GGMLBlock>(new Conv2d(n_gn, n_in, {1, 1}, {1, 1}, {0, 0}, {1, 1}, false));
        }
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        // x: [n, n_in, h, w]
        // return: [n, n_out, h, w]

        if (use_midblock_gn) {
            auto pool_0 = std::dynamic_pointer_cast<Conv2d>(blocks["pool.0"]);
            auto pool_1 = std::dynamic_pointer_cast<GroupNorm>(blocks["pool.1"]);
            auto pool_3 = std::dynamic_pointer_cast<Conv2d>(blocks["pool.3"]);

            auto p = pool_0->forward(ctx, x);
            p      = pool_1->forward(ctx, p);
            p      = ggml_relu_inplace(ctx->ggml_ctx, p);
            p      = pool_3->forward(ctx, p);

            x = ggml_add(ctx->ggml_ctx, x, p);
        }

        auto conv_0 = std::dynamic_pointer_cast<Conv2d>(blocks["conv.0"]);
        auto conv_2 = std::dynamic_pointer_cast<Conv2d>(blocks["conv.2"]);
        auto conv_4 = std::dynamic_pointer_cast<Conv2d>(blocks["conv.4"]);

        auto h = conv_0->forward(ctx, x);
        h      = ggml_relu_inplace(ctx->ggml_ctx, h);
        h      = conv_2->forward(ctx, h);
        h      = ggml_relu_inplace(ctx->ggml_ctx, h);
        h      = conv_4->forward(ctx, h);

        if (n_in != n_out) {
            auto skip = std::dynamic_pointer_cast<Conv2d>(blocks["skip"]);
            LOG_DEBUG("skip");
            x = skip->forward(ctx, x);
        }

        h = ggml_add(ctx->ggml_ctx, h, x);
        h = ggml_relu_inplace(ctx->ggml_ctx, h);
        return h;
    }
};

class TinyEncoder : public UnaryBlock {
    int in_channels = 3;
    int channels    = 64;
    int z_channels  = 4;
    int num_blocks  = 3;

public:
    TinyEncoder(int z_channels = 4, bool use_midblock_gn = false)
        : z_channels(z_channels) {
        int index                       = 0;
        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, channels, {3, 3}, {1, 1}, {1, 1}));
        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new TAEBlock(channels, channels));

        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, channels, {3, 3}, {2, 2}, {1, 1}, {1, 1}, false));
        for (int i = 0; i < num_blocks; i++) {
            blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new TAEBlock(channels, channels));
        }

        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, channels, {3, 3}, {2, 2}, {1, 1}, {1, 1}, false));
        for (int i = 0; i < num_blocks; i++) {
            blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new TAEBlock(channels, channels));
        }

        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, channels, {3, 3}, {2, 2}, {1, 1}, {1, 1}, false));
        for (int i = 0; i < num_blocks; i++) {
            blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new TAEBlock(channels, channels, use_midblock_gn));
        }

        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, z_channels, {3, 3}, {1, 1}, {1, 1}));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        // x: [n, in_channels, h, w]
        // return: [n, z_channels, h/8, w/8]

        for (int i = 0; i < num_blocks * 3 + 6; i++) {
            auto block = std::dynamic_pointer_cast<UnaryBlock>(blocks[std::to_string(i)]);

            x = block->forward(ctx, x);
        }

        return x;
    }
};

class TinyDecoder : public UnaryBlock {
    int z_channels   = 4;
    int channels     = 64;
    int out_channels = 3;
    int num_blocks   = 3;

public:
    TinyDecoder(int z_channels = 4, bool use_midblock_gn = false)
        : z_channels(z_channels) {
        int index = 0;

        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(z_channels, channels, {3, 3}, {1, 1}, {1, 1}));
        index++;  // nn.ReLU()

        for (int i = 0; i < num_blocks; i++) {
            blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new TAEBlock(channels, channels, use_midblock_gn));
        }
        index++;  // nn.Upsample()
        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, channels, {3, 3}, {1, 1}, {1, 1}, {1, 1}, false));

        for (int i = 0; i < num_blocks; i++) {
            blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new TAEBlock(channels, channels));
        }
        index++;  // nn.Upsample()
        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, channels, {3, 3}, {1, 1}, {1, 1}, {1, 1}, false));

        for (int i = 0; i < num_blocks; i++) {
            blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new TAEBlock(channels, channels));
        }
        index++;  // nn.Upsample()
        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, channels, {3, 3}, {1, 1}, {1, 1}, {1, 1}, false));

        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new TAEBlock(channels, channels));
        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, out_channels, {3, 3}, {1, 1}, {1, 1}));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* z) override {
        // z: [n, z_channels, h, w]
        // return: [n, out_channels, h*8, w*8]

        auto h = ggml_ext_scale(ctx->ggml_ctx, z, 1.0f / 3.0f);
        h      = ggml_tanh_inplace(ctx->ggml_ctx, h);
        h      = ggml_ext_scale(ctx->ggml_ctx, h, 3.0f);

        for (int i = 0; i < num_blocks * 3 + 10; i++) {
            if (blocks.find(std::to_string(i)) == blocks.end()) {
                if (i == 1) {
                    h = ggml_relu_inplace(ctx->ggml_ctx, h);
                } else {
                    h = ggml_upscale(ctx->ggml_ctx, h, 2, GGML_SCALE_MODE_NEAREST);
                }
                continue;
            }
            auto block = std::dynamic_pointer_cast<UnaryBlock>(blocks[std::to_string(i)]);

            h = block->forward(ctx, h);
        }

        return h;
    }
};

class TPool : public UnaryBlock {
public:
    int stride;
    TPool(int channels, int stride)
        : stride(stride) {
        blocks["conv"] = std::shared_ptr<GGMLBlock>(new Conv2d(channels * stride, channels, {1, 1}, {1, 1}, {0, 0}, {1, 1}, false));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        auto conv = std::dynamic_pointer_cast<UnaryBlock>(blocks["conv"]);
        auto h    = x;
        if (stride != 1) {
            h = ggml_reshape_4d(ctx->ggml_ctx, h, h->ne[0], h->ne[1], h->ne[2] * stride, h->ne[3] / stride);
        }
        h = conv->forward(ctx, h);
        return h;
    }
};

class TGrow : public UnaryBlock {
public:
    int stride;
    TGrow(int channels, int stride)
        : stride(stride) {
        blocks["conv"] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, channels * stride, {1, 1}, {1, 1}, {0, 0}, {1, 1}, false));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        auto conv = std::dynamic_pointer_cast<UnaryBlock>(blocks["conv"]);
        auto h    = conv->forward(ctx, x);
        if (stride != 1) {
            h = ggml_reshape_4d(ctx->ggml_ctx, h, h->ne[0], h->ne[1], h->ne[2] / stride, h->ne[3] * stride);
        }
        return h;
    }
};

class MemBlock : public GGMLBlock {
    bool has_skip_conv = false;

public:
    MemBlock(int channels, int out_channels)
        : has_skip_conv(channels != out_channels) {
        blocks["conv.0"] = std::shared_ptr<GGMLBlock>(new Conv2d(channels * 2, out_channels, {3, 3}, {1, 1}, {1, 1}));
        blocks["conv.2"] = std::shared_ptr<GGMLBlock>(new Conv2d(out_channels, out_channels, {3, 3}, {1, 1}, {1, 1}));
        blocks["conv.4"] = std::shared_ptr<GGMLBlock>(new Conv2d(out_channels, out_channels, {3, 3}, {1, 1}, {1, 1}));
        if (has_skip_conv) {
            blocks["skip"] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, out_channels, {1, 1}, {1, 1}, {0, 0}, {1, 1}, false));
        }
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* past) {
        // x: [n, channels, h, w]
        auto conv0 = std::dynamic_pointer_cast<Conv2d>(blocks["conv.0"]);
        auto conv1 = std::dynamic_pointer_cast<Conv2d>(blocks["conv.2"]);
        auto conv2 = std::dynamic_pointer_cast<Conv2d>(blocks["conv.4"]);

        auto h = ggml_concat(ctx->ggml_ctx, x, past, 2);
        h      = conv0->forward(ctx, h);
        h      = ggml_relu_inplace(ctx->ggml_ctx, h);
        h      = conv1->forward(ctx, h);
        h      = ggml_relu_inplace(ctx->ggml_ctx, h);
        h      = conv2->forward(ctx, h);

        auto skip = x;
        if (has_skip_conv) {
            auto skip_conv = std::dynamic_pointer_cast<Conv2d>(blocks["skip"]);
            skip           = skip_conv->forward(ctx, x);
        }
        h = ggml_add_inplace(ctx->ggml_ctx, h, skip);
        h = ggml_relu_inplace(ctx->ggml_ctx, h);
        return h;
    }
};

class WideMemBlock : public GGMLBlock {
    bool has_skip_conv = false;

public:
    WideMemBlock(int channels, int out_channels)
        : has_skip_conv(channels != out_channels) {
        int groups       = std::max(1, out_channels / 64);
        blocks["conv.0"] = std::shared_ptr<GGMLBlock>(new Conv2d(channels * 2, out_channels, {1, 1}, {1, 1}));
        blocks["conv.2"] = std::shared_ptr<GGMLBlock>(new Conv2d_grouped(out_channels, out_channels, groups, {3, 3}, {1, 1}, {1, 1}));
        blocks["conv.4"] = std::shared_ptr<GGMLBlock>(new Conv2d(out_channels, out_channels, {1, 1}, {1, 1}));
        blocks["conv.6"] = std::shared_ptr<GGMLBlock>(new Conv2d_grouped(out_channels, out_channels, groups, {3, 3}, {1, 1}, {1, 1}));
        if (has_skip_conv) {
            blocks["skip"] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, out_channels, {1, 1}, {1, 1}, {0, 0}, {1, 1}, false));
        }
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* past) {
        // x: [n, channels, h, w]
        auto conv0 = std::dynamic_pointer_cast<Conv2d>(blocks["conv.0"]);
        auto conv1 = std::dynamic_pointer_cast<Conv2d_grouped>(blocks["conv.2"]);
        auto conv2 = std::dynamic_pointer_cast<Conv2d>(blocks["conv.4"]);
        auto conv3 = std::dynamic_pointer_cast<Conv2d_grouped>(blocks["conv.6"]);

        auto h = ggml_concat(ctx->ggml_ctx, x, past, 2);
        h      = conv0->forward(ctx, h);
        h      = ggml_relu_inplace(ctx->ggml_ctx, h);
        h      = conv1->forward(ctx, h);
        h      = ggml_relu_inplace(ctx->ggml_ctx, h);
        h      = conv2->forward(ctx, h);
        h      = ggml_relu_inplace(ctx->ggml_ctx, h);
        h      = conv3->forward(ctx, h);

        auto skip = x;
        if (has_skip_conv) {
            auto skip_conv = std::dynamic_pointer_cast<Conv2d>(blocks["skip"]);
            skip           = skip_conv->forward(ctx, x);
        }
        h = ggml_add_inplace(ctx->ggml_ctx, h, skip);
        h = ggml_relu_inplace(ctx->ggml_ctx, h);
        return h;
    }
};

ggml_tensor*
patchify(ggml_context* ctx,
         ggml_tensor* x,
         int64_t patch_size,
         int64_t b = 1) {
    // x: [f, b*c, h*q, w*r]
    // return: [f, b*c*r*q, h, w]
    if (patch_size == 1) {
        return x;
    }
    int64_t r = patch_size;
    int64_t q = patch_size;

    int64_t W = x->ne[0];
    int64_t H = x->ne[1];
    int64_t C = x->ne[2];
    int64_t f = x->ne[3];

    int64_t w = W / r;
    int64_t h = H / q;

    x = ggml_reshape_4d(ctx, x, W, q, h, C * f);                         // [W, q, h, C*f]
    x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));  // [W, h, q, C*f]
    x = ggml_reshape_4d(ctx, x, r, w, h, q * C * f);                     // [r, w, h, q*C*f]
    x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 1, 2, 0, 3));  // [w, h, r, q*C*f]
    x = ggml_reshape_4d(ctx, x, w, h, r * q * C, f);                     // [f, b*c*r*q, h, w]

    return x;
}

ggml_tensor* unpatchify(ggml_context* ctx,
                        ggml_tensor* x,
                        int64_t patch_size,
                        int64_t b = 1) {
    // x: [f, b*c*r*q, h, w]
    // return: [f, b*c, h*q, w*r]
    if (patch_size == 1) {
        return x;
    }
    int64_t r = patch_size;
    int64_t q = patch_size;
    int64_t c = x->ne[2] / b / q / r;
    int64_t f = x->ne[3];
    int64_t h = x->ne[1];
    int64_t w = x->ne[0];

    x = ggml_reshape_4d(ctx, x, w, h, r, q * c * b * f);                 // [q*c*b*f, r, h, w]
    x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 2, 0, 1, 3));  // [r, w, h, q*c*b*f]
    x = ggml_reshape_4d(ctx, x, r * w, h, q, c * b * f);                 // [c*b*f, q, h, r*w]
    x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));  // [r*w, q, h, c*b*f]
    x = ggml_reshape_4d(ctx, x, r * w, q * h, c * b, f);

    return x;
}

struct WorkItem {
    ggml_tensor* xt;
    int block_idx;
};

struct SequentialDecoderState {
    std::map<int, ggml_tensor*> mem_single;
};

struct SequentialEncoderState {
    std::map<int, ggml_tensor*> mem_single;
    std::map<int, std::vector<ggml_tensor*>> mem_pool;
};

class TinyVideoEncoder : public UnaryBlock {
    int in_channels = 3;
    int hidden      = 64;
    int z_channels  = 4;
    int num_blocks  = 3;
    int num_layers  = 3;
    int patch_size  = 1;

    int total_blocks = 0;
    int relu_idx     = 0;

public:
    int t_downscale = 1;
    TinyVideoEncoder(int z_channels = 4, int patch_size = 1, std::vector<bool> time_downscale = {true, true, false})
        : z_channels(z_channels), patch_size(patch_size) {
        t_downscale = 1;
        for (bool downscale : time_downscale) {
            if (downscale) {
                t_downscale *= 2;
            }
        }
        int index                       = 0;
        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels * patch_size * patch_size, hidden, {3, 3}, {1, 1}, {1, 1}));
        relu_idx                        = index++;  // nn.ReLU()
        for (int i = 0; i < num_layers; i++) {
            int stride                      = time_downscale[i] ? 2 : 1;
            blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new TPool(hidden, stride));
            blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(hidden, hidden, {3, 3}, {2, 2}, {1, 1}, {1, 1}, false));
            for (int j = 0; j < num_blocks; j++) {
                blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new MemBlock(hidden, hidden));
            }
        }
        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(hidden, z_channels, {3, 3}, {1, 1}, {1, 1}));
        total_blocks                    = index;
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* z) override {
        auto first_conv = std::dynamic_pointer_cast<Conv2d>(blocks["0"]);

        if (patch_size > 1) {
            z = patchify(ctx->ggml_ctx, z, patch_size, 1);
        }

        auto h = first_conv->forward(ctx, z);
        h      = ggml_relu_inplace(ctx->ggml_ctx, h);

        int index = 2;
        for (int i = 0; i < num_layers; i++) {
            auto pool = std::dynamic_pointer_cast<UnaryBlock>(blocks[std::to_string(index++)]);
            auto conv = std::dynamic_pointer_cast<UnaryBlock>(blocks[std::to_string(index++)]);

            h = pool->forward(ctx, h);
            h = conv->forward(ctx, h);
            for (int j = 0; j < num_blocks; j++) {
                auto block = std::dynamic_pointer_cast<MemBlock>(blocks[std::to_string(index++)]);
                auto mem   = ggml_ext_pad_ext(ctx->ggml_ctx, ctx->backend, h, 0, 0, 0, 0, 0, 0, 1, 0);
                mem        = ggml_view_4d(ctx->ggml_ctx, mem, h->ne[0], h->ne[1], h->ne[2], h->ne[3], h->nb[1], h->nb[2], h->nb[3], 0);
                h          = block->forward(ctx, h, mem);
            }
        }
        auto last_conv = std::dynamic_pointer_cast<Conv2d>(blocks[std::to_string(index)]);
        h              = last_conv->forward(ctx, h);

        return h;
    }

    ggml_tensor* forward_seq_single_step(GGMLRunnerContext* ctx,
                                         SequentialEncoderState& state,
                                         std::vector<WorkItem>& work_stack) {
        while (!work_stack.empty()) {
            WorkItem item = work_stack.back();
            work_stack.pop_back();

            ggml_tensor* xt = item.xt;
            int i           = item.block_idx;

            if (i >= total_blocks) {
                if (patch_size > 1) {
                    xt = unpatchify(ctx->ggml_ctx, xt, patch_size, 1);
                }
                return xt;
            }

            if (i == relu_idx) {
                xt = ggml_relu_inplace(ctx->ggml_ctx, xt);
                work_stack.push_back({xt, i + 1});
                continue;
            }

            std::string key = std::to_string(i);
            auto block      = blocks[key];

            if (auto mem_block = std::dynamic_pointer_cast<MemBlock>(block)) {
                ggml_tensor* prev_mem = state.mem_single[i];
                if (prev_mem == nullptr) {
                    prev_mem = ggml_dup_tensor(ctx->ggml_ctx, xt);
                    prev_mem = ggml_scale(ctx->ggml_ctx, prev_mem, 0.);
                }
                ggml_tensor* xt_next = mem_block->forward(ctx, xt, prev_mem);
                state.mem_single[i]  = xt;

                work_stack.push_back({xt_next, i + 1});
            } else if (auto pool = std::dynamic_pointer_cast<TPool>(block)) {
                state.mem_pool[i].push_back(xt);

                if ((int)state.mem_pool[i].size() == pool->stride) {
                    ggml_tensor* cat_input = ggml_ext_vec_concat(ctx->ggml_ctx, state.mem_pool[i], 3);
                    ggml_tensor* xt_next   = pool->forward(ctx, cat_input);

                    state.mem_pool[i].clear();
                    work_stack.push_back({xt_next, i + 1});
                }
            } else if (auto unary_block = std::dynamic_pointer_cast<UnaryBlock>(block)) {
                ggml_tensor* xt_next = unary_block->forward(ctx, xt);
                work_stack.push_back({xt_next, i + 1});
            }
        }

        return nullptr;  // Work stack exhausted
    }

    ggml_tensor* forward_seq(GGMLRunnerContext* ctx,
                             ggml_tensor* z) {
        SequentialEncoderState state;
        std::vector<WorkItem> work_stack;

        if (patch_size > 1) {
            z = patchify(ctx->ggml_ctx, z, patch_size, 1);
        }

        const std::vector<ggml_tensor*>& latent_frames = ggml_ext_chunk(ctx->ggml_ctx, z, z->ne[3], 3);

        for (auto it = latent_frames.rbegin(); it != latent_frames.rend(); ++it) {
            work_stack.push_back({*it, 0});
        }

        std::vector<ggml_tensor*> output_frames;

        while (!work_stack.empty()) {
            ggml_tensor* out_frame = forward_seq_single_step(ctx, state, work_stack);
            if (out_frame != nullptr) {
                output_frames.push_back(out_frame);
            }
        }

        auto h = ggml_ext_vec_concat(ctx->ggml_ctx, output_frames, 3);
        return h;
    }
};



class TinyVideoDecoder : public UnaryBlock {
    int z_channels               = 4;
    int out_channels             = 3;
    int num_blocks               = 3;
    static const int num_layers  = 3;
    int channels[num_layers + 1] = {256, 128, 64, 64};
    int patch_size               = 1;
    int t_upscale                = 1;
    bool is_wide                 = false;

    int total_blocks   = 0;
    int clamp_idx      = 0;
    int relu1_idx      = 0;
    int relu_final_idx = 0;
    std::set<int> upsample_indices;

public:
    TinyVideoDecoder(int z_channels = 4, int patch_size = 1, std::vector<bool> time_upscale = {false, true, true}, bool is_wide = false)
        : z_channels(z_channels), patch_size(patch_size), is_wide(is_wide) {
        t_upscale = 1;
        if (is_wide) {
            channels[0] = 1024;
            channels[1] = 512;
            channels[2] = 256;
        }

        for (bool upscale : time_upscale) {
            if (upscale) {
                t_upscale *= 2;
            }
        }
        clamp_idx                       = 0;
        int index                       = 1;  // Clamp()
        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(z_channels, channels[0], {3, 3}, {1, 1}, {1, 1}));
        relu1_idx                       = index++;  // nn.ReLU()
        for (int i = 0; i < num_layers; i++) {
            int stride = time_upscale[i] ? 2 : 1;
            for (int j = 0; j < num_blocks; j++) {
                if (is_wide) {
                    blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new WideMemBlock(channels[i], channels[i]));
                } else {
                    blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new MemBlock(channels[i], channels[i]));
                }
            }
            upsample_indices.insert(index++);  // Nearest-neighbor spatial upsample slot
            blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new TGrow(channels[i], stride));
            blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(channels[i], channels[i + 1], {3, 3}, {1, 1}, {1, 1}, {1, 1}, false));
        }
        relu_final_idx                  = index++;  // nn.ReLU()
        blocks[std::to_string(index++)] = std::shared_ptr<GGMLBlock>(new Conv2d(channels[num_layers], out_channels * patch_size * patch_size, {3, 3}, {1, 1}, {1, 1}));

        total_blocks = index;
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* z) override {
        auto first_conv = std::dynamic_pointer_cast<Conv2d>(blocks["1"]);

        // Clamp()
        auto h = ggml_ext_scale(ctx->ggml_ctx,
                                ggml_tanh_inplace(ctx->ggml_ctx,
                                                  ggml_ext_scale(ctx->ggml_ctx, z, 1.0f / 3.0f)),
                                3.0f,
                                true);

        h         = first_conv->forward(ctx, h);
        h         = ggml_relu_inplace(ctx->ggml_ctx, h);
        int index = 3;
        for (int i = 0; i < num_layers; i++) {
            for (int j = 0; j < num_blocks; j++) {
                auto mem = ggml_ext_pad_ext(ctx->ggml_ctx, ctx->backend, h, 0, 0, 0, 0, 0, 0, 1, 0);
                mem      = ggml_view_4d(ctx->ggml_ctx, mem, h->ne[0], h->ne[1], h->ne[2], h->ne[3], h->nb[1], h->nb[2], h->nb[3], 0);
                if (is_wide) {
                    auto block = std::dynamic_pointer_cast<WideMemBlock>(blocks[std::to_string(index++)]);
                    h          = block->forward(ctx, h, mem);
                } else {
                    auto block = std::dynamic_pointer_cast<MemBlock>(blocks[std::to_string(index++)]);
                    h          = block->forward(ctx, h, mem);
                }
            }
            // upsample
            index++;
            h          = ggml_upscale(ctx->ggml_ctx, h, 2, GGML_SCALE_MODE_NEAREST);
            auto block = std::dynamic_pointer_cast<UnaryBlock>(blocks[std::to_string(index++)]);
            h          = block->forward(ctx, h);
            block      = std::dynamic_pointer_cast<UnaryBlock>(blocks[std::to_string(index++)]);
            h          = block->forward(ctx, h);
        }
        h = ggml_relu_inplace(ctx->ggml_ctx, h);

        auto last_conv = std::dynamic_pointer_cast<Conv2d>(blocks[std::to_string(++index)]);
        h              = last_conv->forward(ctx, h);
        if (patch_size > 1) {
            h = unpatchify(ctx->ggml_ctx, h, patch_size, 1);
        }
        // shape(W, H, 3, (t_upscale - 1) + T) => shape(W, H, 3, T)
        h = ggml_view_4d(ctx->ggml_ctx, h, h->ne[0], h->ne[1], h->ne[2], h->ne[3] - (t_upscale - 1), h->nb[1], h->nb[2], h->nb[3], (t_upscale - 1) * h->nb[3]);
        return h;
    }

    ggml_tensor* forward_seq_single_step(GGMLRunnerContext* ctx,
                                         SequentialDecoderState& state,
                                         std::vector<WorkItem>& work_stack) {
        while (!work_stack.empty()) {
            WorkItem item = work_stack.back();
            work_stack.pop_back();

            ggml_tensor* xt = item.xt;
            int i           = item.block_idx;

            if (i >= total_blocks) {
                if (patch_size > 1) {
                    xt = unpatchify(ctx->ggml_ctx, xt, patch_size, 1);
                }
                return xt;
            }

            if (i == clamp_idx) {
                xt = ggml_ext_scale(ctx->ggml_ctx,
                                    ggml_tanh_inplace(ctx->ggml_ctx,
                                                      ggml_ext_scale(ctx->ggml_ctx, xt, 1.0f / 3.0f)),
                                    3.0f, true);
                work_stack.push_back({xt, i + 1});
                continue;
            }
            if (i == relu1_idx || i == relu_final_idx) {
                xt = ggml_relu_inplace(ctx->ggml_ctx, xt);
                work_stack.push_back({xt, i + 1});
                continue;
            }
            if (upsample_indices.count(i)) {
                xt = ggml_upscale(ctx->ggml_ctx, xt, 2, GGML_SCALE_MODE_NEAREST);
                work_stack.push_back({xt, i + 1});
                continue;
            }

            std::string key = std::to_string(i);
            auto block      = blocks[key];

            if (auto mem_block = std::dynamic_pointer_cast<MemBlock>(block)) {
                ggml_tensor* prev_mem = state.mem_single[i];
                if (prev_mem == nullptr) {
                    prev_mem = ggml_dup_tensor(ctx->ggml_ctx, xt);
                    prev_mem = ggml_scale(ctx->ggml_ctx, prev_mem, 0.);
                }
                ggml_tensor* xt_next = mem_block->forward(ctx, xt, prev_mem);
                state.mem_single[i]  = xt;

                work_stack.push_back({xt_next, i + 1});
            } else if (auto wide_mem_block = std::dynamic_pointer_cast<WideMemBlock>(block)) {
                ggml_tensor* prev_mem = state.mem_single[i];
                if (prev_mem == nullptr) {
                    prev_mem = ggml_dup_tensor(ctx->ggml_ctx, xt);
                    prev_mem = ggml_scale(ctx->ggml_ctx, prev_mem, 0.);
                }
                ggml_tensor* xt_next = wide_mem_block->forward(ctx, xt, prev_mem);
                state.mem_single[i]  = xt;

                work_stack.push_back({xt_next, i + 1});
            } else if (auto tgrow = std::dynamic_pointer_cast<TGrow>(block)) {
                ggml_tensor* xt_grown = tgrow->forward(ctx, xt);
                int stride            = tgrow->stride;

                // Push chunked sub-frames onto stack in REVERSE order
                // so that sub-frame 0 is popped and processed first
                for (int s = stride - 1; s >= 0; --s) {
                    ggml_tensor* sub_frame = xt_grown;
                    if (stride > 1) {
                        int64_t chunk_channels = xt_grown->ne[3] / stride;
                        size_t offset          = s * chunk_channels * xt_grown->nb[3];

                        sub_frame = ggml_view_4d(ctx->ggml_ctx, xt_grown,
                                                 xt_grown->ne[0], xt_grown->ne[1], xt_grown->ne[2], chunk_channels,
                                                 xt_grown->nb[1], xt_grown->nb[2], xt_grown->nb[3],
                                                 offset);
                    }
                    work_stack.push_back({sub_frame, i + 1});
                }
            } else if (auto unary_block = std::dynamic_pointer_cast<UnaryBlock>(block)) {
                ggml_tensor* xt_next = unary_block->forward(ctx, xt);
                work_stack.push_back({xt_next, i + 1});
            }
        }

        return nullptr;  // Work stack exhausted
    }

    ggml_tensor* forward_seq(GGMLRunnerContext* ctx,
                             ggml_tensor* x) {
        SequentialDecoderState state;
        std::vector<WorkItem> work_stack;

        const std::vector<ggml_tensor*>& latent_frames = ggml_ext_chunk(ctx->ggml_ctx, x, x->ne[3], 3);

        for (auto it = latent_frames.rbegin(); it != latent_frames.rend(); ++it) {
            work_stack.push_back({*it, 0});
        }

        std::vector<ggml_tensor*> output_frames;

        while (!work_stack.empty()) {
            ggml_tensor* out_frame = forward_seq_single_step(ctx, state, work_stack);
            if (out_frame != nullptr) {
                output_frames.push_back(out_frame);
            }
        }

        auto h = ggml_ext_vec_concat(ctx->ggml_ctx, output_frames, 3);
        // shape(W, H, 3, (t_upscale - 1) + T) => shape(W, H, 3, T)
        h = ggml_view_4d(ctx->ggml_ctx, h,
                         h->ne[0], h->ne[1], h->ne[2], h->ne[3] - (t_upscale - 1),
                         h->nb[1], h->nb[2], h->nb[3],
                         (t_upscale - 1) * h->nb[3]);
        return h;
    }
};

class TAEHV : public GGMLBlock {
protected:
    bool decode_only;
    SDVersion version;
    bool is_wide;

public:
    bool parallel                    = false;
    int z_channels                   = 16;
    std::vector<bool> time_downscale = {true, true, false};
    std::vector<bool> time_upscale   = {false, true, true};

public:
    TAEHV(bool decode_only = true, SDVersion version = VERSION_WAN2, bool is_wide = false)
        : decode_only(decode_only), version(version), is_wide(is_wide) {
        int patch = 1;
        if (version == VERSION_WAN2_2_TI2V) {
            z_channels = 48;
            patch      = 2;
        } else if (sd_version_is_hunyuan_video(version)) {
            z_channels = 32;
            patch      = 2;
        } else if (sd_version_is_ltxav(version)) {
            z_channels     = 128;
            patch          = 4;
            time_downscale = {true, true, true};
            time_upscale   = {true, true, true};
        }
        blocks["decoder"] = std::shared_ptr<GGMLBlock>(new TinyVideoDecoder(z_channels, patch, time_upscale, is_wide));
        if (!decode_only) {
            blocks["encoder"] = std::shared_ptr<GGMLBlock>(new TinyVideoEncoder(z_channels, patch, time_downscale));
        }
    }

    ggml_tensor* decode(GGMLRunnerContext* ctx, ggml_tensor* z) {
        auto decoder = std::dynamic_pointer_cast<TinyVideoDecoder>(blocks["decoder"]);
        if (sd_version_is_wan(version) || sd_version_is_hunyuan_video(version) || sd_version_is_ltxav(version)) {
            // (W, H, C, T) -> (W, H, T, C)
            z = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, z, 0, 1, 3, 2));
        }
        auto result = parallel ? decoder->forward(ctx, z) : decoder->forward_seq(ctx, z);
        if (sd_version_is_wan(version) || sd_version_is_hunyuan_video(version) || sd_version_is_ltxav(version)) {
            // (W, H, T, C) -> (W, H, C, T)
            result = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, result, 0, 1, 3, 2));
        }
        return result;
    }

    ggml_tensor* encode(GGMLRunnerContext* ctx, ggml_tensor* x) {
        auto encoder = std::dynamic_pointer_cast<TinyVideoEncoder>(blocks["encoder"]);
        if (sd_version_is_wan(version) || sd_version_is_hunyuan_video(version) || sd_version_is_ltxav(version)) {
            // (W, H, T, C) -> (W, H, C, T)
            x = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));
        }
        int64_t num_frames = x->ne[3];
        if (num_frames % encoder->t_downscale) {
            // pad to multiple of encoder->t_downscale at the end
            auto last_frame = ggml_view_4d(ctx->ggml_ctx, x, x->ne[0], x->ne[1], x->ne[2], 1, x->nb[1], x->nb[2], x->nb[3], (num_frames - 1) * x->nb[3]);
            for (int i = 0; i < encoder->t_downscale - num_frames % encoder->t_downscale; i++) {
                x = ggml_concat(ctx->ggml_ctx, x, last_frame, 3);
            }
        }
        x = parallel ? encoder->forward(ctx, x) : encoder->forward_seq(ctx, x);
        if (sd_version_is_wan(version) || sd_version_is_hunyuan_video(version) || sd_version_is_ltxav(version)) {
            // (W, H, C, T) -> (W, H, T, C)
            x = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));
        }
        return x;
    }
};

class TAESD : public GGMLBlock {
protected:
    bool decode_only;
    bool taef2 = false;

public:
    int z_channels = 4;

public:
    TAESD(bool decode_only = true, SDVersion version = VERSION_SD1)
        : decode_only(decode_only) {
        bool use_midblock_gn = false;
        taef2                = sd_version_uses_flux2_vae(version);

        if (sd_version_is_dit(version)) {
            z_channels = 16;
        }
        if (taef2) {
            z_channels      = 32;
            use_midblock_gn = true;
        }
        blocks["decoder.layers"] = std::shared_ptr<GGMLBlock>(new TinyDecoder(z_channels, use_midblock_gn));

        if (!decode_only) {
            blocks["encoder.layers"] = std::shared_ptr<GGMLBlock>(new TinyEncoder(z_channels, use_midblock_gn));
        }
    }

    ggml_tensor* decode(GGMLRunnerContext* ctx, ggml_tensor* z) {
        auto decoder = std::dynamic_pointer_cast<TinyDecoder>(blocks["decoder.layers"]);
        if (taef2) {
            z = unpatchify(ctx->ggml_ctx, z, 2);
        }
        return decoder->forward(ctx, z);
    }

    ggml_tensor* encode(GGMLRunnerContext* ctx, ggml_tensor* x) {
        auto encoder = std::dynamic_pointer_cast<TinyEncoder>(blocks["encoder.layers"]);
        auto z       = encoder->forward(ctx, x);
        if (taef2) {
            z = patchify(ctx->ggml_ctx, z, 2);
        }
        return z;
    }
};

struct TinyImageAutoEncoder : public VAE {
    TAESD taesd;
    bool decode_only = false;

    TinyImageAutoEncoder(ggml_backend_t backend,
                         const String2TensorStorage& tensor_storage_map,
                         const std::string prefix,
                         bool decoder_only                                   = true,
                         SDVersion version                                   = VERSION_SD1,
                         std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : VAE(version, backend, "tae", weight_manager),
          decode_only(decoder_only),
          taesd(decoder_only, version) {
        scale_input = false;
        taesd.init(params_ctx, tensor_storage_map, prefix);
    }

    std::string get_desc() override {
        return "taesd";
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        taesd.get_param_tensors(tensors, weight_prefix);
    }

    sd::Tensor<float> vae_output_to_latents(const sd::Tensor<float>& vae_output, std::shared_ptr<RNG> rng) override {
        SD_UNUSED(rng);
        return vae_output;
    }

    sd::Tensor<float> diffusion_to_vae_latents(const sd::Tensor<float>& latents) override {
        return latents;
    }

    sd::Tensor<float> vae_to_diffusion_latents(const sd::Tensor<float>& latents) override {
        return latents;
    }

    int get_encoder_output_channels(int input_channels) {
        return taesd.z_channels;
    }

    ggml_cgraph* build_graph(const sd::Tensor<float>& z_tensor, bool decode_graph) {
        ggml_cgraph* gf  = ggml_new_graph(compute_ctx);
        ggml_tensor* z   = make_input(z_tensor);
        auto runner_ctx  = get_context();
        ggml_tensor* out = decode_graph ? taesd.decode(&runner_ctx, z) : taesd.encode(&runner_ctx, z);
        ggml_build_forward_expand(gf, out);
        return gf;
    }

    sd::Tensor<float> _compute(const int n_threads,
                               const sd::Tensor<float>& z_tensor,
                               bool decode_graph) override {
        auto get_graph = [&]() -> ggml_cgraph* {
            return build_graph(z_tensor, decode_graph);
        };

        return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false, false, false), z_tensor.dim());
    }
};

struct TinyVideoAutoEncoder : public VAE {
    TAEHV taehv;
    bool decode_only = false;
    bool is_wide     = false;

    TinyVideoAutoEncoder(ggml_backend_t backend,
                         const String2TensorStorage& tensor_storage_map,
                         const std::string prefix,
                         bool decoder_only                                   = true,
                         SDVersion version                                   = VERSION_WAN2,
                         std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : VAE(version, backend, "tae", weight_manager),
          decode_only(decoder_only) {
        for (auto tensor_storage : tensor_storage_map) {
            if (tensor_storage.first.find(prefix + ".3.conv.6.weight") != std::string::npos) {
                is_wide = true;
                break;
            }
        }
        taehv       = TAEHV(decoder_only, version, is_wide);
        scale_input = false;
        taehv.init(params_ctx, tensor_storage_map, prefix);
    }

    std::string get_desc() override {
        return "taehv";
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        taehv.get_param_tensors(tensors, weight_prefix);
    }

    sd::Tensor<float> vae_output_to_latents(const sd::Tensor<float>& vae_output, std::shared_ptr<RNG> rng) override {
        SD_UNUSED(rng);
        return vae_output;
    }

    sd::Tensor<float> diffusion_to_vae_latents(const sd::Tensor<float>& latents) override {
        return latents;
    }

    sd::Tensor<float> vae_to_diffusion_latents(const sd::Tensor<float>& latents) override {
        return latents;
    }

    int get_encoder_output_channels(int input_channels) {
        return taehv.z_channels;
    }

    ggml_cgraph* build_graph(const sd::Tensor<float>& z_tensor, bool decode_graph) {
        ggml_cgraph* gf = nullptr;
        ggml_tensor* z  = make_input(z_tensor);
        if (decode_graph) {
            int64_t passes = taehv.parallel ? 1 : z->ne[3];
            gf             = ggml_new_graph_custom(compute_ctx, (is_wide ? 4096 : 2048) * passes, false);
        } else {
            int64_t frames = z->ne[2];
            int64_t factor = sd_version_is_minimax_h3(version) ? 20 : sd_version_is_ltxav(version) ? 8
                                                                                                   : 4;
            int64_t passes = taehv.parallel ? 1 : (frames + factor - 1) / factor;
            gf             = ggml_new_graph_custom(compute_ctx, 2048 * passes, false);
        }
        auto runner_ctx  = get_context();
        ggml_tensor* out = decode_graph ? taehv.decode(&runner_ctx, z) : taehv.encode(&runner_ctx, z);
        ggml_build_forward_expand(gf, out);
        return gf;
    }

    sd::Tensor<float> _compute(const int n_threads,
                               const sd::Tensor<float>& z_tensor,
                               bool decode_graph) override {
        auto get_graph = [&]() -> ggml_cgraph* {
            return build_graph(z_tensor, decode_graph);
        };

        return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false, false, false), z_tensor.dim());
    }
};

#endif  // __SD_MODEL_VAE_TAE_HPP__
