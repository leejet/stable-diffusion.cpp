#ifndef __LTXV_HPP__
#define __LTXV_HPP__

#include "common_block.hpp"

namespace LTXV {

    enum class SpatialPadding { ZEROS, REFLECT };

    class CausalConv3d : public GGMLBlock {
    protected:
        int time_kernel_size;
        int spatial_kernel_size;
        SpatialPadding spatial_padding;

    public:
        CausalConv3d(int64_t in_channels,
                     int64_t out_channels,
                     int kernel_size                  = 3,
                     std::tuple<int, int, int> stride = {1, 1, 1},
                     int dilation                     = 1,
                     bool bias                        = true,
                     SpatialPadding padding_mode      = SpatialPadding::ZEROS) {
            // Python reference: self.time_kernel_size = kernel_size[0] — the full temporal kernel.
            // Earlier revisions of this file used `kernel_size / 2` which under-padded by a factor of 2 for k>=3
            // and padded 1 frame when k=1/2 where no padding was expected. Match Python verbatim.
            time_kernel_size    = kernel_size;
            spatial_kernel_size = kernel_size;
            spatial_padding     = padding_mode;
            // When using reflect padding we do it manually in forward(), so the inner Conv3d
            // must run with spatial padding=0. For zeros mode the Conv3d handles padding itself.
            int conv_pad_hw = (padding_mode == SpatialPadding::ZEROS) ? (kernel_size / 2) : 0;
            blocks["conv"]  = std::shared_ptr<GGMLBlock>(new Conv3d(in_channels,
                                                                     out_channels,
                                                                     {kernel_size, kernel_size, kernel_size},
                                                                     stride,
                                                                     {0, conv_pad_hw, conv_pad_hw},
                                                                     {dilation, 1, 1},
                                                                     bias));
        }

        // Helper: replicate the given single-frame tensor `count` times along the depth axis.
        // Returns a [IW, IH, count, N*IC] tensor. count must be >= 1.
        static ggml_tensor* repeat_frame(ggml_context* ctx, ggml_tensor* frame, int count) {
            auto out = frame;
            for (int i = 1; i < count; i++) {
                out = ggml_concat(ctx, out, frame, 2);
            }
            return out;
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             bool causal = true) {
            // x logical shape: [N*IC, ID, IH, IW] (Python order); ggml ne: [IW, IH, ID, N*IC]
            auto conv    = std::dynamic_pointer_cast<Conv3d>(blocks["conv"]);
            auto ggml_cx = ctx->ggml_ctx;

            int pad_front = 0;
            int pad_back  = 0;
            if (causal) {
                pad_front = time_kernel_size - 1;
            } else {
                pad_front = (time_kernel_size - 1) / 2;
                pad_back  = (time_kernel_size - 1) / 2;
            }

            if (pad_front > 0 || pad_back > 0) {
                // Extract first frame as a [IW, IH, 1, N*IC] view on x along the depth axis (ne[2]).
                auto first_frame = ggml_view_4d(ggml_cx, x,
                                                x->ne[0], x->ne[1], 1, x->ne[3],
                                                x->nb[1], x->nb[2], x->nb[3], 0);
                first_frame      = ggml_cont(ggml_cx, first_frame);

                if (pad_front > 0) {
                    auto front_pad = repeat_frame(ggml_cx, first_frame, pad_front);
                    x              = ggml_concat(ggml_cx, front_pad, x, 2);
                }
                if (pad_back > 0) {
                    auto last_frame = ggml_view_4d(ggml_cx, x,
                                                   x->ne[0], x->ne[1], 1, x->ne[3],
                                                   x->nb[1], x->nb[2], x->nb[3], (x->ne[2] - 1) * x->nb[2]);
                    last_frame      = ggml_cont(ggml_cx, last_frame);
                    auto back_pad   = repeat_frame(ggml_cx, last_frame, pad_back);
                    x               = ggml_concat(ggml_cx, x, back_pad, 2);
                }
            }

            // Spatial reflect padding (H, W by k/2 each side). nn.Conv3d with padding_mode='reflect'
            // mirrors the edge rows/cols: [a,b,c,d] with pad=1 → [b,a,b,c,d,c].
            if (spatial_padding == SpatialPadding::REFLECT) {
                int pad = spatial_kernel_size / 2;
                if (pad > 0) {
                    GGML_ASSERT(pad == 1 && "reflect padding only implemented for kernel=3 (pad=1)");
                    x = ggml_cont(ggml_cx, x);
                    int64_t W = x->ne[0], H = x->ne[1], T = x->ne[2], C = x->ne[3];
                    // H-axis reflect: top = row 1, bottom = row H-2.
                    auto row_top = ggml_cont(ggml_cx, ggml_view_4d(ggml_cx, x, W, 1, T, C,
                                                                   x->nb[1], x->nb[2], x->nb[3], 1 * x->nb[1]));
                    auto row_bot = ggml_cont(ggml_cx, ggml_view_4d(ggml_cx, x, W, 1, T, C,
                                                                   x->nb[1], x->nb[2], x->nb[3], (H - 2) * x->nb[1]));
                    x = ggml_concat(ggml_cx, row_top, x, 1);
                    x = ggml_concat(ggml_cx, x, row_bot, 1);
                    x = ggml_cont(ggml_cx, x);
                    W = x->ne[0]; H = x->ne[1]; T = x->ne[2]; C = x->ne[3];
                    // W-axis reflect: left = col 1, right = col W-2.
                    auto col_left  = ggml_cont(ggml_cx, ggml_view_4d(ggml_cx, x, 1, H, T, C,
                                                                     x->nb[1], x->nb[2], x->nb[3], 1 * x->nb[0]));
                    auto col_right = ggml_cont(ggml_cx, ggml_view_4d(ggml_cx, x, 1, H, T, C,
                                                                     x->nb[1], x->nb[2], x->nb[3], (W - 2) * x->nb[0]));
                    x = ggml_concat(ggml_cx, col_left, x, 0);
                    x = ggml_concat(ggml_cx, x, col_right, 0);
                }
            }

            x = conv->forward(ctx, x);
            return x;
        }
    };

};

#endif