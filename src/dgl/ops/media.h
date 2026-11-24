#ifndef __SD_DGL_OPS_MEDIA_H__
#define __SD_DGL_OPS_MEDIA_H__

#include "dgl/models/fwd.h"

#include "core/image_tensor.h"
#include "core/media.h"
#include "core/option_types.h"
#include "core/tensor.hpp"

namespace sd::dgl::ops {

    sd::Audio* waveform_to_sd_audio(const models::AudioVAE* sd,
                                    const sd::Tensor<float>& waveform);
    sd::Tensor<float> build_ltxv_video_positions(int64_t width,
                                                 int64_t height,
                                                 int64_t target_latent_frames,
                                                 int64_t keyframe_latent_frames,
                                                 int keyframe_frame_idx,
                                                 int keyframe_pixel_frames,
                                                 int fps,
                                                 int spatial_scale,
                                                 int temporal_scale,
                                                 bool causal_temporal_positioning);
    sd::Tensor<float> prepare_minimax_h3_reference_waveform(const sd::Audio& audio,
                                                            int target_sample_rate);
    sd::Tensor<float> build_s2v_audio_window(const sd::Tensor<float>& stacked, int64_t batch_frames);
}  // namespace sd::dgl::ops

#endif  // __SD_DGL_OPS_MEDIA_H__
