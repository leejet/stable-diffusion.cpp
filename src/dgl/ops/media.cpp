#include "dgl/ops/media.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "conditioning/wan_audio.h"
#include "dgl/models/audio_vae.h"
#include "dgl/models/fwd.h"
#include "model/vae/audio_vae.hpp"
#include "model_definition.h"
#include "runtime/denoiser.hpp"

namespace sd::dgl::ops {

    sd::Audio* waveform_to_sd_audio(const models::AudioVAE* sd,
                                    const sd::Tensor<float>& waveform) {
        if (sd == nullptr || waveform.empty()) {
            return nullptr;
        }

        int64_t sample_count = waveform.shape()[0];
        int64_t channels     = waveform.shape().size() > 1 ? waveform.shape()[1] : 1;
        if (sample_count <= 0 || channels <= 0) {
            return nullptr;
        }

        sd::Audio* audio = (sd::Audio*)malloc(sizeof(sd::Audio));
        if (audio == nullptr) {
            return nullptr;
        }

        audio->sample_rate  = static_cast<uint32_t>(sd->audio_vae_model != nullptr ? sd->audio_vae_model->output_sample_rate() : 0);
        audio->channels     = static_cast<uint32_t>(channels);
        audio->sample_count = static_cast<uint64_t>(sample_count);
        size_t sample_bytes = waveform.numel() * sizeof(float);
        audio->data         = (float*)malloc(sample_bytes);
        if (audio->data == nullptr) {
            free(audio);
            return nullptr;
        }

        auto wavaform_t = waveform.permute({1, 0, 2, 3});
        std::memcpy(audio->data, wavaform_t.data(), sample_bytes);

        return audio;
    }

    static float ltxv_latent_corner_to_pixel_frame(int64_t corner_index,
                                                   int temporal_scale,
                                                   bool causal_temporal_positioning) {
        float pixel_t = static_cast<float>(corner_index * temporal_scale);
        if (causal_temporal_positioning) {
            pixel_t = std::max(0.f, pixel_t + 1.f - static_cast<float>(temporal_scale));
        }
        return pixel_t;
    }

    static void set_ltxv_video_position(sd::Tensor<float>* positions,
                                        int64_t token,
                                        float t_start,
                                        float t_end,
                                        float h_start,
                                        float h_end,
                                        float w_start,
                                        float w_end) {
        positions->index(0, 0, token, 0) = t_start;
        positions->index(1, 0, token, 0) = t_end;
        positions->index(0, 1, token, 0) = h_start;
        positions->index(1, 1, token, 0) = h_end;
        positions->index(0, 2, token, 0) = w_start;
        positions->index(1, 2, token, 0) = w_end;
    }

    sd::Tensor<float> build_ltxv_video_positions(int64_t width,
                                                 int64_t height,
                                                 int64_t target_latent_frames,
                                                 int64_t keyframe_latent_frames,
                                                 int keyframe_frame_idx,
                                                 int keyframe_pixel_frames,
                                                 int fps,
                                                 int spatial_scale,
                                                 int temporal_scale,
                                                 bool causal_temporal_positioning) {
        GGML_ASSERT(width > 0 && height > 0 && target_latent_frames > 0);
        GGML_ASSERT(keyframe_latent_frames > 0);
        GGML_ASSERT(fps > 0);

        int64_t total_tokens = width * height * (target_latent_frames + keyframe_latent_frames);
        sd::Tensor<float> positions({2, 3, total_tokens, 1});
        int64_t token = 0;

        for (int64_t t = 0; t < target_latent_frames; t++) {
            float t_start = ltxv_latent_corner_to_pixel_frame(t, temporal_scale, causal_temporal_positioning) / static_cast<float>(fps);
            float t_end   = ltxv_latent_corner_to_pixel_frame(t + 1, temporal_scale, causal_temporal_positioning) / static_cast<float>(fps);
            for (int64_t h = 0; h < height; h++) {
                float h_start = static_cast<float>(h * spatial_scale);
                float h_end   = static_cast<float>((h + 1) * spatial_scale);
                for (int64_t w = 0; w < width; w++) {
                    float w_start = static_cast<float>(w * spatial_scale);
                    float w_end   = static_cast<float>((w + 1) * spatial_scale);
                    set_ltxv_video_position(&positions, token++, t_start, t_end, h_start, h_end, w_start, w_end);
                }
            }
        }

        for (int64_t t = 0; t < keyframe_latent_frames; t++) {
            float t_start = static_cast<float>(keyframe_frame_idx + t * temporal_scale);
            float t_end   = static_cast<float>(keyframe_frame_idx + (t + 1) * temporal_scale);
            if (keyframe_pixel_frames == 1) {
                t_end = t_start + 1.f;
            }
            t_start /= static_cast<float>(fps);
            t_end /= static_cast<float>(fps);
            for (int64_t h = 0; h < height; h++) {
                float h_start = static_cast<float>(h * spatial_scale);
                float h_end   = static_cast<float>((h + 1) * spatial_scale);
                for (int64_t w = 0; w < width; w++) {
                    float w_start = static_cast<float>(w * spatial_scale);
                    float w_end   = static_cast<float>((w + 1) * spatial_scale);
                    set_ltxv_video_position(&positions, token++, t_start, t_end, h_start, h_end, w_start, w_end);
                }
            }
        }

        return positions;
    }

    sd::Tensor<float> prepare_minimax_h3_reference_waveform(const sd::Audio& audio,
                                                            int target_sample_rate) {
        if (audio.data == nullptr || audio.sample_count == 0 || audio.channels == 0 || audio.sample_rate == 0) {
            return {};
        }
        uint64_t output_samples = static_cast<uint64_t>(std::llround(
            static_cast<long double>(audio.sample_count) * target_sample_rate / audio.sample_rate));
        output_samples          = std::max<uint64_t>(1, output_samples);
        uint64_t padded_samples = (output_samples + 799) / 800 * 800;
        // Keep stereo streams planar for the mono-per-stream audio encoder:
        // [samples, 1, stereo, batch]. This avoids flattening interleaved L/R
        // storage into alternating samples when the encoder folds streams into
        // its batch dimension.
        sd::Tensor<float> waveform({static_cast<int64_t>(padded_samples), 1, 2, 1});

        for (uint64_t i = 0; i < output_samples; ++i) {
            long double source_pos = static_cast<long double>(i) * audio.sample_rate / target_sample_rate;
            uint64_t source0       = std::min<uint64_t>(static_cast<uint64_t>(source_pos), audio.sample_count - 1);
            uint64_t source1       = std::min<uint64_t>(source0 + 1, audio.sample_count - 1);
            float fraction         = static_cast<float>(source_pos - source0);
            for (uint32_t channel = 0; channel < 2; ++channel) {
                uint32_t source_channel = audio.channels == 1 ? 0 : std::min<uint32_t>(channel, audio.channels - 1);
                float a                 = audio.data[source0 * audio.channels + source_channel];
                float b                 = audio.data[source1 * audio.channels + source_channel];
                waveform.index(static_cast<int64_t>(i), 0, channel, 0) =
                    std::clamp(a + (b - a) * fraction, -1.f, 1.f);
            }
        }
        return waveform;
    }

    sd::Tensor<float> build_s2v_audio_window(const sd::Tensor<float>& stacked, int64_t batch_frames) {
        const int64_t embed_dim  = stacked.shape()[0];
        const int64_t in_frames  = stacked.shape()[1];
        const int64_t num_layers = stacked.shape()[2];
        if (embed_dim <= 0 || in_frames <= 0 || num_layers <= 0 || batch_frames <= 0) {
            return {};
        }
        std::vector<float> layer_first(static_cast<size_t>(num_layers) * in_frames * embed_dim);
        for (int64_t l = 0; l < num_layers; ++l) {
            for (int64_t f = 0; f < in_frames; ++f) {
                const float* src = stacked.data() + l * embed_dim * in_frames + f * embed_dim;
                std::copy_n(src,
                            static_cast<size_t>(embed_dim),
                            layer_first.data() + (static_cast<size_t>(l) * in_frames + f) * embed_dim);
            }
        }
        sd::wan_audio::BucketPlan plan;
        std::vector<float> buckets = sd::wan_audio::build_audio_buckets(layer_first.data(),
                                                                        static_cast<int>(num_layers),
                                                                        static_cast<int>(in_frames),
                                                                        static_cast<int>(embed_dim),
                                                                        static_cast<int>(batch_frames),
                                                                        &plan);
        if (buckets.empty() || plan.bucket_frames < batch_frames) {
            return {};
        }
        // Reorder frame-major buckets into sd::Tensor's [dim, frame, layer] layout.
        sd::Tensor<float> window({embed_dim, batch_frames, num_layers});
        for (int64_t f = 0; f < batch_frames; ++f) {
            for (int64_t l = 0; l < num_layers; ++l) {
                const float* src = buckets.data() + (static_cast<size_t>(f) * num_layers + l) * embed_dim;
                float* dst       = window.data() + l * embed_dim * batch_frames + f * embed_dim;
                std::copy_n(src, static_cast<size_t>(embed_dim), dst);
            }
        }
        return window;
    }

}  // namespace sd::dgl::ops
