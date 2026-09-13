#ifndef __SD_CONDITIONING_AUDIO_PROCESSING_HPP__
#define __SD_CONDITIONING_AUDIO_PROCESSING_HPP__

// Wan2.2-S2V audio windowing, ported from ComfyUI comfy_extras/nodes_wan.py
// (linear_interpolation + get_audio_embed_bucket_fps, m=0).
//
// Input is the wav2vec2 hidden states stacked per layer [num_layers, in_frames, dim]
// at the encoder frame rate (50 Hz). The frames are interpolated to video_rate
// (30 Hz), bucketed to fps (16) frames with zero padding past the audio end, and
// split into chunks of batch_frames = latent_t * 4 frames (one per diffusion chunk).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

namespace AudioProcessing {

    // Chunk/padding math of get_audio_embed_bucket_fps (m=0).
    struct BucketPlan {
        int audio_frames;         // input frames at video_rate (30 Hz)
        int batch_frames;         // pixel frames per chunk (latent_t * 4)
        int video_rate;           // timeline rate of audio_frames (30 Hz)
        int fps;                  // bucket frame rate (16 fps)
        int num_chunks;           // ComfyUI num_repeat
        int bucket_frames;        // total bucket frames = num_chunks * batch_frames
        int padded_audio_frames;  // audio_frames plus zero padding applied
    };

    inline BucketPlan plan_buckets(int audio_frames, int batch_frames, int video_rate = 30, int fps = 16) {
        BucketPlan plan;
        plan.audio_frames  = audio_frames;
        plan.batch_frames  = batch_frames;
        plan.video_rate    = video_rate;
        plan.fps           = fps;
        const double scale = static_cast<double>(video_rate) / fps;
        // min_batch_num = int(audio_frame_num / (batch_frames * scale)) + 1
        plan.num_chunks    = static_cast<int>(audio_frames / (batch_frames * scale)) + 1;
        plan.bucket_frames = plan.num_chunks * batch_frames;
        // padd_audio_num = ceil(bucket_frames / fps * video_rate) - audio_frame_num
        plan.padded_audio_frames = static_cast<int>(
            std::ceil(plan.bucket_frames / static_cast<double>(fps) * video_rate));
        return plan;
    }

    // Bucket frame index (fps timeline) -> source frame index (video_rate timeline).
    // get_sample_indices with fixed_start=0 reduces to round-half-even(i * video_rate / fps),
    // matching numpy's default rounding.
    inline int bucket_source_frame(int bucket_frame, int video_rate = 30, int fps = 16) {
        return static_cast<int>(std::nearbyint(static_cast<double>(bucket_frame) * video_rate / fps));
    }

    // torch.nn.functional.interpolate size computation: output_len = int(in_len / input_fps * output_fps)
    inline int interpolated_frame_count(int in_frames, int input_fps = 50, int output_fps = 30) {
        return static_cast<int>(in_frames / static_cast<double>(input_fps) * output_fps);
    }

    // torch.nn.functional.interpolate(mode='linear', align_corners=True) along the frame
    // dimension. in: [num_layers, in_frames, dim], out: [num_layers, out_frames, dim].
    inline std::vector<float> linear_interpolate_frames(const std::vector<float>& in,
                                                        int num_layers,
                                                        int in_frames,
                                                        int dim,
                                                        int out_frames) {
        std::vector<float> out(static_cast<size_t>(num_layers) * out_frames * dim, 0.0f);
        if (in.empty() || in_frames <= 0 || out_frames <= 0 || num_layers <= 0 || dim <= 0) {
            return out;
        }
        const double scale = out_frames > 1 ? static_cast<double>(in_frames - 1) / (out_frames - 1) : 0.0;
        for (int layer = 0; layer < num_layers; ++layer) {
            for (int out_i = 0; out_i < out_frames; ++out_i) {
                const double pos     = out_i * scale;
                const int src0       = static_cast<int>(pos);
                const int src1       = std::min(src0 + 1, in_frames - 1);
                const float frac     = static_cast<float>(pos - src0);
                const float* in_row  = &in[(static_cast<size_t>(layer) * in_frames + src0) * dim];
                const float* in_next = &in[(static_cast<size_t>(layer) * in_frames + src1) * dim];
                float* out_row       = &out[(static_cast<size_t>(layer) * out_frames + out_i) * dim];
                for (int d = 0; d < dim; ++d) {
                    out_row[d] = in_row[d] * (1.0f - frac) + in_next[d] * frac;
                }
            }
        }
        return out;
    }

    // Polyphase FIR from torchaudio.functional.resample: sinc interpolated at output
    // phases, hann-windowed over lowpass_filter_width zero crossings, anti-aliased by
    // clamping the sinc argument to +-lowpass_filter_width after scaling by min(rate)*rolloff.
    // Returns the input unchanged when sample rates are equal, and an empty vector on
    // invalid input.
    inline std::vector<float> resample_audio(const float* samples,
                                             uint64_t sample_count,
                                             uint32_t orig_sample_rate,
                                             uint32_t target_sample_rate) {
        if (samples == nullptr || sample_count == 0 || orig_sample_rate == 0 || target_sample_rate == 0) {
            return {};
        }
        if (orig_sample_rate == target_sample_rate) {
            return std::vector<float>(samples, samples + sample_count);
        }

        constexpr int kLowpassFilterWidth = 6;
        constexpr double kRolloff         = 0.99;
        constexpr double kPi              = 3.14159265358979323846;

        const uint64_t gcd        = std::gcd(static_cast<uint64_t>(orig_sample_rate),
                                             static_cast<uint64_t>(target_sample_rate));
        const int64_t orig_freq   = static_cast<int64_t>(orig_sample_rate / gcd);
        const int64_t new_freq    = static_cast<int64_t>(target_sample_rate / gcd);
        const double base_freq    = static_cast<double>(std::min(orig_freq, new_freq)) * kRolloff;
        const int64_t width       = static_cast<int64_t>(std::ceil(kLowpassFilterWidth * orig_freq / base_freq));
        const int64_t kernel_size = 2 * width + orig_freq;

        std::vector<double> kernel(static_cast<size_t>(new_freq) * kernel_size);
        for (int64_t j = 0; j < new_freq; ++j) {
            for (int64_t i = 0; i < kernel_size; ++i) {
                double t = -static_cast<double>(j) / new_freq + static_cast<double>(i - width) / orig_freq;
                t *= base_freq;
                t                           = std::clamp(t, -static_cast<double>(kLowpassFilterWidth), static_cast<double>(kLowpassFilterWidth));
                const double cos_arg        = std::cos(t * kPi / kLowpassFilterWidth / 2);
                const double window         = cos_arg * cos_arg;
                double s                    = t * kPi;
                const double sinc           = (s == 0.0) ? 1.0 : std::sin(s) / s;
                kernel[j * kernel_size + i] = sinc * window * (base_freq / orig_freq);
            }
        }

        const uint64_t num_phases    = static_cast<uint64_t>(sample_count / orig_freq) + 1;
        const uint64_t target_length = (static_cast<uint64_t>(new_freq) * sample_count +
                                        static_cast<uint64_t>(orig_freq) - 1) /
                                       static_cast<uint64_t>(orig_freq);
        std::vector<float> out(target_length);
        for (uint64_t phase = 0; phase < num_phases; ++phase) {
            const int64_t src_base = static_cast<int64_t>(phase * orig_freq) - width;
            for (int64_t j = 0; j < new_freq; ++j) {
                const uint64_t out_index = phase * new_freq + j;
                if (out_index >= target_length) {
                    break;
                }
                const double* k = &kernel[j * kernel_size];
                double acc      = 0.0;
                for (int64_t i = 0; i < kernel_size; ++i) {
                    const int64_t src = src_base + i;
                    if (src >= 0 && src < static_cast<int64_t>(sample_count)) {
                        acc += samples[src] * k[i];
                    }
                }
                out[out_index] = static_cast<float>(acc);
            }
        }
        return out;
    }

    // Downmix interleaved samples to mono by averaging channels. Returns an empty
    // vector on invalid input.
    inline std::vector<float> downmix_to_mono(const float* interleaved_samples,
                                              uint64_t sample_count,
                                              uint32_t channels) {
        std::vector<float> mono;
        if (interleaved_samples == nullptr || sample_count == 0 || channels == 0) {
            return mono;
        }
        mono.resize(static_cast<size_t>(sample_count));
        if (channels == 1) {
            std::memcpy(mono.data(), interleaved_samples, static_cast<size_t>(sample_count) * sizeof(float));
            return mono;
        }
        const float scale = 1.0f / static_cast<float>(channels);
        for (uint64_t i = 0; i < sample_count; ++i) {
            float sum = 0.0f;
            for (uint32_t c = 0; c < channels; ++c) {
                sum += interleaved_samples[i * channels + c];
            }
            mono[static_cast<size_t>(i)] = sum * scale;
        }
        return mono;
    }

    // Full bucketing: stacked encoder states [num_layers, in_frames, dim] at input_fps ->
    // bucket frames [bucket_frames, num_layers, dim] at fps, with zero frames past the
    // audio end. Chunk c occupies rows [c * batch_frames, (c + 1) * batch_frames).
    // Returns an empty vector on invalid input; the applied plan is stored in *plan_out.
    inline std::vector<float> build_audio_buckets(const float* stacked_states,
                                                  int num_layers,
                                                  int in_frames,
                                                  int dim,
                                                  int batch_frames,
                                                  BucketPlan* plan_out = nullptr,
                                                  int input_fps        = 50,
                                                  int video_rate       = 30,
                                                  int fps              = 16) {
        if (stacked_states == nullptr || num_layers <= 0 || in_frames <= 0 || dim <= 0 || batch_frames <= 0) {
            return {};
        }
        const int audio_frames = interpolated_frame_count(in_frames, input_fps, video_rate);
        if (audio_frames <= 0) {
            return {};
        }
        const std::vector<float> interpolated =
            linear_interpolate_frames(std::vector<float>(stacked_states,
                                                         stacked_states + static_cast<size_t>(num_layers) * in_frames * dim),
                                      num_layers,
                                      in_frames,
                                      dim,
                                      audio_frames);
        const BucketPlan plan = plan_buckets(audio_frames, batch_frames, video_rate, fps);
        if (plan_out != nullptr) {
            *plan_out = plan;
        }
        std::vector<float> buckets(static_cast<size_t>(plan.bucket_frames) * num_layers * dim, 0.0f);
        for (int frame = 0; frame < plan.bucket_frames; ++frame) {
            const int src = bucket_source_frame(frame, video_rate, fps);
            if (src >= plan.audio_frames) {
                continue;  // zero padding past the audio end
            }
            for (int layer = 0; layer < num_layers; ++layer) {
                std::copy_n(interpolated.data() + (static_cast<size_t>(layer) * audio_frames + src) * dim,
                            static_cast<size_t>(dim),
                            buckets.data() + (static_cast<size_t>(frame) * num_layers + layer) * dim);
            }
        }
        return buckets;
    }

}  // namespace AudioProcessing

#endif  // __SD_CONDITIONING_AUDIO_PROCESSING_HPP__
