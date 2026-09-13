#ifndef __SD_CONDITIONING_WAN_AUDIO_H__
#define __SD_CONDITIONING_WAN_AUDIO_H__

#include <vector>

namespace sd::wan_audio {

    // Chunk/padding math of get_audio_embed_bucket_fps (m=0).
    struct BucketPlan {
        int audio_frames;         // input frames at video_rate (30 Hz)
        int batch_frames;         // pixel frames per chunk (latent_t * 4)
        int video_rate;           // timeline rate of audio_frames (30 Hz)
        int fps;                  // bucket frame rate (16 fps)
        int num_chunks;           // number of audio chunks, including trailing padding
        int bucket_frames;        // total bucket frames = num_chunks * batch_frames
        int padded_audio_frames;  // audio_frames plus zero padding applied
    };

    // Full bucketing: stacked encoder states [num_layers, in_frames, dim] at input_fps ->
    // bucket frames [bucket_frames, num_layers, dim] at fps, with zero frames past the
    // audio end. Chunk c occupies rows [c * batch_frames, (c + 1) * batch_frames).
    // Returns an empty vector on invalid input; the applied plan is stored in *plan_out.
    std::vector<float> build_audio_buckets(const float* stacked_states,
                                           int num_layers,
                                           int in_frames,
                                           int dim,
                                           int batch_frames,
                                           BucketPlan* plan_out = nullptr,
                                           int input_fps        = 50,
                                           int video_rate       = 30,
                                           int fps              = 16);

}  // namespace sd::wan_audio

#endif  // __SD_CONDITIONING_WAN_AUDIO_H__
