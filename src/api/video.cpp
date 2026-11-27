#include "api.h"

#include "dgl/ops/video.h"

sd_status_t sd_video_decode(const void* data,
                            size_t size,
                            int fps,
                            size_t max_frames,
                            sd_tensor_t** frames_out,
                            sd_tensor_t** samples_out,
                            uint32_t* sample_rate_out) {
    if (frames_out)
        *frames_out = nullptr;
    if (samples_out)
        *samples_out = nullptr;
    if (sample_rate_out)
        *sample_rate_out = 0;
    return sd::api::call([&] {
        if (!data || !size || !frames_out || fps < 0)
            return sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "data, size and frames_out are required, and fps must not be negative");
        sd::dgl::ops::VideoSource source(data, size);
        auto video = sd::dgl::ops::load_video(source, max_frames, fps, 1, samples_out != nullptr);
        if (samples_out && video.audio.samples) {
            *samples_out = sd_tensor_retain(video.audio.samples.get());
            if (sample_rate_out)
                *sample_rate_out = uint32_t(video.audio.sample_rate);
        }
        *frames_out = sd_tensor_retain(video.frames.get());
        return SD_STATUS_OK;
    });
}
