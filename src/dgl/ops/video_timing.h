#ifndef __SD_DGL_OPS_VIDEO_TIMING_H__
#define __SD_DGL_OPS_VIDEO_TIMING_H__

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sd::dgl::ops {
    struct VideoFrameRate {
        int64_t num = 0;
        int64_t den = 1;
    };

    struct VideoSampling {
        VideoFrameRate rate;
        size_t count = 0;
        int64_t time_ns(size_t frame) const;
        size_t source_index(const std::vector<int64_t>& timestamps, size_t frame) const;
    };

    VideoFrameRate video_frame_rate(double rate);
    bool video_has_fixed_rate(const std::vector<int64_t>& timestamps, int64_t end_ns, VideoFrameRate rate, int64_t tolerance_ns);
    VideoSampling plan_video_sampling(const std::vector<int64_t>& timestamps, int64_t end_ns, VideoFrameRate declared_rate, int64_t tolerance_ns, int64_t fps_num, int64_t fps_den, size_t max_frames, size_t frame_bytes);
}

#endif  // __SD_DGL_OPS_VIDEO_TIMING_H__
