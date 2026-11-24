#ifndef __SD_DGL_OPS_VIDEO_H__
#define __SD_DGL_OPS_VIDEO_H__

#include <filesystem>
#include "dgl/ops/audio.h"
#include "dgl/tensor.h"

namespace sd::dgl::ops {
    struct VideoReadResult {
        std::shared_ptr<sd_tensor_t> frames;
        int64_t fps_num = 24;
        int64_t fps_den = 1;
        AudioReadResult audio;
    };

    struct VideoSaveOptions {
        int64_t fps_num = 24;
        int64_t fps_den = 1;
        AudioReadResult audio;
        bool trim_audio = false;
        bool jpeg       = true;
        int quality     = 90;
        bool pcm16      = false;
        std::string metadata;
    };

    void validate_video_frames(const sd_tensor_t& frames);
    sd::Image video_frame_view(const sd_tensor_t& frames, size_t index);
    VideoReadResult load_frame_directory(const std::filesystem::path& path, size_t max_frames, int64_t fps_num, int64_t fps_den);
    VideoReadResult load_video(const std::filesystem::path& path, size_t max_frames, int64_t fps_num, int64_t fps_den, bool include_audio);
    VideoReadResult load_avi(const std::filesystem::path& path, size_t max_frames, int64_t fps_num, int64_t fps_den, bool include_audio);
    void save_video(const sd_tensor_t& frames, const std::filesystem::path& path, const VideoSaveOptions& options);
    std::vector<uint8_t> encode_avi(const sd_tensor_t& frames, const VideoSaveOptions& options);
}

#endif  // __SD_DGL_OPS_VIDEO_H__
