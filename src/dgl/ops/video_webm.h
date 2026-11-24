#ifndef __SD_DGL_OPS_VIDEO_WEBM_H__
#define __SD_DGL_OPS_VIDEO_WEBM_H__

#include "dgl/ops/video.h"

namespace sd::dgl::ops {
    VideoReadResult load_webm(const std::filesystem::path& path, size_t max_frames, int64_t fps_num, int64_t fps_den, bool include_audio);
}

#endif  // __SD_DGL_OPS_VIDEO_WEBM_H__
