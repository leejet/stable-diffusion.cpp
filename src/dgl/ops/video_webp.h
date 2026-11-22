#ifndef __SD_DGL_OPS_VIDEO_WEBP_H__
#define __SD_DGL_OPS_VIDEO_WEBP_H__

#include "dgl/ops/video.h"

namespace sd::dgl::ops {
    VideoReadResult load_webp(const std::filesystem::path& path, size_t max_frames, int64_t fps_num, int64_t fps_den);
}

#endif  // __SD_DGL_OPS_VIDEO_WEBP_H__
