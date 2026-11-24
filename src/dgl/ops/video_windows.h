#ifndef __SD_DGL_OPS_VIDEO_WINDOWS_H__
#define __SD_DGL_OPS_VIDEO_WINDOWS_H__

#include "dgl/tensor.h"

namespace sd::dgl::ops {
    std::vector<std::shared_ptr<sd_tensor_t>> split_video_frames(const std::shared_ptr<sd_tensor_t>& frames, size_t size, size_t overlap);
    std::shared_ptr<sd_tensor_t> join_video_windows(const std::vector<std::shared_ptr<sd_tensor_t>>& windows, size_t overlap);
}

#endif  // __SD_DGL_OPS_VIDEO_WINDOWS_H__
