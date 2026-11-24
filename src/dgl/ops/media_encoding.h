#ifndef __SD_DGL_OPS_MEDIA_ENCODING_H__
#define __SD_DGL_OPS_MEDIA_ENCODING_H__

#include "dgl/ops/video.h"

namespace sd::dgl::ops {
    std::vector<uint8_t> encode_webm(const sd_tensor_t& frames, const VideoSaveOptions& options);
    void save_media(const std::string& path, const std::vector<uint8_t>& data, const std::string& mime_type);
}

#endif
