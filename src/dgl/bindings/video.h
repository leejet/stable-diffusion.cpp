#ifndef __SD_DGL_BINDINGS_VIDEO_H__
#define __SD_DGL_BINDINGS_VIDEO_H__

#include "dgl/ops/video.h"
#include "dgl/registry.h"

namespace sd::dgl {
    OutputSignature video_signature();
    Value video_result(std::shared_ptr<sd_tensor_t> frames, int64_t fps_num, int64_t fps_den);
    ops::VideoReadResult read_video_dict(const Value& value);
    Value execute_load_video(Values& args, RunState&);
    Value execute_save_video(Values& args, RunState&);
}  // namespace sd::dgl

#endif  // __SD_DGL_BINDINGS_VIDEO_H__
