#ifndef __SD_DGL_BINDINGS_VIDEO_WINDOWS_H__
#define __SD_DGL_BINDINGS_VIDEO_WINDOWS_H__

#include "dgl/value.h"

namespace sd::dgl {
    struct RunState;

    Value split_video_frames(Values& args, RunState&);
    Value join_video_windows(Values& args, RunState&);
}  // namespace sd::dgl

#endif  // __SD_DGL_BINDINGS_VIDEO_WINDOWS_H__
