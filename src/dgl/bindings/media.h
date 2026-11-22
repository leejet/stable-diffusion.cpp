#ifndef __SD_DGL_BINDINGS_MEDIA_H__
#define __SD_DGL_BINDINGS_MEDIA_H__

#include "dgl/registry.h"

namespace sd::dgl {
    OutputSignature media_signature();
    Value execute_encode_image(Values& args, RunState& state);
    Value execute_encode_video(Values& args, RunState& state);
    Value execute_save_media(Values& args, RunState& state);
}

#endif
