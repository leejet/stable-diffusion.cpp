#ifndef __SD_DGL_BINDINGS_IMAGE_H__
#define __SD_DGL_BINDINGS_IMAGE_H__

#include "dgl/value.h"

namespace sd::dgl {
    struct RunState;

    Value execute_image_to_f32(Values& args, RunState&);
    Value execute_image_to_u8(Values& args, RunState&);
    Value execute_load_image(Values& args, RunState&);
    Value execute_load_images(Values& args, RunState&);
    Value execute_save_image(Values& args, RunState&);
}  // namespace sd::dgl

#endif  // __SD_DGL_BINDINGS_IMAGE_H__
