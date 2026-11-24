#ifndef __SD_DGL_BINDINGS_COMPONENT_LOADER_H__
#define __SD_DGL_BINDINGS_COMPONENT_LOADER_H__

#include "dgl/value.h"

namespace sd::dgl {
    struct PreparedResource;

    PreparedResource load_dit(Values& args);
    PreparedResource load_text_encoder(Values& args);
    PreparedResource load_vae(Values& args);
    PreparedResource load_tae(Values& args);
    PreparedResource load_controlnet(Values& args);
    PreparedResource load_vision_encoder(Values& args);
    PreparedResource load_audio_encoder(Values& args);
    PreparedResource load_audio_vae(Values& args);
    PreparedResource load_ip_adapter(Values& args);
    PreparedResource load_photomaker(Values& args);
    PreparedResource load_pulid(Values& args);
    PreparedResource load_animatediff(Values& args);
    PreparedResource load_latent_upscaler(Values& args);
}  // namespace sd::dgl

#endif  // __SD_DGL_BINDINGS_COMPONENT_LOADER_H__
