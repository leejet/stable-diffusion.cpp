#ifndef __SD_DGL_BINDINGS_LATENTS_H__
#define __SD_DGL_BINDINGS_LATENTS_H__

#include "dgl/value.h"

namespace sd::dgl {
    struct RunState;

    Value encode_vision(Values& args, RunState&);
    Value encode_audio(Values& args, RunState&);
    Value encode_ip_adapter(Values& args, RunState&);
    Value upscale_latent(Values& args, RunState&);
    Value audio_vae_encode(Values& args, RunState&);
    Value audio_vae_decode(Values& args, RunState&);
    Value execute_audio_latent_shape(Values& args, RunState&);
    Value execute_ltx_video_positions(Values& args, RunState&);
    Value execute_s2v_audio_window(Values& args, RunState&);
}  // namespace sd::dgl

#endif  // __SD_DGL_BINDINGS_LATENTS_H__
