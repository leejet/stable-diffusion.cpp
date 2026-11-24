#ifndef __SD_DGL_BINDINGS_COMPONENTS_H__
#define __SD_DGL_BINDINGS_COMPONENTS_H__

#include "dgl/value.h"

namespace sd::dgl {
    struct RunState;

    Value encode_prompt(Values& args, RunState&);
    Value photomaker_tokens(Values& args, RunState&);
    Value encode_photomaker_prompt(Values& args, RunState&);
    Value apply_photomaker(Values& args, RunState&);
    Value vae_encode(Values& args, RunState&);
    Value vae_decode(Values& args, RunState&);
    Value latent_shape(Values& args, RunState&);
    Value t_to_sigma(Values& args, RunState&);
    Value denoiser_scalings(Values& args, RunState&);
    Value make_sigmas(Values& args, RunState&);
    Value sample_tensor(Values& args, RunState& state);
    Value execute_load_pulid_embedding(Values& args, RunState&);
    Value execute_vae_to_dit_latents(Values& args, RunState&);
    Value execute_dit_to_vae_latents(Values& args, RunState&);
    Value execute_add_noise(Values& args, RunState&);
}  // namespace sd::dgl

#endif  // __SD_DGL_BINDINGS_COMPONENTS_H__
