#ifndef __SD_BACKEND_FIT_H__
#define __SD_BACKEND_FIT_H__

#include <functional>
#include <string>

#include "core/ggml_graph_cut.h"
#include "model_loader.h"
#include "stable-diffusion.h"

namespace sd {
    struct TilingConfig;
}

namespace sd::backend_fit {

    int64_t available_ram_bytes();
    int64_t available_commit_bytes();

    using AvailableVRAM = std::function<size_t(ggml_backend_dev_t, size_t, size_t)>;

    std::string select_default_backend(const sd::ggml_graph_cut::MaxVramAssignment& budgets, const AvailableVRAM& available_vram = {});

    bool derive_backend_specs(ModelLoader& loader,
                              ggml_type override_wtype,
                              sd::ggml_graph_cut::MaxVramAssignment& budgets,
                              std::string& runtime_spec,
                              std::string& params_spec);

    bool derive_backend_specs(ModelLoader& loader,
                              ggml_type override_wtype,
                              sd::ggml_graph_cut::MaxVramAssignment& budgets,
                              std::string& runtime_spec,
                              std::string& params_spec,
                              const AvailableVRAM& available_vram,
                              size_t max_ram_bytes = 0);

    bool prepare_vae_decode_retry_tiling(sd::TilingConfig& tiling_params,
                                         bool prefer_temporal_tiling,
                                         ggml_status status,
                                         int latent_tile_size_w,
                                         int latent_tile_size_h,
                                         int scale_factor);

}  // namespace sd::backend_fit

#endif  // __SD_BACKEND_FIT_H__
