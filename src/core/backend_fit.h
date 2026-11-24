#ifndef __SD_BACKEND_FIT_H__
#define __SD_BACKEND_FIT_H__

#include <cstdint>
#include <string>

#include "config.h"
#include "core/ggml_graph_cut.h"
#include "model_loader.h"
#include "stable-diffusion.h"

namespace sd::backend_fit {

    enum class MultiGpuMode {
        AUTO,
        ON,
        OFF,
    };

    struct Options {
        MultiGpuMode multi_gpu = MultiGpuMode::AUTO;
        std::string split_mode_spec;
    };

    // Accepts auto, on and off in any case; an empty string means auto.
    bool parse_multi_gpu_mode(const std::string& text, MultiGpuMode& mode);

    // memory_guard_percent is raised from 0 when the plan runs a component on the CPU with its
    // weights read from disk: without the guard nothing would release them between segments.
    bool derive_backend_specs(ModelLoader& loader,
                              ggml_type override_wtype,
                              sd::ggml_graph_cut::MaxVramAssignment& budgets,
                              std::string& runtime_spec,
                              std::string& params_spec,
                              int& memory_guard_percent,
                              const Options& options);

    bool prepare_vae_decode_retry_tiling(sd::TilingConfig& tiling_params,
                                         bool prefer_temporal_tiling,
                                         ggml_status status,
                                         int latent_tile_size_w,
                                         int latent_tile_size_h,
                                         int scale_factor);

    // Physical memory available without swapping, or -1 when the platform cannot report it.
    int64_t available_ram_bytes();

}  // namespace sd::backend_fit

#endif  // __SD_BACKEND_FIT_H__
