#ifndef __SD_BACKEND_FIT_H__
#define __SD_BACKEND_FIT_H__

#include <cstdint>
#include <functional>
#include <string>

#include "core/ggml_graph_cut.h"
#include "model_loader.h"
#include "stable-diffusion.h"

namespace sd {
    struct TilingConfig;
}

namespace sd::backend_fit {

    // Physical memory available without swapping, or -1 when the platform cannot report it.
    int64_t available_ram_bytes();
    int64_t available_commit_bytes();

    using AvailableVRAM = std::function<size_t(ggml_backend_dev_t, size_t, size_t)>;

    std::string select_default_backend(const sd::ggml_graph_cut::MaxVramAssignment& budgets, const AvailableVRAM& available_vram = {});

    enum class MultiGpuMode {
        AUTO,
        ON,
        OFF,
    };

    struct Options {
        MultiGpuMode multi_gpu = MultiGpuMode::AUTO;
        std::string split_mode_spec;
        AvailableVRAM available_vram;
        size_t max_ram_bytes = 0;
    };

    // Accepts auto, on and off in any case; an empty string means auto.
    bool parse_multi_gpu_mode(const std::string& text, MultiGpuMode& mode);

    bool derive_backend_specs(ModelLoader& loader,
                              ggml_type override_wtype,
                              sd::ggml_graph_cut::MaxVramAssignment& budgets,
                              std::string& runtime_spec,
                              std::string& params_spec);

    // memory_guard_percent is raised from 0 when the plan runs a component on the CPU with its
    // weights read from disk: without the guard nothing would release them between segments.
    bool derive_backend_specs(ModelLoader& loader,
                              ggml_type override_wtype,
                              sd::ggml_graph_cut::MaxVramAssignment& budgets,
                              std::string& runtime_spec,
                              std::string& params_spec,
                              int& memory_guard_percent,
                              const Options& options);

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
