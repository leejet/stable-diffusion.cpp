#ifndef __EXAMPLES_COMMON_PROFILE_H__
#define __EXAMPLES_COMMON_PROFILE_H__

#include <cstdint>
#include <string>
#include <vector>

#include "common.h"

// Resource profiles: a few named sets of options that trade speed for a smaller memory footprint.
// A profile only sets options the user did not give on the command line.
namespace sd_profile {

    // From the lightest on resources to the heaviest. The levels after LITE are not selectable from
    // the command line: a server steps down to them when a request runs out of memory. Every level up to
    // LITE_DISK trades memory for time and keeps all the compute on the GPUs; the CPU takes over only
    // from LITE_HYBRID on, the text encoder first and everything at LITE_CPU.
    enum class Level {
        MAX,          // the defaults of the engine
        BALANCED,     // memory guard at 90
        LITE,         // + tiled VAE, unbatched CFG, memory guard at 85, every GPU of the backend used
        LITE_DISK,    // + weights streamed from disk
        LITE_HYBRID,  // + text encoder computed on the CPU
        LITE_CPU,     // + every module computed on the CPU
    };

    const char* level_name(Level level);

    // Accepts max, balanced, lite and auto; for auto, automatic is set and level is left unchanged.
    bool parse(const std::string& text, Level& level, bool& automatic);

    // The next level that needs less memory; the last level returns itself.
    Level lighter(Level level);

    struct Resources {
        int64_t model_bytes   = 0;   // size of the model files, 0 when unknown
        int64_t ram_available = -1;  // RAM a program can still use, -1 when unknown
        int64_t gpu_free      = 0;   // free memory of the discrete GPU with the most, or of the GPUs of one backend together when auto-fit may spread the model over them; 0 without one
        int gpu_count         = 0;   // how many GPUs gpu_free adds up
    };

    // True when --auto-fit plans where the weights go and neither --multi-gpu off nor an explicit --backend or
    // --params-backend keeps a model on one GPU.
    bool may_use_several_gpus(const SDContextParams& ctx);

    // The memory the weights can sit in on the GPUs. With pool, the discrete GPUs of the backend that
    // has the GPU with the most free memory (CUDA0, CUDA1, ...) count together, less 2 GiB of working
    // memory for every GPU after the first; this is how --auto-fit with --multi-gpu spreads a model.
    // Without pool, or with a single GPU, it is the free memory of the discrete GPU or accelerator
    // that has the most. count receives the number of GPUs added up (0 when there is none).
    int64_t usable_gpu_memory(const std::vector<sd_device_info_t>& devices, bool pool, int* count = nullptr);

    // Reads the sizes of the model files and the memory of the machine. Reads no model data.
    Resources probe(const SDContextParams& ctx);

    // Picks MAX, BALANCED or LITE: MAX when the weights and the working memory fit with room to
    // spare in the memory where the compute happens, BALANCED when they fit, LITE otherwise.
    Level choose(const Resources& resources, std::string* reason = nullptr);

    // Sets the options of the level. With respect_explicit, an option found on the command line is
    // left as it is; without it (a server recovering from a failure) the level takes precedence.
    void apply(Level level, SDContextParams& ctx, SDGenerationParams& gen, bool respect_explicit);

    // Resolves --profile (when present) and applies it. Returns false after logging an error when the
    // name is not valid. applied receives the level in effect (MAX without --profile).
    bool apply_from_options(SDContextParams& ctx, SDGenerationParams& gen, Level& applied);

}  // namespace sd_profile

#endif  // __EXAMPLES_COMMON_PROFILE_H__
