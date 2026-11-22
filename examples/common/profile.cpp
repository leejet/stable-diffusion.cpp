#include "profile.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <system_error>
#include <vector>

namespace sd_profile {

    namespace {

        constexpr int64_t GiB = 1024ll * 1024ll * 1024ll;

        std::string format_gib(int64_t bytes) {
            char text[32];
            snprintf(text, sizeof(text), "%.1f GiB", static_cast<double>(bytes) / static_cast<double>(GiB));
            return text;
        }

        int64_t file_size_or_zero(const std::string& path) {
            if (path.empty()) {
                return 0;
            }
            std::error_code error;
            const auto size = std::filesystem::file_size(path, error);
            return error ? 0 : static_cast<int64_t>(size);
        }

        bool mentions_module(const std::string& spec, const char* name) {
            return spec.find(std::string(name) + "=") != std::string::npos;
        }

        // Working memory each GPU after the first needs next to its share of the weights.
        constexpr int64_t EXTRA_GPU_HEADROOM = 2 * GiB;

        // CUDA0 and CUDA1 are one family, Vulkan0 is another: only GPUs of one family share a model.
        std::string backend_family(const char* device_name) {
            std::string family = device_name;
            while (!family.empty() && std::isdigit(static_cast<unsigned char>(family.back()))) {
                family.pop_back();
            }
            return family;
        }

        std::string lowercase(const std::string& text) {
            std::string lowered = text;
            std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return lowered;
        }

        bool multi_gpu_off(const std::string& mode) {
            return lowercase(mode) == "off";
        }

        bool multi_gpu_unset_or_auto(const std::string& mode) {
            const std::string lowered = lowercase(mode);
            return lowered.empty() || lowered == "auto";
        }

    }  // namespace

    const char* level_name(Level level) {
        switch (level) {
            case Level::MAX:
                return "max";
            case Level::BALANCED:
                return "balanced";
            case Level::LITE:
                return "lite";
            case Level::LITE_DISK:
                return "lite-disk";
            case Level::LITE_HYBRID:
                return "lite-hybrid";
            case Level::LITE_CPU:
                return "lite-cpu";
        }
        return "max";
    }

    bool parse(const std::string& text, Level& level, bool& automatic) {
        automatic = false;
        if (text == "max") {
            level = Level::MAX;
        } else if (text == "balanced") {
            level = Level::BALANCED;
        } else if (text == "lite") {
            level = Level::LITE;
        } else if (text == "auto") {
            automatic = true;
        } else {
            return false;
        }
        return true;
    }

    Level lighter(Level level) {
        switch (level) {
            case Level::MAX:
                return Level::BALANCED;
            case Level::BALANCED:
                return Level::LITE;
            case Level::LITE:
                return Level::LITE_DISK;
            case Level::LITE_DISK:
                return Level::LITE_HYBRID;
            case Level::LITE_HYBRID:
            case Level::LITE_CPU:
                return Level::LITE_CPU;
        }
        return Level::LITE_CPU;
    }

    bool may_use_several_gpus(const SDContextParams& ctx) {
        return ctx.auto_fit && ctx.backend.empty() && ctx.params_backend.empty() && !multi_gpu_off(ctx.multi_gpu);
    }

    int64_t usable_gpu_memory(const std::vector<sd_device_info_t>& devices, bool pool, int* count) {
        if (count != nullptr) {
            *count = 0;
        }
        // An integrated GPU draws on the system RAM, so only discrete GPUs and accelerators count.
        const sd_device_info_t* best = nullptr;
        for (const sd_device_info_t& device : devices) {
            const bool counts = device.type == SD_DEVICE_TYPE_GPU || device.type == SD_DEVICE_TYPE_ACCEL;
            if (counts && (best == nullptr || device.free_memory_bytes > best->free_memory_bytes)) {
                best = &device;
            }
        }
        if (best == nullptr || best->free_memory_bytes == 0) {
            return 0;
        }

        int64_t total = static_cast<int64_t>(best->free_memory_bytes);
        int gpus      = 1;
        if (pool && best->type == SD_DEVICE_TYPE_GPU) {
            const std::string family = backend_family(best->name);
            int64_t sum              = 0;
            int members              = 0;
            for (const sd_device_info_t& device : devices) {
                if (device.type == SD_DEVICE_TYPE_GPU && device.free_memory_bytes > 0 && backend_family(device.name) == family) {
                    sum += static_cast<int64_t>(device.free_memory_bytes);
                    ++members;
                }
            }
            const int64_t pooled = sum - static_cast<int64_t>(members - 1) * EXTRA_GPU_HEADROOM;
            if (members > 1 && pooled > total) {
                total = pooled;
                gpus  = members;
            }
        }
        if (count != nullptr) {
            *count = gpus;
        }
        return total;
    }

    Resources probe(const SDContextParams& ctx) {
        Resources resources;
        const std::vector<const std::string*> files = {
            &ctx.model_path, &ctx.diffusion_model_path, &ctx.high_noise_diffusion_model_path,
            &ctx.uncond_diffusion_model_path, &ctx.clip_l_path, &ctx.clip_g_path, &ctx.clip_vision_path,
            &ctx.t5xxl_path, &ctx.llm_path, &ctx.llm_vision_path, &ctx.vae_path, &ctx.control_net_path,
            &ctx.ip_adapter_path, &ctx.motion_module_path, &ctx.photo_maker_path, &ctx.pulid_weights_path};
        for (const std::string* path : files) {
            resources.model_bytes += file_size_or_zero(*path);
        }

        sd_system_memory_t memory;
        if (sd_get_system_memory(&memory) && memory.available_bytes > 0) {
            resources.ram_available = static_cast<int64_t>(memory.available_bytes);
        }

        std::vector<sd_device_info_t> devices;
        for (size_t i = 0; i < sd_get_device_count(); ++i) {
            sd_device_info_t info;
            if (sd_get_device_info(i, &info)) {
                devices.push_back(info);
            }
        }
        resources.gpu_free = usable_gpu_memory(devices, may_use_several_gpus(ctx), &resources.gpu_count);
        return resources;
    }

    Level choose(const Resources& resources, std::string* reason) {
        auto explain = [&](const std::string& text) {
            if (reason != nullptr) {
                *reason = text;
            }
        };
        if (resources.model_bytes <= 0) {
            explain("the size of the model is unknown");
            return Level::MAX;
        }
        if (resources.ram_available < 0 && resources.gpu_free <= 0) {
            explain("the memory of this machine is unknown");
            return Level::MAX;
        }

        const int64_t reserve = GiB + resources.model_bytes / 5;
        const int64_t need    = resources.model_bytes + reserve;
        const int64_t ram     = std::max<int64_t>(resources.ram_available, 0);
        const int64_t fast    = resources.gpu_free > 0 ? resources.gpu_free : ram;

        std::string facts = "model " + format_gib(resources.model_bytes) + " + " + format_gib(reserve) + " working memory, ";
        facts += resources.ram_available >= 0 ? "RAM available " + format_gib(ram) : std::string("RAM unknown");
        if (resources.gpu_free > 0) {
            facts += ", GPU free " + format_gib(resources.gpu_free);
            if (resources.gpu_count > 1) {
                facts += " over " + std::to_string(resources.gpu_count) + " GPUs";
            }
        } else {
            facts += ", no GPU memory reported";
        }

        if (need + need / 4 <= fast && (resources.ram_available < 0 || ram >= GiB)) {
            explain(facts + ": fits with room to spare");
            return Level::MAX;
        }
        if (need <= std::max(fast, ram)) {
            explain(facts + ": fits with little margin");
            return Level::BALANCED;
        }
        explain(facts + ": does not fit, weights will be streamed");
        return Level::LITE;
    }

    void apply(Level level, SDContextParams& ctx, SDGenerationParams& gen, bool respect_explicit) {
        auto free_to_set = [&](const char* option) {
            return !respect_explicit || !option_was_given(option);
        };

        if (level >= Level::BALANCED && free_to_set("--memory-guard")) {
            const int target = level >= Level::LITE ? 85 : 90;
            ctx.memory_guard = ctx.memory_guard <= 0 ? target : std::min(ctx.memory_guard, target);
        }
        if (level >= Level::LITE) {
            gen.vae_tiling_params.enabled = true;
            if (free_to_set("--batched-cfg")) {
                ctx.batched_cfg = false;
            }
            if (free_to_set("--multi-gpu") && multi_gpu_unset_or_auto(ctx.multi_gpu)) {
                ctx.multi_gpu = "on";
            }
        }
        if (level >= Level::LITE_DISK && free_to_set("--params-backend")) {
            ctx.params_backend        = "disk";
            ctx.offload_params_to_cpu = false;
            ctx.eager_load            = false;
        }
        if (level >= Level::LITE_HYBRID &&
            (free_to_set("--backend") || ctx.backend.empty()) &&
            !mentions_module(ctx.backend, "te") && !mentions_module(ctx.backend, "clip")) {
            ctx.clip_on_cpu = true;
        }
        if (level >= Level::LITE_CPU && free_to_set("--backend")) {
            ctx.backend         = "cpu";
            ctx.clip_on_cpu     = false;
            ctx.vae_on_cpu      = false;
            ctx.control_net_cpu = false;
        }
    }

    bool apply_from_options(SDContextParams& ctx, SDGenerationParams& gen, Level& applied) {
        applied = Level::MAX;
        if (ctx.profile.empty()) {
            return true;
        }
        Level level    = Level::MAX;
        bool automatic = false;
        if (!parse(ctx.profile, level, automatic)) {
            LOG_ERROR("error: --profile must be one of max, balanced, lite, auto (got '%s')", ctx.profile.c_str());
            return false;
        }
        if (automatic) {
            std::string reason;
            level = choose(probe(ctx), &reason);
            LOG_INFO("profile auto -> %s (%s)", level_name(level), reason.c_str());
        } else {
            LOG_INFO("profile %s", level_name(level));
        }
        apply(level, ctx, gen, /*respect_explicit=*/true);
        applied = level;
        return true;
    }

}  // namespace sd_profile
