// Test of the resource profiles (examples/common/profile.*): names, the order of the steps, the choice
// made from the size of the model and the free memory, and which options a profile may set.
// Build (from the repository root, after building the library into build/):
//   cc -c thirdparty/zip.c -o zip.o
//   c++ -std=c++17 -Iexamples -Iinclude -Ithirdparty -Isrc test/profile_test.cpp examples/common/common.cpp \
//       examples/common/log.cpp examples/common/media_io.cpp examples/common/media_metadata.cpp \
//       examples/common/generation_metadata.cpp examples/common/profile.cpp zip.o \
//       -Lbuild/bin -lstable-diffusion -Wl,-rpath,build/bin -pthread -o profile_test
// Run: ./profile_test ; exit code 0 = all checks passed.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "common/common.h"
#include "common/profile.h"

using sd_profile::Level;

static int failures = 0;
#define CHECK(cond, msg)               \
    do {                               \
        if (!(cond)) {                 \
            printf("FAIL: %s\n", msg); \
            failures++;                \
        } else {                       \
            printf("ok:   %s\n", msg); \
        }                              \
    } while (0)

static constexpr int64_t GiB = 1024ll * 1024ll * 1024ll;

static sd_profile::Resources resources(double model_gib, double ram_gib, double gpu_gib) {
    sd_profile::Resources r;
    r.model_bytes   = static_cast<int64_t>(model_gib * GiB);
    r.ram_available = ram_gib < 0 ? -1 : static_cast<int64_t>(ram_gib * GiB);
    r.gpu_free      = static_cast<int64_t>(gpu_gib * GiB);
    return r;
}

static sd_device_info_t device(const char* name, sd_device_type_t type, double free_gib) {
    sd_device_info_t info = {};
    snprintf(info.name, sizeof(info.name), "%s", name);
    info.type               = type;
    info.free_memory_bytes  = static_cast<uint64_t>(free_gib * GiB);
    info.total_memory_bytes = info.free_memory_bytes;
    return info;
}

static bool gpu_memory_is(const std::vector<sd_device_info_t>& devices, bool pool, double gib, int expected_count) {
    int count            = -1;
    const int64_t memory = sd_profile::usable_gpu_memory(devices, pool, &count);
    return memory == static_cast<int64_t>(gib * GiB) && count == expected_count;
}

static void check_choice(const char* label, const sd_profile::Resources& r, Level expected) {
    std::string reason;
    const Level got = sd_profile::choose(r, &reason);
    char text[160];
    snprintf(text, sizeof(text), "choose: %s -> %s", label, sd_profile::level_name(expected));
    CHECK(got == expected && !reason.empty(), text);
}

// Parses a command line into fresh parameters, the way sd-cli and sd-server do.
struct Parsed {
    SDContextParams ctx;
    SDGenerationParams gen;
    bool ok = false;

    explicit Parsed(std::vector<const char*> args) {
        args.insert(args.begin(), "test");
        std::vector<ArgOptions> options = {ctx.get_options(), gen.get_options()};
        ok                              = parse_options(static_cast<int>(args.size()), args.data(), options);
    }
};

int main() {
    CHECK(strcmp(sd_profile::level_name(Level::MAX), "max") == 0 &&
              strcmp(sd_profile::level_name(Level::BALANCED), "balanced") == 0 &&
              strcmp(sd_profile::level_name(Level::LITE), "lite") == 0 &&
              strcmp(sd_profile::level_name(Level::LITE_DISK), "lite-disk") == 0 &&
              strcmp(sd_profile::level_name(Level::LITE_HYBRID), "lite-hybrid") == 0 &&
              strcmp(sd_profile::level_name(Level::LITE_CPU), "lite-cpu") == 0,
          "level names");

    Level level    = Level::MAX;
    bool automatic = true;
    CHECK(sd_profile::parse("lite", level, automatic) && level == Level::LITE && !automatic, "parse lite");
    CHECK(sd_profile::parse("balanced", level, automatic) && level == Level::BALANCED && !automatic, "parse balanced");
    CHECK(sd_profile::parse("max", level, automatic) && level == Level::MAX && !automatic, "parse max");
    CHECK(sd_profile::parse("auto", level, automatic) && automatic, "parse auto");
    CHECK(!sd_profile::parse("lite-disk", level, automatic) && !sd_profile::parse("lite-hybrid", level, automatic) &&
              !sd_profile::parse("lite-cpu", level, automatic) && !sd_profile::parse("", level, automatic) &&
              !sd_profile::parse("LITE", level, automatic) && !sd_profile::parse("fast", level, automatic),
          "the steps below lite and unknown names are not accepted");

    CHECK(sd_profile::lighter(Level::MAX) == Level::BALANCED && sd_profile::lighter(Level::BALANCED) == Level::LITE &&
              sd_profile::lighter(Level::LITE) == Level::LITE_DISK && sd_profile::lighter(Level::LITE_DISK) == Level::LITE_HYBRID &&
              sd_profile::lighter(Level::LITE_HYBRID) == Level::LITE_CPU && sd_profile::lighter(Level::LITE_CPU) == Level::LITE_CPU,
          "lighter walks down the steps, the CPU last, and stops at the last one");

    check_choice("unknown model size", resources(0, 8, 0), Level::MAX);
    check_choice("unknown memory", resources(2, -1, 0), Level::MAX);
    check_choice("SD 1.5 f16, 8 GiB RAM", resources(2, 8, 0), Level::MAX);
    check_choice("SD 1.5 f16, 3.6 GiB RAM", resources(2, 3.6, 0), Level::BALANCED);
    check_choice("SD 1.5 f16, 3 GiB RAM", resources(2, 3, 0), Level::LITE);
    check_choice("SDXL f16, 16 GiB RAM", resources(6.9, 16, 0), Level::MAX);
    check_choice("SDXL f16, 10 GiB RAM", resources(6.9, 10, 0), Level::BALANCED);
    check_choice("SDXL f16, 5 GiB RAM, 5.5 GiB VRAM", resources(6.9, 5, 5.5), Level::LITE);
    check_choice("SDXL Q8, 5 GiB RAM, 6 GiB VRAM", resources(3.7, 5, 6), Level::BALANCED);
    check_choice("SDXL f16, 8 GiB RAM, 24 GiB VRAM", resources(6.9, 8, 24), Level::MAX);
    check_choice("fits the GPU but RAM is almost gone", resources(2, 0.5, 12), Level::BALANCED);

    {
        // 14 GiB of weights: too much for one 12 GiB GPU, enough for two of them together.
        check_choice("14 GiB model, one 12 GiB GPU", resources(14, 8, 12), Level::LITE);
        sd_profile::Resources two = resources(14, 8, 22);
        two.gpu_count             = 2;
        std::string reason;
        CHECK(sd_profile::choose(two, &reason) == Level::BALANCED && reason.find("over 2 GPUs") != std::string::npos,
              "two GPUs together can hold weights that one cannot, and the reason says how many were counted");
        sd_profile::Resources roomy = resources(14, 8, 28);
        roomy.gpu_count             = 2;
        CHECK(sd_profile::choose(roomy, &reason) == Level::MAX, "and with room to spare the profile stays at max");
        sd_profile::choose(resources(2, 8, 0), &reason);
        CHECK(reason.find("no GPU memory reported") != std::string::npos, "the reason says when there is no GPU");
    }
    {
        const std::vector<sd_device_info_t> none;
        CHECK(gpu_memory_is(none, true, 0, 0), "no device: no GPU memory");
        CHECK(gpu_memory_is({device("CPU", SD_DEVICE_TYPE_CPU, 8)}, true, 0, 0), "the CPU is not a GPU");
        CHECK(gpu_memory_is({device("CUDA0", SD_DEVICE_TYPE_GPU, 6)}, true, 6, 1) && gpu_memory_is({device("CUDA0", SD_DEVICE_TYPE_GPU, 6)}, false, 6, 1),
              "one GPU: its own free memory, pooled or not");
        const std::vector<sd_device_info_t> two = {device("CUDA0", SD_DEVICE_TYPE_GPU, 12), device("CUDA1", SD_DEVICE_TYPE_GPU, 12)};
        CHECK(gpu_memory_is(two, false, 12, 1), "two GPUs, not pooled: the one with the most");
        CHECK(gpu_memory_is(two, true, 22, 2), "two GPUs, pooled: the sum less 2 GiB of working memory for the second one");
        const std::vector<sd_device_info_t> three = {device("CUDA0", SD_DEVICE_TYPE_GPU, 8), device("CUDA1", SD_DEVICE_TYPE_GPU, 8), device("CUDA2", SD_DEVICE_TYPE_GPU, 8)};
        CHECK(gpu_memory_is(three, true, 20, 3), "three GPUs: 2 GiB less for every GPU after the first");
        CHECK(gpu_memory_is({device("CUDA0", SD_DEVICE_TYPE_GPU, 12), device("CUDA1", SD_DEVICE_TYPE_GPU, 1)}, true, 12, 1),
              "a GPU too small to be worth its working memory adds nothing");
        CHECK(gpu_memory_is({device("CUDA0", SD_DEVICE_TYPE_GPU, 8), device("Vulkan0", SD_DEVICE_TYPE_IGPU, 30)}, true, 8, 1),
              "an integrated GPU draws on the RAM and does not count");
        CHECK(gpu_memory_is({device("CUDA0", SD_DEVICE_TYPE_GPU, 12), device("Vulkan0", SD_DEVICE_TYPE_GPU, 12)}, true, 12, 1),
              "GPUs of different backends are not added together");
        CHECK(gpu_memory_is({device("ACC0", SD_DEVICE_TYPE_ACCEL, 16), device("CUDA0", SD_DEVICE_TYPE_GPU, 8), device("CUDA1", SD_DEVICE_TYPE_GPU, 8)}, true, 16, 1),
              "an accelerator is counted alone");
        CHECK(gpu_memory_is({device("CUDA0", SD_DEVICE_TYPE_GPU, 0)}, true, 0, 0), "a GPU that reports no memory counts as none");
    }
    {
        Parsed p({});
        CHECK(p.ok && sd_profile::may_use_several_gpus(p.ctx), "by default auto-fit may use several GPUs");
        Parsed off({"--multi-gpu", "OFF"});
        CHECK(off.ok && !sd_profile::may_use_several_gpus(off.ctx), "--multi-gpu off keeps one GPU");
        Parsed on({"--multi-gpu", "on"});
        CHECK(on.ok && on.ctx.multi_gpu == "on" && sd_profile::may_use_several_gpus(on.ctx), "--multi-gpu on is read");
        Parsed manual({"--backend", "diffusion=cuda0"});
        CHECK(manual.ok && !sd_profile::may_use_several_gpus(manual.ctx), "a module assigned by hand is left alone");
        Parsed params({"--params-backend", "cpu"});
        CHECK(params.ok && !sd_profile::may_use_several_gpus(params.ctx), "so is a placement of the parameters");
        Parsed no_fit({"--auto-fit", "off"});
        CHECK(no_fit.ok && !sd_profile::may_use_several_gpus(no_fit.ctx), "without auto-fit nothing is planned");
    }
    {
        Parsed good({"--multi-gpu", "Auto"});
        CHECK(good.ok && good.ctx.validate(IMG_GEN, false), "--multi-gpu accepts auto in any case");
        Parsed bad({"--multi-gpu", "sideways"});
        CHECK(bad.ok && !bad.ctx.validate(IMG_GEN, false), "--multi-gpu refuses anything but auto, on and off");
        Parsed passed({"--multi-gpu", "on"});
        const sd_ctx_params_t raw = passed.ctx.to_sd_ctx_params_t();
        CHECK(raw.multi_gpu != nullptr && strcmp(raw.multi_gpu, "on") == 0, "the value reaches the library");
    }
    {
        Parsed p({});
        sd_profile::apply(Level::LITE, p.ctx, p.gen, true);
        CHECK(p.ok && p.ctx.memory_guard == 85 && !p.ctx.batched_cfg && p.gen.vae_tiling_params.enabled,
              "lite sets the guard, unbatched CFG and VAE tiling");
        CHECK(!p.ctx.clip_on_cpu && !p.ctx.vae_on_cpu && !p.ctx.control_net_cpu && p.ctx.backend.empty(),
              "lite keeps every module on the GPU: the text encoder stays where auto-fit puts it");
        CHECK(p.ctx.multi_gpu == "on", "lite spreads the modules over every GPU of the backend");
        sd_profile::apply(Level::LITE, p.ctx, p.gen, true);
        CHECK(p.ctx.memory_guard == 85 && p.ctx.params_backend.empty() && p.ctx.backend.empty() && p.ctx.multi_gpu == "on",
              "applying a level twice changes nothing more, and lite leaves the weights placement to auto-fit");
    }
    {
        Parsed off({"--multi-gpu", "OFF"});
        sd_profile::apply(Level::LITE, off.ctx, off.gen, true);
        CHECK(off.ctx.multi_gpu == "OFF", "lite does not undo --multi-gpu off at startup");
        sd_profile::apply(Level::LITE, off.ctx, off.gen, false);
        CHECK(off.ctx.multi_gpu == "OFF", "nor does a recovery step");
        Parsed automatic({"--multi-gpu", "auto"});
        sd_profile::apply(Level::LITE, automatic.ctx, automatic.gen, true);
        CHECK(automatic.ctx.multi_gpu == "auto", "an explicit --multi-gpu auto is kept at startup");
        sd_profile::apply(Level::LITE, automatic.ctx, automatic.gen, false);
        CHECK(automatic.ctx.multi_gpu == "on", "and a recovery step turns it on");
        Parsed balanced({});
        sd_profile::apply(Level::BALANCED, balanced.ctx, balanced.gen, true);
        CHECK(balanced.ctx.multi_gpu.empty(), "balanced leaves multi-gpu alone");
    }
    {
        bool stays_on_gpu = true;
        for (Level step : {Level::MAX, Level::BALANCED, Level::LITE, Level::LITE_DISK}) {
            for (bool respect_explicit : {true, false}) {
                Parsed p({});
                sd_profile::apply(step, p.ctx, p.gen, respect_explicit);
                stays_on_gpu = stays_on_gpu && !p.ctx.clip_on_cpu && !p.ctx.vae_on_cpu && !p.ctx.control_net_cpu && p.ctx.backend.empty();
            }
        }
        CHECK(stays_on_gpu, "no step down to lite-disk moves a module to the CPU");
    }
    {
        Parsed p({});
        sd_profile::apply(Level::BALANCED, p.ctx, p.gen, true);
        CHECK(p.ctx.memory_guard == 90 && !p.ctx.clip_on_cpu && p.ctx.batched_cfg && !p.gen.vae_tiling_params.enabled,
              "balanced only turns the memory guard on");
    }
    {
        Parsed p({});
        sd_profile::apply(Level::MAX, p.ctx, p.gen, true);
        CHECK(p.ctx.memory_guard == 0 && !p.ctx.clip_on_cpu && p.ctx.batched_cfg, "max changes nothing");
    }
    {
        Parsed p({"--memory-guard", "70", "--batched-cfg", "on"});
        sd_profile::apply(Level::LITE, p.ctx, p.gen, true);
        CHECK(p.ok && p.ctx.memory_guard == 70 && p.ctx.batched_cfg, "options given on the command line win");
        sd_profile::apply(Level::LITE, p.ctx, p.gen, false);
        CHECK(p.ctx.memory_guard == 70 && !p.ctx.batched_cfg, "a recovery step overrides them, but never raises the guard");
    }
    {
        Parsed p({"--memory-guard", "0"});
        sd_profile::apply(Level::LITE, p.ctx, p.gen, true);
        CHECK(p.ctx.memory_guard == 0, "an explicit --memory-guard 0 is respected");
        sd_profile::apply(Level::LITE, p.ctx, p.gen, false);
        CHECK(p.ctx.memory_guard == 85, "a recovery step turns the guard on anyway");
    }
    {
        Parsed p({"--memory-guard", "95"});
        sd_profile::apply(Level::LITE, p.ctx, p.gen, false);
        CHECK(p.ctx.memory_guard == 85, "a recovery step lowers a higher guard");
    }
    {
        Parsed p({"--backend", "te=cuda0"});
        sd_profile::apply(Level::LITE, p.ctx, p.gen, true);
        CHECK(!p.ctx.clip_on_cpu, "an explicit text encoder placement is not overridden by lite");
        sd_profile::apply(Level::LITE_HYBRID, p.ctx, p.gen, false);
        CHECK(!p.ctx.clip_on_cpu, "nor by lite-hybrid, even in a recovery");
    }
    {
        Parsed p({});
        sd_profile::apply(Level::LITE_HYBRID, p.ctx, p.gen, false);
        CHECK(p.ctx.clip_on_cpu && !p.ctx.vae_on_cpu && !p.ctx.control_net_cpu && p.ctx.backend.empty(),
              "lite-hybrid moves only the text encoder to the CPU");
        CHECK(p.ctx.params_backend == "disk" && !p.ctx.eager_load && p.ctx.memory_guard == 85 && !p.ctx.batched_cfg &&
                  p.gen.vae_tiling_params.enabled && p.ctx.multi_gpu == "on",
              "and keeps everything the steps before it set");
        Parsed manual({"--backend", "cuda0"});
        sd_profile::apply(Level::LITE_HYBRID, manual.ctx, manual.gen, true);
        CHECK(!manual.ctx.clip_on_cpu && manual.ctx.backend == "cuda0", "an explicit --backend is kept at startup");
        Parsed clip({"--clip-on-cpu"});
        sd_profile::apply(Level::LITE, clip.ctx, clip.gen, true);
        sd_profile::apply(Level::LITE_DISK, clip.ctx, clip.gen, false);
        CHECK(clip.ctx.clip_on_cpu, "a --clip-on-cpu given on the command line is never undone before lite-cpu");
    }
    {
        Parsed p({"--params-backend", "cpu", "--offload-to-cpu", "--eager-load"});
        sd_profile::apply(Level::LITE_DISK, p.ctx, p.gen, true);
        CHECK(p.ctx.params_backend == "cpu" && p.ctx.offload_params_to_cpu && p.ctx.eager_load,
              "an explicit --params-backend is kept at startup");
        sd_profile::apply(Level::LITE_DISK, p.ctx, p.gen, false);
        CHECK(p.ctx.params_backend == "disk" && !p.ctx.offload_params_to_cpu, "lite-disk streams the weights from disk in a recovery");
        CHECK(!p.ctx.eager_load, "and does not read all of them into memory first");
        CHECK(!p.ctx.clip_on_cpu && p.ctx.backend.empty(), "while the compute stays on the GPU, the text encoder included");
    }
    {
        Parsed p({"--backend", "cuda0"});
        sd_profile::apply(Level::LITE_CPU, p.ctx, p.gen, true);
        CHECK(p.ctx.backend == "cuda0", "an explicit --backend is kept at startup");
        sd_profile::apply(Level::LITE_CPU, p.ctx, p.gen, false);
        CHECK(p.ctx.backend == "cpu" && !p.ctx.clip_on_cpu && p.ctx.params_backend == "disk",
              "lite-cpu computes everything on the CPU in a recovery");
    }
    {
        Parsed p({});
        p.ctx.profile = "bogus";
        Level applied = Level::LITE;
        CHECK(!sd_profile::apply_from_options(p.ctx, p.gen, applied), "an invalid profile name is refused");
        p.ctx.profile.clear();
        CHECK(sd_profile::apply_from_options(p.ctx, p.gen, applied) && applied == Level::MAX && p.ctx.memory_guard == 0,
              "no --profile leaves the options alone");
        p.ctx.profile = "lite";
        CHECK(sd_profile::apply_from_options(p.ctx, p.gen, applied) && applied == Level::LITE && p.ctx.memory_guard == 85 &&
                  !p.ctx.clip_on_cpu,
              "--profile lite is applied, without moving anything to the CPU");
    }

    printf("\n%s (%d failures)\n", failures ? "TEST FAILED" : "ALL TESTS PASSED", failures);
    return failures ? 1 : 0;
}
