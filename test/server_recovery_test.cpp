// Test of the context keeper of the server (examples/server/recovery.*): the ladder of profiles, stepping
// down when a request runs out of memory, climbing back, loading the model again after a failure, the
// HTTP answers for failures, stopping a generation that is running, and use from several threads. No model
// is needed: the library is replaced by scripted answers and the clock by a variable.
// Build (from the repository root, after building the library into build/); the sources go in one command:
//   cc -c thirdparty/zip.c -o zip.o
//   c++ -std=c++17 -Iexamples -Iexamples/server -Iinclude -Ithirdparty -Isrc test/server_recovery_test.cpp
//       examples/server/recovery.cpp examples/common/common.cpp examples/common/log.cpp
//       examples/common/media_io.cpp examples/common/media_metadata.cpp examples/common/generation_metadata.cpp
//       examples/common/profile.cpp zip.o
//       -Lbuild/bin -lstable-diffusion -Wl,-rpath,build/bin -pthread -o server_recovery_test
// Run: ./server_recovery_test ; exit code 0 = all checks passed.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "httplib.h"
#include "recovery.h"

using Level = sd_profile::Level;
using Clock = ContextKeeper::Clock;
using json  = nlohmann::json;

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

static GenerationFailure failure_of(sd_error_code_t code, const char* message) {
    GenerationFailure failure;
    failure.code    = code;
    failure.message = message;
    return failure;
}
static GenerationFailure oom() {
    return failure_of(SD_ERROR_OUT_OF_MEMORY, "unet ran out of memory");
}
static GenerationFailure backend_failure() {
    return failure_of(SD_ERROR_BACKEND, "the backend failed");
}
static GenerationFailure model_file_failure() {
    return failure_of(SD_ERROR_MODEL_FILE, "failed to open the model");
}
static const GenerationFailure ok_result;

// What the keeper asked the library to load, reduced to the options a profile changes.
struct Load {
    int memory_guard;
    bool batched_cfg;
    std::string backend;
    std::string params_backend;
    bool eager_load;
    std::string model_path;
    std::string multi_gpu;
    bool no_fallback;

    bool text_encoder_on_cpu() const {
        return backend.find("te=cpu") != std::string::npos;
    }

    Level level() const {
        if (backend == "cpu") {
            return Level::LITE_CPU;
        }
        if (params_backend == "disk") {
            return text_encoder_on_cpu() ? Level::LITE_HYBRID : Level::LITE_DISK;
        }
        if (memory_guard == 85) {
            return Level::LITE;
        }
        if (memory_guard == 90) {
            return Level::BALANCED;
        }
        return Level::MAX;
    }
};

// A scripted library and clock.
struct Env {
    std::vector<Load> loads;
    std::function<GenerationFailure(const Load&)> load_result;  // code NONE = the model loads
    std::function<void(const Load&)> on_create;                 // runs when the keeper asks the library for a context
    GenerationFailure next_failure;                             // what the library reports for the attempt that failed
    Clock::time_point now = Clock::now();
    bool accelerator      = true;
    // The devices the library lists besides the cpu; empty means that the engine's own choice is all there is.
    std::vector<sd_device_info_t> devices;

    // The contexts the keeper asked the library to stop, in order. on_cancel runs on the thread that called cancel().
    std::mutex cancel_mutex;
    std::vector<sd_ctx_t*> cancel_calls;
    std::function<void(sd_ctx_t*)> on_cancel;

    ContextKeeper::Hooks hooks() {
        ContextKeeper::Hooks hooks;
        hooks.create_context = [this](const sd_ctx_params_t& params, GenerationFailure& failure) -> SDCtxPtr {
            Load load{params.memory_guard,
                      params.batched_cfg,
                      params.backend != nullptr ? params.backend : "",
                      params.params_backend != nullptr ? params.params_backend : "",
                      params.eager_load,
                      params.model_path != nullptr ? params.model_path : "",
                      params.multi_gpu != nullptr ? params.multi_gpu : "",
                      params.disable_backend_fallback};
            loads.push_back(load);
            if (on_create) {
                on_create(load);
            }
            const GenerationFailure result = load_result ? load_result(load) : ok_result;
            if (result.code != SD_ERROR_NONE) {
                failure = result;
                return nullptr;
            }
            // free_sd_ctx() accepts a context that holds no model.
            return SDCtxPtr(static_cast<sd_ctx_t*>(calloc(1, 64)));
        };
        hooks.read_failure    = [this] { return next_failure; };
        hooks.has_accelerator = [this] { return accelerator; };
        hooks.list_devices    = [this] { return devices; };
        hooks.now             = [this] { return now; };
        hooks.cancel_context  = [this](sd_ctx_t* ctx) {
            // Touching the memory of a context the keeper no longer owns is what the address sanitizer reports.
            reinterpret_cast<volatile char*>(ctx)[63] = 1;
            std::lock_guard<std::mutex> lock(cancel_mutex);
            cancel_calls.push_back(ctx);
            if (on_cancel) {
                on_cancel(ctx);
            }
        };
        return hooks;
    }

    void advance(int seconds) {
        now += std::chrono::seconds(seconds);
    }
};

struct Seen {
    std::vector<Level> levels;
    std::vector<bool> tiled;
    std::vector<sd_ctx_t*> contexts;
};

using Script = std::function<GenerationFailure(const ContextKeeper::ContextView&)>;

static bool run_with(ContextKeeper& keeper,
                     Env& env,
                     Seen& seen,
                     const Script& script,
                     GenerationFailure& failure,
                     const CancelToken* token = nullptr) {
    return keeper.run(
        [&](const ContextKeeper::ContextView& view) {
            seen.levels.push_back(view.level);
            seen.tiled.push_back(view.force_vae_tiling);
            seen.contexts.push_back(view.ctx);
            const GenerationFailure result = script(view);
            if (result.code == SD_ERROR_NONE) {
                return true;
            }
            env.next_failure = result;
            return false;
        },
        failure,
        token);
}

static SDContextParams base_params() {
    SDContextParams params;
    params.model_path = "model.safetensors";
    params.n_threads  = 2;
    return params;
}

static ContextKeeper::Settings settings(bool degradation = true, int cooldown = 10) {
    ContextKeeper::Settings s;
    s.degradation      = degradation;
    s.cooldown_seconds = cooldown;
    return s;
}

static int64_t counter(const json& health, const char* name) {
    return health["counters"][name].get<int64_t>();
}

static const GenerationFailure succeed(const ContextKeeper::ContextView&) {
    return ok_result;
}

static void test_ladder() {
    Env env;
    {
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        CHECK(keeper.describe_ladder() == "max > balanced > lite > lite-disk > lite-hybrid > lite-cpu", "ladder with an accelerator");
        CHECK(keeper.can_step_down(), "a full ladder can step down");
    }
    env.accelerator = false;
    {
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        CHECK(keeper.describe_ladder() == "max > balanced > lite > lite-disk",
              "without an accelerator the cpu has nothing to take over: no hybrid and no cpu profile");
    }

    // The configuration of a host that already asks for most of it: steps that change nothing are left out.
    env.accelerator              = true;
    SDContextParams explicit_set = base_params();
    explicit_set.memory_guard    = 90;
    explicit_set.clip_on_cpu     = true;
    explicit_set.params_backend  = "disk";
    {
        ContextKeeper keeper(explicit_set, Level::MAX, settings(), env.hooks());
        CHECK(keeper.describe_ladder() == "max > lite > lite-cpu", "profiles that would load the same configuration are skipped");
    }
    {
        // Weights already streamed from disk, text encoder still on the GPU: the hybrid profile is the step this host lacks.
        SDContextParams disk_only = base_params();
        disk_only.memory_guard    = 90;
        disk_only.params_backend  = "disk";
        ContextKeeper keeper(disk_only, Level::MAX, settings(), env.hooks());
        CHECK(keeper.describe_ladder() == "max > lite > lite-hybrid > lite-cpu",
              "lite-hybrid is kept when it is the only change of placement left");
    }
    {
        ContextKeeper keeper(base_params(), Level::LITE, settings(), env.hooks());
        CHECK(keeper.describe_ladder() == "lite > lite-disk > lite-hybrid > lite-cpu", "the ladder starts at the profile the server was started with");
    }
    {
        ContextKeeper keeper(base_params(), Level::LITE_HYBRID, settings(), env.hooks());
        CHECK(keeper.describe_ladder() == "lite-hybrid > lite-cpu" && keeper.can_step_down(), "the cpu profile comes after the hybrid one");
    }
    {
        // Reading every weight into memory at load time is what streaming them from disk avoids.
        SDContextParams eager = explicit_set;
        eager.eager_load      = true;
        ContextKeeper keeper(eager, Level::MAX, settings(), env.hooks());
        CHECK(keeper.describe_ladder() == "max > lite > lite-disk > lite-cpu",
              "lite-disk is kept when it only differs by not loading every weight at start");
    }
    {
        ContextKeeper keeper(base_params(), Level::LITE_CPU, settings(), env.hooks());
        CHECK(keeper.describe_ladder() == "lite-cpu" && !keeper.can_step_down(), "nothing is lighter than the cpu profile");
    }
    {
        ContextKeeper keeper(base_params(), Level::MAX, settings(false), env.hooks());
        CHECK(!keeper.can_step_down(), "no stepping down when degradation is disabled");
    }
}

static void test_start() {
    {
        Env env;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        CHECK(!keeper.has_model(), "no model before start");
        CHECK(keeper.status_name() == std::string("no_model"), "status no_model before start");
        CHECK(keeper.start(failure), "start loads the model");
        CHECK(keeper.has_model(), "model after start");
        CHECK(env.loads.size() == 1 && env.loads[0].level() == Level::MAX, "one load, with the profile of the server");
        const json health = keeper.health_json();
        CHECK(health["status"] == "ok" && health["profile"] == "max" && health["base_profile"] == "max", "healthy after start");
        CHECK(health["next_promotion_seconds"].is_null(), "nothing to climb back to at the base profile");
        CHECK(health["last_error"].is_null(), "no error yet");
        CHECK(health["model_loaded"] == true && health["degradation"] == true, "model_loaded and degradation reported");
    }
    {
        Env env;
        env.load_result = [](const Load& load) { return load.level() < Level::LITE ? oom() : ok_result; };
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        CHECK(keeper.start(failure), "start steps down when the memory is not enough");
        CHECK(env.loads.size() == 3 && env.loads[2].level() == Level::LITE, "max, balanced and then lite were tried");
        const json health = keeper.health_json();
        CHECK(health["status"] == "degraded" && health["profile"] == "lite" && health["base_profile"] == "max",
              "degraded after a start that had to step down");
        CHECK(health["next_promotion_seconds"] == 10, "the climb back is scheduled");
        CHECK(health["last_error"]["code"] == "out_of_memory", "the lack of memory is the last error");
    }
    {
        // The weights cannot be streamed from disk if the load reads all of them into memory first.
        Env env;
        env.load_result       = [](const Load& load) { return load.eager_load ? oom() : ok_result; };
        SDContextParams eager = base_params();
        eager.eager_load      = true;
        ContextKeeper keeper(eager, Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        CHECK(keeper.start(failure) && env.loads.size() == 4 && env.loads[3].level() == Level::LITE_DISK && !env.loads[3].eager_load,
              "a server started with --eager-load reaches the disk profile without the eager read");
        CHECK(env.loads[0].eager_load && env.loads[2].eager_load, "the lighter profiles above it keep the option");
    }
    {
        // The cpu takes work only when no level that keeps the compute on the GPUs is left.
        Env env;
        env.load_result = [](const Load& load) { return load.level() < Level::LITE_HYBRID ? oom() : ok_result; };
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        CHECK(keeper.start(failure) && env.loads.size() == 5 && env.loads[4].level() == Level::LITE_HYBRID,
              "a model that does not fit is tried at every level that stays on the GPUs before the cpu is involved");
        bool cpu_early = false;
        for (size_t i = 0; i < 4; ++i) {
            cpu_early = cpu_early || !env.loads[i].backend.empty();
        }
        CHECK(!cpu_early, "no level before lite-hybrid places a module on the cpu");
        CHECK(env.loads[4].backend == "te=cpu", "lite-hybrid gives the cpu the text encoder and nothing else");
        CHECK(env.loads[2].multi_gpu == "on" && env.loads[3].multi_gpu == "on" && env.loads[4].multi_gpu == "on",
              "from lite on every GPU of the backend may take part");
        CHECK(env.loads[0].multi_gpu.empty() && env.loads[1].multi_gpu.empty(), "max and balanced leave the choice of GPUs to the engine");
    }
    {
        Env env;
        env.load_result = [](const Load&) { return model_file_failure(); };
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        CHECK(!keeper.start(failure) && env.loads.size() == 1, "a damaged model file is not a reason to step down");
        CHECK(failure.code == SD_ERROR_MODEL_FILE, "the failure of the start is reported");
    }
    {
        Env env;
        env.load_result = [](const Load&) { return oom(); };
        ContextKeeper keeper(base_params(), Level::MAX, settings(false), env.hooks());
        GenerationFailure failure;
        CHECK(!keeper.start(failure) && env.loads.size() == 1, "with degradation disabled the start gives up at once");
    }
    {
        Env env;
        env.load_result = [](const Load& load) { return load.level() == Level::MAX ? backend_failure() : ok_result; };
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        CHECK(keeper.start(failure) && env.loads.size() == 3 && env.loads[1].level() == Level::MAX && env.loads[2].level() == Level::LITE_CPU,
              "a backend failure at start tries the same profile once more and, with no other device to use, goes to the cpu profile");
    }
    {
        Env env;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        Seen seen;
        GenerationFailure failure;
        CHECK(!run_with(keeper, env, seen, succeed, failure) && failure.code == SD_ERROR_INVALID_ARGUMENT,
              "running without a model is refused");
    }
}

static void test_step_down() {
    {
        Env env;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        CHECK(run_with(keeper, env, seen, succeed, failure), "a request that fits succeeds");
        CHECK(seen.levels.size() == 1 && seen.levels[0] == Level::MAX && !seen.tiled[0] && seen.contexts[0] != nullptr,
              "it ran once, at max, without forced tiling");
        const json health = keeper.health_json();
        CHECK(counter(health, "requests") == 1 && counter(health, "succeeded") == 1 && counter(health, "step_downs") == 0,
              "counted as a plain success");
        CHECK(health["busy"] == false, "not busy once the request is over");
    }
    {
        Env env;
        env.accelerator = false;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        // Fits only from the lite-disk profile on.
        const bool ok = run_with(keeper, env, seen, [](const ContextKeeper::ContextView& view) { return view.level < Level::LITE_DISK ? oom() : ok_result; }, failure);
        CHECK(ok, "the request succeeds after stepping down");
        CHECK(seen.levels == std::vector<Level>({Level::MAX, Level::BALANCED, Level::LITE, Level::LITE_DISK}),
              "the same request was run at each profile in turn");
        CHECK(seen.tiled == std::vector<bool>({false, false, true, true}), "tiling is forced from the lite profile on");
        CHECK(seen.contexts[0] != nullptr && seen.contexts[1] != nullptr, "every attempt had a context");
        CHECK(env.loads.size() == 4, "the model was loaded three more times");
        const json health = keeper.health_json();
        CHECK(counter(health, "step_downs") == 3 && counter(health, "recovered") == 1 && counter(health, "rebuilds") == 3 &&
                  counter(health, "failed") == 0,
              "counters of a request that recovered");
        CHECK(health["status"] == "degraded" && health["profile"] == "lite-disk", "the server stays on the profile that worked");
        CHECK(health["next_promotion_seconds"] == 10, "the climb back is scheduled");
        CHECK(health["last_error"]["code"] == "out_of_memory" && health["last_error"]["message"] == "unet ran out of memory",
              "the last error is kept");

        Seen second;
        CHECK(run_with(keeper, env, second, succeed, failure) && second.levels == std::vector<Level>({Level::LITE_DISK}),
              "the next request runs on the profile that worked, with no rebuild");
        CHECK(env.loads.size() == 4, "no extra load for it");
    }
    {
        Env env;
        env.accelerator = false;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        CHECK(!run_with(keeper, env, seen, [](const ContextKeeper::ContextView&) { return oom(); }, failure), "a request that fits nowhere fails");
        CHECK(failure.code == SD_ERROR_OUT_OF_MEMORY && !failure.unavailable && failure.message == "unet ran out of memory",
              "with the out of memory reason, not as unavailable");
        CHECK(seen.levels.size() == 4, "every profile was tried");
        bool moved_to_cpu = false;
        for (const Load& load : env.loads) {
            moved_to_cpu = moved_to_cpu || load.text_encoder_on_cpu() || load.backend == "cpu";
        }
        CHECK(!moved_to_cpu, "without an accelerator no load moves a module to the cpu: it is the only device");
        const json health = keeper.health_json();
        CHECK(health["status"] == "degraded" && health["profile"] == "lite-disk", "the server keeps the lightest context");
        CHECK(counter(health, "failed") == 1, "one failed request");
        Seen later;
        CHECK(run_with(keeper, env, later, succeed, failure), "a smaller request is served afterwards");
    }
    {
        Env env;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        CHECK(!run_with(keeper, env, seen, [](const ContextKeeper::ContextView&) { return failure_of(SD_ERROR_INVALID_ARGUMENT, "bad size"); }, failure), "an invalid request fails");
        CHECK(failure.code == SD_ERROR_INVALID_ARGUMENT && seen.levels.size() == 1 && env.loads.size() == 1,
              "without stepping down");
        const json health = keeper.health_json();
        CHECK(health["last_error"].is_null(), "a mistake of the client is not an error of the server");
        CHECK(counter(health, "failed") == 1, "but it is a failed request");
    }
    {
        // A request that keeps running out of memory is served by the GPUs for as long as a level exists that lets them.
        Env env;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        const bool ok = run_with(keeper, env, seen, [](const ContextKeeper::ContextView& view) { return view.level < Level::LITE_HYBRID ? oom() : ok_result; }, failure);
        CHECK(ok && seen.levels == std::vector<Level>({Level::MAX, Level::BALANCED, Level::LITE, Level::LITE_DISK, Level::LITE_HYBRID}),
              "the request is run at each level in turn, the one that uses the cpu last");
        CHECK(seen.tiled == std::vector<bool>({false, false, true, true, true}), "tiling is forced from the lite profile on");
        bool cpu_early = false;
        for (size_t i = 0; i + 1 < env.loads.size(); ++i) {
            cpu_early = cpu_early || !env.loads[i].backend.empty();
        }
        CHECK(!cpu_early, "no load before the last one places a module on the cpu");
        const json health = keeper.health_json();
        CHECK(health["status"] == "degraded" && health["profile"] == "lite-hybrid", "the server stays on the profile that worked");
    }
    {
        Env env;
        ContextKeeper keeper(base_params(), Level::MAX, settings(false), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        CHECK(!run_with(keeper, env, seen, [](const ContextKeeper::ContextView&) { return oom(); }, failure), "with degradation disabled a request out of memory fails");
        CHECK(failure.code == SD_ERROR_OUT_OF_MEMORY && seen.levels.size() == 1 && env.loads.size() == 1,
              "after one attempt and without loading the model again");
        const json health = keeper.health_json();
        CHECK(health["status"] == "ok" && health["profile"] == "max" && health["degradation"] == false, "the server stays as it was");
        CHECK(health["last_error"]["code"] == "out_of_memory", "and reports why");
    }
    {
        Env env;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        const bool ok = run_with(keeper, env, seen, [](const ContextKeeper::ContextView& view) { return view.level == Level::MAX ? backend_failure() : ok_result; }, failure);
        CHECK(ok && seen.levels == std::vector<Level>({Level::MAX, Level::MAX, Level::LITE_CPU}),
              "a backend failure is tried once more on the same profile and, with no other device to use, then goes to the cpu profile");
        CHECK(env.loads.back().backend == "cpu", "which loads the model on the cpu");
        CHECK(keeper.backend_choice().backend == "cpu", "and the upscaler follows");
    }
}

static void test_promotion() {
    {
        Env env;
        env.accelerator = false;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        run_with(keeper, env, seen, [](const ContextKeeper::ContextView& view) { return view.level < Level::LITE ? oom() : ok_result; }, failure);
        const size_t loads = env.loads.size();

        env.advance(5);
        CHECK(keeper.health_json()["next_promotion_seconds"] == 5, "the time left is reported");
        Seen early;
        run_with(keeper, env, early, succeed, failure);
        CHECK(env.loads.size() == loads && early.levels == std::vector<Level>({Level::LITE}), "no climb before the wait is over");

        env.advance(6);
        Seen climb;
        CHECK(run_with(keeper, env, climb, succeed, failure), "a request after the wait succeeds");
        CHECK(climb.levels == std::vector<Level>({Level::BALANCED}) && env.loads.size() == loads + 1,
              "on the next heavier profile, loaded for it");
        json health = keeper.health_json();
        CHECK(counter(health, "promotions") == 1 && counter(health, "failed_promotions") == 0, "a promotion is counted");
        CHECK(health["status"] == "degraded" && health["profile"] == "balanced", "one step at a time");

        env.advance(11);
        Seen top;
        run_with(keeper, env, top, succeed, failure);
        health = keeper.health_json();
        CHECK(top.levels == std::vector<Level>({Level::MAX}) && health["status"] == "ok" && health["profile"] == "max",
              "back at the profile the server was started with");
        CHECK(health["next_promotion_seconds"].is_null() && counter(health, "promotions") == 2, "and nothing left to climb");
        env.advance(1000);
        Seen quiet;
        run_with(keeper, env, quiet, succeed, failure);
        CHECK(quiet.levels == std::vector<Level>({Level::MAX}), "it never goes above the base profile");
    }
    {
        Env env;
        env.accelerator = false;
        ContextKeeper keeper(base_params(), Level::MAX, settings(true, 0), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        run_with(keeper, env, seen, [](const ContextKeeper::ContextView& view) { return view.level < Level::BALANCED ? oom() : ok_result; }, failure);
        env.advance(100000);
        Seen later;
        run_with(keeper, env, later, succeed, failure);
        CHECK(later.levels == std::vector<Level>({Level::BALANCED}), "a cooldown of 0 never climbs back");
        CHECK(keeper.health_json()["next_promotion_seconds"].is_null(), "and reports no climb");
    }
    {
        // A climb that fails: the first request on the heavier profile runs out of memory.
        Env env;
        env.accelerator = false;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        const Script needs_balanced = [](const ContextKeeper::ContextView& view) {
            return view.level < Level::BALANCED ? oom() : ok_result;
        };
        run_with(keeper, env, seen, needs_balanced, failure);
        env.advance(11);
        Seen failed_climb;
        CHECK(run_with(keeper, env, failed_climb, needs_balanced, failure), "the request is served even if the climb fails");
        CHECK(failed_climb.levels == std::vector<Level>({Level::MAX, Level::BALANCED}), "after going back down");
        json health = keeper.health_json();
        CHECK(counter(health, "promotions") == 1 && counter(health, "failed_promotions") == 1, "a failed climb is counted");
        CHECK(health["next_promotion_seconds"] == 20, "and the wait doubles");
        env.advance(21);
        Seen again;
        run_with(keeper, env, again, needs_balanced, failure);
        CHECK(keeper.health_json()["next_promotion_seconds"] == 40, "again after another failed climb");

        // A climb that holds puts the wait back to its start.
        env.advance(41);
        Seen holds;
        run_with(keeper, env, holds, succeed, failure);
        CHECK(holds.levels == std::vector<Level>({Level::MAX}), "a later climb that succeeds reaches max");
        Seen drop;
        run_with(keeper, env, drop, needs_balanced, failure);
        CHECK(keeper.health_json()["next_promotion_seconds"] == 10, "the wait starts again from the configured one");
    }
    {
        Env env;
        env.accelerator = false;
        ContextKeeper keeper(base_params(), Level::MAX, settings(true, 2000), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        const Script needs_balanced = [](const ContextKeeper::ContextView& view) {
            return view.level < Level::BALANCED ? oom() : ok_result;
        };
        run_with(keeper, env, seen, needs_balanced, failure);
        env.advance(2001);
        run_with(keeper, env, seen, needs_balanced, failure);
        CHECK(keeper.health_json()["next_promotion_seconds"] == 3600, "the wait stops doubling at an hour");
    }
    {
        // The wait counts from the end of a request that had to step down, however long that took.
        Env env;
        env.accelerator = false;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        run_with(keeper, env, seen, [&env](const ContextKeeper::ContextView& view) {
            env.advance(100);
            return view.level < Level::BALANCED ? oom() : ok_result; }, failure);
        CHECK(keeper.health_json()["next_promotion_seconds"] == 10, "a slow request does not use up the wait");
    }
    {
        // The heavier profile cannot be loaded because of the memory: the climb falls back to the one before.
        Env env;
        env.accelerator = false;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        run_with(keeper, env, seen, [](const ContextKeeper::ContextView& view) { return view.level < Level::LITE ? oom() : ok_result; }, failure);
        env.load_result = [](const Load& load) { return load.level() == Level::BALANCED ? oom() : ok_result; };
        env.advance(11);
        Seen climb;
        CHECK(run_with(keeper, env, climb, succeed, failure) && climb.levels == std::vector<Level>({Level::LITE}),
              "a climb that cannot load the profile serves the request one step down");
        const json health = keeper.health_json();
        CHECK(counter(health, "failed_promotions") == 1 && health["next_promotion_seconds"] == 20, "counted as a failed climb");
    }
    {
        // The heavier profile fails for a reason that is not the memory: the profile in use before it comes back.
        Env env;
        env.accelerator = false;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        run_with(keeper, env, seen, [](const ContextKeeper::ContextView& view) { return view.level < Level::LITE ? oom() : ok_result; }, failure);
        env.load_result = [](const Load& load) { return load.level() == Level::BALANCED ? model_file_failure() : ok_result; };
        env.advance(11);
        Seen climb;
        CHECK(run_with(keeper, env, climb, succeed, failure) && climb.levels == std::vector<Level>({Level::LITE}),
              "the request is served on the previous profile");
    }
}

static void test_unavailable() {
    Env env;
    env.accelerator = false;
    ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
    GenerationFailure failure;
    keeper.start(failure);
    env.load_result = [](const Load& load) { return load.level() == Level::BALANCED ? model_file_failure() : ok_result; };
    Seen seen;
    CHECK(!run_with(keeper, env, seen, [](const ContextKeeper::ContextView&) { return oom(); }, failure), "a request fails when the model cannot be loaded again");
    CHECK(failure.unavailable && failure.code == SD_ERROR_MODEL_FILE && failure.retry_after_seconds > 0,
          "as unavailable, with the cause and a time to retry");
    json health = keeper.health_json();
    CHECK(health["status"] == "unavailable", "the health says so");
    CHECK(health["last_error"]["code"] == "model_file", "and why");

    const size_t loads = env.loads.size();
    Seen blocked;
    CHECK(!run_with(keeper, env, blocked, succeed, failure) && failure.unavailable && blocked.levels.empty(),
          "requests wait instead of trying again at once");
    CHECK(env.loads.size() == loads && failure.retry_after_seconds >= 1 && failure.retry_after_seconds <= 16,
          "no load, and the time left is reported");

    env.advance(16);
    env.load_result = nullptr;
    Seen restored;
    CHECK(run_with(keeper, env, restored, succeed, failure), "the model is loaded again after the wait");
    CHECK(env.loads.size() == loads + 1 && env.loads.back().level() == Level::BALANCED, "from the profile it had stepped down to");
    CHECK(keeper.health_json()["status"] == "degraded", "and the server serves again");
}

static void test_reporting() {
    {
        Env env;
        env.accelerator = false;
        ContextKeeper keeper(base_params(), Level::MAX, settings(false), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        run_with(keeper, env, seen, [](const ContextKeeper::ContextView&) { return failure_of(SD_ERROR_OUT_OF_MEMORY, "bad \xff\xfe name"); }, failure);
        std::string text;
        bool threw = false;
        try {
            text = safe_json_dump(keeper.health_json());
        } catch (...) {
            threw = true;
        }
        CHECK(!threw && text.find("out_of_memory") != std::string::npos, "text that is not UTF-8 never breaks the health");
        const json health = keeper.health_json();
        CHECK(health.contains("uptime_seconds") && health.contains("version") && health.contains("counters"), "the health has its fields");
    }

    GenerationFailure f = oom();
    CHECK(failure_http_status(f) == 503 && std::string(failure_code_name(f)) == "out_of_memory", "out of memory is 503");
    f.unavailable = true;
    CHECK(failure_http_status(f) == 503 && std::string(failure_code_name(f)) == "unavailable", "unavailable is 503");
    CHECK(failure_http_status(failure_of(SD_ERROR_INVALID_ARGUMENT, "x")) == 400, "an invalid argument is 400");
    CHECK(failure_http_status(model_file_failure()) == 500 && failure_http_status(backend_failure()) == 500, "model file and backend are 500");
    CHECK(failure_http_status(failure_of(SD_ERROR_UNKNOWN, "x")) == 500 &&
              std::string(failure_code_name(failure_of(SD_ERROR_UNKNOWN, "x"))) == "generation_failed",
          "an unknown failure keeps the code generation_failed");
    CHECK(failure_is_recoverable(SD_ERROR_OUT_OF_MEMORY) && failure_is_recoverable(SD_ERROR_BACKEND) &&
              !failure_is_recoverable(SD_ERROR_MODEL_FILE) && !failure_is_recoverable(SD_ERROR_INVALID_ARGUMENT),
          "only memory and backend failures are worth a lighter profile");

    {
        httplib::Response res;
        GenerationFailure failure   = oom();
        failure.retry_after_seconds = 42;
        send_generation_failure(res, failure);
        const json body = json::parse(res.body);
        CHECK(res.status == 503 && res.get_header_value("Retry-After") == "42", "503 carries Retry-After");
        CHECK(body["code"] == "out_of_memory" && body["retry_after_seconds"] == 42 && body["error"] == "unet ran out of memory",
              "and the code, the wait and the reason in the body");
    }
    {
        httplib::Response res;
        send_generation_failure(res, failure_of(SD_ERROR_INVALID_ARGUMENT, "bad size"));
        const json body = json::parse(res.body);
        CHECK(res.status == 400 && !res.has_header("Retry-After") && body["code"] == "invalid_argument", "400 has no Retry-After");
    }
    {
        httplib::Response res;
        GenerationFailure failure = model_file_failure();
        failure.unavailable       = true;
        send_generation_failure(res, failure);
        const json body = json::parse(res.body);
        CHECK(res.status == 503 && res.has_header("Retry-After") && body["code"] == "unavailable" && body["cause"] == "model_file",
              "unavailable names its cause");
    }
    {
        httplib::Response res;
        send_generation_failure(res, failure_of(SD_ERROR_OUT_OF_MEMORY, "bad \xff\xfe"));
        CHECK(res.status == 503 && !res.body.empty() && json::accept(res.body), "an answer with text that is not UTF-8 is still JSON");
    }
    CHECK(failure_from_exception(std::bad_alloc()).code == SD_ERROR_OUT_OF_MEMORY, "bad_alloc is a lack of memory");
    const GenerationFailure other = failure_from_exception(std::runtime_error("boom"));
    CHECK(other.code == SD_ERROR_UNKNOWN && other.message == "boom", "other exceptions keep their text");
}

static void test_backend_choice() {
    Env env;
    SDContextParams params = base_params();
    params.backend         = "vulkan";
    params.params_backend  = "cpu";
    params.n_threads       = 3;
    ContextKeeper keeper(params, Level::MAX, settings(), env.hooks());
    ContextKeeper::BackendChoice choice = keeper.backend_choice();
    CHECK(choice.backend == "vulkan" && choice.params_backend == "cpu" && choice.n_threads == 3,
          "before a model is loaded, the upscaler uses the options of the server");
    GenerationFailure failure;
    keeper.start(failure);
    Seen seen;
    run_with(keeper, env, seen, [](const ContextKeeper::ContextView& view) { return view.level < Level::LITE_CPU ? oom() : ok_result; }, failure);
    choice = keeper.backend_choice();
    CHECK(choice.backend == "cpu" && choice.n_threads == 3, "after the cpu profile, the upscaler runs on the cpu");
}

static void test_threads() {
    Env env;
    env.accelerator = false;
    ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
    GenerationFailure start_failure;
    keeper.start(start_failure);

    std::atomic<int> inside{0};
    std::atomic<int> max_inside{0};
    std::atomic<bool> done{false};
    std::atomic<int> succeeded{0};
    std::atomic<int> failed{0};

    std::thread observer([&] {
        while (!done) {
            const json health = keeper.health_json();
            (void)health["status"];
            (void)keeper.status_name();
            (void)keeper.backend_choice();
            (void)keeper.supports_image_generation();
        }
    });

    std::vector<std::thread> workers;
    for (int t = 0; t < 4; t++) {
        workers.emplace_back([&, t] {
            for (int i = 0; i < 40; i++) {
                GenerationFailure failure;
                const bool ok = keeper.run(
                    [&](const ContextKeeper::ContextView& view) {
                        const int now_inside = ++inside;
                        int seen_max         = max_inside.load();
                        while (now_inside > seen_max && !max_inside.compare_exchange_weak(seen_max, now_inside)) {
                        }
                        std::this_thread::sleep_for(std::chrono::microseconds(200));
                        --inside;
                        // Now and then the request does not fit on the heavier profiles.
                        if ((i + t) % 7 == 0 && view.level < Level::LITE) {
                            env.next_failure = oom();
                            return false;
                        }
                        return true;
                    },
                    failure);
                (ok ? succeeded : failed)++;
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    done = true;
    observer.join();

    const json health = keeper.health_json();
    CHECK(max_inside.load() == 1, "never two generations at the same time");
    CHECK(succeeded + failed == 160 && counter(health, "requests") == 160, "every request was counted");
    CHECK(health["busy"] == false, "idle at the end");
    CHECK(counter(health, "step_downs") >= 1 && health["status"] == "degraded", "the steps down of one request serve the next ones");
}

// The model catalog of the server asks the keeper to replace the model, so that a model picked there is protected
// like the one given on the command line.
static void test_load_unload() {
    SDContextParams other = base_params();
    other.model_path      = "other.safetensors";

    {
        Env env;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);

        bool model_while_creating = true;
        env.on_create             = [&](const Load&) { model_while_creating = keeper.has_model(); };
        CHECK(keeper.load(other, failure) && keeper.has_model(), "load replaces the model");
        CHECK(env.loads.size() == 2 && env.loads[1].model_path == "other.safetensors" && env.loads[1].level() == Level::MAX,
              "with the one that was asked for, at the profile the server was started with");
        CHECK(!model_while_creating, "the old context is freed before the new one is created");

        Seen seen;
        std::string ran_on;
        CHECK(run_with(
                  keeper, env, seen,
                  [&](const ContextKeeper::ContextView& view) {
                      ran_on = view.params != nullptr ? view.params->model_path : "";
                      return ok_result;
                  },
                  failure),
              "a request runs after the change");
        CHECK(ran_on == "other.safetensors", "on the new model");
    }
    {
        Env env;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        CHECK(keeper.describe_ladder() == "max > balanced > lite > lite-disk > lite-hybrid > lite-cpu", "the ladder of the first model");
        SDContextParams explicit_set = base_params();
        explicit_set.memory_guard    = 90;
        explicit_set.clip_on_cpu     = true;
        explicit_set.params_backend  = "disk";
        CHECK(keeper.load(explicit_set, failure) && keeper.describe_ladder() == "max > lite > lite-cpu",
              "the ladder is built again from the options of the new model");
    }
    {
        Env env;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        keeper.load(other, failure);
        Seen seen;
        int attempts  = 0;
        const bool ok = run_with(keeper, env, seen, [&](const ContextKeeper::ContextView&) { return ++attempts == 1 ? oom() : ok_result; }, failure);
        CHECK(ok && seen.levels == std::vector<Level>({Level::MAX, Level::BALANCED}),
              "a request on the new model that runs out of memory steps down like on the first one");
        CHECK(env.loads.size() == 3 && env.loads[2].model_path == "other.safetensors", "and the lighter profile loads the same file");
        CHECK(keeper.health_json()["status"] == "degraded", "the health reports the lighter profile");
    }
    {
        Env env;
        env.load_result = [](const Load& load) { return load.level() < Level::LITE ? oom() : ok_result; };
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        CHECK(keeper.health_json()["profile"] == "lite", "the first model needed the lite profile");
        env.load_result = nullptr;
        CHECK(keeper.load(other, failure) && env.loads.back().level() == Level::MAX,
              "the next model starts again from the profile the server was started with");
        const json health = keeper.health_json();
        CHECK(health["status"] == "ok" && health["profile"] == "max" && health["next_promotion_seconds"].is_null(),
              "nothing is left of the step down of the previous model");
    }
    {
        Env env;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen served;
        run_with(keeper, env, served, succeed, failure);

        env.load_result        = [](const Load& load) { return load.model_path == "broken.safetensors" ? model_file_failure() : ok_result; };
        SDContextParams broken = base_params();
        broken.model_path      = "broken.safetensors";
        CHECK(!keeper.load(broken, failure) && failure.code == SD_ERROR_MODEL_FILE, "a model that cannot be loaded is reported with its cause");
        CHECK(!keeper.has_model() && keeper.status_name() == std::string("no_model"), "and no model is left loaded");
        const size_t loads = env.loads.size();
        Seen refused;
        CHECK(!run_with(keeper, env, refused, succeed, failure) && failure.code == SD_ERROR_INVALID_ARGUMENT && refused.levels.empty(),
              "requests are refused until a model is loaded");
        CHECK(keeper.load(other, failure) && keeper.has_model() && env.loads.size() == loads + 1, "load can be called again after a failure");
        const json health = keeper.health_json();
        CHECK(counter(health, "requests") == 2 && counter(health, "succeeded") == 1 && counter(health, "failed") == 1,
              "the counters are kept across the changes of model");
    }
    {
        Env env;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        keeper.unload();
        CHECK(!keeper.has_model() && keeper.status_name() == std::string("no_model") && keeper.health_json()["model_loaded"] == false,
              "unload frees the model");
        Seen seen;
        CHECK(!run_with(keeper, env, seen, succeed, failure) && failure.code == SD_ERROR_INVALID_ARGUMENT && seen.levels.empty(),
              "and requests are refused after it");
        keeper.unload();
        CHECK(!keeper.has_model(), "unloading twice is harmless");
        CHECK(keeper.load(base_params(), failure) && keeper.has_model() && env.loads.size() == 2, "a model can be loaded again");
    }
    {
        Env env;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);

        std::atomic<bool> done{false};
        std::atomic<int> readings{0};
        std::vector<std::thread> readers;
        for (int t = 0; t < 3; t++) {
            readers.emplace_back([&] {
                while (!done) {
                    const json health = keeper.health_json();
                    (void)health["status"];
                    (void)keeper.describe_ladder();
                    (void)keeper.can_step_down();
                    (void)keeper.status_name();
                    (void)keeper.has_model();
                    ++readings;
                }
            });
        }
        for (int i = 0; i < 150; i++) {
            if (i % 3 == 0) {
                keeper.unload();
            } else {
                keeper.load(i % 2 == 0 ? other : base_params(), failure);
            }
        }
        done = true;
        for (auto& reader : readers) {
            reader.join();
        }
        CHECK(readings.load() > 0, "the health, the ladder and the status were read while the model changed");
        CHECK(keeper.load(other, failure) && keeper.has_model() &&
                  keeper.describe_ladder() == "max > balanced > lite > lite-disk > lite-hybrid > lite-cpu",
              "and the keeper still works after 150 changes");
    }
}

static GenerationFailure cancelled_failure() {
    return failure_of(SD_ERROR_CANCELLED, "the generation was cancelled");
}

// Waits (at most five seconds, so that a broken keeper fails the test instead of hanging it) for a flag.
static bool wait_for(const std::atomic<bool>& flag) {
    for (int i = 0; i < 5000 && !flag; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return flag;
}

// A generation on its own thread. It goes on until the library is told to stop it (the cancel hook sets
// told_to_stop) or until the test lets it finish (release), like a real generation that checks the flag.
struct Generation {
    std::atomic<bool> entered{false};
    std::atomic<bool> told_to_stop{false};
    std::atomic<bool> release{false};
    sd_ctx_t* ctx = nullptr;
    GenerationFailure failure;
    bool ok = true;
    std::thread thread;
};

static void start_generation(ContextKeeper& keeper, Env& env, const CancelToken* token, Generation& gen) {
    gen.thread = std::thread([&keeper, &env, token, &gen] {
        gen.ok = keeper.run(
            [&](const ContextKeeper::ContextView& view) {
                gen.ctx     = view.ctx;
                gen.entered = true;
                for (int i = 0; i < 5000 && !gen.told_to_stop && !gen.release; i++) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                if (gen.told_to_stop) {
                    env.next_failure = cancelled_failure();
                    return false;
                }
                return true;
            },
            gen.failure,
            token);
    });
}

static void test_cancel() {
    // A job that is generating is stopped through its token.
    {
        Env env;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);

        Generation gen;
        CancelToken token;
        env.on_cancel = [&gen](sd_ctx_t*) { gen.told_to_stop = true; };
        start_generation(keeper, env, &token, gen);
        CHECK(wait_for(gen.entered), "the generation is running");
        CHECK(keeper.health_json()["busy"] == true, "and the keeper reports it busy");

        const bool told = keeper.cancel(token);
        gen.thread.join();
        const json health = keeper.health_json();
        CHECK(told && token.requested, "cancel() reaches the library while the job generates");
        CHECK(env.cancel_calls.size() == 1 && env.cancel_calls[0] == gen.ctx, "and tells the context that job is using");
        CHECK(!gen.ok && gen.failure.code == SD_ERROR_CANCELLED, "the run reports a cancellation");
        CHECK(counter(health, "cancelled") == 1 && counter(health, "failed") == 0 && counter(health, "succeeded") == 0 &&
                  counter(health, "requests") == 1,
              "a cancelled run is counted on its own, not as a failure");
        CHECK(counter(health, "step_downs") == 0 && env.loads.size() == 1 && health["status"] == "ok",
              "it does not make the server step down or load the model again");
        CHECK(health["last_error"].is_null() && health["busy"] == false, "it is not an error of the server, and the keeper is idle");
        CHECK(!keeper.cancel(token) && env.cancel_calls.size() == 1, "once the job has ended, cancel() reaches nobody");

        Seen seen;
        CancelToken next;
        CHECK(run_with(keeper, env, seen, succeed, failure, &next) && seen.contexts.size() == 1 && seen.contexts[0] == gen.ctx,
              "the next job runs on the same context");
        CHECK(counter(keeper.health_json(), "succeeded") == 1, "and succeeds");
        CHECK(!keeper.cancel(next) && env.cancel_calls.size() == 1, "and a stop for it, once it has ended, reaches nobody either");
    }

    // A stop request that arrives before the job reaches the context: the job generates nothing.
    {
        Env env;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);

        CancelToken token;
        CHECK(!keeper.cancel(token) && token.requested, "with nothing running, cancel() only records the request");
        CHECK(env.cancel_calls.empty(), "and the library is not told");
        Seen seen;
        const bool ok     = run_with(keeper, env, seen, succeed, failure, &token);
        const json health = keeper.health_json();
        CHECK(!ok && failure.code == SD_ERROR_CANCELLED && seen.levels.empty(), "a job cancelled before it starts generates nothing");
        CHECK(counter(health, "cancelled") == 1 && counter(health, "failed") == 0 && health["busy"] == false,
              "and counts as cancelled");
    }

    // A job nobody wants any more does not cost a reload of the model.
    {
        Env env;
        env.accelerator = false;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen first;
        run_with(keeper, env, first, [](const ContextKeeper::ContextView& view) { return view.level < Level::LITE ? oom() : ok_result; }, failure);
        const size_t loads = env.loads.size();
        env.advance(100);

        CancelToken token;
        keeper.cancel(token);
        Seen seen;
        run_with(keeper, env, seen, succeed, failure, &token);
        CHECK(env.loads.size() == loads && counter(keeper.health_json(), "promotions") == 0,
              "a cancelled job does not trigger the climb back to a heavier profile");
        Seen later;
        run_with(keeper, env, later, succeed, failure);
        CHECK(env.loads.size() == loads + 1 && later.levels == std::vector<Level>({Level::BALANCED}),
              "the next job does");
    }

    // A stop and a failure that the server would recover from: the stop wins.
    {
        Env env;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);

        CancelToken token;
        Seen seen;
        const Script cancel_then_fail = [&](const ContextKeeper::ContextView&) {
            keeper.cancel(token);
            return oom();
        };
        const bool ok     = run_with(keeper, env, seen, cancel_then_fail, failure, &token);
        const json health = keeper.health_json();
        CHECK(!ok && failure.code == SD_ERROR_CANCELLED, "a job cancelled while it ran out of memory is reported as cancelled");
        CHECK(env.loads.size() == 1 && counter(health, "step_downs") == 0 && seen.levels.size() == 1,
              "and the model is not loaded again for a job nobody waits for");
        CHECK(counter(health, "cancelled") == 1 && counter(health, "failed") == 0, "it is counted as cancelled");
    }

    // A stop that arrives while the model is being loaded again: the job does not run on the new context.
    {
        Env env;
        CancelToken token;
        env.load_result = [&](const Load& load) {
            if (load.level() == Level::BALANCED) {
                token.requested = true;
            }
            return ok_result;
        };
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);

        Seen seen;
        const bool ok     = run_with(keeper, env, seen, [](const ContextKeeper::ContextView&) { return oom(); }, failure, &token);
        const json health = keeper.health_json();
        CHECK(!ok && failure.code == SD_ERROR_CANCELLED, "a job cancelled while the model is loaded again is reported as cancelled");
        CHECK(seen.levels.size() == 1, "and it does not run again on the new context");
        CHECK(counter(health, "step_downs") == 1 && health["profile"] == "balanced" && counter(health, "cancelled") == 1 &&
                  counter(health, "failed") == 0,
              "the server stays on the profile it loaded, and counts the job as cancelled");
    }

    // The library reports a cancellation: never a failure of the server, whoever asked.
    {
        Env env;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        const bool ok     = run_with(keeper, env, seen, [](const ContextKeeper::ContextView&) { return cancelled_failure(); }, failure);
        const json health = keeper.health_json();
        CHECK(!ok && failure.code == SD_ERROR_CANCELLED && counter(health, "cancelled") == 1 && counter(health, "failed") == 0,
              "a cancellation reported without a token is still not a failure");
        CHECK(env.loads.size() == 1 && health["last_error"].is_null(), "and leaves no error behind");
    }

    // A stop for one job never reaches another request.
    {
        Env env;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);

        Generation gen;  // a synchronous request: no token
        env.on_cancel = [&gen](sd_ctx_t*) { gen.told_to_stop = true; };
        start_generation(keeper, env, nullptr, gen);
        CHECK(wait_for(gen.entered), "a request without a token is running");
        CancelToken other;
        CHECK(!keeper.cancel(other) && env.cancel_calls.empty() && !gen.told_to_stop,
              "a stop for a job that is not running does not reach it");
        gen.release = true;
        gen.thread.join();
        CHECK(gen.ok && counter(keeper.health_json(), "succeeded") == 1, "which finishes normally");
    }

    // The library forgets a stop request when a generation begins, so the caller repeats it.
    {
        Env env;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure start_failure;
        keeper.start(start_failure);

        std::atomic<bool> entered{false};
        std::atomic<bool> first_seen{false};
        std::atomic<bool> forgotten{false};
        std::atomic<bool> second_seen{false};
        env.on_cancel = [&](sd_ctx_t*) {
            if (!first_seen) {
                first_seen = true;
            } else {
                second_seen = true;
            }
        };
        CancelToken token;
        GenerationFailure failure;
        bool ok = true;
        std::thread generation([&] {
            ok = keeper.run(
                [&](const ContextKeeper::ContextView&) {
                    entered = true;
                    wait_for(first_seen);
                    forgotten = true;  // what the library does at the start of generate_image()
                    if (!wait_for(second_seen)) {
                        return true;
                    }
                    env.next_failure = cancelled_failure();
                    return false;
                },
                failure,
                &token);
        });
        wait_for(entered);
        keeper.cancel(token);
        wait_for(forgotten);
        keeper.cancel(token);
        generation.join();
        CHECK(!ok && failure.code == SD_ERROR_CANCELLED, "a request repeated after the library forgot the first one stops the job");
        CHECK(env.cancel_calls.size() == 2, "every call reaches the library while the job runs");
    }
}

// Stops arrive from other threads while jobs run, fail and make the server rebuild its context. Under the
// address sanitizer a stop that reached a released context is reported by the hook of Env.
static void test_cancel_threads() {
    Env env;
    env.accelerator = false;
    ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
    GenerationFailure start_failure;
    keeper.start(start_failure);

    constexpr int kThreads = 4;
    constexpr int kJobs    = 40;
    std::vector<std::unique_ptr<CancelToken>> tokens;
    for (int i = 0; i < kThreads * kJobs; i++) {
        tokens.push_back(std::make_unique<CancelToken>());
        // Every fifth job is cancelled before it reaches the keeper, whatever the threads do.
        tokens.back()->requested = i % 5 == 0;
    }

    std::atomic<int> inside{0};
    std::atomic<int> max_inside{0};
    std::atomic<bool> done{false};
    std::atomic<int> succeeded{0};
    std::atomic<int> failed{0};
    std::atomic<int> cancelled{0};

    std::thread canceller([&] {
        std::mt19937 rng(1234);
        while (!done) {
            keeper.cancel(*tokens[rng() % tokens.size()]);
            (void)keeper.health_json();
        }
    });

    std::vector<std::thread> workers;
    for (int t = 0; t < kThreads; t++) {
        workers.emplace_back([&, t] {
            for (int i = 0; i < kJobs; i++) {
                const CancelToken& token = *tokens[t * kJobs + i];
                GenerationFailure failure;
                const bool ok = keeper.run(
                    [&](const ContextKeeper::ContextView& view) {
                        const int now_inside = ++inside;
                        int seen_max         = max_inside.load();
                        while (now_inside > seen_max && !max_inside.compare_exchange_weak(seen_max, now_inside)) {
                        }
                        for (int step = 0; step < 4 && !token.requested; step++) {
                            std::this_thread::sleep_for(std::chrono::microseconds(100));
                        }
                        --inside;
                        if (token.requested) {
                            env.next_failure = cancelled_failure();
                            return false;
                        }
                        if ((i + t) % 7 == 0 && view.level < Level::LITE) {
                            env.next_failure = oom();
                            return false;
                        }
                        return true;
                    },
                    failure,
                    &token);
                if (ok) {
                    succeeded++;
                } else if (failure.code == SD_ERROR_CANCELLED) {
                    cancelled++;
                } else {
                    failed++;
                }
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    done = true;
    canceller.join();

    const json health = keeper.health_json();
    CHECK(max_inside.load() == 1, "with stops arriving, still never two generations at the same time");
    CHECK(succeeded + failed + cancelled == kThreads * kJobs && counter(health, "requests") == kThreads * kJobs,
          "every job ends as succeeded, failed or cancelled");
    CHECK(counter(health, "succeeded") == succeeded && counter(health, "failed") == failed && counter(health, "cancelled") == cancelled,
          "and the counters of the keeper agree with what the callers saw");
    CHECK(cancelled >= kThreads * kJobs / 5 && health["busy"] == false, "the jobs cancelled beforehand were stopped, and the keeper is idle at the end");
}

static sd_device_info_t device_info(const char* name, sd_device_type_t type, uint64_t free_gib) {
    sd_device_info_t info{};
    snprintf(info.name, sizeof(info.name), "%s", name);
    snprintf(info.description, sizeof(info.description), "%s", name);
    info.type               = type;
    info.free_memory_bytes  = free_gib << 30;
    info.total_memory_bytes = free_gib << 30;
    return info;
}

// A card that the library lists once for each backend that supports it, and the integrated GPU of the processor.
static std::vector<sd_device_info_t> laptop_devices() {
    return {device_info("CUDA0", SD_DEVICE_TYPE_GPU, 6),
            device_info("Vulkan0", SD_DEVICE_TYPE_GPU, 6),
            device_info("Vulkan1", SD_DEVICE_TYPE_IGPU, 8)};
}

using Names = std::vector<std::string>;

// The backend each load asked for, in order; "" is the engine's own choice of device.
static Names backends_of(const Env& env) {
    Names names;
    for (const Load& load : env.loads) {
        names.push_back(load.backend);
    }
    return names;
}

// A scripted library where nothing works but the cpu.
static GenerationFailure only_the_cpu_works(const Load& load) {
    return load.backend == "cpu" ? ok_result : backend_failure();
}

// A request that only succeeds on the device named.
static Script runs_only_on(const std::string& device) {
    return [device](const ContextKeeper::ContextView& view) {
        return view.params != nullptr && view.params->backend == device ? ok_result : backend_failure();
    };
}

// A device that fails is not the end of the GPUs: the same configuration is tried once more, then every other device,
// and the cpu only when none of them works. The server goes back to the preferred device when it works again.
static void test_devices() {
    {
        Env env;
        env.devices     = laptop_devices();
        env.load_result = only_the_cpu_works;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        CHECK(keeper.start(failure), "the model loads on the cpu when no device works");
        CHECK(backends_of(env) == Names({"", "", "CUDA0", "Vulkan0", "Vulkan1", "cpu"}),
              "the same configuration again, then every other device best first, and the cpu last");
        const json health = keeper.health_json();
        CHECK(health["profile"] == "lite-cpu" && health["device"] == "cpu" && health["cpu_modules"] == json::array({"all"}),
              "the health says that the cpu is doing all the work");
        CHECK(health["fallback_devices"] == json::array({"CUDA0", "Vulkan0", "Vulkan1"}), "and which devices the server can use");
    }
    {
        Env env;
        env.devices     = laptop_devices();
        env.load_result = [](const Load& load) { return load.backend == "Vulkan0" ? ok_result : backend_failure(); };
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        CHECK(keeper.start(failure) && backends_of(env) == Names({"", "", "CUDA0", "Vulkan0"}),
              "a device that cannot start is replaced by the next GPU, not by the cpu");
        const json health = keeper.health_json();
        CHECK(env.loads.back().level() == Level::MAX, "on the profile the server was already on");
        CHECK(health["profile"] == "max" && health["status"] == "degraded" && health["device"] == "Vulkan0" &&
                  health["cpu_modules"] == json::array(),
              "the health says that it runs on another device, with nothing on the cpu");
        CHECK(health["next_promotion_seconds"] == 10, "the way back to the preferred device is scheduled");
    }
    {
        Env env;
        env.devices     = laptop_devices();
        int calls       = 0;
        env.load_result = [&](const Load&) { return ++calls == 1 ? backend_failure() : ok_result; };
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        CHECK(keeper.start(failure) && backends_of(env) == Names({"", ""}),
              "a failure that does not repeat is cured by loading the same configuration again");
        const json health = keeper.health_json();
        CHECK(health["status"] == "ok" && health["device"] == "auto" && health["next_promotion_seconds"].is_null(),
              "and the server is back to normal at once");
    }
    {
        Env env;
        env.devices = laptop_devices();
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        CHECK(run_with(keeper, env, seen, runs_only_on("Vulkan0"), failure), "a request is served by another GPU when the first one fails");
        CHECK(seen.levels == std::vector<Level>({Level::MAX, Level::MAX, Level::MAX, Level::MAX}), "on the same profile every time");
        CHECK(backends_of(env) == Names({"", "", "CUDA0", "Vulkan0"}), "the same configuration again, then the next device, then the one after");
        CHECK(keeper.backend_choice().backend == "Vulkan0", "the upscaler follows to the device in use");
        const json health = keeper.health_json();
        CHECK(counter(health, "recovered") == 1 && counter(health, "retries") == 1 && counter(health, "device_switches") == 2 &&
                  counter(health, "step_downs") == 0,
              "counted as one retry and two device changes, not as a lighter profile");
        CHECK(health["status"] == "degraded" && health["profile"] == "max" && health["device"] == "Vulkan0" &&
                  health["next_promotion_seconds"] == 10,
              "the server stays on the device that worked and schedules the way back");
        CHECK(health["last_error"]["code"] == "backend", "the failure is the last error");
    }
    {
        Env env;
        env.devices = laptop_devices();
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        int attempts = 0;
        CHECK(run_with(keeper, env, seen, [&](const ContextKeeper::ContextView&) { return ++attempts == 1 ? backend_failure() : ok_result; }, failure), "a request whose failure does not repeat is served");
        CHECK(seen.levels == std::vector<Level>({Level::MAX, Level::MAX}) && backends_of(env) == Names({"", ""}),
              "by the same device, after loading the model again");
        const json health = keeper.health_json();
        CHECK(health["status"] == "ok" && health["device"] == "auto" && counter(health, "retries") == 1 && counter(health, "device_switches") == 0,
              "without leaving it");
    }
    {
        Env env;
        env.devices = laptop_devices();
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        CHECK(!run_with(keeper, env, seen, [](const ContextKeeper::ContextView&) { return backend_failure(); }, failure), "a request that no device and not even the cpu can serve fails");
        CHECK(failure.code == SD_ERROR_BACKEND && !failure.unavailable && seen.levels.size() == 6 && backends_of(env).back() == "cpu",
              "after trying them all, with the reason of the failure");
        const json health = keeper.health_json();
        CHECK(health["profile"] == "lite-cpu" && counter(health, "failed") == 1, "the server stays up on the cpu");
    }
    {
        // What the host wrote itself is not rewritten: the same configuration again, then the cpu.
        Env env;
        env.devices            = laptop_devices();
        env.load_result        = only_the_cpu_works;
        SDContextParams pinned = base_params();
        pinned.backend         = "diffusion=cuda0,vae=cpu";
        ContextKeeper keeper(pinned, Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        CHECK(keeper.start(failure) && backends_of(env) == Names({"diffusion=cuda0,vae=cpu", "diffusion=cuda0,vae=cpu", "cpu"}),
              "a placement of modules written by the host is kept");
        CHECK(keeper.health_json()["fallback_devices"] == json::array(), "and no other device is offered");
    }
    {
        // A device the host named is the one the engine tries first: the others come after it.
        Env env;
        env.devices           = laptop_devices();
        env.load_result       = only_the_cpu_works;
        SDContextParams named = base_params();
        named.backend         = "cuda0";
        ContextKeeper keeper(named, Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        CHECK(keeper.start(failure) && backends_of(env) == Names({"cuda0", "cuda0", "Vulkan0", "Vulkan1", "cpu"}),
              "the device the host named is not tried again as an alternative");
    }
    {
        // A machine on the network is not something to send the model to because a local device failed.
        Env env;
        env.devices     = {device_info("CUDA0", SD_DEVICE_TYPE_GPU, 6), device_info("RPC0", SD_DEVICE_TYPE_GPU, 24)};
        env.load_result = only_the_cpu_works;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        CHECK(keeper.start(failure) && backends_of(env) == Names({"", "", "CUDA0", "cpu"}), "devices of an rpc server are not used on their own");
    }
    {
        // Cards first (the one the vendor backend lists, then the one with the most memory), then accelerators, then the integrated GPU.
        Env env;
        env.devices     = {device_info("Vulkan0", SD_DEVICE_TYPE_GPU, 6),
                           device_info("Vulkan2", SD_DEVICE_TYPE_IGPU, 8),
                           device_info("CUDA0", SD_DEVICE_TYPE_GPU, 6),
                           device_info("CUDA1", SD_DEVICE_TYPE_GPU, 12),
                           device_info("NPU0", SD_DEVICE_TYPE_ACCEL, 4),
                           device_info("CPU", SD_DEVICE_TYPE_CPU, 16)};
        env.load_result = only_the_cpu_works;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        CHECK(keeper.start(failure) && backends_of(env) == Names({"", "", "CUDA1", "CUDA0", "Vulkan0", "NPU0", "Vulkan2", "cpu"}),
              "the devices are tried best first, the cpu device is not one of them");
    }
    {
        // Running out of memory on the device that took over goes down the profiles on that device.
        Env env;
        env.devices     = laptop_devices();
        env.load_result = [](const Load& load) -> GenerationFailure {
            if (load.backend != "Vulkan0") {
                return backend_failure();
            }
            return load.level() < Level::LITE ? oom() : ok_result;
        };
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        CHECK(keeper.start(failure) && backends_of(env) == Names({"", "", "CUDA0", "Vulkan0", "Vulkan0", "Vulkan0"}) &&
                  env.loads[3].level() == Level::MAX && env.loads[4].level() == Level::BALANCED && env.loads[5].level() == Level::LITE,
              "the lighter profiles are tried on the device that is working");
        const json health = keeper.health_json();
        CHECK(health["profile"] == "lite" && health["device"] == "Vulkan0", "and the server stays there");
    }
    {
        // While a GPU is left to try, the engine must not turn a device that fails to start into a quiet cpu fallback.
        Env env;
        env.devices     = laptop_devices();
        env.load_result = only_the_cpu_works;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        bool guarded = true;
        for (size_t i = 0; i + 1 < env.loads.size(); ++i) {
            guarded = guarded && env.loads[i].no_fallback;
        }
        CHECK(guarded && !env.loads.back().no_fallback, "every load that wants a GPU fails when the GPU does not start; the cpu profile asks for the cpu itself");
    }
    {
        Env env;
        ContextKeeper keeper(base_params(), Level::MAX, settings(false), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        CHECK(!env.loads[0].no_fallback, "with degradation disabled the engine keeps its own fallback");
    }
    {
        Env env;
        env.accelerator = false;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        CHECK(!env.loads[0].no_fallback, "and so does a machine without an accelerator, where the cpu is the only device");
    }
    {
        Env env;
        env.devices = laptop_devices();
        ContextKeeper keeper(base_params(), Level::MAX, settings(false), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        CHECK(!run_with(keeper, env, seen, [](const ContextKeeper::ContextView&) { return backend_failure(); }, failure), "with degradation disabled a backend failure fails the request");
        CHECK(failure.code == SD_ERROR_BACKEND && seen.levels.size() == 1 && env.loads.size() == 1 &&
                  keeper.health_json()["fallback_devices"] == json::array(),
              "after one attempt, without loading the model again or offering a device");
    }
    {
        // The preferred device works again: the next climb returns to it.
        Env env;
        env.devices      = laptop_devices();
        bool cuda_broken = true;
        env.load_result  = [&](const Load& load) { return cuda_broken && (load.backend.empty() || load.backend == "CUDA0") ? backend_failure() : ok_result; };
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        CHECK(backends_of(env) == Names({"", "", "CUDA0", "Vulkan0"}), "the server started on the second GPU");
        const size_t loads = env.loads.size();
        cuda_broken        = false;
        env.advance(5);
        Seen early;
        run_with(keeper, env, early, succeed, failure);
        CHECK(env.loads.size() == loads, "no climb before the wait is over");
        env.advance(6);
        Seen climb;
        CHECK(run_with(keeper, env, climb, succeed, failure) && climb.levels == std::vector<Level>({Level::MAX}) && env.loads.size() == loads + 1 &&
                  env.loads.back().backend.empty(),
              "the next request after the wait loads the engine's own choice again");
        const json health = keeper.health_json();
        CHECK(health["status"] == "ok" && health["device"] == "auto" && counter(health, "promotions") == 1 && health["next_promotion_seconds"].is_null(),
              "and the server is back at the preferred device");
        CHECK(keeper.backend_choice().backend.empty(), "the upscaler follows");
    }
    {
        // The preferred device still fails: one attempt on it, then back to the one that worked, and a longer wait.
        Env env;
        env.devices     = laptop_devices();
        env.load_result = [](const Load& load) { return load.backend.empty() || load.backend == "CUDA0" ? backend_failure() : ok_result; };
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        const size_t loads = env.loads.size();
        env.advance(11);
        Seen climb;
        CHECK(run_with(keeper, env, climb, succeed, failure), "the request is served even if the climb fails");
        CHECK(env.loads.size() == loads + 2 && env.loads[loads].backend.empty() && env.loads[loads + 1].backend == "Vulkan0",
              "one attempt on the preferred device, then the device that worked");
        const json health = keeper.health_json();
        CHECK(counter(health, "promotions") == 0 && counter(health, "failed_promotions") == 1 && health["next_promotion_seconds"] == 20 &&
                  health["device"] == "Vulkan0",
              "counted as a failed climb with a longer wait");
    }
    {
        // The cpu took over because every device failed: the climb goes back to the profile it left, on the best device that works.
        Env env;
        env.devices      = laptop_devices();
        bool gpus_broken = true;
        env.load_result  = [&](const Load& load) { return gpus_broken && load.backend != "cpu" ? backend_failure() : ok_result; };
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        CHECK(keeper.health_json()["profile"] == "lite-cpu", "the cpu took over");
        gpus_broken        = false;
        const size_t loads = env.loads.size();
        env.advance(11);
        Seen climb;
        CHECK(run_with(keeper, env, climb, succeed, failure) && climb.levels == std::vector<Level>({Level::MAX}) &&
                  env.loads.size() == loads + 1 && env.loads.back().backend.empty(),
              "the first climb returns to the profile the server had when the device failed, not one step up");
        CHECK(keeper.health_json()["status"] == "ok", "on the preferred device");
    }
    {
        // Only the second GPU works again.
        Env env;
        env.devices      = laptop_devices();
        bool gpus_broken = true;
        bool only_vulkan = false;
        env.load_result  = [&](const Load& load) {
            if (load.backend == "cpu") {
                return ok_result;
            }
            return gpus_broken || (only_vulkan && load.backend != "Vulkan0") ? backend_failure() : ok_result;
        };
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        gpus_broken = false;
        only_vulkan = true;
        env.advance(11);
        Seen climb;
        CHECK(run_with(keeper, env, climb, succeed, failure) && climb.levels == std::vector<Level>({Level::MAX}), "the climb is served");
        const json health = keeper.health_json();
        CHECK(health["status"] == "degraded" && health["profile"] == "max" && health["device"] == "Vulkan0" && counter(health, "promotions") == 1,
              "by the GPU that works, not by the cpu");
    }
    {
        // The cpu took over because the memory ran out on every level: the climb goes up one step at a time.
        Env env;
        env.devices = laptop_devices();
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        run_with(keeper, env, seen, [](const ContextKeeper::ContextView& view) { return view.level < Level::LITE_CPU ? oom() : ok_result; }, failure);
        env.advance(11);
        Seen climb;
        CHECK(run_with(keeper, env, climb, succeed, failure) && climb.levels == std::vector<Level>({Level::LITE_HYBRID}),
              "after running out of memory everywhere the climb is one profile up");
    }
}

int main() {
    sd_set_log_callback([](enum sd_log_level_t, const char*, void*) {}, nullptr);

    test_ladder();
    test_start();
    test_step_down();
    test_devices();
    test_promotion();
    test_unavailable();
    test_reporting();
    test_backend_choice();
    test_threads();
    test_load_unload();
    test_cancel();
    test_cancel_threads();

    printf("\n%s (%d failures)\n", failures ? "TEST FAILED" : "ALL TESTS PASSED", failures);
    return failures ? 1 : 0;
}
