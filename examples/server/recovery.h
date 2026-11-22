#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <json.hpp>
#include "common/common.h"
#include "common/profile.h"
#include "common/resource_owners.hpp"
#include "stable-diffusion.h"

namespace httplib {
    struct Response;
}

// Why a generation could not be served, in terms a client can act on.
struct GenerationFailure {
    sd_error_code_t code = SD_ERROR_NONE;
    std::string message;
    // No usable context at this moment (it is being rebuilt, or could not be); code holds the cause.
    bool unavailable = false;
    // How long a client should wait before trying again; 0 when waiting is not expected to help.
    int retry_after_seconds = 0;
};

// A client's request to stop one generation, shared between the thread that asks and the one that generates.
// Setting it directly only records the wish; ContextKeeper::cancel() also tells the library.
struct CancelToken {
    std::atomic<bool> requested{false};
};

// Out of memory and backend failures are the ones a lighter configuration can get around.
bool failure_is_recoverable(sd_error_code_t code);
// "out_of_memory", "backend", "model_file", "invalid_argument", "unavailable", ..., and
// "generation_failed" for failures without a more specific cause.
const char* failure_code_name(const GenerationFailure& failure);
int failure_http_status(const GenerationFailure& failure);
GenerationFailure out_of_memory_failure();
GenerationFailure failure_from_exception(const std::exception& error);
// 503 (with Retry-After) for lack of memory or of a context, 400 for invalid input, 500 otherwise.
void send_generation_failure(httplib::Response& res, const GenerationFailure& failure);
// Serializes without ever throwing on text that is not valid UTF-8 (library messages can carry file names).
std::string safe_json_dump(const nlohmann::json& value);

// Owns the diffusion context of the server and keeps it serving when memory runs short.
//
// A request that fails because the machine ran out of memory (or because a backend failed) makes the
// keeper free the context, load the model again with the next lighter profile of the ladder
// max > balanced > lite > lite-disk > lite-hybrid > lite-cpu, and run the same request again. The levels up to
// lite-disk keep all the compute on the GPUs; the CPU takes the text encoder at lite-hybrid and everything at
// lite-cpu, so it is the last resort. Once the machine has been quiet for a while, the keeper climbs back one
// step at a time; a climb that fails doubles the wait.
//
// A backend failure (a device that does not start, or fails while computing) is a different matter: a lighter
// profile does not cure it, so the keeper keeps the profile and changes where the model runs. It loads the model
// again with the same configuration once, in case the failure was passing; then it tries every other device in
// turn, the GPUs best first (the same card on another backend, other cards) and the integrated GPU last; the
// CPU takes over only when none of them works. While a GPU is left to try, the engine is told not to fall back to
// the CPU by itself, so that a device that fails to start is not turned into a slow server unnoticed. The
// climb back leads to the preferred device first, and from the CPU straight to the profile it was running when the
// device failed.
//
// The mutex returned by mutex() serializes everything that uses the compute devices: it is held for
// the whole of a run() and by the upscale endpoint. Reading the health or the resources never takes it.
class ContextKeeper {
public:
    using Clock = std::chrono::steady_clock;

    struct Settings {
        bool degradation     = true;
        int cooldown_seconds = 300;  // wait before climbing back one step; 0 = stay where the server stepped down
    };

    // Everything that touches the library or the machine, replaceable so that the logic can be tested
    // without a model. An empty function selects the real behavior.
    struct Hooks {
        // Loads the model; returns null and describes the reason when it cannot. Default: new_sd_ctx.
        std::function<SDCtxPtr(const sd_ctx_params_t&, GenerationFailure&)> create_context;
        // The reason the attempt that just returned false failed. Default: sd_get_last_error().
        std::function<GenerationFailure()> read_failure;
        // Whether a device other than the CPU is present. Default: asks the library.
        std::function<bool()> has_accelerator;
        // The devices of the machine, in the order the library lists them. Default: sd_get_device_info().
        std::function<std::vector<sd_device_info_t>()> list_devices;
        std::function<Clock::time_point()> now;
        // Asks the library to stop the generation running on a context. Default: sd_cancel_generation(SD_CANCEL_ALL).
        std::function<void(sd_ctx_t*)> cancel_context;
    };

    // What an attempt needs to run one generation.
    struct ContextView {
        sd_ctx_t* ctx           = nullptr;
        sd_profile::Level level = sd_profile::Level::MAX;
        // The server stepped down to a lite profile: decode the VAE in tiles whatever the request asked.
        bool force_vae_tiling = false;
        // What the context was created with, the profile of the ladder included. For a stage that loads its own
        // model next to the context (ADetailer), so that it follows the placement the server is running with.
        const SDContextParams* params = nullptr;
    };

    // Runs one generation. Must return false only when the generation itself failed, and must not call
    // the library again after that call: the reason is read on the same thread right after it returns.
    using Attempt = std::function<bool(const ContextView&)>;

    ContextKeeper(const SDContextParams& base_params,
                  sd_profile::Level base_level,
                  const Settings& settings,
                  const Hooks& hooks = Hooks());
    ContextKeeper(const ContextKeeper&)            = delete;
    ContextKeeper& operator=(const ContextKeeper&) = delete;

    // Loads the model, stepping down the ladder when the memory is not enough. Returns false when no
    // profile could load it; failure holds the reason of the last attempt.
    bool start(GenerationFailure& failure);

    // Replaces the model with the one params describes, e.g. when a client picks another from the catalog. The
    // context in use goes first (both would not fit), then the ladder is built again from the profile the
    // server started with, so the new model is protected like the first. Returns false when no profile could
    // load it: no model is loaded then, failure holds the reason of the last attempt, and load() can be called
    // again. Counters and uptime are kept. The caller makes sure no run() is waiting or running.
    bool load(const SDContextParams& params, GenerationFailure& failure);

    // Frees the context. Generation requests fail with "no model loaded" until load() succeeds.
    void unload();

    std::mutex& mutex() {
        return mutex_;
    }

    // Safe to call from any thread, without the mutex. Both stay true while the context is being rebuilt.
    bool has_model() const {
        return model_loaded_.load();
    }
    bool supports_image_generation() const {
        return supports_image_.load();
    }
    bool supports_video_generation() const {
        return supports_video_.load();
    }

    // Runs attempt on the context, recovering from a lack of memory as described above. Returns true when
    // an attempt succeeded.
    //
    // With a token, cancel() can stop this run: failure.code is then SD_ERROR_CANCELLED, the run counts as
    // cancelled (not as failed) and never makes the server step down. A token that was already requested
    // before the run reached the context stops it without generating anything.
    bool run(const Attempt& attempt, GenerationFailure& failure, const CancelToken* token = nullptr);

    // Records the request in the token and, if the run that owns it is generating at this moment, tells the
    // library to stop. Safe from any thread, never waits for the generation, and returns whether the library
    // was told. The library forgets a stop request when a generation begins, so a caller that cannot tell
    // whether the generation has begun yet repeats the call until the job ends (see the cancel route).
    bool cancel(CancelToken& token);

    // The settings the context in use was created with; for the upscale endpoint, which loads its own model.
    struct BackendChoice {
        std::string backend;
        std::string params_backend;
        bool diffusion_conv_direct = false;
        int n_threads              = -1;
    };
    BackendChoice backend_choice() const;

    // The profiles the server can step down through, e.g. "max > lite > lite-cpu".
    std::string describe_ladder() const;
    bool can_step_down() const;
    // The devices tried, in this order, after the engine's own choice failed, e.g. "CUDA0 > Vulkan0 > Vulkan1";
    // empty when none is offered (no accelerator, degradation disabled, or a placement written by the host).
    std::string describe_devices() const;

    nlohmann::json health_json() const;
    // "ok", "degraded", "recovering", "unavailable" or "no_model"
    const char* status_name() const;

private:
    enum class Status { NO_MODEL,
                        OK,
                        DEGRADED,
                        RECOVERING,
                        UNAVAILABLE };

    struct Counters {
        uint64_t requests          = 0;
        uint64_t succeeded         = 0;
        uint64_t failed            = 0;
        uint64_t cancelled         = 0;  // stopped on request; neither a success nor a failure
        uint64_t recovered         = 0;  // requests that succeeded after a step down
        uint64_t step_downs        = 0;
        uint64_t retries           = 0;  // a backend failure answered by loading the same configuration again
        uint64_t device_switches   = 0;  // a backend failure answered by another device, the cpu included
        uint64_t promotions        = 0;
        uint64_t failed_promotions = 0;
        uint64_t rebuilds          = 0;
    };

    struct LastError {
        std::string code;
        std::string message;
        int64_t at = 0;
    };

    // Where the model runs: a profile of the ladder, on a device. Slot 0 leaves the choice of device to the engine;
    // slot k > 0 names devices_[k - 1], which the keeper tries after the engine's own choice failed.
    struct Rung {
        size_t index = 0;
        size_t slot  = 0;
        bool valid   = false;
        bool retry   = false;  // the rung that just failed, loaded again
    };

    std::unique_ptr<SDContextParams> params_for(sd_profile::Level level, size_t slot = 0) const;
    std::string fingerprint(sd_profile::Level level) const;
    bool force_tiling_at(sd_profile::Level level) const;
    std::vector<sd_profile::Level> make_ladder() const;
    std::vector<std::string> make_devices() const;
    // Where to try next after failure at (index, slot); not valid when nothing is left. The same rung again is
    // offered once (retry_available) for a backend failure, then the other devices, then the lightest profile.
    Rung next_rung(size_t index, size_t slot, const GenerationFailure& failure, bool retry_available) const;
    std::string rung_text(size_t index, size_t slot) const;

    bool load_locked(GenerationFailure& failure);
    // Loads the model at (index, slot) and, unless only_this, keeps moving to the next rung while the failure is one
    // a lighter profile or another device can get around. retry_used is shared with the caller, so that one request
    // never loads the same rung again twice.
    bool build_from(size_t index, size_t slot, GenerationFailure& failure, bool& retry_used, bool only_this = false);
    bool try_build(size_t index, size_t slot, GenerationFailure& failure);
    bool restore(GenerationFailure& failure);
    bool step_down(GenerationFailure& failure, bool& retry_used);
    void note_move(size_t from_index, const Rung& to, const GenerationFailure& failure);
    void promote_if_due();
    void release_context();
    void note_build_failure(const GenerationFailure& failure);
    void record_error(const GenerationFailure& failure);
    Status status_locked() const;
    static const char* status_text(Status status);

    // The settings, the hooks and the profile the server started with do not change after construction. The
    // base parameters and the ladder change only in load(), with mutex_ and state_mutex_ both held, so reading
    // them under either one is enough.
    const Settings settings_;
    const Hooks hooks_;
    SDContextParams base_params_;
    const sd_profile::Level base_level_;
    std::vector<sd_profile::Level> ladder_;
    bool accelerator_ = false;
    std::vector<std::string> devices_;  // the devices of slots 1, 2, ...: best first

    // Held for the whole of a run(); guards ctx_ and the parameters it was created with.
    std::mutex mutex_;
    SDCtxPtr ctx_;
    std::unique_ptr<SDContextParams> ctx_params_;

    // Small lock for the figures health reads; always taken after mutex_ when both are needed.
    mutable std::mutex state_mutex_;
    size_t index_ = 0;  // position in ladder_
    size_t slot_  = 0;  // the device of the context in use: 0 = the engine's own choice, k = devices_[k - 1]
    // The profile the server was running when a backend failure sent it to the cpu profile; the climb back from
    // the cpu returns to it. SIZE_MAX when the cpu profile was reached by running out of memory.
    size_t resume_index_ = SIZE_MAX;
    // What the context in use was asked for, as health reports it.
    std::string device_ = "auto";
    std::vector<std::string> cpu_modules_;
    bool has_ctx_         = false;
    bool rebuilding_      = false;
    bool probation_       = false;  // the context in use is a promotion nobody has tried yet
    int cooldown_seconds_ = 300;
    Clock::time_point next_promotion_;
    Clock::time_point restore_not_before_;
    GenerationFailure restore_failure_;  // why the last attempt to load the model again failed
    Counters counters_;
    LastError last_error_;
    BackendChoice backend_choice_;
    const Clock::time_point started_at_;
    int active_runs_ = 0;
    // Set only while an attempt is executing, and cleared before the context can be released, so cancel()
    // never reaches a context that is gone.
    const CancelToken* running_token_ = nullptr;
    sd_ctx_t* running_ctx_            = nullptr;

    std::atomic<bool> model_loaded_{false};
    std::atomic<bool> supports_image_{false};
    std::atomic<bool> supports_video_{false};
};
