#include "run_events.h"

#include "diagnostic.h"

namespace sd::dgl {
    static thread_local RunEvents* active_events = nullptr;

    RunEvents::RunEvents(const RunCallbacks& callbacks, bool legacy_callbacks)
        : callbacks_(callbacks), legacy_progress_(legacy_callbacks) {
        progress_.set_callback(callbacks_.progress, callbacks_.progress_data);
        if (legacy_callbacks && !callbacks_.preview) {
            legacy_preview_      = sd_get_preview_callback();
            legacy_preview_data_ = sd_get_preview_callback_data();
            callbacks_.mode      = sd_get_preview_mode();
            callbacks_.interval  = sd_get_preview_interval();
            callbacks_.denoised  = sd_should_preview_denoised();
            callbacks_.noisy     = sd_should_preview_noisy();
        }
    }

    void RunEvents::check_cancelled() const {
        if (cancelled())
            throw Error(SD_STATUS_CANCELLED, "DGL run cancelled");
    }

    void RunEvents::report(const char* stage, const char* detail, int64_t completed, int64_t total, const char* unit) const {
        check_cancelled();
        progress_.report(stage, detail, completed, total, unit);
        check_cancelled();
    }

    void RunEvents::report_frames(const char* stage, int completed, int total) const {
        report(stage, "", completed, total, "frames");
        if (legacy_progress_ && sd_get_progress_callback())
            sd_get_progress_callback()(completed, total, 0.f, sd_get_progress_callback_data());
        check_cancelled();
    }

    void RunEvents::report_loading(int completed, int total, void* data) {
        const auto& run = *static_cast<RunEvents*>(data);
        // The loader monitors progress while its worker threads are still running.
        if (!run.cancelled())
            run.progress_.report("loading", "", completed, total, "tensors");
    }

    void RunEvents::begin_sample(int total_steps) {
        check_cancelled();
        ++preview_info_.sample_pass;
        preview_info_.total_steps = total_steps;
        if (legacy_preview_)
            sd_begin_preview_pass(total_steps);
        last_progress_us_ = ggml_time_us();
        if (legacy_progress_ || !callbacks_.progress)
            pretty_progress(0, total_steps, 0.f);
        progress_.begin_sample(total_steps, "");
        check_cancelled();
    }

    void RunEvents::sample_step(int completed, int total_steps) {
        if (legacy_progress_ || !callbacks_.progress) {
            const auto now = ggml_time_us();
            pretty_progress(completed, total_steps, (now - last_progress_us_) / 1000000.f);
            last_progress_us_ = now;
        }
        progress_.report_sample(completed, total_steps);
        check_cancelled();
    }

    void RunEvents::preview_bridge(int step, int frame_count, const sd_tensor_t* const* frames, bool noisy, void* userdata) {
        auto& run = *static_cast<RunEvents*>(userdata);
        if (run.cancelled())
            return;
        if (run.callbacks_.preview)
            run.callbacks_.preview(&run.preview_info_, step, frame_count, frames, noisy, run.callbacks_.preview_data);
        else if (run.legacy_preview_)
            run.legacy_preview_(step, frame_count, frames, noisy, run.legacy_preview_data_);
    }

    RunEventsScope::RunEventsScope(RunEvents& events)
        : previous_(active_events),
          byte_progress_(events.uses_legacy_progress() ? nullptr : RunEvents::report_loading, &events),
          console_progress_(events.uses_legacy_progress() || !events.callbacks().progress) {
        active_events = &events;
    }
    RunEventsScope::~RunEventsScope() {
        active_events = previous_;
    }

    void check_run_cancelled() {
        if (active_events)
            active_events->check_cancelled();
    }
    void report_run_progress(const char* stage) {
        if (active_events)
            active_events->report(stage);
    }
    void report_run_frames(const char* stage, int completed, int total) {
        if (active_events)
            active_events->report_frames(stage, completed, total);
    }
}  // namespace sd::dgl
