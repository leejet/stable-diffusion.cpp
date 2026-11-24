#ifndef __SD_DGL_RUN_EVENTS_H__
#define __SD_DGL_RUN_EVENTS_H__

#include <atomic>

#include "core/option_types.h"
#include "core/util.h"
#include "runtime/generation_progress.h"

namespace sd::dgl {
    struct RunCallbacks {
        sd_generation_progress_cb_t progress = nullptr;
        void* progress_data                  = nullptr;
        sd_dgl_preview_cb_t preview          = nullptr;
        void* preview_data                   = nullptr;
        enum preview_t mode                  = PREVIEW_NONE;
        int interval                         = 0;
        bool denoised                        = false;
        bool noisy                           = false;
    };

    class RunEvents {
        std::atomic<bool> cancelled_{false};
        RunCallbacks callbacks_;
        sd::GenerationProgress progress_;
        sd_preview_info_t preview_info_{};
        sd_preview_cb_t legacy_preview_ = nullptr;
        void* legacy_preview_data_      = nullptr;
        bool legacy_progress_           = false;
        int64_t last_progress_us_       = 0;

        static void preview_bridge(int step, int frame_count, const sd_tensor_t* const* frames, bool noisy, void* userdata);

    public:
        RunEvents(const RunCallbacks& callbacks, bool legacy_callbacks);
        void cancel() noexcept { cancelled_.store(true, std::memory_order_relaxed); }
        void set_cancelled(bool cancelled) noexcept { cancelled_.store(cancelled, std::memory_order_relaxed); }
        bool cancelled() const noexcept { return cancelled_.load(std::memory_order_relaxed); }
        void check_cancelled() const;
        void report(const char* stage, const char* detail = "", int64_t completed = 0, int64_t total = 0, const char* unit = "") const;
        void report_frames(const char* stage, int completed, int total) const;
        static void report_loading(int completed, int total, void* data);
        bool uses_legacy_progress() const { return legacy_progress_; }
        void begin_sample(int total_steps);
        void sample_step(int completed, int total_steps);
        const RunCallbacks& callbacks() const { return callbacks_; }
        const sd_preview_info_t& preview_info() const { return preview_info_; }
        bool uses_legacy_preview() const { return legacy_preview_ != nullptr; }
        sd_preview_cb_t preview_callback() const { return callbacks_.preview || legacy_preview_ ? preview_bridge : nullptr; }
    };

    class RunEventsScope {
        RunEvents* previous_;
        ByteProgressScope byte_progress_;
        ByteProgressConsoleScope console_progress_;

    public:
        explicit RunEventsScope(RunEvents& events);
        ~RunEventsScope();
        RunEventsScope(const RunEventsScope&)            = delete;
        RunEventsScope& operator=(const RunEventsScope&) = delete;
    };

    void check_run_cancelled();
    void report_run_progress(const char* stage);
    void report_run_frames(const char* stage, int completed, int total);
}  // namespace sd::dgl

#endif  // __SD_DGL_RUN_EVENTS_H__
