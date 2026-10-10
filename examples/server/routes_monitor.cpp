#include "routes.h"

#include <chrono>
#include <iomanip>
#include <sstream>

#include "async_jobs.h"
#include "common/common.h"
#include "common/log.h"

// --- Stats ---

struct MonitoringMetrics {
    int64_t uptime_seconds      = 0;
    int64_t timestamp           = 0;
    uint64_t total_completed    = 0;
    uint64_t total_failed       = 0;
    size_t queued               = 0;
    size_t generating           = 0;
    // Sync stats (used for combined totals and speed calculation)
    uint64_t sync_completed     = 0;
    uint64_t sync_failed        = 0;
    double   sync_total_seconds = 0.0;
    // Async breakdown (derived: async = total - sync)
    uint64_t async_completed    = 0;
    uint64_t async_failed       = 0;
    double avg_seconds_per_image = 0.0;
    double images_per_second     = 0.0;
    // last-10 rolling window
    double last_10_avg          = 0.0;
    size_t last_10_samples      = 0;
};

// Minimum allowed average seconds per image to prevent inflated
// images_per_second values from sub-millisecond completions.
static constexpr double kMinAvgSecondsPerImage = 0.001;

static MonitoringMetrics collect_metrics(ServerRuntime& runtime) {
    MonitoringMetrics m;

    m.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();

    // Async job stats (under async_job_manager mutex)
    {
        std::lock_guard<std::mutex> lock(runtime.async_job_manager->mutex);
        AsyncJobStats stats = collect_job_stats(*runtime.async_job_manager);
        m.total_completed   = stats.total_completed;
        m.total_failed      = stats.total_failed;
        m.queued            = stats.queued;
        m.generating        = stats.generating;
        m.last_10_avg       = stats.last_10_avg_seconds_per_image();
        m.last_10_samples   = stats.last_10_samples();
    }

    // Read sync stats and combine with async totals
    {
        std::lock_guard<std::mutex> lock(*runtime.sd_ctx_mutex);
        m.sync_completed     = runtime.sync_completed;
        m.sync_failed        = runtime.sync_failed;
        m.sync_total_seconds = runtime.sync_total_seconds;

        // Compute average while holding the lock to avoid data race on
        // sync_completed / sync_total_seconds.
        if (m.sync_completed > 0) {
            m.avg_seconds_per_image = m.sync_total_seconds / static_cast<double>(m.sync_completed);
            if (m.avg_seconds_per_image < kMinAvgSecondsPerImage) {
                m.avg_seconds_per_image = kMinAvgSecondsPerImage;
            }
            m.images_per_second     = 1.0 / m.avg_seconds_per_image;
        }
    }
    m.total_completed += m.sync_completed;
    m.total_failed    += m.sync_failed;

    // Generation speed: prefer sync stats (set above under the lock),
    // fall back to async rolling window if no sync data yet.
    if (m.sync_completed == 0 && m.last_10_samples > 0) {
        m.avg_seconds_per_image = m.last_10_avg;
        if (m.avg_seconds_per_image < kMinAvgSecondsPerImage) {
            m.avg_seconds_per_image = kMinAvgSecondsPerImage;
        }
        m.images_per_second     = (m.avg_seconds_per_image > 0) ? 1.0 / m.avg_seconds_per_image : 0.0;
    }

    // Async breakdown (derived from already-collected totals)
    m.async_completed = m.total_completed - m.sync_completed;
    m.async_failed    = m.total_failed - m.sync_failed;

    return m;
}

// --- JSON serialization ---

static json json_u64(uint64_t val) { return json(val); }

static json json_double(double val, int precision = 4) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << val;
    return json::parse(oss.str());
}

// --- Endpoint registration ---

void register_monitoring_endpoints(httplib::Server& svr, ServerRuntime& runtime) {
    // Capture server start time once (static so it survives across requests).
    static const std::chrono::system_clock::time_point server_start =
        std::chrono::system_clock::now();

    svr.Get("/metrics", [&runtime](const httplib::Request& req, httplib::Response& res) {
        // If --metrics was not passed, inform the user regardless of Accept header.
        if (!runtime.metrics_enabled) {
            res.status = 501;
            res.set_content(
                "The /metrics endpoint is not enabled. "
                "Start the server with --metrics to enable it.",
                "text/plain");
            return;
        }

        // Guard: runtime pointers may be null during server startup / model loading.
        if (runtime.sd_ctx == nullptr || runtime.sd_ctx_mutex == nullptr) {
            res.status = 503;
            res.set_content(
                "Server is initializing. Please retry after model loading completes.",
                "text/plain");
            return;
        }

        // Gate: only respond to clients that explicitly accept JSON.
        std::string accept = req.get_header_value("Accept");
        bool accepts_json = accept.find("application/json") != std::string::npos;
        if (!accepts_json) {
            res.status = 406;
            res.set_content(
                "Accept: application/json is required for /metrics",
                "text/plain");
            return;
        }

        // Compute uptime
        auto now = std::chrono::system_clock::now();
        int64_t uptime_seconds =
            static_cast<int64_t>(std::chrono::duration<double>(
                now - server_start)
                .count());

        // Collect all metrics
        MonitoringMetrics metrics = collect_metrics(runtime);
        metrics.uptime_seconds = uptime_seconds;

        // Build JSON response
        json response;

        // --- Server info ---
        response["server"] = {
            {"uptime_seconds", json_double(static_cast<double>(metrics.uptime_seconds), 1)},
            {"timestamp", metrics.timestamp},
        };

        // --- Processing ---
        response["processing"] = {
            {"total_completed", json_u64(metrics.total_completed)},
            {"total_failed", json_u64(metrics.total_failed)},
            {"active", {
                {"queued", json_u64(metrics.queued)},
                {"generating", json_u64(metrics.generating)},
            }},
            {"sync_completed", json_u64(metrics.sync_completed)},
            {"sync_failed", json_u64(metrics.sync_failed)},
            {"async_completed", json_u64(metrics.async_completed)},
            {"async_failed", json_u64(metrics.async_failed)},
        };

        // --- Generation speed ---
        json last_10_obj = {
            {"avg_seconds_per_image", json_double(metrics.last_10_avg)},
            {"count", json_u64(metrics.last_10_samples)},
        };
        response["generation_speed"] = {
            {"avg_seconds_per_image", json_double(metrics.avg_seconds_per_image)},
            {"images_per_second", json_double(metrics.images_per_second)},
            {"recent_seconds_per_image", last_10_obj},
        };

        // --- Model memory ---
        sd_model_memory_stats_t mem = {};
        json memory;
        if (sd_get_model_memory_stats(runtime.sd_ctx, &mem)) {
            memory = {
                {"total_bytes", json_u64(mem.total_size)},
                {"vram_bytes", json_u64(mem.vram_size)},
                {"ram_bytes", json_u64(mem.ram_size)},
                {"text_encoders_bytes", json_u64(mem.text_encoders_size)},
                {"diffusion_model_bytes", json_u64(mem.diffusion_model_size)},
                {"vae_bytes", json_u64(mem.vae_size)},
                {"control_net_bytes", json_u64(mem.control_net_size)},
                {"extensions_bytes", json_u64(mem.extensions_size)},
            };
        } else {
            memory = nullptr;
        }
        response["model_memory"] = memory;

        res.set_content(response.dump(2), "application/json");
    });
}
