#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "model_catalog.h"
#include "recovery.h"
#include "runtime.h"

enum class AsyncJobKind {
    ImgGen,
    VidGen,
    Dgl,
};

enum class AsyncJobStatus {
    Queued,
    Generating,
    Completed,
    Failed,
    Cancelled,
};

const char* async_job_kind_name(AsyncJobKind kind);
const char* async_job_status_name(AsyncJobStatus status);

struct AsyncJobProgress {
    std::string stage = "queued";
    std::string detail;
    std::string unit;
    int sample_pass                                     = 0;
    int64_t completed                                   = 0;
    int64_t total                                       = 0;
    std::chrono::steady_clock::time_point stage_started = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point finished{};
};

struct AsyncGenerationJob {
    std::string id;
    AsyncJobKind kind     = AsyncJobKind::ImgGen;
    AsyncJobStatus status = AsyncJobStatus::Queued;
    int64_t created_at   = unix_timestamp_now_ms();
    int64_t started_at   = 0;
    int64_t completed_at = 0;
    std::unique_ptr<ServerModelSelection> model;
    json request_body;
    uint64_t model_revision = 0;
    ImgGenJobRequest img_gen;
    VidGenJobRequest vid_gen;
    std::string dgl_source;
    json dgl_result_info;
    std::vector<std::string> result_images_b64;
    json result_generation_info;
    std::string result_media_b64;
    std::string result_media_mime_type;
    int result_frame_count = 0;
    int result_fps         = 0;
    std::string error_code;
    std::string error_message;
    // Set when a client asks to stop the job, queued or generating. While the job generates, the worker repeats
    // the request to the library (see JobCancellationMonitor), because the library forgets a stop that arrives
    // before a generation pass begins.
    CancelToken cancel;
    // Set, with the manager's mutex held, when the guardian stopped the job: it ends with the code halted.
    std::string halted_reason;
    mutable std::mutex progress_mutex;
    AsyncJobProgress progress;
    mutable std::mutex preview_mutex;
    std::string preview_b64;
    int preview_pass          = 0;
    int preview_step          = 0;
    int preview_total_steps   = 0;
};

struct AsyncJobManager {
    std::mutex mutex;
    std::condition_variable cv;
    // Notified, with mutex held, whenever a job that was generating reaches a final state.
    std::condition_variable finished_cv;
    std::unordered_map<std::string, std::shared_ptr<AsyncGenerationJob>> jobs;
    std::unordered_map<std::string, int64_t> expired_jobs;
    std::deque<std::string> queue;
    uint64_t next_id              = 0;
    bool stop                     = false;
    size_t max_pending_jobs       = 64;
    int64_t completed_ttl_seconds = 600;
    int64_t failed_ttl_seconds    = 600;

    size_t synchronous_jobs = 0;
    std::chrono::steady_clock::time_point staging_deadline   = std::chrono::steady_clock::time_point::max();
    std::chrono::steady_clock::time_point residency_deadline = std::chrono::steady_clock::time_point::max();

    void record_context_job_completion(const SDSvrParams& params);
};

void purge_expired_jobs(AsyncJobManager& manager);
size_t count_pending_jobs(const AsyncJobManager& manager);
std::string make_async_job_id(AsyncJobManager& manager);
bool cancel_queued_job(AsyncJobManager& manager, AsyncGenerationJob& job);
// Asks the generating job to stop and waits, at most for the given time, until it has ended. The request is
// repeated while waiting, because the library forgets a stop request that arrives before the generation has
// begun. Returns true when the job has reached a final state. Call it without holding manager.mutex.
bool cancel_generating_job(ServerRuntime& runtime,
                           const std::shared_ptr<AsyncGenerationJob>& job,
                           std::chrono::milliseconds wait);
// The guardian has halted the server: the job that is generating stops and ends with the code halted, a synchronous
// request (A1111, OpenAI) is stopped too, and the queued jobs wait for the resume. Waits a few seconds at most. Call
// it without holding manager.mutex.
void stop_work_for_guardian(ServerRuntime& runtime, const std::string& reason);
json make_async_job_json(const AsyncJobManager& manager, const AsyncGenerationJob& job);
bool parse_native_job_request(const json& body, ServerRuntime& runtime, AsyncGenerationJob& job, std::string& error_message);
bool execute_img_gen_job(ServerRuntime& runtime,
                         AsyncGenerationJob& job,
                         std::vector<std::string>& output_images,
                         json& output_info,
                         GenerationFailure& failure);
bool execute_vid_gen_job(ServerRuntime& runtime,
                         AsyncGenerationJob& job,
                         std::string& output_media_b64,
                         std::string& output_media_mime_type,
                         int& output_frame_count,
                         int& output_fps,
                         json& output_info,
                         GenerationFailure& failure);
void async_job_worker(ServerRuntime& runtime);
