// Extracted from main.cpp during server refactor.

#include "async_jobs.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <thread>

#include "common/log.h"
#include "common/media_io.h"
#include "common/resource_owners.hpp"

class JobCancellationMonitor {
    std::mutex mutex_;
    std::condition_variable cv_;
    bool done_ = false;
    std::thread thread_;

public:
    // The keeper reaches a context only while this job's attempt is generating on it, so the stop cannot get to
    // another request, nor to a context that has been released. The monitor must not outlive keeper or job.
    JobCancellationMonitor(ContextKeeper& keeper, AsyncGenerationJob& job)
        : thread_([this, &keeper, &job]() {
              std::unique_lock<std::mutex> lock(mutex_);
              while (!done_) {
                  if (job.cancel.requested.load()) {
                      // Each generation pass resets the library flag, including ADetailer.
                      keeper.cancel(job.cancel);
                  }
                  cv_.wait_for(lock, std::chrono::milliseconds(10), [this]() { return done_; });
              }
          }) {}

    ~JobCancellationMonitor() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            done_ = true;
        }
        cv_.notify_one();
        thread_.join();
    }
};

static void preview_callback(int step, int frame_count, sd_image_t* frames, bool is_noisy, void* data) {
    (void)is_noisy;
    if (frame_count <= 0 || frames == nullptr || frames[0].data == nullptr) {
        return;
    }

    AsyncGenerationJob* job      = static_cast<AsyncGenerationJob*>(data);
    const sd_image_t& frame      = frames[0];
    const sd_preview_info_t info = sd_get_preview_info();

    auto image_bytes = encode_image_to_vector(EncodedImageFormat::PNG,
                                              frame.data,
                                              frame.width,
                                              frame.height,
                                              frame.channel,
                                              "",
                                              80);
    if (image_bytes.empty()) {
        return;
    }

    std::string b64 = base64_encode(image_bytes);
    std::lock_guard<std::mutex> lock(job->preview_mutex);
    job->preview_b64         = std::move(b64);
    job->preview_step        = step < 0 ? -step : step;
    job->preview_pass        = info.sample_pass;
    job->preview_total_steps = info.total_steps;
}

static void clear_preview_callback() {
    sd_set_preview_callback(nullptr, PREVIEW_NONE, 1, false, false, nullptr);
}

static void set_preview_callback(AsyncGenerationJob& job) {
    preview_t mode = str_to_preview(job.kind == AsyncJobKind::ImgGen
                                        ? job.img_gen.preview_mode.c_str()
                                        : job.vid_gen.preview_mode.c_str());
    if (mode == PREVIEW_COUNT || mode == PREVIEW_NONE) {
        return;
    }
    int interval = job.kind == AsyncJobKind::ImgGen
                       ? job.img_gen.preview_interval
                       : job.vid_gen.preview_interval;
    if (interval <= 0) {
        interval = 1;
    }
    sd_set_preview_callback(preview_callback, mode, interval, true, false, &job);
}

static void update_job_progress(AsyncGenerationJob& job, const char* stage, const char* detail = "", int sample_pass = -1, int64_t completed = 0, int64_t total = 0, const char* unit = "") {
    std::lock_guard<std::mutex> lock(job.progress_mutex);
    auto& progress = job.progress;
    if (sample_pass < 0) {
        sample_pass = progress.sample_pass;
    }
    if (progress.stage != stage || progress.detail != detail || progress.sample_pass != sample_pass) {
        progress.stage_started = std::chrono::steady_clock::now();
    }
    progress.stage       = stage;
    progress.detail      = detail;
    progress.sample_pass = sample_pass;
    progress.completed   = completed;
    progress.total       = total;
    progress.unit        = unit;
}

static void finish_job_progress(AsyncGenerationJob& job) {
    std::lock_guard<std::mutex> lock(job.progress_mutex);
    job.progress.finished = std::chrono::steady_clock::now();
}

class JobCallbacksGuard {
    sd_ctx_t* ctx_;
    AsyncGenerationJob& job_;
    bool detailing_ = false;

    static void progress_callback(const sd_generation_progress_t* progress, void* data) {
        auto& guard = *static_cast<JobCallbacksGuard*>(data);
        update_job_progress(guard.job_, guard.detailing_ ? "adetailer" : progress->stage,
                            guard.detailing_ ? progress->stage : progress->detail,
                            progress->sample_pass, progress->completed, progress->total, progress->unit);
    }

public:
    JobCallbacksGuard(sd_ctx_t* ctx, AsyncGenerationJob& job)
        : ctx_(ctx), job_(job) {
        clear_preview_callback();
        set_preview_callback(job);
        sd_set_generation_progress_callback(ctx_, progress_callback, this);
    }

    ~JobCallbacksGuard() {
        sd_set_generation_progress_callback(ctx_, nullptr, nullptr);
        clear_preview_callback();
    }

    void begin_detailing() {
        clear_preview_callback();
        detailing_ = true;
        update_job_progress(job_, "adetailer");
    }

    JobCallbacksGuard(const JobCallbacksGuard&)            = delete;
    JobCallbacksGuard& operator=(const JobCallbacksGuard&) = delete;
};

const char* async_job_kind_name(AsyncJobKind kind) {
    switch (kind) {
        case AsyncJobKind::ImgGen:
            return "img_gen";
        case AsyncJobKind::VidGen:
            return "vid_gen";
        default:
            return "img_gen";
    }
}

const char* async_job_status_name(AsyncJobStatus status) {
    switch (status) {
        case AsyncJobStatus::Queued:
            return "queued";
        case AsyncJobStatus::Generating:
            return "generating";
        case AsyncJobStatus::Completed:
            return "completed";
        case AsyncJobStatus::Failed:
            return "failed";
        case AsyncJobStatus::Cancelled:
            return "cancelled";
        default:
            return "failed";
    }
}

void purge_expired_jobs(AsyncJobManager& manager) {
    const int64_t now = unix_timestamp_now_ms();

    for (auto it = manager.expired_jobs.begin(); it != manager.expired_jobs.end();) {
        if (it->second <= now) {
            it = manager.expired_jobs.erase(it);
        } else {
            ++it;
        }
    }

    for (auto it = manager.jobs.begin(); it != manager.jobs.end();) {
        const auto& job = it->second;
        if (job->completed_at == 0) {
            ++it;
            continue;
        }

        int64_t ttl_seconds = job->status == AsyncJobStatus::Completed
                                  ? manager.completed_ttl_seconds
                                  : manager.failed_ttl_seconds;
        if (now - job->completed_at >= ttl_seconds * 1000) {
            manager.expired_jobs[job->id] = now + std::max<int64_t>(ttl_seconds, 60) * 1000;
            it                            = manager.jobs.erase(it);
        } else {
            ++it;
        }
    }
}

size_t count_pending_jobs(const AsyncJobManager& manager) {
    size_t pending = 0;
    for (const auto& entry : manager.jobs) {
        if (entry.second->status == AsyncJobStatus::Queued ||
            entry.second->status == AsyncJobStatus::Generating) {
            ++pending;
        }
    }
    return pending;
}

std::string make_async_job_id(AsyncJobManager& manager) {
    std::ostringstream oss;
    oss << "job_" << std::hex << unix_timestamp_now_ms() << "_" << std::setw(8)
        << std::setfill('0') << manager.next_id++;
    return oss.str();
}

bool cancel_queued_job(AsyncJobManager& manager, AsyncGenerationJob& job) {
    auto new_end = std::remove(manager.queue.begin(), manager.queue.end(), job.id);
    if (new_end == manager.queue.end()) {
        return false;
    }

    manager.queue.erase(new_end, manager.queue.end());
    job.cancel.requested.store(true);
    job.status       = AsyncJobStatus::Cancelled;
    job.completed_at = unix_timestamp_now_ms();
    job.result_images_b64.clear();
    job.result_generation_info = nullptr;
    job.result_media_b64.clear();
    job.result_media_mime_type.clear();
    job.result_frame_count = 0;
    job.result_fps         = 0;
    job.error_code         = "cancelled";
    job.error_message      = "job cancelled by client";
    finish_job_progress(job);
    return true;
}

bool cancel_generating_job(ServerRuntime& runtime,
                           const std::shared_ptr<AsyncGenerationJob>& job,
                           std::chrono::milliseconds wait) {
    constexpr std::chrono::milliseconds kRepeatInterval(100);

    AsyncJobManager& manager = *runtime.async_job_manager;
    const auto deadline      = std::chrono::steady_clock::now() + wait;
    for (;;) {
        runtime.keeper->cancel(job->cancel);

        std::unique_lock<std::mutex> lock(manager.mutex);
        const auto next = std::min(deadline, std::chrono::steady_clock::now() + kRepeatInterval);
        if (manager.finished_cv.wait_until(lock, next, [&]() { return job->status != AsyncJobStatus::Generating; })) {
            return true;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
    }
}

json make_async_job_json(const AsyncJobManager& manager, const AsyncGenerationJob& job) {
    json result;
    result["id"]               = job.id;
    result["kind"]             = async_job_kind_name(job.kind);
    result["status"]           = async_job_status_name(job.status);
    result["cancel_requested"] = job.cancel.requested.load();
    result["created"]          = job.created_at;
    result["started"]          = job.started_at == 0 ? json(nullptr) : json(job.started_at);
    result["completed"]        = job.completed_at == 0 ? json(nullptr) : json(job.completed_at);
    result["queue_position"]   = 0;

    {
        std::lock_guard<std::mutex> lock(job.progress_mutex);
        const auto& progress = job.progress;
        const auto end       = progress.finished == std::chrono::steady_clock::time_point{}
                                   ? std::chrono::steady_clock::now()
                                   : progress.finished;
        result["progress"]   = {
            {"stage", progress.stage},
            {"detail", progress.detail},
            {"pass", progress.sample_pass},
            {"completed", progress.total > 0 ? json(progress.completed) : json(nullptr)},
            {"total", progress.total > 0 ? json(progress.total) : json(nullptr)},
            {"unit", progress.unit.empty() ? json(nullptr) : json(progress.unit)},
            {"stage_elapsed_ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - progress.stage_started).count()},
            {"elapsed_ms", job.started_at == 0 ? 0 : std::max<int64_t>(0, (job.completed_at == 0 ? unix_timestamp_now_ms() : job.completed_at) - job.started_at)},
        };
    }

    if (job.status == AsyncJobStatus::Queued) {
        size_t position = 1;
        for (const auto& queued_id : manager.queue) {
            if (queued_id == job.id) {
                result["queue_position"] = position;
                break;
            }
            ++position;
        }
    }

    if (job.status == AsyncJobStatus::Completed) {
        if (job.kind == AsyncJobKind::VidGen) {
            result["result"] = {
                {"output_format", job.vid_gen.output_format},
                {"mime_type", job.result_media_mime_type},
                {"fps", job.result_fps},
                {"frame_count", job.result_frame_count},
                {"b64_json", job.result_media_b64},
                {"generation_info", job.result_generation_info},
            };
        } else {
            json images = json::array();
            for (size_t i = 0; i < job.result_images_b64.size(); ++i) {
                images.push_back({{"index", i}, {"b64_json", job.result_images_b64[i]}, {"generation_info", job.result_generation_info.at(i)}});
            }
            result["result"] = {
                {"output_format", job.img_gen.output_format},
                {"images", images},
            };
        }
        result["error"] = nullptr;
    } else if (job.status == AsyncJobStatus::Failed ||
               job.status == AsyncJobStatus::Cancelled) {
        result["result"] = nullptr;
        result["error"]  = {
            {"code",
             job.error_code.empty()
                 ? (job.status == AsyncJobStatus::Cancelled ? "cancelled" : "generation_failed")
                 : job.error_code},
            {"message", job.error_message},
        };
    } else {
        result["result"] = nullptr;
        result["error"]  = nullptr;
    }

    if (job.status == AsyncJobStatus::Generating) {
        std::lock_guard<std::mutex> lock(job.preview_mutex);
        if (!job.preview_b64.empty()) {
            result["preview"] = {
                {"pass", job.preview_pass},
                {"step", job.preview_step},
                {"total_steps", job.preview_total_steps},
                {"b64_json", job.preview_b64},
            };
        }
    }

    return result;
}

static bool fail_with(GenerationFailure& failure, const char* message) {
    failure         = GenerationFailure();
    failure.code    = SD_ERROR_UNKNOWN;
    failure.message = message;
    return false;
}

static bool fail_cancelled(GenerationFailure& failure) {
    failure         = GenerationFailure();
    failure.code    = SD_ERROR_CANCELLED;
    failure.message = "the generation was cancelled";
    return false;
}

// Runs the attempt on the keeper's context. While it runs, a stop the client asks for reaches the library however
// many generation passes the attempt makes.
static bool run_job(ServerRuntime& runtime,
                    AsyncGenerationJob& job,
                    const ContextKeeper::Attempt& attempt,
                    GenerationFailure& failure) {
    JobCancellationMonitor cancellation(*runtime.keeper, job);
    return runtime.keeper->run(attempt, failure, &job.cancel);
}

bool execute_img_gen_job(ServerRuntime& runtime,
                         AsyncGenerationJob& job,
                         std::vector<std::string>& output_images,
                         json& output_info,
                         GenerationFailure& failure) {
    output_info                      = json::array();
    const sd_img_gen_params_t params = job.img_gen.to_sd_img_gen_params_t();

    SDImageVec results;
    bool adetailer_failed = false;
    const bool generated  = run_job(
        runtime,
        job,
        [&](const ContextKeeper::ContextView& view) {
            sd_img_gen_params_t attempt_params = params;
            if (view.force_vae_tiling) {
                attempt_params.vae_tiling_params.enabled = true;
            }
            adetailer_failed = false;
            JobCallbacksGuard callbacks(view.ctx, job);
            PromptImageCallbackGuard prompt_images(view.ctx, job.img_gen.gen_params);
            sd_image_generation_result_t* produced = nullptr;
            if (!generate_image(view.ctx, &attempt_params, &produced)) {
                return false;
            }
            results.adopt(produced, view.params->lora_model_dir);
            if (job.cancel.requested.load()) {
                return true;
            }
            if (!results.empty() && job.img_gen.adetailer_enabled && !runtime.default_gen_params->ad_model_path.empty()) {
                callbacks.begin_detailing();
                if (!apply_server_adetailer(runtime, view.ctx, *view.params, job.img_gen, attempt_params, results)) {
                    adetailer_failed = true;
                    return false;
                }
            }
            return true;
        },
        failure);
    if (!generated) {
        if (adetailer_failed && failure.code != SD_ERROR_CANCELLED) {
            failure.message = failure.message.empty() ? "ADetailer failed" : "ADetailer failed: " + failure.message;
        }
        return false;
    }
    if (job.cancel.requested.load()) {
        return fail_cancelled(failure);
    }

    const int num_results = results.count();
    if (num_results <= 0) {
        return fail_with(failure, "generate_image returned no results");
    }

    EncodedImageFormat encoded_format = EncodedImageFormat::PNG;
    if (job.img_gen.output_format == "jpeg") {
        encoded_format = EncodedImageFormat::JPEG;
    } else if (job.img_gen.output_format == "webp") {
        encoded_format = EncodedImageFormat::WEBP;
    }

    update_job_progress(job, "encoding", "image", -1, 0, num_results, "images");
    for (int i = 0; i < num_results; ++i) {
        if (results[i].data == nullptr) {
            continue;
        }

        const std::string metadata = job.img_gen.gen_params.embed_image_metadata
                                         ? results.generation_info(i)
                                         : "";
        auto image_bytes           = encode_image_to_vector(encoded_format,
                                                            results[i].data,
                                                            results[i].width,
                                                            results[i].height,
                                                            results[i].channel,
                                                            metadata,
                                                            job.img_gen.output_compression);
        if (image_bytes.empty()) {
            continue;
        }
        output_images.push_back(base64_encode(image_bytes));
        output_info.push_back(json::parse(results.generation_info(i)));
        update_job_progress(job, "encoding", "image", -1, i + 1, num_results, "images");
    }

    if (output_images.empty()) {
        return fail_with(failure, "generate_image returned empty encoded outputs");
    }

    return true;
}

bool execute_vid_gen_job(ServerRuntime& runtime,
                         AsyncGenerationJob& job,
                         std::string& output_media_b64,
                         std::string& output_media_mime_type,
                         int& output_frame_count,
                         int& output_fps,
                         json& output_info,
                         GenerationFailure& failure) {
    const sd_vid_gen_params_t params = job.vid_gen.to_sd_vid_gen_params_t();
    std::string str_params;

    SDImageVec results;
    int num_results             = 0;
    sd_audio_t* generated_audio = nullptr;

    const bool generated = run_job(
        runtime,
        job,
        [&](const ContextKeeper::ContextView& view) {
            sd_vid_gen_params_t attempt_params = params;
            if (view.force_vae_tiling) {
                attempt_params.vae_tiling_params.enabled = true;
            }
            JobCallbacksGuard callbacks(view.ctx, job);
            PromptImageCallbackGuard prompt_images(view.ctx, job.vid_gen.gen_params);
            sd_video_generation_result_t* produced = nullptr;
            if (!generate_video(view.ctx, &attempt_params, &produced)) {
                return false;
            }
            output_fps      = produced->fps;
            generated_audio = produced->audio;
            produced->audio = nullptr;
            results.adopt_video(produced, view.params->lora_model_dir);
            return true;
        },
        failure);
    if (!generated) {
        return false;
    }

    num_results = results.count();
    if (job.cancel.requested.load()) {
        free_sd_audio(generated_audio);
        return fail_cancelled(failure);
    }
    if (num_results <= 0) {
        free_sd_audio(generated_audio);
        return fail_with(failure, "generate_video returned no results");
    }
    json info   = json::parse(results.generation_info(0), nullptr, false);
    output_info = info.is_discarded() ? json::object() : std::move(info);
    if (job.vid_gen.gen_params.embed_image_metadata) {
        str_params = results.generation_info(0);
    }

    update_job_progress(job, "encoding", "video");
    std::vector<uint8_t> video_bytes = create_video_from_sd_images_to_vector(job.vid_gen.output_format,
                                                                             results.data(),
                                                                             num_results,
                                                                             output_fps,
                                                                             job.vid_gen.output_compression,
                                                                             generated_audio,
                                                                             str_params);
    free_sd_audio(generated_audio);
    if (video_bytes.empty()) {
        return fail_with(failure, "failed to encode generated video container");
    }

    output_media_b64       = base64_encode(video_bytes);
    output_media_mime_type = video_mime_type(job.vid_gen.output_format);
    output_frame_count     = num_results;
    return true;
}

void async_job_worker(ServerRuntime& runtime) {
    AsyncJobManager& manager = *runtime.async_job_manager;

    while (true) {
        std::shared_ptr<AsyncGenerationJob> job;
        {
            std::unique_lock<std::mutex> lock(manager.mutex);
            manager.cv.wait(lock, [&]() { return manager.stop || !manager.queue.empty(); });

            if (manager.stop && manager.queue.empty()) {
                break;
            }

            purge_expired_jobs(manager);
            if (manager.queue.empty()) {
                continue;
            }

            const std::string job_id = manager.queue.front();
            manager.queue.pop_front();

            auto it = manager.jobs.find(job_id);
            if (it == manager.jobs.end()) {
                continue;
            }

            job             = it->second;
            job->status     = AsyncJobStatus::Generating;
            job->started_at = unix_timestamp_now_ms();
            update_job_progress(*job, "waiting");
        }

        std::vector<std::string> output_images;
        json output_info;
        std::string output_media_b64;
        std::string output_media_mime_type;
        int output_frame_count = 0;
        int output_fps         = 0;
        GenerationFailure failure;
        bool ok = false;

        try {
            if (job->kind == AsyncJobKind::ImgGen) {
                ok = execute_img_gen_job(runtime, *job, output_images, output_info, failure);
            } else if (job->kind == AsyncJobKind::VidGen) {
                ok = execute_vid_gen_job(runtime,
                                         *job,
                                         output_media_b64,
                                         output_media_mime_type,
                                         output_frame_count,
                                         output_fps,
                                         output_info,
                                         failure);
            } else {
                fail_with(failure, "unsupported job kind");
            }
        } catch (const std::exception& error) {
            // An exception must not end the only thread that serves the queue.
            ok      = false;
            failure = failure_from_exception(error);
            LOG_ERROR("job %s stopped by an exception: %s", job->id.c_str(), error.what());
        } catch (...) {
            ok = false;
            fail_with(failure, "unexpected error while generating");
        }

        {
            std::lock_guard<std::mutex> lock(manager.mutex);
            auto it = manager.jobs.find(job->id);
            if (it == manager.jobs.end()) {
                continue;
            }

            job->completed_at = unix_timestamp_now_ms();
            finish_job_progress(*job);
            if (job->cancel.requested.load()) {
                job->status        = AsyncJobStatus::Cancelled;
                job->error_code    = "cancelled";
                job->error_message = "job cancelled by client";
                LOG_INFO("job %s cancelled", job->id.c_str());
            } else if (ok) {
                job->status                 = AsyncJobStatus::Completed;
                job->result_images_b64      = std::move(output_images);
                job->result_generation_info = std::move(output_info);
                job->result_media_b64       = std::move(output_media_b64);
                job->result_media_mime_type = std::move(output_media_mime_type);
                job->result_frame_count     = output_frame_count;
                job->result_fps             = output_fps;
                job->error_code.clear();
                job->error_message.clear();
            } else {
                const bool cancelled = failure.code == SD_ERROR_CANCELLED;
                job->status          = cancelled ? AsyncJobStatus::Cancelled : AsyncJobStatus::Failed;
                job->error_code      = failure_code_name(failure);
                job->error_message   = cancelled ? "job cancelled by client"
                                                 : (failure.message.empty() ? "unknown generation error" : failure.message);
                job->result_images_b64.clear();
                job->result_generation_info = nullptr;
                job->result_media_b64.clear();
                job->result_media_mime_type.clear();
                job->result_frame_count = 0;
                job->result_fps         = 0;
                if (cancelled) {
                    LOG_INFO("job %s cancelled", job->id.c_str());
                }
            }

            purge_expired_jobs(manager);
            manager.finished_cv.notify_all();
        }
    }
}
