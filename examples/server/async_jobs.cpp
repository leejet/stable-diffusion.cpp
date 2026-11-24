// Extracted from main.cpp during server refactor.

#include "async_jobs.h"

#include <algorithm>
#include <chrono>
#include <cwctype>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string_view>
#include <thread>

#include "common/log.h"
#include "common/media_io.h"
#include "common/resource_owners.hpp"
#include "model_catalog.h"

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

static void preview_callback(int step, int frame_count, const sd_tensor_t* const* tensors, bool is_noisy, void* data) {
    std::vector<SDImage> views;
    for (int i = 0; i < frame_count; ++i)
        views.push_back(tensor_image_view(tensors[i]));
    SDImage* frames = views.data();
    (void)is_noisy;
    if (frame_count <= 0 || frames == nullptr || frames[0].data == nullptr) {
        return;
    }

    AsyncGenerationJob* job      = static_cast<AsyncGenerationJob*>(data);
    const SDImage& frame         = frames[0];
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
    sd_set_preview_callback(nullptr, "none", 1, false, false, nullptr);
}

static void set_preview_callback(AsyncGenerationJob& job) {
    const auto& mode = job.kind == AsyncJobKind::ImgGen ? job.img_gen.preview_mode : job.vid_gen.preview_mode;
    if (mode == "none")
        return;
    int interval = job.kind == AsyncJobKind::ImgGen
                       ? job.img_gen.preview_interval
                       : job.vid_gen.preview_interval;
    if (interval <= 0) {
        interval = 1;
    }
    sd_set_preview_callback(preview_callback, mode.c_str(), interval, true, false, &job);
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

    static void progress_callback(const sd_generation_progress_t* progress, void* data) {
        auto& guard = *static_cast<JobCallbacksGuard*>(data);
        update_job_progress(guard.job_, progress->stage, progress->detail,
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

    JobCallbacksGuard(const JobCallbacksGuard&)            = delete;
    JobCallbacksGuard& operator=(const JobCallbacksGuard&) = delete;
};

const char* async_job_kind_name(AsyncJobKind kind) {
    switch (kind) {
        case AsyncJobKind::ImgGen:
            return "img_gen";
        case AsyncJobKind::VidGen:
            return "vid_gen";
        case AsyncJobKind::Dgl:
            return "dgl";
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
    job.model.reset();
    job.request_body = nullptr;
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
        if (job.kind == AsyncJobKind::Dgl) {
            result["result"] = job.dgl_result_info;
        } else if (job.kind == AsyncJobKind::VidGen) {
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

static bool prepare_native_job(ServerRuntime& runtime, AsyncGenerationJob& job, std::string& error_message) {
    if (job.cancel.requested.load())
        return false;
    const SDMode mode = job.kind == AsyncJobKind::VidGen ? VID_GEN : IMG_GEN;
    if (job.model) {
        auto model = std::move(job.model);
        return set_current_model(runtime, std::move(*model), mode, error_message);
    }
    auto body = std::move(job.request_body);
    if (job.model_revision == runtime.models->revision)
        return true;
    // An earlier queued job can change the model and its defaults before this job starts.
    if (!runtime_supports_generation_mode(runtime, mode)) {
        error_message = unsupported_generation_mode_error(runtime, mode);
        return false;
    }
    try {
        return parse_native_job_request(body, runtime, job, error_message);
    } catch (const std::exception& error) {
        error_message = error.what();
        return false;
    }
}

bool execute_img_gen_job(ServerRuntime& runtime,
                         AsyncGenerationJob& job,
                         std::vector<std::string>& output_images,
                         json& output_info,
                         GenerationFailure& failure) {
    output_info = json::array();

    SDImageVec results;
    {
        // The model switch and the generation share the lock: no other user of the context comes in between.
        std::lock_guard<std::mutex> lock(*runtime.sd_ctx_mutex);
        std::string error_message;
        if (!prepare_native_job(runtime, job, error_message)) {
            if (job.cancel.requested.load()) {
                return fail_cancelled(failure);
            }
            return fail_with(failure, error_message.empty() ? "the model could not be prepared" : error_message.c_str());
        }
        const bool generated = run_job(
            runtime,
            job,
            [&](const ContextKeeper::ContextView& view) {
                JobCallbacksGuard callbacks(view.ctx, job);
                std::string message;
                return generate_server_images(view.ctx, job.img_gen, results, message, view.force_vae_tiling);
            },
            failure);
        if (!generated) {
            return false;
        }
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
        output_info.push_back(results.generation_info(i));
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
    std::string str_params;

    SDImageVec results;
    int num_results          = 0;
    SDAudio* generated_audio = nullptr;

    {
        // The model switch and the generation share the lock: no other user of the context comes in between.
        std::lock_guard<std::mutex> lock(*runtime.sd_ctx_mutex);
        std::string error_message;
        if (!prepare_native_job(runtime, job, error_message)) {
            if (job.cancel.requested.load()) {
                return fail_cancelled(failure);
            }
            return fail_with(failure, error_message.empty() ? "the model could not be prepared" : error_message.c_str());
        }
        const bool generated = run_job(
            runtime,
            job,
            [&](const ContextKeeper::ContextView& view) {
                SDOptPtr params = job.vid_gen.to_video_opt();
                if (view.force_vae_tiling) {
                    sd_opt_set_bool(params.get(), "vae_tiling.enabled", true);
                }
                JobCallbacksGuard callbacks(view.ctx, job);
                PromptImageCallbackGuard prompt_images(view.ctx, job.vid_gen.gen_params);
                sd_result_t* produced = nullptr;
                if (!generate_sd_video(view.ctx, params, &produced)) {
                    return false;
                }
                output_fps      = sd_result_get_fps(produced);
                generated_audio = retain_result_audio(produced);
                results.adopt_video(produced);
                return true;
            },
            failure);
        if (!generated) {
            return false;
        }
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

int64_t dgl_image_frame_count(const sd_tensor_t* tensor) {
    if (!tensor || sd_tensor_type(tensor) != SD_TENSOR_U8)
        return 0;
    const auto rank = sd_tensor_rank(tensor);
    if (rank != 3 && rank != 4)
        return 0;
    const auto channels = sd_tensor_dim(tensor, 0);
    const auto width    = sd_tensor_dim(tensor, 1);
    const auto height   = sd_tensor_dim(tensor, 2);
    if (channels < 1 || channels > 4 || width <= 0 || height <= 0 ||
        sd_tensor_byte_size(tensor) == 0 || sd_tensor_byte_size(tensor) > std::numeric_limits<int>::max())
        return 0;
    return rank == 3 ? 1 : sd_tensor_dim(tensor, 3);
}

static json describe_dgl_media(const sd_dgl_result_t* result) {
    json media       = json::array();
    size_t remaining = 64 * 1024 * 1024;
    auto append      = [&](const sd_dgl_value_t* value, const std::string& name) {
        if (sd_dgl_value_kind(value) != SD_DGL_VALUE_DICT)
            return;
        const auto* data = sd_dgl_value_field(value, "data");
        const auto* mime = sd_dgl_value_field(value, "mime_type");
        if (sd_dgl_value_kind(data) != SD_DGL_VALUE_BYTES || sd_dgl_value_kind(mime) != SD_DGL_VALUE_STRING)
            return;
        const char* mime_text = nullptr;
        size_t mime_size      = 0;
        const void* bytes     = nullptr;
        size_t size           = 0;
        if (sd_dgl_value_get_string(mime, &mime_text, &mime_size) != SD_STATUS_OK ||
            sd_dgl_value_get_bytes(data, &bytes, &size) != SD_STATUS_OK)
            throw std::runtime_error(sd_get_last_error());
        const std::string mime_type(mime_text, mime_size);
        if (mime_type != "image/png" && mime_type != "video/webm" && mime_type != "video/x-msvideo")
            return;
        json item = {{"name", name}, {"mime_type", mime_type}};
        if (size > remaining) {
            item["error"] = {{"code", "inline_too_large"}, {"message", "DGL media exceeds the 64 MiB inline result budget; use media::save"}};
        } else {
            item["b64_json"] = base64_encode(bytes, size);
            remaining -= size;
        }
        media.push_back(std::move(item));
    };
    const auto* root = sd_dgl_result_root(result);
    if (sd_dgl_value_kind(root) == SD_DGL_VALUE_DICT) {
        append(root, "result");
        if (media.empty())
            for (size_t i = 0; i < sd_dgl_value_count(root); ++i) {
                const auto* value = sd_dgl_value_at(root, i);
                if (const auto* name = sd_dgl_value_name(value))
                    append(value, name);
            }
    }
    return media;
}

static json describe_dgl_result(const sd_dgl_result_t* result) {
    const std::string type = sd_dgl_result_type(result);
    json outputs           = json::array();
    for (size_t i = 0; i < sd_dgl_result_count(result); ++i) {
        const auto* name   = sd_dgl_result_name(result, i);
        const auto* tensor = sd_dgl_result_tensor(result, name);
        const auto dtype   = sd_tensor_type(tensor);
        const auto bytes   = sd_tensor_byte_size(tensor);
        std::vector<int64_t> shape;
        for (size_t axis = 0; axis < sd_tensor_rank(tensor); ++axis)
            shape.push_back(sd_tensor_dim(tensor, axis));
        json output      = {{"name", name}, {"dtype", dtype}, {"shape", shape}, {"bytes", bytes}, {"frames", dgl_image_frame_count(tensor)}};
        const auto* data = sd_tensor_data(tensor);
        if (type == "string" && std::string_view(name) == "result") {
            output["value"] = bytes ? std::string(static_cast<const char*>(data), bytes) : "";
        } else if (shape.empty() && bytes != 0) {
            switch (dtype) {
                case SD_TENSOR_U8:
                    output["value"] = *static_cast<const uint8_t*>(data);
                    break;
                case SD_TENSOR_I32:
                    output["value"] = *static_cast<const int32_t*>(data);
                    break;
                case SD_TENSOR_I64:
                    output["value"] = std::to_string(*static_cast<const int64_t*>(data));
                    break;
                case SD_TENSOR_F32:
                    output["value"] = *static_cast<const float*>(data);
                    break;
                case SD_TENSOR_F64:
                    output["value"] = *static_cast<const double*>(data);
                    break;
            }
        }
        outputs.push_back(std::move(output));
    }
    return {{"type", type}, {"outputs", std::move(outputs)}, {"media", describe_dgl_media(result)}};
}

namespace {
    namespace fs = std::filesystem;

    bool allowed_dgl_path(const char* path, sd_dgl_file_access_t access, void* userdata) try {
        const auto& policy = *static_cast<const ServerFileAccess*>(userdata);
        std::error_code error;
        const auto file = fs::u8path(path);
        if (file.empty())
            return false;
#ifdef _WIN32
        if (file.native().compare(0, 4, L"\\\\?\\") == 0 || file.native().compare(0, 4, L"\\\\.\\") == 0)
            return false;
        for (const auto& part : file.relative_path()) {
            const auto name = part.native();
            if (name.empty() || name == L"." || name == L"..")
                continue;
            if (name.back() == L'.' || name.back() == L' ' || name.find(L':') != std::wstring::npos)
                return false;
            auto device = name.substr(0, name.find(L'.'));
            while (!device.empty() && device.back() == L' ')
                device.pop_back();
            std::transform(device.begin(), device.end(), device.begin(), [](wchar_t ch) { return std::towupper(ch); });
            if (device == L"CON" || device == L"PRN" || device == L"AUX" || device == L"NUL" ||
                device == L"CONIN$" || device == L"CONOUT$" ||
                (device.size() == 4 && (device.compare(0, 3, L"COM") == 0 || device.compare(0, 3, L"LPT") == 0) &&
                 std::wstring_view(L"123456789\u00b9\u00b2\u00b3").find(device.back()) != std::wstring_view::npos))
                return false;
        }
#endif
        const auto absolute = fs::absolute(file, error);
        if (error)
            return false;
        const auto resolved = fs::weakly_canonical(absolute, error);
        if (error)
            return false;
        if (access == SD_DGL_FILE_WRITE) {
            if (!model_path_inside(policy.output, resolved))
                return false;
            for (auto part = absolute; part != part.root_path(); part = part.parent_path()) {
                const auto status = fs::symlink_status(part, error);
                if (fs::is_symlink(status) || (error && status.type() != fs::file_type::not_found))
                    return false;
                error.clear();
            }
            const auto status = fs::symlink_status(resolved, error);
            if (status.type() != fs::file_type::not_found) {
                return !error && fs::is_regular_file(status) && fs::hard_link_count(resolved, error) == 1 && !error;
            }
            error.clear();
            fs::create_directories(resolved.parent_path(), error);
            return !error;
        }
        const auto status   = fs::status(resolved, error);
        const bool readable = !error && (fs::is_regular_file(status) || fs::is_directory(status));
        if (access == SD_DGL_FILE_READ)
            return readable;
        const auto key = model_path_key(resolved);
        if (readable || status.type() == fs::file_type::not_found) {
            if (policy.configured_files.count(key))
                return true;
            if (std::any_of(policy.model_bundles.begin(), policy.model_bundles.end(), [&](const fs::path& root) { return model_path_inside(root, resolved); }))
                return true;
        }
        return readable && std::any_of(policy.models.begin(), policy.models.end(), [&](const ModelRoot& root) {
                   return model_path_key(root.path) == key || model_path_inside(root.path, resolved);
               });
    } catch (const std::exception&) {
        return false;
    }
}

void configure_server_file_access(ServerModels& models, const SDSvrParams& params, const SDContextParams& config) {
    auto& policy                 = models.file_access;
    policy                       = {get_model_roots(params), fs::weakly_canonical(fs::current_path()) / "output", {}, {}};
    models.workspace_file_access = policy;
    for (const auto* directory : {&config.lora_model_dir, &config.hires_upscalers_dir})
        if (!directory->empty())
            policy.models.push_back(make_model_root(fs::u8path(*directory), false));
    for (const auto& entry : config.embedding_map)
        policy.configured_files.insert(model_path_key(fs::weakly_canonical(fs::absolute(fs::u8path(entry.second)))));
    add_startup_model_sources(policy, config);
}

bool apply_server_file_access(ServerModels& models, sd_ctx_t* ctx) {
    return sd_ctx_set_file_access_callback(ctx, allowed_dgl_path, &models.file_access) == SD_STATUS_OK &&
           sd_dgl_ctx_set_file_access_callback(sd_ctx_get_dgl_ctx(ctx), allowed_dgl_path, &models.workspace_file_access);
}

static bool execute_dgl_job(ServerRuntime& runtime,
                            AsyncGenerationJob& job,
                            std::shared_ptr<sd_dgl_result_t>& result,
                            json& result_info,
                            GenerationFailure& failure) {
    auto source                 = std::move(job.dgl_source);
    sd_dgl_result_t* raw_result = nullptr;
    std::string error_message;
    {
        std::lock_guard<std::mutex> lock(*runtime.sd_ctx_mutex);
        update_job_progress(job, "executing", "dgl");
        // A script shares the components of the loaded model; without a model it runs on a context of the
        // server's settings that lasts for this job only.
        const bool ran = runtime.keeper->with_context(
            [&](sd_ctx_t* ctx) {
                if (sd_dgl_run(sd_ctx_get_dgl_ctx(ctx), source.c_str(), nullptr, 0, &raw_result) == SD_STATUS_OK) {
                    return true;
                }
                error_message = sd_dgl_error();
                return false;
            },
            failure,
            &runtime.models->base_context);
        if (!ran) {
            if (failure.code == SD_ERROR_NONE) {
                failure.code    = SD_ERROR_UNKNOWN;
                failure.message = error_message.empty() ? "the DGL script failed" : error_message;
            }
            return false;
        }
    }
    result.reset(raw_result, sd_dgl_result_free);
    result_info = describe_dgl_result(result.get());
    return true;
}

void AsyncJobManager::record_context_job_completion(const SDSvrParams& params) {
    const auto now = std::chrono::steady_clock::now();
    if (params.weight_staging_ttl >= 0)
        staging_deadline = now + std::chrono::seconds(params.weight_staging_ttl);
    if (params.weight_residency_ttl >= 0)
        residency_deadline = now + std::chrono::seconds(params.weight_residency_ttl);
}

static void wait_for_job(ServerRuntime& runtime, std::unique_lock<std::mutex>& lock) {
    auto& manager       = *runtime.async_job_manager;
    using Clock         = std::chrono::steady_clock;
    const auto disabled = Clock::time_point::max();
    while (!manager.stop && manager.queue.empty()) {
        const auto deadline = std::min(manager.staging_deadline, manager.residency_deadline);
        if (deadline == disabled || manager.synchronous_jobs != 0) {
            manager.cv.wait(lock);
            continue;
        }
        const auto now = Clock::now();
        if (now < deadline) {
            manager.cv.wait_until(lock, deadline);
            continue;
        }

        // Keep job admission serialized with the idle check and the DGL release.
        const bool unload  = now >= manager.residency_deadline;
        sd_status_t status = SD_STATUS_OK;
        const bool idle    = runtime.keeper->try_with_context([&](sd_ctx_t* ctx) {
            auto* dgl = sd_ctx_get_dgl_ctx(ctx);
            status    = unload ? sd_dgl_ctx_unload_models(dgl) : sd_dgl_ctx_release_staging(dgl);
        });
        if (!idle || status == SD_STATUS_BUSY) {
            manager.cv.wait_for(lock, std::chrono::milliseconds(100));
            continue;
        }
        if (status != SD_STATUS_OK)
            LOG_WARN("idle weight release failed: %s", sd_dgl_error());
        if (unload)
            manager.residency_deadline = disabled;
        manager.staging_deadline = disabled;
    }
}

void async_job_worker(ServerRuntime& runtime) {
    AsyncJobManager& manager = *runtime.async_job_manager;

    while (true) {
        std::shared_ptr<AsyncGenerationJob> job;
        {
            std::unique_lock<std::mutex> lock(manager.mutex);
            wait_for_job(runtime, lock);

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
        std::shared_ptr<sd_dgl_result_t> dgl_result;
        json dgl_result_info;
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
            } else if (job->kind == AsyncJobKind::Dgl) {
                ok = execute_dgl_job(runtime, *job, dgl_result, dgl_result_info, failure);
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
            manager.record_context_job_completion(*runtime.svr_params);
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
                job->dgl_result             = std::move(dgl_result);
                job->dgl_result_info        = std::move(dgl_result_info);
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
