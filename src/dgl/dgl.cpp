#include "stable-diffusion.h"

#include <algorithm>
#include <mutex>

#include "api/api.h"
#include "context.h"
#include "diagnostic.h"
#include "executor.h"
#include "graph.h"
#include "result.h"

namespace {
    using namespace sd::dgl;

    template <class F>
    sd_status_t api_call(F&& operation, sd_status_t failure = SD_STATUS_INVALID_ARGUMENT) {
        return sd::api::call([&] {
            try {
                operation();
                sd::api::clear_error();
                return SD_STATUS_OK;
            } catch (const Error& error) {
                return sd::api::error(error.status, nullptr, error.what());
            }
        },
                             failure);
    }

    std::unique_lock<std::mutex> lock_idle(Context* ctx) {
        if (!ctx)
            throw Error(SD_STATUS_INVALID_ARGUMENT, "context is required");
        std::unique_lock<std::mutex> lock(ctx->mutex, std::try_to_lock);
        if (!lock.owns_lock() || ctx->active_run || ctx->generating)
            throw Error(SD_STATUS_BUSY, "DGL context is busy");
        return lock;
    }

    static void clear_backend_buffer_caches(const Context& ctx) {
        // Idle weight release can refill a cache that runner cleanup already cleared.
        for (const auto& entry : *ctx.backends) {
            sd_backend_clear_buffer_cache(entry.second.get());
        }
    }

    class ActiveRun {
        Context& context_;

    public:
        RunEvents events;

        explicit ActiveRun(Context& context)
            : context_(context), events(context.callbacks, false) {
            context_.active_run = &events;
        }

        ~ActiveRun() {
            std::lock_guard<std::mutex> lock(context_.mutex);
            if (context_.active_run == &events)
                context_.active_run = nullptr;
        }

        void complete(std::unique_ptr<sd_dgl_result_t>& result, sd_dgl_result_t** out_result) {
            std::lock_guard<std::mutex> lock(context_.mutex);
            events.check_cancelled();
            *out_result         = result.release();
            context_.active_run = nullptr;
        }
    };

    Values bind_inputs(const Graph& graph, const sd_dgl_tensor_input_t* inputs, size_t input_count) {
        if (!inputs && input_count)
            throw Error(SD_STATUS_INVALID_ARGUMENT, "tensor input array is required");
        const auto& names = graph.syntax.entry.parameters;
        Values bound;
        for (size_t i = 0; i < input_count; ++i) {
            const auto& input = inputs[i];
            if (!input.name || !*input.name || !input.tensor)
                throw Error(SD_STATUS_INVALID_ARGUMENT, "each tensor input requires a name and tensor");
            if (std::find(names.begin(), names.end(), input.name) == names.end())
                throw Error(SD_STATUS_INVALID_ARGUMENT, "unknown tensor input '" + std::string(input.name) + "'");
            if (bound.count(input.name))
                throw Error(SD_STATUS_INVALID_ARGUMENT, "duplicate tensor input '" + std::string(input.name) + "'");
            auto retained = std::shared_ptr<sd_tensor_t>(sd_tensor_retain(input.tensor), sd_tensor_free);
            bound.emplace(input.name, Value::object(std::move(retained)));
        }
        for (const auto& name : names)
            if (!bound.count(name))
                throw Error(SD_STATUS_INVALID_ARGUMENT, "missing tensor input '" + name + "'");
        return bound;
    }

    std::unique_ptr<sd_dgl_result_t> execute(Context& ctx, RunEvents& events, const Graph& graph, Values inputs) {
        sd::FileAccessScope file_access(ctx.file_access);
        RunEventsScope event_scope(events);
        RunState state{ctx, events};
        ctx.models->clear_host_memory_error();
        try {
            events.report("executing");
            return std::make_unique<sd_dgl_result_t>(run(graph, state, std::move(inputs)));
        } catch (const std::exception& error) {
            events.check_cancelled();
            const auto& memory_error = ctx.models->host_memory_error();
            if (!memory_error.empty())
                throw Error(SD_STATUS_OUT_OF_MEMORY, std::string(error.what()) + ". " + memory_error);
            if (dynamic_cast<const Error*>(&error) || dynamic_cast<const std::bad_alloc*>(&error))
                throw;
            throw Error(SD_STATUS_GENERATION_FAILED, error.what());
        }
    }
}

const char* sd_dgl_builtin_catalog(void) {
    const char* result = nullptr;
    api_call([&] {
        static const std::string catalog = builtin_catalog();
        result                           = catalog.c_str();
    });
    return result;
}

const char* sd_dgl_error(void) {
    return sd_get_last_error();
}

sd_dgl_ctx_t* sd_dgl_ctx_create(void) {
    sd_dgl_ctx_t* result = nullptr;
    api_call([&] { result = new sd_dgl_ctx_t(); });
    return result;
}

sd_dgl_ctx_t* sd_dgl_ctx_create_once(void) {
    sd_dgl_ctx_t* result = nullptr;
    api_call([&] { result = new sd_dgl_ctx_t(false); });
    return result;
}

void sd_dgl_ctx_free(sd_dgl_ctx_t* ctx) {
    delete ctx;
}

bool sd_dgl_ctx_set(sd_dgl_ctx_t* ctx, const char* key, const char* value) {
    return api_call([&] {
               if (!key || !value)
                   throw Error(SD_STATUS_INVALID_ARGUMENT, "key and value are required");
               auto lock = lock_idle(ctx);
               ctx->set(key, value);
           }) == SD_STATUS_OK;
}

sd_status_t sd_dgl_ctx_release_staging(sd_dgl_ctx_t* ctx) {
    return api_call([&] {
        auto lock = lock_idle(ctx);
        ctx->models->release_idle_gpu_staging();
        clear_backend_buffer_caches(*ctx);
    });
}

sd_status_t sd_dgl_ctx_unload_models(sd_dgl_ctx_t* ctx) {
    return api_call([&] {
        auto lock = lock_idle(ctx);
        ctx->unload_models();
        clear_backend_buffer_caches(*ctx);
    });
}

bool sd_dgl_ctx_set_file_access_callback(sd_dgl_ctx_t* ctx, sd_dgl_file_access_cb_t callback, void* userdata) {
    return api_call([&] {
               auto lock        = lock_idle(ctx);
               ctx->file_access = {callback, userdata};
           }) == SD_STATUS_OK;
}

sd_status_t sd_dgl_ctx_set_progress_callback(sd_dgl_ctx_t* ctx, sd_generation_progress_cb_t callback, void* userdata) {
    return api_call([&] {
        auto lock                    = lock_idle(ctx);
        ctx->callbacks.progress      = callback;
        ctx->callbacks.progress_data = userdata;
    });
}

sd_status_t sd_dgl_ctx_set_preview_callback(sd_dgl_ctx_t* ctx, sd_dgl_preview_cb_t callback, const char* mode, int interval, bool denoised, bool noisy, void* userdata) {
    return api_call([&] {
        auto lock          = lock_idle(ctx);
        const auto preview = callback && mode ? str_to_preview(mode) : PREVIEW_NONE;
        if (callback && (!mode || preview == PREVIEW_COUNT))
            throw Error(SD_STATUS_INVALID_ARGUMENT, "preview mode must be none, proj, tae or vae");
        ctx->callbacks.preview      = callback;
        ctx->callbacks.preview_data = userdata;
        ctx->callbacks.mode         = preview;
        ctx->callbacks.interval     = interval;
        ctx->callbacks.denoised     = denoised;
        ctx->callbacks.noisy        = noisy;
    });
}

sd_status_t sd_dgl_ctx_cancel(sd_dgl_ctx_t* ctx) {
    return api_call([&] {
        if (!ctx)
            throw Error(SD_STATUS_INVALID_ARGUMENT, "context is required");
        std::lock_guard<std::mutex> lock(ctx->mutex);
        if (ctx->active_run)
            ctx->active_run->cancel();
    });
}

sd_status_t sd_dgl_check(const char* source, bool allow_tensor_inputs) {
    return api_call([&] {
        if (!source)
            throw Error(SD_STATUS_INVALID_ARGUMENT, "source is required");
        compile(source, operations(), allow_tensor_inputs);
    });
}

sd_status_t sd_dgl_run(sd_dgl_ctx_t* ctx, const char* source, const sd_dgl_tensor_input_t* inputs, size_t input_count, sd_dgl_result_t** result_out) {
    if (result_out)
        *result_out = nullptr;
    return api_call([&] {
        if (!source || !result_out)
            throw Error(SD_STATUS_INVALID_ARGUMENT, "source and result_out are required");
        auto lock = lock_idle(ctx);
        ActiveRun active(*ctx);
        lock.unlock();
        auto graph = compile(source, operations(), true);
        active.events.check_cancelled();
        auto bound  = bind_inputs(graph, inputs, input_count);
        auto result = execute(*ctx, active.events, graph, std::move(bound));
        active.complete(result, result_out);
    });
}
