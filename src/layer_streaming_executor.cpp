#include "layer_streaming_executor.hpp"

#include <cstdlib>

#include "ggml_extend.hpp"
#include "tensor_registry.hpp"

namespace LayerStreaming {

namespace {

// Run a single stage: build graph, compute, call post_compute if set.
// Returns false on compute failure.
bool run_stage(GGMLRunner*   runner,
               int           n_threads,
               const Stage&  stage,
               ggml_tensor** output_out,
               ggml_context* output_ctx,
               bool          free_buffer_after,
               bool          invoke_post_compute) {
    auto get_graph = stage.build_graph;
    // If a post_compute will run, defer the compute-buffer free until after
    // it executes — otherwise ggml_backend_tensor_get on captured output
    // handles would read from a freed allocation.
    const bool will_post   = invoke_post_compute && static_cast<bool>(stage.post_compute);
    const bool free_in_run = free_buffer_after && !will_post;
    if (!runner->GGMLRunner::compute(get_graph,
                                     n_threads,
                                     /*free_compute_buffer_immediately=*/free_in_run,
                                     output_out,
                                     output_ctx,
                                     /*skip_param_offload=*/true)) {
        return false;
    }
    if (will_post) {
        stage.post_compute();
        if (free_buffer_after) {
            runner->free_compute_buffer();
        }
    }
    return true;
}

}  // namespace

bool run_streaming(GGMLRunner*                     runner,
                   int                             n_threads,
                   const StreamingConfig&          cfg,
                   Stage                           input_stage,
                   PerLayerStageFactory            per_layer_stage_factory,
                   Stage                           output_stage,
                   int                             num_layers,
                   std::function<std::string(int)> layer_name_at,
                   int                             start_layer_idx,
                   ggml_tensor**                   output_out,
                   ggml_context*                   output_ctx) {
    (void)cfg;  // Reserved for future executor-level config; per-engine config
                // lives in runner->streaming_engine_ and is already applied.

    if (runner == nullptr || !runner->is_streaming_enabled()) {
        LOG_ERROR("LayerStreaming::run_streaming: runner null or streaming not enabled");
        return false;
    }
    if (start_layer_idx < 0 || start_layer_idx > num_layers) {
        LOG_ERROR("LayerStreaming::run_streaming: invalid start_layer_idx=%d "
                  "(num_layers=%d)",
                  start_layer_idx,
                  num_layers);
        return false;
    }

    int64_t t_start = ggml_time_ms();
    const bool prof_enabled = std::getenv("SDCPP_STREAM_PROFILE") != nullptr;
    auto prof_now = []() { return ggml_time_us(); };
    int64_t prof_wait_us    = 0;
    int64_t prof_load_us    = 0;
    int64_t prof_advance_us = 0;
    int64_t prof_compute_us = 0;
    int64_t prof_evict_us   = 0;
    if (start_layer_idx > 0) {
        LOG_INFO("%s layer-streaming start (layers %d..%d of %d; %d pre-dispatched)",
                 runner->get_desc().c_str(),
                 start_layer_idx, num_layers - 1, num_layers,
                 start_layer_idx);
    } else {
        LOG_INFO("%s layer-streaming start (%d layers)",
                 runner->get_desc().c_str(), num_layers);
    }

    auto analysis = runner->analyze_vram_budget();
    if (analysis.fits_in_vram) {
        LOG_WARN("%s: invoked layer-streaming executor with fitting model; "
                 "expected caller to use coarse path",
                 runner->get_desc().c_str());
    }

    auto& registry = runner->streaming_engine_->get_registry();

    if (!registry.move_layer_to_gpu("_global")) {
        LOG_ERROR("%s: failed to load _global", runner->get_desc().c_str());
        return false;
    }

    // Stage 1: input (e.g. patch embed + timestep embed + pe).
    if (!run_stage(runner,
                   n_threads,
                   input_stage,
                   /*output_out=*/nullptr,
                   /*output_ctx=*/nullptr,
                   /*free_buffer_after=*/true,
                   /*invoke_post_compute=*/true)) {
        LOG_ERROR("%s: Stage 1 (input) failed", runner->get_desc().c_str());
        return false;
    }

    // Prime prefetch for the per-layer loop.
    if (runner->streaming_engine_) {
        runner->streaming_engine_->prime_prefetch(layer_name_at, start_layer_idx, num_layers);
    }

    // Per-layer loop. Layers [0..start_layer_idx) are assumed to have been
    // pre-dispatched by the caller as a chunk-K resident mega-graph.
    for (int layer_idx = start_layer_idx; layer_idx < num_layers; ++layer_idx) {
        std::string layer_name = layer_name_at(layer_idx);

        int64_t t0 = prof_enabled ? prof_now() : 0;

        if (runner->streaming_engine_) {
            runner->streaming_engine_->wait_for_prefetch(layer_name);
        }
        int64_t t1 = prof_enabled ? prof_now() : 0;

        if (!registry.move_layer_to_gpu(layer_name)) {
            LOG_ERROR("%s: failed to load %s",
                      runner->get_desc().c_str(),
                      layer_name.c_str());
            runner->free_compute_buffer();
            return false;
        }
        int64_t t2 = prof_enabled ? prof_now() : 0;

        if (runner->streaming_engine_) {
            runner->streaming_engine_->advance_prefetch(layer_name_at, layer_idx, num_layers);
        }
        int64_t t3 = prof_enabled ? prof_now() : 0;

        Stage layer_stage = per_layer_stage_factory(layer_idx, /*prev_gpu_output=*/nullptr);

        // Don't free buffer between iterations — same shape, reuse the gallocr.
        if (!run_stage(runner,
                       n_threads,
                       layer_stage,
                       /*output_out=*/nullptr,
                       /*output_ctx=*/nullptr,
                       /*free_buffer_after=*/false,
                       /*invoke_post_compute=*/true)) {
            LOG_ERROR("%s: layer %d execution failed",
                      runner->get_desc().c_str(),
                      layer_idx);
            runner->free_compute_buffer();
            return false;
        }
        int64_t t4 = prof_enabled ? prof_now() : 0;

        registry.move_layer_to_cpu(layer_name);
        int64_t t5 = prof_enabled ? prof_now() : 0;

        if (prof_enabled) {
            prof_wait_us    += t1 - t0;
            prof_load_us    += t2 - t1;
            prof_advance_us += t3 - t2;
            prof_compute_us += t4 - t3;
            prof_evict_us   += t5 - t4;
        }
    }

    runner->free_compute_buffer();

    // Stage 3 — writes directly into *output_out via output_ctx; no post_compute.
    if (output_stage.post_compute) {
        LOG_WARN("%s: output_stage.post_compute is set but will not be called; "
                 "the executor writes directly into the caller's buffer",
                 runner->get_desc().c_str());
    }
    if (!run_stage(runner,
                   n_threads,
                   output_stage,
                   output_out,
                   output_ctx,
                   /*free_buffer_after=*/true,
                   /*invoke_post_compute=*/false)) {
        LOG_ERROR("%s: Stage 3 (output) failed", runner->get_desc().c_str());
        return false;
    }

    int64_t t_end = ggml_time_ms();
    LOG_INFO("%s layer-streaming completed in %.2fs (%d streamed layers)",
             runner->get_desc().c_str(),
             (t_end - t_start) / 1000.0,
             num_layers - start_layer_idx);

    if (prof_enabled) {
        int64_t total_us = prof_wait_us + prof_load_us + prof_advance_us +
                           prof_compute_us + prof_evict_us;
        LOG_INFO("[stream-profile] %s %d streamed layers: total=%.2fms "
                 "wait=%.2fms load=%.2fms advance=%.2fms compute=%.2fms evict=%.2fms",
                 runner->get_desc().c_str(),
                 num_layers - start_layer_idx,
                 total_us / 1000.0,
                 prof_wait_us / 1000.0,
                 prof_load_us / 1000.0,
                 prof_advance_us / 1000.0,
                 prof_compute_us / 1000.0,
                 prof_evict_us / 1000.0);
    }

    return true;
}

}  // namespace LayerStreaming
