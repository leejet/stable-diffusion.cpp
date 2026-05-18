#include "layer_streaming_executor.hpp"

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
    if (!runner->GGMLRunner::compute(get_graph,
                                     n_threads,
                                     /*free_compute_buffer_immediately=*/free_buffer_after,
                                     output_out,
                                     output_ctx,
                                     /*skip_param_offload=*/true)) {
        return false;
    }
    if (invoke_post_compute && stage.post_compute) {
        stage.post_compute();
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
                   ggml_tensor**                   output_out,
                   ggml_context*                   output_ctx) {
    (void)cfg;  // Reserved for future executor-level config; per-engine config
                // lives in runner->streaming_engine_ and is already applied.

    if (runner == nullptr || !runner->is_streaming_enabled()) {
        LOG_ERROR("LayerStreaming::run_streaming: runner null or streaming not enabled");
        return false;
    }

    int64_t t_start = ggml_time_ms();
    LOG_INFO("%s layer-streaming start (%d layers)", runner->get_desc().c_str(), num_layers);

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
        runner->streaming_engine_->prime_prefetch(layer_name_at, 0, num_layers);
    }

    // Per-layer loop (chunk-K resident span lands in Task 3).
    for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        std::string layer_name = layer_name_at(layer_idx);

        if (runner->streaming_engine_) {
            runner->streaming_engine_->wait_for_prefetch(layer_name);
        }
        if (!registry.move_layer_to_gpu(layer_name)) {
            LOG_ERROR("%s: failed to load %s",
                      runner->get_desc().c_str(),
                      layer_name.c_str());
            runner->free_compute_buffer();
            return false;
        }
        if (runner->streaming_engine_) {
            runner->streaming_engine_->advance_prefetch(layer_name_at, layer_idx, num_layers);
        }

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

        registry.move_layer_to_cpu(layer_name);
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
    LOG_INFO("%s layer-streaming completed in %.2fs (%d layers)",
             runner->get_desc().c_str(),
             (t_end - t_start) / 1000.0,
             num_layers);

    return true;
}

}  // namespace LayerStreaming
