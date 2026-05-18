#ifndef __LAYER_STREAMING_EXECUTOR_HPP__
#define __LAYER_STREAMING_EXECUTOR_HPP__

#include <functional>
#include <string>

#include "layer_streaming.hpp"

struct GGMLRunner;

namespace LayerStreaming {

// One stage of a streaming dispatch.
//
//   build_graph   - construct a fresh ggml_cgraph for this stage. Capture
//                   output tensor handles by reference (typically into runner
//                   class members) so post_compute can read them back.
//                   ggml_build_forward_expand must be called on the outputs.
//
//   post_compute  - optional. Called after the executor runs the graph.
//                   Use ggml_backend_tensor_get on the captured output
//                   handles to copy results into runner-owned pinned host
//                   buffers. Leave default-constructed (empty std::function)
//                   if no readback is needed; the executor checks
//                   bool(post_compute) before invoking. Skipped by the
//                   executor for intermediate layers inside a chunk-K
//                   resident span — only the final chunk layer's
//                   post_compute fires.
struct Stage {
    std::function<ggml_cgraph*()> build_graph;
    std::function<void()>         post_compute;
};

// Factory called once per layer to produce a fresh Stage.
//
// layer_idx       - 0-based layer index.
// prev_gpu_output - nullptr for streamed layers (factory rebuilds input from
//                   the runner's host buffer via set_backend_tensor_data).
//                   Non-null for layers inside the chunk-K resident span;
//                   the factory should use it directly as this layer's input
//                   so consecutive resident layers chain on-GPU.
using PerLayerStageFactory =
    std::function<Stage(int layer_idx, ggml_tensor* prev_gpu_output)>;

// Entry point. Drives a complete streaming dispatch:
//
//   1. analyze_vram_budget() -> coarse fallback when model fits, else per-layer.
//   2. Load _global params.
//   3. Run input_stage (build -> compute -> post_compute).
//   4. (Caller's responsibility) — if the model dispatches a chunk-K resident
//      mega-graph for layers [0..K-1] before calling run_streaming, it passes
//      start_layer_idx=K so the executor's per-layer loop and prefetch begin
//      at K. The model owns chunk-graph construction, caching, and dispatch;
//      typically via the LayerStreaming::ChunkGraph helper in chunk_graph.hpp.
//   5. Per-layer loop [start_layer_idx, num_layers), with async prefetch.
//   6. Run output_stage (build -> compute), writing into *output_out via
//      output_ctx. output_stage.post_compute is not called — the executor
//      writes results directly into the caller's buffer.
//   7. On any failure: free the per-layer compute buffer if allocated,
//      return false. Layer eviction is the caller's responsibility — the
//      outer pipeline calls offload_streaming_layers() at the per-step
//      level to clean up any GPU-resident layers from this run.
//
// Returns true on success, false on failure. The caller must handle layer
// eviction (typically via offload_streaming_layers() at the per-step boundary).
bool run_streaming(GGMLRunner*                     runner,
                   int                             n_threads,
                   const StreamingConfig&          cfg,
                   Stage                           input_stage,
                   PerLayerStageFactory            per_layer_stage_factory,
                   Stage                           output_stage,
                   int                             num_layers,
                   std::function<std::string(int)> layer_name_at,
                   // First layer index to stream. Callers using chunk-K
                   // resident dispatch pass start_layer_idx = K so the
                   // executor skips the already-dispatched resident block.
                   // Default 0 = stream every layer.
                   int                             start_layer_idx = 0,
                   ggml_tensor**                   output_out      = nullptr,
                   ggml_context*                   output_ctx      = nullptr);

}  // namespace LayerStreaming

#endif  // __LAYER_STREAMING_EXECUTOR_HPP__
