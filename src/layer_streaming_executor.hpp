#ifndef __SD_LAYER_STREAMING_EXECUTOR_HPP__
#define __SD_LAYER_STREAMING_EXECUTOR_HPP__

#include <functional>
#include <string>
#include <ggml.h>

#include "layer_streaming.hpp"

class GGMLRunner;

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
//                   buffers. Skipped by the executor for intermediate
//                   layers inside a chunk-K resident span.
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
//   4. Build/cache and run the chunk-K resident mega-graph if any layers are
//      resident.
//   5. Per-layer loop for non-resident layers, with async prefetch.
//   6. Run output_stage (build -> compute), writing into *output_out via output_ctx.
//   7. On any failure, evict and free buffers, return false.
//
// Returns true on success, false on any failure (with side effects already
// rolled back).
bool run_streaming(GGMLRunner*                     runner,
                   int                             n_threads,
                   const StreamingConfig&          cfg,
                   Stage                           input_stage,
                   PerLayerStageFactory            per_layer_stage_factory,
                   Stage                           output_stage,
                   int                             num_layers,
                   std::function<std::string(int)> layer_name_at,
                   ggml_tensor**                   output_out,
                   ggml_context*                   output_ctx);

}  // namespace LayerStreaming

#endif  // __SD_LAYER_STREAMING_EXECUTOR_HPP__
