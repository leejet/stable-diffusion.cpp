# Adding backend performance probes

`src/core/probe/backend_probe.h` provides the shared measurement machinery. Operator
policy stays separate: `src/core/probe/flash_attn.cpp`, for example, defines attention
shapes and its CPU/CUDA/HIP/Vulkan shortcuts, then calls `sd_run_backend_probe` for unknown
configurations. A future convolution probe can use the same runner without
changing the attention implementation.

For each representative shape, create a `BackendProbeCase`:

- `build(graph, candidate)` creates inputs and either the reference or candidate
  graph in `graph.ctx`, registers input tensors in `graph.inputs`, and assigns
  `graph.output`. Leave the output null if the requested implementation is
  unavailable; do not silently benchmark its fallback as the candidate.
- By default, inputs receive deterministic F32 data, with the same seed for both
  variants. Supply `initialize` for F16 convolution weights, integer inputs, or
  operator-specific values. It runs after allocation, outside timed execution.
  Both variants must receive equivalent logical inputs.
- Default output validation requires matching shapes and contiguous F32 outputs,
  checks finiteness, and applies absolute and relative tolerances. Supply
  `compare` for other output types or validation rules.

Graphs must be safe to execute repeatedly without modifying their inputs.
`BackendProbeGraph` owns its GGML context and buffer, so all return paths release
probe resources. Constructors allocate metadata only; `supported()` can also be
used before a known-architecture shortcut, without allocating device tensors.

`BackendProbeOptions` controls graph metadata capacity, warmup count, measurement
count, measured-time budget, minimum free device memory, comparison tolerances,
and minimum improvement. Warmup and allocation are excluded from timing. The
runner alternates the two variants, synchronizes the backend around execution,
and compares sums of per-case medians. The time budget is checked before each
measurement pair; it cannot interrupt an in-flight backend operation.

Only use `candidate_faster` when `completed` is true. On incomplete probes,
`error` reports why comparison failed and the timing totals may be partial.
The caller handles logging, known architecture rules, fallback policy, and
context-lifetime caching. Keep caches separate by operation (or key a shared
cache by both operation and backend); an attention result must not select a
convolution implementation. If shapes or settings vary within one context,
include them in the corresponding cache key as well.

No convolution policy or CLI option is added by this infrastructure.

Module assignment parsing remains in `src/core/ggml_extend_backend.cpp`, next
to backend assignment parsing. The probe directory does not parse CLI/API options.
