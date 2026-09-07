## Use Flash Attention to save memory and improve speed.

Enabling flash attention for the diffusion model reduces memory usage by varying amounts of MB.
eg.:
 - flux 768x768 ~600mb
 - SD2 768x768 ~1400mb

For most backends, it slows things down, but for cuda it generally speeds it up too.
At the moment, it is only supported for some models and some backends (like cpu, cuda/rocm, metal).

Run by adding `--diffusion-fa` to the arguments and watch for:
```
[INFO ] stable-diffusion.cpp:312  - Using flash attention in the diffusion model
```
and the compute buffer shrink in the debug log:
```
[DEBUG] ggml_runner.cpp:280 - flux compute buffer size: 650.00 MB(VRAM) on CUDA0 (peak across 1 segment)
```
This reports the actual peak compute workspace capacity per backend, including
CPU fallback. It excludes weights and cache buffers. Within a runner lifecycle,
the summary is printed only on the first graph or when backend capacities or the
segment count change.

## Offload weights to the CPU to save VRAM without reducing generation speed.

Using `--offload-to-cpu` allows you to offload weights to the CPU, saving VRAM without reducing generation speed.

## Use params backend to reduce VRAM or RAM usage.

`--params-backend` controls where model parameters are kept. If it is not set, parameters use the same backend as `--backend`, so a GPU runtime backend also keeps parameters in VRAM.

Use CPU params to reduce VRAM usage:

```shell
--backend cuda0 --params-backend cpu
```

This keeps model weights in system RAM and moves them to the runtime backend when needed. In the example CLI/server, `--offload-to-cpu` is a compatibility shortcut that prepends `*=cpu` to `--params-backend` before creating the context, so explicit module assignments can still override it:

```shell
--offload-to-cpu --params-backend te=disk
```

Use disk params to reduce both VRAM and RAM usage:

```shell
--backend cuda0 --params-backend disk
```

This reloads parameters from the model file on demand, retains unpinned compute copies while space permits, and releases them under pressure or at module-run completion. It has the lowest source-memory residency, but can be slower because evicted weights must be read again. `disk` is never selected implicitly; set it explicitly when RAM usage matters more than reload cost.

Per-module assignments can target only the largest modules:

```shell
--backend cuda0 --params-backend diffusion=disk,te=cpu,vae=cpu
```

See [backend selection](./backend.md) for full syntax.

## Run models that don't fit in VRAM (automatic segmented execution).

`--offload-to-cpu` keeps the source parameters in system RAM and creates compute-side GPU replicas on demand. Unpinned replicas remain resident for reuse, but automatic graph-cut execution evicts them from the last segment backward when the next weight or compute allocation needs space. Disk-backed parameters follow the same policy without retaining a RAM source copy.

When a graph has cut markers and its missing weights plus incremental compute workspace exceed the available device headroom, it runs its fixed segment list in order. A reusable monolithic compute buffer is not counted as a new allocation. An explicit `--max-vram` budget deducts already-resident managed weights and compute/cache buffers registered by every runner sharing the device, so later graph runs remain segmented when the full graph exceeds the budget. The current segment's weights are pinned during compute, and the next parameter-bearing segment is prefetched when the device supports asynchronous transfer. No opt-in streaming flag is required.

- `--max-vram <GiB>` optionally lowers the live-memory limit. A positive value is a managed per-device budget, `0` uses the device's current free memory without an explicit budget, and a negative value snapshots free memory at startup while reserving that many GiB (`--max-vram -1` reserves about 1 GiB). Driver contexts and unrelated external allocations remain outside the managed budget.
- `--disable-prefetch` disables asynchronous next-segment prefetch while retaining synchronous loading, eviction, and segmented execution.
- `--disable-segmented-compute` forces monolithic graph execution for diagnostics or compatibility, even when the automatic memory check would select segments.

Single-device monolithic execution also reclaims unpinned weight replicas before
loading weights or allocating compute workspace, including graphs without cut
markers and runs with `--disable-segmented-compute`. It still respects the managed
device budget and fails if the graph cannot fit after reclamation.

Segment completion releases active workspace use while retaining the runner's
allocator/scheduler capacity. Compatible gallocr reservations are reused across
graphs; idle workspaces can be reclaimed under pressure and are freed at runner
completion. Cross-graph caches survive individual graphs, but cut buffers do not.

The recommended shape for "biggest model my card can host" is:

```shell
sd-cli --diffusion-model flux1-dev.safetensors ... \
       --offload-to-cpu --max-vram -1
```

- `--offload-to-cpu`: params in RAM, staged as needed.
- `--max-vram -1`: reserve about 1 GiB from the startup free-memory snapshot; live free memory can still lower the effective limit for every graph.

Use `--params-backend diffusion=disk` instead when reducing system RAM residency is more important than avoiding repeated model-file reads.

## Use quantization to reduce memory usage.

[quantization](./quantization_and_gguf.md)
