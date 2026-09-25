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

## Use VAE tiling to reduce encode and decode memory usage.

`--vae-tiling` enables spatial tiling for both VAE encoding and decoding. The
default tile size is 256x256 **image pixels**, independent of the VAE scale factor:

```shell
--vae-tiling --vae-tile-size 256x256 --vae-tile-overlap 0.5
```

`--vae-tile-size` accepts one size or `WIDTHxHEIGHT`. A zero dimension uses the
256-pixel default. Sizes are rounded down to a multiple of the VAE scale factor
and capped at the current input dimensions. Explicit sizes below four latent
pixels per axis (or the full axis when it is smaller) are rejected. Encoding and
decoding use the same spatial sizes, without an additional encoding multiplier.
Inputs that fit within a tile are processed as one tile.

For a 512x512 image with the default 50% overlap, both encoding and decoding use
3x3 tiles. A 256-pixel tile corresponds to 32 latent pixels for an 8x VAE, 16 for
a 16x VAE, and 8 for a 32x VAE. Smaller tiles reduce each graph's memory demand,
but overlapping work can increase processing time and tiling can affect image
quality, especially during encoding. Use larger tiles when more context is needed.

`--vae-relative-tile-size` overrides the absolute size on each axis with a positive
value. Values up to and including 1 specify a fraction of the current input size;
values greater than 1 specify a target number of tiles per axis, accounting for
overlap. For example, `0.5x0.5` uses half the width and height in both encode and
decode. The target overlap is clamped to 0 through 0.5 and the actual overlap is
adjusted to fit the image. Size and overlap options require `--vae-tiling`.

**Migration:** `--vae-tile-size` and the C/JSON fields `tile_size_w` and
`tile_size_h` now use image pixels instead of latent units. The C/JSON fields
`tile_size_x/y` have been renamed to `tile_size_w/h`, and `rel_size_x/y` to
`rel_size_w/h`. The command-line option names are unchanged. For example, an old
decode tile size of 32 corresponds to 256 pixels for an 8x VAE or 512 pixels for a
16x VAE. Encoding no longer enlarges explicit or relative tile sizes.

The main VAE decode path retries allocation failures with smaller tiles, even
without `--vae-tiling`. Supported video VAEs first try temporal tiling; spatial
retries use at most 256-pixel tiles initially and then halve the effective tile
dimensions down to the minimum size. Each spatial retry must reduce the effective
tile size. These runtime adjustments do not change the caller's parameters.
Execution failures are not retried, and encoding has no automatic OOM retry.

`--temporal-tiling` remains independent of spatial tiling. MiniMax H3 always uses
spatial tiling (256x256 pixels and 25% overlap by default) and its own temporal
windows. With `--vae-tiling`, its overlap follows `--vae-tile-overlap`; explicit
spatial sizes are honored.

## Offload weights to the CPU to save VRAM without reducing generation speed.

Using `--offload-to-cpu` allows you to offload weights to the CPU, saving VRAM without reducing generation speed.

## Use params backend to reduce VRAM or RAM usage.

`--params-backend` controls where model parameters are kept. If it is not set, auto-fit chooses parameter placement while preserving `--backend`. With `--auto-fit off`, parameters use the same backend as `--backend`, so a GPU runtime backend also keeps parameters in VRAM.

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

When choosing between monolithic and segmented execution, the runner requires
an additional 128 MiB of headroom in both available device memory and any explicit
managed budget. This planning headroom absorbs small allocation estimate changes;
subsequent capacity checks can consume it while still preserving the 512 MiB device
scratch reserve and respecting the managed budget.

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
