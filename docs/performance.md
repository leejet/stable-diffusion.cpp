## Use Flash Attention to save memory and improve speed.

Enabling flash attention for the diffusion model reduces memory usage by varying amounts of MB.
eg.:
 - flux 768x768 ~600mb
 - SD2 768x768 ~1400mb

`--flash-attn on|off|auto` controls flash attention. Its effect on speed depends
on GPU architecture, backend, driver, model, and tensor shapes. Automatic
selection uses backend rules and, for some paths, small probes; it does not
guarantee the fastest end-to-end generation. Compare `on` and `off` with the
same model and generation settings over repeated runs, separating startup/probe
time from steady-state generation time. See [backend selection](backend.md)
for automatic selection rules and per-module overrides.

Watch for:
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

## Tune direct convolution.

`--conv-direct on|off|auto` selects direct convolution or the im2col-based
path. Performance depends on GPU architecture, backend, driver, model, and
tensor shapes, so enabling direct convolution can either improve or reduce speed.

Automatic selection uses small benchmarks on Vulkan and implicit GEMM checks
on other backends. It does not guarantee the fastest end-to-end generation.
Compare `on` and `off` with the same model and generation settings over
repeated runs, keeping flash attention unchanged and separating startup/probe
time from steady-state generation time. See [backend selection](backend.md)
for automatic selection rules and per-module overrides.

## Run conditional and unconditional CFG in one batched UNet forward.

For UNet models, the conditional and unconditional guidance branches are
concatenated into a single batch of two and run through one UNet forward per
step instead of two separate forwards. This is enabled by default whenever the
run qualifies for it.

Use `--batched-cfg off` to force separate conditional and unconditional
forwards.

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

On discrete Vulkan devices, runners reuse released device buffers of the same
size to reduce allocation overhead during segmented execution. This idle buffer
cache is limited to 1 GiB per device and is reclaimed before allocating a new
size or at runner completion. It is outside the managed `--max-vram` budget,
so physical VRAM usage can be higher. Set `GGML_VK_REUSE_BUFFERS=0` to disable
it. Unified-memory and host-memory modes do not use this cache.

Discrete NVIDIA and non-GCN AMD Vulkan devices use a separate transfer queue by
default when available and graphics queue use is disabled, so weight prefetch can overlap compute. Set
`GGML_VK_ASYNC_USE_TRANSFER_QUEUE=0` to disable it.

On discrete NVIDIA devices, allocation chunks can grow up to 4 GiB, bounded by the device's buffer and allocation
limits, to reduce fragmentation within large graph workspaces. Other devices
retain the 1 GiB default. `GGML_VK_SUBALLOCATION_BLOCK_SIZE` overrides this limit
in bytes on any vendor; for example, `1073741824` selects 1 GiB and `4294967296`
selects 4 GiB, subject to device limits. This is a soft chunk limit, not a reservation
or a change to `--max-vram`: a single oversized tensor or exhausting the allocator's
chunk count can require a larger allocation, which may fail.

The recommended shape for "biggest model my card can host" is:

```shell
sd-cli --diffusion-model flux1-dev.safetensors ... \
       --offload-to-cpu --max-vram -1
```

- `--offload-to-cpu`: params in RAM, staged as needed.
- `--max-vram -1`: reserve about 1 GiB from the startup free-memory snapshot; live free memory can still lower the effective limit for every graph.

Use `--params-backend diffusion=disk` instead when reducing system RAM residency is more important than avoiding repeated model-file reads.

## Keep device memory below a pressure threshold (`--memory-guard`).

`--memory-guard <percent>` (e.g. `--memory-guard 90`) makes capacity management
act before allocations approach device limits instead of only when a graph no
longer fits. For every managed device, usage is measured against the
device-reported memory and, when set, the `--max-vram` budget — whichever is
more pressed:

- At or above the threshold, capacity checks evict unpinned weight replicas
  (later segments first, then least recently used) until usage drops below the
  threshold, and asynchronous next-segment prefetch is suspended until pressure
  recovers.
- At 5 points above the threshold the same reduction runs and is reported as
  critical pressure.
- The target is best effort: a graph that still fits is never failed just
  because the threshold cannot be reached, and a warning is logged once per
  device instead.

The guard covers memory managed by the runners (weights, compute workspaces,
caches). Driver contexts and allocations made outside the managed runners stay
invisible to it, so it is not a hard physical cap. On backends that do not
report device memory the guard only tracks the managed `--max-vram` budget.
`0` (the default) disables the guard.

On CPU backends the guard measures system RAM instead: used memory (total minus
the memory available without swapping, plus the allocation a graph is about to
make) against total physical memory. Under pressure it releases unpinned weights
that can be re-read from disk (`--params-backend disk`); weights that live only
in RAM cannot be released, so the guard logs a warning once and the run
continues. RAM is shared with other programs, so the percentage covers the whole
machine, not only this process. The guard never fails a graph on the CPU, and it
does nothing on platforms that cannot report available memory.

Weights that stream from disk are counted before they are loaded, so on the CPU
the guard sees what a graph is about to need and not only what is resident. When
that total would pass the threshold, a graph that has graph cuts runs in segments,
as it does on a GPU: each segment reads its weights, computes, and lets them go
before the next one starts, which keeps the peak well below the whole model. This
only applies when some weights can be released (`--params-backend disk`); if
every weight is resident in RAM, splitting the graph would not lower the peak and
the graph runs in one piece. `--disable-segmented-compute` turns segmentation off.

On a machine without a GPU, `--auto-fit` plans RAM against the CPU backend (see
[backend.md](./backend.md#automatic-placement---auto-fit-onoff)) and, when it
puts a component on disk, sets the guard to 90 unless `--memory-guard` is given.
On Linux, available memory comes from `/proc/meminfo`, which does not reflect the
limit of a container's memory cgroup; inside a container with such a limit, pass
`--memory-guard` and `--params-backend disk` explicitly and leave headroom.

## Pick a resource profile (`--profile`).

`--profile` sets, in one option, the settings that trade speed for a smaller
memory footprint:

| Profile | What it sets |
|---|---|
| `max` | nothing; the defaults of the engine |
| `balanced` | `--memory-guard 90` |
| `lite` | `--memory-guard 85`, `--vae-tiling`, `--batched-cfg off`, and `--multi-gpu on` when `--multi-gpu` is not set or is `auto` (an explicit `off` is kept; with one GPU it changes nothing) |
| `auto` | one of the three above, chosen from the machine (below) |

An option given on the command line always wins over the profile: with
`--profile lite --memory-guard 70` the guard is 70, and `--memory-guard 0` keeps
it off. A profile does not set `--params-backend` or `--backend`, so `--auto-fit`
still decides where the weights live. It does not change the size, the steps or
the sampler of an image; it only changes where the memory goes, and with it the
speed (the tiled VAE is slower, unbatched CFG needs two forwards per step). None
of the profiles you can choose hands compute to the CPU: the GPUs keep all of it.
Before this change `lite` also put the text encoder on the CPU; if you want that,
ask for it with `--clip-on-cpu` or `--backend te=cpu`.

`auto` adds up the size of the model files and a working-memory reserve (1 GiB
plus a fifth of the weights) and compares it with the memory where the compute
happens: the free memory of the discrete GPU with the most, or the available RAM
without one. With several discrete GPUs of the same backend it counts their free
memory together (less 2 GiB of working memory for each GPU after the first), because
`--auto-fit` can spread a model over them (see `--multi-gpu` in
[backend.md](./backend.md#several-gpus---multi-gpu-autoonoff)); `--multi-gpu off`, an
explicit `--backend` or `--params-backend`, or `--auto-fit off` keep the count to one
GPU. `max` when it fits with a quarter to spare, `balanced` when it fits,
`lite` when it does not and the weights will have to be streamed. The decision and
the figures are logged, for example:

```
[INFO ] profile auto -> lite (model 6.9 GiB + 2.4 GiB working memory, RAM available 5.0 GiB, GPU free 5.5 GiB: does not fit, weights will be streamed)
[INFO ] profile auto -> balanced (model 14.0 GiB + 3.8 GiB working memory, RAM available 8.0 GiB, GPU free 22.0 GiB over 2 GPUs: fits with little margin)
```

With no GPU the line says `no GPU memory reported` and the available RAM decides.

The estimate comes from the file sizes (it does not read the models) and does not
account for `--type` conversions at load time. When the sizes or the memory are
unknown, `auto` leaves the defaults alone. The thresholds are heuristics.

### More levels for the server

`sd-server` can go beyond `lite` on its own. When a request runs out of memory, it
loads the model again with the next lighter profile and runs the request again:

| Level | What it sets |
|---|---|
| `lite-disk` | `--params-backend disk` (weights are streamed from the file instead of kept in memory); `--eager-load` is switched off |
| `lite-hybrid` | the text encoder on the CPU (`--clip-on-cpu`); only when the machine has another device |
| `lite-cpu` | everything on the CPU (`--backend cpu`); only when the machine has another device |

These are not values of `--profile`. Up to `lite-disk` all the compute stays on the
GPUs, so a model that is short of memory gets slower but is not handed to the CPU;
the CPU takes work only from `lite-hybrid` on, and all of it only at `lite-cpu`.
The server climbs back one level at a time after `--degradation-cooldown` seconds;
every climb loads the model once, which takes long on a large model. If that cost is
not worth it, start the server on the lighter profile with `--profile lite` and
`--degradation-cooldown 0`, or use `--disable-degradation` to get the error at once.

A device that fails is not a memory problem, so it does not walk down this ladder.
The server tries the same device once more, then every other GPU the engine lists
(graphics cards first, a card through its vendor's backend before the same card
through Vulkan, integrated GPUs last) and takes the CPU only when none is left. While
a GPU is left to try it also stops the engine from falling back to the CPU by itself,
which would hide the failure behind a slow run. `GET /sdcpp/v1/health` shows the
`device`, the `cpu_modules` and the `fallback_devices`, and the log says every move.
See [Failures and recovery](../examples/server/api.md#failures-and-recovery).

## Use quantization to reduce memory usage.

[quantization](./quantization_and_gguf.md)
