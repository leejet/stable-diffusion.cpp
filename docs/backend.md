# Backend selection

`stable-diffusion.cpp` has two backend assignments:

- `--backend` selects the runtime backend used to execute model graphs.
- `--params-backend` selects where model parameters are kept.

If `--params-backend` is not set, parameters use the same backend as their module runtime backend.

## Syntax

A backend assignment can be a single backend name:

```shell
sd-cli -m model.safetensors -p "a cat" --backend cpu
```

This applies to every module that does not have a more specific assignment.

Assignments can also target individual modules:

```shell
sd-cli -m model.safetensors -p "a cat" --backend te=cpu,vae=cuda0,diffusion=vulkan0
```

The same syntax is used for parameter placement:

```shell
sd-cli -m model.safetensors -p "a cat" --backend cuda0 --params-backend te=cpu,vae=cpu
```

`--params-backend` also accepts the special value `disk`:

```shell
sd-cli -m model.safetensors -p "a cat" --backend cuda0 --params-backend disk
```

`--max-vram` can target resolved backend/device names:

```shell
sd-cli -m model.safetensors -p "a cat" --backend diffusion=cuda0,vae=vulkan0 --max-vram cuda0=6,vulkan0=2
```

The budget applies to every module running on that backend.

Module names are case-insensitive. Hyphens and underscores in module names are ignored, so `clip_vision`, `clip-vision`, and `clipvision` are equivalent.

`all=`, `default=`, and `*=` can be used to set the default backend inside a mixed assignment:

```shell
sd-cli -m model.safetensors -p "a cat" --backend all=cuda0,te=cpu
```

## Multiple devices per module (layer split)

A `--backend` module assignment can list several devices separated by `&`:

```shell
sd-cli -m model.safetensors -p "a cat" --backend "diffusion=cuda0&cuda1"
```

The module's transformer blocks are then distributed across the listed devices
in contiguous ranges sized proportionally to each device's free memory (minus a
compute-buffer headroom of about 2 GiB per device), and the
module's graphs are executed with a `ggml_backend_sched` that runs each block
on the device holding its weights, copying the residual stream at the range
boundaries. The first device in the list is the module's main device: it also
holds the non-block tensors (embeddings, final norms, small sub-runners such as
CLIP models or projectors) and the graph inputs/outputs.

Layer split is supported for the `diffusion` and `te` modules. For `te` it
applies to the dominant text encoder (`t5xxl` or the LLM); other modules accept
only a single device. If the module has no recognizable transformer blocks, the
assignment falls back to the first listed device.

`--params-backend` accepts no device lists. If the module has no explicit
params assignment, each block range's parameters are loaded directly to (and,
with `--params-backend diffusion=disk`, released directly from) its own device;
an explicit assignment such as `te=cpu` keeps the parameters on that backend
and stages each range to its device on demand.

Layer split cannot be combined with `--max-vram` graph-cut segmentation or
`--stream-layers` for the split module; those are single-device mechanisms and
are disabled for it.

Use `--list-devices` to see the device names available on the system.

### Row split (`--split-mode row`)

`--split-mode` selects how a multi-device module distributes its weights:
`layer` (the default, described above) or `row`. It accepts a single mode or
per-module assignments:

```shell
sd-cli -m model.safetensors -p "a cat" --backend "diffusion=cuda0&cuda1" --split-mode row
sd-cli -m model.safetensors -p "a cat" --backend "diffusion=cuda0&cuda1,te=cuda0&cuda1" --split-mode diffusion=row,te=layer
```

In row mode the module keeps executing on its main (first listed) device, but
its transformer-block matmul weights are allocated in the backend's row-split
buffer type, which slices each weight's rows across the listed devices in
proportion to free memory and runs those matmuls on all devices in parallel.
Compared to a layer split this uses all GPUs within every layer (instead of
sequentially device by device) at the cost of a cross-device reduction per
matmul - usually the faster option when the devices have fast interconnect.

Row split requires backend support for split buffers and is currently
available on CUDA only; on other backends (or when the listed devices belong
to different backend registries) the module falls back to a layer split.
Embeddings, normalization weights, biases and other non-block tensors stay in
regular buffers on the main device.

Direct ("immediately") LoRA application cannot patch row-split tensors; with
`--split-mode row` the automatic LoRA mode selects runtime application, and an
explicit `--lora-apply-mode immediately` skips the split tensors with a
warning.

## Automatic placement (`--auto-fit`)

`--auto-fit` derives the `diffusion` / `te` / `vae` placements from the model
metadata and the per-device memory budgets, then feeds them into the same
backend assignment mechanism described above (the chosen specs are printed).
`--backend` and `--params-backend` are ignored while auto-fit is enabled.

```shell
sd-cli -m model.safetensors -p "a cat" --auto-fit
sd-cli -m model.safetensors -p "a cat" --auto-fit --max-vram cuda0=8,cuda1=14
sd-cli -m model.safetensors -p "a cat" --auto-fit --split-mode row
```

Budgets reuse `--max-vram`: a positive per-device value caps what auto-fit
plans with on that device, a negative value means "free memory minus that many
GiB", and with no budget set each device's free memory minus a 512 MiB margin
is used. (The same values still drive graph-cut segmented execution for
modules that end up on a single device.)

When everything fits resident, components are simply spread across the
available GPUs. When it does not, auto-fit switches to time-share mode: the
heavy components get `disk` params residency (loaded for their phase, freed
after), and a component too large for any single device is split across all
GPUs with the layer/row split mechanism (`--split-mode` selects which, layer
by default). Components that fit nowhere fall back to the CPU. If a VAE decode
still runs out of memory, tiling is enabled and the decode retried once.

## Modules

| Module | Purpose | Accepted names |
| --- | --- | --- |
| `diffusion` | UNet, DiT, MMDiT, Flux, Wan, Qwen Image, and other diffusion models | `diffusion`, `model`, `unet`, `dit` |
| `te` | Text encoders and conditioners | `te`, `clip`, `text`, `textencoder`, `textencoders`, `conditioner`, `cond`, `llm`, `t5`, `t5xxl` |
| `clip_vision` | CLIP vision encoder | `clip_vision`, `clipvision`, `clip-vision`, `vision` |
| `vae` | VAE and TAE | `vae`, `firststage`, `autoencoder`, `tae` |
| `controlnet` | ControlNet | `controlnet`, `control` |
| `photomaker` | PhotoMaker ID encoder and PhotoMaker LoRA | `photomaker`, `photomakerid`, `pmid`, `photo` |
| `upscaler` | ESRGAN upscaler | `upscaler`, `esrgan`, `hires` |
| `detector` | ADetailer YOLOv8 detector | `detector`, `adetailer`, `yolo` |

`te` is the preferred module name for text encoders. `clip` is kept as an accepted alias because many existing commands and model names use CLIP terminology.

## Backend names

Backend names are resolved against the GGML backend device list. Matching is case-insensitive and accepts exact names or unique prefixes, so common values include names such as:

- `cpu`
- `cuda0`
- `vulkan0`
- `metal`

The special values `auto`, `default`, and an empty backend name select the default backend. The default preference is GPU, then integrated GPU, then CPU.

The special value `gpu` selects the first GPU backend, falling back to the first integrated GPU backend.

The special value `disk` is accepted only by `--params-backend`. `--backend disk` is invalid because `disk` is a parameter residency mode, not a runtime compute backend.

## Runtime backend vs. parameter backend

The runtime backend controls where graph execution runs. The parameter backend controls where model weights are allocated or whether they are reloaded from disk on demand.

For example:

```shell
sd-cli -m model.safetensors -p "a cat" --backend cuda0 --params-backend cpu
```

This runs all modules on `cuda0`, but stores parameters in CPU RAM. During execution, parameters are moved to the runtime backend as needed.

For example:

```shell
sd-cli -m model.safetensors -p "a cat" --backend cuda0 --params-backend disk
```

This runs all modules on `cuda0`, reloads parameters from the model file as needed, and releases those parameter buffers after use.

`disk` is never selected implicitly. If `--params-backend` is not set, parameters use the runtime backend.

Per-module assignments can be mixed:

```shell
sd-cli -m model.safetensors -p "a cat" --backend diffusion=cuda0,te=cpu,vae=cpu --params-backend diffusion=cuda0,te=cpu,vae=cpu
```

This keeps text encoding and VAE execution on CPU while the diffusion model runs on GPU.

## Backend sharing and lifetime

Backends are managed by `SDBackendManager`.

Within one manager, backend instances are cached by resolved backend device name. If multiple modules request the same backend, they share the same `ggml_backend_t`.

For example:

```shell
--backend te=cpu,vae=cpu
```

uses one shared CPU backend for both `te` and `vae` runtime execution.

Runtime and parameter assignments also share the same backend cache. If `--backend diffusion=cuda0` and `--params-backend diffusion=cuda0` resolve to the same device, both use the same backend instance.

`--params-backend disk` does not create a separate backend instance. Parameters are loaded lazily using the module runtime backend.

`SDBackendManager` owns the backend instances and frees them when the context or upscaler is destroyed. Model runners receive non-owning runtime and parameter backend pointers and do not free them.

## Compatibility flags

The example CLI/server still accepts these older CPU placement flags as compatibility aliases:

- `--clip-on-cpu`
- `--vae-on-cpu`
- `--control-net-cpu`
- `--offload-to-cpu`

`--clip-on-cpu`, `--vae-on-cpu`, and `--control-net-cpu` are deprecated. The example argument layer prepends `te=cpu`, `vae=cpu`, and `controlnet=cpu` to `--backend` before creating the context.

`--offload-to-cpu` prepends a CPU default to the parameter assignment in the caller before creating the context:

```shell
--params-backend '*=cpu'
```

Because this default is inserted first, later explicit `--params-backend` entries can still override it, for example `--offload-to-cpu --params-backend te=disk` keeps non-TE parameters on CPU and reloads TE parameters from disk.

Library callers should set `backend` and `params_backend` directly. The old CPU/offload fields are no longer part of the C API. Explicit `--backend` and `--params-backend` assignments are preferred for new commands.
