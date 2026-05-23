# Backend selection

`stable-diffusion.cpp` has two backend assignments:

- `--backend` selects the runtime backend used to execute model graphs.
- `--params-backend` selects the backend used to allocate model parameters.

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

Module names are case-insensitive. Hyphens and underscores in module names are ignored, so `clip_vision`, `clip-vision`, and `clipvision` are equivalent.

`all=`, `default=`, and `*=` can be used to set the default backend inside a mixed assignment:

```shell
sd-cli -m model.safetensors -p "a cat" --backend all=cuda0,te=cpu
```

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

`te` is the preferred module name for text encoders. `clip` is kept as an accepted alias because many existing commands and model names use CLIP terminology.

## Backend names

Backend names are resolved against the GGML backend device list. Matching is case-insensitive and accepts exact names or unique prefixes, so common values include names such as:

- `cpu`
- `cuda0`
- `vulkan0`
- `metal`

The special values `auto`, `default`, and an empty backend name select the default backend. The default preference is GPU, then integrated GPU, then CPU.

The special value `gpu` selects the first GPU backend, falling back to the first integrated GPU backend.

## Runtime backend vs. parameter backend

The runtime backend controls where graph execution runs. The parameter backend controls where model weights are allocated.

For example:

```shell
sd-cli -m model.safetensors -p "a cat" --backend cuda0 --params-backend cpu
```

This runs all modules on `cuda0`, but stores parameters in CPU RAM. During execution, parameters are moved to the runtime backend as needed.

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

`SDBackendManager` owns the backend instances and frees them when the context or upscaler is destroyed. Model runners receive non-owning runtime and parameter backend pointers and do not free them.

## Compatibility flags

The older CPU placement flags are still supported:

- `--clip-on-cpu`
- `--vae-on-cpu`
- `--control-net-cpu`
- `--offload-to-cpu`

`--clip-on-cpu`, `--vae-on-cpu`, and `--control-net-cpu` affect runtime backend assignment only when `--backend` is not set. They map to `te=cpu`, `vae=cpu`, and `controlnet=cpu`.

`--offload-to-cpu` affects parameter backend assignment only when `--params-backend` is not set. It is equivalent to:

```shell
--params-backend cpu
```

Explicit `--backend` and `--params-backend` assignments are preferred for new commands.
