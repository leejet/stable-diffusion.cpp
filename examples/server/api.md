# stable-diffusion.cpp Server APIs

This document describes the server-facing APIs exposed by `examples/server`.

The server currently exposes three API families:

- `OpenAI API` under `/v1/...`
- `Stable Diffusion WebUI API` under `/sdapi/v1/...`
- `sdcpp API` under `/sdcpp/v1/...`

The `sdcpp API` is the native API surface.
Its request schema is the same schema used by `sd_cpp_extra_args`.

Image and video requests can include interleaved images in `prompt` or
`negative_prompt` using `<image:ID>`. The native `sdcpp API` accepts a
`prompt_images` object mapping each ID to a base64 image or image data URI.
For `OpenAI API` and `sdapi`, pass `prompt_images` through `sd_cpp_extra_args`
embedded inside `prompt`. IDs are resolved against that request's images.
See [images in prompts](../../docs/guides/prompt_images.md) for examples and supported
LLM encoders.

Global LoRA rule:

- Server APIs do not parse LoRA tags embedded inside `prompt`.
- `<lora:...>` prompt syntax is intentionally unsupported in `OpenAI API`, `sdapi`, and `sdcpp API`.
- LoRA must be passed through structured API fields when the API supports it.

## Overview

### OpenAI API

Compatibility API shaped like OpenAI image endpoints.

Current generation-related endpoints include:

- `POST /v1/images/generations`
- `POST /v1/images/edits`
- `GET /v1/models`

### Stable Diffusion WebUI API

Compatibility API shaped like the AUTOMATIC1111 / WebUI endpoints.

Current generation-related endpoints include:

- `POST /sdapi/v1/txt2img`
- `POST /sdapi/v1/img2img`
- `GET /sdapi/v1/progress`
- `GET /sdapi/v1/loras`
- `GET /sdapi/v1/upscalers`
- `GET /sdapi/v1/latent-upscale-modes`
- `GET /sdapi/v1/samplers`
- `GET /sdapi/v1/schedulers`
- `GET /sdapi/v1/sd-models`
- `GET /sdapi/v1/options`

### sdcpp API

Native async API for `stable-diffusion.cpp`.

Current endpoints include:

- `GET /sdcpp/v1/capabilities`
- `GET /sdcpp/v1/health`
- `GET /sdcpp/v1/resources`
- `POST /sdcpp/v1/devices/{device}/pressure`
- `POST /sdcpp/v1/guardian/resume`
- `POST /sdcpp/v1/img_gen`
- `GET /sdcpp/v1/jobs/{id}`
- `POST /sdcpp/v1/jobs/{id}/cancel`
- `POST /sdcpp/v1/vid_gen`

## Failures and recovery

A generation that cannot be served is answered the same way by all three API families
(for `sdcpp API` jobs, the same `code` is reported in the `error` of the failed job).

| Status | `code` | Meaning |
| --- | --- | --- |
| `503 Service Unavailable` | `out_of_memory` | The machine ran out of memory, even after the server tried the lighter configurations described below. Retry later, or send a smaller request. |
| `503 Service Unavailable` | `unavailable` | The server has no usable model at this moment, for example because loading it again after a failure did not succeed yet. `cause` holds the reason. |
| `503 Service Unavailable` | `busy` | The model was in use. |
| `503 Service Unavailable` | `halted` | The guardian has stopped all work: a component is reported critical, or device health excludes every compute device (see [Device health and the guardian](#device-health-and-the-guardian)). A generation that was running when the server halted ends the same way. |
| `400 Bad Request` | `invalid_argument` | The request cannot be generated as it is (for example a size the model cannot handle). Retrying it unchanged will not help. |
| `500 Internal Server Error` | `model_file`, `backend`, `generation_failed` | A model file could not be read, a compute device failed, or the generation failed without a more specific reason. |

The body is a JSON object, and a `503` with one of the codes above also carries a `Retry-After` header (the `503`
answers of the [model management](#model-management) endpoints, `model_loading` and `model_not_loaded`, do not):

```json
{
  "error": "unet compute buffer allocation failed (out of memory)",
  "code": "out_of_memory",
  "retry_after_seconds": 30
}
```

| Field | Type | Notes |
| --- | --- | --- |
| `error` | `string` | Human-readable reason |
| `code` | `string` | One of the codes above |
| `cause` | `string` | Only with `unavailable`: the code of the failure that left the server without a model |
| `retry_after_seconds` | `integer` | Only with `503`: how long to wait before trying again; mirrors the `Retry-After` header |

Other `4xx` answers (`400` for a malformed request, `429` when the job queue is full, `401` for a missing API key, ...)
keep their meaning. A `429` carries `Retry-After: 10`.

### Stepping down

A request that fails because the machine ran out of memory (or because a compute device failed) does not end the
server and is not simply refused. The server frees the model, loads it again in a different configuration, and runs
the same request again. An out-of-memory failure moves down the ladder of profiles
`max > balanced > lite > lite-disk > lite-hybrid > lite-cpu`; a device failure moves to another device
(see [Device failures](#device-failures)).

| Profile | What changes |
| --- | --- |
| `balanced` | `--memory-guard 90` |
| `lite` | memory guard 85, VAE decoded in tiles, no batched CFG, `--multi-gpu on` (every GPU of the backend may take a module; an explicit `--multi-gpu off` is kept) |
| `lite-disk` | weights are streamed from disk (`--params-backend disk`) instead of being kept in memory; `--eager-load` is switched off |
| `lite-hybrid` | the text encoder is computed on the CPU; everything else keeps its device |
| `lite-cpu` | everything is computed on the CPU |

Up to `lite-disk` all the computing stays on the GPUs: those profiles save memory by working in smaller pieces and by
reading weights again from RAM or disk, which costs time but does not hand the work to the CPU. The CPU takes a share
only from `lite-hybrid` on, and all of it only at `lite-cpu`, the last resort. These two profiles exist only when the
machine has a device other than the CPU; without one everything already runs on the CPU and the ladder ends at
`lite-disk`.

Steps that would load the same configuration as the previous one are skipped, so the ladder is shorter when
the server was started with some of these options already. The profile the server was started with
(`--profile`, default `max`) is the top of the ladder. From `lite` on, the VAE is decoded in tiles whatever the
request asked.

A request served after a step-down is a normal `200`. When even the lightest profile does not fit, the request
fails with the `503` above, and the server stays up, on the lightest profile that could load the model.

After `--degradation-cooldown` seconds (default `300`) the server climbs back one step, at the beginning of the next
request, which pays for loading the model again: one profile up, or back to the device it started on (see below). A climb
that does not work is undone and the wait doubles (up to an hour); a climb that works resets it. With
`--degradation-cooldown 0` the server stays where it stepped down, and with `--disable-degradation` it never steps
down: such a request is answered with the error at once. `GET /sdcpp/v1/health` reports the profile and the device in use.

The same ladder, and the device rules below, protect the models loaded with `POST /sdcpp/v1/models/load` (see
[Model management](#model-management)): each load starts again from the profile the server was started with, and a
model that does not fit there is loaded on the first lighter profile, or the first device, that works, before the load
is reported as failed.

If the model cannot be loaded again after a failure, requests are answered with `503` / `unavailable` and
the server tries again from the next request on, no more often than every 15 seconds.

### Device failures

A compute device can fail without the memory being short: a driver error, a GPU that is reset or removed, a backend that
does not start. Trimming memory does not cure that, and handing the work to the CPU throws away the GPUs that still work,
so the server first looks for another place to run the same configuration. The order is:

1. the same device once more, because many such failures are passing ones (once per request);
2. each device the engine lists, one at a time on its own: graphics cards first, the vendor's backend (CUDA, HIP, Metal)
   before the same card seen through Vulkan or OpenCL, the one with the most free memory first; accelerators next;
   integrated GPUs last, because they share the system RAM and compute several times slower;
3. the CPU (`lite-cpu`), only when no device is left.

A device failure on the CPU ends the request with the error. An out-of-memory failure on any of these devices first goes
down the ladder of profiles on that same device (see above). Network devices (`RPC...`) are never used as a replacement.
A placement written in `--backend` (modules assigned with `=`, several devices with `&` or `,`) is not rewritten: the
server tries the same configuration once more and then goes to the CPU.

While a device is left to try, the server also tells the engine not to fall back to the CPU by itself
(`disable_backend_fallback`). Without that, a backend that does not start would silently turn the whole model into a CPU
run, and the profile in use would hide it. With `--disable-degradation` the engine keeps its own behavior.

The log says each move (`trying profile lite on Vulkan0 instead of profile lite`) and, at start-up, the order the
server would follow (`a device that fails is replaced by the next one before the cpu takes over: CUDA0 > Vulkan0`). The
same rules apply when the model is loaded at start-up or again after a failure: if the first device does not start, the
server starts on the next one, and the log says which. `GET /sdcpp/v1/health` tells which `device` the model runs on and
which modules the CPU computes (`cpu_modules`).

Going back is the other half. From a GPU the climb makes one attempt: back to the device the server started on when it
is running on another one, otherwise one profile up; if the attempt fails, the server returns to where it was and the
wait doubles. From the CPU, after a device failure, the climb goes straight back to the profile the server had when the
device failed and offers every device again, so the best one that works is found even if the preferred one does not;
when the CPU was reached through memory failures, the climb is one profile up, as usual.

### Device health and the guardian

The engine keeps, for the whole process, a record of how each component is doing: every device the engine lists
(`CUDA0`, `Vulkan1`, `CPU`, ...), the `RAM` and the `disk`. Each one is healthy (level 0), under strain (levels 1 to 3)
or excluded (level 4). These are the default values; `--device-health` changes them (see below).

| Level | What the component offers |
| --- | --- |
| 0 | all of its capacity |
| 1, 2, 3 | 85%, 70%, 55%: the planner gives it that share of its memory, the layer split that share of its capacity, and the CPU computes with 25% fewer threads per level (at least one) |
| 4, excluded | nothing: the engine never computes on it, the planner gives it no module, the server does not load the model on it nor climb back to it, and an excluded CPU takes no work handed over from the GPUs. The RAM is never cut below the share of the last level of strain; an excluded disk is not read during a generation, and a model that would need it fails with `out_of_memory` |

What raises a level, one step at a time: a compute that fails on a device (not for lack of memory: the profiles deal
with that), a backend that does not start, the memory guard that cannot bring the memory below its threshold (it
counts for the device, or for the `RAM` when it guards the CPU), a read of the weights that fails (the `disk`). Events of
one device less than 10 seconds apart count once. Outside watchers report too, with
[`POST /sdcpp/v1/devices/{device}/pressure`](#post-sdcppv1devicesdevicepressure): such a signal holds its level for the time it
lasts, and reaches the other listings of the same card (`CUDA0` and `Vulkan0` of one GPU), which a failure of one backend
does not. A level comes down one step after each quiet period of 300 seconds, counted from the last event and from the
end of the signals that held it; withdrawing a signal does not make the component healthy at once. When the record
changes, the next generation places the model again with the new shares, which costs loading the weights again.

`--device-health` changes the values when the server starts, with any of the keys of
`levels=3,step=15,cpu-step=25,quiet=300,spacing=10` (the defaults): the levels of strain, the share and the CPU
threads given up at each level, the quiet period and the spacing of the events, in seconds. The level after the last
level of strain excludes; each component in `/sdcpp/v1/health` gives its `strain_levels`, so a watcher knows which
level that is. See [Backend selection](../../docs/guides/backend.md#device-health---device-health) for the rules.

The guardian stops all the work of the server when the machine cannot take it: a component is reported `critical`, or
every compute device is excluded (the GPUs and the CPU; the `RAM` and the `disk` do not count). While it is halted:

- the generation in progress stops: a job ends `failed` with the code `halted`, a synchronous request is answered `503`
  with the code `halted`; a generation that finishes before the stop reaches it keeps its result;
- new generations, DGL scripts and model loads are refused with `503`, the code `halted` and a `Retry-After` header: the
  time left to the critical signal, or until the first excluded device comes back, and never less than 10 seconds;
- the queued jobs wait, and run after the resume;
- the server does not load the model again, step down or climb back: a failure that comes with the halt is a stop;
- [`GET /sdcpp/v1/health`](#get-sdcppv1health) answers `503` with `status: "halted"`, the reason and every component; polling
  jobs, cancelling them, unloading the model and the reports still work.

The server resumes by itself when the cause ends (the critical signal runs out or is withdrawn, an excluded device comes
back after its quiet periods), or when an operator says the machine was checked with
[`POST /sdcpp/v1/guardian/resume`](#post-sdcppv1guardianresume). `--guardian off` disables the guardian: device health still
moves the work between devices, but the server never stops all work by itself.

This is a layer of the software, inside the process it watches: a driver that hangs or a crash stop it with everything
else, and lowering the work does not cool a device by itself. A stop that protects people or equipment belongs to
independent, certified hardware, never to this server. A signal that is not repeated runs out, and the server resumes
then: a watcher that stops reporting does not keep the server halted.

## Model management

All three API families share the server's execution context and its component
cache. The command-line model supplies the initial configuration. Native image
and video jobs can select a recipe and component files in the request. Video
requests with an explicit recipe require that recipe to declare
`defaults.video_frames`. The worker probes the selected model family and checks
that it supports the requested mode before applying the selection. A probe or
mode check failure is reported on the job and preserves the current model.
An applied selection remains the
current model for later requests that omit `model`, including A1111 and OpenAI
requests. Each native job reports loading and generation progress through the job API.

Recipe defaults apply when a generation request omits the corresponding
parameters. Image editing endpoints retain their existing input-image size
inference when dimensions are omitted.

Requests that use the current model return HTTP 503 with
`{"error":"No current model is configured","code":"model_not_loaded"}` until
the server has a model configuration. `/v1/models` lists `sd-cpp-local` when a
model is configured. An accepted native job that omits `model` uses the current
model and its defaults when the job starts.

| Method | Endpoint | Response / behavior |
| --- | --- | --- |
| GET | `/sdcpp/v1/recipes` | `{recipes: [...], errors: [...]}` |
| GET | `/sdcpp/v1/model-files` | `{files: [{id, name, category, root, relative_path}], errors: [...]}` |
| POST | `/sdcpp/v1/models/refresh` | Rescan recipes and files; return `{errors: [...]}` |
| GET | `/sdcpp/v1/models/status` | Current model configuration (`configured` or `empty`), recipe, components and revision |

Example `POST /sdcpp/v1/img_gen` request with a model selection.

```json
{
  "prompt": "a photograph of a cat",
  "model": {
    "recipe": "flux1-dev",
    "models": {
      "diffusion_model": "diffusion_models/flux1-dev.gguf",
      "clip_l": "text_encoders/clip_l.safetensors",
      "t5xxl": "text_encoders/t5xxl.safetensors",
      "vae": "vae/ae.safetensors"
    }
  }
}
```

File IDs are opaque strings returned by `model-files`. The first model root
retains the existing relative IDs, such as `checkpoints/nested/model.gguf`.
Additional roots use `@<root-id>/<relative-path>` so identical paths in different
roots remain selectable. Root qualifiers derive from the canonical root path;
adding unrelated roots or refreshing the catalog does not change them. Moving
a root changes its qualifier. Use the returned IDs to select files in generation
requests.

The display `name` omits the physical category directory (including aliases such
as `Stable-diffusion`). Duplicate display paths use the full absolute file path,
with the source directories before the filename and no hash suffix.
`root` is an opaque source identifier, and
`relative_path` is the file path beneath that root. Duplicate canonical files
appear only once, using the first root that discovers them. Categories depend
on directories, never on filename suffixes such as `.vae.safetensors`.

Repeated `--model-dir` arguments merge roots in order; absent arguments default
to `./models`. User recipes override built-ins; the earliest root containing a
valid user recipe wins across roots. Duplicate recipe IDs within one root are
rejected. Directory aliases do not change recipe slot categories.

LoRA lists retain relative `path` IDs for `loras/` in the first model root and
for an explicit `--lora-model-dir`, and qualify other directories. The additive `display_path` field provides a readable
relative path, or the full absolute file path when needed to distinguish duplicates.
Continue sending `path` in structured LoRA requests. `--lora-model-dir` overrides
automatic discovery across all model roots.
Absolute paths, unknown files, wrong categories and
paths escaping the configured root are rejected. An invalid selection returns
HTTP 400 with `code: "invalid_model_config"`. A component that fails to load
after submission reports its error in the job result. Catalog refreshes apply
to later submissions.

This fork also keeps `POST /sdcpp/v1/models/load`, with the same `recipe` and
`models` fields, and `POST /sdcpp/v1/models/unload`, for programs that want to
pick the current model before they send requests, or to get the memory back.
Both return HTTP 409 (`model_busy`) while a request or a queued job is pending.
An invalid selection returns HTTP 400 (`invalid_model_config`) and leaves the
current model in place. When the context for the new model cannot be created,
the response is HTTP 422 (`model_load_failed`) and the server is left without a
model. The weights load with the first generation, as for a model picked by a
request.

The initial command-line selection has `origin: command_line`. Once a catalog
recipe is selected, status reports its ID and component files. The read-only
`runtime_recipe` with ID `@current` represents whichever model is configured at
the time of the query. Omit `model` from the request to use the current model.
Capabilities include the same model information in `runtime` and advertise
`generation_model_selection: true`.

### Recipe feature declarations

Recipes may declare `features` and `disabled_features` arrays. The server starts
with shared defaults, enables features declared by component slots and `features`,
then applies `disabled_features` last. A disabled feature always wins. Unknown
names, non-array values and duplicate names within an array are rejected.
Both fields are optional; existing recipes remain valid.

```json
{
  "features": ["flow_shift", "distilled_guidance"],
  "disabled_features": ["clip_skip", "lora"]
}
```

Default-enabled features are `img2img`, `mask_image`, `lora`, `vae_tiling`,
`cache`, `cache_dit` and `preview`. All other features below default to false:

| Feature | Controls |
| --- | --- |
| `flow_shift` | Flow shift, where used by the selected scheduler |
| `distilled_guidance` | Distilled guidance embedding |
| `shifted_timestep` | SDXL timestep shifting |
| `img_cfg` | Image CFG |
| `slg` | Skip-layer guidance fields |
| `clip_skip` | CLIP Skip |
| `img2img`, `mask_image` | Image initialization/strength and masks |
| `ref_images` | Reference image inputs |
| `ref_videos` | Reference videos as frame arrays |
| `ref_audios` | Standalone reference audio; runtime requires an Audio VAE or Audio Encoder |
| `ref_video_audios` | Soundtracks paired with reference videos; runtime requires an Audio VAE |
| `controlnet` | Control image and ControlNet strength |
| `control_image` | Control image without a separate ControlNet (FLUX Canny/Depth) |
| `ip_adapter` | IP-Adapter image and strength |
| `init_image`, `end_image` | Video start/end frames |
| `control_frames`, `vace` | Video control frames and VACE strength |
| `high_noise` | High-noise pass, MoE boundary and high-noise LoRAs |
| `lora`, `vae_tiling` | LoRA overrides and VAE tiling |
| `cache`, `cache_unet`, `cache_dit` | Cache panel and architecture-specific algorithms |
| `preview`, `preview_tae` | Preview panel and TAE preview option |

A `control_net` slot enables `controlnet`, `ip_adapter` enables `ip_adapter`,
`high_noise_diffusion_model` enables `high_noise`, and `tae`/`taesd` enables `preview_tae`, unless explicitly disabled.
ControlNet, IP-Adapter, high-noise, TAE preview and audio reference controls
also depend on the selected component files.
PuLID disables SLG while selected.
Image-only and video-only inputs are filtered by the active generation mode.
Eta depends on the selected sampler; SCM fields only apply to DBCache,
TaylorSeer and Cache-DIT. Empty panels and advanced sections are hidden.

`GET /sdcpp/v1/recipes` adds a resolved boolean `effective_features` object to
each recipe. This is response-only metadata, like `source` and slot `categories`;
omit it when writing a user recipe file. The defaults are maintained by the
server, not duplicated by the frontend.

`GET /sdcpp/v1/capabilities` reports `generation_features` for the current model,
along with its generation mode and defaults. With no configured model, those
features are false.
The UI uses the selected catalog recipe's `effective_features` and component
selection for requests that specify `model`.
Known high-noise sampling settings and `moe_boundary` are ignored when no
high-noise stage is loaded, and `vace_strength` is ignored by non-VACE models.
Actual control-frame inputs still require VACE.

These declarations govern the Web UI and advertised capabilities, not API
request validation. Existing generation APIs continue accepting their existing
fields. The Web UI omits unsupported/hidden fields and ignores hidden image
inputs when building requests, so switching recipes cannot submit stale options.

### User recipe format

Contributions to the built-in catalog must follow the
[built-in recipe policy](recipes/README.md). Personal user recipes may define
workflow-specific presets without adding them to the built-in catalog.

Each UTF-8 JSON file in `<model-dir>/recipes/` defines one complete recipe.
The maximum recipe file size is 1 MiB. Unknown fields, invalid types and
duplicate slots are errors. Built-ins in `examples/server/recipes/*.json` use
the same one-recipe-per-file schema:

```json
{
  "schema_version": 1,
  "id": "my-flux",
  "name": "My FLUX",
  "description": "FLUX.1 dev",
  "size_multiple": 16,
  "model_url": "https://huggingface.co/black-forest-labs/FLUX.1-dev",
  "slots": [
    {"id": "diffusion_model", "required": true},
    {"id": "clip_l", "required": true},
    {"id": "t5xxl", "required": true},
    {"id": "vae", "required": false},
    {"id": "tae", "required": false}
  ],
  "required_any": [["vae", "tae"]],
  "defaults": {"width": 1024, "height": 1024, "cfg_scale": 1.0, "distilled_guidance": 3.5, "sampling_method": "euler", "steps": 20}
}
```

IDs contain 1–128 ASCII letters, digits, dots, underscores or hyphens.
`name`, `schema_version: 1` and `slots` are required; `description` and
`model_url`, `required_any`, `defaults`, `context`, `features`, `disabled_features`
and `size_multiple`, `video_constraints` are optional. Each slot requires only `id` and boolean `required`.
The server supplies the default label and search directories for each slot.
An optional nonempty `label` overrides the display name, for example
`{"id": "diffusion_model", "label": "Low-noise diffusion model", "required": true}`.
Legacy `category` fields remain accepted, but must match one of the slot's
supported directories; they cannot redefine or restrict its search directories.
At least one required `model` or `diffusion_model` slot is necessary.

An optional top-level `models` object maps slot IDs to file IDs, providing
recipe-local defaults for component selection. Omitting `model.models` uses
the recipe's defaults. Explicit selections override recipe defaults slot by
slot. An empty string disables the default for an optional slot.

`size_multiple` is a positive integer from 1 to 4096 that declares the width and
height increment used by the web UI. It defaults to `8` when omitted; the UI also
uses `8` before a recipe is selected. For example, `"size_multiple": 16` makes
generated size presets multiples of 16. Exact dimensions that do not match the
increment require correction before submission. This is recipe metadata, not a
generation parameter, and does not change the core model's alignment behavior.

`video_constraints` is optional metadata for the web UI's Frames / Duration
controls. It does not override the runtime's model-specific rules. For example:

```json
"video_constraints": {
  "frame_step": 4,
  "frame_offset": 1,
  "frame_rounding": "floor",
  "min_frames": 1
}
```

Legal frame counts are `frame_step * n + frame_offset`, with integer `n >= 0`,
within the inclusive `min_frames` / `max_frames` range. All fields are optional:

| Field | Default | Accepted values |
| --- | --- | --- |
| `frame_step` | `1` | Integer 1–4096 |
| `frame_offset` | `0` | Integer 0 to `frame_step - 1` |
| `frame_rounding` | `"floor"` | `"floor"` or `"ceil"` |
| `min_frames` | `1` | Integer 1–4096 |
| `max_frames` | `4096` | Integer 1–4096, at least `min_frames` |
| `fixed_fps` | Unset | Integer 1–240; locks the FPS control |

The range must contain at least one legal frame count. Unknown keys are rejected.
Omitting the object adds no frame alignment in the UI. `defaults.fps` only sets
the initial FPS; it does not lock it. The UI multiplies requested duration by FPS,
aligns using the declared rounding direction, then clamps to the legal range.
Duration stays in the UI; generation requests still send only `video_frames` and
`fps`. Switching input modes preserves the aligned length. Changing FPS keeps
frames in Frames mode, and keeps the requested seconds in Duration mode.

`model_url` is an optional HTTPS Hugging Face model repository link, shown beside
the recipe description. Built-ins link to the model's author/publisher, not repacks.
`required_any` contains groups of declared slot IDs. At least one model in each
group must be selected. For example, `"required_any": [["vae", "tae"]]` with both
slots marked `required: false` allows either the full VAE or a compatible TAE.
Selecting both is allowed; the current server uses TAE for encoding/decoding when
TAE is selected. Checkpoint recipes with a bundled VAE do not need this group.
Use complete, model-compatible TAE weights; decoder-only preview weights cannot
replace a full VAE. Pixel-space models and unsupported latent formats have no
TAE slot.

| Slot IDs | Search directories |
| --- | --- |
| `model` | `checkpoints` |
| `diffusion_model`, `high_noise_diffusion_model`, `uncond_diffusion_model` | `diffusion_models` |
| `clip_l`, `clip_g`, `t5xxl`, `llm`, `audio_encoder`, `embeddings_connectors` | `text_encoders` |
| `clip_vision` | `clip_vision` |
| `llm_vision` | `text_encoders` or `clip_vision` |
| `vae`, `audio_vae` | `vae` |
| `tae` (legacy alias: `taesd`) | `vae` or `vae_approx` |
| `tokenizer` | `tokenizers` |
| `control_net` | `controlnet` |
| `ip_adapter` | `ipadapter` |
| `motion_module` | `motion_modules` |
| `photo_maker` | `photomaker` |
| `pulid_weights` | `pulid` |

Built-in recipes use `tae` for the Tiny AutoEncoder slot. TAE selections can use files from either `vae/` or the legacy
`vae_approx/` directory. Existing recipes and load requests using `taesd`
remain supported. Do not specify both `tae` and `taesd` in the same recipe
or load request.

Model files can use `.gguf`, `.safetensors`, `.sft`, `.ckpt`, `.pt`, `.pth`
or `.bin` extensions. The tokenizer category lists `.json` files only.
Custom recipes compose components supported by the backend; they do not add
new model architectures.

Optional `context` settings apply to each model load and override the matching
startup setting. Supported fields are `vae_format` (`auto`, `flux`, `sd3`,
`flux2`, or `wan`), `model_args` (a string in the same format as
`--model-args`; replaces the startup string), and `backend` / `params_backend`
(strings in the same format as `--backend` / `--params-backend`; replace the
startup assignments for that load, so a recipe can place components on
specific devices, e.g. keep a large text encoder on CPU while another recipe
uses the GPU). Other startup settings remain unchanged. Switching to a
recipe that omits these fields restores the startup values. For example, PiD
uses `{"vae_format": "flux"}` and Qwen Image Edit 2511 uses
`{"model_args": "qwen_image_zero_cond_t=true"}`.

Supported defaults:

- Integer `width`/`height` (64–4096), `steps` (1–1000), `video_frames`
  (1–4096), `fps` (1–240), `shifted_timestep` (0–1000), and
  `high_noise_steps` (1–1000 or -1 for automatic).
- Numeric `cfg_scale`, `img_cfg_scale`, `distilled_guidance`, `flow_shift`,
  `high_noise_cfg_scale`, and `high_noise_flow_shift` (0–1000).
- Numeric `strength` (0–1), used for image-to-image denoising, and
  `moe_boundary` (0–1), used for automatic Wan MoE expert switching.
- `custom_sigmas`: an empty array or 2–1001 finite, nonnegative numbers.
  A nonempty array overrides the generated schedule; the sigma count minus one
  determines the actual step count. Include the final zero. An empty array uses
  the generated schedule. Loaded recipe sigmas apply to generation requests
  unless `sample_params.custom_sigmas` overrides them; switching recipes restores
  that recipe's schedule or the startup default.
- Supported `sampling_method`/`scheduler` and
  `high_noise_sampling_method`/`high_noise_scheduler` names.

Other generation parameters continue to use startup defaults unless explicitly
supplied by the generation request. Built-in Wan recipes provide video defaults;
use `/sdcpp/v1/vid_gen` with a video model.
LoRAs are merged from `loras/` and `Lora/` in every model root unless
`--lora-model-dir` is supplied.

Recipe responses fill in each slot's `label`, `category` (the legacy value,
if supplied, or the first search directory), and `categories` (the complete
server-defined list of search directories). Clients should filter model files
using `categories`; `category` remains available for older clients.
For example, `{"id": "tae", "required": false}` expands to
`{"id": "tae", "required": false, "label": "TAE", "category": "vae", "categories": ["vae", "vae_approx"]}`.

Recipe responses also add `source: builtin | user` and, for user recipes, `file`.
Do not include `source`, `file`, or slot `categories` in a recipe file. A valid user recipe
replaces a built-in with the same ID in full; inheritance/partial merging is
not supported. Invalid recipes are skipped. Multiple valid user files sharing
an ID are all skipped, preserving any built-in with that ID. Catalog errors
are returned as `{file, message}` entries.

## `sd_cpp_extra_args`

`sd_cpp_extra_args` is an extension mechanism for the compatibility APIs.

Rules:

- Its JSON schema is the same schema used by the native `sdcpp API`.
- `OpenAI API` and `sdapi` can embed it inside `prompt`.
- `sdcpp API` does not need it, because the request body already uses the native schema directly.

Embedding format:

```text
normal prompt text <sd_cpp_extra_args>{"sample_params":{"sample_steps":28}}</sd_cpp_extra_args>
```

Behavior:

- The server extracts the JSON block.
- The JSON block is parsed using the same field rules as the `sdcpp API`.
- The block is removed from the final prompt before generation.

Supported use:

- extend `OpenAI API` requests with native `stable-diffusion.cpp` controls
- extend `sdapi` requests with native `stable-diffusion.cpp` controls

Unsupported use:

- do not use `sd_cpp_extra_args` with `/sdcpp/v1/*`

## OpenAI API

### Purpose

This family exists for client compatibility.

Use it when you want OpenAI-style request and response shapes.

### Native Extension

`OpenAI API` supports `sd_cpp_extra_args` embedded inside `prompt`.

The embedded JSON follows the `sdcpp API` request schema.

### Supported Fields

#### `POST /v1/images/generations`

Currently supported top-level request fields:

| Field | Type | Notes |
| --- | --- | --- |
| `prompt` | `string` | Required |
| `n` | `integer` | Number of images |
| `size` | `string` | Format `WIDTHxHEIGHT` |
| `output_format` | `string` | `png`, `jpeg`, or `webp` |
| `output_compression` | `integer` | Range is clamped to `0..100` |

Native extension fields:

- any `sdcpp API` fields embedded through `sd_cpp_extra_args` inside `prompt`

Response fields:

| Field | Type | Notes |
| --- | --- | --- |
| `created` | `integer` | Unix timestamp |
| `output_format` | `string` | Final encoded image format |
| `data` | `array<object>` | Generated image list |
| `data[].b64_json` | `string` | Base64-encoded image bytes |

#### `POST /v1/images/edits`

Currently supported multipart form fields:

| Field | Type | Notes |
| --- | --- | --- |
| `prompt` | `string` | Required |
| `image[]` | `file[]` | Preferred image upload field |
| `image` | `file` | Legacy single-image upload field |
| `mask` | `file` | Optional mask image |
| `n` | `integer` | Number of images |
| `size` | `string` | Format `WIDTHxHEIGHT` |
| `output_format` | `string` | `png` or `jpeg` |
| `output_compression` | `integer` | Range is clamped to `0..100` |

Native extension fields:

- any `sdcpp API` fields embedded through `sd_cpp_extra_args` inside `prompt`

Uploaded images are decoded at their original dimensions. The first decoded
image establishes the generation dimensions if `size` is omitted. Input
geometry follows the selected model's default input rules.

Reference encoding follows the model defaults. Custom resizing, cropping and
reference conditioning require a DGL program.

Response fields:

| Field | Type | Notes |
| --- | --- | --- |
| `created` | `integer` | Unix timestamp |
| `output_format` | `string` | Final encoded image format |
| `data` | `array<object>` | Generated image list |
| `data[].b64_json` | `string` | Base64-encoded image bytes |

#### `GET /v1/models`

Response fields:

| Field | Type | Notes |
| --- | --- | --- |
| `data` | `array<object>` | Available local models |
| `data[].id` | `string` | Currently fixed to `sd-cpp-local` |
| `data[].object` | `string` | Currently fixed to `model` |
| `data[].owned_by` | `string` | Currently fixed to `local` |

### Output Options

`OpenAI API` supports response serialization controls such as:

- `output_format`
- `output_compression`

### Notes

- `OpenAI API` is synchronous from the HTTP client's perspective.
- Native async job polling is not exposed through this family.
- Prompt-embedded `<lora:...>` tags are intentionally unsupported.

## Stable Diffusion WebUI API

### Purpose

This family exists for client compatibility with WebUI-style tools.

Use it when you want `txt2img` / `img2img`-style endpoints and response shapes.

### Native Extension

`sdapi` supports `sd_cpp_extra_args` embedded inside `prompt`.

The embedded JSON follows the `sdcpp API` request schema.

This allows `sdapi` clients to use native `stable-diffusion.cpp` controls without changing the outer request format.

### Supported Fields

#### `POST /sdapi/v1/txt2img`

Currently supported request fields:

| Field | Type | Notes |
| --- | --- | --- |
| `prompt` | `string` | Required |
| `negative_prompt` | `string` | Optional |
| `width` | `integer` | Positive image width |
| `height` | `integer` | Positive image height |
| `steps` | `integer` | Sampling steps |
| `cfg_scale` | `number` | Text CFG scale |
| `seed` | `integer` | `-1` means random |
| `batch_size` | `integer` | Number of images |
| `clip_skip` | `integer` | Optional |
| `sampler_name` | `string` | WebUI sampler name |
| `scheduler` | `string` | Scheduler name |
| `lora` | `array<object>` | Structured LoRA list |
| `extra_images` | `array<string>` | Base64 or data URL images |
| `enable_hr` | `boolean` | Enable highres fix for `txt2img` |
| `hr_upscaler` | `string` | `Lanczos`, `Nearest`, a latent mode such as `Latent (nearest-exact)`, or an upscaler model name from `/sdapi/v1/upscalers` |
| `hr_scale` | `number` | Highres scale when resize target is not set |
| `hr_resize_x` | `integer` | Highres target width, `0` to use scale |
| `hr_resize_y` | `integer` | Highres target height, `0` to use scale |
| `hr_steps` | `integer` | Highres second-pass sample steps, `0` to reuse `steps` |
| `denoising_strength` | `number` | Highres denoising strength for `txt2img` |

Native extension fields:

- any `sdcpp API` fields embedded through `sd_cpp_extra_args` inside `prompt`

Response fields:

| Field | Type | Notes |
| --- | --- | --- |
| `images` | `array<string>` | Base64-encoded PNG images |
| `parameters` | `object` | Echo of the parsed outer request body |
| `info` | `string` | Currently empty string |

#### `POST /sdapi/v1/img2img`

Currently supported request fields:

| Field | Type | Notes |
| --- | --- | --- |
| all currently supported `txt2img` fields | same as above | Reused |
| `init_images` | `array<string>` | Base64 or data URL images |
| `mask` | `string` | Base64 or data URL image |
| `inpainting_mask_invert` | `integer` or `boolean` | Treated as invert flag |
| `denoising_strength` | `number` | Clamped to `0.0..1.0` |

Highres fix fields are currently handled for `txt2img`; `img2img` uses `denoising_strength` as image-to-image strength.

Native extension fields:

- any `sdcpp API` fields embedded through `sd_cpp_extra_args` inside `prompt`

Response fields:

| Field | Type | Notes |
| --- | --- | --- |
| `images` | `array<string>` | Base64-encoded PNG images |
| `parameters` | `object` | Echo of the parsed outer request body |
| `info` | `string` | Currently empty string |

#### `GET /sdapi/v1/progress`

Poll this endpoint while a synchronous SDAPI generation request is running.
An optional `id_task` query parameter selects the task with the same `id_task`
provided in the generation request (default: `"sdapi"`). Set
`skip_current_image=true` or `1` to omit the preview image.

`current_image` contains the latest base64 JPEG latent projection, or `null`
when unavailable. `state.sampling_step` is the positive logical step and
`state.sampling_steps` is the actual step count for the current sampling pass,
including schedule and img2img strength adjustments. `state.job_no` is the
zero-based pass index; `state.job_count` includes one pass per batch image and
an additional pass per image when highres fix is enabled. `progress` weights
these passes equally. Sampling progress can reach `1` before final decoding
finishes; `eta_relative` is currently always `0`.

When no SDAPI task is active, or `id_task` does not match, the endpoint returns
`progress=0`, `state.job=""`, zero job/step counts, and `current_image=null`.
Completed and failed requests do not retain an active preview.

#### Discovery / Compatibility Endpoints

Currently exposed:

- `GET /sdapi/v1/loras`
- `GET /sdapi/v1/upscalers`
- `GET /sdapi/v1/latent-upscale-modes`
- `GET /sdapi/v1/samplers`
- `GET /sdapi/v1/schedulers`
- `GET /sdapi/v1/sd-models`
- `GET /sdapi/v1/options`

Response fields:

`GET /sdapi/v1/loras`

| Field | Type | Notes |
| --- | --- | --- |
| `[].name` | `string` | Display name derived from file stem |
| `[].path` | `string` | Relative path under the configured LoRA directory |

`GET /sdapi/v1/upscalers`

| Field | Type | Notes |
| --- | --- | --- |
| `[].name` | `string` | Built-in name or model stem |
| `[].model_name` | `string \| null` | Model family label for model-backed upscalers |
| `[].model_path` | `string \| null` | Absolute model path for model-backed upscalers |
| `[].model_url` | `string \| null` | Currently always null |
| `[].scale` | `integer` | Currently `4` |

Built-in entries include `None`, `Lanczos`, and `Nearest`. Model-backed entries are scanned from the top level of `--hires-upscalers-dir`; subdirectories are not scanned.

`GET /sdapi/v1/latent-upscale-modes`

| Field | Type | Notes |
| --- | --- | --- |
| `[].name` | `string` | WebUI-compatible latent upscale mode name |

Built-in latent modes include `Latent`, `Latent (nearest)`, `Latent (nearest-exact)`, `Latent (antialiased)`, `Latent (bicubic)`, and `Latent (bicubic antialiased)`.

`GET /sdapi/v1/samplers`

| Field | Type | Notes |
| --- | --- | --- |
| `[].name` | `string` | Sampler name |
| `[].aliases` | `array<string>` | Currently contains the same single sampler name |
| `[].options` | `object` | Currently empty object |

`GET /sdapi/v1/schedulers`

| Field | Type | Notes |
| --- | --- | --- |
| `[].name` | `string` | Scheduler name |
| `[].label` | `string` | Same value as `name` |

`GET /sdapi/v1/sd-models`

| Field | Type | Notes |
| --- | --- | --- |
| `[].title` | `string` | Model stem |
| `[].model_name` | `string` | Same value as `title` |
| `[].filename` | `string` | Model filename |
| `[].hash` | `string` | Placeholder compatibility value |
| `[].sha256` | `string` | Placeholder compatibility value |
| `[].config` | `null` | Currently always null |

`GET /sdapi/v1/options`

| Field | Type | Notes |
| --- | --- | --- |
| `samples_format` | `string` | Currently fixed to `png` |
| `sd_model_checkpoint` | `string` | Model stem |

### Notes

- `sdapi` is synchronous from the HTTP client's perspective.
- Prompt-embedded `<lora:...>` tags are intentionally unsupported.

## sdcpp API

### Purpose

This is the native `stable-diffusion.cpp` API.

Use it when you want:

- async job submission
- explicit native parameter control
- frontend-oriented capability discovery

### Job Model

All async generation requests create a job.

Job states:

- `queued`
- `generating`
- `completed`
- `failed`
- `cancelled`

Common job shape:

```json
{
  "id": "job_01HTXYZABC",
  "kind": "img_gen",
  "status": "queued",
  "created": 1775401200000,
  "started": null,
  "completed": null,
  "queue_position": 2,
  "result": null,
  "error": null
}
```

Field types:

| Field | Type |
| --- | --- |
| `id` | `string` |
| `kind` | `string` |
| `status` | `string` |
| `created` | `integer` |
| `started` | `integer \| null` |
| `completed` | `integer \| null` |
| `queue_position` | `integer` |
| `result` | `object \| null` |
| `error` | `object \| null` |
| `progress` | `object` |
| `preview` | `object \| null` |
| `cancel_requested` | `boolean`: `true` once the job has been asked to stop (a job that is generating stays `generating` until it has stopped), `false` otherwise |

`created`, `started`, and `completed` are integer Unix timestamps in
**milliseconds**. Subtract `started` from `completed` and divide by 1000 to get
the elapsed generation time in seconds, including output encoding and excluding
queue time.

`progress` is available independently of the preview setting, for both image
and video jobs. Continue polling until `status` is terminal: a sampling pass
can finish before decoding, ADetailer, or output encoding finishes.

| Field | Type | Notes |
| --- | --- | --- |
| `progress.stage` | `string` | Current execution stage; see below |
| `progress.detail` | `string` | Stage detail, or an empty string |
| `progress.pass` | `integer` | Sampling pass number within the job; zero before sampling |
| `progress.completed` | `integer \| null` | Completed units in the current stage/pass |
| `progress.total` | `integer \| null` | Actual total units, or null when unknown |
| `progress.unit` | `string \| null` | `steps` for sampling, `images` for image output encoding; null when uncounted |
| `progress.stage_elapsed_ms` | `integer` | Elapsed time in the current stage/detail/pass |
| `progress.elapsed_ms` | `integer` | Job execution time, excluding queue time; zero before execution |

Stages include `queued`, `waiting` (waiting for the execution lock), `preparing`,
`conditioning`, `sampling`, `vae_encode`, `vae_decode`, `audio_decode`,
`upscaling`, `adetailer`, and `encoding`. Stages may repeat or be skipped.
Details include `positive_prompt` / `negative_prompt` / `image_condition` for
conditioning, `high_noise` / `low_noise` for two-stage sampling, and `image` /
`video` for output encoding. During ADetailer, `detail` identifies the inner
stage, such as `conditioning` or `sampling`. Clients should tolerate unknown
stage, detail, and unit values.

Sampling counts completed logical steps using the actual schedule, including
strength adjustments. Each new sampling pass increments `pass` and starts at
zero, including batch items, highres passes, high/low-noise passes, and ADetailer
passes. The ratio is **current-pass progress**, not overall job completion.
The progress snapshot is published atomically. Preview metadata still describes
the preview frame, which may be older than the current progress.

Unknown totals are represented by null counts, not an estimated percentage.
For example, prompt encoding can report:

```json
"progress": {
  "stage": "conditioning",
  "detail": "positive_prompt",
  "pass": 0,
  "completed": null,
  "total": null,
  "unit": null,
  "stage_elapsed_ms": 3200,
  "elapsed_ms": 4100
}
```

Elapsed times freeze when a job completes, fails, or is cancelled; the last
stage is retained for inspection. Use `status` to determine the final outcome.
Elapsed time alone does not indicate that computation is advancing.

`preview` sub-fields:

| Field | Type | Notes |
| --- | --- | --- |
| `preview.pass` | `integer` | Sampling pass number, starting at 1 within the job |
| `preview.step` | `integer` | Positive logical sample step within this pass |
| `preview.total_steps` | `integer` | Actual sample steps in this pass, after schedule and strength adjustments |
| `preview.b64_json` | `string` | Base64-encoded PNG of the current preview frame |

The preview is updated in place as sampling progresses; poll `GET /sdcpp/v1/jobs/{id}` to retrieve the latest frame. Only the most recent preview is retained.

`step / total_steps` describes the current sampling pass, not overall job completion. A new batch item, high/low-noise stage, or highres pass increments `pass` and restarts `step`. Samplers with multiple denoiser evaluations per logical step may update a preview more than once at the same step. The frame and its pass/step metadata are published together.

### Endpoints

#### `GET /sdcpp/v1/capabilities`

Returns frontend-friendly capability metadata.

The mode-aware fields are the primary interface. The top-level compatibility fields are deprecated mirrors kept for older clients.

Top-level fields:

| Field | Type | Notes |
| --- | --- | --- |
| `model` | `object` | Loaded model metadata |
| `current_mode` | `string` | The native generation mode mirrored by top-level compatibility fields |
| `supported_modes` | `array<string>` | Supported native modes such as `img_gen` or `vid_gen` |
| `defaults` | `object` | Deprecated compatibility mirror of `defaults_by_mode[current_mode]` |
| `output_formats` | `array<string>` | Deprecated compatibility mirror of `output_formats_by_mode[current_mode]` |
| `features` | `object` | Deprecated compatibility mirror of `features_by_mode[current_mode]` |
| `defaults_by_mode` | `object` | Explicit defaults for each supported mode |
| `output_formats_by_mode` | `object` | Explicit output formats for each supported mode |
| `features_by_mode` | `object` | Explicit feature flags for each supported mode |
| `samplers` | `array<string>` | Available sampling methods |
| `schedulers` | `array<string>` | Available schedulers |
| `loras` | `array<object>` | Available LoRA entries |
| `upscalers` | `array<object>` | Available highres upscalers, built-in and model-backed |
| `preview_modes` | `array<string>` | Available preview modes, e.g. `["none", "proj", "tae", "vae"]` |
| `limits` | `object` | Shared queue and size limits |

`model`

| Field | Type |
| --- | --- |
| `model.name` | `string` |
| `model.stem` | `string` |
| `model.path` | `string` |

Compatibility rules:

- `defaults`, `output_formats`, and `features` are deprecated compatibility mirrors
- those three top-level fields always mirror `current_mode`
- `supported_modes`, `defaults_by_mode`, `output_formats_by_mode`, and `features_by_mode` are the mode-aware fields

Mode-aware objects:

| Field | Type |
| --- | --- |
| `defaults_by_mode.img_gen` | `object` |
| `defaults_by_mode.vid_gen` | `object` |
| `output_formats_by_mode.img_gen` | `array<string>` |
| `output_formats_by_mode.vid_gen` | `array<string>` |
| `features_by_mode.img_gen` | `object` |
| `features_by_mode.vid_gen` | `object` |

Shared nested fields:

`loras`

| Field | Type |
| --- | --- |
| `loras[].name` | `string` |
| `loras[].path` | `string` |
| `loras[].display_path` | `string` (display only) |

`upscalers`

| Field | Type | Notes |
| --- | --- | --- |
| `upscalers[].name` | `string` | Built-in name or model stem; use this value in `hires.upscaler` |
| `upscalers[].model` | `boolean` | True for a model-backed upscaler, false for a built-in scaling filter |

Built-in entries include `None`, `Lanczos`, `Nearest`, `Latent`, `Latent (nearest)`, `Latent (nearest-exact)`, `Latent (antialiased)`, `Latent (bicubic)`, and `Latent (bicubic antialiased)`. Model-backed entries are scanned from the top level of `--hires-upscalers-dir`; subdirectories are not scanned.

`limits`

| Field | Type |
| --- | --- |
| `limits.min_width` | `integer` |
| `limits.max_width` | `integer` |
| `limits.min_height` | `integer` |
| `limits.max_height` | `integer` |
| `limits.max_batch_count` | `integer` |
| `limits.max_queue_size` | `integer` |

Shared default fields used by both `img_gen` and `vid_gen`:

| Field | Type |
| --- | --- |
| `prompt` | `string` |
| `negative_prompt` | `string` |
| `clip_skip` | `integer` |
| `width` | `integer` |
| `height` | `integer` |
| `strength` | `number` |
| `seed` | `integer` |
| `sample_params` | `object` |
| `sample_params.scheduler` | `string` |
| `sample_params.sample_method` | `string` |
| `sample_params.sample_steps` | `integer` |
| `sample_params.eta` | `number \| null` |
| `sample_params.shifted_timestep` | `integer` |
| `sample_params.flow_shift` | `number \| null` |
| `sample_params.guidance.txt_cfg` | `number` |
| `sample_params.guidance.img_cfg` | `number \| null` |
| `sample_params.guidance.distilled_guidance` | `number` |
| `sample_params.guidance.slg.layers` | `array<integer>` |
| `sample_params.guidance.slg.layer_start` | `number` |
| `sample_params.guidance.slg.layer_end` | `number` |
| `sample_params.guidance.slg.scale` | `number` |
| `vae_tiling_params` | `object` |
| `vae_tiling_params.enabled` | `boolean` |
| `vae_tiling_params.temporal_tiling` | `boolean` |
| `vae_tiling_params.tile_size_w` | `integer` |
| `vae_tiling_params.tile_size_h` | `integer` |
| `vae_tiling_params.target_overlap` | `number` |
| `vae_tiling_params.rel_size_w` | `number` |
| `vae_tiling_params.rel_size_h` | `number` |
| `vae_tiling_params.extra_tiling_args` | `string` |
| `cache_mode` | `string` |
| `cache_option` | `string` |
| `scm_mask` | `string` |
| `scm_policy_dynamic` | `boolean` |
| `output_format` | `string` |
| `output_compression` | `integer` |

`vae_tiling_params.tile_size_w` and `tile_size_h` are in **image pixels**, with `0` selecting the 256-pixel default. Both encode and decode use these sizes without an encoding multiplier. Positive `rel_size_w`/`rel_size_h` values override the corresponding absolute size: values up to 1 are dimension fractions, and values greater than 1 are target tile counts. Set `enabled` to use spatial tiling. Sizes are aligned down to the VAE scale factor and capped at the input dimensions; explicit sizes below the minimum supported tile size are rejected. These fields previously used latent units; see [VAE tiling](../../docs/guides/performance.md#use-vae-tiling-to-reduce-encode-and-decode-memory-usage) for migration and OOM retry behavior.

`vae_tiling_params.extra_tiling_args` accepts a key=value list. Supported video VAEs accept `temporal_tile_frames` (alias `temporal_tile_size`, default `4`) and `temporal_tile_overlap` (default `1`). Wan2.2 TI2V 5B uses `temporal_decode_frames` (default `1`) to decode multiple latent frames per forward pass, trading additional memory for throughput; other model versions ignore this option.
LTX and Wan preserve causal state between temporal tiles. Hunyuan Video and TAEHV use overlap blending. MiniMax H3 keeps its model-specific fixed temporal windows because its latent-to-frame mapping is non-linear.

`img_gen`-specific default fields:

| Field | Type |
| --- | --- |
| `batch_count` | `integer` |
| `control_strength` | `number` |
| `control_type` | `string` |
| `ip_adapter_strength` | `number` |
| `hires` | `object` |
| `hires.enabled` | `boolean` |
| `hires.upscaler` | `string` |
| `hires.scale` | `number` |
| `hires.target_width` | `integer` |
| `hires.target_height` | `integer` |
| `hires.steps` | `integer` |
| `hires.denoising_strength` | `number` |
| `hires.custom_sigmas` | `array<number>` |
| `hires.upscale_tile_size` | `integer` |

`vid_gen`-specific default fields:

| Field | Type |
| --- | --- |
| `video_frames` | `integer` |
| `fps` | `integer` |
| `moe_boundary` | `number` |
| `vace_strength` | `number` |
| `high_noise_sample_params` | `object` |
| `high_noise_sample_params.scheduler` | `string` |
| `high_noise_sample_params.sample_method` | `string` |
| `high_noise_sample_params.sample_steps` | `integer` |
| `high_noise_sample_params.eta` | `number \| null` |
| `high_noise_sample_params.shifted_timestep` | `integer` |
| `high_noise_sample_params.flow_shift` | `number \| null` |
| `high_noise_sample_params.guidance.txt_cfg` | `number` |
| `high_noise_sample_params.guidance.img_cfg` | `number \| null` |
| `high_noise_sample_params.guidance.distilled_guidance` | `number` |
| `high_noise_sample_params.guidance.slg.layers` | `array<integer>` |
| `high_noise_sample_params.guidance.slg.layer_start` | `number` |
| `high_noise_sample_params.guidance.slg.layer_end` | `number` |
| `high_noise_sample_params.guidance.slg.scale` | `number` |

Fields returned in `features_by_mode.img_gen`:

- `init_image`
- `mask_image`
- `control_image`
- `ip_adapter_image`
- `ref_images`
- `lora`
- `vae_tiling`
- `hires`
- `cache`
- `cancel_queued`
- `cancel_generating`
- `preview`

Fields returned in `features_by_mode.vid_gen`:

- `init_image`
- `end_image`
- `control_frames`
- `high_noise_sample_params`
- `lora`
- `vae_tiling`
- `cache`
- `cancel_queued`
- `cancel_generating`
- `preview`

`cancel_queued` and `cancel_generating` tell whether [`POST /sdcpp/v1/jobs/{id}/cancel`](#post-sdcppv1jobsidcancel)
can stop a job that is waiting and one that is generating. Both are `true` in this version; a client that must also
talk to older servers checks `cancel_generating` before offering a stop button, because an older server answers `409`
for a job that has started.

#### `GET /sdcpp/v1/health`

Reports whether the server can generate and how it is coping with the memory of the machine. It never waits for a
generation in progress, so it answers quickly even while the server is busy. It needs the API key when one is set.

Example response:

```json
{
  "status": "degraded",
  "model_loaded": true,
  "profile": "lite",
  "base_profile": "max",
  "device": "auto",
  "cpu_modules": [],
  "fallback_devices": ["CUDA0", "Vulkan0"],
  "degradation": true,
  "busy": false,
  "counters": {
    "requests": 12,
    "succeeded": 11,
    "failed": 1,
    "cancelled": 3,
    "recovered": 2,
    "step_downs": 3,
    "retries": 0,
    "device_switches": 0,
    "promotions": 1,
    "failed_promotions": 0,
    "rebuilds": 4
  },
  "last_error": {
    "code": "out_of_memory",
    "message": "unet compute buffer allocation failed (out of memory)",
    "at": 1775401204
  },
  "next_promotion_seconds": 214,
  "uptime_seconds": 3601,
  "version": "stable-diffusion.cpp ...",
  "queue": {
    "pending": 0,
    "waiting": 0,
    "max": 64
  },
  "guardian": {
    "enabled": true,
    "halted": false,
    "reason": "",
    "since": 0,
    "retry_after_seconds": 0
  },
  "devices": [
    {
      "name": "CUDA0",
      "level": 2,
      "strain_levels": 3,
      "share_percent": 70,
      "excluded": false,
      "critical": false,
      "reason": "termico: GPU a 86 C, fascia attenzione, throttling termico",
      "held_ms": 4200,
      "next_step_ms": 0
    },
    {
      "name": "CPU",
      "level": 0,
      "strain_levels": 3,
      "share_percent": 100,
      "excluded": false,
      "critical": false,
      "reason": "",
      "held_ms": 0,
      "next_step_ms": 0
    }
  ]
}
```

| Field | Type | Notes |
| --- | --- | --- |
| `status` | `string` | `ok`, `degraded` (running on a lighter profile than the one it was started with, or on another device than the first one), `recovering` (the model is being loaded), `unavailable` (no usable model at the moment), `no_model` (no model is loaded: the server was started without one, or it was unloaded), or `halted` (the guardian has stopped all work; `guardian` says why) |
| `model_loaded` | `boolean` | |
| `profile` | `string \| null` | Profile in use: `max`, `balanced`, `lite`, `lite-disk`, `lite-hybrid` or `lite-cpu`; `null` without a model |
| `base_profile` | `string` | Profile the server was started with |
| `device` | `string \| null` | Where the model computes: `auto` when the engine chooses the device, the value of `--backend` or the device a device switch named, `cpu` at `lite-cpu`; `null` without a model |
| `cpu_modules` | `array \| null` | The modules the CPU computes next to the GPUs: `text_encoder`, `vae`, `controlnet` (from the profile or from `--clip-on-cpu`, `--vae-on-cpu`, `--control-net-cpu`), or `["all"]` at `lite-cpu`. Empty when the GPUs do all the work; `null` without a model |
| `fallback_devices` | `array` | The devices the server tries, in this order, when the device fails, before it takes the CPU. Empty without a device other than the CPU, with `--disable-degradation`, with `--backend cpu` and with a placement written in `--backend` |
| `degradation` | `boolean` | `false` with `--disable-degradation` |
| `busy` | `boolean` | A generation is running |
| `counters` | `object` | Since the start. `requests`, `succeeded`, `failed`; `cancelled` is the number of generations stopped on request (not failures, and they never make the server step down); `recovered` is the number of requests served after the model had to be loaded again to run them; `step_downs` (a lighter profile after an out-of-memory failure), `retries` (the same configuration tried again after a device failure), `device_switches` (another device, or the CPU, after a device failure), `promotions`, `failed_promotions` and `rebuilds` (times the model was loaded again). A job cancelled while it was still queued never reaches the model and is not counted |
| `last_error` | `object \| null` | Last failure the server saw (`code`, `message`, `at` as a Unix time). Invalid requests are not recorded |
| `next_promotion_seconds` | `integer \| null` | Seconds until the next attempt to climb back; `null` when the server is on its base profile and device or does not climb |
| `uptime_seconds` | `integer` | |
| `version` | `string` | |
| `queue` | `object` | Jobs of the async API: `pending` (queued or generating), `waiting` (queued) and `max` (limit before `429`) |
| `guardian` | `object` | `enabled` (`false` with `--guardian off`), `halted`, `reason` (empty when it is not halted), `since` (Unix time of the start of the halt; 0 when it is not halted) and `retry_after_seconds` (0 when it is not halted) |
| `devices` | `array<object>` | The components of [device health](#device-health-and-the-guardian): `name`; `level` (0 healthy, 1 to `strain_levels` under strain, `strain_levels + 1` excluded); `strain_levels`; `share_percent`, what it offers of its capacity; `excluded`; `critical`; `reason`, the `source: reason` of the signal, or the event, that set the level (empty when it is healthy); `held_ms`, how long the signals that hold the level still last; `next_step_ms`, until the level comes down one step if nothing else happens |

Typical status codes:

- `200 OK`
- `503 Service Unavailable` when `status` is `unavailable` or `halted`; the body is the same, and with `halted` a
  `Retry-After` header says when to ask again

#### `GET /sdcpp/v1/resources`

Reports the memory the server can see right now, for clients that want to size a request before sending it.
It needs the API key when one is set.

Example response:

```json
{
  "system_memory": {
    "total_bytes": 17179869184,
    "available_bytes": 9126805504,
    "process_bytes": 3221225472
  },
  "devices": [
    {
      "name": "CUDA0",
      "description": "NVIDIA GeForce RTX ...",
      "type": "gpu",
      "free_memory_bytes": 3758096384,
      "total_memory_bytes": 6442450944
    }
  ]
}
```

| Field | Type | Notes |
| --- | --- | --- |
| `system_memory` | `object \| null` | RAM of the machine (`total_bytes`, `available_bytes`) and the memory used by the server process (`process_bytes`); `null` where the platform does not report it |
| `devices` | `array<object>` | The compute devices, as listed by `--list-devices` |
| `devices[].type` | `string` | `cpu`, `gpu`, `igpu`, `accel` or `unknown` |

#### `POST /sdcpp/v1/devices/{device}/pressure`

An outside watcher (a thermal monitor such as `termico.py osserva --segnala` of
[`scripts/diagnostica_termica/`](../../scripts/diagnostica_termica/README.md), a supervisor, an operator) reports how a
component is doing. `{device}` is a name that [`GET /sdcpp/v1/health`](#get-sdcppv1health) lists under `devices`, in any
case. It needs the API key when one is set, and it answers while the server is halted, so that a watcher can withdraw
its signal.

Request body:

```json
{"level": 2, "reason": "GPU at 86 C with thermal throttling", "duration_seconds": 60, "source": "termico"}
```

| Field | Type | Notes |
| --- | --- | --- |
| `level` | `integer \| string` | `0` withdraws the signal of this source; `1` to `strain_levels` (3 by default) put the component under strain; `strain_levels + 1` or `"excluded"` exclude it; `"critical"` excludes it and asks the guardian to stop all work while the signal lasts |
| `reason` | `string` | Optional, one line. The health shows it as `source: reason` |
| `duration_seconds` | `integer` | Optional, 1 to 86400, default 60. The signal ends by itself after this time, so a watcher repeats it while the condition lasts |
| `source` | `string` | Optional, default `http`: 1 to 32 letters, digits, `.`, `_` or `-`. Each source holds one signal per component, and a new one replaces the old one; name it when more than one watcher reports |

Other fields are refused. The answer is the component and the guardian, as the health shows them; when the report
halts the server, the generation in progress is stopped at once.

```json
{
  "device": {"name": "CUDA0", "level": 2, "strain_levels": 3, "share_percent": 70, "excluded": false, "critical": false,
             "reason": "termico: GPU at 86 C with thermal throttling", "held_ms": 60000, "next_step_ms": 0},
  "guardian": {"enabled": true, "halted": false, "reason": "", "since": 0, "retry_after_seconds": 0}
}
```

Typical status codes:

- `200 OK`
- `400 Bad Request` with the code `invalid_report`: the body is not valid; `error` says why
- `404 Not Found` with the code `unknown_device`: the server has no such component
- `409 Conflict` with the code `report_refused`: the component holds the signals of 8 sources already

#### `POST /sdcpp/v1/guardian/resume`

An operator says the machine was checked: the critical signals end, the excluded components come back to the last level
of strain (the signals that exclude are lowered to it), and the server resumes unless something else still holds the
halt; the queued jobs run. It needs the API key when one is set, and answers `200` with `guardian` and `devices` as
[`GET /sdcpp/v1/health`](#get-sdcppv1health) shows them. Without a halt it only lowers what excludes.

#### `POST /sdcpp/v1/img_gen`

Submits an async image generation job.

Successful submission returns `202 Accepted`.

Example response:

```json
{
  "id": "job_01HTXYZABC",
  "kind": "img_gen",
  "status": "queued",
  "created": 1775401200000,
  "poll_url": "/sdcpp/v1/jobs/job_01HTXYZABC"
}
```

Response fields:

| Field | Type |
| --- | --- |
| `id` | `string` |
| `kind` | `string` |
| `status` | `string` |
| `created` | `integer` |
| `poll_url` | `string` |

#### `GET /sdcpp/v1/jobs/{id}`

Returns current job status.

Typical status codes:

- `200 OK`
- `404 Not Found`
- `410 Gone`

#### `POST /sdcpp/v1/jobs/{id}/cancel`

Stops a job, whether it is still waiting in the queue or already generating. There is no body: the URL is enough,
and `curl -X POST <url>` works as it is.

What happens depends on the state of the job:

| State of the job | What the server does | Answer |
| --- | --- | --- |
| `queued` | Takes it out of the queue; it never starts. | `200` with the job, `status: "cancelled"` |
| `generating` | Tells the library to stop. The job stays `generating` until execution stops at a cancellation checkpoint. Without `wait` the call answers at once; with `wait` it waits for the job to end. | Without `wait`: `200` with the job (`status: "generating"`, `"cancel_requested": true`) at once; poll the job until it is `cancelled`. With `wait`: `200` with the job (`status: "cancelled"`) as soon as it has ended; `202` with the job (`status: "generating"`, `"cancel_requested": true`) if it is still stopping after `wait` seconds |
| `completed`, `failed`, `cancelled` | Nothing to stop. | `200` with the job as it is |
| unknown id | | `404`; `410` if the job existed but was removed after its retention time |

Query parameter:

| Parameter | Type | Notes |
| --- | --- | --- |
| `wait` | `integer` | Optional. Whole seconds the call may wait for a generating job to end: `0` to `60`; a larger number is shortened to `60`. Without it the call answers at once, as the upstream server does. `0` asks for the stop and answers at once too, but with `202` unless the job had already ended. Anything that is not a whole number of seconds, 0 or more, is `400`. |

Example, stopping a job and waiting up to 10 seconds for it to end:

```shell
curl -s -X POST "http://127.0.0.1:1234/sdcpp/v1/jobs/job_6acf9114_00000000/cancel?wait=10"
```

```json
{
  "id": "job_6acf9114_00000000",
  "kind": "img_gen",
  "status": "cancelled",
  "cancel_requested": true,
  "created": 1775401200,
  "started": 1775401203,
  "completed": 1775401209,
  "queue_position": 0,
  "result": null,
  "error": { "code": "cancelled", "message": "job cancelled by client" }
}
```

Without `wait`, the answer comes at once and the job is still `generating` (`200`). With `wait`, the same answer comes
as `202` when the job is still stopping at the end of the wait:

```json
{
  "id": "job_6acf9114_00000000",
  "kind": "img_gen",
  "status": "generating",
  "cancel_requested": true,
  "created": 1775401200,
  "started": 1775401203,
  "completed": null,
  "queue_position": 0,
  "result": null,
  "error": null
}
```

Keep polling `GET /sdcpp/v1/jobs/{id}` (`cancel_requested` stays `true` while the job is generating) until its status
becomes `cancelled`, or send the same request again: repeating it is harmless. Cancelled jobs have no result.

Typical status codes:

- `200 OK`: the job is in a final state (the answer carries it), or the stop was requested and `wait` was not given
  (the job may still be `generating`)
- `202 Accepted`: the stop was requested, `wait` was given and the job had not ended yet
- `400 Bad Request`: invalid `wait`
- `404 Not Found`
- `409 Conflict`: only for a queued job that left the queue at the very moment of the request; ask again
- `410 Gone`

Notes:

- A generation stops at its next check: at the end of the sampling step it is in, and between the stages around it
  (encoding the prompt, decoding the image, a second pass). On a slow machine one step can take seconds, and the
  first generation after the start can spend a long time reading the weights before it reaches a check, so the stop
  may need longer than `wait`. That is what the `202` is for.
- If the job finishes before the stop takes effect, the answer is the finished job: `200` with
  `status: "completed"` and its result.
- A stopped job returns no images, not even the ones of a batch that were already done.
- A stop is not a failure. The model is not loaded again, the server does not step down to a lighter profile, nothing
  is recorded in `last_error`, and [`GET /sdcpp/v1/health`](#get-sdcppv1health) counts it in `counters.cancelled`.
  The next job runs on the same context.
- Only the jobs of this API can be stopped. The synchronous endpoints (`/sdapi/...`, `/v1/images/...`) are not jobs:
  a request that is waiting for the model behind a job that gets cancelled simply runs afterwards, as usual.
- With `wait`, the call keeps its HTTP connection open for up to `wait` seconds.

### Request Body

Example:

```json
{
  "prompt": "a cat sitting on a chair",
  "negative_prompt": "",
  "clip_skip": -1,
  "width": 1024,
  "height": 1024,
  "strength": 0.75,
  "seed": -1,
  "batch_count": 1,
  "control_strength": 0.9,
  "control_type": "",
  "ip_adapter_strength": 1.0,
  "embed_image_metadata": true,

  "init_image": null,
  "ref_images": [],
  "mask_image": null,
  "control_image": null,
  "ip_adapter_image": null,

  "sample_params": {
    "scheduler": "discrete",
    "sample_method": "euler_a",
    "sample_steps": 28,
    "eta": 1.0,
    "shifted_timestep": 0,
    "custom_sigmas": [],
    "flow_shift": 0.0,
    "guidance": {
      "txt_cfg": 7.0,
      "img_cfg": 7.0,
      "distilled_guidance": 3.5,
      "slg": {
        "layers": [7, 8, 9],
        "layer_start": 0.01,
        "layer_end": 0.2,
        "scale": 0.0
      }
    }
  },

  "lora": [],
  "hires": {
    "enabled": false,
    "upscaler": "Latent",
    "scale": 2.0,
    "target_width": 0,
    "target_height": 0,
    "steps": 0,
    "denoising_strength": 0.7,
    "custom_sigmas": [],
    "upscale_tile_size": 128
  },

  "vae_tiling_params": {
    "enabled": false,
    "temporal_tiling": false,
    "tile_size_w": 0,
    "tile_size_h": 0,
    "target_overlap": 0.5,
    "rel_size_w": 0.0,
    "rel_size_h": 0.0,
    "extra_tiling_args": ""
  },

  "cache_mode": "disabled",
  "cache_option": "",
  "scm_mask": "",
  "scm_policy_dynamic": true,

  "preview": "none",
  "preview_interval": 1,

  "output_format": "png",
  "output_compression": 100
}
```

### LoRA Rules

- The server only accepts explicit LoRA entries from the `lora` field.
- Prompt-embedded `<lora:...>` tags are intentionally unsupported.
- Clients should resolve LoRA usage through the structured `lora` array.

### Image Encoding Rules

Native generation follows model defaults. Custom input geometry and reference
conditioning use DGL tensor operations. See
[Image preprocessing](../../docs/guides/image_preprocessing.md) and
[Image editing](../../docs/guides/edit.md).

The former `image_preprocess` rule field is rejected. Use the DGL execution
endpoint for custom preprocessing.

Any image field accepts:

- a raw base64 string, or
- a data URL such as `data:image/png;base64,...`

Channel expectations:

- `init_image`: native channels (3 or 4); alpha is preserved and applied per model
- `ref_images[]`: native channels (3 or 4); alpha is preserved and applied per model
- `control_image`: 3 channels
- `ip_adapter_image`: 3 channels
- `mask_image`: 1 channel

Models that support RGBA (e.g. Qwen-Image 2.1) use the alpha channel of `init_image`
and `ref_images[]`. RGB-only models drop it, so sending RGBA is safe for every model.

If omitted or null:

- single-image fields clear the corresponding image tensor
- array fields map to an empty C-style array, represented as `pointer = nullptr` and `count = 0`

### Field Mapping Summary

Top-level scalar fields:

| Field | Type |
| --- | --- |
| `prompt` | `string` |
| `negative_prompt` | `string` |
| `clip_skip` | `integer` |
| `width` | `integer` |
| `height` | `integer` |
| `strength` | `number` |
| `seed` | `integer` |
| `batch_count` | `integer` |
| `control_strength` | `number` |
| `control_type` | `string` |
| `ip_adapter_strength` | `number` |
| `embed_image_metadata` | `boolean` |
| `preview` | `string` |
| `preview_interval` | `integer` |

Image fields:

| Field | Type |
| --- | --- |
| `init_image` | `string \| null` |
| `ref_images` | `array<string>` |
| `mask_image` | `string \| null` |
| `control_image` | `string \| null` |
| `ip_adapter_image` | `string \| null` |

LoRA fields:

| Field | Type |
| --- | --- |
| `lora[].path` | `string` |
| `lora[].multiplier` | `number` |
| `lora[].is_high_noise` | `boolean` |

Sampling fields:

| Field | Type |
| --- | --- |
| `sample_params.scheduler` | `string` |
| `sample_params.sample_method` | `string` |
| `sample_params.sample_steps` | `integer` |
| `sample_params.eta` | `number` |
| `sample_params.shifted_timestep` | `integer` |
| `sample_params.custom_sigmas` | `array<number>` |
| `sample_params.flow_shift` | `number` |
| `sample_params.guidance.txt_cfg` | `number` |
| `sample_params.guidance.img_cfg` | `number` |
| `sample_params.guidance.distilled_guidance` | `number` |
| `sample_params.guidance.slg.layers` | `array<integer>` |
| `sample_params.guidance.slg.layer_start` | `number` |
| `sample_params.guidance.slg.layer_end` | `number` |
| `sample_params.guidance.slg.scale` | `number` |

Other native fields:

| Field | Type |
| --- | --- |
| `hires` | `object` |
| `hires.enabled` | `boolean` |
| `hires.upscaler` | `string` |
| `hires.scale` | `number` |
| `hires.target_width` | `integer` |
| `hires.target_height` | `integer` |
| `hires.steps` | `integer` |
| `hires.denoising_strength` | `number` |
| `hires.custom_sigmas` | `array<number>` |
| `hires.upscale_tile_size` | `integer` |
| `vae_tiling_params` | `object` |
| `vae_tiling_params.enabled` | `boolean` |
| `vae_tiling_params.temporal_tiling` | `boolean` |
| `vae_tiling_params.tile_size_w` | `integer` |
| `vae_tiling_params.tile_size_h` | `integer` |
| `vae_tiling_params.target_overlap` | `number` |
| `vae_tiling_params.rel_size_w` | `number` |
| `vae_tiling_params.rel_size_h` | `number` |
| `vae_tiling_params.extra_tiling_args` | `string` |
| `cache_mode` | `string` |
| `cache_option` | `string` |
| `scm_mask` | `string` |
| `scm_policy_dynamic` | `boolean` |

For `hires.upscaler`, use `Lanczos`, `Nearest`, `Latent`, `Latent (nearest)`, `Latent (nearest-exact)`, `Latent (antialiased)`, `Latent (bicubic)`, `Latent (bicubic antialiased)`, or an `upscalers[].name` value from `GET /sdcpp/v1/capabilities`. Model-backed upscalers are resolved as `--hires-upscalers-dir / (name + ext)` and must live directly in that directory. `hires.custom_sigmas`, when present, overrides the generated second-pass hires sigma schedule; otherwise the hires schedule is trimmed by `hires.denoising_strength`.

HTTP-only output fields:

| Field | Type |
| --- | --- |
| `output_format` | `string` |
| `output_compression` | `integer` |

### Optional Field Handling

Optional sampling fields may be omitted.

When omitted, backend defaults apply to these fields:

- `sample_params.scheduler`
- `sample_params.sample_method`
- `sample_params.eta`
- `sample_params.flow_shift`
- `sample_params.guidance.img_cfg`

### Preview Interval Semantics

`preview_interval` controls the period (in sample steps) at which preview frames are generated. The default is `1` (every step). Any non-positive values will be clamped to `1`. Note that if `preview` is set to `"none"`, no previews are produced regardless of the interval.

### Completion Result

Example completed job:

```json
{
  "id": "job_01HTXYZABC",
  "kind": "img_gen",
  "status": "completed",
  "created": 1775401200000,
  "started": 1775401203000,
  "completed": 1775401215237,
  "queue_position": 0,
  "result": {
    "output_format": "png",
    "images": [
      {
        "index": 0,
        "b64_json": "iVBORw0KGgoAAA..."
      }
    ]
  },
  "error": null
}
```

### Failure Result

Example failed job:

```json
{
  "id": "job_01HTXYZABC",
  "kind": "img_gen",
  "status": "failed",
  "created": 1775401200000,
  "started": 1775401203000,
  "completed": 1775401204000,
  "queue_position": 0,
  "result": null,
  "error": {
    "code": "generation_failed",
    "message": "generate_image returned empty results"
  }
}
```

`error.code` is `generation_failed` when the cause is not known, or one of `out_of_memory`, `model_file`,
`backend`, `invalid_argument`, `busy` or `unavailable` (see [Failures and recovery](#failures-and-recovery)).
A job that failed for lack of memory has already been tried on every lighter profile the server could use.

### Cancelled Result

Example cancelled job (it was still waiting in the queue; a job stopped while it was generating has `started` set):

```json
{
  "id": "job_01HTXYZABC",
  "kind": "img_gen",
  "status": "cancelled",
  "created": 1775401200000,
  "started": null,
  "completed": 1775401202000,
  "queue_position": 0,
  "result": null,
  "error": {
    "code": "cancelled",
    "message": "job cancelled by client"
  }
}
```

### Submission Errors

`POST /sdcpp/v1/img_gen` may return:

- `202 Accepted` when the job is created
- `400 Bad Request` for an invalid model selection, an unsupported current model mode, or invalid request parameters
- `503 Service Unavailable` when the request uses the current model and the server has none configured
- `429 Too Many Requests` when the job queue is full (with `Retry-After: 10`)
- `500 Internal Server Error` for unexpected server exceptions during submission

### `vid_gen`

The following section documents the native async contract for video generation.

#### `POST /sdcpp/v1/vid_gen`

Submits an async video generation job.

Successful submission returns `202 Accepted`.

Example response:

```json
{
  "id": "job_01HTXYZVID",
  "kind": "vid_gen",
  "status": "queued",
  "created": 1775401200000,
  "poll_url": "/sdcpp/v1/jobs/job_01HTXYZVID"
}
```

Response fields:

| Field | Type |
| --- | --- |
| `id` | `string` |
| `kind` | `string` |
| `status` | `string` |
| `created` | `integer` |
| `poll_url` | `string` |

### Request Body

Compared with `img_gen`, the `vid_gen` request body:

- `vid_gen` is a single video sequence job, so `batch_count` is not part of the request schema
- `mask_image`, `control_image`, `control_strength`, `ip_adapter_image`, `ip_adapter_strength`, and `embed_image_metadata` are not part of the request schema
- `ref_images` is accepted for MiniMax-H3 Ref2VA conditioning; other video model families currently ignore it
- `vid_gen` adds `end_image`, `control_frames`, `high_noise_sample_params`, `video_frames`, `fps`, `moe_boundary`, and `vace_strength`

Example:

```json
{
  "prompt": "a cat walking through a rainy alley",
  "negative_prompt": "",
  "clip_skip": -1,
  "width": 832,
  "height": 480,
  "strength": 0.75,
  "seed": -1,
  "video_frames": 33,
  "fps": 16,
  "moe_boundary": 0.875,
  "vace_strength": 1.0,

  "init_image": null,
  "end_image": null,
  "ref_images": [],
  "control_frames": [],

  "sample_params": {
    "scheduler": "discrete",
    "sample_method": "euler",
    "sample_steps": 28,
    "eta": 1.0,
    "shifted_timestep": 0,
    "custom_sigmas": [],
    "flow_shift": 0.0,
    "guidance": {
      "txt_cfg": 7.0,
      "img_cfg": 7.0,
      "distilled_guidance": 3.5,
      "slg": {
        "layers": [7, 8, 9],
        "layer_start": 0.01,
        "layer_end": 0.2,
        "scale": 0.0
      }
    }
  },

  "high_noise_sample_params": {
    "scheduler": "discrete",
    "sample_method": "euler",
    "sample_steps": -1,
    "eta": 1.0,
    "shifted_timestep": 0,
    "flow_shift": 0.0,
    "guidance": {
      "txt_cfg": 7.0,
      "img_cfg": 7.0,
      "distilled_guidance": 3.5,
      "slg": {
        "layers": [7, 8, 9],
        "layer_start": 0.01,
        "layer_end": 0.2,
        "scale": 0.0
      }
    }
  },

  "lora": [],

  "vae_tiling_params": {
    "enabled": false,
    "temporal_tiling": false,
    "tile_size_w": 0,
    "tile_size_h": 0,
    "target_overlap": 0.5,
    "rel_size_w": 0.0,
    "rel_size_h": 0.0,
    "extra_tiling_args": ""
  },

  "cache_mode": "disabled",
  "cache_option": "",
  "scm_mask": "",
  "scm_policy_dynamic": true,

  "preview": "none",
  "preview_interval": 1,

  "output_format": "webm",
  "output_compression": 100
}
```

### Reference Image Rules

- `ref_images` contains reference images for MiniMax-H3 Ref2VA conditioning.
- Images retain request order and correspond to `<Picture 1>`, `<Picture 2>`, and so on in the prompt.
- For MiniMax-H3, non-empty `ref_images` cannot be combined with `init_image` or `end_image`. A conflicting API request fails during generation; the WebUI checks this before submission.
- MiniMax-H3 does not support `control_frames`; leave that array empty.
- Other video model families currently ignore `ref_images`.
- `features_by_mode.vid_gen` does not currently advertise `ref_images`; its absence is not an indication that MiniMax-H3 reference images are unsupported.

See [MiniMax-H3 reference conditioning](../../docs/models/minimax_h3.md#reference-to-audio-video-conditioning) for model requirements and prompt examples.

### LoRA Rules

- The server only accepts explicit LoRA entries from the `lora` field.
- Prompt-embedded `<lora:...>` tags are intentionally unsupported.
- `lora[].is_high_noise` controls whether a LoRA applies only to the high-noise stage.

### Image and Frame Encoding Rules

Any image field accepts:

- a raw base64 string, or
- a data URL such as `data:image/png;base64,...`

Channel expectations:

- `init_image`: 3 channels
- `end_image`: 3 channels
- `ref_images[]`: decoded with native channels, then converted to RGB by MiniMax-H3
- `control_frames[]`: 3 channels

Frame ordering rules:

- `control_frames[]` order is the conditioning frame order
- `control_frames[]` is preserved in request order

If omitted or null:

- single-image fields clear the corresponding image tensor
- array fields map to an empty C-style array, represented as `pointer = nullptr` and `count = 0`

### Field Mapping Summary

Top-level scalar fields:

| Field | Type |
| --- | --- |
| `prompt` | `string` |
| `negative_prompt` | `string` |
| `clip_skip` | `integer` |
| `width` | `integer` |
| `height` | `integer` |
| `strength` | `number` |
| `seed` | `integer` |
| `video_frames` | `integer` |
| `fps` | `integer` |
| `moe_boundary` | `number` |
| `vace_strength` | `number` |
| `preview` | `string` |
| `preview_interval` | `integer` |

Image and frame fields:

| Field | Type |
| --- | --- |
| `init_image` | `string \| null` |
| `end_image` | `string \| null` |
| `ref_images` | `array<string>` |
| `control_frames` | `array<string>` |

LoRA fields:

| Field | Type |
| --- | --- |
| `lora[].path` | `string` |
| `lora[].multiplier` | `number` |
| `lora[].is_high_noise` | `boolean` |

Sampling fields:

| Field | Type |
| --- | --- |
| `sample_params.scheduler` | `string` |
| `sample_params.sample_method` | `string` |
| `sample_params.sample_steps` | `integer` |
| `sample_params.eta` | `number` |
| `sample_params.shifted_timestep` | `integer` |
| `sample_params.custom_sigmas` | `array<number>` |
| `sample_params.flow_shift` | `number` |
| `sample_params.guidance.txt_cfg` | `number` |
| `sample_params.guidance.img_cfg` | `number` |
| `sample_params.guidance.distilled_guidance` | `number` |
| `sample_params.guidance.slg.layers` | `array<integer>` |
| `sample_params.guidance.slg.layer_start` | `number` |
| `sample_params.guidance.slg.layer_end` | `number` |
| `sample_params.guidance.slg.scale` | `number` |

High-noise sampling fields:

The two video stages share the scheduler, flow shift and timestep shifting.
`high_noise_sample_params.scheduler`, `.flow_shift` and `.shifted_timestep`
are accepted as aliases for those shared settings in video requests. Image
requests ignore these high-noise fields. Within one video request, a
corresponding `sample_params` field takes precedence. Request aliases override
inherited startup and recipe values; `null` clears the inherited shared setting.

| Field | Type |
| --- | --- |
| `high_noise_sample_params.scheduler` | `string` |
| `high_noise_sample_params.sample_method` | `string` |
| `high_noise_sample_params.sample_steps` | `integer` |
| `high_noise_sample_params.eta` | `number` |
| `high_noise_sample_params.shifted_timestep` | `integer` |
| `high_noise_sample_params.flow_shift` | `number` |
| `high_noise_sample_params.guidance.txt_cfg` | `number` |
| `high_noise_sample_params.guidance.img_cfg` | `number` |
| `high_noise_sample_params.guidance.distilled_guidance` | `number` |
| `high_noise_sample_params.guidance.slg.layers` | `array<integer>` |
| `high_noise_sample_params.guidance.slg.layer_start` | `number` |
| `high_noise_sample_params.guidance.slg.layer_end` | `number` |
| `high_noise_sample_params.guidance.slg.scale` | `number` |

Other native fields:

| Field | Type |
| --- | --- |
| `vae_tiling_params` | `object` |
| `vae_tiling_params.enabled` | `boolean` |
| `vae_tiling_params.temporal_tiling` | `boolean` |
| `vae_tiling_params.tile_size_w` | `integer` |
| `vae_tiling_params.tile_size_h` | `integer` |
| `vae_tiling_params.target_overlap` | `number` |
| `vae_tiling_params.rel_size_w` | `number` |
| `vae_tiling_params.rel_size_h` | `number` |
| `vae_tiling_params.extra_tiling_args` | `string` |
| `cache_mode` | `string` |
| `cache_option` | `string` |
| `scm_mask` | `string` |
| `scm_policy_dynamic` | `boolean` |

HTTP-only output fields:

| Field | Type |
| --- | --- |
| `output_format` | `string` |
| `output_compression` | `integer` |

For `vid_gen`, `output_format` and `output_compression` control container encoding.
`fps` is request metadata for the generated sequence and is echoed in the completed job result.

Allowed `output_format` values:

- `webm`
- `webp`
- `avi`

Output format behavior:

- `output_format` defaults to `webm`
- `webp` means animated WebP
- `avi` means MJPG AVI
- `webm` requires the server to be built with WebM support; otherwise the request returns `400`

### Result Payload

Completed jobs return one encoded container payload, not a list of per-frame images.

Result fields:

- `result.b64_json` contains the whole encoded container file as base64
- `result.mime_type` identifies the media type
- `result.output_format` echoes the selected container format
- `result.fps` echoes the effective playback FPS
- `result.frame_count` reports the actual decoded frame count used to build the container

Expected MIME types:

| `output_format` | `mime_type` |
| --- | --- |
| `webm` | `video/webm` |
| `webp` | `image/webp` |
| `avi` | `video/x-msvideo` |

### Optional Field Handling

Optional sampling fields may be omitted.

When omitted, backend defaults apply to these fields:

- `sample_params.scheduler`
- `sample_params.sample_method`
- `sample_params.eta`
- `sample_params.flow_shift`
- `sample_params.guidance.img_cfg`
- `high_noise_sample_params.scheduler`
- `high_noise_sample_params.sample_method`
- `high_noise_sample_params.eta`
- `high_noise_sample_params.flow_shift`
- `high_noise_sample_params.guidance.img_cfg`

`high_noise_sample_params` may also be omitted entirely.

### Frame Count Semantics

`video_frames` is the requested target length. The core normalizes it according
to the loaded model:

| Model family | Frame alignment | Fixed FPS |
| --- | --- | --- |
| Wan, HunyuanVideo, LingBot | `4n + 1`, round up | Wan 2.2 S2V: 16 |
| LTX | `8n + 1`, round up | None |
| MiniMax-H3 | `17n + 5`, round up, minimum 5 | 24 |

AnimateDiff requires an explicit DGL workflow rather than the native video
generation API.

Wan examples:

- `video_frames = 33` stays `33`
- `video_frames = 34` becomes `37`
- `video_frames = 32` becomes `33`

At 16 FPS, a 5-second target requests 80 frames and aligns to 81 frames for Wan,
or approximately 5.063 seconds. Audio-driven generation and temporal upscaling
can change the final length. Use the completed job's actual decoded `frame_count`
and `fps` for the final playback duration.

### Completion Result

Example completed job:

```json
{
  "id": "job_01HTXYZVID",
  "kind": "vid_gen",
  "status": "completed",
  "created": 1775401200000,
  "started": 1775401203000,
  "completed": 1775401215237,
  "queue_position": 0,
  "result": {
    "output_format": "webm",
    "mime_type": "video/webm",
    "fps": 16,
    "frame_count": 33,
    "b64_json": "GkXfo59ChoEBQveBAULygQRC84EIQo..."
  },
  "error": null
}
```

The response returns the encoded `.webm`, animated `.webp`, or `.avi` container payload directly.

### Failure Result

Example failed job:

```json
{
  "id": "job_01HTXYZVID",
  "kind": "vid_gen",
  "status": "failed",
  "created": 1775401200000,
  "started": 1775401203000,
  "completed": 1775401204000,
  "queue_position": 0,
  "result": null,
  "error": {
    "code": "generation_failed",
    "message": "generate_video returned no results"
  }
}
```

`error.code` takes the same values as for `img_gen` jobs.

### Cancelled Result

Example cancelled job (it was still waiting in the queue; a job stopped while it was generating has `started` set):

```json
{
  "id": "job_01HTXYZVID",
  "kind": "vid_gen",
  "status": "cancelled",
  "created": 1775401200000,
  "started": null,
  "completed": 1775401202000,
  "queue_position": 0,
  "result": null,
  "error": {
    "code": "cancelled",
    "message": "job cancelled by client"
  }
}
```

### Submission Errors

`POST /sdcpp/v1/vid_gen` may return:

- `202 Accepted` when the job is created
- `400 Bad Request` for an invalid model selection, an unsupported current model mode, invalid request parameters, or an unsupported output format
- `503 Service Unavailable` when the request uses the current model and the server has none configured
- `429 Too Many Requests` when the job queue is full (with `Retry-After: 10`)
- `500 Internal Server Error` for unexpected server exceptions during submission

## DGL editor API

These endpoints are available to the local, same-origin Web UI. The server uses
the working directory's ./dgl directory unless --dgl-dir is set.

| Method and path | Request or response |
| --- | --- |
| GET /sdcpp/v1/dgl/scripts | Returns directory and a list of scripts |
| GET /sdcpp/v1/dgl/script?name=example.dgl | Returns the script as source |
| POST /sdcpp/v1/dgl/script | Saves UTF-8 source with the given name. Pass previous_source to update an existing script. A conflict returns HTTP 409. |
| POST /sdcpp/v1/dgl/jobs | Queues the source string and returns a job ID with HTTP 202 |
| GET /sdcpp/v1/dgl/jobs/{id} | Returns the job status, error or result with tensor descriptions and inline encoded media |
| GET /sdcpp/v1/dgl/jobs/{id}/outputs/{index}?frame=0 | Returns a PNG of an image output or a video frame. Output and frame indexes start at zero. |

Scripts share the component cache and execution settings used for generation.
Model reads are restricted to the configured model directories. Script writes
are restricted to output/ in the server's working directory.

### Encoded DGL results

The existing job result has `type`, `outputs` and `media`. `outputs` retains
the flat tensor descriptions. `media` recognizes a root `{data, mime_type}`
dict or such dicts in the root's named fields. `data` must be DGL `bytes`, and
the supported MIME types are `image/png`, `video/webm` and `video/x-msvideo`.
Bare bytes, U8 tensors and more deeply nested dicts are not inferred as media.
The name of a root media dict is `result`.

```json
{
  "type": "dict",
  "outputs": [],
  "media": [
    {"name": "cover", "mime_type": "image/png", "b64_json": "..."},
    {"name": "clip", "mime_type": "video/webm", "b64_json": "..."}
  ]
}
```

The script encodes media once. After execution releases the generation lock,
the server converts returned bytes to Base64 once, before publishing
`completed`. Polling returns the stored result; it does not encode a video,
convert Base64 again, or require frame requests. The frame endpoint remains
available for explicit tensor inspection. Native `/sdcpp/v1/vid_gen` is unchanged.

Inline media share a 64 MiB raw-byte budget per job. An item that exceeds the
remaining budget has no `b64_json`; it instead contains
`error: {code: "inline_too_large", message: "..."}`. Other media and tensor
outputs remain available, and any file already written by `media::save`
remains intact. Encoder failures fail the script through the ordinary DGL
execution error path. WebM encoding rejects nonempty audio instead of silently
removing it. Use AVI encoding for a PCM16 soundtrack.

## Actual generation parameters

Generation responses expose the executed DGL source as `generation_info`,
independently of `embed_image_metadata`. Native async image result items and
OpenAI image items contain each image's source with its resolved seed. Native
video results contain the sequence-wide source. SD API `info.generation_info` is an array
aligned with `info.infotexts` and `info.all_seeds`; the outer `parameters` field
remains the request echo. See [generation metadata](../../docs/guides/generation_metadata.md)
for source semantics and container storage.
