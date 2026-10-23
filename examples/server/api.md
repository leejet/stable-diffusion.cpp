# stable-diffusion.cpp Server APIs

This document describes the server-facing APIs exposed by `examples/server`.

The server currently exposes three API families:

- `OpenAI API` under `/v1/...`
- `Stable Diffusion WebUI API` under `/sdapi/v1/...`
- `sdcpp API` under `/sdcpp/v1/...`

The `sdcpp API` is the native API surface.
Its request schema is the same schema used by `sd_cpp_extra_args`.

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
- `POST /sdcpp/v1/img_gen`
- `GET /sdcpp/v1/jobs/{id}`
- `POST /sdcpp/v1/jobs/{id}/cancel`
- `POST /sdcpp/v1/vid_gen`
- `POST /sdcpp/v1/upscale`

## Model management

Model management is additive: existing generation request bodies, successful
responses, job polling and cancellation remain unchanged. All three API families
use the current loaded context. Starting with the existing model arguments
preserves command-line preloading and its generation defaults; it does not
require a recipe.

Recipe defaults apply when a generation request omits the corresponding
parameters. Image editing endpoints retain their existing input-image size
inference when dimensions are omitted.

Without a loaded model, generation endpoints return HTTP 503 with
`{"error":"No model is loaded","code":"model_not_loaded"}`.
While loading/unloading, endpoints that access the context return HTTP 503
with `code: "model_loading"`. These errors retain the existing string-valued
`error` field. Job polling/cancellation, catalog queries and model status remain
available during loading. Model lists are empty when no model is loaded;
`/v1/models` retains the `sd-cpp-local` ID when ready.

| Method | Endpoint | Response / behavior |
| --- | --- | --- |
| GET | `/sdcpp/v1/recipes` | `{recipes: [...], errors: [...]}` |
| GET | `/sdcpp/v1/model-files` | `{files: [{id, name, category, root, relative_path}], errors: [...]}` |
| POST | `/sdcpp/v1/models/refresh` | Rescan recipes and files; return `{errors: [...]}` |
| GET | `/sdcpp/v1/models/status` | Current state, recipe, selection, error, loading time and optional revision |
| POST | `/sdcpp/v1/models/load` | Load the submitted recipe and selection; return status after completion |
| POST | `/sdcpp/v1/models/unload` | Release the active context; return empty status |

Load request:

```json
{
  "recipe": "flux1-dev",
  "models": {
    "diffusion_model": "diffusion_models/flux1-dev.gguf",
    "clip_l": "text_encoders/clip_l.safetensors",
    "t5xxl": "text_encoders/t5xxl.safetensors",
    "vae": "vae/ae.safetensors"
  }
}
```

File IDs are opaque strings returned by `model-files`. The first model root
retains the existing relative IDs, such as `checkpoints/nested/model.gguf`.
Additional roots use `@<root-id>/<relative-path>` so identical paths in different
roots remain selectable. Root qualifiers derive from the canonical root path;
adding unrelated roots or refreshing the catalog does not change them. Moving
a root changes its qualifier. Do not construct IDs or use display names in load
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
paths escaping the configured root are rejected. Optional slots can be omitted
or set to an empty string. Load/unload returns HTTP 409 (`model_busy`) while
requests or queued jobs are pending. Invalid configurations return HTTP 400
(`invalid_model_config`) before unloading the current context. An initialization
failure returns HTTP 422 (`model_load_failed`) and leaves no active context.
The load HTTP request waits for completion; a separate client can poll status
while it runs. Refreshing the catalog never modifies an already loaded context.

Status uses `state: empty | loading | ready | error`, `recipe` (ID or null),
`models` (slot-to-file-ID mapping), and `error` (string or null). A monotonically
increasing `revision` identifies runtime transitions within this server process.
Command-line preloading keeps `origin: command_line` and `recipe: null` for
compatibility, and adds `runtime_recipe`: a read-only `@command-line` recipe
with all feature declarations enabled. It is only available while that context
is loaded and cannot be submitted to `models/load`. The UI selects it by default
and uses command-line generation defaults, not defaults from a catalog recipe.
Capabilities also include this status as an additive `runtime` field.

`load_elapsed_seconds` is the server-measured duration in seconds (a number or
null). It updates while a model is loading and freezes on success or failure.
Timing starts after selection validation and includes releasing the previous
model. It resets for each load and is null before any load and during/after
unloading. Command-line preloading reports its completed initialization time.
Poll `/sdcpp/v1/models/status` to display elapsed time, including while loading;
timer updates alone do not change `revision`.

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
ControlNet controls are shown when previewing a supporting recipe, even before
a model file is selected. High-noise and TAE controls additionally require their
component to be selected. Runtime capabilities require the component to be loaded.
PuLID disables SLG while selected.
Image-only and video-only inputs are filtered by the active generation mode.
Eta depends on the selected sampler; SCM fields only apply to DBCache,
TaylorSeer and Cache-DIT. Empty panels and advanced sections are hidden.

`GET /sdcpp/v1/recipes` adds a resolved boolean `effective_features` object to
each recipe. This is response-only metadata, like `source` and slot `categories`;
omit it when writing a user recipe file. The defaults are maintained by the
server, not duplicated by the frontend.

`GET /sdcpp/v1/capabilities` adds `generation_features` for the loaded context,
derived from the active recipe and component availability without inferring
features from model-family names. When no model is loaded, every feature is
present and false. The UI can still preview a selected recipe using its
`effective_features` before loading.
Existing `features` / `features_by_mode` keys are retained and their model-specific
values reflect these declarations. A loaded recipe is snapshotted: refreshing
recipe files does not change the active context's capabilities until reloading.
Command-line preloading uses the runtime-only all-features recipe instead of model-family inference
and component paths; variant-specific recipe declarations are more precise.

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
`model_url`, `required_any`, `defaults`, `context`, `features` and `disabled_features` are optional. Each slot requires only `id` and boolean `required`.
The server supplies the default label and search directories for each slot.
An optional nonempty `label` overrides the display name, for example
`{"id": "diffusion_model", "label": "Low-noise diffusion model", "required": true}`.
Legacy `category` fields remain accepted, but must match one of the slot's
supported directories; they cannot redefine or restrict its search directories.
At least one required `model` or `diffusion_model` slot is necessary.

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
use `/sdcpp/v1/vid_gen` after loading a video model.
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
geometry follows `image_preprocess`: references preserve their dimensions by
default, while init and mask use the generation canvas preset.

Reference encoding then follows model presets and `ref_image_args`. To skip
input geometry for references and disable resizing before VAE encoding, include
this in `prompt`:

```text
edit this image <sd_cpp_extra_args>{"image_preprocess":"target=ref,mode=none","ref_image_args":"resize_before_vae=false"}</sd_cpp_extra_args>
```

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
| `preview` | `object \| null` |

`created`, `started`, and `completed` are integer Unix timestamps in
**milliseconds**. Subtract `started` from `completed` and divide by 1000 to get
the elapsed generation time in seconds, including output encoding and excluding
queue time.

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
| `upscale` | `boolean` | Whether a compatible RGB ESRGAN model is available for `POST /sdcpp/v1/upscale` |
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
| `upscalers[].image_upscale` | `boolean` | Whether this model can be selected by `POST /sdcpp/v1/upscale`; false for latent upscalers and built-in filters |

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
| `limits.max_upscale_width` | `integer` |
| `limits.max_upscale_height` | `integer` |

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

`vae_tiling_params.tile_size_w` and `tile_size_h` are in **image pixels**, with `0` selecting the 256-pixel default. Both encode and decode use these sizes without an encoding multiplier. Positive `rel_size_w`/`rel_size_h` values override the corresponding absolute size: values up to 1 are dimension fractions, and values greater than 1 are target tile counts. Set `enabled` to use spatial tiling. Sizes are aligned down to the VAE scale factor and capped at the input dimensions; explicit sizes below the minimum supported tile size are rejected. These fields previously used latent units; see [VAE tiling](../../docs/performance.md#use-vae-tiling-to-reduce-encode-and-decode-memory-usage) for migration and OOM retry behavior.

`vae_tiling_params.extra_tiling_args` accepts a key=value list. Supported video VAEs accept `temporal_tile_frames` (alias `temporal_tile_size`, default `4`) and `temporal_tile_overlap` (default `1`).
LTX and Wan preserve causal state between temporal tiles. Hunyuan Video and TAEHV use overlap blending. MiniMax H3 keeps its model-specific fixed temporal windows because its latent-to-frame mapping is non-linear.

`img_gen`-specific default fields:

| Field | Type |
| --- | --- |
| `batch_count` | `integer` |
| `ref_image_args` | `string` |
| `increase_ref_index` | `boolean` |
| `control_strength` | `number` |
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

#### `POST /sdcpp/v1/upscale`

Runs one RGB ESRGAN upscaler over an image, with no generation involved. Latent upscaler models remain available for hires generation but cannot be used here.

This is the HTTP equivalent of `sd-cli -M upscale`: no diffusion model, text
encoder or sampling is used, so it is fast enough to answer synchronously and
does not create a job.

Request fields:

| Field | Type | Notes |
| --- | --- | --- |
| `image` | `string` | Required. Base64 or data URL image |
| `upscaler` | `string` | A name from `upscalers` with `image_upscale: true`; the first compatible entry when omitted |
| `repeats` | `integer` | Run the upscaler this many times, 1 to 4 (default `1`) |
| `tile_size` | `integer` | Tile size, defaulting to the server's `--upscale-tile-size` |
| `output_format` | `string` | `png`, `jpeg`, or `webp` when built with WebP support (default `png`); unsupported formats return 400 |
| `output_compression` | `integer` | Range is clamped to `0..100` |

Response fields:

| Field | Type | Notes |
| --- | --- | --- |
| `images` | `array<object>` | One image |
| `images[].index` | `integer` | |
| `images[].b64_json` | `string` | Base64-encoded image bytes |
| `upscaler` | `string` | The upscaler actually used |
| `scale` | `integer` | The model's scale factor |
| `repeats` | `integer` | How many times it was run |
| `width` | `integer` | Result width |
| `height` | `integer` | Result height |
| `output_format` | `string` | Final encoded image format |

Typical status codes:

- `200 OK`
- `400 Bad Request` (invalid request, unsupported output format, unreadable image, incompatible upscaler, or output dimensions exceeding the limit)
- `500 Internal Server Error`

Notes:

- Final output dimensions, including all repeats, must not exceed 8192 pixels on either axis (`limits.max_upscale_width` and `limits.max_upscale_height`). Requests exceeding this bound are rejected before upscaling.
- The upscaler models are three-channel; alpha is not preserved.
- The request holds the generation context lock, so an upscale and a
  generation never run on the device at the same time.

#### `POST /sdcpp/v1/jobs/{id}/cancel`

Cancels a queued job immediately, or requests interruption of a generating image or
video job. A generating job returns `200 OK` with `cancel_requested: true` and
remains `generating` until execution stops at a cancellation checkpoint. Continue
polling until its status becomes `cancelled`. Cancelled jobs have no result.
Repeated cancellation requests are safe; terminal jobs are returned unchanged.

Typical status codes:

- `200 OK`
- `404 Not Found`
- `409 Conflict`
- `410 Gone`

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
  "ref_image_args": "",
  "increase_ref_index": false,
  "control_strength": 0.9,
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

Native image/video requests and SDAPI accept `image_preprocess` as a rule string
or array of rule strings. OpenAI-compatible requests can supply it in
`sd_cpp_extra_args`. See [Image preprocessing](../../docs/image_preprocessing.md)
for one-time input geometry, native-resolution decoding, mask alignment, and
`canny=true` for edge detection on any supported image input.

Image generation also accepts `ref_image_args` as a string (for example,
`"resize_before_vae=false"`) in native and SDAPI requests, or through
`sd_cpp_extra_args` in OpenAI-compatible requests. It controls downstream
reference encoding and is independent of input geometry rules.

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

- single-image fields map to an empty `sd_image_t`
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
| `ref_image_args` | `string` |
| `image_preprocess` | `string \| array<string>` |
| `increase_ref_index` | `boolean` |
| `control_strength` | `number` |
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

### Cancelled Result

Example cancelled job:

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
- `400 Bad Request` for an empty body, unsupported model mode, invalid JSON, or invalid generation parameters
- `429 Too Many Requests` when the job queue is full
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

See [MiniMax-H3 reference conditioning](../../docs/minimax_h3.md#reference-to-audio-video-conditioning) for model requirements and prompt examples.

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

- single-image fields map to an empty `sd_image_t`
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

`video_frames` is the requested target length, but the current core video path internally normalizes the effective frame count to the largest `4n + 1` value that does not exceed the requested count.

Examples:

- `video_frames = 33` stays `33`
- `video_frames = 34` becomes `33`
- `video_frames = 32` becomes `29`

The completed job payload includes the actual decoded `frame_count`.

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

### Cancelled Result

Example cancelled job:

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
- `400 Bad Request` for an empty body, unsupported model mode, invalid JSON, invalid generation parameters, or an unsupported output format
- `429 Too Many Requests` when the job queue is full
- `500 Internal Server Error` for unexpected server exceptions during submission
