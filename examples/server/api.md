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
  "created": 1775401200,
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
| `upscalers` | `array<object>` | Available model-backed highres upscalers |
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

`upscalers`

| Field | Type | Notes |
| --- | --- | --- |
| `upscalers[].name` | `string` | Built-in name or model stem; use this value in `hires.upscaler` |

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
| `vae_tiling_params.tile_size_x` | `integer` |
| `vae_tiling_params.tile_size_y` | `integer` |
| `vae_tiling_params.target_overlap` | `number` |
| `vae_tiling_params.rel_size_x` | `number` |
| `vae_tiling_params.rel_size_y` | `number` |
| `cache_mode` | `string` |
| `cache_option` | `string` |
| `scm_mask` | `string` |
| `scm_policy_dynamic` | `boolean` |
| `output_format` | `string` |
| `output_compression` | `integer` |

`img_gen`-specific default fields:

| Field | Type |
| --- | --- |
| `batch_count` | `integer` |
| `auto_resize_ref_image` | `boolean` |
| `increase_ref_index` | `boolean` |
| `control_strength` | `number` |
| `hires` | `object` |
| `hires.enabled` | `boolean` |
| `hires.upscaler` | `string` |
| `hires.scale` | `number` |
| `hires.target_width` | `integer` |
| `hires.target_height` | `integer` |
| `hires.steps` | `integer` |
| `hires.denoising_strength` | `number` |
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
- `ref_images`
- `lora`
- `vae_tiling`
- `hires`
- `cache`
- `cancel_queued`
- `cancel_generating`

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

#### `POST /sdcpp/v1/img_gen`

Submits an async image generation job.

Successful submission returns `202 Accepted`.

Example response:

```json
{
  "id": "job_01HTXYZABC",
  "kind": "img_gen",
  "status": "queued",
  "created": 1775401200,
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

Attempts to cancel an accepted job.

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
  "auto_resize_ref_image": true,
  "increase_ref_index": false,
  "control_strength": 0.9,
  "embed_image_metadata": true,

  "init_image": null,
  "ref_images": [],
  "mask_image": null,
  "control_image": null,

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
    "upscale_tile_size": 128
  },

  "vae_tiling_params": {
    "enabled": false,
    "tile_size_x": 0,
    "tile_size_y": 0,
    "target_overlap": 0.5,
    "rel_size_x": 0.0,
    "rel_size_y": 0.0
  },

  "cache_mode": "disabled",
  "cache_option": "",
  "scm_mask": "",
  "scm_policy_dynamic": true,

  "output_format": "png",
  "output_compression": 100
}
```

### LoRA Rules

- The server only accepts explicit LoRA entries from the `lora` field.
- Prompt-embedded `<lora:...>` tags are intentionally unsupported.
- Clients should resolve LoRA usage through the structured `lora` array.

### Image Encoding Rules

Any image field accepts:

- a raw base64 string, or
- a data URL such as `data:image/png;base64,...`

Channel expectations:

- `init_image`: 3 channels
- `ref_images[]`: 3 channels
- `control_image`: 3 channels
- `mask_image`: 1 channel

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
| `auto_resize_ref_image` | `boolean` |
| `increase_ref_index` | `boolean` |
| `control_strength` | `number` |
| `embed_image_metadata` | `boolean` |

Image fields:

| Field | Type |
| --- | --- |
| `init_image` | `string \| null` |
| `ref_images` | `array<string>` |
| `mask_image` | `string \| null` |
| `control_image` | `string \| null` |

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
| `hires.upscale_tile_size` | `integer` |
| `vae_tiling_params` | `object` |
| `cache_mode` | `string` |
| `cache_option` | `string` |
| `scm_mask` | `string` |
| `scm_policy_dynamic` | `boolean` |

For `hires.upscaler`, use `Lanczos`, `Nearest`, `Latent`, `Latent (nearest)`, `Latent (nearest-exact)`, `Latent (antialiased)`, `Latent (bicubic)`, `Latent (bicubic antialiased)`, or an `upscalers[].name` value from `GET /sdcpp/v1/capabilities`. Model-backed upscalers are resolved as `--hires-upscalers-dir / (name + ext)` and must live directly in that directory.

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

### Completion Result

Example completed job:

```json
{
  "id": "job_01HTXYZABC",
  "kind": "img_gen",
  "status": "completed",
  "created": 1775401200,
  "started": 1775401203,
  "completed": 1775401215,
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
  "created": 1775401200,
  "started": 1775401203,
  "completed": 1775401204,
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
  "created": 1775401200,
  "started": null,
  "completed": 1775401202,
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
  "created": 1775401200,
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
- `ref_images`, `mask_image`, `control_image`, `control_strength`, and `embed_image_metadata` are not part of the request schema
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
    "tile_size_x": 0,
    "tile_size_y": 0,
    "target_overlap": 0.5,
    "rel_size_x": 0.0,
    "rel_size_y": 0.0
  },

  "cache_mode": "disabled",
  "cache_option": "",
  "scm_mask": "",
  "scm_policy_dynamic": true,

  "output_format": "webm",
  "output_compression": 100
}
```

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

Image and frame fields:

| Field | Type |
| --- | --- |
| `init_image` | `string \| null` |
| `end_image` | `string \| null` |
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
  "created": 1775401200,
  "started": 1775401203,
  "completed": 1775401215,
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
  "created": 1775401200,
  "started": 1775401203,
  "completed": 1775401204,
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
  "created": 1775401200,
  "started": null,
  "completed": 1775401202,
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
