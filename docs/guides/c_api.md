# C API

Handwritten equivalents of the generated image/video programs are available in
the [DGL documentation examples](../dgl/README.md).

Use purpose-specific opaque options for model loading, generation and
postprocessing, and
opaque `sd_result_t` objects to read generated outputs. The library allocates and
frees these objects; applications must not allocate them by size or access fields.

## Loading and generation

This C example loads a checkpoint and generates an image. Check every setter:
an unsuccessful setter preserves the previous value, including its unset state.

```c
#include <stdio.h>
#include "stable-diffusion.h"

int generate(const char* model_path) {
    int rc = 1;
    sd_ctx_t* ctx = NULL;
    sd_result_t* result = NULL;
    sd_ctx_opt_t* ctx_opt = sd_ctx_opt_create();
    sd_img_gen_opt_t* image_opt = NULL;
    if (!ctx_opt) goto error;
    if (sd_opt_set_string(ctx_opt, "model_path", model_path) != SD_STATUS_OK ||
        sd_opt_set_int(ctx_opt, "n_threads", 8) != SD_STATUS_OK ||
        sd_ctx_create(ctx_opt, &ctx) != SD_STATUS_OK)
        goto error;
    sd_opt_free(ctx_opt);
    ctx_opt = NULL;

    image_opt = sd_img_gen_opt_create();
    if (!image_opt) goto error;
    if (sd_opt_set_string(image_opt, "prompt", "a small house") != SD_STATUS_OK ||
        sd_opt_set_int(image_opt, "width", 512) != SD_STATUS_OK ||
        sd_opt_set_int(image_opt, "height", 512) != SD_STATUS_OK ||
        sd_opt_set_int(image_opt, "seed", 42) != SD_STATUS_OK ||
        sd_opt_set_int(image_opt, "sample.steps", 20) != SD_STATUS_OK ||
        sd_opt_set_string(image_opt, "sample.method", "euler") != SD_STATUS_OK ||
        sd_generate_image(ctx, image_opt, &result) != SD_STATUS_OK)
        goto error;

    for (size_t i = 0; i < sd_result_get_image_count(result); ++i) {
        const sd_tensor_t* image = sd_result_get_image(result, i);
        const char* info = sd_result_get_generation_info(result, i);
        /* Read pixels with sd_tensor_data(image), and consume the UTF-8 DGL source. */
        (void) image;
        (void) info;
    }
    rc = 0;
    goto cleanup;
error:
    fprintf(stderr, "%s\n", sd_get_last_error());
cleanup:
    sd_result_free(result);
    sd_ctx_free(ctx);
    sd_opt_free(ctx_opt);
    sd_opt_free(image_opt);
    return rc;
}
```

`sd_ctx_create()` retains the model configuration. The first generation call
loads the components it needs and reports any loading errors through its return
status. Weight residency follows `eager_load` and the memory budgets. Capability
queries can probe the model family and reuse the detected type.

Use `sd_ctx_set_model_options()` on an idle context to replace its model
configuration. It copies the options and resets model detection. Later
`sd_generate_image()` and `sd_generate_video()` calls use the new configuration.
Use `sd_ctx_set_model_options_for_generation()` to probe the candidate against
`SD_RESULT_IMAGE` or `SD_RESULT_VIDEO` before applying it. A failed check
preserves the current configuration. Model setters return `SD_STATUS_BUSY`
while model probing is in progress, including from file-access callbacks.
The execution settings and component cache continue to serve generation and the
DGL context borrowed through `sd_ctx_get_dgl_ctx()`. The borrowed pointer stays
valid until `sd_ctx_free()`. Run scripts and generation calls one at a time. Set
file-access callbacks for generated programs and scripts through
`sd_ctx_set_file_access_callback()` and `sd_dgl_ctx_set_file_access_callback()`
respectively.

Use `sd_dgl_ctx_release_staging()` to release idle GPU staging copies backed by
RAM weights. Use `sd_dgl_ctx_unload_models()` to release cached model components
and reload them on demand. Both return `SD_STATUS_BUSY` while the context is
running a script or generation.

Use `sd_generate_video()` with options from `sd_vid_gen_opt_create()` to generate video. Its result
exposes frames through the same image accessors, optional audio through
`sd_result_get_audio()`, and effective frame rate through `sd_result_get_fps()`.
`sd_result_get_kind()` distinguishes image batches from videos. Video generation
metadata is sequence-wide and is returned for every valid frame index.

All option names are aliases of the same opaque `sd_opt_t` type and share
all `sd_opt_set_*()`, `sd_opt_unset()` and `sd_opt_free()` functions. The constructor
permanently binds the object's purpose. Setters reject keys for another purpose
immediately, and operation entry points reject an option object of the wrong
purpose even if empty. For example, `prompt` cannot be set on context options,
and image options cannot be passed to `sd_ctx_create()`. These checks occur at
runtime; the C typedef aliases provide descriptive names rather than distinct
compile-time types.

## Operations

Each operation has its own constructor and uses the shared setters and
`sd_opt_free()`. Creation and execution return `sd_status_t`; output pointers
are required and set to `NULL` on failure. All outputs use `sd_result_t`.

| Option constructor | Operation |
| --- | --- |
| `sd_ctx_opt_create()` | `sd_ctx_create()` |
| `sd_img_gen_opt_create()` | `sd_generate_image()` |
| `sd_vid_gen_opt_create()` | `sd_generate_video()` |
| `sd_upscaler_opt_create()` | `sd_upscaler_create()` |
| `sd_upscale_opt_create()` | `sd_upscale()` |

Image upscaling loads `model_path` with optional `n_threads`, `backend`,
`params_backend`, `conv_direct` and `tile_size` (default 128). Execution takes
an RGB `image`. Its scale is fixed by the loaded model and can be queried with
`sd_upscaler_get_scale()`; metadata records that actual scale. The former
image-upscale factor argument was unused and has been removed.

```c
sd_upscaler_opt_t* load = sd_upscaler_opt_create();
sd_upscaler_t* upscaler = NULL;
sd_upscale_opt_t* opt = NULL;
sd_result_t* result = NULL;
sd_status_t status = load ? sd_opt_set_string(load, "model_path", model_path)
                          : SD_STATUS_OUT_OF_MEMORY;
if (status == SD_STATUS_OK) status = sd_upscaler_create(load, &upscaler);
if (status == SD_STATUS_OK) {
    opt = sd_upscale_opt_create();
    status = sd_opt_set_tensor(opt, "image", image);
    if (status == SD_STATUS_OK) status = sd_upscale(upscaler, opt, &result);
}
/* Consume result on success. */
sd_result_free(result);
sd_opt_free(opt);
sd_upscaler_free(upscaler);
sd_opt_free(load);
```

ADetailer, independent video upscaling, AnimateDiff, PhotoMaker, PuLID,
IP-Adapter, custom preprocessing, SLG, timestep shifting and
automatic post-generation upscaling are explicit DGL workflows. Their former
C operation handles or parameter keys are unavailable. Ordinary ControlNet
remains supported with a configured model and a direct control image.

Contexts retain configuration snapshots and a persistent DGL execution context.
Model metadata is probed lazily and cached with file dependencies. Creation can
succeed before a missing or invalid model is discovered by a capability query
or generation call. Component weights are loaded and reused by dgl.

Reference inputs follow model defaults. Custom reference resizing, cropping,
position indexing and conditioning must be written in DGL; there is no
`ref_image_args` override string.

## String-valued modes

Sampler, scheduler, prediction, RNG, cache, control, hires and VAE option enums
are internal. Set their documented string values through `sd_opt_set_string()`.
Leave `sample.method` and `sample.scheduler` unset, or set them to `"auto"`,
to select defaults for the loaded model and sampler.

The bundled CLI and server retain these modes as strings and pass them directly
to setters. Enum parsers, enum name helpers, parameter configuration layouts and
name tables are internal and are not exported by the shared library.

Preview registration also uses strings:

```c
sd_status_t status = sd_set_preview_callback(callback, "proj", 5,
                                             true, false, user_data);
/* Disable registration when the context is idle. */
sd_set_preview_callback(NULL, "none", 1, false, false, NULL);
```

Preview modes are `none`, `proj`, `tae` and `vae`. A null or unknown mode returns
`SD_STATUS_INVALID_ARGUMENT` and preserves the previous registration. See the
public header for interval, noisy/denoised and callback lifetime semantics.

## Tensor data

All media payloads use opaque `sd_tensor_t` handles, backed by the library's
`sd::Tensor<T>`. The declarations are in `stable-diffusion.h`; no additional
public header is required. Dimensions are ordered from most contiguous to least
contiguous, matching `sd::Tensor`:

| Media | Element type | Shape |
| --- | --- | --- |
| Image | `SD_TENSOR_U8` | `[channels, width, height]` |
| Audio samples | `SD_TENSOR_F32` | `[channels, samples]` |
| Video frames | `SD_TENSOR_U8` | `[channels, width, height, frames]` |

Channels are interleaved. Audio duration is `samples / sample_rate`; sample count
and channel count come from the tensor dimensions. `sd_audio_t` adds `sample_rate`
to a sample tensor. `sd_video_t` adds FPS and an optional audio descriptor to a
video tensor. Image dimensions and channel count are entirely in the tensor.

`sd_tensor_create(type, rank, shape, data, data_size)` copies the supplied bytes.
The byte count must exactly match the element type and dimensions. General
tensors support U8, I32, I64, F32 and F64, rank up to 8, rank-zero scalars and
zero-sized dimensions. Media keys validate their required type, rank and positive
dimensions.

```c
int64_t shape[] = {3, width, height};
sd_tensor_t* image = sd_tensor_create(SD_TENSOR_U8, 3, shape,
                                      pixels, (size_t)width * height * 3);
if (image) {
    sd_status_t status = sd_opt_set_tensor(opt, "init_image", image);
    sd_tensor_free(image); /* opt retains the tensor after a successful setter. */
    (void)status;
}
```

Use `sd_opt_set_tensors()` for image lists such as `ref_images`; individual images
may have different dimensions. Use one 4D tensor for `control_frames` and video
upscaler `frames`. Use `sd_opt_set_audios()` and `sd_opt_set_videos()` for lists
whose items also carry sample-rate/FPS metadata. These setters retain all nested
tensors. Freeing the caller's tensor handles or descriptor arrays does not
invalidate configured options.

`sd_result_get_image()` returns a U8 image/frame tensor. `sd_result_get_audio()`
returns sample-rate metadata and a F32 sample tensor. `sd_result_get_video()`
returns FPS, optional audio and a contiguous 4D tensor, assembled and cached on
first access. Reading individual video frames avoids that assembly and copy.

Tensor data is read-only. `sd_tensor_retain()` extends the lifetime of any tensor,
including result and preview tensors, independently of its original owner;
release each owned reference with `sd_tensor_free()`. Retain/free accept NULL and
preserve the last diagnostic. Preview callbacks receive an array of borrowed
image tensors. Prompt-image callbacks return a borrowed image tensor that stays
valid until the next callback. `sd_preprocess_canny()` returns a new image tensor
and leaves its input unchanged.

## Values, validation and ownership

- Unset keys retain existing model-loading and generation defaults.
  `sd_opt_unset()` restores this state and succeeds for an applicable known absent key.
  Zero, false, empty strings and empty arrays remain explicit values, subject to
  that key's validation. `NULL` strings are invalid; use `unset` for defaults.
- Keys and enum values are case-sensitive. Unknown keys return
  `SD_STATUS_UNKNOWN_KEY`; incorrect setter types return `SD_STATUS_TYPE_MISMATCH`.
  Numeric overflow/range errors and invalid enum names are rejected by setters.
  No automatic conversion occurs between integer, floating-point and boolean values.
- Setters and unset reject keys for another purpose with `SD_STATUS_UNSUPPORTED`,
  even when set to a default-looking value. Loading/generation checks the option
  object purpose and parameter combinations. Model-dependent features are checked at generation:
  for example UCache requires UNet and EasyCache/CacheDIT require DiT. Known
  high-noise sampling settings and `moe_boundary` are ignored without a high-noise
  stage; `vace_strength` is ignored by non-VACE models. Actual control-frame inputs
  still require VACE. Existing size/frame alignment and model-dependent automatic
  defaults are retained.
- Scalar strings, numeric arrays, LoRA paths and embedding names/paths are copied.
  Tensor arrays and audio/video metadata are copied, and tensor references are
  retained. Empty arrays accept a `NULL` pointer and count 0.
- Successful context creation owns the configuration it retains. Free or reuse
  the loading params immediately afterwards. Generation params may be freed or
  changed after the generation call returns. Do not mutate/free an option object
  while another operation is using it.
- Results own their media and metadata independently of the context and params.
  Read-only views remain valid until `sd_result_free()`. Do not free or modify
  buffers through a view. Image results have no audio and return FPS 0. Invalid
  indices return `NULL` with a diagnostic. All new free functions accept `NULL`.
- Output pointers are required and set to `NULL` on failure. Partial batch
  cancellation can return successful completed outputs; full cancellation returns
  `SD_STATUS_CANCELLED` without a result. Other failures distinguish invalid
  arguments, unsupported features, memory exhaustion, loading and generation.
- `sd_get_last_error()` returns a thread-local diagnostic. Status/accessor calls
  clear it on success; free functions preserve it. Read it before another API call.

Option and tensor objects have no public field-layout contract. Future settings
normally add a key while retaining existing key names, types and meanings. To
keep an output after releasing the result, retain its tensor first; release the
retained handle with `sd_tensor_free()` when finished. Result metadata and media
metadata descriptors remain borrowed from the result and should be copied when
needed independently.

## Keys

Types below correspond to `sd_opt_set_<type>()`. `context`, `image` and `video`
refer to model loading and image/video generation. Other operation labels match
the corresponding option constructor, `upscaler` or `upscale`.
Array lengths are supplied to setters; there are no separately writable count keys.

Top-level keys retain their existing parameter meanings. Nested names map as follows:

| Key prefix | Configuration |
| --- | --- |
| `sample.*` | Base sampling settings; `method`, `steps`, `sigmas`, `args` select the sampler, step count, custom schedule and extra arguments. |
| `sample.guidance.*` | Text/image CFG and distilled guidance settings. |
| `cache.*` | [Inference cache](caching.md) settings. |
| `vae_tiling.*` | Spatial/temporal VAE tiling settings. |

The library supplies inference defaults. Omitted server inputs do not override
them. The CLI retains its historical seed default of 42; `--seed -1` requests
a random seed. Other omitted CLI inference options use the library defaults.
Model recipes may supply presets; explicit startup options
and request fields take priority over recipe values. A supplied value equal to
a default is still explicit and undergoes model applicability checks.

Generation defaults include 512×512, a random seed (`-1`), 20 sampling steps,
model-selected sampler/scheduler, text CFG 7, distilled guidance 3.5, strength
0.75, and disabled caching/tiling. Image batches default to 1; videos default
to 6 requested frames at 16 FPS, subject to model alignment. Omitted eta, flow shift
and image CFG retain automatic defaults; do not pass infinity to request defaults.
Use `unset`. Wan dual experts share one `sample.*` configuration and one
sigma sequence. `high_noise_sample.steps` defaults to -1: `sample.steps` is the
total, split using `moe_boundary` (default 0.875). With a nonnegative high-noise
count, the total is `sample.steps + high_noise_sample.steps`; the high-noise
expert runs that prefix and the low-noise expert runs the remaining steps.
Zero selects only the low-noise expert. Without a high-noise model this setting
is ignored. Custom sigmas fix the total; a positive high-noise count is adjusted
to retain the requested low-noise count when possible. Independent high-noise
CFG and sampler overrides still require dgl.

Image and LTX/MiniMax-H3 video hires passes use the same DGL execution path.
Set `hires.enabled=true`. The default is latent bilinear upscaling by 2,
followed by refinement with the original model and prompt conditions.
`hires.steps=0` inherits the base step count; denoising strength defaults to
0.7. The automatic schedule uses `floor(steps / strength)` intervals and keeps
the last `steps + 1` entries. If strength is below 1 and rounding would leave
the full schedule in place, one extra interval is generated so the first sigma
is trimmed. Refinement still runs for the requested step count. `hires.sigmas`
supplies the complete refinement schedule and is used directly.

`hires.target_width` and `hires.target_height` override the scale. Supplying
only one makes a square target. Output dimensions follow model alignment.
Image upscalers are `latent`, `latent_nearest`, `latent_nearest_exact`,
`latent_antialiased`, `latent_bicubic`, `latent_bicubic_antialiased`, `lanczos`,
`nearest`, `model`, or `none`. `model` requires `hires.model_path`; images use
an image upscaler and then VAE re-encoding. Videos accept latent interpolation
or a compatible latent upscaler. LTX learned upscalers determine their output
size and frame count from the checkpoint, overriding target dimensions;
audio duration and keyframe conditioning follow that output. H3 learned
upscalers use the requested target size. Other video families require an
explicit DGL refinement workflow.

Sampler/scheduler values follow the [sampling guide](sampling.md); `"auto"`
requests automatic selection. RNG values are `std_default`, `cuda`, `cpu`, `auto`.
Prediction modes are `auto`, `eps`, `v`, `edm_v`, `flow`, `flux_flow`,
`sefi_flow`, `minit2i_flow`, `sensenova_u1_flow`, `flux3_flow`, `h3_av_flow`,
`ming_image_flow`. Weight types use the names returned by `sd_type_name()`,
plus `auto`. VAE formats are `auto`, `flux`, `sd3`,
`flux2`, `wan`. LoRA apply modes are `auto`, `immediately`, `at_runtime`.
Cache modes are `disabled`, `easycache`, `ucache`, `dbcache`, `taylorseer`,
`cache_dit`, `spectrum`. Control types are `none`, `openpose`, `depth`, `scribble`, `canny`, `normal`,
`segment`, `tile`, `repaint`, with the existing aliases such as `hed` and `lineart`.

Integers must fit the underlying field (32 bits except `seed`, which is signed
64-bit). Dimensions, batch size, video frames, FPS and base steps are positive.
Floating-point settings must be finite and fit float32; strength, denoising
strength, percent/range settings and MOE boundary lie in [0, 1]. Custom sigma
arrays are empty or strictly decreasing nonnegative finite values ending in zero,
with at least two entries.

For SDXL's bundled VAE (no `vae_path`), generation emits
`conv_scale = 0.03125` in the generated DGL to avoid FP16 overflow. Set
`force_sdxl_vae_conv_scale = true` to apply the same scaling to an explicit
`vae_path`. TAE decoding retains its own scaling.

| Key | Setter type | Operation |
| --- | --- | --- |
| `attn_scale` | `float` | context |
| `audio_encoder_path` | `string` | context |
| `audio_vae_path` | `string` | context |
| `auto_fit` | `bool` | context |
| `backend` | `string` | context, upscaler |
| `batch_count` | `int` | image |
| `batched_cfg` | `bool` | context |
| `cache.bn_compute_blocks` | `int` | image, video |
| `cache.end_percent` | `float` | image, video |
| `cache.error_decay_rate` | `float` | image, video |
| `cache.fn_compute_blocks` | `int` | image, video |
| `cache.max_cached_steps` | `int` | image, video. Negative values mean no limit. |
| `cache.max_continuous_cached_steps` | `int` | image, video. Negative values mean no limit. |
| `cache.max_warmup_steps` | `int` | image, video |
| `cache.mode` | `string` | image, video |
| `cache.reset_error_on_compute` | `bool` | image, video |
| `cache.residual_diff_threshold` | `float` | image, video |
| `cache.reuse_threshold` | `float` | image, video |
| `cache.scm_mask` | `string` | image, video |
| `cache.scm_policy_dynamic` | `bool` | image, video |
| `cache.spectrum_flex_window` | `float` | image, video |
| `cache.spectrum_lam` | `float` | image, video |
| `cache.spectrum_m` | `int` | image, video |
| `cache.spectrum_stop_percent` | `float` | image, video |
| `cache.spectrum_w` | `float` | image, video |
| `cache.spectrum_warmup_steps` | `int` | image, video |
| `cache.spectrum_window_size` | `int` | image, video |
| `cache.start_percent` | `float` | image, video |
| `cache.taylorseer_n_derivatives` | `int` | image, video |
| `cache.taylorseer_skip_interval` | `int` | image, video |
| `cache.use_relative_threshold` | `bool` | image, video |
| `circular_x` | `bool` | image, video |
| `circular_y` | `bool` | image, video |
| `clip_g_path` | `string` | context |
| `clip_l_path` | `string` | context |
| `clip_skip` | `int` | image, video. Nonpositive values use the model default. |
| `clip_vision_path` | `string` | context |
| `conditioning_cache_size` | `int` | context |
| `control_frames` | `tensor` | video |
| `control_image` | `tensor` | image |
| `control_net_path` | `string` | context |
| `control_strength` | `float` | image |
| `control_type` | `string` | image |
| `conv_direct` | `bool` | upscaler |
| `conv_direct_mode` | `string` | context |
| `diffusion_conv_direct` | `bool` | context |
| `diffusion_flash_attn` | `bool` | context |
| `diffusion_model_path` | `string` | context |
| `disable_prefetch` | `bool` | context |
| `disable_segmented_compute` | `bool` | context |
| `eager_load` | `bool` | context |
| `embeddings` | `embeddings` | context |
| `embeddings_connectors_path` | `string` | context |
| `enable_mmap` | `bool` | context |
| `end_image` | `tensor` | video |
| `flash_attn` | `bool` | context |
| `flash_attn_mode` | `string` | context |
| `force_sdxl_vae_conv_scale` | `bool` | context |
| `fps` | `int` | video |
| `height` | `int` | image, video |
| `high_noise_diffusion_model_path` | `string` | context |
| `image` | `tensor` | upscale |
| `init_image` | `tensor` | image, video |
| `linear_scale` | `float` | context |
| `llm_path` | `string` | context |
| `llm_vision_path` | `string` | context |
| `lora_apply_mode` | `string` | context |
| `loras` | `loras` | image, video |
| `mask_image` | `tensor` | image |
| `max_ram` | `string` | context |
| `max_vram` | `string` | context |
| `model_args` | `string` | context |
| `model_path` | `string` | context, upscaler |
| `moe_boundary` | `float` | video |
| `high_noise_sample.steps` | `int` | video; -1 or nonnegative |
| `hires.enabled` | `bool` | image, video |
| `hires.upscaler`, `hires.model_path` | `string` | image, video |
| `hires.scale` | `float` | image, video; positive, default 2 |
| `hires.target_width`, `hires.target_height` | `int` | image, video; 0 selects scale |
| `hires.steps` | `int` | image, video; 0 inherits base steps |
| `hires.denoising_strength` | `float` | image, video; 0.0001..1 |
| `hires.upscale_tile_size` | `int` | image model upscaling; default 128 |
| `hires.sigmas` | `floats` | image, video; optional refinement schedule |
| `n_threads` | `int` | context, upscaler |
| `negative_prompt` | `string` | image, video |
| `params_backend` | `string` | context, upscaler |
| `prediction` | `string` | context |
| `prompt` | `string` | image, video |
| `qwen_image_layers` | `int` | image |
| `ref_audios` | `audios` | video |
| `ref_images` | `tensors` | image, video |
| `ref_videos` | `videos` | video |
| `rng_type` | `string` | context |
| `rpc_servers` | `string` | context |
| `sage_attn` | `bool` | context |
| `sample.args` | `string` | image, video |
| `sample.eta` | `float` | image, video |
| `sample.flow_shift` | `float` | image, video |
| `sample.guidance.distilled_guidance` | `float` | image, video |
| `sample.guidance.img_cfg` | `float` | image, video |
| `sample.guidance.txt_cfg` | `float` | image, video |
| `sample.method` | `string` | image, video |
| `sample.scheduler` | `string` | image, video |
| `sample.sigmas` | `floats` | image, video |
| `sample.steps` | `int` | image, video |
| `sampler_rng_type` | `string` | context |
| `seed` | `int` | image, video |
| `split_mode` | `string` | context |
| `strength` | `float` | image, video |
| `t5xxl_path` | `string` | context |
| `tae_preview_only` | `bool` | context |
| `taesd_path` | `string` | context |
| `tensor_type_rules` | `string` | context |
| `tile_size` | `int` | upscaler |
| `tokenizer` | `string` | context |
| `uncond_diffusion_model_path` | `string` | context |
| `vace_strength` | `float` | video |
| `vae_conv_direct` | `bool` | context |
| `vae_format` | `string` | context |
| `vae_path` | `string` | context |
| `vae_tiling.enabled` | `bool` | image, video |
| `vae_tiling.extra_tiling_args` | `string` | image, video |
| `vae_tiling.rel_size_h` | `float` | image, video |
| `vae_tiling.rel_size_w` | `float` | image, video |
| `vae_tiling.target_overlap` | `float` | image, video |
| `vae_tiling.temporal_tiling` | `bool` | image, video |
| `vae_tiling.tile_size_h` | `int` | image, video |
| `vae_tiling.tile_size_w` | `int` | image, video |
| `video_frames` | `int` | video |
| `width` | `int` | image, video |
| `wtype` | `string` | context |

## Migration

This is a breaking C API change. The former public generation parameter layouts,
their initialization/formatting helpers, `new_sd_ctx()`, `free_sd_ctx()`,
`generate_image()`, `generate_video()` and public result containers/destructors
have been removed. Applications and bindings must migrate and rebuild; no
compatibility layer is provided.

Create context/image/video options with the corresponding constructor, populate
them through the shared typed setters, and use `sd_ctx_create()`,
`sd_generate_image()` and `sd_generate_video()`. Read or retain outputs through
`sd_result_get_*()` and `sd_tensor_retain()` and release the result with
`sd_result_free()`. Standalone image upscaling retains its purpose-specific
options and actual-scale queries. ADetailer and standalone video-upscale C
handles have been removed; use explicit DGL workflows.

Generation results now contain the exact executed DGL source. They no longer
contain generation JSON. See [generation results and metadata](generation_metadata.md).
Context creation is lazy, and model errors can first surface in a capability
query or generation call. Context options are copied, while media handles are
retained. Cancellation, progress and preview retain their public contracts.

### DGL region results

DGL regions are ordinary dictionaries with integer `width` and `height` fields
and an I64 `boxes` tensor shaped `[4,N]`. The dedicated region value kind has
been removed. `SD_DGL_VALUE_OPAQUE` remains `11` and `SD_DGL_VALUE_VOID` remains
`12`.

Returning a region dict directly now gives a `SD_DGL_VALUE_DICT` root and flat
outputs `boxes`, `width` and `height`. Returning `{regions=r}` gives the flat
names `regions.boxes`, `regions.width` and `regions.height`. Replace root-region
exports such as `--tensor result boxes.npy` with `--tensor boxes boxes.npy`.
Coordinate tensors now use I64; cast old I32 files explicitly with
`tensor::cast(boxes, "i64")` before passing them to `regions::from_boxes`.

Detection results are also ordinary dicts: F32 `xyxy` `[4,N]`, F32 `scores`
`[N]`, and I64 `class_ids` `[N]`, with at most 4,096 items. Empty results preserve
all three fields and ranks. `detections::detect`, `detections::filter`,
`detections::box_masks` and `detections::to_regions` replace the previous global
detection helpers. Use `detections::detect` followed by `detections::to_regions`
to replace the old direct-to-regions `detect` shortcut; detection executes once.

Use `regions::from_boxes`, `regions::crop`, `regions::paste` and
`regions::plan_mask_crops` for rectangle and mask tasks. The previous global
region function names are no longer registered. See
[DGL rectangular regions](../dgl.md#rectangular-regions) for validation and
composition rules.
