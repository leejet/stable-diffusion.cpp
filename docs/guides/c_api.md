# C API

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
        /* Read pixels with sd_tensor_data(image), and consume the UTF-8 JSON info. */
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
| `sd_vid_upscaler_opt_create()` | `sd_vid_upscaler_create()` |
| `sd_vid_upscale_opt_create()` | `sd_upscale_video()` |
| `sd_adetailer_opt_create()` | `sd_adetailer_create()` |
| `sd_adetail_opt_create()` | `sd_adetail()` |

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

Video-upscaler loading requires `diffusion_model_path`, `vae_path` and
`upscale_factor`; backend/thread/attention settings are optional. The scale must
match both checkpoints (RelayVSR uses 2 or 4). Execution takes RGB `frames` with
equal dimensions. Defaults are FPS 24, seed 42, RelayVSR `window_size` 16, and
SeedVR2 `temporal_chunk_size` 5. Free the model with `sd_vid_upscaler_free()`.

ADetailer loading takes the detector `model_path` and optional backend/thread
settings. Call `sd_adetail(adetailer, diffusion_ctx, opt, &result)` with an
`sd_adetail_opt_t`. Set `image` for the source image, and reuse the applicable
image-generation keys for inpainting: `prompt`, `negative_prompt`, dimensions,
strength, seed, sampling, LoRAs, cache, VAE tiling and circular axes. Inpainting
defaults to 512×512 and strength 0.4. `adetail.prompt` and
`adetail.negative_prompt` supply region prompt overrides; `adetail.args` accepts
the existing detector/mask/inpaint argument string documented by the CLI
`--extra-ad-args` option. Reference/control images, hires and batching are not
ADetailer options. Free the detector with `sd_adetailer_free()`.

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
the corresponding option constructor, such as `vid_upscale` for
`sd_vid_upscale_opt_create()`.
Array lengths are supplied to setters; there are no separately writable count keys.

Top-level keys retain their existing parameter meanings. Nested names map as follows:

| Key prefix | Configuration |
| --- | --- |
| `sample.*` | Base sampling settings; `method`, `steps`, `sigmas`, `args` select the sampler, step count, custom schedule and extra arguments. |
| `high_noise_sample.*` | High-noise sampling settings for two-stage video models. |
| `sample.guidance.*` | Text/image CFG, distilled guidance and SLG settings. |
| `cache.*` | [Inference cache](caching.md) settings. |
| `hires.*` | High-resolution second pass and upscaling settings. |
| `vae_tiling.*` | Spatial/temporal VAE tiling settings. |
| `photomaker.*` | PhotoMaker identity images/embedding and style strength. |
| `pulid.*` | PuLID identity embedding path and weight. |
| `image_preprocess.rules` | [Image preprocessing](image_preprocessing.md) rule string. |

The library supplies inference defaults. CLI/server inputs that are omitted do
not override them. Model recipes may supply presets; explicit startup options
and request fields take priority over recipe values. A supplied value equal to
a default is still explicit and undergoes model applicability checks.

Generation defaults include 512×512, a random seed (`-1`), 20 sampling steps,
model-selected sampler/scheduler, text CFG 7, distilled guidance 3.5, strength
0.75, and disabled caching/hires/tiling. Image batches default to 1; videos default
to 6 requested frames at 16 FPS, subject to model alignment. Omitted eta, flow shift
and image CFG retain automatic defaults; do not pass infinity to request defaults.
Use `unset`. High-noise steps default to `-1` (automatic allocation). The high-noise pass
shares the base scheduler, sigmas, flow shift and timestep shifting; set those
through `sample.*`. Separate high-noise keys for these settings are not supported.

Sampler/scheduler values follow the [sampling guide](sampling.md); `"auto"`
requests automatic selection. RNG values are `std_default`, `cuda`, `cpu`, `auto`.
Prediction modes are `auto`, `eps`, `v`, `edm_v`, `sd3_flow`, `flux_flow`,
`sefi_flow`, `minit2i_flow`, `sensenova_u1_flow`, `flux3_flow`, `h3_av_flow`,
`ming_image_flow`. Weight types use the names returned by `sd_type_name()`,
plus `auto`. VAE formats are `auto`, `flux`, `sd3`,
`flux2`, `wan`. LoRA apply modes are `auto`, `immediately`, `at_runtime`.
Cache modes are `disabled`, `easycache`, `ucache`, `dbcache`, `taylorseer`,
`cache_dit`, `spectrum`. Control types are `none`, `openpose`, `depth`, `scribble`, `canny`, `normal`,
`segment`, `tile`, `repaint`, with the existing aliases such as `hed` and `lineart`. Hires upscalers are `none`, `latent`,
`latent_nearest`, `latent_nearest_exact`, `latent_antialiased`, `latent_bicubic`,
`latent_bicubic_antialiased`, `lanczos`, `nearest`, `model`.

Integers must fit the underlying field (32 bits except `seed`, which is signed
64-bit). Dimensions, batch size, video frames, FPS and base steps are positive.
Floating-point settings must be finite and fit float32; strength, denoising
strength, percent/range settings and MOE boundary lie in [0, 1]. Custom sigma
arrays are empty or strictly decreasing nonnegative finite values ending in zero,
with at least two entries. SLG layer indices are nonnegative integers.

| Key | Setter type | Operation |
| --- | --- | --- |
| `adetail.args` | `string` | adetail |
| `adetail.negative_prompt` | `string` | adetail |
| `adetail.prompt` | `string` | adetail |
| `attn_scale` | `float` | context |
| `audio_encoder_path` | `string` | context |
| `audio_vae_path` | `string` | context |
| `auto_fit` | `bool` | context |
| `backend` | `string` | context, upscaler, vid_upscaler, adetailer |
| `batch_count` | `int` | image |
| `batched_cfg` | `bool` | context |
| `cache.bn_compute_blocks` | `int` | image, video, adetail |
| `cache.end_percent` | `float` | image, video, adetail |
| `cache.error_decay_rate` | `float` | image, video, adetail |
| `cache.fn_compute_blocks` | `int` | image, video, adetail |
| `cache.max_cached_steps` | `int` | image, video, adetail |
| `cache.max_continuous_cached_steps` | `int` | image, video, adetail |
| `cache.max_warmup_steps` | `int` | image, video, adetail |
| `cache.mode` | `string` | image, video, adetail |
| `cache.reset_error_on_compute` | `bool` | image, video, adetail |
| `cache.residual_diff_threshold` | `float` | image, video, adetail |
| `cache.reuse_threshold` | `float` | image, video, adetail |
| `cache.scm_mask` | `string` | image, video, adetail |
| `cache.scm_policy_dynamic` | `bool` | image, video, adetail |
| `cache.spectrum_flex_window` | `float` | image, video, adetail |
| `cache.spectrum_lam` | `float` | image, video, adetail |
| `cache.spectrum_m` | `int` | image, video, adetail |
| `cache.spectrum_stop_percent` | `float` | image, video, adetail |
| `cache.spectrum_w` | `float` | image, video, adetail |
| `cache.spectrum_warmup_steps` | `int` | image, video, adetail |
| `cache.spectrum_window_size` | `int` | image, video, adetail |
| `cache.start_percent` | `float` | image, video, adetail |
| `cache.taylorseer_n_derivatives` | `int` | image, video, adetail |
| `cache.taylorseer_skip_interval` | `int` | image, video, adetail |
| `cache.use_relative_threshold` | `bool` | image, video, adetail |
| `circular_x` | `bool` | image, video, adetail |
| `circular_y` | `bool` | image, video, adetail |
| `clip_g_path` | `string` | context |
| `clip_l_path` | `string` | context |
| `clip_skip` | `int` | image, video, adetail |
| `clip_vision_path` | `string` | context |
| `conditioning_cache_size` | `int` | context |
| `control_frames` | `tensor` | video |
| `control_image` | `tensor` | image |
| `control_net_path` | `string` | context |
| `control_strength` | `float` | image |
| `control_type` | `string` | image |
| `conv_direct` | `bool` | upscaler |
| `conv_direct_mode` | `string` | context, vid_upscaler |
| `diffusion_conv_direct` | `bool` | context |
| `diffusion_flash_attn` | `bool` | context |
| `diffusion_model_path` | `string` | context, vid_upscaler |
| `disable_prefetch` | `bool` | context |
| `disable_segmented_compute` | `bool` | context |
| `eager_load` | `bool` | context |
| `embeddings` | `embeddings` | context |
| `embeddings_connectors_path` | `string` | context |
| `enable_mmap` | `bool` | context |
| `end_image` | `tensor` | video |
| `flash_attn` | `bool` | context |
| `flash_attn_mode` | `string` | context, vid_upscaler |
| `force_sdxl_vae_conv_scale` | `bool` | context |
| `fps` | `int` | video, vid_upscale |
| `frames` | `tensor` | vid_upscale |
| `height` | `int` | image, video, adetail |
| `high_noise_diffusion_model_path` | `string` | context |
| `high_noise_sample.args` | `string` | video |
| `high_noise_sample.eta` | `float` | video |
| `high_noise_sample.guidance.distilled_guidance` | `float` | video |
| `high_noise_sample.guidance.img_cfg` | `float` | video |
| `high_noise_sample.guidance.slg.layer_end` | `float` | video |
| `high_noise_sample.guidance.slg.layer_start` | `float` | video |
| `high_noise_sample.guidance.slg.layers` | `ints` | video |
| `high_noise_sample.guidance.slg.scale` | `float` | video |
| `high_noise_sample.guidance.txt_cfg` | `float` | video |
| `high_noise_sample.method` | `string` | video |
| `high_noise_sample.steps` | `int` | video |
| `hires.denoising_strength` | `float` | image, video |
| `hires.enabled` | `bool` | image, video |
| `hires.model_path` | `string` | image, video |
| `hires.scale` | `float` | image, video |
| `hires.sigmas` | `floats` | image, video |
| `hires.steps` | `int` | image, video |
| `hires.target_height` | `int` | image, video |
| `hires.target_width` | `int` | image, video |
| `hires.upscale_tile_size` | `int` | image, video |
| `hires.upscaler` | `string` | image, video |
| `image` | `tensor` | upscale, adetail |
| `image_preprocess.rules` | `string` | image, video |
| `init_image` | `tensor` | image, video |
| `ip_adapter_image` | `tensor` | image |
| `ip_adapter_path` | `string` | context |
| `ip_adapter_strength` | `float` | image |
| `linear_scale` | `float` | context |
| `llm_path` | `string` | context |
| `llm_vision_path` | `string` | context |
| `lora_apply_mode` | `string` | context |
| `loras` | `loras` | image, video, adetail |
| `mask_image` | `tensor` | image |
| `max_vram` | `string` | context |
| `model_args` | `string` | context |
| `model_path` | `string` | context, upscaler, adetailer |
| `moe_boundary` | `float` | video |
| `motion_module_path` | `string` | context |
| `n_threads` | `int` | context, upscaler, vid_upscaler, adetailer |
| `negative_prompt` | `string` | image, video, adetail |
| `params_backend` | `string` | context, upscaler, vid_upscaler, adetailer |
| `photo_maker_path` | `string` | context |
| `photomaker.id_embed_path` | `string` | image |
| `photomaker.id_images` | `tensors` | image |
| `photomaker.style_strength` | `float` | image |
| `prediction` | `string` | context |
| `prompt` | `string` | image, video, adetail |
| `pulid.id_embedding_path` | `string` | image |
| `pulid.id_weight` | `float` | image |
| `pulid_weights_path` | `string` | context |
| `qwen_image_layers` | `int` | image |
| `ref_audios` | `audios` | video |
| `ref_image_args` | `string` | image, video |
| `ref_images` | `tensors` | image, video |
| `ref_videos` | `videos` | video |
| `rng_type` | `string` | context |
| `rpc_servers` | `string` | context |
| `sage_attn` | `bool` | context |
| `sample.args` | `string` | image, video, adetail |
| `sample.eta` | `float` | image, video, adetail |
| `sample.flow_shift` | `float` | image, video, adetail |
| `sample.guidance.distilled_guidance` | `float` | image, video, adetail |
| `sample.guidance.img_cfg` | `float` | image, video, adetail |
| `sample.guidance.slg.layer_end` | `float` | image, video, adetail |
| `sample.guidance.slg.layer_start` | `float` | image, video, adetail |
| `sample.guidance.slg.layers` | `ints` | image, video, adetail |
| `sample.guidance.slg.scale` | `float` | image, video, adetail |
| `sample.guidance.txt_cfg` | `float` | image, video, adetail |
| `sample.method` | `string` | image, video, adetail |
| `sample.scheduler` | `string` | image, video, adetail |
| `sample.shifted_timestep` | `int` | image, video, adetail |
| `sample.sigmas` | `floats` | image, video, adetail |
| `sample.steps` | `int` | image, video, adetail |
| `sampler_rng_type` | `string` | context |
| `seed` | `int` | image, video, vid_upscale, adetail |
| `split_mode` | `string` | context |
| `strength` | `float` | image, video, adetail |
| `t5xxl_path` | `string` | context |
| `tae_preview_only` | `bool` | context |
| `taesd_path` | `string` | context |
| `temporal_chunk_size` | `int` | vid_upscale |
| `tensor_type_rules` | `string` | context |
| `tile_size` | `int` | upscaler |
| `tokenizer` | `string` | context |
| `uncond_diffusion_model_path` | `string` | context |
| `upscale_factor` | `int` | vid_upscaler |
| `vace_strength` | `float` | video |
| `vae_conv_direct` | `bool` | context |
| `vae_format` | `string` | context |
| `vae_path` | `string` | context, vid_upscaler |
| `vae_tiling.enabled` | `bool` | image, video, adetail |
| `vae_tiling.extra_tiling_args` | `string` | image, video, adetail |
| `vae_tiling.rel_size_h` | `float` | image, video, adetail |
| `vae_tiling.rel_size_w` | `float` | image, video, adetail |
| `vae_tiling.target_overlap` | `float` | image, video, adetail |
| `vae_tiling.temporal_tiling` | `bool` | image, video, adetail |
| `vae_tiling.tile_size_h` | `int` | image, video, adetail |
| `vae_tiling.tile_size_w` | `int` | image, video, adetail |
| `video_frames` | `int` | video |
| `width` | `int` | image, video, adetail |
| `window_size` | `int` | vid_upscale |
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
`sd_result_free()`. ADetailer and image/video upscalers use purpose-specific opt constructors and
return the same opaque result type. Their former public parameter structures,
long positional signatures and bool-returning entry points have been removed.
Device selection and cancellation controls retain their interfaces. Preview
registration now uses a mode string and returns a status.

Native generation JSON retains its existing schema and names, including
`sample_params.sample_method`; translate these to the corresponding option keys
(`sample.method`, etc.) when replaying metadata. See
[generation results and metadata](generation_metadata.md).
