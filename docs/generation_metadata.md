# Generation results and metadata

`generate_image()`, `generate_video()`, `adetail_image()` and `upscale()` return
owned results containing the media and its generation record. This is a breaking
C API change: applications and bindings must update their calls and rebuild.

```c
sd_img_gen_params_t params;
sd_img_gen_params_init(&params);
params.prompt = "a small house";

sd_image_generation_result_t* result = NULL;
if (generate_image(ctx, &params, &result)) {
    for (int i = 0; i < result->count; ++i) {
        const sd_image_result_t* item = &result->items[i];
        /* Consume item->image and item->generation_info together. */
    }
    free_sd_image_generation_result(result);
}
```

`result_out` is required and is set to `NULL` on failure. A successful result
owns its pixels and UTF-8 JSON strings. The video result additionally owns its
audio and contains the effective `fps` and `frame_count`. Results remain valid
after freeing the generation context. Use the matching library result destructor;
do not separately free fields unless transferring ownership and clearing them.
Both result destructors accept `NULL`.

## Reproduction parameters

The `sdcpp.generation/v1` JSON is a snapshot of generation parameters for the
same loaded context and the same input images, masks, reference media and other
assets. It is created after `GenerationRequest` resolves the request, before
preprocessing or sampling. A failed call returns no result or metadata.

Fields follow the C generation parameter structs: `prompt`, `negative_prompt`,
`width`, `height`, `seed`, `sample_params`, `hires`, `cache`,
`vae_tiling_params`, `image_preprocess`, and the image/video-specific options.
`kind` distinguishes image and video parameters. There is no inferred `task`.
Enums use their API string names. Seeds are decimal strings to preserve their
full integer range. Omitted optional fields retain the parameter initializer's
defaults, including automatic values represented by infinity in the C API.

Disabled optional features are omitted entirely: `cache`, `hires`, VAE tiling,
zero/nonpositive `shifted_timestep`, inactive SLG, false circular axes, and empty
preprocessing/reference arguments. SLG layers remain present when `slg_uncond`
is enabled even with zero SLG scale. Enabled caches store their mode's parameters;
TaylorSeer retains the shared DBCache thresholds and limits used by the runtime.
Image CFG at its neutral value of 1 is omitted and resolves back to 1 by default.

Image strength, ControlNet/IP-Adapter strength and PhotoMaker settings are stored
when their corresponding input images are supplied. PuLID requires an embedding
path. Layer count is stored for layered generation; video VACE and high-noise
settings require those models. High-noise settings omit scheduler, custom sigmas,
flow shift and timestep shifting because that pass shares the base configuration.
Explicit nonempty argument strings are retained without rewriting their contents.
Core sampling values such as seed, sampler, scheduler, steps and guidance remain
explicit even when equal to their defaults. ADetailer always retains its inpaint
strength because its input crops are created later.
Model-required VAE tiling is resolved before serialization, so forced tiling
(such as MiniMax-H3) remains present even if the caller did not enable it.

The request fixes the seed, dimensions, clip skip, guidance, sampler, scheduler
and eta before taking the snapshot. `custom_sigmas` is stored only when supplied
as an input. Automatically computed schedules, preprocessing transforms,
conditioning/cache decisions, detector regions and output statistics are not
stored: they are recomputed from the same parameters and inputs.

Each image records the seed used to sample its latent. `batch_count` is omitted;
replay image parameters with `batch_count = 1`. Layers or frames decoded from
the same latent share its seed, so layered generation reproduces that group of
layers. Video frames share the video request. Partial batch cancellation returns
metadata only for completed outputs. Full cancellation returns no result.

The current image pipeline prepares input latents once using the starting seed
and shares them across a batch. With a stochastic VAE encoder, replaying a later
image alone using its sampling seed can change those input latents. Per-image
seeds therefore do not guarantee independent reproduction for these paths.

To replay a record, initialize the matching C parameter struct, populate its
fields from the JSON, restore the same input buffers, and call generation with
the same context. Keep custom-sigma/SLG arrays and string storage alive for the
call. Paths supplied as generation parameters (LoRAs, hires models and identity
embeddings) are retained. `models` identifies successfully loaded context model
files by component (`model`, `diffusion_model`, `clip_l`, `vae`, etc.), using
filenames with extensions and omitting directories. Optional components disabled
for the request and preview-only TAE are omitted. WebUI text includes `Model`
(the diffusion model when supplied separately, otherwise the checkpoint, without
the extension) and `VAE` for a separate active VAE. These names are identifiers,
not hashes or sufficient configuration to rebuild a context; replay still requires
the same model files and context settings. The JSON does not embed media buffers
or model weights.

When `--lora-model-dir` is configured, CLI/server output records express LoRA
paths relative to that directory, preserving subdirectories and file extensions
and using `/` separators. This applies to file metadata, API responses and
ADetailer records. Resolve these paths against the same LoRA directory before
passing them back to the C API. Core results retain the supplied loading paths.
Files outside the directory retain `..` components; paths on another volume,
or records without a configured LoRA directory, retain their original paths.

## Postprocessing

Upscale records `upscale.upscale_factor`. ADetailer records `adetailer` and
`inpaint_params`, with a concrete starting seed chosen before detection. These
operations require the same upscaler/detector context and source image on replay.
They do not store detection results or per-region generation traces.

The CLI/server append operation parameter snapshots to `postprocess` in call
order. The source generation parameters remain unchanged. Frame postprocessing
uses the same operation parameters for each video frame. Standalone
postprocessing does not invent source diffusion parameters for an input image.

## File metadata

Image encoders accept the native record and derive WebUI infotext from it.
Images use the existing metadata locations and combined text layout:
`<WebUI infotext>, SDCPP: <native JSON>`. Standalone postprocessing without
source diffusion parameters stores only native JSON. Video metadata stores
only native JSON.

| Container | Metadata location |
| --- | --- |
| PNG | `tEXt`, keyword `parameters` |
| JPEG | COM segment, `parameters` label followed by a NUL byte |
| WebP image / animated WebP | XMP property `sdcpp:parameters` |
| AVI | RIFF INFO `ICMT` |
| WebM | `COMMENT` tag |

Inspect images or videos with `sd-cli -M metadata --image <file>`; the same
command supports all containers above, with `--metadata-format json` for a
structured report. Video metadata inspection does not require a model or video
decoder.
Matroska containers are also supported, including `.webm` outputs whose audio
codec causes the muxer to select the Matroska document type.

The XMP namespace is `https://github.com/leejet/stable-diffusion.cpp/ns/1.0/`.
PNG and JPEG retain the existing UTF-8 text payload convention. Memory encoding
and file saving use the same implementation. The core result and API
`generation_info` fields remain pure JSON; the combined layout applies only
to embedded image metadata.

WebUI conversion is lossy: known samplers and schedules have explicit name
mappings; unsupported ones use `SDCPP Sampler`/`SDCPP Schedule` fields. Hires
keeps the base request size and separate configured hires size/steps. The native
record retains the full generation parameters. Standalone postprocessing has no WebUI
infotext because its original diffusion settings are unknown.

WebUI prompt text appends ordinary LoRAs as `<lora:filename_without_extension:weight>`.
Directory components are omitted to match WebUI's filename lookup; the native
`loras` array retains relative paths and weights. High-noise-only LoRAs and names
that cannot be represented by this tag syntax remain in the native record only.
The native prompt is not modified, and postprocessing LoRAs are not added to the
source generation prompt.

JPEG requires the COM payload, including the label and terminating NUL bytes,
to fit within 65,533 bytes. Larger records fail encoding with an error rather
than being truncated. Use PNG or WebP for larger records.

Disabling file metadata omits both image representations, but does not disable
the core generation record or its API response. Native async image results and
OpenAI image response items contain `generation_info`; video results contain a
sequence-level `generation_info`. The SD API `info` object contains per-output
`generation_info` alongside derived `infotexts` and each image's seed. Its outer
`parameters` field continues to echo the request.
