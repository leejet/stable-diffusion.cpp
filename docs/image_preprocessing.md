# Image preprocessing

Use `--image-preprocess` to transform each image input once, before generation:

```sh
sd-cli ... \
  --image-preprocess "target=init,mode=crop-resize,filter=lanczos,antialias=true" \
  --image-preprocess "target=mask,filter=nearest-exact" \
  --image-preprocess "target=ref,index=0,mode=fit-pad,width=768,height=768,filter=bicubic"
```

CLI and server image loaders decode at the original resolution. The generation
entry point merges input defaults with user rules and prepares one transformed
image per input. The original pipeline then consumes those images, including
its mandatory canvas adaptation, reference resizing, and encoder preprocessing.

```text
native-resolution image
  -> input defaults + user overrides
  -> one input transform
  -> original generation pipeline and model-specific processing
```

These rules do not override internal VAE, CLIP/VLM, ControlNet, or pixel-patch preprocessing.
`--ref-image-args` retains its existing meaning and runs after this input transform.

## Inputs and defaults

| `target` | Input | Default geometry | Indexed? |
| --- | --- | --- | --- |
| `init` | img2img image or video first frame | Center crop to the generation aspect ratio, then resize | No |
| `end` | Video last frame | Center crop, then resize | No |
| `mask` | Inpainting mask | Inherit init geometry; otherwise center crop, then resize | No |
| `control` | Control image | Center crop, then resize | No |
| `ref` | Reference images | Preserve source dimensions | Yes |
| `ip-adapter` | IP-Adapter image | Preserve source dimensions | No |
| `id` | PhotoMaker identity images | Preserve source dimensions | Yes |
| `control-frame` | Control video frames | Center crop, then resize | Yes |

Canvas defaults use the aligned generation dimensions. Reference, IP-Adapter,
and identity inputs use their original dimensions unless overridden. Default
resampling is nearest for images and nearest-exact for masks.

These defaults are shared by CLI, server, and C API. Moving geometry out of
the loaders replaces the previous CLI/server BOX/sRGB resizing, so default
pixels are not guaranteed to match earlier builds.

Reference video and audio preprocessing are outside these image rules.
Preprocessing options apply to `img_gen` and `vid_gen`, not standalone upscale
or ADetailer mode. ADetailer clears the user's rules for its internal crops.

## Rules

Rules are comma-separated `key=value` lists. Repeat the CLI option or separate
rules with semicolons. Every rule requires a `target` and at least one option.
Rule syntax and input compatibility are checked when image/video generation
starts. Unknown keys, invalid values, duplicate keys in a rule, missing images,
and out-of-range indices cause generation to fail with an error log.

Omit `index` to configure every image of that type; otherwise use a zero-based
index. CLI directory inputs follow filename order. Indexed rules override
type-wide rules field by field, regardless of order. At equal specificity,
the last value for a field wins. `auto` selects the input preset.

| `mode` | Input transform |
| --- | --- |
| `auto` | Use the input's default geometry |
| `none` | Keep source dimensions without resizing, cropping, or padding |
| `stretch` | Resize to the target dimensions |
| `crop` | Crop a target-sized rectangle without resizing; fail if the source is too small |
| `crop-resize` | Crop to the target aspect ratio, then resize |
| `fit-pad` | Fit the entire image inside the target dimensions, preserving aspect ratio, then pad |

`width` and `height` must be specified together as positive integers. They
override the input transform's dimensions, not the generation or encoder size.
For a native-size preset, specifying dimensions without a mode selects stretch.
`mode=none` with explicit dimensions different from the source is contradictory
and is rejected.

`anchor=center|top|bottom|left|right` selects crop/padding placement.
`pad_color=#RRGGBB` or `#RRGGBBAA` selects padding, defaulting to opaque black.
A grayscale mask uses the first color component.

`filter=auto|nearest|nearest-exact|bilinear|bicubic|lanczos` selects resampling.
`antialias=auto|true|false` enables antialiasing automatically for filtered
downscaling; explicit true requires bilinear, bicubic, or Lanczos.
Filtered RGBA resizing uses premultiplied alpha.

`canny=true|false` enables edge detection for any supported image target,
defaulting to `false`. It runs once after geometry, before the original
generation pipeline, including with `mode=none`. Grayscale, grayscale-alpha,
RGB, and RGBA inputs are supported; alpha is preserved.

Each input has its own Canny setting. Indexed rules can enable or disable it
for individual references, identity images, or video control frames.

```sh
--image-preprocess "target=init,mode=fit-pad,canny=true"
--image-preprocess "target=ref,index=0,mode=none,canny=true"
--image-preprocess "target=control-frame,index=2,canny=true"
```

Init and mask sources must have the same dimensions. The mask inherits the
init crop, resize, and padding coordinates, while retaining its own filter,
padding value, and Canny setting. Conflicting mask geometry is rejected. An
omitted mask remains absent until the original pipeline creates its default mask.

## Downstream behavior

`mode=none` only skips the input geometry transform. For example:

```sh
--image-preprocess "target=init,mode=none" \
--image-preprocess "target=ref,mode=none"
```

The init image is still adapted to the generation canvas by the original
pipeline. Reference images still follow `--ref-image-args` and model-specific
resizing. CLIP retains its fixed input dimensions and normalization. HiDream-O1
retains its original pixel-reference and visual preprocessing.

Existing sharing between consumers is preserved: for example, Wan img2video
uses the same adapted first frame for VAE conditioning and CLIP. High-resolution
passes reuse the prepared images and apply their original size adaptation;
they do not apply the user's crop a second time.

To disable reference resizing before VAE encoding, use
`--ref-image-args "resize_before_vae=false"` or the server field
`"ref_image_args": "resize_before_vae=false"`. This is separate from
`target=ref,mode=none`, which only skips input geometry. Model constraints
still apply.

## Server requests

Native image/video requests and SDAPI accept `image_preprocess` as a string or
an array of rule strings:

```json
{
  "image_preprocess": [
    "target=init,mode=fit-pad,filter=bicubic",
    "target=mask,filter=nearest-exact",
    "target=ref,index=0,mode=none"
  ]
}
```

OpenAI-compatible requests accept it through
`<sd_cpp_extra_args>{...}</sd_cpp_extra_args>` in the prompt.
Request rules replace server-default rules. Generation metadata records the
user rules; image encodings and channel conventions are unchanged.

## C API

Set `image_preprocess` on the existing image/video generation parameters.
The `generate_image()` and `generate_video()` signatures are unchanged:

```c
sd_img_gen_params_t params;
sd_img_gen_params_init(&params);
/* Set prompt, original-resolution input images, and generation options. */
params.image_preprocess.rules = "target=init,mode=crop-resize,filter=lanczos;"
                                "target=mask,filter=nearest-exact";
bool ok = generate_image(ctx, &params, &images, &count);
```

Both generation parameter initializers set `image_preprocess.rules` to `NULL`,
selecting input presets. Rule strings are borrowed for the synchronous call.
The library owns temporary transformed pixels; caller images and arrays are
not modified. Add `canny=true` to the desired target's rule in
`image_preprocess.rules` to enable Canny.

The parameter structs have grown; applications and bindings must be rebuilt.
