# Usage

For detailed command-line arguments, run:

```bash
./bin/sd-cli -h
```

Logging defaults to `info`. Use `--log-level <level>` to select `debug`, `verbose`,
`info`, `warn`, or `error` (from most to least detailed). Each level includes
messages at that level and all less detailed levels. `-v` and `--verbose` are
equivalent to `--log-level verbose`. If repeated, the last logging option wins.

`--flash-attn on|off|auto` defaults to `auto`, selecting flash attention for each
compute backend during context creation. See [Backend selection](../../docs/backend.md#flash-attention---flash-attn-onoffauto)
for probing behavior.
Use module assignments such as `--flash-attn auto,te=off,vae=on` to override
individual modules.

`--conv-direct on|off|auto` also defaults to `auto`, enabling direct convolution
on backends with implicit GEMM implementations. Use `--conv-direct auto,vae=off`
for module overrides; supported modules are `diffusion`, `vae`, and `controlnet`.
See [Direct convolution](../../docs/backend.md#direct-convolution---conv-direct-onoffauto).

For direct image repair or automatic post-generation YOLOv8 detection followed by cropped inpainting, see
[ADetailer](../../docs/adetailer.md).

Use repeatable `--image-preprocess` rules to select resizing, cropping, padding,
and resampling separately for each image input. Add `canny=true` to any input
rule for edge detection. See
[Image preprocessing](../../docs/image_preprocessing.md) for input selectors,
input defaults, downstream model processing, mask alignment, and examples.

Metadata mode inspects PNG, JPEG, WebP (including animated WebP), AVI, WebM, and Matroska
container metadata without loading any model. Use `--image` for both images and videos:

```bash
./bin/sd-cli -M metadata --image ./output.png
./bin/sd-cli -M metadata --image ./output.jpg --metadata-format json
./bin/sd-cli -M metadata --image ./output.webm
./bin/sd-cli -M metadata --image ./output.avi --metadata-format json
./bin/sd-cli -M metadata --image ./output.png --metadata-raw
./bin/sd-cli -M metadata --image ./output.png --metadata-all
```

For completely black or white images or videos, NaNs, and the `--linear-scale` /
`--attn-scale` workaround, see [Troubleshooting](../../docs/troubleshooting.md).

Generation outputs embed the core [generation record](../../docs/generation_metadata.md).
Images retain the native JSON and an additional WebUI-compatible infotext.
`--disable-image-metadata` omits both from files.
