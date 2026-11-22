# Usage

From the build directory, show the full list of command-line arguments:

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe -h
```

**Linux/macOS**

```sh
./bin/sd-cli -h
```

See [sampling methods and schedulers](../../docs/guides/sampling.md) for supported
names, defaults, and custom sigma schedules.

The CLI defaults to seed 42 for repeatable runs. Pass `--seed -1` to request a
random seed. Other omitted inference options use the
[library defaults](../../docs/guides/c_api.md#keys).
Video generation defaults to 6 requested frames before model
alignment. The conditioning cache uses the library default. Zero and false are
explicit inputs; invalid values are reported rather than treated as omitted.

IP-Adapter, PhotoMaker, PuLID, AnimateDiff, SLG, NitroFusion, ADetailer and video
upscaling use explicit DGL workflows. Their former native CLI options and the
`adetailer` / `vid_upscale` modes have been removed. See the
[model and component guides](../../docs/models/README.md) for scripts.

`--ref-video` accepts a frame directory or a video file. Directories retain
filename order and 24 fps; files are decoded through DGL `video::load` at
24 fps. Supported embedded audio is retained unless a matching
`--ref-video-audio` WAV overrides it. `--control-video` accepts the same inputs;
files are sampled at the generation `--fps` (default 16), without audio, and
`--video-frames` limits decoding after resampling. Existing VACE frame padding
is unchanged. File formats are AVI, static/animated WebP, and this project's
independent-frame VP8 WebM. VP9 and Opus are unsupported; see
[DGL media I/O](../../docs/dgl.md) for codec and build requirements.

Logging defaults to `info`. Use `--log-level <level>` to select `debug`, `verbose`,
`info`, `warn`, or `error` (from most to least detailed). Each level includes
messages at that level and all less detailed levels. `-v` and `--verbose` are
equivalent to `--log-level verbose`. If repeated, the last logging option wins.
Native image/video generation and upscaling print the translated DGL program
before execution at debug level (`--log-level debug`).

Run a [DGL program](../../docs/dgl.md) with `sd-cli -M dgl --script program.dgl`.
DGL mode accepts the same logging options and sends logs to stderr.
Weight loading displays a progress bar with transfer speed; sampling displays
its step progress. Context progress callbacks remain active during loading.
`--run-once on|off` controls model release after last use and defaults to `on`.

`--flash-attn on|off|auto` defaults to `auto`, selecting flash attention for each
compute backend during context creation. See [Backend selection](../../docs/guides/backend.md#flash-attention---flash-attn-onoffauto)
for probing behavior.
Use module assignments such as `--flash-attn auto,te=off,vae=on` to override
individual modules.

`--conv-direct on|off|auto` also defaults to `auto`, enabling direct convolution
on backends with implicit GEMM implementations. Use `--conv-direct auto,vae=off`
for module overrides; supported modules are `diffusion`, `vae`, and `controlnet`.
See [Direct convolution](../../docs/guides/backend.md#direct-convolution---conv-direct-onoffauto).

For direct image repair or automatic post-generation YOLOv8 detection followed by cropped inpainting, see
[ADetailer](../../docs/models/adetailer.md).

Use repeatable `--image-preprocess` rules to select resizing, cropping, padding,
and resampling separately for each image input. Add `canny=true` to any input
rule for edge detection. See
[Image preprocessing](../../docs/guides/image_preprocessing.md) for input selectors,
input defaults, downstream model processing, mask alignment, and examples.

Metadata mode inspects PNG, JPEG, WebP (including animated WebP), AVI, WebM, and Matroska
container metadata without loading any model. Use `--image` for both images and videos:

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe -M metadata --image .\output.png
.\bin\Release\sd-cli.exe `
  -M metadata `
  --image .\output.jpg `
  --metadata-format json
.\bin\Release\sd-cli.exe -M metadata --image .\output.webm
.\bin\Release\sd-cli.exe `
  -M metadata `
  --image .\output.avi `
  --metadata-format json
.\bin\Release\sd-cli.exe `
  -M metadata `
  --image .\output.png `
  --metadata-raw
.\bin\Release\sd-cli.exe `
  -M metadata `
  --image .\output.png `
  --metadata-all
```

**Linux/macOS**

```sh
./bin/sd-cli -M metadata --image ./output.png
./bin/sd-cli \
  -M metadata \
  --image ./output.jpg \
  --metadata-format json
./bin/sd-cli -M metadata --image ./output.webm
./bin/sd-cli \
  -M metadata \
  --image ./output.avi \
  --metadata-format json
./bin/sd-cli \
  -M metadata \
  --image ./output.png \
  --metadata-raw
./bin/sd-cli \
  -M metadata \
  --image ./output.png \
  --metadata-all
```

For completely black or white images or videos, NaNs, and the `--linear-scale` /
`--attn-scale` workaround, see [Troubleshooting](../../docs/troubleshooting.md).

Generation outputs embed the core [generation record](../../docs/guides/generation_metadata.md).
Images retain the native JSON and an additional WebUI-compatible infotext.
`--disable-image-metadata` omits both from files.
