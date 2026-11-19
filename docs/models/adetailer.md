# ADetailer

`sd-cli` can run a YOLOv8 object detector on an existing or newly generated
image, and `sd-server` can run it after image generation. Both perform a cropped
inpaint pass for every detected object. The implementation supports YOLOv8
detection checkpoints. YOLOv8 segmentation and MediaPipe models are not
supported yet.

## Model files

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| Base image model | [Compatible image-generation model](README.md) | `-m` / model component options | Required for the inpainting pass |
| YOLOv8 detector | Converted Safetensors checkpoint prepared below | `--ad-model` | Required; detection checkpoints only |

### Prepare the detector

Run the conversion command from the source repository root, then place the
converted file at the path used by `--ad-model`.

Ultralytics checkpoints must be converted before use. The converter writes the Safetensors detector used by sd.cpp.

**Windows (PowerShell)**

```powershell
python scripts\convert_yolov8_to_safetensors.py face_yolov8n.pt face_yolov8n.safetensors
```

**Linux/macOS**

```sh
python scripts/convert_yolov8_to_safetensors.py face_yolov8n.pt face_yolov8n.safetensors
```

The converter requires Python packages `ultralytics`, `torch`, and
`safetensors`.
Only YOLOv8 detection checkpoints are accepted.
PyTorch checkpoints use pickle internally, so only convert `.pt` files from a
trusted source.

## Usage

CLI examples use the [shared path and shell conventions](README.md#example-conventions).

### CLI

#### Repair an existing image

Use the dedicated `adetailer` mode to detect and repair objects in an existing
image:

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -M adetailer `
  -m ..\models\checkpoints\model.safetensors `
  --ad-model ..\models\detectors\face_yolov8n.safetensors `
  -i input.png `
  -p "detailed portrait photo" `
  --negative-prompt "deformed face" `
  --steps 24 `
  --cfg-scale 6 `
  --strength 0.4 `
  --sampling-method dpm++2m `
  --scheduler karras `
  --extra-ad-args "confidence=0.3,inpaint_padding=32,mask_blur=4" `
  -o repaired.png
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -M adetailer \
  -m ../models/checkpoints/model.safetensors \
  --ad-model ../models/detectors/face_yolov8n.safetensors \
  -i input.png \
  -p "detailed portrait photo" \
  --negative-prompt "deformed face" \
  --steps 24 \
  --cfg-scale 6 \
  --strength 0.4 \
  --sampling-method dpm++2m \
  --scheduler karras \
  --extra-ad-args confidence=0.3,inpaint_padding=32,mask_blur=4 \
  -o repaired.png
```

This mode reuses the normal image-generation options for the detail pass:

- `--init-img`, `--output`, `--prompt`, and `--negative-prompt`
- `--steps`, `--cfg-scale`, `--sampling-method`, and `--scheduler`
- `--strength`, `--seed`, LoRA settings, VAE tiling, and backend assignments
- `--width` and `--height`, which also resize the input when specified

`--ad-prompt` and `--ad-negative-prompt` optionally override the normal prompts.
Values provided in `--extra-ad-args`, such as `steps`, `cfg_scale`,
`denoising_strength`, or `inpaint_width`, take precedence over inherited values.

#### Repair generated images

ADetailer can also run automatically after normal image generation:

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -m ..\models\checkpoints\model.safetensors `
  --ad-model ..\models\detectors\face_yolov8n.safetensors `
  -p "portrait photo" `
  --ad-prompt "[PROMPT], detailed face" `
  --ad-negative-prompt "" `
  --extra-ad-args "confidence=0.3,denoising_strength=0.4,inpaint_width=512,inpaint_height=512"
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -m ../models/checkpoints/model.safetensors \
  --ad-model ../models/detectors/face_yolov8n.safetensors \
  -p "portrait photo" \
  --ad-prompt "[PROMPT], detailed face" \
  --ad-negative-prompt "" \
  --extra-ad-args confidence=0.3,denoising_strength=0.4,inpaint_width=512,inpaint_height=512
```

The same options can be supplied when starting `sd-server`. When `--ad-model` is set, its detector runs after OpenAI-compatible and native asynchronous image generation. SDAPI requests enable ADetailer through `alwayson_scripts`, as described below. If no object is detected, the original image is returned unchanged. RGBA output keeps its original alpha channel, including when detected regions are repaired. Detector initialization or detail-pass failures are reported as request failures.

Server image-generation requests limit base, hires, and detail-pass dimensions to 4096 pixels per side. ADetailer step overrides are limited to 1–100; `steps=0` in `extra_ad_args` keeps inheriting the main generation steps. These limits also apply to inherited dimensions and `extra_ad_args` overrides. Requests exceeding the limits return a parameter error before generation. CLI limits are unchanged.

### SDAPI requests

`/sdapi/v1/txt2img` and `/sdapi/v1/img2img` accept the WebUI ADetailer argument format:

```json
{
  "prompt": "portrait photo",
  "alwayson_scripts": {
    "ADetailer": {
      "args": [true, false, {
        "ad_model": "face_yolov8n.pt",
        "ad_prompt": "[PROMPT], detailed face",
        "ad_use_steps": true,
        "ad_steps": 20
      }]
    }
  }
}
```

The detector must be configured at server startup with `--ad-model`. `ad_model` must match that file's basename; the original `.pt` basename is also accepted for a converted detector. Requests cannot load arbitrary detector paths. Omitting the script, setting its first argument to `false`, selecting `"None"`, or setting `ad_tab_enable` to `false` disables detailing. Dictionary-only argument arrays are also accepted. Multiple enabled dictionaries run in order using the configured detector.

Supported settings include prompts, confidence, mask filtering by area or confidence, mask count and area ratios, dilation/erosion, offsets, mask merging/inversion, blur, denoising strength, masked or full-image inpainting, padding, and optional overrides for inpaint dimensions, steps, CFG, sampler/scheduler, and CLIP skip. The corresponding `ad_use_*` flags control whether overrides apply. Otherwise dimensions, steps, CFG, sampler, scheduler, and CLIP skip inherit the main request. Denoising strength defaults to `0.4`. Pixel mask settings are limited to the larger output dimension, with a minimum limit of 32 and a maximum of 16384 pixels.

Model/VAE switching, non-default noise multipliers, face restoration, ControlNet, and skipping the initial img2img pass are not supported; active requests for these features return an error. Unknown settings and invalid active values also return errors. During detailing, `/sdapi/v1/progress` remains active with an `adetailer` job; its percentage and preview stay fixed until detailing finishes.

## Options

An empty ADetailer prompt inherits the main prompt. `[PROMPT]` inserts the main
prompt, `[SEP]` assigns different prompts to consecutive masks, and `[SKIP]`
skips the corresponding mask.

All settings other than the detector path and prompts are passed through
`--extra-ad-args` as a comma-separated `key=value` list:

| Key | Default | Description |
| --- | ---: | --- |
| `input_size` | `640` | Square YOLO input size; must be a multiple of 32 |
| `confidence` | `0.3` | Detection confidence threshold |
| `nms` | `0.45` | NMS IoU threshold |
| `max_detections` | `100` | Maximum detections retained after NMS |
| `mask_k_largest` | `0` | Keep only the largest K masks; zero keeps all |
| `mask_filter_method` | `area` | Rank retained masks by `area` or `confidence` |
| `mask_min_ratio` | `0` | Minimum bbox area relative to the image |
| `mask_max_ratio` | `1` | Maximum bbox area relative to the image |
| `dilate_erode` | `4` | Positive values dilate; negative values erode |
| `x_offset`, `y_offset` | `0` | Mask offset in pixels; positive Y moves upward |
| `mask_mode` | `none` | `none`, `merge`, or `merge_invert` |
| `merge_masks`, `invert_mask` | `false` | Boolean alternatives to `mask_mode` |
| `mask_blur` | `4` | Final composite feather radius |
| `inpaint_padding` | `32` | Padding around the detected region |
| `inpaint_only_masked` | `true` | Crop to the detected region; `false` uses the full image with the detection mask |
| `inpaint_width`, `inpaint_height` | mode-specific | `512x512` after generation; input/output size in `adetailer` mode |
| `denoising_strength` | mode-specific | `0.4` after generation; inherits `--strength` in `adetailer` mode |
| `steps` | `0` | Detail steps; zero inherits the main generation |
| `cfg_scale` | `-1` | Detail CFG; a negative value inherits the main generation |
| `sample_method` | inherited | Detail sampler name |
| `scheduler` | inherited | Detail scheduler name |
| `sort_by` | `none` | `none`, `left_to_right`, `center_to_edge`, or `area` |

Multiple masks are processed serially. Each completed inpaint becomes the input
for the next mask, and the seed is incremented by the mask index. Use
`mask_mode=merge` to process all detections in one inpaint pass.

The detector uses the `detector` backend module. For example, keep detection on
the CPU while diffusion runs on CUDA:

**Windows (PowerShell) arguments**

```text
--backend "diffusion=cuda0,detector=cpu"
```

**Linux/macOS arguments**

```text
--backend diffusion=cuda0,detector=cpu
```

## Notes and limitations

YOLOv8 segmentation and MediaPipe detectors are not supported. See the
[server API reference](../../examples/server/api.md) for request schemas.
