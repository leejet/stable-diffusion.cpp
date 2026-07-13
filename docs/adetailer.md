# ADetailer

`sd-cli` can run a YOLOv8 object detector on an existing or newly generated
image and perform a cropped inpaint pass for every detected object. The first
implementation supports YOLOv8 detection checkpoints. YOLOv8 segmentation and
MediaPipe models are not supported yet.

## Convert a detector

Ultralytics checkpoints must be converted before use. The converter fuses
BatchNorm into convolution layers and writes a safetensors file with the weight
names expected by the native GGML implementation.

```bash
python scripts/convert_yolov8_to_safetensors.py face_yolov8n.pt face_yolov8n.safetensors
```

The converter requires Python packages `ultralytics`, `torch`, and
`safetensors`.
Only YOLOv8 detection checkpoints are accepted.
PyTorch checkpoints use pickle internally, so only convert `.pt` files from a
trusted source.

## Repair an existing image

Use the dedicated `adetailer` mode to detect and repair objects in an existing
image:

```bash
./bin/sd-cli \
  -M adetailer \
  -m model.safetensors \
  -i input.png \
  -o repaired.png \
  -p "detailed portrait photo" \
  --negative-prompt "deformed face" \
  --steps 24 \
  --cfg-scale 6 \
  --strength 0.4 \
  --sampling-method dpm++2m \
  --scheduler karras \
  --ad-model face_yolov8n.safetensors \
  --extra-ad-args "confidence=0.3,inpaint_padding=32,mask_blur=4"
```

This mode reuses the normal image-generation options for the detail pass:

- `--init-img`, `--output`, `--prompt`, and `--negative-prompt`
- `--steps`, `--cfg-scale`, `--sampling-method`, and `--scheduler`
- `--strength`, `--seed`, LoRA settings, VAE tiling, and backend assignments
- `--width` and `--height`, which also resize the input when specified

`--ad-prompt` and `--ad-negative-prompt` optionally override the normal prompts.
Values provided in `--extra-ad-args`, such as `steps`, `cfg_scale`,
`denoising_strength`, or `inpaint_width`, take precedence over inherited values.

## Repair generated images

ADetailer can also run automatically after normal image generation:

```bash
./bin/sd-cli \
  -m model.safetensors \
  -p "portrait photo" \
  --ad-model face_yolov8n.safetensors \
  --ad-prompt "[PROMPT], detailed face" \
  --ad-negative-prompt "" \
  --extra-ad-args "confidence=0.3,denoising_strength=0.4,inpaint_width=512,inpaint_height=512"
```

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
| `mask_min_ratio` | `0` | Minimum bbox area relative to the image |
| `mask_max_ratio` | `1` | Maximum bbox area relative to the image |
| `dilate_erode` | `4` | Positive values dilate; negative values erode |
| `x_offset`, `y_offset` | `0` | Mask offset in pixels; positive Y moves upward |
| `mask_mode` | `none` | `none`, `merge`, or `merge_invert` |
| `merge_masks`, `invert_mask` | `false` | Boolean alternatives to `mask_mode` |
| `mask_blur` | `4` | Final composite feather radius |
| `inpaint_padding` | `32` | Padding around the detected region |
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

```bash
--backend "diffusion=cuda0,detector=cpu"
```
