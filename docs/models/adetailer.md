# ADetailer

A DGL script runs a YOLOv8 object detector on an existing or newly generated
image, then performs a cropped inpaint pass for each selected object. The implementation supports YOLOv8
detection checkpoints. YOLOv8 segmentation and MediaPipe models are not
supported yet.

**Currently supported only through dgl.**

## Model files

| Component | Weights and downloads | DGL usage | Requirement |
| --- | --- | --- | --- |
| Base image model | [Compatible image-generation model](README.md) | `load_unet`, `load_text_encoder`, `load_vae` | Required for the inpainting pass |
| YOLOv8 detector | Converted Safetensors checkpoint prepared below | `load_detector` | Required; detection checkpoints only |

### Prepare the detector

Run the conversion command from the source repository root, then place the
converted file at the script's `model_path`.

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

### DGL

| Example | Description |
| --- | --- |
| [Repair an existing image](../dgl/adetailer/adetailer-existing-image.dgl) | Detect objects and inpaint selected crops. |
| [Generate and refine](../dgl/adetailer/adetailer-generated-image.dgl) | Generate the starting image, then run the same detail pass. |

#### Run a script

Edit paths, prompts and settings in the script's `var` block, then run from
the build directory. Paths inside the script resolve from the working directory.

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe -M dgl --script ..\docs\dgl\adetailer\adetailer-existing-image.dgl
```

**Linux/macOS**

```sh
./bin/sd-cli -M dgl --script ../docs/dgl/adetailer/adetailer-existing-image.dgl
```

Choose another script from the table to run a different task. Add `--check`
to validate syntax and types without loading weights. See the
[shared run instructions](../dgl/README.md#run-an-example) for build paths
and the server's file-access rules.

The [ADetailer script](../dgl/adetailer/adetailer-existing-image.dgl) loads a YOLOv8 detector
and an inpainting model. Set `confidence` and `nms` on `detections::detect`, and use
`detections::filter` to select the boxes to repaint. The script converts
the boxes to masks, plans crops with `regions::plan_mask_crops`, then assigns prompts and
seeds and blends each inpainted crop into the current image. See
[DGL detail refinement](../dgl.md#detail-refinement) for the data layout.

#### Repair an existing image

The script loads `input_path`, detects objects and repairs their crops.

Edit `detail_prompt`, `negative_prompt`, `seed` and the `settings` dict in the
script. The detector and mask operations are explicit, so filtering and crop
geometry can be changed independently.

#### Repair generated images

This script generates the starting image, then runs the same detail pass.

If no object is detected, `fold` returns the original image unchanged.
The crop/paste helpers preserve the original alpha channel. Detector or detail
failures stop the script. Run the same program through the server's DGL editor;
the removed native ADetailer mode and SDAPI recipe are not execution paths.

## Options

The example's `settings` dict controls crop size, steps, denoising strength,
CFG, sampler, scheduler and mask blur. Each stage has its own parameters:

| Stage | DGL operation | Settings |
| --- | --- | --- |
| Detection | `detections::detect` | `input_size`, `confidence`, `nms`, `max_detections` |
| Selection | `detections::filter` | `top_k`, `top_k_by`, area ratios and ordering |
| Masks | `detections::box_masks` and tensor operations | Dilation, offsets, merging and blur |
| Crop planning | `regions::plan_mask_crops` | `aspect`, `padding` |
| Refinement | `fold(..., process=refine_region)` | Prompts, `seed + region.index`, `settings` |

Crops are processed in order; each repaired image becomes the next iteration's
canvas. To run detection on CPU and diffusion on CUDA, set `backend="cpu"` on
`load_detector` and `backend="cuda0"` on `load_unet`. The script decides prompt
assignment explicitly; it does not parse legacy `[SEP]` / `[SKIP]` directives or
`--extra-ad-args`.

## Notes and limitations

YOLOv8 segmentation and MediaPipe detectors are not supported.
Use the [DGL API](../dgl.md#c-api) to run a complete program from a library caller.
