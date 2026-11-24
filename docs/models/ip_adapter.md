# IP-Adapter

[IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) uses a reference image
to guide SD 1.5 or SDXL generation alongside
the text prompt. Classic and Plus adapters are supported, including combination
with Control Net.

## Model files

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| Base checkpoint | [SD 1.5 or SDXL](sd.md) | `-m` / `--model` | Required; match adapter family |
| CLIP Vision H | `clip_vision_h.safetensors` from [IP-Adapter](https://huggingface.co/h94/IP-Adapter) | `--clip_vision` | Required; ViT-H/14 |
| SD 1.5 adapter | [`models/ip-adapter_sd15.safetensors`](https://huggingface.co/h94/IP-Adapter) | `--ip-adapter` | For SD 1.5 |
| SDXL adapter | [`sdxl_models/ip-adapter_sdxl_vit-h.safetensors`](https://huggingface.co/h94/IP-Adapter) | `--ip-adapter` | For SDXL |
| SD 1.5 Plus | [`models/ip-adapter-plus_sd15.safetensors`](https://huggingface.co/h94/IP-Adapter) | `--ip-adapter` | Alternative for SD 1.5 |
| SDXL Plus | [`sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors`](https://huggingface.co/h94/IP-Adapter) | `--ip-adapter` | Alternative for SDXL |

## Usage

CLI examples use the [shared path and shell conventions](README.md#example-conventions).

### Web UI

Follow the [Web UI guide](../../examples/server/README.md#web-ui) to start the
server, then select the recipe matching your weights. Fill the component slots
below and click **Load models**.
Use the files and compatibility requirements in [Model files](#model-files).

| Recipe | Component slots |
| --- | --- |
| SD 1.5 IP-Adapter / Plus (`sd1-ip-adapter`) | `model`, `clip_vision`, `ip_adapter`, `vae`, `tae` |
| SDXL IP-Adapter / Plus (`sdxl-ip-adapter`) | `model`, `clip_vision`, `ip_adapter`, `vae`, `tae` |

Slot IDs correspond to CLI component options: `diffusion_model` uses
`--diffusion-model`, `llm` uses `--llm`, and `model` uses `-m` / `--model`.
Select optional components when required by your checkpoint or task.

### CLI

#### Example (SD 1.5)

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -m ..\models\checkpoints\sd_v1.5.safetensors `
  --clip_vision ..\models\clip_vision\clip_vision_h.safetensors `
  --ip-adapter ..\models\ipadapter\ip-adapter_sd15.safetensors `
  --ip-adapter-image ..\inputs\reference.png `
  -p "a woman, best quality" `
  -n "lowres, bad anatomy" `
  --cfg-scale 7 `
  --steps 30 `
  --sampling-method dpm++2m `
  --scheduler karras `
  -W 512 `
  -H 512 `
  --ip-adapter-strength 0.8
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -m ../models/checkpoints/sd_v1.5.safetensors \
  --clip_vision ../models/clip_vision/clip_vision_h.safetensors \
  --ip-adapter ../models/ipadapter/ip-adapter_sd15.safetensors \
  --ip-adapter-image ../inputs/reference.png \
  -p "a woman, best quality" \
  -n "lowres, bad anatomy" \
  --cfg-scale 7 \
  --steps 30 \
  --sampling-method dpm++2m \
  --scheduler karras \
  -W 512 \
  -H 512 \
  --ip-adapter-strength 0.8
```

#### Example (SDXL)

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -m ..\models\checkpoints\sdxl.safetensors `
  --clip_vision ..\models\clip_vision\clip_vision_h.safetensors `
  --ip-adapter ..\models\ipadapter\ip-adapter_sdxl_vit-h.safetensors `
  --ip-adapter-image ..\inputs\reference.png `
  -p "a woman, best quality" `
  -n "lowres, bad anatomy" `
  --cfg-scale 6 `
  --steps 25 `
  --sampling-method dpm++2m `
  --scheduler karras `
  -W 1024 `
  -H 1024 `
  --ip-adapter-strength 0.8 `
  --vae-tiling
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -m ../models/checkpoints/sdxl.safetensors \
  --clip_vision ../models/clip_vision/clip_vision_h.safetensors \
  --ip-adapter ../models/ipadapter/ip-adapter_sdxl_vit-h.safetensors \
  --ip-adapter-image ../inputs/reference.png \
  -p "a woman, best quality" \
  -n "lowres, bad anatomy" \
  --cfg-scale 6 \
  --steps 25 \
  --sampling-method dpm++2m \
  --scheduler karras \
  -W 1024 \
  -H 1024 \
  --ip-adapter-strength 0.8 \
  --vae-tiling
```

The SDXL VAE decode at 1024x1024 is memory heavy; add `--vae-tiling` (and
`--offload-to-cpu`) on GPUs with limited VRAM.

#### Plus variants

Use the Plus weights matching your base model. The same IP-Adapter options
apply; no extra flag is needed. Plus can also be combined with Control Net.

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -m ..\models\checkpoints\sd_v1.5.safetensors `
  --clip_vision ..\models\clip_vision\clip_vision_h.safetensors `
  --ip-adapter ..\models\ipadapter\ip-adapter-plus_sd15.safetensors `
  --ip-adapter-image ..\inputs\reference.png `
  -p "a woman, best quality" `
  -n "lowres, bad anatomy" `
  --cfg-scale 7 `
  --steps 30 `
  --sampling-method dpm++2m `
  --scheduler karras `
  -W 512 `
  -H 512 `
  --ip-adapter-strength 0.8
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -m ../models/checkpoints/sd_v1.5.safetensors \
  --clip_vision ../models/clip_vision/clip_vision_h.safetensors \
  --ip-adapter ../models/ipadapter/ip-adapter-plus_sd15.safetensors \
  --ip-adapter-image ../inputs/reference.png \
  -p "a woman, best quality" \
  -n "lowres, bad anatomy" \
  --cfg-scale 7 \
  --steps 30 \
  --sampling-method dpm++2m \
  --scheduler karras \
  -W 512 \
  -H 512 \
  --ip-adapter-strength 0.8
```

The startup log line `IP-Adapter: 16 image tokens` (versus `4` for the
classic adapters) confirms a Plus file was loaded.

#### Combining with Control Net

Add the usual Control Net options to keep the reference appearance while
controlling the pose:

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -m ..\models\checkpoints\sdxl.safetensors `
  --clip_vision ..\models\clip_vision\clip_vision_h.safetensors `
  --ip-adapter ..\models\ipadapter\ip-adapter_sdxl_vit-h.safetensors `
  --control-net ..\models\controlnet\OpenPoseXL2.safetensors `
  --ip-adapter-image ..\inputs\character.png `
  --control-image ..\inputs\pose.png `
  -p "a character, side view" `
  --cfg-scale 6 `
  --steps 25 `
  -W 1024 `
  -H 1024 `
  --ip-adapter-strength 0.9 `
  --control-strength 0.8 `
  --vae-tiling
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -m ../models/checkpoints/sdxl.safetensors \
  --clip_vision ../models/clip_vision/clip_vision_h.safetensors \
  --ip-adapter ../models/ipadapter/ip-adapter_sdxl_vit-h.safetensors \
  --control-net ../models/controlnet/OpenPoseXL2.safetensors \
  --ip-adapter-image ../inputs/character.png \
  --control-image ../inputs/pose.png \
  -p "a character, side view" \
  --cfg-scale 6 \
  --steps 25 \
  -W 1024 \
  -H 1024 \
  --ip-adapter-strength 0.9 \
  --control-strength 0.8 \
  --vae-tiling
```

## Options

- `--ip-adapter <path>` path to the IP-Adapter weight file.
- `--ip-adapter-image <path>` path to the reference image.
- `--ip-adapter-strength <float>` strength of the IP-Adapter injection
  (default 1.0). Lower values let the text prompt dominate; 0.6 to 0.8 is
  a good starting range.
