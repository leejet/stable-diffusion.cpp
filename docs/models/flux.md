# FLUX.1

FLUX.1 dev and schnell generate images from text using CLIP-L, T5-XXL, and the FLUX.1 VAE.

You can run Flux using stable-diffusion.cpp with a GPU that has 6GB or even 4GB of VRAM, without needing to offload to RAM.

## Model files

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| flux | If you don't want to do the conversion yourself, download the preconverted gguf model from [FLUX.1-dev-gguf](https://huggingface.co/leejet/FLUX.1-dev-gguf) or [FLUX.1-schnell](https://huggingface.co/leejet/FLUX.1-schnell-gguf)<br>Otherwise, download flux-dev from [Download](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/flux1-dev.safetensors) or flux-schnell from [Safetensors](https://huggingface.co/black-forest-labs/FLUX.1-schnell/blob/main/flux1-schnell.safetensors) | `--diffusion-model` | Required |
| vae | vae from [Download](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/ae.safetensors) | `--vae` | Use matching VAE |
| clip_l | clip_l from [Download](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/clip_l.safetensors) | `--clip_l` | Required |
| t5xxl | t5xxl from [Download](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp16.safetensors) | `--t5xxl` | Required |

### Optional conversion

Run executable conversion examples from the build directory using the
[shared path and shell conventions](README.md#example-conventions).

#### Convert flux weights

You can download the preconverted gguf weights from [FLUX.1-dev-gguf](https://huggingface.co/leejet/FLUX.1-dev-gguf) or [FLUX.1-schnell](https://huggingface.co/leejet/FLUX.1-schnell-gguf), this way you don't have to do the conversion yourself.

For example:
**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -M convert `
  -m ..\models\unet\flux1-dev.sft `
  --type q8_0 `
  -o ..\models\diffusion_models\flux1-dev-q8_0.gguf `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -M convert \
  -m ../models/unet/flux1-dev.sft \
  --type q8_0 \
  -o ../models/diffusion_models/flux1-dev-q8_0.gguf \
  -v
```

## Usage

CLI examples use the [shared path and shell conventions](README.md#example-conventions).

### Web UI

Follow the [Web UI guide](../../examples/server/README.md#web-ui) to start the
server, then select the recipe matching your weights. Fill the component slots
below and click **Load models**.
Use the files and compatibility requirements in [Model files](#model-files).

| Recipe | Component slots |
| --- | --- |
| FLUX.1 dev (`flux1-dev`) | `diffusion_model`, `clip_l`, `t5xxl`, `vae`, `tae` |
| FLUX.1 schnell (`flux1-schnell`) | `diffusion_model`, `clip_l`, `t5xxl`, `vae`, `tae` |

Slot IDs correspond to CLI component options: `diffusion_model` uses
`--diffusion-model`, `llm` uses `--llm`, and `model` uses `-m` / `--model`.
Select optional components when required by your checkpoint or task.

### CLI

- `--cfg-scale` is recommended to be set to 1.

#### Flux-dev

For example:

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  --diffusion-model ..\models\diffusion_models\flux1-dev-q8_0.gguf `
  --vae ..\models\vae\ae.sft `
  --clip_l ..\models\text_encoders\clip_l.safetensors `
  --t5xxl ..\models\text_encoders\t5xxl_fp16.safetensors `
  -p "a lovely cat holding a sign says 'flux.cpp'" `
  --cfg-scale 1.0 `
  --sampling-method euler `
  --clip-on-cpu `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  --diffusion-model ../models/diffusion_models/flux1-dev-q8_0.gguf \
  --vae ../models/vae/ae.sft \
  --clip_l ../models/text_encoders/clip_l.safetensors \
  --t5xxl ../models/text_encoders/t5xxl_fp16.safetensors \
  -p "a lovely cat holding a sign says 'flux.cpp'" \
  --cfg-scale 1.0 \
  --sampling-method euler \
  --clip-on-cpu \
  -v
```

Using formats of different precisions will yield results of varying quality.

| Type | q8_0  | q4_0  | q4_k  | q3_k  | q2_k |
|---- | ----  |----  |----  |----  |----  |
| **Memory** | 12068.09 MB  | 6394.53 MB | 6395.17 MB | 4888.16 MB  | 3735.73 MB |
| **Result** | ![](../../assets/flux/flux1-dev-q8_0.png) |![](../../assets/flux/flux1-dev-q4_0.png) |![](../../assets/flux/flux1-dev-q4_k.png) |![](../../assets/flux/flux1-dev-q3_k.png) |![](../../assets/flux/flux1-dev-q2_k.png)|

#### Flux-schnell

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  --diffusion-model ..\models\diffusion_models\flux1-schnell-q8_0.gguf `
  --vae ..\models\vae\ae.sft `
  --clip_l ..\models\text_encoders\clip_l.safetensors `
  --t5xxl ..\models\text_encoders\t5xxl_fp16.safetensors `
  -p "a lovely cat holding a sign says 'flux.cpp'" `
  --cfg-scale 1.0 `
  --sampling-method euler `
  --steps 4 `
  --clip-on-cpu `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  --diffusion-model ../models/diffusion_models/flux1-schnell-q8_0.gguf \
  --vae ../models/vae/ae.sft \
  --clip_l ../models/text_encoders/clip_l.safetensors \
  --t5xxl ../models/text_encoders/t5xxl_fp16.safetensors \
  -p "a lovely cat holding a sign says 'flux.cpp'" \
  --cfg-scale 1.0 \
  --sampling-method euler \
  --steps 4 \
  --clip-on-cpu \
  -v
```

| q8_0  |
| ----  |
|![](../../assets/flux/flux1-schnell-q8_0.png) |

#### Run with LoRA

Since many flux LoRA training libraries have used various LoRA naming formats, it is possible that not all flux LoRA naming formats are supported. It is recommended to use LoRA with naming formats compatible with ComfyUI.

##### Flux-dev q8_0 with LoRA

- LoRA model from https://huggingface.co/XLabs-AI/flux-lora-collection/tree/main (using comfy converted version!!!)

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  --diffusion-model ..\models\diffusion_models\flux1-dev-q8_0.gguf `
  --vae ..\models\vae\ae.sft `
  --clip_l ..\models\text_encoders\clip_l.safetensors `
  --t5xxl ..\models\text_encoders\t5xxl_fp16.safetensors `
  --lora-model-dir ..\models\loras `
  -p "a lovely cat holding a sign says 'flux.cpp'<lora:realism_lora_comfy_converted:1>" `
  --cfg-scale 1.0 `
  --sampling-method euler `
  --clip-on-cpu `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  --diffusion-model ../models/diffusion_models/flux1-dev-q8_0.gguf \
  --vae ../models/vae/ae.sft \
  --clip_l ../models/text_encoders/clip_l.safetensors \
  --t5xxl ../models/text_encoders/t5xxl_fp16.safetensors \
  --lora-model-dir ../models/loras \
  -p "a lovely cat holding a sign says 'flux.cpp'<lora:realism_lora_comfy_converted:1>" \
  --cfg-scale 1.0 \
  --sampling-method euler \
  --clip-on-cpu \
  -v
```

![output](../../assets/flux/flux1-dev-q8_0%20with%20lora.png)
