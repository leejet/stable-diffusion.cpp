# FLUX.1 Kontext

FLUX.1 Kontext supports reference-image editing using the FLUX.1 VAE, CLIP-L, and T5-XXL.

You can run Kontext using stable-diffusion.cpp with a GPU that has 6GB or even 4GB of VRAM, without needing to offload to RAM.

## Model files

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| Kontext | If you don't want to do the conversion yourself, download the preconverted gguf model from [FLUX.1-Kontext-dev-GGUF](https://huggingface.co/QuantStack/FLUX.1-Kontext-dev-GGUF)<br>Otherwise, download FLUX.1-Kontext-dev from [Download](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev/blob/main/flux1-kontext-dev.safetensors) | `--diffusion-model` | Required |
| vae | vae from [Download](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/ae.safetensors) | `--vae` | Use matching VAE |
| clip_l | clip_l from [Download](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/clip_l.safetensors) | `--clip_l` | Required |
| t5xxl | t5xxl from [Download](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp16.safetensors) | `--t5xxl` | Required |

### Optional conversion

Run executable conversion examples from the build directory using the
[shared path and shell conventions](README.md#example-conventions).

#### Convert Kontext weights

You can download the preconverted gguf weights from [FLUX.1-Kontext-dev-GGUF](https://huggingface.co/QuantStack/FLUX.1-Kontext-dev-GGUF), this way you don't have to do the conversion yourself.

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -M convert `
  -m ..\models\unet\flux1-kontext-dev.safetensors `
  --type q8_0 `
  -o ..\models\diffusion_models\flux1-kontext-dev-q8_0.gguf `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -M convert \
  -m ../models/unet/flux1-kontext-dev.safetensors \
  --type q8_0 \
  -o ../models/diffusion_models/flux1-kontext-dev-q8_0.gguf \
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
| FLUX.1 Kontext dev (`flux1-kontext`) | `diffusion_model`, `clip_l`, `t5xxl`, `vae`, `tae` |

Slot IDs correspond to CLI component options: `diffusion_model` uses
`--diffusion-model`, `llm` uses `--llm`, and `model` uses `-m` / `--model`.
Select optional components when required by your checkpoint or task.

### CLI

- `--cfg-scale` is recommended to be set to 1.

#### Example

For example:

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  --diffusion-model ..\models\diffusion_models\flux1-kontext-dev-q8_0.gguf `
  --vae ..\models\vae\ae.sft `
  --clip_l ..\models\text_encoders\clip_l.safetensors `
  --t5xxl ..\models\text_encoders\t5xxl_fp16.safetensors `
  -r .\flux1-dev-q8_0.png `
  -p "change 'flux.cpp' to 'kontext.cpp'" `
  --cfg-scale 1.0 `
  --sampling-method euler `
  --clip-on-cpu `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  --diffusion-model ../models/diffusion_models/flux1-kontext-dev-q8_0.gguf \
  --vae ../models/vae/ae.sft \
  --clip_l ../models/text_encoders/clip_l.safetensors \
  --t5xxl ../models/text_encoders/t5xxl_fp16.safetensors \
  -r ./flux1-dev-q8_0.png \
  -p "change 'flux.cpp' to 'kontext.cpp'" \
  --cfg-scale 1.0 \
  --sampling-method euler \
  --clip-on-cpu \
  -v
```

| ref_image | prompt  | output  |
| ---- | ----  |----  |
| ![](../../assets/flux/flux1-dev-q8_0.png) | change 'flux.cpp' to 'kontext.cpp' |![](../../assets/flux/kontext1_dev_output.png) |
