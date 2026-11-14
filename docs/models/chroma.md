# Chroma

Chroma generates images from text using T5-XXL and the FLUX.1 VAE.

You can run Chroma using stable-diffusion.cpp with a GPU that has 6GB or even 4GB of VRAM, without needing to offload to RAM.

## Model files

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| Chroma | If you don't want to do the conversion yourself, download the preconverted gguf model from [silveroxides/Chroma-GGUF](https://huggingface.co/silveroxides/Chroma-GGUF)<br>Otherwise, download chroma's safetensors from [lodestones/Chroma](https://huggingface.co/lodestones/Chroma) | `--diffusion-model` | Required |
| vae | vae from [Download](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/ae.safetensors) | `--vae` | Use matching VAE |
| t5xxl | t5xxl from [Download](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp16.safetensors) | `--t5xxl` | Required |

### Optional conversion

Run executable conversion examples from the build directory using the
[shared path and shell conventions](README.md#example-conventions).

#### Convert Chroma weights

You can download the preconverted gguf weights from [silveroxides/Chroma-GGUF](https://huggingface.co/silveroxides/Chroma-GGUF), this way you don't have to do the conversion yourself.

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -M convert `
  -m ..\models\unet\chroma-unlocked-v40.safetensors `
  --type q8_0 `
  -o ..\models\diffusion_models\chroma-unlocked-v40-q8_0.gguf `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -M convert \
  -m ../models/unet/chroma-unlocked-v40.safetensors \
  --type q8_0 \
  -o ../models/diffusion_models/chroma-unlocked-v40-q8_0.gguf \
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
| Chroma (`chroma`) | `diffusion_model`, `t5xxl`, `vae`, `tae` |

Slot IDs correspond to CLI component options: `diffusion_model` uses
`--diffusion-model`, `llm` uses `--llm`, and `model` uses `-m` / `--model`.
Select optional components when required by your checkpoint or task.

### CLI

#### Example

For example:

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  --diffusion-model ..\models\diffusion_models\chroma-unlocked-v40-q8_0.gguf `
  --vae ..\models\vae\ae.sft `
  --t5xxl ..\models\text_encoders\t5xxl_fp16.safetensors `
  --model-args chroma_use_dit_mask=false `
  -p "a lovely cat holding a sign says 'chroma.cpp'" `
  --cfg-scale 4.0 `
  --sampling-method euler `
  --clip-on-cpu `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  --diffusion-model ../models/diffusion_models/chroma-unlocked-v40-q8_0.gguf \
  --vae ../models/vae/ae.sft \
  --t5xxl ../models/text_encoders/t5xxl_fp16.safetensors \
  --model-args chroma_use_dit_mask=false \
  -p "a lovely cat holding a sign says 'chroma.cpp'" \
  --cfg-scale 4.0 \
  --sampling-method euler \
  --clip-on-cpu \
  -v
```

![](../../assets/flux/chroma_v40.png)
