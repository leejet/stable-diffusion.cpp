# Ovis-Image

Ovis-Image generates images from text using an Ovis 2.5 encoder and the FLUX VAE.

## Model files

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| Ovis-Image-7B | [Safetensors](https://huggingface.co/Comfy-Org/Ovis-Image/tree/main/split_files/diffusion_models)<br>[GGUF](https://huggingface.co/leejet/Ovis-Image-7B-GGUF) | `--diffusion-model` | Required |
| vae | [Safetensors](https://huggingface.co/black-forest-labs/FLUX.1-schnell/tree/main) | `--vae` | Use matching VAE |
| Ovis 2.5 | [Safetensors](https://huggingface.co/Comfy-Org/Ovis-Image/tree/main/split_files/text_encoders) | `--llm` | Use matching text encoder |

## Usage

CLI examples use the [shared path and shell conventions](README.md#example-conventions).

### Web UI

Follow the [Web UI guide](../../examples/server/README.md#web-ui) to start the
server, then select the recipe matching your weights. Fill the component slots
below and click **Load models**.
Use the files and compatibility requirements in [Model files](#model-files).

| Recipe | Component slots |
| --- | --- |
| Ovis Image (`ovis-image`) | `diffusion_model`, `llm`, `vae`, `tae` |

Slot IDs correspond to CLI component options: `diffusion_model` uses
`--diffusion-model`, `llm` uses `--llm`, and `model` uses `-m` / `--model`.
Select optional components when required by your checkpoint or task.

### CLI

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  --diffusion-model ..\models\diffusion_models\ovis_image-Q4_0.gguf `
  --vae ..\models\vae\ae.sft `
  --llm ..\models\text_encoders\ovis_2.5.safetensors `
  -p "a lovely cat" `
  --cfg-scale 5.0 `
  --offload-to-cpu `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  --diffusion-model ../models/diffusion_models/ovis_image-Q4_0.gguf \
  --vae ../models/vae/ae.sft \
  --llm ../models/text_encoders/ovis_2.5.safetensors \
  -p "a lovely cat" \
  --cfg-scale 5.0 \
  --offload-to-cpu \
  -v
```

<img alt="ovis image example" src="../../assets/ovis_image/example.png" />
