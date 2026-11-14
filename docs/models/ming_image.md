# Ming-Image

[Ming-Image](https://github.com/inclusionAI/Ming-Image) 0.1 Design uses a 6B diffusion transformer (DiT), Ling-mini-2.0 for text conditioning, and the Ming-Image VAE. Text-to-image generation with RGBA output is supported.

## Model files

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| Ming-Image 0.1 Design DiT | [Safetensors](https://huggingface.co/Comfy-Org/Ming-Image/tree/main/diffusion_models) | `--diffusion-model` | Required |
| Ling-mini-2.0 BF16 | [Safetensors](https://huggingface.co/Comfy-Org/Ming-Image/tree/main/text_encoders) | `--llm` | Use matching text encoder |
| Ming-Image VAE | [Safetensors](https://huggingface.co/Comfy-Org/Ming-Image/tree/main/vae) | `--vae` | Use matching VAE |
| Ling tokenizer | tokenizer.json: [Download](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design/blob/main/mllm/tokenizer.json) | `--tokenizer` | Required; match text encoder |

The example below uses `ming_image_0.1_design_bf16.safetensors` for the DiT. You can also use `ming_image_0.1_design_int8_convrot.safetensors` with [INT8 convrot support](../guides/int8_convrot.md). Use the BF16 text encoder.

## Usage

CLI examples use the [shared path and shell conventions](README.md#example-conventions).

### Web UI

Follow the [Web UI guide](../../examples/server/README.md#web-ui) to start the
server, then select the recipe matching your weights. Fill the component slots
below and click **Load models**.
Use the files and compatibility requirements in [Model files](#model-files).

| Recipe | Component slots |
| --- | --- |
| Ming Image 0.1 (`ming-image`) | `diffusion_model`, `llm`, `tokenizer`, `vae`, `tae` |

Slot IDs correspond to CLI component options: `diffusion_model` uses
`--diffusion-model`, `llm` uses `--llm`, and `model` uses `-m` / `--model`.
Select optional components when required by your checkpoint or task.

### CLI

#### Text-to-image

Pass the Ling `tokenizer.json` with `--tokenizer` and use the matching Ming-Image VAE. Image dimensions must be multiples of 16. Save the output as PNG to preserve the alpha channel.

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  --diffusion-model ..\models\diffusion_models\ming_image_0.1_design_bf16.safetensors `
  --llm ..\models\text_encoders\ming_image_0.1_ling_mini_2.0_bf16.safetensors `
  --vae ..\models\vae\ming_image_vae_bf16.safetensors `
  --tokenizer ..\models\tokenizers\tokenizer.json `
  -p "A cheerful orange cat sticker, transparent background" `
  --steps 12 `
  --cfg-scale 1 `
  --sampling-method euler `
  --width 1024 `
  --height 1024 `
  --offload-to-cpu `
  -v `
  -o ming_image.png
```

**Linux/macOS**

```sh
./bin/sd-cli \
  --diffusion-model ../models/diffusion_models/ming_image_0.1_design_bf16.safetensors \
  --llm ../models/text_encoders/ming_image_0.1_ling_mini_2.0_bf16.safetensors \
  --vae ../models/vae/ming_image_vae_bf16.safetensors \
  --tokenizer ../models/tokenizers/tokenizer.json \
  -p "A cheerful orange cat sticker, transparent background" \
  --steps 12 \
  --cfg-scale 1 \
  --sampling-method euler \
  --width 1024 \
  --height 1024 \
  --offload-to-cpu \
  -v \
  -o ming_image.png
```
