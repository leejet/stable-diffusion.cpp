# P-H3-Image

P-H3-Image supports text-to-image and reference-image editing. It uses a MiniMax-H3 diffusion transformer, a single-frame VAE, and either Qwen3-VL-4B with a text connector or the MiniMax-H3 Qwen3-VL-32B text encoder. Prompt enhancement is not included.

## Model files

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| P-H3-Image | [Safetensors](https://huggingface.co/PrunaAI/P-H3-Image-Comfy/tree/main/diffusion_models) | `--diffusion-model` | Required |
| P-H3-Image VAE | [Safetensors](https://huggingface.co/PrunaAI/P-H3-Image-Comfy/tree/main/vae) | `--vae` | Use matching VAE |
| Qwen3-VL-4B with the text connector | [Download](https://huggingface.co/PrunaAI/P-H3-Image-Comfy/tree/main/text_encoders) | `--llm` | Use matching text encoder |
| MiniMax-H3 Qwen3-VL-32B text encoder | [Safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/text_encoders)<br>[GGUF](https://huggingface.co/leejet/MiniMax-H3-GGUF/tree/main) | `--llm` | Use matching text encoder |

Alternatively, reuse Qwen3-VL-4B, including its vision tower, with the [separate text connector](https://huggingface.co/PrunaAI/P-H3-Image/blob/main/text_connector/diffusion_pytorch_model.safetensors). Save the connector as `p-h3-image_text_connector.safetensors` and pass it with `--embeddings-connectors`.

The MiniMax-H3 Qwen3-VL-32B encoder does not need a text connector. To use it in
either example below, omit `--embeddings-connectors` and replace `--llm` with:

**Windows (PowerShell) arguments**

```text
--llm ..\models\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf
```

**Linux/macOS arguments**

```text
--llm ../models/text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf
```

Reference-image editing needs the matching encoder's vision tower. If stored
separately, pass it with `--llm_vision`.

Use the P-H3 image VAE; the MiniMax-H3 video VAE is not compatible. No audio VAE is needed.

## Usage

CLI examples use the [shared path and shell conventions](README.md#example-conventions).

### Web UI

Follow the [Web UI guide](../../examples/server/README.md#web-ui) to start the
server, then select the recipe matching your weights. Fill the component slots
below and click **Load models**.
Use the files and compatibility requirements in [Model files](#model-files).

| Recipe | Component slots |
| --- | --- |
| P-H3-Image (`p-h3-image`) | `diffusion_model`, `llm`, `llm_vision`, `embeddings_connectors`, `vae` |

Slot IDs correspond to CLI component options: `diffusion_model` uses
`--diffusion-model`, `llm` uses `--llm`, and `model` uses `-m` / `--model`.
Select optional components when required by your checkpoint or task.

### CLI

#### Text-to-image

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  --diffusion-model ..\models\diffusion_models\p-h3-image_int8_convrot.safetensors `
  --llm ..\models\text_encoders\qwen3vl_4b_bf16.safetensors `
  --embeddings-connectors ..\models\text_encoders\p-h3-image_text_connector.safetensors `
  --vae ..\models\vae\p-h3-image_vae_bf16.safetensors `
  -p "a lovely cat holding a sign says 'p-h3-image.cpp'" `
  --cfg-scale 1.0 `
  --steps 8 `
  --sampling-method euler `
  --sigmas p-h3-image `
  -H 2912 `
  -W 2912 `
  --rng cpu `
  --offload-to-cpu `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  --diffusion-model ../models/diffusion_models/p-h3-image_int8_convrot.safetensors \
  --llm ../models/text_encoders/qwen3vl_4b_bf16.safetensors \
  --embeddings-connectors ../models/text_encoders/p-h3-image_text_connector.safetensors \
  --vae ../models/vae/p-h3-image_vae_bf16.safetensors \
  -p "a lovely cat holding a sign says 'p-h3-image.cpp'" \
  --cfg-scale 1.0 \
  --steps 8 \
  --sampling-method euler \
  --sigmas p-h3-image \
  -H 2912 \
  -W 2912 \
  --rng cpu \
  --offload-to-cpu \
  -v
```

#### Image editing

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  --diffusion-model ..\models\diffusion_models\p-h3-image_int8_convrot.safetensors `
  --llm ..\models\text_encoders\qwen3vl_4b_bf16.safetensors `
  --embeddings-connectors ..\models\text_encoders\p-h3-image_text_connector.safetensors `
  --vae ..\models\vae\p-h3-image_vae_bf16.safetensors `
  -r source.png `
  -r reference.png `
  -p "Edit <Picture 1>. Give the cat the scarf from <Picture 2>, preserving its face and the background." `
  --cfg-scale 1.0 `
  --steps 8 `
  --sampling-method euler `
  --sigmas p-h3-image `
  -H 2912 `
  -W 2912 `
  --rng cpu `
  --offload-to-cpu `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  --diffusion-model ../models/diffusion_models/p-h3-image_int8_convrot.safetensors \
  --llm ../models/text_encoders/qwen3vl_4b_bf16.safetensors \
  --embeddings-connectors ../models/text_encoders/p-h3-image_text_connector.safetensors \
  --vae ../models/vae/p-h3-image_vae_bf16.safetensors \
  -r source.png \
  -r reference.png \
  -p "Edit <Picture 1>. Give the cat the scarf from <Picture 2>, preserving its face and the background." \
  --cfg-scale 1.0 \
  --steps 8 \
  --sampling-method euler \
  --sigmas p-h3-image \
  -H 2912 \
  -W 2912 \
  --rng cpu \
  --offload-to-cpu \
  -v
```

Reference images are numbered in argument order as `<Picture 1>`, `<Picture 2>`, and so on. A plain edit instruction gets `Edit <Picture 1>. ` prepended automatically. Prompts starting with `Edit <Picture N>` or `subject_definitions:` are used directly.

Use dimensions divisible by 32. The upstream default is 2912 × 2912; smaller canvases reduce memory use. Use CFG 1 and `--sigmas p-h3-image` for the upstream 8-step schedule. Keep the default VAE tiling size of 256 pixels.
