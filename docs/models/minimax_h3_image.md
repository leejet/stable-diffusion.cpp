# MiniMax-H3-Image

MiniMax-H3-Image supports text-to-image generation and reference-image editing
with the same checkpoint. It reuses the MiniMax-H3 text encoder and uses a
dedicated image VAE.

## Model files

Pass the three MiniMax-H3-Image components separately:

- `--diffusion-model`: MiniMax-H3-Image diffusion transformer
- `--vae`: MiniMax-H3-Image VAE
- `--llm`: the MiniMax-H3 Qwen3-VL-32B text encoder checkpoint, or a Qwen3-VL-4B/8B with a text connector

Use the MiniMax-H3 text encoder, including its Qwen3-VL vision tower and
DeepStack mergers. An existing MiniMax-H3 text encoder can be reused. If the
vision tower is stored separately, pass it with `--llm_vision`.

The P-H3-Image text encoder can be used instead: Qwen3-VL-4B with the P-H3
text connector. It needs much less memory (about 5 GB for the INT8 ConvRot
file, versus 18 GB for the 32B encoder in Q4_K_M) and works for both
text-to-image and reference-image editing. Pass the P-H3-Image text encoder,
which includes the connector, with `--llm`. Alternatively, pass Qwen3-VL-4B
with `--llm` and the [separate text connector](https://huggingface.co/PrunaAI/P-H3-Image/blob/main/text_connector/diffusion_pytorch_model.safetensors)
with `--embeddings-connectors`. Reference-image editing needs the Qwen3-VL-4B
vision tower in the text encoder or in `--llm_vision`.

A [ClipProj projection](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3)
can replace the 32B encoder with a plain Qwen3-VL-4B or Qwen3-VL-8B: pass the
small encoder with `--llm` and the matching matrix with
`--embeddings-connectors`. It needs much less memory (about 4.5 GB with a 4B
INT8 encoder, versus 18 GB for the 32B encoder in Q4_K_M). The `mmh3-4b-*`
matrices expect a Qwen3-VL-4B encoder and the `mmh3-8b-*` matrices a
Qwen3-VL-8B encoder; match the two. Reference-image editing needs the encoder's
vision tower in the text encoder or in `--llm_vision`. ClipProj is a community
proof of concept: outputs are close to, not identical to, the 32B encoder. See
the [model card](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3) for
measurements and limitations.

Use a single-file diffusion checkpoint with the six-step PDD output heads
included, such as `minimax_h3_image_pruned_bf16.safetensors` or its INT8 ConvRot
version. Use the MiniMax-H3-Image VAE; the MiniMax-H3 video VAE and P-H3-Image
VAE are incompatible. No audio VAE is needed.

### Download weights

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| minimax_h3_image_pruned | [Safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/tree/main/diffusion_models) | `--diffusion-model` | Required |
| qwen3vl_32b_minimax_h3 | [Safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/tree/main/text_encoders)<br>[GGUF](https://huggingface.co/leejet/MiniMax-H3-GGUF/tree/main) | `--llm` | Use matching text encoder |
| P-H3-Image text encoder (Qwen3-VL-4B with the text connector) | [Safetensors](https://huggingface.co/PrunaAI/P-H3-Image-Comfy/tree/main/text_encoders) | `--llm` | Use matching text encoder |
| ClipProj projection (for a plain Qwen3-VL-4B or Qwen3-VL-8B) | [Safetensors](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3) | `--embeddings-connectors` | Pair with a matching Qwen3-VL-4B/8B `--llm` |
| vae | [Safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/tree/main/vae) | `--vae` | Use matching VAE |

## Usage

### Web UI

Follow the [Web UI guide](../../examples/server/README.md#web-ui) to start the
server, then select the recipe matching your weights. Fill the component slots
below and click **Load models**.
Use the files and compatibility requirements in [Model files](#model-files).

| Recipe | Component slots |
| --- | --- |
| MiniMax H3 Image (`minimax-h3-image`) | `diffusion_model`, `llm`, `llm_vision`, `embeddings_connectors`, `vae` |

Slot IDs correspond to CLI component options: `diffusion_model` uses
`--diffusion-model`, `llm` uses `--llm`, and `model` uses `-m` / `--model`.
Select optional components when required by your checkpoint or task.

### DGL

| Example | Description |
| --- | --- |
| [Text to image](../dgl/minimax_h3_image/minimax-h3-image-txt2img.dgl) | MiniMax-H3-Image generation. |
| [Image editing](../dgl/minimax_h3_image/minimax-h3-image-edit.dgl) | Edit a reference image. |

#### Run a script

Edit paths, prompts and settings in the script's `var` block, then run from
the build directory. Paths inside the script resolve from the working directory.

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe -M dgl --script ..\docs\dgl\minimax_h3_image\minimax-h3-image-txt2img.dgl
```

**Linux/macOS**

```sh
./bin/sd-cli -M dgl --script ../docs/dgl/minimax_h3_image/minimax-h3-image-txt2img.dgl
```

Choose another script from the table to run a different task. Add `--check`
to validate syntax and types without loading weights. See the
[shared run instructions](../dgl/README.md#run-an-example) for build paths
and the server's file-access rules.

### Legacy CLI

CLI examples use the [shared path and shell conventions](README.md#example-conventions).

#### Text-to-image

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  --diffusion-model ..\models\diffusion_models\minimax_h3_image_pruned_bf16.safetensors `
  --vae ..\models\vae\minimax_h3_image_vae_fp16.safetensors `
  --llm ..\models\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf `
  -p "A silver tabby kitten holding a sign reading 'sd.cpp', natural daylight" `
  --cfg-scale 1.0 `
  --steps 6 `
  --sampling-method euler `
  --sigmas minimax-h3-image `
  -W 1024 `
  -H 1024 `
  --rng cpu `
  --offload-to-cpu `
  --diffusion-fa `
  --vae-tiling `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  --diffusion-model ../models/diffusion_models/minimax_h3_image_pruned_bf16.safetensors \
  --vae ../models/vae/minimax_h3_image_vae_fp16.safetensors \
  --llm ../models/text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  -p "A silver tabby kitten holding a sign reading 'sd.cpp', natural daylight" \
  --cfg-scale 1.0 \
  --steps 6 \
  --sampling-method euler \
  --sigmas minimax-h3-image \
  -W 1024 \
  -H 1024 \
  --rng cpu \
  --offload-to-cpu \
  --diffusion-fa \
  --vae-tiling \
  -v
```

With the P-H3-Image text encoder, replace the `--llm` argument:

**Windows (PowerShell) arguments**

```text
--llm ..\models\text_encoders\p-h3-image_text_encoder_int8_convrot.safetensors
```

**Linux/macOS arguments**

```text
--llm ../models/text_encoders/p-h3-image_text_encoder_int8_convrot.safetensors
```

With a ClipProj projection, replace the `--llm` argument with the small
encoder plus the matching matrix:

**Windows (PowerShell) arguments**

```text
--llm ..\models\text_encoders\qwen3vl_4b_bf16.safetensors `
  --embeddings-connectors ..\models\text_encoders\mmh3-4b-ClipProj-v3.1.safetensors
```

**Linux/macOS arguments**

```text
--llm ../models/text_encoders/qwen3vl_4b_bf16.safetensors \
  --embeddings-connectors ../models/text_encoders/mmh3-4b-ClipProj-v3.1.safetensors
```

Use Euler, CFG 1, and `--sigmas minimax-h3-image` for the model's six-step
schedule. The default discrete scheduler also produces this schedule with
`--steps 6`. Prompt expansion is not included.

#### Reference-image editing

Add `--ref-image` (or `-r`) for reference-image editing:

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  --diffusion-model ..\models\diffusion_models\minimax_h3_image_pruned_bf16.safetensors `
  --vae ..\models\vae\minimax_h3_image_vae_fp16.safetensors `
  --llm ..\models\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf `
  -p "Edit <Picture 1>. Put a red scarf on the cat, preserving its face and the background." `
  -r ..\inputs\ernie_image\turbo_example.png `
  --cfg-scale 1.0 `
  --steps 6 `
  --sampling-method euler `
  --sigmas minimax-h3-image `
  -W 1024 `
  -H 1024 `
  --rng cpu `
  --offload-to-cpu `
  --diffusion-fa `
  --vae-tiling `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  --diffusion-model ../models/diffusion_models/minimax_h3_image_pruned_bf16.safetensors \
  --vae ../models/vae/minimax_h3_image_vae_fp16.safetensors \
  --llm ../models/text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  -p "Edit <Picture 1>. Put a red scarf on the cat, preserving its face and the background." \
  -r ../inputs/ernie_image/turbo_example.png \
  --cfg-scale 1.0 \
  --steps 6 \
  --sampling-method euler \
  --sigmas minimax-h3-image \
  -W 1024 \
  -H 1024 \
  --rng cpu \
  --offload-to-cpu \
  --diffusion-fa \
  --vae-tiling \
  -v
```

`--ref-image` can be repeated. Reference images are numbered in argument order
as `<Picture 1>`, `<Picture 2>`, and so on, with up to 16 references supported
by the model. Prompts are passed through without an automatic editing prefix.

Reference images keep their aspect ratio. The text encoder sees a 1024-pixel
short edge, while VAE inputs use a short edge between 1024 and 2048 pixels.
Small references are enlarged and dimensions are rounded to multiples of 32.

## Notes and limitations

- Width and height are aligned upward to a multiple of 32.
- Start at 1024 × 1024. Larger canvases and more references increase memory use.
- Use quantized weights, `--offload-to-cpu`, and `--vae-tiling` to reduce GPU
  memory requirements.
