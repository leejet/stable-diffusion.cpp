# Z-Image L2P

[L2P](https://nju-pcalab.github.io/projects/L2P/) (Latent-to-Pixel) generates images
directly in pixel space using Z-Image-Turbo and Qwen3 4B. No VAE is required.

## Model files

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| L2P (1K) | [Safetensors](https://huggingface.co/zhen-nan/L2P/tree/main) | `--diffusion-model` | Required |
| Qwen3 4b | [Safetensors](https://huggingface.co/Comfy-Org/z_image_turbo/tree/main/split_files/text_encoders)<br>[GGUF](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/tree/main) | `--llm` | Use matching text encoder |

## Usage

CLI examples use the [shared path and shell conventions](README.md#example-conventions).

### Web UI

Follow the [Web UI guide](../../examples/server/README.md#web-ui) to start the
server, then select the recipe matching your weights. Fill the component slots
below and click **Load models**.
Use the files and compatibility requirements in [Model files](#model-files).

| Recipe | Component slots |
| --- | --- |
| Z-Image L2P (`z-image-l2p`) | `diffusion_model`, `llm` |

Slot IDs correspond to CLI component options: `diffusion_model` uses
`--diffusion-model`, `llm` uses `--llm`, and `model` uses `-m` / `--model`.
Select optional components when required by your checkpoint or task.

### CLI

#### Text-to-image

Do not pass `--vae`. Image dimensions must be multiples of 16. The reference pipeline uses 30 steps with a CFG scale of 2.0.

The released checkpoint keeps part of the transformer in F32. Pass `--type bf16` to load it in BF16, as the reference pipeline runs it; this reduces the diffusion model from about 18.6 GB to 11.8 GB.

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  --diffusion-model ..\models\diffusion_models\model-1k-merge.safetensors `
  --llm ..\models\text_encoders\qwen_3_4b.safetensors `
  --type bf16 `
  -p "an origami pig on fire in the middle of a dark room with a pentagram on the floor" `
  --cfg-scale 2.0 `
  --steps 30 `
  --sampling-method euler `
  -W 1024 `
  -H 1024 `
  -v `
  -o z_image_l2p.png
```

**Linux/macOS**

```sh
./bin/sd-cli \
  --diffusion-model ../models/diffusion_models/model-1k-merge.safetensors \
  --llm ../models/text_encoders/qwen_3_4b.safetensors \
  --type bf16 \
  -p "an origami pig on fire in the middle of a dark room with a pentagram on the floor" \
  --cfg-scale 2.0 \
  --steps 30 \
  --sampling-method euler \
  -W 1024 \
  -H 1024 \
  -v \
  -o z_image_l2p.png
```
