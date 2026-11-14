# MiniT2I

MiniT2I uses a MiniT2I diffusion transformer and `google/flan-t5-large` as the text encoder.

## Model files

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| MiniT2I diffusion model | [Safetensors](https://huggingface.co/MiniT2I/MiniT2I/tree/main/minit2i-b-16/transformer) (`diffusion_pytorch_model.safetensors`) | `--diffusion-model` | Required |
| flan-t5-large text encoder | [Safetensors](https://huggingface.co/google/flan-t5-large/tree/main) (`model.safetensors`) | `--t5xxl` | Required |

## Usage

CLI examples use the [shared path and shell conventions](README.md#example-conventions).

### Web UI

Follow the [Web UI guide](../../examples/server/README.md#web-ui) to start the
server, then select the recipe matching your weights. Fill the component slots
below and click **Load models**.
Use the files and compatibility requirements in [Model files](#model-files).

| Recipe | Component slots |
| --- | --- |
| MiniT2I (`minit2i`) | `diffusion_model`, `t5xxl` |

Slot IDs correspond to CLI component options: `diffusion_model` uses
`--diffusion-model`, `llm` uses `--llm`, and `model` uses `-m` / `--model`.
Select optional components when required by your checkpoint or task.

### CLI

#### CUDA

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  --diffusion-model ..\models\diffusion_models\minit2i\diffusion_pytorch_model.safetensors `
  --t5xxl ..\models\text_encoders\flan-t5-large\model.safetensors `
  --prompt "a cat" `
  --steps 100 `
  --cfg-scale 6 `
  --sampling-method euler `
  --width 512 `
  --height 512 `
  --seed 42 `
  --output minit2i_cuda.png
```

**Linux/macOS**

```sh
./bin/sd-cli \
  --diffusion-model ../models/diffusion_models/minit2i/diffusion_pytorch_model.safetensors \
  --t5xxl ../models/text_encoders/flan-t5-large/model.safetensors \
  --prompt "a cat" \
  --steps 100 \
  --cfg-scale 6 \
  --sampling-method euler \
  --width 512 \
  --height 512 \
  --seed 42 \
  --output minit2i_cuda.png
```

#### macOS with Metal

**macOS (Metal)**

```sh
./bin/sd-cli \
  --diffusion-model ../models/diffusion_models/minit2i/diffusion_pytorch_model.safetensors \
  --t5xxl ../models/text_encoders/flan-t5-large/model.safetensors \
  --prompt "a cat" \
  --steps 100 \
  --cfg-scale 6 \
  --sampling-method euler \
  --width 512 \
  --height 512 \
  --rng cpu \
  --threads 8 \
  --backend metal \
  --seed 42 \
  --output minit2i_metal.png
```
