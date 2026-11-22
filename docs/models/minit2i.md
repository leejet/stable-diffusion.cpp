# MiniT2I

MiniT2I uses a MiniT2I diffusion transformer and `google/flan-t5-large` as the text encoder.

## Model files

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| MiniT2I diffusion model | [Safetensors](https://huggingface.co/MiniT2I/MiniT2I/tree/main/minit2i-b-16/transformer) (`diffusion_pytorch_model.safetensors`) | `--diffusion-model` | Required |
| flan-t5-large text encoder | [Safetensors](https://huggingface.co/google/flan-t5-large/tree/main) (`model.safetensors`) | `--t5xxl` | Required |

## Usage

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

### DGL

| Example | Description |
| --- | --- |
| [CUDA](../dgl/minit2i/minit2i-cuda-txt2img.dgl) | Text-to-image generation on CUDA. |
| [Metal](../dgl/minit2i/minit2i-metal-txt2img.dgl) | Run with `--threads 8`. |

#### Run a script

Edit paths, prompts and settings in the script's `var` block, then run from
the build directory. Paths inside the script resolve from the working directory.

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe -M dgl --script ..\docs\dgl\minit2i\minit2i-cuda-txt2img.dgl
```

**Linux/macOS**

```sh
./bin/sd-cli -M dgl --script ../docs/dgl/minit2i/minit2i-cuda-txt2img.dgl
```

Choose another script from the table to run a different task. Add `--check`
to validate syntax and types without loading weights. See the
[shared run instructions](../dgl/README.md#run-an-example) for build paths
and the server's file-access rules.

### Legacy CLI

CLI examples use the [shared path and shell conventions](README.md#example-conventions).

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
