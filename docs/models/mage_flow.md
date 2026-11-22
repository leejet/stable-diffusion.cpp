# Mage-Flow

[Mage-Flow](https://github.com/microsoft/Mage) uses a 4B native-resolution multimodal diffusion transformer, Qwen3-VL for text and image conditioning, and the 128-channel Mage-VAE. Both text-to-image and instruction-based image editing checkpoints are supported.

## Model files

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| Mage-Flow diffusion | Mage-Flow diffusion from [Download](https://huggingface.co/Comfy-Org/Mage-Flow/tree/main/diffusion_models) | `--diffusion-model` | Required |
| Mage-Flow vae | Mage-Flow vae from [Download](https://huggingface.co/Comfy-Org/Mage-Flow/tree/main/vae) | `--vae` | Use matching VAE |
| Qwen3-VL 4B | [Safetensors](https://huggingface.co/Comfy-Org/Krea-2/tree/main/text_encoders)<br>[GGUF](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/tree/main) | `--llm` | Use matching text encoder |

## Usage

### Web UI

Follow the [Web UI guide](../../examples/server/README.md#web-ui) to start the
server, then select the recipe matching your weights. Fill the component slots
below and click **Load models**.
Use the files and compatibility requirements in [Model files](#model-files).

| Recipe | Component slots |
| --- | --- |
| Mage Flow (`mage-flow`) | `diffusion_model`, `llm`, `vae` |
| Mage Flow Turbo (`mage-flow-turbo`) | `diffusion_model`, `llm`, `vae` |
| Mage Flow Edit (`mage-flow-edit`) | `diffusion_model`, `llm`, `vae`, `llm_vision` |

Slot IDs correspond to CLI component options: `diffusion_model` uses
`--diffusion-model`, `llm` uses `--llm`, and `model` uses `-m` / `--model`.
Select optional components when required by your checkpoint or task.

### DGL

| Example | Description |
| --- | --- |
| [Text to image](../dgl/mage_flow/mage-flow-turbo-txt2img.dgl) | Mage-Flow Turbo. |
| [Image editing](../dgl/mage_flow/mage-flow-edit.dgl) | Mage-Flow Edit. |

#### Run a script

Edit paths, prompts and settings in the script's `var` block, then run from
the build directory. Paths inside the script resolve from the working directory.

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe -M dgl --script ..\docs\dgl\mage_flow\mage-flow-turbo-txt2img.dgl
```

**Linux/macOS**

```sh
./bin/sd-cli -M dgl --script ../docs/dgl/mage_flow/mage-flow-turbo-txt2img.dgl
```

Choose another script from the table to run a different task. Add `--check`
to validate syntax and types without loading weights. See the
[shared run instructions](../dgl/README.md#run-an-example) for build paths
and the server's file-access rules.

### Legacy CLI

CLI examples use the [shared path and shell conventions](README.md#example-conventions).

#### Text-to-image

Use 30 steps for Base models and 4 steps with `--cfg-scale 1` for Turbo models. Image dimensions must be multiples of 16; the official checkpoints are trained for native resolutions from 512 to 2048 pixels.

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  --diffusion-model ..\models\diffusion_models\Mage-Flow-Turbo.safetensors `
  --llm ..\models\text_encoders\Qwen3-VL-4B-Instruct-Q4_K_M.gguf `
  --vae ..\models\vae\mage_vae.safetensors `
  -p "a lovely cat holding a sign says 'mage.cpp'" `
  --cfg-scale 1.0 `
  --steps 4 `
  --offload-to-cpu `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  --diffusion-model ../models/diffusion_models/Mage-Flow-Turbo.safetensors \
  --llm ../models/text_encoders/Qwen3-VL-4B-Instruct-Q4_K_M.gguf \
  --vae ../models/vae/mage_vae.safetensors \
  -p "a lovely cat holding a sign says 'mage.cpp'" \
  --cfg-scale 1.0 \
  --steps 4 \
  --offload-to-cpu \
  -v
```

<img width="256" alt="Mage-Flow example" src="../../assets/mage_flow/example.png" />

#### Image editing

Mage-Flow-Edit accepts one or more reference images. The CLI and C API send each
image to both Qwen3-VL and the diffusion transformer. The vision copy follows
the encoder's native preprocessing and patch alignment; the VAE copy is resized
to the requested output resolution. Use a [DGL script](../guides/edit.md#custom-reference-processing)
to customize these two inputs.

For the Turbo edit checkpoint, use 4 steps and `--cfg-scale 1`.

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  --diffusion-model ..\models\diffusion_models\Mage-Flow-Edit.safetensors `
  --llm ..\models\text_encoders\Qwen3-VL-4B-Instruct-Q4_K_M.gguf `
  --llm_vision ..\models\text_encoders\Qwen3-VL-4B-Instruct-mmproj-BF16.gguf `
  --vae ..\models\vae\mage_vae.safetensors `
  -r ..\inputs\flux\flux1-dev-q8_0.png `
  -p "change 'flux.cpp' to 'mage.cpp'" `
  --cfg-scale 4.0 `
  --sampling-method euler `
  --offload-to-cpu `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  --diffusion-model ../models/diffusion_models/Mage-Flow-Edit.safetensors \
  --llm ../models/text_encoders/Qwen3-VL-4B-Instruct-Q4_K_M.gguf \
  --llm_vision ../models/text_encoders/Qwen3-VL-4B-Instruct-mmproj-BF16.gguf \
  --vae ../models/vae/mage_vae.safetensors \
  -r ../inputs/flux/flux1-dev-q8_0.png \
  -p "change 'flux.cpp' to 'mage.cpp'" \
  --cfg-scale 4.0 \
  --sampling-method euler \
  --offload-to-cpu \
  -v
```

<img width="256" alt="Mage-Flow-Edit example" src="../../assets/mage_flow/edit_example.png" />
