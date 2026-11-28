# Chroma1-Radiance

Chroma1-Radiance generates images directly in pixel space using T5-XXL. No VAE is required.

## Model files

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| Chroma1-Radiance | [Safetensors](https://huggingface.co/lodestones/Chroma1-Radiance/tree/main)<br>[GGUF](https://huggingface.co/silveroxides/Chroma1-Radiance-GGUF/tree/main) | `--diffusion-model` | Required |
| t5xxl | [Safetensors](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp16.safetensors) | `--t5xxl` | Required |

## Usage

### Web UI

Follow the [Web UI guide](../../examples/server/README.md#web-ui) to start the
server, then select the recipe matching your weights. Fill the component slots
below and click **Load models**.
Use the files and compatibility requirements in [Model files](#model-files).

| Recipe | Component slots |
| --- | --- |
| Chroma Radiance (`chroma-radiance`) | `diffusion_model`, `t5xxl` |

Slot IDs correspond to CLI component options: `diffusion_model` uses
`--diffusion-model`, `llm` uses `--llm`, and `model` uses `-m` / `--model`.
Select optional components when required by your checkpoint or task.

### DGL

| Example | Description |
| --- | --- |
| [Text to image](../dgl/chroma_radiance/chroma1-radiance-v0-4-txt2img.dgl) | Generate pixels directly, without a VAE. |

#### Run a script

Edit paths, prompts and settings in the script's `var` block, then run from
the build directory. Paths inside the script resolve from the working directory.

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe -M dgl --script ..\docs\dgl\chroma_radiance\chroma1-radiance-v0-4-txt2img.dgl
```

**Linux/macOS**

```sh
./bin/sd-cli -M dgl --script ../docs/dgl/chroma_radiance/chroma1-radiance-v0-4-txt2img.dgl
```

Choose another script from the table to run a different task. Add `--check`
to validate syntax and types without loading weights. See the
[shared run instructions](../dgl/README.md#run-an-example) for build paths
and the server's file-access rules.

### Legacy CLI

CLI examples use the [shared path and shell conventions](README.md#example-conventions).

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  --diffusion-model ..\models\diffusion_models\Chroma1-Radiance-v0.4-Q8_0.gguf `
  --t5xxl ..\models\text_encoders\clip\t5xxl_fp16.safetensors `
  -p "a lovely cat holding a sign says 'chroma  radiance cpp'" `
  --cfg-scale 4.0 `
  --sampling-method euler `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  --diffusion-model ../models/diffusion_models/Chroma1-Radiance-v0.4-Q8_0.gguf \
  --t5xxl ../models/text_encoders/clip/t5xxl_fp16.safetensors \
  -p "a lovely cat holding a sign says 'chroma  radiance cpp'" \
  --cfg-scale 4.0 \
  --sampling-method euler \
  -v
```

<img alt="Chroma1-Radiance" src="../../assets/flux/chroma1-radiance.png" />
