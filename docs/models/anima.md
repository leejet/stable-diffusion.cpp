# Anima

Anima generates images from text using a Qwen3 text encoder and a compatible Qwen Image VAE.

## Model files

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| Anima | [Safetensors](https://huggingface.co/circlestone-labs/Anima/tree/main/split_files/diffusion_models)<br>[GGUF](https://huggingface.co/Bedovyy/Anima-GGUF/tree/main)<br>gguf Anima2: [GGUF](https://huggingface.co/JusteLeo/Anima2-GGUF/tree/main) | `--diffusion-model` | Required |
| vae | [Safetensors](https://huggingface.co/circlestone-labs/Anima/tree/main/split_files/vae) | `--vae` | Use matching VAE |
| Qwen3-0.6B-Base | [Safetensors](https://huggingface.co/circlestone-labs/Anima/tree/main/split_files/text_encoders)<br>[GGUF](https://huggingface.co/mradermacher/Qwen3-0.6B-Base-GGUF/tree/main) | `--llm` | Use matching text encoder |

## Usage

### Web UI

Follow the [Web UI guide](../../examples/server/README.md#web-ui) to start the
server, then select the recipe matching your weights. Fill the component slots
below and click **Load models**.
Use the files and compatibility requirements in [Model files](#model-files).

| Recipe | Component slots |
| --- | --- |
| Anima (`anima`) | `diffusion_model`, `llm`, `vae`, `tae` |
| Anima Edit (`anima-edit`) | `diffusion_model`, `llm`, `vae`, `tae` |

Slot IDs correspond to CLI component options: `diffusion_model` uses
`--diffusion-model`, `llm` uses `--llm`, and `model` uses `-m` / `--model`.
Select optional components when required by your checkpoint or task.

For Anima Edit, select a matching community editing LoRA separately and provide
reference images. The references are VAE-encoded for diffusion conditioning;
selecting the recipe does not load the LoRA automatically.

### DGL

| Example | Description |
| --- | --- |
| [Text to image](../dgl/anima/anima-preview-txt2img.dgl) | Anima Preview. |

#### Run a script

Edit paths, prompts and settings in the script's `var` block, then run from
the build directory. Paths inside the script resolve from the working directory.

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe -M dgl --script ..\docs\dgl\anima\anima-preview-txt2img.dgl
```

**Linux/macOS**

```sh
./bin/sd-cli -M dgl --script ../docs/dgl/anima/anima-preview-txt2img.dgl
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
  --diffusion-model ..\models\diffusion_models\anima-preview.safetensors `
  --vae ..\models\vae\qwen_image_vae.safetensors `
  --llm ..\models\text_encoders\qwen_3_06b_base.safetensors `
  -p "a lovely cat holding a sign says 'anima.cpp'" `
  --cfg-scale 6.0 `
  --sampling-method euler `
  --offload-to-cpu `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  --diffusion-model ../models/diffusion_models/anima-preview.safetensors \
  --vae ../models/vae/qwen_image_vae.safetensors \
  --llm ../models/text_encoders/qwen_3_06b_base.safetensors \
  -p "a lovely cat holding a sign says 'anima.cpp'" \
  --cfg-scale 6.0 \
  --sampling-method euler \
  --offload-to-cpu \
  -v
```

<img alt="anima image example" src="../../assets/anima/example.png" />
