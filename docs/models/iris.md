# Iris-3B

Iris-3B generates images directly in pixel space and uses Qwen3-VL-4B-Instruct as the text encoder. No VAE is required.

## Model files

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| Iris-3B | [Safetensors](https://huggingface.co/speridlabs/iris-3b/tree/main) (`model.safetensors` in the root directory) | `--diffusion-model` | Required |
| Qwen3-VL-4B-Instruct | [Safetensors](https://huggingface.co/Comfy-Org/Krea-2/tree/main/text_encoders)<br>[GGUF](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/tree/main) | `--llm` | Use matching text encoder |

## Usage

CLI examples use the [shared path and shell conventions](README.md#example-conventions).

### Web UI

Follow the [Web UI guide](../../examples/server/README.md#web-ui) to start the
server, then select the recipe matching your weights. Fill the component slots
below and click **Load models**.
Use the files and compatibility requirements in [Model files](#model-files).

| Recipe | Component slots |
| --- | --- |
| Iris 3B (`iris-3b`) | `diffusion_model`, `llm` |

Slot IDs correspond to CLI component options: `diffusion_model` uses
`--diffusion-model`, `llm` uses `--llm`, and `model` uses `-m` / `--model`.
Select optional components when required by your checkpoint or task.

### CLI

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  --diffusion-model ..\models\diffusion_models\iris-3b.safetensors `
  --llm ..\models\text_encoders\Qwen3-VL-4B-Instruct-Q4_K_M.gguf `
  -p "a lovely cat" `
  --cfg-scale 3 `
  -H 1024 `
  -W 1024
```

**Linux/macOS**

```sh
./bin/sd-cli \
  --diffusion-model ../models/diffusion_models/iris-3b.safetensors \
  --llm ../models/text_encoders/Qwen3-VL-4B-Instruct-Q4_K_M.gguf \
  -p "a lovely cat" \
  --cfg-scale 3 \
  -H 1024 \
  -W 1024
```

<img width="256" alt="iris-3b example" src="../../assets/iris/example.png" />

## Notes and limitations

- Width and height must be multiples of 16. Do not pass `--vae`.
- Captions are limited to 300 tokens including the assistant-turn suffix. Positive and negative prompts use the same template.
