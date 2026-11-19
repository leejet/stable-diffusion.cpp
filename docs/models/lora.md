# LoRA

LoRA weights adapt a compatible base model. Select a LoRA by name and strength
using `<lora:NAME:STRENGTH>` in the prompt.

## Model files

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| Base model | [Compatible model checkpoint](README.md) | `-m` / model component options | Required |
| LoRA weights | `marblesh.safetensors` or `marblesh.ckpt` for the example | `--lora-model-dir` and `<lora:NAME:STRENGTH>` in the prompt | Match the base model family |

## Usage

CLI examples use the [shared path and shell conventions](README.md#example-conventions).

### Web UI

Load a compatible base model with its recipe, then select the LoRA for generation.
See [Web UI guide](../../examples/server/README.md#web-ui) for the shared workflow.

### CLI

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -m ..\models\checkpoints\v1-5-pruned-emaonly.safetensors `
  --lora-model-dir ..\models\loras `
  -p "a lovely cat<lora:marblesh:1>"
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -m ../models/checkpoints/v1-5-pruned-emaonly.safetensors \
  --lora-model-dir ../models/loras \
  -p "a lovely cat<lora:marblesh:1>"
```

The example loads `../models/loras/marblesh.safetensors` or
`../models/loras/marblesh.ckpt` (use backslashes on Windows).

## Options

### LoRA apply mode

There are two ways to apply LoRA: **immediately** and **at_runtime**. You can specify it using the `--lora-apply-mode` parameter.

By default, the mode is selected automatically:

* If the model weights contain any quantized parameters, the **at_runtime** mode is used;
* Otherwise, the **immediately** mode is used.

The **immediately** mode may have precision and compatibility issues with quantized parameters, but it usually offers faster inference speed and, in some cases, lower memory usage.
In contrast, the **at_runtime** mode provides better compatibility and higher precision, but inference may be slower and memory usage may be higher in some cases.

## Notes and limitations

Choose LoRA weights trained for the base model. See the
[WebUI-style LoRA syntax](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#lora).
