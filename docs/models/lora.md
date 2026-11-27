# LoRA

LoRA weights adapt a compatible base model. Select a LoRA by name and strength
using `<lora:NAME:STRENGTH>` in the prompt.

## Model files

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| Base model | [Compatible model checkpoint](README.md) | `-m` / model component options | Required |
| LoRA weights | `marblesh.safetensors` or `marblesh.ckpt` for the example | `--lora-model-dir` and `<lora:NAME:STRENGTH>` in the prompt | Match the base model family |

## Usage

### Web UI

Load a compatible base model with its recipe, then select the LoRA for generation.
See [Web UI guide](../../examples/server/README.md#web-ui) for the shared workflow.

### DGL

| Example | Description |
| --- | --- |
| [SD 1.5 with LoRA](../dgl/lora/sd1-5-lora.dgl) | Apply the same LoRA to the UNet and text encoder. |

#### Run a script

Edit paths, prompts and settings in the script's `var` block, then run from
the build directory. Paths inside the script resolve from the working directory.

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe -M dgl --script ..\docs\dgl\lora\sd1-5-lora.dgl
```

**Linux/macOS**

```sh
./bin/sd-cli -M dgl --script ../docs/dgl/lora/sd1-5-lora.dgl
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

The default **auto** mode chooses **immediately** for unquantized and INT8
tensorwise weights on the compute backend without row splitting. It chooses
**at_runtime** for other quantized formats, offloaded weights, or row-split
weights. For scripts, see
[DGL LoRA combinations](../dgl.md#lora-combinations).

The **immediately** mode may have precision and compatibility issues with quantized parameters, but it usually offers faster inference speed and, in some cases, lower memory usage.
For INT8 tensorwise and convrot weights, **immediately** merges and requantizes
the affected weights. See the [INT8 convrot guide](../guides/int8_convrot.md).
In contrast, the **at_runtime** mode provides better compatibility and higher precision, but inference may be slower and memory usage may be higher in some cases.

## Notes and limitations

Choose LoRA weights trained for the base model. See the
[WebUI-style LoRA syntax](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#lora).
