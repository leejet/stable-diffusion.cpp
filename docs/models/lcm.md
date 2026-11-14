# LCM and LCM-LoRA

LCM checkpoints and LCM-LoRAs support generation with a small number of steps.
The example below applies an LCM-LoRA to SD 1.5.

## Model files

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| Base model | [Stable Diffusion 1.5 checkpoint](sd.md) | `-m` / `--model` | Required for the SD 1.5 LoRA example |
| LCM-LoRA | [LCM-LoRA SD 1.5](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5) | `--lora-model-dir` and prompt | Match the LoRA to the base model |

## Usage

CLI examples use the [shared path and shell conventions](README.md#example-conventions).

### Web UI

Follow the [Web UI guide](../../examples/server/README.md#web-ui) to start the
server, then select the recipe matching your weights. Fill the component slots
below and click **Load models**.
Use the files and compatibility requirements in [Model files](#model-files).

| Recipe | Component slots |
| --- | --- |
| SD 1.x (`sd1-checkpoint`) | `model`, `vae`, `tae` |
| SDXL (`sdxl-checkpoint`) | `model`, `vae`, `tae` |

Slot IDs correspond to CLI component options: `diffusion_model` uses
`--diffusion-model`, `llm` uses `--llm`, and `model` uses `-m` / `--model`.
Select optional components when required by your checkpoint or task.

Select the matching LCM checkpoint or apply a compatible LCM-LoRA, then set the
sampling method, CFG, and steps for that model.

### CLI

#### LCM-LoRA with SD 1.5

Add `<lora:lcm-lora-sdv1-5:1>` to the prompt:

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -m ..\models\checkpoints\v1-5-pruned-emaonly.safetensors `
  --lora-model-dir ..\models\loras `
  -p "a lovely cat<lora:lcm-lora-sdv1-5:1>" `
  --steps 4 `
  --cfg-scale 1 `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -m ../models/checkpoints/v1-5-pruned-emaonly.safetensors \
  --lora-model-dir ../models/loras \
  -p "a lovely cat<lora:lcm-lora-sdv1-5:1>" \
  --steps 4 \
  --cfg-scale 1 \
  -v
```

#### LCM-LoRA comparison

| without LCM-LoRA (--cfg-scale 7)  | with LCM-LoRA (--cfg-scale 1)  |
| ----  |----    |
| ![](../../assets/without_lcm.png) |![](../../assets/with_lcm.png)  |

## Options

Use `--cfg-scale 1.0` and 2-8 steps for the LCM-LoRA workflow. The original
example uses `euler_a`; `lcm` sampling is also supported. See
[sampling methods and schedulers](../guides/sampling.md) for their defaults.
