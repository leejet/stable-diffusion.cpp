# SenseNova U1.5

SenseNova U1.5 is an 8B MoT model that performs diffusion directly in RGB pixel
space. It does not require a separate text encoder or VAE.

## Model files

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| SenseNova U1.5 8B MoT | [Safetensors](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT) | `--model` | Required |

Pass the complete downloaded repository directory to `--model`. The directory
must contain `model.safetensors.index.json`, every referenced Safetensors shard,
and the tokenizer files.

## Usage

CLI examples use the [shared path and shell conventions](README.md#example-conventions).

### Web UI

Follow the server instructions for
[sharded checkpoints](../../examples/server/README.md#sharded-checkpoints).
Pass the complete repository directory with `--model` when starting the server,
using the same model path as the CLI examples below. The browser then uses
**Command-line model (all features)** for that loaded checkpoint.

### CLI

#### CUDA

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  --model ..\models\checkpoints\SenseNova-U1.5-8B-MoT `
  --prompt "a red cube on a white background" `
  --steps 50 `
  --cfg-scale 4 `
  --flow-shift 3 `
  --sampling-method euler `
  --width 2048 `
  --height 2048 `
  --rng cuda `
  --seed 42 `
  --output output.png
```

**Linux/macOS**

```sh
./bin/sd-cli \
  --model ../models/checkpoints/SenseNova-U1.5-8B-MoT \
  --prompt "a red cube on a white background" \
  --steps 50 \
  --cfg-scale 4 \
  --flow-shift 3 \
  --sampling-method euler \
  --width 2048 \
  --height 2048 \
  --rng cuda \
  --seed 42 \
  --output output.png
```

## Notes and limitations

- To match the official non-thinking text-to-image pipeline, use 50 Euler
  steps, CFG 4, flow shift 3, seed 42, CUDA RNG, and an empty negative prompt.
- Width and height must be multiples of 32. The trained 1:1 resolution is
  2048x2048; lower resolutions are useful for smoke tests but are outside the
  training buckets.
- The SenseNova prompt template and unconditional prompt are built
  automatically.
- This implementation supports non-thinking text-to-image generation. Image
  editing, visual understanding, interleaved generation, and thinking-mode
  prompt expansion are not implemented.
