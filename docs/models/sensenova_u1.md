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

### Web UI

Follow the server instructions for
[sharded checkpoints](../../examples/server/README.md#sharded-checkpoints).
Pass the complete repository directory with `--model` when starting the server,
using the same model path as the CLI examples below. The browser then uses
**Command-line model (all features)** for that loaded checkpoint.

### DGL

| Example | Description |
| --- | --- |
| [Text to image](../dgl/sensenova_u1/sensenova-u1-cuda-txt2img.dgl) | SenseNova U1.5 on CUDA. |

#### Run a script

Edit paths, prompts and settings in the script's `var` block, then run from
the build directory. Paths inside the script resolve from the working directory.

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe -M dgl --script ..\docs\dgl\sensenova_u1\sensenova-u1-cuda-txt2img.dgl
```

**Linux/macOS**

```sh
./bin/sd-cli -M dgl --script ../docs/dgl/sensenova_u1/sensenova-u1-cuda-txt2img.dgl
```

Choose another script from the table to run a different task. Add `--check`
to validate syntax and types without loading weights. See the
[shared run instructions](../dgl/README.md#run-an-example) for build paths
and the server's file-access rules.

#### Prompt formatting

The [DGL example](../dgl/sensenova_u1/sensenova-u1-cuda-txt2img.dgl)
uses the default system message for its positive prompt and passes the complete
negative input with `raw=true`. Edit the user text inside that raw string to
change its negative prompt. An empty negative still includes the user/assistant
markers and `<img>`, and omits the positive template's `<think>` segment.
Native CLI/C API generation builds this input automatically.

For custom positive instructions, pass `system="your instruction"` to
`encode_prompt`; `system=""` removes the system segment but keeps the positive
assistant template. For a completely custom tokenizer input use `raw=true`
without `system`. See [DGL prompt formatting](../dgl.md#independent-generation-components).

### Legacy CLI

CLI examples use the [shared path and shell conventions](README.md#example-conventions).

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
