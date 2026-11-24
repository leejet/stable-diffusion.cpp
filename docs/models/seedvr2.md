# SeedVR2

SeedVR2 upscales images and videos through dgl. Support is experimental.

**Currently supported only through dgl.**

## Model files

| Component | Weights and downloads | DGL usage | Requirement |
| --- | --- | --- | --- |
| Diffusion model | [ComfyUI SeedVR2](https://huggingface.co/Comfy-Org/SeedVR2) | `load_dit` | Choose 3B or 7B Sharp below |
| SeedVR2 VAE | `seedvr2_ema_vae_fp16.safetensors` from the same repository | `load_vae` | Required; shared by both variants |

These [ComfyUI safetensors weights](https://huggingface.co/Comfy-Org/SeedVR2)
load directly, without conversion or a separate text encoder:

| Model | `load_dit` | `load_vae` |
| --- | --- | --- |
| 3B | `seedvr2_3b_int8_convrot.safetensors` | `seedvr2_ema_vae_fp16.safetensors` |
| 7B Sharp | `seedvr2_7b_sharp_int8_convrot.safetensors` | `seedvr2_ema_vae_fp16.safetensors` |

Use the matching SeedVR2 VAE for either model. The 7B model requires more memory.

## Usage

### DGL

| Example | Description |
| --- | --- |
| [Upscale an image](../dgl/seedvr2/seedvr2-3b-image-upscale.dgl) | Use the 3B diffusion model and matching SeedVR2 VAE. |
| [Upscale a video](../dgl/seedvr2/seedvr2-3b-video-upscale.dgl) | Process frame windows and retain the source audio. |

#### Run a script

Edit paths, prompts and settings in the script's `var` block, then run from
the build directory. Paths inside the script resolve from the working directory.

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe -M dgl --script ..\docs\dgl\seedvr2\seedvr2-3b-image-upscale.dgl
```

**Linux/macOS**

```sh
./bin/sd-cli -M dgl --script ../docs/dgl/seedvr2/seedvr2-3b-image-upscale.dgl
```

Choose another script from the table to run a different task. Add `--check`
to validate syntax and types without loading weights. See the
[shared run instructions](../dgl/README.md#run-an-example) for build paths
and the server's file-access rules.

The [SeedVR2 script](../dgl/seedvr2/seedvr2-3b-video-upscale.dgl) loads the DiT and
causal VAE independently. The script controls output resolution with
bicubic resizing and pads each window to dimensions divisible by 16 and
a frame count of `4n+1`. It uses the VAE posterior mode, the checkpoint's
positive embedding, and a one-step Euler sampler. The condition contains
the scaled VAE latent and a one-channel task mask. Each decoded window is
cropped to its original frame count and target dimensions before stitching.
The intermediate float tensors must fit the DGL tensor size limit of 256 MiB
per tensor. For large output resolutions, use shorter pixel windows.

CLI examples use the [shared path and shell conventions](README.md#example-conventions).

#### Upscale an image

The image script reads `input_path` and writes a PNG with `image::save`.

#### Upscale a video

Set `input_path` to numbered frames or a video supported by `video::load`.

Set `fps` to the desired frame rate; the example uses 24. Frame directories
are read in filename order. Video output is AVI, and the script passes the
source audio separately to `video::save`.

## Notes and limitations

- `factor` is a positive integer scale applied to pixel dimensions before
  VAE encoding. Both 3B and 7B Sharp use the matching SeedVR2 VAE.
- `window_size` controls processing granularity. The video example uses 1 to
  process each frame independently and reduce memory; 5, 9, 13, … include
  temporal context at higher memory cost. Each window is padded to `4n+1`
  frames and cropped after decoding. Chunk boundaries may be visible.
- Sampling uses the checkpoint's positive embedding and an explicit one-step
  Euler flow; no text encoder is needed.
- Input/output order and frame count are preserved. Input, output and
  intermediate tensors must fit the 256 MiB per-tensor limit. For larger
  output frames, reduce window size; long clips need incremental I/O.
- The existing DGL workflow was tested with 3B and 7B Sharp on Vulkan. The
  documentation examples in this collection are also statically checked;
  this does not establish image quality or throughput on every backend.

Model reference: [ByteDance-Seed/SeedVR](https://github.com/ByteDance-Seed/SeedVR).
