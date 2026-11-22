# RelayVSR

[RelayVSR](https://github.com/kopperx/RelayVSR) upscales images and videos by
2x or 4x through separately loaded diffusion and VAE components.

**Currently supported only through dgl.**

## Model files

| Component | Weights and downloads | DGL usage | Requirement |
| --- | --- | --- | --- |
| RelayVSR diffusion model | [ComfyUI RelayVSR](https://huggingface.co/Comfy-Org/RelayVSR) | `load_dit` | Match the requested 2x or 4x scale |
| RelayVSR VAE | Matching scale-specific VAE from the same repository | `load_vae` | Required; ordinary Wan VAE is incompatible |

[ComfyUI safetensors weights](https://huggingface.co/Comfy-Org/RelayVSR) load
directly. Select the matching pair for the requested scale:

| Scale | `load_dit` | `load_vae` |
| --- | --- | --- |
| 2 | `relayvsr_x2_lightx2v_int8_convrot.safetensors` | `relayvsr_flash_vae_x2_fp32.safetensors` |
| 4 | `relayvsr_x4_int8_convrot.safetensors` | `relayvsr_flash_vae_x4_fp32.safetensors` |

No separate text encoder or ordinary Wan VAE is needed.

## Usage

### DGL

| Example | Description |
| --- | --- |
| [Upscale a video](../dgl/relay_vsr/relayvsr-x2-video-upscale.dgl) | Process overlapping windows with the 2x model. |
| [Upscale an image](../dgl/relay_vsr/relayvsr-x2-image-upscale.dgl) | Run the same 2x model on one image. |

#### Run a script

Edit paths, prompts and settings in the script's `var` block, then run from
the build directory. Paths inside the script resolve from the working directory.

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe -M dgl --script ..\docs\dgl\relay_vsr\relayvsr-x2-video-upscale.dgl
```

**Linux/macOS**

```sh
./bin/sd-cli -M dgl --script ../docs/dgl/relay_vsr/relayvsr-x2-video-upscale.dgl
```

Choose another script from the table to run a different task. Add `--check`
to validate syntax and types without loading weights. See the
[shared run instructions](../dgl/README.md#run-an-example) for build paths
and the server's file-access rules.

Set `input_path` to a numbered-frame directory or a video supported by
`video::load`.

Frame directories are read in filename order. Set the script's `fps` to the
desired output rate. `video::save` writes AVI and receives the original audio
separately. The [single-image script](../dgl/relay_vsr/relayvsr-x2-image-upscale.dgl)
uses `image::load` / `image::save` instead.

## Notes and limitations

- Choose matching 2× or 4× diffusion/VAE weights. Their architecture determines
  the output scale; ordinary Wan VAE weights are incompatible.
- `window_size = 16` splits the input into windows with one overlapping frame.
  The script samples endpoint keyframes for each window, decodes against the
  source frames, then joins windows by dropping duplicate overlap frames.
- Sampling is an explicit one-step flow with `[1, 0]` sigmas and Euler. There
  is no separate native video-upscale sampler configuration.
- Loader attention and convolution settings use their normal auto defaults.
  See [backend selection](../guides/backend.md) for placement.
- `video::load` limits decoded input to 256 MiB. Window outputs use the general
  1 GiB tensor limit, and joined video may use up to 4 GiB. Long clips or large
  frames need a more granular input/output program.
