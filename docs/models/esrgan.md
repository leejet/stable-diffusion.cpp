# ESRGAN and MoSR

ESRGAN and compatible MoSR weights upscale images. **Chaining generation and
upscaling is currently supported only through DGL**, as shown below. Standalone
CLI upscaling remains available with `-M upscale --upscale-model FILE -i INPUT`.

## Model files

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| Base model | [Stable Diffusion checkpoint](sd.md) | `-m` / `--model` | Required for generation below |
| ESRGAN upscaler | [RealESRGAN_x4plus_anime_6B.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth) | `load_upscaler` (DGL) / `--upscale-model` (standalone CLI) | Choose a compatible upscaler |
| MoSR upscaler | [4xNomos2_hq_mosr.pth](https://github.com/Phhofm/models/releases/download/4xNomos2_hq_mosr/4xNomos2_hq_mosr.pth) | `load_upscaler` (DGL) / `--upscale-model` (standalone CLI) | Alternative; requires the PixelShuffle upsampler |

## Usage

### DGL

| Example | Description |
| --- | --- |
| [Generate and upscale](../dgl/esrgan/sd1-5-txt2img-esrgan.dgl) | SD 1.5 generation followed by RealESRGAN. |
| [Upscale an image](../dgl/esrgan/realesrgan-x4plus-anime-6b-upscale.dgl) | Run RealESRGAN on an existing image. |

#### Run a script

Edit paths, prompts and settings in the script's `var` block, then run from
the build directory. Paths inside the script resolve from the working directory.

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe -M dgl --script ..\docs\dgl\esrgan\sd1-5-txt2img-esrgan.dgl
```

**Linux/macOS**

```sh
./bin/sd-cli -M dgl --script ../docs/dgl/esrgan/sd1-5-txt2img-esrgan.dgl
```

Choose another script from the table to run a different task. Add `--check`
to validate syntax and types without loading weights. See the
[shared run instructions](../dgl/README.md#run-an-example) for build paths
and the server's file-access rules.

## Notes and limitations

RGBA output retains its alpha channel during model upscaling and hires fix.
The upscaler processes RGB; alpha is resized with bilinear interpolation and
recombined with the result. The model architecture is detected from the weights.
