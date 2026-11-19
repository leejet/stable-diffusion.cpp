# ESRGAN and MoSR

ESRGAN and compatible MoSR weights upscale generated images. Pass the chosen
upscaler with `--upscale-model`.

## Model files

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| Base model | [Stable Diffusion checkpoint](sd.md) | `-m` / `--model` | Required for generation below |
| ESRGAN upscaler | [RealESRGAN_x4plus_anime_6B.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth) | `--upscale-model` | Choose a compatible upscaler |
| MoSR upscaler | [4xNomos2_hq_mosr.pth](https://github.com/Phhofm/models/releases/download/4xNomos2_hq_mosr/4xNomos2_hq_mosr.pth) | `--upscale-model` | Alternative; requires the PixelShuffle upsampler |

## Usage

CLI examples use the [shared path and shell conventions](README.md#example-conventions).

### CLI

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -m ..\models\checkpoints\v1-5-pruned-emaonly.safetensors `
  --upscale-model ..\models\upscale_models\RealESRGAN_x4plus_anime_6B.pth `
  -p "a lovely cat"
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -m ../models/checkpoints/v1-5-pruned-emaonly.safetensors \
  --upscale-model ../models/upscale_models/RealESRGAN_x4plus_anime_6B.pth \
  -p "a lovely cat"
```

## Notes and limitations

RGBA output retains its alpha channel during model upscaling and hires fix.
The upscaler processes RGB; alpha is resized with bilinear interpolation and
recombined with the result. The model architecture is detected from the weights.
