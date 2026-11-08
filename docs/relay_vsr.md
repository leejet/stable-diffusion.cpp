# RelayVSR image and video super-resolution

[RelayVSR](https://github.com/kopperx/RelayVSR) upscales images and videos by
2x or 4x with `-M vid_upscale`.

## Weights

[ComfyUI safetensors weights](https://huggingface.co/Comfy-Org/RelayVSR) load
directly. Select the matching pair for the requested scale:

| `--upscale-factor` | `--diffusion-model` | `--vae` |
| --- | --- | --- |
| 2 | `relayvsr_x2_lightx2v_int8_convrot.safetensors` | `relayvsr_flash_vae_x2_fp32.safetensors` |
| 4 | `relayvsr_x4_int8_convrot.safetensors` | `relayvsr_flash_vae_x4_fp32.safetensors` |

No separate text encoder or ordinary Wan VAE is needed.

## Usage

For a video, extract numbered frames into a directory, then run:

```sh
sd-cli -M vid_upscale \
  --diffusion-model models/diffusion_models/relayvsr_x2_lightx2v_int8_convrot.safetensors \
  --vae models/vae/relayvsr_flash_vae_x2_fp32.safetensors \
  --input-frames input_frames --upscale-factor 2 --fps 24 \
  --vsr-window-size 16 -s 42 -o output.avi
```

Frames are read in filename order. Set `--fps` to the source video's frame rate.
Video outputs support AVI, WebM, and animated WebP. To save individual frames,
use an output pattern such as `-o output/%04d.png`.

For a single image, replace `--input-frames input_frames` with `-i input.png`
and use `-o output.png` (or JPEG/WebP). No FPS setting is needed for an image.
`-i` / `--init-img` and `--input-frames` are mutually exclusive.

## Settings and limits

- `--upscale-factor` is required. Use 2 or 4 with the corresponding weights.
  Output width and height are exactly the input dimensions times the scale.
- `--vsr-window-size` sets the reference-frame window, including both endpoints.
  The default 16 selects reference frames 0, 15, 30, and so on, and always
  includes the final frame. Every input frame appears once in the output.
- `--flash-attn auto` and `--conv-direct auto` are enabled by default. See
  [backend selection](backend.md) to choose a device or weight placement.
- Sampler, scheduler, and step settings do not configure RelayVSR.
  `--vsr-temporal-chunk-size` is a SeedVR2 option.
- Long videos require enough system memory to hold the input and output frames.
  Audio is not copied to the output.
