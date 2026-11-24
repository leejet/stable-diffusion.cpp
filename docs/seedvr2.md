# SeedVR2 image and video super-resolution

SeedVR2 upscales images and videos with `-M vid_upscale`. Support is experimental.

## Weights

These [ComfyUI safetensors weights](https://huggingface.co/Comfy-Org/SeedVR2)
load directly, without conversion or a separate text encoder:

| Model | `--diffusion-model` | `--vae` |
| --- | --- | --- |
| 3B | `seedvr2_3b_int8_convrot.safetensors` | `seedvr2_ema_vae_fp16.safetensors` |
| 7B Sharp | `seedvr2_7b_sharp_int8_convrot.safetensors` | `seedvr2_ema_vae_fp16.safetensors` |

Use the matching SeedVR2 VAE for either model. The 7B model requires more memory.

## Upscale an image

```sh
sd-cli -M vid_upscale \
  --diffusion-model models/diffusion_models/seedvr2_3b_int8_convrot.safetensors \
  --vae models/vae/seedvr2_ema_vae_fp16.safetensors \
  -i input.png --upscale-factor 2 -s 42 -o restored.png
```

`-i` / `--init-img` accepts a single image. Outputs can be PNG, JPEG, or WebP.
No FPS or temporal chunk setting is needed for an image.

## Upscale a video

Extract the video into numbered frames, then pass the frame directory:

```sh
mkdir frames
ffmpeg -i input.mp4 -fps_mode passthrough frames/%06d.png
sd-cli -M vid_upscale \
  --diffusion-model models/diffusion_models/seedvr2_3b_int8_convrot.safetensors \
  --vae models/vae/seedvr2_ema_vae_fp16.safetensors \
  --input-frames frames --upscale-factor 2 --fps 24 \
  --vsr-temporal-chunk-size 1 -s 42 -o restored.avi
```

Set `--fps` to the source video's frame rate; 24 is only the example above.
Frames are read in filename order. Use either `-i` or `--input-frames`, not both.

Video outputs support AVI, WebM, and animated WebP. To save individual frames,
use an output pattern such as `-o restored/%06d.png`.

## Settings and limits

- `--upscale-factor` is required and must be a positive integer. Output width
  and height are exactly the input dimensions multiplied by this value.
- `--vsr-temporal-chunk-size` controls how many video frames are processed
  together. Valid values are 1, 5, 9, 13, and so on; the default is 5. A value of
  1 processes each frame independently and uses less memory. Larger values use
  neighboring frames but require more memory. Chunk boundaries may be visible.
- Frame count and order are preserved, including a shorter final chunk. Audio
  is not copied to the output.
- If you run out of GPU memory, try `--vsr-temporal-chunk-size 1`. A five-frame
  chunk at 1728x960 output resolution exceeded 24 GB of GPU memory in testing.
- `--flash-attn auto` and `--conv-direct auto` are enabled by default. See
  [backend selection](backend.md) to choose a device or weight placement.
- SeedVR2 does not use sampler, scheduler, or step settings.
  `--vsr-window-size` is a RelayVSR option and does not affect SeedVR2.

3B has been tested on CUDA and CPU; 7B has been tested on CUDA with small inputs.
Other backends have not been tested with this integration.

Model reference: [ByteDance-Seed/SeedVR](https://github.com/ByteDance-Seed/SeedVR).
