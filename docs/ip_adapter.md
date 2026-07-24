# IP-Adapter

stable-diffusion.cpp supports [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)
image-prompt conditioning for SD 1.5 and SDXL. Given a reference image,
IP-Adapter transfers the subject and appearance of that image into the
generation, alongside the text prompt.

IP-Adapter encodes the reference image with a CLIP-Vision (ViT-H/14)
encoder, projects the embedding into a few image tokens, and injects them
through a decoupled cross-attention added to every attn2 layer of the
UNet. It composes with Control Net, so a reference image (appearance) and
an OpenPose hint (pose) can be combined in a single generation.

## Required weights

1. A base SD 1.5 or SDXL model.
2. A CLIP-Vision (ViT-H/14) image encoder, passed with `--clip_vision`
   (for example `clip_vision_h.safetensors`).
3. An IP-Adapter weight file, passed with `--ip-adapter`. The `vit-h`
   variants reuse the same ViT-H encoder as above. From
   [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter):
   - SD 1.5: `models/ip-adapter_sd15.safetensors`
   - SDXL: `sdxl_models/ip-adapter_sdxl_vit-h.safetensors`

## Options

- `--ip-adapter <path>` path to the IP-Adapter weight file.
- `--ip-adapter-image <path>` path to the reference image.
- `--ip-adapter-strength <float>` strength of the IP-Adapter injection
  (default 1.0). Lower values let the text prompt dominate; 0.6 to 0.8 is
  a good starting range.

## Example (SD 1.5)

```
sd-cli -m ..\models\sd_v1.5.safetensors --clip_vision ..\models\clip_vision_h.safetensors --ip-adapter ..\models\ip-adapter_sd15.safetensors --ip-adapter-image ..\assets\reference.png --ip-adapter-strength 0.8 -p "a woman, best quality" -n "lowres, bad anatomy" --cfg-scale 7 --steps 30 --sampling-method dpm++2m --scheduler karras -W 512 -H 512
```

## Example (SDXL)

```
sd-cli -m ..\models\sdxl.safetensors --clip_vision ..\models\clip_vision_h.safetensors --ip-adapter ..\models\ip-adapter_sdxl_vit-h.safetensors --ip-adapter-image ..\assets\reference.png --ip-adapter-strength 0.8 -p "a woman, best quality" -n "lowres, bad anatomy" --cfg-scale 6 --steps 25 --sampling-method dpm++2m --scheduler karras -W 1024 -H 1024 --diffusion-fa --vae-tiling
```

The SDXL VAE decode at 1024x1024 is memory heavy; add `--vae-tiling` (and
`--offload-to-cpu`) on GPUs with limited VRAM.

## Combining with Control Net

Add the usual Control Net options to keep the reference appearance while
controlling the pose:

```
sd-cli -m ..\models\sdxl.safetensors --clip_vision ..\models\clip_vision_h.safetensors --ip-adapter ..\models\ip-adapter_sdxl_vit-h.safetensors --ip-adapter-image ..\assets\character.png --ip-adapter-strength 0.9 --control-net ..\models\OpenPoseXL2.safetensors --control-image ..\assets\pose.png --control-strength 0.8 -p "a character, side view" --cfg-scale 6 --steps 25 -W 1024 -H 1024 --diffusion-fa --vae-tiling
```
