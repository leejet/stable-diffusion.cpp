# How to Use

PiD is NVIDIA's Pixel Diffusion Decoder. It replaces the usual VAE decode or decode-then-upscale path with a pixel-space diffusion decoder conditioned on a
source latent and text prompt.

In stable-diffusion.cpp, PiD currently runs as an image edit pipeline: provide a reference image with `-r`/`--ref-image`, encode that image with a matching VAE, then let the PiD diffusion model decode/upscale directly to RGB.

## Download weights

- Download PiD
    - safetensors: https://huggingface.co/Comfy-Org/PixelDiT/tree/main/diffusion_models
- Download Gemma 2 2B
    - safetensors: https://huggingface.co/Comfy-Org/PixelDiT/tree/main/text_encoders
- Download the VAE that matches the PiD checkpoint backbone
    - safetensors: https://huggingface.co/nvidia/PiD/tree/main/checkpoints
    - Flux / Z-Image PiD: use the Flux VAE and pass `--vae-format flux`
    - SD3 PiD: use the SD3 VAE and pass `--vae-format sd3`
    - Flux.2 PiD: use the Flux.2 VAE and pass `--vae-format flux2`

The official PiD model card should be checked before use. At the time of the initial PiD release, the official weights are under the NSCLv1 non-commercial license.

## Examples

```
.\bin\Release\sd-cli.exe --diffusion-model ..\..\ComfyUI\models\diffusion_models\pid_flux1_512_to_2048_4step_bf16.safetensors --llm "..\..\ComfyUI\models\text_encoders\gemma_2_2b_it_elm_bf16.safetensors" --vae ..\..\ComfyUI\models\vae\ae.sft --vae-format flux --cfg-scale 1.0  -p "a lovely cat" -r ..\assets\ernie_image\turbo_example.png --diffusion-fa -v --steps 4 -H 2048 -W 2048 --rng cpu
```

Before:

<img width="256" alt="ERNIE-Image Turbo example" src="../assets/ernie_image/turbo_example.png" />

After:
<img width="1024" alt="PiD example" src="../assets/pid/example.png" />

## Notes

- `-r`/`--ref-image` is required. PiD uses the first reference image as the source latent condition.
- `--vae-format` should match the VAE latent layout used by the PiD checkpoint. This is important when using standalone VAE files because the PiD diffusion
  checkpoint alone does not identify the VAE format.
