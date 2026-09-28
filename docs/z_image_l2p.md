# Z-Image L2P

[L2P](https://nju-pcalab.github.io/projects/L2P/) (Latent-to-Pixel) transfers Z-Image-Turbo to pixel space. The VAE is replaced by 16x16 patch tokenization on RGB pixels and a small convolutional local decoder that turns the last DiT hidden states back into pixels, so no VAE is needed. The text encoder is the Qwen3-4B used by Z-Image-Turbo.

## Download weights

- Download L2P (1K)
    - safetensors: https://huggingface.co/zhen-nan/L2P/tree/main
- Download Qwen3 4b
    - safetensors: https://huggingface.co/Comfy-Org/z_image_turbo/tree/main/split_files/text_encoders
    - gguf: https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/tree/main

## Text-to-image

Do not pass `--vae`. Image dimensions must be multiples of 16. The reference pipeline uses 30 steps with a CFG scale of 2.0.

The released checkpoint keeps part of the transformer in F32. Pass `--type bf16` to load it in BF16, as the reference pipeline runs it; this reduces the diffusion model from about 18.6 GB to 11.8 GB.

```bash
.\bin\Release\sd-cli.exe --diffusion-model ..\models\diffusion_models\model-1k-merge.safetensors --llm ..\models\text_encoders\qwen_3_4b.safetensors -p "an origami pig on fire in the middle of a dark room with a pentagram on the floor" --cfg-scale 2.0 --steps 30 --sampling-method euler -W 1024 -H 1024 --diffusion-fa --type bf16 -v -o z_image_l2p.png
```
