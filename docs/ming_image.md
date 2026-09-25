# Ming-Image

[Ming-Image](https://github.com/inclusionAI/Ming-Image) 0.1 Design uses a 6B diffusion transformer (DiT), Ling-mini-2.0 for text conditioning, and the Ming-Image VAE. Text-to-image generation with RGBA output is supported.

## Download weights

- Download Ming-Image 0.1 Design DiT
    - safetensors: https://huggingface.co/Comfy-Org/Ming-Image/tree/main/diffusion_models
- Download Ling-mini-2.0 BF16
    - safetensors: https://huggingface.co/Comfy-Org/Ming-Image/tree/main/text_encoders
- Download Ming-Image VAE
    - safetensors: https://huggingface.co/Comfy-Org/Ming-Image/tree/main/vae
- Download Ling tokenizer
    - tokenizer.json: https://huggingface.co/inclusionAI/Ming-Image-0.1-Design/blob/main/mllm/tokenizer.json

The example below uses `ming_image_0.1_design_bf16.safetensors` for the DiT. You can also use `ming_image_0.1_design_int8_convrot.safetensors` with [INT8 convrot support](int8_convrot.md). Use the BF16 text encoder.

## Text-to-image

Pass the Ling `tokenizer.json` with `--tokenizer` and use the matching Ming-Image VAE. Image dimensions must be multiples of 16. Save the output as PNG to preserve the alpha channel.

```bash
.\bin\Release\sd-cli.exe --diffusion-model ..\models\diffusion_models\ming_image_0.1_design_bf16.safetensors --llm ..\models\text_encoders\ming_image_0.1_ling_mini_2.0_bf16.safetensors --vae ..\models\vae\ming_image_vae_bf16.safetensors --tokenizer ..\models\text_encoders\tokenizer.json -p "A cheerful orange cat sticker, transparent background" --width 1024 --height 1024 --steps 12 --cfg-scale 1 --sampling-method euler --diffusion-fa -v --offload-to-cpu -o ming_image.png
```
