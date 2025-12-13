# How to Use

You can run Z-Image with stable-diffusion.cpp on GPUs with 4GB of VRAM — or even less.

## Download weights

- Download Z-Image-Turbo
    - safetensors: https://huggingface.co/Comfy-Org/z_image_turbo/tree/main/split_files/diffusion_models
    - gguf: https://huggingface.co/leejet/Z-Image-Turbo-GGUF/tree/main
- Download vae
    - safetensors: https://huggingface.co/black-forest-labs/FLUX.1-schnell/tree/main
- Download Qwen3 4b
    - safetensors: https://huggingface.co/Comfy-Org/z_image_turbo/tree/main/split_files/text_encoders
    - gguf: https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/tree/main

## Examples

```
.\bin\Release\sd-cli.exe --diffusion-model  z_image_turbo-Q3_K.gguf --vae ..\..\ComfyUI\models\vae\ae.sft  --llm ..\..\ComfyUI\models\text_encoders\Qwen3-4B-Instruct-2507-Q4_K_M.gguf -p "A cinematic, melancholic photograph of a solitary hooded figure walking through a sprawling, rain-slicked metropolis at night. The city lights are a chaotic blur of neon orange and cool blue, reflecting on the wet asphalt. The scene evokes a sense of being a single component in a vast machine. Superimposed over the image in a sleek, modern, slightly glitched font is the philosophical quote: 'THE CITY IS A CIRCUIT BOARD, AND I AM A BROKEN TRANSISTOR.' -- moody, atmospheric, profound, dark academic" --cfg-scale 1.0 -v --offload-to-cpu --diffusion-fa -H 1024 -W 512
```

<img width="256" alt="z-image example" src="../assets/z_image/q3_K.png" />

## Comparison of Different Quantization Types

| bf16 | q8_0 | q6_K | q5_0 | q4_K | q4_0 | q3_K | q2_K|
|---|---|---|---|---|---|---|---|
| <img width="256" alt="bf16" src="../assets/z_image/bf16.png" /> | <img width="256" alt="q8_0" src="../assets/z_image/q8_0.png" /> | <img width="256" alt="q6_K" src="../assets/z_image/q6_K.png" /> | <img width="256" alt="q5_0" src="../assets/z_image/q5_0.png" />  | <img width="256" alt="q4_K" src="../assets/z_image/q4_K.png" /> | <img width="256" alt="q4_0" src="../assets/z_image/q4_0.png" /> | <img width="256" alt="q3_K" src="../assets/z_image/q3_K.png" /> | <img width="256" alt="q2_K" src="../assets/z_image/q2_K.png" /> |