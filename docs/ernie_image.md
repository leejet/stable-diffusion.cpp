# How to Use

You can run ERNIE-Image with stable-diffusion.cpp on GPUs with 4GB of VRAM — or even less.

## Download weights

- Download ERNIE-Image-Turbo
    - safetensors: https://huggingface.co/Comfy-Org/ERNIE-Image/tree/main/diffusion_models
    - gguf: https://huggingface.co/unsloth/ERNIE-Image-Turbo-GGUF/tree/main
- Download ERNIE-Image
    - safetensors: https://huggingface.co/Comfy-Org/ERNIE-Image/tree/main/diffusion_models
    - gguf: https://huggingface.co/unsloth/ERNIE-Image-GGUF/tree/main
- Download vae
    - safetensors: https://huggingface.co/Comfy-Org/ERNIE-Image/tree/main/vae
- Download ministral 3b
    - safetensors: https://huggingface.co/Comfy-Org/ERNIE-Image/tree/main/text_encoders
    - gguf: https://huggingface.co/unsloth/Ministral-3-3B-Instruct-2512-GGUF/tree/main

## Examples

### ERNIE-Image-Turbo

```
.\bin\Release\sd-cli.exe --diffusion-model  ..\..\ComfyUI\models\diffusion_models\ernie-image-turbo.safetensors --vae ..\..\ComfyUI\models\vae\flux2_ae.safetensors  --llm ..\..\ComfyUI\models\text_encoders\ministral-3-3b.safetensors -p "a lovely cat" --cfg-scale 1.0 --steps 8 -v --offload-to-cpu --diffusion-fa
```

<img width="256" alt="ERNIE-Image Turbo example" src="../assets/ernie_image/turbo_example.png" />

### ERNIE-Image

```
.\bin\Release\sd-cli.exe --diffusion-model  ..\..\ComfyUI\models\diffusion_models\ernie-image-UD-Q4_K_M.gguf --vae ..\..\ComfyUI\models\vae\flux2_ae.safetensors  --llm ..\..\ComfyUI\models\text_encoders\ministral-3-3b.safetensors -p "a lovely cat" --cfg-scale 5.0 -v --offload-to-cpu --diffusion-fa
```

<img width="256" alt="ERNIE-Image example" src="../assets/ernie_image/example.png" />
