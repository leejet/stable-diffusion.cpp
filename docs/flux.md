# How to Use

You can run Flux using stable-diffusion.cpp with a GPU that has 6GB or even 4GB of VRAM, without needing to offload to RAM.

## Download weights

- Download flux
    - If you don't want to do the conversion yourself, download the preconverted gguf model from [FLUX.1-dev-gguf](https://huggingface.co/leejet/FLUX.1-dev-gguf) or [FLUX.1-schnell](https://huggingface.co/leejet/FLUX.1-schnell-gguf)
    - Otherwise, download flux-dev from https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/flux1-dev.safetensors or flux-schnell from https://huggingface.co/black-forest-labs/FLUX.1-schnell/blob/main/flux1-schnell.safetensors
- Download vae from https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/ae.safetensors
- Download clip_l from https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/clip_l.safetensors
- Download t5xxl from https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp16.safetensors

## Convert flux weights

You can download the preconverted gguf weights from [FLUX.1-dev-gguf](https://huggingface.co/leejet/FLUX.1-dev-gguf) or [FLUX.1-schnell](https://huggingface.co/leejet/FLUX.1-schnell-gguf), this way you don't have to do the conversion yourself.

Using fp16 will lead to overflow, but ggml's support for bf16 is not yet fully developed. Therefore, we need to convert flux to gguf format here, which also saves VRAM. For example:
```
.\bin\Release\sd.exe -M convert -m ..\..\ComfyUI\models\unet\flux1-dev.sft -o ..\models\flux1-dev-q8_0.gguf -v --type q8_0
```

## Run

- `--cfg-scale` is recommended to be set to 1. 

### Flux-dev
For example:

```
 .\bin\Release\sd.exe --diffusion-model  ..\models\flux1-dev-q8_0.gguf --vae ..\models\ae.sft --clip_l ..\models\clip_l.safetensors --t5xxl ..\models\t5xxl_fp16.safetensors  -p "a lovely cat holding a sign says 'flux.cpp'" --cfg-scale 1.0 --sampling-method euler -v
```

Using formats of different precisions will yield results of varying quality.

| Type | q8_0  | q4_0  | q4_k  | q3_k  | q2_k |
|---- | ----  |----  |----  |----  |----  |
| **Memory** | 12068.09 MB  | 6394.53 MB | 6395.17 MB | 4888.16 MB  | 3735.73 MB |
| **Result** | ![](../assets/flux/flux1-dev-q8_0.png) |![](../assets/flux/flux1-dev-q4_0.png) |![](../assets/flux/flux1-dev-q4_k.png) |![](../assets/flux/flux1-dev-q3_k.png) |![](../assets/flux/flux1-dev-q2_k.png)|



### Flux-schnell


```
 .\bin\Release\sd.exe --diffusion-model  ..\models\flux1-schnell-q8_0.gguf --vae ..\models\ae.sft --clip_l ..\models\clip_l.safetensors --t5xxl ..\models\t5xxl_fp16.safetensors  -p "a lovely cat holding a sign says 'flux.cpp'" --cfg-scale 1.0 --sampling-method euler -v --steps 4
```

| q8_0  |
| ----  |
|![](../assets/flux/flux1-schnell-q8_0.png) |

## Run with LoRA

Since many flux LoRA training libraries have used various LoRA naming formats, it is possible that not all flux LoRA naming formats are supported. It is recommended to use LoRA with naming formats compatible with ComfyUI.

### Flux-dev q8_0 with LoRA

- LoRA model from https://huggingface.co/XLabs-AI/flux-lora-collection/tree/main (using comfy converted version!!!)

```
.\bin\Release\sd.exe --diffusion-model  ..\models\flux1-dev-q8_0.gguf --vae ...\models\ae.sft --clip_l ..\models\clip_l.safetensors --t5xxl ..\models\t5xxl_fp16.safetensors  -p "a lovely cat holding a sign says 'flux.cpp'<lora:realism_lora_comfy_converted:1>" --cfg-scale 1.0 --sampling-method euler -v --lora-model-dir ../models
```

![output](../assets/flux/flux1-dev-q8_0%20with%20lora.png)
