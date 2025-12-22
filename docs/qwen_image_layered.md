# How to Use

## Download weights

- Download Qwen Image Layered
    - safetensors: https://huggingface.co/Comfy-Org/Qwen-Image-Layered_ComfyUI/tree/main/split_files/diffusion_models
    - gguf: https://huggingface.co/QuantStack/Qwen-Image-Layered-GGUF/tree/main
- Download vae
    - safetensors: https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files/vae
- Download qwen_2.5_vl 7b
    - safetensors: https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files/text_encoders
    - gguf: https://huggingface.co/mradermacher/Qwen2.5-VL-7B-Instruct-GGUF/tree/main

## Examples

### Text to Layered Image

This model generates 4 RGBA layers from a text prompt.

```
.\bin\Release\sd-cli.exe --diffusion-model  ..\..\ComfyUI\models\diffusion_models\Qwen-Image-Layered-Q8_0.gguf --vae ..\..\ComfyUI\models\vae\qwen_image_vae.safetensors  --llm ..\..\ComfyUI\models\text_encoders\Qwen2.5-VL-7B-Instruct-Q8_0.gguf  -p "a lovely cat" --cfg-scale 2.5 --sampling-method euler -v --offload-to-cpu -H 1024 -W 1024 --diffusion-fa --flow-shift 3
```

The output will consist of multiple images corresponding to the generated layers.

```