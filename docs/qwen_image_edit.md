# How to Use

## Download weights

- Download Qwen Image
    - Qwen Image Edit
        - safetensors: https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/tree/main/split_files/diffusion_models
        - gguf: https://huggingface.co/QuantStack/Qwen-Image-Edit-GGUF/tree/main
    - Qwen Image Edit 2509
        - safetensors: https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/tree/main/split_files/diffusion_models
        - gguf: https://huggingface.co/QuantStack/Qwen-Image-Edit-2509-GGUF/tree/main
- Download vae
    - safetensors: https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files/vae
- Download qwen_2.5_vl 7b
    - safetensors: https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files/text_encoders
    - gguf: https://huggingface.co/mradermacher/Qwen2.5-VL-7B-Instruct-GGUF/tree/main

## Examples

### Qwen Image Edit

```
.\bin\Release\sd.exe --diffusion-model  ..\..\ComfyUI\models\diffusion_models\Qwen_Image_Edit-Q8_0.gguf --vae ..\..\ComfyUI\models\vae\qwen_image_vae.safetensors  --qwen2vl ..\..\ComfyUI\models\text_encoders\qwen_2.5_vl_7b.safetensors --cfg-scale 2.5 --sampling-method euler -v --offload-to-cpu --diffusion-fa --flow-shift 3 -r ..\assets\flux\flux1-dev-q8_0.png -p "change 'flux.cpp' to 'edit.cpp'" --seed 1118877715456453
```

<img alt="qwen_image_edit" src="../assets/qwen/qwen_image_edit.png" />


### Qwen Image Edit 2509

```
.\bin\Release\sd.exe --diffusion-model  ..\..\ComfyUI\models\diffusion_models\Qwen-Image-Edit-2509-Q4_K_S.gguf --vae ..\..\ComfyUI\models\vae\qwen_image_vae.safetensors  --qwen2vl ..\..\ComfyUI\models\text_encoders\Qwen2.5-VL-7B-Instruct-Q8_0.gguf --qwen2vl_vision ..\..\ComfyUI\models\text_encoders\Qwen2.5-VL-7B-Instruct.mmproj-Q8_0.gguf --cfg-scale 2.5 --sampling-method euler -v --offload-to-cpu --diffusion-fa --flow-shift 3 -r ..\assets\flux\flux1-dev-q8_0.png -p "change 'flux.cpp' to 'Qwen Image Edit 2509'"
```

<img alt="qwen_image_edit_2509" src="../assets/qwen/qwen_image_edit_2509.png" />