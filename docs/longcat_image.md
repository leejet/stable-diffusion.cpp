# How to Use

LongCat-Image uses a LongCat diffusion transformer, the FLUX VAE, and Qwen2.5-VL as the LLM text encoder.

## Download weights

- Download LongCat Image
    - safetensors: https://huggingface.co/Comfy-Org/LongCat-Image/tree/main/split_files/diffusion_models
    - gguf: https://huggingface.co/vantagewithai/LongCat-Image-GGUF/tree/main/comfy
- Download LongCat Image Edit
    - LongCat Image Edit Turbo: https://huggingface.co/meituan-longcat/LongCat-Image-Edit-Turbo
    - gguf: https://huggingface.co/vantagewithai/LongCat-Image-Edit-GGUF/tree/main
- Download vae
    - safetensors: https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/ae.safetensors
- Download qwen_2.5_vl 7b
    - safetensors: https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files/text_encoders
    - gguf: https://huggingface.co/mradermacher/Qwen2.5-VL-7B-Instruct-GGUF/tree/main
    - For image editing with GGUF text encoders, also download the matching mmproj file and pass it with `--llm_vision`.

## Run

LongCat uses quoted text for character-level text rendering. Put target text inside single quotes, double quotes, or Chinese quotes.

### LongCat Image

```
.\bin\Release\sd-cli.exe --diffusion-model  ..\..\ComfyUI\models\diffusion_models\LongCat-Image-Q4_K_M.gguf --vae ..\..\ComfyUI\models\vae\ae.sft --llm ..\..\ComfyUI\models\text_encoders\Qwen2.5-VL-7B-Instruct-Q8_0.gguf -p "a lovely cat holding a sign says 'longcat.cpp'" --cfg-scale 5.0 --sampling-method euler --flow-shift 3 -v --offload-to-cpu --diffusion-fa
```

<img alt="longcat example" src="../assets/longcat/example.png" />
