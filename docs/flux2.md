# How to Use

## Download weights

- Download FLUX.2-dev
    - gguf: https://huggingface.co/city96/FLUX.2-dev-gguf/tree/main
- Download vae
    - safetensors: https://huggingface.co/black-forest-labs/FLUX.2-dev/tree/main
- Download Mistral-Small-3.2-24B-Instruct-2506-GGUF
    - gguf: https://huggingface.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF/tree/main

## Examples

```
.\bin\Release\sd-cli.exe --diffusion-model  ..\..\ComfyUI\models\diffusion_models\flux2-dev-Q4_K_S.gguf --vae ..\..\ComfyUI\models\vae\flux2_ae.safetensors  --llm ..\..\ComfyUI\models\text_encoders\Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf -r .\kontext_input.png -p "change 'flux.cpp' to 'flux2-dev.cpp'" --cfg-scale 1.0 --sampling-method euler -v --diffusion-fa --offload-to-cpu
```

<img alt="flux2 example" src="../assets/flux2/example.png" />



