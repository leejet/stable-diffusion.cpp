# How to Use

Iris-3B generates images directly in pixel space and uses Qwen3-VL-4B-Instruct as the text encoder. No VAE is required.

## Download weights

- Download Iris-3B
    - safetensors: https://huggingface.co/speridlabs/iris-3b/tree/main (`model.safetensors` in the root directory)
- Download Qwen3-VL-4B-Instruct
    - safetensors: https://huggingface.co/Comfy-Org/Krea-2/tree/main/text_encoders
    - gguf: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/tree/main

## Examples

```
.\bin\Release\sd-cli.exe --diffusion-model ..\models\diffusion_models\iris-3b.safetensors --llm ..\models\text_encoders\Qwen3-VL-4B-Instruct-Q4_K_M.gguf -p "a lovely cat" --cfg-scale 3 -H 1024 -W 1024 --diffusion-fa
```

<img width="256" alt="iris-3b example" src="../assets/iris/example.png" />

## Notes

- Width and height must be multiples of 16. Do not pass `--vae`.
- Captions are limited to 300 tokens including the assistant-turn suffix. Positive and negative prompts use the same template.
