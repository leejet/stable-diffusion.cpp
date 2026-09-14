# How to Use

Lens uses a Lens diffusion transformer, the FLUX.2 VAE, and GPT-OSS-20B as the LLM text encoder.

## Download weights

- Download Lens
    - safetensors: https://huggingface.co/Comfy-Org/Lens/tree/main/diffusion_models
- Download Lens Turbo
    - safetensors: https://huggingface.co/Comfy-Org/Lens/tree/main/diffusion_models
- Download vae
    - safetensors: https://huggingface.co/black-forest-labs/FLUX.2-dev/tree/main
- Download GPT-OSS-20B
    - gguf: https://huggingface.co/unsloth/gpt-oss-20b-GGUF/tree/main
- Download GPT-OSS-20B tokenizer.json
    - https://huggingface.co/openai/gpt-oss-20b/tree/main

Lens and Lens Turbo require an external GPT-OSS `tokenizer.json` matching the text encoder checkpoint. Save it as `tokenizer_gpt_oss.json` and pass it with `--tokenizer`; the tokenizer is not embedded in sd.cpp. See [JSON tokenizers](tokenizers.md) for CLI and C API usage.

## Examples

### Lens

```
.\bin\Release\sd-cli.exe --diffusion-model ..\models\diffusion_models\lens_bf16.safetensors --llm "..\models\text_encoders\gpt-oss-20b-UD-Q8_K_XL.gguf" --tokenizer ..\models\tokenizers\tokenizer_gpt_oss.json --vae ..\models\vae\flux2_ae.safetensors --cfg-scale 5.0  -p "A crystal dragon soaring through an aurora borealis sky, its entire body made of transparent faceted crystal refracting the green and purple aurora light into rainbow spectra, ice particles trailing from its wings, high fantasy digital art" --diffusion-fa -v
```

<img width="256" alt="Lens example" src="../assets/lens/example.png" />

### Lens Turbo

```
.\bin\Release\sd-cli.exe --diffusion-model ..\models\diffusion_models\lens_turbo_bf16.safetensors --llm "..\models\text_encoders\gpt-oss-20b-UD-Q8_K_XL.gguf" --tokenizer ..\models\tokenizers\tokenizer_gpt_oss.json --vae ..\models\vae\flux2_ae.safetensors --cfg-scale 1.0  -p "A crystal dragon soaring through an aurora borealis sky, its entire body made of transparent faceted crystal refracting the green and purple aurora light into rainbow spectra, ice particles trailing from its wings, high fantasy digital art" --diffusion-fa -v --steps 4
```

<img width="256" alt="Lens Turbo example" src="../assets/lens/turbo_example.png" />
