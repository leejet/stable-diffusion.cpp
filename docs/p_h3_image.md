# How to Use

P-H3-Image supports text-to-image and reference-image editing. It uses a MiniMax-H3 diffusion transformer, a single-frame VAE, and Qwen3-VL-4B with a text connector. Prompt enhancement is not included.

## Download weights

- Download P-H3-Image
    - safetensors: https://huggingface.co/PrunaAI/P-H3-Image-Comfy/tree/main/diffusion_models
- Download P-H3-Image VAE
    - safetensors: https://huggingface.co/PrunaAI/P-H3-Image-Comfy/tree/main/vae
- Download Qwen3-VL-4B with the text connector
    - https://huggingface.co/PrunaAI/P-H3-Image-Comfy/tree/main/text_encoders

Alternatively, reuse Qwen3-VL-4B, including its vision tower, with the [separate text connector](https://huggingface.co/PrunaAI/P-H3-Image/blob/main/text_connector/diffusion_pytorch_model.safetensors). Save the connector as `p-h3-image_text_connector.safetensors` and pass it with `--embeddings-connectors`.

Use the P-H3 image VAE; the MiniMax-H3 video VAE is not compatible. No audio VAE is needed.

## Examples

### Text-to-image

```
.\bin\Release\sd-cli.exe --diffusion-model ..\models\diffusion_models\p-h3-image_int8_convrot.safetensors --llm ..\models\text_encoders\qwen3vl_4b_bf16.safetensors --embeddings-connectors ..\models\text_encoders\p-h3-image_text_connector.safetensors --vae ..\models\vae\p-h3-image_vae_bf16.safetensors --cfg-scale 1.0 --steps 8 --sampling-method euler --sigmas p-h3-image -p "a lovely cat holding a sign says 'p-h3-image.cpp'" -v --offload-to-cpu --rng cpu -H 2912 -W 2912
```

### Image editing

```
.\bin\Release\sd-cli.exe --diffusion-model ..\models\diffusion_models\p-h3-image_int8_convrot.safetensors --llm ..\models\text_encoders\qwen3vl_4b_bf16.safetensors --embeddings-connectors ..\models\text_encoders\p-h3-image_text_connector.safetensors --vae ..\models\vae\p-h3-image_vae_bf16.safetensors --cfg-scale 1.0 --steps 8 --sampling-method euler --sigmas p-h3-image -r source.png -r reference.png -p "Edit <Picture 1>. Give the cat the scarf from <Picture 2>, preserving its face and the background." -v --offload-to-cpu --rng cpu -H 2912 -W 2912
```

Reference images are numbered in argument order as `<Picture 1>`, `<Picture 2>`, and so on. A plain edit instruction gets `Edit <Picture 1>. ` prepended automatically. Prompts starting with `Edit <Picture N>` or `subject_definitions:` are used directly.

Use dimensions divisible by 32. The upstream default is 2912 × 2912; smaller canvases reduce memory use. Use CFG 1 and `--sigmas p-h3-image` for the upstream 8-step schedule. Keep the default VAE tiling size of 256 pixels.
