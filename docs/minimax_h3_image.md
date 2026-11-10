# MiniMax-H3-Image

MiniMax-H3-Image supports text-to-image generation and reference-image editing
with the same checkpoint. It reuses the MiniMax-H3 text encoder and uses a
dedicated image VAE.

## Model files

Pass the three MiniMax-H3-Image components separately:

- `--diffusion-model`: MiniMax-H3-Image diffusion transformer
- `--vae`: MiniMax-H3-Image VAE
- `--llm`: the MiniMax-H3 Qwen3-VL-32B text encoder checkpoint

Use the MiniMax-H3 text encoder, including its Qwen3-VL vision tower and
DeepStack mergers. An existing MiniMax-H3 text encoder can be reused. If the
vision tower is stored separately, pass it with `--llm_vision`.

Use a single-file diffusion checkpoint with the six-step PDD output heads
included, such as `minimax_h3_image_pruned_bf16.safetensors` or its INT8 ConvRot
version. Use the MiniMax-H3-Image VAE; the MiniMax-H3 video VAE and P-H3-Image
VAE are incompatible. No audio VAE or P-H3 text connector is needed.

### Download weights

- Download minimax_h3_image_pruned
    - safetensors: https://huggingface.co/Comfy-Org/MiniMax-H3-Image/tree/main/diffusion_models
- Download qwen3vl_32b_minimax_h3
    - safetensors: https://huggingface.co/Comfy-Org/MiniMax-H3-Image/tree/main/text_encoders
    - gguf: https://huggingface.co/leejet/MiniMax-H3-GGUF/tree/main
- Download vae
    - safetensors: https://huggingface.co/Comfy-Org/MiniMax-H3-Image/tree/main/vae

## Text-to-image

```sh
.\bin\Release\sd-cli.exe --diffusion-model ..\models\diffusion_models\minimax_h3_image_pruned_bf16.safetensors --vae ..\models\vae\minimax_h3_image_vae_fp16.safetensors --llm ..\models\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf -p "A silver tabby kitten holding a sign reading 'sd.cpp', natural daylight" --cfg-scale 1.0 --steps 6 --sampling-method euler --sigmas minimax-h3-image -v -W 1024 -H 1024 --offload-to-cpu --rng cpu --diffusion-fa --vae-tiling
```

Use Euler, CFG 1, and `--sigmas minimax-h3-image` for the model's six-step
schedule. The default discrete scheduler also produces this schedule with
`--steps 6`. Prompt expansion is not included.

## Reference-image editing

Add `--ref-image` (or `-r`) for reference-image editing:

```sh
.\bin\Release\sd-cli.exe --diffusion-model ..\models\diffusion_models\minimax_h3_image_pruned_bf16.safetensors --vae ..\models\vae\minimax_h3_image_vae_fp16.safetensors --llm ..\models\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf -p "Edit <Picture 1>. Put a red scarf on the cat, preserving its face and the background." -r ..\assets\ernie_image\turbo_example.png --cfg-scale 1.0 --steps 6 --sampling-method euler --sigmas minimax-h3-image -v -W 1024 -H 1024 --offload-to-cpu --rng cpu --diffusion-fa --vae-tiling
```

`--ref-image` can be repeated. Reference images are numbered in argument order
as `<Picture 1>`, `<Picture 2>`, and so on, with up to 16 references supported
by the model. Prompts are passed through without an automatic editing prefix.

Reference images keep their aspect ratio. The text encoder sees a 1024-pixel
short edge, while VAE inputs use a short edge between 1024 and 2048 pixels.
Small references are enlarged and dimensions are rounded to multiples of 32.

## Shape and runtime notes

- Width and height are aligned upward to a multiple of 32.
- Start at 1024 × 1024. Larger canvases and more references increase memory use.
- Use quantized weights, `--offload-to-cpu`, and `--vae-tiling` to reduce GPU
  memory requirements.
