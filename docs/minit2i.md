# How to Use

MiniT2I uses a MiniT2I diffusion transformer and `google/flan-t5-large` as the text encoder.

## Download weights

- Download MiniT2I diffusion model
    - safetensors: https://huggingface.co/MiniT2I/minit2i-b-16/tree/main/transformer (`diffusion_pytorch_model.safetensors`)
- Download flan-t5-large text encoder
    - safetensors: https://huggingface.co/google/flan-t5-large/tree/main (`model.safetensors`)

## Examples

### Mac Metal

```
./bin/sd-cli \
  --backend metal \
  --diffusion-model ../models/minit2i/diffusion_pytorch_model.safetensors \
  --t5xxl ../models/flan-t5-large/model.safetensors \
  --prompt "a cat" \
  --steps 100 \
  --cfg-scale 6 \
  --width 512 \
  --height 512 \
  --seed 42 \
  --sampling-method euler \
  --rng cpu \
  --output minit2i_metal.png \
  --threads 8
```

### CUDA with diffusion flash attention

```
./bin/sd-cli \
  --diffusion-model ../models/minit2i/diffusion_pytorch_model.safetensors \
  --t5xxl ../models/flan-t5-large/model.safetensors \
  --prompt "a cat" \
  --steps 100 \
  --cfg-scale 6 \
  --width 512 \
  --height 512 \
  --seed 42 \
  --sampling-method euler \
  --diffusion-fa \
  --output minit2i_cuda.png
```
