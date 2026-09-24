# How to Use

You can run PixArt-α / PixArt-Σ with stable-diffusion.cpp.

PixArt is a DiT-based text-to-image model family conditioned by a T5-XXL text
encoder and a standard SD1.x-style VAE.

## Download weights

- Download the transformer (diffusion model)
    - PixArt-Σ XL-2 1024-MS: https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS/tree/main/transformer
    - PixArt-α XL-2 1024-MS: https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS/tree/main/transformer
- Download the T5-XXL text encoder
    - safetensors: https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS/tree/main/text_encoder
- Download the VAE
    - safetensors: https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS/tree/main/vae
    - any SD1.x-compatible VAE also works
- Tokenizer: the T5 vocabulary is embedded; no extra tokenizer file is needed.

## Examples

```
.\bin\Release\sd-cli.exe --diffusion-model ..\models\diffusion_models\pixart_sigma_xl2_1024_ms.safetensors --t5xxl ..\models\text_encoders\t5xxl.safetensors --vae ..\models\vae\pixart_vae.safetensors -p "a lovely cat" --cfg-scale 4.5 -W 1024 -H 1024 --steps 20 -v
```

## Notes

- PixArt-Σ checkpoints compute 2D sincos positional embeddings at runtime;
  the trained grid is 64x64 patches with an interpolation scale of 2.
  For checkpoints trained at a different resolution, the positional embedding
  parameters can be adjusted via model args:
  `--model-args "pixart_pos_embed_base_size=<trained grid>,pixart_interpolation_scale=<scale>"`
  (e.g. `pixart_pos_embed_base_size=32,pixart_interpolation_scale=1` for
  PixArt-α XL-2 512).
- Checkpoints carrying resolution/aspect-ratio micro-condition weights are
  detected but those conditions are not applied yet; a warning is logged and
  generation proceeds with the timestep embedding only.
- The transformer predicts 8 channels (noise + learned variance); only the
  noise half is used for sampling, matching the reference implementation.
