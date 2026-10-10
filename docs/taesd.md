## Using TAESD to faster decoding

You can use TAESD to accelerate the decoding of latent images by following these steps:

- Download the model [weights](https://huggingface.co/madebyollin/taesd/blob/main/diffusion_pytorch_model.safetensors).

Or curl

```bash
curl -L -O https://huggingface.co/madebyollin/taesd/resolve/main/diffusion_pytorch_model.safetensors
```

- Specify the model path using the `--taesd PATH` parameter. example:

```bash
sd-cli -m ../models/v1-5-pruned-emaonly.safetensors -p "a lovely cat" --taesd ../models/diffusion_pytorch_model.safetensors
```

### Qwen-Image and wan (TAEHV)

sd.cpp also supports [TAEHV](https://github.com/madebyollin/taehv) (#937), which can be used for Qwen-Image and wan.

- For **Qwen-Image and wan2.1 and wan2.2-A14B**, download the wan2.1 tae [safetensors weights](https://github.com/madebyollin/taehv/blob/main/safetensors/taew2_1.safetensors)
  
  Or curl
  
  ```bash
  curl -L -O https://github.com/madebyollin/taehv/raw/refs/heads/main/safetensors/taew2_1.safetensors
  ```

- For **wan2.2-TI2V-5B**, use the wan2.2 tae [safetensors weights](https://github.com/madebyollin/taehv/blob/main/safetensors/taew2_2.safetensors)
  
  Or curl
  
  ```bash
  curl -L -O https://github.com/madebyollin/taehv/raw/refs/heads/main/safetensors/taew2_2.safetensors
  ```

Then simply replace the `--vae xxx.safetensors` with `--tae xxx.safetensors` in the commands. If it still out of VRAM, add `--vae-conv-direct` to your command though might be slower.

### Qwen Image 2.1 (TAEQI2.1)

For Qwen Image 2.1, use [taeqi2_1](https://github.com/madebyollin/taesd), which supports 64-channel latents, 16x spatial scaling, and RGBA images.

Download the official [safetensors weights](https://huggingface.co/madebyollin/taeqi2_1/blob/main/taeqi2_1.safetensors) directly; no conversion is needed:

```bash
curl -L -o taeqi2_1.safetensors https://huggingface.co/madebyollin/taeqi2_1/resolve/main/taeqi2_1.safetensors
```

Replace `--vae PATH` with `--taesd taeqi2_1.safetensors` in the [Qwen Image 2.1 examples](qwen_image_2.1.md) to use it for encoding and decoding. For previews only, keep `--vae PATH` and add `--taesd taeqi2_1.safetensors --taesd-preview-only --preview tae`.
