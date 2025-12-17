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
