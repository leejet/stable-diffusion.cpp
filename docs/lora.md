## LoRA

- You can specify the directory where the lora weights are stored via `--lora-model-dir`. If not specified, the default is the current working directory.

- LoRA is specified via prompt, just like [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#lora).

Here's a simple example:

```
./bin/sd -m ../models/v1-5-pruned-emaonly.safetensors -p "a lovely cat<lora:marblesh:1>" --lora-model-dir ../models
```

`../models/marblesh.safetensors` or `../models/marblesh.ckpt` will be applied to the model

# Support matrix

> ℹ️ CUDA `get_rows` support is defined here:  
> [ggml-org/ggml/src/ggml-cuda/getrows.cu#L156](https://github.com/ggml-org/ggml/blob/7dee1d6a1e7611f238d09be96738388da97c88ed/src/ggml-cuda/getrows.cu#L156)  
> Currently only the basic types + Q4/Q5/Q8 are implemented. K-quants are **not** supported.

NOTE: The other backends may have different support.

| Quant / Type | CUDA | Vulkan |
|--------------|------|--------|
| F32          | ✔️   | ✔️   |
| F16          | ✔️   | ✔️   |
| BF16         | ✔️   | ✔️   |
| I32          | ✔️   | ❌   |
| Q4_0         | ✔️   | ✔️   |
| Q4_1         | ✔️   | ✔️   |
| Q5_0         | ✔️   | ✔️   |
| Q5_1         | ✔️   | ✔️   |
| Q8_0         | ✔️   | ✔️   |
| Q2_K         | ❌   | ❌   |
| Q3_K         | ❌   | ❌   |
| Q4_K         | ❌   | ❌   |
| Q5_K         | ❌   | ❌   |
| Q6_K         | ❌   | ❌   |
| Q8_K         | ❌   | ❌   |
| IQ1_S        | ❌   | ✔️   |
| IQ1_M        | ❌   | ✔️   |
| IQ2_XXS      | ❌   | ✔️   |
| IQ2_XS       | ❌   | ✔️   |
| IQ2_S        | ❌   | ✔️   |
| IQ3_XXS      | ❌   | ✔️   |
| IQ3_S        | ❌   | ✔️   |
| IQ4_XS       | ❌   | ✔️   |
| IQ4_NL       | ❌   | ✔️   |
| MXFP4        | ❌   | ✔️   |
