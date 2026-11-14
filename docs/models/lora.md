## LoRA

- You can specify the directory where the lora weights are stored via `--lora-model-dir`. If not specified, the default is the current working directory.

- LoRA is specified via prompt, just like [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#lora).

Here's a simple example:

```
./bin/sd-cli -m ../models/v1-5-pruned-emaonly.safetensors -p "a lovely cat<lora:marblesh:1>" --lora-model-dir ../models
```

`../models/marblesh.safetensors` or `../models/marblesh.ckpt` will be applied to the model

# Lora Apply Mode

There are two ways to apply LoRA: **immediately** and **at_runtime**. You can specify it using the `--lora-apply-mode` parameter.

By default, the mode is selected automatically:

* If the model weights contain any quantized parameters, the **at_runtime** mode is used;
* Otherwise, the **immediately** mode is used.

The **immediately** mode may have precision and compatibility issues with quantized parameters, but it usually offers faster inference speed and, in some cases, lower memory usage.
In contrast, the **at_runtime** mode provides better compatibility and higher precision, but inference may be slower and memory usage may be higher in some cases.

