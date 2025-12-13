## Download weights

- download original weights(.ckpt or .safetensors). For example
    - Stable Diffusion v1.4 from https://huggingface.co/CompVis/stable-diffusion-v-1-4-original
    - Stable Diffusion v1.5 from https://huggingface.co/runwayml/stable-diffusion-v1-5
    - Stable Diffuison v2.1 from https://huggingface.co/stabilityai/stable-diffusion-2-1
    - Stable Diffusion 3 2B from https://huggingface.co/stabilityai/stable-diffusion-3-medium

### txt2img example

```sh
./bin/sd-cli -m ../models/sd-v1-4.ckpt -p "a lovely cat"
# ./bin/sd-cli -m ../models/v1-5-pruned-emaonly.safetensors -p "a lovely cat"
# ./bin/sd-cli -m ../models/sd_xl_base_1.0.safetensors --vae ../models/sdxl_vae-fp16-fix.safetensors -H 1024 -W 1024 -p "a lovely cat" -v
# ./bin/sd-cli -m ../models/sd3_medium_incl_clips_t5xxlfp16.safetensors -H 1024 -W 1024 -p 'a lovely cat holding a sign says \"Stable Diffusion CPP\"' --cfg-scale 4.5 --sampling-method euler -v --clip-on-cpu
# ./bin/sd-cli --diffusion-model  ../models/flux1-dev-q3_k.gguf --vae ../models/ae.sft --clip_l ../models/clip_l.safetensors --t5xxl ../models/t5xxl_fp16.safetensors  -p "a lovely cat holding a sign says 'flux.cpp'" --cfg-scale 1.0 --sampling-method euler -v --clip-on-cpu
# ./bin/sd-cli -m  ..\models\sd3.5_large.safetensors --clip_l ..\models\clip_l.safetensors --clip_g ..\models\clip_g.safetensors --t5xxl ..\models\t5xxl_fp16.safetensors  -H 1024 -W 1024 -p 'a lovely cat holding a sign says \"Stable diffusion 3.5 Large\"' --cfg-scale 4.5 --sampling-method euler -v --clip-on-cpu
```

Using formats of different precisions will yield results of varying quality.

| f32  | f16  |q8_0  |q5_0  |q5_1  |q4_0  |q4_1  |
| ----  |----  |----  |----  |----  |----  |----  |
| ![](../assets/f32.png) |![](../assets/f16.png) |![](../assets/q8_0.png) |![](../assets/q5_0.png) |![](../assets/q5_1.png) |![](../assets/q4_0.png) |![](../assets/q4_1.png) |

### img2img example

- `./output.png` is the image generated from the above txt2img pipeline


```
./bin/sd-cli -m ../models/sd-v1-4.ckpt -p "cat with blue eyes" -i ./output.png -o ./img2img_output.png --strength 0.4
```

<p align="center">
  <img src="../assets/img2img_output.png" width="256x">
</p>