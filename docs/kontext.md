# How to Use

You can run Kontext using stable-diffusion.cpp with a GPU that has 6GB or even 4GB of VRAM, without needing to offload to RAM.

## Download weights

- Download Kontext
    - If you don't want to do the conversion yourself, download the preconverted gguf model from [FLUX.1-Kontext-dev-GGUF](https://huggingface.co/QuantStack/FLUX.1-Kontext-dev-GGUF)
    - Otherwise, download FLUX.1-Kontext-dev from https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev/blob/main/flux1-kontext-dev.safetensors
- Download vae from https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/ae.safetensors
- Download clip_l from https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/clip_l.safetensors
- Download t5xxl from https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp16.safetensors

## Convert Kontext weights

You can download the preconverted gguf weights from [FLUX.1-Kontext-dev-GGUF](https://huggingface.co/QuantStack/FLUX.1-Kontext-dev-GGUF), this way you don't have to do the conversion yourself.

```
.\bin\Release\sd-cli.exe -M convert -m ..\..\ComfyUI\models\unet\flux1-kontext-dev.safetensors -o ..\models\flux1-kontext-dev-q8_0.gguf -v --type q8_0
```

## Run

- `--cfg-scale` is recommended to be set to 1. 

### Example
For example:

```
 .\bin\Release\sd-cli.exe -r .\flux1-dev-q8_0.png --diffusion-model  ..\models\flux1-kontext-dev-q8_0.gguf --vae ..\models\ae.sft --clip_l ..\models\clip_l.safetensors --t5xxl ..\models\t5xxl_fp16.safetensors -p "change 'flux.cpp' to 'kontext.cpp'" --cfg-scale 1.0 --sampling-method euler -v --clip-on-cpu
```


| ref_image | prompt  | output  |
| ---- | ----  |----  |
| ![](../assets/flux/flux1-dev-q8_0.png) | change 'flux.cpp' to 'kontext.cpp' |![](../assets/flux/kontext1_dev_output.png) |



