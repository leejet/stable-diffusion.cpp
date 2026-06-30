# How to Use

You can run Chroma using stable-diffusion.cpp with a GPU that has 6GB or even 4GB of VRAM, without needing to offload to RAM.

## Download weights

- Download Chroma
    - If you don't want to do the conversion yourself, download the preconverted gguf model from [silveroxides/Chroma-GGUF](https://huggingface.co/silveroxides/Chroma-GGUF)
    - Otherwise, download chroma's safetensors from [lodestones/Chroma](https://huggingface.co/lodestones/Chroma)
- Download vae from https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/ae.safetensors
- Download t5xxl from https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp16.safetensors

## Convert Chroma weights

You can download the preconverted gguf weights from [silveroxides/Chroma-GGUF](https://huggingface.co/silveroxides/Chroma-GGUF), this way you don't have to do the conversion yourself.

```
.\bin\Release\sd-cli.exe -M convert -m ..\..\ComfyUI\models\unet\chroma-unlocked-v40.safetensors -o ..\models\chroma-unlocked-v40-q8_0.gguf -v --type q8_0
```

## Run

### Example
For example:

```
 .\bin\Release\sd-cli.exe --diffusion-model  ..\models\chroma-unlocked-v40-q8_0.gguf --vae ..\models\ae.sft --t5xxl ..\models\t5xxl_fp16.safetensors  -p "a lovely cat holding a sign says 'chroma.cpp'" --cfg-scale 4.0 --sampling-method euler -v --chroma-disable-dit-mask --clip-on-cpu
```

![](../assets/flux/chroma_v40.png)



