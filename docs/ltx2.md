# How to Use

## Download weights

- Download LTX-2.3
    - safetensors: https://huggingface.co/Kijai/LTX2.3_comfy/tree/main/diffusion_models
    - gguf: https://huggingface.co/unsloth/LTX-2.3-GGUF/tree/main
- Download gemma-3-12b-it
    - gguf: https://huggingface.co/unsloth/gemma-3-12b-it-GGUF/tree/main
- Download embeddings connectors
    - safetensors: https://huggingface.co/unsloth/LTX-2.3-GGUF/tree/main/text_encoders
- Download vae
    - safetensors: https://huggingface.co/unsloth/LTX-2.3-GGUF/tree/main/vae
- Download audio vae
    - safetensors: https://huggingface.co/unsloth/LTX-2.3-GGUF/tree/main/vae

## Examples

### LTX-2.3 dev T2V

```
.\bin\Release\sd-cli.exe -M vid_gen --diffusion-model  ..\..\ComfyUI\models\diffusion_models\ltx-2.3-22b-dev-UD-Q4_K_M.gguf --vae ..\..\ComfyUI\models\vae\ltx-2.3-22b-dev_video_vae.safetensors --audio-vae ..\..\ComfyUI\models\vae\ltx-2.3-22b-dev_audio_vae.safetensors --llm ..\..\ComfyUI\models\text_encoders\gemma-3-12b-it-qat-UD-Q4_K_XL.gguf --embeddings-connectors ..\..\ComfyUI\models\text_encoders\ltx-2.3-22b-dev_embeddings_connectors.safetensors  -p "a lovely cat" --cfg-scale 6.0 --sampling-method euler -v -n "worst quality, low quality, blurry, distorted, artifacts" -W 1280 -H 720 --diffusion-fa --offload-to-cpu --video-frames 33 --fps 24 -o t2v.webm
```

<video
  src="../assets/ltx2/t2v.webm"
  controls
  muted
  style="max-width: 100%; height: auto;"></video>

### LTX-2.3 dev I2V

```
.\bin\Release\sd-cli.exe -M vid_gen --diffusion-model  ..\..\ComfyUI\models\diffusion_models\ltx-2.3-22b-dev-UD-Q4_K_M.gguf --vae ..\..\ComfyUI\models\vae\ltx-2.3-22b-dev_video_vae.safetensors --audio-vae ..\..\ComfyUI\models\vae\ltx-2.3-22b-dev_audio_vae.safetensors --llm ..\..\ComfyUI\models\text_encoders\gemma-3-12b-it-qat-UD-Q4_K_XL.gguf --embeddings-connectors ..\..\ComfyUI\models\text_encoders\ltx-2.3-22b-dev_embeddings_connectors.safetensors  -p "a lovely cat" --cfg-scale 6.0 --sampling-method euler -v  -W 1280 -H 720 --diffusion-fa --offload-to-cpu --video-frames 33 -i ..\assets\ernie_image\turbo_example.png -o i2v.webm
```

<video
  src="../assets/ltx2/i2v.webm"
  controls
  muted
  style="max-width: 100%; height: auto;"></video>

### LTX-2.3 dev FLF2V

```
.\bin\Release\sd-cli.exe -M vid_gen --diffusion-model  ..\..\ComfyUI\models\diffusion_models\ltx-2.3-22b-dev-UD-Q4_K_M.gguf --vae ..\..\ComfyUI\models\vae\ltx-2.3-22b-dev_video_vae.safetensors --audio-vae ..\..\ComfyUI\models\vae\ltx-2.3-22b-dev_audio_vae.safetensors --llm ..\..\ComfyUI\models\text_encoders\gemma-3-12b-it-qat-UD-Q4_K_XL.gguf --embeddings-connectors ..\..\ComfyUI\models\text_encoders\ltx-2.3-22b-dev_embeddings_connectors.safetensors  -p "glass flower blossom" --cfg-scale 6.0 --sampling-method euler -v  -W 1280 -H 720 --diffusion-fa --offload-to-cpu --video-frames 33 --init-img ..\..\ComfyUI\input\start_image.png --end-img ..\..\ComfyUI\input\end_image.png -o flf2v.webm
```

<video
  src="../assets/ltx2/flf2v.webm"
  controls
  muted
  style="max-width: 100%; height: auto;"></video>