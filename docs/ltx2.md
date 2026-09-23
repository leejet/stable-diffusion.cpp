# How to Use

Both LTX-2.3 and LTX-2.5 are supported. The two share a transformer, video VAE and audio
VAE architecture; LTX-2.5 drops the video FFN biases, adds a learned keyframe
absolute-position embedding, and pairs with a Gemma 4 text encoder instead of Gemma 3.
Everything is detected from the weights, so the command lines differ only in which files
you pass.

# LTX-2.3

## Download weights

### LTX-2.3

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
- Download LTX spatial latent upscaler
    - safetensors: https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors

### LTX-2.5

- Download LTX-2.5
    - safetensors: https://huggingface.co/Lightricks/LTX-2.5/tree/main/diffusion_models
    - gguf: https://huggingface.co/vantagewithai/LTX-2.5-GGUF/tree/main
- Download the text encoder. This is a Gemma 4 12B fine-tuned for LTX with the text
  projection bundled in, so no separate `--embeddings-connectors` file is needed. Google's
  stock Gemma 4 is not a substitute.
    - safetensors: https://huggingface.co/Lightricks/LTX-2.5/blob/main/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
- Download the video vae. Use the **conv** variant: `ltx-2.5-video-vae-conv-bf16.safetensors`.
  The default `ltx-2.5-video-vae-bf16.safetensors` is a diffusion decoder, which is not
  implemented here.
    - safetensors: https://huggingface.co/Lightricks/LTX-2.5/blob/main/vae/ltx-2.5-video-vae-conv-bf16.safetensors
- Download the audio vae
    - safetensors: https://huggingface.co/Lightricks/LTX-2.5/blob/main/vae/ltx-2.5-audio-vae-bf16.safetensors
- Download the LTX spatial latent upscaler
    - safetensors: https://huggingface.co/Lightricks/LTX-2.5/blob/main/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors

To run the text encoder quantized, convert it once with sd-cli:

```
.\bin\Release\sd-cli.exe -M convert -m ..\models\text_encoders\gemma4-12b-with-proj-ltx-2.5-bf16.safetensors --type q8_0 -o ..\models\text_encoders\gemma4-12b-with-proj-ltx-2.5-Q8_0.gguf
```

## Examples

### LTX-2.3 dev T2V

```
.\bin\Release\sd-cli.exe -M vid_gen --diffusion-model  ..\models\diffusion_models\ltx-2.3-22b-dev-UD-Q4_K_M.gguf --vae ..\models\vae\ltx-2.3-22b-dev_video_vae.safetensors --audio-vae ..\models\vae\ltx-2.3-22b-dev_audio_vae.safetensors --llm ..\models\text_encoders\gemma-3-12b-it-qat-UD-Q4_K_XL.gguf --embeddings-connectors ..\models\text_encoders\ltx-2.3-22b-dev_embeddings_connectors.safetensors  -p "a lovely cat" --cfg-scale 6.0 --sampling-method euler -v -n "worst quality, low quality, blurry, distorted, artifacts" -W 1280 -H 720 --diffusion-fa --offload-to-cpu --video-frames 33 --fps 24 -o t2v.webm
```

<video
  src="../assets/ltx2/t2v.webm"
  controls
  muted
  style="max-width: 100%; height: auto;"></video>

### LTX-2.3 dev I2V

```
.\bin\Release\sd-cli.exe -M vid_gen --diffusion-model  ..\models\diffusion_models\ltx-2.3-22b-dev-UD-Q4_K_M.gguf --vae ..\models\vae\ltx-2.3-22b-dev_video_vae.safetensors --audio-vae ..\models\vae\ltx-2.3-22b-dev_audio_vae.safetensors --llm ..\models\text_encoders\gemma-3-12b-it-qat-UD-Q4_K_XL.gguf --embeddings-connectors ..\models\text_encoders\ltx-2.3-22b-dev_embeddings_connectors.safetensors  -p "a lovely cat" --cfg-scale 6.0 --sampling-method euler -v  -W 1280 -H 720 --diffusion-fa --offload-to-cpu --video-frames 33 -i ..\assets\ernie_image\turbo_example.png -o i2v.webm
```

<video
  src="../assets/ltx2/i2v.webm"
  controls
  muted
  style="max-width: 100%; height: auto;"></video>

### LTX-2.3 dev FLF2V

```
.\bin\Release\sd-cli.exe -M vid_gen --diffusion-model  ..\models\diffusion_models\ltx-2.3-22b-dev-UD-Q4_K_M.gguf --vae ..\models\vae\ltx-2.3-22b-dev_video_vae.safetensors --audio-vae ..\models\vae\ltx-2.3-22b-dev_audio_vae.safetensors --llm ..\models\text_encoders\gemma-3-12b-it-qat-UD-Q4_K_XL.gguf --embeddings-connectors ..\models\text_encoders\ltx-2.3-22b-dev_embeddings_connectors.safetensors  -p "glass flower blossom" --cfg-scale 6.0 --sampling-method euler -v  -W 1280 -H 720 --diffusion-fa --offload-to-cpu --video-frames 33 --init-img ..\..\ComfyUI\input\start_image.png --end-img ..\..\ComfyUI\input\end_image.png -o flf2v.webm
```

<video
  src="../assets/ltx2/flf2v.webm"
  controls
  muted
  style="max-width: 100%; height: auto;"></video>

### LTX-2.3 spatial latent upscale

LTX spatial latent upscale runs a model-backed x2 latent upsampler between the low-resolution video pass and the high-resolution refine pass. `-W` and `-H` are the pre-upscale generation size; the spatial upsampler produces x2 latent dimensions.

Put `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` under the directory passed to `--hires-upscalers-dir`, then use the model name without path or extension in `--hires-upscaler`.

```
.\bin\Release\sd-cli.exe -M vid_gen --diffusion-model  ..\models\diffusion_models\ltx-2.3-22b-dev-UD-Q4_K_M.gguf --vae ..\models\vae\ltx-2.3-22b-dev_video_vae.safetensors --audio-vae ..\models\vae\ltx-2.3-22b-dev_audio_vae.safetensors --llm ..\models\text_encoders\gemma-3-12b-it-qat-UD-Q4_K_XL.gguf --embeddings-connectors ..\models\text_encoders\ltx-2.3-22b-dev_embeddings_connectors.safetensors --hires-upscalers-dir ..\models\latent_upscale_models --hires-upscaler ltx-2.3-spatial-upscaler-x2-1.1 --hires --hires-steps 4 -p "a lovely cat" --cfg-scale 6.0 --sampling-method euler -v  -W 640 -H 360 --diffusion-fa --offload-to-cpu --video-frames 33 -i ..\assets\ernie_image\turbo_example.png -o hires_i2v.webm
```

By default, the hires refine pass uses the main sampler and scheduler, then trims the second-pass sigma schedule by `--hires-denoising-strength` (`0.7` by default). To reproduce a ComfyUI-style explicit refine schedule, pass custom hires sigmas:

```
--hires-sigmas "0.85,0.725,0.421875,0.0"
```

<video
  src="../assets/ltx2/hires_i2v.webm"
  controls
  muted
  style="max-width: 100%; height: auto;"></video>

### LTX-2.5 dev T2V

```
.\bin\Release\sd-cli.exe -M vid_gen --diffusion-model ..\models\diffusion_models\ltx-2.5-22b-dev-transformer-Q8_0.gguf --vae ..\models\vae\ltx-2.5-video-vae-conv-bf16.safetensors --audio-vae ..\models\vae\ltx-2.5-audio-vae-bf16.safetensors --llm ..\models\text_encoders\gemma4-12b-with-proj-ltx-2.5-Q8_0.gguf -p "A wide aerial shot of a red vintage convertible driving along a coastal cliff road at sunset, waves crashing below" --cfg-scale 3.0 --sampling-method euler -v -n "worst quality, low quality, blurry, distorted, artifacts" -W 1280 -H 720 --diffusion-fa --offload-to-cpu --video-frames 121 --fps 24 -o t2v.webm
```

### LTX-2.5 dev I2V

```
.\bin\Release\sd-cli.exe -M vid_gen --diffusion-model ..\models\diffusion_models\ltx-2.5-22b-dev-transformer-Q8_0.gguf --vae ..\models\vae\ltx-2.5-video-vae-conv-bf16.safetensors --audio-vae ..\models\vae\ltx-2.5-audio-vae-bf16.safetensors --llm ..\models\text_encoders\gemma4-12b-with-proj-ltx-2.5-Q8_0.gguf -p "a lovely cat blinking slowly, gentle camera push in" --cfg-scale 3.0 --sampling-method euler -v -W 1280 -H 720 --diffusion-fa --offload-to-cpu --video-frames 121 -i ..\assets\ernie_image\turbo_example.png -o i2v.webm
```

### LTX-2.5 spatial latent upscale

Works exactly like the LTX-2.3 upscaler described below; put
`ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` under `--hires-upscalers-dir` and
pass its name without path or extension to `--hires-upscaler`.

```
.\bin\Release\sd-cli.exe -M vid_gen --diffusion-model ..\models\diffusion_models\ltx-2.5-22b-dev-transformer-Q8_0.gguf --vae ..\models\vae\ltx-2.5-video-vae-conv-bf16.safetensors --audio-vae ..\models\vae\ltx-2.5-audio-vae-bf16.safetensors --llm ..\models\text_encoders\gemma4-12b-with-proj-ltx-2.5-Q8_0.gguf --hires-upscalers-dir ..\models\latent_upscale_models --hires-upscaler ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0 --hires --hires-steps 6 -p "a lovely cat" --cfg-scale 3.0 --sampling-method euler -v -W 640 -H 360 --diffusion-fa --offload-to-cpu --video-frames 121 -o hires_t2v.webm
```

## Not implemented

- The diffusion video decoder (`ltx-2.5-video-vae-bf16.safetensors`). Use the conv VAE.
- The temporal latent upscaler and the duration head (`--auto-duration`); pass
  `--video-frames` explicitly.