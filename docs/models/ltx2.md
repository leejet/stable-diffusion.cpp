# LTX-2

LTX-2.3 and LTX-2.5 support text-to-video and image-to-video generation.
LTX-2.3 uses Gemma 3 with separate embedding connectors. LTX-2.5 requires its
LTX-specific Gemma 4 encoder with the text projection included and the conv video VAE.

## Model files

### LTX-2.3

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| LTX-2.3 transformer | [Safetensors](https://huggingface.co/Kijai/LTX2.3_comfy/tree/main/diffusion_models)<br>[GGUF](https://huggingface.co/unsloth/LTX-2.3-GGUF/tree/main) | `--diffusion-model` | Required |
| Gemma 3 12B text encoder | [GGUF](https://huggingface.co/unsloth/gemma-3-12b-it-GGUF/tree/main) | `--llm` | Required for 2.3 |
| LTX-2.3 connectors | [Safetensors](https://huggingface.co/unsloth/LTX-2.3-GGUF/tree/main/text_encoders) | `--embeddings-connectors` | Required for 2.3 |
| LTX-2.3 video VAE | [Safetensors](https://huggingface.co/unsloth/LTX-2.3-GGUF/tree/main/vae) | `--vae` | Required for these examples |
| LTX-2.3 audio VAE | [Safetensors](https://huggingface.co/unsloth/LTX-2.3-GGUF/tree/main/vae) | `--audio-vae` | For audio output |
| LTX-2.3 spatial upscaler | [Safetensors](https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors) | `--hires-upscalers-dir` | For latent spatial upscaling |

### LTX-2.5

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| LTX-2.5 transformer | [Safetensors](https://huggingface.co/Lightricks/LTX-2.5/tree/main/diffusion_models)<br>[GGUF](https://huggingface.co/vantagewithai/LTX-2.5-GGUF/tree/main) | `--diffusion-model` | Required |
| LTX Gemma 4 12B with projection | projection bundled in, so no separate `--embeddings-connectors` file is needed. Google's<br>stock Gemma 4 is not a substitute.<br>[Safetensors](https://huggingface.co/Lightricks/LTX-2.5/blob/main/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors) | `--llm` | Required; stock Gemma 4 is incompatible |
| LTX-2.5 conv video VAE | The default `ltx-2.5-video-vae-bf16.safetensors` is a diffusion decoder, which is not<br>implemented here.<br>[Safetensors](https://huggingface.co/Lightricks/LTX-2.5/blob/main/vae/ltx-2.5-video-vae-conv-bf16.safetensors) | `--vae` | Use the conv variant |
| LTX-2.5 audio VAE | [Safetensors](https://huggingface.co/Lightricks/LTX-2.5/blob/main/vae/ltx-2.5-audio-vae-bf16.safetensors) | `--audio-vae` | For audio output |
| LTX-2.5 spatial upscaler | [Safetensors](https://huggingface.co/Lightricks/LTX-2.5/blob/main/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors) | `--hires-upscalers-dir` | For latent spatial upscaling |

### Optional conversion

Run executable conversion examples from the build directory using the
[shared path and shell conventions](README.md#example-conventions).

#### Quantize the LTX-2.5 text encoder

To run the text encoder quantized, convert it once with sd-cli:

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -M convert `
  -m ..\models\text_encoders\gemma4-12b-with-proj-ltx-2.5-bf16.safetensors `
  --type q8_0 `
  -o ..\models\text_encoders\gemma4-12b-with-proj-ltx-2.5-Q8_0.gguf
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -M convert \
  -m ../models/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  --type q8_0 \
  -o ../models/text_encoders/gemma4-12b-with-proj-ltx-2.5-Q8_0.gguf
```

## Usage

CLI examples use the [shared path and shell conventions](README.md#example-conventions).

### Web UI

Follow the [Web UI guide](../../examples/server/README.md#web-ui) to start the
server, then select the recipe matching your weights. Fill the component slots
below and click **Load models**.
Use the files and compatibility requirements in [Model files](#model-files).

| Recipe | Component slots |
| --- | --- |
| LTX 2.3 dev (`ltx23-dev`) | `diffusion_model`, `llm`, `embeddings_connectors`, `vae`, `tae`, `audio_vae` |
| LTX 2.3 Distilled 1.1 (`ltx23-distilled`) | `diffusion_model`, `llm`, `embeddings_connectors`, `vae`, `tae`, `audio_vae` |
| LTX 2.5 dev (`ltx25-dev`) | `diffusion_model`, `llm`, `vae`, `tae`, `audio_vae` |

Slot IDs correspond to CLI component options: `diffusion_model` uses
`--diffusion-model`, `llm` uses `--llm`, and `model` uses `-m` / `--model`.
Select optional components when required by your checkpoint or task.

### CLI

#### LTX-2.3 dev T2V

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -M vid_gen `
  --diffusion-model ..\models\diffusion_models\ltx-2.3-22b-dev-UD-Q4_K_M.gguf `
  --vae ..\models\vae\ltx-2.3-22b-dev_video_vae.safetensors `
  --audio-vae ..\models\vae\ltx-2.3-22b-dev_audio_vae.safetensors `
  --llm ..\models\text_encoders\gemma-3-12b-it-qat-UD-Q4_K_XL.gguf `
  --embeddings-connectors ..\models\text_encoders\ltx-2.3-22b-dev_embeddings_connectors.safetensors `
  -p "a lovely cat" `
  -n "worst quality, low quality, blurry, distorted, artifacts" `
  --cfg-scale 6.0 `
  --sampling-method euler `
  -W 1280 `
  -H 720 `
  --video-frames 33 `
  --fps 24 `
  --offload-to-cpu `
  -v `
  -o t2v.webm
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -M vid_gen \
  --diffusion-model ../models/diffusion_models/ltx-2.3-22b-dev-UD-Q4_K_M.gguf \
  --vae ../models/vae/ltx-2.3-22b-dev_video_vae.safetensors \
  --audio-vae ../models/vae/ltx-2.3-22b-dev_audio_vae.safetensors \
  --llm ../models/text_encoders/gemma-3-12b-it-qat-UD-Q4_K_XL.gguf \
  --embeddings-connectors ../models/text_encoders/ltx-2.3-22b-dev_embeddings_connectors.safetensors \
  -p "a lovely cat" \
  -n "worst quality, low quality, blurry, distorted, artifacts" \
  --cfg-scale 6.0 \
  --sampling-method euler \
  -W 1280 \
  -H 720 \
  --video-frames 33 \
  --fps 24 \
  --offload-to-cpu \
  -v \
  -o t2v.webm
```

<video
  src="../../assets/ltx2/t2v.webm"
  controls
  muted
  style="max-width: 100%; height: auto;"></video>

#### LTX-2.3 dev I2V

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -M vid_gen `
  --diffusion-model ..\models\diffusion_models\ltx-2.3-22b-dev-UD-Q4_K_M.gguf `
  --vae ..\models\vae\ltx-2.3-22b-dev_video_vae.safetensors `
  --audio-vae ..\models\vae\ltx-2.3-22b-dev_audio_vae.safetensors `
  --llm ..\models\text_encoders\gemma-3-12b-it-qat-UD-Q4_K_XL.gguf `
  --embeddings-connectors ..\models\text_encoders\ltx-2.3-22b-dev_embeddings_connectors.safetensors `
  -p "a lovely cat" `
  -i ..\inputs\ernie_image\turbo_example.png `
  --cfg-scale 6.0 `
  --sampling-method euler `
  -W 1280 `
  -H 720 `
  --video-frames 33 `
  --offload-to-cpu `
  -v `
  -o i2v.webm
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -M vid_gen \
  --diffusion-model ../models/diffusion_models/ltx-2.3-22b-dev-UD-Q4_K_M.gguf \
  --vae ../models/vae/ltx-2.3-22b-dev_video_vae.safetensors \
  --audio-vae ../models/vae/ltx-2.3-22b-dev_audio_vae.safetensors \
  --llm ../models/text_encoders/gemma-3-12b-it-qat-UD-Q4_K_XL.gguf \
  --embeddings-connectors ../models/text_encoders/ltx-2.3-22b-dev_embeddings_connectors.safetensors \
  -p "a lovely cat" \
  -i ../inputs/ernie_image/turbo_example.png \
  --cfg-scale 6.0 \
  --sampling-method euler \
  -W 1280 \
  -H 720 \
  --video-frames 33 \
  --offload-to-cpu \
  -v \
  -o i2v.webm
```

<video
  src="../../assets/ltx2/i2v.webm"
  controls
  muted
  style="max-width: 100%; height: auto;"></video>

#### LTX-2.3 dev FLF2V

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -M vid_gen `
  --diffusion-model ..\models\diffusion_models\ltx-2.3-22b-dev-UD-Q4_K_M.gguf `
  --vae ..\models\vae\ltx-2.3-22b-dev_video_vae.safetensors `
  --audio-vae ..\models\vae\ltx-2.3-22b-dev_audio_vae.safetensors `
  --llm ..\models\text_encoders\gemma-3-12b-it-qat-UD-Q4_K_XL.gguf `
  --embeddings-connectors ..\models\text_encoders\ltx-2.3-22b-dev_embeddings_connectors.safetensors `
  -p "glass flower blossom" `
  --init-img ..\inputs\start_image.png `
  --end-img ..\inputs\end_image.png `
  --cfg-scale 6.0 `
  --sampling-method euler `
  -W 1280 `
  -H 720 `
  --video-frames 33 `
  --offload-to-cpu `
  -v `
  -o flf2v.webm
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -M vid_gen \
  --diffusion-model ../models/diffusion_models/ltx-2.3-22b-dev-UD-Q4_K_M.gguf \
  --vae ../models/vae/ltx-2.3-22b-dev_video_vae.safetensors \
  --audio-vae ../models/vae/ltx-2.3-22b-dev_audio_vae.safetensors \
  --llm ../models/text_encoders/gemma-3-12b-it-qat-UD-Q4_K_XL.gguf \
  --embeddings-connectors ../models/text_encoders/ltx-2.3-22b-dev_embeddings_connectors.safetensors \
  -p "glass flower blossom" \
  --init-img ../inputs/start_image.png \
  --end-img ../inputs/end_image.png \
  --cfg-scale 6.0 \
  --sampling-method euler \
  -W 1280 \
  -H 720 \
  --video-frames 33 \
  --offload-to-cpu \
  -v \
  -o flf2v.webm
```

<video
  src="../../assets/ltx2/flf2v.webm"
  controls
  muted
  style="max-width: 100%; height: auto;"></video>

#### LTX-2.3 spatial latent upscale

LTX spatial latent upscale runs a model-backed x2 latent upsampler between the low-resolution video pass and the high-resolution refine pass. `-W` and `-H` are the pre-upscale generation size; the spatial upsampler produces x2 latent dimensions.

Put `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` under the directory passed to `--hires-upscalers-dir`, then use the model name without path or extension in `--hires-upscaler`.

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -M vid_gen `
  --diffusion-model ..\models\diffusion_models\ltx-2.3-22b-dev-UD-Q4_K_M.gguf `
  --vae ..\models\vae\ltx-2.3-22b-dev_video_vae.safetensors `
  --audio-vae ..\models\vae\ltx-2.3-22b-dev_audio_vae.safetensors `
  --llm ..\models\text_encoders\gemma-3-12b-it-qat-UD-Q4_K_XL.gguf `
  --embeddings-connectors ..\models\text_encoders\ltx-2.3-22b-dev_embeddings_connectors.safetensors `
  --hires-upscalers-dir ..\models\latent_upscale_models `
  -p "a lovely cat" `
  -i ..\inputs\ernie_image\turbo_example.png `
  --cfg-scale 6.0 `
  --sampling-method euler `
  -W 640 `
  -H 360 `
  --video-frames 33 `
  --hires-upscaler ltx-2.3-spatial-upscaler-x2-1.1 `
  --hires `
  --hires-steps 4 `
  --offload-to-cpu `
  -v `
  -o hires_i2v.webm
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -M vid_gen \
  --diffusion-model ../models/diffusion_models/ltx-2.3-22b-dev-UD-Q4_K_M.gguf \
  --vae ../models/vae/ltx-2.3-22b-dev_video_vae.safetensors \
  --audio-vae ../models/vae/ltx-2.3-22b-dev_audio_vae.safetensors \
  --llm ../models/text_encoders/gemma-3-12b-it-qat-UD-Q4_K_XL.gguf \
  --embeddings-connectors ../models/text_encoders/ltx-2.3-22b-dev_embeddings_connectors.safetensors \
  --hires-upscalers-dir ../models/latent_upscale_models \
  -p "a lovely cat" \
  -i ../inputs/ernie_image/turbo_example.png \
  --cfg-scale 6.0 \
  --sampling-method euler \
  -W 640 \
  -H 360 \
  --video-frames 33 \
  --hires-upscaler ltx-2.3-spatial-upscaler-x2-1.1 \
  --hires \
  --hires-steps 4 \
  --offload-to-cpu \
  -v \
  -o hires_i2v.webm
```

By default, the hires refine pass uses the main sampler and scheduler, then trims the second-pass sigma schedule by `--hires-denoising-strength` (`0.7` by default). To reproduce a ComfyUI-style explicit refine schedule, pass custom hires sigmas:

**Windows (PowerShell) arguments**

```text
--hires-sigmas "0.85,0.725,0.421875,0.0"
```

**Linux/macOS arguments**

```text
--hires-sigmas 0.85,0.725,0.421875,0.0
```

<video
  src="../../assets/ltx2/hires_i2v.webm"
  controls
  muted
  style="max-width: 100%; height: auto;"></video>

#### LTX-2.5 dev T2V

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -M vid_gen `
  --diffusion-model ..\models\diffusion_models\ltx-2.5-22b-dev-transformer-Q8_0.gguf `
  --vae ..\models\vae\ltx-2.5-video-vae-conv-bf16.safetensors `
  --audio-vae ..\models\vae\ltx-2.5-audio-vae-bf16.safetensors `
  --llm ..\models\text_encoders\gemma4-12b-with-proj-ltx-2.5-Q8_0.gguf `
  -p "A wide aerial shot of a red vintage convertible driving along a coastal cliff road at sunset, waves crashing below" `
  -n "worst quality, low quality, blurry, distorted, artifacts" `
  --cfg-scale 3.0 `
  --sampling-method euler `
  -W 1280 `
  -H 720 `
  --video-frames 121 `
  --fps 24 `
  --offload-to-cpu `
  -v `
  -o t2v.webm
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -M vid_gen \
  --diffusion-model ../models/diffusion_models/ltx-2.5-22b-dev-transformer-Q8_0.gguf \
  --vae ../models/vae/ltx-2.5-video-vae-conv-bf16.safetensors \
  --audio-vae ../models/vae/ltx-2.5-audio-vae-bf16.safetensors \
  --llm ../models/text_encoders/gemma4-12b-with-proj-ltx-2.5-Q8_0.gguf \
  -p "A wide aerial shot of a red vintage convertible driving along a coastal cliff road at sunset, waves crashing below" \
  -n "worst quality, low quality, blurry, distorted, artifacts" \
  --cfg-scale 3.0 \
  --sampling-method euler \
  -W 1280 \
  -H 720 \
  --video-frames 121 \
  --fps 24 \
  --offload-to-cpu \
  -v \
  -o t2v.webm
```

#### LTX-2.5 dev I2V

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -M vid_gen `
  --diffusion-model ..\models\diffusion_models\ltx-2.5-22b-dev-transformer-Q8_0.gguf `
  --vae ..\models\vae\ltx-2.5-video-vae-conv-bf16.safetensors `
  --audio-vae ..\models\vae\ltx-2.5-audio-vae-bf16.safetensors `
  --llm ..\models\text_encoders\gemma4-12b-with-proj-ltx-2.5-Q8_0.gguf `
  -p "a lovely cat blinking slowly, gentle camera push in" `
  -i ..\inputs\ernie_image\turbo_example.png `
  --cfg-scale 3.0 `
  --sampling-method euler `
  -W 1280 `
  -H 720 `
  --video-frames 121 `
  --offload-to-cpu `
  -v `
  -o i2v.webm
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -M vid_gen \
  --diffusion-model ../models/diffusion_models/ltx-2.5-22b-dev-transformer-Q8_0.gguf \
  --vae ../models/vae/ltx-2.5-video-vae-conv-bf16.safetensors \
  --audio-vae ../models/vae/ltx-2.5-audio-vae-bf16.safetensors \
  --llm ../models/text_encoders/gemma4-12b-with-proj-ltx-2.5-Q8_0.gguf \
  -p "a lovely cat blinking slowly, gentle camera push in" \
  -i ../inputs/ernie_image/turbo_example.png \
  --cfg-scale 3.0 \
  --sampling-method euler \
  -W 1280 \
  -H 720 \
  --video-frames 121 \
  --offload-to-cpu \
  -v \
  -o i2v.webm
```

#### LTX-2.5 spatial latent upscale

Works like the LTX-2.3 upscaler described above; put
`ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` under `--hires-upscalers-dir` and
pass its name without path or extension to `--hires-upscaler`.

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -M vid_gen `
  --diffusion-model ..\models\diffusion_models\ltx-2.5-22b-dev-transformer-Q8_0.gguf `
  --vae ..\models\vae\ltx-2.5-video-vae-conv-bf16.safetensors `
  --audio-vae ..\models\vae\ltx-2.5-audio-vae-bf16.safetensors `
  --llm ..\models\text_encoders\gemma4-12b-with-proj-ltx-2.5-Q8_0.gguf `
  --hires-upscalers-dir ..\models\latent_upscale_models `
  -p "a lovely cat" `
  --cfg-scale 3.0 `
  --sampling-method euler `
  -W 640 `
  -H 360 `
  --video-frames 121 `
  --hires-upscaler ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0 `
  --hires `
  --hires-steps 6 `
  --offload-to-cpu `
  -v `
  -o hires_t2v.webm
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -M vid_gen \
  --diffusion-model ../models/diffusion_models/ltx-2.5-22b-dev-transformer-Q8_0.gguf \
  --vae ../models/vae/ltx-2.5-video-vae-conv-bf16.safetensors \
  --audio-vae ../models/vae/ltx-2.5-audio-vae-bf16.safetensors \
  --llm ../models/text_encoders/gemma4-12b-with-proj-ltx-2.5-Q8_0.gguf \
  --hires-upscalers-dir ../models/latent_upscale_models \
  -p "a lovely cat" \
  --cfg-scale 3.0 \
  --sampling-method euler \
  -W 640 \
  -H 360 \
  --video-frames 121 \
  --hires-upscaler ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0 \
  --hires \
  --hires-steps 6 \
  --offload-to-cpu \
  -v \
  -o hires_t2v.webm
```

## Notes and limitations

Not implemented:

- The diffusion video decoder (`ltx-2.5-video-vae-bf16.safetensors`). Use the conv VAE.
- The temporal latent upscaler and the duration head (`--auto-duration`); pass
  `--video-frames` explicitly.
