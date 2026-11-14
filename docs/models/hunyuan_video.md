# HunyuanVideo 1.5

HunyuanVideo 1.5 uses a HunyuanVideo diffusion transformer, a causal video VAE, Qwen2.5-VL 7B for the main text conditioning,
and ByT5 Small GlyphXL for glyph-aware text conditioning.

## Model files

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| HunyuanVideo 1.5 | [Safetensors](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/tree/main/split_files/diffusion_models) | `--diffusion-model` | Required |
| vae | [Safetensors](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/tree/main/split_files/vae) | `--vae` | Use matching VAE |
| qwen_2.5_vl 7b | [Safetensors](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files/text_encoders)<br>[GGUF](https://huggingface.co/mradermacher/Qwen2.5-VL-7B-Instruct-GGUF/tree/main) | `--llm` | Use matching text encoder |
| byt5 small glyphxl | safetensros: [Download](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/tree/main/split_files/text_encoders) | `--t5xxl` | Required |

## Usage

CLI examples use the [shared path and shell conventions](README.md#example-conventions).

### Web UI

Follow the [Web UI guide](../../examples/server/README.md#web-ui) to start the
server, then select the recipe matching your weights. Fill the component slots
below and click **Load models**.
Use the files and compatibility requirements in [Model files](#model-files).

| Recipe | Component slots |
| --- | --- |
| HunyuanVideo 1.5 T2V (`hunyuan-video-15-t2v`) | `diffusion_model`, `llm`, `t5xxl`, `vae`, `tae` |
| HunyuanVideo 1.5 I2V (`hunyuan-video-15-i2v`) | `diffusion_model`, `llm`, `t5xxl`, `vae`, `tae` |

Use the T2V checkpoint with the T2V recipe. For I2V, select the I2V checkpoint
and recipe, then supply the starting image in Video Generation.

Slot IDs correspond to CLI component options: `diffusion_model` uses
`--diffusion-model`, `llm` uses `--llm`, and `model` uses `-m` / `--model`.
Select optional components when required by your checkpoint or task.

### CLI

#### Text-to-video example

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -M vid_gen `
  --diffusion-model ..\models\diffusion_models\hunyuanvideo1.5_720p_t2v_fp16.safetensors `
  --vae ..\models\vae\hunyuanvideo15_vae_fp16.safetensors `
  --llm ..\models\text_encoders\qwen_2.5_vl_7b.safetensors `
  --t5xxl ..\models\text_encoders\byt5_small_glyphxl_fp16.safetensors `
  -p "a lovely cat" `
  --cfg-scale 6.0 `
  --sampling-method euler `
  -W 1280 `
  -H 720 `
  --video-frames 33 `
  --offload-to-cpu `
  --vae-tiling `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -M vid_gen \
  --diffusion-model ../models/diffusion_models/hunyuanvideo1.5_720p_t2v_fp16.safetensors \
  --vae ../models/vae/hunyuanvideo15_vae_fp16.safetensors \
  --llm ../models/text_encoders/qwen_2.5_vl_7b.safetensors \
  --t5xxl ../models/text_encoders/byt5_small_glyphxl_fp16.safetensors \
  -p "a lovely cat" \
  --cfg-scale 6.0 \
  --sampling-method euler \
  -W 1280 \
  -H 720 \
  --video-frames 33 \
  --offload-to-cpu \
  --vae-tiling \
  -v
```

<video src=../../assets/hunyuan_video/hy1.5_t2v.mp4 controls="controls" muted="muted" type="video/mp4"></video>
