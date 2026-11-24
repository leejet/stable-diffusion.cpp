# MiniMax-H3

MiniMax-H3 jointly generates video and stereo audio with a packed diffusion
transformer. The implementation supports text-to-audio-video (T2VA), optional
first-frame conditioning (I2VA), first/last-frame conditioning (FL2VA), and
image/video/audio reference conditioning (Ref2VA).

## Model files

Pass the four MiniMax-H3 components separately:

- `--diffusion-model`: MiniMax-H3 diffusion transformer
- `--vae`: MiniMax-H3 video VAE
- `--audio-vae`: MiniMax-H3 audio VAE
- `--llm`: the MiniMax-H3 Qwen3-VL-32B text encoder, or Qwen3-VL-4B with the P-H3 text connector

For the 32B option, use the MiniMax-H3 variant: Qwen3-VL-32B truncated to 50
language layers and exported without the final language-model normalization.

For the smaller 4B option, pass the P-H3-Image text encoder, which bundles the
connector, with `--llm`. Alternatively, pass Qwen3-VL-4B with `--llm` and the
[separate P-H3 text connector](https://huggingface.co/PrunaAI/P-H3-Image/blob/main/text_connector/diffusion_pytorch_model.safetensors)
with `--embeddings-connectors`. For example, replace the `--llm` argument in the
commands below with either:

**Windows (PowerShell) arguments**

```text
--llm ..\models\text_encoders\p-h3-image_text_encoder_int8_convrot.safetensors
```

**Linux/macOS arguments**

```text
--llm ../models/text_encoders/p-h3-image_text_encoder_int8_convrot.safetensors
```

or:

**Windows (PowerShell) arguments**

```text
--llm ..\models\text_encoders\qwen3vl_4b.gguf `
  --embeddings-connectors ..\models\text_encoders\p-h3-image_text_connector.safetensors
```

**Linux/macOS arguments**

```text
--llm ../models/text_encoders/qwen3vl_4b.gguf \
  --embeddings-connectors ../models/text_encoders/p-h3-image_text_connector.safetensors
```

Reference-image and reference-video conditioning needs the matching encoder's
Qwen3-VL vision tower, including the three DeepStack mergers. If stored
separately, pass it with `--llm_vision`. Keep using the MiniMax-H3 video and audio
VAEs with either text encoder.

In DGL, use `load_text_encoder` with `family = "minimax_h3"`. For a separate
4B connector or vision projector, pass `connectors_path` or `vision_path` to
that loader. The native generation API trims decoded audio to the duration of
the output video. In scripts, save decoded frames and audio with
`video::save(path, {frames=frames, fps_num=24, fps_den=1}, audio=audio, trim_audio=true)`
to match the generated clip duration.
For video presentation, pass frame tensors to `encode_prompt` as
`presentation=[{video={frames=frames, fps_num=24, fps_den=1}, stride=12}]`. Frame rate and sampling
stride are explicit; VAE reference latents remain a separate input.

Both the original time-embedder DiT and the smaller AdaLN curve-table variant
are detected from their weights.

### Download weights

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| minimax_h3_fl2va/minimax_h3_ref2va | [Safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/diffusion_models)<br>[GGUF](https://huggingface.co/leejet/MiniMax-H3-GGUF/tree/main) | `--diffusion-model` | Required |
| qwen3vl_32b_minimax_h3 | [Safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/text_encoders)<br>[GGUF](https://huggingface.co/leejet/MiniMax-H3-GGUF/tree/main) | `--llm` | Use matching text encoder |
| P-H3-Image text encoder (Qwen3-VL-4B with the text connector) | [Safetensors](https://huggingface.co/PrunaAI/P-H3-Image-Comfy/tree/main/text_encoders) | `--llm` | Use matching text encoder |
| vae | [Safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) | `--vae` | Use matching VAE |
| audio vae | [Safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) | `--audio-vae` | Required |

## Usage

### Web UI

Follow the [Web UI guide](../../examples/server/README.md#web-ui) to start the
server, then select the recipe matching your weights. Fill the component slots
below and click **Load models**.
Use the files and compatibility requirements in [Model files](#model-files).

| Recipe | Component slots |
| --- | --- |
| MiniMax H3 FL2VA (`minimax-h3-fl2va`) | `diffusion_model`, `llm`, `llm_vision`, `embeddings_connectors`, `vae`, `tae`, `audio_vae` |
| MiniMax H3 Ref2VA (`minimax-h3-ref2va`) | `diffusion_model`, `llm`, `llm_vision`, `embeddings_connectors`, `vae`, `tae`, `audio_vae` |
| HyperFlow H3 Flash FL2VA (`hyperflow-h3-flash-fl2va`) | `diffusion_model`, `llm`, `llm_vision`, `embeddings_connectors`, `vae`, `tae`, `audio_vae` |
| HyperFlow H3 Flash Ref2VA (`hyperflow-h3-flash-ref2va`) | `diffusion_model`, `llm`, `llm_vision`, `embeddings_connectors`, `vae`, `tae`, `audio_vae` |

Slot IDs correspond to CLI component options: `diffusion_model` uses
`--diffusion-model`, `llm` uses `--llm`, and `model` uses `-m` / `--model`.
Select optional components when required by your checkpoint or task.

### DGL

| Example | Description |
| --- | --- |
| [Text to audio/video](../dgl/minimax_h3/minimax-h3-text-to-av.dgl) | Generate video and its soundtrack together. |
| [First/last-frame conditioning](../dgl/minimax_h3/minimax-h3-first-last-frame.dgl) | Guide the video with endpoint images. |
| [Reference to audio/video](../dgl/minimax_h3/minimax-h3-reference-to-av.dgl) | Use image, video and audio references. |
| [Interpolated latent hires](../dgl/minimax_h3/minimax-h3-latent-hires.dgl) | Resize the draft latent, then run a second sampling pass. |
| [Learned latent hires](../dgl/minimax_h3/minimax-h3-learned-hires.dgl) | Use a learned latent upscaler between sampling passes. |

#### Run a script

Edit paths, prompts and settings in the script's `var` block, then run from
the build directory. Paths inside the script resolve from the working directory.

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe -M dgl --script ..\docs\dgl\minimax_h3\minimax-h3-text-to-av.dgl
```

**Linux/macOS**

```sh
./bin/sd-cli -M dgl --script ../docs/dgl/minimax_h3/minimax-h3-text-to-av.dgl
```

Choose another script from the table to run a different task. Add `--check`
to validate syntax and types without loading weights. See the
[shared run instructions](../dgl/README.md#run-an-example) for build paths
and the server's file-access rules.

#### Latent upscaling

Use `load_latent_upscaler` to load the learned spatial upscaler.
`upscale_latent` takes the video part of the draft diffusion latent
and returns a diffusion latent for the refine pass. Pass the target width
and height in latent units. A resize target uses even dimensions
(multiples of 32 in pixel units).

```dgl
upscaler = load_latent_upscaler("minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors")
bigger = upscale_latent(upscaler, draft_av.video, width = 76, height = 42)
```

### Legacy CLI

CLI examples use the [shared path and shell conventions](README.md#example-conventions).

#### Text-to-audio-video

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -M vid_gen `
  --diffusion-model ..\models\diffusion_models\minimax_h3_fl2va-Q4_K_M.gguf `
  --vae ..\models\vae\minimax_h3_video_vae_fp16.safetensors `
  --audio-vae ..\models\vae\minimax_h3_audio_vae_fp32.safetensors `
  --llm ..\models\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf `
  -p "A cute American Shorthair silver tabby kitten surfs on a tropical ocean wave, riding a white surfboard with the clear text 'sd.cpp' on it. Cinematic tracking shot, realistic water, bright sunlight, smooth motion, and consistent character appearance. Add upbeat tropical surf-rock background music with cheerful drums and guitar, synchronized with the kitten`鈥檚 energetic surfing." `
  --cfg-scale 1.0 `
  -W 864 `
  -H 480 `
  --fps 24 `
  --video-frames 56 `
  --rng cpu `
  --offload-to-cpu `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -M vid_gen \
  --diffusion-model ../models/diffusion_models/minimax_h3_fl2va-Q4_K_M.gguf \
  --vae ../models/vae/minimax_h3_video_vae_fp16.safetensors \
  --audio-vae ../models/vae/minimax_h3_audio_vae_fp32.safetensors \
  --llm ../models/text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  -p "A cute American Shorthair silver tabby kitten surfs on a tropical ocean wave, riding a white surfboard with the clear text 'sd.cpp' on it. Cinematic tracking shot, realistic water, bright sunlight, smooth motion, and consistent character appearance. Add upbeat tropical surf-rock background music with cheerful drums and guitar, synchronized with the kitten鈥檚 energetic surfing." \
  --cfg-scale 1.0 \
  -W 864 \
  -H 480 \
  --fps 24 \
  --video-frames 56 \
  --rng cpu \
  --offload-to-cpu \
  -v
```

<video src=../../assets/minimax-h3/t2av.mp4 controls="controls" muted="muted" type="video/mp4"></video>

Omitting `--audio-vae` still runs the joint diffusion model but produces video without a
decoded audio track.

#### First/last-frame conditioning

Add `--init-img` for I2VA, or both `--init-img` and `--end-img` for FL2VA:

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -M vid_gen `
  --diffusion-model ..\models\diffusion_models\minimax_h3_fl2va-Q4_K_M.gguf `
  --vae ..\models\vae\minimax_h3_video_vae_fp16.safetensors `
  --audio-vae ..\models\vae\minimax_h3_audio_vae_fp32.safetensors `
  --llm ..\models\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf `
  -p "a lovely cat" `
  -i ..\inputs\ernie_image\turbo_example.png `
  --cfg-scale 1.0 `
  -W 864 `
  -H 480 `
  --fps 24 `
  --video-frames 56 `
  --rng cpu `
  --offload-to-cpu `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -M vid_gen \
  --diffusion-model ../models/diffusion_models/minimax_h3_fl2va-Q4_K_M.gguf \
  --vae ../models/vae/minimax_h3_video_vae_fp16.safetensors \
  --audio-vae ../models/vae/minimax_h3_audio_vae_fp32.safetensors \
  --llm ../models/text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  -p "a lovely cat" \
  -i ../inputs/ernie_image/turbo_example.png \
  --cfg-scale 1.0 \
  -W 864 \
  -H 480 \
  --fps 24 \
  --video-frames 56 \
  --rng cpu \
  --offload-to-cpu \
  -v
```

<video src=../../assets/minimax-h3/i2av.mp4 controls="controls" muted="muted" type="video/mp4"></video>

#### Reference-to-audio-video conditioning

Ref2VA accepts any combination of reference images, reference videos, paired
video soundtracks, and standalone audio references:

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -M vid_gen `
  --diffusion-model ..\models\diffusion_models\minimax_h3_ref2va_pruned-Q4_K_M.gguf `
  --vae ..\models\vae\minimax_h3_video_vae_fp16.safetensors `
  --audio-vae ..\models\vae\minimax_h3_audio_vae_fp32.safetensors `
  --llm ..\models\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf `
  -p "Use the cat from <Picture 1> as the main character. Keep the cat`鈥檚 appearance, fur color, facial features, and identity consistent with the reference image. Create a 2-second cinematic video: start with an extreme close-up shot of the cat`鈥檚 face, focusing on its cute expression and detailed fur texture. The camera slowly rotates around the cat`鈥檚 head, creating a dynamic reveal. Then smoothly pull back and zoom out to reveal the full scene: the cat is standing confidently on a surfboard, riding ocean waves. Water splashes around the board, sea breeze gently moves the cat`鈥檚 fur, and the cat maintains a cute and fearless expression while surfing. Smooth camera movement, cinematic orbit shot, seamless zoom-out transition, low-angle wide shot, realistic ocean environment, golden sunlight, dynamic waves, high-quality realistic style, natural motion, no distortion, keep the cat`鈥檚 identity unchanged." `
  -r ..\inputs\ernie_image\turbo_example.png `
  --cfg-scale 1.0 `
  -W 864 `
  -H 480 `
  --fps 24 `
  --video-frames 56 `
  --rng cpu `
  --offload-to-cpu `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -M vid_gen \
  --diffusion-model ../models/diffusion_models/minimax_h3_ref2va_pruned-Q4_K_M.gguf \
  --vae ../models/vae/minimax_h3_video_vae_fp16.safetensors \
  --audio-vae ../models/vae/minimax_h3_audio_vae_fp32.safetensors \
  --llm ../models/text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  -p "Use the cat from <Picture 1> as the main character. Keep the cat鈥檚 appearance, fur color, facial features, and identity consistent with the reference image. Create a 2-second cinematic video: start with an extreme close-up shot of the cat鈥檚 face, focusing on its cute expression and detailed fur texture. The camera slowly rotates around the cat鈥檚 head, creating a dynamic reveal. Then smoothly pull back and zoom out to reveal the full scene: the cat is standing confidently on a surfboard, riding ocean waves. Water splashes around the board, sea breeze gently moves the cat鈥檚 fur, and the cat maintains a cute and fearless expression while surfing. Smooth camera movement, cinematic orbit shot, seamless zoom-out transition, low-angle wide shot, realistic ocean environment, golden sunlight, dynamic waves, high-quality realistic style, natural motion, no distortion, keep the cat鈥檚 identity unchanged." \
  -r ../inputs/ernie_image/turbo_example.png \
  --cfg-scale 1.0 \
  -W 864 \
  -H 480 \
  --fps 24 \
  --video-frames 56 \
  --rng cpu \
  --offload-to-cpu \
  -v
```

<video src=../../assets/minimax-h3/r2av.mp4 controls="controls" muted="muted" type="video/mp4"></video>

`--ref-image`, `--ref-video`, and `--ref-audio` can each be repeated. A
reference video can be a directory of image frames sorted lexicographically,
treated as 24 fps, or a file decoded and sampled at 24 fps. File inputs support
AVI, static/animated WebP and this project's independent-frame VP8 WebM;
see [DGL media I/O](../dgl.md) for codec and build requirements.
Reference files retain supported embedded audio by default. Repeated
`--ref-video-audio` WAV files are paired by index with repeated `--ref-video`
inputs and replace their embedded audio; the replaced track is not decoded.
WAV PCM (8/16/24/32-bit) and 32/64-bit floating-point samples are accepted; audio is converted to stereo 32 kHz during
generation.

Reference inputs are presented to Qwen3-VL in image, video, then audio order.
Videos are sampled at 2 fps for the Qwen presentation while their full 24 fps
latents condition the diffusion transformer. Paired video and audio references
share the same timeline. Ref2VA cannot be combined with `--init-img` or
`--end-img` in one request.

Reference images preserve their aspect ratio and are downscaled only when
their pixel area exceeds the generation canvas. Each axis is then snapped to
the nearest 32-pixel multiple, with a minimum of 32. An axis may grow by up to
16 pixels during alignment. Custom reference sizes and preprocessing require a
[DGL program](../guides/edit.md#custom-reference-processing).

The C API exposes the same inputs through `ref_images`, `ref_videos`, and
the `ref_audios` video-option key. Each `sd_video_t` supplies its own
frame rate and optional soundtrack; non-24-fps inputs are resampled internally.

#### Sol-H3-VAE decoding

Pass a [Sol-H3-VAE checkpoint](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae)
with `--vae` to use the distilled video decoder. The ComfyUI FP16 and INT8
ConvRot checkpoints include the MiniMax-H3 encoder for image and video inputs.
Supply `--audio-vae` as usual to decode the soundtrack.

Sol decodes the whole clip by default. Use `--vae-tiling` for spatial tiling or
`--temporal-tiling` for temporal tiling. Either can reduce memory use but may
affect output quality because the decoder has less context at tile boundaries.

Temporal tiles default to 10 latent frames with 5 latent frames of overlap.
Set `temporal_tile_frames` and `temporal_tile_overlap` in
`--extra-tiling-args` to override them. Both must be multiples of 5 latent
frames, and overlap must be smaller than the tile. The last tile may be shorter.
Circular decoding requires spatial tiling on each circular axis, with a tile
size smaller than the input on that axis.

#### Latent hires second pass (two-stage sampling)

MiniMax-H3 supports `--hires` with two interchangeable second-stage upscalers:

- an interpolating latent upscaler (`Latent`, `Latent (nearest)`,
  `Latent (bicubic)`, ...): after the first pass finishes, the packed AV latent
  is resampled in latent space to `--hires-width` x `--hires-height` and
  refined by a second sampling pass;
- a model-backed upscaler (`--hires-upscaler <name>` + `--hires-upscalers-dir`):
  a H3-shaped learned latent upscaler checkpoint runs between the passes,
  like the LTX spatial upsampler. This is the sd.cpp equivalent of the
  two-sampler workflows used for H3 in node-based frontends (e.g. the
  community 3D-conv latent upscaler from
  [LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)).

Draft at a low megapixel count (e.g. 0.4 MP) and refine at a higher one
(e.g. 0.8 MP) instead of paying for the full resolution in every step:

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -M vid_gen `
  --diffusion-model ..\models\diffusion_models\minimax_h3_fl2va-Q4_K_M.gguf `
  --vae ..\models\vae\minimax_h3_video_vae_fp16.safetensors `
  --audio-vae ..\models\vae\minimax_h3_audio_vae_fp32.safetensors `
  --llm ..\models\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf `
  -p "a lovely cat" `
  --cfg-scale 1.0 `
  -W 864 `
  -H 480 `
  --video-frames 56 `
  --rng cpu `
  --hires `
  --hires-upscaler "Latent (bicubic)" `
  --hires-width 1216 `
  --hires-height 672 `
  --hires-steps 4 `
  --hires-denoising-strength 0.4 `
  --diffusion-fa `
  --offload-to-cpu `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -M vid_gen \
  --diffusion-model ../models/diffusion_models/minimax_h3_fl2va-Q4_K_M.gguf \
  --vae ../models/vae/minimax_h3_video_vae_fp16.safetensors \
  --audio-vae ../models/vae/minimax_h3_audio_vae_fp32.safetensors \
  --llm ../models/text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  -p "a lovely cat" \
  --cfg-scale 1.0 \
  -W 864 \
  -H 480 \
  --video-frames 56 \
  --rng cpu \
  --hires \
  --hires-upscaler "Latent (bicubic)" \
  --hires-width 1216 \
  --hires-height 672 \
  --hires-steps 4 \
  --hires-denoising-strength 0.4 \
  --diffusion-fa \
  --offload-to-cpu \
  -v
```

or with the learned upscaler:

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -M vid_gen ... `
  --hires-upscalers-dir ..\models\latent_upscale_models `
  --hires `
  --hires-upscaler h3-latent-upscaler-3d `
  --hires-width 1216 `
  --hires-height 672 `
  --hires-steps 4 `
  --hires-denoising-strength 0.4
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -M vid_gen ... \
  --hires-upscalers-dir ../models/latent_upscale_models \
  --hires \
  --hires-upscaler h3-latent-upscaler-3d \
  --hires-width 1216 \
  --hires-height 672 \
  --hires-steps 4 \
  --hires-denoising-strength 0.4
```

Notes for this mode:

- The learned upscaler expects a second normalization of the diffusion latent,
  matching its training convention. The runner applies and reverses this extra
  transform internally; callers should pass the usual H3 diffusion latents.
- The frame count is never changed by the second stage, so the packed audio
  latent length stays valid and the generated soundtrack is preserved.
- Width and height still have to be multiples of 32; the hires target is
  aligned like any other resolution. The learned upscaler additionally wants
  an even latent size on both axes, which 32-pixel targets provide.
- `--hires-sigmas` can replace `--hires-denoising-strength` when the second
  stage should resume from an explicit sigma schedule, which is how most
  two-stage H3 workflows split the sampler.
- Keyframe conditioning (`--init-img` / `--end-img`) is not available together
  with hires yet: those latents are encoded at the first stage resolution and
  would have to be re-encoded for the hires canvas. Use T2VA or Ref2VA.
- The second stage runs at the larger latent, so it is the memory-hungry half;
  combine it with quantization, `--vae-tiling` and `--offload-to-cpu` on
  small cards.
- The learned upscaler is scale-conditional: the target-to-source ratio is
  fed into the network as an embedding, 1.2-2.0x is the trained domain, and
  downscaling is rejected.

#### HyperFlow H3 Flash

[HyperFlow H3 Flash](https://huggingface.co/VideoRebirthAI/hyperflow-h3-flash)
uses the shared components and input options described in this guide. Replace
`--diffusion-model` with the matching HyperFlow FL2VA or Ref2VA checkpoint and
use these sampling options:

**Windows (PowerShell) arguments**

```text
--sampling-method euler `
  --cfg-scale 1 `
  --sigmas hyperflow-h3-flash-6
```

**Linux/macOS arguments**

```text
--sampling-method euler \
  --cfg-scale 1 \
  --sigmas hyperflow-h3-flash-6
```

Use `hyperflow-h3-flash-4` or `hyperflow-h3-flash-8` for 4 or 8 steps. These
presets also work with `--hires-sigmas`; their length determines the step count.
They already include video flow shift 12, so keep the default flow shift.

## Notes and limitations

- Width and height are aligned upward to a multiple of 32.
- Frame count is aligned upward to the `17k + 5` grid, with a minimum of 5.
- MiniMax-H3 runs at 24 fps; another requested value is overridden.
- Attention head grouping is controlled by
  `--model-args minimax_h3_attn_chunk_budget=2`, in GiB (default: `2`).
  The budget estimates the size of a single FP32 Q/K tensor per group:
  `head_dim * tokens * batch * 4 bytes * heads_per_group`. The group size adapts
  to the sequence length, with a minimum of one head; inputs that fit use all
  heads together. Set `0` to disable grouping, or a smaller budget such as
  `0.25` to reduce peak activation memory for long sequences at the cost of
  extra launches and concatenation. Each head still attends to the full sequence.
  This is a grouping target, not a cap on total attention memory or VRAM usage;
  other intermediates, backend workspace, and weights require additional memory.
  `--max-vram` and `--offload-to-cpu` control weight residency, but the active
  block's working memory must also fit within the VRAM budget.
- The default video flow shift is 12. The audio stream is mapped internally to
  its shift of 3, so the regular samplers can operate on the packed AV latent.
  CLI and C API translation keep this mapping consistent when `--flow-shift`
  is overridden, including hires refinement.
