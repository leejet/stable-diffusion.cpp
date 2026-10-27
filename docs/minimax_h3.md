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
- `--llm`: the MiniMax-H3 Qwen3-VL-32B text encoder checkpoint

The text encoder must be the MiniMax-H3 variant: Qwen3-VL-32B truncated to 50
language layers and exported without the final language-model normalization.
Its Qwen3-VL vision tower, including the three DeepStack mergers, must also be
present. If the vision tower is stored separately, pass it with `--llm_vision`.

Both the original time-embedder DiT and the smaller AdaLN curve-table variant
are detected from their weights.

### Download weights

- Download minimax_h3_fl2va/minimax_h3_ref2va
    - safetensors: https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/diffusion_models
    - gguf: https://huggingface.co/leejet/MiniMax-H3-GGUF/tree/main
- Download qwen3vl_32b_minimax_h3
    - safetensors: https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/text_encoders
    - gguf: https://huggingface.co/leejet/MiniMax-H3-GGUF/tree/main
- Download vae
    - safetensors: https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae
- Download audio vae
    - safetensors: https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae

## Text-to-audio-video

```sh
.\bin\Release\sd-cli.exe -M vid_gen --diffusion-model  ..\models\diffusion_models\minimax_h3_fl2va-Q4_K_M.gguf --vae ..\models\vae\minimax_h3_video_vae_fp16.safetensors --audio-vae ..\models\vae\minimax_h3_audio_vae_fp32.safetensors --llm ..\models\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf -p "A cute American Shorthair silver tabby kitten surfs on a tropical ocean wave, riding a white surfboard with the clear text 'sd.cpp' on it. Cinematic tracking shot, realistic water, bright sunlight, smooth motion, and consistent character appearance. Add upbeat tropical surf-rock background music with cheerful drums and guitar, synchronized with the kitten’s energetic surfing." --cfg-scale 1.0 -v -W 864 -H 480 --offload-to-cpu --rng cpu --fps 24 --video-frames 56
```

<video src=../assets/minimax-h3/t2av.mp4 controls="controls" muted="muted" type="video/mp4"></video>

Omitting `--audio-vae` still runs the joint diffusion model but produces video without a
decoded audio track.

## First/last-frame conditioning

Add `--init-img` for I2VA, or both `--init-img` and `--end-img` for FL2VA:

```sh
.\bin\Release\sd-cli.exe -M vid_gen --diffusion-model  ..\models\diffusion_models\minimax_h3_fl2va-Q4_K_M.gguf --vae ..\models\vae\minimax_h3_video_vae_fp16.safetensors --audio-vae ..\models\vae\minimax_h3_audio_vae_fp32.safetensors --llm ..\models\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf -p "a lovely cat" -i ..\assets\ernie_image\turbo_example.png --cfg-scale 1.0 -v -W 864 -H 480 --offload-to-cpu --rng cpu --fps 24 --video-frames 56
```

<video src=../assets/minimax-h3/i2av.mp4 controls="controls" muted="muted" type="video/mp4"></video>

## Reference-to-audio-video conditioning

Ref2VA accepts any combination of reference images, reference videos, paired
video soundtracks, and standalone audio references:

```sh
.\bin\Release\sd-cli.exe -M vid_gen --diffusion-model  ..\models\diffusion_models\minimax_h3_ref2va_pruned-Q4_K_M.gguf --vae ..\models\vae\minimax_h3_video_vae_fp16.safetensors --audio-vae ..\models\vae\minimax_h3_audio_vae_fp32.safetensors --llm ..\models\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf -p "Use the cat from <Picture 1> as the main character. Keep the cat’s appearance, fur color, facial features, and identity consistent with the reference image. Create a 2-second cinematic video: start with an extreme close-up shot of the cat’s face, focusing on its cute expression and detailed fur texture. The camera slowly rotates around the cat’s head, creating a dynamic reveal. Then smoothly pull back and zoom out to reveal the full scene: the cat is standing confidently on a surfboard, riding ocean waves. Water splashes around the board, sea breeze gently moves the cat’s fur, and the cat maintains a cute and fearless expression while surfing. Smooth camera movement, cinematic orbit shot, seamless zoom-out transition, low-angle wide shot, realistic ocean environment, golden sunlight, dynamic waves, high-quality realistic style, natural motion, no distortion, keep the cat’s identity unchanged." -r ..\assets\ernie_image\turbo_example.png --cfg-scale 1.0 -v -W 864 -H 480 --offload-to-cpu --rng cpu --fps 24 --video-frames 56
```

<video src=../assets/minimax-h3/r2av.mp4 controls="controls" muted="muted" type="video/mp4"></video>

`--ref-image`, `--ref-video`, and `--ref-audio` can each be repeated. A
reference video is a directory of image frames sorted lexicographically and is
treated as 24 fps. Repeated `--ref-video-audio` WAV files are paired by index
with repeated `--ref-video` inputs. WAV PCM (8/16/24/32-bit) and 32/64-bit
floating-point samples are accepted; audio is converted to stereo 32 kHz by the
pipeline.

Reference inputs are presented to Qwen3-VL in image, video, then audio order.
Videos are sampled at 2 fps for the Qwen presentation while their full 24 fps
latents condition the diffusion transformer. Paired video and audio references
share the same timeline. Ref2VA cannot be combined with `--init-img` or
`--end-img` in one request.

Reference images keep their aspect ratio and are only downscaled when they
exceed the configured limit; neither mode scales a reference up. The limit is
selected by `ref_image_size`. The default, `match`, uses the pixel area of the
requested generation canvas. Pass `--ref-image-args "ref_image_size=max"` (or
`ref_image_size` in `sd_vid_gen_params_t.ref_image_args`) to use a 2048 px short
edge instead, which preserves more identity detail at a higher sampling cost:
reference latents are attended at every sampling step.

Whichever limit applies, each axis is then snapped to the nearest 32-pixel
multiple. An axis of at least 16 pixels can therefore grow by up to 16 pixels
past the scaled size (a 48-pixel axis becomes 64), and references are never
narrower than 32 pixels.
`resize_before_vae=false` is not supported here; MiniMax-H3 keeps references on
the 32-pixel grid and reports a warning.

The C API exposes the same inputs through `ref_images`, `ref_videos`, and
`ref_audios` in `sd_vid_gen_params_t`. Each `sd_ref_video_t` supplies its own
frame rate and optional soundtrack; non-24-fps inputs are resampled internally.

## Shape and runtime notes

- Width and height are aligned upward to a multiple of 32.
- Frame count is aligned upward to the `17k + 5` grid, with a minimum of 5.
- MiniMax-H3 runs at 24 fps; another requested value is overridden.
- The default video flow shift is 12. The audio stream is mapped internally to
  its shift of 3, so the regular samplers can operate on the packed AV latent.

## Notes for VRAM-limited and older GPUs (measured on Tesla V100 32 GB)

Everything below was measured on a single Tesla V100-PCIE-32 GB (sm_70: no
bf16/fp8/int8 tensor cores, so quantized-attention kernels are out), Windows,
driver 581.80, CUDA 12.4 toolchain, Q4_K_M DiT with a Q4_K_M Qwen3-VL text
encoder, at commit `d8fb10c` (2026-09). Treat it as a calibration point for
other 16-32 GB cards.

### Flags that matter

With the default attention path the packed AV transformer materializes huge
attention matrices and a 32 GB card OOMs at 1024x576 above 39 frames.
Diffusion flash attention plus CPU offload and mmap changes that completely:

| Flag | Effect (measured) |
|---|---|
| Diffusion flash attention | 1024x576 ceiling on 32 GB: 39 -> 124+ frames |
| `--offload-to-cpu` | All ~35 GB of weights move to host RAM; idle GPU footprint drops 17.5 GB -> ~10 MiB; sampling at 1024x576x56 only 1.7% slower |
| `--mmap` | Cold-start end-to-end 1236 s -> 286 s (4.3x) on a 1024x576x39 probe; output bit-identical (PSNR infinite, SSIM 1.0, identical audio SHA256) |

At `d8fb10c` this combination was `--diffusion-fa --offload-to-cpu --mmap`.
On current master the attention switch is `--flash-attn off,diffusion=on`
(the old flag still works but warns) and `--auto-fit` is on by default. Do not
add explicit `--params-backend` assignments on top of offload: an explicit
params backend disables the automatic placement that makes this combination
work.

### Measured envelope

Single V100, 8 sampling steps, 24 fps. Upper block: no LoRA, seed 3001.
Lower block: community Turbo LoRA at multiplier 1.0, seed 4101, CLI.

| Canvas | Frames | Duration | Sampling | Wall clock | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| 1024x576 | 39 | 1.63 s | 258 s | 294 s | - |
| 1024x576 | 56 | 2.33 s | 249 s | 306 s | - |
| 1024x576 | 124 | 5.17 s | 643 s | 753 s | - |
| 1280x720 | 39 | 1.63 s | 279 s | 331 s | - |
| 1280x720 | 124 | 5.17 s | 1221 s | 1390 s | - |
| 1536x864 | 39 | 1.63 s | 425 s | 496 s | - |
| 1344x768 + Turbo LoRA | 175 | 7.29 s | - | 47.8 min | 29,486 MiB |
| 1344x768 + Turbo LoRA | 192 | 7.96 s | - | 54.7 min | 30,925 MiB |
| 1344x768 + Turbo LoRA | 209 | 8.71 s | - | 60.0 min | 32,206 MiB |

Observations that should transfer to other small GPUs:

- Peak VRAM at 1344x768 grows by ~85 MiB per frame; 209 frames (~8.7 s) is the
  practical ceiling on 32 GB with everything else offloaded.
- Throughput is compute-bound: per-step time is independent of step count, and
  with offload the per-step weight transfer over PCIe accounts for under 4% of
  the step. Step-count reduction is therefore the only large speed lever - the
  8-step community Turbo LoRA sampled 1024x576x39 in 261 s versus 649 s at 20
  steps without it (2.5x).
- A second V100 does not speed up a single job (layer split runs sequentially
  and row split is unavailable for the CUDA backend), but running two
  generations in parallel, one per GPU, measured 1.86x throughput.

### Audio delivery gotcha

The `.webm` written for audio-video outputs muxes the audio track as raw PCM,
which WebM-profile players drop (#2233). Remux before delivery:

```bash
ffmpeg -i in.webm -c:v copy -c:a libopus out.webm   # real WebM
ffmpeg -i in.webm -c:v libx264 -c:a aac out.mp4     # MP4 delivery
```

A full deployment recipe for V100-class hardware (startup scripts, VRAM and
duration tables, measured pitfalls, LoRA conversion tools) is maintained at
[noahark/minimax-h3-v100-deploy](https://github.com/noahark/minimax-h3-v100-deploy).
