# RelayVSR video super-resolution

RelayVSR uses a Wan model to sample sparse reference latents, followed by the
Flash conditional decoder to reconstruct every input frame. The model code
follows [RelayVSR](https://github.com/kopperx/RelayVSR); the loader maps
[ComfyUI repackaged weights](https://huggingface.co/Comfy-Org/RelayVSR) to the
original module names.

Use `-M vid_upscale` with an ordered directory of RGB images. Filenames are
sorted lexicographically. Supply the source frame rate with `--fps`. Output
keeps the input frame count. Use a video output supported by the CLI, such as
`.avi` or `.webm`, or a numbered PNG output such as `output/%04d.png`.

The scale is an explicit, required argument. It is not read from checkpoint
metadata. Use matching Wan and Flash checkpoints:

| Scale | `--diffusion-model` | `--vae` |
| --- | --- | --- |
| 2 | `relayvsr_x2_lightx2v_int8_convrot.safetensors` | `relayvsr_flash_vae_x2_fp32.safetensors` |
| 4 | `relayvsr_x4_int8_convrot.safetensors` | `relayvsr_flash_vae_x4_fp32.safetensors` |

```sh
sd-cli -M vid_upscale \
  --diffusion-model models/diffusion_models/relayvsr_x2_lightx2v_int8_convrot.safetensors \
  --vae models/vae/relayvsr_flash_vae_x2_fp32.safetensors \
  --input-frames input_frames --upscale-factor 2 --fps 24 \
  --vsr-window-size 16 -s 42 -o output.avi
```

`--vsr-window-size` includes both reference endpoints. The default 16 samples
references at frame indices 0, 15, 30, and so on, always including the last
frame. Shared endpoints update Flash history in the following window but are
emitted only once. Single-frame input is also supported.

Wan performs one flow-matching Euler step at timestep 1000 with guidance 1,
using the prompt embedding stored in the checkpoint. No text encoder or
ordinary Wan VAE is needed. Generic generation sampling settings do not
configure this fixed sampling path.

The Flash scale cannot be verified from its tensor shapes: the x2 and x4 files
have identical shapes. The caller must select the correct pair. Wan's input
projection is checked against the requested scale. Inputs are replicate-padded
internally and outputs are cropped to exactly the input size times the scale.

INT8 convrot uses the existing [quantized-weight execution path](int8_convrot.md).
Backend and parameter placement use `--backend` and `--params-backend`, with
Wan assigned to `diffusion` and Flash to `vae`. CUDA builds with CUTLASS use
fused window attention on compute capability 8.0 or newer. It loads spatial
neighborhoods directly, with FP16 Q/K/V and FP32 accumulation, and preserves
Wan's boundary-shifted windows and Flash's boundary-truncated windows.
Other backends use bounded spatial tiles with ordinary attention operators.
Set `GGML_CUDA_DISABLE_WINDOW_ATTENTION=1` to use that fallback on CUDA for
comparison. Floating-point results can differ between these implementations.

Library callers initialize `sd_video_upscaler_ctx_params_t` with
`sd_video_upscaler_ctx_params_init()`, set the model paths, explicit scale and
backend options, and create a `video_upscaler_ctx_t` with
`new_video_upscaler_ctx()`. Model architecture is detected from
weight names and shapes; the scale remains caller-supplied.

Initialize `sd_video_upscale_params_t` with `sd_video_upscale_params_init()`,
then set the RGB frames, frame count, FPS, seed and RelayVSR reference window.
Call `upscale_video()` and free the result with
`free_sd_video_generation_result()`. Defaults are 24 FPS, seed 42 and a
16-frame reference window. No sampler or scheduler selection is required.

Each call starts fresh
temporal state. Model caches are bounded by the temporal windows; the public
API currently holds the complete input and output frame arrays in host memory.
Audio is not copied by this path.
