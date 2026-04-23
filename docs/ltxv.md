# LTX-Video 2 support (work in progress)

This document tracks the `feat/ltx-video` branch which ports
[Lightricks LTX-Video 2](https://huggingface.co/Lightricks/LTX-Video) to
stable-diffusion.cpp. The port is a 1:1 translation of the diffusers
reference implementation (video-only path):

- `src/diffusers/models/transformers/transformer_ltx2.py` (LTX-2 joint a/v transformer)
- `src/diffusers/models/autoencoders/autoencoder_kl_ltx2.py` (LTX-2 video VAE)
- `src/diffusers/pipelines/ltx2/pipeline_ltx2.py` (scheduler + glue)

**Scope:** VIDEO generation only. The transformer loads all audio weights so
checkpoints open cleanly, but the forward path skips the audio self-attention,
audio cross-attention, audio-to-video and video-to-audio cross attention,
audio FFN, and audio output projection (equivalent to diffusers'
`isolate_modalities=True` + discarding the audio output). The audio VAE and
vocoder are not ported.

## Status

| Area | State |
|---|---|
| Weight detection (model.cpp) | done — keys on `audio_scale_shift_table` / `av_cross_attn_video_scale_shift` / `audio_proj_in` / `audio_time_embed` |
| Transformer — video path | implemented; CPU + CUDA builds clean |
| Transformer — audio branches (weight slots) | registered so LTX-2 checkpoints open cleanly |
| Transformer — audio forward | intentionally skipped (video-only mode) |
| PerChannelRMSNorm | implemented |
| LTX2AdaLayerNormSingle with configurable `num_mod_params` (6 or 9) | implemented |
| Gated attention (`to_gate_logits` + 2·σ) | implemented |
| 3-D RoPE "interleaved" | implemented (patch-boundary midpoint, `vae_scale_factors=(8,32,32)`, `causal_offset=1`, fps scaling) |
| 3-D RoPE "split" | **not yet** — falls back to interleaved layout |
| `cross_attn_mod` (9-param modulation) | implemented (forward skips text Q modulation gate when off) |
| `prompt_modulation` (LTX-2.3) | slots registered; forward path does not consume `temb_prompt` yet |
| Video VAE encoder | implemented |
| Video VAE decoder (with timestep conditioning) | implemented |
| VAE runtime `causal` flag | implemented |
| VAE `conv_shortcut` as plain Conv3d (no temporal padding) | implemented |
| VAE `PerChannelRMSNorm` | implemented |
| Downsampler variants (spatial / temporal / spatiotemporal / conv) | implemented |
| Pixel-shuffle 3D up/downsample exact ordering | **simplified reshape — needs verification** |
| T5-XXL conditioner | hooked via `T5CLIPEmbedder(use_mask=true, is_umt5=false)` |
| Flow-match scheduler + `default_flow_shift` | hooked (shift=3; LTX diffusers uses dynamic shift) |
| Latent shape: 128 channels, spatial/32, temporal/8 | wired |
| Frame rounding 8k+1 | wired |
| Audio generation | **not supported** |
| End-to-end video generation | **pending hardware validation** |

## Known simplifications and TODOs

1. **Pixel-shuffle 3D in the VAE.** The decoder's upsampler produces a tensor
   with `C_in * 8` channels and diffusers re-interleaves those channels with
   the (T, H, W) axes in a specific permute order. The current code uses a
   direct `ggml_reshape_4d` which is equivalent only when the channel groups
   are laid out the way ggml stores them. This is the most likely place
   artifacts will appear first. Same issue in the encoder's pixel-unshuffle
   patchify path (`patch_size=4, patch_size_t=1`).

2. **Split RoPE.** LTX-2 introduced a `split` rope variant in addition to the
   legacy `interleaved` mode. Only `interleaved` is implemented here; if the
   LTX-2 checkpoint you're using declares `rope_type = "split"` in its
   config, output will be incorrect. Split-rope requires reshaping Q/K to
   `[B, H, T, D/2]` before rotation.

3. **`cross_attn_mod` (LTX-2.X) prompt modulation gate.** The transformer
   block registers the 9-param `scale_shift_table` and the forward path
   applies shift/scale to the normed Q for the prompt cross-attention, but
   the LTX-2.3 `prompt_modulation` branch (where `temb_prompt` adds an extra
   scale/shift to the KV) is not yet applied.

4. **Flow-match shift.** Diffusers computes a per-shape dynamic shift; we set
   a fixed `default_flow_shift = 3.0`. If your hardware tests show overly
   blurry or over-sharpened output, this is the first knob to turn.

5. **Latents mean/std.** LTX-2's `latents_mean` / `latents_std` buffers are
   not yet consumed by `diffusion_to_vae_latents` / `vae_to_diffusion_latents`
   (the current implementations are identity). Plug them in once the smoke
   test proves the graph is otherwise correct.

6. **Audio path.** Audio self-attention, audio cross-attention to text,
   audio↔video cross-attention, and audio FFN all have their weight slots
   registered but the forward path skips them. Add them back when audio
   generation is prioritised.

7. **VAE scale factor.** Defaulted to **32** (patch_size=4 × 2³) in
   `vae.hpp:get_scale_factor`. If the LTX-2 "video" checkpoint you're using
   has all four down-blocks spatio-temporal (→ 4 × 2⁴ = 64), bump this.

## Testing

Text-to-video smoke test (DGX / CUDA):

```bash
./sd-cli \
    --model  /path/to/ltx2_video.safetensors \
    --vae    /path/to/ltx2_vae.safetensors \
    --t5xxl  /path/to/t5xxl.safetensors \
    -W 704 -H 480 --video-frames 25 \
    -p "a cat wearing sunglasses driving a convertible" \
    --cfg-scale 3.0 --steps 30 \
    -o /tmp/ltx2_test.webp \
    --verbose
```

Expected behaviour when things go wrong:

| Symptom | Most likely cause |
|---|---|
| `error: unexpected model type` | detection in `model.cpp` missed one of the LTX-2 keys (audio_scale_shift_table et al.) |
| `wrong shape` on a transformer weight | stride/mod-param count mismatch — verify `cross_attn_mod` flag |
| Crash inside `ggml_reshape` in the VAE | pixel-shuffle simplification (§1) |
| Output that looks like pure noise | flow-match shift (§4), latents mean/std (§5), or split rope missing (§2) |
| Output that looks blurry but shape is right | gate_logits factor of 2 / qk-norm weight loading; also check `cross_attn_mod` code path |
| `wrong shape` on a VAE weight | LTX-2 "Video" checkpoint may use `spatio_temporal_scaling=(True,True,True,False)` — override in `LTX2VideoEncoder3d` ctor |

## References

- LTX-Video model card: https://huggingface.co/Lightricks/LTX-Video
- Diffusers reference (LTX-2): https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/ltx2
- Upstream sd.cpp LTX-1 WIP (obsolete): https://github.com/leejet/stable-diffusion.cpp/pull/491
