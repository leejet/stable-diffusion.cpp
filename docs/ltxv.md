# LTX-Video 2.3 support — end-to-end validated

Branch: `feat/ltx-video` in
<https://github.com/mudler/stable-diffusion.cpp>. Ports Lightricks' LTX-2.3
22B audio-video foundation model (`Lightricks/LTX-2.3`) to
stable-diffusion.cpp, video-only path.

## Status — end-to-end pipeline works

Validated on an NVIDIA GB10 (Grace Blackwell, CUDA 13, 119 GB unified memory)
with `ltx-2.3-22b-distilled.safetensors` (46 GB BF16):

| Stage | Result |
|---|---|
| Version detection (`model.cpp`) | `VERSION_LTXV2` detected on `audio_scale_shift_table` / `audio_patchify_proj` / `audio_adaln_single` / `av_ca_video_scale_shift_adaln_single` / `video_embeddings_connector` |
| Weight registration | 4444 transformer + 170 VAE tensors registered — **zero missing, zero shape mismatches** vs. the 22B checkpoint (verified offline) |
| Checkpoint load | 46 GB BF16 loads in ~9 s, all 5947 tensors parse cleanly (audio_vae / vocoder / text_embedding_projection ignored) |
| Transformer forward | 48 layers × 32 heads × 128 head-dim (inner_dim 4096), 2 sampling steps complete in 2.26 s (1.13 s/step) on GB10 — 128 MB compute buffer |
| VAE decode | 9-block encoder/decoder with per-channel RMS norm; 2 latent frames → 9 output frames in 0.99 s — 1.77 GB compute buffer |
| End-to-end | 704×480×9 WebP written to disk; **3.64 s wall time for 2-step run, 5.45 s for 4-step run** |
| Quantization | BF16 46 GB → q8_0 28.3 GB (≈50 % reduction) via `sd-cli -M convert --type q8_0` in 9.6 s |
| Quantized inference | q8_0 GGUF loads + runs vid_gen end-to-end successfully |

## What's in the code

**Transformer (`src/ltxv.hpp`)**
- `LTX2VideoTransformer3DModel` — 48 layers; inner 4096 (32×128), cross-attn dim 4096, caption 4096
- `LTXAttention` — qk_norm_across_heads, always-on gated attention (`to_gate_logits` + 2·σ), interleaved and split RoPE variants
- `LTX2VideoTransformerBlock` — per-block `scale_shift_table` (9, dim), `prompt_scale_shift_table` (2, dim), `scale_shift_table_a2v_ca_video/audio` (5, dim/audio_dim), `audio_scale_shift_table` (9, audio_dim), `audio_prompt_scale_shift_table` (2, audio_dim). Forward path runs **only** video self-attn + prompt cross-attn + FF; audio self-attn, a2v/v2a cross-attn and audio FFN are loaded but skipped (isolate_modalities=True).
- `AdaLayerNormSingle` with configurable `num_mod_params`
- `EmbeddingsConnector` — 128 learnable registers + 8 transformer_1d_blocks (gated self-attn + FF) for both video and audio
- Split 3-D RoPE (video-axis F/H/W, dim/6 freqs per axis, vae_scale_factors (8, 32, 32), `causal_offset=1`, fps scaling, pair-swap rotation)
- Stub `LTXV2Conditioner` returning zero embeddings of shape `[1, 128, 4096]`

**VAE (`src/ltxv.hpp`)**
- 9-block encoder: res×4 @128, spatial↓(1,2,2) 128→256, res×6 @256, temporal↓(2,1,1) 256→512, res×4 @512, st↓(2,2,2) 512→1024, res×2 @1024, st↓(2,2,2) 1024→1024, res×2 @1024
- Decoder is the exact mirror
- `VAEResBlock` is the LTX-2.3 simplified shape (two `CausalConv3d` with silu gates, no norms, no timestep modulation)
- `CausalConv3d` uses `conv.weight` / `conv.bias` names, hardcoded F16 dtype so it stays within the CUDA `ggml_cuda_op_im2col_3d` accepted types
- `VAEUpsampler` pixel-shuffle drops the first `st_t − 1` frames after each temporal upsample so `f_out = (f_in − 1) × st_t + 1` composes across all upsamples

**Pipeline wiring (`src/stable-diffusion.cpp` etc.)**
- `VERSION_LTXV2` / `sd_version_is_ltxv2` / `sd_version_is_dit` entry
- VAE factory arm builds `LTXV::LTXVVAERunner`
- FLOW_PRED with `default_flow_shift = 3.0`
- Latent channels 128, VAE scale factor 32, temporal compression 8
- Frame count padded to 8k+1 (LTX-2.3 I/O spec)
- Ignore prefixes: `audio_vae.`, `vocoder.`, `text_embedding_projection.`

## Known quality gaps (load is correct, quality is not)

Current output is a valid 704×480×9 WebP but nearly uniform (≈ 660 bytes).
Expected: the pipeline runs, but quality is blocked by these items:

1. **Text encoder is a stub.** LTX-2.3 uses a custom multilingual encoder
   that is not included in the 22B safetensors file (only the aggregate
   `text_embedding_projection` is). Porting the real encoder is the
   biggest single remaining item for usable output.

2. **Pixel-shuffle ordering (VAE down/up-samplers).** I use a direct
   `ggml_reshape_4d` where diffusers does
   `.permute(0,1,5,2,6,3,7,4).flatten(…)`. The total element count and
   the shape both match, but the channel-pixel grouping is wrong, so
   decoded pixels are mixed across spatial positions.

3. **Latent stats unused.** `vae.per_channel_statistics.mean-of-means` /
   `std-of-means` are loaded but `diffusion_to_vae_latents` /
   `vae_to_diffusion_latents` are still identity functions. LTX-2.3's
   VAE latents are normalised — plugging these in is a straightforward
   element-wise op once the stats are on CPU.

4. **Flow-match shift defaults.** Set to 3.0 as a placeholder. LTX-2.3
   distilled uses 8 steps, CFG=1 — tuning needed after encoder is real.

5. **Audio branch loaded but unused.** About half of the 40 GB
   transformer parameter buffer is audio-related weights. Add the forward
   paths when audio generation becomes a priority.

## How to run the e2e test

```bash
# On the GPU host:
./sd-cli -M vid_gen \
  -m /path/to/ltx-2.3-22b-distilled.safetensors \
  -p "a cat walking across a grassy field" \
  -W 704 -H 480 --video-frames 9 \
  --steps 4 --cfg-scale 1 \
  -o /tmp/ltx23.webp \
  --seed 42 -v

# Quantize to q8_0 GGUF (28 GB, runs end-to-end):
./sd-cli -M convert \
  -m /path/to/ltx-2.3-22b-distilled.safetensors \
  -o /path/to/ltx-2.3-22b-distilled-q8_0.gguf \
  --type q8_0 -v

# Inference from the GGUF:
./sd-cli -M vid_gen \
  -m /path/to/ltx-2.3-22b-distilled-q8_0.gguf \
  -p "a cat walking across a grassy field" \
  -W 704 -H 480 --video-frames 9 \
  --steps 4 --cfg-scale 1 \
  -o /tmp/ltx23_q8.webp --seed 42
```

## References

- LTX-2.3 model card: https://huggingface.co/Lightricks/LTX-2.3
- Diffusers LTX-2.0 reference (not an exact match for 2.3):
  https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_ltx2.py
- Upstream ltx-pipelines (Lightricks):
  https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-pipelines
