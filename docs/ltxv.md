# LTX-Video 2.3 support (work in progress)

This document tracks the `feat/ltx-video` branch which is pivoting the
stable-diffusion.cpp port towards
[Lightricks LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) (22B
audio-video foundation model), video-only generation.

## State of the port

Architecture was initially modelled on diffusers' `transformer_ltx2.py` +
`autoencoder_kl_ltx2.py`, then rebased onto the actual LTX-2.3 22B
checkpoint (`ltx-2.3-22b-dev.safetensors`, read via safetensors header
inspection). The rebase surfaced a lot of divergence — tracking it here.

### What matches the checkpoint

- 48 transformer blocks, inner_dim=4096 (32 × 128), audio_inner_dim=2048
  (32 × 64)
- Gated attention on every attention (video + audio + cross-modal) —
  `to_gate_logits` weight present with output dim 32 in every attn layer
- `cross_attn_mod = True` → `scale_shift_table` has 9 mod params
  (=36864/4096); `audio_cross_attn_mod = True` → 9 audio mod params
  (=18432/2048)
- `prompt_modulation = True` (LTX-2.3) — `prompt_adaln_single` /
  `audio_prompt_adaln_single` present with 2 mod params
- Block-level names match (`attn1.to_q/k/v`, `attn1.q_norm/k_norm`,
  `attn1.to_gate_logits`, `to_out.0`, `ff.net.0.proj`, `ff.net.2`)
- Top-level names match (post-rename): `adaln_single`, `audio_adaln_single`,
  `patchify_proj`, `audio_patchify_proj`, `proj_out`, `audio_proj_out`,
  `scale_shift_table`, `audio_scale_shift_table`, `prompt_adaln_single`,
  `audio_prompt_adaln_single`, `av_ca_video_scale_shift_adaln_single`,
  `av_ca_audio_scale_shift_adaln_single`, `av_ca_a2v_gate_adaln_single`,
  `av_ca_v2a_gate_adaln_single`

### What is still divergent (TODO)

**Transformer:**
1. **`video_embeddings_connector` / `audio_embeddings_connector`** — LTX-2.3
   replaces the simple `PixArtAlphaTextProjection` with a full prompt
   re-embedder: 128 learnable registers + 8 self-attention transformer_1d_blocks.
   The code currently registers `caption_projection` (LTX-2.0 style, 2
   linear layers) which will FAIL to load on a 2.3 checkpoint. Needs a
   new `EmbeddingsConnector` block.
2. **Split RoPE** — not yet implemented; only interleaved is wired.
3. **Prompt modulation forward path** (`prompt_adaln_single`) — weights
   load but forward doesn't apply the prompt scale/shift to KV.

**VAE (much bigger mismatch):**
4. **9 encoder down_blocks + 9 decoder up_blocks** — current code has 4.
5. **`block_out_channels` starts at 128** (code has 256) — scale
   progression is different for LTX-2.3.
6. **First VAE channel count** — `vae.encoder.conv_in.conv.weight` is
   `[128, 48, 3, 3, 3]` → 48 input channels (patch_size=4, in_channels=3
   → 3*16=48). Matches our math but confirm the output of the first conv
   is 128 not 256.
7. **`vae.decoder.conv_in`** is `[1024, 128, 3, 3, 3]` → deepest latent
   channel width is 1024 (not 2048 as LTX-2.0 defaults suggest).

**Weight loading:**
8. Default constructor still uses LTX-2.0 configs (num_layers=48 ok but
   VAE config wrong). The transformer dims are correct for LTX-2.3 too.
9. Tensor name for `to_gate_logits` output bias may be `attn1.to_gate_logits.bias`
   — currently registered as `blocks["to_gate_logits"]` child of LTX2Attention,
   so path is `...attn1.to_gate_logits.bias` — **this should be OK**.

**Pipeline:**
10. Flow-match scheduler defaults (shift, num_steps) for LTX-2.3 not
    tuned; the distilled checkpoint ships with `steps=8, cfg=1`.
11. Latent stats (mean/std) — LTX-2.3 may have non-unit latent stats.
    Not yet parsed from checkpoint.
12. Frame-count constraint: **dimensions must be divisible by 32,
    frame count must be 8k+1** — the wiring in
    `GenerationRequest` uses 8k+1 for LTX so that's correct.

## Testing

Do not expect weight loading to succeed on LTX-2.3 yet — the
`video_embeddings_connector` + VAE architecture changes need to land
first. Current code will likely error with "missing tensor
video_embeddings_connector.learnable_registers" or similar.

The architecture investigation artifact is committed so future sessions
can resume without re-reading the 46GB checkpoint header.

## References

- LTX-2.3 model card: https://huggingface.co/Lightricks/LTX-2.3
- LTX-2.3 `ltx-2.3-22b-dev.safetensors` — 5947 tensors, 46GB, merged
  single-file release with `audio_vae.*` + `vae.*` + `model.diffusion_model.*`
- Upstream `ltx-pipelines` package (reference impl):
  https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-pipelines
- Diffusers LTX-2.0 reference (was the starting point; config keys
  don't match 2.3 one-to-one): https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_ltx2.py

## Input/Output Requirements (from LTX-2.3 model card)

- Width & height divisible by 32
- Frame count divisible by 8, plus 1 (i.e. 8k+1 for integer k≥0)
- Non-compliant inputs must be padded with -1 in pixel space then cropped
  to the desired output dimensions
