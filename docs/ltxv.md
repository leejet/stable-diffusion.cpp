# LTX-Video 2.3 support — conditional text-to-video works end-to-end

Branch: `feat/ltx-video` in
<https://github.com/mudler/stable-diffusion.cpp>. Ports Lightricks' LTX-2.3
22B audio-video foundation model (`Lightricks/LTX-2.3`) to
stable-diffusion.cpp, video-only path. **Text conditioning wired via a
native Gemma-3-12B port** so prompts actually steer the output.

## Status — prompts generate the thing you asked for

Validated on an NVIDIA GB10 (Grace Blackwell, CUDA 13, 119 GB unified memory)
with `ltx-2.3-22b-distilled.safetensors` (46 GB BF16) + Gemma-3-12B-it
(24 GB BF16) as text encoder:

| Stage | Result |
|---|---|
| LTX version detection (`model.cpp`) | `VERSION_LTXV2` detected on `audio_scale_shift_table` / `audio_patchify_proj` / `audio_adaln_single` / `av_ca_video_scale_shift_adaln_single` / `video_embeddings_connector` |
| Weight registration | 4444 transformer + 170 VAE + 4 text_embedding_projection tensors registered — **zero missing, zero shape mismatches** vs. the 22B checkpoint |
| Checkpoint load | 46 GB BF16 loads in ~9 s; audio_vae / vocoder ignored (video-only pipeline) |
| Gemma-3-12B text encoder | Loads + runs in 5 s on GB10; 49-layer hidden states match HuggingFace to bf16 precision; `text_embedding_projection.video_aggregate_embed` output: std=6.828 (HF: 6.830) |
| Transformer forward | 48 layers × 32 heads × 128 head-dim (inner_dim 4096), 8 distilled steps in 123 s on GB10 |
| VAE decode | 9-block decoder with per-channel RMS norm + proper 3-D depth-to-space; 16-frame latent → 121-frame video in 16 s |
| End-to-end | 704×480×9 WebP in ~14 s; 768×512×121 WebP in ~140 s on GB10; **prompts generate the described subject** (cat → cat, dragon → dragon, etc.) |
| Quantization | BF16 46 GB → q8_0 28.3 GB via `sd-cli -M convert --type q8_0` in 9.6 s; q8_0 GGUF runs end-to-end |

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

## Numerical correctness — resolved

Nine bugs were diagnosed and fixed by working backwards from the VAE output
(and later the text-conditioning path) using graph-level probes. Each one is
noted here because the same mistake is easy to make again porting future
video VAE/DiT stacks:

1. **EmbeddingsConnector pre-norm.** Reference
   `_BasicTransformerBlock1D.forward` does `rms_norm(hidden_states)` before
   both attn1 and ff (and a final `rms_norm` after the stack). We had
   bare `x = x + attn(x); x = x + ff(x)` — residuals compounded across 8
   blocks and drove the connector output to std≈1e12, exploding cross-attn
   in every transformer block.

2. **Final `norm_out` before the scale/shift + `proj_out`.** Reference
   `LTXModel._process_output` is
   `x = norm_out(x); x = x * (1 + scale) + shift; x = proj_out(x)`.
   Without the LayerNorm the post-block activation (std≈285 after 48
   layers) leaked into the predicted velocity and the sampler diverged.
   Transformer output std went from 57 → 1.0 after adding `ggml_norm`.

3. **VAE `conv_norm_out` + SiLU before `conv_out`.** The reference decoder
   ends with `sample = conv_norm_out(sample); sample = silu(sample);
   sample = conv_out(sample)`. We were skipping the PixelNorm+SiLU, so
   output pixels were O(1000) instead of O(1).

4. **Latent per-channel normalisation.** `vae.per_channel_statistics.*`
   is now materialised to CPU and applied in `diffusion_to_vae_latents`
   (`x * std + mean`) / `vae_to_diffusion_latents` (`(x - mean) / std`).

5. **VAE depth-to-space ordering.** `ggml_reshape_4d` alone doesn't
   implement einops `b (c p1 p2 p3) f h w -> b c (f p1) (h p2) (w p3)` —
   the sub-indices come out in the wrong order. Replaced with a proper
   `depth_to_space_3d` helper that decomposes the channel axis through
   permute+cont passes so p3 lands inner-of-W, p2 inner-of-H, p1
   inner-of-F. Eliminated the visible banding.

6. **Gemma-3 49-layer concat layout.** `ggml_concat(hidden_all[i],
   axis=0)` produces a flat axis with layer-slow / hidden-fast ordering,
   but HF's `reshape(B, T, D*L)` produces hidden-slow / layer-fast.
   `text_embedding_projection.video_aggregate_embed` was trained for the
   HF layout — a transposed input made the projection output essentially
   noise and all prompts generated the same scene. Fixed by stacking
   along axis 2 → permute(2, 0, 1, 3) → reshape to [D*L, T, 1].

7. **EmbeddingsConnector register layout.** Reference
   `_replace_padded_with_learnable_registers` produces a **fixed
   128-token** output with real text at positions [0..L-1] and
   `learnable_registers[L..127]` at [L..127]. We were concatenating
   registers+text to 128+L tokens in the wrong order. Rewrote the
   connector's register path.

8. **Double attention scaling in Gemma-3.** Gemma-3 uses
   `scale = 1/sqrt(query_pre_attn_scalar) = 1/sqrt(head_dim)` for the
   12B variant — and `ggml_ext_attention_ext` applies the same
   `1/sqrt(d_head)` internally. Applying both multiplied the softmax
   temperature by 1/16, collapsing attention to near-uniform and
   producing a persistent ~sqrt(D) "attention sink" outlier at the same
   hidden dim for every layer. Dropping the explicit Q scale made the
   Gemma forward match HF to bf16 precision.

9. **Two different patchify conventions in `ops.py` vs `sampling.py`.**
   `DepthToSpaceUpsample` (intermediate upsamplers) uses
   `b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)` — p3 (w-stride)
   innermost in the channel axis. `ops.py::unpatchify` (the decoder's
   final 4×4 un-patch) uses
   `b (c p r q) f h w -> b c (f p) (h q) (w r)` — q (h_patch) innermost.
   We were reusing the upsampler helper for the final unpatchify, which
   silently transposed every 4×4 output block and left a visible fine-
   scale hatching artefact that survived every diffusion step. Added a
   dedicated `depth_to_space_3d_patch` that swaps the inner (p_w, p_h)
   pair of the channel axis before delegating, matching the reference
   layout exactly.

Cross-checked against the 22B checkpoint's embedded config
(`safetensors __metadata__["config"]["vae"]`): `norm_layer=pixel_norm`,
`spatial_padding_mode=zeros`, `timestep_conditioning=false`,
`causal_decoder=false`, patch_size=4, and none of the `compress_all`
decoder blocks sets `residual=True` — so the residual skip from
`DepthToSpaceUpsample` is correctly absent here.

End-to-end result: prompts now actually generate the described content.
Seed 42 with *"a cat walking across a grassy field"* produces exactly
that. Per-layer Gemma hidden states match HF to bf16 noise; the
projected cross-attention features match HF (min/max/std 0.0%/0.2%/0.03%
different).

## Remaining items (future sessions)

1. **Audio branch.** Roughly half of the LTX transformer buffer is
   audio-related (`audio_attn1/2`, `audio_to_video_attn`,
   `video_to_audio_attn`, `audio_embeddings_connector`,
   `audio_scale_shift_table`, etc.). Adding joint audio+video generation
   also needs the `audio_vae` (102 tensors), the HiFi-GAN-style
   `vocoder` (1227 tensors), and the BWE upsampler. Non-trivial.

2. **Schedule for non-distilled variants.** The 22B non-distilled model
   uses LTX2Scheduler (token-count-dependent shift, stretched to a
   terminal value). Only the distilled 8-step table is wired up today.

3. **Quantised Gemma.** Gemma-3-12B is 24 GB in BF16. A q8_0 or q4_k
   conversion would drop it to ~12 GB / ~7 GB — useful for smaller
   hardware. The existing sd-cli `-M convert` path should handle it.

## How to run the e2e test

First, grab the two model artefacts:

```bash
# LTX-2.3 distilled 22B (46 GB BF16 safetensors):
hf download Lightricks/LTX-2.3 ltx-2.3-22b-distilled.safetensors \
    --local-dir ltxv-models

# Gemma-3-12B-it (tokenizer.model + 5x safetensors shards, ~24 GB BF16):
hf download google/gemma-3-12b-it --local-dir gemma-3-12b-it
```

Then run with the distilled 8-step schedule (auto-selected when
`--steps 8` is passed on an ltxv2 model):

```bash
./sd-cli -M vid_gen \
  -m ltxv-models/ltx-2.3-22b-distilled.safetensors \
  --text-encoder gemma-3-12b-it \
  -p "a cat walking across a grassy field" \
  -W 704 -H 480 --video-frames 9 \
  --steps 8 --cfg-scale 1 \
  -o /tmp/ltx23.webp --seed 42

# Official distilled shape (768x512, 121 frames, ~140 s on GB10):
./sd-cli -M vid_gen \
  -m ltxv-models/ltx-2.3-22b-distilled.safetensors \
  --text-encoder gemma-3-12b-it \
  -p "a cat walking across a grassy field" \
  -W 768 -H 512 --video-frames 121 \
  --steps 8 --cfg-scale 1 \
  -o /tmp/ltx23.webp --seed 42

# Without --text-encoder: LTX runs unconditionally (zero embeddings),
# pipeline still produces valid frames but ignores the prompt.

# Quantise the LTX DiT to q8_0 GGUF (46 GB -> 28 GB):
./sd-cli -M convert \
  -m ltxv-models/ltx-2.3-22b-distilled.safetensors \
  -o ltxv-models/ltx-2.3-22b-distilled-q8_0.gguf \
  --type q8_0
```

## References

- LTX-2.3 model card: https://huggingface.co/Lightricks/LTX-2.3
- Diffusers LTX-2.0 reference (not an exact match for 2.3):
  https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_ltx2.py
- Upstream ltx-pipelines (Lightricks):
  https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-pipelines
