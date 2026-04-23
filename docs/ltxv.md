# LTX-Video support (work in progress)

This document tracks the `feat/ltx-video` branch which ports
[Lightricks LTX-Video](https://huggingface.co/Lightricks/LTX-Video) to
stable-diffusion.cpp. The port is a 1:1 translation of the diffusers
reference implementation:

- `src/diffusers/models/transformers/transformer_ltx.py` (LTX 13B transformer)
- `src/diffusers/models/autoencoders/autoencoder_kl_ltx.py` (CausalVideoAutoencoder)
- `src/diffusers/pipelines/ltx/pipeline_ltx.py` (scheduler + text-encoder glue)

## Status

| Area | State |
|---|---|
| Weight detection (model.cpp) | done — keys on `scale_shift_table` / `adaln_single` / `caption_projection` |
| Transformer 28-layer DiT | implemented, CPU build clean |
| 3D RoPE (F, H, W, dim//6 per axis) | implemented |
| qk-norm-across-heads (RMSNorm on full inner_dim) | implemented |
| AdaLN-single 6-way modulation | implemented |
| Final `scale_shift_table` + `embedded_timestep` | implemented |
| CausalConv3d causal/non-causal padding | implemented |
| Video autoencoder encoder | implemented |
| Video autoencoder decoder | implemented (with timestep conditioning) |
| Pixel-shuffle 3D up/downsample | **simplified reshape — needs verification** |
| T5-XXL conditioner (not UMT5) | hooked via `T5CLIPEmbedder(use_mask=true, is_umt5=false)` |
| Flow-match scheduler + `default_flow_shift` | hooked (shift=3; LTX diffusers uses dynamic shift) |
| Latent-shape / temporal compression = 8 | wired |
| 128 latent channels / spatial compression = 32 | wired |
| End-to-end video generation | **pending hardware validation** |

## Known simplifications and TODOs

1. **Pixel-shuffle 3D in the VAE.** The decoder's upsampler produces a tensor
   with `C_in * 8` channels and diffusers re-interleaves those channels with
   the (T, H, W) axes in a specific order
   (`.permute(0,1,5,2,6,3,7,4).flatten(6,7).flatten(4,5).flatten(2,3)`).
   The current code uses a direct `ggml_reshape_4d` which is equivalent
   only when the channel groups are laid out the way ggml stores them.
   This is the most likely place output artifacts will appear first.

2. **`upsample_residual` / `upsample_factor`.** LTX 0.9.5 uses the residual
   path in some up-blocks. The current decoder ignores the residual; add it
   if checkpoint weights refer to an `.upsample_residual.*` submodule.

3. **Flow-match shift.** Diffusers computes a per-shape dynamic shift; we set
   a fixed `default_flow_shift = 3.0`. If your hardware tests show overly
   blurry or over-sharpened output, this is the first knob to turn.

4. **Latents mean/std.** LTX's `latents_mean` and `latents_std` buffers are
   not yet consumed by `diffusion_to_vae_latents` / `vae_to_diffusion_latents`
   (the current implementations are identity). Official LTX checkpoints ship
   these tensors; plug them in once the smoke test proves the graph is
   otherwise correct.

5. **Posterior splitting.** The encoder's `conv_out` produces
   `latent_channels + 1` channels (diffusers then replicates the last channel
   to reach `2 * latent_channels - 1`). The encode path is wired for future
   training/i2v use — text-to-video only needs the decoder.

6. **LTX-2 variants** (`transformer_ltx2.py`, `autoencoder_kl_ltx2.py`) are
   not yet supported. The current port targets the 13B "LTX-Video"
   architecture. LTX-2 adds audio and a larger head count; those need their
   own `SDVersion` entry and parameterisation pass.

## Testing

Text-to-video smoke test (DGX / CUDA):

```bash
# 1. Convert a diffusers LTX checkpoint to a sd.cpp-friendly safetensors.
#    You need: transformer + VAE + T5-XXL text encoder.
#    The converter respects the stable-diffusion.cpp tensor namespace:
#      model.diffusion_model.* / first_stage_model.* / text_encoders.t5xxl.transformer.*

./sd-cli \
    --model  /path/to/ltxv_13b.safetensors \
    --vae    /path/to/ltxv_vae.safetensors \
    --t5xxl  /path/to/t5xxl.safetensors \
    -W 704 -H 480 --video-frames 25 \
    -p "a cat wearing sunglasses driving a convertible" \
    --cfg-scale 3.0 --steps 30 \
    -o /tmp/ltxv_test.webp \
    --verbose
```

Expected behaviour when things go wrong:

| Symptom | Most likely cause |
|---|---|
| `error: unexpected model type` | detection in `model.cpp` missed one of the three LTX keys |
| Crash inside `ggml_reshape` in the VAE | pixel-shuffle simplification (§1) |
| Output that looks like noise end-to-end | flow-match shift (§3) or latents mean/std (§4) |
| Output that looks like a blurry photograph but shape is right | qk-norm-across-heads numerics; check `norm_q.weight` loads into full inner_dim |
| `wrong shape` error on a specific tensor name | diffusers weight name doesn't match; need an entry in `name_conversion.cpp` |

## References

- Upstream WIP (now stale): https://github.com/leejet/stable-diffusion.cpp/pull/491
- LTX model card: https://huggingface.co/Lightricks/LTX-Video
- Diffusers reference: https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/ltx
