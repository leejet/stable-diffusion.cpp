#!/usr/bin/env python3
"""Dump tiny LTX-2 VAE reference tensors for C++/GGML parity testing.

Strategy mirrors dump_reference.py / dump_gemma.py: instantiate a tiny VideoEncoder
and VideoDecoder with deterministic tamed weights, run one forward pass each on
fixed inputs, and save per-block intermediate outputs and the state_dict.

Tiny config exercises one of each encoder block type (compress_space_res,
compress_time_res, res_x) and a matching decoder (res_x, compress_time, compress_space)
plus the output AdaLN + PerChannelStatistics (un)normalize.

Usage:
    /home/ilintar/venv/bin/python dump_vae.py
"""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch

from safetensors.torch import save_file

from ltx_core.model.video_vae.video_vae import VideoEncoder, VideoDecoder
from ltx_core.model.video_vae.enums import NormLayerType, LogVarianceType, PaddingModeType

# -------- Config --------

SEED = 0
OUT_DIR = pathlib.Path("/tmp/vae_ref")
TENSOR_DIR = OUT_DIR / "tensors"

# Tiny VAE config. patch_size=2 (vs standard 4) to keep spatial dims small.
# encoder: compress_space_res(×2 ch) then compress_time_res(×2 ch) then res_x(1 layer).
# decoder: res_x(1 layer), compress_time, compress_space (reversed during construction).
IN_CHANNELS       = 3
LATENT_CHANNELS   = 8
DECODER_BASE_CH   = 8  # decoder conv_in goes 128 -> 8 * 8 = 64 (with *8 multiplier)
PATCH_SIZE        = 2
NORM_LAYER        = NormLayerType.PIXEL_NORM
LOG_VAR           = LogVarianceType.UNIFORM
PADDING_ENC       = PaddingModeType.ZEROS
PADDING_DEC       = PaddingModeType.REFLECT

# Video shape: 1 + 8*k frames required by encoder's validator. 1 + 8*1 = 9 → F=9.
# Spatial must divide by (patch_size * 2 * 2) = 8 for one compress_space_res + one compress_time_res.
# H = W = 16 is the minimum that divides 8 cleanly after patchify.
BATCH, F_IN, H_IN, W_IN = 1, 9, 16, 16

DECODE_TIMESTEP = 0.05  # Gemma/LTX-2 conventional decoder timestep


# -------- Utility --------

@dataclass
class Manifest:
    entries: List[Dict] = field(default_factory=list)

    def add(self, name: str, t: torch.Tensor):
        self.entries.append({"name": name, "shape": list(t.shape), "dtype": "f32"})

    def dump(self, path: pathlib.Path):
        path.write_text(json.dumps({"entries": self.entries}, indent=2))


def save_tensor(t: torch.Tensor, name: str, manifest: Manifest):
    safe = name.replace("/", "__")
    arr = t.detach().to(torch.float32).contiguous().cpu().numpy()
    arr.tofile(TENSOR_DIR / f"{safe}.bin")
    manifest.add(name, t)


def tame_(model: torch.nn.Module):
    """Deterministic, finite weights. Reuses the pattern from dump_reference.py.

    - 1D params (biases, norm weights, scale_shift_tables, per_channel_scale*):
      zero-initialized. Gemma-style convention where (1+w) is used as the effective
      scale works the same for VAE's ResnetBlock3D AdaLN (hidden * (1 + scale) + shift).
    - 2D/3D/4D/5D params (linears, convs): Kaiming-ish with std=1/sqrt(fan_in).
    - PerChannelStatistics buffers: std-of-means = 1.0, mean-of-means = 0.0 so
      normalize/un_normalize become identity + scale.
    """
    g = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() <= 1:
                # 0-D (scalars like timestep_scale_multiplier) or 1-D (biases, norm weights,
                # scale_shift_tables, per_channel_scale*) — zero init.
                p.zero_()
                # `timestep_scale_multiplier` needs to be exactly the trained value (1000.0)
                # to match the denoiser scale, but a zero multiplier just zeroes the timestep
                # embedding; the parity path still exercises the rest of the block math.
            else:
                fan_in = max(1, p.numel() // p.shape[0])
                std = 1.0 / math.sqrt(fan_in)
                p.normal_(mean=0.0, std=std, generator=g)
        # PerChannelStatistics is a buffer (not a parameter), init separately:
        for name, buf in model.named_buffers():
            if "std-of-means" in name:
                buf.fill_(1.0)
            elif "mean-of-means" in name:
                buf.fill_(0.0)


def build_encoder() -> VideoEncoder:
    return VideoEncoder(
        convolution_dimensions=3,
        in_channels=IN_CHANNELS,
        out_channels=LATENT_CHANNELS,
        encoder_blocks=[
            ("compress_space_res", {"multiplier": 2}),
            ("compress_time_res",  {"multiplier": 2}),
            ("res_x",              {"num_layers": 1}),
        ],
        patch_size=PATCH_SIZE,
        norm_layer=NORM_LAYER,
        latent_log_var=LOG_VAR,
        encoder_spatial_padding_mode=PADDING_ENC,
    )


def build_decoder() -> VideoDecoder:
    # Encoder reduces temporal by 2 and spatial by patch_size*2 = 4. Decoder matches inverse.
    return VideoDecoder(
        convolution_dimensions=3,
        in_channels=LATENT_CHANNELS,
        out_channels=IN_CHANNELS,
        decoder_blocks=[
            # order is reversed inside VideoDecoder ctor — this is the encoder-side order:
            ("compress_space",   {"multiplier": 1}),
            ("compress_time",    {"multiplier": 1}),
            ("res_x",            {"num_layers": 1}),
        ],
        patch_size=PATCH_SIZE,
        norm_layer=NORM_LAYER,
        causal=False,
        timestep_conditioning=True,
        decoder_spatial_padding_mode=PADDING_DEC,
        base_channels=DECODER_BASE_CH,
    )


# -------- Main --------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TENSOR_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    encoder = build_encoder().eval()
    decoder = build_decoder().eval()

    tame_(encoder)
    tame_(decoder)

    # Input video: (B=1, C=3, F=9, H=16, W=16).
    rng = np.random.default_rng(SEED)
    video_np = rng.standard_normal((BATCH, IN_CHANNELS, F_IN, H_IN, W_IN), dtype=np.float32)
    video = torch.from_numpy(video_np).clone()

    manifest = Manifest()
    save_tensor(video, "video_in", manifest)
    print(f"input: shape={tuple(video.shape)}")

    # --- Encoder forward ---
    with torch.no_grad():
        x = video
        # Replicate VideoEncoder.forward manually so we can cache intermediates.
        from ltx_core.model.video_vae.ops import patchify
        x = patchify(x, patch_size_hw=PATCH_SIZE, patch_size_t=1)
        save_tensor(x, "enc_post_patchify", manifest)

        x = encoder.conv_in(x)
        save_tensor(x, "enc_post_conv_in", manifest)

        for i, blk in enumerate(encoder.down_blocks):
            x = blk(x)
            save_tensor(x, f"enc_block_{i}", manifest)

        x = encoder.conv_norm_out(x)
        save_tensor(x, "enc_post_norm", manifest)
        x = encoder.conv_act(x)
        x = encoder.conv_out(x)
        save_tensor(x, "enc_post_conv_out", manifest)

        # Replicate UNIFORM latent_log_var path: means = x[:, :-1], logvar = x[:, -1:].
        if LOG_VAR == LogVarianceType.UNIFORM:
            means  = x[:, :-1, ...]
            logvar = x[:, -1:, ...]
            # (We save just the means and the final normalized latent; don't need logvar for parity.)
            save_tensor(means, "enc_means_preNorm", manifest)
            latent = encoder.per_channel_statistics.normalize(means)
            save_tensor(latent, "latent", manifest)
        else:
            raise RuntimeError("only UNIFORM supported in dumper")

    print(f"latent: shape={tuple(latent.shape)} mean={latent.mean().item():.4f} std={latent.std().item():.4f}")

    # --- Decoder forward (deterministic path; no noise, fixed timestep) ---
    timestep = torch.full((BATCH,), DECODE_TIMESTEP, dtype=torch.float32)
    with torch.no_grad():
        y = decoder.per_channel_statistics.un_normalize(latent)
        save_tensor(y, "dec_post_unnorm", manifest)

        # Match the real decoder.forward: self.causal=False is set by the configurator,
        # so every conv call uses causal=False. Earlier versions of this dumper relied
        # on the default causal=True which diverged from actual behavior and masked a
        # conv1/conv2 mismatch in the C++ port.
        y = decoder.conv_in(y, causal=False)
        save_tensor(y, "dec_post_conv_in", manifest)

        # TimestepEmbedder probe: feed the exact `timestep` used below, save the 256-dim
        # result so the C++ side can byte-diff its TimestepEmbedder output against Python's.
        # Uses the inner time_embedder (embedding_dim=256) from the res_x block.
        te_probe = decoder.up_blocks[0].time_embedder(
            timestep=timestep.flatten(), hidden_dtype=y.dtype)
        save_tensor(te_probe, "te_probe_up0", manifest)

        # Intermediate after each up_block (reversed decoder config).
        # Probe INSIDE the first res_x block: dump the pixel_norm(conv_in_output) to verify
        # the norm path is byte-exact. This is the Python `hidden_states = self.norm1(x)`
        # inside the first ResnetBlock3D of up_blocks[0].
        from ltx_core.model.common.normalization import PixelNorm
        probe_block = decoder.up_blocks[0].res_blocks[0]
        y_pre = y  # still conv_in output here; save a copy.
        y_norm1 = probe_block.norm1(y_pre)
        save_tensor(y_norm1, "dec_resblock0_post_norm1", manifest)
        # Also save post_adaln1 (just the modulation, no silu/conv yet).
        ts_embed_block = decoder.up_blocks[0].time_embedder(
            timestep=timestep.flatten(), hidden_dtype=y_pre.dtype
        ).view(BATCH, -1, 1, 1, 1)
        ada_probe = probe_block.scale_shift_table[None, ..., None, None, None] + ts_embed_block.reshape(
            BATCH, 4, -1, 1, 1, 1
        )
        sh1, sc1, sh2, sc2 = ada_probe.unbind(dim=1)
        y_adaln1 = y_norm1 * (1 + sc1) + sh1
        save_tensor(y_adaln1, "dec_resblock0_post_adaln1", manifest)
        y_silu1  = probe_block.non_linearity(y_adaln1)
        y_conv1  = probe_block.conv1(y_silu1, causal=False)
        save_tensor(y_conv1, "dec_resblock0_post_conv1", manifest)
        y_norm2  = probe_block.norm2(y_conv1)
        save_tensor(y_norm2, "dec_resblock0_post_norm2", manifest)

        # Build the timestep embedding that UNetMidBlock3D would use internally for
        # the res_x block. The scale multiplier is a learned scalar (we zero-inited it).
        # The parity comparison only verifies the *forward* math; if the multiplier is
        # 0 then the time embedding collapses. That's fine since we verify tracewise.
        # Inject a timestep only when calling the block (passed through forward()).

        # Replicate VideoDecoder.forward partial path.
        # Important: the decoder's up_blocks list is REVERSED of the config list.
        # Our config: [compress_space, compress_time, res_x]. After reversing:
        # [res_x, compress_time, compress_space]. So up_blocks[0] is the res_x.

        # Timestep scale is used inside last_time_embedder; but res_x UNetMidBlock3D
        # has its own time_embedder. Pass raw timestep; each block handles scaling.

        for i, blk in enumerate(decoder.up_blocks):
            # Only res_x (UNetMidBlock3D) accepts timestep; up/down sample blocks don't.
            from ltx_core.model.video_vae.resnet import UNetMidBlock3D
            if isinstance(blk, UNetMidBlock3D):
                y = blk(y, causal=False, timestep=timestep)
            else:
                y = blk(y, causal=False)
            save_tensor(y, f"dec_block_{i}", manifest)

        # Final AdaLN output + conv_norm_out: this is the `last_scale_shift_table` + time_embedder path.
        ada = decoder.last_scale_shift_table[None, ..., None, None, None] + decoder.last_time_embedder(
            timestep=(timestep * decoder.timestep_scale_multiplier).flatten(),
            hidden_dtype=y.dtype,
        ).view(BATCH, 2, -1, 1, 1, 1)
        shift, scale = ada.unbind(dim=1)
        y = decoder.conv_norm_out(y)
        save_tensor(y, "dec_post_pixel_norm", manifest)
        y = y * (1 + scale) + shift
        save_tensor(y, "dec_post_ada", manifest)
        y = decoder.conv_act(y)
        y = decoder.conv_out(y, causal=False)
        save_tensor(y, "dec_post_conv_out", manifest)

        from ltx_core.model.video_vae.ops import unpatchify
        y = unpatchify(y, patch_size_hw=PATCH_SIZE, patch_size_t=1)
        save_tensor(y, "video_out", manifest)

    print(f"decoded: shape={tuple(y.shape)} mean={y.mean().item():.4f} std={y.std().item():.4f}")

    # --- State dict: concatenate encoder + decoder + per_channel_statistics under "vae." prefix. ---
    prefixed = {}
    for k, v in encoder.state_dict().items():
        prefixed[f"vae.encoder.{k}"] = v.to(torch.float32).contiguous()
    for k, v in decoder.state_dict().items():
        prefixed[f"vae.decoder.{k}"] = v.to(torch.float32).contiguous()
    # PerChannelStatistics is registered inside both encoder & decoder AND also dumped under
    # a top-level `vae.per_channel_statistics.*` path (matching the real checkpoint convention,
    # per VAE_ENCODER_COMFY_KEYS_FILTER). We keep all three copies so encoder/decoder
    # blocks can load from either the nested or the canonical path.
    pcs = encoder.per_channel_statistics
    for bufname, buf in pcs.named_buffers():
        # .clone() to sever storage sharing with the nested copies — safetensors
        # refuses to dump multiple keys pointing at the same underlying buffer.
        prefixed[f"vae.per_channel_statistics.{bufname}"] = buf.detach().to(torch.float32).clone().contiguous()

    save_file(prefixed, str(OUT_DIR / "state_dict.safetensors"))
    (OUT_DIR / "tensor_names.txt").write_text("\n".join(sorted(prefixed.keys())) + "\n")
    manifest.dump(OUT_DIR / "manifest.json")

    (OUT_DIR / "config.json").write_text(json.dumps({
        "in_channels":         IN_CHANNELS,
        "latent_channels":     LATENT_CHANNELS,
        "decoder_base_ch":     DECODER_BASE_CH,
        "patch_size":          PATCH_SIZE,
        "norm_layer":          NORM_LAYER.value,
        "log_var":             LOG_VAR.value,
        "batch":               BATCH,
        "frames":              F_IN,
        "height":              H_IN,
        "width":               W_IN,
        "decode_timestep":     DECODE_TIMESTEP,
        "encoder_blocks":      [
            ["compress_space_res", {"multiplier": 2}],
            ["compress_time_res",  {"multiplier": 2}],
            ["res_x",              {"num_layers": 1}],
        ],
        "decoder_blocks":      [
            ["compress_space",     {"multiplier": 1}],
            ["compress_time",      {"multiplier": 1}],
            ["res_x",              {"num_layers": 1}],
        ],
    }, indent=2))

    print(f"\nDone. Wrote {len(manifest.entries)} tensors under {OUT_DIR}.")
    print(f"State dict: {len(prefixed)} keys → {OUT_DIR}/state_dict.safetensors")


if __name__ == "__main__":
    main()
