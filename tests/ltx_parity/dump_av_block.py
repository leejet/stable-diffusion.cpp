#!/usr/bin/env python3
"""Dump LTX-2 AV transformer block weights, inputs, and outputs for C++ parity testing.

Strategy: instantiate a TINY AV-enabled BasicAVTransformerBlock with the SAME flags
as the production 22B model (cross_attention_adaln=True, apply_gated_attention=True,
audio dim=64, video dim=128). Run preprocessor + block forward on deterministic random
inputs and dump:

  - block state_dict as named .bin files (raw fp32 bytes per parameter)
  - all TransformerArgs fields needed by our C++ forward_av (video + audio sides)
  - block outputs (vx_out, ax_out)

Layout is chosen so raw bytes are interpretable as ggml ne (column-major fastest).
Python's torch tensor [B, T, dim] memory layout matches ggml ne [dim, T, B], so we
write `.numpy().tobytes()` of contiguous tensors directly.

Outputs:
    /tmp/ltx_av_block_ref/
        manifest.json   -- catalogue {name -> {shape, path}}
        weights/*.bin   -- block parameters
        inputs/*.bin    -- TransformerArgs fields
        outputs/*.bin   -- vx_out, ax_out

Usage:
    /home/ilintar/venv/bin/python tests/ltx_parity/dump_av_block.py
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch

from ltx_core.model.transformer.adaln import AdaLayerNormSingle
from ltx_core.model.transformer.attention import AttentionFunction
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.transformer import (
    BasicAVTransformerBlock,
    TransformerConfig,
)
from ltx_core.model.transformer.transformer_args import (
    MultiModalTransformerArgsPreprocessor,
)


# ============================================================================
# Config — tiny dims, FULL feature flags matching 22B
# ============================================================================

SEED = 0xA1C0DE
OUT_DIR = pathlib.Path("/tmp/ltx_av_block_ref")
W_DIR = OUT_DIR / "weights"
IN_DIR = OUT_DIR / "inputs"
OUT_DIR_T = OUT_DIR / "outputs"

# ---- Modality dims ----
# Video: heads × d_head = inner_dim. Tiny but matches the 22B's "heads, d_head"
# pattern (32×128 there → inner_dim 4096; we use 4×32 → 128).
VIDEO_HEADS   = 4
VIDEO_D_HEAD  = 32
VIDEO_DIM     = VIDEO_HEADS * VIDEO_D_HEAD       # 128
# In production 22B: cross_attention_dim == video.dim. The cross_attention_adaln
# path applies prompt_scale_shift_table[shape (2, video.dim)] elementwise to the
# context, which requires context_dim == video.dim. Mirror that here.
VIDEO_CTX_DIM = VIDEO_DIM

# Audio: 22B uses 32×64 (smaller per-head) → inner_dim 2048. We mirror with 4×16 → 64.
AUDIO_HEADS   = 4
AUDIO_D_HEAD  = 16
AUDIO_DIM     = AUDIO_HEADS * AUDIO_D_HEAD       # 64
AUDIO_CTX_DIM = AUDIO_DIM                         # same constraint as video

# Sequence lengths (B=1).
B          = 1
F_LAT      = 2          # video frames
H_LAT      = 3
W_LAT      = 4
T_VIDEO    = F_LAT * H_LAT * W_LAT  # 24 video tokens
T_AUDIO    = 5                       # audio tokens
S_VIDEO    = 6                       # video text-context tokens
S_AUDIO    = 4                       # audio text-context tokens

# ---- Feature flags (match 22B) ----
CROSS_ATTENTION_ADALN  = True
APPLY_GATED_ATTENTION  = True
ROPE_TYPE              = LTXRopeType.INTERLEAVED   # block-level test; INTERLEAVED is well-tested in our C++ path
NORM_EPS               = 1e-6

# ---- Positional embedding params ----
VIDEO_POS_DIMS    = 3                              # (frame, h, w)
AUDIO_POS_DIMS    = 1                              # frame index only
VIDEO_MAX_POS     = [20, 2048, 2048]
AUDIO_MAX_POS     = [20]
USE_MIDDLE_INDICES_GRID = True
POS_EMB_THETA           = 10000.0
DOUBLE_PRECISION_ROPE   = False
TIMESTEP_SCALE_MULT     = 1000
AV_CA_TS_SCALE_MULT     = 1


# ============================================================================
# Manifest helper
# ============================================================================


@dataclass
class Manifest:
    weights: Dict[str, Dict] = field(default_factory=dict)
    inputs:  Dict[str, Dict] = field(default_factory=dict)
    outputs: Dict[str, Dict] = field(default_factory=dict)
    config:  Dict          = field(default_factory=dict)

    def _add(self, target: Dict, name: str, t: torch.Tensor, dest: pathlib.Path):
        arr = t.detach().to(torch.float32).contiguous().cpu().numpy()
        path = dest / (name.replace("/", "__") + ".bin")
        path.write_bytes(arr.tobytes())
        target[name] = {"shape": list(arr.shape), "dtype": "float32",
                         "nbytes": arr.nbytes, "path": str(path.relative_to(OUT_DIR))}

    def add_weight(self, name: str, t: torch.Tensor):
        self._add(self.weights, name, t, W_DIR)

    def add_input(self, name: str, t: torch.Tensor):
        self._add(self.inputs, name, t, IN_DIR)

    def add_output(self, name: str, t: torch.Tensor):
        self._add(self.outputs, name, t, OUT_DIR_T)

    def write(self, path: pathlib.Path):
        path.write_text(json.dumps({
            "config": self.config,
            "weights": self.weights,
            "inputs":  self.inputs,
            "outputs": self.outputs,
        }, indent=2, default=str))


# ============================================================================
# Build positions for a modality
# ============================================================================


def make_video_positions(F: int, H: int, W: int, device, fps: float = 24.0) -> torch.Tensor:
    """Generate INDICES_GRID for video positions. Shape: (B, n_pos_dims=3, T_total).
    With use_middle_indices_grid=True we need a (B, n_pos_dims, T, 2) tensor where
    [..., 0] = start index and [..., 1] = end index of the indices grid cell.
    For a non-VAE-compressed test, start==end == [frame_idx, h, w].
    """
    # Build coords [F*H*W, 3] in (f, h, w) order, then transpose to [3, T].
    coords = []
    for f in range(F):
        for h in range(H):
            for w in range(W):
                coords.append([f, h, w])
    coords = torch.tensor(coords, dtype=torch.float32, device=device).t()  # [3, T]
    # Replicate start==end along last axis for use_middle_indices_grid=True.
    grid = coords.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, 2).contiguous()  # [B, 3, T, 2]
    return grid


def make_audio_positions(T: int, device) -> torch.Tensor:
    """Audio positions are 1-D (frame index)."""
    coords = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(0)  # [1, T]
    grid = coords.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, 2).contiguous()   # [B, 1, T, 2]
    return grid


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    W_DIR.mkdir(exist_ok=True)
    IN_DIR.mkdir(exist_ok=True)
    OUT_DIR_T.mkdir(exist_ok=True)

    device = torch.device("cpu")
    dtype  = torch.float32

    manifest = Manifest()
    manifest.config = {
        "video": {"dim": VIDEO_DIM, "heads": VIDEO_HEADS, "d_head": VIDEO_D_HEAD,
                   "ctx_dim": VIDEO_CTX_DIM, "T": T_VIDEO, "S": S_VIDEO,
                   "F": F_LAT, "H": H_LAT, "W": W_LAT},
        "audio": {"dim": AUDIO_DIM, "heads": AUDIO_HEADS, "d_head": AUDIO_D_HEAD,
                   "ctx_dim": AUDIO_CTX_DIM, "T": T_AUDIO, "S": S_AUDIO},
        "B": B,
        "cross_attention_adaln": CROSS_ATTENTION_ADALN,
        "apply_gated_attention": APPLY_GATED_ATTENTION,
        "rope_type": ROPE_TYPE.value,
        "norm_eps": NORM_EPS,
        "video_max_pos": VIDEO_MAX_POS,
        "audio_max_pos": AUDIO_MAX_POS,
        "use_middle_indices_grid": USE_MIDDLE_INDICES_GRID,
        "pos_emb_theta": POS_EMB_THETA,
        "timestep_scale_multiplier": TIMESTEP_SCALE_MULT,
        "av_ca_timestep_scale_multiplier": AV_CA_TS_SCALE_MULT,
        "audio_cross_attention_dim": AUDIO_CTX_DIM,
        "cross_pe_max_pos": max(VIDEO_MAX_POS[0], AUDIO_MAX_POS[0]),
        "seed": SEED,
    }

    # ---- Build the AV block ----
    video_cfg = TransformerConfig(
        dim=VIDEO_DIM, heads=VIDEO_HEADS, d_head=VIDEO_D_HEAD,
        context_dim=VIDEO_CTX_DIM,
        apply_gated_attention=APPLY_GATED_ATTENTION,
        cross_attention_adaln=CROSS_ATTENTION_ADALN,
    )
    audio_cfg = TransformerConfig(
        dim=AUDIO_DIM, heads=AUDIO_HEADS, d_head=AUDIO_D_HEAD,
        context_dim=AUDIO_CTX_DIM,
        apply_gated_attention=APPLY_GATED_ATTENTION,
        cross_attention_adaln=CROSS_ATTENTION_ADALN,
    )
    block = BasicAVTransformerBlock(
        idx=0,
        video=video_cfg,
        audio=audio_cfg,
        rope_type=ROPE_TYPE,
        norm_eps=NORM_EPS,
        attention_function=AttentionFunction.DEFAULT,
    ).to(device).eval()

    # Random init all params (skipping bias init that some Linear layers default to zero).
    with torch.no_grad():
        for p in block.parameters():
            p.uniform_(-0.05, 0.05)

    # Dump every block parameter.
    for name, p in block.state_dict().items():
        manifest.add_weight(name, p)

    # ---- Build the AV preprocessors (model-level adaln modules) ----
    # AdaLayerNormSingle with the right embedding_coefficient values.
    # adaln_embedding_coefficient(cross_attention_adaln=True) = 9, False = 6.
    coef_main = 9 if CROSS_ATTENTION_ADALN else 6
    video_adaln = AdaLayerNormSingle(VIDEO_DIM, embedding_coefficient=coef_main).to(device).eval()
    audio_adaln = AdaLayerNormSingle(AUDIO_DIM, embedding_coefficient=coef_main).to(device).eval()
    video_prompt_adaln = AdaLayerNormSingle(VIDEO_DIM, embedding_coefficient=2).to(device).eval() if CROSS_ATTENTION_ADALN else None
    audio_prompt_adaln = AdaLayerNormSingle(AUDIO_DIM, embedding_coefficient=2).to(device).eval() if CROSS_ATTENTION_ADALN else None
    av_ca_video_ss_adaln = AdaLayerNormSingle(VIDEO_DIM, embedding_coefficient=4).to(device).eval()
    av_ca_audio_ss_adaln = AdaLayerNormSingle(AUDIO_DIM, embedding_coefficient=4).to(device).eval()
    av_ca_a2v_gate_adaln = AdaLayerNormSingle(VIDEO_DIM, embedding_coefficient=1).to(device).eval()
    av_ca_v2a_gate_adaln = AdaLayerNormSingle(AUDIO_DIM, embedding_coefficient=1).to(device).eval()

    with torch.no_grad():
        for m in [video_adaln, audio_adaln, video_prompt_adaln, audio_prompt_adaln,
                  av_ca_video_ss_adaln, av_ca_audio_ss_adaln,
                  av_ca_a2v_gate_adaln, av_ca_v2a_gate_adaln]:
            if m is None:
                continue
            for p in m.parameters():
                p.uniform_(-0.05, 0.05)

    # ---- Build patchify projections (linear layers feeding the preprocessor) ----
    # in_channels arbitrary (we'll provide already-patchified x by setting in_channels=dim and using identity-like init? No, just pick small in_channels and let it project up).
    VIDEO_IN_CHANNELS = 16
    AUDIO_IN_CHANNELS = 16
    video_patchify = torch.nn.Linear(VIDEO_IN_CHANNELS, VIDEO_DIM, bias=True).to(device).eval()
    audio_patchify = torch.nn.Linear(AUDIO_IN_CHANNELS, AUDIO_DIM, bias=True).to(device).eval()
    with torch.no_grad():
        for p in video_patchify.parameters(): p.uniform_(-0.05, 0.05)
        for p in audio_patchify.parameters(): p.uniform_(-0.05, 0.05)

    # ---- Build preprocessors ----
    cross_pe_max_pos = max(VIDEO_MAX_POS[0], AUDIO_MAX_POS[0])
    video_prep = MultiModalTransformerArgsPreprocessor(
        patchify_proj=video_patchify,
        adaln=video_adaln,
        cross_scale_shift_adaln=av_ca_video_ss_adaln,
        cross_gate_adaln=av_ca_a2v_gate_adaln,
        inner_dim=VIDEO_DIM,
        max_pos=VIDEO_MAX_POS,
        num_attention_heads=VIDEO_HEADS,
        cross_pe_max_pos=cross_pe_max_pos,
        use_middle_indices_grid=USE_MIDDLE_INDICES_GRID,
        audio_cross_attention_dim=AUDIO_CTX_DIM,
        timestep_scale_multiplier=TIMESTEP_SCALE_MULT,
        double_precision_rope=DOUBLE_PRECISION_ROPE,
        positional_embedding_theta=POS_EMB_THETA,
        rope_type=ROPE_TYPE,
        av_ca_timestep_scale_multiplier=AV_CA_TS_SCALE_MULT,
        caption_projection=None,
        prompt_adaln=video_prompt_adaln,
    )
    audio_prep = MultiModalTransformerArgsPreprocessor(
        patchify_proj=audio_patchify,
        adaln=audio_adaln,
        cross_scale_shift_adaln=av_ca_audio_ss_adaln,
        cross_gate_adaln=av_ca_v2a_gate_adaln,
        inner_dim=AUDIO_DIM,
        max_pos=AUDIO_MAX_POS,
        num_attention_heads=AUDIO_HEADS,
        cross_pe_max_pos=cross_pe_max_pos,
        use_middle_indices_grid=USE_MIDDLE_INDICES_GRID,
        audio_cross_attention_dim=AUDIO_CTX_DIM,
        timestep_scale_multiplier=TIMESTEP_SCALE_MULT,
        double_precision_rope=DOUBLE_PRECISION_ROPE,
        positional_embedding_theta=POS_EMB_THETA,
        rope_type=ROPE_TYPE,
        av_ca_timestep_scale_multiplier=AV_CA_TS_SCALE_MULT,
        caption_projection=None,
        prompt_adaln=audio_prompt_adaln,
    )

    # ---- Build modalities ----
    video_latent = torch.randn(B, T_VIDEO, VIDEO_IN_CHANNELS, dtype=dtype, device=device)
    audio_latent = torch.randn(B, T_AUDIO, AUDIO_IN_CHANNELS, dtype=dtype, device=device)
    video_context = torch.randn(B, S_VIDEO, VIDEO_CTX_DIM, dtype=dtype, device=device)
    audio_context = torch.randn(B, S_AUDIO, AUDIO_CTX_DIM, dtype=dtype, device=device)
    # No context mask → full attention. The preprocessor's _prepare_attention_mask
    # only converts non-float masks to additive log-space bias; passing None is
    # the cleanest way to skip masking entirely on both python and C++ sides.
    video_ctx_mask = None
    audio_ctx_mask = None
    video_timesteps = torch.tensor([0.7], dtype=dtype, device=device)
    audio_timesteps = torch.tensor([0.5], dtype=dtype, device=device)
    video_sigma     = torch.tensor([0.7], dtype=dtype, device=device)
    audio_sigma     = torch.tensor([0.5], dtype=dtype, device=device)
    video_positions = make_video_positions(F_LAT, H_LAT, W_LAT, device)
    audio_positions = make_audio_positions(T_AUDIO, device)

    video_modality = Modality(
        latent=video_latent,
        context=video_context,
        context_mask=video_ctx_mask,
        timesteps=video_timesteps.unsqueeze(0).unsqueeze(0),  # [B, 1, 1] for AdaLN flatten
        sigma=video_sigma,
        positions=video_positions,
        attention_mask=None,
        enabled=True,
    )
    audio_modality = Modality(
        latent=audio_latent,
        context=audio_context,
        context_mask=audio_ctx_mask,
        timesteps=audio_timesteps.unsqueeze(0).unsqueeze(0),
        sigma=audio_sigma,
        positions=audio_positions,
        attention_mask=None,
        enabled=True,
    )

    # ---- Run preprocessors ----
    video_args = video_prep.prepare(video_modality, cross_modality=audio_modality)
    audio_args = audio_prep.prepare(audio_modality, cross_modality=video_modality)

    # ---- Dump TransformerArgs ----
    # video.
    manifest.add_input("video__x",                  video_args.x)
    manifest.add_input("video__context",            video_args.context)
    if video_args.context_mask is not None:
        manifest.add_input("video__context_mask",   video_args.context_mask)
    manifest.add_input("video__timesteps",          video_args.timesteps)
    if video_args.prompt_timestep is not None:
        manifest.add_input("video__prompt_timestep", video_args.prompt_timestep)
    pe_v_cos, pe_v_sin = video_args.positional_embeddings
    manifest.add_input("video__pe_cos", pe_v_cos)
    manifest.add_input("video__pe_sin", pe_v_sin)
    if video_args.cross_positional_embeddings is not None:
        cpe_v_cos, cpe_v_sin = video_args.cross_positional_embeddings
        manifest.add_input("video__cross_pe_cos", cpe_v_cos)
        manifest.add_input("video__cross_pe_sin", cpe_v_sin)
    if video_args.cross_scale_shift_timestep is not None:
        manifest.add_input("video__cross_scale_shift_timestep", video_args.cross_scale_shift_timestep)
    if video_args.cross_gate_timestep is not None:
        manifest.add_input("video__cross_gate_timestep", video_args.cross_gate_timestep)

    # audio.
    manifest.add_input("audio__x",                  audio_args.x)
    manifest.add_input("audio__context",            audio_args.context)
    if audio_args.context_mask is not None:
        manifest.add_input("audio__context_mask",   audio_args.context_mask)
    manifest.add_input("audio__timesteps",          audio_args.timesteps)
    if audio_args.prompt_timestep is not None:
        manifest.add_input("audio__prompt_timestep", audio_args.prompt_timestep)
    pe_a_cos, pe_a_sin = audio_args.positional_embeddings
    manifest.add_input("audio__pe_cos", pe_a_cos)
    manifest.add_input("audio__pe_sin", pe_a_sin)
    if audio_args.cross_positional_embeddings is not None:
        cpe_a_cos, cpe_a_sin = audio_args.cross_positional_embeddings
        manifest.add_input("audio__cross_pe_cos", cpe_a_cos)
        manifest.add_input("audio__cross_pe_sin", cpe_a_sin)
    if audio_args.cross_scale_shift_timestep is not None:
        manifest.add_input("audio__cross_scale_shift_timestep", audio_args.cross_scale_shift_timestep)
    if audio_args.cross_gate_timestep is not None:
        manifest.add_input("audio__cross_gate_timestep", audio_args.cross_gate_timestep)

    # ---- Run the block forward ----
    with torch.no_grad():
        video_out, audio_out = block(video=video_args, audio=audio_args)

    manifest.add_output("video__x_out", video_out.x)
    manifest.add_output("audio__x_out", audio_out.x)

    # ---- Write manifest ----
    manifest.write(OUT_DIR / "manifest.json")
    print(f"[OK] Wrote {len(manifest.weights)} weights, "
          f"{len(manifest.inputs)} inputs, {len(manifest.outputs)} outputs to {OUT_DIR}")
    print(f"video.x_out shape={tuple(video_out.x.shape)}  audio.x_out shape={tuple(audio_out.x.shape)}")


if __name__ == "__main__":
    main()
