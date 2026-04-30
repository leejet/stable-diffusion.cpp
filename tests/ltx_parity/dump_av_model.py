#!/usr/bin/env python3
"""Dump a tiny LTX-2 AudioVideo model + inputs + outputs for C++ parity testing.

Build a deterministic-random LTXModel(model_type=AudioVideo, num_layers=2) with
the SAME flags as the production 22B (cross_attention_adaln=True,
apply_gated_attention=True), run forward(video, audio), and dump everything
needed for a C++ side-by-side comparison.

Outputs:
    /tmp/ltx_av_model_ref/
        manifest.json   -- catalogue
        weights/*.bin   -- every model parameter
        inputs/*.bin    -- modality fields (latents, contexts, sigmas, etc.) PLUS
                           pre-scaled timesteps that LTXModel.forward_av expects
                           (caller pre-scales, mirroring the existing video-only
                           parity test pattern). Also dumps PE cos/sin pairs.
        outputs/*.bin   -- vx_out, ax_out

Usage:
    /home/ilintar/venv/bin/python tests/ltx_parity/dump_av_model.py
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Dict

import torch

from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.model.transformer.attention import AttentionFunction
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.model import LTXModel, LTXModelType
from ltx_core.model.transformer.rope import LTXRopeType


# ============================================================================
# Config
# ============================================================================

SEED = 0xCA11AB1E
OUT_DIR = pathlib.Path("/tmp/ltx_av_model_ref")
W_DIR   = OUT_DIR / "weights"
IN_DIR  = OUT_DIR / "inputs"
OUT_DIR_T = OUT_DIR / "outputs"

# Tiny dims, full-feature flags. Audio inner_dim = audio.heads * audio.d_head.
# Production 22B uses video 32×128, audio 32×64; we mirror with 4×32 / 4×16.
VIDEO_HEADS, VIDEO_D_HEAD = 4, 32
AUDIO_HEADS, AUDIO_D_HEAD = 4, 16
VIDEO_DIM = VIDEO_HEADS * VIDEO_D_HEAD     # 128
AUDIO_DIM = AUDIO_HEADS * AUDIO_D_HEAD     # 64

# Production has cross_attention_dim == video.dim and audio_cross_attention_dim
# == audio.dim (the cross_attention_adaln modulation requires it).
VIDEO_CROSS_ATTN_DIM = VIDEO_DIM
AUDIO_CROSS_ATTN_DIM = AUDIO_DIM

VIDEO_IN_CHANNELS  = 16
VIDEO_OUT_CHANNELS = 16
AUDIO_IN_CHANNELS  = 8
AUDIO_OUT_CHANNELS = 8

NUM_LAYERS = 2

CROSS_ATTENTION_ADALN  = True
APPLY_GATED_ATTENTION  = True
ROPE_TYPE              = LTXRopeType.INTERLEAVED
NORM_EPS               = 1e-6

VIDEO_MAX_POS  = [20, 2048, 2048]
AUDIO_MAX_POS  = [20]
USE_MIDDLE_INDICES_GRID = True
POS_EMB_THETA           = 10000.0
DOUBLE_PRECISION_ROPE   = False
TIMESTEP_SCALE_MULT     = 1000
AV_CA_TS_SCALE_MULT     = 1

B = 1
F_LAT, H_LAT, W_LAT = 2, 3, 4              # video patch grid
T_VIDEO   = F_LAT * H_LAT * W_LAT          # 24 video tokens
T_AUDIO   = 5
S_VIDEO   = 6
S_AUDIO   = 4


# ============================================================================
# Manifest helper
# ============================================================================


@dataclass
class Manifest:
    weights: Dict[str, Dict] = field(default_factory=dict)
    inputs:  Dict[str, Dict] = field(default_factory=dict)
    outputs: Dict[str, Dict] = field(default_factory=dict)
    config:  Dict           = field(default_factory=dict)

    def _add(self, target, name, t, dest):
        arr = t.detach().to(torch.float32).contiguous().cpu().numpy()
        path = dest / (name.replace("/", "__") + ".bin")
        path.write_bytes(arr.tobytes())
        target[name] = {"shape": list(arr.shape), "path": str(path.relative_to(OUT_DIR))}

    def add_w(self, name, t): self._add(self.weights, name, t, W_DIR)
    def add_i(self, name, t): self._add(self.inputs,  name, t, IN_DIR)
    def add_o(self, name, t): self._add(self.outputs, name, t, OUT_DIR_T)

    def write(self, path):
        path.write_text(json.dumps({
            "config": self.config,
            "weights": self.weights,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }, indent=2, default=str))


# ============================================================================
# Position grids
# ============================================================================


def make_video_positions(F: int, H: int, W: int, device) -> torch.Tensor:
    coords = []
    for f in range(F):
        for h in range(H):
            for w in range(W):
                coords.append([f, h, w])
    coords = torch.tensor(coords, dtype=torch.float32, device=device).t()  # [3, T]
    grid = coords.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, 2).contiguous()  # [B, 3, T, 2]
    return grid


def make_audio_positions(T: int, device) -> torch.Tensor:
    coords = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(0)  # [1, T]
    grid = coords.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, 2).contiguous()
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
        "video": {"in_channels": VIDEO_IN_CHANNELS, "out_channels": VIDEO_OUT_CHANNELS,
                   "dim": VIDEO_DIM, "heads": VIDEO_HEADS, "d_head": VIDEO_D_HEAD,
                   "ctx_dim": VIDEO_CROSS_ATTN_DIM, "T": T_VIDEO, "S": S_VIDEO,
                   "F": F_LAT, "H": H_LAT, "W": W_LAT},
        "audio": {"in_channels": AUDIO_IN_CHANNELS, "out_channels": AUDIO_OUT_CHANNELS,
                   "dim": AUDIO_DIM, "heads": AUDIO_HEADS, "d_head": AUDIO_D_HEAD,
                   "ctx_dim": AUDIO_CROSS_ATTN_DIM, "T": T_AUDIO, "S": S_AUDIO},
        "B": B,
        "num_layers": NUM_LAYERS,
        "cross_attention_adaln": CROSS_ATTENTION_ADALN,
        "apply_gated_attention": APPLY_GATED_ATTENTION,
        "rope_type": ROPE_TYPE.value,
        "norm_eps": NORM_EPS,
        "video_max_pos": VIDEO_MAX_POS,
        "audio_max_pos": AUDIO_MAX_POS,
        "timestep_scale_multiplier": TIMESTEP_SCALE_MULT,
        "av_ca_timestep_scale_multiplier": AV_CA_TS_SCALE_MULT,
        "audio_cross_attention_dim": AUDIO_CROSS_ATTN_DIM,
        "cross_pe_max_pos": max(VIDEO_MAX_POS[0], AUDIO_MAX_POS[0]),
        "seed": SEED,
    }

    # ---- Build the LTXModel ----
    model = LTXModel(
        model_type=LTXModelType.AudioVideo,
        num_attention_heads=VIDEO_HEADS,
        attention_head_dim=VIDEO_D_HEAD,
        in_channels=VIDEO_IN_CHANNELS,
        out_channels=VIDEO_OUT_CHANNELS,
        num_layers=NUM_LAYERS,
        cross_attention_dim=VIDEO_CROSS_ATTN_DIM,
        norm_eps=NORM_EPS,
        attention_type=AttentionFunction.DEFAULT,
        positional_embedding_theta=POS_EMB_THETA,
        positional_embedding_max_pos=VIDEO_MAX_POS,
        timestep_scale_multiplier=TIMESTEP_SCALE_MULT,
        use_middle_indices_grid=USE_MIDDLE_INDICES_GRID,
        audio_num_attention_heads=AUDIO_HEADS,
        audio_attention_head_dim=AUDIO_D_HEAD,
        audio_in_channels=AUDIO_IN_CHANNELS,
        audio_out_channels=AUDIO_OUT_CHANNELS,
        audio_cross_attention_dim=AUDIO_CROSS_ATTN_DIM,
        audio_positional_embedding_max_pos=AUDIO_MAX_POS,
        av_ca_timestep_scale_multiplier=AV_CA_TS_SCALE_MULT,
        rope_type=ROPE_TYPE,
        double_precision_rope=DOUBLE_PRECISION_ROPE,
        apply_gated_attention=APPLY_GATED_ATTENTION,
        caption_projection=None,
        audio_caption_projection=None,
        cross_attention_adaln=CROSS_ATTENTION_ADALN,
    ).to(device).eval()

    # Random init.
    with torch.no_grad():
        for p in model.parameters():
            p.uniform_(-0.05, 0.05)

    # Dump every state_dict tensor.
    for name, p in model.state_dict().items():
        manifest.add_w(name, p)
    print(f"weights: {len(manifest.weights)}")

    # ---- Build modalities and run forward ----
    video_latent  = torch.randn(B, T_VIDEO, VIDEO_IN_CHANNELS, dtype=dtype, device=device)
    audio_latent  = torch.randn(B, T_AUDIO, AUDIO_IN_CHANNELS, dtype=dtype, device=device)
    video_context = torch.randn(B, S_VIDEO, VIDEO_CROSS_ATTN_DIM, dtype=dtype, device=device)
    audio_context = torch.randn(B, S_AUDIO, AUDIO_CROSS_ATTN_DIM, dtype=dtype, device=device)
    video_sigma   = torch.tensor([0.7], dtype=dtype, device=device)
    audio_sigma   = torch.tensor([0.5], dtype=dtype, device=device)

    video_modality = Modality(
        latent=video_latent,
        context=video_context,
        context_mask=None,
        timesteps=video_sigma.view(B, 1, 1),
        sigma=video_sigma,
        positions=make_video_positions(F_LAT, H_LAT, W_LAT, device),
        attention_mask=None,
        enabled=True,
    )
    audio_modality = Modality(
        latent=audio_latent,
        context=audio_context,
        context_mask=None,
        timesteps=audio_sigma.view(B, 1, 1),
        sigma=audio_sigma,
        positions=make_audio_positions(T_AUDIO, device),
        attention_mask=None,
        enabled=True,
    )

    perturbations = BatchedPerturbationConfig.empty(B)
    with torch.no_grad():
        v_out, a_out = model(video=video_modality, audio=audio_modality, perturbations=perturbations)

    # ---- Inputs for the C++ side ----
    # video and audio latents (pre-patchify; LTXModel.forward_av runs patchify itself).
    manifest.add_i("video__latent", video_latent)
    manifest.add_i("audio__latent", audio_latent)
    manifest.add_i("video__context", video_context)
    manifest.add_i("audio__context", audio_context)

    # Pre-scaled timesteps. Our C++ LTXModel.forward_av takes them already scaled
    # (mirroring the existing video-only LTXModel.forward convention).
    v_t_self      = video_sigma * TIMESTEP_SCALE_MULT
    a_t_self      = audio_sigma * TIMESTEP_SCALE_MULT
    v_t_prompt_self = v_t_self                          # cross_attention_adaln uses same σ scaling
    a_t_prompt_self = a_t_self
    # Cross-modality timesteps follow MultiModalTransformerArgsPreprocessor:
    # cross_modality.sigma * timestep_scale_multiplier for the scale_shift adaln,
    # cross_modality.sigma * av_ca_timestep_scale_multiplier for the gate adaln.
    v_t_cross_ss   = audio_sigma * TIMESTEP_SCALE_MULT
    a_t_cross_ss   = video_sigma * TIMESTEP_SCALE_MULT
    v_t_cross_gate = audio_sigma * AV_CA_TS_SCALE_MULT
    a_t_cross_gate = video_sigma * AV_CA_TS_SCALE_MULT

    manifest.add_i("video__t_self",        v_t_self)
    manifest.add_i("audio__t_self",        a_t_self)
    manifest.add_i("video__t_prompt_self", v_t_prompt_self)
    manifest.add_i("audio__t_prompt_self", a_t_prompt_self)
    manifest.add_i("video__t_cross_ss",    v_t_cross_ss)
    manifest.add_i("audio__t_cross_ss",    a_t_cross_ss)
    manifest.add_i("video__t_cross_gate",  v_t_cross_gate)
    manifest.add_i("audio__t_cross_gate",  a_t_cross_gate)

    # Positional embeddings — re-derive via the model's preprocessors so we get
    # exactly what the python forward saw. Then dump cos/sin separately so the
    # C++ side can pack them into our [inner_dim, T, 2] layout.
    video_args = model.video_args_preprocessor.prepare(video_modality, audio_modality)
    audio_args = model.audio_args_preprocessor.prepare(audio_modality, video_modality)
    pe_v_cos, pe_v_sin = video_args.positional_embeddings
    pe_a_cos, pe_a_sin = audio_args.positional_embeddings
    cpe_v_cos, cpe_v_sin = video_args.cross_positional_embeddings
    cpe_a_cos, cpe_a_sin = audio_args.cross_positional_embeddings
    manifest.add_i("video__pe_cos", pe_v_cos)
    manifest.add_i("video__pe_sin", pe_v_sin)
    manifest.add_i("audio__pe_cos", pe_a_cos)
    manifest.add_i("audio__pe_sin", pe_a_sin)
    manifest.add_i("video__cross_pe_cos", cpe_v_cos)
    manifest.add_i("video__cross_pe_sin", cpe_v_sin)
    manifest.add_i("audio__cross_pe_cos", cpe_a_cos)
    manifest.add_i("audio__cross_pe_sin", cpe_a_sin)

    manifest.add_o("video__x_out", v_out)
    manifest.add_o("audio__x_out", a_out)

    manifest.write(OUT_DIR / "manifest.json")
    print(f"[OK] Wrote {len(manifest.weights)} weights, {len(manifest.inputs)} inputs, "
          f"{len(manifest.outputs)} outputs to {OUT_DIR}")
    print(f"video.x_out shape={tuple(v_out.shape)}  audio.x_out shape={tuple(a_out.shape)}")


if __name__ == "__main__":
    main()
