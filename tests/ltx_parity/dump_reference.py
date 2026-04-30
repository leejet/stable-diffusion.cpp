#!/usr/bin/env python3
"""Dump LTX-2 reference tensors for C++/GGML parity testing.

Strategy: instantiate a TINY LTX-2 model (2 layers, small dims) with deterministic
random weights, run a single forward pass on fixed inputs, and write every intermediate
tensor (post-each-block, post-AdaLN, post-patchify, final output) to
/tmp/ltx_ref/tensors/ as raw fp32 bytes. Also dump the state_dict as safetensors so the
C++ side can load the exact same weights.

Usage:
    /home/ilintar/venv/bin/python dump_reference.py

Outputs:
    /tmp/ltx_ref/manifest.json         -- catalogue of every dumped tensor
    /tmp/ltx_ref/state_dict.safetensors -- model weights
    /tmp/ltx_ref/tensors/*.bin          -- raw fp32 bytes, one file per tensor
    /tmp/ltx_ref/tensor_names.txt       -- state_dict.keys() for name-mapping verification
"""

from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
from safetensors.torch import save_file

from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.model.transformer.adaln import AdaLayerNormSingle
from ltx_core.model.transformer.attention import Attention, AttentionFunction
from ltx_core.model.transformer.feed_forward import FeedForward
from ltx_core.model.transformer.model import LTXModel, LTXModelType
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.rope import (
    LTXRopeType,
    apply_rotary_emb,
    generate_freq_grid_pytorch,
    precompute_freqs_cis,
)
from ltx_core.model.transformer.timestep_embedding import (
    PixArtAlphaCombinedTimestepSizeEmbeddings,
)
from ltx_core.guidance.perturbations import BatchedPerturbationConfig

# -------- Config --------

SEED = 0
OUT_DIR = pathlib.Path("/tmp/ltx_ref")
TENSOR_DIR = OUT_DIR / "tensors"

# Tiny model config — deliberately small so every tensor is cheap to dump.
INNER_DIM = 128
NUM_HEADS = 4
HEAD_DIM = 32  # NUM_HEADS * HEAD_DIM = INNER_DIM
NUM_LAYERS = 2
IN_CHANNELS = 16
OUT_CHANNELS = 16
CROSS_ATTN_DIM = 128  # keep == INNER_DIM to avoid needing caption_projection for now
NORM_EPS = 1e-6

# Toy latent (F, H, W) — small but with at least 2 frames to exercise temporal axis.
F_LAT, H_LAT, W_LAT = 2, 4, 6
BATCH = 1
FPS = 24.0

# Synthetic text context.
CONTEXT_LEN = 8


# -------- Utility --------


@dataclass
class Manifest:
    entries: List[Dict] = field(default_factory=list)

    def add(self, name: str, tensor: torch.Tensor, notes: str = ""):
        t = tensor.detach().to(torch.float32).contiguous().cpu()
        # Flatten name → filename by replacing '/' with '__' so everything lives in one dir.
        fname = name.replace("/", "__") + ".bin"
        path = TENSOR_DIR / fname
        path.write_bytes(t.numpy().tobytes())
        self.entries.append(
            {
                "name": name,
                "shape": list(t.shape),
                "dtype": "float32",
                "nbytes": t.numel() * 4,
                "path": str(path.relative_to(OUT_DIR)),
                "notes": notes,
            }
        )

    def dump(self, path: pathlib.Path):
        path.write_text(json.dumps({"entries": self.entries}, indent=2))


def seeded_randn(shape, seed_offset=0):
    g = torch.Generator().manual_seed(SEED + seed_offset)
    return torch.randn(shape, generator=g, dtype=torch.float32)


# -------- Dumpers --------


def dump_rope():
    """Dump RoPE freqs_cis + apply_rotary_emb result for a known grid."""
    # 3D positions, middle-grid form: shape [B, n_pos_dims, T, 2] with (start, end) pairs.
    F, H, W = F_LAT, H_LAT, W_LAT
    T = F * H * W
    positions = torch.zeros(BATCH, 3, T, 2, dtype=torch.float32)
    idx = 0
    for f in range(F):
        for h in range(H):
            for w in range(W):
                # Time axis divided by fps per ltx_pipelines/utils/tools.py:135.
                positions[0, 0, idx, 0] = f / FPS
                positions[0, 0, idx, 1] = (f + 1) / FPS
                positions[0, 1, idx, 0] = h
                positions[0, 1, idx, 1] = h + 1
                positions[0, 2, idx, 0] = w
                positions[0, 2, idx, 1] = w + 1
                idx += 1

    cos, sin = precompute_freqs_cis(
        positions,
        dim=INNER_DIM,
        out_dtype=torch.float32,
        theta=10000.0,
        max_pos=[20, 2048, 2048],
        use_middle_indices_grid=True,
        num_attention_heads=NUM_HEADS,
        rope_type=LTXRopeType.SPLIT,
        freq_grid_generator=generate_freq_grid_pytorch,
    )

    # Apply to a known q tensor so we can diff both the pe itself and the post-rotation output.
    q = seeded_randn((BATCH, T, INNER_DIM), seed_offset=100)
    q_rot = apply_rotary_emb(q, (cos, sin), LTXRopeType.SPLIT)

    m = {}
    m["rope/positions"] = positions
    m["rope/cos"] = cos
    m["rope/sin"] = sin
    m["rope/q_in"] = q
    m["rope/q_rotated"] = q_rot
    return m


def dump_scheduler():
    """LTX2Scheduler output for a few representative configurations.
    Keys: 'schedule/tokens_{N}_steps_{S}' → sigma array of length S+1.
    """
    scheduler = LTX2Scheduler()
    cases = [
        # (tokens, steps, stretch, terminal)
        (1024, 10, True, 0.1),   # small latent (BASE_SHIFT anchor)
        (1024, 30, True, 0.1),
        (4096, 10, True, 0.1),   # MAX_SHIFT anchor
        (4096, 40, True, 0.1),   # typical LTX-2 default
        (2560, 30, True, 0.1),   # interpolated
        (4096, 8,  False, 0.1),  # no stretch path
    ]
    out = {}
    for tokens, steps, stretch, terminal in cases:
        # LTX2Scheduler expects a `latent` tensor to derive tokens from shape[2:].
        # Fake one with product(shape[2:]) == tokens.
        fake_latent = torch.zeros(1, 1, tokens)
        sigmas = scheduler.execute(
            steps=steps, latent=fake_latent,
            max_shift=2.05, base_shift=0.95,
            stretch=stretch, terminal=terminal,
        )
        key = f"schedule/tokens{tokens}_steps{steps}_stretch{int(stretch)}"
        out[key] = sigmas.detach().float()
    return out


def dump_adaln():
    """AdaLayerNormSingle: t → (modulation[B, coeff, dim], embedded[B, dim])."""
    torch.manual_seed(SEED + 2)
    adaln = AdaLayerNormSingle(embedding_dim=INNER_DIM, embedding_coefficient=6).eval()

    # Fixed timestep σ ∈ (0, 1). Python applies *1000 externally; mirror that here.
    sigma = torch.tensor([0.42], dtype=torch.float32)
    t_scaled = sigma * 1000.0

    with torch.no_grad():
        modulation, embedded = adaln(t_scaled, hidden_dtype=torch.float32)

    # Extract sub-weights for loading into C++. The isolated AdaLN test weights are not loaded into
    # the full LTXRunner, so the prefix only needs to be unique w.r.t. the full-model weights.
    sd = {f"adaln_standalone.{k}": v.detach().float() for k, v in adaln.state_dict().items()}

    return {
        "adaln/sigma": sigma,
        "adaln/t_scaled": t_scaled,
        "adaln/modulation": modulation,
        "adaln/embedded": embedded,
    }, sd


def dump_full_model():
    """Tiny LTXModel (VideoOnly) forward, dumping per-block outputs."""
    torch.manual_seed(SEED + 3)

    # Stash a helper to tame magnitudes for parity testing. With default init, scale_shift_table is
    # torch.empty(...) (uninitialised memory — random garbage) and many Linears have Kaiming init
    # which, compounded across blocks with AdaLN * (1 + scale) + shift modulation, produces values
    # that overflow fp32 (output becomes NaN). We don't care about the semantics of the weights —
    # only that C++ and Python compute the SAME function on the SAME weights — so we replace them
    # with bounded random values post-construction.

    model = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=NUM_HEADS,
        attention_head_dim=HEAD_DIM,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        num_layers=NUM_LAYERS,
        cross_attention_dim=CROSS_ATTN_DIM,
        norm_eps=NORM_EPS,
        attention_type=AttentionFunction.PYTORCH,
        positional_embedding_theta=10000.0,
        positional_embedding_max_pos=[20, 2048, 2048],
        timestep_scale_multiplier=1000,
        use_middle_indices_grid=True,
        rope_type=LTXRopeType.SPLIT,
        double_precision_rope=False,
        apply_gated_attention=False,
        caption_projection=None,  # cross_attention_dim == inner_dim so no projection needed
        cross_attention_adaln=False,
    ).eval()

    # Tame weights: small random gaussians with scale 1/sqrt(dim), scale_shift_table zeroed so the
    # forward is well-conditioned even with two randomly-initialised blocks stacked.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "scale_shift_table" in name:
                p.zero_()
                continue
            # q_norm / k_norm RMSNorm weights must be ~1 to act as normalisers (not kill signal).
            if name.endswith("q_norm.weight") or name.endswith("k_norm.weight"):
                p.fill_(1.0)
                continue
            if p.dim() == 1:
                # biases → zero
                p.zero_()
            else:
                # Kaiming-ish: std ~ 1/sqrt(fan_in)
                fan_in = p.shape[1] if p.dim() >= 2 else p.numel()
                p.normal_(0.0, 1.0 / (fan_in ** 0.5))

    # Synthetic inputs.
    F, H, W = F_LAT, H_LAT, W_LAT
    T = F * H * W
    latent = seeded_randn((BATCH, IN_CHANNELS, F, H, W), seed_offset=200)
    sigma = torch.tensor([0.5], dtype=torch.float32)
    context = seeded_randn((BATCH, CONTEXT_LEN, CROSS_ATTN_DIM), seed_offset=300)

    # Build positions in (B, n_pos_dims, T, 2) middle-grid form.
    positions = torch.zeros(BATCH, 3, T, 2, dtype=torch.float32)
    idx = 0
    for f in range(F):
        for h in range(H):
            for w in range(W):
                positions[0, 0, idx, 0] = f / FPS
                positions[0, 0, idx, 1] = (f + 1) / FPS
                positions[0, 1, idx, 0] = h
                positions[0, 1, idx, 1] = h + 1
                positions[0, 2, idx, 0] = w
                positions[0, 2, idx, 1] = w + 1
                idx += 1

    # LTX's Modality carries the latent pre-patchify (shape [B, C, F, H, W] → flat [B, T, C]).
    # patchify_proj is Linear(in_channels, inner_dim) so we need [B, T, C] for input.
    latent_flat = latent.permute(0, 2, 3, 4, 1).reshape(BATCH, T, IN_CHANNELS)

    # For pure T2V, per-token timesteps = sigma broadcast.
    timesteps = sigma.view(BATCH, 1).expand(BATCH, T).contiguous()

    # Positions shape the preprocessor wants is [B, 3, T] (no middle-grid pair dim) when
    # use_middle_indices_grid=False, or [B, 3, T, 2] when True. Our positions are already the
    # [B, 3, T, 2] form. Good.
    modality = Modality(
        latent=latent_flat,
        sigma=sigma,
        timesteps=timesteps,
        positions=positions,
        context=context,
        enabled=True,
        context_mask=None,
        attention_mask=None,
    )

    # Instrument: intercept transformer_blocks outputs so we can dump per-block.
    per_block_outputs = {}
    orig_forwards = []
    for i, blk in enumerate(model.transformer_blocks):
        orig = blk.forward
        orig_forwards.append(orig)

        def make_capture(idx, original):
            def capture(video=None, audio=None, perturbations=None):
                out_video, out_audio = original(video=video, audio=audio, perturbations=perturbations)
                per_block_outputs[f"block_{idx:02d}_out"] = out_video.x.detach().float().clone()
                return out_video, out_audio
            return capture

        blk.forward = make_capture(i, orig)

    with torch.no_grad():
        vx, _ = model(video=modality, audio=None, perturbations=BatchedPerturbationConfig.empty(BATCH))

    # Also capture post-patchify result by running patchify_proj manually (same computation).
    with torch.no_grad():
        patchified = model.patchify_proj(latent_flat)
        tm_mod, tm_embedded = model.adaln_single(timesteps.flatten() * 1000.0, hidden_dtype=torch.float32)
        tm_mod = tm_mod.view(BATCH, -1, tm_mod.shape[-1])
        tm_embedded = tm_embedded.view(BATCH, -1, tm_embedded.shape[-1])

    # Also save the unflattened latent in [C, F, H, W] order (batch=1 squeezed).
    # Memory layout: W innermost → matches ggml ne=[W, H, F, C] which is what LTXRunner::build_graph
    # expects at its entry point. Convert by squeezing batch dim from the original [B=1, C, F, H, W].
    latent_unflat = latent.squeeze(0)  # [C, F, H, W]

    # Velocity output comes out of the Python model as [B, T, C=out_channels]. Also save the
    # unflattened [C, F, H, W] form so C++ can compare without reshaping.
    vx_unflat = vx.reshape(BATCH, F, H, W, OUT_CHANNELS).permute(0, 4, 1, 2, 3).squeeze(0)  # [C, F, H, W]

    tensors = {
        "model/latent_in": latent_flat,
        "model/latent_unflat": latent_unflat,
        "model/sigma": sigma,
        "model/timesteps_per_token": timesteps,
        "model/context_in": context,
        "model/positions": positions,
        "model/patchify_out": patchified,
        "model/adaln_modulation": tm_mod,
        "model/adaln_embedded_timestep": tm_embedded,
        "model/velocity_out": vx,
        "model/velocity_out_unflat": vx_unflat,
    }
    for k, v in per_block_outputs.items():
        tensors[f"model/{k}"] = v

    # Use the sd.cpp convention: DiT weights live under "model.diffusion_model.".
    # Pairs with LTXRunner's default prefix so the C++ loader reads names verbatim.
    sd = {f"model.diffusion_model.{k}": v.detach().float() for k, v in model.state_dict().items()}

    # --- Single-step Euler parity ----------------------------------------------------------------
    # Starting from the noisy latent + the same velocity we just computed, run ONE deterministic
    # Euler step using the LTX2Scheduler with 10 steps at σ=0.5 (which falls between sigmas[k]
    # and sigmas[k+1] for some k — we pick the step endpoints manually so C++ gets exact inputs).
    # This validates the (σ_next - σ) * v formula through the denoiser↔DiT integration boundary.
    sched = LTX2Scheduler()
    sched_sigmas = sched.execute(steps=10, latent=torch.zeros(1, 1, T), stretch=True, terminal=0.1)

    # Pick one adjacent sigma pair. sigmas[4] is reasonably mid-trajectory for 10 steps.
    step_idx = 4
    sigma_cur  = sched_sigmas[step_idx].item()
    sigma_next = sched_sigmas[step_idx + 1].item()

    # The model was just run at σ=0.5; for the Euler test, re-run at σ_cur (a schedule value).
    # The `vx` we already have is at σ=0.5 which doesn't match; redo the forward with sigma_cur.
    timesteps_step = torch.tensor([sigma_cur], dtype=torch.float32).view(BATCH, 1).expand(BATCH, T).contiguous()
    modality_step  = Modality(
        latent=latent_flat,
        sigma=torch.tensor([sigma_cur], dtype=torch.float32),
        timesteps=timesteps_step,
        positions=positions,
        context=context,
        enabled=True,
        context_mask=None,
        attention_mask=None,
    )
    with torch.no_grad():
        v_step, _ = model(video=modality_step, audio=None, perturbations=BatchedPerturbationConfig.empty(BATCH))

    # Euler step: x_next = x + (σ_next - σ) * v (LTX-2 predicts velocity directly).
    x_next = latent_flat + (sigma_next - sigma_cur) * v_step

    # Also dump the unflattened form for C++ convenience.
    x_next_unflat = x_next.reshape(BATCH, F, H, W, IN_CHANNELS).permute(0, 4, 1, 2, 3).squeeze(0)  # [C, F, H, W]
    v_step_unflat = v_step.reshape(BATCH, F, H, W, OUT_CHANNELS).permute(0, 4, 1, 2, 3).squeeze(0)

    tensors["euler/sigma_cur"]  = torch.tensor([sigma_cur], dtype=torch.float32)
    tensors["euler/sigma_next"] = torch.tensor([sigma_next], dtype=torch.float32)
    tensors["euler/v_step"]     = v_step
    tensors["euler/v_step_unflat"] = v_step_unflat
    tensors["euler/x_next"]     = x_next
    tensors["euler/x_next_unflat"] = x_next_unflat

    return tensors, sd


def dump_full_model_v2(num_layers: int = NUM_LAYERS,
                       zero_scale_shift_table: bool = True,
                       prefix: str = "model.diffusion_model_v2",
                       tensor_prefix: str = "v2model",
                       seed_offset: int = 4):
    """Tiny LTXModel (VideoOnly) with V2 features enabled:
      - cross_attention_adaln=True (adds prompt_scale_shift_table, prompt_adaln_single,
        extends scale_shift_table to 9 coeffs, routes CA through apply_cross_attention_adaln)
      - apply_gated_attention=True (adds to_gate_logits on attn1 and attn2)
    State-dict is saved under `prefix` so multiple variants can coexist in the same file.

    Args:
        num_layers: how many transformer blocks to stack.  Deeper values exercise
            cross-layer drift (e.g. the real 22B DiT has 48).
        zero_scale_shift_table: if False, initialise all scale_shift_table /
            prompt_scale_shift_table weights with bounded random values so the
            modulation path (AdaLN multiply/shift + CA mod) is actually exercised
            — the default True path is too well-conditioned to surface sign/layout
            bugs in the (1+scale) and shift-kv branches.
    """
    torch.manual_seed(SEED + seed_offset)

    model = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=NUM_HEADS,
        attention_head_dim=HEAD_DIM,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        num_layers=num_layers,
        cross_attention_dim=CROSS_ATTN_DIM,
        norm_eps=NORM_EPS,
        attention_type=AttentionFunction.PYTORCH,
        positional_embedding_theta=10000.0,
        positional_embedding_max_pos=[20, 2048, 2048],
        timestep_scale_multiplier=1000,
        use_middle_indices_grid=True,
        rope_type=LTXRopeType.SPLIT,
        double_precision_rope=False,
        apply_gated_attention=True,
        caption_projection=None,
        cross_attention_adaln=True,
    ).eval()

    # Tame weights, same recipe as V1. The sst branch is optional: leaving it zero
    # means AdaLN modulation degenerates to identity, which hides bugs in the
    # (1 + scale) path and the CA-AdaLN shift_kv / scale_kv broadcast.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "scale_shift_table" in name:
                if zero_scale_shift_table:
                    p.zero_()
                else:
                    # Keep magnitudes small so the stacked modulation doesn't explode
                    # across layers. scale_shift_table rows are added to a (0, 1]-ish
                    # AdaLN output; 0.05 keeps the post-modulation scale in ~[0.95, 1.05].
                    p.normal_(0.0, 0.05)
                continue
            if name.endswith("q_norm.weight") or name.endswith("k_norm.weight"):
                p.fill_(1.0)
                continue
            if p.dim() == 1:
                p.zero_()
            else:
                fan_in = p.shape[1] if p.dim() >= 2 else p.numel()
                p.normal_(0.0, 1.0 / (fan_in ** 0.5))

    F, H, W = F_LAT, H_LAT, W_LAT
    T = F * H * W
    latent = seeded_randn((BATCH, IN_CHANNELS, F, H, W), seed_offset=400)
    sigma = torch.tensor([0.5], dtype=torch.float32)
    context = seeded_randn((BATCH, CONTEXT_LEN, CROSS_ATTN_DIM), seed_offset=500)

    positions = torch.zeros(BATCH, 3, T, 2, dtype=torch.float32)
    idx = 0
    for f in range(F):
        for h in range(H):
            for w in range(W):
                positions[0, 0, idx, 0] = f / FPS
                positions[0, 0, idx, 1] = (f + 1) / FPS
                positions[0, 1, idx, 0] = h
                positions[0, 1, idx, 1] = h + 1
                positions[0, 2, idx, 0] = w
                positions[0, 2, idx, 1] = w + 1
                idx += 1

    latent_flat = latent.permute(0, 2, 3, 4, 1).reshape(BATCH, T, IN_CHANNELS)
    timesteps = sigma.view(BATCH, 1).expand(BATCH, T).contiguous()
    modality = Modality(
        latent=latent_flat,
        sigma=sigma,
        timesteps=timesteps,
        positions=positions,
        context=context,
        enabled=True,
        context_mask=None,
        attention_mask=None,
    )

    per_block_outputs = {}
    for i, blk in enumerate(model.transformer_blocks):
        orig = blk.forward

        def make_capture(idx, original):
            def capture(video=None, audio=None, perturbations=None):
                out_video, out_audio = original(video=video, audio=audio, perturbations=perturbations)
                per_block_outputs[f"block_{idx:02d}_out"] = out_video.x.detach().float().clone()
                return out_video, out_audio
            return capture

        blk.forward = make_capture(i, orig)

    with torch.no_grad():
        vx, _ = model(video=modality, audio=None, perturbations=BatchedPerturbationConfig.empty(BATCH))

    with torch.no_grad():
        patchified = model.patchify_proj(latent_flat)
        tm_mod, tm_embedded = model.adaln_single(timesteps.flatten() * 1000.0, hidden_dtype=torch.float32)
        tm_mod = tm_mod.view(BATCH, -1, tm_mod.shape[-1])
        tm_embedded = tm_embedded.view(BATCH, -1, tm_embedded.shape[-1])
        # V2 extra: prompt_adaln output driven by sigma (× scale_mult = 1000).
        p_mod, _ = model.prompt_adaln_single(
            (sigma * 1000.0).flatten(), hidden_dtype=torch.float32
        )
        p_mod = p_mod.view(BATCH, -1, p_mod.shape[-1])

    latent_unflat = latent.squeeze(0)
    vx_unflat = vx.reshape(BATCH, F, H, W, OUT_CHANNELS).permute(0, 4, 1, 2, 3).squeeze(0)

    tensors = {
        f"{tensor_prefix}/latent_in": latent_flat,
        f"{tensor_prefix}/latent_unflat": latent_unflat,
        f"{tensor_prefix}/sigma": sigma,
        f"{tensor_prefix}/timesteps_per_token": timesteps,
        f"{tensor_prefix}/context_in": context,
        f"{tensor_prefix}/positions": positions,
        f"{tensor_prefix}/patchify_out": patchified,
        f"{tensor_prefix}/adaln_modulation": tm_mod,
        f"{tensor_prefix}/adaln_embedded_timestep": tm_embedded,
        f"{tensor_prefix}/prompt_modulation": p_mod,
        f"{tensor_prefix}/velocity_out": vx,
        f"{tensor_prefix}/velocity_out_unflat": vx_unflat,
    }
    for k, v in per_block_outputs.items():
        tensors[f"{tensor_prefix}/{k}"] = v

    sd = {f"{prefix}.{k}": v.detach().float() for k, v in model.state_dict().items()}
    return tensors, sd


# -------- Main --------


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TENSOR_DIR.mkdir(parents=True, exist_ok=True)

    torch.use_deterministic_algorithms(False)  # some ops (layernorm) aren't deterministic
    torch.manual_seed(SEED)

    manifest = Manifest()
    state_dict: Dict[str, torch.Tensor] = {}

    print("[1/4] RoPE …")
    for name, t in dump_rope().items():
        manifest.add(name, t)

    print("[2/4] LTX2Scheduler …")
    for name, t in dump_scheduler().items():
        manifest.add(name, t)

    print("[3/4] AdaLayerNormSingle …")
    adaln_tensors, adaln_sd = dump_adaln()
    for name, t in adaln_tensors.items():
        manifest.add(name, t)
    state_dict.update(adaln_sd)

    print("[4/5] Full LTXModel (tiny, V1) …")
    model_tensors, model_sd = dump_full_model()
    for name, t in model_tensors.items():
        manifest.add(name, t)
    state_dict.update(model_sd)

    print("[5/6] Full LTXModel (tiny, V2: cross_attention_adaln + apply_gated_attention) …")
    model_v2_tensors, model_v2_sd = dump_full_model_v2()
    for name, t in model_v2_tensors.items():
        manifest.add(name, t)
    state_dict.update(model_v2_sd)

    # Deep V2: 8 layers + non-zero scale_shift_table so accumulated modulation drift
    # surfaces.  The original V2 dump is too gentle (only 2 layers, zeroed sst) to
    # catch bugs that only matter when modulation is non-trivial.
    print("[6/6] Full LTXModel (tiny, V2-deep: 8 layers, non-zero scale_shift_table) …")
    v2_deep_tensors, v2_deep_sd = dump_full_model_v2(
        num_layers=8,
        zero_scale_shift_table=False,
        prefix="model.diffusion_model_v2_deep",
        tensor_prefix="v2deep",
        seed_offset=7,
    )
    for name, t in v2_deep_tensors.items():
        manifest.add(name, t)
    state_dict.update(v2_deep_sd)

    # Safetensors requires contiguous CPU tensors.
    sd_contig = {k: v.contiguous().cpu() for k, v in state_dict.items()}
    save_file(sd_contig, str(OUT_DIR / "state_dict.safetensors"))

    manifest_path = OUT_DIR / "manifest.json"
    manifest.dump(manifest_path)

    with (OUT_DIR / "tensor_names.txt").open("w") as f:
        for name in sorted(state_dict.keys()):
            t = state_dict[name]
            f.write(f"{name}\t{list(t.shape)}\t{t.dtype}\n")

    print(f"Done. Wrote {len(manifest.entries)} tensors under {OUT_DIR}.")
    print(f"State dict: {len(state_dict)} keys → {OUT_DIR}/state_dict.safetensors")
    print(f"Manifest: {manifest_path}")
    print(f"Name inventory: {OUT_DIR}/tensor_names.txt")


if __name__ == "__main__":
    main()
