#!/usr/bin/env python3
"""Dump tiny LTX-2 Connector V1 reference tensors for C++/GGML parity testing.

Covers:
- FeatureExtractorV1 (masked norm + aggregate_embed Linear)
- Embeddings1DConnector (2× BasicTransformerBlock1D + final rms_norm, with
  num_learnable_registers weights present but unused on all-ones mask path)
- PixArtAlphaTextProjection (caption_projection inside the DiT)

Usage:
    /home/ilintar/venv/bin/python dump_connector.py
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

from ltx_core.text_encoders.gemma.embeddings_connector import Embeddings1DConnector
from ltx_core.text_encoders.gemma.embeddings_processor import (
    EmbeddingsProcessor,
    convert_to_additive_mask,
)
from ltx_core.text_encoders.gemma.feature_extractor import FeatureExtractorV1
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.text_projection import PixArtAlphaTextProjection


SEED = 0

# Two variants to exercise different connector paths:
#   - "nopad"  (default): SEQ_LEN=8, NUM_REGISTERS=4, mask all-ones.  Register
#                         replacement is a no-op (reals fill everything) —
#                         covers the "skip concat" branch in C++.
#   - "padded" (env CONNECTOR_VARIANT=padded): SEQ_LEN=8, NUM_REGISTERS=8,
#                         T_REAL=3 with left-padded mask [0,0,0,0,0,1,1,1].
#                         Register replacement moves reals to the front and
#                         fills positions [T_REAL, num_reg) with the trailing
#                         slice of learnable_registers — this is the path the
#                         production conditioner/LTX2ConnectorRunner now takes
#                         when T_real < num_registers.
import os
VARIANT = os.environ.get("CONNECTOR_VARIANT", "nopad")
assert VARIANT in ("nopad", "padded")

OUT_DIR = pathlib.Path("/tmp/connector_ref" if VARIANT == "nopad" else "/tmp/connector_ref_padded")
TENSOR_DIR = OUT_DIR / "tensors"

# Tiny config (mirrors real LTX-2 head_dim=128 for fp16-stable attention;
# 2 heads keeps inner_dim small enough for fast parity).
NUM_HEADS = 2
HEAD_DIM = 32
INNER_DIM = NUM_HEADS * HEAD_DIM  # 64
NUM_LAYERS = 2
ROPE_THETA = 10_000.0
ROPE_MAX_POS = [1]

FEAT_NUM_LAYERS = 5  # fake "embed + 4 transformer layers"
FLAT_DIM = INNER_DIM * FEAT_NUM_LAYERS  # 80

CAPTION_CHANNELS = INNER_DIM  # 64
CAPTION_HIDDEN = 128          # DiT inner dim (larger than connector)
CAPTION_OUT = CAPTION_HIDDEN  # default: = hidden_size

BATCH = 1

if VARIANT == "nopad":
    NUM_REGISTERS = 4
    SEQ_LEN = 8        # > num_reg so register replacement is a no-op
    T_REAL = 8         # entire SEQ_LEN is real tokens
else:  # padded
    NUM_REGISTERS = 8
    SEQ_LEN = 8        # == num_reg (Python requires SEQ_LEN % NUM_REGISTERS == 0)
    T_REAL = 3         # left-padded: only last 3 positions are real

assert SEQ_LEN % NUM_REGISTERS == 0


@dataclass
class Manifest:
    entries: List[Dict] = field(default_factory=list)

    def add(self, name: str, t: torch.Tensor):
        self.entries.append({"name": name, "shape": list(t.shape), "dtype": "f32"})

    def dump(self, path: pathlib.Path):
        path.write_text(json.dumps({"entries": self.entries}, indent=2))


def save_tensor(t: torch.Tensor, name: str, manifest: Manifest):
    safe_name = name.replace("/", "__")
    arr = t.detach().to(torch.float32).contiguous().cpu().numpy()
    arr.tofile(TENSOR_DIR / f"{safe_name}.bin")
    manifest.add(name, t)


def tame_(model: torch.nn.Module):
    g = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 1:
                # RMSNorm weights (standard, not Gemma's (1+w)) etc.: keep at 1 so
                # effective scale is identity at init. For plain biases zero is
                # also fine; we just keep the default shape.
                if "norm" in name.lower() or "weight" == name.split(".")[-1] and p.shape[0] == INNER_DIM:
                    p.fill_(1.0)
                else:
                    p.zero_()
            elif p.dim() == 2:
                fan_in = p.shape[1]
                std = 1.0 / math.sqrt(fan_in)
                p.normal_(mean=0.0, std=std, generator=g)
            else:
                p.normal_(mean=0.0, std=0.02, generator=g)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TENSOR_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    # --- Build modules (tiny). ---
    aggregate_embed = torch.nn.Linear(FLAT_DIM, INNER_DIM, bias=False)
    feature_extractor = FeatureExtractorV1(aggregate_embed=aggregate_embed, is_av=False)

    connector = Embeddings1DConnector(
        attention_head_dim=HEAD_DIM,
        num_attention_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        positional_embedding_theta=ROPE_THETA,
        positional_embedding_max_pos=ROPE_MAX_POS,
        causal_temporal_positioning=False,
        num_learnable_registers=NUM_REGISTERS,
        rope_type=LTXRopeType.INTERLEAVED,
        # True = numpy fp64 linspace + pow cast to fp32 at the end. Matches our
        # C++ fp64 path byte-exactly. With False, torch's fp32 pow drifts 1 ULP
        # at the tail of the grid, causing ~5e-2 cos/sin diffs we can't reproduce.
        double_precision_rope=True,
        apply_gated_attention=False,
    )

    caption_projection = PixArtAlphaTextProjection(
        in_features=CAPTION_CHANNELS,
        hidden_size=CAPTION_HIDDEN,
        out_features=CAPTION_OUT,
        act_fn="gelu_tanh",
    )

    # Tame weights deterministically.
    tame_(feature_extractor)
    tame_(connector)
    tame_(caption_projection)

    # Cast to float32 (the tame() doesn't touch registers which default to bfloat16).
    with torch.no_grad():
        if hasattr(connector, "learnable_registers"):
            g = torch.Generator().manual_seed(SEED + 1)
            connector.learnable_registers.data = (
                torch.rand(NUM_REGISTERS, INNER_DIM, generator=g) * 2.0 - 1.0
            ).to(torch.float32)

    feature_extractor.eval()
    connector.eval()
    caption_projection.eval()

    # --- Build inputs. ---
    rng = np.random.default_rng(SEED + 2)
    # Pretend 49-layer stack (tiny): [B, T, D=INNER_DIM, L=FEAT_NUM_LAYERS]
    stacked = torch.tensor(
        rng.normal(loc=0.0, scale=1.0, size=(BATCH, SEQ_LEN, INNER_DIM, FEAT_NUM_LAYERS)),
        dtype=torch.float32,
    )
    # Binary attention mask.  Left-padded when VARIANT="padded": first
    # (SEQ_LEN - T_REAL) positions are pad (0), last T_REAL are real (1).
    attention_mask = torch.ones((BATCH, SEQ_LEN), dtype=torch.int64)
    if VARIANT == "padded":
        attention_mask[:, : SEQ_LEN - T_REAL] = 0
        # Zero-out the padded positions in the stacked input too, matching what
        # the real HF pipeline feeds (padded tokens have zero embeddings after
        # feature extraction since Gemma's pad_token embedding is unused in the
        # text-to-video pipeline — FeatureExtractor masks them out anyway).
        stacked[:, : SEQ_LEN - T_REAL, :, :] = 0

    manifest = Manifest()
    save_tensor(stacked, "stacked_in", manifest)
    save_tensor(attention_mask.to(torch.float32), "attention_mask", manifest)

    # --- 1. Feature extractor. ---
    with torch.no_grad():
        feat_out, _ = feature_extractor(stacked, attention_mask, padding_side="left")
    save_tensor(feat_out, "feat_ext_out", manifest)
    print(f"  feat_ext_out shape={tuple(feat_out.shape)} "
          f"mean={feat_out.mean().item():.4f} std={feat_out.std().item():.4f}")

    # --- 2. Connector. ---
    additive_mask = convert_to_additive_mask(attention_mask, feat_out.dtype)
    # Run connector piece-by-piece to capture intermediates.
    with torch.no_grad():
        hs = feat_out
        am = additive_mask
        # Register replacement (no-op for all-ones mask, but exercises the path).
        if connector.num_learnable_registers:
            hs, am = connector._replace_padded_with_learnable_registers(hs, am)
        save_tensor(hs, "after_registers", manifest)

        indices_grid = torch.arange(hs.shape[1], dtype=torch.float32)
        indices_grid = indices_grid[None, None, :]
        from ltx_core.model.transformer.rope import (
            generate_freq_grid_np,
            generate_freq_grid_pytorch,
            precompute_freqs_cis,
        )
        freq_gen = generate_freq_grid_np if connector.double_precision_rope else generate_freq_grid_pytorch
        freqs_cis = precompute_freqs_cis(
            indices_grid=indices_grid,
            dim=connector.inner_dim,
            out_dtype=hs.dtype,
            theta=connector.positional_embedding_theta,
            max_pos=connector.positional_embedding_max_pos,
            num_attention_heads=connector.num_attention_heads,
            rope_type=connector.rope_type,
            freq_grid_generator=freq_gen,
        )
        cos_f, sin_f = freqs_cis
        save_tensor(cos_f, "rope_cos", manifest)
        save_tensor(sin_f, "rope_sin", manifest)

        for i, block in enumerate(connector.transformer_1d_blocks):
            hs = block(hs, attention_mask=am, pe=freqs_cis)
            save_tensor(hs, f"conn_block_{i}_out", manifest)
            print(f"  conn_block_{i}_out shape={tuple(hs.shape)} "
                  f"mean={hs.mean().item():.4f} std={hs.std().item():.4f}")

        from ltx_core.utils import rms_norm
        hs = rms_norm(hs)
        save_tensor(hs, "conn_final_out", manifest)
        print(f"  conn_final_out shape={tuple(hs.shape)} "
              f"mean={hs.mean().item():.4f} std={hs.std().item():.4f}")

    # --- 3. Caption projection. ---
    with torch.no_grad():
        caption_out = caption_projection(hs)
    save_tensor(caption_out, "caption_proj_out", manifest)
    print(f"  caption_proj_out shape={tuple(caption_out.shape)} "
          f"mean={caption_out.mean().item():.4f} std={caption_out.std().item():.4f}")

    # --- Save state dict under C++-friendly keys. ---
    state: Dict[str, torch.Tensor] = {}
    # Feature extractor
    state["feature_extractor.aggregate_embed.weight"] = (
        feature_extractor.aggregate_embed.weight.detach().to(torch.float32).contiguous()
    )
    # Connector parameters
    for key, value in connector.state_dict().items():
        state[f"connector.{key}"] = value.detach().to(torch.float32).contiguous()
    # Caption projection
    for key, value in caption_projection.state_dict().items():
        state[f"caption_projection.{key}"] = value.detach().to(torch.float32).contiguous()

    save_file(state, str(OUT_DIR / "state_dict.safetensors"))
    (OUT_DIR / "tensor_names.txt").write_text("\n".join(sorted(state.keys())) + "\n")
    manifest.dump(OUT_DIR / "manifest.json")

    (OUT_DIR / "config.json").write_text(json.dumps({
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "inner_dim": INNER_DIM,
        "num_layers": NUM_LAYERS,
        "num_registers": NUM_REGISTERS,
        "rope_theta": ROPE_THETA,
        "rope_max_pos": ROPE_MAX_POS,
        "feat_num_layers": FEAT_NUM_LAYERS,
        "flat_dim": FLAT_DIM,
        "caption_channels": CAPTION_CHANNELS,
        "caption_hidden": CAPTION_HIDDEN,
        "caption_out": CAPTION_OUT,
        "seq_len": SEQ_LEN,
        "batch": BATCH,
    }, indent=2))

    print(f"\nDone. {len(manifest.entries)} tensors → {OUT_DIR}")
    print(f"State dict: {len(state)} keys → {OUT_DIR}/state_dict.safetensors")


if __name__ == "__main__":
    main()
