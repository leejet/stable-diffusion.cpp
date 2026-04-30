#!/usr/bin/env python3
"""Dump tiny Gemma 3 reference tensors for C++/GGML parity testing.

Strategy mirrors dump_reference.py: instantiate a tiny Gemma3TextModel (6 layers so
sliding-window pattern triggers at layer 5, small dims) with deterministic tamed
weights, run one forward pass on fixed input_ids, and write every intermediate tensor
(embedding-post-scale, per-layer output, all-layer stack, final norm) to
/tmp/gemma_ref/tensors/ as raw fp32 bytes. Also write the state_dict as safetensors
so the C++ side can load identical weights.

Usage:
    /home/ilintar/venv/bin/python dump_gemma.py
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
from transformers import Gemma3TextConfig, Gemma3TextModel

# -------- Config --------

SEED = 0
OUT_DIR = pathlib.Path("/tmp/gemma_ref")
TENSOR_DIR = OUT_DIR / "tensors"

# Select config via GEMMA_PARITY_VARIANT env var: "tiny" (default) or "deep".
# The tiny variant is fast but only exercises 6 stacked layers with tamed weights.
# The deep variant scales to 24 layers × 512 hidden to stress-test accumulated drift
# and the full sliding/global interleave pattern (same sliding_window_pattern=6 as
# the real Gemma 3 12B). Shared code path in both — differences are pure scaling.
import os
VARIANT = os.environ.get("GEMMA_PARITY_VARIANT", "tiny")

if VARIANT == "deep":
    NUM_LAYERS = 24
    HIDDEN_SIZE = 512
    NUM_HEADS = 8
    NUM_KV_HEADS = 4
    HEAD_DIM = 64
    INTERMEDIATE_SIZE = 1024
    VOCAB_SIZE = 1024
    SLIDING_WINDOW = 16
    SLIDING_WINDOW_PATTERN = 6
    SEQ_LEN = 32  # > sliding_window so the sliding mask actually bites
    # Each "deep" run reuses /tmp/gemma_ref but under a distinct tensor prefix so
    # test_gemma_parity.cpp can load both files without key collision.
    TENSOR_PREFIX_MODEL = "text_encoder_deep.model"
    TENSOR_TAG_PREFIX   = "deep_"  # applied to tensor output filenames
else:
    # Tiny Gemma 3 config. 6 layers so layer index 5 is the first (and only) global
    # layer under the (i+1)%6 rule — exercises both sliding and full paths.
    NUM_LAYERS = 6
    HIDDEN_SIZE = 128
    NUM_HEADS = 4
    NUM_KV_HEADS = 2
    HEAD_DIM = 32  # NOTE: != HIDDEN_SIZE / NUM_HEADS. Matches Gemma's non-standard head_dim.
    INTERMEDIATE_SIZE = 256
    VOCAB_SIZE = 512
    SLIDING_WINDOW = 4
    SLIDING_WINDOW_PATTERN = 6
    SEQ_LEN = 8
    TENSOR_PREFIX_MODEL = "text_encoder.model"
    TENSOR_TAG_PREFIX   = ""

RMS_EPS = 1e-6
ROPE_THETA = 1_000_000.0
ROPE_LOCAL_THETA = 10_000.0
BATCH = 1

TENSOR_PREFIX = TENSOR_PREFIX_MODEL  # Our LLM wrapper stores TextModel under .model,
                                      # so the full key is prefix.model.<tensor_name>.


# -------- Utility --------


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
    """Apply deterministic, finite weights. Mirrors dump_reference.py's approach:
    RMSNorm weights = 1, linears ~= Kaiming with a smaller gain.
    """
    g = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 1:
                # All 1D params: RMS weights, LN weights, biases. Gemma uses RMS with
                # `weight = zeros` + `(1 + weight)` pattern; see below — keep at 0 so
                # effective scale is 1.0 at init.
                p.zero_()
            elif p.dim() == 2:
                fan_in = p.shape[1]
                std = 1.0 / math.sqrt(fan_in)
                p.normal_(mean=0.0, std=std, generator=g)
            else:
                p.normal_(mean=0.0, std=0.02, generator=g)


# -------- Main --------


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TENSOR_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    # Real Gemma 3 12B config has rope_scaling={"rope_type": "linear", "factor": 8.0}
    # applied to full_attention layers only (HuggingFace gemma3 config.json).  Mirror that
    # here in the deep variant so C++ parity actually exercises the scaling path.  The
    # tiny variant keeps scaling disabled (factor=1) for faster iteration / backward compat.
    rope_scaling = {"rope_type": "linear", "factor": 8.0} if VARIANT == "deep" else None
    config = Gemma3TextConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        rms_norm_eps=RMS_EPS,
        rope_theta=ROPE_THETA,
        rope_local_base_freq=ROPE_LOCAL_THETA,
        rope_scaling=rope_scaling,
        sliding_window=SLIDING_WINDOW,
        sliding_window_pattern=SLIDING_WINDOW_PATTERN,
        max_position_embeddings=1024,
        attention_bias=False,
        attn_logit_softcapping=None,
        final_logit_softcapping=None,
        query_pre_attn_scalar=HEAD_DIM,  # 1/sqrt(head_dim) scaling
        hidden_activation="gelu_pytorch_tanh",
    )

    print("Config summary:")
    print(f"  layer_types: {config.layer_types}")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  head_dim:    {config.head_dim}")
    print(f"  sliding_window: {config.sliding_window}")

    model = Gemma3TextModel(config)
    model.eval()
    tame_(model)

    # Fixed input ids.
    rng = np.random.default_rng(SEED)
    input_ids = torch.tensor(rng.integers(low=0, high=VOCAB_SIZE, size=(BATCH, SEQ_LEN)), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    manifest = Manifest()
    save_tensor(input_ids.to(torch.float32), f"{TENSOR_TAG_PREFIX}input_ids", manifest)  # store as f32 for simplicity

    # Forward with output_hidden_states=True to capture every layer.
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    # out.hidden_states is a tuple: (embedding_out, layer_0_out, layer_1_out, ..., layer_N_out)
    # So len == num_layers + 1. First element is post-embed-scale.
    hidden_states = out.hidden_states
    assert len(hidden_states) == NUM_LAYERS + 1, f"got {len(hidden_states)} hidden states"

    for i, h in enumerate(hidden_states):
        tag = f"{TENSOR_TAG_PREFIX}hs_{i:02d}" if i > 0 else f"{TENSOR_TAG_PREFIX}hs_embed"
        save_tensor(h, tag, manifest)
        if VARIANT == "tiny" or i % 4 == 0 or i == len(hidden_states) - 1:
            # Keep logs short for the deep variant.
            print(f"  {tag}: shape={tuple(h.shape)} mean={h.mean().item():.4f} std={h.std().item():.4f}")

    # Final norm'd output (post model.norm, which is Gemma's output rms).
    save_tensor(out.last_hidden_state, f"{TENSOR_TAG_PREFIX}last_hidden_state", manifest)

    # Stacked all-(N+1)-layer tensor as LTX-2 consumes it:
    #   torch.stack(hidden_states, dim=-1) -> [B, T, H, N+1]
    stacked = torch.stack(hidden_states, dim=-1)
    save_tensor(stacked, f"{TENSOR_TAG_PREFIX}all_layers_stacked", manifest)
    print(f"  all_layers_stacked: shape={tuple(stacked.shape)}")

    # Write state dict with our C++-side prefix convention.
    # If a prior run (e.g. "tiny" → then "deep") already wrote a state_dict with a
    # different prefix, merge instead of overwriting so both variants live in one
    # safetensors file and the C++ test can load either config path on demand.
    state_dict = model.state_dict()
    prefixed = {f"{TENSOR_PREFIX}.{k}": v.to(torch.float32).contiguous() for k, v in state_dict.items()}
    sd_path = OUT_DIR / "state_dict.safetensors"
    if sd_path.exists():
        try:
            from safetensors.torch import load_file
            existing = load_file(str(sd_path))
            for k, v in existing.items():
                if k.startswith(f"{TENSOR_PREFIX}."):
                    continue  # replace our own prefix on re-run
                prefixed[k] = v
            print(f"  merged {len(existing)} existing tensors into new state_dict")
        except Exception as e:
            print(f"  warning: could not merge existing state_dict ({e}); overwriting")
    save_file(prefixed, str(sd_path))
    (OUT_DIR / "tensor_names.txt").write_text("\n".join(sorted(prefixed.keys())) + "\n")
    manifest.dump(OUT_DIR / f"manifest{'_deep' if VARIANT == 'deep' else ''}.json")

    # Also dump config JSON so C++ side can cross-check shapes if needed.
    (OUT_DIR / "config.json").write_text(json.dumps({
        "num_layers": NUM_LAYERS,
        "hidden_size": HIDDEN_SIZE,
        "num_heads": NUM_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "intermediate_size": INTERMEDIATE_SIZE,
        "vocab_size": VOCAB_SIZE,
        "rms_norm_eps": RMS_EPS,
        "sliding_window": SLIDING_WINDOW,
        "sliding_window_pattern": SLIDING_WINDOW_PATTERN,
        "rope_theta_global": ROPE_THETA,
        "rope_theta_local": ROPE_LOCAL_THETA,
        "seq_len": SEQ_LEN,
        "batch": BATCH,
        "embed_scale": math.sqrt(HIDDEN_SIZE),
        "layer_types": config.layer_types,
        "tensor_prefix": TENSOR_PREFIX,
    }, indent=2))

    print(f"\nDone. Wrote {len(manifest.entries)} tensors under {OUT_DIR}.")
    print(f"State dict: {len(prefixed)} keys → {OUT_DIR}/state_dict.safetensors")
    print(f"Manifest:   {OUT_DIR}/manifest.json")
    print(f"Name list:  {OUT_DIR}/tensor_names.txt")


if __name__ == "__main__":
    main()
