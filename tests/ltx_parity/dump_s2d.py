#!/usr/bin/env python3
"""Dump reference space-to-depth / depth-to-space outputs along each of the three
stride axes (W, H, T) as standalone test vectors.

Each case applies a single-axis stride=2 split (the building block that will be
composed to give full 3D SpaceToDepth). We dump both the input and the expected
output so the C++ side can verify its ggml reshape+permute chain byte-exact.

Output: /tmp/s2d_ref/tensors/*.bin + manifest.json + config.json.
Usage:  /home/ilintar/venv/bin/python dump_s2d.py
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
from einops import rearrange

OUT_DIR = pathlib.Path("/tmp/s2d_ref")
TENSOR_DIR = OUT_DIR / "tensors"

# Distinct primes where possible so any mis-axis mixup shows up immediately.
B, C, T, H, W = 1, 3, 4, 6, 8   # after: (T/2, H, W), (T, H/2, W), (T, H, W/2) per case
FACTOR = 2


@dataclass
class Manifest:
    entries: List[Dict] = field(default_factory=list)
    def add(self, name, t): self.entries.append({"name": name, "shape": list(t.shape), "dtype": "f32"})
    def dump(self, p): p.write_text(json.dumps({"entries": self.entries}, indent=2))


def save(t: torch.Tensor, name: str, mf: Manifest):
    arr = t.detach().to(torch.float32).contiguous().cpu().numpy()
    arr.tofile(TENSOR_DIR / f"{name}.bin")
    mf.add(name, t)


def s2d_W(x: torch.Tensor, p3: int) -> torch.Tensor:
    # [B, C, T, H, W*p3] -> [B, C*p3, T, H, W]
    return rearrange(x, "b c t h (w p3) -> b (c p3) t h w", p3=p3)


def s2d_H(x: torch.Tensor, p2: int) -> torch.Tensor:
    # [B, C, T, H*p2, W] -> [B, C*p2, T, H, W]
    return rearrange(x, "b c t (h p2) w -> b (c p2) t h w", p2=p2)


def s2d_T(x: torch.Tensor, p1: int) -> torch.Tensor:
    # [B, C, T*p1, H, W] -> [B, C*p1, T, H, W]
    return rearrange(x, "b c (t p1) h w -> b (c p1) t h w", p1=p1)


def s2d_full(x: torch.Tensor, p1: int, p2: int, p3: int) -> torch.Tensor:
    # [B, C, T*p1, H*p2, W*p3] -> [B, C*p1*p2*p3, T, H, W]
    return rearrange(x, "b c (t p1) (h p2) (w p3) -> b (c p1 p2 p3) t h w",
                     p1=p1, p2=p2, p3=p3)


def d2s_W(x: torch.Tensor, p3: int) -> torch.Tensor:
    return rearrange(x, "b (c p3) t h w -> b c t h (w p3)", p3=p3)


def d2s_H(x: torch.Tensor, p2: int) -> torch.Tensor:
    return rearrange(x, "b (c p2) t h w -> b c t (h p2) w", p2=p2)


def d2s_T(x: torch.Tensor, p1: int) -> torch.Tensor:
    return rearrange(x, "b (c p1) t h w -> b c (t p1) h w", p1=p1)


def d2s_full(x: torch.Tensor, p1: int, p2: int, p3: int) -> torch.Tensor:
    return rearrange(x, "b (c p1 p2 p3) t h w -> b c (t p1) (h p2) (w p3)",
                     p1=p1, p2=p2, p3=p3)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TENSOR_DIR.mkdir(parents=True, exist_ok=True)
    mf = Manifest()
    torch.manual_seed(0)

    # --- Axis-W primitive ---
    x_w = torch.randn(B, C, T, H, W * FACTOR)
    save(x_w, "input_axisW", mf)
    save(s2d_W(x_w, FACTOR), "expected_axisW", mf)

    # --- Axis-H primitive ---
    x_h = torch.randn(B, C, T, H * FACTOR, W)
    save(x_h, "input_axisH", mf)
    save(s2d_H(x_h, FACTOR), "expected_axisH", mf)

    # --- Axis-T primitive ---
    x_t = torch.randn(B, C, T * FACTOR, H, W)
    save(x_t, "input_axisT", mf)
    save(s2d_T(x_t, FACTOR), "expected_axisT", mf)

    # --- Full 3D (stride=(2,2,2)) composition ---
    x_all = torch.randn(B, C, T * FACTOR, H * FACTOR, W * FACTOR)
    save(x_all, "input_full222", mf)
    save(s2d_full(x_all, FACTOR, FACTOR, FACTOR), "expected_full222", mf)

    # --- Stride=(1,2,2) (what compress_space_res uses) ---
    x_122 = torch.randn(B, C, T, H * FACTOR, W * FACTOR)
    save(x_122, "input_full122", mf)
    save(s2d_full(x_122, 1, FACTOR, FACTOR), "expected_full122", mf)

    # --- Stride=(2,1,1) (compress_time_res) ---
    x_211 = torch.randn(B, C, T * FACTOR, H, W)
    save(x_211, "input_full211", mf)
    save(s2d_full(x_211, FACTOR, 1, 1), "expected_full211", mf)

    # --- DepthToSpace (single-axis + composed) ---
    # Input for axis primitives: [B, C_large, T, H, W] where C_large = C * factor.
    dx_w = torch.randn(B, C * FACTOR, T, H, W)
    save(dx_w, "dinput_axisW", mf)
    save(d2s_W(dx_w, FACTOR), "dexpected_axisW", mf)

    dx_h = torch.randn(B, C * FACTOR, T, H, W)
    save(dx_h, "dinput_axisH", mf)
    save(d2s_H(dx_h, FACTOR), "dexpected_axisH", mf)

    dx_t = torch.randn(B, C * FACTOR, T, H, W)
    save(dx_t, "dinput_axisT", mf)
    save(d2s_T(dx_t, FACTOR), "dexpected_axisT", mf)

    dx_222 = torch.randn(B, C * (FACTOR ** 3), T, H, W)
    save(dx_222, "dinput_full222", mf)
    save(d2s_full(dx_222, FACTOR, FACTOR, FACTOR), "dexpected_full222", mf)

    dx_122 = torch.randn(B, C * (FACTOR ** 2), T, H, W)
    save(dx_122, "dinput_full122", mf)
    save(d2s_full(dx_122, 1, FACTOR, FACTOR), "dexpected_full122", mf)

    dx_211 = torch.randn(B, C * FACTOR, T, H, W)
    save(dx_211, "dinput_full211", mf)
    save(d2s_full(dx_211, FACTOR, 1, 1), "dexpected_full211", mf)

    # --- PixelNorm (dim=1 RMS) ---
    eps = 1e-8
    pn_in = torch.randn(B, 5, T, H, W)  # C=5 to exercise a non-power-of-2 channel
    pn_out = pn_in / torch.sqrt((pn_in ** 2).mean(dim=1, keepdim=True) + eps)
    save(pn_in,  "pn_input", mf)
    save(pn_out, "pn_expected", mf)

    # --- PerChannelStatistics ---
    # Random mu and sigma (sigma > 0). Buffers shape [C] as in the real VAE.
    c_pcs = 6
    pcs_in    = torch.randn(B, c_pcs, T, H, W)
    pcs_mu    = torch.randn(c_pcs)
    pcs_sigma = torch.rand(c_pcs) + 0.5        # keep away from zero
    save(pcs_in,        "pcs_input",   mf)
    save(pcs_mu,        "pcs_mu",      mf)
    save(pcs_sigma,     "pcs_sigma",   mf)
    save((pcs_in - pcs_mu.view(1, c_pcs, 1, 1, 1)) / pcs_sigma.view(1, c_pcs, 1, 1, 1),
         "pcs_normalize_expected", mf)
    save((pcs_in * pcs_sigma.view(1, c_pcs, 1, 1, 1)) + pcs_mu.view(1, c_pcs, 1, 1, 1),
         "pcs_unnormalize_expected", mf)

    mf.dump(OUT_DIR / "manifest.json")
    (OUT_DIR / "config.json").write_text(json.dumps({
        "B": B, "C": C, "T": T, "H": H, "W": W, "FACTOR": FACTOR,
        "pn_C": 5, "pn_eps": eps,
        "pcs_C": c_pcs,
    }, indent=2))
    print(f"wrote {len(mf.entries)} tensors under {OUT_DIR}")


if __name__ == "__main__":
    main()
