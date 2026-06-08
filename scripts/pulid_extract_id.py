"""
Precompute a PuLID-Flux identity embedding from a single source portrait.

Writes a gguf file (a single tensor `pulid_id`) that stable-diffusion.cpp's
`--pulid-id-embedding` flag consumes. See docs/pulid.md for the format and
overall PuLID-Flux flow.

This script intentionally lives outside the C++ build: identity extraction
needs insightface + EVA-CLIP-L + IDFormer, which are PyTorch-only stacks
that would be impractical to reimplement in ggml just to run once per
source person. The C++ side downstream of this file is cross-vendor and
backend-agnostic.

Dependencies (recommended: vendor rather than pip-install due to upstream
packaging quirks):
  - torch + safetensors
  - The ToTheBeginning/PuLID repository's `pulid/pipeline_flux.py` and
    its sibling packages (`flux/`, `eva_clip/`, `models/`). Put them on
    PYTHONPATH or sys.path before running this script.
  - insightface, facexlib (PuLID pipeline pulls these in)
  - numpy, Pillow

Usage:
  python pulid_extract_id.py \\
    --portrait /path/to/source-photo.jpg \\
    --pulid-weights /path/to/pulid_flux_v0.9.1.safetensors \\
    --out /path/to/source.pulidembd

The portrait must contain a clearly visible face. insightface's antelopev2
detector will be auto-downloaded on first run.
"""

from __future__ import annotations

import argparse
import os
import sys


def _make_minimal_flux_skeleton(device):
    """PuLIDPipeline expects a `dit` (Flux transformer) to attach its
    PerceiverAttentionCA modules to during construction. We never run a
    forward pass on it -- the encoders alone (which is what we actually
    need) live on the pipeline object, not the dit. So we instantiate a
    real Flux skeleton with default params and never load its weights."""
    import torch
    from flux.model import Flux
    from flux.util import configs

    with torch.device("cpu"):
        model = Flux(configs["flux-dev"].params).to(torch.bfloat16)
    return model


def extract(portrait_path: str, pulid_weights: str) -> "torch.Tensor":
    import numpy as np
    import torch
    from PIL import Image
    from pulid.pipeline_flux import PuLIDPipeline

    if torch.cuda.is_available():
        device, onnx_provider = "cuda", "gpu"
    else:
        device, onnx_provider = "cpu", "cpu"

    print(f"device={device}", flush=True)

    print("constructing minimal Flux skeleton (no weights loaded)", flush=True)
    dit = _make_minimal_flux_skeleton(device)

    print("instantiating PuLIDPipeline", flush=True)
    pulid = PuLIDPipeline(dit=dit, device=device,
                          weight_dtype=torch.bfloat16,
                          onnx_provider=onnx_provider)

    print(f"loading PuLID weights from {pulid_weights}", flush=True)
    # PuLIDPipeline.load_pretrain expects a "version" string used to construct
    # the default filename when pretrain_path is None. We pass the file
    # directly so the version string is informational only.
    pulid.load_pretrain(pretrain_path=pulid_weights, version="v0.9.1")

    print(f"extracting ID embedding from {portrait_path}", flush=True)
    face_img = np.array(Image.open(portrait_path).convert("RGB"))
    id_embedding, _ = pulid.get_id_embedding(face_img)
    print(f"id embedding shape={tuple(id_embedding.shape)} dtype={id_embedding.dtype}",
          flush=True)

    if id_embedding.ndim == 3 and id_embedding.shape[0] == 1:
        id_embedding = id_embedding[0]
    return id_embedding


def write_embd(tensor, out_path: str, dtype_choice: str) -> None:
    import gguf
    import torch

    if tensor.ndim != 2:
        raise ValueError(f"expected (num_tokens, token_dim); got {tuple(tensor.shape)}")
    num_tokens, token_dim = tensor.shape

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # The embedding ships as a standard gguf container holding a single tensor
    # named "pulid_id". numpy is row-major (num_tokens, token_dim); gguf stores
    # dims reversed, so stable-diffusion.cpp reads it back as
    # ne[0]=token_dim, ne[1]=num_tokens (see load_pulid_id_embedding).
    writer = gguf.GGUFWriter(out_path, arch="pulid")
    writer.add_uint32("pulid.version", 1)

    if dtype_choice == "fp16":
        arr = tensor.to(torch.float16).contiguous().cpu().numpy()
        writer.add_tensor("pulid_id", arr)
    elif dtype_choice == "fp32":
        arr = tensor.to(torch.float32).contiguous().cpu().numpy()
        writer.add_tensor("pulid_id", arr)
    elif dtype_choice == "bf16":
        raw = tensor.to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy()
        writer.add_tensor("pulid_id", raw,
                          raw_shape=(int(num_tokens), int(token_dim)),
                          raw_dtype=gguf.GGMLQuantizationType.BF16)
    else:
        raise ValueError(f"unknown --dtype {dtype_choice}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"wrote {out_path}: gguf, tensor pulid_id [{token_dim}, {num_tokens}] {dtype_choice}",
          flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--portrait", required=True,
                    help="Path to the source portrait image (JPG/PNG).")
    ap.add_argument("--pulid-weights", required=True,
                    help="Path to pulid_flux_v0.9.x.safetensors.")
    ap.add_argument("--out", required=True,
                    help="Output path for the .pulidembd binary.")
    ap.add_argument("--dtype", default="fp16",
                    choices=["fp16", "bf16", "fp32"],
                    help="Storage dtype (default fp16; produces ~131 KB).")
    args = ap.parse_args()

    if not os.path.exists(args.portrait):
        print(f"ERROR: portrait not found at {args.portrait}", file=sys.stderr)
        return 2
    if not os.path.exists(args.pulid_weights):
        print(f"ERROR: PuLID weights not found at {args.pulid_weights}", file=sys.stderr)
        return 3

    embedding = extract(args.portrait, args.pulid_weights)
    write_embd(embedding, args.out, args.dtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
