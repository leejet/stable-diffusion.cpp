"""
Precompute a PuLID-Flux identity embedding from a single source portrait.

Writes a gguf file (a single tensor `pulid_id`) that stable-diffusion.cpp's
`--pulid-id-embedding` flag consumes.

Dependencies (recommended: vendor rather than pip-install due to upstream
packaging quirks):
  - torch + safetensors
  - The ToTheBeginning/PuLID repository's `pulid/` package and `eva_clip/`.
    Put them on PYTHONPATH or sys.path before running this script.
  - insightface, facexlib, torchvision, opencv-python, huggingface_hub, gguf
  - numpy, Pillow

Usage:
  python scripts/pulid_extract_id.py \\
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
from types import SimpleNamespace


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

    # PuLIDPipeline only attaches pulid_ca attributes to `dit` during
    # construction; get_id_embedding() never runs Flux, so a dummy object is
    # enough and avoids importing/building a Flux skeleton.
    print("instantiating PuLIDPipeline with a dummy Flux object", flush=True)
    dit = SimpleNamespace()
    pulid = PuLIDPipeline(dit=dit,
                          device=device,
                          weight_dtype=torch.bfloat16,
                          onnx_provider=onnx_provider)

    print(f"loading PuLID weights from {pulid_weights}", flush=True)
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
