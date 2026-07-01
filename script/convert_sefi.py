#!/usr/bin/env python3
"""Convert a SeFi-Image diffusers checkpoint into a single sd.cpp-compatible safetensors.

Operates on raw safetensors bytes so any dtype (BF16, F32, ...) is preserved exactly.
No numpy or torch dependency required.

Usage:
    python3 script/convert_sefi.py <sefi_diffusers_dir> <output.safetensors>
"""

import argparse
import json
import os
import re
import struct
import sys


_LINEAR_TO_LIN = re.compile(r"\.linear\.")
_SHARED_MOD_PREFIXES = (
    "double_stream_modulation_img",
    "double_stream_modulation_txt",
    "single_stream_modulation",
)


def rewrite_transformer_key(key: str) -> str:
    if key.startswith("backbone."):
        key = key[len("backbone."):]
    elif key.startswith("dual_time_embed."):
        return key

    if any(key.startswith(prefix + ".") for prefix in _SHARED_MOD_PREFIXES):
        key = _LINEAR_TO_LIN.sub(".lin.", key, count=1)

    if key == "context_embedder.weight":
        return "txt_in.weight"
    if key == "context_embedder.bias":
        return "txt_in.bias"
    if key == "x_embedder.weight":
        return "img_in.weight"
    if key == "x_embedder.bias":
        return "img_in.bias"

    if key == "proj_out.weight":
        return "final_layer.linear.weight"
    if key == "proj_out.bias":
        return "final_layer.linear.bias"
    if key == "norm_out.linear.weight":
        return "final_layer.adaLN_modulation.1.weight"
    if key == "norm_out.linear.bias":
        return "final_layer.adaLN_modulation.1.bias"

    m = re.match(r"transformer_blocks\.(\d+)\.(.*)$", key)
    if m:
        return _rewrite_double_stream(m.group(1), m.group(2))
    m = re.match(r"single_transformer_blocks\.(\d+)\.(.*)$", key)
    if m:
        return _rewrite_single_stream(m.group(1), m.group(2))

    return key


def _rewrite_double_stream(idx: str, tail: str) -> str:
    dst = f"double_blocks.{idx}."
    mapping = {
        "norm1.linear.weight":          "img_mod.lin.weight",
        "norm1_context.linear.weight":  "txt_mod.lin.weight",
        "attn.norm_q.weight":           "img_attn.norm.query_norm.scale",
        "attn.norm_k.weight":           "img_attn.norm.key_norm.scale",
        "attn.norm_added_q.weight":     "txt_attn.norm.query_norm.scale",
        "attn.norm_added_k.weight":     "txt_attn.norm.key_norm.scale",
        "attn.to_out.0.weight":         "img_attn.proj.weight",
        "attn.to_add_out.weight":       "txt_attn.proj.weight",
        "ff.net.0.proj.weight":         "img_mlp.0.weight",
        "ff.net.2.weight":              "img_mlp.2.weight",
        "ff_context.net.0.proj.weight": "txt_mlp.0.weight",
        "ff_context.net.2.weight":      "txt_mlp.2.weight",
        "ff.linear_in.weight":          "img_mlp.0.weight",
        "ff.linear_out.weight":         "img_mlp.2.weight",
        "ff_context.linear_in.weight":  "txt_mlp.0.weight",
        "ff_context.linear_out.weight": "txt_mlp.2.weight",
    }
    return dst + mapping.get(tail, tail)


# QKV triplets to fuse on output: source tails -> target fused tail.
# Each tuple is (q_tail, k_tail, v_tail, fused_target_tail).
QKV_DOUBLE_TRIPLETS = [
    ("attn.to_q.weight",       "attn.to_k.weight",       "attn.to_v.weight",       "img_attn.qkv.weight"),
    ("attn.add_q_proj.weight", "attn.add_k_proj.weight", "attn.add_v_proj.weight", "txt_attn.qkv.weight"),
]


def _rewrite_single_stream(idx: str, tail: str) -> str:
    dst = f"single_blocks.{idx}."
    mapping = {
        "norm.linear.weight":          "modulation.lin.weight",
        "attn.norm_q.weight":          "norm.query_norm.scale",
        "attn.norm_k.weight":          "norm.key_norm.scale",
        "attn.to_qkv_mlp_proj.weight": "linear1.weight",
        "attn.to_out.weight":          "linear2.weight",
    }
    return dst + mapping.get(tail, tail)




def read_safetensors_header(path: str):
    """Return (header dict, data start byte offset)."""
    with open(path, "rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        hdr_bytes = f.read(hdr_len)
    return json.loads(hdr_bytes), 8 + hdr_len


def collect_shard_paths(directory: str, weight_pattern: str):
    index_path = os.path.join(directory, f"{weight_pattern}.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path) as f:
            idx = json.load(f)
        return sorted({os.path.join(directory, n) for n in idx["weight_map"].values()})
    single = os.path.join(directory, f"{weight_pattern}.safetensors")
    if not os.path.isfile(single):
        raise FileNotFoundError(f"No checkpoint at {directory}: missing {weight_pattern}")
    return [single]


def stage_tensors_for_section(section_dir: str, rewrite_fn):
    """Return a list of (new_key, shard_path, data_start_offset, info_dict) entries.

    A "qkv_fuse" pseudo-entry with three source descriptors is emitted when a
    transformer_blocks.* split q/k/v triplet is found, so the writer can fuse
    them into a single output tensor.
    """
    entries = []
    # First, index all raw keys per shard so we can detect qkv triplets.
    raw_by_block = {}  # block_idx -> {tail: (key, shard_path, data_off, info)}
    raw_others = []
    for shard_path in collect_shard_paths(section_dir, "diffusion_pytorch_model"):
        hdr, data_off = read_safetensors_header(shard_path)
        for key, info in hdr.items():
            if key == "__metadata__":
                continue
            m = re.match(r"backbone\.transformer_blocks\.(\d+)\.(.*)$", key)
            if m and any(m.group(2) in trip[:3] for trip in QKV_DOUBLE_TRIPLETS):
                idx = m.group(1)
                raw_by_block.setdefault(idx, {})[m.group(2)] = (key, shard_path, data_off, info)
            else:
                raw_others.append((key, shard_path, data_off, info))

    for key, shard_path, data_off, info in raw_others:
        new_key = rewrite_fn(key)
        # Swap the (scale, shift) halves to (shift, scale) at conversion time so
        # the on-disk weight matches BFL flux ordering and the runtime stays
        # version-agnostic. norm_out.linear weight shape is [2*dim, dim] and bias
        # is [2*dim]; both split along axis 0 (outermost == row-major outer).
        if new_key in ("final_layer.adaLN_modulation.1.weight",
                       "final_layer.adaLN_modulation.1.bias"):
            info = dict(info)
            info["_chunk_swap_halves"] = True
        entries.append((new_key, shard_path, data_off, info))

    for block_idx, tails in raw_by_block.items():
        for q_tail, k_tail, v_tail, fused_tail in QKV_DOUBLE_TRIPLETS:
            if q_tail in tails and k_tail in tails and v_tail in tails:
                q = tails[q_tail]; k = tails[k_tail]; v = tails[v_tail]
                # Validate shapes match.
                q_shape = q[3]["shape"]; k_shape = k[3]["shape"]; v_shape = v[3]["shape"]
                if q_shape != k_shape or q_shape != v_shape:
                    raise ValueError(f"qkv shape mismatch at block {block_idx} {q_tail}: q={q_shape} k={k_shape} v={v_shape}")
                fused_shape = [q_shape[0] * 3] + list(q_shape[1:])
                fused_info = {
                    "dtype": q[3]["dtype"],
                    "shape": fused_shape,
                    "_qkv_sources": [q, k, v],  # pseudo field consumed by writer
                }
                entries.append((f"double_blocks.{block_idx}.{fused_tail}",
                                None, None, fused_info))
                del tails[q_tail]; del tails[k_tail]; del tails[v_tail]
        # Anything left in tails was an unmatched single - pass through.
        for tail, payload in tails.items():
            entries.append((rewrite_fn(payload[0]),) + payload[1:])
    return entries


_DTYPE_BYTES = {
    "BF16": 2, "F16": 2, "F32": 4, "F64": 8,
    "U8": 1, "I8": 1, "I16": 2, "I32": 4, "I64": 8,
    "BOOL": 1,
}


def _total_bytes(info: dict) -> int:
    if "_qkv_sources" in info:
        elems = 1
        for d in info["shape"]:
            elems *= d
        return elems * _DTYPE_BYTES[info["dtype"]]
    start, end = info["data_offsets"]
    return end - start


def write_consolidated(out_path: str, entries):
    """Write a single safetensors file by streaming raw bytes from each shard.

    For qkv-fused entries, q/k/v are concatenated along axis 0 (row-major), so a
    simple byte-level concatenation produces the correct fused layout for any
    standard dtype.
    """
    entries = sorted(entries, key=lambda e: e[0])

    new_header = {}
    cur_offset = 0
    for new_key, shard_path, data_off, info in entries:
        size = _total_bytes(info)
        new_header[new_key] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": [cur_offset, cur_offset + size],
        }
        cur_offset += size

    header_json = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    pad = (-len(header_json)) % 8
    header_json = header_json + (b" " * pad)

    def copy_range(src_path, src_data_off, src_info, out, byte_range=None):
        start, end = src_info["data_offsets"]
        if byte_range is not None:
            sub_start, sub_end = byte_range
            start, end = start + sub_start, start + sub_end
        with open(src_path, "rb") as src:
            src.seek(src_data_off + start)
            remaining = end - start
            while remaining > 0:
                chunk = src.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise IOError(f"Truncated tensor in {src_path}")
                out.write(chunk)
                remaining -= len(chunk)

    with open(out_path, "wb") as out:
        out.write(struct.pack("<Q", len(header_json)))
        out.write(header_json)
        for new_key, shard_path, data_off, info in entries:
            if "_qkv_sources" in info:
                for q_entry in info["_qkv_sources"]:
                    _, src_path, src_data_off, src_info = q_entry
                    copy_range(src_path, src_data_off, src_info, out)
            elif info.get("_chunk_swap_halves"):
                size = _total_bytes(info)
                half = size // 2
                if size != half * 2:
                    raise ValueError(f"{new_key}: odd byte size {size} cannot be split into halves")
                copy_range(shard_path, data_off, info, out, byte_range=(half, size))
                copy_range(shard_path, data_off, info, out, byte_range=(0, half))
            else:
                copy_range(shard_path, data_off, info, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", help="SeFi diffusers checkpoint directory")
    parser.add_argument("output", help="Output transformer safetensors path (load via --diffusion-model)")
    args = parser.parse_args()

    transformer_entries = stage_tensors_for_section(
        os.path.join(args.input_dir, "transformer"), rewrite_transformer_key)

    print(f"Transformer tensors: {len(transformer_entries)}")
    print(f"Writing {len(transformer_entries)} tensors -> {args.output}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_consolidated(args.output, transformer_entries)
    print(f"Done. Output size: {os.path.getsize(args.output) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
