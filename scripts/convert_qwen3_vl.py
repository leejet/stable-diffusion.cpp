#!/usr/bin/env python3
"""Convert a Qwen3-VL HF safetensors checkpoint into a sd.cpp-loadable form.

The HF dump prefixes text-tower keys with ``model.language_model.`` and
vision-tower keys with ``model.visual.``. sd.cpp expects ``model.<rest>`` for
the text side; the vision side is converted by sd.cpp's own
``convert_qwen3_vl_vision_name`` and is left as-is here.

Operates on raw safetensors bytes so any dtype (BF16/F16/F32) is preserved.

Usage:
    python3 scripts/convert_qwen3_vl.py <hf_qwen3_vl_dir_or_safetensors> <output.safetensors>
"""

import argparse
import json
import os
import struct
import sys


def rewrite_key(key: str) -> str:
    if key.startswith("model.language_model."):
        return "model." + key[len("model.language_model."):]
    return key


def read_safetensors_header(path: str):
    with open(path, "rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        hdr_bytes = f.read(hdr_len)
    return json.loads(hdr_bytes), 8 + hdr_len


def collect_shard_paths(path: str):
    if os.path.isdir(path):
        index_path = os.path.join(path, "model.safetensors.index.json")
        if os.path.isfile(index_path):
            with open(index_path) as f:
                idx = json.load(f)
            return sorted({os.path.join(path, n) for n in idx["weight_map"].values()})
        single = os.path.join(path, "model.safetensors")
        if os.path.isfile(single):
            return [single]
        raise FileNotFoundError(f"No Qwen3-VL safetensors in {path}")
    if os.path.isfile(path):
        return [path]
    raise FileNotFoundError(path)


def stage_tensors(input_path: str):
    entries = []
    for shard_path in collect_shard_paths(input_path):
        hdr, data_off = read_safetensors_header(shard_path)
        for key, info in hdr.items():
            if key == "__metadata__":
                continue
            entries.append((rewrite_key(key), shard_path, data_off, info))
    return entries


def write_consolidated(out_path: str, entries):
    entries = sorted(entries, key=lambda e: e[0])

    new_header = {}
    cur_offset = 0
    for new_key, shard_path, data_off, info in entries:
        start, end = info["data_offsets"]
        size = end - start
        new_header[new_key] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": [cur_offset, cur_offset + size],
        }
        cur_offset += size

    header_json = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    pad = (-len(header_json)) % 8
    header_json = header_json + (b" " * pad)

    with open(out_path, "wb") as out:
        out.write(struct.pack("<Q", len(header_json)))
        out.write(header_json)
        for new_key, shard_path, data_off, info in entries:
            start, end = info["data_offsets"]
            with open(shard_path, "rb") as src:
                src.seek(data_off + start)
                remaining = end - start
                while remaining > 0:
                    chunk = src.read(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        raise IOError(f"Truncated tensor in {shard_path}")
                    out.write(chunk)
                    remaining -= len(chunk)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="HF Qwen3-VL directory or single safetensors file")
    parser.add_argument("output", help="Output single safetensors path")
    args = parser.parse_args()

    entries = stage_tensors(args.input)
    print(f"Tensors: {len(entries)}")
    print(f"Writing -> {args.output}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_consolidated(args.output, entries)
    print(f"Done. Output size: {os.path.getsize(args.output) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
