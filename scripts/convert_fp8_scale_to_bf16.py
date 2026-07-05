#!/usr/bin/env python
import argparse
import json
import math
import os
import struct
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open


FLOAT_DTYPES = {
    "BF16",
    "F16",
    "F32",
    "F64",
    "F8_E4M3",
    "F8_E4M3FN",
    "F8_E5M2",
}

FP8_DTYPES = {
    "F8_E4M3",
    "F8_E4M3FN",
    "F8_E5M2",
}

DTYPE_SIZES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


def read_safetensors_header(path: Path):
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = f.read(header_len).decode("utf-8").rstrip()
    return json.loads(header)


def numel(shape):
    return math.prod(shape) if shape else 1


def scale_key_for_weight(name: str):
    if name.endswith(".weight"):
        return name[:-len(".weight")] + ".weight_scale"
    if name.endswith("weight"):
        return name + "_scale"
    return None


def tensor_nbytes(dtype: str, shape):
    return numel(shape) * DTYPE_SIZES[dtype]


def build_output_plan(header):
    entries = {k: v for k, v in header.items() if k != "__metadata__"}
    paired_scale_keys = set()
    plan = []

    for name, info in entries.items():
        scale_key = scale_key_for_weight(name)
        if info["dtype"] in FP8_DTYPES and scale_key in entries:
            paired_scale_keys.add(scale_key)

    for name, info in entries.items():
        if name in paired_scale_keys:
            continue

        dtype = info["dtype"]
        shape = info["shape"]
        scale_key = scale_key_for_weight(name)

        if dtype in FP8_DTYPES and scale_key in entries:
            scale_info = entries[scale_key]
            plan.append(
                {
                    "name": name,
                    "source_dtype": dtype,
                    "output_dtype": "BF16",
                    "shape": shape,
                    "mode": "fp8_scaled_weight",
                    "scale_key": scale_key,
                }
            )
            continue

        if dtype in FLOAT_DTYPES:
            plan.append(
                {
                    "name": name,
                    "source_dtype": dtype,
                    "output_dtype": "BF16",
                    "shape": shape,
                    "mode": "float_to_bf16",
                }
            )
        else:
            plan.append(
                {
                    "name": name,
                    "source_dtype": dtype,
                    "output_dtype": dtype,
                    "shape": shape,
                    "mode": "copy",
                }
            )

    metadata = dict(header.get("__metadata__", {}) or {})
    metadata["format"] = "pt"
    metadata["conversion"] = "fp8_weight_scale_to_bf16"

    output_header = {"__metadata__": metadata}
    offset = 0
    for item in plan:
        size = tensor_nbytes(item["output_dtype"], item["shape"])
        output_header[item["name"]] = {
            "dtype": item["output_dtype"],
            "shape": item["shape"],
            "data_offsets": [offset, offset + size],
        }
        offset += size

    return plan, output_header, offset


def write_tensor_bytes(out, tensor):
    tensor = tensor.detach().cpu().contiguous()
    if tensor.numel() == 0:
        return
    if tensor.dtype == torch.bfloat16:
        tensor.view(torch.uint16).numpy().tofile(out)
    elif tensor.dtype in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)):
        tensor.view(torch.uint8).numpy().tofile(out)
    else:
        tensor.numpy().tofile(out)


def scale_view_for_chunk(scale, chunk, first_dim_start=0, first_dim_end=None):
    scale = scale.to(torch.float32)

    if scale.numel() == 1:
        return scale.reshape((1,) * chunk.ndim)

    if chunk.ndim > 0 and scale.ndim == 1:
        if first_dim_end is not None and scale.shape[0] >= first_dim_end:
            scale = scale[first_dim_start:first_dim_end]
        if scale.shape[0] == chunk.shape[0]:
            return scale.reshape((scale.shape[0],) + (1,) * (chunk.ndim - 1))

    return scale


def write_scaled_fp8_weight(out, weight, scale, chunk_rows):
    if weight.ndim == 0:
        result = weight.to(torch.float32) * scale_view_for_chunk(scale, weight)
        write_tensor_bytes(out, result.to(torch.bfloat16))
        return

    rows = weight.shape[0]
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        chunk = weight[start:end].to(torch.float32)
        scale_view = scale_view_for_chunk(scale, chunk, start, end)
        result = chunk * scale_view
        write_tensor_bytes(out, result.to(torch.bfloat16))


def write_float_as_bf16(out, tensor, chunk_rows):
    if tensor.dtype == torch.bfloat16:
        write_tensor_bytes(out, tensor)
        return

    if tensor.ndim == 0:
        write_tensor_bytes(out, tensor.to(torch.bfloat16))
        return

    rows = tensor.shape[0]
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        write_tensor_bytes(out, tensor[start:end].to(torch.bfloat16))


def convert(input_path: Path, output_path: Path, chunk_rows: int, dry_run: bool):
    header = read_safetensors_header(input_path)
    plan, output_header, data_size = build_output_plan(header)

    source_counts = Counter(item["source_dtype"] for item in plan)
    output_counts = Counter(item["output_dtype"] for item in plan)
    scaled_count = sum(item["mode"] == "fp8_scaled_weight" for item in plan)
    dropped_scales = sum(item["mode"] == "fp8_scaled_weight" for item in plan)
    header_bytes = json.dumps(output_header, separators=(",", ":")).encode("utf-8")
    expected_size = 8 + len(header_bytes) + data_size

    print(f"input:  {input_path}")
    print(f"output: {output_path}")
    print(f"tensors written: {len(plan)}")
    print(f"scaled fp8 weights dequantized: {scaled_count}")
    print(f"weight_scale tensors dropped: {dropped_scales}")
    print(f"source dtypes: {dict(sorted(source_counts.items()))}")
    print(f"output dtypes: {dict(sorted(output_counts.items()))}")
    print(f"expected output size: {expected_size / (1024 ** 3):.2f} GiB")

    if dry_run:
        return

    if output_path.exists():
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        raise FileExistsError(f"{tmp_path} already exists; remove it or choose another output")

    with safe_open(str(input_path), framework="pt", device="cpu") as sf, tmp_path.open("wb") as out:
        out.write(struct.pack("<Q", len(header_bytes)))
        out.write(header_bytes)

        for index, item in enumerate(plan, 1):
            name = item["name"]
            print(f"[{index:04d}/{len(plan):04d}] {name} -> {item['output_dtype']}")

            tensor = sf.get_tensor(name)
            if item["mode"] == "fp8_scaled_weight":
                scale = sf.get_tensor(item["scale_key"])
                write_scaled_fp8_weight(out, tensor, scale, chunk_rows)
            elif item["mode"] == "float_to_bf16":
                write_float_as_bf16(out, tensor, chunk_rows)
            else:
                write_tensor_bytes(out, tensor)

        actual_size = out.tell()

    if actual_size != expected_size:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"wrote {actual_size} bytes, expected {expected_size} bytes")

    tmp_path.replace(output_path)
    print("done")


def main():
    parser = argparse.ArgumentParser(
        description="Convert an fp8 safetensors checkpoint with weight_scale tensors to bf16."
    )
    parser.add_argument("--input", default="ideogram4_fp8.safetensors", type=Path)
    parser.add_argument("--output", default="ideogram4_bf16.safetensors", type=Path)
    parser.add_argument("--chunk-rows", default=1024, type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if args.chunk_rows < 1:
        raise ValueError("--chunk-rows must be >= 1")
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if args.overwrite and output_path.exists():
        output_path.unlink()

    convert(input_path, output_path, args.chunk_rows, args.dry_run)


if __name__ == "__main__":
    main()
