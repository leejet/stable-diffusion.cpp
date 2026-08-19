#!/usr/bin/env python3
"""Merge selected tensors from multiple safetensors files without loading weights.

Edit ``OUTPUT_PATH`` and ``SOURCE_RULES`` below, then run:

    python scripts/merge_safetensors.py

Each source rule uses regular expressions against complete tensor names.
``include`` is required and matches when any expression succeeds. ``exclude``
wins over ``include``. Expressions are evaluated with ``re.search``.
"""

import json
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

OUTPUT_PATH = Path(r".minimax_h3_fl2va_pruned_bf16.safetensors")

SOURCE_RULES = [
    {
        "path": Path(r".minimax_h3_fl2va_bf16.safetensors"),
        "include": [r".*"],
        "exclude": [r".*adaln_proj\.linear.*", r"time_embedder.*"],
    },
    {
        "path": Path(r".minimax_h3_fl2va_pruned_int8_convrot.safetensors"),
        "include": [r"^.*adaln_proj\.linear.*", "adaln_t_table"],
        "exclude": [],
    },
]

# Safetensors metadata is optional. Set this to a dict[str, str] if needed.
OUTPUT_METADATA = None

# Refuse to replace an existing output unless explicitly enabled.
OVERWRITE_OUTPUT = False

# Only tensor headers and this fixed-size buffer are held in memory.
COPY_BUFFER_SIZE = 8 * 1024 * 1024
PROGRESS_INTERVAL = 1024 * 1024 * 1024
MAX_HEADER_SIZE = 256 * 1024 * 1024


@dataclass(frozen=True)
class TensorEntry:
    name: str
    source_path: Path
    source_data_offset: int
    source_start: int
    source_end: int
    dtype: str
    shape: list[int]

    @property
    def size(self) -> int:
        return self.source_end - self.source_start


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def read_exact(file: BinaryIO, size: int, description: str) -> bytes:
    data = file.read(size)
    if len(data) != size:
        raise ValueError(f"truncated {description}: expected {size} bytes, got {len(data)}")
    return data


def read_safetensors_header(path: Path) -> tuple[dict, int, int]:
    file_size = path.stat().st_size
    with path.open("rb") as file:
        header_size = struct.unpack("<Q", read_exact(file, 8, f"header size in {path}"))[0]
        if header_size == 0 or header_size > MAX_HEADER_SIZE:
            raise ValueError(
                f"invalid header size in {path}: {header_size} "
                f"(limit: {MAX_HEADER_SIZE})"
            )
        header_bytes = read_exact(file, header_size, f"header in {path}")

    try:
        header = json.loads(header_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid safetensors JSON header in {path}: {error}") from error
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header in {path} is not an object")

    data_offset = 8 + header_size
    if data_offset > file_size:
        raise ValueError(f"safetensors data offset is past end of file: {path}")
    return header, data_offset, file_size


def parse_tensor_entry(
    name: str,
    info: object,
    source_path: Path,
    source_data_offset: int,
    source_file_size: int,
) -> TensorEntry:
    if not isinstance(info, dict):
        raise ValueError(f"{source_path}: tensor {name!r} has an invalid header entry")

    dtype = info.get("dtype")
    shape = info.get("shape")
    offsets = info.get("data_offsets")
    if not isinstance(dtype, str):
        raise ValueError(f"{source_path}: tensor {name!r} has an invalid dtype")
    if not isinstance(shape, list) or not all(
        isinstance(dimension, int) and dimension >= 0 for dimension in shape
    ):
        raise ValueError(f"{source_path}: tensor {name!r} has an invalid shape")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(offset, int) for offset in offsets)
    ):
        raise ValueError(f"{source_path}: tensor {name!r} has invalid data offsets")

    start, end = offsets
    if start < 0 or end < start or source_data_offset + end > source_file_size:
        raise ValueError(
            f"{source_path}: tensor {name!r} byte range [{start}, {end}) "
            "is outside the file"
        )

    return TensorEntry(
        name=name,
        source_path=source_path,
        source_data_offset=source_data_offset,
        source_start=start,
        source_end=end,
        dtype=dtype,
        shape=list(shape),
    )


def compile_patterns(rule_index: int, field: str, values: object) -> list[re.Pattern[str]]:
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise TypeError(f"SOURCE_RULES[{rule_index}][{field!r}] must be a list of strings")
    try:
        return [re.compile(value) for value in values]
    except re.error as error:
        raise ValueError(
            f"invalid regex in SOURCE_RULES[{rule_index}][{field!r}]: {error}"
        ) from error


def collect_entries() -> list[TensorEntry]:
    if not SOURCE_RULES:
        raise ValueError("SOURCE_RULES must contain at least one source")

    entries: list[TensorEntry] = []
    selected_by_name: dict[str, TensorEntry] = {}
    header_cache: dict[Path, tuple[dict, int, int]] = {}

    for rule_index, rule in enumerate(SOURCE_RULES):
        if not isinstance(rule, dict) or "path" not in rule or "include" not in rule:
            raise TypeError(
                f"SOURCE_RULES[{rule_index}] must contain 'path' and 'include'"
            )

        source_path = Path(rule["path"])
        if not source_path.is_file():
            raise FileNotFoundError(f"source file does not exist: {source_path}")
        source_path = source_path.resolve()

        include = compile_patterns(rule_index, "include", rule["include"])
        exclude = compile_patterns(rule_index, "exclude", rule.get("exclude", []))
        if not include:
            raise ValueError(f"SOURCE_RULES[{rule_index}]['include'] must not be empty")

        if source_path not in header_cache:
            header_cache[source_path] = read_safetensors_header(source_path)
        header, data_offset, file_size = header_cache[source_path]

        matched = 0
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if not any(pattern.search(name) for pattern in include):
                continue
            if any(pattern.search(name) for pattern in exclude):
                continue

            entry = parse_tensor_entry(name, info, source_path, data_offset, file_size)
            previous = selected_by_name.get(name)
            if previous is not None:
                raise ValueError(
                    f"tensor {name!r} was selected more than once:\n"
                    f"  first:  {previous.source_path}\n"
                    f"  second: {source_path}"
                )
            selected_by_name[name] = entry
            print(f"entry {entry}")
            entries.append(entry)
            matched += 1

        print(f"Rule {rule_index}: selected {matched} tensors from {source_path}")
        if matched == 0:
            raise ValueError(
                f"SOURCE_RULES[{rule_index}] did not select any tensors; check its regexes"
            )

    if not entries:
        raise ValueError("no tensors were selected")
    return entries


def build_output_header(entries: list[TensorEntry]) -> tuple[bytes, int]:
    header: dict[str, object] = {}
    if OUTPUT_METADATA is not None:
        if not isinstance(OUTPUT_METADATA, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in OUTPUT_METADATA.items()
        ):
            raise TypeError("OUTPUT_METADATA must be None or a dict[str, str]")
        header["__metadata__"] = OUTPUT_METADATA

    output_offset = 0
    for entry in entries:
        header[entry.name] = {
            "dtype": entry.dtype,
            "shape": entry.shape,
            "data_offsets": [output_offset, output_offset + entry.size],
        }
        output_offset += entry.size

    header_bytes = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    header_bytes += b" " * (-len(header_bytes) % 8)
    return header_bytes, output_offset


def copy_tensor(source: BinaryIO, output: BinaryIO, entry: TensorEntry) -> None:
    source.seek(entry.source_data_offset + entry.source_start)
    remaining = entry.size
    while remaining:
        chunk = source.read(min(COPY_BUFFER_SIZE, remaining))
        if not chunk:
            raise OSError(
                f"unexpected end of file while copying {entry.name!r} "
                f"from {entry.source_path}"
            )
        output.write(chunk)
        remaining -= len(chunk)


def write_output(entries: list[TensorEntry]) -> None:
    if COPY_BUFFER_SIZE <= 0:
        raise ValueError("COPY_BUFFER_SIZE must be positive")

    output_path = OUTPUT_PATH.resolve()
    source_paths = {entry.source_path.resolve() for entry in entries}
    if output_path in source_paths:
        raise ValueError("OUTPUT_PATH must not be one of the source files")
    if output_path.exists() and not OVERWRITE_OUTPUT:
        raise FileExistsError(
            f"output already exists: {output_path}; set OVERWRITE_OUTPUT = True to replace it"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(output_path.name + ".partial")
    if partial_path.exists():
        raise FileExistsError(
            f"partial output already exists: {partial_path}; remove it before retrying"
        )

    header_bytes, tensor_bytes = build_output_header(entries)
    print(
        f"Writing {len(entries)} tensors ({format_bytes(tensor_bytes)}) to {output_path}"
    )

    current_source_path: Path | None = None
    current_source: BinaryIO | None = None
    copied = 0
    next_progress = PROGRESS_INTERVAL
    try:
        with partial_path.open("xb") as output:
            output.write(struct.pack("<Q", len(header_bytes)))
            output.write(header_bytes)

            try:
                for entry in entries:
                    if entry.source_path != current_source_path:
                        if current_source is not None:
                            current_source.close()
                        current_source = entry.source_path.open("rb")
                        current_source_path = entry.source_path

                    copy_tensor(current_source, output, entry)
                    copied += entry.size
                    if PROGRESS_INTERVAL > 0 and copied >= next_progress:
                        print(
                            f"  copied {format_bytes(copied)} / "
                            f"{format_bytes(tensor_bytes)}"
                        )
                        while next_progress <= copied:
                            next_progress += PROGRESS_INTERVAL
            finally:
                if current_source is not None:
                    current_source.close()

        if copied != tensor_bytes:
            raise OSError(f"copied {copied} tensor bytes, expected {tensor_bytes}")
        os.replace(partial_path, output_path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise

    print(f"Done: {output_path} ({format_bytes(output_path.stat().st_size)})")


def main() -> None:
    entries = collect_entries()
    write_output(entries)


if __name__ == "__main__":
    main()
