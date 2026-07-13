#!/usr/bin/env python3
"""Convert an Ultralytics YOLOv8 detection checkpoint for sd.cpp ADetailer."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert an Ultralytics YOLOv8 detection .pt checkpoint to safetensors."
    )
    parser.add_argument("input", type=Path, help="input YOLOv8 detection checkpoint")
    parser.add_argument("output", type=Path, help="output safetensors path")
    parser.add_argument(
        "--input-size", type=int, default=640, help="detector input size metadata (default: 640)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.input_size < 32 or args.input_size % 32 != 0:
        raise ValueError("--input-size must be a positive multiple of 32")
    if args.output.suffix.lower() != ".safetensors":
        raise ValueError("output path must use the .safetensors extension")

    try:
        import torch
        from safetensors.torch import save_file
        from ultralytics import YOLO
        from ultralytics.nn.modules.head import Detect
    except ImportError as exc:
        raise SystemExit("conversion requires ultralytics, torch, and safetensors") from exc

    torch_load = torch.load

    def load_trusted_checkpoint(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return torch_load(*load_args, **load_kwargs)

    torch.load = load_trusted_checkpoint
    try:
        yolo = YOLO(str(args.input))
    finally:
        torch.load = torch_load
    network = yolo.model
    if not isinstance(network.model[-1], Detect) or network.model[-1].__class__.__name__ != "Detect":
        raise ValueError("only YOLOv8 detection checkpoints are supported; segmentation is not yet supported")

    network.eval()
    network.fuse()
    state_dict = network.state_dict()
    required = {
        "model.0.conv.weight",
        "model.22.cv2.0.2.weight",
        "model.22.cv3.0.2.weight",
    }
    missing = sorted(required.difference(state_dict))
    if missing:
        raise ValueError(f"checkpoint does not match the supported YOLOv8 layout; missing {missing}")

    tensors = {}
    for name, tensor in state_dict.items():
        if not name.startswith("model.") or ".bn." in name or name.endswith("dfl.conv.weight"):
            continue
        if not (name.endswith(".weight") or name.endswith(".bias")):
            continue
        dtype = torch.float16 if name.endswith(".weight") else torch.float32
        tensors[name] = tensor.detach().to(device="cpu", dtype=dtype).contiguous()

    metadata = {
        "format": "pt",
        "yolov8.variant": "detect",
        "yolov8.input_size": str(args.input_size),
        "yolov8.num_classes": str(int(network.model[-1].nc)),
        "yolov8.reg_max": str(int(network.model[-1].reg_max)),
        "yolov8.names": json.dumps(yolo.names, ensure_ascii=False),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output), metadata=metadata)
    print(f"wrote {args.output}: {len(tensors)} tensors")


if __name__ == "__main__":
    main()
