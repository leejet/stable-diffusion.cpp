# LTX-2 parity tests

Block-by-block numerical parity between the C++/GGML LTX-2 port and the reference
PyTorch implementation in `/devel/tools/diffusion/LTX-2/packages/ltx-core/`.

## How it works

`dump_reference.py` instantiates a **tiny, deterministic** LTX-2 transformer with fixed
random weights (seed=0) and runs a forward pass on a fixed input. It writes:

- `/tmp/ltx_ref/manifest.json` — catalogue of every dumped tensor (name, shape, dtype, offset)
- `/tmp/ltx_ref/state_dict.safetensors` — all model weights in a standard format
- `/tmp/ltx_ref/tensors/*.bin` — each intermediate tensor as raw float32 bytes

The "tiny" model is small enough (2 layers, inner_dim=128) to run in milliseconds on CPU
and make it easy to dump every intermediate without filling the disk. That scope is
deliberate: parity at tiny dims transfers to full-size models because every block is
tested exhaustively.

A matching C++ test (to be written) loads `state_dict.safetensors`, replays the same
input, and diffs every intermediate tensor against the reference. Tolerances:
- F32: 1e-5 absolute, 1e-4 relative
- BF16/FP16 C++ path: 1e-2 absolute, 5e-3 relative

## Run

```bash
/home/ilintar/venv/bin/python dump_reference.py
```

## What's NOT covered (yet)

- **Gemma 3 text encoder** — needs a Gemma 3 checkpoint. Deferred; we dump a synthetic
  `context` tensor (random but fixed) as a placeholder.
- **VAE** — separate dumper planned once the C++ VAE is building.
- **Sampler loop** — a separate script, not this one. This one tests a single forward call.
