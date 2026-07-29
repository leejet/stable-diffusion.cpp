# Externally quantized safetensors

Some checkpoints ship weights that were already quantized by another toolchain,
storing the payload as raw bytes plus sidecar tensors that describe how to
dequantize it. Such files are loaded directly: each weight is repacked once, at
load time, into a native ggml type, so inference uses the ordinary kernels and no
backend-specific support is required.

| On-disk form | Repacked to | Lossless? |
| --- | --- | --- |
| ConvRot `convrot_w4a4` (int4) | `Q4_0` | yes, apart from F32→F16 on the scale |
| ComfyUI `int8_tensorwise`, ConvRot with `linear_dtype: int8` | `Q8_0` | yes, apart from F32→F16 on the scale |
| bitsandbytes NF4 | `F32` | yes (the codebook is non-uniform, so no 4-bit ggml type matches it) |

Because NF4 decodes to F32, pass `--type` to requantize it to something compact,
e.g. `--type q4_K`. The int4/int8 paths are already compact and need no `--type`.

## ConvRot

[ConvRot](https://arxiv.org/abs/2512.03673) is a 4-bit scheme for diffusion
transformers. It multiplies each weight by a fixed orthogonal rotation before
quantizing, which spreads outliers across the group and makes 4-bit weights far
more accurate.

The consequence for inference is that the stored weight is `W·H`, not `W`. Since

```
y = x·Wᵀ = x·(W'·H)ᵀ = (x·H)·W'ᵀ
```

the activation must be rotated by the same `H` before the matmul. Skipping this
does not degrade quality gracefully — it produces noise.

`H` is a block-diagonal regular-Hadamard transform over `convrot_groupsize` input
channels (256 by default; the group size must be a power of four). It is built
from the radix-4 core

```
M = [[ 1,  1,  1, -1],
     [ 1,  1, -1,  1],
     [ 1, -1,  1,  1],
     [-1,  1,  1,  1]]
```

Kronecker-composed `log4(groupsize)` times and scaled by `1/sqrt(groupsize)`.
That makes `H` symmetric as well as orthogonal, so no transpose bookkeeping is
needed. The rotation is applied as a single matmul against a shared constant
matrix, costing `groupsize / out_features` extra FLOPs — under 5% for a
6144-wide layer.

Note that activations stay in F32 here, so this is W4A16 rather than the paper's
W4A4: lower throughput than a dedicated 4-bit-activation kernel, but higher
accuracy.

A layer is only rotated when the checkpoint marks it and `in_features` divides
the group size; ConvRot leaves the remaining layers unrotated and omits the
marker, which is honoured.

**LoRA is not supported on ConvRot layers.** LoRA deltas are trained against the
unrotated weight and cannot be mixed into a rotated activation, so adapters are
skipped on those layers with a warning rather than applied incorrectly.

## Recognized layouts

ComfyUI-style, read from the safetensors `__metadata__` key
`_quantization_metadata`:

```json
{"format_version": "1.0",
 "layers": {"model.diffusion_model.blocks.0.attn.wq":
            {"format": "convrot_w4a4", "convrot_groupsize": 256}}}
```

with `<layer>.weight` holding the packed payload and `<layer>.weight_scale`
holding one F32 scale per output channel.

bitsandbytes NF4 is detected from its own sidecars —
`<layer>.weight.quant_state.bitsandbytes__nf4` (which carries the logical shape
the packed tensor has lost), `.absmax`, `.quant_map`, and the nested pair used
when the absmax is itself double-quantized.

Byte tensors that no configuration claims are dropped, so leftover metadata blobs
do not reach the model.

## Example

Krea-2 Turbo quantized to ConvRot int4:

```
sd-cli --diffusion-model Krea2_INT8.safetensors \
       --llm Qwen3-VL-4B-Instruct-Q4_K_M.gguf \
       --vae wan_2.1_vae.safetensors \
       -p "a lovely cat holding a sign says 'krea2.cpp'" \
       --diffusion-fa -v
```

At ~4.5 bits per weight the diffusion model is roughly a third the size of Q8_0.
If it still does not fit, cap what stays resident with `-ngl` (see
[backend.md](backend.md)).
