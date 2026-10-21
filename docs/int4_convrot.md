# INT4 Convrot Safetensors

sd.cpp can load and execute ComfyUI packed int4 convrot safetensors directly. Two formats are supported:

- `convrot_w4a4`: int4 weights with int4-class activations.
- `asym_w4a8_int8`: int4 codebook weights with int8 activations.

The packed int4 weights are not converted to another weight type at load time. Weights consume roughly half a byte per parameter, and loading performs no reblocking step and writes no cache files.

This requires the packed int4 convrot extensions in the patched GGML.
Builds with `SD_USE_UPSTREAM_GGML=ON` reject these files during loading.

## Checkpoint format

Each quantized linear module contains a packed weight tensor and a U8 quantization configuration blob:

- `<module>.weight`: a byte-packed I8 tensor holding two 4-bit values per byte, `K / 2` bytes per output row.
- `<module>.comfy_quant`: a U8 tensor containing the JSON quantization configuration.

A `convrot_w4a4` module additionally contains:

- `<module>.weight_scale`: one floating-point scale per output row. A two-dimensional `[out_features, 1]` representation is normalized to one dimension while loading.

An `asym_w4a8_int8` module additionally contains:

- `<module>.weight_codebook`: 16 floating-point codebook entries shared by the module.
- `<module>.weight_s_channel`: one floating-point scale per output row.
- `<module>.weight_s_rel`: one F8_E4M3 relative scale per 16 input features per output row.

The configuration blob for `convrot_w4a4` has this form:

```json
{
  "format": "convrot_w4a4",
  "convrot_groupsize": 64
}
```

For `asym_w4a8_int8` the blob also carries the codebook quantization group size:

```json
{
  "format": "asym_w4a8_int8",
  "group_size": 16,
  "convrot_groupsize": 64
}
```

The convrot group size must be a power of four, must divide the input feature dimension, and must be `64` or `256`. The input feature dimension must additionally be a multiple of `32`.

## How int4 convrot works

The rotation follows the same scheme as INT8 convrot: a normalized regular Hadamard transform is applied offline to the weights and at runtime to the activations, preserving the linear operation while spreading outliers across each feature group. See [INT8 Convrot Safetensors](./int8_convrot.md) for the rotation details.

For an original floating-point linear layer `Y = X W^T + b`, the checkpoint stores packed nibbles of the rotated weights. At runtime sd.cpp rotates and quantizes the activations once per input, reusing the packed result across all linear layers that share the same input and group size.

### convrot_w4a4

The rotated weights are quantized to signed 4-bit values in `[-7, 7]` with one scale per output row. At runtime the activations are quantized to int8, which keeps the dot product on an exact integer grid: the 4-bit weights sign-extend to int8 without error, the accumulation is exact in 32-bit integers, and the output is reconstructed as

```text
Y[r, o] ~= A[r, o] * s_x[r] * s_w[o] + b[o]
```

This variant introduces no additional weight error beyond the int4 quantization itself.

### asym_w4a8_int8

Each nibble indexes a 16-entry per-module codebook. The effective weight is

```text
W_rot[o, i] = codebook[code[o, i]] * s_channel[o] * s_rel[o, i / 16]
```

There is no integer grid for the codebook, so the kernels requantize the codebook to int8 when a block starts and compute the dot product with int8 integer instructions. The per-group relative scales are applied as a floating-point correction after each group. The result stays within a small tolerance of the floating-point reference instead of matching it bit for bit.

## Backend support

Both int4 formats run on CPU, NVIDIA CUDA, HIP (ROCm), and Vulkan, with convrot group sizes `64` and `256`.

On HIP, plain `int8_tensorwise` models currently fall back to CPU execution; this is tracked by the separate INT8 HIP/BLAS changes and does not affect int4 models.

LoRA adapters are applied at runtime without modifying the packed int4 weights. The int4 convrot path computes the base linear output, while LoRA, LoHa, LoKr, and raw weight-difference adapters compute their corrections from the original, unrotated activation and add them to the base output. `--lora-apply-mode auto` selects this path for models containing packed int4 weights.

## Example

ComfyUI int4 convrot safetensors can be passed to `--diffusion-model` without conversion:

```powershell
.\bin\Release\sd-cli.exe --diffusion-model ..\models\diffusion_models\flux-2-klein-4b_int4_convrot.safetensors --vae ..\models\vae\flux2-klein-vae.safetensors --llm ..\models\text_encoders\qwen_3_4b_int8_convrot.safetensors -p "a lovely cat holding a sign says 'sd.cpp'" --steps 8 --cfg-scale 1 --diffusion-fa -v --vae-tiling
```
