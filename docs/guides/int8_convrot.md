# INT8 Convrot Safetensors

DGL versions of the generation examples are linked below. Edit each script's
`var` block and follow the [DGL run instructions](../dgl/README.md).


sd.cpp can load and execute ComfyUI `int8_tensorwise` safetensors with `convrot` metadata directly. The stored INT8 weights are not converted to another weight type at load time.

This uses sd.cpp's INT8 tensorwise/convrot operators through the patched GGML
extension API. Schemas and graph builders live in sd.cpp, alongside the
backend kernels and fusion rules; GGML supplies generic dispatch and workspace
services. See [operator extensions](../development/ggml_extensions.md) for the implementation
and the preserved INT8 API entry points.
Builds with `SD_USE_UPSTREAM_GGML=ON` reject these files during loading.

## Checkpoint format

Each quantized linear module contains the following tensors:

- `<module>.weight`: an I8 weight matrix.
- `<module>.weight_scale`: one floating-point scale for each output row. ComfyUI's two-dimensional `[out_features, 1]` representation is normalized to a one-dimensional tensor while loading.
- `<module>.comfy_quant`: a U8 tensor containing the JSON quantization configuration.

A supported configuration has this form:

```json
{
  "format": "int8_tensorwise",
  "convrot": true,
  "convrot_groupsize": 256
}
```

When `convrot` is enabled, `convrot_groupsize` defaults to `256` (H256). The group size must be a power of four no greater than `256` and must divide the input feature dimension.

## How INT8 convrot works

Convrot combines an offline rotation of the weights with the same rotation of the activations at runtime. The rotation uses a normalized regular Hadamard matrix constructed recursively from

```text
     [ 1  1  1 -1 ]
     [ 1  1 -1  1 ]
H4 = [ 1 -1  1  1 ] / 2
     [-1  1  1  1 ]
```

For a group size `G = 4^n`, the transform is the normalized Kronecker power of `H4`. It is applied independently to every contiguous group of `G` input features. The resulting block-diagonal rotation matrix `R` is orthogonal and symmetric, so `R R^T = I`.

For an original floating-point linear layer

```text
Y = X W^T + b
```

the checkpoint stores a rotated weight matrix `W_rot = W R`, quantized per output row. At runtime sd.cpp computes `X_rot = X R`. Ignoring quantization error,

```text
X_rot W_rot^T = X R (W R)^T = X R R^T W^T = X W^T
```

The rotation therefore preserves the linear operation. Its purpose is to spread isolated large values across each feature group, reducing the effect of outliers on tensorwise INT8 quantization.

### Weight quantization

The rotated weights are quantized offline with one scale per output row:

```text
s_w[o]      = max_i(abs(W_rot[o, i])) / 127
Q_w[o, i]   = clamp(round(W_rot[o, i] / s_w[o]), -127, 127)
```

`Q_w` is stored in `<module>.weight`, and `s_w` is stored in `<module>.weight_scale`.

### Runtime activation quantization

For every activation row, sd.cpp applies the group-wise Hadamard rotation and then calculates one dynamic scale across the entire rotated row:

```text
s_x[r]      = max_i(abs(X_rot[r, i])) / 127
Q_x[r, i]   = clamp(round(X_rot[r, i] / s_x[r]), -127, 127)
```

The matrix multiplication accumulates into signed 32-bit integers:

```text
A[r, o] = sum_i(Q_x[r, i] * Q_w[o, i])
```

The floating-point output is reconstructed as

```text
Y[r, o] ~= A[r, o] * s_x[r] * s_w[o] + b[o]
```

The packed runtime activation tensor contains the I8 activation rows and their floating-point row scales. Linear layers with an input scale of `1` that share the same input and convrot group size reuse this packed tensor, avoiding repeated rotation and activation quantization within the graph. For other input scales, activations are scaled before explicit convrot quantization, and the linear output is unscaled before adding bias.

## Backend support

- CPU provides the portable regular Hadamard, activation quantization, INT8 matrix multiplication, and scale restoration implementations.
- Metal supports INT8 tensorwise and convrot on Apple silicon (Apple GPU family 7 or newer). It keeps weights in INT8 and rotates and quantizes activations on the GPU. On macOS/iOS 26 or newer, it probes Metal TensorOps `matmul2d` with INT8 inputs and INT32 cooperative results. It combines 1024-product partial sums using wrapping integer addition to preserve exact accumulation on older GPUs, then applies scales and bias in registers. Older systems or unavailable TensorOps pipelines use SIMD-group matrix multiplication: INT8 tiles are converted to FP16, and FP32 partial sums cover at most 256 products before being combined as integers, preserving the I32 accumulation contract. Convrot groups 1, 4, 16, 64, and 256 are supported. These kernels require the patched GGML Metal extension services and are enabled by `SD_METAL=ON`.
- CUDA uses CUTLASS for eligible shapes when enabled, and cuBLAS for other eligible shapes on Turing and newer. HIP uses hipBLAS or hipBLASLt on supported targets, including RDNA2, RDNA3, RDNA3.5, RDNA4 (`gfx120x`), and CDNA.
- CUDA, HIP, and MUSA share a tiled INT8 dot-product fallback. It handles devices without INT8 BLAS and projections whose output width is not divisible by four. It uses packed byte-dot instructions where available.
- Vulkan supports unrotated rowwise, H64, and H256 activation quantization and packed-activation INT8 matrix multiplication with input widths divisible by 64. It requires integer dot product support and uses cooperative matrices when available. Unsupported configurations use the scheduler's CPU fallback.

`--lora-apply-mode auto` selects immediate merging for INT8 tensorwise models
on the compute backend without row splitting. Use `--lora-apply-mode at_runtime`
to compute each adapter's correction from the original activation and add it to
the INT8 linear output.

Immediate mode merges an adapter into INT8 tensorwise weights on first use.
For convrot weights, sd.cpp rotates the adapter update before adding it to the
dequantized weights. It then recalculates the weight
scales and requantizes the result to INT8. The merge runs on the selected
backend when it supports the operation, with ggml's usual CPU fallback. The
additional quantization step can change the output compared with runtime
application.

The shared CUDA/HIP/MUSA convrot activation paths support H64 and H256.

## Packed INT4 convrot

ComfyUI `convrot_w4a4` and `asym_w4a8_int8` safetensors load directly, including
checkpoints that mix packed INT4, INT8 tensorwise, and floating-point layers.
Packed weights retain their checkpoint byte layout; they are not expanded or
converted to GGUF Q4 blocks at load time.

- `convrot_w4a4`: signed 4-bit weights, low nibble first, with F32 per-output
  `weight_scale`. Activations use the same INT8 convrot quantization as above;
  the format name does not select a native INT4 activation kernel.
- `asym_w4a8_int8`: 4-bit indices into `weight_codebook` (16 F32 values), F32
  `weight_s_channel`, and F8_E4M3 `weight_s_rel` per 16 input features. The
  weight group size must be 16 and defaults to 16 when omitted.

Both formats imply convrot rotation even without a `convrot` flag; missing
`convrot_groupsize` defaults to 256.

Both formats reuse the convrot activation cache when the input scale is `1` and
produce F32 outputs with optional bias.

W4A8 multiplies the codebook value by the group's relative scale, rounds to nearest even,
and clamps to `[-127, 127]`. The linear operation decodes weights into temporary workspace,
then runs the same INT8 GEMM as W8A8 with `weight_s_channel` as the per-output
scale. Eligible large CUDA projections decode slices and write GEMM results
directly into the corresponding output columns. CPU, CUDA/HIP/MUSA, Vulkan,
and Metal provide weight decoders. The checkpoint's packed weights stay in
their original layout.

If the decoded W4A8 weights exceed the Vulkan device's `maxStorageBufferRange`,
the linear operation decodes slices of output channels into shared workspace
and writes each GEMM result directly to the corresponding output columns. An
`8192 × 24576` projection needs 192 MiB after decoding and can run on a device
with a 128 MiB storage-buffer range. The packed weights, activations, scales,
and FP32 output must still fit their respective bindings. Projections that
exceed those limits fall back to CPU.

W4A4 uses packed-weight kernels. Eligible CUDA SM80+ shapes can use CUTLASS mixed
S8 x S4 GEMM. Vulkan uses cooperative matrices where supported and requires a
logical input width divisible by 64. Use `GGML_CUDA_W4_FORCE_DP4A=1` or
`GGML_VK_W4_FORCE_DP4A=1` to select packed byte dots for W4A4 on the corresponding
backend.

LoRA application remains at runtime, using the original unrotated activations
and logical weight dimensions. Packed weights and their codebook/scale tensors
are excluded from weight type conversion. `SD_USE_UPSTREAM_GGML` builds reject
these formats because they require the application extension API.

## Runtime requantization to INT8

Use `--linear-type i8` to run eligible Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K,
Q4_K, Q5_K, Q6_K, and Q8_K linear layers through INT8 GEMM on CUDA, HIP, MUSA,
and Vulkan. A bare mode applies to all models in the context, including
diffusion models and text encoders loaded with `--llm`. The default is `auto`,
which currently behaves like `raw`. With
`--linear-type raw`, each layer uses its original execution path. Native INT8
weights use their INT8 path in all three modes. For floating-point checkpoints,
use `--type q4_0` (or another supported weight type) to quantize eligible weights
at load time.

Use module assignments to apply a mode selectively. For example,
`--linear-type dit=i8` selects runtime conversion for the diffusion module.
The text encoder and other modules use `auto`. Use `--linear-type i8,te=raw` to
enable conversion globally while keeping the text encoders on their original
paths. Module names and aliases are shared with [backend selection](backend.md).
Module assignments override the default regardless of their order. Within
the same string, the last value for a module or the default wins.

Runtime conversion uses H256 for weights and activations. Eligible linear
layers have an input width divisible by 256. Vulkan also requires integer
dot products and compatible storage buffer sizes and alignment. The C API option is
`linear_type`. In DGL, set the mode on a model loader, for example
`load_dit(path, linear_type = "i8")`. To override loader settings by module,
use `sd_dgl_ctx_set(ctx, "linear_type", "dit=i8")` or pass
`--linear-type dit=i8` when running the script from the CLI.

At runtime, sd.cpp dequantizes the weight blocks, rotates the weights, and
requantizes them to INT8 with one scale per output channel. It applies the
same rotation to the activations before quantizing each input row. The
converted matrix uses temporary GPU memory and feeds the standard INT8 kernels
and their fusions. The checkpoint file stays in its original format.
Requantization changes numerical results, and conversion overhead depends on
matrix shape. Row-split models and unsupported layers use their original
execution paths.

## Example

ComfyUI INT8 convrot safetensors can be passed to `--diffusion-model` without conversion:

```powershell
.\bin\Release\sd-cli.exe --diffusion-model ..\models\diffusion_models\krea2_turbo_int8_convrot.safetensors --llm ..\models\text_encoders\Qwen3-VL-4B-Instruct-Q4_K_M.gguf --vae ..\models\vae\wan_2.1_vae.safetensors -p "a lovely cat holding a sign says 'krea2.cpp'" --steps 8 --cfg-scale 1 -v --offload-to-cpu
```
