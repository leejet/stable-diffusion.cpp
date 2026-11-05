# INT8 Convrot Safetensors

sd.cpp can load and execute ComfyUI `int8_tensorwise` safetensors with `convrot` metadata directly. The stored INT8 weights are not converted to another weight type at load time.

This uses sd.cpp's INT8 tensorwise/convrot operators through the patched GGML
extension API. Schemas and graph builders live in sd.cpp, alongside the
backend kernels and fusion rules; GGML supplies generic dispatch and workspace
services. See [operator extensions](ggml_extensions.md) for the implementation
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

The convrot group size must be a power of four and must divide the input feature dimension. The commonly used configuration is H256, with `convrot_groupsize` set to `256`.

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
- NVIDIA CUDA devices with compute capability 7.5 or newer use the native accelerated path. For H256, CUDA fuses the rotation, row-wise maximum reduction, and activation quantization. Eligible shapes use the CUTLASS INT8 implementation when enabled; other supported shapes use cuBLAS for I8 x I8 to I32 GEMM and a CUDA kernel for scale restoration and bias addition.
- AMD HIP devices in the CDNA, RDNA3 (including RDNA3.5), and RDNA4 families use the same INT8/H256 kernels with hipBLAS for I8 x I8 to I32 GEMM. Other AMD architectures fall back to CPU for INT8 matrix multiplication.
- MUSA shares the CUDA-style H256 activation kernels. INT8 GEMM falls back to CPU because the MUSA backend does not advertise INT8 BLAS support.
- Vulkan provides H256 activation quantization and INT8 matrix multiplication when the build and device support accelerated packed INT8 dot products. Unsupported configurations and GPU backends without these kernels use the backend scheduler to fall back to CPU.

LoRA adapters are applied at runtime without modifying the INT8 weights. The INT8 convrot path computes the base linear output, while LoRA, LoHa, LoKr, and raw weight-difference adapters compute their output corrections from the original, unrotated activation and add them to the base output. `--lora-apply-mode auto` selects this path for models containing INT8 tensorwise weights. If `immediately` is requested, sd.cpp falls back to runtime application because merging an adapter would require dequantizing and rotating its weight update, then recalculating the per-row scales and requantizing the result.

The shared CUDA/HIP/MUSA convrot activation paths support H64 and H256. Vulkan also supports H64 activation quantization; its INT8 matrix multiplication path currently requires H256.

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
  metadata must specify `group_size: 16`. CPU evaluates the floating-point
  codebook; GPU kernels requantize it to INT8 with `amax / 127`, introducing a
  small additional rounding error.

Both formats imply convrot rotation even without a `convrot` flag; missing
`convrot_groupsize` defaults to 256.

Both formats reuse the convrot activation cache when the input scale is `1` and
produce F32 outputs with optional bias. CPU provides a portable implementation.
CUDA/HIP/MUSA use shared packed-weight kernels without requiring INT8 BLAS.
With CUTLASS enabled, NVIDIA
SM80 and newer devices use mixed-input S8 x S4 GEMM for eligible W4A4 shapes,
reusing the INT8 scale epilogue without expanding the full weight tensor. W4A8
uses CUTLASS integer MMA per 16-value group with FP32 scale restoration.
Other CUDA shapes and builds retain the packed byte-dot path. Set
`GGML_CUDA_W4_FORCE_DP4A=1` to compare against that path. Supported AMD targets
use packed byte-dot instructions, with portable byte arithmetic for other targets.
Vulkan requires integer dot product support and at least 13312 bytes of shared
memory for W4A4 or 16452 bytes for W4A8. GPU W4 kernels require the logical input
width to be divisible by 64; H64 and H256 are supported. Other shapes use CPU
fallback. HIP W4A8 selects wave64 WMMA by the active compilation target's
intrinsic support, preferring the GFX12 fragment layout over the GFX11 layout.
There is no GPU architecture allowlist: newer targets supporting either
intrinsic automatically use its matching layout. Selection happens per device
image, including multi-architecture builds; targets supporting neither intrinsic
retain the integer-dot fallback. W4A4 retains integer dot products on HIP.

LoRA application remains at runtime, using the original unrotated activations
and logical weight dimensions. Packed weights and their codebook/scale tensors
are excluded from weight type conversion. `SD_USE_UPSTREAM_GGML` builds reject
these formats because they require the application extension API.

## Example

ComfyUI INT8 convrot safetensors can be passed to `--diffusion-model` without conversion:

```powershell
.\bin\Release\sd-cli.exe --diffusion-model ..\models\diffusion_models\krea2_turbo_int8_convrot.safetensors --llm ..\models\text_encoders\Qwen3-VL-4B-Instruct-Q4_K_M.gguf --vae ..\models\vae\wan_2.1_vae.safetensors -p "a lovely cat holding a sign says 'krea2.cpp'" --steps 8 --cfg-scale 1 -v --offload-to-cpu
```
