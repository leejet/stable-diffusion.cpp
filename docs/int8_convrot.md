# INT8 Convrot Safetensors

sd.cpp can load and execute ComfyUI `int8_tensorwise` safetensors with `convrot` metadata directly. The stored INT8 weights are not converted to another weight type at load time.

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

The packed runtime activation tensor contains the I8 activation rows and their floating-point row scales. Linear layers that share the same input and convrot group size reuse this packed tensor, avoiding repeated rotation and activation quantization within the graph.

## Backend support

- CPU provides the portable regular Hadamard, activation quantization, INT8 matrix multiplication, and scale restoration implementations.
- NVIDIA CUDA devices with compute capability 7.5 or newer use the native accelerated path. For H256, CUDA fuses the rotation, row-wise maximum reduction, and activation quantization. It uses cuBLAS for I8 x I8 to I32 GEMM and a CUDA kernel for scale restoration and bias addition.
- Vulkan and other GPU backends do not currently have dedicated INT8 convrot kernels. They use the backend scheduler to fall back to CPU, which is expected to be substantially slower than the CUDA path.

LoRA adapters are applied at runtime without modifying the INT8 weights. The INT8 convrot path computes the base linear output, while LoRA, LoHa, LoKr, and raw weight-difference adapters compute their output corrections from the original, unrotated activation and add them to the base output. `--lora-apply-mode auto` selects this path for models containing INT8 tensorwise weights. If `immediately` is requested, sd.cpp falls back to runtime application because merging an adapter would require dequantizing and rotating its weight update, then recalculating the per-row scales and requantizing the result.

The dedicated CUDA convrot activation path currently requires a group size of `256`; other supported group sizes use CPU execution.

## Example

ComfyUI INT8 convrot safetensors can be passed to `--diffusion-model` without conversion:

```powershell
.\bin\Release\sd-cli.exe --diffusion-model ..\models\diffusion_models\krea2_turbo_int8_convrot.safetensors --llm ..\models\text_encoders\Qwen3-VL-4B-Instruct-Q4_K_M.gguf --vae ..\models\vae\wan_2.1_vae.safetensors -p "a lovely cat holding a sign says 'krea2.cpp'" --steps 8 --cfg-scale 1 --diffusion-fa -v --offload-to-cpu
```
