# SageAttention CUDA device code

Adapted from [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention),
revision `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`, under Apache-2.0 (see LICENSE).

`attn.cuh` contains the device kernel from
`csrc/qattn/qk_int_sv_f16_cuda_sm80.cu`; `attn_fp8.cuh` adapts
`csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh`. The remaining headers come from `csrc/`
and `csrc/qattn/attn_utils.cuh`. PyTorch includes and host wrappers were removed;
include paths and guards were adjusted, device helpers were placed in the
`sageattention` namespace, and `div_ceil` was renamed to avoid macro collisions.
Unused `<cuda/pipeline>` includes were removed so multi-architecture builds can
also compile targets below SM70; the backend rejects SageAttention below SM80.
Predicated loads now zero unused FP16 V lanes: masked probabilities cannot
suppress NaNs left in shared memory by previous kernels.

The sd.cpp adapter in `../sage_attention.cu` supplies FP32 K smoothing and native CUDA
per-thread INT8 quantization, matching the token grouping in upstream
`sageattention/triton/quant_per_thread.py`. V is quantized per channel and
transposed, padded with zeros, and permuted according to
`csrc/fused/fused.cu`. Its FP8 range is 448 for SageAttention2 and 2.25 for
SageAttention2++. Zero channels use a positive scale to avoid division by zero.
SageAttention2 uses an FP32 instruction buffer; 2++ uses an FP16 instruction
buffer added to FP32 accumulators. The SM80 compatibility path retains per-warp
INT8 Q/K and FP16 V with FP32 accumulation.

The adapter requests workspace and the current stream through public ggml extension APIs; it has no Python, PyTorch,
or Triton dependency. The public output is converted back to FP32.
