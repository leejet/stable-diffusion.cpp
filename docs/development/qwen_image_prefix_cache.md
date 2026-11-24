# Qwen Image 2.1 prefix cache

For cache options and memory estimates, see the
[Qwen Image 2.1 usage guide](../models/qwen_image_2.1.md#prefix-cache).

## Cache behavior

By default, the first denoising call for each fixed condition saves the text and reference-image keys and values from every transformer layer. Later calls only compute the target-image tokens. Positive and negative conditions use separate caches, which are released when sampling ends.

Set `qwen_image_2_1_prefix_cache_type` in `--model-args` to `auto` or a type name using the same parser and case-sensitive names as `--type`:

- `auto` (default): use FP16 only when Flash Attention is enabled, Sage Attention is disabled, the attention scale is unchanged, and every attention operation in the cache-writing or cache-reading graph selects Flash Attention after backend support checks. If an operation falls back, rebuild the prefix in FP32 before executing and keep FP32 for the rest of that sampling run.
- `f32`: always store FP32 keys and values.
- `f16`: always store FP16 keys and values, including with ordinary attention or custom attention scaling. This saves cache memory but can introduce additional rounding error.
- Other types, such as `bf16`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q4_K`, `q6_K`, `iq4_nl`, and `iq4_xs`: use the requested storage type if the ggml build provides runtime conversion to and from FP32. Quantization is lossy and must be selected explicitly; `auto` never selects a quantized type.

Cache data is packed into contiguous rows of `hidden_size` elements before conversion, so 256-element quantization blocks work with the model's 128-element attention heads without padding. The type's block size must divide `hidden_size`. Unknown types, types lacking runtime conversion (for example `q8_1` and several IQ formats), and incompatible block sizes are ignored with a warning, leaving the previous setting or the default `auto` unchanged.

For example, use `--model-args qwen_image_2_1_prefix_cache_type=q8_0` to enable 8-bit cache storage. Cached keys and values are converted back to the attention input type before concatenating with the current target tokens. This reduces persistent cache memory; attention working buffers still use floating-point values, and conversion adds work on each step. Backends without the required conversion operations use the existing CPU fallback.

For the default 32-layer model, a prefix of 4096 tokens takes approximately the following memory per condition, excluding weights, working buffers, and allocation overhead:

| Cache type | Memory |
| --- | ---: |
| `f32` | 4 GiB |
| `f16` | 2 GiB |
| `q8_0` | 1.0625 GiB |
| `q4_0` | 0.5625 GiB |

The runner accounts for the cache when checking the memory budget. If a cached execution runs out of memory, it releases the prefix caches, disables caching for the rest of that sampling run, and retries the full sequence once. Per-step conditioning extensions currently use the full-sequence path.

Disable this optimization with `--model-args qwen_image_2_1_prefix_cache=false`. It reuses step-independent activations; numerical results can still differ slightly because the matrix sizes change.
