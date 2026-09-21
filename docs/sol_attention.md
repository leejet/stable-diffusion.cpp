# Sol-Attn

`--sol-attn` enables native CUDA Sol-Attn in the diffusion model, including the
high-noise diffusion model when present. It uses the shared attention dispatcher
without classifying tokens as text, images, or video. Python, PyTorch, Triton,
and CuTe DSL are not needed to build or run it.

This implementation follows the diagonal-threshold algorithm in
[NVlabs/Sana's Sol-Attn](https://github.com/NVlabs/Sana/tree/sol-engine/techniques/sparse_backends/sol_attn).
It summarizes 64-token KV blocks, selects exact blocks using proxy scores and
an online threshold, and approximates the remaining blocks using their K means
and V sums. Adjacent blocks remain exact. Both contributions share an online
softmax normalizer. Q/K/V and probability tiles use BF16 Tensor Cores with FP32
accumulation; the BF16 result is returned through the existing FP32 interface.

## Build

Use patched GGML, CUDA Toolkit 12.0 or newer, and an NVIDIA GPU with compute
capability 8.0 or newer. Compile kernels for the target GPU:

```sh
cmake -S . -B build -DSD_CUDA=ON -DSD_USE_UPSTREAM_GGML=OFF
cmake --build build --config Release
```

The feature is compiled with the CUDA backend; no separate build option is
required. Upstream GGML and non-CUDA backends do not support it. A system GGML
must provide the matching patched API and CUDA implementation. Tensor-parallel
row splitting is not supported; layer splitting requires supported devices.

## Use

Add `--sol-attn` to an existing generation command:

```sh
sd-cli ... --sol-attn
sd-cli ... --sol-attn --sol-attn-tau 1.0
```

The default threshold coefficient is `1.0`. Larger coefficients select fewer
blocks for exact attention. The coefficient must be finite; zero does not mean
dense attention. Omit `--sol-attn` to disable the feature.

The native kernel supports unmasked, noncausal attention with head dimension
128, equal Q/K/V sequence lengths and head counts, and multiple batches. Other
attention operations fall back to FlashAttention when available, then ordinary
attention. Existing attention scaling overrides remain effective. `--fa` and
`--diffusion-fa` may be used together with Sol-Attn; `--sage-attn` is mutually
exclusive. Text encoders and VAEs retain their existing attention selection.

Initialization reports an error if the requested diffusion backend cannot run
Sol-Attn. Graph logs report the number of Sol-Attn and FlashAttention nodes and
warn when no Sol-Attn nodes are selected. CUDA execution errors are not silently
converted into dense attention.

This is approximate attention. Validate quality and end-to-end speed with the
same prompt, seed, dimensions, frame count, and sampling settings. Include
packing, preprocessing, offload, and decode time in comparisons. Short sequences
may not benefit. Upstream combined pipeline speedups are not measurements of
this native kernel. Exact-covariance thresholds, text sinks, Morton ordering,
and step/layer schedules are not implemented.

## Validation

On an RTX 4090 with CUDA 12.4, Wan 2.1 T2V 1.3B was tested at 832x480,
33 frames, 20 Euler steps, seed 42, CFG 6, and flow shift 3, using the prompt
`a lovely cat` and the same negative prompt for every run:

| Attention | Sampling time | Total process time |
| --- | ---: | ---: |
| FlashAttention | 45.73 s | 74.63 s |
| Sol-Attn, tau 1 | 37.17 s | 66.20 s |
| Sol-Attn, tau 0 | 40.66 s | 68.50 s |

These are single-run measurements. The graph selected 30 Sol-Attn nodes and
30 FlashAttention nodes. At tau 1, sampled video frames showed washed-out
colors and reduced detail. Tau 0 improved clarity in this example, but still
changed the composition. Neither setting guarantees the baseline's quality.
For this Wan command, `--sol-attn --sol-attn-tau 0` is a more conservative
starting point. In the one-frame case, tau 1 increased warm sampling time from
0.140 to 0.148 seconds per step.

Validation also covered 15 numerical reference cases, 11 layout/scaling/fallback
cases, CUDA memory checking, and 36 existing SageAttention regression cases.
CLI and server CUDA builds and the upstream GGML CPU library build passed.
Other GPU architectures, multi-GPU execution, and other models have not been
tested.

## Library API

Configure Sol-Attn in `sd_ctx_params_t` before creating the context:

```cpp
sd_ctx_params_t params;
sd_ctx_params_init(&params);
// Set model paths and other context options here.
params.sol_attn = true;
params.sol_attn_tau = 1.0f;
sd_ctx_t* ctx = new_sd_ctx(&params);
```

`sd_ctx_params_init` defaults `sol_attn` to false and `sol_attn_tau` to 1.0.
`new_sd_ctx` returns null for a nonfinite threshold, unavailable requested
backends, or a conflict with SageAttention. The context owns a copy of these
settings; changing the input structure after creation does not reconfigure it.
Applications must be rebuilt against the updated `sd_ctx_params_t` definition.
