## Quantization

You can specify the model weight type using the `--type` parameter. The weights are automatically converted when loading the model.

- `f16` for 16-bit floating-point
- `f32` for 32-bit floating-point
- `q8_0` for 8-bit integer quantization
- `q5_0` or `q5_1` for 5-bit integer quantization
- `q4_0` or `q4_1` for 4-bit integer quantization


### Memory Requirements of Stable Diffusion 1.x

| precision | f32  | f16  |q8_0  |q5_0  |q5_1  |q4_0  |q4_1  |
| ----         | ----  |----  |----  |----  |----  |----  |----  |
|  **Memory** (txt2img - 512 x 512) | ~2.8G | ~2.3G | ~2.1G | ~2.0G | ~2.0G | ~2.0G | ~2.0G |
|  **Memory** (txt2img - 512 x 512) *with Flash Attention* | ~2.4G | ~1.9G | ~1.6G | ~1.5G | ~1.5G | ~1.5G | ~1.5G |

## Convert to GGUF

You can also convert weights in the formats `ckpt/safetensors/diffusers` to gguf and perform quantization in advance, avoiding the need for quantization every time you load them.

For example:

```sh
./bin/sd-cli -M convert -m ../models/v1-5-pruned-emaonly.safetensors -o  ../models/v1-5-pruned-emaonly.q8_0.gguf -v --type q8_0
```

## Automatic per-tensor quantization (`--att`)

`-M convert` with `--tensor-type-rules` lets you assign a different quant type per tensor, but the rules have to be written by hand. The `--att` (auto-tensor-type) flag *derives* them: given a diffusion model, an imatrix, and a bits-per-weight (bpw) budget, it measures each candidate type's real output error on captured activations and solves for the lowest-error per-tensor assignment that fits the budget. It runs *during a normal generation* (same ergonomics as `--imat-out`) and writes a `--tensor-type-rules` string when the run finishes. The goal is to make aggressive sub-4-bpw quants usable on image models, where a naive uniform quant tends to destroy quality (especially fine typography).

```sh
# One run: collect an imatrix AND search a 3.5-bpw assignment
./bin/sd-cli --diffusion-model model.gguf --vae vae.safetensors --llm te.gguf \
  -p "a cafe menu reading COFFEE 3\$ ..." --steps 20 -s 42 \
  --imat-out model.imatrix \
  --att "out=rules.txt,bpw=3.5" -o probe.png

# Then quantize with the derived rules
./bin/sd-cli -M convert --diffusion-model model.gguf --imat-in model.imatrix \
  --tensor-type-rules "$(cat rules.txt)" -o model-att35.gguf
```

`--att` requires a standalone `--diffusion-model`, and works best with `--imat-in` (or `--imat-out` to collect in the same run); without an imatrix the candidate quantization falls back to uniform importance.

### How it works

1. **Inventory.** The diffusion model's 2D weights (prefix `model.diffusion_model.`) are grouped into `(role, layer-bucket)` classes. A *role* is the tensor name with its block index replaced by `#` (e.g. `...layers.#.attention.qkv.weight`); each role's layers are split into early/middle/late *buckets* (`buckets=`, default 3) so depth-dependent sensitivity is captured. A few representative layers per bucket (`reps=`) are the capture targets; tensors below `min-elements=` (40k) keep the source type.
2. **Capture.** During generation the backend eval callback intercepts `GGML_OP_MUL_MAT` and, for a target weight, copies a strided token subset of the F32 input activations and the reference output — `samples=` samples spaced by `stride=` occurrences so different denoise steps/cfg passes are represented (`max-tokens=` bounds host RAM). It also chains to the imatrix collector, so `--imat-out` works in the same run.
3. **Cost matrix.** Each captured weight is read from the model file and quantized to every candidate type with the imatrix — identical to how `-M convert` would quantize it. Each quantized weight is re-run through a real matmul on the captured activations and scored by **per-row relative L2** (`||ref − quant||² / ||ref||²`), which is scale-free so low-magnitude residual-stream writers are not underweighted.
4. **Optimization.** An element-weighted multi-choice knapsack (DP) picks one type per `(role, bucket)` to minimize total weighted error subject to the total bpw landing in `[target − tol-low, target + tol-high]`. Tensors that stay unquantized (norms, sub-`min-elements` weights, unmatched tensors) keep their source type and count as fixed overhead — `-M convert` without `--type` leaves unmatched tensors alone, which the bpw accounting relies on.
5. **Output.** A comma-joined `--tensor-type-rules` string with `^…$`-anchored regexes and per-bucket layer alternations, ready to feed straight into `-M convert`.

### Options

`--att` takes comma-separated `key=value` options; `out=` and `bpw=` are required. Capture/DP shape: `types=a|b|c` (candidate ladder, default `q2_K,iq2_s,iq3_xxs,iq3_s,q3_K,iq4_xs,iq4_nl,q4_K,q5_K,q6_K,q8_0`), `buckets=`, `reps=`, `samples=`, `stride=`, `max-tokens=`, `min-elements=`, `threads=`, `tol-high=`, `tol-low=`.

Quality knobs:

- `sigma=uniform|low` (**default `low`**) — `low` keeps each weight's most-recent (lowest-sigma, detail-formation) occurrences in a ring buffer instead of sampling the whole denoise trajectory. On Z-Image this lowered output error ~14% vs `uniform`, consistently across the Vulkan and ROCm backends and independent of the sample count.
- `cost=mean|tail|maxmean` (+ `tail-q=`, `tail-lambda=`) — reduce each capture's per-row error by a high quantile or `mean + λ·max` instead of the mean, to penalize worst-case tensors.
- `role-floor=<bpw>` (+ `floor-roles=a|b|c`) — force a minimum bpw for sensitive roles (attention/adaLN etc.). Off by default; can starve other tensors when the budget is tight.
- `topk=<K>` — emit the top-K distinct assignments (spread across the bpw band) plus a `.candidates` manifest, so you can generate with each and keep whichever looks best.

See `docs/AUTO_TYPE_FINDINGS.md` for design notes and the tuning experiment behind these defaults.