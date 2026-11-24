## Caching

DGL versions of the generation examples are linked below. Edit each script's
`var` block and follow the [DGL run instructions](../dgl/README.md).


Caching methods accelerate diffusion inference by reusing intermediate computations when changes between steps are small.

### Conditioning Cache

Conditioning results are cached per model context using an LRU cache. The default
capacity is **4 entries for `sd-cli`, `sd-server` and the C API**.
Set `--conditioning-cache-size N` to change the limit; `0` disables caching.
For example, `sd-cli -m model.safetensors -p "a cat" --conditioning-cache-size 8`
increases the CLI cache capacity to 8 entries. The C API option is
`conditioning_cache_size`, set with `sd_opt_set_int()` on context options.
This cache is independent of the diffusion-step `--cache-mode` options below.

### Cache Modes

| Mode | Target | Description |
|------|--------|-------------|
| `ucache` | UNET models | Condition-level caching with error tracking |
| `easycache` | DiT models | Condition-level cache |
| `dbcache` | DiT models | Block-level L1 residual threshold |
| `taylorseer` | DiT models | Taylor series approximation |
| `cache-dit` | DiT models | Combined DBCache + TaylorSeer |
| `spectrum` | UNET and DiT models | Chebyshev + Taylor output forecasting |

### UCache (UNET Models)

UCache caches the residual difference (output - input) and reuses it when input changes are below threshold.

```bash
sd-cli -m model.safetensors -p "a cat" --cache-mode ucache --cache-option "threshold=1.5"
```


#### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `threshold` | Error threshold for reuse decision | 1.0 |
| `start` | Start caching at this percent of steps | 0.15 |
| `end` | Stop caching at this percent of steps | 0.95 |
| `decay` | Error decay rate (0-1) | 1.0 |
| `relative` | Scale threshold by output norm (0/1) | 1 |
| `reset` | Reset error after computing (0/1) | 1 |

#### Reset Parameter

The `reset` parameter controls error accumulation behavior:

- `reset=1` (default): Resets accumulated error after each computed step. More aggressive caching, works well with most samplers.
- `reset=0`: Keeps error accumulated. More conservative, recommended for `euler_a` sampler.

### EasyCache (DiT Models)

Condition-level caching for DiT models. Caches and reuses outputs when input changes are below threshold.

```bash
--cache-mode easycache --cache-option "threshold=0.3"
```

#### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `threshold` | Input change threshold for reuse | 0.2 |
| `start` | Start caching at this percent of steps | 0.15 |
| `end` | Stop caching at this percent of steps | 0.95 |

### Cache-DIT (DiT Models)

For DiT models like FLUX and QWEN, these modes cache each model condition's
raw output. This path does not perform block-level caching.

#### DBCache

Reuses whole-model input/output residuals when input changes pass the L1 difference threshold:

```bash
--cache-mode dbcache --cache-option "threshold=0.25,warmup=4"
```

#### TaylorSeer

Uses Taylor series approximation to predict each condition's raw model output:

```bash
--cache-mode taylorseer
```

#### Cache-DIT (Combined)

Combines DBCache and TaylorSeer:

```bash
--cache-mode cache-dit
```

#### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `Fn`, `Bn` | Legacy options: ignored with a warning; block counts are not supported | No effect |
| `threshold` | L1 residual difference threshold | 0.08 |
| `warmup` | Steps before caching starts | 8 |

Explicit `Fn`/`Bn` settings, including their former defaults, produce a warning.
The C API keys `cache.fn_compute_blocks` and `cache.bn_compute_blocks` remain
accepted for compatibility and also warn when set. They have no effect on cache
execution. Omitted options do not warn.

#### SCM Options

Steps Computation Mask controls which steps can be cached:

```bash
--scm-mask "1,1,1,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,1"
```

Mask values: `1` = compute, `0` = can cache.

The native `--scm-policy` option is retained for compatibility. Its `dynamic`
and `static` values do not change this condition-level implementation: a mask
value of `1` forces computation; `0` permits reuse subject to the cache's gates
and available history.

```bash
--scm-policy dynamic
```

### Spectrum (UNET and DiT Models)

Spectrum uses Chebyshev polynomial fitting blended with Taylor extrapolation to predict denoised outputs, skipping entire forward passes. Based on the paper [Spectrum: Adaptive Spectral Feature Forecasting for Efficient Diffusion Sampling](https://github.com/tingyu215/Spectrum).

```bash
sd-cli -m model.safetensors -p "a cat" --cache-mode spectrum
```


#### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `w` | Chebyshev vs Taylor blend weight (0=Taylor, 1=Chebyshev) | 0.40 |
| `m` | Chebyshev polynomial degree | 3 |
| `lam` | Ridge regression regularization | 1.0 |
| `window` | Initial window size (compute every N steps) | 2 |
| `flex` | Window growth per computed step after warmup | 0.50 |
| `warmup` | Steps to always compute before caching starts | 4 |
| `stop` | Stop caching at this fraction of total steps | 0.9 |

### Performance Tips

- Start with default thresholds and adjust based on output quality
- Lower threshold = better quality, less speedup
- Higher threshold = more speedup, potential quality loss
- More steps generally means more caching opportunities
