## Caching

Caching methods accelerate diffusion inference by reusing intermediate computations when changes between steps are small.

### Cache Modes

| Mode | Target | Description |
|------|--------|-------------|
| `ucache` | UNET models | Condition-level caching with error tracking |
| `easycache` | DiT models | Condition-level cache |
| `dbcache` | DiT models | Block-level L1 residual threshold |
| `taylorseer` | DiT models | Taylor series approximation |
| `cache-dit` | DiT models | Combined DBCache + TaylorSeer |

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

For DiT models like FLUX and QWEN, use block-level caching modes.

#### DBCache

Caches blocks based on L1 residual difference threshold:

```bash
--cache-mode dbcache --cache-option "threshold=0.25,warmup=4"
```

#### TaylorSeer

Uses Taylor series approximation to predict block outputs:

```bash
--cache-mode taylorseer
```

#### Cache-DIT (Combined)

Combines DBCache and TaylorSeer:

```bash
--cache-mode cache-dit --cache-preset fast
```

#### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `Fn` | Front blocks to always compute | 8 |
| `Bn` | Back blocks to always compute | 0 |
| `threshold` | L1 residual difference threshold | 0.08 |
| `warmup` | Steps before caching starts | 8 |

#### Presets

Available presets: `slow`, `medium`, `fast`, `ultra` (or `s`, `m`, `f`, `u`).

```bash
--cache-mode cache-dit --cache-preset fast
```

#### SCM Options

Steps Computation Mask controls which steps can be cached:

```bash
--scm-mask "1,1,1,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,1"
```

Mask values: `1` = compute, `0` = can cache.

| Policy | Description |
|--------|-------------|
| `dynamic` | Check threshold before caching |
| `static` | Always cache on cacheable steps |

```bash
--scm-policy dynamic
```

### Performance Tips

- Start with default thresholds and adjust based on output quality
- Lower threshold = better quality, less speedup
- Higher threshold = more speedup, potential quality loss
- More steps generally means more caching opportunities
