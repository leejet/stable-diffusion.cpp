# SPEED — Spectral Progressive Diffusion sampler

SPEED (`--sampling-method speed_flow`) accelerates flow-model sampling by running the early denoising steps at a **reduced spatial resolution**, then progressively expanding to full resolution as the trajectory approaches the data manifold. Reference: [Xiao et al., arXiv:2605.18736](https://arxiv.org/abs/2605.18736); reference C++/Python: [howardhx/speed](https://github.com/howardhx/speed), [ruwwww/ComfyUI-SPEED](https://github.com/ruwwww/ComfyUI-SPEED).

Between segments, the low-resolution latent is expanded via a spectral basis (DCT-II by default) — the low-frequency coefficients are preserved and the newly-exposed high-frequency bands are filled with Gaussian noise scaled by the current sigma. The flow-matching time at each transition is rescaled by `kappa` (paper Eq. 5-6) so the base sampler continues on a consistent schedule after resize.

## Quick usage

```sh
# Flux dev / schnell — default configuration
./bin/sd-cli --sampling-method speed_flow ...

# Explicit scales; use ':' as the intra-list separator
./bin/sd-cli --sampling-method speed_flow \
    --extra-sample-args "speed_scales=0.5:1.0"

# Manual transition point (overrides delta-optimal)
./bin/sd-cli --sampling-method speed_flow \
    --extra-sample-args "speed_scales=0.5:1.0,speed_manual_sigmas=0.7"
```

## Extra sampler args

All parsed out of `--extra-sample-args`. The list is comma-separated at the top level; within a single value use `:` (or `/`, `|`) as the intra-value separator.

| Arg | Default | Description |
|---|---|---|
| `speed_scales` | `0.5:1.0` | Colon-separated resolution fractions, strictly increasing, ending at `1.0`. E.g. `0.25:0.5:1.0` for a 3-level schedule. |
| `speed_levels` | *(unset)* | Shortcut: `speed_levels=2` → `{0.5, 1.0}`; `=3` → `{0.25, 0.5, 1.0}`. Ignored if `speed_scales` is set. |
| `speed_delta` | `0.01` | Noise-dominated tolerance (paper Eq. 9). Smaller values push transitions later in the trajectory. |
| `speed_manual_sigmas` | *(unset)* | Colon-separated sigma thresholds, one per transition (length `= scales.size() - 1`). Bypasses `speed_delta` and the power-spectrum formula. Values must be strictly decreasing and each in `(0, 1)`. |
| `speed_preset` | *auto* | Power-spectrum preset. Auto-selected from the loaded model: `wan21` (`A=219.485, β=2.423`) for WAN, `flux` (`A=203.615, β=1.915`) for every other flow model. Values fitted from the respective VAEs in the reference impl's `configs.yaml`. Set explicitly to override. |
| `speed_A`, `speed_beta` | *(preset)* | Power-spectrum override for custom models. Overrides `speed_preset`. |
| `speed_transform` | `dct` | Spectral basis for the resolution expansion. `dct` handles any ratio; `fft` also handles any ratio, provided both spatial dims are powers of 2. |
| `speed_seed` | `0` | Seed for the Gaussian noise placed into the new high-frequency bands at each transition. |

## Typical speedups

Measured locally on an RTX 3060 12 GiB. Numbers are sampling-time speedups over `--sampling-method euler` at the same seed and step count.

| Model | Resolution | Steps | Config | Speedup |
|---|---|---|---|---|
| Flux schnell Q4_K | 1024×1024 | 8 | `speed_scales=0.5:1.0` (delta-optimal) | ~1.6× |
| Qwen Image Q2_K | 1024×1024 | 20 | `speed_scales=0.5:1.0` (delta-optimal) | ~1.7× |
| Z-Image turbo Q8 | 1024×688 | 8 | `speed_scales=0.5:1.0,speed_manual_sigmas=0.7` | ~3.5× |

Compute savings scale with the number of low-resolution steps in the schedule. Models with front-loaded (shift-heavy) sigma schedules — Flux schnell, Z-Image turbo — often benefit from `speed_manual_sigmas` to push the transition into the middle of the trajectory rather than the very end.

## Compatibility

- `--offload-to-cpu`, `--max-vram`, `--stream-layers`: fully compatible. The graph-cut planner builds a second cached plan for the full-resolution shape at the transition (~5 ms overhead).
- `--cache-mode easycache` (and other caches): compatible. The cache runtime tolerates the resolution transition; the cache heuristic skips only steps where the shape context is stable.
- Non-flow denoisers (SD 1.5, SDXL): SPEED targets flow models specifically. If dispatched on an ε-prediction model it will run but the kappa/sigma rescale math is not valid — do not use.

## Reference-latent models (Qwen Image Edit, Flux Kontext, etc.)

**SPEED runs on ref-latent models but reduces fidelity to the source image.** A warning is logged when this combination is detected:

```
[WARN ] SPEED sampler with reference latents (edit / ref-guided model): the low-resolution
        segments see a downscaled reference so fine detail from the source image (backgrounds,
        textures, layout beyond broad structure) is likely to drift.
```

Why: the reference latents are the low-frequency structure the edit is supposed to preserve. During SPEED's low-resolution segments the reference is downscaled to match the working latent, so the model conditions on a coarser version of the reference — it retains the semantic subject and the requested edit direction, but loses fine detail (backgrounds, textures, exact framing). Composition remains plausible but is not faithful to the source.

Practical guidance:

- For **creative / style edits** where subject and edit intent matter more than reproducing the source scene: SPEED is fine.
- For **fidelity edits** (localized inpaint-style changes, preserving specific backgrounds or text): use `euler` or another full-resolution sampler.
- To trade off, use `speed_manual_sigmas` with a threshold in the middle of the trajectory so only 2–3 early steps run at low resolution.

## Known limits (v1)

- Naive O(N²) 2D DCT — ~1.8 s per direction on a 128×86×16 latent. Fine for image-sized latents; video and very large latents would need a proper FFT-based DCT.
- Only image latents (4-D `[W, H, C, N]`). Video (5-D) is not handled.
- Multi-resolution triggers 2 ggml graph builds per generation, cached thereafter.
- Inpainting masks with `!=` full-resolution shape are silently skipped at low-res steps (the mask blend guard checks the shape and falls through when it can't broadcast).

## References

- Paper: Xiao et al., *"SPEED: Spectral Progressive Diffusion for training-free acceleration"*, [arXiv:2605.18736](https://arxiv.org/abs/2605.18736)
- Official code: <https://github.com/howardhx/speed>
- ComfyUI implementation: <https://github.com/ruwwww/ComfyUI-SPEED>
