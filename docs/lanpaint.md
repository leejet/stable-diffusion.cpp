## LanPaint Inpainting

LanPaint is a training-free inpainting sampler that gives diffusion models
"think mode": instead of denoising in one pass, it runs up to N inner Langevin
steps per sampling step, letting the model reconcile the prompt-driven region
with the known (kept) region before committing to the update. It is a port of
the [ComfyUI LanPaint extension](https://github.com/scraed/LanPaint)
(official implementation of
["LanPaint: Training-Free Diffusion Inpainting with Asymptotically Exact and Fast Conditional Sampling"](https://arxiv.org/abs/2502.03491),
TMLR 2025), with the parameter defaults of the v2.1.0 ComfyUI node.

In essence, the inpainting problem is modeled as a conditional stochastic
process, and LanPaint integrates it numerically with
[Langevin dynamics](https://en.wikipedia.org/wiki/Langevin_dynamics). The
drift of the process is assembled from the underlying model's own denoising
solutions, i.e. its inferred solutions of the unconditional version of the
same process, so the conditional problem is solved without retraining. Each
inner iteration is one integration step of this process, and each step costs
one model evaluation; that is the exchange of extra compute for inpainting
quality: up to `1 + n_steps` model evaluations per sampling step instead of one.

### Usage

Provide an init image and a mask and enable LanPaint:

```bash
sd-cli -m model.safetensors -i images/input.png --mask images/mask.png \
    -p "a cozy living room, photorealistic" \
    --strength 1.0 --sampling-method euler --lanpaint -v
```

Mask convention (same as the rest of `stable-diffusion.cpp`): **white (255)
marks the region that gets repainted, black (0) marks the region that is kept.**
Masks are binarized at load time and evaluated at latent resolution. When no
mask is given, LanPaint is inactive and sampling proceeds like the plain
sampler.

The inner loop needs both a conditional and an unconditional model branch.
Guidance-distilled models without an unconditional branch (for example
turbo/schnell variants) run, but the keep-region guidance degenerates and the
LanPaint benefit largely disappears.

### Parameters

| Flag | Description | Default |
|------|-------------|---------|
| `--lanpaint` | enable LanPaint | off |
| `--lanpaint-n-steps` | number of inner Langevin steps per sampling step ("turns of thinking"); the effective count ramps down near the end of the schedule | 5 |
| `--lanpaint-lambda` | strength of the keep-region (bidirectional) guidance | 5.0 |
| `--lanpaint-beta` | time-scale ratio of the keep-region branch | 1.0 |
| `--lanpaint-step-size` | Langevin step size; scaled by the remaining noise fraction of the current step | 0.2 |
| `--lanpaint-early-stop` | skip the inner loop for the last N sampling steps (they only polish with a plain evaluation) | 1 |
| `--lanpaint-min-step-frac` | when the remaining noise fraction drops below this value, the step size is pinned there and the inner-step count ramps down to zero | 1.0 |
| `--lanpaint-prompt-first` | Prompt First mode: sets the BIG guidance scale to -0.5, emphasizing prompt following over mask-boundary coherence | off (Image First) |
| `--lanpaint-cfg-big` | explicit BIG guidance scale override; by default it resolves to `--cfg-scale` (Image First) or -0.5 (Prompt First) | auto |

`Image First` (default) and `Prompt First` change how strongly the inner loop
is anchored to the already-known image versus the prompt. Use Image First for
seamless object insertion in the known surroundings; use Prompt First when the
repainted region should follow the prompt even at the cost of boundary
coherence.

LanPaint inherits every per-step feature of the outer sampler: conditioning, Control
Net, IP-Adapter, reference latents, video masks, previews and cancellation.

### Cost

Each inner step costs one model evaluation (conditional + unconditional), so a
run needs up to `steps x (1 + n_steps)` evaluations. At INFO log level the
planned count is printed before sampling starts:

```
LanPaint: 16 outer steps, up to 5 inner steps per outer step, 68 model evaluations planned (16 without LanPaint)
```

At VERBOSE level every inner step is logged. With the defaults, `early_stop`
and the `min_step_frac` ramp already keep the tail of the schedule cheap.
Recommended `--lanpaint-n-steps` range: 2-8 (the upstream default of 5 is a
good quality/speed balance).

### Supported samplers and models

- Samplers: `euler` (recommended), `euler_a`, `heun`, `dpm2`, `dpm++2m`,
  `dpm++2mv2`. Other samplers are rejected with an error.
- Models: everything using the standard denoiser noise scaling, including
  SD1.x/SD2.x (eps, v-prediction, EDM), SDXL, SD3/SD3.5, FLUX, Chroma,
  Qwen-Image, Wan, HunyuanVideo, and LTX video models.
- Not supported (rejected with an error):
  - MiniT2I and SeFi (custom latent scaling / dual time conventions),
  - SenseNova U1.5 (its noise scaling discards the latent, which would
    overwrite the kept region),
  - MiniMax-H3 and LTX-AV (audio rows follow a per-stream noise schedule the
    inner loop does not model).
- Sampling caches (`--cache-mode`, for example) are disabled under LanPaint:
  the inner loop's repeated evaluations at one step index break their reuse
  accounting.

