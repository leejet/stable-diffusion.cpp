# Sampling methods and schedulers

The sampling method controls how each denoising step updates the image or video
latent. The scheduler chooses the noise levels (sigmas) used across those steps.
Choose both according to the [model guide](../models/README.md) for your checkpoint.

## Usage

For example, using the Stable Diffusion 1.5 checkpoint from the
[quick start](../../README.md#quick-start):

```sh
./sd-cli -m ./models/v1-5-pruned-emaonly.safetensors \
  --sampling-method euler_a --scheduler discrete --steps 20 -p "a lovely cat"
```

Run this from the directory containing `sd-cli`. On Windows, use `sd-cli.exe`
with your shell's path and line-continuation syntax.

| CLI option | Purpose |
| --- | --- |
| `--sampling-method NAME` | Select the sampling method. |
| `--scheduler NAME` | Select the sigma scheduler. |
| `--steps N` | Set the number of sampling steps. |
| `--high-noise-sampling-method NAME` | Select the high-noise sampler for two-stage video models. |
| `--sigmas VALUES_OR_PRESET` | Supply a custom schedule instead of generating one with the scheduler. |
| `--hires-sigmas VALUES_OR_PRESET` | Supply a custom schedule for the highres fix second pass. |

The names below are the exact, case-sensitive CLI values. Run `./sd-cli --help`
to check the options supported by your installed build.

For native server requests, use `sample_params.sample_method` and
`sample_params.scheduler`. See the [server API reference](../../examples/server/api.md)
for request fields and capability discovery.

## Sampling methods

| Method | CLI value |
| --- | --- |
| Euler | `euler` |
| Euler ancestral | `euler_a` |
| Heun | `heun` |
| DPM2 | `dpm2` |
| DPM++ 2S ancestral | `dpm++2s_a` |
| DPM++ 2M | `dpm++2m` |
| DPM++ 2M v2 (modified) | `dpm++2mv2` |
| iPNDM | `ipndm` |
| iPNDM V | `ipndm_v` |
| LCM | `lcm` |
| DDIM trailing | `ddim_trailing` |
| TCD | `tcd` |
| RES Multistep | `res_multistep` |
| RES 2S | `res_2s` |
| ER-SDE | `er_sde` |
| Euler CFG++ | `euler_cfg_pp` |
| Euler ancestral CFG++ | `euler_a_cfg_pp` |
| Euler GE (gradient estimation) | `euler_ge` |
| DPM++ 2M SDE | `dpm++2m_sde` |
| DPM++ 2M SDE with Brownian tree noise | `dpm++2m_sde_bt` |
| LMS | `lms` |

## Schedulers

| Scheduler | CLI value |
| --- | --- |
| Discrete | `discrete` |
| Karras | `karras` |
| Exponential | `exponential` |
| AYS | `ays` |
| GITS | `gits` |
| SGM Uniform | `sgm_uniform` |
| Simple | `simple` |
| SmoothStep | `smoothstep` |
| KL Optimal | `kl_optimal` |
| LCM | `lcm` |
| Bong Tangent | `bong_tangent` |
| LTX-2 | `ltx2` |
| Logit Normal | `logit_normal` |
| FLUX.2 | `flux2` |
| FLUX | `flux` |
| Beta | `beta` |
| LLaDA-Image | `llada_image` |

`normal` is accepted as an alias for `discrete`.

## Defaults and model compatibility

Omitting `--sampling-method` and `--scheduler` uses defaults selected for the
model and sampler. A Web UI recipe can supply its own defaults.

- For EDM v-prediction checkpoints, the automatic scheduler is `exponential`.
  Otherwise, the `lcm` and `tcd` samplers default to the `lcm` scheduler, and
  `ddim_trailing` defaults to `simple`. Other samplers use the model's default
  scheduler.
- AYS has reference schedules for SD 1.x, SDXL, and SVD. It is not designed for
  SD 2.x or other model families.
- GITS requires at least two steps.
- Model-specific schedules should follow the matching guide, such as
  [FLUX.2](../models/flux2.md), [LTX-2](../models/ltx2.md),
  [Ideogram4](../models/ideogram4.md), or [LLaDA-Image](../models/llada_image.md).
- Some models use specialized sampling paths or schedules, such as
  [SeFi-Image](../models/sefi_image.md). Changing a generic sampler or scheduler
  option may not change that model's behavior.
- For few-step checkpoints or adapters, keep the sampler, step count, guidance,
  and schedule specified in the model guide. See [LCM/LCM-LoRA](../models/lcm.md)
  for an example.

## Custom sigma schedules

`--sigmas` accepts comma-separated noise levels or one of the presets below.
Supply descending values ending in zero when specifying your own schedule, for
example `--sigmas "14.61,7.8,3.5,0.0"`. The values replace the generated schedule;
the number of sampling steps becomes the number of values minus one.

| Preset | Steps | Model guide |
| --- | --- | --- |
| `qwen-image-2.1-turbo` | 8 | [Qwen Image 2.1 Turbo](../models/qwen_image_2.1.md#turbo) |
| `p-h3-image` | 8 | [P-H3-Image](../models/p_h3_image.md) |
| `minimax-h3-image` | 6 | [MiniMax-H3-Image](../models/minimax_h3_image.md) |
| `hyperflow-h3-flash-4` | 4 | [HyperFlow H3 Flash](../models/minimax_h3.md#hyperflow-h3-flash) |
| `hyperflow-h3-flash-6` | 6 | [HyperFlow H3 Flash](../models/minimax_h3.md#hyperflow-h3-flash) |
| `hyperflow-h3-flash-8` | 8 | [HyperFlow H3 Flash](../models/minimax_h3.md#hyperflow-h3-flash) |

`--hires-sigmas` accepts the same format and presets for the highres fix second
pass. Use each preset with the checkpoint and settings in its model guide.
