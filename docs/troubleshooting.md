# Troubleshooting

## Completely black or white images or videos / NaNs

Some ggml backends can encounter numerical overflow during inference, producing
NaN (not-a-number) values. This can result in completely black or white images or videos.
Whether it happens can depend on the backend, device, model, and weight format.

Known overflow issues have been addressed as far as possible, but the maintainer
has limited hardware and cannot test every combination. Some cases may therefore
still need a manual workaround.

These options are supported by both `sd-cli` and `sd-server`. If you encounter
this problem, add them to your CLI generation command or server startup command:

```sh
--linear-scale 0.0078125 --attn-scale 0.0078125
```

For `sd-server`, restart the server after changing these startup options. Run the
same prompt and seed again to see whether the output recovers. If the problem
persists, try smaller positive values, for example:

```sh
--linear-scale 0.00390625 --attn-scale 0.00390625
```

These options reduce intermediate values and compensate afterwards to preserve
the intended output scale:

- `--linear-scale` scales Linear inputs before matrix multiplication and rescales
  the result.
- `--attn-scale` scales attention keys and values (K/V). It takes effect only in
  the Flash Attention path, where `--fa` or `--diffusion-fa` is enabled and the
  backend supports it.

The two values can be set independently and apply across model components. The
default `0` preserves each model's built-in settings; `1` explicitly disables the
corresponding scaling. Overrides must be finite positive values. C API users can
set `linear_scale` and `attn_scale` in `sd_ctx_params_t`.

If the problem persists after trying the relevant steps above,
[submit a bug report](https://github.com/leejet/stable-diffusion.cpp/issues/new?template=bug_report.yml).
Include your full command, backend and hardware, model and weight format, logs,
and the scale values you tried with their results.
