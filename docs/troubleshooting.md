# Troubleshooting

## Video model used in image generation mode

If generation reports that a model cannot be run with `generate_image()`, add
`--mode vid_gen` to the CLI command. `--video-frames` alone does not select video
mode. Video models require this mode even when generating a single frame.
Library callers must use `generate_video()` for these models; use
`sd_ctx_supports_image_generation()` and `sd_ctx_supports_video_generation()` to
check the available generation modes.

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
  the Flash Attention path, when selected and supported by the backend.

The two values can be set independently and apply across model components. The
default `0` preserves each model's built-in settings; `1` explicitly disables the
corresponding scaling. Overrides must be finite positive values. C API users can
set `linear_scale` and `attn_scale` in `sd_ctx_params_t`.

If the problem persists after trying the relevant steps above,
[submit a bug report](https://github.com/leejet/stable-diffusion.cpp/issues/new?template=bug_report.yml).
Include your full command, backend and hardware, model and weight format, logs,
and the scale values you tried with their results.

## Windows: process exits immediately with 0xC0000135 (STATUS_DLL_NOT_FOUND)

Running a Windows ROCm prebuilt (`sd-cli.exe` / `sd-server.exe`) in a shell
where the HIP runtime is not on `PATH` can terminate before printing anything,
with exit code `0xC0000135` (`STATUS_DLL_NOT_FOUND`). This is a loader failure,
not an inference crash: `stable-diffusion.dll` statically imports
`hipblas.dll` and `libhipblaslt.dll` (which in turn pull in `rocblas.dll`,
`rocsolver.dll` and the rest of the hipBLAS chain), while the release archive
only bundles the core runtime DLLs (`amd_comgr.dll`, `amdhip64_7.dll` and
`rocm_kpack.dll`) alongside the executables.

Do not confuse this with `0xC000001D` (`STATUS_ILLEGAL_INSTRUCTION`), which
indicates a CPU instruction set problem instead. The failure is also
independent of generation options such as `--audio-vae`: the same binary
succeeds with those options once the runtime is visible.

Fix: prepend the directories containing the HIP runtime DLLs to `PATH` before
starting the executable. Common sources:

- a Windows ROCm runtime/SDK installation (add its `bin` directory to `PATH`,
  or run inside the environment its installer creates);
- the Python ROCm distributions, which install the runtime under
  `<python home>\Lib\site-packages\_rocm_sdk_libraries\bin` and
  `_rocm_sdk_core\bin`.

Prefer adding those directories to `PATH` over copying individual DLLs.
`rocblas.dll` and `libhipblaslt.dll` look for their data directories
(`rocblas\` and `hipblaslt\`) next to themselves, so copying only the DLLs can
get past the loader and still fail later. If you do copy, copy the DLLs
together with those sibling data directories.

Verify the fix with:

```powershell
.\sd-cli.exe --version
```

which should print the version string and exit with code 0. If it still exits
with `0xC0000135`, inspect the executable's imports (`dumpbin /dependents
sd-cli.exe` or the [Dependencies](https://github.com/lucasg/Dependencies)
tool) to identify which DLL is missing.
