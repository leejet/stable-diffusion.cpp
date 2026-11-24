# Troubleshooting

## Windows: the program exits immediately without printing anything

On Windows, an `sd-cli.exe` or `sd-server.exe` that ends at once with no output
(for example `--list-devices` prints nothing), or an application that reports an
exit code such as `3221225781`, usually means that the Microsoft Visual C++
Redistributable is missing or too old. The pre-built binaries need the
2015-2022 (x64) redistributable **version 14.44 or newer**; Microsoft requires the
installed version to be at least as recent as the build tools that produced the
binaries ([details](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)).

Windows reports this only through the exit code, normally without an
"Application Error" event. In PowerShell:

```powershell
.\sd-cli.exe --list-devices
$LASTEXITCODE    # -1073741515 (0xC0000135, the same as 3221225781): a DLL was not found
(Get-Item C:\Windows\System32\msvcp140.dll).VersionInfo.FileVersion    # must be 14.44 or newer
```

Install the latest redistributable from <https://aka.ms/vc14/vc_redist.x64.exe>
and run the command again. The other start-up codes, such as `0xC0000139` (a DLL
lacks a required function: mixed versions in one folder or an old runtime),
`0xC000007B` (32-bit and 64-bit files mixed) and `0xC0000428` (file signature
rejected by a security policy), are listed in the
[Italian guide](./integrazione_progetti.md#28-problemi-comuni-e-come-leggerli).

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
set the `linear_scale` and `attn_scale` context-option keys with `sd_opt_set_float()`.

If the problem persists after trying the relevant steps above,
[submit a bug report](https://github.com/leejet/stable-diffusion.cpp/issues/new?template=bug_report.yml).
Include your full command, backend and hardware, model and weight format, logs,
and the scale values you tried with their results.

## Windows: process exits immediately with 0xC0000135 (STATUS_DLL_NOT_FOUND)

`sd-cli.exe` / `sd-server.exe` from the Windows ROCm prebuilt may exit with
`0xC0000135` (`STATUS_DLL_NOT_FOUND`) and no output when the HIP runtime is
not on `PATH`. `stable-diffusion.dll` statically imports `hipblas.dll` and
`libhipblaslt.dll`, but the release archive only bundles `amd_comgr.dll`,
`amdhip64_7.dll` and `rocm_kpack.dll`. This is a loader failure, unrelated to
`--audio-vae` or other generation options; `0xC000001D`
(`STATUS_ILLEGAL_INSTRUCTION`) would instead indicate a CPU instruction issue.

Fix: prepend the ROCm SDK `bin` directories to `PATH` — with the Python ROCm
wheels these are `<python home>\Lib\site-packages\_rocm_sdk_libraries\bin` and
`_rocm_sdk_core\bin`. Do not copy individual DLLs: `rocblas.dll` and
`libhipblaslt.dll` also need their sibling `rocblas\` and `hipblaslt\` data
directories.

Verify:

```powershell
.\sd-cli.exe --version
'0x{0:X8}' -f $LASTEXITCODE
```

Expect the version string and `0x00000000`. If it still fails, inspect the
whole DLL chain (`sd-cli.exe` -> `stable-diffusion.dll` -> ...): `dumpbin
/dependents` lists only one image's direct imports, so use
[Dependencies](https://github.com/lucasg/Dependencies) or also check
`stable-diffusion.dll` to find the missing module.
