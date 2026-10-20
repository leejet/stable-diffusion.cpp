# Model selection

Start without model arguments to select models in the web UI:

```sh
sd-server --model-dir ./models
```

From a source checkout, you can also use `sh ./start-server.sh` on Linux/macOS
or `.\start-server.bat` on Windows. The launchers look for the server relative
to the script: beside the script, then in `bin/Release/`, `bin/`,
`build/bin/Release/`, and `build/bin/`, in that order.
For a different build directory, set `SD_SERVER_PATH` to the executable path:

```sh
SD_SERVER_PATH="./build_vulkan/bin/sd-server" sh ./start-server.sh --listen-port 8080
```

```powershell
$env:SD_SERVER_PATH = ".\build_vulkan\bin\Release\sd-server.exe"
.\start-server.bat --listen-port 8080
```

Binary release archives include both launchers. Keep the launchers under version
control and put personal settings in a separate file beside them:

- Windows batch: `start-server.local.bat`
- Linux/macOS: `start-server.local.sh`

These local files are ignored by Git and excluded from release archives. They
are loaded automatically after the launcher's defaults. They can override
`SD_SERVER_PATH`, `MODEL_DIR`, and `COMMANDLINE_ARGS`; there is no need to edit
the tracked launchers. For example:

```bat
@echo off
set "MODEL_DIR=D:\stable-diffusion-webui\models"
set "COMMANDLINE_ARGS=--listen-port 8080 --offload-to-cpu"
```

```sh
MODEL_DIR="/data/stable-diffusion-webui/models"
COMMANDLINE_ARGS="--listen-port 8080 --offload-to-cpu"
```

Each launcher prints the resolved command before execution and forwards the
server's exit code.

`SD_SERVER_PATH` inherits the environment value by default; leave it empty for
automatic discovery, or set it in the local file to select another executable.
Settings are applied in this order: launcher defaults, local configuration,
then invocation arguments. Repeated `--model-dir` arguments add roots, so they
do not replace the `MODEL_DIR` root. Local configuration files are shell scripts
and use the corresponding shell's syntax.

The launchers preserve the current working directory, so the default
`MODEL_DIR="./models"` is relative to where you start the launcher. Arguments
passed when invoking the launcher follow these defaults. Repeated `--model-dir`
arguments add search roots rather than replacing `MODEL_DIR`; set `MODEL_DIR` in
the local configuration to change the first root. Other single-value options can override the defaults.
In the shell script, `COMMANDLINE_ARGS` is a
whitespace-separated list; pass values containing spaces as quoted arguments
when invoking the script. `MODEL_DIR` itself supports spaces in both scripts.
A leading `--version` bypasses the configured defaults. An explicit
`SD_SERVER_PATH` is used without falling back to another build if it is invalid.

Choose a recipe, select the required component files, and click **Load models**.
Recipe and component selectors support searching by name or relative file ID.
Use the arrow keys and Enter to select, or Escape to cancel a search. With
**Apply recipe defaults** enabled, selecting a recipe immediately applies its
declared generation defaults while preserving prompts, input images and fields
not specified by the recipe. Disable it to keep your current parameters when
switching recipes. This preference defaults to enabled and is saved locally.
Model loading and server-info refreshes do not reapply defaults or reset the form.
Recipes declaring `video_frames` also switch the form to Video Generation;
other recipes switch to Image Generation. Generation is enabled after the
selected recipe is loaded. Changing a selection does not reload the model
until you click Load.
The server can also start with the existing model arguments shown below;
existing API clients then work without selecting a recipe. The UI automatically
selects **Command-line model (all features)** and uses the startup generation
defaults. This runtime-only recipe exposes all controls for the supported
image/video mode. Selecting another recipe previews a model switch; selecting
the command-line entry again restores its generation defaults. The entry is
removed when the command-line context is unloaded or replaced.

Built-in recipes cover SD 1/2/3/3.5, SDXL and distilled checkpoints;
FLUX 1/2/3, Chroma, Qwen Image and editing variants; Z-Image, Anima, ERNIE,
LENS, PiD and other supported image families; AnimateDiff, Wan, LTX,
HunyuanVideo, LingBot Video and MiniMax-H3. Model-specific text encoder
labels identify the expected encoder family and size. Loading a video model enables **Video Generation**; I2V also requires a
start image. The Models panel can be collapsed and remembers its open state.
Its header shows elapsed loading time from the status API and keeps the final
time after success or failure, including after a page refresh.
Built-ins are embedded in the
binary from [recipes/](recipes/), with one JSON file per recipe. Adding or
editing a built-in requires rebuilding the server. User recipes are read from
`<model-dir>/recipes/*.json`. A valid user recipe with the same ID replaces
the entire built-in recipe. Across roots, the first valid user recipe wins;
overrides are logged. Duplicate user IDs within one root are rejected; invalid
recipes are reported without removing a valid recipe from another root or the
built-in version. Slots need only `id` and
`required`; the server supplies labels and search directories. Use `label`
only when a recipe needs a more specific display name.

Recipes use `features` to enable specialized controls and `disabled_features`
to turn off shared defaults. Disabled features take precedence. The UI hides
unsupported fields, empty sections and irrelevant sampler/cache options, and
omits hidden values from generation requests. See the
[feature declarations](api.md#recipe-feature-declarations) for the defaults
and complete list of supported names.

Built-in additions and changes must follow the [recipe policy](recipes/README.md).
Split recipes by task weights or added capabilities, not by style or acceleration
LoRAs. Quantizations, file formats and compatible layer-count variants share a
recipe. Native encoders, projectors and connectors stay in the model's recipe.
Community editing recipes require selecting the matching LoRA separately; a
recipe does not enable it automatically.

See [generation defaults and sources](recipes/defaults.md) for the selected
publisher recommendations and reference-code defaults, with step defaults capped
at 20. Recipes link to the
original publisher's Hugging Face repository where one is available. DiT
recipes default to Euler, except for models requiring a specialized sampler.

Compatible recipes offer a TAE slot alongside VAE. When the checkpoint does not
bundle a VAE, select at least one of VAE or TAE. The Models panel shows a reminder
below the model selectors until this requirement is met. A selected TAE is used for
encoding and decoding, including when a full VAE is also selected; use complete,
model-compatible TAE weights rather than decoder-only preview weights. Qwen
Image 2.1 uses the same recipe for its full VAE and TAEQI2.1 alternatives.

The following variants have dedicated recipes:

| Model | Recipe IDs / notes |
| --- | --- |
| Z-Image L2P | `z-image-l2p`; no VAE |
| LLaDA-Image | `llada-image` (base) and `llada-image-turbo`; use matching encoders and connectors |
| PixArt Alpha | `pixart-alpha-512`, `pixart-alpha-1024`; Alpha VAE scale and positional settings |
| PiD / PiD 1.5 | `pid` / `pid-15` for Flux or Z-Image VAE; append `-sd3`, `-flux2`, or `-qwen-image` for other latent formats |
| SDXL adapters | `sdxl-ip-adapter` supports classic and Plus ViT-H adapters; `sdxl-controlnet` requires compatible SDXL weights |
| FLUX.1 Fill / Depth | `flux1-fill` uses Init Image and Mask Image; `flux1-depth` uses a precomputed depth map in Control Image |
| SD2 / SDXL inpainting | `sd2-inpainting`, `sdxl-inpainting` |
| SD-Turbo | `sd-turbo`; single-file SD2.1-derived checkpoint |
| Community editing LoRAs | `anima-edit`, `krea2-raw-edit`, `krea2-turbo-edit`; apply a matching LoRA after loading |
| HunyuanVideo 1.5 | `hunyuan-video-15-t2v`, `hunyuan-video-15-i2v`; choose the task's checkpoint |
| Wan 2.1 / compatible single Wan 2.2 experts | `wan21-t2v`, `wan21-i2v`; the I2V recipe also accepts compatible SkyReels V2 weights |
| Qwen Image 2.1 Turbo | `qwen-image-21-turbo`; standalone Turbo weights with the fixed 8-step schedule, Euler and CFG 1 |
| MOSS Image | `moss-image-base`, `moss-image-sft`; choose the corresponding checkpoint |

PiD always requires a reference image and uses the first one. Its recipes expose
reference inputs rather than ordinary image-to-image inputs. IP-Adapter recipes
expose a separate IP-Adapter image and strength. FLUX Canny/Depth consume Control
Image directly without a separate ControlNet; the UI does not calculate edge
or depth maps. Their strength is controlled by guidance, not ControlNet strength.
Krea2 edit recipes use the default `krea2_ostris_edit` reference preset, so the
editing LoRA must match that convention. Ordinary Anima and Krea2 recipes do not
expose reference-image editing inputs.

Fill/Depth guidance defaults follow the [Diffusers FLUX documentation](https://huggingface.co/docs/diffusers/api/pipelines/flux).
SD-Turbo uses the [model card's one-step configuration](https://huggingface.co/stabilityai/sd-turbo),
with CFG disabled using scale 1 in sd.cpp.

SVD is not included: legacy model recognition and UNet branches remain, but the
current runner/conditioning setup does not provide a dedicated SVD image-conditioning
path. Adding a recipe alone would not establish a working SVD pipeline.

Reference images, masks, audio, first/last video frames, and generation options
such as upscaling remain generation inputs. Hardware and memory options remain
startup arguments. Tiny SD/Tiny SD2 and Diffusers LCM directories must first be
converted into single-file checkpoints for the file catalog. LCM checkpoints
and LoRAs use the matching SD family recipe; select LCM sampling and adjust
steps/CFG for the chosen weights.
Metadata inspection, conversion, standalone upscaling and ADetailer are CLI
workflows rather than model-loading recipes. ESRGAN and LTX latent upscalers
are selected through generation upscaling settings with `--hires-upscalers-dir`;
textual inversion uses the startup `--embd-dir` option.

Repeat `--model-dir` to share multiple model directories, including an existing
SD WebUI model directory:

```sh
sd-server --model-dir ./models --model-dir /data/stable-diffusion-webui/models
```

With no `--model-dir`, the server uses `./models`. Explicit roots replace that
default and are scanned in argument order. Duplicate canonical roots and file
paths are scanned once. Every root can contain a `recipes/` directory.
Directory names are matched case-insensitively on all platforms:

| Category | Directory names |
| --- | --- |
| Checkpoints | `checkpoints`, `Stable-diffusion` |
| VAE | `vae`, `VAE` |
| LoRA | `loras`, `Lora` |
| ControlNet | `controlnet`, `ControlNet` |

Other categories keep their existing directory names. Classification uses only
the containing directory: `*.vae.safetensors`, `*.vae.ckpt`, and similar names
inside `checkpoints/` or `Stable-diffusion/` remain checkpoints.

Place model files in category directories, for example:

```text
models/
  checkpoints/
  diffusion_models/
  text_encoders/
  vae/
  clip_vision/
  loras/
  recipes/
    my-flux.json
```

Subdirectories within a model category are supported. Use **Refresh files &
recipes** after adding files or changing recipes. Scanning lists file names
without loading model weights. Category membership does not guarantee
compatibility; the model loader performs the final compatibility check.
Symlinks resolving outside the model directory are rejected. Dropdown labels
omit the category prefix while retaining subdirectory names. Identical display
paths are displayed as full file paths, with source directories before the
filename instead of a hash suffix. The same applies to duplicate LoRA paths. File IDs include a
root qualifier for additional roots; clients should use the returned IDs.
Empty optional categories do not display missing-file hints.

Place Tiny AutoEncoder (TAE) files in `vae/` and select them in the **TAE**
slot. The legacy `vae_approx/` directory is also scanned for TAE selections.

LoRAs are discovered automatically in both `loras/` and `Lora/` under every
model root. Use `--lora-model-dir` to replace automatic discovery with one
explicit LoRA directory. Duplicate names remain separate selectable entries. The LoRA selectors support
searching by name or path with the same keyboard controls as model selectors.

The server keeps one model context loaded. Loading or unloading is rejected
while generation requests or queued jobs are pending. A switch validates the
selection before unloading the old context, then loads the replacement.
If loading fails, the service stays online with no active model and reports
the error. Backend and memory options supplied at startup apply to subsequent
loads. Recipe defaults apply to new loads; explicitly supplied generation
request parameters take precedence.

See [the API reference](api.md#model-management) for recipe fields, supported
slots, management endpoints, and compatibility behavior.

# Example

The following example starts `sd-server` with a standalone diffusion model, VAE, and LLM text encoder:

```
.\bin\Release\sd-server.exe --diffusion-model  ..\models\diffusion_models\z_image_turbo_bf16.safetensors --vae ..\models\vae\ae.sft  --llm ..\models\text_encoders\qwen_3_4b.safetensors --offload-to-cpu -v --cfg-scale 1.0
```

What this example does:

* `--diffusion-model` selects the standalone diffusion model
* `--vae` selects the VAE decoder
* `--llm` selects the text encoder / language model used by this pipeline
* `--offload-to-cpu` reduces VRAM pressure by keeping weights in RAM when possible
* `-v` enables verbose logging (equivalent to `--log-level verbose`)
* `--cfg-scale 1.0` sets the default CFG scale for generation

Logging defaults to `info`. Use `--log-level <level>` to select `debug`, `verbose`,
`info`, `warn`, or `error` (from most to least detailed). Each level includes
messages at that level and all less detailed levels. `-v` and `--verbose` are
equivalent to `--log-level verbose`. If repeated, the last logging option wins.

After the server starts successfully:

* the web UI is available at `http://127.0.0.1:1234/`
* the native async API is available under `/sdcpp/v1/...`
* the compatibility APIs are available under `/v1/...` and `/sdapi/v1/...`

If you want to use a different host or port, pass:

```bash
--listen-ip <ip> --listen-port <port>
```

# Frontend

## Build with Frontend

The server can optionally build the web frontend and embed it into the binary as `gen_index_html.h`.

### Requirements

Install the following tools:

* **Node.js** ≥ 20
  https://nodejs.org/

* **pnpm** ≥ 10
  Install via npm:

```bash
npm install -g pnpm
```

Verify installation:

```bash
node -v
pnpm -v
```

### Install frontend dependencies

Go to the frontend directory and install dependencies:

```bash
cd examples/server/frontend
pnpm install
```

### Build the server with CMake

Enable the frontend build option when configuring CMake:

```bash
cmake -B build -DSD_SERVER_BUILD_FRONTEND=ON
cmake --build build --config Release
```

If `pnpm` is available, the build system will automatically run:

```
pnpm run build
pnpm run build:header
```

and embed the generated frontend into the server binary.

## Frontend Repository

The web frontend is maintained in a **separate repository**, https://github.com/leejet/sdcpp-webui.

If you want to modify the UI or frontend logic, please submit pull requests to the **frontend repository**.

This repository (`stable-diffusion.cpp`) only vendors the frontend periodically. Changes from the frontend repo are synchronized:

* approximately **every 1–2 weeks**, or
* when there are **major frontend updates**

Because of this, frontend changes will **not appear here immediately** after being merged upstream.

## Using an external frontend

By default, the server uses the **embedded frontend** generated during the build (`gen_index_html.h`).

You can also serve a custom frontend file instead of the embedded one by using:

```bash
--serve-html-path <path-to-index.html>
```

For example:

```bash
sd-server --serve-html-path ./index.html
```

In this case, the server will load and serve the specified `index.html` file instead of the embedded frontend. This is useful when:

* developing or testing frontend changes
* using a custom UI
* avoiding rebuilding the binary after frontend modifications

# Usage

For detailed command-line arguments, run:

```bash
./bin/sd-server -h
```

For completely black or white images or videos, NaNs, and the `--linear-scale` /
`--attn-scale` startup options, see [Troubleshooting](../../docs/troubleshooting.md).
