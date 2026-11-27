# stable-diffusion.cpp Server

`sd-server` provides a Web UI for image and video generation, model management,
and HTTP APIs. Download or build the server before starting; see the
[build guide](../../docs/build.md) and [frontend build instructions](#build-with-frontend).

- [Web UI](#web-ui): start the server, select a model, and generate.
- [Model management](#model-management): model directories, recipes, and startup models.
- [Server configuration](#server-configuration): launcher settings, network, logging, and help.
- [API](#api): native and compatibility endpoints.
- [Frontend development](#frontend-development): build or customize the Web UI.

Inference settings follow this priority: request fields, explicit server startup
options, model recipe presets, then [library defaults](../../docs/guides/c_api.md#keys).
Omitted request fields retain the configured override. In native JSON requests,
`null` clears an optional inference field so the library selects its value;
`0`, `false`, empty strings and empty numeric arrays are explicit values.
Capability `defaults` and `defaults_by_mode` contain configured overrides, with
`null` for optional settings left to the library. Effective settings are returned
in generation metadata. Output formats, queue limits and listen addresses keep
server-local defaults.

## Web UI

### Start the server

From the repository root, start the launcher. It finds the server executable
and uses the repository's `models/` directory:

**Windows (PowerShell)**

```powershell
.\start-server.bat
```

You can also double-click `start-server.bat` in File Explorer to start the
server. Keep the console window open while using the Web UI.

**Linux/macOS**

```sh
sh ./start-server.sh
```

To start the executable directly instead, run from the build directory:

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-server.exe --model-dir ..\models
```

**Linux/macOS**

```sh
./bin/sd-server --model-dir ../models
```

The Windows path above is for a Visual Studio Release build. Adjust the path
for another build configuration or an extracted release archive. Open
<http://127.0.0.1:1234/> after the server starts, and keep the server running
while using the browser.

### Select a model

1. Download the weights and required components listed in the matching
   [model guide](../../docs/models/README.md). Put them in the category folders
   under `models/`, such as `checkpoints/`, `diffusion_models/`, `text_encoders/`,
   `tokenizers/`, and `vae/`. See [Model directories](#model-directories) below.
2. Click **Model** below the prompt to open model selection.
3. Choose the matching **Recipe**. Keep **Apply recipe defaults** enabled to
   start with that recipe's generation settings.
4. Select each component in its named field: for example, **Diffusion model**,
   the model-specific text encoder field, and **VAE**. Required fields have an
   asterisk. A VAE/TAE requirement needs one compatible decoder. Match all files
   to the chosen checkpoint and variant, as described in its model guide.
5. If a file is missing from a dropdown, check its category folder and click
   **Refresh files & recipes**. Selectors support searching by name or path.

Recipe and component selectors support searching by name or relative file ID.
Use the arrow keys and Enter to select, or Escape to cancel a search. With
**Apply recipe defaults** enabled, selecting a recipe immediately applies its
declared generation defaults while preserving prompts, input images and fields
not specified by the recipe. Disable it to keep your current parameters when
switching recipes. This preference defaults to enabled and is saved locally.
Recipes declaring `video_frames` also switch the form to Video Generation;
other recipes switch to Image Generation. Select the required components to
enable **Generate**. The job loads any needed components and reports the
loading progress alongside sampling and output encoding.

### Generate an image or video

1. Enter a prompt, for example `a lovely cat` for an image model.
2. Open **Sample & Guidance** to set **Method**, **Scheduler**, **Steps**, and
   **CFG Scale** when the model guide specifies values different from the recipe
   defaults. Set the output dimensions in **Aspect Ratio & Resolution**.
3. Supply inputs for the chosen task. Editing recipes use **Reference Images**;
   IP-Adapter uses **IP-Adapter Image**. For image-to-video, use video generation
   and upload a **Start Frame**; add an **End Frame** only for a compatible
   first/last-frame workflow. Set the video length and FPS as required by the model.
4. Click **Generate**. Wait for denoising, decoding, and output encoding to
   finish, then preview or save the result. Sampling reaching 100% does not mean
   decoding has finished.

The generation status bar shows the current stage and elapsed time even when
Preview Mode is `none`. During sampling it shows completed steps for the current
pass; prompt encoding and VAE decoding show an activity indicator when their
total work is unknown. The canvas keeps the submitted output aspect ratio, and
stage changes keep the same layout. Editing the next request's resolution does
not resize the current preview or result.

Settings will offer **Notify me when generation finishes** (upcoming; requires
a frontend update). When enabled, the UI requests browser notification
permission and shows a notification when a generation job completes, fails, or
is cancelled, so you do not need to watch the tab while waiting. The
preference is saved locally. Notifications require granting browser
notification permission; if permission was previously denied, reset it in the
browser's site settings for this site before turning the setting on.

### LoRAs

To apply a LoRA, open **LoRA**, select a compatible file, and set its
**Strength**. Place files under `models/loras/` or `models/Lora/`. Browser
requests use this selector; prompt tags such as `<lora:name:1>` are CLI syntax
and are not parsed by the native server API.

### DGL editor

Open the Web UI and select **DGL** in the header. The workspace follows the
server's `--listen-ip` setting. The default `127.0.0.1` limits access to the
server machine. When the server listens on a LAN interface, open the Web UI
using the server's LAN IP and port to use the workspace from another device.
Use **Search scripts** to filter the script list, then select a saved script or
choose **New**. **Save** updates the current file or asks for a name for a new
script. Use **Save as…** to save a copy. The editor reopens the last saved script
used in that browser when it is still available. **Run** submits the current
editor text to the generation queue, including unsaved edits. The status bar
below the editor shows progress and keeps the final status and elapsed time.
The server terminal displays weight-loading throughput and sampling speed.

When a script returns values, **View result** appears in the status bar. Open it
to inspect text, scalars, tensor metadata, and encoded media. PNG appears as an
image, WebM as a video player, and AVI as a download. WebM encoding is silent.
Use AVI for a PCM16 soundtrack.

Use `image::encode` or `video::encode` and return their `{data, mime_type}`
dicts, either directly or as named fields of the root dict. To save the same
encoded output, call `media::save("output/cover.png", cover)` before returning
it. Saving is optional. The job result contains the same encoded bytes, and
polling returns them without re-encoding.

The editor underlines supported media paths and shows compact previews below
their lines. Previews align with the code indentation, and files on the same
line appear side by side when space permits. Media loads near the visible area
of the editor. An output path without a file stays collapsed, and a completed
run refreshes its preview when the file becomes available. Drag the handle in
a preview's lower-right corner to resize it, or double-click the handle to reset it.
Click an image or video preview to open it at a larger size. Expanded videos
open at the current playback position. Click the picture to play or pause, or
drag the timeline below it to seek. AVI path previews support MJPEG and
uncompressed 24-bit RGB video with up to 64 MiB of PCM or 32-bit float audio.
The speaker button mutes the preview. To adjust volume in an inline preview,
hover or focus the speaker button to open the volume slider.
The editor previews files under `input/`, `assets/`, and the configured output
directory (`output/` by default). For example, use `assets/reference.png` or
`output/clip.webm` with the default configuration. Turn off **Media view** in
the editor toolbar to hide previews. Start the server with `--no-media-view`
to disable file previews at the server.

DGL jobs share an execution context for the lifetime of the server. Compatible
scripts can reuse loaded model components across runs. The context can release
idle weights under memory pressure.

Scripts are listed and saved in the server's working directory under
`dgl/` by default. Use `--dgl-dir` to select another directory. Script paths
are relative to the server's working directory. Model files and their
dependencies must reside under one of the configured `--model-dir` roots.
Write results under `output/` by default. For example,
`image::save("output/result.png", image)` saves a PNG. Use `--output-dir` to
choose a dedicated directory for DGL writes and previews. For example, start
the server with `--model-dir ../models --output-dir renders` and use
`image::save("renders/result.png", image)` to save and preview the PNG.
Relative paths still resolve from the server's working directory. Absolute
paths under the configured output directory also work. Writes outside that
directory are denied.

## Model management

### Model directories

Repeat `--model-dir` to share multiple model directories, including an existing
SD WebUI model directory. Run these examples from the build directory:

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-server.exe --model-dir ..\models --model-dir D:\stable-diffusion-webui\models
```

**Linux/macOS**

```sh
./bin/sd-server --model-dir ../models --model-dir /data/stable-diffusion-webui/models
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
  tokenizers/
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

### Recipes

A recipe describes a model's components, supported capabilities, and generation
defaults. See the [model guides](../../docs/models/README.md) for compatible
recipes and weights. Text encoder labels in the selectors identify the expected
encoder family and size.

#### Built-in and user recipes

Built-in recipes are embedded in the binary from [recipes/](recipes/), with one JSON file per recipe. Adding or
editing a built-in requires rebuilding the server. User recipes are read from
`<model-dir>/recipes/*.json`. A valid user recipe with the same ID replaces
the entire built-in recipe. Across roots, the first valid user recipe wins;
overrides are logged. Duplicate user IDs within one root are rejected; invalid
recipes are reported without removing a valid recipe from another root or the
built-in version. Slots need only `id` and
`required`; the server supplies labels and search directories. Use `label`
only when a recipe needs a more specific display name.

Built-in additions and changes must follow the [recipe policy](recipes/README.md).
Split recipes by task weights or added capabilities, not by style or acceleration
LoRAs. Quantizations, file formats and compatible layer-count variants share a
recipe. Native encoders, projectors and connectors stay in the model's recipe.
Community editing recipes require selecting the matching LoRA separately; a
recipe does not enable it automatically.

#### Generation defaults and controls

See [generation defaults and sources](recipes/defaults.md) for the selected
publisher recommendations and reference-code defaults, with step defaults capped
at 20. Recipes link to the
original publisher's Hugging Face repository where one is available. DiT
recipes default to Euler, except for models requiring a specialized sampler.

Recipes use `features` to enable specialized controls and `disabled_features`
to turn off shared defaults. Disabled features take precedence. The UI hides
unsupported fields, empty sections and irrelevant sampler/cache options, and
omits hidden values from generation requests. See the
[feature declarations](api.md#recipe-feature-declarations) for the defaults
and complete list of supported names.

#### VAE and TAE

Compatible recipes offer a TAE slot alongside VAE. When the checkpoint does not
bundle a VAE, select at least one of VAE or TAE. The Models panel shows a reminder
below the model selectors until this requirement is met. A selected TAE is used for
encoding and decoding, including when a full VAE is also selected; use complete,
model-compatible TAE weights rather than decoder-only preview weights. Qwen
Image 2.1 uses the same recipe for its full VAE and TAEQI2.1 alternatives.

### Models loaded at startup

You can pass the same model-loading arguments used by `sd-cli` to `sd-server`.
Explicit model paths can be outside `--model-dir`. A startup model directory
includes the dependencies inside that directory. A startup shard index also
permits the shards it names within the index directory. Native generation accepts
LoRAs from `--lora-model-dir` and upscalers from `--hires-upscalers-dir`.
For example, run this Z-Image Turbo configuration from the build directory:

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-server.exe `
  --model-dir ..\models `
  --diffusion-model ..\models\diffusion_models\z_image_turbo_bf16.safetensors `
  --vae ..\models\vae\ae.sft `
  --llm ..\models\text_encoders\qwen_3_4b.safetensors `
  --offload-to-cpu `
  -v `
  --cfg-scale 1.0
```

**Linux/macOS**

```sh
./bin/sd-server \
  --model-dir ../models \
  --diffusion-model ../models/diffusion_models/z_image_turbo_bf16.safetensors \
  --vae ../models/vae/ae.sft \
  --llm ../models/text_encoders/qwen_3_4b.safetensors \
  --offload-to-cpu \
  -v \
  --cfg-scale 1.0
```

`--diffusion-model`, `--vae`, and `--llm` select the model components.
`--offload-to-cpu` keeps weights in RAM when possible to reduce VRAM pressure,
`-v` enables verbose logging, and `--cfg-scale 1.0` sets the default CFG scale.

The model specified on the command line is the initial model for API clients.
Its components load when needed. The UI offers **Current server model** with
generation defaults and controls for the active configuration. Generating with
a catalog recipe makes that selection the current model.

#### Identity inputs

PhotoMaker and PuLID use the **DGL editor**. Prepare the inputs using the
[PhotoMaker](../../docs/models/photo_maker.md) or [PuLID](../../docs/models/pulid.md)
guide, then set model and identity-input paths in the corresponding script.
The former identity-input startup arguments have been removed. Scripts read
their own inputs under the server's file-access rules.

#### Sharded checkpoints

For a checkpoint loaded from a complete repository directory, pass its model
loading arguments when starting `sd-server`, as in the model's CLI example.
Keep the index, tokenizer, and all shards together. The browser then uses
**Current server model** for that repository. Select the repository
using the startup option so the model loader can resolve all its shards.

### Switching models

Choose a recipe and component files for the next generation. A queued job that
specifies a recipe makes it the current model when it starts. Later requests
without a recipe use the current model. The server shares one execution context
with the DGL editor and reuses matching model components within its startup
memory budgets. A load failure is reported on the job that requested the model.

See [the API reference](api.md#model-management) for recipe fields, supported
slots, management endpoints, and compatibility behavior.

### Limitations

Tiny SD/Tiny SD2 and Diffusers LCM directories must first be
converted into single-file checkpoints for the file catalog. LCM checkpoints
and LoRAs use the matching SD family recipe; select LCM sampling and adjust
steps/CFG for the chosen weights.
Metadata inspection, conversion, standalone upscaling and ADetailer are CLI
workflows rather than model-loading recipes. ESRGAN and LTX latent upscalers
are selected through generation upscaling settings with `--hires-upscalers-dir`;
textual inversion uses the startup `--embd-dir` option.

SVD is not available through the current server pipeline.

## Server configuration

Reference images, masks, audio, first/last video frames, and generation options
such as upscaling are supplied with generation requests. Hardware and memory
options are startup arguments and apply to subsequent model loads.

### Launcher settings

Both source checkouts and binary release archives include `start-server.bat`
and `start-server.sh`. The launchers search for the server beside the script,
then in `bin/Release/`, `bin/`, `build/bin/Release/`, and `build/bin/`, in that order.
They print the resolved command and forward the server's exit code.

#### Custom executable

For a different build directory, set `SD_SERVER_PATH` to the executable path.
Run from the repository root:

**Windows (PowerShell)**

```powershell
$env:SD_SERVER_PATH = ".\build_vulkan\bin\Release\sd-server.exe"
.\start-server.bat --listen-port 8080
```

**Linux/macOS**

```sh
SD_SERVER_PATH="./build_vulkan/bin/sd-server" sh ./start-server.sh --listen-port 8080
```

`SD_SERVER_PATH` inherits the environment value by default. Leave it empty for
automatic discovery. An explicit invalid path does not fall back to another build.

#### Local configuration

Keep personal settings in a file beside the launchers: `start-server.local.bat`
on Windows or `start-server.local.sh` on Linux/macOS. These files are ignored by
Git and excluded from release archives. They are loaded automatically and can
set `SD_SERVER_PATH`, `MODEL_DIR`, and `COMMANDLINE_ARGS`.

**Windows (`start-server.local.bat`)**

```bat
@echo off
set "MODEL_DIR=D:\stable-diffusion-webui\models"
set "COMMANDLINE_ARGS=--listen-port 8080 --offload-to-cpu"
```

**Linux/macOS (`start-server.local.sh`)**

```sh
MODEL_DIR="/data/stable-diffusion-webui/models"
COMMANDLINE_ARGS="--listen-port 8080 --offload-to-cpu"
```

Settings are applied in this order: launcher defaults, local configuration,
then invocation arguments. Local files are shell scripts and use the
corresponding shell's syntax.

The launchers preserve the current working directory, so the default
`MODEL_DIR="./models"` is relative to where you start the launcher. To change
the first model root, set `MODEL_DIR` in the local configuration. Repeated
`--model-dir` arguments add search roots; other single-value options can override
the defaults.

In the shell script, `COMMANDLINE_ARGS` is a whitespace-separated list. Pass
values containing spaces as quoted invocation arguments instead. `MODEL_DIR`
itself supports spaces in both scripts. A leading `--version` bypasses the
configured defaults.

### Network and job queue

| Option | Default | Purpose |
| --- | --- | --- |
| `--listen-ip <ip>` | `127.0.0.1` | Server listen address |
| `--listen-port <port>` | `1234` | Server listen port |
| `--dgl-dir <path>` | `dgl` | Script directory for the DGL editor |
| `--output-dir <path>` | `output` | Directory for DGL writes and output previews |
| `--no-media-view` | Off | Disable the media view endpoint |
| `--max-pending-jobs <count>` | `64` | Maximum async jobs queued or generating |
| `--completed-ttl-seconds <seconds>` | `600` | Retention time for completed async jobs |

### Access

The default listener accepts connections from the server machine. A listener
bound to a LAN interface also accepts connections from devices that can reach
it. You can open the Web UI from a link on another site. Browser API requests
are same-origin by default. Use an IP address, `localhost`, or the hostname
configured as the listen address when connecting directly.

| Option | Default | Purpose |
| --- | --- | --- |
| `--api-key <key>` | none | Key clients must send as `Authorization: Bearer <key>` or `X-Api-Key: <key>`. Separate multiple keys with commas. |
| `--api-key-file <file>` | none | The same, one key per line, so keys don't show in the process list |
| `--cors-origins <origins>` | none | Additional browser origins allowed to make cross-origin requests, comma-separated |

With a key, the page at `/` still loads. Enter the key in **Settings → Server API Key**
and select **Apply**. The Web UI saves it for that server in the current browser tab.
Leave **Base URL** blank when connecting to the server that serves the Web UI.
DGL scripts and media previews use the server that serves the Web UI.
Authentication failures return `401` with a `WWW-Authenticate` Bearer challenge.
Protected responses include `Cache-Control: no-store`.

The server rejects requests from other browser origins with `403` unless their
origin is listed in `--cors-origins`. For a reverse proxy that preserves the
public Host header, include its full public origin in that list, such as
`https://sd.example.com`. The origin policy applies to DGL and media requests
as well as the generation APIs.

Startup fails if an access option is missing its value, a list is empty
(such as `--api-key ,`), or the key file contains no keys.

### Idle weight release

| Option | Default | Memory released when idle |
| --- | --- | --- |
| `--weight-staging-ttl <seconds>` | `-1` | GPU staging copies of weights backed by RAM, including components placed on CPU by `--params-backend cpu` or auto-fit. The next request uploads the needed weights from RAM. |
| `--weight-residency-ttl <seconds>` | `-1` | Cached model components, including RAM weights, GPU weights, staging copies, and component caches. The next request loads the needed components from their files. |

Both options also clear idle reusable buffers on supported backends after
releasing weights.

For either option, `-1` disables release, `0` releases as soon as the shared
context is idle, and a positive integer sets the idle time in seconds. The
timer starts when the last job finishes. Running and queued generation or DGL
jobs defer release. Status and progress requests do not reset the timer. If
both deadlines are due, the server unloads the model components in one step.
The configured model and the model catalog remain available after unloading.

### Logging

Logging defaults to `info`. Use `--log-level <level>` to select `debug`, `verbose`,
`info`, `warn`, or `error` (from most to least detailed). Each level includes
messages at that level and all less detailed levels. `-v` and `--verbose` are
equivalent to `--log-level verbose`. If repeated, the last logging option wins.

### Command-line help

Run from the build directory for the complete list of server and generation options:

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-server.exe --help
```

**Linux/macOS**

```sh
./bin/sd-server --help
```

## API

The server exposes these API families at the same host and port as the Web UI:

| API | Path prefix |
| --- | --- |
| Native async API | `/sdcpp/v1/...` |
| OpenAI-compatible API | `/v1/...` |
| Stable Diffusion WebUI-compatible API | `/sdapi/v1/...` |

See the [API reference](api.md) for endpoints, request fields, and examples.

## Frontend development

### Build with frontend

The server can build the frontend and embed it in the binary. Install
[Node.js](https://nodejs.org/) 20 or newer and pnpm 10 or newer:

```sh
npm install -g pnpm
node -v
pnpm -v
```

Run the following commands from the repository root. They work in PowerShell
and POSIX shells. Initialize the frontend submodule if it is missing, then
configure and build with CMake:

```sh
git submodule update --init examples/server/frontend
cmake -S . -B build -DSD_SERVER_BUILD_FRONTEND=ON
cmake --build build --config Release
```

Add the backend options from the [build guide](../../docs/build.md) when
configuring CMake. The build installs frontend dependencies, runs `pnpm run build`
and `pnpm run build:header`, and embeds `gen_index_html.h` in `sd-server`.
If pnpm is unavailable, CMake can use an existing generated header; without
either, the server builds without the embedded Web UI.

### Using an external frontend

Use `--serve-html-path` to serve a custom `index.html` instead of the embedded
frontend. For example, place it at the repository root and run from the build
directory:

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-server.exe --model-dir ..\models --serve-html-path ..\index.html
```

**Linux/macOS**

```sh
./bin/sd-server --model-dir ../models --serve-html-path ../index.html
```

This lets you develop or test the frontend without rebuilding the server binary.

### Frontend repository

The Web UI is maintained in the separate
[sdcpp-webui repository](https://github.com/leejet/sdcpp-webui), included here as
a Git submodule. Submit UI and frontend logic changes to that repository.
Changes appear here when the submodule is updated.
