# Model and component guides

Each guide covers weights, setup, and runnable examples for a model or component.
For shared generation options, device configuration, and troubleshooting, see the
[documentation index](../README.md).

When adding model support, follow the
[model documentation requirements](../../CONTRIBUTING.md#model-documentation)
for guide structure, examples, and index updates.

## Example conventions

Run the executable examples from the CMake build directory, for example after
`cd build` from the repository root. Each guide gives both command forms:

| Platform | CLI path from the build directory | Server path |
| --- | --- | --- |
| Windows, Visual Studio Release | `.\bin\Release\sd-cli.exe` | `.\bin\Release\sd-server.exe` |
| Linux/macOS | `./bin/sd-cli` | `./bin/sd-server` |

Windows examples use PowerShell backticks for line continuation. Linux/macOS
examples use POSIX shell backslashes. Backend-specific examples identify their
requirements; Metal examples apply to macOS only. A different CMake generator,
build configuration, or extracted release archive can use another executable
location; substitute that path while keeping the arguments.

Keep weights under the repository's `models/` directory, organized into the
categories described in the server guide. From the build directory these paths
start with `../models/` on Linux/macOS and `..\models\` on Windows.

Provide input images, masks, audio, or frame directories at the example paths,
or change the corresponding arguments. Paths under `../inputs/` refer to local
inputs you supply. Filenames in commands identify the example weights; adjust
paths to match your downloads. Conversion and preprocessing scripts identify
when they must be run from the source repository root instead.

The [server Web UI guide](../../examples/server/README.md#web-ui) covers file
placement, server startup, model loading, inputs, and generation. Each model
guide lists its compatible recipes and components. Choose files from the same
variant; a VAE or TAE selection must match that model.

## Image generation

- [Anima](anima.md)
- [Boogu Image](boogu_image.md)
- [Chroma](chroma.md)
- [Chroma1-Radiance](chroma_radiance.md)
- [Distilled SD and SDXL models](distilled_sd.md)
- [ERNIE-Image](ernie_image.md)
- [Fibo 2.0](fibo2.md)
- [FLUX.1](flux.md)
- [FLUX.2](flux2.md)
- [FLUX.3 Image](flux3.md)
- [HiDream-O1-Image](hidream_o1_image.md)
- [Ideogram4](ideogram4.md)
- [Iris-3B](iris.md)
- [Krea2](krea2.md)
- [Lens](lens.md)
- [LLaDA-Image](llada_image.md)
- [LongCat Image](longcat_image.md)
- [Mage-Flow](mage_flow.md)
- [Ming-Image Design](ming_image.md)
- [MiniMax-H3-Image](minimax_h3_image.md)
- [MiniT2I](minit2i.md)
- [MOSS-Image](moss_image.md)
- [Ovis-Image](ovis_image.md)
- [P-H3-Image](p_h3_image.md)
- [PiD](pid.md)
- [PixArt](pixart.md)
- [Qwen Image](qwen_image.md)
- [Qwen Image 2.1](qwen_image_2.1.md)
- [SeFi-Image](sefi_image.md)
- [SenseNova U1.5](sensenova_u1.md)
- [Stable Diffusion 1.x, 2.x, and SDXL, including Turbo variants](sd.md)
- [Stable Diffusion 3 / 3.5](sd3.md)
- [Z-Image](z_image.md)
- [Z-Image L2P](z_image_l2p.md)

## Image editing

See the [image editing guide](../guides/edit.md) for shared reference-image options
and model presets. Models that support both generation and editing use the same
model guide.

- [Boogu Image Edit](boogu_image.md#boogu-image-edit)
- [Fibo 2.0 Dev / Edit Turbo](fibo2.md#fibo-20-edit-turbo)
- [FLUX.1-Kontext-dev](kontext.md)
- [FLUX.2](flux2.md)
- [FLUX.3 Image](flux3.md)
- [LLaDA-Image editing](llada_image.md#image-editing)
- [LongCat Image Edit](longcat_image.md)
- [Mage-Flow-Edit](mage_flow.md#image-editing)
- [MiniMax-H3-Image reference-image editing](minimax_h3_image.md#reference-image-editing)
- [Qwen Image 2.1 editing](qwen_image_2.1.md#image-editing)
- [Qwen Image Edit series](qwen_image_edit.md)

## Video generation

- [HunyuanVideo 1.5](hunyuan_video.md)
- [LingBot-Video](lingbot_video.md)
- [LTX-2.3 / LTX-2.5](ltx2.md)
- [MiniMax-H3](minimax_h3.md)
- [Wan2.1 / Wan2.2](wan.md)

## Image and video super-resolution

- [ESRGAN and MoSR](esrgan.md)
- [RelayVSR](relay_vsr.md)
- [SeedVR2](seedvr2.md)

## LoRAs, adapters, and auxiliary models

- [ADetailer](adetailer.md)
- [AnimateDiff](animatediff.md)
- [IP-Adapter](ip_adapter.md)
- [LCM / LCM-LoRA](lcm.md)
- [LoRA](lora.md)
- [PhotoMaker](photo_maker.md)
- [PuLID-Flux](pulid.md)
- [TAESD and related tiny autoencoders](taesd.md)
