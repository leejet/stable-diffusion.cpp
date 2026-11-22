# DGL examples

Choose a runnable script for [image generation and editing](#image-generation-and-editing),
[video and audio generation](#video-and-audio-generation), or
[adapters, refinement and upscaling](#adapters-refinement-and-upscaling).
Each model name links to its guide for weights and setup.

## Run an example

Open the script's `var` block and set the model paths, input files, prompt and
output path. Run the following commands from the CMake build directory. Relative
paths in the script resolve from that directory.

**Windows (PowerShell, Visual Studio Release)**

```powershell
.\bin\Release\sd-cli.exe -M dgl --script ..\docs\dgl\sd\sd1-5-txt2img.dgl --check
.\bin\Release\sd-cli.exe -M dgl --script ..\docs\dgl\sd\sd1-5-txt2img.dgl
```

**Linux/macOS**

```sh
./bin/sd-cli -M dgl --script ../docs/dgl/sd/sd1-5-txt2img.dgl --check
./bin/sd-cli -M dgl --script ../docs/dgl/sd/sd1-5-txt2img.dgl
```

The first command checks the script without loading weights. The second runs it
and writes to the output path in the script. Set context options such as
`--max-vram` and `--auto-fit` on the command line. In the server's DGL editor,
choose model paths under a configured model directory and save results under
`output/`. See the [server guide](../../examples/server/README.md#dgl-editor).

For a large video input, set `max_frames` on `video::load` to limit how many
frames it decodes. See [video input formats and limits](../dgl.md#reading-video)
and [NumPy tensor files](../dgl.md#tensor-files).

## Model examples

### Image generation and editing

| Model guide | Scripts |
| --- | --- |
| [Anima](../models/anima.md) | [Anima Preview text to image](anima/anima-preview-txt2img.dgl) |
| [Boogu Image](../models/boogu_image.md) | [Text to image](boogu_image/boogu-image-base-txt2img.dgl) · [Image editing](boogu_image/boogu-image-edit.dgl) |
| [Chroma](../models/chroma.md) | [Unlocked v40 text to image](chroma/chroma-unlocked-v40-txt2img.dgl) |
| [Chroma1-Radiance](../models/chroma_radiance.md) | [Text to image](chroma_radiance/chroma1-radiance-v0-4-txt2img.dgl) |
| [Distilled Stable Diffusion models](../models/distilled_sd.md) | [SSD1B](distilled_sd/ssd1b-txt2img.dgl) · [SDXS-512-DreamShaper](distilled_sd/sdxs-512-dreamshaper-txt2img.dgl) |
| [ERNIE-Image](../models/ernie_image.md) | [ERNIE-Image Turbo](ernie_image/ernie-image-turbo-txt2img.dgl) · [ERNIE-Image](ernie_image/ernie-image-txt2img.dgl) |
| [Fibo 2.0](../models/fibo2.md) | [Dev: text to image](fibo2/fibo2-dev-txt2img.dgl) · [Dev: image editing](fibo2/fibo2-dev-edit.dgl) · [Edit Turbo](fibo2/fibo2-edit-turbo.dgl) |
| [FLUX.1](../models/flux.md) | [FLUX.1 dev](flux/flux1-dev-txt2img.dgl) · [FLUX.1 schnell](flux/flux1-schnell-txt2img.dgl) · [FLUX.1 dev with LoRA](flux/flux1-dev-lora.dgl) |
| [FLUX.2 dev](../models/flux2.md) | [Image editing](flux2/flux2-dev-edit.dgl) |
| [FLUX.2 Klein 4B](../models/flux2.md) | [Text to image](flux2/flux2-klein-4b-txt2img.dgl) · [Image editing](flux2/flux2-klein-4b-edit.dgl) · [Base text to image](flux2/flux2-klein-base-4b-txt2img.dgl) |
| [FLUX.2 Klein 9B](../models/flux2.md) | [Text to image](flux2/flux2-klein-9b-txt2img.dgl) · [Image editing](flux2/flux2-klein-9b-edit.dgl) · [Base text to image](flux2/flux2-klein-base-9b-txt2img.dgl) |
| [FLUX.3 Image](../models/flux3.md) | [Klein: text to image](flux3/flux3-image-klein-txt2img.dgl) · [Klein: image editing](flux3/flux3-image-klein-edit.dgl) · [Klein Base: text to image](flux3/flux3-image-klein-base-txt2img.dgl) |
| [HiDream-O1-Image](../models/hidream_o1_image.md) | [Text to image](hidream_o1_image/hidream-o1-image-dev-txt2img.dgl) |
| [Ideogram4](../models/ideogram4.md) | [Text to image](ideogram4/ideogram4-txt2img.dgl) |
| [Iris-3B](../models/iris.md) | [Text to image](iris/iris-3b-txt2img.dgl) |
| [FLUX.1 Kontext](../models/kontext.md) | [Edit with FLUX.1 Kontext dev](kontext/flux1-kontext-dev-edit.dgl) |
| [Krea2](../models/krea2.md) | [Raw text to image](krea2/krea2-raw-txt2img.dgl) |
| [Lens](../models/lens.md) | [Lens](lens/lens-txt2img.dgl) · [Lens Turbo](lens/lens-turbo-txt2img.dgl) |
| [LLaDA-Image](../models/llada_image.md) | [Text to image](llada_image/llada-image-turbo-txt2img.dgl) · [Image editing](llada_image/llada-image-turbo-edit.dgl) |
| [LongCat Image](../models/longcat_image.md) | [Text to image](longcat_image/longcat-image-txt2img.dgl) |
| [Mage-Flow](../models/mage_flow.md) | [Text to image](mage_flow/mage-flow-turbo-txt2img.dgl) · [Image editing](mage_flow/mage-flow-edit.dgl) |
| [Ming-Image](../models/ming_image.md) | [0.1 Design text to image](ming_image/ming-image-0-1-design-txt2img.dgl) |
| [MiniMax-H3-Image](../models/minimax_h3_image.md) | [Text to image](minimax_h3_image/minimax-h3-image-txt2img.dgl) · [Image editing](minimax_h3_image/minimax-h3-image-edit.dgl) |
| [MiniT2I](../models/minit2i.md) | [CUDA](minit2i/minit2i-cuda-txt2img.dgl) · [Metal](minit2i/minit2i-metal-txt2img.dgl) (`--threads 8`) |
| [MOSS-Image](../models/moss_image.md) | [Base: text to image](moss_image/moss-image-base-txt2img.dgl) · [SFT: text to image](moss_image/moss-image-sft-txt2img.dgl) · [Inline images](moss_image/moss-image-sft-inline-images.dgl) · [Image editing (HQ)](moss_image/moss-image-edit.dgl) · [Image editing (chinoll)](moss_image/moss-image-edit-chinoll.dgl) |
| [Ovis-Image](../models/ovis_image.md) | [Text to image](ovis_image/ovis-image-txt2img.dgl) |
| [P-H3-Image](../models/p_h3_image.md) | [Text to image](p_h3_image/p-h3-image-txt2img.dgl) · [Image editing](p_h3_image/p-h3-image-edit.dgl) |
| [PiD](../models/pid.md) | [512 to 2048 pixels](pid/pid-flux1-512-to-2048-txt2img.dgl) |
| [PixArt](../models/pixart.md) | [Sigma XL2 at 1024 pixels](pixart/pixart-sigma-xl2-1024-ms-txt2img.dgl) |
| [Qwen Image](../models/qwen_image.md) | [Text to image](qwen_image/qwen-image-txt2img.dgl) |
| [Qwen Image 2.1](../models/qwen_image_2.1.md) | [Text to image](qwen_image_2.1/qwen-image-2-1-txt2img.dgl) · [Image editing](qwen_image_2.1/qwen-image-2-1-edit.dgl) |
| [Qwen Image Edit](../models/qwen_image_edit.md) | [Qwen Image Edit](qwen_image_edit/qwen-image-edit.dgl) · [Qwen Image Edit 2509](qwen_image_edit/qwen-image-edit-2509.dgl) · [Qwen Image Edit 2511](qwen_image_edit/qwen-image-edit-2511.dgl) |
| [Stable Diffusion – generation](../models/sd.md) | [SD 1.4](sd/sd1-4-txt2img.dgl) · [SD 1.5](sd/sd1-5-txt2img.dgl) · [SDXL](sd/sdxl-txt2img.dgl) · [NitroFusion](sd/nitrosd-realism-nitrofusion.dgl) |
| [Stable Diffusion – image conditioning](../models/sd.md) | [SD 1.4 image to image](sd/sd1-4-img2img.dgl) · [SD 1.5 ControlNet](sd/sd1-5-controlnet.dgl) · [SD 1.5 inpainting](sd/sd1-5-inpainting.dgl) |
| [Stable Diffusion 3](../models/sd3.md) | [SD 3.5 Large text to image](sd3/sd3-5-large-txt2img.dgl) |
| [SeFi-Image](../models/sefi_image.md) | [1B Turbo](sefi_image/sefi-1b-turbo-txt2img.dgl) · [1B Base](sefi_image/sefi-1b-base-txt2img.dgl) · [5B Turbo](sefi_image/sefi-5b-turbo-txt2img.dgl) (`--max-vram 8`) · [5B RL](sefi_image/sefi-image-5b-rl.dgl) |
| [SenseNova U1.5](../models/sensenova_u1.md) | [Text to image on CUDA](sensenova_u1/sensenova-u1-cuda-txt2img.dgl) |
| [Z-Image](../models/z_image.md) | [Turbo: text to image](z_image/z-image-turbo-txt2img.dgl) · [Base: text to image](z_image/z-image-base-txt2img.dgl) · [Turbo with ControlNet](z_image/z-image-turbo-controlnet.dgl) |
| [Z-Image L2P](../models/z_image_l2p.md) | [Text to image](z_image_l2p/z-image-l2p-txt2img.dgl) |

### Video and audio generation

| Model guide | Scripts |
| --- | --- |
| [AnimateDiff – text to video](../models/animatediff.md) | [Text to video with CPU offload](animatediff/animatediff-realisticvision-v6-txt2video-cpu-offload.dgl) · [Text to video, 16 frames](animatediff/animatediff-realisticvision-v6-txt2video-16-frames.dgl) · [Text to video with a VRAM budget](animatediff/animatediff-realisticvision-v6-txt2video-vram-budget.dgl) (`--max-vram 2.0`) · [Domain Adapter LoRA](animatediff/animatediff-realisticvision-v6-domain-lora.dgl) |
| [AnimateDiff – image to video](../models/animatediff.md) | [Generate the initial image](animatediff/realisticvision-v6-txt2img.dgl) · [Image to video](animatediff/animatediff-realisticvision-v6-img2video.dgl) |
| [HunyuanVideo 1.5](../models/hunyuan_video.md) | [720p text to video](hunyuan_video/hunyuanvideo1-5-720p-txt2video.dgl) |
| [LingBot-Video](../models/lingbot_video.md) | [Dens 1.3B text to video](lingbot_video/lingbot-video-dens-1-3b-txt2video.dgl) |
| [LTX-2.3](../models/ltx2.md) | [2.3 text to video](ltx2/ltx-2-3-dev-txt2video.dgl) · [2.3 image to video](ltx2/ltx-2-3-dev-img2video.dgl) · [2.3 first/last frames](ltx2/ltx-2-3-dev-first-last-frame.dgl) · [2.3 spatial latent upscale](ltx2/ltx-2-3-spatial-upscale.dgl) |
| [LTX-2.5](../models/ltx2.md) | [2.5 text to video](ltx2/ltx-2-5-dev-txt2video.dgl) · [2.5 image to video](ltx2/ltx-2-5-dev-img2video.dgl) · [2.5 spatial latent upscale](ltx2/ltx-2-5-spatial-upscale.dgl) |
| [MiniMax-H3](../models/minimax_h3.md) | [Text to audio/video](minimax_h3/minimax-h3-text-to-av.dgl) · [First/last-frame conditioning](minimax_h3/minimax-h3-first-last-frame.dgl) · [Reference to audio/video](minimax_h3/minimax-h3-reference-to-av.dgl) · [Interpolated latent hires](minimax_h3/minimax-h3-latent-hires.dgl) · [Learned latent hires](minimax_h3/minimax-h3-learned-hires.dgl) |
| [Wan 2.1](../models/wan.md) | [Text to video, 1.3B](wan/wan2-1-t2v-1-3b.dgl) · [Text to video, 14B](wan/wan2-1-t2v-14b.dgl) · [Image to video, 14B](wan/wan2-1-i2v-14b.dgl) · [First/last frames, 14B](wan/wan2-1-first-last-frame-14b.dgl) |
| [Wan 2.1 VACE 1.3B](../models/wan.md) | [Text to video](wan/wan2-1-vace-1-3b-txt2video.dgl) · [Reference to video](wan/wan2-1-vace-1-3b-ref2video.dgl) · [Video to video](wan/wan2-1-vace-1-3b-video2video.dgl) |
| [Wan 2.1 VACE 14B](../models/wan.md) | [Text to video](wan/wan2-1-vace-14b-txt2video.dgl) · [Reference to video](wan/wan2-1-vace-14b-ref2video.dgl) · [Video to video](wan/wan2-1-vace-14b-video2video.dgl) |
| [Wan 2.2 A14B](../models/wan.md) | [Text to video](wan/wan2-2-t2v-a14b.dgl) · [Image to video](wan/wan2-2-i2v-a14b.dgl) · [Single image](wan/wan2-2-t2v-a14b-image.dgl) · [Text to video with LoRA](wan/wan2-2-t2v-a14b-lora.dgl) · [First/last frames](wan/wan2-2-first-last-frame-a14b.dgl) |
| [Wan 2.2 TI2V 5B](../models/wan.md) | [Text to video](wan/wan2-2-ti2v-5b-txt2video.dgl) · [Image to video](wan/wan2-2-ti2v-5b-img2video.dgl) |
| [Wan 2.2 S2V](../models/wan.md) | [Speech to video, 14B](wan/wan2-2-s2v-14b.dgl) |
| [Wan 2.2 VACE Fun A14B](../models/wan.md) | [Reference to video](wan/wan2-2-vace-fun-a14b-ref2video.dgl) |

### Adapters, refinement and upscaling

| Model guide | Scripts |
| --- | --- |
| [ADetailer](../models/adetailer.md) | [Refine an existing image](adetailer/adetailer-existing-image.dgl) · [Generate and refine](adetailer/adetailer-generated-image.dgl) |
| [ESRGAN and MoSR](../models/esrgan.md) | [Generate and upscale](esrgan/sd1-5-txt2img-esrgan.dgl) · [Upscale an image](esrgan/realesrgan-x4plus-anime-6b-upscale.dgl) |
| [IP-Adapter](../models/ip_adapter.md) | [SD 1.5 Classic](ip_adapter/ip-adapter-sd1-5.dgl) · [SDXL Classic](ip_adapter/ip-adapter-sdxl.dgl) · [SD 1.5 Plus](ip_adapter/ip-adapter-plus-sd1-5.dgl) · [SDXL with ControlNet](ip_adapter/ip-adapter-sdxl-controlnet.dgl) |
| [LCM and LCM-LoRA](../models/lcm.md) | [LCM-LoRA with SD 1.5](lcm/lcm-lora-sd1-5.dgl) |
| [LoRA](../models/lora.md) | [SD 1.5 UNet and text encoder](lora/sd1-5-lora.dgl) |
| [PhotoMaker](../models/photo_maker.md) | [PhotoMaker v1](photo_maker/photomaker-v1-sdxl.dgl) · [PhotoMaker v2](photo_maker/photomaker-v2-sdxl.dgl) |
| [PuLID-Flux](../models/pulid.md) | [Identity-guided generation](pulid/pulid-flux1-schnell.dgl) |
| [RelayVSR](../models/relay_vsr.md) | [Upscale a video](relay_vsr/relayvsr-x2-video-upscale.dgl) · [Upscale an image](relay_vsr/relayvsr-x2-image-upscale.dgl) |
| [SeedVR2](../models/seedvr2.md) | [Upscale an image](seedvr2/seedvr2-3b-image-upscale.dgl) · [Upscale a video](seedvr2/seedvr2-3b-video-upscale.dgl) |
| [TAESD](../models/taesd.md) | [SD 1.5 decoding with TAESD](taesd/sd1-5-taesd-preview.dgl). The CLI writes the final PNG. Intermediate previews require a [host preview callback](../dgl.md#progress-preview-and-cancellation) in `tae` mode. |
