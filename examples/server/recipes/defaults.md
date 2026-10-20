# Built-in generation defaults

Reviewed against model-publisher cards and reference implementations on 2026-10-20.

These are starting values, not limits on supported resolutions or generation settings.
Built-in step defaults above 20 are reduced to 20; the cited sources may recommend more steps.
Users can still choose a higher step count.
When several aspect ratios or ranges are published, the selected example or square bucket is noted below.
A dash means no recipe override: the available primary sources did not establish a fixed value, or the size depends on the input image.
Ordinary DiT recipes explicitly use Euler; PiD retains its LCM sampling path.
Guidance-distilled models use `cfg_scale: 1` to disable external CFG in sd.cpp. FLUX embedded guidance is stored separately as `distilled_guidance`.
For community edit adapters and other optional extensions, these are the base-model settings unless the source specifies otherwise.

| Recipe | CFG | Steps | Width × height | Source / notes |
| --- | --- | --- | --- | --- |
| `anima` | 4 | 20 | 1024 × 1024 | [Source](https://huggingface.co/circlestone-labs/Anima) The official ranges are 30–50 steps / CFG 4–5; CFG 4 selects the lower end. Edit uses the same base; tune the chosen editing LoRA separately. |
| `anima-edit` | 4 | 20 | 1024 × 1024 | [Source](https://huggingface.co/circlestone-labs/Anima) The official ranges are 30–50 steps / CFG 4–5; CFG 4 selects the lower end. Edit uses the same base; tune the chosen editing LoRA separately. |
| `animatediff-sd15` | 7.5 | 20 | 512 × 512 | [Source](https://raw.githubusercontent.com/guoyww/AnimateDiff/main/configs/prompts/1_animate/1_3_animate_ToonYou.yaml)  |
| `boogu-image` | 4 | 20 | 1024 × 1024 | [Source](https://huggingface.co/Boogu/Boogu-Image-0.1-Base)  |
| `boogu-image-edit` | 5 | 20 | 1024 × 1024 | [Source](https://huggingface.co/Boogu/Boogu-Image-0.1-Edit)  |
| `chroma` | 3 | 20 | 1024 × 1024 | [Source](https://huggingface.co/lodestones/Chroma1-HD/raw/main/README.md)  |
| `chroma-radiance` | 3.5 | 20 | 1024 × 1024 | [Source](https://huggingface.co/lodestones/Chroma1-Radiance/raw/main/inference_reference_code.py)  |
| `cosxl-edit` | — | — | — | [Source](https://huggingface.co/stabilityai/cosxl) No numerical generation defaults in the author’s model card; previous test values were removed. |
| `ernie-image` | 4 | 20 | 1024 × 1024 | [Source](https://huggingface.co/baidu/ERNIE-Image)  |
| `ernie-image-turbo` | 1 | 8 | 1024 × 1024 | [Source](https://huggingface.co/baidu/ERNIE-Image-Turbo)  |
| `flex2` | 1 | 20 | 1024 × 1024 | [Source](https://huggingface.co/ostris/Flex.2-preview)  |
| `flux1-canny` | 1 | 20 | 1024 × 1024 | [Source](https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev)  |
| `flux1-depth` | 1 | 20 | 1024 × 1024 | [Source](https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev)  |
| `flux1-dev` | 1 | 20 | 1024 × 1024 | [Source](https://huggingface.co/black-forest-labs/FLUX.1-dev)  |
| `flux1-fill` | 1 | 20 | 1232 × 1632 | [Source](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev)  |
| `flux1-kontext` | 1 | 20 | — | [Source](https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/pipelines/flux/pipeline_flux_kontext.py) Output size normally follows the reference image; no fixed size is imposed by this recipe. |
| `flux1-pulid` | 1 | 20 | 1024 × 1024 | [Source](https://huggingface.co/black-forest-labs/FLUX.1-dev) FLUX.1-dev defaults; identity conditioning is provided by PuLID. |
| `flux1-schnell` | 1 | 4 | 1024 × 1024 | [Source](https://huggingface.co/black-forest-labs/FLUX.1-schnell)  |
| `flux2-dev` | 1 | 20 | 1024 × 1024 | [Source](https://huggingface.co/black-forest-labs/FLUX.2-dev)  |
| `flux2-klein-4b` | 1 | 4 | 1024 × 1024 | [Source](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)  |
| `flux2-klein-9b` | 1 | 4 | 1024 × 1024 | [Source](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)  |
| `flux2-klein-base-4b` | 4 | 20 | 1024 × 1024 | [Source](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)  |
| `flux2-klein-base-9b` | 4 | 20 | 1024 × 1024 | [Source](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B)  |
| `flux3-klein` | 1 | 16 | 1024 × 1024 | [Source](https://raw.githubusercontent.com/black-forest-labs/flux-3-image-klein/main/README.md)  |
| `flux3-klein-base` | 4 | 20 | 1024 × 1024 | [Source](https://raw.githubusercontent.com/black-forest-labs/flux-3-image-klein/main/src/flux3/pipeline.py)  |
| `hidream-o1-dev` | 1 | 20 | 2048 × 2048 | [Source](https://huggingface.co/HiDream-ai/HiDream-O1-Image-Dev)  |
| `hunyuan-video-15-i2v` | 6 | 20 | 1280 × 720 | [Source](https://huggingface.co/tencent/HunyuanVideo-1.5) 720p base I2V defaults; CFG/step-distilled and 480p checkpoints use different settings. |
| `hunyuan-video-15-t2v` | 6 | 20 | 1280 × 720 | [Source](https://huggingface.co/tencent/HunyuanVideo-1.5) 720p base T2V defaults; CFG-distilled / 480p checkpoints need their published settings. |
| `ideogram4` | 7 | 20 | 1024 × 1024 | [Source](https://raw.githubusercontent.com/ideogram-oss/ideogram4/main/docs/inference.md) Constant-guidance pipeline defaults; this recipe does not reproduce the CLI’s per-step guidance schedule. |
| `instruct-pix2pix` | — | 10 | — | [Source](https://huggingface.co/timbrooks/instruct-pix2pix) Official quick-start example; output size follows the input image. |
| `iris-3b` | 3 | 20 | 1024 × 1024 | [Source](https://huggingface.co/speridlabs/iris-3b)  |
| `krea2-raw` | 3.5 | 20 | 1024 × 1024 | [Source](https://huggingface.co/krea/Krea-2-Raw) Base model defaults; edit requires a matching community LoRA with its own settings. |
| `krea2-raw-edit` | 3.5 | 20 | 1024 × 1024 | [Source](https://huggingface.co/krea/Krea-2-Raw) Base model defaults; edit requires a matching community LoRA with its own settings. |
| `krea2-turbo` | 1 | 8 | 2048 × 2048 | [Source](https://huggingface.co/krea/Krea-2-Turbo) Official CLI example; edit requires a matching community LoRA. |
| `krea2-turbo-edit` | 1 | 8 | 2048 × 2048 | [Source](https://huggingface.co/krea/Krea-2-Turbo) Official CLI example; edit requires a matching community LoRA. |
| `lens` | 5 | 20 | 1440 × 1440 | [Source](https://huggingface.co/microsoft/Lens) Official model card: RL model, default base_resolution 1440. |
| `lens-turbo` | 1 | 4 | 1440 × 1440 | [Source](https://huggingface.co/microsoft/Lens-Turbo) Official model card: 4-step distilled model, default base_resolution 1440. |
| `lingbot-video-dense` | 3 | 20 | 832 × 480 | [Source](https://huggingface.co/robbyant/lingbot-video-dense-1.3b) Official first-stage inference example; optional refiner is separate. |
| `lingbot-video-moe` | 3 | 20 | 832 × 480 | [Source](https://huggingface.co/robbyant/lingbot-video-moe-30b-a3b) Official inference example. |
| `llada-image` | 5 | 20 | 1024 × 1024 | [Source](https://huggingface.co/inclusionAI/LLaDA-Image)  |
| `llada-image-turbo` | 1 | 4 | 1024 × 1024 | [Source](https://huggingface.co/inclusionAI/LLaDA-Image-Turbo)  |
| `longcat-image` | 4 | 20 | 1344 × 768 | [Source](https://huggingface.co/meituan-longcat/LongCat-Image)  |
| `longcat-image-edit` | 4.5 | 20 | — | [Source](https://huggingface.co/meituan-longcat/LongCat-Image-Edit) Reference-image-dependent output size; no fixed size is imposed. |
| `ltx23-dev` | 3 | 20 | 768 × 512 | [Source](https://raw.githubusercontent.com/Lightricks/LTX-2/main/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py) Official single-stage parameters, before optional latent upscaling. |
| `ltx23-distilled` | 1 | 8 | 768 × 512 | [Source](https://raw.githubusercontent.com/Lightricks/LTX-2/main/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py) Model card specifies 8 steps / CFG 1; size uses official pipeline’s first stage. |
| `ltx25-dev` | 3 | 20 | 768 × 512 | [Source](https://raw.githubusercontent.com/Lightricks/LTX-2/main/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py) Official non-distilled pipeline parameters for model versions >= 2.4, before optional upscaling. |
| `mage-flow` | 5 | 20 | 1024 × 1024 | [Source](https://huggingface.co/microsoft/Mage-Flow)  |
| `mage-flow-edit` | 5 | 20 | 1024 × 1024 | [Source](https://huggingface.co/microsoft/Mage-Flow)  |
| `mage-flow-turbo` | 1 | 4 | 1024 × 1024 | [Source](https://huggingface.co/microsoft/Mage-Flow)  |
| `ming-image` | 1 | 12 | 2048 × 2048 | [Source](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design)  |
| `minimax-h3-fl2va` | 1 | — | 1344 × 768 | [Source](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3) Guidance-distilled; official 768p canvas. No fixed step count is prescribed here; the upstream API counts sigma grid points rather than sd.cpp denoising steps. |
| `minimax-h3-ref2va` | 1 | — | 1344 × 768 | [Source](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3) Guidance-distilled; official 768p canvas. No fixed step count is prescribed here; the upstream API counts sigma grid points rather than sd.cpp denoising steps. |
| `minit2i` | 2.5 | 20 | 512 × 512 | [Source](https://huggingface.co/MiniT2I/MiniT2I/raw/main/pipeline.py) B/16 example and native model size; L/16 uses CFG 6. |
| `moss-image-base` | 3 | 20 | 1024 × 1024 | [Source](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base)  |
| `moss-image-sft` | 3 | 20 | 1024 × 1024 | [Source](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)  |
| `nitrosd-realism` | 1 | 1 | 1024 × 1024 | [Source](https://huggingface.co/ChenDY/NitroFusion)  |
| `ovis-image` | 5 | 20 | 1024 × 1024 | [Source](https://huggingface.co/ATH-MaaS/Ovis-Image-7B)  |
| `pid` | 1 | 4 | 2048 × 2048 | [Source](https://huggingface.co/nvidia/PiD) Standalone PiD decoding at 2K; guidance is handled by the upstream generator. |
| `pid-15` | 1 | 4 | 2048 × 2048 | [Source](https://huggingface.co/nvidia/PiD) Standalone PiD decoding at 2K; guidance is handled by the upstream generator. |
| `pid-15-flux2` | 1 | 4 | 2048 × 2048 | [Source](https://huggingface.co/nvidia/PiD) Standalone PiD decoding at 2K; guidance is handled by the upstream generator. |
| `pid-15-qwen-image` | 1 | 4 | 2048 × 2048 | [Source](https://huggingface.co/nvidia/PiD) Standalone PiD decoding at 2K; guidance is handled by the upstream generator. |
| `pid-15-sd3` | 1 | 4 | 2048 × 2048 | [Source](https://huggingface.co/nvidia/PiD) Standalone PiD decoding at 2K; guidance is handled by the upstream generator. |
| `pid-flux2` | 1 | 4 | 2048 × 2048 | [Source](https://huggingface.co/nvidia/PiD) Standalone PiD decoding at 2K; guidance is handled by the upstream generator. |
| `pid-qwen-image` | 1 | 4 | 2048 × 2048 | [Source](https://huggingface.co/nvidia/PiD) Standalone PiD decoding at 2K; guidance is handled by the upstream generator. |
| `pid-sd3` | 1 | 4 | 2048 × 2048 | [Source](https://huggingface.co/nvidia/PiD) Standalone PiD decoding at 2K; guidance is handled by the upstream generator. |
| `pixart-alpha-1024` | 4.5 | 20 | 1024 × 1024 | [Source](https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/pipelines/pixart_alpha/pipeline_pixart_alpha.py)  |
| `pixart-alpha-512` | 4.5 | 20 | 512 × 512 | [Source](https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/pipelines/pixart_alpha/pipeline_pixart_alpha.py)  |
| `pixart-sigma` | 4.5 | 20 | 1024 × 1024 | [Source](https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/pipelines/pixart_alpha/pipeline_pixart_sigma.py)  |
| `qwen-image` | 4 | 20 | 1328 × 1328 | [Source](https://huggingface.co/Qwen/Qwen-Image) Official square resolution bucket; true_cfg_scale maps to cfg_scale. |
| `qwen-image-21` | 1 | 20 | 2048 × 2048 | [Source](https://raw.githubusercontent.com/QwenLM/Qwen-Image-2.1/main/README.md) Original checkpoint defaults; select the matching TAEQI2.1 to replace its VAE. |
| `qwen-image-21-turbo` | 1 | 8 | 2048 × 2048 | [Model card](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo), [fixed sigmas](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/model_index.json). Standalone Turbo weights; the fixed 8-step schedule is applied on model load. |
| `qwen-image-edit` | 4 | 20 | — | [Source](https://huggingface.co/Qwen/Qwen-Image-Edit) Official editing example; output dimensions follow reference-image aspect ratio. |
| `qwen-image-edit-2509` | 4 | 20 | — | [Source](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) Official editing example; output dimensions follow reference-image aspect ratio. |
| `qwen-image-edit-2511` | 4 | 20 | — | [Source](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) Official editing example; output dimensions follow reference-image aspect ratio. |
| `qwen-image-layered` | 4 | 20 | 640 × 640 | [Source](https://huggingface.co/Qwen/Qwen-Image-Layered) Official recommended 640 resolution bucket; layered RGBA VAE is required. |
| `sd-turbo` | 1 | 1 | 512 × 512 | [Source](https://huggingface.co/stabilityai/sd-turbo)  |
| `sd1-checkpoint` | 7.5 | 20 | 512 × 512 | [Source](https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py)  |
| `sd1-controlnet` | 7.5 | 20 | 512 × 512 | [Source](https://huggingface.co/lllyasviel/control_v11p_sd15_canny)  |
| `sd1-inpainting` | 7.5 | 20 | 512 × 512 | [Source](https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py)  |
| `sd1-ip-adapter` | 7.5 | 20 | 512 × 512 | [Source](https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py) SD 1.x base pipeline defaults; adapter strength is configured separately. |
| `sd2-512-checkpoint` | 7.5 | 20 | 512 × 512 | [Source](https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py)  |
| `sd2-768-checkpoint` | 7.5 | 20 | 768 × 768 | [Source](https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py)  |
| `sd2-inpainting` | 7.5 | 20 | 512 × 512 | [Source](https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py)  |
| `sd3-medium` | 7 | 20 | 1024 × 1024 | [Source](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)  |
| `sd35-large` | 3.5 | 20 | 1024 × 1024 | [Source](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)  |
| `sd35-medium` | 4.5 | 20 | 1024 × 1024 | [Source](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)  |
| `sdxl-checkpoint` | 5 | 20 | 1024 × 1024 | [Source](https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py)  |
| `sdxl-controlnet` | 5 | 20 | 1024 × 1024 | [Source](https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py) SDXL pipeline defaults; compatible ControlNet weights are required. |
| `sdxl-inpainting` | 8 | 20 | 1024 × 1024 | [Source](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/raw/main/README.md)  |
| `sdxl-ip-adapter` | 5 | 20 | 1024 × 1024 | [Source](https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py) SDXL base pipeline defaults; adapter strength is configured separately. |
| `sdxl-photomaker` | — | — | 1024 × 1024 | [Source](https://huggingface.co/TencentARC/PhotoMaker) Use settings for the selected compatible SDXL checkpoint; no unique CFG/steps are prescribed by the model card. |
| `sdxl-turbo` | 1 | 1 | 512 × 512 | [Source](https://huggingface.co/stabilityai/sdxl-turbo)  |
| `sdxs` | 1 | 1 | 512 × 512 | [Source](https://huggingface.co/IDKiro/sdxs-512-dreamshaper)  |
| `sefi-image` | 4 | 20 | — | [Source](https://huggingface.co/SeFi-Image/SeFi-Image-5B-Base) Base/RL family defaults; Turbo checkpoints require 4 steps / CFG 1. |
| `sensenova-u15` | 4 | 20 | 2048 × 2048 | [Source](https://raw.githubusercontent.com/OpenSenseNova/SenseNova-U1/main/examples/README.md)  |
| `wan21-flf2v` | 5 | 20 | 1280 × 720 | [Source](https://raw.githubusercontent.com/Wan-Video/Wan2.1/main/generate.py)  |
| `wan21-i2v` | 5 | 20 | 832 × 480 | [Source](https://raw.githubusercontent.com/Wan-Video/Wan2.1/main/generate.py) 480p base defaults; also accepts compatible SkyReels V2 and single Wan 2.2 I2V experts. |
| `wan21-t2v` | 5 | 20 | 1280 × 720 | [Source](https://raw.githubusercontent.com/Wan-Video/Wan2.1/main/generate.py) 14B base defaults; single Wan 2.2 experts and acceleration LoRAs need matching settings. |
| `wan21-vace` | 5 | 20 | 832 × 480 | [Source](https://raw.githubusercontent.com/Wan-Video/Wan2.1/main/generate.py)  |
| `wan22-i2v` | 3.5 | 20 | 1280 × 720 | [Source](https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/wan/configs/wan_i2v_A14B.py)  |
| `wan22-s2v` | 4.5 | 20 | — | [Source](https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/wan/configs/wan_s2v_14B.py) The official S2V pipeline derives output dimensions from the input. |
| `wan22-t2v` | 3 | 20 | 1280 × 720 | [Source](https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/wan/configs/wan_t2v_A14B.py)  |
| `wan22-ti2v-5b` | 5 | 20 | 1280 × 704 | [Source](https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/wan/configs/wan_ti2v_5B.py)  |
| `z-image` | 4 | 20 | 720 × 1280 | [Source](https://huggingface.co/Tongyi-MAI/Z-Image)  |
| `z-image-l2p` | — | — | — | [Source](https://huggingface.co/zhen-nan/L2P) No numerical inference recommendations in the author’s model card; previous test values were removed. |
| `z-image-turbo` | 1 | 9 | 1024 × 1024 | [Source](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)  |

TAE alternatives use the model-compatible weights listed by [TAESD](https://github.com/madebyollin/taesd) and [TAEHV](https://github.com/madebyollin/taehv).
TAE slots are omitted for pixel-space models, PiD, layered RGBA decoding and Mage-VAE paths that the current runtime does not map to a compatible TAE.
