# SageAttention

`--sage-attn` enables native CUDA SageAttention in the diffusion model, including
the high-noise diffusion model when present. Python, PyTorch, and Triton are not
required at build time or runtime.

The CUDA backend automatically selects a kernel supported by both the GPU and
the compiled CUDA toolkit:

| GPU / toolkit | Implementation |
| --- | --- |
| SM89 or newer, CUDA 12.8 or newer (except SM90) | SageAttention2++: per-thread INT8 Q/K, FP8 PV, FP16 instruction accumulation with an FP32 buffer |
| SM89 or newer, CUDA 12.4 or newer; SM90 also uses this path with newer toolkits | SageAttention2: per-thread INT8 Q/K, FP8 PV, two-level FP32 accumulation |
| SM80 or newer, CUDA 12.0 or newer | INT8 Q/K, FP16 PV compatibility path |

The FP8 paths smooth K, quantize V per channel, and pad and permute V for FP8
Tensor Cores. The 2++ path uses the upstream V scale limit of 2.25 to avoid
overflow in its FP16 instruction accumulator. The public output remains FP32.
These are the upstream **INT8** SageAttention2/2++ variants; the paper's INT4
variant and Hopper-specific WGMMA kernel are not implemented here.

## Build

Use the bundled patched GGML, CUDA Toolkit 12.0 or newer, and an NVIDIA GPU with
compute capability 8.0 or newer. Compile kernels for the GPU being used.

```sh
cmake -S . -B build -DSD_CUDA=ON -DSD_USE_UPSTREAM_GGML=OFF
cmake --build build --config Release
```

No separate SageAttention build option is needed. Upstream GGML builds do not
support it. A system GGML must include the matching patched API and CUDA
backend. Enabling `--sage-attn` with an unavailable build or diffusion device
reports an error. Building with CUDA 12.4 selects SageAttention2 on an RTX 4090;
rebuild with CUDA 12.8 or newer to use SageAttention2++.

## Use

Replace `--diffusion-fa` with `--sage-attn` in an existing command. For example,
from the build directory:

```powershell
.\bin\Release\sd-cli.exe -M vid_gen --diffusion-model  ..\models\diffusion_models\Wan2.2-T2V-A14B-LowNoise-Q8_0.gguf --high-noise-diffusion-model  ..\models\diffusion_models\Wan2.2-T2V-A14B-HighNoise-Q8_0.gguf --vae ..\models\vae\wan_2.1_vae.safetensors --t5xxl ..\models\text_encoders\umt5-xxl-encoder-Q8_0.gguf  -p "a lovely cat" --cfg-scale 3.5 --sampling-method euler --steps 10 --high-noise-cfg-scale 3.5 --high-noise-sampling-method euler --high-noise-steps 8 -v -n "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，
形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" -W 832 -H 480 --diffusion-fa --offload-to-cpu --video-frames 33 --sage-attn
```

SageAttention currently handles unmasked attention with head dimensions from
1 through 128, including grouped-query attention, different query/key lengths,
and multiple batches. Dimensions below 64 are zero-padded to 64; dimensions
between 65 and 127 are zero-padded to 128. The original softmax scale is preserved,
and the output is cropped back to the original dimension. Other attention
operations fall back to FlashAttention when supported, then ordinary attention.
SageAttention takes precedence in diffusion
when combined with `--fa` or `--diffusion-fa`; `--fa` continues to control other
modules. Existing attention scaling overrides remain effective.

Attention quantization changes numerical results. Compare image quality and
end-to-end generation time using the same seed, dimensions, and sampling
settings. Compare sampling steps after the first step for warmed-up inference
speed, and report model loading and first-step initialization separately.
Quantization, smoothing, and format conversion costs are included in generation
time, so short sequences may not benefit.

Library callers set `sd_ctx_params_t.sage_attn = true` before `new_sd_ctx()`,
like `diffusion_flash_attn`. Context creation fails if the requested feature is
unavailable. Initialize the parameter structure with `sd_ctx_params_init()`.
Rebuild library callers against the updated public header.
