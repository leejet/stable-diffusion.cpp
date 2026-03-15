<p align="center">
  <img src="./assets/logo.png" width="360x">
</p>

# Cacheable stable-diffusion.cpp (Fork for Streaming API)

**Diffusion model (SD, Flux, Wan, ...) inference in pure C/C++**

This repository is a fork of [leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), modified to introduce a **Condition Caching (Streaming API)**.
While the upstream repo excels at stateless generation, this fork is specifically enhanced for **real-time video generation** and **high-throughput img2img streaming applications** where heavy text encoder re-evaluations (e.g., Qwen/LLM for Flux.2) become devastating bottlenecks.

By leveraging this fork's C API extensions, you can cache prompt conditions and reference images, skipping the LLM layers entirely on subsequent frames.

---

## 🚀 What's New in this Fork?

We added the **Streaming API Extensions** to the C API.
These functions allow you to encode text conditions and reference images exactly once, preserving them in a persistent GGML context. The cached representations can then be looped through `sd_img2img_with_cond` to radically increase Video-to-Video throughput.

Note: `sd_encode_condition` requires `width` and `height` parameters to match the output resolution of the subsequent `sd_img2img_with_cond()` call. These values are used for positional embeddings in SDXL/SD3 architectures and are unused but still required for Flux.

- For details on the architecture and caching mechanism: [Streaming API Design](./docs/streaming_api_design.md)

## 📚 Documentation

Detailed documentation tailored for using this repository in your own projects:

- 💻 **[C API & Streaming API Reference](./docs/c_api_reference.md)**: How to integrate the library into C/C++ projects, and full usage of the Condition Caching API.
- 🐚 **[Command-Line Interface (CLI) Guide](./docs/cli_reference.md)**: A complete reference guide for the `sd-cli` tool.
- ⚙️ **[Build Guide](./docs/build.md)**: Instructions on how to compile the project (CMake, CUDA, Vulkan, Metal).
- ⚡ **[Performance Optimization](./docs/performance.md)**: Tips for reducing VRAM and increasing generation speed.

*(Note: Additional model-specific documentation from the upstream repository is available in the `docs/` folder, such as `flux.md`, `sd3.md`, `lora.md`, etc.)*

---

## Upstream Features

This fork retains 100% compatibility with all the amazing features developed by the original `stable-diffusion.cpp` contributors:

- Plain C/C++ implementation based on [ggml](https://github.com/ggml-org/ggml), working similarly to llama.cpp.
- Super lightweight and without external dependencies.
- **Supported Models**: SD1.x, SD2.x, SDXL, SD3, FLUX.1/FLUX.2, Qwen-Image, Z-Image, Wan2.1/2.2, PhotoMaker, and more.
- **Supported Backends**: CPU (AVX2/AVX512), CUDA, Vulkan, Metal, OpenCL, SYCL.
- **Supported Formats**: Pytorch checkpoints (`.ckpt`/`.pth`), Safetensors (`.safetensors`), GGUF (`.gguf`).
- Flash Attention for aggressive memory usage optimization.
- LoRA support, ControlNet, LCM, ESRGAN upscaling, and TAESD faster latent decoding.

## Quick Start

### 1. Build from Source

Since you will likely integrate this as a backend for another project, we recommend building from source. For full instructions, see the upstream [Build Guide](./docs/build.md).

```sh
# Example: Building with Vulkan acceleration and Shared Libraries (C API)
mkdir build && cd build
cmake .. -DSD_VULKAN=ON -DSD_BUILD_SHARED_LIBS=ON
cmake --build . --config Release
# After a successful build, the CLI binary is at: build/bin/sd-cli
# The shared library is at: build/stable-diffusion.dll (Windows) or build/libstable-diffusion.so (Linux)
```

### 2. Standard CLI Usage

Download a core model file (e.g., `v1-5-pruned-emaonly.safetensors` from Hugging Face).

```sh
./bin/sd-cli -m ../models/v1-5-pruned-emaonly.safetensors -p "a lovely cat"
```

For detailed arguments and use-cases (like img2img or LoRA), check out the [CLI Guide](./docs/cli_reference.md).

### 3. Streaming API Quick Start (C/C++)

The key benefit of this fork is condition caching. Here is a minimal example:

```cpp
#include "stable-diffusion.h"

// 1. Initialize context (once)
sd_ctx_params_t ctx_params;
sd_ctx_params_init(&ctx_params);
ctx_params.diffusion_model_path = "flux-2-klein-4b.gguf";
ctx_params.vae_path = "ae.safetensors";
ctx_params.llm_path = "qwen3-4b.gguf";
ctx_params.flash_attn = true;
sd_ctx_t* ctx = new_sd_ctx(&ctx_params);

// 2. Encode prompt ONCE (the expensive LLM step)
sd_condition_t* cond = sd_encode_condition(ctx, "cinematic oil painting", "", 512, 512);

// 3. Process each video frame cheaply (no re-encoding)
while (streaming) {
    sd_image_t frame = get_next_frame();
    sd_image_t result = sd_img2img_with_cond(ctx, frame, cond, NULL, 0, 0.75f, 4, 1.0f, -1, NULL);
    render(result);
    free(result.data);
    free(frame.data);
}

// 4. Cleanup
sd_free_condition(cond);
free_sd_ctx(ctx);
```

For a full working example including reference image caching, see [`examples/stream_img2img/main.cpp`](./examples/stream_img2img/main.cpp).

## References

As this is a fork, all credits for the base architecture belong to the respective original project creators:

- [leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
- [ggml](https://github.com/ggml-org/ggml)
- [diffusers](https://github.com/huggingface/diffusers)
- [stable-diffusion](https://github.com/CompVis/stable-diffusion)
- [sd3-ref](https://github.com/Stability-AI/sd3-ref)
- [stable-diffusion-stability-ai](https://github.com/Stability-AI/stablediffusion)
- [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [k-diffusion](https://github.com/crowsonkb/k-diffusion)
- [latent-consistency-model](https://github.com/luosiallen/latent-consistency-model)
- [generative-models](https://github.com/Stability-AI/generative-models/)
- [PhotoMaker](https://github.com/TencentARC/PhotoMaker)
- [Wan2.1](https://github.com/Wan-Video/Wan2.1)
- [Wan2.2](https://github.com/Wan-Video/Wan2.2)
