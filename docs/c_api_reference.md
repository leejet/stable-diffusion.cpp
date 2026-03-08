# C API & Streaming API Reference

This document serves as a comprehensive guide on how to integrate and use the C API of `stable-diffusion.cpp`, focusing particularly on the **Streaming API Extensions** introduced in this fork for real-time video processing and condition caching.

## 1. Integration & Setup

To use the C API of `stable-diffusion.cpp` in a C/C++ project, include the `stable-diffusion.h` header located in the `include/` directory.

When building your project, link against the generated `stable-diffusion` library. The necessary dependencies (like `ggml`) should be linked automatically if included as a subdirectory or package via CMake.

```cpp
#include "stable-diffusion.h"
// C API is declared under extern "C", so it works seamlessly inside C and C++ environments.
```

## 2. Core Data Structures

The main structures utilized by the standard C API include:

### `sd_image_t`
Holds a raw pixel-space image. Its fields are:
```c
typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t channel; // always 3 (RGB)
    uint8_t* data;    // caller-owned heap buffer; free with free(data)
} sd_image_t;
```

### `sd_ctx_t`
This is an opaque pointer representing the loaded Stable Diffusion model context. It encapsulates the underlying GGML logic and memory allocations.

### `sd_ctx_params_t`
Configuration parameters used when initializing a new context via `new_sd_ctx()`. Key fields include:
- `model_path`: Path to a standalone model (`.safetensors`, `.gguf`, etc.)
- `diffusion_model_path`: Path to a standalone diffusion model (e.g., for Flux)
- `vae_path`, `clip_l_path`, `t5xxl_path`, `llm_path`: Paths to individual component models
- `n_threads`: Number of CPU threads (`-1` for auto-detect)
- `flash_attn`: Enable Flash Attention for VRAM savings
- `offload_params_to_cpu`: Keep weights in RAM, load per-layer on demand

*Refer to `stable-diffusion.h` for the full list of fields and use `sd_ctx_params_init()` to initialize defaults.*

### `sd_img_gen_params_t`
Configuration for standard image generation. Key fields include `prompt`, `width`, `height`, `sample_params`, `strength` (for img2img), `ref_images`, `control_image`, and `cache`.
*Refer to `stable-diffusion.h` for the full list and use `sd_img_gen_params_init()` to initialize defaults.*

## 3. Standard API Usage

The basic workflow for single image generation (txt2img/img2img):

1. **Initialize the Context:**
   ```cpp
   sd_ctx_params_t params;
   sd_ctx_params_init(&params); // Initialize defaults

   // Pass the path to your checkpoint (.safetensors, .gguf, etc.)
   params.model_path = "path/to/model.safetensors";
   params.n_threads = -1;    // auto-detect CPU cores
   params.flash_attn = true; // recommended for VRAM savings

   sd_ctx_t* sd_ctx = new_sd_ctx(&params);
   if (!sd_ctx) {
       // Failed to load model
       return 1;
   }
   ```

2. **Setup Generation Parameters:**
   ```cpp
   sd_img_gen_params_t gen_params;
   sd_img_gen_params_init(&gen_params); // Initialize defaults

   gen_params.prompt = "a beautiful landscape, oil painting";
   gen_params.width = 512;
   gen_params.height = 512;
   gen_params.sample_params.sample_steps = 20;
   gen_params.sample_params.guidance.txt_cfg = 7.0f;
   gen_params.seed = 42;
   ```

3. **Generate Image:**
   ```cpp
   // The returned sd_image_t pointer contains an array of images (based on batch_count).
   // Returns NULL on failure (e.g., OOM, invalid params).
   sd_image_t* results = generate_image(sd_ctx, &gen_params);

   if (results) {
       for (int i = 0; i < gen_params.batch_count; i++) {
           if (results[i].data) {
               // process results[i]...
               free(results[i].data);
           }
       }
       free(results);
   }
   ```

4. **Cleanup:**
   ```cpp
   free_sd_ctx(sd_ctx);
   ```

---

## 4. 🚀 Streaming API Extensions (Condition Caching)

The standard API executes text encoding, VAE image encoding, diffusion processing, and VAE decoding in a single function call. For processing a video stream (where the prompt or reference image remains static across frames), this constant re-encoding incurs a massive performance penalty.

The **Streaming API** resolves this by splitting the pipeline, allowing you to cache conditions (like embeddings) and pass them directly into a hot-loop.

### Extended Opaque Types

- `sd_condition_t`: Holds cached text prompt embeddings (both `cond` and `uncond`). Backed by a persistent GGML storage context.
- `sd_image_latent_t`: Holds a cached reference image latent. Backed by a persistent GGML storage context.

**Error handling:** All `sd_encode_*` functions return `NULL` on failure (e.g., if the model context is invalid, or if GGML memory allocation fails). Always check the return value before entering the hot-loop.

### Preparing Caches (Run Once)

To start streaming, you must pre-calculate the text condition and any reference images you require outside of the hot-loop.

#### Encoding Text
```cpp
// Encodes the prompt via the model's text encoder (CLIP, T5, or Qwen/LLM).
// Both positive and negative prompts are encoded and stored together.
// Returns NULL on failure.
sd_condition_t* cond = sd_encode_condition(
    sd_ctx,
    "A cinematic video of a running dog",  // positive prompt
    "low quality, blurry",                 // negative prompt (or "" for empty)
    1024,                                  // output width (for positional embeddings)
    1024                                   // output height
);
if (!cond) { /* handle error */ }
```
*Note: `width` and `height` are required for architectures that incorporate resolution into positional embeddings during text encoding (SDXL, SD3). They must match the `input_frame` dimensions used in the subsequent `sd_img2img_with_cond()` call. For Flux architectures these values are unused, but the parameters are still required.*

#### Encoding Reference Images (optional)
```cpp
// Encodes an image through the VAE and returns persistent latents.
// Pass NULL/0 for ref_latents/n_ref_latents in sd_img2img_with_cond if not needed.
// Returns NULL on failure.
sd_image_latent_t* ref_latent = sd_encode_ref_image(
    sd_ctx,
    &my_reference_sd_image
);
if (!ref_latent) { /* handle error */ }
```

### The Hot Loop (Run Per Frame)

Use `sd_img2img_with_cond` to perform diffusion on the incoming video frame using the cached conditions. This severely reduces latency per frame.

**Ownership rules for `sd_img2img_with_cond`:**
- `input_frame` — caller owns the buffer. The function does **not** take ownership; free `input_frame.data` after each call.
- Return value — the `sd_image_t` is returned by value; its `data` buffer is heap-allocated and **must** be freed by the caller. `data` will be `NULL` on failure.

```cpp
// Without reference images (pass NULL and 0):
// sd_image_t output = sd_img2img_with_cond(sd_ctx, frame, cond, NULL, 0, ...);

// With one or more reference images:
sd_image_latent_t* ref_latents[] = { ref_latent };

while (streaming) {
    sd_image_t input_frame = get_next_video_frame(); // User implementation

    // Diffusion + decode step only (text encoding and ref image encoding are skipped)
    sd_image_t output_frame = sd_img2img_with_cond(
        sd_ctx,
        input_frame,             // The incoming frame to denoise
        cond,                    // Cached text conditions
        ref_latents,             // Cached reference latents (or NULL if unused)
        1,                       // Number of reference latents (or 0 if unused)
        0.75f,                   // Strength (0.0 = input image, 1.0 = full generation)
        4,                       // Sample steps (keep low for speed, e.g. 4 for Flux.2-klein)
        1.0f,                    // CFG Scale
        42                       // RNG Seed (-1 for random)
    );

    if (output_frame.data) {
        render_frame(output_frame); // User implementation
        free(output_frame.data);
    }

    free(input_frame.data);
}
```

### Cleanup

Once streaming is finished, you must free the cached structures to prevent GGML memory leaks.

```cpp
sd_free_condition(cond);
sd_free_image_latent(ref_latent);
free_sd_ctx(sd_ctx);
```

For a complete working example, please refer to: [`examples/stream_img2img/main.cpp`](../examples/stream_img2img/main.cpp).
