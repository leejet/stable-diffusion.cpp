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

### `sd_ctx_t`
This is an opaque pointer representing the loaded Stable Diffusion model context. It encapsulates the underlying GGML logic and memory allocations.

### `sd_ctx_params_t`
Configuration parameters used when initializing a new context via `new_sd_ctx()`. Includes configurations for VAE decoding threads, model quantization, and RNG backend preferences.

### `sd_img_gen_params_t`
Configuration mapped for standard image generation parameters like `prompt`, `negative_prompt`, `width`, `height`, `sample_steps`, `cfg_scale`, `seed`.

## 3. Standard API Usage

The basic workflow for single image generation (txt2img/img2img):

1. **Initialize the Context:**
   ```cpp
   sd_ctx_params_t params;
   sd_ctx_params_init(&params); // Initialize defaults

   // Pass the path to your checkpoint (.safetensors, .gguf, etc.)
   sd_ctx_t* ctx = new_sd_ctx("path/to/model.safetensors", params);
   ```

2. **Setup Generation Parameters:**
   ```cpp
   sd_img_gen_params_t gen_params;
   sd_img_gen_params_init(&gen_params); // Initialize defaults
   
   gen_params.prompt = "a beautiful landscape, oil painting";
   gen_params.width = 512;
   gen_params.height = 512;
   gen_params.sample_steps = 20;
   gen_params.cfg_scale = 7.0f;
   ```

3. **Generate Image:**
   ```cpp
   // The returned sd_image_t contains width, height, channel count and a raw pixels buffer.
   sd_image_t result = generate_image(ctx, &gen_params);

   // The caller must free result.data when finished.
   free(result.data);
   ```

4. **Cleanup:**
   ```cpp
   free_sd_ctx(ctx);
   ```

---

## 4. 🚀 Streaming API Extensions (Condition Caching)

The standard API executes text encoding, VAE image encoding, diffusion processing, and VAE decoding in a single function call. For processing a video stream (where the prompt or reference image remains static across frames), this constant re-encoding incurs a massive performance penalty.

The **Streaming API** resolves this by splitting the pipeline, allowing you to cache conditions (like embeddings) and pass them directly into a hot-loop.

### Extended Opaque Types

- `sd_condition_t`: Holds cached text prompt embeddings.
- `sd_image_latent_t`: Holds cached reference image latents.

### Preparing Caches (Run Once)

To start streaming, you must pre-calculate the text condition and any reference images you require outside of the hot-loop.

#### Encoding Text
```cpp
// Encodes the prompt via the model's text encoder (CLIP, T5, or Qwen/LLM).
// The resulting sd_condition_t* is bound to persistent memory.
sd_condition_t* cond = sd_encode_condition(
    ctx, 
    "A cinematic video of a running dog", 
    "low quality, blurry"
);
```

#### Encoding Reference Images (For in-context conditioning or ControlNets)
```cpp
// Encodes an image through the VAE and returns latents.
sd_image_latent_t* ref_latent = sd_encode_ref_image(
    ctx, 
    &my_reference_sd_image
);
```

### The Hot Loop (Run Per Frame)

Use `sd_img2img_with_cond` to perform diffusion on the incoming video frame using the cached conditions. This severely reduces latency per frame.

```cpp
sd_image_latent_t* ref_latents[] = { ref_latent }; // Can pass multiple references if needed.

while (streaming) {
    sd_image_t input_frame = get_next_video_frame(); // User implementation
    
    // Diffusion step only
    sd_image_t output_frame = sd_img2img_with_cond(
        ctx,
        input_frame,             // The incoming frame to denoise
        cond,                    // Cached text conditions
        ref_latents,             // Cached reference latents
        1,                       // Number of reference latents
        0.75f,                   // Strength (0.0 = input image, 1.0 = full generation)
        4,                       // Sample steps (Keep low for speed, e.g. 4 for Flux.2-klein)
        1.0f,                    // CFG Scale
        42                       // RNG Seed (-1 for random)
    );
    
    render_frame(output_frame); // User implementation
    
    free(input_frame.data);
    free(output_frame.data);
}
```

### Cleanup

Once streaming is finished, you must free the cached structures to prevent GGML memory leaks.

```cpp
sd_free_condition(cond);
sd_free_image_latent(ref_latent);
free_sd_ctx(ctx);
```

For a complete working example, please refer to: `examples/stream_img2img/main.cpp`.
