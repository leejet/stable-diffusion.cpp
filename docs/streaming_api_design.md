# Streaming API Extension: Design & Motivation

This document summarizes the context, motivation, and design decisions behind the **Streaming API Extensions** added to this fork of `stable-diffusion.cpp`.

> **Target audience:** This document is written for **developers and AI agents** who will maintain, extend, or merge upstream changes into this repository. If you are looking for usage instructions and code examples, see [`docs/c_api_reference.md`](./c_api_reference.md) instead.

## 1. Background & The Core Problem

The original `stable-diffusion.cpp` implementation is optimized for generating single images (or a batch of images) at a time. The standard C API functions, such as `generate_image()`, take a prompt string and perform the entire inference pipeline from scratch:
1. **Text Encoding:** Encoding the prompt using text encoders (CLIP, T5, or Qwen/LLM).
2. **Reference Image Encoding:** VAE encoding of reference images, if any.
3. **Input Frame Encoding:** VAE encoding of the initial latent space (for `img2img`).
4. **Diffusion:** Denoising the latent space (DiT / UNet steps).
5. **VAE Decoding:** Decoding the final latent space back to pixel space.

**The Problem:**
When processing a continuous video stream (real-time `img2img` generation) at a high framerate, steps 1 and 2 (Text Encoding and Reference Image Encoding) are exactly the same for every frame. 
However, the original API forces these steps to run repeatedly. For heavy LLM-based text encoders (like Qwen3 in Flux.2-klein), this redundant text encoding can take several seconds *per frame*, completely destroying the throughput required for real-time video streaming.

## 2. Objective

To achieve maximum throughput for video streams or real-time `img2img` applications using models like Flux.1/2, SD3, and others, we need to **cache the results of the text and reference image encoders**, and reuse them for every incoming video frame.

The goal was to introduce a new **"Streaming API" (Condition Caching API)** that separates the preparation phases from the hot-loop diffusion phase.

## 3. Design Principles

To ensure this fork remains maintainable and easily mergeable with the upstream `stable-diffusion.cpp` repository, the following principles were strictly adhered to:

1. **Non-Destructive Extension:** We do not modify the core logic of the existing `generate_image()` or `txt2img()` functions. The Streaming API is entirely additive.
2. **Model Agnostic:** The caching mechanism hooks into the generic `cond_stage_model` and `first_stage_model` interfaces, ensuring it works seamlessly across SD1.5, SDXL, SD3, and Flux architectures.
3. **C API Abstraction (Pimpl Idiom):** The internal C++ / ggml structures are hidden from the user. We expose opaque pointers (`sd_condition_t*` and `sd_image_latent_t*`) in `include/stable-diffusion.h` to allow usage across different languages via FFI (Rust, Python, etc.).
4. **Physical Separation of Code:** All new structural definitions and implementation logic for the Streaming API are grouped **at the very bottom** of `src/stable-diffusion.cpp`. This prevents ugly merge conflicts if the upstream author deeply modifies the internal classes (since git handles append-only changes exceedingly well).

## 4. Architecture & Memory Management

The biggest technical challenge was GGML memory management. In the original `sd.cpp`, a single temporary context (`work_ctx`) is allocated, used for the entire pipeline, and then completely freed.

To support caching, we separated the memory into two lifecycles:
*   **Computation Context:** Temporary memory used during the encoding process.
*   **Storage Context (`storage_ctx`):** Persistent memory attached to our opaque pointer wrappers. 

When `sd_encode_condition()` or `sd_encode_ref_image()` is called:
1. A temporary `work_ctx` is created for computation.
2. The model generates the condition tensors.
3. A persistent `storage_ctx` is created.
4. We use `ggml_dup_tensor()` and manually copy the memory (via `memcpy`) and names (via `ggml_set_name()`) from the computation context to the storage context.
5. The temporary context is freed, returning the wrapped storage context to the user.

**Attention to AI Agents:** 
When modifying the Streaming API, be exceptionally careful with `ggml_context` ownership. Always ensure tensors passed into the main diffusion loop belong to a context that stays alive for the duration of the loop, and do not accidentally hold pointers to a freed `work_ctx`.

## 5. Summary of New Components

You can find the full API signature in `include/stable-diffusion.h` under the `// --- Streaming API Extensions ---` block.

- `sd_condition_t`: Opaque struct holding cached prompt embeddings.
- `sd_image_latent_t`: Opaque struct holding cached reference image latents.
- `sd_encode_condition(sd_ctx, prompt, negative_prompt, width, height)`: Encodes prompts into a cacheable object. The `width` and `height` parameters are required for architectures (SDXL, SD3) that bake resolution into positional embeddings during text encoding. They must match the output resolution passed to the subsequent `sd_img2img_with_cond()` call. These parameters have no effect on Flux architectures but must still be provided.
- `sd_encode_ref_image()`: Encodes images into a cacheable object.
- `sd_img2img_with_cond()`: The hot-loop function that takes cached conditions and an input frame, running only the diffusion and decode steps.
- `sd_free_condition()` / `sd_free_image_latent()`: Memory cleanup handlers.

## 6. Known Limitations

The current Streaming API has a few hardcoded defaults to prioritize performance and simplify the initial implementation. These limitations may be addressed in future updates:

- **Distilled Guidance (Flux-specific hardcoding):** `guidance.distilled_guidance` is currently hardcoded to `3.5f` inside `sd_img2img_with_cond()`. For **Flux** models this is the intended operating value. For **non-Flux models** (SD1.5, SDXL, SD3), the `distilled_guidance` field is not consumed by those architectures' diffusion logic, so it has no practical effect—the function is still usable with those models. However, `sd_img2img_with_cond()` is primarily designed and tested for Flux. Use with other architectures may have untested edge cases.
- **Fixed Samplers:** The hot-loop function `sd_img2img_with_cond()` always uses the model's default `sample_method` and `scheduler` (via `sd_get_default_sample_method()` / `sd_get_default_scheduler()`). Custom samplers like `lcm` cannot currently be injected via the C API for this specific function.
- **Negative Prompt Null/Empty Handling:** If `negative_prompt` is `NULL` or an empty string `""`, the API encodes an empty string to generate the `uncond` embeddings. Internally, both paths encode the empty string identically—there is no behavioral difference between `NULL` and `""`. This correctly prevents crashes with CFG > 1.0.
- **Width/Height Dependency:** The `width` and `height` parameters passed to `sd_encode_condition()` must match the dimensions of `input_frame` passed to `sd_img2img_with_cond()`. See [`docs/c_api_reference.md`](./c_api_reference.md) for the full explanation of when this matters (SDXL, SD3) and when it doesn't (Flux).
