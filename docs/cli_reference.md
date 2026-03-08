# CLI Reference Guide

The `sd-cli` executable provides a command-line interface to interact with diffusion models such as Stable Diffusion, SDXL, SD3, and Flux using `stable-diffusion.cpp`.

This guide provides a structured overview of the `sd-cli` parameters and common use cases.

## General Usage

```bash
./bin/sd-cli [options] -m <model_path> -p "<prompt>"
```

### Basic Options

*   `-m`, `--model <path>`: The path to the standalone diffusion model (`.safetensors`, `.gguf`, etc.). This is **required** or must be paired with specific network weights (like `--diffusion-model`).
*   `-p`, `--prompt "<text>"`: The text description of the image you want to generate.
*   `-n`, `--negative-prompt "<text>"`: Text description of what you *do not* want (default: empty).
*   `-o`, `--output <path>`: Path to write the output image (default: `./output.png`). For video sequences, you can use format specifiers like `output_%03d.png`.
*   `-H`, `--height <pixels>`: The height of the generated image (default: 512).
*   `-W`, `--width <pixels>`: The width of the generated image (default: 512).
*   `--steps <int>`: Number of sampling denoising steps (default: 20). Higher usually means better quality but slower.
*   `--cfg-scale <float>`: Unconditional guidance scale or Classifier-Free Guidance. Controls how strongly the image should follow the prompt (default: 7.0).

### Advanced Components (Text Encoders, LLMs, and VAEs)

Some models require loading their encoders and auto-encoders separated from the base standalone diffusion model:

*   `--clip_l <path>` / `--clip_g <path>` / `--t5xxl <path>`: Path to individual text encoders (often used by SD3 or Flux).
*   `--llm <path>`: Path to a generic LLM text encoder (e.g., `qwenvl2.5` for Qwen-Image or `mistral-small3.2` for Flux2).
*   `--vae <path>`: Path to a standalone VAE model if it is not embedded in the main model.
*   `--taesd <path>`: Path to a Tiny AutoEncoder for significantly faster, but lower quality, image decoding. Useful for fast previews.

### Hardware & Execution Modes

*   `-t`, `--threads <int>`: Number of CPU threads to use during computation. Default is `-1` (auto-detect physical cores).
*   `--offload-to-cpu`: Store weights in RAM and load them to VRAM on demand (useful for low VRAM systems).
*   `--fa`: Enable Flash Attention to aggressively save VRAM during generation (highly recommended).
*   `--type <type>`: Override weight precision type. E.g., `f32`, `f16`, `q4_0`, `q8_0`.

## Common Workflows

### 1. Generating a Single Image (txt2img)

```bash
./bin/sd-cli -m ../models/v1-5-pruned-emaonly.safetensors -p "a highly detailed photorealistic futuristic city"
```

### 2. Image-to-Image (img2img)

To modify an existing image rather than generating from scratch.

*   `-i`, `--init-img <path>`: Path to the base image to modify.
*   `--strength <float>`: The strength of the unnoising process. `0.0` leaves the image entirely unchanged, and `1.0` destroys all previous pixel information to generate a brand new image based on the prompt (default: 0.75).

```bash
./bin/sd-cli -m models/v1-5.safetensors -i my_photo.png -p "a cyberpunk city" --strength 0.6
```

### 3. Using LoRA Models

You can apply multiple Low-Rank Adaptation (LoRA) models on top of your main model to fine-tune styles.

*   `--lora-model-dir <dir>`: Directory containing your `.safetensors` LoRA weights.
*   To trigger the LoRA, use the standard notation inside your prompt: `<lora:name_of_file_without_extension:multiplier>`.

```bash
./bin/sd-cli -m models/v1-5.safetensors --lora-model-dir ./loras -p "a cat playing piano <lora:pixel_art_style:0.8>"
```

### 4. Flux & SD3 Usage

For larger models like SD3 or Flux, architecture requires supplying individual encoder weights and reducing CFG scale or steps based on distillation.

```bash
# Example for Flux.1 Schnell
./bin/sd-cli \
  --diffusion-model ./models/flux1-schnell.gguf \
  --vae ./models/ae.safetensors \
  --clip_l ./models/clip_l.safetensors \
  --t5xxl ./models/t5xxl_fp16.safetensors \
  -p "a wide shot of a dog on the beach" \
  -W 1024 -H 1024 \
  --steps 4 \
  --cfg-scale 1.0 \
  --sampling-method euler
```

### 5. ControlNets

ControlNets allow you to constrain generation using structural features like edges or depth maps.

*   `--control-net <path>`: Path to the ControlNet model.
*   `--control-image <path>`: Path to the conditioning layout (e.g., a canny edge image).

```bash
./bin/sd-cli -m models/v1-5.safetensors --control-net models/control_v11p_sd15_canny.safetensors --control-image room_edges.png -p "a modern living room"
```

## Advanced Samplers & Schedulers

*   `--sampling-method <name>`: Sampler algorithm. Favorites include `euler_a`, `dpm++2m`, `lcm` (default depends on the architecture).
*   `--scheduler <name>`: Define how the denoising schedule steps are distributed (e.g., `discrete`, `karras`, `exponential`, `sgm_uniform`).

To view the complete, raw list of CLI flags and fallback behaviors, run `./bin/sd-cli --help`.
