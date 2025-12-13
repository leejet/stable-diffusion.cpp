# Run

```
usage: ./bin/sd-server  [options]

Svr Options:
  -l, --listen-ip <string>    server listen ip (default: 127.0.0.1)
  --listen-port <int>         server listen port (default: 1234)
  -v, --verbose               print extra info
  --color                     colors the logging tags according to level
  -h, --help                  show this help message and exit

Context Options:
  -m, --model <string>                     path to full model
  --clip_l <string>                        path to the clip-l text encoder
  --clip_g <string>                        path to the clip-g text encoder
  --clip_vision <string>                   path to the clip-vision encoder
  --t5xxl <string>                         path to the t5xxl text encoder
  --llm <string>                           path to the llm text encoder. For example: (qwenvl2.5 for qwen-image, mistral-small3.2 for flux2, ...)
  --llm_vision <string>                    path to the llm vit
  --qwen2vl <string>                       alias of --llm. Deprecated.
  --qwen2vl_vision <string>                alias of --llm_vision. Deprecated.
  --diffusion-model <string>               path to the standalone diffusion model
  --high-noise-diffusion-model <string>    path to the standalone high noise diffusion model
  --vae <string>                           path to standalone vae model
  --taesd <string>                         path to taesd. Using Tiny AutoEncoder for fast decoding (low quality)
  --control-net <string>                   path to control net model
  --embd-dir <string>                      embeddings directory
  --lora-model-dir <string>                lora model directory
  --tensor-type-rules <string>             weight type per tensor pattern (example: "^vae\.=f16,model\.=q8_0")
  --photo-maker <string>                   path to PHOTOMAKER model
  --upscale-model <string>                 path to esrgan model.
  -t, --threads <int>                      number of threads to use during computation (default: -1). If threads <= 0, then threads will be set to the number of
                                           CPU physical cores
  --chroma-t5-mask-pad <int>               t5 mask pad size of chroma
  --vae-tile-overlap <float>               tile overlap for vae tiling, in fraction of tile size (default: 0.5)
  --flow-shift <float>                     shift value for Flow models like SD3.x or WAN (default: auto)
  --vae-tiling                             process vae in tiles to reduce memory usage
  --force-sdxl-vae-conv-scale              force use of conv scale on sdxl vae
  --offload-to-cpu                         place the weights in RAM to save VRAM, and automatically load them into VRAM when needed
  --control-net-cpu                        keep controlnet in cpu (for low vram)
  --clip-on-cpu                            keep clip in cpu (for low vram)
  --vae-on-cpu                             keep vae in cpu (for low vram)
  --diffusion-fa                           use flash attention in the diffusion model
  --diffusion-conv-direct                  use ggml_conv2d_direct in the diffusion model
  --vae-conv-direct                        use ggml_conv2d_direct in the vae model
  --chroma-disable-dit-mask                disable dit mask for chroma
  --chroma-enable-t5-mask                  enable t5 mask for chroma
  --type                                   weight type (examples: f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0, q2_K, q3_K, q4_K). If not specified, the default is the
                                           type of the weight file
  --rng                                    RNG, one of [std_default, cuda, cpu], default: cuda(sd-webui), cpu(comfyui)
  --sampler-rng                            sampler RNG, one of [std_default, cuda, cpu]. If not specified, use --rng
  --prediction                             prediction type override, one of [eps, v, edm_v, sd3_flow, flux_flow, flux2_flow]
  --lora-apply-mode                        the way to apply LoRA, one of [auto, immediately, at_runtime], default is auto. In auto mode, if the model weights
                                           contain any quantized parameters, the at_runtime mode will be used; otherwise,
                                           immediately will be used.The immediately mode may have precision and
                                           compatibility issues with quantized parameters, but it usually offers faster inference
                                           speed and, in some cases, lower memory usage. The at_runtime mode, on the
                                           other hand, is exactly the opposite.
  --vae-tile-size                          tile size for vae tiling, format [X]x[Y] (default: 32x32)
  --vae-relative-tile-size                 relative tile size for vae tiling, format [X]x[Y], in fraction of image size if < 1, in number of tiles per dim if >=1
                                           (overrides --vae-tile-size)
```