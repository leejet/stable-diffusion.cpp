# Run

```
usage: ./bin/sd  [options]

Options:
  -m, --model <string>                     path to full model
  --clip_l <string>                        path to the clip-l text encoder
  --clip_g <string>                        path to the clip-g text encoder
  --clip_vision <string>                   path to the clip-vision encoder
  --t5xxl <string>                         path to the t5xxl text encoder
  --qwen2vl <string>                       path to the qwen2vl text encoder
  --qwen2vl_vision <string>                path to the qwen2vl vit
  --diffusion-model <string>               path to the standalone diffusion model
  --high-noise-diffusion-model <string>    path to the standalone high noise diffusion model
  --vae <string>                           path to standalone vae model
  --taesd <string>                         path to taesd. Using Tiny AutoEncoder for fast decoding (low quality)
  --control-net <string>                   path to control net model
  --embd-dir <string>                      embeddings directory
  --lora-model-dir <string>                lora model directory
  -i, --init-img <string>                  path to the init image
  --end-img <string>                       path to the end image, required by flf2v
  --tensor-type-rules <string>             weight type per tensor pattern (example: "^vae\.=f16,model\.=q8_0")
  --photo-maker <string>                   path to PHOTOMAKER model
  --pm-id-images-dir <string>              path to PHOTOMAKER input id images dir
  --pm-id-embed-path <string>              path to PHOTOMAKER v2 id embed
  --mask <string>                          path to the mask image
  --control-image <string>                 path to control image, control net
  --control-video <string>                 path to control video frames, It must be a directory path. The video frames inside should be stored as images in
                                           lexicographical (character) order. For example, if the control video path is
                                           `frames`, the directory contain images such as 00.png, 01.png, ... etc.
  -o, --output <string>                    path to write result image to (default: ./output.png)
  -p, --prompt <string>                    the prompt to render
  -n, --negative-prompt <string>           the negative prompt (default: "")
  --upscale-model <string>                 path to esrgan model.
  -t, --threads <int>                      number of threads to use during computation (default: -1). If threads <= 0, then threads will be set to the number of
                                           CPU physical cores
  --upscale-repeats <int>                  Run the ESRGAN upscaler this many times (default: 1)
  -H, --height <int>                       image height, in pixel space (default: 512)
  -W, --width <int>                        image width, in pixel space (default: 512)
  --steps <int>                            number of sample steps (default: 20)
  --high-noise-steps <int>                 (high noise) number of sample steps (default: -1 = auto)
  --clip-skip <int>                        ignore last layers of CLIP network; 1 ignores none, 2 ignores one layer (default: -1). <= 0 represents unspecified,
                                           will be 1 for SD1.x, 2 for SD2.x
  -b, --batch-count <int>                  batch count
  --chroma-t5-mask-pad <int>               t5 mask pad size of chroma
  --video-frames <int>                     video frames (default: 1)
  --fps <int>                              fps (default: 24)
  --timestep-shift <int>                   shift timestep for NitroFusion models (default: 0). recommended N for NitroSD-Realism around 250 and 500 for
                                           NitroSD-Vibrant
  --cfg-scale <float>                      unconditional guidance scale: (default: 7.0)
  --img-cfg-scale <float>                  image guidance scale for inpaint or instruct-pix2pix models: (default: same as --cfg-scale)
  --guidance <float>                       distilled guidance scale for models with guidance input (default: 3.5)
  --slg-scale <float>                      skip layer guidance (SLG) scale, only for DiT models: (default: 0). 0 means disabled, a value of 2.5 is nice for sd3.5
                                           medium
  --skip-layer-start <float>               SLG enabling point (default: 0.01)
  --skip-layer-end <float>                 SLG disabling point (default: 0.2)
  --eta <float>                            eta in DDIM, only for DDIM and TCD (default: 0)
  --high-noise-cfg-scale <float>           (high noise) unconditional guidance scale: (default: 7.0)
  --high-noise-img-cfg-scale <float>       (high noise) image guidance scale for inpaint or instruct-pix2pix models (default: same as --cfg-scale)
  --high-noise-guidance <float>            (high noise) distilled guidance scale for models with guidance input (default: 3.5)
  --high-noise-slg-scale <float>           (high noise) skip layer guidance (SLG) scale, only for DiT models: (default: 0)
  --high-noise-skip-layer-start <float>    (high noise) SLG enabling point (default: 0.01)
  --high-noise-skip-layer-end <float>      (high noise) SLG disabling point (default: 0.2)
  --high-noise-eta <float>                 (high noise) eta in DDIM, only for DDIM and TCD (default: 0)
  --strength <float>                       strength for noising/unnoising (default: 0.75)
  --pm-style-strength <float>
  --control-strength <float>               strength to apply Control Net (default: 0.9). 1.0 corresponds to full destruction of information in init image
  --moe-boundary <float>                   timestep boundary for Wan2.2 MoE model. (default: 0.875). Only enabled if `--high-noise-steps` is set to -1
  --flow-shift <float>                     shift value for Flow models like SD3.x or WAN (default: auto)
  --vace-strength <float>                  wan vace strength
  --vae-tile-overlap <float>               tile overlap for vae tiling, in fraction of tile size (default: 0.5)
  --vae-tiling                             process vae in tiles to reduce memory usage
  --force-sdxl-vae-conv-scale              force use of conv scale on sdxl vae
  --offload-to-cpu                         place the weights in RAM to save VRAM, and automatically load them into VRAM when needed
  --control-net-cpu                        keep controlnet in cpu (for low vram)
  --clip-on-cpu                            keep clip in cpu (for low vram)
  --vae-on-cpu                             keep vae in cpu (for low vram)
  --diffusion-fa                           use flash attention in the diffusion model
  --diffusion-conv-direct                  use ggml_conv2d_direct in the diffusion model
  --vae-conv-direct                        use ggml_conv2d_direct in the vae model
  --canny                                  apply canny preprocessor (edge detection)
  -v, --verbose                            print extra info
  --color                                  colors the logging tags according to level
  --chroma-disable-dit-mask                disable dit mask for chroma
  --chroma-enable-t5-mask                  enable t5 mask for chroma
  --increase-ref-index                     automatically increase the indices of references images based on the order they are listed (starting with 1).
  --disable-auto-resize-ref-image          disable auto resize of ref images
  -M, --mode                               run mode, one of [img_gen, vid_gen, upscale, convert], default: img_gen
  --type                                   weight type (examples: f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0, q2_K, q3_K, q4_K). If not specified, the default is the
                                           type of the weight file
  --rng                                    RNG, one of [std_default, cuda], default: cuda
  -s, --seed                               RNG seed (default: 42, use random seed for < 0)
  --sampling-method                        sampling method, one of [euler, euler_a, heun, dpm2, dpm++2s_a, dpm++2m, dpm++2mv2, ipndm, ipndm_v, lcm, ddim_trailing,
                                           tcd] (default: euler for Flux/SD3/Wan, euler_a otherwise)
  --prediction                             prediction type override, one of [eps, v, edm_v, sd3_flow, flux_flow]
  --scheduler                              denoiser sigma scheduler, one of [discrete, karras, exponential, ays, gits, smoothstep, sgm_uniform, simple], default:
                                           discrete
  --skip-layers                            layers to skip for SLG steps (default: [7,8,9])
  --high-noise-sampling-method             (high noise) sampling method, one of [euler, euler_a, heun, dpm2, dpm++2s_a, dpm++2m, dpm++2mv2, ipndm, ipndm_v, lcm,
                                           ddim_trailing, tcd] default: euler for Flux/SD3/Wan, euler_a otherwise
  --high-noise-scheduler                   (high noise) denoiser sigma scheduler, one of [discrete, karras, exponential, ays, gits, smoothstep, sgm_uniform,
                                           simple], default: discrete
  --high-noise-skip-layers                 (high noise) layers to skip for SLG steps (default: [7,8,9])
  -r, --ref-image                          reference image for Flux Kontext models (can be used multiple times)
  -h, --help                               show this help message and exit
  --vae-tile-size                          tile size for vae tiling, format [X]x[Y] (default: 32x32)
  --vae-relative-tile-size                 relative tile size for vae tiling, format [X]x[Y], in fraction of image size if < 1, in number of tiles per dim if >=1
                                           (overrides --vae-tile-size)
```