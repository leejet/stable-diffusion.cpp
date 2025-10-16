# Run

```
usage: ./bin/sd [arguments]

arguments:
  -h, --help                         show this help message and exit
  -M, --mode [MODE]                  run mode, one of: [img_gen, vid_gen, upscale, convert], default: img_gen
  -t, --threads N                    number of threads to use during computation (default: -1)
                                     If threads <= 0, then threads will be set to the number of CPU physical cores
  --offload-to-cpu                   place the weights in RAM to save VRAM, and automatically load them into VRAM when needed
  -m, --model [MODEL]                path to full model
  --diffusion-model                  path to the standalone diffusion model
  --high-noise-diffusion-model       path to the standalone high noise diffusion model
  --clip_l                           path to the clip-l text encoder
  --clip_g                           path to the clip-g text encoder
  --clip_vision                      path to the clip-vision encoder
  --t5xxl                            path to the t5xxl text encoder
  --qwen2vl                          path to the qwen2vl text encoder
  --qwen2vl_vision                   path to the qwen2vl vit
  --vae [VAE]                        path to vae
  --taesd [TAESD_PATH]               path to taesd. Using Tiny AutoEncoder for fast decoding (low quality)
  --control-net [CONTROL_PATH]       path to control net model
  --embd-dir [EMBEDDING_PATH]        path to embeddings
  --upscale-model [ESRGAN_PATH]      path to esrgan model. For img_gen mode, upscale images after generate, just RealESRGAN_x4plus_anime_6B supported by now
  --upscale-repeats                  Run the ESRGAN upscaler this many times (default 1)
  --type [TYPE]                      weight type (examples: f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0, q2_K, q3_K, q4_K)
                                     If not specified, the default is the type of the weight file
  --tensor-type-rules [EXPRESSION]   weight type per tensor pattern (example: "^vae\.=f16,model\.=q8_0")
  --lora-model-dir [DIR]             lora model directory
  -i, --init-img [IMAGE]             path to the init image, required by img2img
  --mask [MASK]                      path to the mask image, required by img2img with mask
  -i, --end-img [IMAGE]              path to the end image, required by flf2v
  --control-image [IMAGE]            path to image condition, control net
  -r, --ref-image [PATH]             reference image for Flux Kontext models (can be used multiple times)
  --control-video [PATH]             path to control video frames, It must be a directory path.
                                     The video frames inside should be stored as images in lexicographical (character) order
                                     For example, if the control video path is `frames`, the directory contain images such as 00.png, 01.png, 鈥?etc.
  --increase-ref-index               automatically increase the indices of references images based on the order they are listed (starting with 1).
  -o, --output OUTPUT                path to write result image to (default: ./output.png)
  -p, --prompt [PROMPT]              the prompt to render
  -n, --negative-prompt PROMPT       the negative prompt (default: "")
  --cfg-scale SCALE                  unconditional guidance scale: (default: 7.0)
  --img-cfg-scale SCALE              image guidance scale for inpaint or instruct-pix2pix models: (default: same as --cfg-scale)
  --guidance SCALE                   distilled guidance scale for models with guidance input (default: 3.5)
  --slg-scale SCALE                  skip layer guidance (SLG) scale, only for DiT models: (default: 0)
                                     0 means disabled, a value of 2.5 is nice for sd3.5 medium
  --eta SCALE                        eta in DDIM, only for DDIM and TCD: (default: 0)
  --skip-layers LAYERS               Layers to skip for SLG steps: (default: [7,8,9])
  --skip-layer-start START           SLG enabling point: (default: 0.01)
  --skip-layer-end END               SLG disabling point: (default: 0.2)
  --scheduler {discrete, karras, exponential, ays, gits, smoothstep, sgm_uniform, simple} Denoiser sigma scheduler (default: discrete)
  --sampling-method {euler, euler_a, heun, dpm2, dpm++2s_a, dpm++2m, dpm++2mv2, ipndm, ipndm_v, lcm, ddim_trailing, tcd}
                                     sampling method (default: "euler" for Flux/SD3/Wan, "euler_a" otherwise)
  --timestep-shift N                 shift timestep for NitroFusion models, default: 0, recommended N for NitroSD-Realism around 250 and 500 for NitroSD-Vibrant
  --steps  STEPS                     number of sample steps (default: 20)
  --high-noise-cfg-scale SCALE       (high noise) unconditional guidance scale: (default: 7.0)
  --high-noise-img-cfg-scale SCALE   (high noise) image guidance scale for inpaint or instruct-pix2pix models: (default: same as --cfg-scale)
  --high-noise-guidance SCALE        (high noise) distilled guidance scale for models with guidance input (default: 3.5)
  --high-noise-slg-scale SCALE       (high noise) skip layer guidance (SLG) scale, only for DiT models: (default: 0)
                                     0 means disabled, a value of 2.5 is nice for sd3.5 medium
  --high-noise-eta SCALE             (high noise) eta in DDIM, only for DDIM and TCD: (default: 0)
  --high-noise-skip-layers LAYERS    (high noise) Layers to skip for SLG steps: (default: [7,8,9])
  --high-noise-skip-layer-start      (high noise) SLG enabling point: (default: 0.01)
  --high-noise-skip-layer-end END    (high noise) SLG disabling point: (default: 0.2)
  --high-noise-scheduler {discrete, karras, exponential, ays, gits, smoothstep, sgm_uniform, simple} Denoiser sigma scheduler (default: discrete)
  --high-noise-sampling-method {euler, euler_a, heun, dpm2, dpm++2s_a, dpm++2m, dpm++2mv2, ipndm, ipndm_v, lcm, ddim_trailing, tcd}
                                     (high noise) sampling method (default: "euler_a")
  --high-noise-steps  STEPS          (high noise) number of sample steps (default: -1 = auto)
                                     SLG will be enabled at step int([STEPS]*[START]) and disabled at int([STEPS]*[END])
  --strength STRENGTH                strength for noising/unnoising (default: 0.75)
  --control-strength STRENGTH        strength to apply Control Net (default: 0.9)
                                     1.0 corresponds to full destruction of information in init image
  -H, --height H                     image height, in pixel space (default: 512)
  -W, --width W                      image width, in pixel space (default: 512)
  --rng {std_default, cuda}          RNG (default: cuda)
  -s SEED, --seed SEED               RNG seed (default: 42, use random seed for < 0)
  -b, --batch-count COUNT            number of images to generate
  --prediction {eps, v, edm_v, sd3_flow, flux_flow} Prediction type override
  --clip-skip N                      ignore last layers of CLIP network; 1 ignores none, 2 ignores one layer (default: -1)
                                     <= 0 represents unspecified, will be 1 for SD1.x, 2 for SD2.x
  --vae-tiling                       process vae in tiles to reduce memory usage
  --vae-tile-size [X]x[Y]            tile size for vae tiling (default: 32x32)
  --vae-relative-tile-size [X]x[Y]   relative tile size for vae tiling, in fraction of image size if < 1, in number of tiles per dim if >=1 (overrides --vae-tile-size)
  --vae-tile-overlap OVERLAP         tile overlap for vae tiling, in fraction of tile size (default: 0.5)
  --force-sdxl-vae-conv-scale        force use of conv scale on sdxl vae
  --vae-on-cpu                       keep vae in cpu (for low vram)
  --clip-on-cpu                      keep clip in cpu (for low vram)
  --diffusion-fa                     use flash attention in the diffusion model (for low vram)
                                     Might lower quality, since it implies converting k and v to f16.
                                     This might crash if it is not supported by the backend.
  --diffusion-conv-direct            use Conv2d direct in the diffusion model
                                     This might crash if it is not supported by the backend.
  --vae-conv-direct                  use Conv2d direct in the vae model (should improve the performance)
                                     This might crash if it is not supported by the backend.
  --control-net-cpu                  keep controlnet in cpu (for low vram)
  --canny                            apply canny preprocessor (edge detection)
  --color                            colors the logging tags according to level
  --chroma-disable-dit-mask          disable dit mask for chroma
  --chroma-enable-t5-mask            enable t5 mask for chroma
  --chroma-t5-mask-pad  PAD_SIZE     t5 mask pad size of chroma
  --video-frames                     video frames (default: 1)
  --fps                              fps (default: 24)
  --moe-boundary BOUNDARY            timestep boundary for Wan2.2 MoE model. (default: 0.875)
                                     only enabled if `--high-noise-steps` is set to -1
  --flow-shift SHIFT                 shift value for Flow models like SD3.x or WAN (default: auto)
  --vace-strength                    wan vace strength
  --photo-maker                      path to PHOTOMAKER model
  --pm-id-images-dir [DIR]           path to PHOTOMAKER input id images dir
  --pm-id-embed-path [PATH]          path to PHOTOMAKER v2 id embed
  --pm-style-strength                strength for keeping PHOTOMAKER input identity (default: 20)
  -v, --verbose                      print extra info
```