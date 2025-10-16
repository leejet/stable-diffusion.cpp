## Use Flash Attention to save memory and improve speed.

Enabling flash attention for the diffusion model reduces memory usage by varying amounts of MB.
eg.:
 - flux 768x768 ~600mb
 - SD2 768x768 ~1400mb

For most backends, it slows things down, but for cuda it generally speeds it up too.
At the moment, it is only supported for some models and some backends (like cpu, cuda/rocm, metal).

Run by adding `--diffusion-fa` to the arguments and watch for:
```
[INFO ] stable-diffusion.cpp:312  - Using flash attention in the diffusion model
```
and the compute buffer shrink in the debug log:
```
[DEBUG] ggml_extend.hpp:1004 - flux compute buffer size: 650.00 MB(VRAM)
```

## Offload weights to the CPU to save VRAM without reducing generation speed.

Using `--offload-to-cpu` allows you to offload weights to the CPU, saving VRAM without reducing generation speed.

## Use quantization to reduce memory usage.

[quantization](./quantization_and_gguf.md)