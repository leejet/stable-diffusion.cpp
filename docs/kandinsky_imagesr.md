# Kandinsky ImageSR 2.0

Kandinsky ImageSR uses the same transformer and FLUX VAE for both 2× and 4× super-resolution. Download the transformer configuration, all transformer shards and their index, and the VAE from the [official model bundle](https://huggingface.co/kandinskylab/Kandinsky-ImageSR-2.0).

```sh
hf download kandinskylab/Kandinsky-ImageSR-2.0 --include 'transformer/*' 'vae/*' --local-dir models/Kandinsky-ImageSR-2.0
sd-cli -M upscale -i input.png --upscale-model models/Kandinsky-ImageSR-2.0 --upscale-factor 2 -o output.png
```

Use `--upscale-factor 4` for 4×, the default. Both scales use the baked zero-text conditioning and take one flow step per tile. The timestep is 0.15 for 2× and 0.244 for 4×. Tiles overlap by 25% and have a fixed output size of 1024×1024. The output is cropped to the requested multiple of the input dimensions.

The C API accepts the bundle directory in `new_upscaler_ctx` and the scale in `upscale`. For the server, place the bundle directory under `--hires-upscalers-dir` and use its directory name as the `upscaler` in [the upscale endpoint](../examples/server/api.md#post-sdcppv1upscale).
