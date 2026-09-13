# How to Use

SenseNova U1.5 is an 8B MoT model that performs diffusion directly in RGB pixel
space. It does not require a separate text encoder or VAE.

## Download weights

- Download SenseNova U1.5 8B MoT
    - safetensors: https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT

Pass the complete downloaded repository directory to `--model`. The directory
must contain `model.safetensors.index.json`, every referenced Safetensors shard,
and the tokenizer files.

## Examples

### CUDA

```bash
./bin/sd-cli \
  --model /path/to/SenseNova-U1.5-8B-MoT \
  --prompt "a red cube on a white background" \
  --width 2048 \
  --height 2048 \
  --steps 50 \
  --cfg-scale 4 \
  --flow-shift 3 \
  --seed 42 \
  --sampling-method euler \
  --rng cuda \
  --fa \
  --output output.png
```

## Notes

- To match the official non-thinking text-to-image pipeline, use 50 Euler
  steps, CFG 4, flow shift 3, seed 42, CUDA RNG, and an empty negative prompt.
- Width and height must be multiples of 32. The trained 1:1 resolution is
  2048x2048; lower resolutions are useful for smoke tests but are outside the
  training buckets.
- The SenseNova prompt template and unconditional prompt are built
  automatically.
- This implementation supports non-thinking text-to-image generation. Image
  editing, visual understanding, interleaved generation, and thinking-mode
  prompt expansion are not implemented.
