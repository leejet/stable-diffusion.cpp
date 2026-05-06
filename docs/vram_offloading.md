# VRAM Offloading

Run models larger than your GPU memory by offloading weights to CPU RAM during generation.

## Offload Modes

Use `--offload-mode <mode>` to select the offloading strategy:

| Mode | Description | VRAM Usage | Speed | Quality |
|------|-------------|------------|-------|---------|
| `none` | Everything stays on GPU (default) | Highest | Fastest | No penalty |
| `cond_only` | Offload text encoder after conditioning | High | Near-full speed — only a brief reload between conditioning and diffusion | No penalty |
| `cond_diffusion` | Offload both text encoder and diffusion model between stages | Medium | Slower — model is reloaded to GPU each diffusion step | No penalty |
| `aggressive` | Aggressively offload all components when not in use | Low | Slowest of the non-streaming modes — frequent CPU↔GPU transfers | No penalty |
| `layer_streaming` | Stream transformer layers one-by-one through GPU | Lowest | Depends on model size (see below) | No penalty when using coarse-stage; per-layer streaming is lossless for most architectures |

The `--offload-to-cpu` flag is a shortcut that picks a reasonable offload mode automatically.

## Layer Streaming

Layer streaming is the most memory-efficient mode. Instead of loading the entire diffusion model into VRAM, it loads one transformer block at a time.

### How it works

1. **Coarse-stage**: If the model fits in VRAM (e.g., quantized models), all layers are loaded at once and the full graph is executed normally. This is as fast as `--offload-mode none` with no quality penalty — the only overhead is the initial CPU→GPU weight transfer.
2. **Per-layer streaming**: If the model doesn't fit (e.g., bf16 models on small GPUs), each transformer block is loaded, executed as a mini-graph, then offloaded back to CPU before the next block. This uses minimal VRAM but is significantly slower due to per-step CPU↔GPU transfers. Output quality is identical to full-model execution — the computation is mathematically equivalent, just split across separate graph evaluations.

The mode is chosen automatically based on available VRAM.

### Supported architectures

- Flux (double_blocks + single_blocks)
- ZImage / Z-Image-Turbo (context_refiner + noise_refiner + layers)
- MMDiT / SD3 (joint_blocks)
- UNet / SD1.x / SDXL (input_blocks + middle_block + output_blocks)
- Anima (blocks)
- WAN (blocks + vace_blocks)
- Qwen Image (transformer_blocks)

### Examples

#### ZImage-Turbo Q8 with layer streaming

```
sd-cli --diffusion-model z_image_turbo-Q8_0.gguf \
  --llm Qwen3-4b-Z-Engineer-V2.gguf \
  --vae ae.safetensors \
  -p "a cat" --cfg-scale 1.0 --diffusion-fa \
  -H 1024 -W 688 -s 42 \
  --offload-mode layer_streaming -v
```

The Q8 model (6.7 GB) fits in a 12 GB GPU, so coarse-stage streaming is used automatically:
```
[INFO ] z_image model fits in VRAM, using coarse-stage streaming
[INFO ] z_image coarse-stage streaming completed in 1.66s
```

#### Flux-dev Q4 with layer streaming

```
sd-cli --diffusion-model flux1-dev-q4_0.gguf \
  --vae ae.safetensors \
  --clip_l clip_l.safetensors \
  --t5xxl t5xxl_fp16.safetensors \
  -p "a lovely cat" --cfg-scale 1.0 --sampling-method euler \
  --offload-mode layer_streaming -v
```

#### SD1.5 with aggressive offloading

```
sd-cli -m sd-v1-4.ckpt \
  -p "a photograph of an astronaut riding a horse" \
  --offload-mode aggressive -v
```

## Combining with other options

- `--diffusion-fa`: Flash attention reduces VRAM further. Recommended with all offload modes. No quality penalty.
- `--clip-on-cpu`: Run CLIP text encoder on CPU. Saves VRAM but slows conditioning. No quality penalty.
- Quantized models (`q4_0`, `q8_0`, etc.) reduce model size, making coarse-stage streaming more likely (faster). **Quantization does reduce output quality** — lower bit depths produce softer details and may introduce artifacts. See [quantization](./quantization_and_gguf.md) for quality comparisons. `q8_0` is nearly indistinguishable from full precision; `q4_0` and below show visible degradation on fine details.

## Quality impact summary

| Technique | Quality Impact |
|-----------|---------------|
| `--offload-mode` (any mode) | **None** — offloading only changes where weights are stored, not the computation |
| `--diffusion-fa` (flash attention) | **None** — mathematically equivalent, just more memory-efficient |
| `--clip-on-cpu` | **None** — same computation on CPU instead of GPU |
| Quantization (`q8_0`) | **Negligible** — nearly identical to full precision |
| Quantization (`q4_0`, `q4_k`) | **Minor** — slight softening, fine details may differ |
| Quantization (`q3_k`, `q2_k`) | **Noticeable** — visible quality loss, best for previews or VRAM-constrained setups |

## Troubleshooting

- **OOM during generation**: Try a more aggressive mode. `layer_streaming` uses the least VRAM.
- **Slow generation**: Coarse-stage streaming (model fits in VRAM) is nearly as fast as no offloading. Per-layer streaming is slower due to CPU-GPU transfers each step. Using quantized models often lets you stay in coarse-stage mode.
- **Black or corrupted output**: This is a bug. Please report it with the model, offload mode, and resolution used.
