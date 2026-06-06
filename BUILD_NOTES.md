# LTX 2.3 & Wan 2.2 — Complete Optimization Findings

**Hardware:** 4× GTX 1080 Ti (11 GB VRAM each), 32 GB RAM, CUDA 12.8  
**Date:** June 1-5, 2026  
**Repo:** `/home/daniel/GitHub/stable-diffusion.cpp`

---

## 1. LTX 2.3 DISTILLED — PROVEN WINNING CONFIG

### Model Stack (30 GB total)
| Component | File | Size |
|---|---|---|
| Diffusion | `ltx-2.3-22b-distilled-1.1-UD-Q4_K_M.gguf` | 16 GB |
| Video VAE | `ltx-2.3-22b-distilled_video_vae.safetensors` | 1.4 GB |
| Audio VAE | `ltx-2.3-22b-distilled_audio_vae.safetensors` | 308 MB |
| LLM | `gemma-3-12b-it-qat-UD-Q4_K_XL.gguf` | 7 GB |
| Embeddings | `ltx-2.3-22b-distilled_embeddings_connectors.safetensors` | 2.2 GB |

### Optimal GPU Flags
```
--backend diffusion=cuda2,te=cuda3,vae=cuda1
--params-backend cpu --max-vram -0.5
--cfg-scale 1.1 --sampling-method euler --scheduler ltx2 --steps 8
--diffusion-fa
--extra-sample-args "stretch=true,max_shift=2.0,base_shift=0.95,terminal=0.1"
--vae-tile-size 16x16 --vae-tile-overlap 0.25
--temporal-tile_frames=2,temporal_tile_overlap=0
--vae-conv-direct
```

### Optimal CPU VAE Flags (when GPU OOMs)
```
--vae-tile-size 64x64 --vae-tile-overlap 0.50
--temporal-tile_frames=8,temporal_tile_overlap=2
--threads 28 --backend ...,vae=cpu
```

### CRITICAL Rules
1. **Never use `--mmap`** — locks 27 GB RAM with zero benefit
2. **Never use EasyCache with 8 steps** — 0 skips every single run
3. **Rotated Canvas**: LTX trained on landscape. For portrait output, render landscape (1280×704 or 864×480) then ffmpeg `transpose=1` to rotate
4. **LoRA adapter (7.1 GB) is a dead end** on 11 GB GPUs — adds too much weight memory

### Performance Summary

| Resolution | Frames | Duration | Denoising | VAE Decode | Total |
|---|---|---|---|---|---|
| 704×1280 (720p) | 129 | 5.2s | 1716s | 267s | 2196s |
| 480×864 (480p) | 201 | 8.0s | 1079s | 131s | 1709s |
| 480×864 (480p) | 249 | 10.0s | 1411s | 165s | 2326s |
| 864×480 | 81 | 3.2s | 520s | 52s | 677s |
| 832×480 T2V | 81 | 3.2s | 417s | 67s | **580s** |

### GPU VAE Tile Ceilings (11 GB)
| Spatial | Temporal | Overlap | Decode (81f) | Notes |
|---|---|---|---|---|
| 16×16 | 2 | 0 | 52s | Stable baseline |
| 16×16 | 3 | 1 | 69s | Max at 480p only |
| **8×8** | **12** | **3** | **67s** | 🏆 Best slow-motion |
| 24×24 | 2 | 0 | 46s | Largest GPU tile |
| 32×32 | any | any | OOM | 13+ GB alloc |
| 256×256 | any | any | OOM | 34.5 GB alloc |

### CPU VAE Tile Ceilings (32 GB RAM)
| Tiles | Temp | Overlap | Decode (81f) | Notes |
|---|---|---|---|---|
| 256×256 | 24 | 6 | OOM | 44.7 GB alloc |
| 64×64 | 8 | 2 | ~1500s | Works |
| 128×128 | 8 | 2 | OOM | 29+ GB |

### FLF2V / I2V (81f, 480p)
| Mode | Denoising | VAE | Total |
|---|---|---|---|
| FLF2V (start+end) | 573s | 52s | 718s |
| I2V (start only) | 520s | 52s | 664s |
| T2V (no image) | 417s | 67s | **580s** |

---

## 2. --SAVE-LATENT / --LOAD-LATENT (Implemented June 4)

Two new CLI flags for staged execution:
```
--save-latent output/latent.bin    → 2-6 MB file
--load-latent output/latent.bin    → 7-60s VAE-only (100× faster than full pipe)
```

**Use case:** Generate latent once (28 min), replay VAE with different tile configs in seconds.

### Modified files
- `src/tensor_ggml.hpp` — added `save_tensor_to_file_as_tensor<T>()`
- `include/stable-diffusion.h` — added `save_latent_path` / `load_latent_path`
- `examples/common/common.h` — added string fields
- `examples/common/common.cpp` — CLI flags + C struct population
- `src/stable-diffusion.cpp` — pipeline if/else + save point
- `examples/cli/main.cpp` — validation bypass

---

## 3. WAN 2.2 TI2V — DIAGNOSED BOTTLENECKS

### Model Stack (10.9 GB total)
| Component | File | Size |
|---|---|---|
| Diffusion | `Wan2.2-TI2V-5B-Q8_0.gguf` | 5.1 GB |
| VAE | `Wan2.2_VAE.safetensors` | 1.4 GB |
| Text Encoder | `umt5-xxl-encoder-Q6_K.gguf` | 4.4 GB |

### Root Cause: Wan VAE Has No Temporal Tiling

**Source analysis** (`src/wan.hpp`): Wan builds a single 215K-node compute graph for ALL frames. LTX breaks work into 20K-node graphs with streaming overlap. Wan also has a dead-code frame-by-frame path (`build_graph_partial` at line 1217) disabled with `if (true)` — abandoned due to a causal padding bug at chunk boundaries.

| Metric | Wan VAE | LTX VAE |
|---|---|---|
| Graph nodes (21f) | 215,040 | 20,480 × 7 tiles |
| Decoder channels | **1024** | ~128 |
| CausalConv3d layers | **34** | ~15 |
| Latent channels | 48 | 128 |
| Temporal tiling | NONE (dead code) | Streaming, 4f/tile |
| `--vae-tiling` flag | Partially ignored | Full support |
| Per-tile GPU time | 103s | 22s |

### GPU VAE Speed (81f, 16×16 tiles)
- 10 temporal groups × 103s = **1030s (17 min)** VAE alone
- 20 denoising steps × 36s = 770s
- Total: ~1800s (30 min) for 3.2s of video

### Wan Fix Recommendations
1. Fix and enable `decode_partial` dead code (causal padding bug at line 1267)
2. Implement proper temporal tiling like LTX's `decode_temporal_tiled_streaming`
3. Reduce decoder channels or use DepthToSpace upsampling
4. **TAEHV** (`--tae`) as immediate workaround — 4-6× VAE speedup per GitHub discussions

### Sources
- [GH Discussion #868](https://github.com/leejet/stable-diffusion.cpp/discussions/868) — Wan VAE memory explosion
- [GH Discussion #701](https://github.com/leejet/stable-diffusion.cpp/discussions/701) — Tile size performance data
- [GH PR #484](https://github.com/leejet/stable-diffusion.cpp/pull/484) — VAE tiling improvements

---

## 4. GLOBAL FINDINGS

### Flags Tested — What Works
| Flag | Effect |
|---|---|
| `--diffusion-fa` | **Essential** — Flash Attention |
| `--max-vram -0.5` | Graph-cut segmentation |
| `--params-backend cpu` | Params in RAM, streamed to GPU |
| `--extra-sample-args stretch=...` | ~2% denoising speedup |
| `--threads 28` | 2.2% faster than 14 (CPU workloads) |

### Flags Tested — What DOESN'T Help
| Flag | Finding |
|---|---|
| `--mmap` | **HARMFUL** — locks 27 GB RAM |
| `--cache-mode easycache` (8 steps) | 0 skips — dead weight |
| `--vae-conv-direct` (CPU) | Zero difference |
| `--vae-conv-direct` (GPU) | Negligible (~2%) |
| `--lora-apply-mode` (quantized) | OOM from +7 GB adapter |
| Multi-GPU backend | Doesn't distribute (single GPU used) |
| `--threads 14` vs 28 (GPU VAE) | Negligible difference |

### Sampler Comparison
| Sampler | EasyCache | Steps | Time |
|---|---|---|---|
| `euler` | Works (15/30 skips) | 30 | 771s |
| `euler_a` | **Broken** (0/30 skips) | 30 | 1091s |

### Per-tile VRAM Formula
```
VRAM = spatial_area × temporal_frames × constant
16×16 × 2 = 512 units (baseline)
8×8 × 12 = 768 units (same budget!)
```

### VAE Decode Time Scaling
- **VRAM**: constant regardless of video length (sequential tiles)
- **Time**: linear with frame count
- CPU VAE is 3-25× slower than GPU, but handles larger tiles

---

## 5. SCRIPTS

All scripts in repo root accept: `[prompt] [frames] [seed] [output] [portrait] [threads] [vae] [steps]`

| Script | Res | VAE default | Use case |
|---|---|---|---|
| `run_720p.sh` | 1280×704 | GPU 16×16 | Max quality 720p |
| `run_480p.sh` | 864×480 | GPU 16×16 | General 480p |
| `run_anime_480p.sh` | 864×480 | GPU 16×16 | Anime/action |

```bash
./run_anime_480p.sh "prompt" 201 42 out.webm true 14 gpu 8    # GPU VAE
./run_anime_480p.sh "prompt" 201 42 out.webm true 14 cpu 8     # CPU VAE (quality)
```
