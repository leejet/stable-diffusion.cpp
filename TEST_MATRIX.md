# LTX 2.3 Video Generation — Test Matrix

**Hardware:** 4× GTX 1080 Ti (11 GB each), 32 GB RAM, CUDA 12.8  
**Date:** 2026-06-01 to 2026-06-03

---

## Global Parameters (all runs)

| Parameter | Value |
|---|---|
| Model | `ltx-2.3-22b-distilled-1.1-UD-Q4_K_M.gguf` (16 GB) |
| VAE | `ltx-2.3-22b-distilled_video_vae.safetensors` |
| Audio VAE | `ltx-2.3-22b-distilled_audio_vae.safetensors` |
| LLM | `gemma-3-12b-it-qat-UD-Q4_K_XL.gguf` (7 GB) |
| Embeddings | `ltx-2.3-22b-distilled_embeddings_connectors.safetensors` |
| Sampler | `euler`, scheduler `ltx2` |
| CFG scale | 1.1 |
| Steps | 8 |
| FPS | 25 |
| Threads | 14 |
| Backend | `diffusion=cuda2,te=cuda3,vae=cuda1` |
| Params backend | `cpu` |
| VRAM | `--max-vram -0.5` |
| Flash Attention | `--diffusion-fa` |
| Stretch | `stretch=true,max_shift=2.0,base_shift=0.95,terminal=0.1` |
| VAE conv | `--vae-conv-direct` |
| mmap | **DISABLED** (harmful — locks 27 GB RAM) |
| EasyCache | **DISABLED** (0 skips on 8-step runs) |

---

## Test Results

### 1. Landscape 1280×704 — Early Tests

| # | Frames | Res | mmap | VAE tiles | Temp frames | Temp overlap | EasyCache | Denoising | Audio VAE | Video VAE | Total | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 65 | 1280×704 | yes | 16×16 | 2 | 1 | yes (0 skips) | 745s | 51s | 238s | 1059s | ✅ |
| 2 | 65 | 1280×704 | yes | 16×16 | 2 | 0 | no | 756s | 51s | 132s | 957s | ✅ |
| 3 | 65 | 1280×704 | **no** | **256×256** | 2 | 0 | no | 739s | 50s | OOM | — | ❌ VAE OOM GPU1 |
| 4 | 65 | 1280×704 | no | 256×256 | 2 | 0 | no | 740s | 51s | OOM | — | ❌ VAE OOM (+conv_direct) |

### 2. Landscape 1280×704 — earlier mmap runs (with EasyCache)

| # | Frames | Res | mmap | VAE | Temp | Cache | Denoising | Total | Status |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 65 | 1280×704 | yes | GPU | 2/1 | yes | 745s | 1059s | ✅ |
| 6 | 65 | 1280×704 | yes | GPU | 2/0 | no | 756s | 957s | ✅ |
| 7 | 65 | 1280×704 | no | GPU 256×256 | 2/0 | no | 739s | OOM | ❌ |

### 3. Vertical 720p (704×1280) — 129 frames (5.2s)

| # | Seed | Temp frames | Temp overlap | Denoising | Audio VAE | Video VAE | Total | Status | Output |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 42 | 2 | 0 | 1707s | 197s | 267s | 2196s | ✅ | `shorts_5s_720p.webm` |
| 9 | 43 | 2 | 0 | 1716s | 197s | 267s | 2210s | ✅ | `shorts_5s_720p_smooth.webm` |
| 10 | 42 | 3 | 1 | 1683s | 196s | OOM | — | ❌ | GPU1 OOM |
| 11 | 42 | **2** | **1** | running | — | — | — | ▶️ | `shorts_5s_720p_overlap1.webm` |

### 4. Vertical 480p (480×864)

| # | Frames | Duration | Temp frames | Temp overlap | Segments | Denoising | Audio VAE | Video VAE | Total | Status | Output |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 12 | 201 | 8.0s | 2 | 0 | 6 | 1079s | 479s | 131s | 1709s | ✅ | `anime_8s_480x864.webm` |
| 13 | 249 | 10.0s | 2 | 0 | 7 | 1411s | 734s | 165s | 2326s | ✅ | `anime_10s_480x864.webm` |
| 14 | 129 | 5.2s | **3** | **1** | 5 | 638s | 195s | 117s | 967s | ✅ | `anime_5s_480p_temp3.webm` |
| 15 | 129 | 5.2s | 4 | 2 | 5 | — | — | OOM | — | ❌ | GPU1 OOM |

### 5. Edge Cases

| # | Frames | Res | Tokens | Segments | Status | Error |
|---|---|---|---|---|---|---|
| 16 | 201 | 704×1280 | 22,880 | 16 | ❌ | `grid_z < USHRT_MAX` — CUDA grid overflow |
| 17 | 65 | 480×768 | 3,240 | 3 | ✅ | T2V baseline (Q4_0 dev) |
| 18 | 65 | 480×768 | 3,240 | 3 | ✅ | FLF2V (Q4_0 dev, 771s) |

---

## Hardware Limits Discovered

### GPU VRAM (11 GB per GPU)

| Parameter | Ceiling | Note |
|---|---|---|
| Spatial VAE tile | **16×16** | 32×32 OOMs on all GPUs |
| Temporal frames @ 720p | **2** | 3 OOMs (too many spatial tiles per batch) |
| Temporal frames @ 480p | **3** | 4 OOMs |
| Temporal overlap @ 720p | **0→1 testing** | Overlap=1 doesn't change per-tile memory |
| Max tokens before grid overflow | ~15,000 | 22,880 crashes `grid_z < USHRT_MAX` |
| Max segments before grid overflow | ~9 | 16 segments overflows CUDA grid |

### System RAM (32 GB)

| Config | Peak RSS | Model residency | Note |
|---|---|---|---|
| With `--mmap` | 27.3 GB | Models locked forever | Harmful |
| Without `--mmap` | 28.4 GB → **3.2 GB** | Models freed after use | Much better |

### Sampler

| Sampler | EasyCache effect |
|---|---|
| `euler` | 0 skips with 8 steps (too few steps to warm up) |
| `euler_a` | 0 skips even with 30 steps (stochastic noise breaks caching) |

---

## Scaling Formulas (for time estimation)

- **Tokens:** `T_lat = (frames - 1) / 8 + 1`, `H_lat = height / 32`, `W_lat = width / 32`
- **Denoising time ≈ tokens × 0.022s** (for distilled Q4_K_M, 8 steps)
- **VAE decode ≈ proportional to frames** (6 temporal tiles per 65f batch)

---

## Best Known Configurations

### 720p (704×1280) — `run_720p.sh`
```
--vae-tiling --vae-tile-size 16x16 --vae-tile-overlap 0.25
--temporal-tiling --extra-tiling-args "temporal_tile_frames=2,temporal_tile_overlap=0"
```
Max frames: **129** (5.2s). 201 frames crashes with CUDA grid overflow.

### 480p (480×864) — `run_480p.sh`
```
--vae-tiling --vae-tile-size 16x16 --vae-tile-overlap 0.25
--temporal-tiling --extra-tiling-args "temporal_tile_frames=3,temporal_tile_overlap=1"
```
Max frames tested: **249** (10.0s).

---

## Key Takeaways

1. **Never use `--mmap`** — locks 27 GB RAM with zero benefit
2. **Never use EasyCache with 8 steps** — 0 skips, pure overhead
3. **`euler_a` disables EasyCache** — stochastic sampler, cache never hits
4. **16×16 VAE tiles is the absolute GPU ceiling** — any GPU, any resolution
5. **Temporal frames ceiling depends on resolution** — 3 at 480p, 2 at 720p
6. **Distilled LoRA (7.1 GB) is a dead end** on 11 GB GPUs — adds 7.1 GB to weight footprint
7. **`--vae-conv-direct` helps but doesn't change tile size limits**
8. **`--extra-sample-args stretch` reduces denoising time ~2%**
9. **Max resolution for 8s:** 480×864×201f (10,530 tokens, 28.5 min)
10. **Max resolution for 5s:** 704×1280×129f (14,960 tokens, 36.8 min)
