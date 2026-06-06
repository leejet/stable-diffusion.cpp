#!/bin/bash
# 480p (864×480 or 480×864) video generation
# Multi-GPU: diffusion=cuda2, te=cuda3, vae=cuda1
# Usage: ./run_480p.sh [prompt] [frames] [seed] [output] [portrait] [threads] [vae] [steps]

PROMPT="${1:-Cinematic high quality video, smooth motion, consistent details.}"
FRAMES="${2:-201}"
SEED="${3:-42}"
OUTPUT="${4:-output/480p_f${FRAMES}_s${SEED}.webm}"
PORTRAIT="${5:-false}"
THREADS="${6:-14}"
VAE="${7:-gpu}"
STEPS="${8:-8}"

SAMPLING_BASE="max_shift=2.0,base_shift=0.95,terminal=0.1"

if [ "$PORTRAIT" = "true" ]; then
  W=480; H=864
  SAMPLE_ARGS="stretch=true,$SAMPLING_BASE"
  ORIENT="9:16 portrait (native vertical)"
else
  W=864; H=480
  SAMPLE_ARGS="stretch=false,$SAMPLING_BASE"
  ORIENT="16:9 landscape (native)"
fi

if [ "$VAE" = "cpu" ]; then
  VAE_BACKEND="vae=cpu"
  VAE_TILES="--vae-tile-size 32x32 --vae-tile-overlap 0.25"
  VAE_TEMP=""
  VAE_THREADS=28
  VAE_LABEL="CPU (32x32, native temporal)"
else
  VAE_BACKEND="vae=cuda1"
  VAE_TILES="--vae-tile-size 14x14 --vae-tile-overlap 0.25"
  VAE_TEMP="--extra-tiling-args temporal_tile_frames=4,temporal_tile_overlap=1"
  VAE_THREADS="$THREADS"
  VAE_LABEL="GPU (14x14, temp=4/1 — defaults)"
fi

echo "=== 480p Generation ==="
echo "Orientation: $ORIENT (${W}x${H})"
echo "Frames:      $FRAMES ($(python3 -c "print($FRAMES/25)")s)"
echo "Tokens:      $(python3 -c "T=($FRAMES-1)//8+1; print(T*($H//32)*($W//32))")"
echo "Seed:        $SEED   Steps: $STEPS   Threads: $VAE_THREADS"
echo "VAE:         $VAE_LABEL"
echo "Output:      $OUTPUT"
echo ""

nohup numactl --interleave=all ./build-cuda/bin/sd-cli -M vid_gen \
  --diffusion-model models/ltx-2.3-22b-distilled-1.1-UD-Q4_K_M.gguf \
  --vae models/ltx-2.3-22b-distilled_video_vae.safetensors \
  --audio-vae models/ltx-2.3-22b-distilled_audio_vae.safetensors \
  --llm models/gemma-3-12b-it-qat-UD-Q4_K_XL.gguf \
  --embeddings-connectors models/ltx-2.3-22b-distilled_embeddings_connectors.safetensors \
  -p "$PROMPT" \
  --cfg-scale 1.1 --sampling-method euler --scheduler ltx2 --steps "$STEPS" \
  -W "$W" -H "$H" --video-frames "$FRAMES" --fps 25 \
  --seed "$SEED" --threads "$VAE_THREADS" \
  --backend diffusion=cuda2,te=cuda3,$VAE_BACKEND \
  --params-backend cpu --max-vram -0.5 --diffusion-fa \
  --mmap \
  --extra-sample-args "$SAMPLE_ARGS" \
  --vae-tiling $VAE_TILES \
  ${VAE_TEMP:+--temporal-tiling $VAE_TEMP} \
  --audio-vae-cpu \
  --vae-conv-direct \
  --save-latent "${OUTPUT%.*}.latent.bin" \
  -o "$OUTPUT" \
  > "${OUTPUT%.*}.log" 2>&1 &
echo "PID: $! | Log: ${OUTPUT%.*}.log"