#!/bin/bash
# Wan 2.2 TI2V — Image-to-Video Generation
# Usage: ./run_wan.sh [prompt] [frames] [seed] [output] [portrait] [threads] [vae]
#   portrait: "true" (default) = 9:16, "false" = 16:9
#   vae: "gpu" (default, slow but works) or "cpu" (larger tiles, even slower)
#   steps: 20 (default, Wan designed for 20 steps)

PROMPT="${1:-High-octane anime action sequence. Two cyberpunk shinobi warriors clash in a neon-lit futuristic dojo...}"
FRAMES="${2:-81}"
SEED="${3:-42}"
OUTPUT="${4:-output/wan_f${FRAMES}_s${SEED}.webm}"
PORTRAIT="${5:-true}"
THREADS="${6:-14}"
VAE="${7:-gpu}"
STEPS="${8:-20}"

W=864
H=480

if [ "$PORTRAIT" = "true" ]; then
  FINAL_PROMPT="Rotated 90 degrees counter-clockwise camera shot. $PROMPT"
  TEMP_OUTPUT="${OUTPUT%.*}.tmp.webm"
  ORIENT="9:16 portrait (rotated canvas)"
else
  FINAL_PROMPT="$PROMPT"
  TEMP_OUTPUT="$OUTPUT"
  ORIENT="16:9 landscape (native)"
fi

if [ "$VAE" = "cpu" ]; then
  # Fallback: full VAE on CPU (slow but works if TAEHV unavailable)
  VAE_FLAGS="--vae models/wan/Wan2.2_VAE.safetensors"
  VAE_BACKEND="vae=cpu"
  VAE_TILES="--vae-tile-size 64x64 --vae-tile-overlap 0.50"
  VAE_TEMP="temporal_tile_frames=8,temporal_tile_overlap=2"
  VAE_THREADS=28
  VAE_LABEL="CPU VAE (64x64, very slow)"
else
  # TAEHV: Tiny AutoEncoder — 62× faster than full VAE!
  VAE_FLAGS="--tae models/taew2_2.safetensors"
  VAE_BACKEND="vae=cuda1"
  VAE_TILES=""
  VAE_TEMP=""
  VAE_THREADS="$THREADS"
  VAE_LABEL="TAEHV (16.6s, GPU)"
fi

echo "=== Wan 2.2 TI2V Generation ==="
echo "Orientation: $ORIENT (${W}x${H})"
echo "Frames: $FRAMES ($(python3 -c "print($FRAMES/25)")s)"
echo "Seed: $SEED  Steps: $STEPS  Threads: $VAE_THREADS"
echo "VAE: $VAE_LABEL"
echo "Output: $OUTPUT"
echo ""

numactl --interleave=all ./build-cuda/bin/sd-cli -M vid_gen \
  --diffusion-model models/wan/Wan2.2-TI2V-5B-Q8_0.gguf \
  $VAE_FLAGS \
  --t5xxl models/wan/umt5-xxl-encoder-Q6_K.gguf \
  -p "$FINAL_PROMPT" \
  -i input/scene_start.png \
  --sampling-method euler --scheduler auto --steps "$STEPS" \
  -W "$W" -H "$H" --video-frames "$FRAMES" --fps 25 \
  --seed "$SEED" --threads "$VAE_THREADS" \
  --backend diffusion=cuda2,te=cuda3,$VAE_BACKEND \
  --params-backend cpu --max-vram -0.5 \
  --diffusion-fa \
  $VAE_TILES \
  ${VAE_TEMP:+--temporal-tiling --extra-tiling-args "$VAE_TEMP"} \
  ${VAE_TILES:+--vae-conv-direct} \
  --save-latent "${OUTPUT%.*}.latent.bin" \
  -o "$TEMP_OUTPUT"

if [ "$PORTRAIT" = "true" ]; then
  echo ""
  echo "=== Post-Processing: Rotating to 9:16 portrait ==="
  ffmpeg -y -i "$TEMP_OUTPUT" -vf "transpose=1" -c:v libvpx-vp9 -pix_fmt yuv420p "$OUTPUT" 2>/dev/null
  rm "$TEMP_OUTPUT"
  echo "Done! Portrait: $OUTPUT"
else
  echo "Done! Landscape: $OUTPUT"
fi
