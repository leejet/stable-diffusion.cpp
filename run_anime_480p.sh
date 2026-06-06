#!/bin/bash
# 480p Anime Action Video
# Rotated Canvas: renders 864x480 landscape, rotates to 480x864 portrait
# Usage: ./run_anime_480p.sh [prompt] [frames] [seed] [output] [portrait] [threads] [vae]
#   portrait: "true" (default) = 9:16, "false" = 16:9
#   vae: "gpu" (default, fast) or "cpu" (quality, slow)
#   steps: 8 (default), try 12-16 for denoising quality

PROMPT="${1:-High-octane anime action sequence. Two cyberpunk shinobi warriors clash in a neon-lit futuristic dojo. One throws a devastating punch, sending the other flying across the room. Camera pans dynamically tracking the airborne fighter as he extends his cybernetic arm and fires an energy blast back at his opponent.}"
FRAMES="${2:-201}"
SEED="${3:-42}"
OUTPUT="${4:-output/anime_f${FRAMES}_s${SEED}.webm}"
PORTRAIT="${5:-true}"
THREADS="${6:-14}"
VAE="${7:-gpu}"
STEPS="${8:-8}"

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
  VAE_BACKEND="vae=cpu"
  VAE_TILES="--vae-tile-size 256x256 --vae-tile-overlap 0.25"
  VAE_TEMP="temporal_tile_frames=24,temporal_tile_overlap=6"
  VAE_THREADS=28
  VAE_LABEL="CPU (256x256, quality)"
else
  VAE_BACKEND="vae=cuda1"
  VAE_TILES="--vae-tile-size 16x16 --vae-tile-overlap 0.25"
  VAE_TEMP="temporal_tile_frames=2,temporal_tile_overlap=0"
  VAE_THREADS="$THREADS"
  VAE_LABEL="GPU (16x16, fast)"
fi

echo "=== 480p Anime Action Generation ==="
echo "Orientation: $ORIENT (${W}x${H})"
echo "Frames: $FRAMES ($(python3 -c "print($FRAMES/25)")s)"
echo "Seed: $SEED  Steps: $STEPS  Threads: $VAE_THREADS"
echo "VAE: $VAE_LABEL"
echo "Output: $OUTPUT"
echo ""

numactl --interleave=all ./build-cuda/bin/sd-cli -M vid_gen \
  --diffusion-model models/ltx-2.3-22b-distilled-1.1-UD-Q4_K_M.gguf \
  --vae models/ltx-2.3-22b-distilled_video_vae.safetensors \
  --audio-vae models/ltx-2.3-22b-distilled_audio_vae.safetensors \
  --llm models/gemma-3-12b-it-qat-UD-Q4_K_XL.gguf \
  --embeddings-connectors models/ltx-2.3-22b-distilled_embeddings_connectors.safetensors \
  -p "$FINAL_PROMPT" \
  --cfg-scale 1.1 \
  --sampling-method euler --scheduler ltx2 --steps "$STEPS" \
  -W "$W" -H "$H" --video-frames "$FRAMES" --fps 25 \
  --seed "$SEED" --threads "$VAE_THREADS" \
  --backend diffusion=cuda2,te=cuda3,$VAE_BACKEND \
  --params-backend cpu --max-vram -0.5 \
  --diffusion-fa \
  --extra-sample-args "stretch=true,max_shift=2.0,base_shift=0.95,terminal=0.1" \
  --vae-tiling $VAE_TILES \
  --temporal-tiling --extra-tiling-args "$VAE_TEMP" \
  --vae-conv-direct \
  --save-latent "${OUTPUT%.*}.latent.bin" \
  -o "$TEMP_OUTPUT"

# Rotate to portrait if needed
if [ "$PORTRAIT" = "true" ]; then
  echo ""
  echo "=== Post-Processing: Rotating to 9:16 portrait ==="
  ffmpeg -y -i "$TEMP_OUTPUT" -vf "transpose=1" -c:v libvpx-vp9 -pix_fmt yuv420p "$OUTPUT" 2>/dev/null
  rm "$TEMP_OUTPUT"
  echo "Done! Portrait video: $OUTPUT"
else
  echo "Done! Landscape video: $OUTPUT"
fi
