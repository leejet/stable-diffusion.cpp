#!/usr/bin/env bash
# Run LTX-2.3 22B-dev video generation with the current local model paths.
# All placement / lazy-load / tensor-split is auto-detected (defaults to ON),
# so this script only needs the prompt and output path.
#
# Usage:
#   ./run_ltx2.sh "<prompt>" [output.webm]
#
# Override knobs by editing the variables below, or by exporting them before
# the call:
#   STEPS=20 SEED=99 ./run_ltx2.sh "a sunset" out.webm

set -euo pipefail

# --- Models (current local paths) -----------------------------------------
DIT="/media/ilintar/D_SSD/models/ltx-2/ltx-2.3-22b-dev-Q6_K.gguf"
LLM="/media/ilintar/D_SSD/models/ltx-2/gemma-3-12b-it-UD-Q8_K_XL.gguf"
VAE="/media/ilintar/D_SSD/models/ltx-2/ltx-2.3-22b-dev_video_vae.safetensors"
EMB="/media/ilintar/D_SSD/models/ltx-2/ltx-2.3-22b-dev_embeddings_connectors.safetensors"
TOK="/media/ilintar/D_SSD/models/ltx-2/gemma_tokenizer.json"

# --- Core video parameters (override via env or edit here) ----------------
WIDTH="${WIDTH:-480}"
HEIGHT="${HEIGHT:-320}"
FRAMES="${FRAMES:-25}"
FPS="${FPS:-25}"
STEPS="${STEPS:-30}"
SEED="${SEED:-7}"

# --- Sampling / guidance --------------------------------------------------
CFG_SCALE="${CFG_SCALE:-3.0}"
RESCALE_SCALE="${RESCALE_SCALE:-0.7}"
STG_SCALE="${STG_SCALE:-1.0}"
STG_BLOCKS="${STG_BLOCKS:-[28]}"

# --- Args -----------------------------------------------------------------
PROMPT="${1:-a cinematic photograph of a sunset over the ocean}"
OUTPUT="${2:-output.webm}"
NEG_PROMPT="${NEG_PROMPT:-}"

# --- Resolve sd-cli path --------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SD_CLI="${SD_CLI:-${SCRIPT_DIR}/build/bin/sd-cli}"
if [[ ! -x "$SD_CLI" ]]; then
    echo "sd-cli not found at $SD_CLI — build it first (cmake --build build -j)" >&2
    exit 1
fi

echo "running LTX-2 vid_gen → $OUTPUT"
echo "  ${WIDTH}x${HEIGHT}, ${FRAMES} frames @ ${FPS} fps, ${STEPS} steps, seed=${SEED}"
echo "  cfg=${CFG_SCALE}, rescale=${RESCALE_SCALE}, stg=${STG_SCALE} blocks=${STG_BLOCKS}"

NEG_ARGS=()
if [[ -n "$NEG_PROMPT" ]]; then
    NEG_ARGS=(-n "$NEG_PROMPT")
fi

SD_QUIET_UNKNOWN_TENSORS=1 exec "$SD_CLI" -M vid_gen \
    --diffusion-model "$DIT" \
    --llm "$LLM" \
    --vae "$VAE" \
    -m "$EMB" \
    --gemma-tokenizer "$TOK" \
    -W "$WIDTH" -H "$HEIGHT" --video-frames "$FRAMES" --fps "$FPS" \
    --steps "$STEPS" --seed "$SEED" \
    --cfg-scale "$CFG_SCALE" --rescale-scale "$RESCALE_SCALE" \
    --stg-scale "$STG_SCALE" --stg-blocks "$STG_BLOCKS" \
    --diffusion-fa \
    --mmap \
    -p "$PROMPT" \
    "${NEG_ARGS[@]}" \
    -o "$OUTPUT"
