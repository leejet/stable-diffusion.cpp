#!/bin/bash
# End-to-end LTX-2.3 test script for DGX.
# Run as: ssh dgx.casa 'bash -s' < /tmp/ltxv_test.sh

set -e
set -o pipefail

SD_CLI=~/ltxv-sd-cpp/build-cuda/bin/sd-cli
MODEL=~/ltxv-models/ltx-2.3-22b-distilled.safetensors
OUT=/tmp/ltx23_out

mkdir -p "$OUT"
echo "=============================================="
echo "[1/3] vid_gen BF16 (no quant) — dry run"
echo "=============================================="
$SD_CLI -M vid_gen \
    -m "$MODEL" \
    -p "a cat walking across a grassy field" \
    -W 704 -H 480 --video-frames 9 \
    --steps 1 --cfg-scale 1 \
    -o "$OUT/dryrun.webp" \
    --seed 42 \
    -v 2>&1 | tail -80

echo ""
echo "=============================================="
echo "[2/3] Quantize to q8_0"
echo "=============================================="
$SD_CLI -M convert \
    -m "$MODEL" \
    -o "$OUT/ltx23_q8_0.gguf" \
    --type q8_0 \
    -v 2>&1 | tail -30

echo ""
echo "=============================================="
echo "[3/3] vid_gen with q8_0 GGUF"
echo "=============================================="
$SD_CLI -M vid_gen \
    -m "$OUT/ltx23_q8_0.gguf" \
    -p "a cat walking across a grassy field" \
    -W 704 -H 480 --video-frames 9 \
    --steps 4 --cfg-scale 1 \
    -o "$OUT/q8_output.webp" \
    --seed 42 \
    -v 2>&1 | tail -80

echo ""
echo "=============================================="
echo "Outputs in $OUT:"
ls -la "$OUT/"
