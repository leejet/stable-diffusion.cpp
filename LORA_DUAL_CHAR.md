# LTX2.3-IC-LORA-Dual-Character

**Source:** https://civitai.com/models/2500098
**Developer:** Maque AI (麻雀 AI)
**License:** Apache-2.0
**File:** `models/LTX2.3-IC-LORA-Dual-Character.safetensors` (313 MB, rank 32)

## Purpose
Identity Consistency LoRA for two-person dialogue, character interaction, and multi-shot cinematic scenes.

## Best For
- Two-person dialogue scenes (subtle movements, conversation)
- Ancient Fantasy, Modern, 3D Anime styles
- I2V and T2V with structured multi-shot prompts

## Limitations
- Complex physical interactions (fighting, wrestling) may cause deformation
- High-intensity action not recommended

## Recommended Parameters
- Resolution: 16:9 (768×512 or 1280×720)
- Duration: ≥10s, 24 FPS
- LoRA weight: 0.6 - 1.0 (standalone), 0.3 - 0.5 (stacked)
- Steps: 8+ (model default)
- CFG: 1.0-1.1 (distilled model)

## Prompt Structure
[Scene] + [Characters] + [Storyboard shots with camera language]

## Usage with sd-cli
```
--lora-model-dir models/
-p "<lora:LTX2.3-IC-LORA-Dual-Character.safetensors:0.8> [Scene] ... [Characters] ... Shot 1..."
```
