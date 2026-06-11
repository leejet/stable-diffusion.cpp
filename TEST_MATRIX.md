# LTX 2.3 Test Matrix — Complete Generation Log

> All tests: 4× GTX 1080 Ti (200W), row-split pairs, 8 steps, Euler/LTX2, CFG 1.1, 25fps, GPU VAE 14×14 t4/o1, `--diffusion-fa --audio-vae-cpu --vae-conv-direct --mmap`

## Quick Reference — All Tests

| # | Style | LoRA | Duration | Resolution | Tokens | Time | Seed | Pair | Output |
|---|---|---|---|---|---|---|---|---|---|
| C1 | Cyberpunk Hook | cyberpunk | 3s (73f) | 640×1152 | 7,200 | 16.2m | 42 | 2,3 | `shot1_3s_640x1152.webm` |
| C2 | Cyberpunk Drone | cyberpunk | 4s (97f) | 640×1152 | 9,360 | 20.7m | 42 | 0,1 | `shot2_4s_640x1152.webm` |
| C3 | Cyberpunk Comms | cyberpunk | 5s (121f) | 576×1056 | 9,504 | 24.0m | 43 | 2,3 | `shot3_5s_576x1056.webm` |
| C4 | Cyberpunk Sprint | cyberpunk | 6s (145f) | 512×928 | 8,816 | 19.7m | 44 | 0,1 | `shot4_6s_512x928.webm` |
| C5 | Cyberpunk Command | cyberpunk | 6s (145f) | 512×928 | 8,816 | 21.7m | 45 | 2,3 | `shot5_6s_512x928.webm` |
| C6 | Cyberpunk Rooftop | cyberpunk | 6s (145f) | 416×736 | 5,681 | 13.8m | 46 | 0,1 | `shot6_6s_416x736.webm` |
| D1 | Ancient Fantasy Celestial | dialog | 5s (121f) | 576×1056 | 9,504 | 23.6m | 101 | 0,1 | `dialog_test_5s_576x1056.webm` |
| D2 | Modern Office | dialog | 6s (145f) | 512×928 | 8,816 | 24.8m | 102 | 2,3 | `dialog_test_6s_512x928.webm` |
| D3 | Ancient Fantasy 10s | dialog | 10s (249f) | 416×736 | 9,568 | 24.1m | 103 | 0,1 | `dialog_test_10s_416x736.webm` |
| D4 | Coffee Shop | dialog | 5s (121f) | 576×1056 | 9,504 | 23.8m | 104 | 0,1 | `dialog_d4_cafe_5s_576x1056.webm` |
| D5 | Rooftop Terrace | dialog | 6s (145f) | 512×928 | 8,816 | 24.1m | 105 | 0,1 | `dialog_d5_rooftop_6s_512x928.webm` |
| D6 | Library Meeting | dialog | 10s (249f) | 416×736 | 9,568 | 🔄 | 106 | 0,1 | `dialog_d6_library_10s_416x736.webm` |
| A1 | Training Duel | none | 4s (97f) | 640×1152 | 9,360 | 21.7m | 201 | 2,3 | `anime_shonen_4s_640x1152.webm` |
| A2 | Warehouse Fight | none | 5s (121f) | 576×1056 | 9,504 | 23.8m | 202 | 2,3 | `anime_shonen_a2_fight_5s_576x1056.webm` |
| A3 | Power Transformation | none | 6s (145f) | 512×928 | 8,816 | 23.5m | 203 | 2,3 | `anime_shonen_a3_transform_6s_512x928.webm` |
| A4 | Academy Episode | none | 10s (249f) | 416×736 | 9,568 | 🔄 | 204 | 2,3 | `anime_shonen_a4_academy_10s_416x736.webm` |

**Totals (completed):** 14 videos | **Total time:** ~5.3 hours | **Avg time:** 22.0 min/video

---

## Cyberpunk Shots (C1-C6)

All use trigger: `C6b4rP8nk, cinematic cyberpunk neo-noir aesthetic,`
LoRA: `<lora:loras/styles/cyberpunk/Cinematic_sci-fi-cyberpunk.safetensors:0.8>`
Full prompts in: `models/loras/styles/cyberpunk/FirstVideo.txt`

### C1 — The Hook 3s
```
Prompt (abbreviated): Low-angle crane shot ascends, revealing a character in dark leather
trench coat silhouetted against futuristic cityscape. Rain, pink/cyan neon, character whips
head toward camera with intense gaze.
Pair: 2,3 | Seed: 42 | Time: 972.9s (16.2m) | Tokens: 7,200
```

### C2 — Drone Corridor 4s
```
Prompt (abbreviated): Wide-angle handheld tracking shot through narrow grimy industrial
corridor. Concrete walls with blue neon, tactical drone with red sensors gliding above,
reflections in puddles.
Pair: 0,1 | Seed: 42 | Time: 1240.4s (20.7m) | Tokens: 9,360
```

### C3 — Dialogue/Comms 5s
```
Prompt (abbreviated): Close-up tracking camera arcs around figure in dark tactical streetwear.
Teal/orange light, cybernetic neck implants. Character taps comms implant, mutters: "The
encryption is holding, I need a clearer path forward now!"
Pair: 2,3 | Seed: 43 | Time: 1440.9s (24.0m) | Tokens: 9,504
```

### C4 — Soldier Sprinting 6s
```
Prompt (abbreviated): Fast-tracking handheld camera follows soldier in dark tactical jacket
sprinting down rain-slicked urban alley. Boots splash through water, neon signs create
shadows, soldier reaches for weapon.
Pair: 0,1 | Seed: 44 | Time: ~1182s (19.7m) | Tokens: 8,816
```

### C5 — Command Center 6s
```
Prompt (abbreviated): Rapid handheld crane down through high-tech command center. Character
slams hand onto holographic terminal, red alert flashes across monitor, crimson light on
focused gaze.
Pair: 2,3 | Seed: 45 | Time: ~1302s (21.7m) | Tokens: 8,816
```

### C6 — Rooftop Cliffhanger 6s
```
Prompt (abbreviated): Low-angle dolly push-in on character on wet rooftop looking at skyline.
Rain streams down, cityscape pulses pink and cyan. Character presses comms, shouts: "It's a
trap, they already knew I was coming!"
Pair: 0,1 | Seed: 46 | Time: ~828s (13.8m) | Tokens: 5,681
```

---

## Dual Character Dialog (D1-D6)

LoRA: `<lora:loras/dialog/LTX2.3-IC-LORA-Dual-Character.safetensors:0.8>`
Prompt structure: `[Scene] + [Characters: Female/Male descriptions] + [Shots with camera + timing]`
All dialog in English.

### D1 — Ancient Fantasy Celestial 5s
```
[Scene] Ancient Chinese celestial realm, pink and gold clouds at sunset, ethereal floating
pavilions, warm light piercing through mist.
[Characters] Female: silver hairpins, teal silk robes with gold embroidery, luminous jade
pendant. Male: black hair with jade crown, dark blue warrior robes with silver trim.
Shot 1 (Wide, 2s): Camera descends through clouds revealing two figures on marble bridge.
Shot 2 (Medium, 3s): Camera arcs. Female extends hand with pendant, speaks softly. Male nods.
Pair: 0,1 | Seed: 101 | Time: 1417.4s (23.6m) | Tokens: 9,504
```

### D2 — Modern Office 6s
```
[Scene] Glass-walled corporate office at dusk, skyline through windows, warm amber lamps
vs cool blue monitors, rain on windows.
[Characters] Female: tailored navy blazer, silver-streaked bob, intense eyes. Male: grey
hoodie over collared shirt, nervous, adjusting glasses.
Shot 1 (Wide, 2s): Both at holographic display table.
Shot 2 (Medium, 4s): Female points at data, speaking rapidly. Male's eyes widen, types
frantically on transparent keyboard.
Pair: 2,3 | Seed: 102 | Time: 1490.2s (24.8m) | Tokens: 8,816
```

### D3 — Ancient Fantasy Extended 10s
```
[Scene] Celestial realm, pink/purple sunset clouds, golden sunlight, ethereal atmosphere.
[Characters] Female: bun with silver hairpins, sky-blue flowing gown with gold lotus,
holding jade bottle. Male: silver crown, moon-white immortal robe with cloud embroidery,
white jade whisk.
Shot 1 (Wide, 3s): Overhead, two immortals above clouds.
Shot 2 (Medium, 4s): Camera pushes in. Female turns, whispers softly.
Shot 3 (Close-up, 3s): Male looks at bottle, replies warmly with gentle nod.
Pair: 0,1 | Seed: 103 | Time: 1444.4s (24.1m) | Tokens: 9,568
```

### D4 — Coffee Shop 5s
```
[Scene] Cozy urban coffee shop at golden hour, sunlight through large windows, exposed
brick walls with art, espresso machines humming.
[Characters] Sarah: curly auburn hair, cream knitted sweater, nervously turning coffee cup.
James: clean-shaven, navy blazer over white t-shirt, old leather journal.
Shot 1 (Wide, 2s): They sit across small wooden table by window, steam rising from drinks.
Shot 2 (Medium, 3s): Camera pushes in. Sarah looks up with hesitant smile, speaks softly.
James leans forward, journal forgotten.
Pair: 0,1 | Seed: 104 | Time: 1430.0s (23.8m) | Tokens: 9,504
```

### D5 — Rooftop Terrace 6s
```
[Scene] Rooftop garden at twilight, string lights, city skyline against purple-orange sky,
potted herbs, wine bottle on iron table.
[Characters] Elena: sleek dark bob, burgundy silk blouse, silver pendant, confident but tired.
Marcus: close-cropped hair, stubble, brown leather jacket over grey henley, nervous.
Shot 1 (Wide, 2s): At railing, city lights below, comfortable silence.
Shot 2 (Medium, 4s): Camera dollies sideways. Elena turns, sets glass down, speaks quietly.
Marcus's jaw tightens, then he meets her eyes.
Pair: 0,1 | Seed: 105 | Time: 1444.1s (24.1m) | Tokens: 8,816
```

### D6 — Library Meeting 10s 🔄
```
[Scene] Quiet library reading room at golden hour, tall arched windows, dust motes in
sunbeams, towering wooden bookshelves, grandfather clock ticking.
[Characters] Claire: wire-rimmed glasses, honey-blonde braid, cream cardigan, poetry book.
Daniel: dark curly hair, navy peacoat, holding same poetry book, hesitant half-smile.
Shot 1 (Wide, 3s): Camera dollies through aisles. Claire reads in alcove. Daniel notices.
Shot 2 (Medium, 4s): Daniel approaches, clears throat. Claire looks up. He gestures at book.
Shot 3 (Close-up, 3s): Hands tracing same passage. Claire looks up, eyes bright.
Pair: 0,1 | Seed: 106 | Tokens: 9,568
```

---

## Anime Shonen (A1-A4)

No LoRA. Pure prompt-driven 2D anime style.
Prompt structure: `[Style tags] + [Scene] + [Characters] + [Shots with camera + timing]`

### A1 — Training Duel 4s
```
High-octane shonen anime action, 2D anime, vibrant cel shading, dynamic speed lines.
[Scene] Ancient Japanese temple courtyard at golden hour, cherry blossom petals swirling.
[Characters] Young warrior: spiky black hair, damaged orange gi with blue undershirt, red
headband. Mentor: silver hair ponytail, calm eyes, white haori over dark hakama, katana.
Shot 1 (Wide, 2s): Warrior stands fists clenched, staring down mentor. Cherry blossoms blow.
Shot 2 (Dynamic pan, 3s): Fast whip-pan. Student lunges with afterimages. Mentor sidesteps.
Pair: 2,3 | Seed: 201 | Time: 1301.9s (21.7m) | Tokens: 9,360
```

### A2 — Warehouse Fight 5s
```
Intense shonen anime fight, 2D anime, dramatic lighting, impact frames, motion blur.
[Scene] Abandoned industrial district at night, full moon through broken windows, sparks
from electrical panels, rain on cracked concrete.
[Characters] Hero: wild silver-white hair, piercing blue eyes, black vest over torn red
shirt, bandaged fists. Villain: tall, dark coat, cold grey eyes, cybernetic arms with
red energy conduits.
Shot 1 (Wide, 2s): Fighters face off, rain dripping, villain's arms humming.
Shot 2 (Fast action, 3s): Hero dashes, throws rapid punches. Villain blocks with metal
forearms, sparks exploding, counter-attacks with devastating palm strike.
Pair: 2,3 | Seed: 202 | Time: 1429.8s (23.8m) | Tokens: 9,504
```

### A3 — Power Transformation 6s
```
Epic shonen anime transformation, 2D anime, dramatic energy effects, glowing aura, particles.
[Scene] Desolate mountain peak under storm, dark clouds, lightning, ancient stone pillars
with glowing runes, wind whipping.
[Characters] Young warrior: spiky hair turning electric blue, tattered clothes, body covered
in glowing blue vein markings, eyes blazing white.
Shot 1 (Wide, 2s): Warrior kneels in rune circle, body trembling as energy builds.
Shot 2 (Dynamic close-up, 4s): Camera orbits. Head snaps up, eyes blazing. Shockwave erupts,
pillars shatter. Hair ignites blue, roaring aura engulfs body, debris floats upward.
Pair: 2,3 | Seed: 203 | Time: 1411.9s (23.5m) | Tokens: 8,816
```

### A4 — Academy Episode 10s 🔄
```
Shonen anime multi-shot episode, 2D anime, vibrant colors, cinematic, emotional storytelling.
[Scene] Futuristic academy courtyard, floating platforms with light bridges, holographic
cherry blossoms, holographic clock tower, golden afternoon.
[Characters] Riku: messy dark blue hair, bright orange eyes, navy/silver academy uniform,
floating mechanical fox drone. Aiko: long black hair in high ponytail with red ribbon,
violet eyes, slim silver bracelet.
Shot 1 (Wide, 3s): Riku runs across light bridge waving. Aiko rolls eyes but smiles.
Shot 2 (Medium, 4s): They walk together. Riku gestures excitedly, drone mirrors movements.
Shot 3 (Close-up, 3s): At fountain. Riku pulls out glowing crystal. Aiko's bracelet pulses.
Pair: 2,3 | Seed: 204 | Tokens: 9,568
```

---

## Key

- 🔄 = currently generating
- ✅ = completed
- ❌ = failed

## Command Templates

```bash
# Pair A (GPUs 0,1):
./run_pair.sh "0,1" "PROMPT" FRAMES W H SEED OUTPUT [dialog|cyberpunk|""]

# Pair B (GPUs 2,3):
./run_pair.sh "2,3" "PROMPT" FRAMES W H SEED OUTPUT [dialog|cyberpunk|""]
```

## Key Findings

1. **~9,500 tokens takes ~23-24 min regardless of resolution** — whether 5s at 576×1056 or 10s at 416×736, it's the same denoising cost
2. **No-LoRA is slightly faster** — anime (no LoRA): 1302-1430s. Dialog (with LoRA): 1417-1490s. The 312 MB LoRA adds ~2 min of overhead
3. **5 segments without LoRA vs 6 with** — cleaner graph-cut, marginally less memory fragmentation
4. **10s at 416×736 is the sweet spot** — max duration at max token budget, with time nearly identical to 5s high-res
5. **Dual Character LoRA works for English dialog** — handles modern and fantasy settings equally well
6. **Anime shonen style achievable without LoRA** — strong prompt engineering with [Style] + [Scene] + [Characters] + [Shots] structure works
