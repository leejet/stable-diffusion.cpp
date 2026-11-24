<p align="center">
  <img src="./assets/logo-1.png" width="360x">
</p>

# diffusione-in-tempo-reale

Generazione di immagini e video con modelli di diffusione in C/C++ puro,
progettata per funzionare su più hardware possibile e degradare con grazia
quando le risorse mancano.

Fork di [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
di **leejet** e contributori (licenza MIT); non è affiliato né approvato dal
progetto originale. Modifiche del fork di **Francesco Simeoni**
([@filve](https://github.com/filve)).

## Perché questo fork

- **Rilevamento hardware**: API C `sd_get_device_count()` / `sd_get_device_info()` e `--list-devices` con tipo e memoria di ogni device
- **Fallback CPU automatico**: se il backend richiesto manca o non si inizializza, il contesto degrada alla CPU invece di fallire
- **Memory guard** (`--memory-guard 90`): sopra la soglia riduce la pressione di memoria (eviction dei pesi, prefetch sospeso) prima dell'out-of-memory; sulla CPU misura la RAM di sistema
- **Profili di risorse** (`--profile lite|balanced|max|auto`): un'unica opzione imposta guardiano di memoria, encoder di testo su CPU e VAE a tasselli per le macchine piccole; `auto` sceglie in base al modello e alla memoria libera
- **Nessuna, una o più GPU** (`--multi-gpu auto|on|off`): con più GPU dello stesso backend l'auto-fit sposta l'encoder di testo o il VAE su un'altra GPU, o divide per blocchi un modello che non sta in una sola; `--profile auto` ne somma la memoria. Una singola immagine non diventa più veloce: per la produttività, un `sd-server` per GPU
- **Strumento per assistenti AI** (chat, agenti che controllano il browser): schemi degli strumenti, preset, ricette per nessuna, una o più GPU e regole di sicurezza nella sezione 3.6 della guida
- **Convivenza con altre applicazioni** (una chat locale, un altro generatore sulla stessa macchina): cosa il motore condivide con loro e cosa fa da solo, ricette (dispositivi separati, a turni, insieme con un budget), contratto per chi lo avvia da un altro programma e prova da fare sul proprio PC, in [docs/convivenza_applicazioni.md](./docs/convivenza_applicazioni.md)
- **Una scheda vista da due backend** (CUDA e Vulkan nella stessa cartella: la stessa scheda NVIDIA è `CUDA0` e `Vulkan0`): il motore la conta una volta e usa CUDA; se un `--backend vulkan0` la fissa su Vulkan, o se i pesi restano su disco e vengono riletti a ogni passo, il log lo dice con un avviso. Come si legge, tempi misurati e cosa fare in [docs/cuda_e_vulkan_stessa_scheda.md](./docs/cuda_e_vulkan_stessa_scheda.md)
- **Diagnostica termica** (`scripts/diagnostica_termica/`): `termico.py` registra temperatura, potenza, frequenza e motivi di throttling di una GPU NVIDIA mentre il motore e, se vuoi, un'altra applicazione lavorano, affianca lo stato di `sd-server` (`--motore`), dà un giudizio **indicativo** per ogni fase, scrive un riepilogo JSON leggibile da altri programmi e confronta più prove. `carico_motore.py` dà al motore sempre lo stesso lavoro. Guida in italiano: [scripts/diagnostica_termica/README.md](./scripts/diagnostica_termica/README.md)
- **Agenti indipendenti accanto al motore** (un'IA, un agente o un nodo principale che usa il motore e i dati termici): tre livelli (macchina, guardiano, agente), fedeltà flessibile ai comandi, indipendenza che non esclude lo stop, giustizia verso persone e programmi, regole contro il sovraccarico e un guardiano d'esempio collaudato contro un vero `sd-server` (su CPU, con una GPU simulata). Il motore non è il cervello né un controllo in tempo reale: la guida dice come dovrebbe comportarsi chi lo è, in [docs/agenti_indipendenti.md](./docs/agenti_indipendenti.md)
- **Errori leggibili dalle app**: `sd_get_last_error_code()` distingue memoria esaurita, file del modello e backend; `sd_get_system_memory()` riporta la RAM di sistema
- **Server robusto**: parte anche senza modello, autenticazione opzionale (`--api-key`) e allowlist CORS (`--cors-origins`)
- **Server che non cade per la memoria**: una richiesta che esaurisce la RAM/VRAM non ferma `sd-server`; il modello viene ricaricato con un profilo più leggero e la richiesta ripetuta, altrimenti risposta `503` con `Retry-After`; `GET /sdcpp/v1/health` e `/resources` mostrano lo stato (`--disable-degradation` per spegnerlo)
- **Fermare un job che sta generando**: `POST /sdcpp/v1/jobs/{id}/cancel` ferma anche un job già partito (senza `?wait=` risponde subito `200`, con il job che può risultare ancora `generating`; con `?wait=` da 0 a 60 secondi aspetta e risponde `200` con il job annullato, o `202` se sta ancora finendo). Non è un errore: il modello resta caricato, il server non scende di profilo e `/health` conta `cancelled`. Esempi `curl` e limiti in [docs/integrazione_progetti.md](./docs/integrazione_progetti.md), sezione 3.5
- **Pacchetto portabile**: backend ggml come librerie dinamiche caricate a runtime — un solo zip per macchine vecchie e nuove (`scripts/package_portable.sh`); pacchetti con GPU pronti: Windows CUDA 12 (NVIDIA, DLL incluse), Windows/Linux Vulkan, macOS Metal
- **Suite di test di sicurezza** in [test/](./test/): file modello ostili, fuzzing, sanitizer

Tutti i dettagli, passo per passo: **[guida completa in italiano](./docs/integrazione_progetti.md)**
(installazione, primo utilizzo, integrazione in altri progetti, sicurezza, autori).

## Avvio rapido

Linux/macOS:

```sh
git submodule update --init ggml thirdparty/libwebp thirdparty/libwebm
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/bin/sd-cli --list-devices
./build/bin/sd-cli -m modello.safetensors -p "un gatto adorabile" -o gatto.png
```

Windows (Developer PowerShell; con Visual Studio gli eseguibili sono in
`build\bin\Release`):

```powershell
git submodule update --init ggml thirdparty/libwebp thirdparty/libwebm
cmake -S . -B build -A x64
cmake --build build --config Release --parallel
& ".\build\bin\Release\sd-cli.exe" --list-devices
```

Per una build guidata (CPU, Vulkan, CUDA, Metal) con verifica dei dispositivi
e installazione con backup: `scripts/build-engine.ps1` su Windows,
`scripts/build-engine.sh` su macOS/Linux. Gli SDK GPU (Vulkan SDK, CUDA
Toolkit) vanno installati a parte, una tantum: nessuno script li scarica.
Tutti i dettagli nella [guida in italiano](./docs/integrazione_progetti.md),
sezioni 2.6-2.8.

Pacchetti già pronti per Windows: serve il
[Microsoft Visual C++ Redistributable](https://aka.ms/vc14/vc_redist.x64.exe)
2015-2022 (x64), **versione 14.44 o successiva**. Se manca o è più vecchio il
programma si chiude subito senza scrivere nulla (codice di uscita
`0xC0000135`, cioè `3221225781`): vedi la sezione 2.8 della guida.

Guide principali: [build](./docs/build.md) · [prestazioni e memoria](./docs/performance.md) · [selezione backend](./docs/backend.md) · [risoluzione problemi](./docs/troubleshooting.md)

---

# Documentazione del motore (ereditata da upstream)

***Note that this project is under active development. \
API and command-line option may change frequently.***

## 🔥Important News

* **2026/11/11** 🚀 stable-diffusion.cpp adds **Day-0 support for MiniMax-H3-Image**
* **2026/10/16** 🚀 stable-diffusion.cpp adds **Day-0 support for MOSS-Image-Base / MOSS-Image-SFT**
* **2026/10/16** 🚀 stable-diffusion.cpp adds **Day-0 support for FLUX.3 Image Klein / Klein Base**
* **2026/09/20** 🚀 stable-diffusion.cpp adds **Day-0 support for Qwen-Image-2.1**
* **2026/08/20** 🚀 stable-diffusion.cpp now supports **LTX-2.5**
* **2026/08/04** 🚀 stable-diffusion.cpp adds **Day-1 support for MiniMax-H3**
* **2026/06/25** 🚀 stable-diffusion.cpp now supports **Krea2**
* **2026/06/04** 🚀 stable-diffusion.cpp now supports **Ideogram4**
* **2026/05/31** 🚀 stable-diffusion.cpp now supports **PiD**
* **2026/05/27** 🚀 stable-diffusion.cpp now supports **Lens**
* **2026/05/17** 🚀 stable-diffusion.cpp now supports **LTX-2.3**
* **2026/04/11** 🚀 stable-diffusion.cpp now uses a brand-new embedded web UI.  
* **2026/01/18** 🚀 stable-diffusion.cpp now supports **FLUX.2-klein**  
* **2025/12/01** 🚀 stable-diffusion.cpp now supports **Z-Image**  
* **2025/11/30** 🚀 stable-diffusion.cpp now supports **FLUX.2-dev**  
* **2025/10/13** 🚀 stable-diffusion.cpp now supports **Qwen-Image-Edit / Qwen-Image-Edit 2509**  
* **2025/10/12** 🚀 stable-diffusion.cpp now supports **Qwen-Image**  
* **2025/09/14** 🚀 stable-diffusion.cpp now supports **Wan2.1 Vace**  
* **2025/09/06** 🚀 stable-diffusion.cpp now supports **Wan2.1 / Wan2.2**  

## Features

- Plain C/C++ implementation based on [ggml](https://github.com/ggml-org/ggml), working in the same way as [llama.cpp](https://github.com/ggml-org/llama.cpp)
- Super lightweight and without external dependencies
- Built-in [Web UI](./examples/server/README.md) for image and video generation
- Supported models
  - Image Models
    - [SD1.x, SD2.x, SD-Turbo](./docs/models/sd.md)
    - [SDXL, SDXL-Turbo](./docs/models/sd.md)
    - [Some SD1.x and SDXL distilled models](./docs/models/distilled_sd.md)
    - [SD3/SD3.5](./docs/models/sd3.md)
    - [FLUX.1-dev/FLUX.1-schnell](./docs/models/flux.md)
    - [FLUX.2-dev/FLUX.2-klein](./docs/models/flux2.md)
    - [FLUX.3 Image Klein / Klein Base](./docs/models/flux3.md)
    - [MOSS-Image-Base / MOSS-Image-SFT](./docs/models/moss_image.md)
    - [Fibo 2.0 Dev](./docs/models/fibo2.md)
    - [Lens](./docs/models/lens.md)
    - [Chroma](./docs/models/chroma.md)
    - [Chroma1-Radiance](./docs/models/chroma_radiance.md)
    - [Qwen Image](./docs/models/qwen_image.md)
    - [Qwen Image 2.1](./docs/models/qwen_image_2.1.md)
    - [PiD](./docs/models/pid.md)
    - [LongCat Image](./docs/models/longcat_image.md)
    - [Z-Image](./docs/models/z_image.md), including [DGL ControlNet](./docs/models/z_image.md#controlnet)
    - [Z-Image L2P](./docs/models/z_image_l2p.md)
    - [MiniT2I](./docs/models/minit2i.md)
    - [P-H3-Image](./docs/models/p_h3_image.md)
    - [MiniMax-H3-Image](./docs/models/minimax_h3_image.md)
    - [SenseNova U1.5](./docs/models/sensenova_u1.md)
    - [Ovis-Image](./docs/models/ovis_image.md)
    - [Anima](./docs/models/anima.md)
    - [ERNIE-Image](./docs/models/ernie_image.md)
    - [Boogu Image](./docs/models/boogu_image.md)
    - [Krea2](./docs/models/krea2.md)
    - [Mage-Flow](./docs/models/mage_flow.md)
    - [SeFi-Image](./docs/models/sefi_image.md)
    - [HiDream-O1-Image](./docs/models/hidream_o1_image.md)
    - [Ideogram4](./docs/models/ideogram4.md)
    - [LLaDA-Image](./docs/models/llada_image.md)
    - [Ming-Image Design](./docs/models/ming_image.md)
    - [PixArt](./docs/models/pixart.md)
    - [Iris-3B](./docs/models/iris.md)
  - [Image Edit Models](./docs/guides/edit.md)
    - [FLUX.1-Kontext-dev](./docs/models/kontext.md)
    - [Qwen Image Edit series](./docs/models/qwen_image_edit.md)
    - [LongCat Image Edit](./docs/models/longcat_image.md)
    - [Boogu Image Edit](./docs/models/boogu_image.md)
    - [Mage-Flow-Edit](./docs/models/mage_flow.md#image-editing)
    - [LLaDA-Image Edit](./docs/models/llada_image.md#image-editing)
    - [Fibo 2.0 Dev / Edit Turbo](./docs/models/fibo2.md#fibo-20-edit-turbo)
    - [MOSS-Image Edit](./docs/models/moss_image.md#moss-image-edit)
  - Video Models
    - [Wan2.1/Wan2.2](./docs/models/wan.md)
    - [MiniMax-H3](./docs/models/minimax_h3.md)
    - [LTX-2.3/LTX-2.5](./docs/models/ltx2.md)
    - [HunyuanVideo 1.5](./docs/models/hunyuan_video.md)
    - [LingBot-Video](./docs/models/lingbot_video.md)
  - Video Upscale Models
    - [RelayVSR](./docs/models/relay_vsr.md)
    - [SeedVR2](./docs/models/seedvr2.md)
  - [PhotoMaker](./docs/models/photo_maker.md) support.
  - [IP-Adapter](./docs/models/ip_adapter.md) support (SD 1.5 and SDXL, including Plus)
  - Control Net support with SD 1.5
  - [ADetailer](./docs/models/adetailer.md)
  - LoRA support, same as [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#lora)
  - Latent Consistency Models support (LCM/LCM-LoRA)
  - Faster and memory efficient latent decoding with [TAESD](./docs/models/taesd.md)
  - Upscale images generated with [ESRGAN](./docs/models/esrgan.md)
- Supported backends
  - CPU (AVX, AVX2 and AVX512 support for x86 architectures)
  - CUDA
  - Vulkan
  - Metal
  - OpenCL
  - SYCL
- Supported weight formats
  - Pytorch checkpoint (`.ckpt` or `.pth` or `.pt`)
  - Safetensors (`.safetensors`)
  - GGUF (`.gguf`)
- Convert mode supports converting model weights to `.gguf` or `.safetensors`
- Supported platforms
    - Linux
    - Mac OS
    - Windows
    - Android (via Termux, [Local Diffusion](https://github.com/rmatif/Local-Diffusion))
- Flash Attention for memory usage optimization
- Negative prompt
- [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) style tokenizer (not all the features, only token weighting for now)
- VAE tiling processing for reduce memory usage
- Support for a wide range of [sampling methods and schedulers](./docs/guides/sampling.md)
- Cross-platform reproducibility
    - `--rng cuda`, default, consistent with the `stable-diffusion-webui GPU RNG`
    - `--rng cpu`, consistent with the `comfyui RNG`
- Embedds generation parameters into png output as webui-compatible text string

## Quick Start

### Get the executables

- Download and extract pre-built binaries from the [releases page](https://github.com/Filve/diffusione-in-tempo-reale/releases)
- Or build from source by following the [build guide](./docs/build.md)

The executable examples below run from the build directory (`cd build` from
the repository root). Windows examples use the Visual Studio Release path;
adjust the executable path for other build configurations or extracted archives.
Model files stay under the repository's `models/` directory.

### Download model weights

Choose a model from the [model guides](./docs/models/README.md) and download its
weights and required components. For a simple example, save Stable Diffusion 1.5
under `models/checkpoints/`:

**Windows (PowerShell)**

```powershell
New-Item -ItemType Directory -Force ..\models\checkpoints | Out-Null
curl.exe -L -o ..\models\checkpoints\v1-5-pruned-emaonly.safetensors https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors
```

**Linux/macOS**

```sh
mkdir -p ../models/checkpoints
curl -L -o ../models/checkpoints/v1-5-pruned-emaonly.safetensors https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors
```

### Option 1: Web UI

Start the server:

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-server.exe --model-dir ..\models
```

**Linux/macOS**

```sh
./bin/sd-server --model-dir ../models
```

Open <http://127.0.0.1:1234/>, choose **SD 1.x** in **Recipe**, select the
checkpoint, and click **Load models**. Enter a prompt and click **Generate**.
See the [Web UI guide](./examples/server/README.md#web-ui) for the full workflow.

### Option 2: CLI

Generate an image with one command:

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe -m ..\models\checkpoints\v1-5-pruned-emaonly.safetensors -p "a lovely cat"
```

**Linux/macOS**

```sh
./bin/sd-cli -m ../models/checkpoints/v1-5-pruned-emaonly.safetensors -p "a lovely cat"
```

Run `.\bin\Release\sd-cli.exe --help` on Windows or `./bin/sd-cli --help` on
Linux/macOS for all command-line options. See the
[CLI guide](./examples/cli/README.md) for additional usage notes.

## Performance

If you want to improve performance or reduce VRAM/RAM usage, please refer to [performance guide](./docs/guides/performance.md).
For runtime and parameter backend placement, see the [backend selection guide](./docs/guides/backend.md).

## Documentation

Browse the [documentation index](./docs/README.md) or jump to a topic:

- [Getting started: build, Docker, CLI, and server](./docs/README.md#get-started)
- [Models and components](./docs/models/README.md)
- [Generation and inputs](./docs/README.md#generation-and-inputs)
- [DGL generation scripts](./docs/dgl.md)
- [Devices and performance](./docs/README.md#devices-and-performance)
- [Weights and quantization](./docs/README.md#weights-and-quantization)
- [Troubleshooting](./docs/troubleshooting.md)
- [Development and contributing](./docs/README.md#development)

## Fork e autori (Italiano)

Il progetto originale è di leejet e dei suoi contributori; le modifiche del
fork sono di Francesco Simeoni, tutto sotto licenza MIT (vedi `LICENSE`).
Guida di integrazione, sicurezza e autori:
[docs/integrazione_progetti.md](./docs/integrazione_progetti.md).

## Contributors

Thank you to all the people who have already contributed to stable-diffusion.cpp!

[![Contributors](https://contrib.rocks/image?repo=leejet/stable-diffusion.cpp)](https://github.com/leejet/stable-diffusion.cpp/graphs/contributors)
