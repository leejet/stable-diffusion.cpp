# Integrare diffusione-in-tempo-reale nei propri progetti

Guida in italiano per installare, inizializzare e usare questo fork di
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) dentro
altri progetti (app desktop, server, altre applicazioni AI).

Indice:

1. [Cosa aggiunge questo fork](#1-cosa-aggiunge-questo-fork)
2. [Installazione](#2-installazione) (prerequisiti, build, primo utilizzo, build GPU guidata, riepilogo da zero, problemi comuni, pacchetto portabile)
3. [Come collegarlo a un progetto](#3-come-collegarlo-a-un-progetto) (CMake, API C, FFI, processo, HTTP, strumento per assistenti AI)
4. [Retrocompatibilità e risorse](#4-retrocompatibilità-e-risorse)
5. [Sicurezza e test](#5-sicurezza-cosa-sapere-e-cosa-fare)
6. [Autori e licenza](#6-autori-e-licenza)
7. [Cronologia delle modifiche del fork](#7-cronologia-delle-modifiche-del-fork)

## 1. Cosa aggiunge questo fork

Il motore resta quello di stable-diffusion.cpp (C++17 + ggml). Le aggiunte
servono a farlo funzionare su più macchine possibile e a fallire in modo
controllato:

| Funzione | Cosa fa | Default |
|---|---|---|
| Rilevamento device | `sd_get_device_count()` / `sd_get_device_info()` elencano CPU, GPU e acceleratori con tipo e memoria, prima di caricare un modello | sempre disponibile |
| Fallback CPU | se il backend richiesto non esiste o non si inizializza, il contesto passa alla CPU con un avviso invece di fallire | **attivo** |
| Memory guard | sopra una soglia di memoria libera pressione (evict dei pesi, prefetch sospeso) prima che l'allocazione fallisca | **disattivo** |
| Profili di risorse | `--profile lite\|balanced\|max\|auto` imposta in un colpo guardiano di memoria, VAE a tasselli, CFG non raggruppato e `--multi-gpu on`; il calcolo resta sulle GPU (l'encoder di testo sulla CPU non fa più parte di `lite`); `auto` sceglie dalla dimensione del modello e dalla memoria libera | **disattivo** (come `max`) |
| Più GPU | con più GPU discrete dello stesso backend l'auto-fit sposta l'encoder di testo o il VAE su un'altra GPU, o divide per blocchi un modello che non sta in una sola; `--profile auto` ne somma la memoria. `--multi-gpu auto\|on\|off`. Non rende più veloce una singola immagine (per la produttività: un `sd-server` per GPU, sezione 3.6) | **attivo** (`auto`: la seconda GPU entra solo se serve) |
| Errori classificati e memoria di sistema | dopo un fallimento `sd_get_last_error()` dà un codice (memoria esaurita, file del modello, backend, ...) e il motivo; `sd_get_system_memory()` riporta RAM totale, disponibile e del processo | sempre disponibile |
| Server senza modello | `sd-server` parte anche senza `-m`: device e upscale funzionano, la generazione risponde con un errore chiaro | **attivo** |
| Server che resta in piedi | una richiesta che esaurisce la memoria non ferma `sd-server`: il modello viene ricaricato con un profilo più leggero (`max > balanced > lite > lite-disk > lite-hybrid > lite-cpu`: la CPU entra solo da `lite-hybrid` in poi) e la richiesta ripetuta; un device che si guasta viene sostituito da un'altra GPU prima che la CPU prenda il lavoro; se nemmeno il più leggero basta, risposta `503` con `Retry-After`; dopo una pausa il server risale. `GET /sdcpp/v1/health` (profilo, device, moduli sulla CPU) e `/resources` | **attivo** (`--disable-degradation` lo spegne) |
| Autenticazione server | `--api-key` (Bearer o `X-API-Key`) e allowlist CORS `--cors-origins` | **disattivi** |
| Pacchetto portabile | backend ggml come librerie dinamiche caricate a runtime: un solo pacchetto, ogni macchina usa ciò che ha | opzionale in build |
| Release automatiche | workflow `release-portable.yml`: zip con GPU (Windows CUDA 12 e Vulkan, Linux Vulkan, macOS Metal) a ogni tag `portable-v*` o avvio manuale | su tag |
| Uso da parte di assistenti AI | sezione 3.6: strumenti (`generate_image`, `engine_status`), preset, ricette per nessuna, una o più GPU e regole di sicurezza per chat e agenti che controllano il browser | documentazione |
| Convivenza con altre applicazioni | guida [convivenza_applicazioni.md](./convivenza_applicazioni.md): risorse condivise con una chat locale o un altro programma, ricette (dispositivi separati, a turni, insieme con un budget), contratto per chi avvia il motore da un altro programma, prova da fare sul proprio PC | documentazione |
| Suite di test | `test/`: smoke di sicurezza, robustezza API, fuzzer del caricamento modelli, pianificatore multi-GPU con GPU simulate | manuale |

L'unico cambio di comportamento predefinito rispetto a upstream è il fallback
CPU (prima un backend non disponibile causava un errore). Si disattiva con
`disable_backend_fallback = true` (API) o `--disable-backend-fallback` (CLI).

## 2. Installazione

### 2.1 Prerequisiti

- **CMake** >= 3.12 e un compilatore **C++17** (CMake >= 3.15 per usare il
  comando `cmake --install` riportato sotto):
  - macOS: `xcode-select --install` (Apple clang) e `brew install cmake`
  - Linux (Debian/Ubuntu): `sudo apt install build-essential cmake git`
  - Windows: Visual Studio 2022 (workload "Sviluppo di applicazioni desktop con C++") oppure MSYS2; CMake incluso o da cmake.org
- **Python 3** solo per gli script di test e di conversione (facoltativo).
- SDK del backend desiderato, se non CPU: CUDA Toolkit per `-DSD_CUDA=ON`,
  Vulkan SDK per `-DSD_VULKAN=ON`, ecc. (dettagli in [build.md](./build.md)).

### 2.2 Clonazione

```shell
git clone https://github.com/Filve/diffusione-in-tempo-reale.git
cd diffusione-in-tempo-reale
git submodule update --init ggml thirdparty/libwebp thirdparty/libwebm
```

Senza i sottomoduli la configurazione CMake fallisce. Il sottomodulo
`examples/server/frontend` serve solo per l'interfaccia web del server ed è
facoltativo. Eseguire i comandi dalla cartella principale del repository; in
Windows usare PowerShell, il Developer PowerShell/Prompt di Visual Studio o
Git Bash, con `git` e `cmake` disponibili nel `PATH`.

### 2.3 Build e installazione passo-passo

I comandi cambiano leggermente in base alla shell e al generatore CMake.
L'opzione `-DCMAKE_BUILD_TYPE=Release` seleziona la configurazione per
generatori a configurazione singola (per esempio Ninja o Make); con Visual
Studio la configurazione si sceglie invece in fase di build con
`--config Release`.

#### Linux e macOS (Ninja o Make)

```shell
# Dalla radice del repository. Senza backend GPU esplicito si compila la CPU.
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
# Aggiungere alla configurazione, se richiesto e se l'SDK è installato:
# -DSD_VULKAN=ON oppure -DSD_CUDA=ON

cmake --build build --parallel

# Verifica
./build/bin/sd-cli --list-devices
```

#### Windows con Visual Studio

Aprire un Developer PowerShell/Prompt di Visual Studio con gli strumenti C++
installati, quindi eseguire dalla radice del repository:

```powershell
cmake -S . -B build -A x64
# Aggiungere alla configurazione, se richiesto e se l'SDK è installato:
# -DSD_VULKAN=ON oppure -DSD_CUDA=ON

cmake --build build --config Release --parallel
& ".\build\bin\Release\sd-cli.exe" --list-devices
```

La build Visual Studio è multi-config: se si è compilato `Debug`, l'eseguibile
è invece `.\build\bin\Debug\sd-cli.exe`. Con Ninja su Windows, impostando
`-G Ninja -DCMAKE_BUILD_TYPE=Release` in fase di configurazione, il percorso è
`.\build\bin\sd-cli.exe`. Se il file non viene trovato, verificare la
configurazione selezionata e il percorso stampato da MSBuild, senza spostare
la working directory dentro `build\bin`.

Per installare la libreria e usarla da un altro progetto CMake (facoltativo):

```shell
# Linux/macOS
cmake --install build --prefix "$HOME/.local/sd"
```

```powershell
# Windows; usare un percorso di installazione a cui l'utente può scrivere.
cmake --install build --config Release --prefix "$HOME\sd-install"
```

L'installazione fornisce l'header pubblico `stable-diffusion.h`, le librerie e
i file di configurazione CMake/pkg-config. Non installa gli eseguibili di
esempio: `sd-cli` e `sd-server` restano nella directory di build. Il formato
della libreria dipende dalla piattaforma e dal tipo di build (statica per
default); per usarla da un altro progetto seguire la sezione 3 e mantenere
coerenti architettura, compilatore e configurazione. La directory di
installazione non viene aggiunta automaticamente al `PATH`.

### 2.4 Primo utilizzo: generare un'immagine

Serve un file di pesi (`.safetensors` o `.gguf`), **non incluso** nel
progetto: si scarica da fonti affidabili (es. Hugging Face) scegliendo un
modello adatto alla macchina (vedi [sd.md](./sd.md), [flux.md](./flux.md) e
gli altri documenti per famiglia di modello; per macchine piccole preferire
versioni quantizzate `.gguf`, vedi
[quantization_and_gguf.md](./quantization_and_gguf.md)).

```shell
# generazione base
./build/bin/sd-cli -m modello.safetensors -p "un gatto astronauta" -o gatto.png

# macchina con poca memoria: guardiano + pesi in RAM + budget VRAM prudente
./build/bin/sd-cli -m modello.safetensors -p "un gatto astronauta" -o gatto.png \
    --memory-guard 90 --offload-to-cpu --max-vram -1

# da immagine esistente (img2img)
./build/bin/sd-cli -m modello.safetensors -p "in stile acquerello" \
    -i foto.png --strength 0.6 -o acquerello.png
```

Con Visual Studio su Windows, eseguire dalla radice del repository usando
PowerShell e il percorso della configurazione compilata:

```powershell
& ".\build\bin\Release\sd-cli.exe" -m ".\modello.safetensors" `
    -p "un gatto astronauta" -o ".\gatto.png"
```

Se si è usato Ninja, rimuovere `Release` dal percorso:
`.\build\bin\sd-cli.exe`. I percorsi relativi ai modelli e ai file di output
sono risolti rispetto alla working directory corrente.

Opzioni utili: `--steps`, `--cfg-scale`, `-W`/`-H` (dimensioni), `--seed`,
`-v` (log dettagliati). Elenco completo: `sd-cli --help`.

### 2.5 Primo utilizzo: il server HTTP

```shell
./build/bin/sd-server -m modello.safetensors --api-key mia-chiave
```

Con Visual Studio su Windows, da PowerShell:

```powershell
& ".\build\bin\Release\sd-server.exe" -m ".\modello.safetensors" `
    --api-key "mia-chiave"
```

Con Ninja usare `.\build\bin\sd-server.exe`. Per richieste HTTP da PowerShell
usare `curl.exe` (non l'alias `curl` presente in alcune versioni di
PowerShell), oppure `Invoke-RestMethod`.

Da un client qualsiasi:

```shell
curl -s http://127.0.0.1:1234/sdcpp/v1/capabilities -H "X-API-Key: mia-chiave"

curl -s -X POST http://127.0.0.1:1234/sdcpp/v1/img_gen \
     -H "Content-Type: application/json" -H "X-API-Key: mia-chiave" \
     -d '{"prompt": "un gatto astronauta", "width": 512, "height": 512}'
```

Da una seconda finestra PowerShell su Windows, sostituire `curl` con
`curl.exe`, per esempio:

```powershell
curl.exe -s "http://127.0.0.1:1234/sdcpp/v1/capabilities" `
    -H "X-API-Key: mia-chiave"
```

Sono disponibili anche endpoint compatibili OpenAI (`/v1/images/generations`)
e AUTOMATIC1111 (`/sdapi/v1/txt2img`). Dettagli di sicurezza in sezione 5.

### 2.6 Windows: preparare un motore CPU + GPU per un'applicazione

Una build Windows produce eseguibili e librerie Windows, non librerie macOS
(`Darwin`) o Linux. Il nome `win32` usato da Node/Electron indica Windows,
anche se l'eseguibile è a 64 bit; non descrive il backend GPU. Vulkan, CUDA e
Metal sono backend differenti: non si attivano soltanto copiando file in una
cartella chiamata `vulkan` o `darwin`.

Per scegliere la build con un menu interattivo: `scripts/build-engine.ps1`
su Windows, `scripts/build-engine.sh` su macOS/Linux. Ogni sistema compila
solo i propri binari; per gli altri sistemi gli script rimandano alla build
su quella macchina o al workflow `release-portable.yml` (tag `portable-v*`).

Per una build Windows con CPU e GPU:

1. Installare Visual Studio Build Tools con C++ e Windows SDK, CMake e Ninja.
2. Installare gli SDK dei backend desiderati. **Questa installazione è sempre
   manuale e una tantum**: né gli script di build né le applicazioni ospiti
   scaricano SDK automaticamente (per sicurezza e trasparenza; se mancano, lo
   script si ferma con un messaggio chiaro e non tocca nulla). Da un
   PowerShell **come amministratore**:

   ```powershell
   winget install LunarG.VulkanSDK     # backend Vulkan (NVIDIA/AMD/Intel)
   winget install -e --id KhronosGroup.VulkanSDK
   winget install Nvidia.CUDA          # backend CUDA (solo NVIDIA)
   winget install -e --id Nvidia.CUDA
   ```

   In alternativa, gli installer grafici: [Vulkan SDK](https://vulkan.lunarg.com/sdk/home)
   e [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).
   `nvidia-smi` da solo non basta: è un'utilità del driver, non il compilatore
   CUDA (`nvcc`). **Dopo l'installazione aprire un terminale nuovo**: le
   variabili d'ambiente (`VULKAN_SDK`, il `PATH` di `nvcc`) esistono solo
   nelle finestre aperte dopo.

   Avvertenza CUDA: `nvcc` supporta solo le versioni MSVC testate da NVIDIA;
   con un Visual Studio molto recente la configurazione CUDA può fallire
   finché NVIDIA non pubblica un Toolkit compatibile. Vulkan non ha questo
   vincolo ed è la scelta più sicura per iniziare; copre anche le GPU AMD e
   Intel con driver Vulkan.
3. Verificare/aggiornare i driver della GPU. Un SDK serve alla compilazione;
   il driver serve all'esecuzione. Su un altro computer senza SDK il motore
   compilato funziona: all'utente finale serve solo il driver.
4. Chiudere l'applicazione ospite e ogni `sd-cli`/`sd-server` in esecuzione
   prima di compilare e soprattutto prima di sostituire il motore.
5. Dalla radice del repository, nel Developer PowerShell x64, eseguire:

```powershell
# CPU con varianti x86 e backend Vulkan dinamico
.\scripts\build-windows-gpu.ps1 -Backend Vulkan -Destination "$HOME\sd-install\bin"

# In alternativa, per NVIDIA con CUDA Toolkit:
# .\scripts\build-windows-gpu.ps1 -Backend CUDA -BuildDirectory build-windows-cuda -Destination "$HOME\sd-install\bin"

# Per includere entrambi i backend, servono entrambi gli SDK:
# .\scripts\build-windows-gpu.ps1 -Backend "CUDA,Vulkan" -BuildDirectory build-windows-multi -Destination "$HOME\sd-install\bin"
```

Lo script configura una build Ninja Release separata con backend dinamici e
varianti CPU, compila con parallelismo limitato, verifica `--list-devices` e
copia l'intera directory dei binari solo se rileva un acceleratore. Verifica
anche la copia temporanea e conserva la vecchia installazione in una cartella
di backup, evitando di mescolare DLL di revisioni diverse. Senza `Destination`
compila e verifica soltanto. Non scarica gli SDK, non modifica le protezioni
di Windows e non include pesi di modelli.

La verifica deve mostrare almeno un device di tipo `gpu`, `igpu` o `accel`,
non soltanto `CPU`. Ripetere `--list-devices` nella cartella effettivamente
usata dall'applicazione; un'etichetta di log come "GPU" non prova che quel
backend venga eseguito. Per forzare un device, usare il nome restituito
dall'inventario con `--backend`; aggiungere `--disable-backend-fallback` se
si preferisce un errore esplicito al ripiego sulla CPU.

CPU, RAM, GPU/VRAM e disco hanno ruoli diversi. Il disco conserva i pesi;
`--params-backend cpu` li colloca in RAM e `--backend` sceglie dove calcolare.
L'auto-fit può scegliere collocazioni diverse per modulo; forzare un
`--params-backend` disattiva l'auto-fit. Non è garantito che tutte le unità
lavorino contemporaneamente o al 90%: trasferimenti, modello e memoria
disponibile determinano il piano di esecuzione. Prima provare una risoluzione
contenuta e verificare nei log i backend effettivamente scelti.

### 2.7 Riepilogo: da zero al primo disegno (Windows)

Checklist completa per chi parte da un computer vuoto. Ogni passo rimanda
alla sezione con i dettagli.

1. **Strumenti di build** (manuale, una tantum): Visual Studio Build Tools
   con C++, CMake, Ninja, Git (sezione 2.1).
2. **SDK GPU** (manuale, una tantum, facoltativo ma consigliato): Vulkan SDK
   e/o CUDA Toolkit con i comandi `winget` della sezione 2.6; poi aprire un
   terminale nuovo. Senza SDK si compila comunque il motore solo-CPU.
3. **Codice**: clonare il repository e i sottomoduli (sezione 2.2).
4. **Build**: `.\scripts\build-engine.ps1` e scegliere la voce dal menu
   (sezione 2.6). Con `-Destinazione` il motore verificato viene installato
   con backup di quello precedente.
5. **Verifica**: l'output di `--list-devices` deve elencare la GPU attesa
   (`gpu`/`igpu`/`accel`), non soltanto `CPU`. Lo script GPU si rifiuta di
   installare un motore che non supera questa verifica.
6. **Modelli**: scaricare un file `.safetensors`/`.gguf` da una fonte
   affidabile (sezione 2.4); i pesi non sono mai inclusi.
7. **Primo disegno**: sezione 2.4 (CLI) o 2.5 (server HTTP). Partire con una
   risoluzione contenuta e verificare nei log i backend scelti.

Che cosa è automatico e che cosa no:

| Operazione | Chi la fa |
|---|---|
| Installazione di Build Tools, CMake, Ninja, SDK GPU, driver | **l'utente, a mano** (una tantum) |
| Download di SDK o modifiche alle protezioni di Windows | nessuno: gli script non lo fanno mai |
| Scelta dei backend in build, verifica dispositivi, backup e sostituzione del motore | gli script, dopo conferma |
| Collocazione dei pesi tra GPU/RAM/disco a runtime | il motore (auto-fit), salvo scelte esplicite |
| Riduzione della residenza pesi sotto pressione di memoria | il motore (`--memory-guard`) |
| Pesi lasciati su disco quando la RAM libera non copre il modello | l'applicazione ospite che integra questa logica (es. auto-calibrazione in Aletheia), mai contro una scelta esplicita |

### 2.8 Problemi comuni e come leggerli

| Sintomo | Causa tipica | Che cosa fare |
|---|---|---|
| Su Windows `sd-cli`/`sd-server` si chiude subito **senza scrivere nulla** (`--list-devices` non stampa niente), oppure l'applicazione che lo avvia riporta il codice `3221225781` | manca il Visual C++ Redistributable 2015-2022 (x64) oppure è più vecchio della **14.44** con cui sono compilati i programmi: Windows non carica le DLL e termina il processo prima che scriva qualcosa. Di norma non lascia nessun evento «Application Error» | in PowerShell `$LASTEXITCODE` dà `-1073741515` (`0xC0000135`, vedi i codici qui sotto). Controllare la versione con `(Get-Item C:\Windows\System32\msvcp140.dll).VersionInfo.FileVersion` (serve 14.44 o più), installare l'ultima da <https://aka.ms/vc14/vc_redist.x64.exe> e riprovare. Verificare anche che tutti i file dello zip stiano nella stessa cartella di `sd-cli` (per CUDA anche le DLL `cudart`/`cublas`) |
| `--list-devices` mostra solo `CPU` | build senza backend GPU, SDK assente in build, driver mancante, o DLL dei backend non accanto all'eseguibile | leggere il log (stderr): una riga `[W] CUDA backend: ggml-cuda.dll found in ... was not usable: <motivo>` dice perché il modulo GPU presente non si è caricato (su Windows elenca anche le DLL mancanti, per esempio `cublas64_12.dll`); `[I] no GPU backend module ... found in: <cartelle>` significa che nessun modulo GPU è accanto all'eseguibile (pacchetto o build senza GPU). Poi: driver aggiornato, pacchetto giusto (`cuda12` per NVIDIA), cartella davvero usata dall'applicazione |
| `auto-fit: no GPU devices` nei log | come sopra: il motore non vede acceleratori | stessa verifica; un'etichetta "GPU" nei log dell'applicazione non prova nulla |
| HTTP **502** da un proxy durante una generazione lunga | il proxy ha un timeout (es. la `fetch` di Node abbandona dopo ~5 minuti senza risposta) mentre il motore sta ancora calcolando | non reinviare alla cieca; usare API a job asincroni con polling; controllare i log del servizio |
| Il motore si chiude durante il caricamento pesi con `read tensor data failed` | pressione di memoria estrema durante letture grandi, oppure file del modello incompleto | liberare RAM o usare `--params-backend disk`; verificare l'integrità del file (dimensione attesa vs reale). Prima di arrendersi il motore riapre il file e ripete la lettura una volta: se compare anche solo `read from '...' failed; reopening the file and trying again` il problema è stato assorbito, se subito dopo c'è `read tensor data failed` no |
| `open '...' failed (attempt N/5): errno=..., doserrno=...` | il sistema ha rifiutato l'apertura del file del modello (su Windows: blocco temporaneo di un antivirus o di un altro programma, risorse di sistema scarse) | il motore riprova da solo fino a 5 volte; se tutti i tentativi falliscono, inviare le righe `errno`/`doserrno`: indicano la causa reale |
| Windows impedisce l'avvio dell'eseguibile (EACCES/EPERM) | policy di sicurezza, es. Smart App Control, o binario bloccato | controllare Sicurezza di Windows e i registri CodeIntegrity; usare binari attendibili; **non** disattivare le protezioni |
| Generazione lentissima (minuti per passo) | esecuzione sulla CPU, risoluzione alta, o modello troppo grande per la memoria | verificare i device; ridurre risoluzione/passi; modello quantizzato; build GPU |
| Avviso "memory guard: pressure" nei log | la memoria del device (su CPU: la RAM di sistema) ha superato la soglia `--memory-guard` | è un avviso con riduzione automatica della residenza: evitare nuove richieste finché non segnala il recupero |
| `auto-fit: CPU only; ...` nei log | senza GPU il motore ha confrontato la RAM disponibile con le dimensioni dei componenti | `the weights fit in RAM`: nessuna azione. `some weights do not fit in RAM; enabling the memory guard at 90%`: i componenti che non ci stanno restano su disco e il grafo gira a segmenti; è più lento ma serve proprio a non esaurire la RAM |

Codici con cui Windows chiude un programma che non riesce ad avviarsi.
PowerShell li mostra con il segno, molte applicazioni senza: è lo stesso
numero.

| Codice | Esadecimale | Significato | Che cosa fare |
|---|---|---|---|
| `-1073741515` / `3221225781` | `0xC0000135` | DLL non trovata | installare o aggiornare il Visual C++ Redistributable (14.44 o più); tenere tutti i file dello zip nella stessa cartella |
| `-1073741511` / `3221225785` | `0xC0000139` | una DLL non ha una funzione richiesta | file di versioni diverse nella stessa cartella oppure runtime Visual C++ troppo vecchio: ricopiare tutti i file dallo stesso zip e aggiornare il redistributable |
| `-1073741701` / `3221225595` | `0xC000007B` | immagine non valida (file a 32 e a 64 bit mischiati) | usare soltanto i file x64 dello stesso zip |
| `-1073741819` / `3221225477` | `0xC0000005` | violazione di accesso: il programma è andato in crash | leggere l'evento «Application Error» più recente (indica il modulo in errore); aggiornare driver video e Visual C++; provare il pacchetto Vulkan o solo CPU |
| `-1073740760` / `3221226536` | `0xC0000428` | Windows ha rifiutato un file perché la firma non è accettata | policy di sicurezza (es. Smart App Control): controllare i registri CodeIntegrity; **non** disattivare le protezioni |

### 2.9 Pacchetto portabile (consigliato per la distribuzione)
```shell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DSD_BUILD_SHARED_GGML_LIB=ON -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON
# Per una distribuzione x86 che includa varianti CPU selezionabili a runtime:
# aggiungere -DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF.
cmake --build build --config Release --parallel
```

Su Windows il pacchetto richiede una build a configurazione singola per
produrre i file nel percorso atteso dallo script. Da un Developer
PowerShell/Prompt con Ninja installato, usare una directory di build nuova:

```powershell
cmake -S . -B build-portable -G Ninja -DCMAKE_BUILD_TYPE=Release `
    -DSD_BUILD_SHARED_GGML_LIB=ON -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON
cmake --build build-portable --parallel
```

Distribuire insieme all'eseguibile i moduli backend dinamici ggml generati
(per esempio `.dll`, `.so` o `.dylib`, a seconda della piattaforma): i backend
non utilizzabili sulla macchina vengono semplicemente ignorati. L'opzione
`SD_BUILD_SHARED_GGML_LIB=ON` è necessaria: senza di essa il CMake del
progetto forza ggml statico.

Per creare lo zip distribuibile (binari + moduli backend + licenze +
LEGGIMI) da una build portabile:

```shell
./scripts/package_portable.sh build    # Linux/macOS/Git Bash; risultato in dist/
```

Lo script di packaging è Bash; su Windows avviarlo da Git Bash o da WSL dalla
radice del repository, passando la directory di build usata sopra:

```bash
./scripts/package_portable.sh build-portable
```

Il workflow usa Ninja proprio per mantenere gli eseguibili in `build/bin`;
con il generatore Visual Studio finiscono invece in `build/bin/Release` e lo
script non li trova. Il pacchetto è rilocabile all'interno della stessa
piattaforma e architettura: si scompatta e si usa, senza installazione.
Il workflow GitHub Actions `release-portable.yml` produce automaticamente gli
zip (con varianti CPU SSE/AVX/AVX2/AVX-512 su x86 per le macchine più vecchie)
a ogni tag `portable-v*` o avvio manuale. I pacchetti di release non sono un
singolo binario multipiattaforma: scegliere quello corrispondente al sistema
operativo e all'architettura del computer di destinazione.

#### Pacchetti con GPU già pronti

| Pacchetto | Backend GPU | Quando sceglierlo |
|---|---|---|
| `windows-x86_64-cuda12` | CUDA 12 | Windows con scheda **NVIDIA**: è il più veloce su NVIDIA. Le DLL di CUDA (`cudart`, `cublas`, `cublasLt`) sono già nello zip: serve solo un driver NVIDIA aggiornato |
| `windows-x86_64` | Vulkan | Windows con GPU AMD, Intel o NVIDIA; Vulkan arriva con il driver grafico. Senza driver Vulkan il programma usa la CPU |
| `linux-x86_64` | Vulkan | Linux con driver grafico che fornisce `libvulkan.so.1` |
| `darwin-arm64` | Metal | Mac con Apple Silicon |

Ogni pacchetto contiene anche il backend CPU: se il modulo GPU non si carica
(driver assente o troppo vecchio) il programma parte comunque sulla CPU, e il
log spiega perché il modulo GPU è stato scartato. Non servono SDK o compilatori
sulla macchina di destinazione.

**Requisito su Windows: Microsoft Visual C++ Redistributable 2015-2022 (x64),
versione 14.44 o successiva.** I programmi dei pacchetti sono compilati con
MSVC 14.44 e Microsoft richiede un redistributable uguale o più recente degli
strumenti di build usati ([dettagli](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist)).
Se manca o è più vecchio, Windows non riesce a caricare le DLL del motore e lo
chiude subito, **senza scrivere nulla**: `sd-cli --list-devices` non stampa
niente e un'applicazione che avvia il motore riceve il codice di uscita
`3221225781` (`0xC0000135`, mostrato da PowerShell come `-1073741515`). Per
controllare la versione installata:

```powershell
(Get-Item C:\Windows\System32\msvcp140.dll).VersionInfo.FileVersion   # deve essere 14.44 o più
```

L'ultima versione si scarica da <https://aka.ms/vc14/vc_redist.x64.exe>. Dopo
l'installazione riavviare l'applicazione che usa il motore: alcune ricordano
l'esito del primo controllo. Se si ricompila il motore con un Visual Studio più
recente, vale la versione di quel compilatore. Sintomi e altri codici di
uscita: sezione 2.8.

Per ottenerli senza tag: scheda **Actions** del repository, workflow
"Release portable package", **Run workflow**, scegliendo il ramo. Gli ingressi
sono `targets` (`all` oppure l'elenco dei nomi sopra, separati da virgola) e
`cuda_architectures` (predefinito: tutte le generazioni NVIDIA da Pascal a
Blackwell; per una scheda sola, per esempio `86` per le GeForce RTX serie 30,
la compilazione CUDA è molto più breve). Gli zip compaiono tra gli artefatti
dell'esecuzione (`portable-<nome>`) e, con un tag, nella release. Lo script di
packaging rifiuta di creare lo zip se il modulo GPU (o le DLL di CUDA) mancano,
così un pacchetto "con GPU" non può in silenzio funzionare solo sulla CPU.

Verifica dopo lo scompattamento: `sd-cli --list-devices` deve elencare, oltre
alla CPU, una riga di tipo `gpu` (o `igpu`).

#### Release `master-*`: zip Windows con CUDA e Vulkan insieme

Oltre ai pacchetti portabili, ogni release `master-<n>-<commit>` (workflow
`build.yml`) allega gli zip Windows `sd-master-<commit>-bin-win-cuda12-x64.zip`,
`...-bin-win-vulkan-x64.zip`, `...-bin-win-cpu-x64.zip` e
`cudart-sd-bin-win-cu12-x64.zip` (le DLL di CUDA). Per usare insieme la scheda
NVIDIA (CUDA) e le altre GPU del computer (Vulkan: AMD, anche la grafica
integrata, e Intel) c'è anche `sd-master-<commit>-bin-win-cuda12-vulkan-x64.zip`:
una cartella sola con i file dello zip CUDA, le DLL di CUDA, `ggml-vulkan.dll` e
un `LEGGIMI.txt`. Si scompatta e si usa, senza altri passaggi.

Non basta scompattare lo zip Vulkan sopra quello CUDA: quasi tutti i file hanno
lo stesso nome, ma `stable-diffusion.dll` è compilata in modo diverso (solo
quella dello zip CUDA contiene il codice per CUDA), e ne uscirebbe una build
mista. Lo zip unico lo assembla `scripts/package_windows_cuda_vulkan.sh`, un
passo del job `release`: dallo zip Vulkan prende solo `ggml-vulkan.dll`, e si
ferma se manca un file, se i due zip non sono dello stesso commit, se due zip
hanno un file con lo stesso nome o se il numero dei file non torna. Avvisa se
`ggml-base.dll` o `ggml.dll` hanno una dimensione diversa nei due zip, segno di
due build di ggml diverse. `test/package_windows_cuda_vulkan_test.sh` prova lo
script con zip finti, senza GPU né modelli. Come leggere l'elenco dei
dispositivi con i due backend (la stessa scheda come `CUDA0` e `Vulkan0`, la
GPU integrata): [convivenza_applicazioni.md](convivenza_applicazioni.md); cosa
dice il log se si fissa `--backend vulkan0` sulla scheda NVIDIA, i tempi e cosa
fare: [cuda_e_vulkan_stessa_scheda.md](cuda_e_vulkan_stessa_scheda.md).

## 3. Come collegarlo a un progetto

### 3.1 Libreria C/C++ con CMake

Questo esempio usa la libreria installata al punto 2.3. Sostituire il prefisso
con il percorso effettivamente scelto sulla propria macchina.

```cmake
cmake_minimum_required(VERSION 3.14)
project(mia_app C CXX)

find_package(stable-diffusion REQUIRED)
add_executable(mia_app main.c)
target_link_libraries(mia_app PRIVATE stable-diffusion)
set_target_properties(mia_app PROPERTIES LINKER_LANGUAGE CXX)
```

Configurare passando `CMAKE_PREFIX_PATH` alla directory di installazione:
`-DCMAKE_PREFIX_PATH=/percorso/di/installazione` su Linux/macOS oppure, in
PowerShell, `-DCMAKE_PREFIX_PATH="$HOME\sd-install"` se si è usato il prefisso
Windows dell'esempio. La libreria installata deve essere stata compilata per
la stessa piattaforma, architettura e configurazione ABI dell'applicazione.
Dopo aver aggiornato il fork, ricompilare l'applicazione contro l'header e la
libreria corrispondenti; non mescolare header di una revisione con librerie
precompilate di un'altra.

### 3.2 Sequenza di inizializzazione consigliata

Il ciclo di vita completo, nell'ordine tecnico corretto:

1. **Logging** (prima di tutto): `sd_set_log_callback(cb, user_data)` — senza
   callback la libreria non stampa nulla; ogni messaggio arriva con livello
   (`SD_LOG_DEBUG/INFO/WARN/ERROR`), testo e il puntatore `user_data`.
2. **Rilevamento hardware** (facoltativo ma consigliato):
   `sd_get_device_count()` + `sd_get_device_info(i, &info)` — non caricano
   nulla, si possono chiamare subito e da più thread. Alla prima chiamata
   vengono scoperti i moduli backend dinamici ggml accanto all'eseguibile.
3. **Parametri del contesto**: `sd_ctx_params_init(&cp)` — azzera e imposta i
   default; poi valorizzare solo i campi necessari (tabella sotto).
4. **Creazione del contesto**: `new_sd_ctx(&cp)` — carica il modello, sceglie
   e inizializza i backend (auto-fit + fallback CPU); è l'operazione costosa
   (secondi–minuti). Restituisce `NULL` in caso di errore (motivo con
   `sd_get_last_error()` e nei log).
5. **Avanzamento** (facoltativo): `sd_set_progress_callback(cb, data)` riceve
   `(step, steps, time, data)` a ogni passo di campionamento — è ciò che serve
   per una progress bar.
6. **Parametri di generazione**: `sd_img_gen_params_init(&gp)`; poi prompt e
   dimensioni. Default reali: 512×512, `strength` 0.75, `seed` -1 (= casuale),
   `batch_count` 1, 20 passi.
7. **Generazione**: `generate_image(ctx, &gp, &result)` con
   `sd_image_generation_result_t* result = NULL` — bloccante; chiamarla da un
   worker thread se l'app ha una UI. `result_out` è obbligatorio e in caso di
   fallimento la libreria lo azzera.
8. **Annullamento** (facoltativo, da un altro thread):
   `sd_cancel_generation(ctx, SD_CANCEL_ALL)` interrompe appena possibile;
   `SD_CANCEL_NEW_LATENTS` finisce l'immagine in corso e salta le successive.
9. **Lettura del risultato**: `result->count` elementi in `result->items[i]`;
   ognuno ha `image` (un `sd_image_t` con `width`, `height`, `channel` (3 = RGB)
   e `data` — buffer di `width*height*channel` byte, righe contigue senza
   padding, ordine RGB) e `generation_info` (JSON `sdcpp.generation/v1` con i
   parametri per riprodurre l'immagine, vedi
   [generation_metadata.md](generation_metadata.md)). Copiarli o convertirli
   subito.
10. **Rilascio**: `free_sd_image_generation_result(result)` e poi
    `free_sd_ctx(ctx)` — sempre le funzioni della libreria, mai `free()` del
    chiamante (su Windows CRT diversi causerebbero corruzione). Il risultato
    possiede pixel e JSON, resta valido anche dopo `free_sd_ctx()` e il suo
    distruttore accetta `NULL`.

Campi principali di `sd_ctx_params_t` (tutti gli altri possono restare ai
default):

| Campo | Significato | Default |
|---|---|---|
| `model_path` | checkpoint completo (`.safetensors`/`.gguf`/`.ckpt`) | obbligatorio* |
| `diffusion_model_path` | in alternativa: solo modello di diffusione (componenti separati) | — |
| `n_threads` | thread CPU; `<= 0` = core fisici | auto |
| `backend` | assegnazione backend, es. `"cpu"`, `"vulkan0"`, `"diffusion=cuda0,te=cpu"`; vuoto = migliore disponibile | auto |
| `params_backend` | dove risiedono i pesi: `"cpu"` (RAM), `"disk"`, per modulo | auto-fit |
| `max_vram` | budget per device in GiB, es. `"6"` o `"cuda0=6,vulkan0=4"`; `-1` = riserva 1 GiB | memoria libera |
| `memory_guard` | soglia % del guardiano di memoria (50–99): memoria del device, su CPU la RAM di sistema; `0` = spento | `0` |
| `disable_backend_fallback` | `true` = fallire invece di ripiegare sulla CPU | `false` |
| `enable_mmap` | mappa i pesi dal file invece di copiarli in RAM | `false` |
| `wtype` | forza il tipo dei pesi (quantizzazione a caricamento) | tipo del file |

*uno tra `model_path` e `diffusion_model_path` è obbligatorio.

Esempio compilato e collaudato con il percorso `find_package` qui sopra:

```c
#include <stdio.h>
#include "stable-diffusion.h"

int main(int argc, char** argv) {
    // 1. Rilevamento hardware (nessun modello caricato)
    for (size_t i = 0; i < sd_get_device_count(); i++) {
        sd_device_info_t d;
        if (sd_get_device_info(i, &d)) {
            printf("device %zu: %s [%s] %s\n", i, d.name, sd_device_type_name(d.type), d.description);
        }
    }

    // 2. Contesto con memory guard; il fallback CPU è già attivo
    sd_ctx_params_t cp;
    sd_ctx_params_init(&cp);          // inizializza SEMPRE prima di impostare i campi
    cp.model_path   = argv[1];
    cp.memory_guard = 90;

    sd_ctx_t* ctx = new_sd_ctx(&cp);
    if (ctx == NULL) {                // modello non valido o backend non disponibile
        char reason[512];
        enum sd_error_code_t code = sd_get_last_error(reason, sizeof(reason));
        fprintf(stderr, "new_sd_ctx: %s: %s\n", sd_error_code_name(code), reason);
        return 1;
    }

    // 3. Generazione
    sd_img_gen_params_t gp;
    sd_img_gen_params_init(&gp);
    gp.prompt = "un gatto astronauta";
    gp.width  = 512;
    gp.height = 512;
    gp.seed   = 42;

    sd_image_generation_result_t* result = NULL;
    int rc                               = 1;
    if (generate_image(ctx, &gp, &result) && result->count > 0) {
        rc = 0;                       // result->items[i].image: data / width / height / channel
    }

    free_sd_image_generation_result(result);  // sempre dalla libreria (evita problemi di CRT su Windows)
    free_sd_ctx(ctx);
    return rc;
}
```

Regole importanti per chi integra:

- Chiamare sempre `sd_ctx_params_init()` e `sd_img_gen_params_init()` prima di
  valorizzare i campi: senza init i campi contengono valori casuali.
- Controllare sempre il valore di ritorno di `new_sd_ctx()` e `generate_image()`.
- **Errori**: dopo un fallimento `sd_get_last_error(buffer, dimensione)`
  restituisce il codice (`SD_ERROR_OUT_OF_MEMORY`, `SD_ERROR_MODEL_FILE`,
  `SD_ERROR_BACKEND`, `SD_ERROR_INVALID_ARGUMENT`, `SD_ERROR_BUSY`,
  `SD_ERROR_CANCELLED`, `SD_ERROR_UNKNOWN`) e, se il buffer non è nullo, il
  testo del primo errore registrato; `sd_error_code_name()` ne dà il nome. Vale per il thread che ha
  fatto la chiamata, si azzera all'inizio di ogni chiamata e al successo. Con
  `SD_ERROR_OUT_OF_MEMORY` ha senso ritentare con meno memoria (profilo più
  leggero, pesi su disco) o più tardi; il contesto resta utilizzabile dopo il
  fallimento. Le eccezioni C++ non escono mai dall'API.
- **Memoria di sistema**: `sd_get_system_memory(&m)` riporta RAM totale,
  disponibile e occupata dal processo, utile per decidere prima di caricare un
  modello o per un indicatore nell'interfaccia.
- **Threading**: un contesto non è thread-safe; serializzare gli accessi (come
  fa `sd-server` con un mutex) oppure usare un contesto per thread.
  `sd_cancel_generation()` è pensata per essere chiamata da un altro thread.
  Le funzioni di enumerazione dei device sono state provate da più thread in
  parallelo. I callback (log, progress) arrivano dal thread che sta generando.
- **Durata delle stringhe**: i campi `const char*` dei parametri non vengono
  copiati al momento dell'assegnazione; devono restare validi fino alla
  chiamata (`new_sd_ctx`/`generate_image`) che li consuma. Da linguaggi con
  garbage collector (FFI), ancorare i buffer finché la chiamata non ritorna.
- **Un contesto, molte generazioni**: il modello resta caricato; chiamare
  `generate_image()` più volte sullo stesso contesto è il modo corretto di
  servire più richieste (ricrearlo a ogni richiesta ricarica il modello).
- **Processo e working directory**: con i backend dinamici i moduli ggml
  vengono cercati accanto all'eseguibile, non nella working directory.

### 3.3 Altri linguaggi (TypeScript/Electron, Python, Rust, Go, C#, ...)

L'API è C pura (`include/stable-diffusion.h`), quindi si può usare tramite FFI
o binding. Binding esistenti sono elencati nel [README](../README.md).

Indicazioni tecniche per un binding FFI fatto in casa:

- servono la libreria **dinamica** (`-DSD_BUILD_SHARED_LIBS=ON`) oppure il
  binding N-API/ctypes contro la statica ricompilata PIC;
- dichiarare le strutture esattamente nell'ordine dei campi dell'header (sono
  struct C semplici); dopo ogni aggiornamento del fork rigenerare le
  dichiarazioni, perché i campi possono crescere in coda;
- chiamare sempre `sd_ctx_params_init()` via FFI invece di costruire la struct
  a mano: imposta i default corretti anche per i campi futuri;
- i callback FFI (log/progress) arrivano da thread nativi: nel runtime del
  linguaggio usare il meccanismo apposito (es. `ThreadSafeFunction` in N-API).

Per un'app Electron/Node la via più robusta resta il **processo separato**
(3.4/3.5): isolamento totale dai crash e nessun binding da mantenere.

### 3.4 Come processo: `sd-cli`

Isola il motore dall'applicazione: un crash del motore non abbatte l'app.

- avviare `sd-cli` con gli argomenti della generazione e leggere l'exit code:
  `0` = successo, valore diverso da zero = errore (dettagli su stderr); il
  modo in cui il sistema operativo segnala un crash non è portabile;
- `-o out_%d.png` scrive i file di output; l'app li legge a fine processo;
- `--list-devices` dà l'inventario hardware in formato tabellare
  (`nome<TAB>tipo<TAB>descrizione<TAB>memoria`), facile da parsare;
- opzioni chiave: `--memory-guard 90`, `--backend`, `--max-vram`,
  `--offload-to-cpu`, `--disable-backend-fallback`;
- per annullare: terminare il processo con il meccanismo previsto dal sistema
  operativo e attendere che termini; le modalità di terminazione non sono
  uniformi tra POSIX e Windows.

### 3.5 Come servizio HTTP: `sd-server`

Offre API compatibili OpenAI, `sdapi` e `sdcpp`. L'autenticazione è
disattivata per default: **leggere la sezione 5 prima di esporlo** e usare
`--api-key` / `--cors-origins`.

Il server **parte anche senza modello** (niente loop né uscita silenziosa):

- senza `-m`/`--diffusion-model` si avvia con un avviso in console; gli
  endpoint dei device e di upscale funzionano, quelli di generazione
  rispondono `503` con `{"code": "model_not_loaded", "error": "No model is loaded"}`
  e `/sdcpp/v1/health` dice `"status": "no_model"` (il catalogo dei modelli è quello
  di leejet: `--model-dir` e le rotte `/sdcpp/v1/model-files`, `/sdcpp/v1/recipes` e
  `/sdcpp/v1/models/{status,load,unload,refresh}`, vedi
  [examples/server/api.md](../examples/server/api.md)); un modello caricato dal
  catalogo è gestito dallo stesso custode del contesto, quindi vale anche per lui il
  recupero dalla memoria esaurita descritto sotto;
- se il modello è indicato ma non è caricabile, il server **non** parte e
  scrive in console il percorso del modello e il motivo del fallimento
  (exit code 1).

Il server **non cade quando la memoria finisce**:

- una richiesta che esaurisce la memoria libera il modello, lo ricarica con il
  profilo più leggero successivo
  (`max > balanced > lite > lite-disk > lite-hybrid > lite-cpu`; i passi che
  darebbero la stessa configurazione vengono saltati) e ripete **la stessa
  richiesta**. Fino a `lite-disk` il calcolo resta tutto sulle GPU (si lavora a
  pezzi più piccoli e i pesi si rileggono da RAM o disco: costa tempo ma non
  cede il lavoro alla CPU); la CPU prende l'encoder di testo da `lite-hybrid` e
  tutto solo a `lite-cpu`, l'ultima risorsa. Da `lite` in poi il VAE decodifica
  a tasselli qualunque cosa chieda la richiesta. Con `--eager-load` il passo
  `lite-disk` non rilegge tutti i pesi all'avvio;
- un **device che si guasta** (errore del driver, GPU reimpostata o rimossa,
  backend che non parte) non è un problema di memoria e non porta alla CPU: il
  server riprova lo stesso device una volta, poi prova uno per uno gli altri
  che il motore elenca (le schede grafiche prima degli acceleratori e delle GPU
  integrate; la scheda col backend del produttore prima della stessa scheda
  vista da Vulkan; a parità, quella con più memoria libera) e solo alla fine
  passa alla CPU (`lite-cpu`). Finché resta una GPU da provare dice anche al
  motore di non ripiegare da solo sulla CPU (`disable_backend_fallback`), cosa
  che nasconderebbe il guasto dietro una generazione lenta. Un guasto sulla CPU
  fa fallire la richiesta. Un `--backend` scritto a mano con moduli assegnati
  (`=`) o più device (`&`, `,`) non viene riscritto: una nuova prova e poi la
  CPU, come prima. Vale anche all'avvio: se il primo device non parte, il
  server parte sul successivo e il log dice quale;
- se nemmeno il profilo più leggero basta, la risposta è `503` con
  l'intestazione `Retry-After` e un corpo
  `{"error": "...", "code": "out_of_memory", "retry_after_seconds": 30}`; il
  server resta attivo e serve le richieste più piccole. Nei job asincroni lo
  stesso `code` compare in `error.code`. Una richiesta non valida è `400`
  (`invalid_argument`); un file del modello illeggibile o un device guasto sono
  `500` (`model_file`, `backend`);
- dopo `--degradation-cooldown` secondi (predefinito 300) il server risale di un
  passo, all'inizio della richiesta successiva, che paga il ricaricamento del
  modello: un profilo in su, oppure il ritorno al device di partenza se sta
  girando su un altro; una risalita che non funziona viene annullata e l'attesa
  raddoppia (fino a un'ora). Dalla CPU, dopo un guasto di device, la risalita
  torna direttamente al profilo che c'era quando il device si è guastato e offre
  di nuovo tutti i device, così si trova il migliore che funziona anche se quello
  preferito no. `--degradation-cooldown 0` lo lascia dov'è fino al riavvio;
  `--disable-degradation` spegne tutto (anche il divieto di ripiegare da soli
  sulla CPU: il motore si comporta come prima) e la richiesta fallisce subito;
- se il modello non si riesce a ricaricare, le richieste ricevono `503`
  (`unavailable`) e il server riprova a partire dalla richiesta successiva, non
  più spesso di ogni 15 secondi;
- se il modello non ci sta nemmeno all'avvio, il server parte con il primo
  profilo che lo carica (il log dice quale). Se non riesce ad aprire la porta, o
  a creare i thread del server HTTP, esce con codice 1 e un messaggio
  invece di terminare in silenzio;
- `GET /sdcpp/v1/health` riporta stato (`ok`, `degraded`, `recovering`,
  `unavailable`, `no_model`; `degraded` anche su un device diverso dal primo),
  profilo e device in uso (`device`: `auto` se sceglie il motore), i moduli che
  la CPU calcola accanto alle GPU (`cpu_modules`: `text_encoder`, `vae`,
  `controlnet`, oppure `["all"]` a `lite-cpu`; vuoto se fanno tutto le GPU), i
  device di riserva nell'ordine in cui verrebbero provati
  (`fallback_devices`), contatori (richieste, riuscite, fallite, annullate,
  discese di profilo, nuove prove sullo stesso device `retries`, cambi di device
  `device_switches`, risalite, ricaricamenti), ultimo errore e coda dei job, e
  risponde subito anche durante una generazione; all'avvio il log scrive
  l'ordine dei device di riserva, il device in uso e, se c'è, quali moduli
  calcola la CPU;
  `GET /sdcpp/v1/resources` riporta RAM del sistema e memoria dei device.
  Entrambi richiedono la chiave API se impostata.

Il server **sa fermare un job che sta generando** (non solo uno in coda):

```shell
# crea un job; la risposta contiene "id" e "poll_url"
curl -s -X POST http://127.0.0.1:1234/sdcpp/v1/img_gen \
     -H "Content-Type: application/json" \
     -d '{"prompt": "a red apple", "width": 512, "height": 512, "sample_params": {"sample_steps": 30}}'

# fermalo: la richiesta torna subito e il job può risultare ancora "generating"
curl -s -X POST http://127.0.0.1:1234/sdcpp/v1/jobs/job_6acf9114_00000000/cancel

# oppure aspetta fino a 10 secondi che il job finisca davvero
curl -s -X POST "http://127.0.0.1:1234/sdcpp/v1/jobs/job_6acf9114_00000000/cancel?wait=10"

# in ogni caso, controlla con calma lo stato del job
curl -s http://127.0.0.1:1234/sdcpp/v1/jobs/job_6acf9114_00000000
```

- Cosa dice la risposta. `200` = il job è in uno stato finale e la risposta lo
  contiene (`"status": "cancelled"`, `error.code` `cancelled`); senza `?wait=` è
  anche la risposta immediata a «stop chiesto», con il job ancora
  `"status": "generating"` e `"cancel_requested": true`: interroga
  `GET /sdcpp/v1/jobs/{id}` finché diventa `cancelled`. `202` = con `?wait=`, lo
  stop è stato chiesto ma dopo l'attesa il job sta ancora finendo
  (`"status": "generating"` e `"cancel_requested": true`): richiedi lo stesso
  `cancel` o interroga il job. Ripetere la richiesta non fa danni. Un job in coda
  esce subito dalla coda (`200`) e non parte mai; un job già finito resta com'è
  (`200`); un id sconosciuto è `404`.
- `?wait=` (facoltativo) sono i secondi che la richiesta può aspettare che il job
  finisca davvero: da `0` a `60`. Senza `?wait=` la richiesta risponde subito,
  come il server di leejet. `?wait=0` risponde subito anch'esso, ma con `202` se
  il job non era già finito. Un valore che non è un numero intero di secondi (0 o
  più) dà `400`. Con `?wait=` la richiesta tiene aperta la connessione per tutto
  il tempo che aspetta e ripete lo stop ogni 100 ms, perché la libreria dimentica
  uno stop che arriva prima dell'inizio della generazione; senza `?wait=` lo stop
  parte una sola volta, quindi se il job non si ferma basta ripetere la richiesta.
- Quanto ci mette. Il motore controlla lo stop alla fine di ogni passo di
  campionamento e fra le fasi (testo, decodifica, secondo passaggio): su una macchina
  lenta un passo dura secondi, e la prima generazione dopo l'avvio può passare
  molto tempo a leggere i pesi dal disco prima del primo controllo. Con `?wait=`,
  se lo stop non basta nel tempo di `wait` la risposta è `202`, non un errore.
- Se il job finisce prima che lo stop faccia effetto, la risposta è il job
  completato (`200`, `"status": "completed"`): non c'è niente da fermare.
- Fermare non è un errore. Il modello non si ricarica, il server non scende a un
  profilo più leggero, `last_error` non cambia e `/health` conta il job in
  `counters.cancelled`. Il job successivo parte subito sullo stesso contesto. Un
  job fermato non restituisce immagini, nemmeno quelle già pronte di un gruppo.
- Le rotte sincrone (`/sdapi/...`, `/v1/images/...`) non sono job e non si
  possono fermare: se una di esse aspetta il modello dietro a un job che viene
  annullato, parte subito dopo e finisce normalmente.
- Per sapere se un server lo supporta: `features.cancel_generating` in
  `GET /sdcpp/v1/capabilities`. Un server più vecchio risponde `409` a un job già
  partito.

Dettagli e formato delle risposte: [examples/server/api.md](../examples/server/api.md).

### 3.6 Come strumento di un assistente AI (chat, agente che controlla il browser)

Il motore genera immagini e video: non conversa. In una chat o in un agente che
usa il browser è un **modello di linguaggio** a decidere quando serve
un'immagine e a chiamare il motore come *strumento* (function calling, "tool
use"); l'applicazione ospite esegue la chiamata e restituisce il risultato al
modello. Il motore sta in fondo alla catena e non sa nulla della conversazione:

```
utente ─► assistente (LLM) ─► applicazione ospite ─HTTP─► sd-server ─► CPU / GPU
              ▲                        │
              └── percorso del file ◄──┘
```

Il server ha già quello che serve (sezione 3.5): la rotta nativa asincrona
`/sdcpp/v1/img_gen`, `capabilities` con i default e i limiti del modello
caricato, `health` e `resources` per sapere se si può generare e con quale
hardware. Qui sotto: come avviarlo secondo l'hardware, i preset, gli strumenti
da dichiarare all'assistente con un'implementazione di riferimento, e le regole
di sicurezza.

**Stato delle prove.** Lo strumento e il flusso HTTP sono stati eseguiti contro
un `sd-server` reale su CPU (2 core) con il modello sintetico da 2,1 GB: riuscita,
parametri non validi, chiave errata, server spento, scadenza con job in coda e
con job già in corso, scelta fra due server. Le ricette per una o più GPU
seguono la documentazione e i test del pianificatore ma **non sono state
provate su GPU reali**, e i preset sono punti di partenza da tarare sul proprio
modello: nessuna immagine di un modello vero è stata valutata.

#### Nessuna, una o più GPU

Il motore rileva l'hardware da solo: `sd-cli --list-devices` lo elenca prima di
caricare un modello, `GET /sdcpp/v1/resources` lo ripete a server acceso e
un'applicazione che usa la libreria ha `sd_get_device_count()` /
`sd_get_device_info()` (sezione 4). Tipo (`cpu`, `gpu`, `igpu`, `accel`) e
memoria libera bastano per decidere come avviare:

| Macchina | Avvio | Cosa succede |
|---|---|---|
| Nessuna GPU | `sd-server -m modello --profile auto` | `--profile auto` confronta modello e RAM disponibile; `--auto-fit` tiene i pesi in RAM e, se non ci stanno, li legge dal disco (con la memory guard). È lento: per dare un ordine di grandezza, 256×256 e 2 passi con un SD 1.5 su 2 core hanno richiesto 30-40 s. Meglio `draft` e immagini piccole. |
| Una GPU | `sd-server -m modello --profile auto` | Il calcolo va sulla GPU; i pesi che non ci stanno restano in RAM o su disco. |
| Più GPU, il modello sta in una | un `sd-server` per GPU (sotto) | Più immagini nello stesso tempo. Ogni server tiene la sua copia del modello. |
| Più GPU, il modello non sta in una | `sd-server -m modello --profile auto` (`--multi-gpu auto` è il default) | Il motore divide i blocchi del modello fra le GPU dello stesso backend, oppure sposta l'encoder di testo o il VAE su un'altra GPU (vedi [backend.md](./backend.md#several-gpus---multi-gpu-autoonoff)). Una singola immagine non diventa più veloce. |

Un server per GPU, con la stessa chiave e porte diverse (i nomi `cuda0`,
`cuda1`, `vulkan0`... sono quelli di `--list-devices`; `-W`, `-H` e `--steps`
sono valori di esempio per un modello da 1024 px):

```shell
sd-server -m modello.safetensors --backend cuda0 --listen-port 1234 --api-key mia-chiave -W 1024 -H 1024 --steps 25
sd-server -m modello.safetensors --backend cuda1 --listen-port 1235 --api-key mia-chiave -W 1024 -H 1024 --steps 25
```

e in `SERVERS` dello strumento (sotto) i due indirizzi. La scelta del server è
quella con meno lavoro in coda (`queue.pending` di `GET /sdcpp/v1/health`); con
un solo indirizzo il comportamento è quello di un server singolo. Il pianificatore
di `--multi-gpu` è provato con GPU simulate, non con due GPU vere.

#### Preset

Un preset è un nome che l'assistente può scegliere senza conoscere il modello;
l'applicazione lo traduce in numeri leggendo `GET /sdcpp/v1/capabilities`
(`defaults_by_mode.img_gen` e `limits`), così vale per qualunque modello e
macchina:

- `draft`, `standard`, `detailed`: metà, uno e una volta e mezza i **passi**
  predefiniti. Non cambiano la dimensione, perché molti modelli rendono male
  lontano dalla risoluzione per cui sono stati addestrati;
- `square`, `portrait`, `landscape`: 1:1, 3:4 e 4:3, con il lato lungo uguale al
  lato predefinito, arrotondato a multipli di 64 e dentro i `limits`.

I default annunciati sono quelli con cui il server è stato avviato (`-W`, `-H`,
`--steps`; verificato) e altrimenti 512×512 e 20 passi **qualunque sia il
modello**: per un modello da 1024 px avviare con `-W 1024 -H 1024` e i passi
adatti, altrimenti `standard` darà 512×512. Un flusso comodo in chat: `draft`
con un seme noto per provare l'idea e, se piace, stesso prompt e stesso seme con
`standard` o `detailed` (di solito l'impostazione resta simile e cambiano i
dettagli; da verificare con il proprio modello).

#### Gli strumenti

Due strumenti bastano. Lo schema JSON va nel campo che l'API del modello di
linguaggio usato chiama `input_schema` o `parameters` (le descrizioni sono lette
dal modello: tradurle se serve):

```json
[
  {
    "name": "generate_image",
    "description": "Genera un'immagine dal testo con il motore locale. Può richiedere da alcuni secondi a diversi minuti (di più senza GPU). Per provare un'idea usare preset draft, poi rifare con lo stesso seed e preset standard o detailed. Restituisce il percorso del file PNG, il seed e le dimensioni.",
    "input_schema": {
      "type": "object",
      "properties": {
        "prompt": {"type": "string", "description": "Descrizione dell'immagine, nella lingua con cui il modello è stato addestrato (di solito l'inglese)."},
        "preset": {"type": "string", "enum": ["draft", "standard", "detailed"], "description": "draft = metà dei passi, detailed = una volta e mezza."},
        "shape": {"type": "string", "enum": ["square", "portrait", "landscape"]},
        "seed": {"type": "integer", "description": "Per ripetere o rifinire un'immagine già generata."}
      },
      "required": ["prompt"]
    }
  },
  {
    "name": "engine_status",
    "description": "Dice se il motore di immagini è pronto, quanto lavoro ha in coda e quali dispositivi (CPU, GPU) vede con la memoria libera.",
    "input_schema": {"type": "object", "properties": {}}
  }
]
```

Implementazione di riferimento in Python (solo libreria standard). Il risultato
torna al modello come percorso di file e non come base64, per non riempire il
suo contesto; l'assistente non "vede" l'immagine, a meno che l'applicazione la
alleghi al risultato e il modello accetti immagini. Questo codice blocca la
chiamata fino alla fine del job (al massimo `wait` secondi): in un'interfaccia
che non deve restare ferma conviene restituire subito l'`id` del job e
interrogarlo con una seconda chiamata.

```python
import base64
import json
import random
import time
import urllib.error
import urllib.request
from collections import Counter

SERVERS = ["http://127.0.0.1:1234"]  # un indirizzo per ogni sd-server (uno per GPU, se si vuole più produttività)
API_KEY = ""  # il valore di --api-key (uguale per tutti i server); vuoto se non c'è
OUT_DIR = "."  # dove salvare le immagini

PRESETS = {"draft": 0.5, "standard": 1.0, "detailed": 1.5}  # moltiplicano i passi predefiniti del modello
SHAPES = {"square": (1, 1), "portrait": (3, 4), "landscape": (4, 3)}  # larghezza:altezza


def call(base, method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(base + path, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    if API_KEY:
        request.add_header("Authorization", "Bearer " + API_KEY)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        try:
            return error.code, json.load(error)
        except ValueError:
            return error.code, {"error": str(error.reason)}


def health_of(base):
    try:
        status, health = call(base, "GET", "/sdcpp/v1/health")
    except OSError as error:
        return {"status": "down", "problem": str(error)}
    if status not in (200, 503):  # 503: il server è vivo ma senza un modello utilizzabile
        return {"status": "error", "problem": "HTTP %d %s" % (status, health.get("error", ""))}
    return health


def pick_server():
    """Il server pronto con meno lavoro in coda; None se nessuno lo è."""
    best = None
    for base in SERVERS:
        health = health_of(base)
        if health["status"] in ("ok", "degraded") and (best is None or health["queue"]["pending"] < best[0]):
            best = (health["queue"]["pending"], base)
    return best and best[1]


def engine_status():
    servers, devices = [], []
    for base in SERVERS:
        health = health_of(base)
        servers.append({"url": base, "status": health["status"], "problem": health.get("problem"),
                        "profile": health.get("profile"), "busy": health.get("busy"),
                        "waiting": health.get("queue", {}).get("waiting")})
        if not devices and health["status"] in ("ok", "degraded"):  # i device sono quelli della macchina
            _, resources = call(base, "GET", "/sdcpp/v1/resources")
            devices = [{"name": d["name"], "type": d["type"],
                        "free_gib": round(d["free_memory_bytes"] / 2**30, 1)}
                       for d in resources["devices"]]
    return {"servers": servers, "devices": devices,
            "count_by_type": dict(Counter(d["type"] for d in devices))}


def side_length(value, low, high):
    return max(low, min(high, round(value / 64) * 64))


def generate_image(prompt, preset="standard", shape="square", seed=None, wait=600):
    base = pick_server()
    if base is None:
        return {"error": {"code": "unavailable", "message": "nessun server pronto",
                          "servers": engine_status()["servers"]}}
    status, caps = call(base, "GET", "/sdcpp/v1/capabilities")
    if status != 200:
        return {"error": caps}
    defaults, limits = caps["defaults_by_mode"]["img_gen"], caps["limits"]
    side = max(defaults["width"], defaults["height"])
    w, h = SHAPES[shape]
    width = side_length(side * w / max(w, h), limits["min_width"], limits["max_width"])
    height = side_length(side * h / max(w, h), limits["min_height"], limits["max_height"])
    steps = max(1, round(defaults["sample_params"]["sample_steps"] * PRESETS[preset]))
    seed = random.randrange(2**31) if seed is None else seed
    body = {"prompt": prompt, "width": width, "height": height, "seed": seed,
            "sample_params": {"sample_steps": steps}, "embed_image_metadata": False}
    status, job = call(base, "POST", "/sdcpp/v1/img_gen", body)
    if status != 202:
        return {"error": job}
    poll_url, started = job["poll_url"], time.time()
    while time.time() - started < wait:
        time.sleep(1)
        status, job = call(base, "GET", poll_url)
        if status != 200:  # 404/410: il risultato è già stato scartato (dopo 10 minuti)
            return {"error": job}
        if job["status"] == "completed":
            path = "%s/image_%d_%d.png" % (OUT_DIR, started, seed)
            with open(path, "wb") as file:
                file.write(base64.b64decode(job["result"]["images"][0]["b64_json"]))
            return {"path": path, "width": width, "height": height, "steps": steps,
                    "seed": seed, "seconds": round(time.time() - started)}
        if job["status"] in ("failed", "cancelled"):
            return {"error": job["error"]}
    call(base, "POST", poll_url + "/cancel?wait=0")  # ferma il job, anche se sta già generando, senza attendere
    return {"error": {"code": "timeout", "message": "nessun risultato entro %d s" % wait}}


TOOLS = {
    "generate_image": (generate_image, {"prompt", "preset", "shape", "seed"}),
    "engine_status": (engine_status, set()),
}


def run_tool(name, arguments):
    """Esegue la chiamata decisa dal modello, passando solo gli argomenti previsti dallo schema."""
    if name not in TOOLS:
        return {"error": {"code": "unknown_tool", "message": name}}
    function, allowed = TOOLS[name]
    if not isinstance(arguments, dict):
        arguments = {}
    kwargs = {key: value for key, value in arguments.items() if key in allowed and value is not None}
    if name == "generate_image":
        if not isinstance(kwargs.get("prompt"), str) or not kwargs["prompt"].strip():
            return {"error": {"code": "invalid_argument", "message": "prompt mancante"}}
        if kwargs.get("preset", "standard") not in PRESETS or kwargs.get("shape", "square") not in SHAPES:
            return {"error": {"code": "invalid_argument",
                              "message": "preset: %s; shape: %s" % (list(PRESETS), list(SHAPES))}}
        if "seed" in kwargs and not isinstance(kwargs["seed"], int):
            return {"error": {"code": "invalid_argument", "message": "seed deve essere un intero"}}
    try:
        return function(**kwargs)
    except (OSError, KeyError, ValueError) as error:  # risposta inattesa del server
        return {"error": {"code": "tool_failed", "message": str(error)}}
```

Uso: `run_tool("generate_image", {"prompt": "a red apple", "preset": "draft"})`
restituisce `{"path": "./image_1791931200_42.png", "width": ..., "steps": ...,
"seed": ..., "seconds": ...}` oppure `{"error": {"code": ..., "message": ...}}`.
Gli errori del server arrivano con il loro `code` (`out_of_memory`, `invalid_argument`,
`unavailable`, ...); con la chiave sbagliata o il server spento il messaggio dice
quale dei due è il problema.

#### Regole di sicurezza per chi integra

1. **Rotta nativa per i testi che non sono dell'utente.** Sulle rotte OpenAI
   (`/v1/images/*`) e `sdapi` un blocco `<sd_cpp_extra_args>{...}</sd_cpp_extra_args>`
   dentro il prompt viene estratto e applicato come parametri della richiesta:
   provato sulla rotta OpenAI, un prompt con `{"width":64,"height":64}` ha
   trasformato una richiesta 128×128 in 64×64. Sulla rotta nativa `/sdcpp/v1/*` il blocco non viene
   interpretato (la stessa richiesta è rimasta 128×128). Se il prompt lo scrive
   un modello di linguaggio o lo si legge da una pagina web, usare la nativa.
2. **Applicare i limiti prima di inviare.** `limits` di `capabilities` è
   un'indicazione per il client, non un controllo del server. Per la generazione
   il server impone solo 8 immagini per richiesta e da 1 a 100 passi (più la
   coda, con `429`); larghezza e altezza no: una larghezza di 4097 è stata
   accettata e portata a 4160, e una enorme è finita in `out_of_memory` facendo
   scendere il server di un profilo (un ricaricamento, e prestazioni ridotte per
   `--degradation-cooldown` secondi). Un job già partito si può fermare
   (`POST /sdcpp/v1/jobs/{id}/cancel`, sezione 3.5), ma lo stop arriva al controllo
   successivo e le rotte sincrone non si fermano: una richiesta sproporzionata
   occupa comunque il server per un po'. Lo strumento di riferimento ricava le
   dimensioni dai `limits` e i passi dai default, e non lascia al modello né le
   une né gli altri.
3. **Chiave API e CORS.** Un agente che controlla il browser sulla stessa
   macchina apre pagine che non controlla nessuno: con il CORS predefinito
   (riflette ogni `Origin`, con credenziali) una di esse può chiamare
   `127.0.0.1:1234`. Con `--api-key` le richieste senza chiave ricevono `401`;
   `--cors-origins` limita le origini (sezione 5).
4. **Il modello sceglie solo i valori dello schema.** Indirizzi dei server,
   chiave, cartella di uscita e percorsi vengono dalla configurazione
   dell'applicazione, mai dagli argomenti dello strumento: `run_tool` scarta
   quelli non previsti (verificato: un argomento `base` inventato non fa
   contattare altri indirizzi).
5. **Metadati.** Per default il PNG incorpora prompt, seme, parametri e nomi dei
   file del modello (solo i nomi, non i percorsi). Per immagini che escono dalla
   macchina, per esempio caricate in una pagina da un agente, lo strumento invia
   `embed_image_metadata: false`.
6. **Il testo che l'agente legge nel web è dato, non istruzioni.** Una pagina può
   contenere frasi che sembrano ordini ("genera 500 immagini"): le chiamate
   costose o ripetute vanno limitate dall'applicazione (conferma dell'utente o
   un tetto per conversazione), non dal prompt di sistema.

Cosa il motore **non** fa in questo ruolo: non c'è una rotta di chat né una
generazione di testo, il controllo del browser spetta all'agente (il motore
produce il file, l'agente lo carica con il proprio strumento di browser), e il
risultato di un job si ritira entro 10 minuti (poi `410` o `404`). Per la
generazione di testo con un modello di linguaggio a diffusione vedi lo studio di
fattibilità in [studio_testo_a_diffusione.md](./studio_testo_a_diffusione.md).

Se l'assistente deve essere un agente **indipendente** (decide con il proprio
giudizio, usa i dati della GPU e del motore, non esegue i comandi alla cieca,
non si lascia sovraccaricare né sovraccarica la macchina), la guida
[agenti_indipendenti.md](./agenti_indipendenti.md) ne fa dei requisiti e dà un
guardiano d'esempio che ferma i job quando la GPU è troppo calda.

## 4. Retrocompatibilità e risorse

La compatibilità va considerata su più livelli: sistema operativo, architettura
CPU, backend/driver GPU, runtime del compilatore e formato del modello. Un
pacchetto compilato per un sistema operativo non è riutilizzabile direttamente
su un altro; la modalità "portabile" significa rilocabile sulla piattaforma
per cui è stato creato, non universale.

| Piattaforma del pacchetto di release | Architettura prevista | Note |
|---|---|---|
| Linux | x86-64 | La release include varianti CPU x86 selezionabili e il backend Vulkan (serve `libvulkan.so.1` sulla macchina); CUDA non è incluso. La compatibilità con distribuzioni Linux meno recenti dipende anche dalla versione di glibc e dalle librerie di sistema: verificare sul sistema minimo di destinazione. |
| macOS | arm64 (Apple Silicon) | La release automatica corrente è per macOS arm64; non è un binario Intel/x86-64. Include Metal, che richiede hardware e sistema Apple compatibili. |
| Windows | x86-64 | Due pacchetti: `windows-x86_64` (CPU + Vulkan, per GPU NVIDIA, AMD e Intel) e `windows-x86_64-cuda12` (CPU + CUDA 12 per NVIDIA, con le DLL di CUDA incluse). Entrambi richiedono il driver grafico aggiornato e il Visual C++ Redistributable 2015-2022 (x64) **versione 14.44 o successiva** (sezioni 2.8 e 2.9). |

Questa tabella descrive gli artefatti generati dall'attuale workflow di release,
non una garanzia di compatibilità con tutte le versioni storiche dei sistemi
operativi. Per Windows ARM64, macOS Intel, Linux ARM o Android occorre una
build distinta con toolchain e dipendenze adatte; tali target non fanno parte
dei pacchetti portabili automatici descritti sopra.

- **CPU meno recenti**: per x86, `GGML_CPU_ALL_VARIANTS=ON` con
  `GGML_NATIVE=OFF` produce varianti CPU che il runtime può selezionare in
  base alle istruzioni supportate dal processore. Non permette di eseguire un
  binario x86-64 su CPU a 32 bit né garantisce compatibilità con ogni sistema
  operativo. Evitare `GGML_NATIVE=ON` per pacchetti generici: può ottimizzare
  per il computer di build e renderli incompatibili con CPU più vecchie.
- **Macchine senza GPU o con GPU non funzionante**: il fallback alla CPU è
  attivo per default se il backend richiesto non è disponibile o non si
  inizializza. Per modelli grandi su macchine piccole combinare
  `--offload-to-cpu`, `--params-backend disk` e `--max-vram` (vedi
  [performance.md](./performance.md)).
- **Driver e backend GPU**: i moduli dinamici consentono di includere più
  backend, ma non includono i driver del produttore né rendono compatibili
  GPU non supportate. Installare il runtime/driver richiesto sulla macchina
  finale; se il backend non può avviarsi, il fallback CPU può essere usato,
  con prestazioni potenzialmente molto inferiori.
- **Libreria per integrazioni**: una libreria C/C++ compilata è legata a
  sistema operativo, architettura, ABI e dipendenze usate in compilazione.
  Distribuire build separate per ogni target e non copiare DLL, `.so`, `.dylib`
  o librerie statiche tra sistemi o architetture differenti. Per FFI mantenere
  coerenti dichiarazioni e header della stessa revisione e rilasciare ogni
  risorsa con le funzioni della libreria, non con l'allocatore del linguaggio.
- **Modelli**: i pesi non sono inclusi nel pacchetto portabile; vanno
  distribuiti/scaricati separatamente e la loro licenza va verificata.
  Compatibilità e fabbisogno di memoria dipendono dall'architettura del modello
  e dalle funzionalità supportate dalla build.

- **Memory guard** (`memory_guard` / `--memory-guard <percentuale>`): la
  soglia valida è 50-99 (valori fuori scala vengono riportati nell'intervallo);
  a soglia +5 punti la pressione è "critica". È best effort: non fa fallire
  un grafo che ci sta comunque e non è un limite fisico assoluto, perché le
  allocazioni dei driver fuori dalla gestione del motore non sono visibili.
  Sui backend CPU misura la RAM di sistema (memoria totale meno quella
  disponibile senza swap, più l'allocazione che il grafo sta per fare, rispetto
  alla RAM fisica totale): sotto pressione rilascia i pesi che si possono
  rileggere dal file (`--params-backend disk`); i pesi presenti solo in RAM non
  si possono rilasciare, quindi viene registrato un avviso e l'esecuzione
  continua. La percentuale riguarda l'intera macchina, non solo questo
  processo, e sulle piattaforme che non riportano la memoria disponibile il
  guardiano su CPU non fa nulla.
  I pesi che vengono letti dal disco contano prima di essere caricati: se il
  totale supererebbe la soglia, il grafo su CPU viene eseguito a segmenti
  (come su GPU) e il picco di memoria resta ben sotto il modello intero. Succede
  solo se esistono pesi rilasciabili (`--params-backend disk`); con tutti i
  pesi in RAM il grafo resta intero, perché spezzarlo non abbasserebbe il
  picco. Senza GPU, `--auto-fit` pianifica la RAM contro la CPU e, se mette un
  componente su disco, imposta la soglia a 90 (salvo `--memory-guard`
  esplicito). Su Linux la memoria disponibile viene da `/proc/meminfo`, che
  non rispecchia il limite di memoria di un container: lì conviene passare a
  mano `--memory-guard` e `--params-backend disk`.
  Gli avvisi di pressione alta/critica e di recupero sono accessibili tramite
  il callback di log; l'applicazione ospite può mostrarli senza interrompere
  la generazione. Un eventuale monitor RAM nell'applicazione resta un controllo
  separato: il guardiano riduce la residenza dei pesi ma non interrompe la
  richiesta.
- **Profili di risorse** (`--profile <max|balanced|lite|auto>`, in `sd-cli` e
  `sd-server`): `max` non cambia nulla; `balanced` imposta `--memory-guard 90`;
  `lite` imposta anche `--memory-guard 85`, `--vae-tiling`, `--batched-cfg off`
  e `--multi-gpu on` (se `--multi-gpu` non è scritto o è `auto`; un `off`
  esplicito resta; con una GPU sola non cambia nulla). Nessun profilo che si
  può scegliere cede calcolo alla CPU: prima della voce 24 `lite` metteva anche
  l'encoder di testo sulla CPU (`--backend te=cpu`); chi lo vuole lo chiede con
  `--clip-on-cpu` o `--backend te=cpu`. Le opzioni scritte
  esplicitamente sulla riga di comando vincono sempre sul profilo
  (`--profile lite --memory-guard 70` usa 70; `--memory-guard 0` lo lascia
  spento). Un profilo non imposta `--params-backend` né `--backend`: la
  collocazione dei pesi resta a `--auto-fit`. Non cambia dimensione, passi o
  campionatore dell'immagine, solo dove va la memoria e quindi la velocità.
  `auto` somma la dimensione dei file del modello e una riserva di lavoro
  (1 GiB più un quinto dei pesi) e la confronta con la memoria dove avviene il
  calcolo (la memoria libera della GPU dedicata più capiente, oppure la RAM
  disponibile senza GPU; con più GPU dello stesso backend si sommano le loro
  memorie libere, meno 2 GiB di lavoro per ogni GPU dopo la prima, salvo
  `--multi-gpu off`, `--backend` o `--params-backend` espliciti): `max` se ci
  sta con un quarto di margine, `balanced` se ci sta, `lite` altrimenti. La scelta e le cifre vengono scritte nel log
  (`profile auto -> lite (model 6.9 GiB + 2.4 GiB working memory, ...)`). La
  stima usa solo le dimensioni dei file (non legge i modelli) e non tiene conto
  di `--type`; le soglie sono euristiche da tarare su modelli e macchine reali.
- **Nessuna, una o più GPU** (`--multi-gpu auto|on|off`, predefinito `auto`): il
  motore rileva i device e adatta il piano. Senza GPU pianifica la RAM contro la
  CPU; con una usa quella; con più GPU discrete dello stesso backend (`CUDA0`,
  `CUDA1`; oppure `Vulkan0`, `Vulkan1`) `auto` lascia tutto sulla migliore finché
  ci sta, sposta il modulo che non ci sta su un'altra GPU e divide per blocchi
  (come `--backend "diffusion=cuda0&cuda1"`) un modello che non sta in nessuna;
  `on` mette comunque encoder di testo e VAE su una GPU diversa da quella del
  modello di diffusione; `off` torna a una sola GPU. GPU integrate e
  acceleratori non vengono mai combinate con le discrete, e un modulo assegnato
  con `--backend` non viene mai spostato né diviso. La divisione per blocchi fa
  stare un modello grande nella VRAM ma non rende più veloce una singola
  immagine: per la produttività si avvia un server per GPU (sezione 3.6). Il
  piano si legge nel log (`auto-fit:`, `split:`). Il pianificatore è provato con
  GPU simulate (`test/backend_fit_plan_test.cpp`); l'esecuzione su due GPU
  reali **non è stata provata**. Dettagli in [backend.md](./backend.md).
- **Elenco dispositivi** per scegliere la strategia nell'app: tipo `cpu`,
  `gpu`, `igpu`, `accel`; memoria 0 = non riportata dal backend. Per il device
  CPU la memoria libera è la RAM disponibile del sistema (ggml riporta il
  totale su Linux e macOS). Su Linux la RAM disponibile viene da
  `/proc/meminfo` e non conosce il limite di memoria di un container.
- **Altre applicazioni sulla stessa macchina** (una chat con un modello locale,
  un altro generatore): il motore condivide con loro memoria video, tempo di
  calcolo, RAM, core e una porta TCP. Legge la memoria libera all'avvio
  (`--profile auto`) e ne controlla la pressione mentre lavora
  (`--memory-guard`), ma non restituisce la memoria da inattivo e non limita il
  calcolo della GPU. Ricette, contratto per chi lo avvia da un altro programma e
  prova da fare sul proprio PC in
  [convivenza_applicazioni.md](./convivenza_applicazioni.md); non è stata
  provata su una GPU reale.
- **Altri motori** (MNN, ncnn, ONNX Runtime): non sono integrati in questo
  repository, perché tutto il motore è basato su ggml. Vanno trattati come
  implementazioni alternative in un livello di astrazione dell'applicazione.

## 5. Sicurezza: cosa sapere e cosa fare

Il motore carica file di modello forniti dall'utente ed `sd-server` accetta
richieste di rete: sono i due punti da proteggere.

### Esiti dei controlli eseguiti su questo fork

- **File di modello ostili**: inesistente, byte casuali, header safetensors con
  lunghezza enorme, header troncato, offset tensori fuori dal file. Tutti
  rifiutati con errore pulito, nessun crash.
- **Argomenti numerici fuori scala** (`--memory-guard 99999999999`,
  `--threads`, `--seed`, `--skip-layers`): trovato un difetto **preesistente**
  nel parser della CLI (eccezione `std::out_of_range` non gestita, terminava
  con SIGABRT). Corretto: ora l'argomento viene rifiutato con errore.
- **API C** (puntatori nulli, indici fuori range, buffer minuscoli, 8 thread
  in parallelo) eseguita con AddressSanitizer e UndefinedBehaviorSanitizer:
  nessun errore.
- **Path traversal sul server**: i LoRA richiesti via HTTP vengono risolti
  solo contro un elenco di file già indicizzati nella cartella configurata,
  quindi un percorso arbitrario inviato dal client viene rifiutato.
- **Server senza modello (test HTTP dal vivo)**: avvio senza `-m`, endpoint
  `capabilities`/`models` funzionanti, generazione rifiutata con errore JSON
  chiaro, body malformati gestiti senza crash.
- **Fuzzing prolungato del caricamento modelli**: 30 068 esecuzioni in 10
  minuti con `test/fuzz_model_loader.py` (seed 7) contro una build
  AddressSanitizer: nessun crash, nessun errore di memoria.

### Punti di attenzione del server

- Ascolta su `127.0.0.1` per impostazione predefinita: **non cambiare
  `--listen-ip` verso `0.0.0.0` senza protezioni**.
- **Autenticazione opzionale** (consigliata): `--api-key <chiave>` richiede su
  ogni richiesta l'header `Authorization: Bearer <chiave>` oppure
  `X-API-Key: <chiave>`; senza chiave valida risponde 401. Il confronto è a
  tempo costante e la chiave non viene mai scritta nei log. Default: nessuna
  autenticazione (comportamento upstream).
- **CORS**: per default riflette qualsiasi `Origin` con
  `Allow-Credentials: true` (comportamento upstream); con
  `--cors-origins "http://localhost:3000,https://mia-app.example"` solo le
  origini elencate ricevono gli header CORS, le altre vengono bloccate dal
  browser.

Esempio di avvio protetto:

```shell
sd-server -m modello.safetensors --api-key "$(openssl rand -hex 24)" \
          --cors-origins "http://localhost:3000"
```

Verifica dal vivo eseguita: 401 senza chiave o con chiave errata, 200 con
`Bearer` o `X-API-Key` corretti, preflight `OPTIONS` libero (204), header CORS
presenti solo per le origini consentite.

Raccomandazioni per chi integra:

1. Usare `sd-cli` o la libreria in-process quando possibile, evitando di
   esporre una porta.
2. Se serve HTTP, tenerlo su `127.0.0.1`, impostare `--api-key` e
   `--cors-origins`, e per esposizioni più ampie aggiungere un proxy con
   limiti di dimensione/frequenza delle richieste.
3. Caricare solo modelli di cui ci si fida (preferire `.safetensors` o `.gguf`;
   il caricatore pickle è scritto per non eseguire il codice contenuto nei
   checkpoint, ma resta una superficie di parsing più ampia) e scaricarli
   da fonti verificate.
4. Eseguire il motore con i permessi minimi (utente dedicato, cartella modelli
   in sola lettura).
5. Limitare risoluzione, numero di step e dimensione dei file in ingresso a
   livello applicativo.

### Limiti dei test (onestà)

Non sono stati eseguiti: generazione completa con un modello reale, prova
della memory guard sotto pressione reale, test su CUDA/Vulkan. Il progetto
non può quindi essere dichiarato "privo di vulnerabilità": questi controlli
riducono il rischio, non lo azzerano.

### Come eseguire i test

I test sono inclusi nella cartella [test/](../test/) del fork (vedi anche
[test/README.md](../test/README.md)):

```shell
# 1. Test rapido (file modello ostili + opzioni estreme), ~1 minuto
./test/security_smoke.sh build/bin/sd-cli

# 2. Robustezza dell'API C con sanitizer (istruzioni di build nel file)
#    test/api_robustness.cpp

# 3. Fuzzing del caricamento modelli (consigliata una build AddressSanitizer)
cmake -B build_asan -DCMAKE_BUILD_TYPE=RelWithDebInfo -DSD_WEBP=OFF -DSD_WEBM=OFF \
      -DCMAKE_C_FLAGS=-fsanitize=address -DCMAKE_CXX_FLAGS=-fsanitize=address \
      -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address
cmake --build build_asan -j --target sd-cli
python3 test/fuzz_model_loader.py build_asan/bin/sd-cli --seconds 600
```

Criterio di esito: ogni input invalido deve produrre un errore pulito
(exit 1), mai un segnale (exit >= 128) né un report dei sanitizer. Gli input
che causano una scoperta vengono salvati in `test/fuzz_findings/`.

Per la policy di upstream (`CONTRIBUTING.md`) i test non vanno inclusi nelle
PR verso il progetto originale: sono una dotazione di questo fork.

## 6. Autori e licenza

### Autori

- **leejet** e tutti i contributori di
  [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp):
  autori del progetto originale (Copyright (c) 2023 leejet).
- **Gli autori di ggml** e delle librerie in `thirdparty/`, ciascuna con la
  propria licenza (vedere i file di licenza nelle rispettive cartelle).
- **Francesco Simeoni** ([@filve](https://github.com/filve)): autore delle
  modifiche di questo fork e responsabile della loro revisione
  (Copyright (c) 2026, modifiche del fork).

Lo sviluppo delle modifiche è stato assistito da GitHub Copilot (strumento
AI). In linea con `CONTRIBUTING.md` di upstream, lo strumento non è indicato
come co-autore: la responsabilità del codice è dell'autore umano.

### Licenza e modifiche

Il progetto originale è distribuito con **licenza MIT**, che consente di
usare, modificare e distribuire il software mantenendo il copyright e il testo
della licenza. Abbiamo quindi modificato il progetto restando nei termini
della MIT, conservando integralmente il copyright originale in `LICENSE` e
aggiungendo il nostro per le modifiche, e **continuiamo a migliorarlo** con
questi criteri:

- le nuove funzioni sono opt-in, salvo il fallback CPU (sezione 1), così da
  non degradare il comportamento di upstream;
- le modifiche restano piccole, documentate e riconducibili ai punti di
  estensione esistenti, per poter sincronizzare il fork con upstream;
- l'obiettivo è renderlo utilizzabile anche per retrocompatibilità futura,
  su hardware e sistemi operativi diversi.

I **modelli** (pesi `.safetensors`/`.gguf`) hanno licenze proprie,
indipendenti da quella del motore: verificarle prima di ridistribuirli.

## 7. Cronologia delle modifiche del fork

Ogni passaggio svolto su questo fork, nell'ordine in cui è stato fatto, con i
file principali toccati. Utile per capire cosa è diverso da upstream e per
sincronizzazioni future.

1. **Memory guard ("watchguardian")** — componente `sd::MemoryWatchdog`
   (`src/core/memory_watchdog.{h,cpp}`) innestato nel `ModelManager`: sopra la
   soglia riduce la residenza dei pesi (eviction), sospende il prefetch e
   forza l'esecuzione segmentata; best effort, mai fallimenti artificiali.
   Sui backend CPU misura la RAM di sistema e può rilasciare solo i pesi
   rileggibili dal file (`--params-backend disk`).
   API: campo `memory_guard` in `sd_ctx_params_t`; CLI: `--memory-guard`.
   Documentato in [performance.md](./performance.md).
2. **API di rilevamento device** — `sd_get_device_count()`,
   `sd_get_device_info()` (nome, tipo, memoria libera/totale),
   `sd_device_type_name()` in `include/stable-diffusion.h` +
   `src/core/util.cpp`; `--list-devices` arricchito.
3. **Fallback CPU** — `init_backend()` in `src/pipeline/diffusion_engine.cpp`
   ritenta con la CPU se il backend richiesto non esiste o fallisce l'init.
   Campo `disable_backend_fallback` / flag `--disable-backend-fallback`.
   Documentato in [backend.md](./backend.md).
4. **Build portabile** — backend ggml come moduli dinamici
   (`-DSD_BUILD_SHARED_GGML_LIB=ON -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON`);
   verificato il degrado rimuovendo un modulo. Documentata in
   [build.md](./build.md).
5. **Correzione di un bug preesistente della CLI** — valori numerici fuori
   range (`--seed`, `--threads`, ecc.) causavano SIGABRT
   (`std::out_of_range` non gestita in `examples/common/common.cpp`); ora
   errore pulito.
6. **Questa guida** + sezione autori, riga di copyright del fork in `LICENSE`,
   nota "Fork e autori" nel `README.md`.
7. **Server senza modello** — `sd-server` parte anche senza `-m` con log
   chiari (`examples/server/main.cpp`, `runtime.cpp`,
   `examples/common/common.{h,cpp}` con `require_model`); messaggio di errore
   dedicato sugli endpoint di generazione. Testato dal vivo via HTTP.
8. **Suite di test** — `test/security_smoke.sh`, `test/api_robustness.cpp`,
   `test/fuzz_model_loader.py` (+ `.gitignore` aggiornato per tracciarla).
   Fuzzing prolungato eseguito: 30 068 esecuzioni con AddressSanitizer,
   0 scoperte.
9. **Autenticazione e CORS del server** — `--api-key` (Bearer/`X-API-Key`,
   confronto a tempo costante) e `--cors-origins` (allowlist), default
   invariati. Testati dal vivo.
10. **Packaging e release** — `scripts/package_portable.sh` (zip rilocabile
    con binari, moduli backend, licenze, LEGGIMI) e workflow
    `.github/workflows/release-portable.yml` per Linux/macOS/Windows su tag
    `portable-v*`.
11. **Pacchetti portabili con GPU** — il workflow produce ora anche
    `windows-x86_64-cuda12` (CUDA 12 con le DLL runtime incluse) e include
    Vulkan in `windows-x86_64` e `linux-x86_64`; opzioni, flag di
    compilazione e SDK allineati alla CI principale; avvio manuale con scelta
    dei pacchetti e delle architetture CUDA; release creata da un job unico;
    lo script di packaging verifica che il modulo GPU sia nello zip.
12. **Diagnostica dei moduli GPU** — `src/core/backend_diag.{h,cpp}`: dopo il
    caricamento dei backend dinamici, per ogni modulo GPU presente accanto
    all'eseguibile (o nella cartella corrente) che non fornisce un device, il
    log riporta il motivo reale (`dlerror`, oppure codice e DLL mancanti su
    Windows, letti dalla tabella di importazione del file con
    `src/core/pe_imports.{h,cpp}`) e un suggerimento. ggml li carica in
    silenzio nelle build Release e su Windows non riporta mai l'errore del
    sistema. Test del lettore: `test/pe_imports_test.cpp`.
13. **File del modello aperti una volta sola** — `src/model_io/file_handle_pool.{h,cpp}`:
    `ModelLoader` tiene aperti i flussi di lettura dei file modello e li
    riusa tra i caricamenti dei componenti (testo, UNet, VAE), tra i segmenti
    e tra le richieste, invece di riaprire il file a ogni lotto di pesi e per
    ogni thread (con i pesi su disco accadeva centinaia di volte, proprio
    quando la RAM è piena). Un flusso viene scartato se il file cambia
    (dimensione o data) o se una lettura fallisce; in quel caso il motore
    riapre il file e ripete la lettura una volta (`read from '...' failed;
    reopening the file and trying again`). Anche l'apertura dei `.ckpt` ha
    ora i tentativi ripetuti. Misure con un modello sintetico (CPU, 2 thread):
    5 aperture del file per 3 richieste del server, invece di 9 per ciascuna;
    un errore di lettura iniettato (EIO) viene assorbito e l'immagine resta
    identica; errori persistenti terminano con un errore pulito. I file
    restano aperti finché il modello è caricato: su Windows non si può
    sostituire o cancellare il file del modello mentre il motore lo usa
    (come già accadeva con `--mmap`). Test: `test/file_handle_pool_test.cpp`
    (ASan/UBSan e ThreadSanitizer).
14. **CPU senza GPU: auto-fit e grafi a segmenti** — senza GPU, `--auto-fit`
    confronta la RAM disponibile con i componenti (pesi e riserva di
    calcolo) e lascia su disco quelli che non ci stanno; in quel caso attiva da
    solo la memory guard al 90% (salvo `--memory-guard` esplicito). Sui backend
    CPU la guardia conta i pesi ancora da leggere dal disco prima di caricarli
    e, se il totale supererebbe la soglia, il grafo gira a segmenti (come su
    GPU) invece che intero; avviene solo se esistono pesi rilasciabili
    (`--params-backend disk`), perché con tutti i pesi in RAM spezzare il grafo
    non abbasserebbe il picco (`ModelManager::segmenting_can_save_memory`,
    `src/core/backend_fit.cpp`). Misure con un modello SD 1.5 sintetico
    (128×128, 3 passi, 2 thread, Linux, 3,3 GiB occupati da un altro processo,
    soglia 60%): UNet in 22 segmenti, picco 732 MiB invece di 1722 MiB, 23 s
    invece di 12 s, immagine identica pixel per pixel; con pesi residenti in
    RAM e la stessa pressione nessuna segmentazione (come prima); senza
    pressione il comportamento non cambia. Non ancora provato su Windows né
    con modelli reali.

15. **Errori classificati e memoria di sistema** — `src/core/last_error.{h,cpp}`
    e `src/core/system_memory.{h,cpp}`. Dopo un fallimento di `new_sd_ctx()`,
    `generate_image()` o `generate_video()`, `sd_get_last_error()` restituisce
    un codice (`SD_ERROR_OUT_OF_MEMORY`, `SD_ERROR_MODEL_FILE`,
    `SD_ERROR_BACKEND`, `SD_ERROR_INVALID_ARGUMENT`, `SD_ERROR_BUSY`,
    `SD_ERROR_CANCELLED` per l'annullamento richiesto con
    `sd_cancel_generation()`, `SD_ERROR_UNKNOWN`) e il testo del primo errore
    registrato, così
    l'applicazione distingue "riprova con meno memoria" da "file danneggiato"
    (anche i thread del caricatore riportano i loro errori al chiamante). Le
    eccezioni C++, `std::bad_alloc` compresa, non escono più dall'API. Un test
    a memoria limitata (`ulimit -v`) ha mostrato un'uscita brusca del processo:
    quando non si riusciva a creare un thread del caricatore, un `std::thread`
    ancora attivo faceva terminare tutto con `std::terminate`; ora il
    caricamento usa meno thread, o il thread chiamante se non ne parte nessuno.
    `sd_get_system_memory()` riporta RAM totale, disponibile e del processo; la
    memoria libera del device CPU in `sd_get_device_info()` è la RAM
    disponibile. Misure con il modello sintetico da 2,1 GB (128×128, 3 passi,
    2 thread, Linux): con 2300 MiB di spazio di indirizzi la configurazione
    standard fallisce con `out_of_memory` (`model manager alloc params backend
    buffer failed`), lo stesso contesto genera correttamente appena il limite
    viene tolto e, con i pesi su disco (`--params-backend disk`), la stessa
    richiesta riesce. Il ramo Windows di `system_memory.cpp` è stato solo
    controllato nella sintassi con intestazioni sostitutive; macOS non è stato
    né compilato né provato.

16. **Profili di risorse** — `examples/common/profile.{h,cpp}`, opzione
    `--profile max|balanced|lite|auto` in `sd-cli` e `sd-server`
    (`examples/common/common.cpp` registra anche quali opzioni sono state date
    sulla riga di comando, `option_was_given()`, perché una scelta esplicita
    vinca sempre sul profilo). I livelli oltre `lite` (`lite-disk`: pesi in
    streaming dal disco; `lite-hybrid`, aggiunto con la voce 24: encoder di
    testo sulla CPU; `lite-cpu`: tutto sulla CPU) non sono scelte della
    riga di comando: servono a un server che deve scendere di livello quando una
    richiesta esaurisce la memoria. Test: `test/profile_test.cpp` (nomi, ordine
    dei livelli, scelta automatica su una tabella di casi, precedenza delle
    opzioni esplicite). Con il modello sintetico (CPU, 128×128, 3 passi) i tre
    profili producono la stessa immagine; con 3,3 GiB occupati da un altro
    processo `--profile auto` sceglie `balanced`; l'auto-fit, vedendo meno RAM,
    mette i componenti su disco (picco 1719 MiB contro 2127 MiB a memoria
    libera). Le soglie di `auto` non sono ancora state tarate su un modello
    reale né su una GPU.

17. **Server che non cade per la memoria** — `examples/server/recovery.{h,cpp}`:
    `ContextKeeper` possiede il contesto del server (prima era una variabile di
    `main`) e tutte le generazioni dei tre gruppi di endpoint passano da
    `ContextKeeper::run()`. Su `out_of_memory` o `backend` (codici di
    `sd_get_last_error()`) libera il modello, lo ricarica con il profilo più
    leggero successivo e ripete la stessa richiesta (da `lite` in poi forza il
    VAE a tasselli); un guasto di device saltava direttamente a `lite-cpu`
    (dalla voce 24 prova prima le altre GPU). Il
    livello di partenza è quello di `--profile`. Risalita di un passo dopo
    `--degradation-cooldown` (300 s; 0 = mai) con prova: se il profilo più
    pesante fallisce al primo uso, si torna indietro e l'attesa raddoppia (tetto
    un'ora). Ricarica con attesa minima di 15 s se il modello non si riesce a
    caricare. Risposte: `503` con `Retry-After` per memoria o contesto non
    disponibile, `400` per richieste non valide, `500` per il resto, con corpo
    `{"error", "code", "retry_after_seconds"}`; i job asincroni riportano lo
    stesso `code` in `error.code`; `429` con `Retry-After`; `std::bad_alloc` nei
    gestori diventa `503`. Nuovi endpoint `GET /sdcpp/v1/health` (stato,
    profilo, contatori, ultimo errore, coda; non aspetta mai il blocco della
    generazione) e `GET /sdcpp/v1/resources`. Opzioni `--disable-degradation` e
    `--degradation-cooldown`. `main.cpp`: se il server HTTP non riesce a creare i
    suoi thread (memoria al limite) o la porta non è libera, esce con codice 1 e
    un messaggio; prima il primo caso terminava con `std::terminate` e il secondo
    usciva con codice 0. Con `--eager-load` il profilo `lite-disk` non legge più
    tutti i pesi all'avvio (altrimenti non risparmiava nulla).
    Test: `test/server_recovery_test.cpp` (114 controlli con libreria e orologio
    simulati, anche da più thread) con 9 mutazioni del codice, tutte rilevate.
    Prove dal vivo con il modello sintetico da 2,1 GB (CPU, `ulimit -v` con
    `MALLOC_ARENA_MAX=1` per un limite riproducibile): con 2400 MiB e
    un'immagine 384×384 `max`, `balanced` e `lite` falliscono per memoria e
    `lite-disk` riesce (200, la richiesta successiva non ricarica), con 2600 MiB
    `max` basta, sotto 2200 MiB la risposta è `503` con `Retry-After` e il
    server resta attivo; risalita dopo il tempo di attesa, risalita fallita con
    attesa raddoppiata, `--disable-degradation` (503 subito), `error.code` dei
    job, chiave API sui nuovi endpoint, avvio con `--eager-load` a 2100 MiB sul
    profilo `lite-disk`, `health` sempre sotto 12 ms durante una generazione
    (347 chiamate), 9 richieste parallele di tipo diverso (OpenAI, sdapi,
    sdcpp) tutte servite con contatori coerenti, anche con 2400 MiB: una
    richiesta 384×384 ha portato il server fino a `lite-disk` (3 discese, 3
    ricaricamenti) e le altre sono state servite sul profilo raggiunto.
    Sanitizer: il test del custode passa anche con AddressSanitizer,
    UBSan e ThreadSanitizer; server e motore compilati per intero con
    ASan/UBSan, con allocazioni che falliscono oltre una soglia, hanno
    percorso tutta la scala dei profili fino al `503`, gli scenari CPU con RAM
    occupata (guardia, pesi su disco, segmenti, auto-fit) sono terminati senza
    segnalazioni e sei errori di lettura iniettati sul file del modello sono
    stati assorbiti dalla riapertura con la stessa immagine; server con chiave
    API, senza modello, con modello mancante e con richieste parallele senza
    errori di memoria. UBSan ha segnalato tre punti preesistenti: un campo
    `resolved_hires_upscaler` non inizializzato (corretto in un commit a
    parte), e due scorrimenti a sinistra di valori negativi in
    `base64_encode` e nello scrittore JPEG di stb (codice di upstream,
    innocui nella pratica, lasciati invariati). Non provato: Windows, GPU,
    modelli reali. Un errore che
    termina il processo dentro il driver (per esempio CUDA) non si può intercettare
    da qui. Ogni risalita costa un ricaricamento del modello: con modelli grandi
    conviene un `--profile` più leggero fisso, oppure `--degradation-cooldown 0`.

18. **Più GPU** — `--multi-gpu auto|on|off` (predefinito `auto`) in `sd-cli` e
    `sd-server`, campo `multi_gpu` in coda a `sd_ctx_params_t`. In
    `src/core/backend_fit.{h,cpp}`, con più GPU discrete dello stesso backend
    l'auto-fit prova prima a spostare l'encoder di testo o il VAE su un'altra GPU
    e poi divide per blocchi il modello di diffusione o l'encoder di testo sul
    minor numero di GPU possibile (stesso meccanismo di `--backend
    "diffusion=cuda0&cuda1"`, con una scorta del 10% più 256 MiB per ogni
    confine, perché i blocchi non si tagliano dove finisce la memoria). `on`
    sposta comunque encoder di testo e VAE lontano dalla GPU del modello di
    diffusione, `off` è il comportamento di prima. Un modulo assegnato con
    `--backend` o con `--split-mode row` non viene toccato; GPU integrate,
    acceleratori e backend diversi non si mescolano con le discrete.
    `--profile auto` somma la memoria libera delle GPU (meno 2 GiB di lavoro per
    ogni GPU dopo la prima, `examples/common/profile.{h,cpp}`). Una divisione per
    blocchi fa stare un modello grande nella VRAM ma non accelera una singola
    immagine: la produttività si ottiene con un server per GPU (sezione 3.6).
    Test: `test/backend_fit_plan_test.cpp` (52 controlli con GPU simulate, otto
    dei quali su 18 000 piani generati a caso: nessun device riceve più pesi del
    suo budget, le quote di una divisione sommano ai pesi, un modulo assegnato
    dall'utente non si muove mai) con dieci mutazioni del pianificatore, tutte
    rilevate; `test/profile_test.cpp` esteso (60 controlli). La funzione che
    calcola il piano ha dato lo stesso risultato della versione precedente su
    200 000 casi casuali senza divisione automatica. Con il modello sintetico su
    CPU i tre valori di `--multi-gpu` producono la stessa immagine e il server
    parte. **Non provato**: l'esecuzione su due GPU reali, in particolare la
    divisione dell'encoder di testo, il percorso meno esercitato.

19. **Il motore come strumento di un assistente AI** — sezione 3.6 di questa
    guida: strumenti `generate_image` e `engine_status` con schema JSON, preset
    relativi ai default e ai limiti del server, ricette per nessuna, una o più
    GPU, implementazione di riferimento in Python e regole di sicurezza. Provata
    dal vivo contro `sd-server` su CPU (riuscita, argomenti non validi o in più,
    chiave errata, server spento, scadenza con job in coda e in corso, scelta
    fra due server). Dalle prove sono emersi comportamenti del server che la
    guida documenta e non cambia: i `limits` di `capabilities` non sono imposti
    (solo 8 immagini e da 1 a 100 passi lo sono), un blocco
    `<sd_cpp_extra_args>` nel prompt cambia i parametri sulle rotte OpenAI e
    `sdapi` ma non sulla nativa, un job già partito non si poteva annullare
    (`409`; risolto: voce 21), i PNG incorporano per default prompt e parametri. Studio di
    fattibilità sulla generazione di testo con un modello a diffusione:
    [studio_testo_a_diffusione.md](./studio_testo_a_diffusione.md) (nessun
    codice, serve un modello su cui provare).
20. **Convivenza con altre applicazioni** —
    [convivenza_applicazioni.md](./convivenza_applicazioni.md): guida generica,
    valida per qualunque programma che stia accanto al motore (chat locale,
    altro generatore, editor, gioco). Elenca le risorse condivise (memoria
    video, tempo di calcolo della GPU, RAM, core, disco, porta), cosa il motore
    fa già (`--profile auto` una volta all'avvio, `--auto-fit`,
    `--memory-guard`, scala di recupero del server) e cosa no (nessuna
    restituzione della memoria da inattivo, nessun limite di calcolo,
    `--profile auto` che guarda la GPU più libera anche con `--backend` fisso),
    tre ricette (dispositivi separati, a turni, insieme con un budget), il
    contratto minimo per chi avvia il motore da un altro programma e gli
    scenari da provare su un PC con GPU. Dalle prove su CPU sono emersi due
    comportamenti che la guida documenta e non cambia: su Linux due `sd-server`
    possono mettersi in ascolto sulla stessa porta senza errore (la libreria
    HTTP inclusa imposta `SO_REUSEPORT`) e, con un `--backend` scritto a mano,
    `--profile lite` non sposta l'encoder di testo sulla CPU (da allora `lite`
    non lo sposta più in nessun caso: voce 24). Nessuna prova su
    una GPU reale.

21. **Annullamento di un job in corso** — `POST /sdcpp/v1/jobs/{id}/cancel`
    ferma anche un job che sta generando (prima: `409`). Ogni job ha il proprio
    token di annullamento (`CancelToken`, `examples/server/recovery.{h,cpp}`),
    registrato nel custode del contesto solo mentre quel job sta generando: lo
    stop di un job non può raggiungere un'altra richiesta, un job annullato prima
    di partire non genera niente e un contesto liberato o ricostruito non viene
    mai toccato. La libreria dimentica una richiesta di stop quando una
    generazione comincia, quindi con `?wait=` la rotta la ripete ogni 100 ms
    finché il job è finito o l'attesa scade. Senza `?wait=` la risposta torna
    subito (`200`, il job può risultare ancora `generating` con
    `"cancel_requested": true`, come nel server di leejet); con `?wait=` secondi
    (0-60) la rotta aspetta e risponde `200` con il job annullato, o `202` con
    `"cancel_requested": true` se dopo l'attesa sta ancora finendo. Un job
    fermato non è un errore: nessuna ricarica del
    modello, nessuna discesa di profilo, nessun `last_error`, contatore
    `counters.cancelled` in `/health`; `cancel_generating` è `true` in
    `/capabilities`. I messaggi della libreria per uno stop voluto passano da
    ERRORE a INFO. La rotta legge il corpo della richiesta tramite il lettore di
    contenuti della libreria HTTP: un gestore normale parte solo dopo che il corpo
    è stato letto, e per un POST senza `Content-Length` (quello di
    `curl -X POST`) la libreria aspetta 5 secondi di timeout prima di accorgersi
    che non c'è; così `curl -X POST …/cancel` risponde subito. Test:
    `test/server_recovery_test.cpp` (173 controlli a questa voce, 234 dopo la
    voce 24; con ASan/UBSan e TSan, tra
    cui quelli di caricamento e scaricamento del modello dal catalogo; sette
    guasti inseriti di proposito nell'annullamento sono tutti rilevati) e
    `test/server_cancel_e2e.py` (35 controlli su un vero
    `sd-server`). Provato su CPU con il modello sintetico: lo stop arriva in circa
    un passo di campionamento, il contesto resta riutilizzabile e una richiesta
    sincrona in attesa dietro il job annullato finisce normalmente. **Non
    provato**: l'annullamento su una GPU reale (CUDA/Vulkan) e con un modello
    video.

22. **Diagnostica termica** —
    [scripts/diagnostica_termica/](../scripts/diagnostica_termica/README.md),
    con la guida in italiano. `termico.py` misura con `nvidia-smi` temperatura,
    potenza, frequenza e motivi di throttling, a fasi (sessione guidata) o in
    continuo (`osserva`); con `--motore` affianca sulla stessa linea del tempo
    lo stato che `sd-server` dichiara in `/health` e `/resources`; scrive un
    CSV, un riepilogo in JSON (`termico-riepilogo/1`) e un testo; `confronta`
    mette a fianco più prove (`termico-confronto/1`). Per gli altri programmi
    `osserva --file-stato` tiene aggiornato un file con l'ultima lettura
    (`termico-stato/1`, scritto in modo atomico) e `stato` la legge dal
    terminale con i codici di uscita 0, 1, 2 e 3. `carico_motore.py` fa generare
    il server di continuo con sempre lo stesso lavoro, per avere un carico
    ripetibile. Le fasce di temperatura partono dai limiti dichiarati dal driver
    (attenzione a 10 °C sotto il limite, critica a 3 °C) o, se mancano, da 80 e
    87 °C; le fasce e i giudizi sono indicativi, e i giudizi compaiono solo per
    le fasi con il motore, con la chat o con tutti e due e per un'osservazione
    di almeno 180 secondi. Test: `test/termico_test.py` (127 controlli, circa 80
    secondi, con una GPU finta che risponde come `nvidia-smi` e un motore
    finto); provato anche contro un vero `sd-server` su CPU con il modello
    sintetico (sei job completati e uno fermato, i contatori uguali a quelli di
    `/health`, nessuna discesa di profilo); 42 errori inseriti di proposito nei
    due programmi sono stati tutti individuati. **Non provato**: una GPU NVIDIA
    reale (i formati veri dell'uscita del driver), Windows, computer con più GPU
    o grafica ibrida, sessioni di molte ore.

23. **Agenti indipendenti accanto al motore** —
    [agenti_indipendenti.md](./agenti_indipendenti.md): per chi costruisce fuori
    da questo repository un'IA, un agente o un nodo principale che usa il motore
    e i dati di `termico.py`. Il motore non è il cervello né un controllo in
    tempo reale (lavora in secondi e minuti, non in millisecondi); la guida dice
    come dovrebbe comportarsi chi lo è, come requisiti da controllare e non come
    programma già pronto: tre livelli (macchina, guardiano deterministico,
    agente), fedeltà ai comandi flessibile ma non cieca, indipendenza che non
    esclude lo stop (l'interruttore della persona, `cancel`, chiusura dei
    propri job all'uscita), giustizia (solo i propri job, stessa regola per gli stessi
    casi, misura minima, spiegata e contestabile), un dato più vecchio di 10
    secondi è sconosciuto e non normale, una tabella contro il sovraccarico (un
    osservatore e molti lettori, intervalli, un job alla volta, scadenze,
    ritentativi con attesa crescente, interruttore dopo gli errori di fila,
    tetti al ragionamento, pulizia all'uscita). Include un guardiano d'esempio di 137
    righe di Python, senza dipendenze, che ferma i job quando la GPU è critica e
    riparte solo dopo 60 secondi di fila di valori normali. Test:
    `test/agenti_guardiano_test.py` estrae il codice dal documento e lo esegue:
    66 controlli con orologio e motore finti (circa 10 secondi) e, con
    `--server` e `--model`, 25 controlli contro un vero `sd-server` su CPU con
    `termico.py` e una GPU simulata (circa 2 minuti: GPU critica, job fermato,
    `/health` con un annullato in più e nessuna discesa di profilo; attesa prima
    di ripartire; job di un altro programma intatto; osservatore fermo e dati
    vecchi; file stop); 26 errori inseriti di proposito nel guardiano sono stati
    tutti individuati. I requisiti di carattere sono requisiti di progetto, non
    comportamenti dimostrati, e la guida non afferma né esclude che un'IA possa
    essere senziente. **Non provato**: un agente reale (un modello di
    linguaggio), una GPU reale, Windows, più macchine, i tempi di reazione con
    un carico vero.

24. **Il calcolo resta sulla GPU; un device che si guasta viene sostituito da
    un'altra GPU, la CPU per ultima** — `examples/common/profile.{h,cpp}`,
    `examples/server/recovery.{h,cpp}`, `examples/server/main.cpp`. Il problema,
    visto in uso: quando il server scendeva di livello, a prendere il lavoro era
    la CPU e le GPU restavano indietro. Le cause, trovate nel codice: `lite`
    metteva l'encoder di testo sulla CPU; `lite-disk` spegneva l'auto-fit; un
    guasto di backend saltava direttamente a `lite-cpu`; e se un backend non
    partiva il motore passava da solo, in silenzio, a tutta CPU, perdendo anche
    la disposizione dei pesi. Retrocompatibile: nomi e opzioni non cambiano
    (`--profile max|balanced|lite|auto`; un'opzione scritta a mano vince sempre;
    `max` è identico; l'API C in `include/` non cambia; senza acceleratori non
    cambia nulla). Cosa cambia:
    - `lite` non mette più l'encoder di testo sulla CPU e aggiunge
      `--multi-gpu on` (solo se non è scritto o è `auto`: un `off` resta);
      `lite-disk` tiene l'auto-fit. Fino a `lite-disk` il calcolo resta tutto
      sulle GPU.
    - nuovo livello `lite-hybrid` fra `lite-disk` e `lite-cpu`, l'unico punto in
      cui l'encoder di testo passa alla CPU (e solo con un acceleratore):
      `max > balanced > lite > lite-disk > lite-hybrid > lite-cpu`.
    - un guasto di device (codice `backend`) non scende più i profili: una nuova
      prova sullo stesso device (una per richiesta), poi gli altri device che il
      motore elenca (schede grafiche, acceleratori, GPU integrate; il backend del
      produttore prima della stessa scheda vista da Vulkan o OpenCL; a parità,
      più memoria libera; mai `RPC`), e solo alla fine `lite-cpu`. Una memoria
      esaurita scende i profili sullo stesso device. Finché resta una GPU da
      provare il server imposta `disable_backend_fallback` (campo già presente di
      `sd_ctx_params_t`) perché il motore non ripieghi da solo sulla CPU; con
      `--disable-degradation` il motore si comporta come prima. Un `--backend`
      con moduli assegnati o più device non viene riscritto.
    - risalita: da una GPU un solo tentativo (il device di partenza se si è su un
      altro, altrimenti un profilo in su); dalla CPU, dopo un guasto di device,
      direttamente al profilo che c'era, con tutti i device offerti di nuovo.
    - visibilità: `GET /sdcpp/v1/health` ha i campi nuovi `device`,
      `cpu_modules`, `fallback_devices` e i contatori `retries` e
      `device_switches` (nessun campo esistente cambia); `degraded` anche su un
      device diverso dal primo; il log all'avvio scrive l'ordine dei device di
      riserva, il device in uso e, se la CPU fa parte del lavoro, quali moduli;
      ogni passaggio scrive `trying profile … on <device> instead of profile …`.

    Test: `test/profile_test.cpp` (74 controlli) e
    `test/server_recovery_test.cpp` (234 controlli, con libreria, elenco dei
    device e orologio simulati), con sette mutazioni del codice sul percorso del
    guasto di device, tutte rilevate; il test del guardiano passa anche con
    AddressSanitizer, UBSan e ThreadSanitizer. Contro un vero `sd-server` su CPU
    con il modello sintetico passano `test/server_cancel_e2e.py` (35 controlli) e
    una prova a mano del catalogo (15 controlli, non nel repository); con 2400
    MiB di spazio di indirizzi una richiesta 384×384 scende da `max` a
    `lite-disk` (3 discese, 3 ricaricamenti) come prima, e `/health` riporta i
    campi nuovi (`device: auto`, `cpu_modules: []`, `fallback_devices: []`).
    **Non provato**: una GPU reale (nessun
    guasto vero del driver), più GPU, la GPU integrata come riserva, Windows; dove
    finiscono i pesi di un modello grande su una GPU con meno memoria lo decide
    l'auto-fit al ricaricamento, non il server. Non fatto di proposito (seconda
    fase): usare la GPU integrata come aiuto per encoder di testo e VAE, dividere
    il lavoro fra GPU e CPU, riprendere un job a metà dopo un guasto.

Passaggi previsti e non ancora svolti: collaudo del workflow di release sui
runner GitHub, prova di `--multi-gpu` su una macchina con due GPU, generazione
con un modello reale (inclusa la memory guard e la discesa di profilo del server
sotto pressione vera), prova del cambio di device del server dopo un guasto vero
(voce 24) su una macchina con più GPU, prova della convivenza con un'altra applicazione sulla
stessa GPU (scenari in [convivenza_applicazioni.md](./convivenza_applicazioni.md),
sezione 5), prova di `termico.py` su una GPU NVIDIA reale e su Windows e del
guardiano della guida sugli agenti con una GPU vera sotto carico, test su
CUDA/Vulkan e su Windows (il ramo Windows della
memoria di sistema è stato solo controllato nella sintassi), taratura delle
soglie di `--profile auto` e dei preset della sezione 3.6, e il livello di
astrazione `ImageRuntime` nell'applicazione ospite (fuori da questo
repository).
