# Convivenza con altre applicazioni

Questa guida è per chi usa `sd-cli`, `sd-server` o la libreria **accanto ad altre
applicazioni che chiedono le stesse risorse**: una chat con un modello di
linguaggio locale (llama.cpp, LM Studio, un modello nel browser con WebGPU...),
un altro generatore, un editor video, un gioco. Non dipende da nessuna
applicazione in particolare. Dice cosa il motore condivide con gli altri, cosa
fa da solo per non pestare i piedi, cosa invece non fa, e come configurarlo (o
avviarlo da un programma ospite) perché la convivenza funzioni.

Per installare il motore e collegarlo a un progetto vedi la
[guida di integrazione](./integrazione_progetti.md); per le opzioni di memoria
vedi [performance.md](./performance.md) e [backend.md](./backend.md).

## In breve

- Il motore è un processo a sé e **non sa nulla delle altre applicazioni**. Con
  loro condivide solo risorse fisiche: memoria video, tempo di calcolo della
  GPU, RAM, core della CPU, disco e una porta TCP.
- All'avvio **legge la memoria libera** e si adatta (`--profile auto`); mentre
  lavora **tiene d'occhio la pressione** sulla memoria (`--memory-guard`); se la
  memoria finisce `sd-server` **non si ferma** e riprova con una configurazione
  più leggera.
- **Non** restituisce la memoria video quando è inattivo, **non** cambia profilo
  se l'altra applicazione parte dopo di lui e **non** limita l'uso di calcolo
  della GPU. Per restituire la memoria video si chiude il processo.
- Tre modi di convivere, dal più sicuro al più delicato:
  [dispositivi separati](#31-dispositivi-separati),
  [a turni](#32-a-turni-stessa-gpu-mai-insieme),
  [insieme con un budget](#33-insieme-stessa-gpu-nello-stesso-momento).
- Nessuna prova è stata fatta con una GPU reale: la sezione
  [6](#6-cosa-è-verificato-e-cosa-no) dice cosa è stato controllato e cosa no.

## 1. Cosa si condivide

| Risorsa | Come la usa il motore | Se finisce | Come si limita |
|---|---|---|---|
| Memoria video (VRAM) | pesi del modello, spazio di calcolo e cache dei grafi | l'allocazione fallisce (`out_of_memory`) e `sd-server` scende di profilo; con alcuni driver NVIDIA su Windows non fallisce ma rallenta (vedi [3.3](#33-insieme-stessa-gpu-nello-stesso-momento)) | `--profile`, `--memory-guard`, `--max-vram`, `--auto-fit` |
| Tempo di calcolo della GPU | una generazione alla volta, con tutta la GPU | due programmi sulla stessa GPU si rallentano a vicenda; il motore non ha un limite di utilizzo | non sovrapporre i lavori, o usare GPU diverse |
| RAM di sistema | pesi tenuti in RAM (`--offload-to-cpu`, backend CPU) e buffer di lavoro | swap, o terminazione di un processo da parte del sistema | `--params-backend disk`, `--memory-guard`, `--profile lite` |
| Core della CPU | thread di calcolo: tutti i core fisici se non indichi altro | una chat su CPU e il motore si contendono i core | `--threads N` |
| Disco | lettura del modello all'avvio e, con `--params-backend disk`, a ogni richiesta | avvio e richieste lenti | modello su un SSD |
| Porta TCP | `sd-server` ascolta su `127.0.0.1:1234` | vedi [4.1](#41-porta-e-istanze) | `--listen-port` |

La memoria libera che il motore legge è quella che il driver riporta per
l'intero dispositivo, quindi **comprende già ciò che occupano le altre
applicazioni**. È per questo che `--profile auto`, `--auto-fit` e
`--memory-guard` "vedono" una chat già caricata senza saperne nulla. Sulla CPU
la guardia misura la RAM dell'intera macchina, non quella del solo processo (e
non il limite di un cgroup, se sei in un container: vedi
[performance.md](./performance.md#keep-device-memory-below-a-pressure-threshold---memory-guard)).

## 2. Cosa fa già il motore, e cosa no

| Meccanismo | Quando agisce | Dettagli |
|---|---|---|
| `--profile auto` | **una volta**, all'avvio del processo: confronta modello più riserva (1 GiB più un quinto dei pesi) con la memoria libera della GPU discreta più libera (o di più GPU dello stesso backend sommate), o con la RAM se non c'è una GPU; sceglie `max`, `balanced` o `lite` e lo scrive nel log (`profile auto -> ...`) | [performance.md](./performance.md#pick-a-resource-profile---profile) |
| `--auto-fit` | al caricamento del modello: decide dove mettere ogni componente in base alla memoria libera di quel momento | [backend.md](./backend.md#automatic-placement---auto-fit-onoff) |
| `--memory-guard N` | durante la generazione: oltre N% di uso (il più alto fra memoria riportata dal dispositivo e budget `--max-vram`) libera i pesi non fissati e sospende il prefetch | [performance.md](./performance.md#keep-device-memory-below-a-pressure-threshold---memory-guard) |
| `--max-vram` | tetto gestito per dispositivo (`cuda0=6`, `vulkan0=2`, oppure un valore solo); non è un limite fisico | [backend.md](./backend.md#syntax) |
| Scala di recupero di `sd-server` | dopo un errore di memoria: scende di un livello lungo `max > balanced > lite > lite-disk > lite-hybrid > lite-cpu` (fino a `lite-disk` il calcolo resta sulle GPU), ricarica il modello e ripete la richiesta; dopo un errore di dispositivo riprova lo stesso, poi prova le altre GPU e solo alla fine la CPU; risponde `503` con `Retry-After` solo se non resta nulla da provare; risale dopo `--degradation-cooldown` secondi | [api.md](../examples/server/api.md#failures-and-recovery) |
| `--multi-gpu` | con più GPU discrete dello stesso backend usa la seconda solo se serve (o sempre, con `on`) | [backend.md](./backend.md#several-gpus---multi-gpu-autoonoff) |
| Stato leggibile dall'esterno | `sd-cli --list-devices`; `GET /sdcpp/v1/health`; `GET /sdcpp/v1/resources` | [api.md](../examples/server/api.md) |

Cosa il motore **non** fa:

1. **Non restituisce la memoria da inattivo.** `sd-server` tiene il modello
   caricato tra una richiesta e l'altra e non ha un comando per scaricarlo
   (nessun `unload`, nessun timeout di inattività). L'unico modo di liberare la
   memoria video è terminare il processo; `sd-cli` la restituisce da solo alla
   fine di ogni esecuzione.
2. **Non rivaluta `--profile auto` mentre vive.** Se l'altra applicazione parte
   dopo, il motore se ne accorge solo quando la guardia vede la pressione
   salire oltre la soglia, o quando una richiesta va in errore e il server
   scende di profilo.
3. **Non limita il calcolo della GPU:** non esiste un'opzione per cedere una
   percentuale di tempo all'altra applicazione.
4. **`--profile auto` guarda la GPU discreta con più memoria libera**, anche se
   hai fissato il motore su un'altra con `--backend`. Se lo fissi su un
   dispositivo diverso dal più libero, indica un profilo esplicito.
5. **Non impedisce a una seconda istanza di usare la stessa porta** (su Linux,
   vedi [4.1](#41-porta-e-istanze)).

## 3. Ricette

| Situazione | Ricetta |
|---|---|
| L'altra applicazione può usare un altro dispositivo (seconda GPU, GPU integrata, CPU) | [3.1 dispositivi separati](#31-dispositivi-separati) |
| Una sola GPU e i due modelli non ci stanno insieme con margine | [3.2 a turni](#32-a-turni-stessa-gpu-mai-insieme) |
| Una sola GPU e i due modelli ci stanno entrambi con margine | [3.3 insieme](#33-insieme-stessa-gpu-nello-stesso-momento) |
| L'altra applicazione calcola sulla CPU | [3.4 la CPU](#34-laltra-applicazione-calcola-sulla-cpu) |

### 3.1 Dispositivi separati

È la configurazione più sicura: se i dispositivi sono davvero distinti, le due
applicazioni non si contendono né la memoria video né il tempo di calcolo della
GPU. Una GPU integrata, però, usa la RAM di sistema.

1. Elenca i dispositivi. Una riga per dispositivo, con campi separati da
   tabulazioni: nome, tipo, descrizione, memoria libera e totale. I nomi sono
   quelli accettati da `--backend`.

   ```shell
   sd-cli --list-devices
   ```

   ```
   CUDA0   gpu   NVIDIA GeForce RTX ...   5.20 GiB free / 6.00 GiB total
   ```

   (esempio: i valori dipendono dalla macchina; il tipo è `cpu`, `gpu`, `igpu`,
   `accel` o `unknown`).

2. Fissa il motore su un dispositivo e dagli una porta tutta sua:

   ```shell
   sd-server -m model.safetensors --backend cuda0 --listen-port 18234
   ```

3. Fai usare un altro dispositivo all'altra applicazione, con le sue opzioni.
   Per limitare i dispositivi che il motore vede, senza usare `--backend`,
   puoi impostare le variabili d'ambiente dei backend prima di avviarlo:
   `CUDA_VISIBLE_DEVICES=1` gli fa vedere solo la seconda GPU CUDA, e
   `GGML_VK_VISIBLE_DEVICES=1` fa usare al backend Vulkan solo il dispositivo
   con quell'indice di enumerazione (si possono dare più indici separati da
   virgola; un indice inesistente fa fallire l'avvio con un errore). In
   PowerShell: `$env:CUDA_VISIBLE_DEVICES="1"`.

Da sapere:

- **Senza `--backend`** la scelta spetta all'auto-fit, che con più GPU discrete
  preferisce quella con più memoria libera. Il dispositivo predefinito, quando
  non c'è un piano, è la prima GPU discreta, poi la GPU integrata, poi la CPU.
  Su un portatile con GPU integrata e discreta l'ordine dei dispositivi Vulkan
  non è stato verificato: leggi sempre `--list-devices` prima di fissare un
  nome.
- **Un `--backend` scritto a mano vince sul profilo.** Con
  `--profile lite --backend cuda0` il profilo imposta guardia, VAE a tasselli e
  CFG non raggruppato; `--multi-gpu on` non sposta nulla, perché un modulo
  assegnato con `--backend` non viene mai spostato. L'encoder di testo resta su
  `cuda0`, come con ogni profilo che si può scegliere. Se vuoi la CPU, scrivilo
  tu: `--backend all=cuda0,te=cpu`.
- **Con più GPU** e nessun `--backend`, il pianificatore può usare la seconda
  GPU del motore se il modello non sta nella prima. Per tenere il motore su una
  GPU sola: `--multi-gpu off`. Un modulo assegnato con `--backend` non viene mai
  spostato.
- **CUDA e Vulkan nella stessa cartella** (per esempio per avere anche la GPU
  AMD; la release `master-*` ha già lo zip `...-bin-win-cuda12-vulkan-x64.zip`,
  vedi [integrazione_progetti.md](integrazione_progetti.md), sezione 2.9): la
  stessa scheda NVIDIA compare due volte, come `CUDA0` e `Vulkan0`. Il
  motore la riconosce dall'indirizzo PCI che entrambi i backend riportano, la
  conta una volta e usa CUDA; nel piano la riga di `Vulkan0` dice
  `same GPU as CUDA0, not used automatically`. La GPU integrata (`Vulkan1` su un
  portatile AMD) non viene scelta finché una scheda ha memoria libera, perché la
  sua «memoria» è la RAM di sistema, e non ospita mai i pesi. Si possono
  comunque nominare con `--backend`, ma sulla stessa scheda Vulkan può essere
  molto più lento di CUDA: se lo nomini (o lo fa un programma al posto tuo) il
  log lo dice con un avviso. Non aggiunge velocità: il motore non divide un
  modulo fra backend diversi. Log, tempi misurati e cosa fare:
  [cuda_e_vulkan_stessa_scheda.md](./cuda_e_vulkan_stessa_scheda.md).
- **La GPU integrata** usa la RAM di sistema: per `--profile auto` non conta
  come memoria veloce (contano solo GPU discrete e acceleratori), e per una
  chat sulla GPU integrata la risorsa contesa è la RAM, non la memoria video.

### 3.2 A turni (stessa GPU, mai insieme)

1. L'altra applicazione scarica il suo modello (o si chiude).
2. Avvia il motore. `--profile auto` legge la memoria ora libera e sceglie di
   conseguenza.
3. Genera.
4. Termina il processo `sd-server`. La memoria video torna libera quando il
   processo finisce, non prima.
5. L'altra applicazione ricarica il suo modello.

Da sapere:

- Il motore non può chiedere all'altra applicazione di scaricare il suo modello,
  né il contrario: serve qualcuno che coordini (l'utente o il programma
  ospite).
- `POST /sdcpp/v1/jobs/{id}/cancel` ferma un job **in coda** (subito) e uno **in
  esecuzione** (al passo di campionamento successivo): senza `?wait=` risposta
  subito `200` (il job può risultare ancora `generating`), con `?wait=` risposta
  `200` con il job annullato, oppure `202` se sta ancora finendo (vedi
  [integrazione_progetti.md](./integrazione_progetti.md), sezione 3.5). Il modello
  resta in memoria e il server non scende di profilo, quindi è il modo giusto per
  fare posto all'altra applicazione senza spegnere il server. Le richieste
  sincrone (`/sdapi`, `/v1/images`) non sono job e non si fermano; un server più
  vecchio risponde `409` a un job già partito.
- Per richieste isolate `sd-cli` è la scelta più semplice: carica il modello,
  genera e restituisce tutto uscendo. Ogni esecuzione paga il caricamento del
  modello.

### 3.3 Insieme (stessa GPU, nello stesso momento)

1. Avvia **prima l'altra applicazione** e aspetta che abbia finito di caricare
   il suo modello: il motore misura la memoria libera quando parte.
2. Avvia il motore e leggi la riga `profile auto -> ...` del log, che dice cosa
   ha visto e cosa ha deciso:

   ```shell
   sd-server -m model.safetensors --profile auto --listen-port 18234
   ```

3. Se vuoi un margine per la crescita dell'altra applicazione (per esempio una
   chat la cui cache cresce con il contesto), aggiungi `--max-vram -1`, che
   riserva circa 1 GiB della memoria libera al momento dell'avvio, oppure un
   budget fisso per dispositivo, come `--max-vram cuda0=3`.
4. Tieni accesa la guardia: `--memory-guard 85`. Il profilo `balanced` la imposta
   a 90 e `lite` a 85, quindi con `--profile auto` c'è già a meno che il motore
   scelga `max`, che non la imposta.
5. Se non sei sicuro, parti da `--profile lite`: VAE a tasselli, CFG non
   raggruppato, tutte le GPU del backend; il calcolo resta sulle GPU. Più lento,
   ma con un picco di memoria più basso di `max`. Per liberare ancora memoria
   video a costo di core e tempo aggiungi `--clip-on-cpu` (l'encoder di testo
   sulla CPU). Con `--backend` esplicito vedi la nota in
   [3.1](#31-dispositivi-separati).

Cosa aspettarsi:

- Se l'altra applicazione cresce dopo, la memoria libera cala. La guardia libera
  i pesi del motore quando l'uso supera la soglia, ma non può ridurre quelli
  dell'altra applicazione.
- Quando le due applicazioni calcolano nello stesso momento i tempi si
  allungano per entrambe.
- Se la memoria finisce, `sd-server` scende di profilo e ripete la richiesta
  (che riesce, più lenta; fino a `lite-disk` il calcolo resta sulla GPU),
  oppure risponde `503` con `Retry-After`.
- `--max-vram` non è un limite fisico: i contesti del driver e le allocazioni
  fuori dai gestori di memoria del motore non ne fanno parte.
- **Driver NVIDIA su Windows.** Dalla versione 536.40 un programma CUDA che
  supera la memoria video libera può continuare usando la RAM di sistema invece
  di fallire, a velocità più bassa; dalla 546.01 si può disattivare per singolo
  programma (Pannello di controllo NVIDIA, Gestisci impostazioni 3D,
  Impostazioni programma, "CUDA - Sysmem Fallback Policy", "Prefer No Sysmem
  Fallback"; il programma va riavviato) ([NVIDIA](#fonti)). Con
  l'impostazione "Driver Default" un'allocazione troppo grande può diventare un
  rallentamento invece di un errore: la scala di recupero di `sd-server`, che
  reagisce all'errore, non scatterebbe. Come il motore si comporti in quel caso
  non è stato provato.

### 3.4 L'altra applicazione calcola sulla CPU

Il conflitto è su RAM e core, non sulla memoria video. Metti il motore sulla GPU
e limita i suoi thread con `--threads N` (se omesso, o se `N` è 0 o negativo,
usa tutti i core fisici). Con `--profile lite` il calcolo resta sulla GPU. La CPU
entra solo se il server scende a `lite-hybrid` (l'encoder di testo: un lavoro
breve a inizio richiesta, ma toglie core alla chat in quel momento) o a
`lite-cpu` (tutto), oppure se la chiedi tu con `--clip-on-cpu`.

## 4. Contratto minimo per chi avvia il motore da un altro programma

### 4.1 Porta e istanze

- **Indica sempre `--listen-port`** e scegli una porta libera. La porta
  predefinita del motore, 1234, è anche quella del server locale di
  LM Studio ([LM Studio](#fonti)).
- **Non contare su un errore di "porta occupata"** per accorgerti che c'è già
  un'istanza. Su Linux due `sd-server` hanno ascoltato sulla stessa porta senza
  alcun errore e le richieste si sono divise fra i due (12 e 12 in una prova).
  La causa è `SO_REUSEPORT`, che la libreria HTTP inclusa in `thirdparty/`
  imposta di default; il server non lo cambia. I job asincroni creati su
  un'istanza non esistono sull'altra. Su Windows il comportamento non è stato
  verificato.
- **Controlla prima tu** con `GET /sdcpp/v1/health` sulla porta che vuoi usare:
  `200` con un JSON che ha `status` e `version` significa che c'è già un server
  del motore; `401` che c'è un server che chiede una chiave API; un `404` di
  solito indica un altro server; un errore di connessione, che la porta è
  libera.
- **Una porta per istanza.** Per usare più GPU in parallelo si avviano più
  processi, ciascuno con il proprio `--backend cudaN` e la propria porta (vedi
  [backend.md](./backend.md#several-gpus---multi-gpu-autoonoff)).
- `--listen-ip` è `127.0.0.1` per default: lascialo così se serve solo in
  locale. Se una pagina web chiama il motore direttamente, imposta
  `--cors-origins` con l'origine della pagina (di default il server accetta
  qualsiasi origine) e `--api-key`. Vedi
  [sicurezza](./integrazione_progetti.md#5-sicurezza-cosa-sapere-e-cosa-fare).

### 4.2 Vita del processo figlio

- Chi avvia il motore deve anche chiuderlo, **anche quando esce in modo
  anomalo**. Un `sd-server` rimasto vivo tiene il modello e la memoria video; alla
  prossima partenza la porta risulta già in uso oppure, su Linux, la nuova
  istanza si affianca alla vecchia (vedi sopra). Su Windows un processo figlio non termina con il genitore a meno che il
  genitore non lo metta in un oggetto Job con la chiusura automatica; su Linux
  si può usare un gruppo di processi o `PR_SET_PDEATHSIG`.
- Per sapere quando è pronto, sonda `GET /sdcpp/v1/health` fino a che risponde
  `ok` o `degraded` (il server può generare). `recovering` vuol dire che sta
  caricando il modello. La rotta risponde subito anche mentre il server genera.

### 4.3 Prima di un lavoro pesante

- Prima dell'avvio usa `sd-cli --list-devices`; a server acceso
  `GET /sdcpp/v1/resources` dà la memoria libera dei dispositivi e la RAM della
  macchina. Servono a decidere se partire, rimandare o chiedere all'utente di
  chiudere qualcosa.
- La risposta non dice quanta memoria tiene il server su un dispositivo: per
  saperlo confronta la memoria usata prima e dopo l'avvio.
- Una pagina web (WebGPU) non ha modo di leggere la memoria video libera: fra
  le proprietà di `GPUAdapterInfo` (fornitore, architettura, dispositivo,
  descrizione e poco altro) non c'è la memoria ([MDN](#fonti)). La misura spetta
  al programma ospite, o al motore con `/resources`.

### 4.4 Risposte da gestire

| Risposta | Significato | Cosa fare |
|---|---|---|
| `503` `out_of_memory` | memoria esaurita anche con la configurazione più leggera | aspetta `Retry-After` secondi, libera memoria nell'altra applicazione o invia una richiesta più piccola |
| `503` `unavailable` | il server sta ricaricando il modello dopo un errore (`cause` dice quale) | riprova dopo `Retry-After` |
| `503` `busy` | il modello era in uso | riprova dopo `Retry-After` |
| `429` | coda piena (`queue.max` in `/health`, 64) | riprova dopo `Retry-After` (10 secondi) |
| `202` su `cancel?wait=…` | lo stop è stato chiesto e dopo l'attesa il job sta ancora finendo (`cancel_requested: true`); senza `?wait=` la risposta è invece subito `200`, con il job ancora `generating` | ripeti `cancel` o interroga il job finché diventa `cancelled` |
| `409` su `cancel` | server più vecchio: il job sta generando e non si può fermare | aspetta o termina il processo |
| `401` | manca la chiave API | `Authorization: Bearer <chiave>` oppure `X-API-Key` |
| `400` `invalid_argument` | la richiesta non è generabile così | non ripeterla uguale |
| `200` dopo una discesa di profilo o un cambio di device | servita con un profilo più leggero o su un altro device | `/health` mostra `status: degraded`, il `profile` e il `device` in uso (e `cpu_modules` se la CPU fa parte del lavoro); dopo `--degradation-cooldown` il server prova a risalire |

Il corpo di un `503` è `{"error": "...", "code": "...", "retry_after_seconds": N}`
(vedi [api.md](../examples/server/api.md#failures-and-recovery)). Un `503` che
arriva dopo che il server era già sceso fino al livello più leggero può voler
dire che un'altra applicazione ha preso la memoria: confronta `/resources` con
i valori che avevi prima.

### 4.5 Il log

Il log è scritto per le persone, non è un'interfaccia stabile: per le decisioni
automatiche usa `/health` e `/resources`. Conserva però l'output del processo
figlio e mostralo all'utente quando qualcosa va storto. Le righe
`profile auto -> ...`, `auto-fit plan:`, `memory guard: ...` e
`trying profile ...` dicono cosa il motore ha visto e perché ha ridotto
qualcosa. Con `-v` il log è più dettagliato.

Se il programma che avvia il motore è un agente che decide con il proprio
giudizio (un assistente, un modello di linguaggio, un nodo principale), le regole
di questa sezione valgono anche per lui. Come si comporta, perché non deve
sovraccaricare né farsi sovraccaricare e come si ferma un job quando la GPU è
troppo calda sono in [agenti_indipendenti.md](./agenti_indipendenti.md).

## 5. Provarlo sul proprio PC

È la prova che manca (sezione [6](#6-cosa-è-verificato-e-cosa-no)). Servono il
motore, l'altra applicazione e un modello per ciascuno.

1. Registra la memoria video in una finestra a parte. Su Windows con una GPU
   NVIDIA, in PowerShell:

   ```powershell
   nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader -l 2 | ForEach-Object { $_; Add-Content vram.csv $_ }
   ```

   Su Linux lo stesso comando, con `| tee -a vram.csv` al posto del
   `ForEach-Object`. Senza una GPU NVIDIA, Gestione attività (Prestazioni, GPU)
   mostra la memoria dedicata e quella condivisa.

   Con una GPU NVIDIA c'è anche un programma pronto,
   [`termico.py`](../scripts/diagnostica_termica/README.md), che registra la
   memoria video insieme a temperatura, potenza, frequenza e motivi di
   throttling, e con `--motore` affianca lo stato che `sd-server` dichiara in
   `/health` e `/resources`. La sua sessione guidata ha già le fasi di questa
   prova (solo il motore, solo l'altra applicazione, tutte e due insieme) e
   alla fine scrive un riepilogo e un file JSON da allegare, se qualcosa va
   storto, insieme ai log.

2. Ripeti gli scenari e annota i valori:

   | # | Cosa fai | Cosa annoti |
   |---|---|---|
   | A | Solo l'altra applicazione, con il modello caricato: a riposo e mentre genera | `memory.used` nei due casi, tempo di una risposta |
   | B | Solo il motore con `--profile auto`: a riposo e mentre genera | `memory.used`, riga `profile auto -> ...`, tempo di un'immagine |
   | C | Prima A, poi avvia il motore con `--profile auto` | profilo scelto; `memory.used` rispetto ad A più B |
   | D | Con C acceso, genera un'immagine mentre l'altra applicazione genera | tempi di entrambe; se una delle due fallisce o se il driver si reimposta |
   | E | Chiudi il motore | `memory.used` torna al valore di A? |
   | F | (facoltativo) ripeti D con `--profile lite` o `--max-vram -1` | quanta memoria si guadagna e quanto tempo si perde |

3. La prova è riuscita se nessuna delle due applicazioni termina o si blocca; se
   il motore finisce la generazione (anche più lenta) oppure risponde `503` con
   `Retry-After`; se dopo la chiusura del motore la memoria torna ai valori di
   A; e se il driver non si reimposta (nessuno schermo nero, nessun errore di
   dispositivo perso).
4. Se qualcosa va storto salva `vram.csv`, il log del motore (con `-v`), il JSON
   di `GET /sdcpp/v1/health` e di `GET /sdcpp/v1/resources`, e scrivi modello e
   scheda grafica usati.

## 6. Cosa è verificato e cosa no

Verificato, su Linux, **solo CPU**, con un modello sintetico:

- I test dei profili, del pianificatore multi-GPU (con GPU simulate) e della
  scala di recupero del server passano: non servono né una GPU né un modello.
- Un programma ospite ha avviato `sd-server` come processo figlio con
  `--profile auto --memory-guard 90`, ne ha letto `/sdcpp/v1/health` e ha
  ottenuto un'immagine PNG valida.
- Con un carico di RAM esterno e la soglia abbassata, la guardia segnala la
  pressione (`memory guard: ...`) e la generazione termina.
- La forma delle risposte di `/health`, di `/resources` e del `503`.
- `--profile lite` non sposta l'encoder di testo sulla CPU, con o senza un
  `--backend` scritto a mano: lo fa solo `lite-hybrid` (eseguito con la logica
  dei profili, `test/profile_test.cpp`).
- La scala dei device del server (nuova prova sullo stesso device, altre GPU, CPU
  per ultima, risalita) con elenco dei device e orologio simulati
  (`test/server_recovery_test.cpp`).
- Due `sd-server` sulla stessa porta: entrambi in ascolto, richieste divise.

**Non** verificato:

- Qualunque prova con una GPU reale (CUDA o Vulkan): consumo di memoria video,
  effetto di `--max-vram` e della guardia, tempi.
- La convivenza con una chat reale (llama.cpp, LM Studio, WebGPU).
- Windows: condivisione della porta, chiusura del processo figlio.
- L'ordine dei dispositivi Vulkan su un portatile con GPU integrata e discreta.
- La corrispondenza degli indirizzi PCI fra CUDA e Vulkan su altre macchine
  reali: su un portatile con RTX 3060 e GPU integrata AMD il piano scrive
  `same GPU as CUDA0` per `Vulkan0` e non per la GPU integrata (log di un
  utente, vedi [cuda_e_vulkan_stessa_scheda.md](./cuda_e_vulkan_stessa_scheda.md));
  per il resto la logica è provata con dispositivi simulati.
- Il comportamento del motore con il ripiego sulla RAM dei driver NVIDIA.
- Il passaggio a un'altra GPU dopo un guasto vero del driver (con due schede, o
  con la GPU integrata come ultima riserva prima della CPU).
- Il costo del ricaricamento di un modello reale.
- Quanto si scalda la macchina con il motore e l'altra applicazione accesi per
  molti minuti: lo misura [`termico.py`](../scripts/diagnostica_termica/README.md),
  che finora è stato provato solo con una GPU simulata.

## Fonti

- NVIDIA, [System Memory Fallback for Stable Diffusion](https://nvidia.custhelp.com/app/answers/detail/a_id/5490/~/system-memory-fallback-for-stable-diffusion):
  versione 536.40, rallentamento invece di errore, impostazione per programma da
  546.01.
- LM Studio, [Get up and running with the LM Studio API](https://lmstudio.ai/docs/developer/rest/quickstart):
  server locale su `http://localhost:1234` per default.
- MDN, [GPUAdapterInfo](https://developer.mozilla.org/docs/Web/API/GPUAdapterInfo):
  proprietà esposte da un adattatore WebGPU (nessuna riguarda la memoria).
