# Diagnostica termica della GPU

Questa cartella contiene due strumenti, scritti solo con la libreria standard di Python, per capire **come il tuo
PC regge il motore**: quanto si scalda la GPU, quanta potenza usa, a che frequenza lavora e se il driver la sta
frenando. Si usa a riposo, con il motore acceso, con una chat accanto, con tutto insieme.

| File | Cosa fa |
|---|---|
| [`termico.py`](./termico.py) | Misura temperatura, potenza, frequenza e motivi di throttling con `nvidia-smi`, a fasi o in continuo; può affiancare lo stato del motore (`sd-server`) sulla stessa linea del tempo; scrive CSV, un riepilogo in JSON leggibile da altri programmi e un testo; confronta più prove. |
| [`carico_motore.py`](./carico_motore.py) | Fa generare `sd-server` di continuo con sempre lo stesso lavoro, per avere un carico ripetibile durante la prova. |

## In breve

- **Solo lettura, salvo `--segnala`.** `termico.py` non cambia nessuna impostazione della GPU, non avvia carichi, non
  ferma programmi. Esegue `nvidia-smi` con interrogazioni di sola lettura e, se glielo chiedi, fa richieste `GET` al
  motore (sezione [10](#10-cosa-fa-sul-tuo-pc-esattamente)). Solo con `osserva --segnala` manda anche al motore la fascia
  della GPU, e il motore le toglie lavoro (sezione [7.1](#71-segnalare-la-fascia-al-motore---segnala)).
- **Non è un test di velocità né uno stress test.** Il carico lo generi tu, con il motore vero; lo strumento guarda
  cosa succede mentre lavora.
- **Le fasce di temperatura sono indicative**, non garanzie del produttore: partono dai limiti che il driver dice
  di avere, se li dice (sezione [6.4](#64-le-fasce-di-temperatura)). Lo stesso vale per i giudizi
  ("sostenibile", "al limite"...).
- Funziona con **GPU NVIDIA** (serve `nvidia-smi`, che c'è con il driver). Con AMD, Intel o Apple non parte.
- **Mai provato su una GPU reale** da chi l'ha scritto: è stato collaudato con una GPU finta e con un vero
  `sd-server` su CPU. La sezione [11](#11-cosa-è-verificato-e-cosa-no) dice cosa è stato controllato e cosa no:
  se il tuo driver risponde in modo diverso, il programma dovrebbe dirlo invece di inventare, e un tuo riscontro
  è utile.

I cinque comandi:

| Comando | Quando usarlo |
|---|---|
| `guidata` | La prova completa a fasi (riposo, motore, chat, insieme, raffreddamento). È il comando predefinito. |
| `osserva` | Registrare per un tempo o finché non premi Ctrl+C, senza fasi; può tenere aggiornato un file con l'ultima lettura per altri programmi. |
| `stato` | Una sola lettura in JSON; il codice di uscita dice la fascia. Per script e agenti. |
| `riepilogo` | Rifare il riepilogo da un CSV, anche di una prova interrotta (per esempio perché il PC si è spento). |
| `confronta` | Mettere a confronto due o più prove. |

## 1. A cosa serve e a cosa no

Serve a rispondere, con numeri, a domande come:

- La GPU del mio portatile regge il motore acceso a lungo, o dopo qualche minuto si frena?
- Quanta velocità perde quando si scalda (la frequenza media cala)?
- Posso tenere acceso il motore e una chat insieme senza farla scaldare troppo?
- Il portatile sta meglio con l'alimentatore collegato? Dopo aver pulito le ventole? Con un supporto che lo alza?
- Mentre faceva caldo il motore ha cambiato profilo, ha avuto errori, ha finito la memoria video?

**Non** serve a:

- Misurare quanto è veloce la GPU (per quello conta il tempo per immagine; `carico_motore.py` lo stampa).
- Proteggere l'hardware. La GPU si protegge da sola (il driver la rallenta e, in casi estremi, la spegne); lo
  strumento osserva e avvisa a schermo, non decide niente al posto tuo e non fa niente da solo.
- Dare una garanzia: i numeri dicono come si è comportato **quel PC, in quella stanza, quel giorno**.

## 2. Cosa serve

- Windows o Linux con una GPU NVIDIA e il suo driver (porta con sé `nvidia-smi`).
- Python 3.8 o più recente. Nessun pacchetto da installare. Se vedi `(base)` nel prompt hai già conda e quindi
  Python; su Windows può servire `py -3` al posto di `python`.
- Per le fasi con carico: il motore (`sd-server` o `sd-cli`) e un modello.

Non serve nessuna installazione: basta il file. Da una finestra di PowerShell (Windows) o di un terminale (Linux),
nella cartella del progetto:

```powershell
python scripts\diagnostica_termica\termico.py --versione
python scripts\diagnostica_termica\termico.py --help
```

Negli esempi che seguono si scrive `python scripts/diagnostica_termica/termico.py` con la barra normale; su
Windows va bene anche così in PowerShell.

## 3. Prima prova (due minuti)

Prima della prova lunga controlla che lo strumento veda la tua GPU:

```shell
python scripts/diagnostica_termica/termico.py stato
```

Risponde con un JSON. Se vedi il nome della tua scheda, la temperatura e `"fascia": {"nome": "normale", ...}`, è
tutto a posto. I campi sono spiegati nella sezione [9](#9-per-altri-programmi-json-csv-e-stato). Il codice di
uscita è `0` per la fascia normale, `1` per attenzione, `2` per critica, `3` se non è riuscito a leggere la GPU.

Se compare `nvidia-smi non trovato`, vai alla sezione [12](#12-problemi-comuni).

## 4. La prova completa, passo per passo

### 4.1 Prima di cominciare

Il confronto fra prove vale solo se le condizioni sono le stesse. Prima di ogni prova:

- scegli **dove** la fai (stessa stanza, stessa superficie: un portatile sul letto si scalda più che su un tavolo) e
  segnalo nel nome della prova (`--nome`);
- **collega l'alimentatore** originale, a meno che tu non voglia provare proprio la batteria;
- scegli il **piano di alimentazione** di Windows e lascialo uguale (per esempio "Prestazioni elevate");
- chiudi i programmi che usano la GPU e che non fanno parte della prova (browser con video, giochi, altri
  generatori);
- parti da **PC freddo o comunque a riposo**: la fase RIPOSO serve proprio a vedere da dove si parte.

### 4.2 Le cinque fasi

```shell
python scripts/diagnostica_termica/termico.py guidata --nome portatile-alimentatore
```

Il programma ti guida con messaggi a schermo, una fase alla volta. Le fasi sono queste:

| Fase | Cosa fai | Come finisce |
|---|---|---|
| RIPOSO | Niente: PC fermo, né motore né chat. | Da sola, dopo 120 secondi (`--auto-secondi`). **Un Invio premuto per sbaglio viene ignorato.** |
| MOTORE | Avvii il motore e lo fai generare di continuo (sezione [4.3](#43-il-carico-del-motore)). La chat resta spenta. | Premi Invio per iniziare a registrare e Invio di nuovo per fermare. Meglio **almeno 3 minuti**. |
| CHAT | Fermi il motore e usi soltanto la chat, facendola rispondere di continuo. | Come MOTORE. |
| INSIEME | Motore e chat lavorano nello stesso momento. | Come MOTORE. |
| RAFFREDDAMENTO | Fermi tutto e lasci il PC fermo. | Da sola, dopo 120 secondi. |

Non servono tutte: `--fasi motore,raffreddamento` fa solo quelle due (l'ordine è quello che scrivi; i nomi validi
sono `riposo`, `motore`, `chat`, `insieme`, `raffreddamento`).

Ogni 10 secondi vedi una riga con lo stato, per esempio (numeri inventati):

```text
  [MOTORE 2:10] 74 C  76.8 W  1596 MHz  97 %  4310 MiB  limite_potenza  | motore ok/max occupato
```

Quando la temperatura entra nella fascia di attenzione (o in quella critica) compare un avviso, ripetuto al più
ogni 30 secondi; se il driver sta già rallentando la GPU per il caldo lo dice. Con **Ctrl+C**, in qualsiasi
momento, la prova si chiude e il riepilogo di quanto raccolto viene scritto lo stesso.

Perché 3 minuti? Sotto i 3 minuti non si può dire se la temperatura si stabilizza o continua a salire: il programma
chiama quelle fasi "troppo corte" e non le giudica. Un portatile può avere bisogno di 5-10 minuti per arrivare al
suo equilibrio: se a fine fase la temperatura sta ancora salendo, il giudizio è "da verificare" e conviene
ripetere la fase più a lungo.

### 4.3 Il carico del motore

La fase MOTORE ha senso se il motore lavora **di continuo e sempre allo stesso modo**. Se hai già un tuo modo
(un'applicazione che genera di continuo), usalo. Altrimenti:

1. Avvia il server in un'altra finestra, con la stessa configurazione che userai davvero:

   ```shell
   sd-server -m modello.safetensors --listen-port 1234
   ```

2. Quando `termico.py` ti dice di avviare il carico, lancia in un'altra finestra:

   ```shell
   python scripts/diagnostica_termica/carico_motore.py --durata 300
   ```

   Manda a `sd-server` un job dopo l'altro (sempre lo stesso: stessa immagine, stesso seme, stessi passi),
   stampa il tempo di ognuno e, allo scadere dei secondi o con Ctrl+C, **ferma il job in corso** con
   `POST /sdcpp/v1/jobs/{id}/cancel`. Opzioni: `--url`, `--chiave-api`, `--larghezza`, `--altezza`, `--passi`,
   `--seme`, `--prompt`. Il tempo per immagine dipende dal modello e dalla GPU: per confrontare due prove, o due
   PC, usa lo stesso modello e gli stessi valori.

3. Premi Invio in `termico.py` poco dopo l'avvio del carico, e di nuovo Invio un po' prima della sua fine.

Con `sd-cli` si può fare qualcosa di simile con `-b N`, che genera N immagini di fila con il modello caricato una
sola volta; per fermarlo però si chiude il programma, mentre `carico_motore.py` dura quanto dici e ferma il job in
corso in modo pulito. In ogni caso non lanciare `sd-cli` in un ciclo: caricherebbe ogni volta il modello dal disco,
e quello non è il carico che ti interessa.

### 4.4 Con la chat o un'altra applicazione

Le fasi CHAT e INSIEME presuppongono un'altra applicazione che usa la GPU (una chat con un modello locale, per
esempio). `termico.py` non la conosce e non la controlla: la fai lavorare tu, e lui misura la GPU. Per la fase
INSIEME conviene che le due cose lavorino **davvero in contemporanea**: avvia la chat con una domanda lunga e,
mentre risponde, lascia girare il motore. Sul conflitto di memoria fra le due applicazioni vedi
[convivenza_applicazioni.md](../../docs/convivenza_applicazioni.md).

### 4.5 Durante la prova

- Non toccare altre impostazioni (ventole, piano di alimentazione, overclock) finché non finisce.
- Se la temperatura entra nella fascia critica e non scende, **ferma il carico**: è un avviso, non un comando, e
  decidi tu. Se il PC fa rumori strani, odore di bruciato o si spegne, ferma tutto e lascia raffreddare.
- Se il PC si spegne o si riavvia da solo durante la prova, è già un risultato: il CSV è completo fino all'ultimo
  secondo (viene scritto riga per riga), il `.json` è aggiornato all'ultima fase finita (e dice
  `chiuso_regolarmente: false`) e `riepilogo` ricostruisce la tabella dal CSV (sezione
  [7](#7-osservare-senza-fasi-osserva)). Non ripetere la stessa prova con lo stesso carico finché non hai capito
  perché è successo.

Alla fine, il programma stampa il riepilogo e scrive nella cartella indicata con `--uscita` (di default quella da
cui l'hai lanciato) tre file con lo stesso nome, `termico-<nome>-<data>-<ora>`:

| File | Contenuto |
|---|---|
| `.csv` | Un campione al secondo, per fare grafici (Excel, LibreOffice...). |
| `.json` | Il riepilogo per i programmi, e il file da allegare per condividere la prova. |
| `.txt` | Il riepilogo che hai visto a schermo. |

## 5. Affiancare il motore (`--motore`)

Aggiungendo l'indirizzo di `sd-server`, ogni campione della GPU porta con sé lo stato del motore nello stesso
istante: così si vede se il motore stava lavorando, quanti job ha fatto, se ha cambiato profilo o ha avuto errori
**mentre** la GPU si scaldava.

```shell
python scripts/diagnostica_termica/termico.py guidata --motore http://127.0.0.1:1234 --nome portatile-alimentatore
```

Se `sd-server` è stato avviato con `--api-key`, passa la chiave con `--chiave-api CHIAVE` oppure, meglio su un PC
condiviso (gli argomenti dei programmi sono visibili agli altri utenti), con la variabile d'ambiente
`TERMICO_CHIAVE_API`. La chiave viene mandata soltanto al motore e **non è mai scritta** nei file né a schermo.

Cosa viene letto, e con che frequenza (in un thread a parte, così un motore lento non rallenta le letture della
GPU):

| Richiesta | Ogni | Che cosa ne esce |
|---|---|---|
| `GET /sdcpp/v1/health` | 2 secondi (`--ogni-motore`) | stato (`ok`, `degraded`...), profilo in uso, se sta generando, job in coda, contatori (richieste, riuscite, fallite, annullate, discese di profilo, ricaricamenti), tempo di attività |
| `GET /sdcpp/v1/resources` | ogni 3 letture di `health` | RAM di sistema disponibile e memoria video libera della GPU (il primo dispositivo di tipo GPU) |

Descrizione dei campi: [examples/server/api.md](../../examples/server/api.md). Il riepilogo aggiunge una tabella
"MOTORE (sd-server) durante le fasi" con, per ogni fase: quanto del tempo il motore era raggiungibile e occupato,
quali profili ha usato, quanti job ha concluso/fallito/annullato **durante la fase** (differenza fra i contatori
a inizio e a fine fase), quante discese di profilo e ricaricamenti, la coda massima e i minimi di RAM e memoria
video libera.

Come leggerla: una discesa di profilo (`discese`) significa che il motore ha ridotto la configurazione perché
**la memoria** non bastava, non perché faceva caldo; se compare insieme a una memoria video libera quasi a zero, il
problema è la memoria e si risolve come descritto in [convivenza_applicazioni.md](../../docs/convivenza_applicazioni.md).
Se il motore è stato riavviato a metà fase (il tempo di attività torna indietro) i contatori non sono
confrontabili e lo strumento lo segnala invece di dare numeri sbagliati. Se il motore non risponde (spento, porta
sbagliata, chiave errata) la GPU si registra lo stesso: il CSV riporta `0` in `motore_raggiungibile` e il
programma lo dice all'inizio.

## 6. Leggere i risultati

### 6.1 La tabella

Esempio di riepilogo di una prova completa con il motore affiancato. **I numeri sono inventati**, servono solo a
mostrare il formato:

```text
GPU: NVIDIA GeForce RTX 3060 Laptop GPU, driver 576.52  (prova: portatile)
fase            durata  temp C ini/fine/max  dT a 60s (C)  W med/max  MHz med/min  util% med  VRAM MiB max
RIPOSO          1:59    47/48/48             +1            9.5/9.5    210/210      3          520
MOTORE          3:59    48/76/76             +15           76.5/78.5  1578/1536    97         4310
CHAT            3:19    50/63/63             +8            42.0/42.0  1480/1480    81         5200
INSIEME         3:59    55/88/88             +18           74.2/79.0  1509/1377    99         5850
RAFFREDDAMENTO  1:59    88/55/88             -25           11.0/11.0  300/300      4          520

fase            motivi di throttling (% del tempo)   giudizio (indicativo)
RIPOSO          nessuno                              -
MOTORE          limite_potenza 100%                  da_verificare
CHAT            limite_potenza 89%                   sostenibile
INSIEME         limite_potenza 100%, termico_sw 45%  non_sostenibile
RAFFREDDAMENTO  nessuno                              -

MOTORE (sd-server) durante le fasi
fase     raggiungibile  occupato  profili  job ok/falliti/annullati  discese  coda max  RAM libera min MiB  VRAM libera min MiB
MOTORE   100%           100%      max      17/0/0                    0        1         7961                640
INSIEME  100%           100%      max      17/0/0                    0        1         7961                48

Fasce di temperatura (INDICATIVE, non garanzie del produttore): attenzione da 80 C, critica da 87 C.
Origine: valori generici per un portatile: il driver non dichiara i suoi limiti.

Note sui giudizi (indicativi):
  MOTORE: a fine fase la temperatura saliva ancora: serve una prova più lunga per vedere dove si ferma
  CHAT: la temperatura si è stabilizzata sotto la fascia di attenzione
  INSIEME: ha toccato la fascia critica (88 C, da 87 C)
  INSIEME: il driver ha rallentato la GPU per il caldo nel 45% del tempo
```

Leggi la prima tabella come "che cosa è successo alla GPU", la seconda come "perché e che giudizio ne esce", la terza
(solo con `--motore`) come "che cosa stava facendo il motore nel frattempo".

| Colonna | Significato |
|---|---|
| `durata` | Minuti:secondi registrati. |
| `temp C ini/fine/max` | Temperatura della GPU all'inizio, alla fine e al massimo della fase. |
| `dT a 60s (C)` | Di quanti gradi è salita nei primi 60 secondi: dice quanto è veloce a scaldarsi. |
| `W med/max` | Potenza assorbita, media e massima. Su alcuni portatili il driver non la dà (`n/d`). |
| `MHz med/min` | Frequenza del processore grafico, media e minima: se cala col tempo, la GPU sta perdendo velocità. |
| `util% med` | Quanto era occupata la GPU in media. |
| `VRAM MiB max` | Memoria video usata al massimo (da tutti i programmi, non solo dal motore). |
| `motivi di throttling` | Per quanta parte del tempo il driver ha dichiarato ciascun motivo di frenata (6.2). |
| `giudizio` | Giudizio indicativo sulle fasi con carico (6.3). Riposo e raffreddamento non si giudicano. |

### 6.2 I motivi di throttling

Il driver dice **perché** sta tenendo la GPU sotto la frequenza massima. I nomi usati qui e i bit di NVML
corrispondenti:

| Nome | Bit | Che cosa dice il driver | Come leggerlo |
|---|---|---|---|
| `limite_potenza` | `0x4` | È stato raggiunto il limite di potenza (software). | Molto comune sotto carico sui portatili: la GPU lavora al massimo della potenza concessa. Non è un guasto, ma limita la velocità. |
| `termico_sw` | `0x20` | Il driver rallenta per la temperatura. | Segnale di caldo: la GPU sta perdendo velocità per questo. |
| `termico_hw` | `0x40` | Rallentamento termico a livello hardware. | Segnale di caldo serio. |
| `rallentamento_hw` | `0x8` | Rallentamento hardware (può dipendere da temperatura, alimentazione o segnali esterni). | Non dovrebbe durare a lungo; guarda gli altri motivi. |
| `freno_potenza_hw` | `0x80` | Un freno di potenza esterno (per esempio l'alimentatore o la batteria) limita la GPU. | Sospetta un alimentatore non adatto o la batteria. |
| `clock_fissato` | `0x2` | Le frequenze sono fissate da un'impostazione o da un programma. | Normale solo se l'hai impostato tu. |
| `sync_boost` | `0x10` | Frequenze allineate fra più GPU. | Raro. |
| `clock_display` | `0x100` | Frequenze legate al display. | Raro. |

Il bit `0x1` (GPU inattiva) non è mostrato: a riposo è normale. Se il tuo driver non rende leggibili i motivi,
la colonna dice `n/d` e lo strumento lo scrive all'inizio. Il campo letto si chiama `clocks_throttle_reasons.active`
nei driver meno recenti e `clocks_event_reasons.active` in quelli recenti: il programma prova entrambi.

La riga "throttling per il caldo" del confronto e il giudizio contano solo `termico_sw` e `termico_hw`: il limite
di potenza è normale sotto carico e non dice nulla sul raffreddamento.

### 6.3 I giudizi

Solo per le fasi con carico (MOTORE, CHAT, INSIEME, e OSSERVAZIONE). Sono **indicativi**: una regola semplice,
non una perizia.

| Giudizio | Quando |
|---|---|
| `troppo_corta` | La fase è durata meno di 3 minuti: non si può dire se la temperatura si stabilizza. |
| `sostenibile` | Si è stabilizzata sotto la fascia di attenzione e il driver non ha frenato per il caldo. |
| `da_verificare` | È sotto l'attenzione, ma a fine fase saliva ancora di 1,5 °C al minuto o più: ripeti la fase più a lungo. |
| `al_limite` | È entrata nella fascia di attenzione, oppure il driver ha frenato per il caldo (anche poco). |
| `non_sostenibile` | Ha toccato la fascia critica, oppure il driver ha frenato per il caldo per il 20% del tempo o più. |

Ogni giudizio ha sotto la tabella la spiegazione in una riga ("ha toccato la fascia critica (88 C, da 87 C)").

### 6.4 Le fasce di temperatura

Le fasce servono a dare un nome alla temperatura: `normale`, `attenzione`, `critica`. **Sono indicative.**

- Se il driver dichiara i suoi limiti (`nvidia-smi -q -d TEMPERATURE`, voce *GPU Slowdown Temp*, oppure *GPU Max
  Operating Temp*), l'attenzione scatta 10 °C sotto il limite e la critica 3 °C sotto.
- Se non li dichiara, si usano valori generici da portatile: attenzione da **80 °C**, critica da **87 °C**.
- Se le conosci meglio tu, forzale: `--attenzione 75 --critica 83`. Il riepilogo dice sempre da dove vengono
  (`origine`).

Una GPU da portatile che lavora a 75-82 °C sotto carico è comune e, finché il driver non frena per il caldo, spesso
accettabile; la temperatura da sola non dice se la GPU sta soffrendo: guarda anche frequenza e motivi di throttling.

### 6.5 Alcuni casi tipici

| Cosa vedi | Che cosa vuol dire | Che cosa puoi provare |
|---|---|---|
| Temperatura che sale e si ferma sotto l'attenzione; `limite_potenza` al 100%; frequenza costante | La GPU è limitata dalla potenza concessa, non dal caldo. È il caso sano. | Niente. |
| Frequenza media che cala nel tempo e `termico_sw`/`termico_hw` diversi da zero | Il raffreddamento non basta: la GPU perde velocità a causa del caldo. | Pulire le ventole, alzare il portatile o usare un supporto con ventole, ridurre il carico (profilo `lite`, immagini più piccole), controllare il piano di alimentazione. |
| `freno_potenza_hw` | Un limite esterno di potenza. | Alimentatore originale e collegato bene; non a batteria. |
| `dT a 60s` molto alto (più di 20 °C) e subito attenzione | Si scalda in fretta: dissipatore o pasta termica in cattivo stato, ventole bloccate, ambiente caldo. | Confrontare con la stessa prova dopo la pulizia (sezione [8](#8-confrontare-più-prove-confronta)). |
| Memoria video usata vicina al totale, `discese` maggiori di zero nel motore | Il problema è la memoria, non il caldo. | [convivenza_applicazioni.md](../../docs/convivenza_applicazioni.md). |
| Temperatura bassa ma `util%` basso e job lenti | Il collo di bottiglia non è la GPU (disco, RAM, CPU). | Non è un tema termico. |

## 7. Osservare senza fasi (`osserva`)

Per registrare mentre usi il PC normalmente, o mentre un altro programma pilota il motore:

```shell
python scripts/diagnostica_termica/termico.py osserva --durata 1800 --ogni 5 --nome sera --motore http://127.0.0.1:1234
```

Senza `--durata` registra finché non premi Ctrl+C (o il programma riceve un segnale di terminazione): il riepilogo
viene scritto lo stesso. `--ogni` è l'intervallo in secondi fra due campioni (da 0,2; di default 1). Per sessioni di
ore usa 5 o 10 secondi: i campioni stanno in memoria e il riepilogo viene ricalcolato ogni minuto.

Due cose utili per non perdere la prova:

- Il file `.json` viene scritto **subito** e riaggiornato ogni minuto con `chiuso_regolarmente: false`; alla chiusura
  diventa `true`. Se il PC si spegne, quel campo dice che la prova è stata interrotta.
- Il `.csv` è completo fino all'ultimo campione scritto. Con `riepilogo` si ricostruisce la tabella:

  ```shell
  python scripts/diagnostica_termica/termico.py riepilogo termico-sera-20261014-213000.csv --json riepilogo.json
  ```

  Prende GPU e fasce dal `.json` che gli sta accanto, se c'è, e avverte che la prova non è stata chiusa
  regolarmente. Una riga rimasta a metà in fondo al file viene scartata.

`osserva --file-stato stato.json` riscrive a ogni campione un file con l'ultima lettura, per altri programmi
(sezione [9](#9-per-altri-programmi-json-csv-e-stato)).

Una cautela: interrogare la GPU ogni secondo con `nvidia-smi` è leggero, ma su alcuni portatili con grafica ibrida
può impedire alla GPU di spegnersi quando non serve. Non lasciare `osserva` acceso per ore a batteria se non ti
serve.

### 7.1 Segnalare la fascia al motore (`--segnala`)

Con `--segnala` il programma non si limita a guardare: a ogni campione manda a `sd-server` la fascia della GPU come
livello di salute del suo device, e il motore le toglie lavoro un gradino alla volta (85%, 70%, 55% di quello che può
tenere, con i valori di partenza) e lo passa agli altri device; una GPU esclusa non calcola più niente finché non si
riprende.

```shell
python scripts/diagnostica_termica/termico.py osserva --motore http://127.0.0.1:1234 --segnala
```

| Cosa vede `termico.py` | Livello mandato |
|---|---|
| fascia normale | nessuno (un segnale mandato prima si ritira) |
| fascia normale, ma il driver rallenta la GPU per il caldo | 1 |
| fascia di attenzione | 1 |
| attenzione con throttling termico | 2 |
| fascia critica | 3 |
| critica con throttling termico hardware | escluso |

- **Quale device.** Senza nome, il programma cerca la GPU fra i device del motore (`GET /sdcpp/v1/resources`) dal nome
  che dà `nvidia-smi`; la stessa scheda vista da CUDA e da Vulkan va bene con l'uno o l'altro nome, perché il motore
  riconosce la scheda e passa il segnale anche all'altro. Con due schede uguali, o se il nome non si trova, indica il
  device: `--segnala CUDA0` (i nomi li elenca `GET /sdcpp/v1/health`, sotto `devices`).
- **Un motore con altri livelli.** Il motore può avere più o meno di tre livelli di sforzo (`--device-health levels=N`):
  il programma li legge da `GET /sdcpp/v1/health` e non va mai oltre l'ultimo, perché il livello dopo è l'esclusione.
  Con due livelli la fascia critica vale 2, e lo dice all'inizio; l'esclusione arriva solo con il throttling hardware.
- **Quanto dura.** Il segnale vale qualche campione (5 secondi con `--ogni 1`) e si ripete finché la fascia resta.
  Tornata la GPU normale, e quando il programma si ferma, il segnale si ritira; il livello poi scende di un gradino
  ogni 5 minuti di calma (con i valori di partenza), non di colpo. Se il programma cade, il segnale scade da solo
  dopo pochi secondi.
- **Cosa non fa.** Non ferma mai il motore da solo: manda al massimo "escluso", e il motore si ferma del tutto solo se
  non gli resta nessun device su cui calcolare (sezione "Device health and the guardian" della
  [guida dell'API](../../examples/server/api.md#device-health-and-the-guardian)).
- **Errori.** Un motore che non risponde, la chiave API mancante o un device sconosciuto si dicono una volta sola, e
  la misura va avanti; un motore lento non rallenta i campioni. La chiave è la stessa di `--motore` (`--chiave-api`
  o `TERMICO_CHIAVE_API`).

Le fasce restano indicative (sezione [6.4](#64-le-fasce-di-temperatura)): se sul tuo portatile il driver rallenta la GPU
prima della fascia di attenzione, il programma lo segnala lo stesso con il livello 1.

## 8. Confrontare più prove (`confronta`)

```shell
python scripts/diagnostica_termica/termico.py confronta prima.json dopo.json
```

Accetta file `.json` e `.csv` (anche mischiati; con un `.csv` prende GPU e fasce dal `.json` accanto). La prima
è il riferimento. Con due prove la tabella ha anche la colonna "differenza" (seconda meno prima); con tre o più
solo i valori. Esempio, con numeri inventati (stessa prova prima e dopo la pulizia delle ventole):

```text
CONFRONTO FRA PROVE
  A = prima-pulizia, NVIDIA GeForce RTX 3060 Laptop GPU, driver 576.52
  B = dopo-pulizia, NVIDIA GeForce RTX 3060 Laptop GPU, driver 576.52

Fase MOTORE
                               A                B            differenza
temp max (C)                   87               76           -11
temp ultimi 60 s (C)           87.0             76.0         -11.0
potenza media (W)              76.0             76.0         +0.0
clock medio (MHz)              1423             1602         +179
clock minimo (MHz)             1290             1590         +300
utilizzo GPU medio (%)         97               97           +0
throttling per il caldo (%)    49               0            -49
throttling per la potenza (%)  100              100          +0
giudizio (indicativo)          non_sostenibile  sostenibile
```

Qui la pulizia ha tolto il rallentamento per il caldo e ha restituito circa 180 MHz di frequenza media.

Usi tipici: prima e dopo la pulizia delle ventole o il cambio della pasta termica; con e senza alimentatore; con
e senza supporto; due profili del motore (`--profile lite` e `--profile max`); due piani di alimentazione. Per
fare un confronto che vale, ripeti **la stessa prova** (stesse fasi, stesso carico, stessa durata, stessa stanza)
cambiando una sola cosa. Se le due prove sono di GPU o driver diversi il programma te lo scrive: il confronto
vale come ordine di grandezza.

`--json confronto.json` salva il confronto in un formato leggibile da altri programmi.

## 9. Per altri programmi (JSON, CSV e `stato`)

Tutti i file sono pensati per essere letti da altri programmi (uno script, un'applicazione, un agente). I formati
sono **versionati** nel campo `formato` (`termico-stato/1`, `termico-riepilogo/1`, `termico-confronto/1`): chi legge
ignora i campi che non conosce, perché in futuro ne possono arrivare altri senza cambiare la versione. I valori
mancanti sono `null` (nei file) o vuoti (nel CSV), mai zero inventato.

### 9.1 `stato`: una lettura

```shell
python scripts/diagnostica_termica/termico.py stato --motore http://127.0.0.1:1234
```

Stampa sempre **un JSON** (anche in caso di errore: allora c'è un campo `errore`):

```json
{
  "formato": "termico-stato/1",
  "istante": "2026-10-14T21:30:12",
  "istante_unix": 1791999012.4,
  "gpu": {
    "indice": 0, "nome": "NVIDIA GeForce RTX 3060 Laptop GPU", "temp_c": 74.0, "potenza_w": 76.8,
    "clock_mhz": 1596.0, "util_pct": 97.0, "vram_usata_mib": 4310.0, "pstate": "P2",
    "motivi_throttling": ["limite_potenza"]
  },
  "fascia": {"nome": "normale", "indicativo": true, "attenzione_da_c": 80.0, "critica_da_c": 87.0, "origine": "..."},
  "tendenza_c_al_minuto": null,
  "motore": {
    "raggiungibile": true, "stato": "ok", "profilo": "max", "occupato": true, "coda": 1, "in_attesa": 0,
    "job_riusciti": 12, "job_falliti": 0, "job_annullati": 0, "discese_di_profilo": 0, "ricaricamenti": 1,
    "ram_disponibile_mib": 8704.0, "vram_libera_mib": 3584.0
  },
  "avvisi": []
}
```

(Numeri inventati.) `motore` è `null` se non hai passato `--motore`; se il motore non risponde è
`{"raggiungibile": false, "errore": "..."}` e negli `avvisi` compare una riga. `motivi_throttling` è `null` se il
driver non li dice. `tendenza_c_al_minuto` richiede una storia di almeno 30 secondi (e 10 campioni): in `stato`,
che fa una lettura sola, è sempre `null`; in `osserva --file-stato` c'è dopo mezzo minuto circa.

**Codice di uscita** (stessa convenzione dei plugin di monitoraggio): `0` fascia normale, `1` attenzione, `2` critica,
`3` lettura impossibile o uso sbagliato del comando. Il codice parla solo della **temperatura della GPU**; lo stato
del motore è nel JSON.

Leggerlo:

```python
import json, subprocess, sys
r = subprocess.run([sys.executable, "scripts/diagnostica_termica/termico.py", "stato"], capture_output=True, text=True)
quadro = json.loads(r.stdout)            # c'è sempre un JSON
if "errore" in quadro:                   # codice di uscita 3: nessun dato
    ...
fascia = quadro["fascia"]["nome"]        # "normale", "attenzione" o "critica" (indicativa)
```

```powershell
$testo = python scripts\diagnostica_termica\termico.py stato
$codice = $LASTEXITCODE
$quadro = $testo | ConvertFrom-Json
"$($quadro.fascia.nome) $($quadro.gpu.temp_c) C (codice $codice)"
```

```shell
python3 scripts/diagnostica_termica/termico.py stato > stato.json; echo "codice $?"
jq -r '.fascia.nome, .gpu.temp_c' stato.json
```

### 9.2 `osserva --file-stato`: l'ultima lettura, sempre aggiornata

```shell
python scripts/diagnostica_termica/termico.py osserva --file-stato stato.json --motore http://127.0.0.1:1234
```

Il file ha lo stesso formato di `stato` (con la tendenza) ed è riscritto a ogni campione **per intero o per niente**:
chi lo legge non lo trova mai a metà (su Windows, se in quell'istante il file è aperto da un altro programma, la
scrittura riprova subito dopo). Regole per chi lo legge:

- **Un dato vecchio non è un dato.** Confronta `istante_unix` con l'ora attuale (`time.time()`): se la lettura ha
  più di una decina di secondi, l'osservatore non sta più girando, e la temperatura va trattata come sconosciuta, non
  come "l'ultima che ho visto".
- Se il file non c'è, o non si legge, è la stessa cosa: dato mancante.
- Non aspettare il file in un ciclo che blocca il resto del programma; leggilo quando serve decidere.

### 9.3 Il riepilogo (`.json`)

Un oggetto con `formato: "termico-riepilogo/1"` e questi campi:

| Campo | Contenuto |
|---|---|
| `chiuso_regolarmente` | `true` se la prova è stata chiusa dal programma (anche con Ctrl+C); `false` se è stata interrotta di colpo. |
| `nome`, `creato`, `aggiornato` | Etichetta data da te, inizio e ultimo salvataggio (ora locale). |
| `gpu` | `indice`, `nome`, `driver`, `limite_potenza_w`, `clock_max_mhz`, `temp_rallentamento_c`, `temp_spegnimento_c`, `temp_max_operativa_c` (ciò che il driver dichiara; `null` se non lo dice). |
| `fasce` | `attenzione_da_c`, `critica_da_c`, `origine`, e `indicativo: true`. |
| `motore` | `url` (senza eventuali credenziali), `raggiungibile_all_avvio`, `versione`, `profilo_base`; `null` senza `--motore`. |
| `fasi_previste` | I nomi delle fasi che la prova intendeva fare. |
| `fasi` | Una voce per fase registrata (sotto). |

Ogni voce di `fasi`: `nome`, `durata_s`, `campioni`; `temp_c` (`inizio`, `fine`, `max`, `media`, `ultimi_60s_mediana`,
`variazione_dopo_60s`, `pendenza_finale_c_min`); `potenza_w` (`media`, `max`); `clock_mhz` (`media`, `min`, `max`);
`util_gpu_pct` (`media`); `vram_usata_mib` (`max`); `throttling` (`leggibile`, `motivi_pct`, `termico_pct`); `fasce`
(`secondi_oltre_attenzione`, `secondi_oltre_critica`, `tempo_alla_fascia_attenzione_s`); `ancora_in_salita`;
`valutazione` (`giudizio`, `indicativo`, `note`; `null` per riposo e raffreddamento); `motore` (`campioni`,
`raggiungibile_pct`, `occupato_pct`, `stati`, `profili`, `coda_max`, `ram_disponibile_min_mib`,
`vram_libera_min_mib`, `riavviato`, `job_riusciti`, `job_falliti`, `job_annullati`, `discese_di_profilo`,
`ricaricamenti`; `null` senza `--motore`).

### 9.4 Il CSV

Una riga per campione, con la stessa colonna del tempo per la GPU e per il motore:

| Colonne | Contenuto |
|---|---|
| `istante`, `fase`, `t_s` | Ora locale del campione, nome della fase, secondi dall'inizio della fase. |
| `temp_c`, `potenza_w`, `clock_sm_mhz`, `util_gpu_pct`, `vram_usata_mib`, `pstate` | Lettura della GPU. Vuoto = il driver non l'ha data. |
| `motivi_hex`, `motivi` | Maschera dei motivi di throttling in esadecimale e i nomi corrispondenti (vuoti se non c'è alcun limite). |
| `motore_raggiungibile` | `1` risposto, `0` non risposto, vuoto = senza `--motore`. |
| `motore_stato`, `motore_profilo`, `motore_occupato`, `motore_coda` | Dal `health` del motore. |
| `motore_richieste`, `motore_riuscite`, `motore_fallite`, `motore_annullate`, `motore_discese`, `motore_ricaricamenti`, `motore_uptime_s` | Contatori del motore dall'avvio. |
| `ram_disponibile_mib`, `vram_libera_mib` | Da `resources`. |

Il motore è letto ogni 2 secondi: le colonne del motore ripetono l'ultima lettura finché non ne arriva una nuova;
se per più di 10 secondi non arriva niente, restano vuote.

## 10. Cosa fa sul tuo PC, esattamente

`termico.py` esegue soltanto queste interrogazioni di `nvidia-smi` (nessuna delle quali cambia qualcosa):

```text
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader,nounits -i 0
nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits -i 0
nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits -i 0
nvidia-smi -q -d TEMPERATURE -i 0
nvidia-smi --query-gpu=clocks_throttle_reasons.active --format=csv,noheader,nounits -i 0   (o clocks_event_reasons.active)
nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,utilization.gpu,memory.used,pstate,... -i 0   (a ogni campione)
```

(`-i 0` è la GPU scelta con `--gpu`.) Rete: **solo** se passi `--motore`, richieste `GET` a
`<indirizzo>/sdcpp/v1/health` e `/sdcpp/v1/resources` dell'indirizzo che indichi, senza passare da proxy; con
`osserva --segnala` anche `POST <indirizzo>/sdcpp/v1/devices/{device}/pressure` a ogni campione (sezione
[7.1](#71-segnalare-la-fascia-al-motore---segnala)); niente altro, niente telemetria. File: scrive `.csv`, `.json`, `.txt` e l'eventuale file di stato soltanto nella cartella di
uscita (il nome della prova viene ripulito: non può far uscire i file dalla cartella). Non modifica né legge altro.

`carico_motore.py` parla solo con il server che gli indichi: `POST /sdcpp/v1/img_gen`, `GET /sdcpp/v1/jobs/{id}` e
`POST /sdcpp/v1/jobs/{id}/cancel`. Non salva immagini.

## 11. Cosa è verificato e cosa no

**Verificato**, con 161 controlli automatici in [`test/termico_test.py`](../../test/termico_test.py) (due minuti e
mezzo; una GPU finta che risponde come `nvidia-smi`, un motore finto, e il programma lanciato davvero): calcolo di
fasce, giudizi, pendenze e riepiloghi; lettura e scrittura del CSV (anche con righe troncate); `stato` con tutti i
codici di uscita; `osserva` con file di stato letto in continuazione senza mai trovarlo a metà; `riepilogo` di una
prova interrotta con un segnale di terminazione; la sessione guidata pilotata come da una persona (compreso l'Invio
premuto per sbaglio durante una fase automatica); `confronta`; motore spento, motore che si spegne a metà, chiave API
giusta e sbagliata, GPU che risponde a intermittenza o per niente; la chiave non compare mai nell'uscita né nei file;
`osserva --segnala` con motori finti a tre, due e un livello di sforzo, e contro un vero `sd-server` in
[`test/server_guardian_e2e.py`](../../test/server_guardian_e2e.py), anche con `--device-health levels=2`. Inoltre
`termico.py` e `carico_motore.py` sono stati provati contro un vero `sd-server` (build per CPU, modello
sintetico, GPU simulata) mentre generava, con uno script di prova che non fa parte del repository: sei job completati
e uno fermato con `cancel`, i contatori letti da `termico.py` uguali a quelli di `/health`, nessuna discesa di
profilo. Per vedere che i controlli servano a qualcosa, durante lo sviluppo sono stati inseriti di proposito 42 errori
nei due programmi (una soglia sbagliata, un bit del throttling dimenticato, una chiave API non inviata, ...) e tutti
sono stati individuati; anche lo script che li inserisce non fa parte del repository.

**Non verificato**:

- una **GPU NVIDIA reale**: i formati veri dell'uscita del driver (in particolare la voce *GPU Slowdown Temp* di
  `nvidia-smi -q -d TEMPERATURE` e i nomi dei campi dei motivi di throttling nel tuo driver) sono quelli
  documentati, non visti; se il tuo driver risponde in altro modo il programma lo dice (`n/d`, fasce generiche)
  invece di inventare;
- **Windows**: il rilevamento dell'Invio da tastiera (`msvcrt`), la codifica della console e la ricerca di
  `nvidia-smi.exe` sono scritti per Windows ma non sono stati eseguiti;
- computer con **più GPU** o con grafica ibrida (si sceglie con `--gpu N`; l'indice è quello di `nvidia-smi`, che può
  non coincidere con i nomi `CUDA0`, `CUDA1` del motore);
- **sessioni di molte ore**.

## 12. Problemi comuni

| Sintomo | Causa probabile | Cosa fare |
|---|---|---|
| `nvidia-smi non trovato` | Driver NVIDIA assente o `nvidia-smi` fuori dal percorso. | In PowerShell `where.exe nvidia-smi`. Cerca `C:\Windows\System32\nvidia-smi.exe` o `C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe`; indica il file con `--nvidia-smi PERCORSO`. |
| `nvidia-smi non risponde per la GPU 0` | Indice sbagliato o driver bloccato. | Prova `nvidia-smi` da solo; con più GPU usa `--gpu 1`. |
| Python non si trova / `python` apre il Microsoft Store | Alias di Windows. | Usa `py -3` oppure il Python di conda (`(base)`). |
| Lettere accentate strane in `cmd.exe` | Codifica della console. | Usa PowerShell o Windows Terminal, oppure `chcp 65001`. |
| `n/d` nella potenza | Il driver non dà la potenza su quel portatile. | Normale: il resto funziona. |
| Motivi di throttling `n/d` | Il driver non li rende leggibili. | Lo strumento lo dice all'inizio. Il resto funziona. |
| Il motore "non risponde" | Indirizzo o porta sbagliati, server spento, chiave API errata. | Controlla con `curl http://127.0.0.1:1234/sdcpp/v1/health`; con `--api-key` serve `--chiave-api`. |
| "nvidia-smi non risponde" a metà prova | Dieci letture fallite di fila (driver in difficoltà, GPU persa). | La prova si chiude da sola e scrive i file. Se succede sotto carico, è già un'informazione. |
| Un Invio premuto troppo presto | Nelle fasi automatiche viene ignorato. | Aspetta la richiesta successiva. |
| La temperatura non cambia fra le fasi | La GPU non ha lavorato (programma sbagliato, il carico usa la CPU). | Guarda `util%` e `VRAM MiB`: se sono bassi, il carico non è sulla GPU. |

## 13. Condividere una prova

Per far vedere una prova ad altri (per esempio in una segnalazione sul progetto), allega il file `.json`, oppure
incolla il riepilogo. Dentro ci sono: il nome della GPU e la versione del driver, i numeri misurati, l'etichetta
che hai dato con `--nome`, l'ora locale e, se hai usato `--motore`, l'indirizzo del server **senza eventuali
credenziali** (di solito `http://127.0.0.1:1234`). Non c'è il nome del PC, il tuo nome utente né la chiave API. Se
l'etichetta o l'indirizzo dicono qualcosa che non vuoi condividere, cambiali nel file prima di allegarlo.

Per capire una prova con meno dati, bastano il riepilogo, il modello usato, il profilo del motore e come era
collegato il PC (alimentatore o batteria).

## 14. Riferimento delle opzioni

Le opzioni vanno **dopo** il comando (`termico.py osserva --durata 60`). Senza comando parte `guidata` (le opzioni
della prima versione funzionano ancora).

| Opzione | Comandi | Significato |
|---|---|---|
| `--gpu N` | tutti | Indice della GPU secondo `nvidia-smi` (default 0). |
| `--nvidia-smi PERCORSO` | tutti | Percorso di `nvidia-smi`, se non si trova da solo. |
| `--attenzione C`, `--critica C` | tutti | Temperature da cui la fascia è "attenzione" e "critica" (vincono sui limiti del driver). `--avviso` è il vecchio nome di `--attenzione`. |
| `--motore URL` | guidata, osserva, stato | Affianca lo stato di `sd-server` (per esempio `http://127.0.0.1:1234`). |
| `--chiave-api CHIAVE` | guidata, osserva, stato | Chiave API di `sd-server`; oppure variabile `TERMICO_CHIAVE_API`. |
| `--ogni-motore S` | guidata, osserva | Secondi fra due letture del motore (default 2). |
| `--nome ETICHETTA` | guidata, osserva | Etichetta della prova; entra nel nome dei file. |
| `--uscita CARTELLA` | guidata, osserva | Dove scrivere i file (default: cartella corrente; viene creata se manca). |
| `--ogni S` | guidata, osserva | Secondi fra due campioni (default 1, minimo 0,2). |
| `--fasi LISTA` | guidata | Fasi da fare, separate da virgola (default tutte). |
| `--auto-secondi S` | guidata | Durata delle fasi automatiche (default 120). |
| `--durata S` | osserva | Secondi da registrare (default: fino a Ctrl+C). |
| `--file-stato FILE` | osserva | File JSON con l'ultima lettura, riscritto a ogni campione. |
| `--segnala [DEVICE]` | osserva | Manda al motore (serve `--motore`) la fascia della GPU come livello di salute del device; senza `DEVICE` lo cerca dal nome della GPU (sezione 7.1). |
| `--json FILE` | riepilogo, confronta | Scrive anche il risultato in JSON. |

`carico_motore.py`: `--url` (default `http://127.0.0.1:1234`), `--chiave-api`, `--durata` (default 180),
`--larghezza`, `--altezza` (default 512), `--passi` (default 20), `--seme` (default 1), `--prompt`.

Codici di uscita: `termico.py stato` 0-3 come spiegato nella sezione 9.1; gli altri comandi, 0 se tutto è andato
bene e 1 se la prova è stata interrotta da un errore (per esempio `nvidia-smi` che smette di rispondere), 3 per un
uso sbagliato delle opzioni. `carico_motore.py`: 0 se ha completato almeno un job e nessuno è fallito, 1 altrimenti.

## Per approfondire

- [convivenza_applicazioni.md](../../docs/convivenza_applicazioni.md): il motore accanto a una chat o a un altro
  programma sulla stessa GPU (memoria, profili, `/health` e `/resources`).
- [integrazione_progetti.md](../../docs/integrazione_progetti.md): collegare il motore a un progetto, e la nuova
  interruzione dei job in corso.
- [examples/server/api.md](../../examples/server/api.md): l'interfaccia HTTP del server.
- [agenti_indipendenti.md](../../docs/agenti_indipendenti.md): come un'intelligenza artificiale, un agente o un
  nodo principale esterno può usare questi dati senza sovraccaricare la macchina né il motore.
