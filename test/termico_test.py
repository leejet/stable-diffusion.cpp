#!/usr/bin/env python3
"""Test di scripts/diagnostica_termica/termico.py con una GPU finta (un nvidia-smi scritto qui) e un motore finto.

Non serve nessuna GPU, nessun driver e nessun modello: la GPU e il motore sono processi e server locali che il
test crea e distrugge. Serve solo Python 3.8 o più recente, senza pacchetti.

  python3 test/termico_test.py            tutto (circa 1-2 minuti)
  python3 test/termico_test.py --veloce   solo le prove delle funzioni, senza lanciare il programma

Codice di uscita 0 = tutti i controlli superati. Su Windows la prova della sessione guidata (che usa uno
pseudo-terminale) e quella del segnale di terminazione vengono saltate.
"""
import argparse
import contextlib
import csv
import http.server
import importlib.util
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse

sys.dont_write_bytecode = True  # provare gli script non deve lasciare __pycache__ accanto a loro

QUI = os.path.dirname(os.path.abspath(__file__))
TERMICO = os.path.join(QUI, "..", "scripts", "diagnostica_termica", "termico.py")
CARICO = os.path.join(QUI, "..", "scripts", "diagnostica_termica", "carico_motore.py")

errori = 0


def controlla(condizione, messaggio):
    global errori
    if condizione:
        print("ok:   " + messaggio, flush=True)
    else:
        print("FAIL: " + messaggio, flush=True)
        errori += 1
    return condizione


def carica_modulo():
    spec = importlib.util.spec_from_file_location("termico", TERMICO)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


# ---------------------------------------------------------------------------------------------------------
# GPU finta: un file .py che termico.py esegue al posto di nvidia-smi (--nvidia-smi file.py).
# Cosa risponde lo decidono variabili d'ambiente e file nella cartella FAKE_DIR, che il test può cambiare
# mentre il programma gira: temp (gradi), mask (maschera dei motivi in esadecimale).
# ---------------------------------------------------------------------------------------------------------
GPU_FINTA = r'''
import os, sys, time

modo = os.environ.get("FAKE_MODE", "ok")
cartella = os.environ["FAKE_DIR"]
argomenti = sys.argv[1:]


def contenuto(nome):
    try:
        with open(os.path.join(cartella, nome)) as f:
            return f.read().strip()
    except OSError:
        return None


if "-q" in argomenti:
    if os.environ.get("FAKE_LIMITI") == "1":
        print("==============NVSMI LOG==============")
        print("Attached GPUs                             : 1")
        print("GPU 00000000:01:00.0")
        print("    Temperature")
        print("        GPU Current Temp                  : 45 C")
        print("        GPU T.Limit Temp                  : 42 C")
        print("        GPU Shutdown Temp                 : 98 C")
        print("        GPU Slowdown Temp                 : 90 C")
        print("        GPU Max Operating Temp            : 87 C")
        print("        GPU Target Temperature            : N/A")
    sys.exit(0)

campi = ""
for a in argomenti:
    if a.startswith("--query-gpu="):
        campi = a.split("=", 1)[1]
partenza = os.path.join(cartella, "t0")
if not os.path.exists(partenza):
    with open(partenza, "w") as f:
        f.write(repr(time.time()))
with open(partenza) as f:
    t = time.time() - float(f.read())

if modo == "dead" and campi not in ("name,driver_version", "clocks_throttle_reasons.active", "clocks_event_reasons.active"):
    print("Unable to determine the device handle for GPU0: Unknown Error", file=sys.stderr)
    sys.exit(9)
if modo == "flaky" and "temperature.gpu" in campi:  # una lettura su tre fallisce, mai dieci di fila
    percorso_conteggio = os.path.join(cartella, "letture")
    try:
        with open(percorso_conteggio) as f:
            letture = int(f.read())
    except (OSError, ValueError):
        letture = 0
    with open(percorso_conteggio, "w") as f:
        f.write(str(letture + 1))
    if letture % 3 == 2:
        print("Unable to determine the device handle for GPU0: Unknown Error", file=sys.stderr)
        sys.exit(9)

valori = []
for campo in campi.split(","):
    if campo == "name":
        valori.append("NVIDIA GeForce RTX 3060 Laptop GPU")
    elif campo == "driver_version":
        valori.append("576.52")
    elif campo == "power.limit":
        valori.append("80.00")
    elif campo == "clocks.max.sm":
        valori.append("2100")
    elif campo == "temperature.gpu":
        fissa = contenuto("temp")
        valori.append(fissa if fissa else str(int(min(88, 45 + t * 0.6))))
    elif campo == "power.draw":
        valori.append("[N/A]" if os.environ.get("FAKE_POTENZA_NA") == "1" else "%.2f" % (10 + t * 0.5))
    elif campo == "clocks.sm":
        valori.append(str(int(max(1200, 1700 - t * 5))))
    elif campo == "utilization.gpu":
        valori.append("97")
    elif campo == "memory.used":
        valori.append("4100")
    elif campo == "pstate":
        valori.append("P2")
    elif campo == "clocks_throttle_reasons.active":
        if modo in ("evt", "nomask"):
            print('Field "%s" is not a valid field to query.' % campo)
            sys.exit(2)
        valori.append("0x" + (contenuto("mask") or "0000000000000001"))
    elif campo == "clocks_event_reasons.active":
        if modo == "nomask":
            print('Field "%s" is not a valid field to query.' % campo)
            sys.exit(2)
        valori.append("0x" + (contenuto("mask") or "0000000000000004"))
    else:
        print('Field "%s" is not a valid field to query.' % campo)
        sys.exit(2)
print(", ".join(valori))
'''


class Gpu:
    """Una cartella di lavoro con la GPU finta dentro."""

    def __init__(self, base):
        self.cartella = tempfile.mkdtemp(prefix="gpu-", dir=base)
        self.file = os.path.join(self.cartella, "nvidia-smi-finto.py")
        with open(self.file, "w") as f:
            f.write(GPU_FINTA)

    def imposta(self, nome, valore):
        percorso = os.path.join(self.cartella, nome)
        if valore is None:
            if os.path.exists(percorso):
                os.remove(percorso)
        else:
            with open(percorso, "w") as f:
                f.write(str(valore))

    def ambiente(self, **extra):
        env = dict(os.environ, FAKE_DIR=self.cartella)
        env.pop("TERMICO_CHIAVE_API", None)
        env.update({k: str(v) for k, v in extra.items()})
        return env

    def riparti(self):
        self.imposta("t0", None)


NOME_GPU_FINTA = "NVIDIA GeForce RTX 3060 Laptop GPU"


# ---------------------------------------------------------------------------------------------------------
# Motore finto: risponde come sd-server a /sdcpp/v1/health, /sdcpp/v1/resources e alle segnalazioni
# (POST /sdcpp/v1/devices/{device}/pressure), e ricorda ogni richiesta.
# ---------------------------------------------------------------------------------------------------------
class MotoreFinto:
    DISPOSITIVI = ("CPU", "CUDA0", "Vulkan0", "RAM", "disk")

    def __init__(self, chiave=None, stato="ok", profilo="max", descrizione_gpu="gpu", ritardo_segnali=0.0):
        self.chiave = chiave
        self.stato = stato
        self.profilo = profilo
        self.descrizione_gpu = descrizione_gpu
        self.ritardo_segnali = ritardo_segnali
        self.richieste = []
        self.segnali = []  # (device, corpo) di ogni segnalazione accettata, in ordine
        self.avvio = time.time()
        motore = self

        class Gestore(http.server.BaseHTTPRequestHandler):
            def log_message(self, *argomenti):
                pass

            def rispondi(self, codice, corpo):
                dati = json.dumps(corpo).encode()
                self.send_response(codice)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(dati)))
                self.end_headers()
                self.wfile.write(dati)

            def senza_diritto(self):
                return motore.chiave and self.headers.get("Authorization") != "Bearer " + motore.chiave

            def do_GET(self):
                motore.richieste.append(("GET", self.path))
                if self.senza_diritto():
                    return self.rispondi(401, {"error": "unauthorized"})
                passato = time.time() - motore.avvio
                if self.path == "/sdcpp/v1/health":
                    riuscite = int(passato / 1.5)
                    return self.rispondi(503 if motore.stato == "unavailable" else 200, {
                        "status": motore.stato, "model_loaded": True, "profile": motore.profilo,
                        "base_profile": "max", "degradation": True, "busy": True,
                        "counters": {"requests": riuscite + 1, "succeeded": riuscite, "failed": 0, "cancelled": 0,
                                     "recovered": 0, "step_downs": 0, "promotions": 0, "failed_promotions": 0,
                                     "rebuilds": 1},
                        "last_error": None, "next_promotion_seconds": None, "uptime_seconds": int(passato) + 100,
                        "version": "stable-diffusion.cpp finto", "queue": {"pending": 1, "waiting": 0, "max": 64}})
                if self.path == "/sdcpp/v1/resources":
                    return self.rispondi(200, {
                        "system_memory": {"total_bytes": 17179869184, "available_bytes": 9126805504,
                                          "process_bytes": 3221225472},
                        "devices": [{"name": "CPU", "description": "cpu", "type": "cpu",
                                     "free_memory_bytes": 8000000000, "total_memory_bytes": 17179869184},
                                    {"name": "CUDA0", "description": motore.descrizione_gpu, "type": "gpu",
                                     "free_memory_bytes": 3758096384, "total_memory_bytes": 6442450944},
                                    {"name": "Vulkan0", "description": motore.descrizione_gpu, "type": "gpu",
                                     "free_memory_bytes": 3758096384, "total_memory_bytes": 6442450944}]})
                self.rispondi(404, {"error": "not found"})

            def altro(self):
                motore.richieste.append((self.command, self.path))
                self.rispondi(405, {"error": "method not allowed"})

            def do_POST(self):
                parti = self.path.split("/")
                if len(parti) != 6 or parti[1:4] != ["sdcpp", "v1", "devices"] or parti[5] != "pressure":
                    return self.altro()
                motore.richieste.append(("POST", self.path))
                if self.senza_diritto():
                    return self.rispondi(401, {"error": "unauthorized"})
                corpo = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"null")
                time.sleep(motore.ritardo_segnali)
                dispositivo = urllib.parse.unquote(parti[4])
                if dispositivo not in MotoreFinto.DISPOSITIVI:
                    return self.rispondi(404, {"error": "no device is named " + dispositivo, "code": "unknown_device"})
                motore.segnali.append((dispositivo, corpo))
                self.rispondi(200, {"device": {"name": dispositivo, "level": 0}, "guardian": {"enabled": True, "halted": False}})

            do_PUT = do_DELETE = do_PATCH = altro

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Gestore)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = "http://127.0.0.1:%d" % self.server.server_address[1]

    def ferma(self):
        self.server.shutdown()
        self.server.server_close()


# ---------------------------------------------------------------------------------------------------------
# Lancio del programma
# ---------------------------------------------------------------------------------------------------------
def lancia(gpu, argomenti, timeout=60, env=None, **extra):
    comando = [sys.executable, "-I", TERMICO] + argomenti
    inizio = time.time()
    r = subprocess.run(comando, capture_output=True, text=True, encoding="utf-8", timeout=timeout,
                       env=env or gpu.ambiente(**extra), stdin=subprocess.DEVNULL)
    r.durata = time.time() - inizio
    return r


def con_gpu(gpu, argomenti):
    """Aggiunge --nvidia-smi dopo il comando (le opzioni appartengono al comando, non al programma)."""
    return argomenti[:1] + ["--nvidia-smi", gpu.file] + argomenti[1:]


def file_con(cartella, estensione):
    trovati = sorted(f for f in os.listdir(cartella) if f.endswith(estensione))
    return [os.path.join(cartella, f) for f in trovati]


def leggi_csv(percorso):
    with open(percorso, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def leggi_json(percorso):
    with open(percorso, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------------------
# Prove delle funzioni
# ---------------------------------------------------------------------------------------------------------
def campione(t, temp, mask=0x4, motore=None, watt=60.0, mhz=1500.0, util=95.0, mib=4000.0):
    return {"t": float(t), "istante": "2026-01-01T00:00:%02d" % (int(t) % 60), "temp": temp, "watt": watt,
            "mhz": mhz, "util": util, "mib": mib, "pstate": "P2", "mask": mask, "motore": motore}


def prove_funzioni(tm):
    print("\n-- funzioni --")
    controlla(tm.num("71") == 71.0 and tm.num("[N/A]") is None and tm.num("nan") is None and tm.num(None) is None
              and tm.num("inf") is None, "num: legge i numeri e scarta [N/A], nan e inf")
    controlla(tm.maschera("0x0000000000000024") == 0x24 and tm.maschera("") is None and tm.maschera("[N/A]") is None,
              "maschera: legge l'esadecimale e scarta il resto")
    controlla(tm.nomi_motivi(0x24) == "limite_potenza+termico_sw" and tm.nomi_motivi(0x1) == "" and
              tm.nomi_motivi(None) == "", "nomi_motivi: il bit 0x1 (GPU inattiva) non è un limite")
    controlla(tm.nomi_motivi(0xC0) == "termico_hw+freno_potenza_hw", "nomi_motivi: termico hardware e freno di potenza")

    retta = [(t, 50 + 0.1 * t) for t in range(0, 61)]
    controlla(abs(tm.pendenza_al_minuto(retta) - 6.0) < 1e-6, "pendenza_al_minuto: 0,1 C al secondo sono 6 C al minuto")
    controlla(abs(tm.pendenza_al_minuto([(t, 70.0) for t in range(0, 61)])) < 1e-9, "pendenza_al_minuto: temperatura ferma = 0")
    controlla(tm.pendenza_al_minuto(retta[:5]) is None and tm.pendenza_al_minuto([(t, 1) for t in range(0, 20)]) is None,
              "pendenza_al_minuto: pochi punti o meno di 30 secondi = nessuna pendenza")

    generiche = tm.calcola_fasce({}, None, None)
    controlla(generiche["attenzione_da_c"] == 80 and generiche["critica_da_c"] == 87 and generiche["indicativo"] is True
              and "generici" in generiche["origine"], "calcola_fasce: senza limiti dal driver, valori generici e indicativi")
    dal_driver = tm.calcola_fasce({"temp_rallentamento_c": 90.0, "temp_max_operativa_c": 87.0}, None, None)
    controlla(dal_driver["attenzione_da_c"] == 80 and dal_driver["critica_da_c"] == 87 and "dichiarato dal driver" in dal_driver["origine"],
              "calcola_fasce: dal limite di rallentamento: attenzione 10 C sotto, critica 3 C sotto")
    solo_massimo = tm.calcola_fasce({"temp_max_operativa_c": 87.0}, None, None)
    controlla(solo_massimo["attenzione_da_c"] == 77 and solo_massimo["critica_da_c"] == 84 and "GPU Max Operating Temp" in solo_massimo["origine"],
              "calcola_fasce: senza rallentamento usa la temperatura massima operativa")
    tue = tm.calcola_fasce({"temp_rallentamento_c": 90.0}, 70.0, 75.0)
    controlla(tue["attenzione_da_c"] == 70 and tue["critica_da_c"] == 75 and tue["origine"] == "indicate da te",
              "calcola_fasce: le soglie indicate dall'utente vincono")
    una = tm.calcola_fasce({"temp_rallentamento_c": 90.0}, None, 85.0)
    controlla(una["attenzione_da_c"] == 80 and una["critica_da_c"] == 85 and "in parte indicate da te" in una["origine"],
              "calcola_fasce: una sola soglia indicata, l'altra dal driver")

    f = generiche
    controlla([tm.fascia_di(x, f) for x in (None, 79.9, 80.0, 86.9, 87.0, 95)] ==
              [None, "normale", "attenzione", "attenzione", "critica", "critica"], "fascia_di: i confini sono inclusi verso l'alto")

    v = tm.valuta
    controlla(v("RIPOSO", 300, 40, 0, False, f) is None and v("RAFFREDDAMENTO", 300, 40, 0, False, f) is None,
              "valuta: riposo e raffreddamento non si giudicano")
    controlla(v("MOTORE", 170, 60, 0, False, f)["giudizio"] == "troppo_corta" and v("MOTORE", 180, 60, 0, False, f)["giudizio"] == "sostenibile",
              "valuta: meno di 3 minuti = troppo corta, da 3 minuti si giudica")
    controlla(v("MOTORE", 300, 70, 0, False, f)["giudizio"] == "sostenibile", "valuta: stabile sotto l'attenzione = sostenibile")
    controlla(v("MOTORE", 300, 70, 0, True, f)["giudizio"] == "da_verificare", "valuta: ancora in salita = da verificare")
    controlla(v("MOTORE", 300, 82, 0, False, f)["giudizio"] == "al_limite", "valuta: fascia di attenzione = al limite")
    controlla(v("MOTORE", 300, 70, 3, False, f)["giudizio"] == "al_limite", "valuta: un po' di throttling per il caldo = al limite")
    controlla(v("INSIEME", 300, 88, 0, False, f)["giudizio"] == "non_sostenibile", "valuta: fascia critica = non sostenibile")
    controlla(v("INSIEME", 300, 70, 20, False, f)["giudizio"] == "non_sostenibile", "valuta: throttling per il caldo dal 20% = non sostenibile")
    controlla(v("CHAT", 300, 70, 0, False, f)["indicativo"] is True and v("OSSERVAZIONE", 300, 70, 0, False, f) is not None,
              "valuta: il giudizio è indicativo e vale anche per l'osservazione")

    # una fase che sale da 45 a 93 gradi in 240 secondi (0,2 C al secondo), con il caldo che frena la GPU dopo i 180 secondi
    salita = [campione(t, 45 + t / 5.0, mask=(0x24 if t >= 180 else 0x4)) for t in range(0, 241)]
    r = tm.riassumi_fase("MOTORE", salita, f)
    controlla(r["temp_c"]["inizio"] == 45 and r["temp_c"]["fine"] == 93 and r["temp_c"]["max"] == 93, "riassumi_fase: temperatura iniziale, finale e massima")
    controlla(r["temp_c"]["variazione_dopo_60s"] == 12.0, "riassumi_fase: variazione dopo 60 secondi")
    controlla(abs(r["temp_c"]["pendenza_finale_c_min"] - 12.0) < 0.01 and r["ancora_in_salita"] is True,
              "riassumi_fase: pendenza finale 12 C al minuto = ancora in salita")
    controlla(r["throttling"]["motivi_pct"].get("termico_sw") == 25 and r["throttling"]["termico_pct"] == 25 and
              r["throttling"]["motivi_pct"].get("limite_potenza") == 100,
              "riassumi_fase: percentuali dei motivi di throttling e del throttling per il caldo (0x24 contiene anche il limite di potenza)")
    controlla(r["fasce"]["tempo_alla_fascia_attenzione_s"] == 175.0 and r["fasce"]["secondi_oltre_attenzione"] == 66.0 and
              r["fasce"]["secondi_oltre_critica"] == 31.0,
              "riassumi_fase: quando entra nelle fasce e per quanto resta (66 e 31 campioni da un secondo)")
    controlla(r["valutazione"]["giudizio"] == "non_sostenibile", "riassumi_fase: il giudizio tiene conto di fasce e throttling")
    controlla(tm.riassumi_fase("MOTORE", [], f) is None, "riassumi_fase: nessun campione = nessuna fase")
    # il throttling per il caldo è 0x20 (software) e 0x40 (hardware); il freno di potenza 0x80 non è caldo
    hw = [campione(t, 70.0, mask=(0x40 if t < 60 else 0x4)) for t in range(0, 241)]
    r = tm.riassumi_fase("MOTORE", hw, f)
    controlla(r["throttling"]["termico_pct"] == 24 and r["throttling"]["motivi_pct"]["termico_hw"] == 24 and
              r["valutazione"]["giudizio"] == "non_sostenibile", "riassumi_fase: il rallentamento termico hardware (0x40) conta come caldo")
    freno = [campione(t, 70.0, mask=0x80) for t in range(0, 241)]
    r = tm.riassumi_fase("MOTORE", freno, f)
    controlla(r["throttling"]["termico_pct"] == 0 and r["throttling"]["motivi_pct"] == {"freno_potenza_hw": 100} and
              r["valutazione"]["giudizio"] == "sostenibile", "riassumi_fase: il freno di potenza hardware (0x80) non è caldo, ma si vede fra i motivi")
    un_campione = [campione(t, 70.0, mask=(0x20 if t == 100 else 0x4)) for t in range(0, 241)]
    r = tm.riassumi_fase("MOTORE", un_campione, f)
    controlla(r["throttling"]["termico_pct"] == 1 and r["valutazione"]["giudizio"] == "al_limite",
              "riassumi_fase: anche un solo campione con il caldo che frena si vede (1%), e rende la fase 'al limite'")
    corta = [campione(t, 45 + t / 5.0) for t in range(0, 150)]
    r = tm.riassumi_fase("MOTORE", corta, f)
    controlla(r["ancora_in_salita"] is False and r["valutazione"]["giudizio"] == "troppo_corta",
              "riassumi_fase: sotto i 3 minuti non si dice 'ancora in salita': la fase è solo troppo corta")
    senza = [campione(t, None, mask=None, watt=None, mhz=None, util=None, mib=None) for t in range(0, 5)]
    r = tm.riassumi_fase("MOTORE", senza, f)
    controlla(r["temp_c"]["max"] is None and r["throttling"]["leggibile"] is False and r["potenza_w"]["media"] is None,
              "riassumi_fase: valori mancanti ([N/A]) restano mancanti, senza errori")
    stabile = [campione(t, 70.0, mask=0x4) for t in range(0, 241)]
    r = tm.riassumi_fase("MOTORE", stabile, f)
    controlla(r["ancora_in_salita"] is False and r["valutazione"]["giudizio"] == "sostenibile" and
              r["throttling"]["termico_pct"] == 0 and r["fasce"]["tempo_alla_fascia_attenzione_s"] is None,
              "riassumi_fase: temperatura stabile = sostenibile")

    def m(uptime, riuscite, fallite=0, occupato=True, ok=True):
        if not ok:
            return {"ok": False, "errore": "irraggiungibile"}
        return {"ok": True, "stato": "ok", "profilo": "max", "occupato": occupato, "coda": 1, "riuscite": riuscite,
                "fallite": fallite, "annullate": 0, "discese": 0, "ricaricamenti": 1, "uptime": uptime,
                "ram_disp_mib": 9000.0 - riuscite, "vram_libera_mib": 3500.0}
    fase = [campione(i, 70.0, motore=m(100 + i, 5 + i, fallite=(1 if i >= 3 else 0))) for i in range(0, 6)]
    r = tm.riassumi_motore(fase)
    controlla(r["job_riusciti"] == 5 and r["job_falliti"] == 1 and r["occupato_pct"] == 100 and r["raggiungibile_pct"] == 100 and
              r["profili"] == ["max"] and r["riavviato"] is False and r["ram_disponibile_min_mib"] == 8990.0,
              "riassumi_motore: job fatti durante la fase (differenza dei contatori), occupazione, minimi")
    riavviato = [campione(0, 70.0, motore=m(500, 40)), campione(1, 70.0, motore=m(501, 41)), campione(2, 70.0, motore=m(3, 0))]
    r = tm.riassumi_motore(riavviato)
    controlla(r["riavviato"] is True and r["job_riusciti"] is None, "riassumi_motore: server riavviato a metà fase = contatori non confrontabili")
    spento = [campione(i, 70.0, motore=m(0, 0, ok=False)) for i in range(0, 3)]
    controlla(tm.riassumi_motore(spento) == {"campioni": 3, "raggiungibile_pct": 0}, "riassumi_motore: motore che non risponde mai")
    meta = [campione(0, 70.0, motore=m(1, 0)), campione(1, 70.0, motore=m(0, 0, ok=False))]
    controlla(tm.riassumi_motore(meta)["raggiungibile_pct"] == 50, "riassumi_motore: motore che risponde a metà")
    controlla(tm.riassumi_motore([campione(0, 70.0)]) is None, "riassumi_motore: senza motore affiancato = nessun riepilogo del motore")

    controlla(tm.url_senza_credenziali("http://utente:segreto@127.0.0.1:1234/") == "http://127.0.0.1:1234" and
              tm.url_senza_credenziali("http://[::1]:80") == "http://[::1]:80" and tm.normalizza_url(" 127.0.0.1:1234/ ") == "http://127.0.0.1:1234",
              "url: niente credenziali nel riepilogo, indirizzi IPv6, schema aggiunto se manca")
    controlla(tm.nome_sicuro("../../etc/passwd") == "etc-passwd" and tm.nome_sicuro("portatile 3060") == "portatile-3060" and
              tm.nome_sicuro("///") == "", "nome_sicuro: il nome di una prova non può uscire dalla cartella")

    # CSV: scrittura e rilettura danno gli stessi campioni, anche con valori mancanti e con il motore
    cartella = tempfile.mkdtemp(prefix="csv-")
    try:
        percorso = os.path.join(cartella, "prova.csv")
        originali = [campione(0, 60.0, motore=m(100, 3)), campione(1, 61.0, mask=None, watt=None, motore={"ok": False, "errore": "x"}),
                     campione(2, 62.0, motore=None)]
        with open(percorso, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(tm.COLONNE_CSV)
            for c in originali:
                w.writerow(tm.riga_csv("MOTORE", c))
            fh.write("2026-01-01T00:00:03,MOTORE,3.0,6")  # riga troncata: il PC si è spento mentre scriveva
        lette = tm.campioni_da_csv(percorso)
        controlla(len(lette) == 1 and lette[0][0] == "MOTORE" and len(lette[0][1]) == 3, "CSV: la riga troncata in fondo viene scartata")
        a, b, c = lette[0][1]
        controlla(a["temp"] == 60.0 and a["motore"]["riuscite"] == 3 and a["motore"]["profilo"] == "max" and a["motore"]["occupato"] is True and
                  a["mask"] == 0x4 and a["motore"]["ram_disp_mib"] == 8997.0, "CSV: i valori del motore e della GPU tornano uguali")
        controlla(b["watt"] is None and b["mask"] is None and b["motore"] == {"ok": False, "errore": "irraggiungibile"} and c["motore"] is None,
                  "CSV: valore mancante = vuoto; motore irraggiungibile e motore non affiancato restano distinti")
        senza_intestazione = os.path.join(cartella, "altro.csv")
        with open(senza_intestazione, "w") as fh:
            fh.write("a,b\n1,2\n")
        try:
            tm.campioni_da_csv(senza_intestazione)
            rilevato = False
        except ValueError as errore:
            rilevato = "non è un CSV di termico.py" in str(errore)
        controlla(rilevato, "CSV: un file qualunque viene rifiutato con un messaggio chiaro")

        provvisorio = os.path.join(cartella, "x.json")
        tm.scrivi_json(provvisorio, {"a": 1})
        controlla(leggi_json(provvisorio) == {"a": 1} and not os.path.exists(provvisorio + ".tmp"),
                  "scrivi_json: scrive per intero e non lascia file provvisori")
    finally:
        shutil.rmtree(cartella, ignore_errors=True)

    # il quadro: tendenza della temperatura negli ultimi 60 secondi e avvisi
    info = {"indice": 0, "nome": "GPU di prova"}
    recenti = [campione(t, 60 + 0.1 * t) for t in range(0, 61)]
    quadro = tm.costruisci_quadro(info, f, recenti[-1], recenti)
    controlla(abs(quadro["tendenza_c_al_minuto"] - 6.0) < 0.01 and any("rapida salita" in a for a in quadro["avvisi"]) and
              quadro["fascia"]["nome"] == "normale", "costruisci_quadro: 6 C al minuto = tendenza e avviso di rapida salita")
    lenta = [campione(t, 60 + 0.01 * t) for t in range(0, 61)]
    quadro = tm.costruisci_quadro(info, f, lenta[-1], lenta)
    controlla(abs(quadro["tendenza_c_al_minuto"] - 0.6) < 0.01 and quadro["avvisi"] == [], "costruisci_quadro: una salita lenta non è un avviso")
    quadro = tm.costruisci_quadro(info, f, campione(0, 70.0), None)
    controlla(quadro["tendenza_c_al_minuto"] is None, "costruisci_quadro: con una sola lettura non c'è tendenza")
    vecchie = [campione(t, 40 + t) for t in range(0, 61)] + [campione(200 + t, 70.0) for t in range(0, 61)]
    quadro = tm.costruisci_quadro(info, f, vecchie[-1], vecchie)
    controlla(abs(quadro["tendenza_c_al_minuto"]) < 1e-6, "costruisci_quadro: la tendenza guarda solo gli ultimi 60 secondi")

    prova_a = {"formato": tm.FORMATO_RIEPILOGO, "nome": "prima", "creato": "x", "chiuso_regolarmente": True,
               "gpu": {"nome": "GPU", "driver": "1"},
               "fasi": [tm.riassumi_fase("MOTORE", [campione(t, 80.0, mhz=1400.0, watt=70.0) for t in range(0, 200)], f)]}
    prova_b = {"formato": tm.FORMATO_RIEPILOGO, "nome": "dopo", "creato": "y", "chiuso_regolarmente": True,
               "gpu": {"nome": "GPU", "driver": "1"},
               "fasi": [tm.riassumi_fase("MOTORE", [campione(t, 74.0, mhz=1550.0, watt=71.0) for t in range(0, 200)], f),
                        tm.riassumi_fase("CHAT", [campione(t, 60.0) for t in range(0, 20)], f)]}
    conf = tm.confronta_prove([prova_a, prova_b], ["A", "B"])
    controlla(conf["fasi"]["MOTORE"]["temp_max_c"] == [80.0, 74.0] and conf["fasi"]["MOTORE"]["clock_medio_mhz"] == [1400.0, 1550.0] and
              conf["fasi"]["CHAT"]["temp_max_c"] == [None, 60.0], "confronta_prove: valori per prova; una fase che manca in una prova resta vuota")
    testo = "\n".join(tm.testo_confronto(conf))
    controlla("-6" in testo and "+150" in testo and "differenza" in testo and "A = prima" in testo, "testo_confronto: differenza fra due prove")
    altra = json.loads(json.dumps(prova_b))
    altra["gpu"] = {"nome": "Altra GPU", "driver": "2"}
    testo = "\n".join(tm.testo_confronto(tm.confronta_prove([prova_a, altra, prova_b], ["A", "B", "C"])))
    controlla("GPU o driver diversi" in testo and "differenza" not in testo, "testo_confronto: GPU diverse = avviso; tre prove = niente colonna di differenza")


def prove_funzioni_segnala(tm):
    print("\n-- funzioni di --segnala --")
    L = tm.livello_segnale
    controlla(L("normale", 0x1) == 0 and L("normale", None) == 0, "livello_segnale: fascia normale = nessun segnale (il bit 0x1 non è un motivo)")
    controlla(L("normale", 0x20) == 1 and L("normale", 0x40) == 1,
              "livello_segnale: il driver che rallenta la GPU per il caldo vale un gradino anche in fascia normale")
    controlla(L("attenzione", 0x4) == 1 and L("attenzione", 0x20) == 2 and L("attenzione", 0x40) == 2,
              "livello_segnale: attenzione = 1, con throttling termico = 2")
    controlla(L("critica", 0x1) == 3 and L("critica", 0x20) == 3 and L("critica", 0x40) == 4,
              "livello_segnale: critica = 3, con throttling termico hardware = escluso (4)")
    controlla(L(None, 0x40) is None, "livello_segnale: temperatura illeggibile = nessuna segnalazione")
    motivo = tm.motivo_segnale({"temp": 86.0, "mask": 0x20}, "attenzione")
    controlla("86 C" in motivo and "attenzione" in motivo and "throttling termico" in motivo and "\n" not in motivo,
              "motivo_segnale: temperatura, fascia e throttling in una riga (%s)" % motivo)

    D = tm.dispositivo_del_motore
    portatile = [{"name": "CPU", "description": "cpu", "type": "cpu"},
                 {"name": "CUDA0", "description": NOME_GPU_FINTA, "type": "gpu"},
                 {"name": "Vulkan0", "description": NOME_GPU_FINTA, "type": "gpu"},
                 {"name": "Vulkan1", "description": "AMD Radeon(TM) Graphics", "type": "igpu"}]
    controlla(D(portatile, NOME_GPU_FINTA) == "CUDA0",
              "dispositivo_del_motore: la GPU si riconosce dal nome; vista da CUDA e da Vulkan va bene la prima (il motore passa il segnale all'altra)")
    controlla(D(portatile[:1] + portatile[2:], NOME_GPU_FINTA.lower()) == "Vulkan0",
              "dispositivo_del_motore: senza CUDA va bene Vulkan, e le maiuscole non contano")
    controlla(D(portatile, "NVIDIA RTX A2000") is None, "dispositivo_del_motore: una GPU che il motore non vede = nessun device")
    gemelle = [{"name": "CUDA0", "description": NOME_GPU_FINTA, "type": "gpu"}, {"name": "CUDA1", "description": NOME_GPU_FINTA, "type": "gpu"}]
    controlla(D(gemelle, NOME_GPU_FINTA) is None, "dispositivo_del_motore: due schede uguali non si distinguono dal nome: il device va indicato")
    controlla(D([], NOME_GPU_FINTA) is None and D(None, NOME_GPU_FINTA) is None, "dispositivo_del_motore: nessun elenco = nessun device")
    controlla(tm.durata_segnale(1.0) == 5 and tm.durata_segnale(0.3) == 5 and tm.durata_segnale(4.0) == 12,
              "durata_segnale: qualche campione, mai meno di 5 secondi")


# ---------------------------------------------------------------------------------------------------------
# Prove del programma, lanciato come lo lancerebbe una persona (o un altro programma)
# ---------------------------------------------------------------------------------------------------------
def prove_stato(gpu, base):
    print("\n-- stato --")
    gpu.imposta("temp", 60)
    gpu.imposta("mask", None)
    r = lancia(gpu, con_gpu(gpu, ["stato"]))
    d = json.loads(r.stdout)
    controlla(r.returncode == 0 and d["formato"] == "termico-stato/1" and d["fascia"]["nome"] == "normale" and
              d["fascia"]["indicativo"] is True and d["motore"] is None and d["gpu"]["temp_c"] == 60 and
              d["gpu"]["nome"].startswith("NVIDIA") and d["gpu"]["motivi_throttling"] == [] and d["avvisi"] == [],
              "stato: 60 C = fascia normale, codice di uscita 0, JSON completo")
    controlla(d["fascia"]["attenzione_da_c"] == 80 and d["fascia"]["critica_da_c"] == 87 and "generici" in d["fascia"]["origine"],
              "stato: senza limiti dal driver usa le fasce generiche e lo dice")
    controlla(abs(d["istante_unix"] - time.time()) < 30, "stato: istante_unix permette di riconoscere una lettura vecchia senza leggere l'ora locale")
    gpu.imposta("temp", 82)
    r = lancia(gpu, con_gpu(gpu, ["stato"]))
    controlla(r.returncode == 1 and json.loads(r.stdout)["fascia"]["nome"] == "attenzione", "stato: 82 C = attenzione, codice di uscita 1")
    gpu.imposta("temp", 88)
    gpu.imposta("mask", "0000000000000060")
    r = lancia(gpu, con_gpu(gpu, ["stato"]))
    d = json.loads(r.stdout)
    controlla(r.returncode == 2 and d["fascia"]["nome"] == "critica" and d["gpu"]["motivi_throttling"] == ["termico_sw", "termico_hw"] and
              "temperatura nella fascia critica" in d["avvisi"] and "il driver sta rallentando la GPU per il caldo" in d["avvisi"],
              "stato: 88 C = critica, codice 2, throttling per il caldo segnalato fra gli avvisi")
    gpu.imposta("mask", None)
    gpu.imposta("temp", 85)
    r = lancia(gpu, con_gpu(gpu, ["stato"]), FAKE_LIMITI="1")
    d = json.loads(r.stdout)
    controlla(r.returncode == 1 and d["fascia"]["attenzione_da_c"] == 80 and d["fascia"]["critica_da_c"] == 87 and
              "dichiarato dal driver" in d["fascia"]["origine"] and "GPU Slowdown Temp 90" in d["fascia"]["origine"],
              "stato: i limiti letti con nvidia-smi -q -d TEMPERATURE (rallentamento 90 C) danno le fasce 80 e 87")
    gpu.imposta("temp", 65)
    r = lancia(gpu, con_gpu(gpu, ["stato", "--attenzione", "60", "--critica", "70"]))
    d = json.loads(r.stdout)
    controlla(r.returncode == 1 and d["fascia"]["origine"] == "indicate da te", "stato: --attenzione e --critica vincono sui limiti")
    r = lancia(gpu, con_gpu(gpu, ["stato", "--attenzione", "90", "--critica", "80"]))
    controlla(r.returncode == 3 and "più bassa" in r.stderr, "stato: soglie incoerenti = uso sbagliato, codice 3 (il 2 è la fascia critica)")
    r = lancia(gpu, con_gpu(gpu, ["stato", "--non-esiste"]))
    controlla(r.returncode == 3, "stato: un'opzione sconosciuta esce con 3, non con 2")
    r = lancia(gpu, con_gpu(gpu, ["osserva", "--ogni", "0.05"]))
    controlla(r.returncode == 3 and "0,2 secondi" in r.stderr, "osserva: intervalli sotto 0,2 secondi rifiutati")
    un_file = os.path.join(base, "un-file")
    with open(un_file, "w") as fh:
        fh.write("x")
    r = lancia(gpu, con_gpu(gpu, ["osserva", "--durata", "1", "--uscita", os.path.join(un_file, "sotto")]))
    controlla(r.returncode == 1 and "Non riesco a scrivere" in r.stderr and "Traceback" not in r.stderr,
              "osserva: una cartella di uscita impossibile = messaggio chiaro, senza traceback")

    r = lancia(gpu, ["stato", "--nvidia-smi", os.path.join(base, "non-esiste")])
    d = json.loads(r.stdout)
    controlla(r.returncode == 3 and "errore" in d and "nvidia-smi non trovato" in d["errore"],
              "stato: nvidia-smi assente = codice 3 e un JSON con l'errore (chi legge trova sempre JSON)")
    r = lancia(gpu, con_gpu(gpu, ["stato"]), FAKE_MODE="dead")
    controlla(r.returncode == 3 and "errore" in json.loads(r.stdout), "stato: GPU che non risponde = codice 3 e JSON con l'errore")

    r = lancia(gpu, con_gpu(gpu, ["stato"]), FAKE_MODE="nomask")
    controlla(r.returncode == 0 and json.loads(r.stdout)["gpu"]["motivi_throttling"] is None,
              "stato: driver che non dice i motivi di throttling = null, non un elenco vuoto")
    r = lancia(gpu, con_gpu(gpu, ["stato"]), FAKE_MODE="evt")
    controlla("limite_potenza" in json.loads(r.stdout)["gpu"]["motivi_throttling"],
              "stato: i driver nuovi (clocks_event_reasons) si leggono come i vecchi")
    r = lancia(gpu, con_gpu(gpu, ["stato"]), FAKE_POTENZA_NA="1")
    controlla(json.loads(r.stdout)["gpu"]["potenza_w"] is None, "stato: potenza [N/A] = null")


def prove_stato_motore(gpu, base):
    print("\n-- stato con il motore --")
    gpu.imposta("temp", 60)
    motore = MotoreFinto()
    try:
        r = lancia(gpu, con_gpu(gpu, ["stato", "--motore", motore.url]))
        d = json.loads(r.stdout)
        m = d["motore"]
        controlla(r.returncode == 0 and m["raggiungibile"] is True and m["stato"] == "ok" and m["profilo"] == "max" and m["occupato"] is True and
                  m["coda"] == 1 and m["ram_disponibile_mib"] == 8704.0 and m["vram_libera_mib"] == 3584.0 and d["avvisi"] == [],
                  "stato --motore: stato, profilo, coda, RAM e VRAM libere (la VRAM è quella della GPU, non della CPU)")
        r = lancia(gpu, con_gpu(gpu, ["stato", "--motore", motore.url.replace("http://", "")]))
        controlla(json.loads(r.stdout)["motore"]["raggiungibile"] is True, "stato --motore: l'indirizzo senza http:// funziona")
        richieste = set(motore.richieste)
        controlla(richieste <= {("GET", "/sdcpp/v1/health"), ("GET", "/sdcpp/v1/resources")} and len(richieste) == 2,
                  "solo lettura: al motore arrivano soltanto GET di /health e /resources (%s)" % sorted(richieste))
        motore.stato = "unavailable"
        r = lancia(gpu, con_gpu(gpu, ["stato", "--motore", motore.url]))
        d = json.loads(r.stdout)
        controlla(d["motore"]["stato"] == "unavailable" and "il motore è in stato unavailable" in d["avvisi"],
                  "stato --motore: il 503 di /health ha un corpo valido e si legge (stato unavailable)")
    finally:
        motore.ferma()
    chiuso = lancia(gpu, con_gpu(gpu, ["stato", "--motore", motore.url]))
    d = json.loads(chiuso.stdout)
    controlla(d["motore"] == {"raggiungibile": False, "errore": "irraggiungibile"} and
              any("non è raggiungibile" in a for a in d["avvisi"]) and chiuso.returncode == 0,
              "stato --motore: motore spento = raggiungibile false e un avviso, la lettura della GPU resta valida")

    protetto = MotoreFinto(chiave="segreto-123")
    try:
        r = lancia(gpu, con_gpu(gpu, ["stato", "--motore", protetto.url]))
        controlla(json.loads(r.stdout)["motore"]["errore"] == "chiave API mancante o errata", "chiave API: senza chiave il motore risponde 401 e lo si dice")
        r = lancia(gpu, con_gpu(gpu, ["stato", "--motore", protetto.url, "--chiave-api", "sbagliata"]))
        controlla(json.loads(r.stdout)["motore"]["raggiungibile"] is False, "chiave API: una chiave sbagliata non passa")
        r = lancia(gpu, con_gpu(gpu, ["stato", "--motore", protetto.url, "--chiave-api", "segreto-123"]))
        controlla(json.loads(r.stdout)["motore"]["raggiungibile"] is True, "chiave API: con --chiave-api passa")
        r = lancia(gpu, con_gpu(gpu, ["stato", "--motore", protetto.url]), TERMICO_CHIAVE_API="segreto-123")
        controlla(json.loads(r.stdout)["motore"]["raggiungibile"] is True and "segreto-123" not in r.stdout + r.stderr,
                  "chiave API: anche dalla variabile TERMICO_CHIAVE_API, e non compare nell'uscita")
    finally:
        protetto.ferma()


class LettoreDiStato(threading.Thread):
    """Legge in continuazione il file di stato, come farebbe un altro programma: non deve mai trovarlo a metà."""

    def __init__(self, percorso):
        super().__init__(daemon=True)
        self.percorso = percorso
        self.fermo = threading.Event()
        self.letture = 0
        self.rotte = 0

    def run(self):
        while not self.fermo.is_set():
            try:
                with open(self.percorso, encoding="utf-8") as f:
                    testo = f.read()
            except OSError:
                continue
            try:
                json.loads(testo)
                self.letture += 1
            except ValueError:
                self.rotte += 1


def prove_osserva(gpu, base):
    print("\n-- osserva --")
    cartella = os.path.join(base, "uscita", "annidata")
    file_stato = os.path.join(base, "stato.json")
    gpu.imposta("temp", None)
    gpu.imposta("mask", None)
    gpu.riparti()
    motore = MotoreFinto()
    lettore = LettoreDiStato(file_stato)
    lavoro = os.path.join(base, "cartella-di-lavoro")
    os.makedirs(lavoro)
    try:
        lettore.start()
        comando = [sys.executable, "-I", TERMICO, "osserva", "--nvidia-smi", gpu.file, "--durata", "5", "--ogni", "0.5",
                   "--ogni-motore", "0.5", "--uscita", cartella, "--nome", "../prova 1", "--file-stato", file_stato,
                   "--motore", motore.url]
        r = subprocess.run(comando, capture_output=True, text=True, encoding="utf-8", timeout=60, env=gpu.ambiente(),
                           stdin=subprocess.DEVNULL, cwd=lavoro)
    finally:
        lettore.fermo.set()
        lettore.join(5)
        motore.ferma()
    controlla(r.returncode == 0 and "RIEPILOGO" in r.stdout and "Fasce di temperatura (INDICATIVE" in r.stdout and
              "MOTORE (sd-server) durante le fasi" in r.stdout and "motivi di throttling (% del tempo)" in r.stdout,
              "osserva: termina da sola dopo --durata e stampa il riepilogo con il motore")
    controlla(os.listdir(lavoro) == [] and os.path.exists(os.path.join(base, "uscita")) and not os.path.exists(os.path.join(base, "prova-1")),
              "osserva: scrive solo nella cartella di uscita (creata anche se annidata); un nome come ../prova non esce dalla cartella")
    csv_files, json_files, txt_files = (file_con(cartella, e) for e in (".csv", ".json", ".txt"))
    controlla(len(csv_files) == len(json_files) == len(txt_files) == 1 and os.path.basename(csv_files[0]).startswith("termico-prova-1-"),
              "osserva: un .csv, un .json e un .txt con lo stesso nome (%s)" % [os.path.basename(x) for x in csv_files])
    righe = leggi_csv(csv_files[0])
    tm = carica_modulo()
    controlla(list(righe[0].keys()) == tm.COLONNE_CSV, "osserva: le colonne del CSV sono quelle documentate")
    controlla(len(righe) >= 8 and {x["fase"] for x in righe} == {"OSSERVAZIONE"}, "osserva: un campione ogni 0,5 secondi per 5 secondi (%d righe)" % len(righe))
    controlla(all(x["motore_raggiungibile"] == "1" for x in righe) and righe[-1]["motore_stato"] == "ok" and righe[-1]["motore_profilo"] == "max" and
              righe[-1]["ram_disponibile_mib"] == "8704" and righe[-1]["vram_libera_mib"] == "3584" and righe[-1]["motore_occupato"] == "1",
              "osserva: nel CSV le colonne del motore sono sulla stessa riga della GPU")
    controlla(righe[0]["temp_c"] != "" and righe[0]["potenza_w"] != "" and righe[0]["pstate"] == "P2" and righe[0]["motivi_hex"] == "0x1" and
              righe[0]["motivi"] == "", "osserva: i valori della GPU sono nel CSV (il bit 0x1 non è un motivo)")
    riepilogo = leggi_json(json_files[0])
    fase = riepilogo["fasi"][0]
    controlla(riepilogo["formato"] == "termico-riepilogo/1" and riepilogo["chiuso_regolarmente"] is True and riepilogo["nome"] == "../prova 1" and
              riepilogo["gpu"]["nome"].startswith("NVIDIA") and riepilogo["gpu"]["limite_potenza_w"] == 80.0 and
              riepilogo["gpu"]["clock_max_mhz"] == 2100.0 and riepilogo["fasce"]["indicativo"] is True and
              riepilogo["fasi_previste"] == ["OSSERVAZIONE"] and fase["nome"] == "OSSERVAZIONE",
              "osserva: il JSON ha formato, GPU, fasce indicative e la fase")
    controlla(riepilogo["motore"]["url"] == motore.url and riepilogo["motore"]["raggiungibile_all_avvio"] is True and
              fase["motore"]["raggiungibile_pct"] == 100 and fase["motore"]["job_riusciti"] >= 1 and fase["motore"]["occupato_pct"] == 100,
              "osserva: il JSON riporta il motore (indirizzo, job fatti durante la fase)")
    controlla(fase["valutazione"]["giudizio"] == "troppo_corta" and fase["valutazione"]["indicativo"] is True,
              "osserva: 5 secondi sono troppo pochi per un giudizio, e il programma lo dice")
    with open(txt_files[0], encoding="utf-8") as f:
        controlla("OSSERVAZIONE" in f.read(), "osserva: il .txt contiene il riepilogo")
    stato = leggi_json(file_stato)
    controlla(stato["formato"] == "termico-stato/1" and stato["gpu"]["temp_c"] is not None and stato["motore"]["raggiungibile"] is True and
              "tendenza_c_al_minuto" in stato and stato["fascia"]["indicativo"] is True,
              "osserva: il file di stato ha l'ultima lettura, la fascia e il motore")
    controlla(lettore.letture >= 5 and lettore.rotte == 0,
              "osserva: il file di stato non si vede mai a metà (%d letture, %d rotte)" % (lettore.letture, lettore.rotte))
    controlla(not os.path.exists(file_stato + ".tmp"), "osserva: nessun file provvisorio rimasto")

    gpu.riparti()
    cartella = os.path.join(base, "uscita2")
    r = lancia(gpu, con_gpu(gpu, ["osserva", "--durata", "3", "--ogni", "0.5", "--uscita", cartella]), FAKE_POTENZA_NA="1")
    riepilogo = leggi_json(file_con(cartella, ".json")[0])
    righe = leggi_csv(file_con(cartella, ".csv")[0])
    controlla(r.returncode == 0 and riepilogo["motore"] is None and riepilogo["fasi"][0]["motore"] is None and
              all(x["motore_raggiungibile"] == "" for x in righe) and all(x["potenza_w"] == "" for x in righe) and
              riepilogo["fasi"][0]["potenza_w"]["media"] is None and "n/d" in r.stdout,
              "osserva: senza --motore le colonne del motore restano vuote; potenza [N/A] = vuoto nel CSV e n/d nel testo")


def prove_osserva_guasti(gpu, base):
    print("\n-- osserva: guasti --")
    gpu.imposta("temp", None)
    # GPU che risponde una volta su tre
    gpu.riparti()
    gpu.imposta("letture", None)
    cartella = os.path.join(base, "flaky")
    r = lancia(gpu, con_gpu(gpu, ["osserva", "--durata", "4", "--ogni", "0.3", "--uscita", cartella]), FAKE_MODE="flaky")
    righe = leggi_csv(file_con(cartella, ".csv")[0])
    controlla(r.returncode == 0 and 6 <= len(righe) <= 10, "GPU con letture che falliscono ogni tanto: la registrazione va avanti (%d righe)" % len(righe))

    # GPU che smette di rispondere
    gpu.riparti()
    cartella = os.path.join(base, "morta")
    r = lancia(gpu, con_gpu(gpu, ["osserva", "--ogni", "0.2", "--uscita", cartella]), FAKE_MODE="dead")
    riepilogo = leggi_json(file_con(cartella, ".json")[0])
    controlla(r.returncode == 1 and "nvidia-smi non risponde" in r.stdout and riepilogo["chiuso_regolarmente"] is True and riepilogo["fasi"] == [],
              "GPU che non risponde: dopo 10 letture fallite di fila si ferma con un messaggio, codice 1, i file restano validi")

    # motore che sparisce a metà
    gpu.riparti()
    motore = MotoreFinto()
    cartella = os.path.join(base, "motore-spento")
    timer = threading.Timer(2.5, motore.ferma)
    timer.start()
    try:
        r = lancia(gpu, con_gpu(gpu, ["osserva", "--durata", "6", "--ogni", "0.5", "--ogni-motore", "0.5", "--uscita", cartella,
                                      "--motore", motore.url]))
    finally:
        timer.cancel()
    righe = leggi_csv(file_con(cartella, ".csv")[0])
    stati = [x["motore_raggiungibile"] for x in righe]
    riepilogo = leggi_json(file_con(cartella, ".json")[0])
    controlla(r.returncode == 0 and len(righe) >= 10 and stati[0] == "1" and stati[-1] == "0" and
              0 < riepilogo["fasi"][0]["motore"]["raggiungibile_pct"] < 100,
              "motore che si spegne a metà: la GPU si registra lo stesso, il CSV mostra quando è sparito (%s)" % "".join(stati))

    # motore spento fin dall'inizio
    gpu.riparti()
    cartella = os.path.join(base, "motore-assente")
    r = lancia(gpu, con_gpu(gpu, ["osserva", "--durata", "3", "--ogni", "0.5", "--ogni-motore", "0.5", "--uscita", cartella,
                                  "--motore", motore.url]))
    righe = leggi_csv(file_con(cartella, ".csv")[0])
    riepilogo = leggi_json(file_con(cartella, ".json")[0])
    controlla(r.returncode == 0 and "non risponde" in r.stdout and all(x["motore_raggiungibile"] == "0" for x in righe) and
              riepilogo["motore"]["raggiungibile_all_avvio"] is False,
              "motore mai raggiungibile: si dice all'inizio, il CSV riporta 0 su ogni riga, la GPU è registrata")

    # chiave API: usata, mai scritta
    protetto = MotoreFinto(chiave="segreto-123")
    try:
        gpu.riparti()
        cartella = os.path.join(base, "con-chiave")
        r = lancia(gpu, con_gpu(gpu, ["osserva", "--durata", "3", "--ogni", "0.5", "--ogni-motore", "0.5", "--uscita", cartella,
                                      "--motore", protetto.url, "--chiave-api", "segreto-123"]))
        righe = leggi_csv(file_con(cartella, ".csv")[0])
        tutto = r.stdout + r.stderr + "".join(open(f, encoding="utf-8").read() for e in (".csv", ".json", ".txt") for f in file_con(cartella, e))
        controlla(all(x["motore_raggiungibile"] == "1" for x in righe) and "segreto-123" not in tutto,
                  "chiave API: il motore risponde e la chiave non compare né nell'uscita né nei file")
        gpu.riparti()
        cartella = os.path.join(base, "senza-chiave")
        r = lancia(gpu, con_gpu(gpu, ["osserva", "--durata", "3", "--ogni", "0.5", "--ogni-motore", "0.5", "--uscita", cartella,
                                      "--motore", protetto.url]))
        controlla("chiave API mancante o errata" in r.stdout, "chiave API: se manca, il programma lo dice invece di tacere")
    finally:
        protetto.ferma()


def osserva_con_passi(gpu, argomenti, passi, durata_passo, env=None, timeout=90):
    """Lancia osserva e intanto cambia la GPU finta: passi è una lista di (temperatura, maschera esadecimale)."""
    fermo = threading.Event()

    def cambia():
        for temp, mask in passi:
            gpu.imposta("temp", temp)
            gpu.imposta("mask", mask.zfill(16))
            if fermo.wait(durata_passo):
                return

    temp, mask = passi[0]
    gpu.imposta("temp", temp)
    gpu.imposta("mask", mask.zfill(16))
    gpu.riparti()
    comando = [sys.executable, "-I", TERMICO, "osserva", "--nvidia-smi", gpu.file] + argomenti
    p = subprocess.Popen(comando, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                         env=env or gpu.ambiente(), stdin=subprocess.DEVNULL)
    cambiatore = threading.Thread(target=cambia, daemon=True)
    cambiatore.start()
    try:
        uscita, errori_uscita = p.communicate(timeout=timeout)
    finally:
        fermo.set()
        cambiatore.join(5)
        gpu.imposta("mask", None)
        gpu.imposta("temp", None)
    return p.returncode, uscita, errori_uscita


def livelli_di(segnali):
    """I livelli ricevuti, senza ripetizioni di fila."""
    livelli = []
    for _, corpo in segnali:
        if not livelli or livelli[-1] != corpo["level"]:
            livelli.append(corpo["level"])
    return livelli


def prove_segnala(gpu, base):
    print("\n-- osserva --segnala --")
    r = lancia(gpu, con_gpu(gpu, ["osserva", "--segnala", "--durata", "1", "--uscita", os.path.join(base, "segnala-0")]))
    controlla(r.returncode != 0 and "--motore" in r.stderr, "--segnala senza --motore viene rifiutato")

    # Tutte le fasce, una dopo l'altra, e di nuovo normale. La GPU finta legge temperatura e maschera da due file:
    # ogni passo cambia solo ciò che fra due passi vicini non dà un livello diverso a metà scrittura.
    motore = MotoreFinto(descrizione_gpu=NOME_GPU_FINTA)
    try:
        passi = [(60, "1"), (82, "4"), (82, "20"), (88, "1"), (88, "40"), (88, "1"), (60, "1")]
        codice, uscita, _ = osserva_con_passi(gpu, ["--segnala", "--motore", motore.url, "--ogni", "0.3", "--durata", "12",
                                                    "--uscita", os.path.join(base, "segnala-1")], passi, 1.5)
        segnali = list(motore.segnali)
    finally:
        motore.ferma()
    livelli = livelli_di(segnali)
    controlla(codice == 0 and livelli == [1, 2, 3, "excluded", 3, 0],
              "--segnala: attenzione 1, con throttling termico 2, critica 3, con throttling hardware escluso, di nuovo 3, poi il ritiro (%s)" % livelli)
    controlla(segnali and all(d == "CUDA0" for d, _ in segnali), "--segnala: il device si trova dal nome della GPU fra quelli del motore")
    attivi = [corpo for _, corpo in segnali if corpo["level"] != 0]
    controlla(attivi and all(corpo["source"] == "termico" and corpo["duration_seconds"] == 5 for corpo in attivi),
              "--segnala: fonte termico, durata di qualche campione (5 s)")
    controlla(sum(1 for corpo in attivi if corpo["level"] == 1) >= 2, "--segnala: il segnale si ripete a ogni campione finché la fascia resta")
    doppio = next((corpo for corpo in attivi if corpo["level"] == 2), {})
    controlla("82 C" in doppio.get("reason", "") and "throttling termico" in doppio.get("reason", ""),
              "--segnala: il motivo dice temperatura e throttling (%s)" % doppio.get("reason"))
    controlla(segnali and segnali[-1][1] == {"level": 0, "source": "termico"} and sum(1 for _, c in segnali if c["level"] == 0) == 1,
              "--segnala: tornata normale, il segnale si ritira una volta sola")
    controlla("CUDA0" in uscita and "segnal" in uscita.lower(), "--segnala: all'inizio dice a quale device del motore segnala")

    # Device indicato, chiave API, ritiro alla fine.
    protetto = MotoreFinto(chiave="k-77", descrizione_gpu=NOME_GPU_FINTA)
    try:
        codice, uscita, errori_uscita = osserva_con_passi(
            gpu, ["--segnala", "Vulkan0", "--motore", protetto.url, "--chiave-api", "k-77", "--ogni", "0.3", "--durata", "3",
                  "--uscita", os.path.join(base, "segnala-2")], [(82, "4")], 3)
        segnali = list(protetto.segnali)
    finally:
        protetto.ferma()
    controlla(codice == 0 and segnali and all(d == "Vulkan0" for d, _ in segnali) and livelli_di(segnali) == [1, 0],
              "--segnala Vulkan0: si usa il device indicato, e alla fine il segnale si ritira (%s)" % livelli_di(segnali))
    controlla("k-77" not in uscita + errori_uscita, "--segnala: la chiave API passa e non compare nell'uscita")

    protetto = MotoreFinto(chiave="k-77", descrizione_gpu=NOME_GPU_FINTA)
    try:
        codice, uscita, _ = osserva_con_passi(gpu, ["--segnala", "--motore", protetto.url, "--ogni", "0.3", "--durata", "3",
                                                    "--uscita", os.path.join(base, "segnala-3")], [(82, "4")], 3)
    finally:
        protetto.ferma()
    controlla(codice == 0 and uscita.count("Segnalazione al motore rifiutata: chiave API mancante o errata") == 1,
              "--segnala senza la chiave: il programma lo dice, una volta, e continua a misurare")

    # Il motore non conosce la GPU, o il device indicato.
    estraneo = MotoreFinto()
    try:
        codice, uscita, _ = osserva_con_passi(gpu, ["--segnala", "--motore", estraneo.url, "--ogni", "0.3", "--durata", "3",
                                                    "--uscita", os.path.join(base, "segnala-4")], [(82, "4")], 3)
        nessuno = list(estraneo.segnali)
        codice_ignoto, uscita_ignoto, _ = osserva_con_passi(
            gpu, ["--segnala", "Nessuno9", "--motore", estraneo.url, "--ogni", "0.3", "--durata", "3",
                  "--uscita", os.path.join(base, "segnala-5")], [(82, "4")], 3)
    finally:
        estraneo.ferma()
    controlla(codice == 0 and not nessuno and "--segnala NOME" in uscita and NOME_GPU_FINTA in uscita,
              "--segnala: una GPU che il motore non elenca = nessun segnale, e il programma dice come indicarla")
    controlla(codice_ignoto == 0 and uscita_ignoto.count("Nessuno9") >= 1 and uscita_ignoto.count("non conosce") == 1,
              "--segnala Nessuno9: il motore non lo conosce, il programma lo dice una volta e continua")

    # Un motore lento non rallenta le letture della GPU.
    lento = MotoreFinto(descrizione_gpu=NOME_GPU_FINTA, ritardo_segnali=2.0)
    cartella = os.path.join(base, "segnala-6")
    try:
        codice, _, _ = osserva_con_passi(gpu, ["--segnala", "--motore", lento.url, "--ogni", "0.3", "--durata", "4",
                                               "--uscita", cartella], [(88, "1")], 4)
    finally:
        lento.ferma()
    righe = leggi_csv(file_con(cartella, ".csv")[0])
    controlla(codice == 0 and len(righe) >= 10, "--segnala con un motore lento: i campioni della GPU non rallentano (%d righe in 4 s)" % len(righe))


def prove_segnali(gpu, base):
    if os.name == "nt":
        print("\n-- segnali: saltate su Windows --")
        return
    print("\n-- segnali e interruzioni --")
    gpu.imposta("temp", None)
    gpu.riparti()
    cartella = os.path.join(base, "sigterm")
    comando = [sys.executable, "-I", TERMICO, "osserva", "--nvidia-smi", gpu.file, "--ogni", "0.5", "--uscita", cartella]
    p = subprocess.Popen(comando, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                         env=gpu.ambiente(), stdin=subprocess.DEVNULL)
    time.sleep(3.5)
    p.send_signal(signal.SIGTERM)
    uscita, _ = p.communicate(timeout=30)
    riepilogo = leggi_json(file_con(cartella, ".json")[0])
    controlla(p.returncode == 0 and "Interrotto." in uscita and riepilogo["chiuso_regolarmente"] is True and
              riepilogo["fasi"][0]["campioni"] >= 4, "SIGTERM: la prova si chiude, il riepilogo viene scritto (%d campioni)" % riepilogo["fasi"][0]["campioni"])

    gpu.riparti()
    cartella = os.path.join(base, "sigkill")
    comando[-1] = cartella
    p = subprocess.Popen(comando, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=gpu.ambiente(), stdin=subprocess.DEVNULL)
    time.sleep(3.5)
    p.kill()
    p.wait()
    csv_file, json_file = file_con(cartella, ".csv")[0], file_con(cartella, ".json")[0]
    controlla(len(leggi_csv(csv_file)) >= 4 and leggi_json(json_file)["chiuso_regolarmente"] is False,
              "processo ucciso (come un PC che si spegne): il CSV è completo fino all'ultimo campione e il JSON dice che non è stato chiuso")
    r = lancia(gpu, ["riepilogo", csv_file, "--json", os.path.join(cartella, "ricostruito.json")])
    ricostruito = leggi_json(os.path.join(cartella, "ricostruito.json"))
    controlla(r.returncode == 0 and "OSSERVAZIONE" in r.stdout and "non è stata chiusa regolarmente" in r.stdout and
              ricostruito["fasi"][0]["campioni"] >= 4 and ricostruito["chiuso_regolarmente"] is False and
              ricostruito["gpu"]["nome"].startswith("NVIDIA"),
              "riepilogo: ricostruisce la prova dal CSV, prende GPU e fasce dal .json accanto e avverte che non era chiusa")
    chiusa = os.path.join(base, "uscita", "annidata")
    r = lancia(gpu, ["riepilogo", file_con(chiusa, ".csv")[0]])
    controlla(r.returncode == 0 and "non è stata chiusa regolarmente" not in r.stdout and "MOTORE (sd-server)" in r.stdout,
              "riepilogo: una prova chiusa regolarmente non porta l'avvertimento, e mostra il motore")
    r = lancia(gpu, ["riepilogo", file_con(chiusa, ".json")[0]])
    controlla(r.returncode == 0 and "OSSERVAZIONE" in r.stdout, "riepilogo: accetta anche un .json")
    r = lancia(gpu, ["riepilogo", os.path.join(base, "non-esiste.csv")])
    controlla(r.returncode != 0 and "non-esiste.csv" in r.stderr, "riepilogo: un file che non c'è = messaggio con il nome del file")


def prova_guidata(gpu, base):
    print("\n-- sessione guidata --")
    r = lancia(gpu, ["--nvidia-smi", gpu.file, "--fasi", "riposo", "--uscita", os.path.join(base, "vecchio")])
    controlla(r.returncode == 0 and "Interrotto." in r.stdout and "RIEPILOGO" in r.stdout and
              leggi_json(file_con(os.path.join(base, "vecchio"), ".json")[0])["chiuso_regolarmente"] is True,
              "senza comando parte la sessione guidata (le opzioni della prima versione funzionano ancora); a input finito si chiude con ordine")
    r = lancia(gpu, ["guidata", "--nvidia-smi", gpu.file, "--fasi", "riposo,sconosciuta"])
    controlla(r.returncode != 0 and "sconosciuta" in r.stderr, "guidata: una fase sconosciuta viene rifiutata")
    if os.name == "nt":
        print("saltata su Windows: la sessione guidata con un pseudo-terminale")
        return
    import pty
    import select
    cartella = os.path.join(base, "guidata")
    gpu.imposta("temp", None)
    gpu.riparti()
    comando = [sys.executable, "-I", TERMICO, "guidata", "--nvidia-smi", gpu.file, "--fasi", "riposo,motore", "--auto-secondi", "4",
               "--ogni", "0.5", "--uscita", cartella, "--nome", "guida"]
    pid, fd = pty.fork()
    if pid == 0:
        os.execve(sys.executable, comando, gpu.ambiente())
    testo = ""
    passo = 0
    quando = {}
    inizio = time.time()
    while time.time() - inizio < 60:
        pronto, _, _ = select.select([fd], [], [], 0.1)
        if pronto:
            try:
                dati = os.read(fd, 4096)
            except OSError:
                break
            if not dati:
                break
            testo += dati.decode("utf-8", "replace")
        ora = time.time()
        if passo == 0 and "Premi Invio per iniziare" in testo:
            os.write(fd, b"\n")
            quando["riposo"] = ora
            passo = 1
        elif passo == 1 and ora - quando["riposo"] >= 1.5:
            os.write(fd, b"\n")  # Invio sbagliato durante la fase automatica: non deve fermarla
            passo = 2
        elif passo == 2 and "Quando il carico è partito" in testo:
            quando["fine_riposo"] = ora
            os.write(fd, b"\n")
            quando["motore"] = ora
            passo = 3
        elif passo == 3 and ora - quando["motore"] >= 3.0:
            os.write(fd, b"\n")
            passo = 4
    _, stato = os.waitpid(pid, 0)
    riepilogo = leggi_json(file_con(cartella, ".json")[0])
    nomi = [f["nome"] for f in riepilogo["fasi"]]
    controlla(os.waitstatus_to_exitcode(stato) == 0 and nomi == ["RIPOSO", "MOTORE"] and riepilogo["fasi_previste"] == ["RIPOSO", "MOTORE"] and
              riepilogo["chiuso_regolarmente"] is True, "guidata: due fasi registrate, in ordine, e prova chiusa (%s)" % nomi)
    controlla(quando.get("fine_riposo", 0) - quando.get("riposo", 1e9) >= 3.5 and riepilogo["fasi"][0]["durata_s"] >= 3.0,
              "guidata: un Invio durante la fase automatica non la ferma (durata %.1f s di 4)" % riepilogo["fasi"][0]["durata_s"])
    controlla(riepilogo["fasi"][1]["durata_s"] >= 2.0 and "=== Fase 1/2: RIPOSO ===" in testo and "=== Fase 2/2: MOTORE ===" in testo and
              "Registrazione ferma" in testo and "RIEPILOGO" in testo, "guidata: la fase a mano parte e si ferma con Invio; il testo guida le fasi")
    controlla(riepilogo["fasi"][0]["valutazione"] is None and riepilogo["fasi"][1]["valutazione"]["giudizio"] == "troppo_corta",
              "guidata: il riposo non si giudica; il carico troppo breve è 'troppo corta'")


def prove_confronto(gpu, base):
    print("\n-- confronta --")
    uno = file_con(os.path.join(base, "uscita", "annidata"), ".json")[0]
    due = file_con(os.path.join(base, "uscita2"), ".csv")[0]
    destinazione = os.path.join(base, "confronto.json")
    r = lancia(gpu, ["confronta", uno, due, "--json", destinazione])
    confronto = leggi_json(destinazione)
    controlla(r.returncode == 0 and "CONFRONTO FRA PROVE" in r.stdout and "A = ../prova 1" in r.stdout and "Fase OSSERVAZIONE" in r.stdout and
              "differenza" in r.stdout and "temp max (C)" in r.stdout and confronto["formato"] == "termico-confronto/1" and
              len(confronto["prove"]) == 2 and "temp_max_c" in confronto["fasi"]["OSSERVAZIONE"],
              "confronta: un .json e un .csv insieme, tabella con differenza e JSON per i programmi")
    controlla("job riusciti" in r.stdout, "confronta: le righe del motore compaiono se almeno una prova lo aveva")
    r = lancia(gpu, ["confronta", uno])
    controlla(r.returncode != 0 and "almeno due" in r.stderr, "confronta: con un solo file spiega che ne servono due")
    sbagliato = os.path.join(base, "sbagliato.json")
    with open(sbagliato, "w") as f:
        f.write('{"a": 1}')
    r = lancia(gpu, ["confronta", uno, sbagliato])
    controlla(r.returncode != 0 and "non è un riepilogo di termico.py" in r.stderr, "confronta: un .json di altro genere viene rifiutato")
    for comando in (["--help"], ["stato", "--help"], ["osserva", "--help"], ["guidata", "--help"], ["riepilogo", "--help"], ["confronta", "--help"],
                    ["--versione"]):
        r = lancia(gpu, comando)
        if r.returncode != 0:
            controlla(False, "aiuto: %s esce con 0" % " ".join(comando))
            break
    else:
        controlla(True, "aiuto: --help di ogni comando e --versione escono con 0")


class MotoreConJob:
    """Un sd-server finto che accetta job: un job finisce dopo `durata_job` secondi; cancel lo ferma."""

    def __init__(self, durata_job=1.3, chiave=None, rifiuti=()):
        self.durata_job = durata_job
        self.chiave = chiave
        self.rifiuti = list(rifiuti)
        self.job = {}
        self.richieste = []
        self.blocco = threading.Lock()
        motore = self

        class Gestore(http.server.BaseHTTPRequestHandler):
            def log_message(self, *argomenti):
                pass

            def rispondi(self, codice, corpo, intestazioni=()):
                dati = json.dumps(corpo).encode()
                self.send_response(codice)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(dati)))
                for nome, valore in intestazioni:
                    self.send_header(nome, valore)
                self.end_headers()
                self.wfile.write(dati)

            def descrivi(self, identificativo):
                voce = motore.job[identificativo]
                if voce["annullato"]:
                    stato = "cancelled"
                elif time.time() - voce["inizio"] >= motore.durata_job:
                    stato = "completed"
                else:
                    stato = "generating"
                corpo = {"id": identificativo, "kind": "img_gen", "status": stato, "created": int(voce["inizio"]),
                         "result": None, "error": None}
                if stato == "completed":
                    corpo["result"] = {"images": [{"index": 0, "b64_json": "iVBORw0KGgo="}]}
                if stato == "cancelled":
                    corpo["error"] = {"code": "cancelled", "message": "job cancelled by client"}
                return corpo

            def autorizzato(self):
                if motore.chiave and self.headers.get("Authorization") != "Bearer " + motore.chiave:
                    self.rispondi(401, {"error": {"code": "unauthorized", "message": "api key required"}})
                    return False
                return True

            def do_POST(self):
                lunghezza = int(self.headers.get("Content-Length") or 0)
                corpo = self.rfile.read(lunghezza).decode() if lunghezza else ""
                with motore.blocco:
                    motore.richieste.append(("POST", self.path, corpo))
                    if not self.autorizzato():
                        return
                    if self.path == "/sdcpp/v1/img_gen":
                        if motore.rifiuti:
                            return self.rispondi(motore.rifiuti.pop(0), {"error": {"code": "busy", "message": "coda piena"}},
                                                 [("Retry-After", "1")])
                        identificativo = "job_%d" % (len(motore.job) + 1)
                        motore.job[identificativo] = {"inizio": time.time(), "annullato": False}
                        return self.rispondi(202, {"id": identificativo, "kind": "img_gen", "status": "queued",
                                                   "created": int(time.time()), "poll_url": "/sdcpp/v1/jobs/" + identificativo})
                    corrispondenza = self.path.split("?")[0].split("/")
                    if len(corrispondenza) == 6 and corrispondenza[-1] == "cancel" and corrispondenza[4] in motore.job:
                        motore.job[corrispondenza[4]]["annullato"] = True
                        return self.rispondi(200, self.descrivi(corrispondenza[4]))
                    self.rispondi(404, {"error": {"code": "not_found", "message": "no"}})

            def do_GET(self):
                with motore.blocco:
                    motore.richieste.append(("GET", self.path, ""))
                    if not self.autorizzato():
                        return
                    identificativo = self.path.rsplit("/", 1)[-1]
                    if self.path.startswith("/sdcpp/v1/jobs/") and identificativo in motore.job:
                        return self.rispondi(200, self.descrivi(identificativo))
                    self.rispondi(404, {"error": {"code": "not_found", "message": "no"}})

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Gestore)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = "http://127.0.0.1:%d" % self.server.server_address[1]

    def ferma(self):
        self.server.shutdown()
        self.server.server_close()

    def invii(self):
        return [json.loads(c) for m, p, c in self.richieste if m == "POST" and p == "/sdcpp/v1/img_gen" and c]

    def annullamenti(self):
        return [p for m, p, _ in self.richieste if m == "POST" and p.startswith("/sdcpp/v1/jobs/") and "/cancel" in p]


def lancia_carico(argomenti, timeout=60, **extra):
    env = dict(os.environ)
    env.pop("TERMICO_CHIAVE_API", None)
    env.update({k: str(v) for k, v in extra.items()})
    return subprocess.run([sys.executable, "-I", CARICO] + argomenti, capture_output=True, text=True, encoding="utf-8",
                          timeout=timeout, env=env, stdin=subprocess.DEVNULL)


def prove_carico(base):
    print("\n-- carico_motore.py --")
    motore = MotoreConJob(durata_job=4.0)
    try:
        r = lancia_carico(["--url", motore.url, "--durata", "6", "--passi", "7", "--larghezza", "256", "--altezza", "192",
                           "--seme", "9", "--prompt", "ciao"])
    finally:
        motore.ferma()
    invii = motore.invii()
    controlla(r.returncode == 0 and "job 1: completato in" in r.stdout and "fermato alla scadenza del tempo" in r.stdout and
              "1 job completati" in r.stdout and "1 fermati, 0 falliti" in r.stdout,
              "carico: un job completato e uno fermato quando scade il tempo (%s)" % r.stdout.strip().splitlines()[-1:])
    controlla(len(invii) == 2 and all(x == invii[0] for x in invii) and invii[0]["width"] == 256 and invii[0]["height"] == 192 and
              invii[0]["seed"] == 9 and invii[0]["sample_params"]["sample_steps"] == 7 and invii[0]["prompt"] == "ciao" and
              invii[0]["batch_count"] == 1, "carico: ogni job è identico agli altri e porta le dimensioni, il seme e i passi indicati")
    controlla(len(motore.annullamenti()) == 1 and motore.annullamenti()[0].endswith("/cancel?wait=10"),
              "carico: alla scadenza il job in corso si ferma con POST .../cancel")

    if os.name != "nt":
        motore = MotoreConJob(durata_job=30.0)
        try:
            p = subprocess.Popen([sys.executable, "-I", CARICO, "--url", motore.url, "--durata", "60"], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding="utf-8", stdin=subprocess.DEVNULL)
            time.sleep(3.0)
            p.send_signal(signal.SIGINT)
            uscita, _ = p.communicate(timeout=30)
        finally:
            motore.ferma()
        controlla(len(motore.annullamenti()) == 1 and "Interrotto: fermo il job in corso" in uscita and "Finito dopo" in uscita and
                  p.returncode == 1, "carico: con Ctrl+C ferma il job in corso e stampa quanto ha fatto (nessun job finito = codice 1)")

    motore = MotoreConJob(durata_job=1.0, rifiuti=[429, 503])
    try:
        r = lancia_carico(["--url", motore.url, "--durata", "6"])
    finally:
        motore.ferma()
    controlla(r.returncode == 0 and r.stdout.count("il server chiede di aspettare 1 s") == 2 and "(HTTP 429)" in r.stdout and
              "(HTTP 503)" in r.stdout and "job 1: completato in" in r.stdout,
              "carico: 429 e 503 con Retry-After si aspettano e si riprova")

    motore = MotoreConJob(durata_job=1.0, chiave="k-123")
    try:
        r = lancia_carico(["--url", motore.url, "--durata", "3"])
        controlla(r.returncode != 0 and "chiave API" in r.stderr and len(motore.invii()) == 1,
                  "carico: senza la chiave richiesta dal server si ferma subito dicendolo")
        r = lancia_carico(["--url", motore.url, "--durata", "3"], TERMICO_CHIAVE_API="k-123")
        controlla(r.returncode == 0 and "job 1: completato" in r.stdout and "k-123" not in r.stdout + r.stderr,
                  "carico: la chiave dalla variabile d'ambiente passa, e non compare nell'uscita")
        r = lancia_carico(["--url", motore.url, "--durata", "3", "--chiave-api", "k-123"])
        controlla(r.returncode == 0, "carico: la chiave con --chiave-api passa")
    finally:
        motore.ferma()

    spec = importlib.util.spec_from_file_location("carico_motore", CARICO)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    modulo.ATTESA_SENZA_RISPOSTA = 0.05
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            modulo.main(["--url", "http://127.0.0.1:1", "--durata", "30"])
        messaggio = None
    except SystemExit as uscita:
        messaggio = str(uscita.code)
    controlla(messaggio is not None and "Il server non risponde" in messaggio, "carico: server spento = dopo cinque tentativi si ferma con un messaggio")
    r = lancia_carico(["--durata", "0"])
    controlla(r.returncode == 2 and "devono avere senso" in r.stderr, "carico: valori assurdi rifiutati")


def prove_programma(base):
    gpu = Gpu(base)
    prove_stato(gpu, base)
    prove_stato_motore(gpu, base)
    prove_osserva(gpu, base)
    prove_osserva_guasti(gpu, base)
    prove_segnala(gpu, base)
    prove_segnali(gpu, base)
    prova_guidata(gpu, base)
    prove_confronto(gpu, base)
    prove_carico(base)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--veloce", action="store_true", help="solo le prove delle funzioni")
    args = parser.parse_args()
    tm = carica_modulo()
    prove_funzioni(tm)
    prove_funzioni_segnala(tm)
    if not args.veloce:
        base = tempfile.mkdtemp(prefix="termico-test-")
        try:
            prove_programma(base)
        finally:
            shutil.rmtree(base, ignore_errors=True)
    print()
    if errori:
        print("%d CONTROLLI FALLITI" % errori)
        return 1
    print("TUTTO SUPERATO (0 controlli falliti)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
