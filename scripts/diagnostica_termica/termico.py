#!/usr/bin/env python3
"""Diagnostica termica della GPU NVIDIA per il motore di generazione di immagini.

Misura temperatura, potenza, frequenza e motivi di throttling con nvidia-smi, a fasi, e può affiancare alla
stessa linea del tempo lo stato del motore (sd-server: /sdcpp/v1/health e /sdcpp/v1/resources).

Non cambia nessuna impostazione della GPU, non avvia nessun carico e scrive soltanto nella cartella di uscita.
Al motore chiede soltanto il suo stato, salvo con osserva --segnala: allora gli manda anche la fascia della GPU
come livello di salute del device (POST /sdcpp/v1/devices/{device}/pressure), e il motore sposta il lavoro.
Serve Python 3.8 o più recente; nessun pacchetto da installare.

Comandi (guida completa: README.md di questa cartella):
  guidata    sessione a fasi (riposo, motore, chat, insieme, raffreddamento); è il comando predefinito
  osserva    registra senza fasi per un tempo o fino a Ctrl+C, e può tenere aggiornato un file di stato
  stato      una sola lettura in JSON; il codice di uscita dice la fascia (0 normale, 1 attenzione, 2 critica)
  riepilogo  rifà il riepilogo da un CSV, anche di una prova interrotta
  confronta  mette a confronto più prove (CSV o JSON)
"""
import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

VERSIONE = "1.0"
FORMATO_RIEPILOGO = "termico-riepilogo/1"
FORMATO_STATO = "termico-stato/1"
FORMATO_CONFRONTO = "termico-confronto/1"

CAMPI = ["temperature.gpu", "power.draw", "clocks.sm", "utilization.gpu", "memory.used", "pstate"]
CAMPI_MOTIVI = ["clocks_throttle_reasons.active", "clocks_event_reasons.active"]
MAX_ERRORI_DI_FILA = 10

# Bit di NVML (nvmlClocksEventReason*). Il bit 0x1 (GPU inattiva) è normale a riposo: non è un limite.
MOTIVI = [
    (0x2, "clock_fissato"),
    (0x4, "limite_potenza"),
    (0x8, "rallentamento_hw"),
    (0x10, "sync_boost"),
    (0x20, "termico_sw"),
    (0x40, "termico_hw"),
    (0x80, "freno_potenza_hw"),
    (0x100, "clock_display"),
]
BIT_TERMICI = 0x20 | 0x40

FASI = {
    "riposo": ("RIPOSO", "Lascia il PC fermo: niente motore, niente chat. "
                         "Si ferma da sola: non premere Invio finché non ricompare la richiesta.", True),
    "motore": ("MOTORE", "Avvia il motore e fallo generare di continuo. La chat resta spenta.", False),
    "chat": ("CHAT", "Ferma il motore. Usa solo la chat, facendola rispondere di continuo.", False),
    "insieme": ("INSIEME", "Fai lavorare motore e chat nello stesso momento.", False),
    "raffreddamento": ("RAFFREDDAMENTO", "Ferma tutto e lascia il PC fermo. "
                                         "Si ferma da sola: non premere Invio finché non ricompare la richiesta.", True),
}
FASI_DI_CARICO = ("MOTORE", "CHAT", "INSIEME", "OSSERVAZIONE")

# Fasce di temperatura. Sono indicative: partono dai limiti che il driver dichiara, se li dichiara.
ATTENZIONE_GENERICA = 80.0
CRITICA_GENERICA = 87.0
MARGINE_ATTENZIONE = 10.0
MARGINE_CRITICA = 3.0
SOGLIA_ANCORA_IN_SALITA = 1.5  # gradi al minuto, misurati sugli ultimi 60 secondi della fase
DURATA_MINIMA_GIUDIZIO = 180   # secondi (3 minuti): con meno non si può dire se la temperatura si stabilizza

COLONNE_GPU = ["istante", "fase", "t_s", "temp_c", "potenza_w", "clock_sm_mhz", "util_gpu_pct",
               "vram_usata_mib", "pstate", "motivi_hex", "motivi"]
COLONNE_MOTORE = ["motore_raggiungibile", "motore_stato", "motore_profilo", "motore_occupato", "motore_coda",
                  "motore_richieste", "motore_riuscite", "motore_fallite", "motore_annullate", "motore_discese",
                  "motore_ricaricamenti", "motore_uptime_s", "ram_disponibile_mib", "vram_libera_mib"]
COLONNE_CSV = COLONNE_GPU + COLONNE_MOTORE
# chiave interna del motore -> colonna del CSV
CHIAVI_MOTORE = [("stato", "motore_stato"), ("profilo", "motore_profilo"), ("coda", "motore_coda"),
                 ("richieste", "motore_richieste"), ("riuscite", "motore_riuscite"), ("fallite", "motore_fallite"),
                 ("annullate", "motore_annullate"), ("discese", "motore_discese"),
                 ("ricaricamenti", "motore_ricaricamenti"), ("uptime", "motore_uptime_s"),
                 ("ram_disp_mib", "ram_disponibile_mib"), ("vram_libera_mib", "vram_libera_mib")]


# ----------------------------------------------------------------------------------------------------------
# Numeri e statistiche
# ----------------------------------------------------------------------------------------------------------

def num(testo):
    try:
        valore = float(testo)
    except (TypeError, ValueError):
        return None
    return valore if valore == valore and abs(valore) != float("inf") else None


def maschera(testo):
    try:
        return int(testo, 16)
    except (TypeError, ValueError):
        return None


def nomi_motivi(valore):
    if valore is None:
        return ""
    return "+".join(nome for bit, nome in MOTIVI if valore & bit)


def fmt(valore, cifre=0):
    return "n/d" if valore is None else f"{valore:.{cifre}f}"


def arr(valore, cifre=1):
    return None if valore is None else round(valore, cifre)


def validi(valori):
    return [v for v in valori if v is not None]


def media(valori):
    v = validi(valori)
    return sum(v) / len(v) if v else None


def minimo(valori):
    v = validi(valori)
    return min(v) if v else None


def massimo(valori):
    v = validi(valori)
    return max(v) if v else None


def mediana(valori):
    v = validi(valori)
    return statistics.median(v) if v else None


def pct(k, n):
    """Percentuale intera; se c'è anche solo un campione non scende sotto 1."""
    return 0 if not k or not n else max(1, 100 * k // n)


def durata_testo(secondi):
    s = int(secondi)
    return f"{s // 60}:{s % 60:02d}"


def pendenza_al_minuto(punti):
    """Pendenza, in unità al minuto, della retta che segue meglio i punti (secondi, valore)."""
    if len(punti) < 10 or punti[-1][0] - punti[0][0] < 30:
        return None
    n = len(punti)
    medio_t = sum(t for t, _ in punti) / n
    medio_v = sum(v for _, v in punti) / n
    den = sum((t - medio_t) ** 2 for t, _ in punti)
    if den < 1e-9:
        return None
    return 60.0 * sum((t - medio_t) * (v - medio_v) for t, v in punti) / den


# ----------------------------------------------------------------------------------------------------------
# nvidia-smi
# ----------------------------------------------------------------------------------------------------------

def trova_nvidia_smi(indicato=None):
    if indicato:
        return indicato if os.path.exists(indicato) else None
    percorso = shutil.which("nvidia-smi")
    if percorso:
        return percorso
    for candidato in (
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        r"C:\Windows\System32\nvidia-smi.exe",
    ):
        if os.path.exists(candidato):
            return candidato
    return None


def esegui(nvsmi, argomenti):
    # Un file .py indicato con --nvidia-smi si esegue con lo stesso Python: serve alle prove con una GPU finta.
    comando = [sys.executable, nvsmi] if nvsmi.lower().endswith(".py") else [nvsmi]
    try:
        return subprocess.run(comando + argomenti, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None


def interroga(nvsmi, gpu, campi):
    r = esegui(nvsmi, ["--query-gpu=" + ",".join(campi), "--format=csv,noheader,nounits", "-i", str(gpu)])
    if r is None:
        return None
    testo = (r.stdout or "").strip()
    if r.returncode != 0 or not testo or "valid field" in testo or "valid field" in (r.stderr or ""):
        return None
    parti = [p.strip() for p in testo.splitlines()[0].split(",")]
    return parti if len(parti) == len(campi) else None


def leggi(nvsmi, gpu, campo_motivi):
    campi = CAMPI + ([campo_motivi] if campo_motivi else [])
    p = interroga(nvsmi, gpu, campi)
    if p is None:
        return None
    return {
        "temp": num(p[0]),
        "watt": num(p[1]),
        "mhz": num(p[2]),
        "util": num(p[3]),
        "mib": num(p[4]),
        "pstate": p[5],
        "mask": maschera(p[6]) if campo_motivi else None,
    }


RE_LIMITE_TERMICO = re.compile(
    r"^\s*(GPU Slowdown Temp|GPU Shutdown Temp|GPU Max Operating Temp)\s*:\s*(\d+(?:\.\d+)?)\s*C\b", re.M)


def limiti_termici(nvsmi, gpu):
    """Limiti di temperatura che il driver dichiara (nvidia-smi -q -d TEMPERATURE); mancano se non li dice."""
    limiti = {"temp_rallentamento_c": None, "temp_spegnimento_c": None, "temp_max_operativa_c": None}
    r = esegui(nvsmi, ["-q", "-d", "TEMPERATURE", "-i", str(gpu)])
    if r is None or r.returncode != 0:
        return limiti
    chiavi = {"GPU Slowdown Temp": "temp_rallentamento_c", "GPU Shutdown Temp": "temp_spegnimento_c",
              "GPU Max Operating Temp": "temp_max_operativa_c"}
    for nome, valore in RE_LIMITE_TERMICO.findall(r.stdout or ""):
        limiti[chiavi[nome]] = num(valore)
    return limiti


def info_gpu(nvsmi, gpu):
    base = interroga(nvsmi, gpu, ["name", "driver_version"])
    if base is None:
        return None
    info = {"indice": gpu, "nome": base[0], "driver": base[1], "limite_potenza_w": None, "clock_max_mhz": None}
    for chiave, campo in (("limite_potenza_w", "power.limit"), ("clock_max_mhz", "clocks.max.sm")):
        valore = interroga(nvsmi, gpu, [campo])
        info[chiave] = num(valore[0]) if valore else None
    info.update(limiti_termici(nvsmi, gpu))
    return info


def campo_dei_motivi(nvsmi, gpu):
    return next((c for c in CAMPI_MOTIVI if interroga(nvsmi, gpu, [c]) is not None), None)


# ----------------------------------------------------------------------------------------------------------
# Fasce di temperatura (indicative)
# ----------------------------------------------------------------------------------------------------------

def calcola_fasce(info, attenzione=None, critica=None):
    info = info or {}
    limite, nome_limite = None, None
    for chiave, nome in (("temp_rallentamento_c", "GPU Slowdown Temp"), ("temp_max_operativa_c", "GPU Max Operating Temp")):
        if info.get(chiave):
            limite, nome_limite = info[chiave], nome
            break
    if limite:
        att_auto, crit_auto = limite - MARGINE_ATTENZIONE, limite - MARGINE_CRITICA
        origine = (f"limite dichiarato dal driver ({nome_limite} {limite:.0f} C): "
                   f"attenzione {MARGINE_ATTENZIONE:.0f} C sotto, critica {MARGINE_CRITICA:.0f} C sotto")
    else:
        att_auto, crit_auto = ATTENZIONE_GENERICA, CRITICA_GENERICA
        origine = "valori generici per un portatile: il driver non dichiara i suoi limiti"
    if attenzione is not None and critica is not None:
        origine = "indicate da te"
    elif attenzione is not None or critica is not None:
        origine = "in parte indicate da te, per il resto " + origine
    return {
        "indicativo": True,
        "origine": origine,
        "attenzione_da_c": attenzione if attenzione is not None else att_auto,
        "critica_da_c": critica if critica is not None else crit_auto,
        "rallentamento_gpu_c": info.get("temp_rallentamento_c"),
        "spegnimento_gpu_c": info.get("temp_spegnimento_c"),
    }


def fascia_di(temp, fasce):
    if temp is None:
        return None
    if temp >= fasce["critica_da_c"]:
        return "critica"
    if temp >= fasce["attenzione_da_c"]:
        return "attenzione"
    return "normale"


# ----------------------------------------------------------------------------------------------------------
# Motore (sd-server): letture con GET; le segnalazioni di --segnala sono le sole POST
# ----------------------------------------------------------------------------------------------------------

def normalizza_url(testo):
    testo = testo.strip().rstrip("/")
    if "://" not in testo:
        testo = "http://" + testo
    return testo


def url_senza_credenziali(url):
    parti = urllib.parse.urlsplit(url)
    host = parti.hostname or ""
    if ":" in host:
        host = "[" + host + "]"
    return f"{parti.scheme}://{host}" + (f":{parti.port}" if parti.port else "")


def apri_json(richiesta, chiave, timeout):
    """(codice HTTP, corpo JSON); (None, None) se il motore non risponde."""
    if chiave:
        richiesta.add_header("Authorization", "Bearer " + chiave)
    # Il motore è di solito su questo stesso PC: niente proxy di sistema.
    apri = urllib.request.build_opener(urllib.request.ProxyHandler({})).open
    try:
        with apri(richiesta, timeout=timeout) as risposta:
            return risposta.status, json.loads(risposta.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as errore:  # /health risponde 503 con lo stesso corpo quando non può generare
        try:
            return errore.code, json.loads(errore.read().decode("utf-8", "replace"))
        except (ValueError, OSError):
            return errore.code, None
    except (urllib.error.URLError, OSError, ValueError):
        return None, None


def chiedi_json(url, chiave, percorso, timeout):
    return apri_json(urllib.request.Request(url + percorso, headers={"Accept": "application/json"}), chiave, timeout)


def manda_json(url, chiave, percorso, corpo, timeout):
    richiesta = urllib.request.Request(url + percorso, data=json.dumps(corpo).encode("utf-8"), method="POST",
                                       headers={"Accept": "application/json", "Content-Type": "application/json"})
    return apri_json(richiesta, chiave, timeout)


def leggi_salute(url, chiave, timeout=3.0):
    codice, salute = chiedi_json(url, chiave, "/sdcpp/v1/health", timeout)
    if codice is None:
        return {"ok": False, "errore": "irraggiungibile"}
    if codice == 401:
        return {"ok": False, "errore": "chiave API mancante o errata"}
    if not isinstance(salute, dict) or "status" not in salute:
        return {"ok": False, "errore": f"risposta inattesa (HTTP {codice})"}
    contatori = salute.get("counters") or {}
    coda = salute.get("queue") or {}
    return {
        "ok": True,
        "stato": salute.get("status"),
        "profilo": salute.get("profile"),
        "profilo_base": salute.get("base_profile"),
        "occupato": bool(salute.get("busy")),
        "coda": coda.get("pending"),
        "attesa": coda.get("waiting"),
        "richieste": contatori.get("requests"),
        "riuscite": contatori.get("succeeded"),
        "fallite": contatori.get("failed"),
        "annullate": contatori.get("cancelled"),
        "discese": contatori.get("step_downs"),
        "ricaricamenti": contatori.get("rebuilds"),
        "uptime": salute.get("uptime_seconds"),
        "versione": salute.get("version"),
    }


def leggi_risorse(url, chiave, timeout=3.0):
    codice, r = chiedi_json(url, chiave, "/sdcpp/v1/resources", timeout)
    if codice != 200 or not isinstance(r, dict):
        return None
    ram = (r.get("system_memory") or {}).get("available_bytes")
    vram = next((d.get("free_memory_bytes") for d in r.get("devices") or []
                 if d.get("type") in ("gpu", "igpu") and d.get("free_memory_bytes") is not None), None)
    return {"ram_disp_mib": None if ram is None else ram / 1048576,
            "vram_libera_mib": None if vram is None else vram / 1048576}


class OsservatoreMotore:
    """Interroga il motore in un thread a parte, così un motore lento non rallenta le letture della GPU."""

    def __init__(self, url, chiave, intervallo=2.0, ogni_risorse=3):
        self.url = url
        self.chiave = chiave
        self.intervallo = intervallo
        self.ogni_risorse = ogni_risorse
        self._ultimo = None
        self._risorse = {"ram_disp_mib": None, "vram_libera_mib": None}
        self._fermo = threading.Event()
        self._thread = threading.Thread(target=self._ciclo, daemon=True)

    def avvia(self):
        self._thread.start()

    def ferma(self):
        self._fermo.set()
        self._thread.join(timeout=5)

    def _ciclo(self):
        giro = 0
        while not self._fermo.is_set():
            lettura = leggi_salute(self.url, self.chiave)
            if lettura["ok"] and giro % self.ogni_risorse == 0:
                self._risorse = leggi_risorse(self.url, self.chiave) or self._risorse
            giro += 1
            if lettura["ok"]:
                lettura.update(self._risorse)
            self._ultimo = (time.monotonic(), lettura)
            self._fermo.wait(self.intervallo)

    def istantanea(self, attesa=0.0):
        """L'ultima lettura (una copia), None se non ce n'è ancora o se è troppo vecchia per essere vera."""
        fine = time.monotonic() + attesa
        while self._ultimo is None and time.monotonic() < fine:
            time.sleep(0.05)
        ultimo = self._ultimo
        if ultimo is None or time.monotonic() - ultimo[0] > max(10.0, 5 * self.intervallo):
            return None
        return dict(ultimo[1])


# ----------------------------------------------------------------------------------------------------------
# Segnalazioni al motore (osserva --segnala): la fascia della GPU diventa il livello di salute del suo device.
# Il motore toglie lavoro al device un gradino alla volta e, se lo esclude, lo sposta altrove; il segnale dura
# qualche campione e si ripete finché la condizione resta, così scade da solo se questo programma si ferma.
# ----------------------------------------------------------------------------------------------------------

FONTE_SEGNALE = "termico"
LIVELLO_ESCLUSO = 4  # il livello dopo i tre di sforzo del motore
BIT_TERMICO_HW = 0x40


def livello_segnale(fascia, mask):
    """Il livello che fascia e motivi di throttling valgono per il motore; None se la temperatura non si legge."""
    if fascia is None:
        return None
    termico = (mask or 0) & BIT_TERMICI
    if fascia == "critica":
        return LIVELLO_ESCLUSO if (mask or 0) & BIT_TERMICO_HW else 3
    if fascia == "attenzione":
        return 2 if termico else 1
    # Il driver che rallenta la GPU per il caldo sotto la fascia di attenzione: le fasce sono più alte del suo limite.
    return 1 if termico else 0


def motivo_segnale(c, fascia):
    mask = c.get("mask") or 0
    motivo = f"GPU a {c['temp']:.0f} C, fascia {fascia}"
    if mask & BIT_TERMICO_HW:
        motivo += ", throttling termico hardware"
    elif mask & BIT_TERMICI:
        motivo += ", throttling termico"
    return motivo


def durata_segnale(ogni):
    """Qualche campione, mai meno di 5 secondi."""
    return max(5, int(math.ceil(3 * ogni)))


def dispositivo_del_motore(dispositivi, nome_gpu):
    """Il device del motore per la GPU, dal nome che il motore ne dà (/sdcpp/v1/resources); None se non c'è o se
    due schede uguali non si distinguono. La stessa scheda vista da CUDA e da Vulkan va bene con l'uno o l'altro
    nome: il motore riconosce la scheda e passa il segnale anche all'altro."""
    voluto = (nome_gpu or "").strip().lower()
    trovati = [d.get("name") for d in dispositivi or [] if isinstance(d, dict) and d.get("name") and
               d.get("type") in ("gpu", "igpu") and str(d.get("description") or "").strip().lower() == voluto]
    if not voluto or not trovati:
        return None
    backend = [re.sub(r"\d+$", "", nome) for nome in trovati]
    if len(set(backend)) != len(backend):
        return None
    return trovati[0]


class SegnalatoreMotore:
    """Manda al motore il livello della GPU in un thread a parte, così un motore lento non rallenta le letture.
    Un livello sopra 0 si ripete a ogni campione; tornato 0 il segnale si ritira una volta, e così alla fine."""

    def __init__(self, url, chiave, dispositivo, nome_gpu, durata):
        self.url = url
        self.chiave = chiave
        self.dispositivo = dispositivo  # None: si cerca dal nome della GPU
        self.nome_gpu = nome_gpu
        self.durata = durata
        self._voluto = None             # (livello, motivo) dell'ultimo campione, non ancora mandato
        self._attivo = False            # il motore tiene un segnale di questa fonte
        self._fermo = False
        self._annunciato = False
        self._avvisi = set()
        self._ultima_ricerca = None
        self._condizione = threading.Condition()
        self._thread = threading.Thread(target=self._ciclo, daemon=True)

    def avvia(self):
        self._thread.start()

    def aggiorna(self, livello, motivo):
        with self._condizione:
            self._voluto = (livello, motivo)
            self._condizione.notify()

    def ferma(self):
        with self._condizione:
            self._fermo = True
            self._condizione.notify()
        self._thread.join(timeout=10)

    def _avvisa(self, chiave, testo):
        if chiave not in self._avvisi:
            self._avvisi.add(chiave)
            print("  " + testo, flush=True)

    def _ciclo(self):
        self._trova_dispositivo()
        while True:
            with self._condizione:
                while self._voluto is None and not self._fermo:
                    self._condizione.wait()
                voluto, self._voluto = self._voluto, None
                fermo = self._fermo
            if fermo:
                if self._attivo:
                    self._invia(0, "")
                return
            self._invia(*voluto)

    def _trova_dispositivo(self):
        if self.dispositivo is None and (self._ultima_ricerca is None or time.monotonic() - self._ultima_ricerca >= 10):
            self._ultima_ricerca = time.monotonic()
            codice, risorse = chiedi_json(self.url, self.chiave, "/sdcpp/v1/resources", 3.0)
            if codice == 200 and isinstance(risorse, dict):
                dispositivi = risorse.get("devices") or []
                self.dispositivo = dispositivo_del_motore(dispositivi, self.nome_gpu)
                if self.dispositivo is None:
                    nomi = ", ".join(str(d.get("name")) for d in dispositivi if isinstance(d, dict)) or "nessuno"
                    self._avvisa("gpu_ignota", f"Nel motore non trovo la GPU {self.nome_gpu} fra i device ({nomi}): "
                                               "indica il suo nome con --segnala NOME.")
            else:
                self._avvisa_errore(codice, risorse)
        if self.dispositivo is not None and not self._annunciato:
            self._annunciato = True
            print(f"  Segnalo al motore la GPU come device {self.dispositivo}.", flush=True)
        return self.dispositivo

    def _avvisa_errore(self, codice, risposta):
        if codice is None:
            self._avvisa("irraggiungibile", "Segnalazione al motore non riuscita: il motore non risponde. Riprovo al prossimo campione.")
        elif codice == 401:
            self._avvisa("401", "Segnalazione al motore rifiutata: chiave API mancante o errata (--chiave-api o TERMICO_CHIAVE_API).")
        elif codice == 404:
            self._avvisa("404", f"Il motore non conosce il device {self.dispositivo}: i nomi sono quelli di "
                                "/sdcpp/v1/health (CPU, CUDA0, Vulkan0...).")
        else:
            errore = risposta.get("error") if isinstance(risposta, dict) else None
            self._avvisa(f"http{codice}", f"Segnalazione al motore rifiutata (HTTP {codice})" + (f": {errore}" if errore else "."))

    def _invia(self, livello, motivo):
        if livello == 0 and not self._attivo:
            return
        dispositivo = self._trova_dispositivo()
        if dispositivo is None:
            return
        if livello == 0:
            corpo = {"level": 0, "source": FONTE_SEGNALE}
        else:
            corpo = {"level": "excluded" if livello == LIVELLO_ESCLUSO else livello, "source": FONTE_SEGNALE,
                     "reason": motivo, "duration_seconds": self.durata}
        percorso = "/sdcpp/v1/devices/" + urllib.parse.quote(dispositivo, safe="") + "/pressure"
        codice, risposta = manda_json(self.url, self.chiave, percorso, corpo, 3.0)
        if codice == 200:
            self._attivo = livello != 0
        else:
            self._avvisa_errore(codice, risposta)


# ----------------------------------------------------------------------------------------------------------
# Riepilogo
# ----------------------------------------------------------------------------------------------------------

def valuta(nome, durata, temp_max, termico_pct, ancora_in_salita, fasce):
    """Giudizio indicativo su una fase di carico. Le altre fasi (riposo, raffreddamento) non si giudicano."""
    if nome not in FASI_DI_CARICO:
        return None
    note = []
    if durata < DURATA_MINIMA_GIUDIZIO:
        giudizio = "troppo_corta"
        note.append("meno di 3 minuti: troppo poco per dire se la temperatura si stabilizza")
    elif (temp_max is not None and temp_max >= fasce["critica_da_c"]) or termico_pct >= 20:
        giudizio = "non_sostenibile"
        if temp_max is not None and temp_max >= fasce["critica_da_c"]:
            note.append(f"ha toccato la fascia critica ({temp_max:.0f} C, da {fasce['critica_da_c']:.0f} C)")
        if termico_pct >= 20:
            note.append(f"il driver ha rallentato la GPU per il caldo nel {termico_pct}% del tempo")
    elif (temp_max is not None and temp_max >= fasce["attenzione_da_c"]) or termico_pct > 0:
        giudizio = "al_limite"
        if temp_max is not None and temp_max >= fasce["attenzione_da_c"]:
            note.append(f"è entrata nella fascia di attenzione ({temp_max:.0f} C, da {fasce['attenzione_da_c']:.0f} C)")
        if termico_pct > 0:
            note.append(f"il driver ha rallentato la GPU per il caldo nel {termico_pct}% del tempo")
    elif ancora_in_salita:
        giudizio = "da_verificare"
        note.append("a fine fase la temperatura saliva ancora: serve una prova più lunga per vedere dove si ferma")
    else:
        giudizio = "sostenibile"
        note.append("la temperatura si è stabilizzata sotto la fascia di attenzione")
    return {"giudizio": giudizio, "indicativo": True, "note": note}


def riassumi_motore(campioni):
    letture = [c["motore"] for c in campioni if c.get("motore")]
    if not letture:
        return None
    buone = [m for m in letture if m.get("ok")]
    r = {"campioni": len(letture), "raggiungibile_pct": 100 * len(buone) // len(letture)}
    if not buone:
        return r
    r["occupato_pct"] = 100 * sum(1 for m in buone if m.get("occupato")) // len(buone)
    r["stati"] = sorted({m["stato"] for m in buone if m.get("stato")})
    r["profili"] = sorted({m["profilo"] for m in buone if m.get("profilo")})
    r["coda_max"] = massimo([m.get("coda") for m in buone])
    r["ram_disponibile_min_mib"] = arr(minimo([m.get("ram_disp_mib") for m in buone]))
    r["vram_libera_min_mib"] = arr(minimo([m.get("vram_libera_mib") for m in buone]))
    riavviato = any(a.get("uptime") is not None and b.get("uptime") is not None and b["uptime"] < a["uptime"]
                    for a, b in zip(buone, buone[1:]))
    r["riavviato"] = riavviato
    for chiave, nome in (("riuscite", "job_riusciti"), ("fallite", "job_falliti"), ("annullate", "job_annullati"),
                         ("discese", "discese_di_profilo"), ("ricaricamenti", "ricaricamenti")):
        prima, dopo = buone[0].get(chiave), buone[-1].get(chiave)
        r[nome] = None if riavviato or prima is None or dopo is None else dopo - prima
    return r


def riassumi_fase(nome, campioni, fasce):
    if not campioni:
        return None
    t_fine = campioni[-1]["t"]
    temp = [c["temp"] for c in campioni]
    prima = next(iter(validi(temp)), None)
    ultima = next(iter(reversed(validi(temp))), None)
    dopo60 = next((c["temp"] for c in campioni if c["t"] >= 60 and c["temp"] is not None), None)
    pendenza = pendenza_al_minuto([(c["t"], c["temp"]) for c in campioni
                                   if c["temp"] is not None and c["t"] >= t_fine - 60])
    ancora = bool(t_fine >= DURATA_MINIMA_GIUDIZIO and pendenza is not None and pendenza >= SOGLIA_ANCORA_IN_SALITA)
    maschere = [c["mask"] for c in campioni if c["mask"] is not None]
    motivi = {}
    for bit, nome_motivo in MOTIVI:
        k = sum(1 for m in maschere if m & bit)
        if k:
            motivi[nome_motivo] = pct(k, len(maschere))
    termico = pct(sum(1 for m in maschere if m & BIT_TERMICI), len(maschere))
    passo = mediana([b["t"] - a["t"] for a, b in zip(campioni, campioni[1:])]) or 1.0
    oltre_att = [c for c in campioni if c["temp"] is not None and c["temp"] >= fasce["attenzione_da_c"]]
    oltre_crit = [c for c in campioni if c["temp"] is not None and c["temp"] >= fasce["critica_da_c"]]
    temp_max = massimo(temp)
    return {
        "nome": nome,
        "durata_s": arr(t_fine),
        "campioni": len(campioni),
        "temp_c": {
            "inizio": prima, "fine": ultima, "max": temp_max, "media": arr(media(temp)),
            "ultimi_60s_mediana": arr(mediana([c["temp"] for c in campioni if c["t"] >= t_fine - 60])),
            "variazione_dopo_60s": None if prima is None or dopo60 is None else dopo60 - prima,
            "pendenza_finale_c_min": arr(pendenza, 2),
        },
        "potenza_w": {"media": arr(media([c["watt"] for c in campioni])), "max": massimo([c["watt"] for c in campioni])},
        "clock_mhz": {"media": arr(media([c["mhz"] for c in campioni]), 0),
                      "min": minimo([c["mhz"] for c in campioni]), "max": massimo([c["mhz"] for c in campioni])},
        "util_gpu_pct": {"media": arr(media([c["util"] for c in campioni]), 0)},
        "vram_usata_mib": {"max": massimo([c["mib"] for c in campioni])},
        "throttling": {"leggibile": bool(maschere), "motivi_pct": motivi, "termico_pct": termico},
        "fasce": {
            "secondi_oltre_attenzione": arr(len(oltre_att) * passo),
            "secondi_oltre_critica": arr(len(oltre_crit) * passo),
            "tempo_alla_fascia_attenzione_s": arr(oltre_att[0]["t"]) if oltre_att else None,
        },
        "ancora_in_salita": ancora,
        "valutazione": valuta(nome, t_fine, temp_max, termico, ancora, fasce),
        "motore": riassumi_motore(campioni),
    }


def costruisci_riepilogo(meta, risultati, chiuso, fasi_previste=None):
    return {
        "formato": FORMATO_RIEPILOGO,
        "strumento": {"nome": "termico.py", "versione": VERSIONE},
        "creato": meta["creato"],
        "aggiornato": dt.datetime.now().isoformat(timespec="seconds"),
        "chiuso_regolarmente": chiuso,
        "nome": meta.get("nome"),
        "gpu": meta["gpu"],
        "fasce": meta["fasce"],
        "motore": meta.get("motore"),
        "fasi_previste": fasi_previste or [nome for nome, _ in risultati],
        "fasi": [r for r in (riassumi_fase(nome, campioni, meta["fasce"]) for nome, campioni in risultati) if r],
    }


# ----------------------------------------------------------------------------------------------------------
# Testo
# ----------------------------------------------------------------------------------------------------------

def tabella(righe):
    larghezze = [max(len(str(r[i])) for r in righe) for i in range(len(righe[0]))]
    return ["  ".join(str(c).ljust(larghezze[i]) for i, c in enumerate(r)).rstrip() for r in righe]


def motivi_testo(throttling):
    if not throttling["leggibile"]:
        return "n/d"
    return ", ".join(f"{nome} {valore}%" for nome, valore in throttling["motivi_pct"].items()) or "nessuno"


def righe_fase(f):
    t, p, c = f["temp_c"], f["potenza_w"], f["clock_mhz"]
    return [
        f["nome"], durata_testo(f["durata_s"]),
        f"{fmt(t['inizio'])}/{fmt(t['fine'])}/{fmt(t['max'])}",
        "n/d" if t["variazione_dopo_60s"] is None else f"{t['variazione_dopo_60s']:+.0f}",
        f"{fmt(p['media'], 1)}/{fmt(p['max'], 1)}",
        f"{fmt(c['media'])}/{fmt(c['min'])}",
        fmt(f["util_gpu_pct"]["media"]),
        fmt(f["vram_usata_mib"]["max"]),
    ]


def righe_throttling(f):
    giudizio = f["valutazione"]["giudizio"] if f["valutazione"] else "-"
    return [f["nome"], motivi_testo(f["throttling"]), giudizio]


def righe_motore(f):
    m = f["motore"]
    if not m:
        return None
    if "occupato_pct" not in m:
        return [f["nome"], f"{m['raggiungibile_pct']}%", "-", "-", "-", "-", "-", "-", "-"]
    job = "n/d" if m["job_riusciti"] is None else f"{m['job_riusciti']}/{m['job_falliti']}/{m['job_annullati']}"
    return [
        f["nome"], f"{m['raggiungibile_pct']}%", f"{m['occupato_pct']}%", ",".join(m["profili"]) or "-",
        job, fmt(m["discese_di_profilo"]), fmt(m["coda_max"]), fmt(m["ram_disponibile_min_mib"]),
        fmt(m["vram_libera_min_mib"]),
    ]


def testo_riepilogo(r):
    gpu = r["gpu"]
    fasce = r["fasce"]
    righe = [f"GPU: {gpu['nome']}, driver {gpu['driver']}"]
    if r.get("nome"):
        righe[0] += f"  (prova: {r['nome']})"
    intestazione = ["fase", "durata", "temp C ini/fine/max", "dT a 60s (C)", "W med/max", "MHz med/min", "util% med",
                    "VRAM MiB max"]
    righe += tabella([intestazione] + [righe_fase(f) for f in r["fasi"]])
    righe += [""] + tabella([["fase", "motivi di throttling (% del tempo)", "giudizio (indicativo)"]] +
                            [righe_throttling(f) for f in r["fasi"]])
    con_motore = [x for x in (righe_motore(f) for f in r["fasi"]) if x]
    if con_motore:
        righe += ["", "MOTORE (sd-server) durante le fasi"]
        righe += tabella([["fase", "raggiungibile", "occupato", "profili", "job ok/falliti/annullati", "discese",
                           "coda max", "RAM libera min MiB", "VRAM libera min MiB"]] + con_motore)
    righe += ["", f"Fasce di temperatura (INDICATIVE, non garanzie del produttore): attenzione da "
                  f"{fasce['attenzione_da_c']:.0f} C, critica da {fasce['critica_da_c']:.0f} C.",
              f"Origine: {fasce['origine']}."]
    note = [(f["nome"], n) for f in r["fasi"] if f["valutazione"] for n in f["valutazione"]["note"]]
    if note:
        righe += ["", "Note sui giudizi (indicativi):"] + [f"  {nome}: {nota}" for nome, nota in note]
    if not r.get("chiuso_regolarmente", True):
        righe += ["", "La prova non è stata chiusa regolarmente: il riepilogo è ricostruito da quanto era stato scritto."]
    return righe


# ----------------------------------------------------------------------------------------------------------
# File
# ----------------------------------------------------------------------------------------------------------

def scrivi_json(percorso, dati):
    """Scrive per intero o non scrive: chi legge non vede mai un file a metà."""
    provvisorio = percorso + ".tmp"
    with open(provvisorio, "w", encoding="utf-8", newline="\n") as f:
        json.dump(dati, f, ensure_ascii=False, indent=2)
        f.write("\n")
    for tentativo in range(5):
        try:
            os.replace(provvisorio, percorso)
            return
        except PermissionError:  # su Windows fallisce se un altro programma ha il file aperto in quell'istante
            if tentativo == 4:
                raise
            time.sleep(0.05)


def cella(valore, cifre=0):
    return "" if valore is None else f"{valore:.{cifre}f}"


def riga_csv(fase, c):
    m = c.get("motore")
    if m is None:
        motore = [""] * len(COLONNE_MOTORE)
    elif not m.get("ok"):
        motore = ["0"] + [""] * (len(COLONNE_MOTORE) - 1)
    else:
        valori = {colonna: m.get(chiave) for chiave, colonna in CHIAVI_MOTORE}
        valori["motore_raggiungibile"] = 1
        valori["motore_occupato"] = 1 if m.get("occupato") else 0
        motore = []
        for colonna in COLONNE_MOTORE:
            v = valori.get(colonna)
            motore.append("" if v is None else (f"{v:.0f}" if colonna.endswith("_mib") else v))
    return [
        c["istante"], fase, f"{c['t']:.1f}", cella(c["temp"]), cella(c["watt"], 2), cella(c["mhz"]),
        cella(c["util"]), cella(c["mib"]), c["pstate"],
        "" if c["mask"] is None else f"0x{c['mask']:x}", nomi_motivi(c["mask"]),
    ] + motore


def intero(testo):
    v = num(testo)
    return None if v is None else int(v)


def campioni_da_csv(percorso):
    """Rilegge un CSV scritto da questo strumento: [(nome della fase, [campioni])] nell'ordine del file."""
    fasi = {}
    with open(percorso, newline="", encoding="utf-8") as f:
        lettore = csv.DictReader(f)
        mancanti = [c for c in COLONNE_GPU if c not in (lettore.fieldnames or [])]
        if mancanti:
            raise ValueError("non è un CSV di termico.py (mancano le colonne: " + ", ".join(mancanti) + ")")
        for riga in lettore:
            t = num(riga["t_s"])
            if t is None or any(riga.get(colonna) is None for colonna in COLONNE_GPU):
                continue  # riga troncata: il PC può essersi spento mentre scriveva
            c = {"t": t, "istante": riga["istante"], "temp": num(riga["temp_c"]), "watt": num(riga["potenza_w"]),
                 "mhz": num(riga["clock_sm_mhz"]), "util": num(riga["util_gpu_pct"]),
                 "mib": num(riga["vram_usata_mib"]), "pstate": riga["pstate"], "mask": maschera(riga["motivi_hex"]),
                 "motore": None}
            raggiungibile = riga.get("motore_raggiungibile", "")
            if raggiungibile != "":
                if raggiungibile == "1":
                    c["motore"] = {
                        "ok": True,
                        "stato": riga["motore_stato"] or None,
                        "profilo": riga["motore_profilo"] or None,
                        "occupato": riga["motore_occupato"] == "1",
                        "coda": intero(riga["motore_coda"]),
                        "richieste": intero(riga["motore_richieste"]),
                        "riuscite": intero(riga["motore_riuscite"]),
                        "fallite": intero(riga["motore_fallite"]),
                        "annullate": intero(riga["motore_annullate"]),
                        "discese": intero(riga["motore_discese"]),
                        "ricaricamenti": intero(riga["motore_ricaricamenti"]),
                        "uptime": intero(riga["motore_uptime_s"]),
                        "ram_disp_mib": num(riga["ram_disponibile_mib"]),
                        "vram_libera_mib": num(riga["vram_libera_mib"]),
                    }
                else:
                    c["motore"] = {"ok": False, "errore": "irraggiungibile"}
            fasi.setdefault(riga["fase"], []).append(c)
    return list(fasi.items())


def nome_sicuro(testo):
    return re.sub(r"[^A-Za-z0-9_-]+", "-", testo).strip("-")[:40]


# ----------------------------------------------------------------------------------------------------------
# Input da tastiera (sessione guidata)
# ----------------------------------------------------------------------------------------------------------

def invio_premuto():
    if os.name == "nt":
        import msvcrt

        premuto = False
        while msvcrt.kbhit():
            if msvcrt.getwch() in ("\r", "\n"):
                premuto = True
        return premuto
    import select

    pronto, _, _ = select.select([sys.stdin], [], [], 0)
    if pronto:
        sys.stdin.readline()
        return True
    return False


def svuota_input():
    try:
        if os.name == "nt":
            import msvcrt

            while msvcrt.kbhit():
                msvcrt.getwch()
        elif sys.stdin.isatty():
            import termios

            termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass


# ----------------------------------------------------------------------------------------------------------
# Sessione di registrazione (guidata e osserva)
# ----------------------------------------------------------------------------------------------------------

def riga_live(nome, c):
    testo = (f"  [{nome} {durata_testo(c['t'])}] {fmt(c['temp'])} C  {fmt(c['watt'], 1)} W  {fmt(c['mhz'])} MHz  "
             f"{fmt(c['util'])} %  {fmt(c['mib'])} MiB  {nomi_motivi(c['mask']) or '-'}")
    m = c.get("motore")
    if m is not None:
        if m.get("ok"):
            testo += f"  | motore {m['stato']}/{m.get('profilo') or '-'}" + (" occupato" if m.get("occupato") else " libero")
        else:
            testo += f"  | motore: {m['errore']}"
    return testo


def crea_sessione(args, nvsmi, info, campo_motivi, osservatore, fasi_previste):
    try:
        return Sessione(args, nvsmi, info, campo_motivi, osservatore, fasi_previste)
    except OSError as errore:
        if osservatore is not None:
            osservatore.ferma()
        sys.exit(f"Non riesco a scrivere in {args.uscita}: {errore}")


class Sessione:
    def __init__(self, args, nvsmi, info, campo_motivi, osservatore, fasi_previste):
        self.args = args
        self.nvsmi = nvsmi
        self.campo_motivi = campo_motivi
        self.osservatore = osservatore
        self.fasi_previste = fasi_previste
        self.fasce = calcola_fasce(info, args.attenzione, args.critica)
        self.meta = {"creato": dt.datetime.now().isoformat(timespec="seconds"), "nome": args.nome,
                     "gpu": info, "fasce": self.fasce, "motore": None}
        self.risultati = []
        os.makedirs(args.uscita, exist_ok=True)
        stile = "termico" + ("-" + nome_sicuro(args.nome) if args.nome and nome_sicuro(args.nome) else "")
        self.base = os.path.join(args.uscita, stile + "-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.file_csv = open(self.base + ".csv", "w", newline="", encoding="utf-8")
        self.scrittore = csv.writer(self.file_csv)
        self.scrittore.writerow(COLONNE_CSV)
        self.file_csv.flush()
        self.salva(False)

    def descrivi_motore(self, attesa=3.0):
        """Annota nel riepilogo quale motore è stato affiancato alla misura, e dice se risponde."""
        if self.osservatore is None:
            return
        lettura = self.osservatore.istantanea(attesa)
        self.meta["motore"] = {"url": url_senza_credenziali(self.osservatore.url),
                               "raggiungibile_all_avvio": bool(lettura and lettura.get("ok")),
                               "versione": (lettura or {}).get("versione"),
                               "profilo_base": (lettura or {}).get("profilo_base")}
        if lettura and lettura.get("ok"):
            print(f"Motore: {self.meta['motore']['url']} risponde (stato {lettura['stato']}, "
                  f"profilo {lettura.get('profilo') or '-'}).")
        else:
            motivo = (lettura or {}).get("errore", "nessuna risposta")
            print(f"Motore: {self.meta['motore']['url']} non risponde ({motivo}). Registro la sola GPU; "
                  "riprovo a ogni campione.")
        self.salva(False)

    def riepilogo(self, chiuso):
        return costruisci_riepilogo(self.meta, self.risultati, chiuso, self.fasi_previste)

    def salva(self, chiuso):
        scrivi_json(self.base + ".json", self.riepilogo(chiuso))

    def chiudi(self):
        self.file_csv.close()
        if self.osservatore is not None:
            self.osservatore.ferma()

    def registra(self, nome, fermati, durata_max=None, dopo_campione=None):
        """Registra una fase: un campione ogni args.ogni secondi finché fermati() è vero o passa durata_max."""
        campioni = []
        self.risultati.append((nome, campioni))
        errori = 0
        ultimo_avviso = -1e9
        ultima_riga = -1e9
        t0 = time.monotonic()
        while True:
            inizio = time.monotonic()
            t = inizio - t0
            if durata_max is not None and t >= durata_max:
                break
            c = leggi(self.nvsmi, self.args.gpu, self.campo_motivi)
            if c is None:
                errori += 1
                if errori >= MAX_ERRORI_DI_FILA:
                    raise RuntimeError("nvidia-smi non risponde")
            else:
                errori = 0
                c["t"] = t
                c["istante"] = dt.datetime.now().isoformat(timespec="milliseconds")
                c["motore"] = self.osservatore.istantanea() if self.osservatore else None
                campioni.append(c)
                self.scrittore.writerow(riga_csv(nome, c))
                self.file_csv.flush()
                if t - ultima_riga >= 10 or len(campioni) == 1:
                    ultima_riga = t
                    print(riga_live(nome, c), flush=True)
                fascia = fascia_di(c["temp"], self.fasce)
                if fascia != "normale" and fascia is not None and t - ultimo_avviso >= 30:
                    ultimo_avviso = t
                    if fascia == "critica":
                        print(f"  CRITICA: {c['temp']:.0f} C (fascia critica da {self.fasce['critica_da_c']:.0f} C). "
                              "Se non scende, ferma il carico.", flush=True)
                    else:
                        print(f"  ATTENZIONE: {c['temp']:.0f} C (attenzione da {self.fasce['attenzione_da_c']:.0f} C). "
                              "Se continua a salire, ferma il carico.", flush=True)
                if c["mask"] is not None and c["mask"] & BIT_TERMICI and t - ultimo_avviso >= 30:
                    ultimo_avviso = t
                    print("  Il driver sta rallentando la GPU per il caldo.", flush=True)
                if dopo_campione:
                    dopo_campione(t, c, campioni)
            while True:
                if fermati():
                    return
                if time.monotonic() >= inizio + self.args.ogni:
                    break
                time.sleep(0.05)


# ----------------------------------------------------------------------------------------------------------
# Quadro (stato corrente)
# ----------------------------------------------------------------------------------------------------------

def costruisci_quadro(info, fasce, c, recenti=None):
    """Una lettura della GPU con la fascia e, se c'è, il motore: quello che serve a chi deve decidere."""
    mask = c["mask"]
    punti = [(x["t"], x["temp"]) for x in (recenti or []) if x["temp"] is not None and x["t"] >= c["t"] - 60]
    tendenza = pendenza_al_minuto(punti)
    fascia = fascia_di(c["temp"], fasce)
    avvisi = []
    if fascia == "critica":
        avvisi.append("temperatura nella fascia critica")
    if mask is not None and mask & BIT_TERMICI:
        avvisi.append("il driver sta rallentando la GPU per il caldo")
    if tendenza is not None and tendenza >= 3.0:
        avvisi.append(f"temperatura in rapida salita ({tendenza:+.1f} C al minuto)")
    m = c.get("motore")
    motore = None
    if m is not None:
        if m.get("ok"):
            motore = {
                "raggiungibile": True, "stato": m["stato"], "profilo": m.get("profilo"), "occupato": m["occupato"],
                "coda": m.get("coda"), "in_attesa": m.get("attesa"), "job_riusciti": m.get("riuscite"),
                "job_falliti": m.get("fallite"), "job_annullati": m.get("annullate"),
                "discese_di_profilo": m.get("discese"), "ricaricamenti": m.get("ricaricamenti"),
                "ram_disponibile_mib": arr(m.get("ram_disp_mib")), "vram_libera_mib": arr(m.get("vram_libera_mib")),
            }
            if m["stato"] != "ok":
                avvisi.append(f"il motore è in stato {m['stato']}")
        else:
            motore = {"raggiungibile": False, "errore": m["errore"]}
            avvisi.append("il motore non è raggiungibile: " + m["errore"])
    return {
        "formato": FORMATO_STATO,
        "istante": c["istante"],
        "istante_unix": round(time.time(), 3),
        "gpu": {
            "indice": info["indice"], "nome": info["nome"], "temp_c": c["temp"], "potenza_w": c["watt"],
            "clock_mhz": c["mhz"], "util_pct": c["util"], "vram_usata_mib": c["mib"], "pstate": c["pstate"],
            "motivi_throttling": None if mask is None else [n for b, n in MOTIVI if mask & b],
        },
        "fascia": {"nome": fascia, "indicativo": True, "attenzione_da_c": fasce["attenzione_da_c"],
                   "critica_da_c": fasce["critica_da_c"], "origine": fasce["origine"]},
        "tendenza_c_al_minuto": arr(tendenza, 2),
        "motore": motore,
        "avvisi": avvisi,
    }


CODICE_DI_USCITA = {"normale": 0, "attenzione": 1, "critica": 2}


# ----------------------------------------------------------------------------------------------------------
# Comandi
# ----------------------------------------------------------------------------------------------------------

def prepara(args):
    """Trova nvidia-smi, legge la GPU e avvia l'osservatore del motore. Esce con un messaggio se non può."""
    nvsmi = trova_nvidia_smi(args.nvidia_smi)
    if not nvsmi:
        sys.exit("nvidia-smi non trovato: serve il driver NVIDIA installato (o indica il file con --nvidia-smi).")
    info = info_gpu(nvsmi, args.gpu)
    if info is None:
        sys.exit(f"nvidia-smi non risponde per la GPU {args.gpu}.")
    campo_motivi = campo_dei_motivi(nvsmi, args.gpu)
    osservatore = None
    if args.motore:
        osservatore = OsservatoreMotore(normalizza_url(args.motore), args.chiave_api, args.ogni_motore)
        osservatore.avvia()
    return nvsmi, info, campo_motivi, osservatore


def chiudi_e_riferisci(sessione, interrotto_da_errore=False):
    sessione.chiudi()
    sessione.salva(True)
    riepilogo = sessione.riepilogo(True)
    righe = testo_riepilogo(riepilogo)
    print("\nRIEPILOGO")
    print("\n".join(righe))
    with open(sessione.base + ".txt", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(righe) + "\n")
    print(f"\nFile scritti in {os.path.abspath(os.path.dirname(sessione.base))}:")
    print(f"  {os.path.basename(sessione.base)}.csv   un campione al secondo, per i grafici")
    print(f"  {os.path.basename(sessione.base)}.json  riepilogo per i programmi (e per condividere la prova)")
    print(f"  {os.path.basename(sessione.base)}.txt   questo riepilogo")
    print("Per condividere la prova allega il file .json, oppure incolla il riepilogo qui sopra.")
    return 1 if interrotto_da_errore else 0


def al_segnale(*_):
    raise KeyboardInterrupt()


def comando_guidata(args):
    fasi = [x.strip() for x in args.fasi.split(",") if x.strip()]
    sconosciute = [x for x in fasi if x not in FASI]
    if sconosciute:
        sys.exit("fasi sconosciute: " + ", ".join(sconosciute) + " (valide: " + ", ".join(FASI) + ")")
    nvsmi, info, campo_motivi, osservatore = prepara(args)
    sessione = crea_sessione(args, nvsmi, info, campo_motivi, osservatore, [FASI[x][0] for x in fasi])
    print(f"GPU: {info['nome']}, driver {info['driver']}")
    if not campo_motivi:
        print("Con questo driver i motivi di throttling non sono leggibili: li salto.")
    print(f"Fasce di temperatura (indicative): attenzione da {sessione.fasce['attenzione_da_c']:.0f} C, "
          f"critica da {sessione.fasce['critica_da_c']:.0f} C. Origine: {sessione.fasce['origine']}.")
    sessione.descrivi_motore()
    print("Ctrl+C in qualsiasi momento: il riepilogo di quanto raccolto viene stampato lo stesso.")
    esito = 0
    try:
        for numero, chiave in enumerate(fasi, 1):
            nome, istruzione, auto = FASI[chiave]
            print(f"\n=== Fase {numero}/{len(fasi)}: {nome} ===")
            print(istruzione)
            svuota_input()
            if auto:
                input(f"Premi Invio per iniziare ({args.auto_secondi} secondi). ")
                sessione.registra(nome, lambda: False, durata_max=args.auto_secondi)
            else:
                input("Quando il carico è partito premi Invio per iniziare a registrare "
                      "(meglio almeno 3 minuti), poi Invio di nuovo per fermare. ")
                sessione.registra(nome, invio_premuto)
                print("Registrazione ferma: ora puoi fermare il carico.")
            sessione.salva(False)
    except (KeyboardInterrupt, EOFError):
        print("\nInterrotto.")
    except (RuntimeError, OSError) as errore:
        print(f"\nErrore: {errore}.")
        esito = 1
    return chiudi_e_riferisci(sessione, esito == 1)


def comando_osserva(args):
    if args.segnala is not None and not args.motore:
        sys.exit("--segnala manda la fascia della GPU al motore: indica anche --motore URL.")
    nvsmi, info, campo_motivi, osservatore = prepara(args)
    sessione = crea_sessione(args, nvsmi, info, campo_motivi, osservatore, ["OSSERVAZIONE"])
    print(f"GPU: {info['nome']}, driver {info['driver']}")
    print(f"Fasce di temperatura (indicative): attenzione da {sessione.fasce['attenzione_da_c']:.0f} C, "
          f"critica da {sessione.fasce['critica_da_c']:.0f} C. Origine: {sessione.fasce['origine']}.")
    sessione.descrivi_motore()
    segnalatore = None
    if args.segnala is not None:
        segnalatore = SegnalatoreMotore(normalizza_url(args.motore), args.chiave_api, args.segnala or None, info["nome"],
                                        durata_segnale(args.ogni))
        print(f"Segnalazioni al motore: la fascia diventa il livello di salute del device (attenzione 1, con throttling "
              f"termico 2, critica 3, critica con throttling hardware esclusa), ripetuta a ogni campione con una durata "
              f"di {segnalatore.durata} s; tornata normale, e alla fine, il segnale si ritira.")
        segnalatore.avvia()
    if args.file_stato:
        print(f"Stato aggiornato a ogni campione in {args.file_stato} (un file più vecchio di qualche secondo non vale).")
    print("Ctrl+C per fermare: il riepilogo viene scritto lo stesso." if not args.durata
          else f"Si ferma da sola dopo {args.durata} secondi (o con Ctrl+C).")
    try:
        signal.signal(signal.SIGTERM, al_segnale)
    except (ValueError, OSError, AttributeError):
        pass
    ultimo_salvataggio = [time.monotonic()]

    stato_non_scritto = [False]

    def dopo_campione(t, c, campioni):
        if segnalatore is not None:
            fascia = fascia_di(c["temp"], sessione.fasce)
            livello = livello_segnale(fascia, c["mask"])
            if livello is not None:
                segnalatore.aggiorna(livello, motivo_segnale(c, fascia))
        if args.file_stato:
            try:
                scrivi_json(args.file_stato, costruisci_quadro(info, sessione.fasce, c, campioni[-70:]))
                stato_non_scritto[0] = False
            except OSError as errore:
                if not stato_non_scritto[0]:
                    print(f"  Il file di stato non si può scrivere ({errore}): riprovo al prossimo campione.", flush=True)
                stato_non_scritto[0] = True
        if time.monotonic() - ultimo_salvataggio[0] >= 60:
            ultimo_salvataggio[0] = time.monotonic()
            sessione.salva(False)

    esito = 0
    try:
        sessione.registra("OSSERVAZIONE", lambda: False, durata_max=args.durata or None, dopo_campione=dopo_campione)
    except KeyboardInterrupt:
        print("\nInterrotto.")
    except (RuntimeError, OSError) as errore:
        print(f"\nErrore: {errore}.")
        esito = 1
    finally:
        if segnalatore is not None:
            segnalatore.ferma()
    return chiudi_e_riferisci(sessione, esito == 1)


def comando_stato(args):
    def esci_con_errore(messaggio):
        print(json.dumps({"formato": FORMATO_STATO, "errore": messaggio}, ensure_ascii=False, indent=2))
        print(messaggio, file=sys.stderr)
        return 3

    nvsmi = trova_nvidia_smi(args.nvidia_smi)
    if not nvsmi:
        return esci_con_errore("nvidia-smi non trovato")
    info = info_gpu(nvsmi, args.gpu)
    if info is None:
        return esci_con_errore(f"nvidia-smi non risponde per la GPU {args.gpu}")
    c = leggi(nvsmi, args.gpu, campo_dei_motivi(nvsmi, args.gpu))
    if c is None:
        return esci_con_errore("lettura della GPU non riuscita")
    c["t"] = 0.0
    c["istante"] = dt.datetime.now().isoformat(timespec="seconds")
    c["motore"] = None
    if args.motore:
        url = normalizza_url(args.motore)
        c["motore"] = leggi_salute(url, args.chiave_api)
        if c["motore"]["ok"]:
            c["motore"].update(leggi_risorse(url, args.chiave_api) or {})
    fasce = calcola_fasce(info, args.attenzione, args.critica)
    quadro = costruisci_quadro(info, fasce, c)
    print(json.dumps(quadro, ensure_ascii=False, indent=2))
    return CODICE_DI_USCITA.get(quadro["fascia"]["nome"], 3)


def carica_prova(percorso, args):
    """Una prova da un .json di riepilogo o da un .csv; il .csv usa il .json accanto per GPU e fasce, se c'è."""
    if percorso.lower().endswith(".json"):
        with open(percorso, encoding="utf-8") as f:
            dati = json.load(f)
        if not str(dati.get("formato", "")).startswith("termico-riepilogo/"):
            raise ValueError("non è un riepilogo di termico.py")
        return dati
    risultati = campioni_da_csv(percorso)
    accanto = os.path.splitext(percorso)[0] + ".json"
    meta = {}
    if os.path.exists(accanto):
        try:
            with open(accanto, encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError):
            meta = {}
    info = meta.get("gpu") or {"indice": 0, "nome": "GPU sconosciuta", "driver": "?"}
    if args.attenzione is None and args.critica is None and meta.get("fasce"):
        fasce = meta["fasce"]
    else:
        fasce = calcola_fasce(info, args.attenzione, args.critica)
    chiuso = bool(meta.get("chiuso_regolarmente"))
    meta = {"creato": meta.get("creato") or dt.datetime.fromtimestamp(os.path.getmtime(percorso)).isoformat(timespec="seconds"),
            "nome": meta.get("nome"), "gpu": info, "fasce": fasce, "motore": meta.get("motore")}
    return costruisci_riepilogo(meta, risultati, chiuso)


def comando_riepilogo(args):
    try:
        prova = carica_prova(args.file, args)
    except (OSError, ValueError) as errore:
        sys.exit(f"{args.file}: {errore}")
    if not prova["fasi"]:
        sys.exit(f"{args.file}: nessun campione.")
    print("\n".join(testo_riepilogo(prova)))
    if args.json:
        scrivi_json(args.json, prova)
        print(f"\nRiepilogo scritto in {args.json}")
    return 0


METRICHE_CONFRONTO = [
    ("temp_max_c", "temp max (C)", lambda f: f["temp_c"]["max"], 0),
    ("temp_ultimi_60s_c", "temp ultimi 60 s (C)", lambda f: f["temp_c"]["ultimi_60s_mediana"], 1),
    ("potenza_media_w", "potenza media (W)", lambda f: f["potenza_w"]["media"], 1),
    ("clock_medio_mhz", "clock medio (MHz)", lambda f: f["clock_mhz"]["media"], 0),
    ("clock_min_mhz", "clock minimo (MHz)", lambda f: f["clock_mhz"]["min"], 0),
    ("util_media_pct", "utilizzo GPU medio (%)", lambda f: f["util_gpu_pct"]["media"], 0),
    ("throttling_termico_pct", "throttling per il caldo (%)",
     lambda f: f["throttling"]["termico_pct"] if f["throttling"]["leggibile"] else None, 0),
    ("throttling_potenza_pct", "throttling per la potenza (%)",
     lambda f: f["throttling"]["motivi_pct"].get("limite_potenza", 0) if f["throttling"]["leggibile"] else None, 0),
    ("job_riusciti", "job riusciti", lambda f: (f.get("motore") or {}).get("job_riusciti"), 0),
    ("job_falliti", "job falliti", lambda f: (f.get("motore") or {}).get("job_falliti"), 0),
    ("motore_occupato_pct", "motore occupato (%)", lambda f: (f.get("motore") or {}).get("occupato_pct"), 0),
]


def confronta_prove(prove, etichette):
    nomi_fasi = []
    for prova in prove:
        for f in prova["fasi"]:
            if f["nome"] not in nomi_fasi:
                nomi_fasi.append(f["nome"])
    fasi = {}
    for nome in nomi_fasi:
        per_prova = [next((f for f in prova["fasi"] if f["nome"] == nome), None) for prova in prove]
        metriche = {}
        for chiave, _, estrai, _ in METRICHE_CONFRONTO:
            valori = [None if f is None else estrai(f) for f in per_prova]
            if any(v is not None for v in valori):
                metriche[chiave] = valori
        metriche["giudizio"] = [(f["valutazione"] or {}).get("giudizio") if f else None for f in per_prova]
        fasi[nome] = metriche
    return {
        "formato": FORMATO_CONFRONTO,
        "prove": [{"etichetta": e, "nome": p.get("nome"), "gpu": p["gpu"]["nome"], "driver": p["gpu"]["driver"],
                   "creato": p.get("creato"), "chiuso_regolarmente": p.get("chiuso_regolarmente")}
                  for e, p in zip(etichette, prove)],
        "fasi": fasi,
    }


def testo_confronto(c):
    etichette = [p["etichetta"] for p in c["prove"]]
    righe = ["CONFRONTO FRA PROVE"]
    for p in c["prove"]:
        descrizione = f"  {p['etichetta']} = {p['nome'] or '(senza nome)'}, {p['gpu']}, driver {p['driver']}"
        if p["chiuso_regolarmente"] is False:
            descrizione += "  [non chiusa regolarmente]"
        righe.append(descrizione)
    if len({(p["gpu"], p["driver"]) for p in c["prove"]}) > 1:
        righe.append("  Attenzione: GPU o driver diversi. Il confronto vale come ordine di grandezza, non come misura esatta.")
    due = len(c["prove"]) == 2
    for nome, metriche in c["fasi"].items():
        righe += ["", f"Fase {nome}"]
        tab = [["", *etichette] + (["differenza"] if due else [])]
        for chiave, etichetta, _, cifre in METRICHE_CONFRONTO:
            if chiave not in metriche:
                continue
            valori = metriche[chiave]
            riga = [etichetta] + [fmt(v, cifre) for v in valori]
            if due:
                a, b = valori
                riga.append("n/d" if a is None or b is None else f"{b - a:+.{cifre}f}")
            tab.append(riga)
        giudizi = metriche["giudizio"]
        if any(giudizi):
            tab.append(["giudizio (indicativo)"] + [g or "-" for g in giudizi] + ([""] if due else []))
        righe += tabella(tab)
    return righe


def comando_confronta(args):
    if len(args.file) < 2:
        sys.exit("servono almeno due file da confrontare.")
    prove = []
    for percorso in args.file:
        try:
            prova = carica_prova(percorso, args)
        except (OSError, ValueError) as errore:
            sys.exit(f"{percorso}: {errore}")
        if not prova["fasi"]:
            sys.exit(f"{percorso}: nessun campione.")
        prove.append(prova)
    etichette = [chr(ord("A") + i) if i < 26 else str(i + 1) for i in range(len(prove))]
    confronto = confronta_prove(prove, etichette)
    for p, percorso in zip(confronto["prove"], args.file):
        p["file"] = os.path.basename(percorso)
        if not p["nome"]:
            p["nome"] = os.path.splitext(os.path.basename(percorso))[0]
    print("\n".join(testo_confronto(confronto)))
    if args.json:
        scrivi_json(args.json, confronto)
        print(f"\nConfronto scritto in {args.json}")
    return 0


# ----------------------------------------------------------------------------------------------------------
# Riga di comando
# ----------------------------------------------------------------------------------------------------------

class Parser(argparse.ArgumentParser):
    def error(self, message):
        """Un uso sbagliato esce con 3 ("sconosciuto"), non con 2: per `stato` il 2 vuol dire fascia critica."""
        self.print_usage(sys.stderr)
        self.exit(3, f"{self.prog}: errore: {message}\n")


def crea_parser():
    ap = Parser(
        prog="termico.py", formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.split("Comandi")[0].strip(),
        epilog="Senza comando parte la sessione guidata. Guida completa: README.md di questa cartella.")
    ap.add_argument("--versione", action="version", version=f"termico.py {VERSIONE}")
    sotto = ap.add_subparsers(dest="comando", metavar="comando")

    gpu = argparse.ArgumentParser(add_help=False)
    gpu.add_argument("--gpu", type=int, default=0, help="indice della GPU (default 0)")
    gpu.add_argument("--nvidia-smi", dest="nvidia_smi", metavar="PERCORSO", help="percorso di nvidia-smi, se non si trova da solo")
    gpu.add_argument("--attenzione", "--avviso", dest="attenzione", type=float, metavar="C",
                     help="temperatura da cui la fascia è 'attenzione' (default: dai limiti del driver, o 80)")
    gpu.add_argument("--critica", type=float, metavar="C",
                     help="temperatura da cui la fascia è 'critica' (default: dai limiti del driver, o 87)")

    motore = argparse.ArgumentParser(add_help=False)
    motore.add_argument("--motore", metavar="URL", help="affianca lo stato di sd-server, per esempio http://127.0.0.1:1234")
    motore.add_argument("--chiave-api", dest="chiave_api", metavar="CHIAVE",
                        default=os.environ.get("TERMICO_CHIAVE_API"),
                        help="chiave API di sd-server (oppure variabile TERMICO_CHIAVE_API)")
    motore.add_argument("--ogni-motore", dest="ogni_motore", type=float, default=2.0, metavar="S",
                        help="ogni quanti secondi interrogare il motore (default 2)")

    registrazione = argparse.ArgumentParser(add_help=False)
    registrazione.add_argument("--nome", help="etichetta della prova (entra nel nome dei file)")
    registrazione.add_argument("--uscita", default=".", metavar="CARTELLA", help="dove scrivere i file (default: cartella corrente)")
    registrazione.add_argument("--ogni", type=float, default=1.0, metavar="S", help="secondi fra due campioni (default 1)")

    guidata = sotto.add_parser("guidata", parents=[gpu, motore, registrazione], help="sessione a fasi (predefinito)")
    guidata.add_argument("--fasi", default="riposo,motore,chat,insieme,raffreddamento",
                         help="fasi da eseguire, separate da virgola (default: tutte)")
    guidata.add_argument("--auto-secondi", type=int, default=120, metavar="S", help="durata delle fasi automatiche (default 120)")
    guidata.set_defaults(esegui=comando_guidata)

    osserva = sotto.add_parser("osserva", parents=[gpu, motore, registrazione], help="registra senza fasi")
    osserva.add_argument("--durata", type=int, default=0, metavar="S", help="secondi da registrare (default: fino a Ctrl+C)")
    osserva.add_argument("--segnala", nargs="?", const="", default=None, metavar="DEVICE",
                         help="manda al motore (--motore) la fascia della GPU come livello di salute del suo device: "
                              "attenzione 1, con throttling termico 2, critica 3, critica con throttling hardware "
                              "esclusa; DEVICE è il nome nel motore (CUDA0, Vulkan0...), senza si trova dal nome della GPU")
    osserva.add_argument("--file-stato", dest="file_stato", metavar="FILE",
                         help="file JSON riscritto a ogni campione con l'ultima lettura (per altri programmi)")
    osserva.set_defaults(esegui=comando_osserva)

    stato = sotto.add_parser("stato", parents=[gpu, motore], help="una lettura in JSON; il codice di uscita è la fascia")
    stato.set_defaults(esegui=comando_stato)

    riepilogo = sotto.add_parser("riepilogo", parents=[gpu], help="rifà il riepilogo da un CSV")
    riepilogo.add_argument("file", help="CSV (o JSON) di una prova")
    riepilogo.add_argument("--json", metavar="FILE", help="scrive anche il riepilogo in JSON")
    riepilogo.set_defaults(esegui=comando_riepilogo)

    confronta = sotto.add_parser("confronta", parents=[gpu], help="confronta due o più prove")
    confronta.add_argument("file", nargs="+", help="CSV o JSON delle prove (la prima è il riferimento)")
    confronta.add_argument("--json", metavar="FILE", help="scrive anche il confronto in JSON")
    confronta.set_defaults(esegui=comando_confronta)
    return ap


def controlla_argomenti(ap, args):
    if getattr(args, "attenzione", None) is not None and getattr(args, "critica", None) is not None \
            and args.attenzione >= args.critica:
        ap.error("--attenzione deve essere più bassa di --critica")
    for nome in ("ogni", "ogni_motore"):
        if getattr(args, nome, 1.0) < 0.2:
            ap.error("gli intervalli non possono scendere sotto 0,2 secondi: nvidia-smi non regge di più")
    if getattr(args, "durata", 0) < 0:
        ap.error("--durata non può essere negativa")


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or (argv[0].startswith("-") and argv[0] not in ("-h", "--help", "--versione")):
        argv.insert(0, "guidata")  # come nella prima versione: i comandi vecchi funzionano ancora
    ap = crea_parser()
    args = ap.parse_args(argv)
    if not getattr(args, "esegui", None):
        ap.print_help()
        return 0
    controlla_argomenti(ap, args)
    try:
        return args.esegui(args)
    except KeyboardInterrupt:
        print("\nInterrotto.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
