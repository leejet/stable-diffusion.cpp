#!/usr/bin/env python3
"""Prova del guardiano d'esempio di docs/agenti_indipendenti.md (sezione 9): esegue il codice che sta nel documento.

Senza opzioni usa un orologio finto e un motore finto: non servono né un modello né una GPU e dura una decina di secondi.
Con --server e --model prova anche la catena vera, quella che conta: un vero sd-server (su qualunque backend, anche la
CPU), termico.py con la GPU finta di test/termico_test.py e il guardiano che ferma un job quando la GPU diventa critica.

  python3 test/agenti_guardiano_test.py
  python3 test/agenti_guardiano_test.py --server build/bin/sd-server --model /percorso/modello.safetensors

Opzioni per il server (un backend, --flash-attn...) vanno dopo --extra, una per volta: --extra=--flash-attn --extra=on.
Codice di uscita 0 = tutti i controlli superati. Serve Python 3.8 o più recente, senza pacchetti.
"""
import argparse
import contextlib
import http.server
import importlib.util
import io
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

sys.dont_write_bytecode = True  # provare gli script non deve lasciare __pycache__ accanto a loro

QUI = os.path.dirname(os.path.abspath(__file__))
DOCUMENTO = os.path.join(QUI, "..", "docs", "agenti_indipendenti.md")
TERMICO = os.path.join(QUI, "..", "scripts", "diagnostica_termica", "termico.py")
TITOLO_GUARDIANO = "## 9. Un guardiano d'esempio"
RICHIESTA_LUNGA = {"prompt": "a", "width": 64, "height": 64, "seed": 3, "sample_params": {"sample_steps": 100}}  # il massimo del server
RICHIESTA_BREVE = {"prompt": "b", "width": 64, "height": 64, "seed": 4, "sample_params": {"sample_steps": 4}}

errori = 0


def controlla(condizione, messaggio):
    global errori
    if condizione:
        print("ok:   " + messaggio, flush=True)
    else:
        print("FAIL: " + messaggio, flush=True)
        errori += 1
    return condizione


def blocchi_python_del_documento():
    """I blocchi ```python della sezione 9: il guardiano e l'esempio d'uso."""
    with open(DOCUMENTO, encoding="utf-8") as file:
        testo = file.read()
    assert TITOLO_GUARDIANO in testo, "la sezione del guardiano non c'è più nel documento: aggiorna il test"
    blocchi = re.findall(r"```python\n(.*?)\n```", testo.split(TITOLO_GUARDIANO, 1)[1], re.S)
    assert len(blocchi) >= 2, "servono il blocco del guardiano e quello d'uso"
    return blocchi[0], blocchi[1]


def carica_guardiano(codice):
    spazio = {"__name__": "guardiano"}
    exec(compile(codice, DOCUMENTO, "exec"), spazio)
    return spazio


# ---------------------------------------------------------------------------------------------------------
# Parte 1: orologio finto e motore finto
# ---------------------------------------------------------------------------------------------------------
class OrologioSenzaFine(Exception):
    pass


class Orologio:
    """Sostituisce il modulo time dentro il guardiano: sleep() fa avanzare il tempo senza aspettare.

    Un guardiano che non si arrende mai farebbe girare il test all'infinito: oltre cinque minuti di tempo finto
    (le prove più lunghe ne usano un minuto e mezzo) il test si interrompe e lo segnala."""

    LIMITE_S = 300

    def __init__(self, cosa_succede=None):
        self.t = 1000.0
        self.cosa_succede = cosa_succede

    def monotonic(self):
        return self.t

    def time(self):
        return self.t

    def sleep(self, secondi):
        self.t += secondi
        if self.t - 1000.0 > self.LIMITE_S:
            raise OrologioSenzaFine("il guardiano non finisce: oltre %d secondi di tempo finto" % self.LIMITE_S)
        if self.cosa_succede:
            self.cosa_succede(self)

    def strftime(self, formato):
        return time.strftime(formato)


class MotoreFinto:
    """Risponde come sd-server ai pochi percorsi usati dal guardiano e ricorda cosa gli è arrivato."""

    def __init__(self):
        self.job = {"status": "generating"}
        self.codice_job = 200
        self.cancel = []
        self.letture = 0
        self.codice_invio = 202
        self.intestazioni_invio = {}
        self.cancella_davvero = True
        self.autorizzazioni = []
        motore = self

        class Gestore(http.server.BaseHTTPRequestHandler):
            def log_message(self, *argomenti):
                pass

            def _json(self, codice, corpo, intestazioni=None):
                dati = json.dumps(corpo).encode()
                self.send_response(codice)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(dati)))
                for nome, valore in (intestazioni or {}).items():
                    self.send_header(nome, valore)
                self.end_headers()
                self.wfile.write(dati)

            def do_GET(self):
                motore.autorizzazioni.append(self.headers.get("Authorization"))
                motore.letture += 1
                self._json(motore.codice_job, motore.job)

            def do_POST(self):
                motore.autorizzazioni.append(self.headers.get("Authorization"))
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                if self.path.endswith("/cancel?wait=0"):
                    motore.cancel.append(self.path)
                    if motore.cancella_davvero:
                        motore.job = {"status": "cancelled"}
                    self._json(202, {"status": "generating", "cancel_requested": True})
                elif self.path == "/sdcpp/v1/img_gen":
                    self._json(motore.codice_invio, {"id": "job1", "poll_url": "/sdcpp/v1/jobs/job1"}, motore.intestazioni_invio)
                else:
                    self._json(404, {})

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Gestore)
        self.url = "http://127.0.0.1:%d" % self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def chiudi(self):
        self.server.shutdown()
        self.server.server_close()


def prove_con_orologio_finto(ns, cartella):
    print("\n-- guardiano, con orologio e motore finti --")
    ns["FILE_STATO"] = os.path.join(cartella, "stato.json")
    ns["FILE_STOP"] = os.path.join(cartella, "stop")
    registro = []
    registra_vero = ns["registra"]
    ns["registra"] = lambda **campi: registro.append(campi)

    def scrivi(fascia, istante, motivi=None):
        quadro = {"formato": "termico-stato/1", "istante_unix": istante, "gpu": {"motivi_throttling": motivi},
                  "fascia": {"nome": fascia}}
        with open(ns["FILE_STATO"], "w", encoding="utf-8") as file:
            json.dump(quadro, file)

    def nuovo():
        return ns["Guardiano"]()

    # livello(): come si legge una lettura di termico.py
    livello = ns["livello"]
    controlla(livello(None) == "sconosciuto", "livello(None) = sconosciuto")
    controlla(livello({}) == "sconosciuto", "livello({}) = sconosciuto")
    controlla(livello([1, 2]) == "sconosciuto", "livello(lista) = sconosciuto")
    controlla(livello({"fascia": {"nome": "normale"}, "gpu": {}}) == "normale", "fascia normale")
    controlla(livello({"fascia": {"nome": "attenzione"}, "gpu": {"motivi_throttling": None}}) == "attenzione",
              "attenzione, con i motivi di throttling null")
    controlla(livello({"fascia": {"nome": "critica"}, "gpu": {"motivi_throttling": []}}) == "critica", "critica")
    controlla(livello({"fascia": {"nome": "normale"}, "gpu": {"motivi_throttling": ["termico_sw"]}}) == "attenzione",
              "normale + termico_sw = attenzione")
    controlla(livello({"fascia": {"nome": "normale"}, "gpu": {"motivi_throttling": ["termico_hw"]}}) == "attenzione",
              "normale + termico_hw = attenzione")
    controlla(livello({"fascia": {"nome": "normale"}, "gpu": {"motivi_throttling": ["limite_potenza"]}}) == "normale",
              "il limite di potenza non è caldo")
    controlla(livello({"fascia": {"nome": "critica"}, "gpu": {"motivi_throttling": ["termico_sw"]}}) == "critica",
              "la critica resta critica")
    controlla(livello({"fascia": {"nome": "boh"}, "gpu": {}}) == "sconosciuto", "un nome di fascia inventato = sconosciuto")
    controlla(livello({"fascia": {"nome": "normale"}}) == "sconosciuto", "senza il blocco gpu = sconosciuto")
    controlla(livello({"fascia": "normale", "gpu": {}}) == "sconosciuto", "fascia che non è un oggetto = sconosciuto")

    # leggi_stato(): un dato vecchio non è un dato
    leggi = ns["leggi_stato"]
    controlla(leggi() is None, "file di stato mancante = nessun dato")
    for contenuto, perche in (("{non json", "JSON rotto"), ("[1]", "JSON che non è un oggetto"), ("{}", "senza istante_unix")):
        with open(ns["FILE_STATO"], "w", encoding="utf-8") as file:
            file.write(contenuto)
        controlla(leggi() is None, "%s = nessun dato" % perche)
    scrivi("normale", time.time())
    controlla(leggi() is not None, "lettura fresca = dati")
    scrivi("normale", time.time() - 9)
    controlla(leggi() is not None, "lettura di 9 secondi = dati")
    scrivi("normale", time.time() - 11)
    controlla(leggi() is None, "lettura di 11 secondi = nessun dato")
    scrivi("normale", "ieri")
    controlla(leggi() is None, "istante_unix non numerico = nessun dato")

    # Guardiano.decidi(): le regole
    riposo = ns["RIPOSO_S"]
    d = nuovo().decidi("normale", 0)
    controlla(d["nuovi_lavori"] and not d["ferma_il_lavoro_in_corso"] and d["motivo"] == "normale",
              "partenza a freddo: si può lavorare subito")
    g = nuovo()
    d = g.decidi("sconosciuto", 0)
    controlla(not d["nuovi_lavori"] and not d["ferma_il_lavoro_in_corso"] and d["motivo"] == "sconosciuto",
              "dato sconosciuto: niente nuovi lavori, ma il lavoro in corso non si ferma")
    controlla(g.decidi("normale", 5)["nuovi_lavori"], "da sconosciuto a normale si riparte subito (non c'è stata nessuna allerta)")
    g = nuovo()
    d = g.decidi("attenzione", 0)
    controlla(not d["nuovi_lavori"] and not d["ferma_il_lavoro_in_corso"], "attenzione: niente nuovi lavori, il corrente continua")
    esiti = [g.decidi("normale", 10 + t) for t in (0, riposo // 2, riposo - 1)]
    controlla(all(not e["nuovi_lavori"] and e["motivo"] == "riposo" for e in esiti), "tornata normale da meno di RIPOSO_S: ancora riposo")
    controlla(g.decidi("normale", 10 + riposo)["nuovi_lavori"], "normale da RIPOSO_S secondi: si riparte")
    d = nuovo().decidi("critica", 0)
    controlla(d["ferma_il_lavoro_in_corso"] and not d["nuovi_lavori"], "critica: si ferma il lavoro")
    g = nuovo()
    g.decidi("attenzione", 0)
    riparte = False
    for k in range(1, 60):  # la lettura passa avanti e indietro ogni 5 secondi, per cinque minuti
        if g.decidi("normale" if k % 2 else "attenzione", 5 * k)["nuovi_lavori"]:
            riparte = True
    controlla(not riparte, "lettura che sfarfalla attorno alla soglia: non riparte mai")
    g = nuovo()
    g.decidi("critica", 0)
    g.decidi("normale", 10)
    g.decidi("normale", 10 + riposo - 10)
    g.decidi("sconosciuto", 10 + riposo)
    g.decidi("normale", 15 + riposo)
    controlla(not g.decidi("normale", 15 + riposo + riposo - 5)["nuovi_lavori"], "dopo un buco nei dati l'attesa riparte da capo")
    controlla(g.decidi("normale", 15 + 2 * riposo)["nuovi_lavori"], "... e finisce RIPOSO_S secondi dopo il buco")
    tetto = ns["MAX_LAVORI_ALL_ORA"]
    g = nuovo()
    g.partiti.extend(100 + k for k in range(tetto))
    d = g.decidi("normale", 200)
    controlla(not d["nuovi_lavori"] and d["motivo"] == "tetto_orario", "tetto orario raggiunto: niente nuovi lavori")
    d = g.decidi("critica", 200)
    controlla(d["ferma_il_lavoro_in_corso"] and d["motivo"] == "critica", "con il tetto raggiunto la critica resta critica")
    g = nuovo()
    g.partiti.extend(100 + k for k in range(tetto))
    d = g.decidi("normale", 100 + 3600 + tetto + 1)
    controlla(d["nuovi_lavori"] and len(g.partiti) == 0, "dopo un'ora i lavori vecchi non contano più")
    g = nuovo()
    g.partiti.extend(100 + k for k in range(tetto))
    controlla(not g.decidi("normale", 100 + 3600 - 10)["nuovi_lavori"], "... ma a un'ora meno dieci secondi contano ancora")
    g = nuovo()
    g.partiti.extend(100 + k for k in range(tetto - 1))
    controlla(g.decidi("normale", 200)["nuovi_lavori"], "un lavoro sotto il tetto: si può")

    # l'interruttore della persona
    g = nuovo()
    controlla(g.decidi("normale", 0)["nuovi_lavori"], "senza file stop: si lavora")
    open(ns["FILE_STOP"], "w").close()
    d = g.decidi("normale", 1)
    controlla(d == {"nuovi_lavori": False, "ferma_il_lavoro_in_corso": True, "motivo": "stop"},
              "con il file stop: niente lavori e si ferma")
    controlla(g.decidi("sconosciuto", 2)["ferma_il_lavoro_in_corso"], "il file stop vale anche con il dato sconosciuto")
    os.remove(ns["FILE_STOP"])
    controlla(g.decidi("normale", 3)["nuovi_lavori"], "tolto il file stop si riparte (non c'era stata nessuna allerta)")

    # registra(): una riga JSON leggibile
    uscita = io.StringIO()
    with contextlib.redirect_stdout(uscita):
        registra_vero(evento="prova", motivo="è tutto ok")
    riga = json.loads(uscita.getvalue())
    controlla(riga["evento"] == "prova" and riga["motivo"] == "è tutto ok"
              and re.match(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d$", riga["quando"]), "registra: una riga JSON con l'ora")
    controlla("è" in uscita.getvalue(), "registra: le lettere accentate restano leggibili")

    # avvia_lavoro(): parte solo se il guardiano lo permette
    motore = MotoreFinto()
    ns["MOTORE"] = motore.url
    scrivi("normale", time.time())
    g = nuovo()
    percorso, perche = ns["avvia_lavoro"](g, {"prompt": "x"})
    controlla(percorso == "/sdcpp/v1/jobs/job1" and perche is None and len(g.partiti) == 1, "avvia_lavoro: parte e conta il lavoro")
    controlla(registro and registro[-1]["evento"] == "avvio" and registro[-1]["motivo"] == "normale" and registro[-1]["nuovi_lavori"],
              "avvia_lavoro: scrive nel registro la decisione e il motivo")
    controlla(motore.autorizzazioni[-1] is None, "senza chiave API non c'è l'intestazione Authorization")
    ns["CHIAVE_API"] = "segreta"
    ns["avvia_lavoro"](nuovo(), {"prompt": "x"})
    ns["chiama"]("GET", "/sdcpp/v1/jobs/job1")
    controlla(motore.autorizzazioni[-2:] == ["Bearer segreta"] * 2, "con la chiave API: Authorization su POST e su GET")
    ns["CHIAVE_API"] = ""
    scrivi("attenzione", time.time())
    percorso, perche = ns["avvia_lavoro"](g, {"prompt": "x"})
    controlla(percorso is None and "attenzione" in perche and len(g.partiti) == 1, "avvia_lavoro: in attenzione non parte (%s)" % perche)
    controlla(registro[-1]["motivo"] == "attenzione" and not registro[-1]["nuovi_lavori"], "avvia_lavoro: anche il rifiuto è nel registro")
    scrivi("normale", time.time())
    open(ns["FILE_STOP"], "w").close()
    percorso, perche = ns["avvia_lavoro"](nuovo(), {"prompt": "x"})
    controlla(percorso is None and perche == "non ora: stop", "avvia_lavoro: con il file stop non parte (%s)" % perche)
    os.remove(ns["FILE_STOP"])
    g = nuovo()
    motore.codice_invio, motore.intestazioni_invio = 429, {"Retry-After": "7"}
    percorso, perche = ns["avvia_lavoro"](g, {"prompt": "x"})
    controlla(percorso is None and "7 s" in perche and not g.partiti, "avvia_lavoro: 429 con Retry-After 7 (%s)" % perche)
    motore.codice_invio, motore.intestazioni_invio = 503, {}
    percorso, perche = ns["avvia_lavoro"](g, {"prompt": "x"})
    controlla(percorso is None and "10 s" in perche, "avvia_lavoro: 503 senza intestazione = 10 secondi (%s)" % perche)
    motore.codice_invio = 400
    percorso, perche = ns["avvia_lavoro"](g, {"prompt": "x"})
    controlla(percorso is None and "400" in perche and not g.partiti, "avvia_lavoro: 400 rifiutato (%s)" % perche)
    motore.chiudi()
    percorso, perche = ns["avvia_lavoro"](g, {"prompt": "x"})
    controlla(percorso is None and "non risponde" in perche, "avvia_lavoro: motore spento (%s)" % perche)

    # segui_lavoro(): con il tempo finto
    def segui(cosa_succede, scadenza_s=600, prepara=None):
        m = MotoreFinto()
        ns["MOTORE"] = m.url
        if prepara:
            prepara(m)
        orologio = Orologio(lambda o: cosa_succede(o, m))
        ns["time"] = orologio
        scrivi("normale", orologio.t)
        try:
            risultato = ns["segui_lavoro"](nuovo(), "/sdcpp/v1/jobs/job1", scadenza_s=scadenza_s)
        finally:
            ns["time"] = time
            m.chiudi()
        return risultato, m, orologio

    def finisce_al_terzo_controllo(o, m):
        scrivi("normale", o.t)
        if m.letture >= 3:
            m.job = {"status": "completed"}

    def critica_al_secondo_controllo(o, m):
        scrivi("critica" if m.letture >= 2 else "normale", o.t)

    def sempre_normale(o, m):
        scrivi("normale", o.t)

    def stop_al_secondo_controllo(o, m):
        scrivi("normale", o.t)
        if m.letture >= 2:
            open(ns["FILE_STOP"], "w").close()

    del registro[:]
    (job, perche), m, o = segui(finisce_al_terzo_controllo)
    controlla(job["status"] == "completed" and perche is None and not m.cancel and not registro,
              "segui: un job che finisce da solo non viene toccato né registrato")
    (job, perche), m, o = segui(critica_al_secondo_controllo)
    controlla(job["status"] == "cancelled" and perche == "gpu_critica",
              "segui: GPU critica -> job fermato (%s, %s)" % (job["status"], perche))
    controlla(m.cancel and set(m.cancel) == {"/sdcpp/v1/jobs/job1/cancel?wait=0"}, "segui: ferma solo il proprio job (%s)" % m.cancel[:1])
    controlla(len(registro) == 1 and registro[0]["motivo"] == "gpu_critica",
              "segui: un solo evento di stop nel registro, anche con più richieste")
    del registro[:]
    (job, perche), m, o = segui(stop_al_secondo_controllo)
    if os.path.exists(ns["FILE_STOP"]):
        os.remove(ns["FILE_STOP"])
    controlla(job["status"] == "cancelled" and perche == "stop", "segui: il file stop ferma il job (%s, %s)" % (job["status"], perche))
    controlla(registro and registro[-1] == {"evento": "stop", "motivo": "stop", "job": "/sdcpp/v1/jobs/job1"},
              "segui: lo stop è nel registro con il job")
    (job, perche), m, o = segui(sempre_normale, scadenza_s=10)
    controlla(job["status"] == "cancelled" and perche == "scaduto", "segui: scadenza -> job fermato (%s, %s)" % (job["status"], perche))
    controlla(o.t - 1000.0 <= 14, "segui: una scadenza di 10 s ferma entro 14 s di tempo finto (%.0f)" % (o.t - 1000.0))

    def ignora_il_cancel(m):
        m.cancella_davvero = False
    (job, perche), m, o = segui(sempre_normale, scadenza_s=10, prepara=ignora_il_cancel)
    controlla(job["status"] == "generating" and perche == "scaduto",
              "segui: stop ignorato dal motore -> lo dice (%s, %s)" % (job["status"], perche))
    controlla(20 <= len(m.cancel) <= 40, "segui: le richieste di stop restano poche (%d)" % len(m.cancel))
    controlla(o.t - 1000.0 <= 10 + 60 + 4, "segui: in tutto al massimo scadenza + 60 secondi (%.0f)" % (o.t - 1000.0))

    def job_scartato(m):
        m.codice_job, m.job = 410, {"error": "gone"}
    (job, perche), m, o = segui(sempre_normale, prepara=job_scartato)
    controlla(perche == "job_sparito" and not m.cancel, "segui: job scartato dal motore (410) = job_sparito")

    spento = MotoreFinto()
    ns["MOTORE"] = spento.url
    spento.chiudi()
    orologio = Orologio(lambda o: scrivi("normale", o.t))
    ns["time"] = orologio
    scrivi("normale", orologio.t)
    try:
        job, perche = ns["segui_lavoro"](nuovo(), "/sdcpp/v1/jobs/job1", scadenza_s=20)
    finally:
        ns["time"] = time
    controlla(perche == "scaduto" and job == {} and orologio.t - 1000.0 <= 20 + 60 + 4,
              "segui: motore spento = finisce entro scadenza + 60 secondi, senza eccezioni (%.0f s)" % (orologio.t - 1000.0))


# ---------------------------------------------------------------------------------------------------------
# Parte 2: un vero sd-server, termico.py e una GPU finta
# ---------------------------------------------------------------------------------------------------------
def carica_modulo(nome, percorso):
    spec = importlib.util.spec_from_file_location(nome, percorso)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def http_json(metodo, url, corpo=None, timeout=60):
    dati = None if corpo is None else json.dumps(corpo).encode()
    richiesta = urllib.request.Request(url, data=dati, method=metodo, headers={"Content-Type": "application/json"})
    apri = urllib.request.build_opener(urllib.request.ProxyHandler({})).open
    try:
        with apri(richiesta, timeout=timeout) as risposta:
            return risposta.status, json.load(risposta)
    except urllib.error.HTTPError as errore:
        return errore.code, json.loads(errore.read().decode() or "{}")


def porta_libera():
    with socket.socket() as presa:
        presa.bind(("127.0.0.1", 0))
        return presa.getsockname()[1]


def avvia_server(args, log):
    porta = porta_libera()
    comando = [args.server, "-m", args.model, "--listen-ip", "127.0.0.1", "--listen-port", str(porta)] + args.extra
    processo = subprocess.Popen(comando, stdout=log, stderr=subprocess.STDOUT)
    url = "http://127.0.0.1:%d" % porta
    inizio = time.time()
    while time.time() - inizio < args.warmup_timeout:
        if processo.poll() is not None:
            raise SystemExit("il server è uscito all'avvio (codice %s): guarda il suo log" % processo.returncode)
        try:
            codice, salute = http_json("GET", url + "/sdcpp/v1/health", timeout=5)
            if codice == 200 and salute.get("model_loaded"):
                return processo, url
        except (OSError, ValueError):
            pass
        time.sleep(1)
    processo.kill()
    raise SystemExit("il server non è pronto dopo %d secondi" % args.warmup_timeout)


def manda_lavoro(url, passi, seme=1):
    codice, job = http_json("POST", url + "/sdcpp/v1/img_gen", {"prompt": "t", "width": 64, "height": 64, "seed": seme,
                                                                "sample_params": {"sample_steps": passi}})
    assert codice == 202, (codice, job)
    return job["id"]


def aspetta_fine(url, id_job, timeout):
    inizio = time.time()
    while time.time() - inizio < timeout:
        codice, job = http_json("GET", url + "/sdcpp/v1/jobs/" + id_job)
        if codice == 200 and job["status"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(0.5)
    raise SystemExit("il job %s non è finito in %d secondi" % (id_job, timeout))


def conta(url):
    return http_json("GET", url + "/sdcpp/v1/health")[1]


def prove_con_server_vero(args, ns, cartella):
    print("\n-- guardiano, con un vero sd-server, termico.py e una GPU finta --")
    termico_test = carica_modulo("termico_test", os.path.join(QUI, "termico_test.py"))
    log = open(os.path.join(cartella, "server.log"), "w")
    server, url = avvia_server(args, log)
    osservatori = []
    try:
        riscaldamento = aspetta_fine(url, manda_lavoro(url, 2), args.warmup_timeout)
        controlla(riscaldamento["status"] == "completed", "il lavoro di riscaldamento del server è completato")

        gpu = termico_test.Gpu(cartella)
        gpu.imposta("temp", 50)
        uscita = os.path.join(cartella, "uscita")
        os.makedirs(uscita)

        def avvia_osservatore():
            processo = subprocess.Popen(
                [sys.executable, "-I", TERMICO, "osserva", "--nvidia-smi", gpu.file, "--motore", url, "--nome", "guardiano",
                 "--uscita", uscita, "--file-stato", os.path.join(uscita, "stato.json"), "--ogni", "1"],
                env=gpu.ambiente(), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            osservatori.append(processo)
            return processo

        osservatore = avvia_osservatore()
        registro = []
        ns["FILE_STATO"], ns["MOTORE"], ns["RIPOSO_S"] = os.path.join(uscita, "stato.json"), url, 6
        ns["FILE_STOP"] = os.path.join(cartella, "stop")
        ns["registra"] = lambda **campi: registro.append(campi)
        for _ in range(40):
            if ns["leggi_stato"]():
                break
            time.sleep(0.5)
        controlla(ns["livello"](ns["leggi_stato"]()) == "normale", "a 50 C il guardiano legge 'normale' dal file scritto da termico.py")

        def segui_in_un_thread(guardiano, percorso, scadenza_s=120):
            esito = {}
            thread = threading.Thread(target=lambda: esito.update(r=ns["segui_lavoro"](guardiano, percorso, scadenza_s=scadenza_s)))
            thread.start()
            return thread, esito

        # A: la GPU diventa critica mentre un lavoro genera
        prima = conta(url)["counters"]
        g = ns["Guardiano"]()
        percorso, perche = ns["avvia_lavoro"](g, RICHIESTA_LUNGA)
        controlla(percorso is not None and perche is None, "A: con la GPU a 50 C il lavoro parte")
        thread, esito = segui_in_un_thread(g, percorso)
        time.sleep(8)
        controlla(conta(url)["busy"] is True, "A: dopo 8 secondi il motore sta ancora generando")
        gpu.imposta("temp", 90)
        thread.join(timeout=90)
        job, fermato = esito.get("r", ({}, None))
        controlla(not thread.is_alive() and job.get("status") == "cancelled" and fermato == "gpu_critica",
                  "A: GPU a 90 C -> il guardiano ferma il job (%s, %s)" % (job.get("status"), fermato))
        dopo = conta(url)
        differenza = {k: dopo["counters"][k] - prima[k] for k in prima}
        controlla(differenza["cancelled"] == 1 and differenza["step_downs"] == 0 and differenza["failed"] == 0
                  and dopo["model_loaded"] and dopo["profile"] == dopo["base_profile"] and dopo["busy"] is False,
                  "A: /health: cancelled +1, nessuna discesa di profilo, nessun fallimento, motore di nuovo libero (%s)" % differenza)
        controlla([r["evento"] for r in registro] == ["avvio", "stop"] and registro[-1]["motivo"] == "gpu_critica",
                  "A: nel registro, l'avvio e lo stop con il motivo")

        # B: tornata normale, si aspetta prima di ripartire
        gpu.imposta("temp", 50)
        time.sleep(3)
        percorso, perche = ns["avvia_lavoro"](g, RICHIESTA_BREVE)
        controlla(percorso is None and "riposo" in perche, "B: tornata a 50 C, il guardiano aspetta ancora (%s)" % perche)
        richieste = conta(url)["counters"]["requests"]
        time.sleep(ns["RIPOSO_S"] + 1)
        percorso, perche = ns["avvia_lavoro"](g, RICHIESTA_BREVE)
        controlla(percorso is not None, "B: dopo RIPOSO_S secondi di fila a normale il lavoro parte")
        job, fermato = ns["segui_lavoro"](g, percorso, scadenza_s=args.warmup_timeout)
        controlla(job["status"] == "completed" and fermato is None, "B: e finisce da solo (%s, %s)" % (job["status"], fermato))
        controlla(conta(url)["counters"]["requests"] == richieste + 1, "B: al motore è arrivata una sola richiesta")

        # E: il job di un altro programma non si tocca
        altro = manda_lavoro(url, 100, seme=9)
        time.sleep(3)
        g2 = ns["Guardiano"]()
        percorso, perche = ns["avvia_lavoro"](g2, RICHIESTA_LUNGA)
        controlla(percorso is not None, "E: il lavoro del guardiano entra in coda dietro a quello dell'altro programma")
        thread, esito = segui_in_un_thread(g2, percorso)
        time.sleep(5)
        gpu.imposta("temp", 90)
        thread.join(timeout=90)
        job, fermato = esito.get("r", ({}, None))
        controlla(job.get("status") == "cancelled" and fermato == "gpu_critica",
                  "E: il job in coda del guardiano viene fermato (%s, %s)" % (job.get("status"), fermato))
        controlla(http_json("GET", url + "/sdcpp/v1/jobs/" + altro)[1].get("status") == "generating",
                  "E: il job dell'altro programma continua a generare")
        http_json("POST", url + "/sdcpp/v1/jobs/%s/cancel?wait=10" % altro)
        controlla(aspetta_fine(url, altro, 60)["status"] == "cancelled", "E: (pulizia) il job dell'altro programma è fermato a mano")

        # C: l'osservatore si ferma e il dato invecchia
        gpu.imposta("temp", 50)
        osservatore.terminate()
        osservatore.wait(timeout=30)
        time.sleep(ns["ETA_MASSIMA_S"] + 2)
        controlla(ns["leggi_stato"]() is None, "C: osservatore fermo da oltre ETA_MASSIMA_S secondi = nessun dato")
        richieste = conta(url)["counters"]["requests"]
        g3 = ns["Guardiano"]()
        percorso, perche = ns["avvia_lavoro"](g3, RICHIESTA_LUNGA)
        controlla(percorso is None and "sconosciuto" in perche and conta(url)["counters"]["requests"] == richieste,
                  "C: con il dato vecchio non parte niente (%s)" % perche)

        # D: senza dati il lavoro in corso continua fino alla sua scadenza, poi si ferma
        id_job = manda_lavoro(url, 100, seme=11)
        inizio = time.time()
        job, fermato = ns["segui_lavoro"](g3, "/sdcpp/v1/jobs/" + id_job, scadenza_s=8)
        controlla(job["status"] == "cancelled" and fermato == "scaduto" and 8 <= time.time() - inizio < 30,
                  "D: senza dati il lavoro continua fino alla scadenza e poi si ferma (%.0f s, %s, %s)"
                  % (time.time() - inizio, job["status"], fermato))

        # F: l'esempio d'uso del documento, così com'è (ridotto a 64x64 e 4 passi)
        avvia_osservatore()
        time.sleep(4)
        gpu.imposta("temp", 50)
        blocco_guardiano, blocco_uso = blocchi_python_del_documento()
        uso = blocco_uso.replace('"width": 512, "height": 512, "sample_params": {"sample_steps": 20}',
                                 '"width": 64, "height": 64, "sample_params": {"sample_steps": 4}')
        controlla(uso != blocco_uso, "F: l'esempio d'uso contiene la richiesta da ridurre (se il documento è cambiato, aggiorna il test)")
        stampato = io.StringIO()
        with contextlib.redirect_stdout(stampato):
            exec(compile(uso, DOCUMENTO, "exec"), dict(ns))
        controlla(stampato.getvalue().strip() == "completed None", "F: l'esempio d'uso del documento gira e stampa 'completed None'")

        # G: l'interruttore della persona
        g4 = ns["Guardiano"]()
        percorso, perche = ns["avvia_lavoro"](g4, RICHIESTA_LUNGA)
        controlla(percorso is not None, "G: con la GPU a 50 C il lavoro lungo parte")
        thread, esito = segui_in_un_thread(g4, percorso)
        time.sleep(5)
        open(ns["FILE_STOP"], "w").close()
        thread.join(timeout=90)
        job, fermato = esito.get("r", ({}, None))
        controlla(not thread.is_alive() and job.get("status") == "cancelled" and fermato == "stop",
                  "G: il file stop ferma il job (%s, %s)" % (job.get("status"), fermato))
        percorso, perche = ns["avvia_lavoro"](g4, RICHIESTA_LUNGA)
        controlla(percorso is None and perche == "non ora: stop", "G: con il file stop non parte niente (%s)" % perche)
        os.remove(ns["FILE_STOP"])
        percorso, perche = ns["avvia_lavoro"](g4, RICHIESTA_BREVE)
        controlla(percorso is not None, "G: tolto il file stop si riparte")
        job, fermato = ns["segui_lavoro"](g4, percorso, scadenza_s=args.warmup_timeout)
        controlla(job["status"] == "completed", "G: e il lavoro finisce (%s)" % job["status"])
    finally:
        for processo in osservatori:
            if processo.poll() is None:
                processo.terminate()
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()


def main():
    parser = argparse.ArgumentParser(description="Prova del guardiano d'esempio di docs/agenti_indipendenti.md")
    parser.add_argument("--server", help="percorso di sd-server: attiva le prove con un vero server")
    parser.add_argument("--model", help="un modello qualsiasi che il server riesca a caricare")
    parser.add_argument("--extra", action="append", default=[], help="opzione in più per il server (una per volta, con =)")
    parser.add_argument("--warmup-timeout", type=int, default=600, help="secondi per caricare il modello e finire un lavoro (default 600)")
    args = parser.parse_args()
    if bool(args.server) != bool(args.model):
        parser.error("--server e --model vanno insieme")
    codice, _ = blocchi_python_del_documento()
    with tempfile.TemporaryDirectory(prefix="guardiano-") as cartella:
        try:
            prove_con_orologio_finto(carica_guardiano(codice), cartella)
        except OrologioSenzaFine as errore:
            controlla(False, "con l'orologio finto: %s" % errore)
        if args.server:
            prove_con_server_vero(args, carica_guardiano(codice), cartella)
        else:
            print("\n(le prove con un vero sd-server si attivano con --server e --model)")
    print("\nTUTTO SUPERATO (0 controlli falliti)" if errori == 0 else "\nFALLITI: %d controlli" % errori)
    return 1 if errori else 0


if __name__ == "__main__":
    sys.exit(main())
