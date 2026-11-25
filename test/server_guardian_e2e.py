#!/usr/bin/env python3
"""End-to-end test of the guardian of sd-server and of the reports of outside watchers, on a real server.

It needs a real model, any model the server can load, on any backend; the jobs are small (64x64). The standard
library is enough.

  python3 test/server_guardian_e2e.py --server build/bin/sd-server --model /path/to/model.safetensors

Through POST /sdcpp/v1/devices/{device}/pressure it checks that:
  - a report that is not valid is refused, and one for a device the server does not have is 404;
  - a level of strain on the RAM leaves the server working;
  - the RAM reported critical while a job generates halts the server: the job ends with the code halted, the queued
    job waits, new work is refused with 503 and Retry-After, and the server resumes by itself when the signal ends;
  - the same while a synchronous request generates: the request is answered 503 with the code halted;
  - every compute device reported critical while a job computes on it: the job stops, the model is not loaded
    anywhere else, and the server stays halted after the signals until an operator resumes it
    (POST /sdcpp/v1/guardian/resume);
  - a second server, with --guardian off and --device-health levels=2,step=20,quiet=5, never halts: a critical report
    only changes device health; its components have two levels of strain (80%, 60%, then excluded) and come back one
    step every 5 s;
  - termico.py osserva --segnala, with the fake GPU of test/termico_test.py, reports to that server: the attention
    band as level 1, and the critical band as level 2, the last of the two, not as an exclusion;
  - sd-cli, next to the server or given with --cli, applies --device-health at start, in the DGL mode too, and
    refuses values against the rules.

The test starts its servers with an API key of its own, to check that the reports need it. Extra options for the
servers (a backend, ...) go after --extra, one each, for example --extra=--backend --extra=cpu. To use a server that
is already running: --url http://127.0.0.1:1234 (add --api-key if it asks for one); the scenario with --guardian off
and the checks of the server log are then skipped, and the server is left with signals of the source "e2e" that end
within two minutes. Exit code 0 = all checks passed.
"""
import argparse
import importlib.util
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

failures = 0
FINAL = ("completed", "failed", "cancelled")


def check(condition, message):
    global failures
    if condition:
        print("ok:   " + message, flush=True)
    else:
        print("FAIL: " + message, flush=True)
        failures += 1
    return condition


class Abort(Exception):
    """The test cannot go on: what comes next depends on what just failed."""


def wait_until(condition, timeout, interval=0.2):
    started = time.time()
    while time.time() - started < timeout:
        if condition():
            return True
        time.sleep(interval)
    return condition()


class Client:
    def __init__(self, base_url, api_key):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def call(self, method, path, body=None, timeout=120, raw=None):
        """Returns the status, the JSON body (or the text when it is not JSON) and the headers."""
        data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = urllib.request.Request(self.base_url + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status, payload, answer_headers = response.status, response.read(), response.headers
        except urllib.error.HTTPError as error:
            status, payload, answer_headers = error.code, error.read(), error.headers
        try:
            parsed = json.loads(payload) if payload else None
        except ValueError:
            parsed = payload.decode(errors="replace")
        return status, parsed, answer_headers

    def health(self):
        return self.call("GET", "/sdcpp/v1/health")

    def pressure(self, device, body):
        return self.call("POST", "/sdcpp/v1/devices/%s/pressure" % urllib.parse.quote(device, safe=""), body)

    def submit(self, steps, seed, size):
        status, job, _ = self.call("POST", "/sdcpp/v1/img_gen", generation_body(steps, seed, size))
        if status != 202:
            raise Abort("the server did not accept a job: %s %s" % (status, job))
        return job["id"]

    def job(self, job_id):
        return self.call("GET", "/sdcpp/v1/jobs/" + job_id)[1]

    def wait_for(self, job_id, wanted, timeout):
        started = time.time()
        while time.time() - started < timeout:
            job = self.job(job_id)
            if job["status"] in wanted:
                return job
            if job["status"] in FINAL:
                raise Abort("job %s ended as %s, expected %s: %s" % (job_id, job["status"], wanted, job.get("error")))
            time.sleep(0.1)
        raise Abort("job %s did not become %s within %d s" % (job_id, wanted, timeout))


def generation_body(steps, seed, size):
    return {"prompt": "a test", "width": size, "height": size, "seed": seed, "sample_params": {"sample_steps": steps}}


def is_halted_refusal(status, body, headers):
    try:
        retry_after = int(headers.get("Retry-After", "0"))
    except ValueError:
        retry_after = 0
    return status == 503 and isinstance(body, dict) and body.get("code") == "halted" and retry_after >= 10


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start_server(args, log, extra=()):
    port = free_port()
    command = [args.server, "-m", args.model, "--listen-ip", "127.0.0.1", "--listen-port", str(port)]
    command += ["--api-key", args.api_key] + args.extra + list(extra)
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    return process, "http://127.0.0.1:%d" % port


def wait_ready(client, process, timeout):
    started = time.time()
    while time.time() - started < timeout:
        if process is not None and process.poll() is not None:
            raise Abort("the server stopped at start-up (exit code %s)" % process.returncode)
        try:
            status, health, _ = client.call("GET", "/sdcpp/v1/health", timeout=5)
            if status == 200 and health.get("status") in ("ok", "degraded"):
                return health
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.5)
    raise Abort("the server was not ready after %d s" % timeout)


def stop_server(process):
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()


def device(answer, name):
    for entry in answer.get("devices", []) if isinstance(answer, dict) else []:
        if entry["name"].lower() == name.lower():
            return entry
    return {}


def run_scenarios(client, args):
    size = args.size
    print("warming up (the first generation may read the weights from disk)...", flush=True)
    warm = client.wait_for(client.submit(2, 1, size), ("completed",), args.warmup_timeout)
    check(warm["status"] == "completed", "warm-up job completed")

    # 1. The health shows the guardian and every component of device health.
    status, health, _ = client.health()
    guardian = health.get("guardian", {}) if isinstance(health, dict) else {}
    check(status == 200 and guardian.get("enabled") is True and guardian.get("halted") is False,
          "the health reports the guardian, on and not halted")
    names = [entry["name"] for entry in health.get("devices", [])]
    check({"CPU", "RAM", "disk"} <= set(names), "and the components of device health: %s" % ", ".join(names))
    check(names and all(entry["level"] == 0 and entry["share_percent"] == 100 and not entry["excluded"] for entry in health["devices"]),
          "all of them healthy at the start")
    compute = [name for name in names if name.lower() not in ("ram", "disk")]
    before = health.get("counters", {})
    start_status = health.get("status")
    if not compute:
        raise Abort("the health lists no compute device")

    # 2. Reports that are not valid.
    status, body, _ = client.pressure("NoSuchDevice0", {"level": 1})
    check(status == 404 and isinstance(body, dict) and body.get("code") == "unknown_device", "a report for a device the server does not have is 404")
    for wrong in ({"level": 9}, {"level": "hot"}, {"level": 1, "duration_seconds": 0}, {"level": 1, "source": "two words"}, {"lvl": 1}):
        status, body, _ = client.pressure("RAM", wrong)
        check(status == 400 and isinstance(body, dict) and body.get("code") == "invalid_report" and body.get("error"),
              "a report that is not valid is 400 with the reason: %s" % json.dumps(wrong))
    status, body, _ = client.call("POST", "/sdcpp/v1/devices/RAM/pressure", raw=b"not json")
    check(status == 400, "a body that is not JSON is 400")
    if client.api_key:
        anonymous = Client(client.base_url, None)
        check(anonymous.pressure("RAM", {"level": 1})[0] == 401, "a report without the API key is refused")
        check(anonymous.call("POST", "/sdcpp/v1/guardian/resume")[0] == 401, "and so is a resume")
    check(device(client.health()[1], "RAM").get("level") == 0, "none of them changed device health")

    # 3. A level of strain: the server keeps working.
    status, body, _ = client.pressure("ram", {"level": 2, "source": "e2e", "reason": "test strain", "duration_seconds": 30})
    ram = body.get("device", {}) if isinstance(body, dict) else {}
    check(status == 200 and ram.get("name") == "RAM" and ram.get("level") == 2 and ram.get("share_percent") == 70,
          "the RAM reported at level 2 (its name in any case) offers 70%% of its capacity (got %s %s)" % (status, body))
    check("e2e: test strain" in ram.get("reason", "") and ram.get("held_ms", 0) > 25000,
          "the reason names the source, and the signal is held for the time given")
    check(isinstance(body, dict) and body.get("guardian", {}).get("halted") is False, "strain does not halt the server")
    done = client.wait_for(client.submit(2, 2, size), ("completed",), args.warmup_timeout)
    check(done["status"] == "completed", "a job completes with the RAM under strain")
    status, body, _ = client.pressure("RAM", {"level": 0, "source": "e2e"})
    ram = body.get("device", {}) if isinstance(body, dict) else {}
    check(status == 200 and ram.get("held_ms") == 0 and ram.get("level") == 2 and ram.get("next_step_ms", 0) > 0,
          "withdrawing the signal leaves the level, which comes down one step after each quiet period")

    # 4. The RAM reported critical while a job generates.
    running = client.submit(args.steps, 3, size)
    client.wait_for(running, ("generating",), 60)
    queued = client.submit(2, 4, size)
    time.sleep(1.0)
    status, body, _ = client.pressure("RAM", {"level": "critical", "source": "e2e", "reason": "test overheat", "duration_seconds": args.critical_seconds})
    halted_at = time.time()
    state = body.get("guardian", {}) if isinstance(body, dict) else {}
    check(status == 200 and state.get("halted") is True and "RAM is critical" in state.get("reason", ""),
          "a critical report halts the server at once (got %s %s)" % (status, body))
    job = {}
    if wait_until(lambda: client.job(running)["status"] in FINAL, 30):
        job = client.job(running)
    error = job.get("error") or {}
    check(job.get("status") == "failed" and error.get("code") == "halted" and "test overheat" in error.get("message", ""),
          "the job that was generating ends with the code halted and the reason (%s)" % json.dumps(error))
    check(time.time() - halted_at < 15, "within seconds of the report (%.1f s)" % (time.time() - halted_at))
    check(client.job(queued)["status"] == "queued", "the queued job waits")
    refused = [
        ("POST", "/sdcpp/v1/img_gen", generation_body(2, 5, size)),
        ("POST", "/sdapi/v1/txt2img", {"prompt": "a test", "width": size, "height": size, "steps": 2}),
        ("POST", "/v1/images/generations", {"prompt": "a test", "size": "%dx%d" % (size, size)}),
        ("POST", "/sdcpp/v1/dgl/jobs", {"source": "x"}),
        ("POST", "/sdcpp/v1/models/load", {}),
    ]
    for method, path, request_body in refused:
        status, answer, headers = client.call(method, path, request_body)
        check(is_halted_refusal(status, answer, headers),
              "%s %s is refused with 503, the code halted and Retry-After (got %s %s)" % (method, path, status, answer))
    status, health, headers = client.health()
    check(is_halted_refusal(status, {"code": "halted"} if health.get("status") == "halted" else {}, headers) and
          health.get("guardian", {}).get("halted") is True, "the health says halted, with 503 and Retry-After")
    check(device(health, "RAM").get("critical") is True, "and shows the RAM critical")
    check(client.call("GET", "/sdcpp/v1/jobs/" + queued)[0] == 200, "polling a job still works")
    check(client.pressure("disk", {"level": 1, "source": "e2e", "duration_seconds": 5})[0] == 200, "and so do reports")
    check(wait_until(lambda: client.health()[0] == 200, args.critical_seconds + 20),
          "the server resumes by itself when the critical signal ends")
    status, health, _ = client.health()
    ram = device(health, "RAM")
    check(health.get("guardian", {}).get("halted") is False and ram.get("excluded") is True and ram.get("critical") is False,
          "the RAM stays excluded for a while, which does not halt the server: it is not a compute device")
    done = client.wait_for(queued, ("completed",), args.warmup_timeout)
    check(done["status"] == "completed", "the job that waited runs and completes")

    # 5. The same while a synchronous request generates.
    outcome = {}

    def synchronous_request():
        outcome["result"] = client.call("POST", "/sdapi/v1/txt2img", {
            "prompt": "a sync test", "width": size, "height": size, "steps": args.steps, "seed": 7,
        }, timeout=args.warmup_timeout)

    thread = threading.Thread(target=synchronous_request)
    thread.start()
    check(wait_until(lambda: client.health()[1].get("busy") is True, 60), "a synchronous request generates")
    time.sleep(1.0)
    client.pressure("RAM", {"level": "critical", "source": "e2e", "reason": "test overheat again", "duration_seconds": args.critical_seconds})
    thread.join(60)
    status, answer, headers = outcome.get("result", (None, None, {}))
    check(is_halted_refusal(status, answer, headers),
          "the synchronous request is stopped and answered 503 with the code halted (got %s %s)" % (status, answer))
    check(wait_until(lambda: client.health()[0] == 200, args.critical_seconds + 20), "the server resumes by itself again")

    # 6. Every compute device reported critical while a job generates on it, as a thermal watcher does: the engine
    # refuses the excluded device, and the server must stop rather than load the model somewhere else. When the
    # signals end the devices are still excluded, so the server stays halted until an operator resumes it.
    running = client.submit(args.steps, 6, size)
    client.wait_for(running, ("generating",), 60)
    time.sleep(1.0)
    for name in compute:
        client.pressure(name, {"level": "critical", "source": "e2e", "reason": "test overheat of " + name, "duration_seconds": args.critical_seconds})
    job = {}
    if wait_until(lambda: client.job(running)["status"] in FINAL, 30):
        job = client.job(running)
    error = job.get("error") or {}
    check(job.get("status") == "failed" and error.get("code") == "halted",
          "with every compute device (%s) critical, the job that computes on them ends with the code halted (%s)" % (", ".join(compute), json.dumps(error)))
    state = {}

    def no_compute_device_left():
        state.update(client.health()[1].get("guardian", {}))
        return "no compute device" in state.get("reason", "")

    check(wait_until(no_compute_device_left, args.critical_seconds + 20) and state.get("halted") is True,
          "when the critical signals end the devices are still excluded: the server stays halted (%s)" % state.get("reason"))
    status, answer, headers = client.call("POST", "/sdcpp/v1/img_gen", generation_body(2, 7, size))
    check(is_halted_refusal(status, answer, headers), "and refuses work")
    status, body, _ = client.call("POST", "/sdcpp/v1/guardian/resume")
    check(status == 200 and isinstance(body, dict) and body.get("guardian", {}).get("halted") is False,
          "an operator who has checked the machine resumes the server")
    check(all(device(body, name).get("level") == device(body, name).get("strain_levels") and device(body, name).get("excluded") is False
              for name in compute),
          "the excluded devices come back at the last level of strain")
    done = client.wait_for(client.submit(2, 8, size), ("completed",), args.warmup_timeout)
    check(done["status"] == "completed", "and the server works again, with the model where it was")
    for name in compute:
        client.pressure(name, {"level": 0, "source": "e2e"})

    # 7. The keeper is where it was.
    status, health, _ = client.health()
    now = health.get("counters", {})
    check(status == 200 and health.get("status") == start_status, "the halts did not move the model: the status is %s as at the start" % health.get("status"))
    check(now.get("step_downs") == before.get("step_downs") and now.get("failed") == before.get("failed") and
          now.get("device_switches") == before.get("device_switches") and now.get("retries") == before.get("retries"),
          "no step down, no device switch, no failure: the work the guardian stopped counts as stopped")
    check(now.get("cancelled", 0) - before.get("cancelled", 0) == 3, "the two jobs and the request that the guardian stopped count as cancelled")


def run_termico(client, args, temperature, mask, band, level):
    """termico.py osserva --segnala with the fake GPU of test/termico_test.py at the given temperature: its report
    reaches the server as that level of the CPU, named explicitly so that the scenario does not depend on the GPUs of
    the machine, and is withdrawn when termico.py ends; the CPU then comes back with the quiet periods of 5 s."""
    here = os.path.dirname(os.path.abspath(__file__))
    termico = os.path.join(here, "..", "scripts", "diagnostica_termica", "termico.py")
    spec = importlib.util.spec_from_file_location("termico_test", os.path.join(here, "termico_test.py"))
    fakes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fakes)
    folder = tempfile.mkdtemp(prefix="termico-e2e-")
    try:
        fake_gpu = os.path.join(folder, "nvidia-smi-finto.py")
        with open(fake_gpu, "w") as handle:
            handle.write(fakes.GPU_FINTA)
        for name, value in (("temp", str(temperature)), ("mask", mask.zfill(16))):
            with open(os.path.join(folder, name), "w") as handle:
                handle.write(value)
        environment = dict(os.environ, FAKE_DIR=folder)
        environment.pop("TERMICO_CHIAVE_API", None)
        command = [sys.executable, "-I", termico, "osserva", "--nvidia-smi", fake_gpu, "--motore", client.base_url,
                   "--chiave-api", args.api_key, "--segnala", "CPU", "--durata", "5", "--ogni", "0.5",
                   "--uscita", os.path.join(folder, "uscita")]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                   env=environment, stdin=subprocess.DEVNULL)
        cpu = {}

        def reported():
            cpu.update(device(client.health()[1], "CPU"))
            return cpu.get("level") == level and cpu.get("held_ms", 0) > 0

        seen = wait_until(reported, 10)
        output, _ = process.communicate(timeout=60)
        check(seen and "termico: GPU a %d C, fascia %s" % (temperature, band) in cpu.get("reason", "") and cpu.get("excluded") is False,
              "termico.py --segnala at %d C: the %s band reaches the server as level %d of 2, with the reason (%s)" %
              (temperature, band, level, cpu.get("reason")))
        after = device(client.health()[1], "CPU")
        check(process.returncode == 0 and after.get("held_ms") == 0 and 0 < after.get("level", 0) <= level,
              "termico.py --segnala: when it ends its signal is withdrawn, and the level does not drop at once")
        check(wait_until(lambda: device(client.health()[1], "CPU").get("level") == 0, 5 * level + 10),
              "termico.py --segnala: the CPU comes back one step every 5 s")
        check(args.api_key not in output, "termico.py --segnala: the API key does not appear in its output")
        return output
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def run_guardian_off(args, log_path):
    """The second server: --guardian off, and the values of device health changed with --device-health."""
    with open(log_path, "wb") as log:
        process, base_url = start_server(args, log, ["--guardian", "off", "--device-health", "levels=2,step=20,quiet=5"])
        client = Client(base_url, args.api_key)
        try:
            health = wait_ready(client, process, args.start_timeout)
            check(health.get("guardian", {}).get("enabled") is False, "--guardian off: the health says the guardian is off")
            check(health.get("devices") and all(entry.get("strain_levels") == 2 for entry in health["devices"]),
                  "--device-health levels=2: every component has two levels of strain")
            shares = []
            for level in (1, 2, 3):
                status, body, _ = client.pressure("disk", {"level": level, "source": "e2e", "duration_seconds": 60})
                disk = body.get("device", {}) if status == 200 else {}
                shares.append((disk.get("share_percent"), disk.get("excluded")))
            check(shares == [(80, False), (60, False), (0, True)],
                  "--device-health step=20: the levels offer 80%% and 60%%, and level 3 excludes (%s)" % shares)
            client.pressure("disk", {"level": 0, "source": "e2e"})
            check(wait_until(lambda: device(client.health()[1], "disk").get("level") == 0, 30),
                  "--device-health quiet=5: withdrawn, the disk comes back one step every 5 s")
            status, body, _ = client.pressure("RAM", {"level": "critical", "source": "e2e", "reason": "test overheat", "duration_seconds": 30})
            check(status == 200 and device({"devices": [body.get("device", {})]}, "RAM").get("critical") is True and
                  body.get("guardian", {}).get("halted") is False,
                  "--guardian off: a critical report reaches device health but does not halt the server")
            done = client.wait_for(client.submit(2, 8, args.size), ("completed",), args.warmup_timeout)
            check(done["status"] == "completed", "--guardian off: the server keeps working")
            run_termico(client, args, 82, "4", "attenzione", 1)
            output = run_termico(client, args, 88, "1", "critica", 2)
            check("2 livelli di sforzo" in output, "termico.py --segnala: the critical band stops at the last of the two levels, and it says so")
        finally:
            stop_server(process)


def run_cli_option(args):
    """--device-health in sd-cli, in the DGL mode that checks a script without loading a model: a value against the
    rules stops the program at start, and a valid one is applied and named in the log."""
    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, "..", "docs", "dgl", "sd", "sd1-5-txt2img.dgl")
    command = [args.cli, "-M", "dgl", "--script", script, "--check", "--device-health"]
    wrong = subprocess.run(command + ["levels=0"], capture_output=True, text=True, timeout=120)
    check(wrong.returncode == 1 and "--device-health: levels=0" in wrong.stdout + wrong.stderr,
          "sd-cli --device-health levels=0: refused at start, with the rules")
    right = subprocess.run(command + ["quiet=600"], capture_output=True, text=True, timeout=120)
    check(right.returncode == 0 and "device health: 3 levels of strain" in right.stdout + right.stderr and
          "a quiet period of 600 s" in right.stdout + right.stderr,
          "sd-cli --device-health quiet=600: applied in the DGL mode too, and named in the log")


def log_lines(path, offset=0):
    with open(path, "rb") as handle:
        handle.seek(offset)
        return handle.read().decode(errors="replace")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default="build/bin/sd-server", help="path to sd-server")
    parser.add_argument("--cli", help="path to sd-cli, for its --device-health (default: next to --server; skipped if it is not there)")
    parser.add_argument("--model", help="model for the servers to load")
    parser.add_argument("--url", help="use a server that is already running instead of starting one")
    parser.add_argument("--api-key", help="API key of the server (default: one of the test's own for the servers it starts)")
    parser.add_argument("--extra", action="append", default=[], help="an extra option for sd-server (repeatable)")
    parser.add_argument("--size", type=int, default=64, help="width and height of the images (default 64, the minimum)")
    parser.add_argument("--steps", type=int, default=100, help="steps of the generations that get stopped, at most 100")
    parser.add_argument("--critical-seconds", type=int, default=6, help="how long the critical reports last (default 6)")
    parser.add_argument("--start-timeout", type=int, default=300, help="seconds to wait for a server to load the model")
    parser.add_argument("--warmup-timeout", type=int, default=900, help="seconds for any job that must complete")
    args = parser.parse_args()
    if not args.url and not args.model:
        parser.error("give --model (and --server), or --url of a running server")
    if not args.url and not args.api_key:
        args.api_key = "e2e-" + secrets.token_hex(8)

    process = None
    log = None
    log_path = None
    if args.url:
        base_url = args.url
    else:
        handle, log_path = tempfile.mkstemp(prefix="sd-server-guardian-", suffix=".log")
        os.close(handle)
        log = open(log_path, "wb")
        process, base_url = start_server(args, log)

    client = Client(base_url, args.api_key)
    try:
        wait_ready(client, process, args.start_timeout)
        run_scenarios(client, args)
    except Abort as error:
        check(False, str(error))
    finally:
        stop_server(process)
        if log is not None:
            log.close()

    off_log_path = None
    if not args.url:
        handle, off_log_path = tempfile.mkstemp(prefix="sd-server-guardian-off-", suffix=".log")
        os.close(handle)
        try:
            run_guardian_off(args, off_log_path)
        except Abort as error:
            check(False, "--guardian off: " + str(error))

    if not args.cli:
        args.cli = os.path.join(os.path.dirname(args.server), "sd-cli" + (".exe" if os.name == "nt" else ""))
    if os.path.isfile(args.cli):
        run_cli_option(args)
    else:
        print("sd-cli not found at %s: its --device-health is not checked" % args.cli)

    if log_path is not None:
        text = log_lines(log_path)
        check("guardian: the server stops all work: RAM is critical: e2e: test overheat" in text,
              "the server log says when and why the server stopped all work")
        check(re.search(r"job \S+ stopped by the guardian", text) is not None, "it names the job that the guardian stopped")
        check("guardian: the cause has ended, the server resumes work" in text, "it says when the server resumed by itself")
        check("guardian: an operator says the machine was checked" in text, "and when an operator resumed it")
        # The engine may meet a device that has just been excluded before the stop reaches it, and says so.
        refused = "the engine does not compute on it"
        for path in (log_path, off_log_path):
            errors = [line for line in re.findall(r"\[E\][^\r\n]*", log_lines(path)) if refused not in line] if path else []
            check(not errors, "the server writes no other error line to its log" + (": " + errors[0] if errors else ""))
        if failures:
            print("server logs kept at " + log_path + (" and " + off_log_path if off_log_path else ""))
        else:
            os.remove(log_path)
            if off_log_path:
                os.remove(off_log_path)

    print("\n%s (%d failures)" % ("TEST FAILED" if failures else "ALL TESTS PASSED", failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
