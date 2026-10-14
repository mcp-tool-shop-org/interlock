"""interlock-relay, stage 1: the driver for the 5090 runs. Nothing here runs without the Publisher's grant.

    uv run --extra relay python scripts/relay_s1.py --grant <id> --prefix relay-s1-<date> [--steps ...]

Before every step the driver unloads the card, waits for it to settle at the idle baseline, and runs the full
preflight. Any refusal stops the run before a model call. Each step runs in its own process, with a VRAM watch
that writes the step's stop file over 30,500 MiB (interleaved runs have no VRAM ceiling of their own).

- ``decode``, ``long``, ``unload``, ``companion``: ``interlock run experiments/relay-s1/<step>.yaml --trace relay``,
  under a counter of the HTTP requests the Ollama adapter sends (below the hook: H1's ground truth). Supervisor
  health checks aren't counted, by the design's definition. Each step writes ``logs/<id>.http.json``.
- ``stop``: the decode spec, stopped through ``receipts/<id>.stop`` once its trace holds 10 model calls. It passes
  only on the operator-stop exit, 77.
- ``import``: the process-level import check. There are 15 triples of child processes, in the order no import,
  idle import, no import. Each child makes 1 warm-up and 3 measured calls with the decode spec's untraced arm, and
  the idle-import child imports ``nemo_relay`` first. The output is ``logs/<prefix>-import.json``.

At the end the card is unloaded, ``interlock check`` runs, and ``logs/<prefix>.driver.json`` records the preflight,
every step, the check, and the host, user and home strings the analysis must not find in a trace. The analysis is
``experiments/interlock-relay/stage1/analyze.py`` in mcp-tool-shop-org/rnd, frozen before the run.
"""

from __future__ import annotations

import argparse
import contextlib
import http.client
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

STEPS = ("decode", "long", "unload", "companion", "stop", "import")
SPEC_DIR = Path("experiments/relay-s1")
RELAY_VERSION = "0.10.1"
OLLAMA_VERSION = "0.35.1"
IDLE_VRAM_MIB = 3500  # the first preflight: the desktop's own use (2,0xx to 2,7xx MiB measured), not a model
SETTLE_MARGIN_MIB = 512
# H2 measures client-side time: a busy CPU adds noise where equivalence is tested.
CPU_QUIET_PCT = 25.0  # later steps: back within this of the first preflight's reading
# On Windows (WDDM) nvidia-smi lists every desktop app that draws on the GPU, with no memory. A compute workload is a
# process whose name says it runs models; any of these on the GPU refuses the run.
WORKLOAD_NAMES = ("python", "ollama", "llama", "comfy", "vllm", "lms", "lm studio", "koboldcpp", "torch", "onnx",
                  "whisper", "tensorrt", "triton")
IMPORT_TRIPLES = 15
STOP_AFTER_SPANS = 10
VRAM_CEILING_MIB = 30500
STEP_TIMEOUT_S = 3600
GRANT_CAP_MIN = 90
STOP_GRACE_S = 120
EXIT_OPERATOR_STOP = 77


class PreflightError(RuntimeError):
    """One preflight item failed. The run stops before any model call."""


# -- the counter below the hook ---------------------------------------------------------------------------------------


_ADAPTER_FILE = os.path.join("interlock", "adapters", "ollama.py")


@contextlib.contextmanager
def count_adapter_requests():
    """Count every ``http.client`` request sent from the Ollama adapter's transport, by path. Requests from anywhere
    else (the supervisor's health checks) aren't counted."""

    counts: dict[str, int] = {}
    lock = threading.Lock()
    original = http.client.HTTPConnection.request

    def counted(self, method, url, *args, **kwargs):
        if any(_ADAPTER_FILE in frame.filename for frame in traceback.extract_stack()):
            path = url.split("?", 1)[0]
            with lock:
                counts[path] = counts.get(path, 0) + 1
        return original(self, method, url, *args, **kwargs)

    http.client.HTTPConnection.request = counted
    try:
        yield counts
    finally:
        http.client.HTTPConnection.request = original


# -- the preflight ----------------------------------------------------------------------------------------------------


class Seams:
    """Every outside read the preflight makes. The defaults are the live ones; tests pass their own."""

    def environ(self) -> dict:
        return dict(os.environ)

    def version(self, package: str) -> str | None:
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return None

    def relay_config_paths(self) -> list:
        from nemo_relay import plugin

        report = plugin.validate({"version": 1, "components": []})
        return list(report.get("config_paths") or [])

    def git(self, root: Path, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)  # noqa: S603, S607

    def ollama(self, path: str) -> dict:
        conn = http.client.HTTPConnection("127.0.0.1", 11434, timeout=5)
        try:
            conn.request("GET", path)
            return json.loads(conn.getresponse().read().decode("utf-8"))
        finally:
            conn.close()

    def unload(self, model: str) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", 11434, timeout=60)
        try:
            body = json.dumps({"model": model, "prompt": "", "stream": False, "keep_alive": 0})
            conn.request("POST", "/api/generate", body=body, headers={"Content-Type": "application/json"})
            conn.getresponse().read()
        finally:
            conn.close()

    def gpu(self) -> dict:
        def query(args: list[str]) -> str:
            return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, check=True).stdout  # noqa: S603, S607

        used = query(["--query-gpu=memory.used", "--format=csv,noheader,nounits"]).strip().splitlines()
        names = [line.split(",", 1)[-1].strip().replace("\\", "/").rsplit("/", 1)[-1]
                 for line in query(["--query-compute-apps=pid,process_name", "--format=csv,noheader"]).splitlines()
                 if line.strip()]
        workloads = sorted({n for n in names if any(w in n.lower() for w in WORKLOAD_NAMES)})
        return {"used_mib": int(used[0]), "workloads": workloads}

    def cpu_percent(self, seconds: float = 5.0) -> float:
        """The system's CPU use, averaged over ``seconds``."""

        import psutil

        return float(psutil.cpu_percent(interval=seconds))

    def listeners(self, port: int) -> int:
        # interlock's own reader: on Windows build 26340 a listening socket reports state 0 to psutil.
        from interlock.ollama_supervisor import listening_pids

        return len(listening_pids(port))

    def interpreter(self) -> Path:
        """The Python every step runs on: ``_spawn`` starts each child with this one."""

        return Path(sys.executable)

    def heartbeat_age_s(self) -> float:
        from interlock.guards import heartbeat_age_s
        from interlock.harness.runner import heartbeat_path

        return heartbeat_age_s(heartbeat_path(None), time.time())


def preflight(root: Path, grant: str | None, seams: Seams, *, first: bool = True,
              baseline_mib: int | None = None) -> dict:
    """Every item, in order. The first failure raises before any model call. Returns what was read.

    Before the first step the whole tree must be clean. Before a later step, everything but ``receipts/`` and
    ``table/`` must be, since the steps before it wrote their receipts there."""

    from interlock.guards import HEARTBEAT_MAX_AGE_S

    facts: dict = {}

    def need(ok: bool, message: str) -> None:
        if not ok:
            raise PreflightError(f"REFUSED: {message}")

    need(bool(grant), "no grant: the Publisher grants the 5090 for this run")
    # interlock refuses a device run from an in-repo interpreter; refuse here, before any step, instead.
    python = seams.interpreter().resolve()
    need(not python.is_relative_to(root.resolve()),
         f"the interpreter {python} is inside the checkout; set UV_PROJECT_ENVIRONMENT to the pinned environment")
    facts["python"] = str(python)
    age = seams.heartbeat_age_s()
    need(age <= HEARTBEAT_MAX_AGE_S, f"the watchdog heartbeat is {age:.0f}s old")
    facts["heartbeat_age_s"] = round(age, 1)
    otel = sorted(k for k in seams.environ() if k.startswith("OTEL_"))
    need(not otel, f"OTEL variables are set: {', '.join(otel)}")
    relay = seams.version("nemo-relay")
    need(relay == RELAY_VERSION, f"nemo-relay is {relay}, not {RELAY_VERSION} (install interlock[relay])")
    facts["nemo_relay"] = relay
    layered = seams.relay_config_paths()
    need(not layered, f"Relay would layer in {layered}")
    scope = ["--", "."] if first else ["--", ".", ":!receipts", ":!table"]
    status = seams.git(root, "status", "--porcelain", *scope).stdout.strip()
    need(not status, "the interlock tree is not clean" + ("" if first else " outside receipts/ and table/"))
    ignored = seams.git(root, "check-ignore", "-q", "logs/relay-s1-probe").returncode == 0
    need(ignored, "logs/ is not git-ignored")
    facts["git_sha"] = seams.git(root, "rev-parse", "HEAD").stdout.strip()
    version = (seams.ollama("/api/version") or {}).get("version")
    need(version == OLLAMA_VERSION, f"Ollama on 11434 is {version}, not {OLLAMA_VERSION}")
    facts["ollama_version"] = version
    loaded = (seams.ollama("/api/ps") or {}).get("models") or []
    need(not loaded, f"{len(loaded)} model(s) already on the card")
    listeners = seams.listeners(11434)
    need(listeners == 1, f"{listeners} listeners on 11434; the supervisor needs exactly one")
    gpu = seams.gpu()
    need(not gpu["workloads"], f"another compute workload is on the GPU: {', '.join(gpu['workloads'])}")
    ceiling = IDLE_VRAM_MIB if baseline_mib is None else baseline_mib + SETTLE_MARGIN_MIB
    need(gpu["used_mib"] <= ceiling, f"{gpu['used_mib']} MiB in use; idle is at most {ceiling}")
    facts["idle_vram_mib"] = gpu["used_mib"]
    cpu = seams.cpu_percent()
    need(cpu <= CPU_QUIET_PCT, f"the CPU is at {cpu:.0f}% over 5 s; at most {CPU_QUIET_PCT:.0f}% before a step")
    facts["cpu_pct"] = round(cpu, 1)
    return facts


# -- the steps --------------------------------------------------------------------------------------------------------


def child_run(step: str, prefix: str, grant: str, root: Path, main=None) -> int:
    """In the step's own process: one traced interlock run under the request counter, then the count is written.

    Each step is its own process so the driver survives the run watcher's os._exit (the stop step, a VRAM breach)
    and can unload and check the card between steps (Argon stage 1 review, B2)."""

    if main is None:
        from interlock.cli import main
    spec = "decode" if step == "stop" else step
    run_id = f"{prefix}-{step}"
    argv = ["run", str(SPEC_DIR / f"{spec}.yaml"), "--root", str(root), "--grant", grant, "--trace", "relay",
            "--run-id", run_id]
    with count_adapter_requests() as counts:
        code = main(argv)
    (root / "logs").mkdir(exist_ok=True)
    record = {"run_id": run_id, "exit": code, "adapter_requests": sum(counts.values()), "by_path": counts}
    (root / "logs" / f"{run_id}.http.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n",
                                                       encoding="utf-8")
    return code


class VramWatch:
    """Reads the card once a second while a step runs. Over the ceiling, it writes the step's stop file once, so the
    run's own watcher ends it cleanly. Interleaved runs have no VRAM ceiling of their own."""

    def __init__(self, seams: Seams, stop_file: Path | None, *, ceiling_mib: int = VRAM_CEILING_MIB,
                 every_s: float = 1.0):
        self.seams, self.stop_file, self.ceiling, self.every = seams, stop_file, ceiling_mib, every_s
        self.peak = 0
        self.breached = False
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="relay-s1-vram", daemon=True)

    def _loop(self) -> None:
        while not self._done.is_set():
            self.read_once()
            self._done.wait(self.every)

    def read_once(self) -> None:
        try:
            used = int(self.seams.gpu()["used_mib"])
        except Exception:  # a failed read is skipped; the next one counts
            return
        self.peak = max(self.peak, used)
        if used > self.ceiling and not self.breached:
            self.breached = True
            if self.stop_file is not None:
                self.stop_file.parent.mkdir(exist_ok=True)
                self.stop_file.write_text(f"VRAM {used} MiB over {self.ceiling}\n", encoding="utf-8")

    def __enter__(self) -> VramWatch:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._done.set()
        self._thread.join(timeout=5)


def _spawn(step: str, prefix: str, grant: str, root: Path, popen):
    argv = [sys.executable, str(Path(__file__).resolve()), "--child-run", step, "--prefix", prefix, "--grant", grant,
            "--root", str(root)]
    return popen(argv, cwd=root)


def _wait_or_stop(proc, stop_file: Path, limit_s: float) -> tuple[int, bool]:
    """Wait for the step. At its time limit, write its stop file so the run's watcher ends it cleanly, then wait the
    grace period; a run that still hasn't ended is killed. Returns the exit code and whether the limit was hit."""

    try:
        return proc.wait(timeout=max(limit_s, 1.0)), False
    except subprocess.TimeoutExpired:
        stop_file.parent.mkdir(exist_ok=True)
        stop_file.write_text("interlock-relay stage 1: the grant's time limit\n", encoding="utf-8")
        try:
            return proc.wait(timeout=STOP_GRACE_S), True
        except subprocess.TimeoutExpired:
            proc.kill()
            return proc.wait(timeout=30), True


def run_step(step: str, prefix: str, grant: str, root: Path, *, seams: Seams, popen=subprocess.Popen,
             watch=VramWatch, limit_s: float = STEP_TIMEOUT_S) -> dict:
    """One traced run in its own process, watched for VRAM and stopped at the time limit."""

    run_id = f"{prefix}-{step}"
    stop_file = root / "receipts" / f"{run_id}.stop"
    proc = _spawn(step, prefix, grant, root, popen)
    with watch(seams, stop_file) as card:
        code, timed_out = _wait_or_stop(proc, stop_file, limit_s)
    return {"run_id": run_id, "exit": code, "vram_peak_mib": card.peak, "vram_breach": card.breached,
            "timed_out": timed_out}


def spans_started(trace: Path) -> int:
    """How many model-call spans the trace has opened so far."""

    try:
        lines = trace.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0
    count = 0
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line still being written
        count += event.get("category") == "llm" and event.get("scope_category") == "start"
    return count


def run_stop(prefix: str, grant: str, root: Path, *, seams: Seams, popen=subprocess.Popen, sleep=time.sleep,
             spans: int = STOP_AFTER_SPANS, timeout_s: float = 300.0, watch=VramWatch,
             limit_s: float = STEP_TIMEOUT_S) -> dict:
    """The decode spec in its own process, stopped through its stop file once the trace holds ``spans`` model calls.

    Waiting on the trace, not a clock, means the stop lands mid-run however fast the card is (Argon, B3). The step
    passes only on the operator-stop exit, 77."""

    run_id = f"{prefix}-stop"
    stop = root / "receipts" / f"{run_id}.stop"
    trace = root / "logs" / f"{run_id}.atof.jsonl"
    proc = _spawn("stop", prefix, grant, root, popen)
    with watch(seams, stop) as card:
        waited = 0.0
        while spans_started(trace) < spans and waited < timeout_s and proc.poll() is None:
            sleep(0.5)
            waited += 0.5
        written = proc.poll() is None
        seen = spans_started(trace)
        if written:
            stop.parent.mkdir(exist_ok=True)
            stop.write_text("interlock-relay stage 1: the stop-file run\n", encoding="utf-8")
        code, _timed_out = _wait_or_stop(proc, stop, limit_s)
    # The stop counts only if it landed mid-run on enough traced calls, and the watch didn't write it (Argon round 2).
    stopped = written and seen >= spans and not card.breached and code == EXIT_OPERATOR_STOP
    return {"run_id": run_id, "exit": code, "expected_exit": EXIT_OPERATOR_STOP, "stop_written": written,
            "spans_before_stop": seen, "stopped": stopped, "vram_peak_mib": card.peak, "vram_breach": card.breached}


def unload_all(seams: Seams) -> list[str]:
    """Unload every model on the card (keep_alive 0). The stop step's os._exit and the import children leave the
    model loaded."""

    names = [m.get("name") or m.get("model") for m in (seams.ollama("/api/ps") or {}).get("models") or []]
    for name in names:
        if name:
            seams.unload(name)
    return [n for n in names if n]


def wait_idle(seams: Seams, *, sleep=time.sleep, timeout_s: float = 90.0, ceiling_mib: int = IDLE_VRAM_MIB) -> float:
    """Wait for an empty card at the idle baseline, with no compute workload on it. A card that doesn't settle
    refuses the next step."""

    waited = 0.0
    while True:
        loaded = (seams.ollama("/api/ps") or {}).get("models") or []
        gpu = seams.gpu()
        used, workloads = gpu["used_mib"], gpu["workloads"]
        if not loaded and used <= ceiling_mib and not workloads:
            return waited
        if waited >= timeout_s:
            raise PreflightError(f"REFUSED: the card did not settle in {timeout_s:.0f}s ({len(loaded)} loaded, "
                                 f"{used} MiB, workloads: {', '.join(workloads) or 'none'})")
        sleep(1.0)
        waited += 1.0


CHILD = r'''
import json, statistics, sys, time
started = time.perf_counter()
args = json.loads(sys.argv[1])
import_s = None
if args["idle_import"]:
    t = time.perf_counter()
    import nemo_relay  # noqa: F401  # idle: imported, never initialised
    import_s = time.perf_counter() - t
import yaml
from interlock.adapters.base import StepRequest
from interlock.adapters.ollama import OllamaAdapter
spec = yaml.safe_load(open(args["spec"], encoding="utf-8"))
arm = next(a for a in spec["arms"] if a["id"] == args["arm"])
keys = ("thinking_level", "num_predict", "num_ctx", "prompt", "keep_alive", "temperature")
request = StepRequest(step_id=arm["id"], task=arm["task"], model=arm["model"], device=arm["device"],
                      engine=arm["engine"], settings={k: arm.get(k) for k in keys})
adapter = OllamaAdapter(device=arm["device"])
ready_s = time.perf_counter() - started
metrics = {"latency_ms": [], "decode_tok_s": [], "prompt_tok_s": []}
for i in range(1 + args["calls"]):
    result = adapter.run(request)
    if not result.ok:
        print(json.dumps({"error": result.error}))
        sys.exit(2)
    if i == 0:
        continue  # the warm-up
    metrics["latency_ms"].append(result.latency_ms)
    metrics["decode_tok_s"].append(result.throughput)
    metrics["prompt_tok_s"].append(result.prompt_tok_s)
print(json.dumps({"metrics": metrics, "import_s": import_s, "ready_s": ready_s}))
'''


def _child(root: Path, *, idle_import: bool, calls: int = 3) -> dict:
    args = {"idle_import": idle_import, "calls": calls, "spec": str(SPEC_DIR / "decode.yaml"), "arm": "decode-off"}
    done = subprocess.run([sys.executable, "-c", CHILD, json.dumps(args)], cwd=root, capture_output=True,  # noqa: S603
                          text=True, check=False, timeout=600)
    if done.returncode != 0:
        raise RuntimeError(f"import-check child failed ({done.returncode}): {done.stdout.strip()} {done.stderr[-400:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def run_import(prefix: str, root: Path, *, triples: int = IMPORT_TRIPLES, child=_child, between=None) -> dict:
    """15 triples of processes: no import, idle import, no import. ``between`` runs before each child (the
    heartbeat re-check in the live driver)."""

    record = {"triples": []}
    for _ in range(triples):
        triple = {}
        for side, idle in (("no_import_before", False), ("idle_import", True), ("no_import_after", False)):
            if between is not None:
                between()
            triple[side] = child(root, idle_import=idle)
        record["triples"].append(triple)
    (root / "logs").mkdir(exist_ok=True)
    (root / "logs" / f"{prefix}-import.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n",
                                                         encoding="utf-8")
    return {"triples": len(record["triples"])}


# -- the whole run ----------------------------------------------------------------------------------------------------


def drive(root: Path, grant: str, prefix: str, steps: tuple[str, ...], *, seams: Seams | None = None,
          step=run_step, stop=run_stop, imports=run_import, check=None, sleep=time.sleep, watch=None,
          cap_min: float = GRANT_CAP_MIN, clock=time.monotonic) -> dict:
    seams = seams or Seams()
    watch = watch or VramWatch
    if check is None:
        def check():
            from interlock.cli import main

            return main(["check", "--root", str(root)]) == 0
    out: dict = {"prefix": prefix, "steps": {}, "needles": {
        "host": socket.gethostname(), "user": os.environ.get("USERNAME") or os.environ.get("USER") or "",
        "home": str(Path.home())}}
    out["preflight"] = preflight(root, grant, seams)
    deadline = clock() + cap_min * 60
    out["cap_min"] = cap_min

    def remaining() -> float:
        left = deadline - clock()
        if left <= 0:
            raise PreflightError(f"REFUSED: the grant's {cap_min:g}-minute cap is reached; no further step starts")
        return left

    try:
        for index, name in enumerate(steps):
            if name not in STEPS:
                raise PreflightError(f"REFUSED: unknown step {name}")
            remaining()
            if index:
                unload_all(seams)
                baseline = out["preflight"]["idle_vram_mib"]
                wait_idle(seams, sleep=sleep, ceiling_mib=baseline + SETTLE_MARGIN_MIB)
                preflight(root, grant, seams, first=False, baseline_mib=baseline)  # idle again; nothing else changed
            if name == "stop":
                out["steps"][name] = result = stop(prefix, grant, root, seams=seams, limit_s=remaining())
                if result.get("vram_breach"):
                    raise PreflightError(f"REFUSED: VRAM passed {VRAM_CEILING_MIB} MiB in the stop run")
                if not result["stopped"]:
                    raise PreflightError(f"REFUSED: the stop run ended {result['exit']}, not by its stop file (77)")
            elif name == "import":
                card = watch(seams, None)

                def recheck(card=card):
                    """Before each child: the heartbeat, and the VRAM watch (a breach stops the check at once)."""

                    from interlock.guards import HEARTBEAT_MAX_AGE_S

                    if card.breached:
                        raise PreflightError(f"REFUSED: VRAM passed {VRAM_CEILING_MIB} MiB in the import check")
                    remaining()  # no child starts past the grant's cap
                    age = seams.heartbeat_age_s()
                    if age > HEARTBEAT_MAX_AGE_S:
                        raise PreflightError(f"REFUSED: the watchdog heartbeat is {age:.0f}s old")

                with card:
                    out["steps"][name] = imports(prefix, root, between=recheck)
                out["steps"][name]["vram_peak_mib"] = card.peak
                if card.breached:
                    raise PreflightError(f"REFUSED: VRAM passed {VRAM_CEILING_MIB} MiB in the import check")
            else:
                out["steps"][name] = result = step(name, prefix, grant, root, seams=seams, limit_s=remaining())
                if result.get("timed_out"):
                    raise PreflightError(f"REFUSED: {name} reached the grant's cap and was stopped")
                if result.get("vram_breach"):
                    raise PreflightError(f"REFUSED: VRAM passed {VRAM_CEILING_MIB} MiB in {name}; the run stopped")
                if result["exit"] != 0:
                    raise PreflightError(f"REFUSED: step {name} exited {result['exit']}; the run stops")
    finally:
        try:
            out["unloaded_at_end"] = unload_all(seams)
        except Exception as exc:  # the record is still written
            out["unloaded_at_end"] = f"failed: {exc}"
        out["check_passed"] = check()
        (root / "logs").mkdir(exist_ok=True)
        (root / "logs" / f"{prefix}.driver.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n",
                                                             encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grant", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--steps", default=",".join(STEPS))
    parser.add_argument("--max-minutes", type=float, default=GRANT_CAP_MIN,
                        help="the grant's cap: no step starts after it, and a running step is stopped at it")
    parser.add_argument("--child-run", choices=("decode", "long", "unload", "companion", "stop"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child_run:
        return child_run(args.child_run, args.prefix, args.grant, args.root.resolve())
    try:
        drive(args.root.resolve(), args.grant, args.prefix, tuple(s for s in args.steps.split(",") if s),
              cap_min=args.max_minutes)
    except PreflightError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
