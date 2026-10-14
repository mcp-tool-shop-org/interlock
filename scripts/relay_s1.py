"""interlock-relay, stage 1: the driver for the 5090 runs. Nothing here runs without the Publisher's grant.

    uv run --extra relay python scripts/relay_s1.py --grant <id> --prefix relay-s1-<date> [--steps ...]

Steps, in order, each refusing before any model call when the preflight fails:

- ``decode``, ``long``, ``unload``, ``companion``: ``interlock run experiments/relay-s1/<step>.yaml --trace relay``,
  in this process, under a counter of the HTTP requests the Ollama adapter sends (below the hook: H1's ground truth).
  Supervisor health checks aren't counted, by the design's definition. Each step writes ``logs/<id>.http.json``.
- ``stop``: the decode spec in a child process, stopped through ``receipts/<id>.stop`` after a delay. The run's
  watcher ends that process with ``os._exit``, so it can't run in this one. Exit 77 is the expected end.
- ``import``: the process-level import check. There are 15 triples of child processes, in the order no import,
  idle import, no import. Each child makes 1 warm-up and 3 measured calls with the decode spec's untraced arm, and
  the idle-import child imports ``nemo_relay`` first. The output is ``logs/<prefix>-import.json``.

Afterwards ``interlock check`` runs, and ``logs/<prefix>.driver.json`` records the preflight, every step's exit,
the check, and the host, user and home strings that the analysis must not find in a trace. The analysis is
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
IDLE_VRAM_MIB = 2400
IMPORT_TRIPLES = 15
STOP_AFTER_S = 60.0
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

    def gpu(self) -> dict:
        def query(args: list[str]) -> str:
            return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, check=True).stdout  # noqa: S603, S607

        used = query(["--query-gpu=memory.used", "--format=csv,noheader,nounits"]).strip().splitlines()
        apps = [line for line in query(["--query-compute-apps=pid", "--format=csv,noheader"]).splitlines() if line]
        return {"used_mib": int(used[0]), "compute_apps": len(apps)}

    def listeners(self, port: int) -> int:
        import psutil

        return sum(1 for c in psutil.net_connections(kind="tcp") if c.status == "LISTEN" and c.laddr.port == port)

    def heartbeat_age_s(self) -> float:
        from interlock.guards import heartbeat_age_s
        from interlock.harness.runner import heartbeat_path

        return heartbeat_age_s(heartbeat_path(None), time.time())


def preflight(root: Path, grant: str | None, seams: Seams, *, first: bool = True) -> dict:
    """Every item, in order. The first failure raises before any model call. Returns what was read.

    Before the first step the whole tree must be clean. Before a later step, everything but ``receipts/`` and
    ``table/`` must be, since the steps before it wrote their receipts there."""

    from interlock.guards import HEARTBEAT_MAX_AGE_S

    facts: dict = {}

    def need(ok: bool, message: str) -> None:
        if not ok:
            raise PreflightError(f"REFUSED: {message}")

    need(bool(grant), "no grant: the Publisher grants the 5090 for this run")
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
    need(gpu["compute_apps"] == 0, f"{gpu['compute_apps']} other process(es) on the GPU")
    need(gpu["used_mib"] <= IDLE_VRAM_MIB,
         f"{gpu['used_mib']} MiB in use; the idle baseline is at most {IDLE_VRAM_MIB}")
    facts["idle_vram_mib"] = gpu["used_mib"]
    return facts


# -- the steps --------------------------------------------------------------------------------------------------------


def run_step(step: str, prefix: str, grant: str, root: Path, main=None) -> dict:
    """One traced interlock run in this process, under the request counter."""

    if main is None:
        from interlock.cli import main
    run_id = f"{prefix}-{step}"
    argv = ["run", str(SPEC_DIR / f"{step}.yaml"), "--root", str(root), "--grant", grant, "--trace", "relay",
            "--run-id", run_id]
    with count_adapter_requests() as counts:
        code = main(argv)
    (root / "logs").mkdir(exist_ok=True)
    record = {"run_id": run_id, "exit": code, "adapter_requests": sum(counts.values()), "by_path": counts}
    (root / "logs" / f"{run_id}.http.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n",
                                                       encoding="utf-8")
    return record


def run_stop(prefix: str, grant: str, root: Path, *, after_s: float = STOP_AFTER_S, popen=subprocess.Popen,
             sleep=time.sleep) -> dict:
    """The decode spec in a child process, stopped through its stop file."""

    run_id = f"{prefix}-stop"
    argv = [sys.executable, "-m", "interlock", "run", str(SPEC_DIR / "decode.yaml"), "--root", str(root), "--grant",
            grant, "--trace", "relay", "--run-id", run_id]
    proc = popen(argv, cwd=root)
    sleep(after_s)
    stop = root / "receipts" / f"{run_id}.stop"
    stop.parent.mkdir(exist_ok=True)
    stop.write_text("interlock-relay stage 1: the stop-file run\n", encoding="utf-8")
    code = proc.wait(timeout=600)
    return {"run_id": run_id, "exit": code, "expected_exit": EXIT_OPERATOR_STOP, "stopped": code == EXIT_OPERATOR_STOP}


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


def drive(root: Path, grant: str, prefix: str, steps: tuple[str, ...], *, seams: Seams | None = None, main=None,
          stop=run_stop, imports=run_import) -> dict:
    seams = seams or Seams()
    if main is None:
        from interlock.cli import main
    out: dict = {"prefix": prefix, "steps": {}, "needles": {
        "host": socket.gethostname(), "user": os.environ.get("USERNAME") or os.environ.get("USER") or "",
        "home": str(Path.home())}}
    out["preflight"] = preflight(root, grant, seams)
    try:
        for step in steps:
            if step not in STEPS:
                raise PreflightError(f"REFUSED: unknown step {step}")
            if step != STEPS[0]:
                preflight(root, grant, seams, first=False)  # the card is idle again; nothing else changed
            if step == "stop":
                out["steps"][step] = stop(prefix, grant, root)
            elif step == "import":
                def recheck():
                    age = seams.heartbeat_age_s()
                    from interlock.guards import HEARTBEAT_MAX_AGE_S

                    if age > HEARTBEAT_MAX_AGE_S:
                        raise PreflightError(f"REFUSED: the watchdog heartbeat is {age:.0f}s old")

                out["steps"][step] = imports(prefix, root, between=recheck)
            else:
                out["steps"][step] = run_step(step, prefix, grant, root, main=main)
                if out["steps"][step]["exit"] != 0:
                    raise PreflightError(f"REFUSED: step {step} exited {out['steps'][step]['exit']}; the run stops")
    finally:
        out["check_passed"] = main(["check", "--root", str(root)]) == 0
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
    args = parser.parse_args(argv)
    try:
        drive(args.root.resolve(), args.grant, args.prefix, tuple(s for s in args.steps.split(",") if s))
    except PreflightError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
