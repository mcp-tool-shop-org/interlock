"""The interlock-relay stage 1 driver (scripts/relay_s1.py), on fakes: the preflight refuses before any model call,
the request counter sees only the adapter's requests, and the steps run in order. No device, no server."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("relay_s1", ROOT / "scripts" / "relay_s1.py")
relay_s1 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(relay_s1)


class Seams(relay_s1.Seams):
    """Every preflight read, healthy unless a test changes one."""

    def __init__(self, **changes):
        self.state = {
            "environ": {}, "nemo-relay": "0.10.1", "config_paths": [], "status": "", "ignored": 0, "sha": "abc",
            "/api/version": {"version": "0.35.1"}, "/api/ps": {"models": []}, "listeners": 1,
            "gpu": {"used_mib": 2008, "workloads": []}, "heartbeat": 5.0, "cpu": 6.0,
        }
        self.state.update(changes)
        self.status_args = []
        self.unloaded = []

    def environ(self):
        return self.state["environ"]

    def version(self, package):
        return self.state[package]

    def relay_config_paths(self):
        return self.state["config_paths"]

    def git(self, root, *args):
        if args[0] == "status":
            self.status_args.append(args)
            return subprocess.CompletedProcess(args, 0, self.state["status"], "")
        if args[0] == "check-ignore":
            return subprocess.CompletedProcess(args, self.state["ignored"], "", "")
        return subprocess.CompletedProcess(args, 0, self.state["sha"] + "\n", "")

    def ollama(self, path):
        return self.state[path]

    def gpu(self):
        return self.state["gpu"]

    def unload(self, model):
        self.unloaded.append(model)
        self.state["/api/ps"] = {"models": []}
        self.state["gpu"] = {"used_mib": 2008, "workloads": []}

    def listeners(self, port):
        return self.state["listeners"]

    def heartbeat_age_s(self):
        return self.state["heartbeat"]

    def cpu_percent(self, seconds=5.0):
        return self.state["cpu"]


def test_a_healthy_preflight_returns_what_it_read(tmp_path: Path):
    facts = relay_s1.preflight(tmp_path, "g-1", Seams())
    assert facts == {"heartbeat_age_s": 5.0, "nemo_relay": "0.10.1", "git_sha": "abc", "ollama_version": "0.35.1",
                     "idle_vram_mib": 2008, "cpu_pct": 6.0}


@pytest.mark.parametrize(
    ("grant", "changes", "message"),
    [
        (None, {}, "no grant"),
        ("g", {"heartbeat": 10_000.0}, "heartbeat"),
        ("g", {"environ": {"OTEL_SERVICE_NAME": "x"}}, "OTEL"),
        ("g", {"nemo-relay": None}, "nemo-relay is None"),
        ("g", {"config_paths": ["user/plugins.toml"]}, "layer in"),
        ("g", {"status": " M src/x.py"}, "not clean"),
        ("g", {"ignored": 1}, "git-ignored"),
        ("g", {"/api/version": {"version": "0.36.0"}}, "0.36.0"),
        ("g", {"/api/ps": {"models": [{"name": "m"}]}}, "already on the card"),
        ("g", {"listeners": 2}, "2 listeners"),
        ("g", {"gpu": {"used_mib": 2008, "workloads": ["python.exe"]}}, "workload is on the GPU: python.exe"),
        ("g", {"gpu": {"used_mib": 9000, "workloads": []}}, "9000 MiB"),
        ("g", {"cpu": 36.0}, "CPU is at 36%"),
    ],
)
def test_each_preflight_item_refuses(tmp_path: Path, grant, changes, message):
    with pytest.raises(relay_s1.PreflightError, match=message):
        relay_s1.preflight(tmp_path, grant, Seams(**changes))


def test_after_the_first_step_receipts_and_table_may_change(tmp_path: Path):
    seams = Seams()
    relay_s1.preflight(tmp_path, "g", seams, first=False)
    assert seams.status_args[-1][-2:] == (":!receipts", ":!table")


def test_the_counter_sees_only_requests_sent_from_the_adapter(tmp_path: Path, monkeypatch):
    import http.client

    sent = []
    monkeypatch.setattr(http.client.HTTPConnection, "request", lambda self, method, url, *a, **k: sent.append(url))
    # A function compiled under the adapter's file name stands in for its transport.
    adapter_like = {}
    exec(compile("def send(conn, url):\n    conn.request('POST', url)\n",  # noqa: S102
                 str(ROOT / "src" / "interlock" / "adapters" / "ollama.py"), "exec"), adapter_like)
    with relay_s1.count_adapter_requests() as counts:
        conn = http.client.HTTPConnection("127.0.0.1", 11434)
        adapter_like["send"](conn, "/api/generate")
        adapter_like["send"](conn, "/api/generate")
        conn.request("GET", "/api/version")  # a health check from elsewhere
    assert counts == {"/api/generate": 2}
    assert sent == ["/api/generate", "/api/generate", "/api/version"]  # the real request still goes out
    assert http.client.HTTPConnection.request.__name__ == "<lambda>"  # restored after the block


def test_the_child_run_runs_the_spec_traced_and_records_the_count(tmp_path: Path):
    seen = []

    def main(argv):
        seen.append(argv)
        return 0

    assert relay_s1.child_run("decode", "p", "g-1", tmp_path, main=main) == 0
    assert seen == [["run", str(Path("experiments/relay-s1/decode.yaml")), "--root", str(tmp_path), "--grant", "g-1",
                     "--trace", "relay", "--run-id", "p-decode"]]
    record = json.loads((tmp_path / "logs" / "p-decode.http.json").read_text(encoding="utf-8"))
    assert record["run_id"] == "p-decode" and record["adapter_requests"] == 0
    relay_s1.child_run("stop", "p", "g", tmp_path, main=main)
    assert seen[-1][1] == str(Path("experiments/relay-s1/decode.yaml")) and seen[-1][-1] == "p-stop"


class Proc:
    def __init__(self, code=0, *, until_stop=None):
        self.code, self.until_stop = code, until_stop

    def poll(self):
        if self.until_stop is not None and not self.until_stop.is_file():
            return None
        return self.code

    def wait(self, timeout):
        return self.code


class NoWatch:
    def __init__(self, seams, stop_file):
        self.peak, self.breached = 0, False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


def test_a_step_runs_in_its_own_process_under_the_vram_watch(tmp_path: Path):
    started = []
    record = relay_s1.run_step("long", "p", "g", tmp_path, seams=Seams(),
                               popen=lambda argv, cwd: started.append(argv) or Proc(0), watch=NoWatch)
    assert record == {"run_id": "p-long", "exit": 0, "vram_peak_mib": 0, "vram_breach": False, "timed_out": False}
    assert started[0][2:4] == ["--child-run", "long"] and started[0][1].endswith("relay_s1.py")


def _spans(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"category": "llm", "scope_category": "start", "uuid": str(i)}) for i in range(n)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_the_stop_step_waits_for_spans_then_stops_and_expects_77(tmp_path: Path):
    stop = tmp_path / "receipts" / "p-stop.stop"
    trace = tmp_path / "logs" / "p-stop.atof.jsonl"
    ticks = []

    def sleep(seconds):
        ticks.append(seconds)
        _spans(trace, 4 * len(ticks))  # the run traces 4 calls per tick

    record = relay_s1.run_stop("p", "g", tmp_path, seams=Seams(), popen=lambda argv, cwd: Proc(77, until_stop=stop),
                               sleep=sleep, watch=NoWatch)
    assert record["stopped"] is True and record["stop_written"] is True
    assert record["spans_before_stop"] >= 10 and stop.is_file()


def test_a_stop_run_that_finishes_first_does_not_pass(tmp_path: Path):
    record = relay_s1.run_stop("p", "g", tmp_path, seams=Seams(), popen=lambda argv, cwd: Proc(0),
                               sleep=lambda s: None, watch=NoWatch)
    assert record["stopped"] is False and record["stop_written"] is False
    assert not (tmp_path / "receipts" / "p-stop.stop").exists()


def test_the_vram_watch_writes_the_stop_file_once_over_the_ceiling(tmp_path: Path):
    stop = tmp_path / "receipts" / "x.stop"
    seams = Seams(gpu={"used_mib": 31_000, "workloads": ["ollama.exe"]})
    watch = relay_s1.VramWatch(seams, stop)
    watch.read_once()
    first = stop.read_text(encoding="utf-8")
    watch.read_once()
    assert watch.breached and watch.peak == 31_000 and "31000" in first
    calm = relay_s1.VramWatch(Seams(), tmp_path / "receipts" / "y.stop")
    calm.read_once()
    assert not calm.breached and not (tmp_path / "receipts" / "y.stop").exists()


def test_unload_all_and_wait_idle(tmp_path: Path):
    seams = Seams(**{"/api/ps": {"models": [{"name": "qwen3:8b"}]},
                     "gpu": {"used_mib": 15000, "workloads": ["ollama.exe"]}})
    assert relay_s1.unload_all(seams) == ["qwen3:8b"] and seams.unloaded == ["qwen3:8b"]
    assert relay_s1.wait_idle(seams, sleep=lambda s: None) == 0.0
    stuck = Seams(**{"/api/ps": {"models": [{"name": "m"}]}})
    with pytest.raises(relay_s1.PreflightError, match="did not settle"):
        relay_s1.wait_idle(stuck, sleep=lambda s: None, timeout_s=3)


def test_the_import_check_runs_fifteen_triples_in_order(tmp_path: Path):
    order = []

    def child(root, *, idle_import):
        order.append(idle_import)
        return {"metrics": {"latency_ms": [1.0, 1.0, 1.0]}, "import_s": 0.1 if idle_import else None}

    checks = []
    relay_s1.run_import("p", tmp_path, child=child, between=lambda: checks.append(1))
    assert order == [False, True, False] * 15
    assert len(checks) == 45
    record = json.loads((tmp_path / "logs" / "p-import.json").read_text(encoding="utf-8"))
    assert set(record["triples"][0]) == {"no_import_before", "idle_import", "no_import_after"}


def _drive(tmp_path, steps, *, seams=None, step=None, stop=None):
    checks = []
    out = relay_s1.drive(
        tmp_path, "g", "p", steps, seams=seams or Seams(),
        step=step or (lambda name, prefix, grant, root, seams, limit_s: {"run_id": f"p-{name}", "exit": 0}),
        stop=stop or (lambda prefix, grant, root, seams, limit_s: {"stopped": True, "exit": 77}),
        imports=lambda prefix, root, between: {"triples": 15},
        check=lambda: checks.append(1) or True, sleep=lambda s: None)
    return out, checks


def test_between_steps_the_card_is_unloaded_settled_and_preflighted(tmp_path: Path):
    seams = Seams()
    out, checks = _drive(tmp_path, ("decode", "stop", "import"), seams=seams)
    assert list(out["steps"]) == ["decode", "stop", "import"]
    assert len(seams.status_args) == 3  # the first preflight, then one before each later step
    assert checks == [1]


def test_a_stop_step_that_did_not_stop_fails_the_run(tmp_path: Path):
    with pytest.raises(relay_s1.PreflightError, match="not by its stop file"):
        _drive(tmp_path, ("stop", "import"),
               stop=lambda prefix, grant, root, seams, limit_s: {"stopped": False, "exit": 0})
    driver = json.loads((tmp_path / "logs" / "p.driver.json").read_text(encoding="utf-8"))
    assert "import" not in driver["steps"]


def test_a_vram_breach_fails_the_run(tmp_path: Path):
    with pytest.raises(relay_s1.PreflightError, match="VRAM passed"):
        _drive(tmp_path, ("companion",),
               step=lambda name, prefix, grant, root, seams, limit_s: {"exit": 77, "vram_breach": True})


def test_a_failed_step_stops_the_run_and_the_record_is_still_written(tmp_path: Path):
    ran = []

    def step(name, prefix, grant, root, seams, limit_s):
        ran.append(name)
        return {"run_id": f"p-{name}", "exit": 1}

    with pytest.raises(relay_s1.PreflightError, match="decode exited 1"):
        _drive(tmp_path, ("decode", "long"), step=step)
    assert ran == ["decode"]
    driver = json.loads((tmp_path / "logs" / "p.driver.json").read_text(encoding="utf-8"))
    assert driver["check_passed"] is True and "decode" in driver["steps"] and "long" not in driver["steps"]
    assert set(driver["needles"]) == {"host", "user", "home"}


def test_a_refused_preflight_runs_nothing(tmp_path: Path):
    ran = []
    with pytest.raises(relay_s1.PreflightError):
        _drive(tmp_path, ("decode",), seams=Seams(listeners=0), step=lambda *a, **k: ran.append(1))
    assert ran == []


class BreachWatch(NoWatch):
    def __init__(self, seams, stop_file):
        super().__init__(seams, stop_file)
        self.breached = True


def test_a_stop_after_the_timeout_with_too_few_spans_does_not_pass(tmp_path: Path):
    stop = tmp_path / "receipts" / "p-stop.stop"
    _spans(tmp_path / "logs" / "p-stop.atof.jsonl", 3)  # the run traced only 3 calls before the timeout
    record = relay_s1.run_stop("p", "g", tmp_path, seams=Seams(), popen=lambda argv, cwd: Proc(77, until_stop=stop),
                               sleep=lambda s: None, timeout_s=2, watch=NoWatch)
    assert record["stop_written"] is True and record["spans_before_stop"] == 3
    assert record["stopped"] is False


def test_a_stop_written_by_a_vram_breach_does_not_pass_and_the_run_refuses(tmp_path: Path):
    stop = tmp_path / "receipts" / "p-stop.stop"
    _spans(tmp_path / "logs" / "p-stop.atof.jsonl", 12)
    record = relay_s1.run_stop("p", "g", tmp_path, seams=Seams(), popen=lambda argv, cwd: Proc(77, until_stop=stop),
                               sleep=lambda s: None, watch=BreachWatch)
    assert record["vram_breach"] is True and record["stopped"] is False
    with pytest.raises(relay_s1.PreflightError, match="VRAM passed .* in the stop run"):
        _drive(tmp_path, ("stop",), stop=lambda prefix, grant, root, seams, limit_s: {"stopped": True, "exit": 77,
                                                                                       "vram_breach": True})


def test_the_idle_wait_also_waits_for_no_process_on_the_gpu(tmp_path: Path):
    busy = Seams(gpu={"used_mib": 2008, "workloads": ["ComfyUI.exe"]})
    with pytest.raises(relay_s1.PreflightError, match="workloads: ComfyUI.exe"):
        relay_s1.wait_idle(busy, sleep=lambda s: None, timeout_s=3)

    class Draining(Seams):
        def __init__(self):
            super().__init__(gpu={"used_mib": 2008, "workloads": ["ollama.exe"]})
            self.reads = 0

        def gpu(self):
            self.reads += 1
            if self.reads > 2:
                self.state["gpu"] = {"used_mib": 2008, "workloads": []}
            return self.state["gpu"]

    assert relay_s1.wait_idle(Draining(), sleep=lambda s: None) == 2.0


def test_a_vram_breach_stops_the_import_check_before_the_next_child(tmp_path: Path):
    children = []

    def imports(prefix, root, between):
        for _ in range(45):
            between()
            children.append(1)
        return {"triples": 15}

    class LateBreach(NoWatch):
        def __init__(self, seams, stop_file):
            super().__init__(seams, stop_file)
            self.calls = 0

        @property
        def breached(self):
            return len(children) >= 4

        @breached.setter
        def breached(self, value):
            pass

    with pytest.raises(relay_s1.PreflightError, match="VRAM passed .* in the import check"):
        relay_s1.drive(tmp_path, "g", "p", ("import",), seams=Seams(), imports=imports, check=lambda: True,
                       sleep=lambda s: None, watch=LateBreach)
    assert len(children) == 4  # the fifth child never started


def test_the_grant_cap_starts_no_step_after_it(tmp_path: Path):
    now = [0.0]
    ran = []

    def step(name, prefix, grant, root, seams, limit_s):
        ran.append((name, limit_s))
        now[0] += 50 * 60  # each step takes 50 minutes
        return {"run_id": f"p-{name}", "exit": 0}

    with pytest.raises(relay_s1.PreflightError, match="90-minute cap"):
        relay_s1.drive(tmp_path, "g", "p", ("decode", "long", "unload"), seams=Seams(), step=step,
                       check=lambda: True, sleep=lambda s: None, clock=lambda: now[0])
    assert [name for name, _ in ran] == ["decode", "long"]
    assert ran[0][1] == 90 * 60 and ran[1][1] == 40 * 60  # the second step gets only what's left


def test_a_step_still_running_at_its_limit_is_stopped_through_its_stop_file(tmp_path: Path):
    stop = tmp_path / "receipts" / "p-decode.stop"

    class Slow:
        def __init__(self):
            self.waits = 0

        def wait(self, timeout):
            self.waits += 1
            if not stop.is_file():
                raise subprocess.TimeoutExpired("interlock", timeout)
            return 77

    record = relay_s1.run_step("decode", "p", "g", tmp_path, seams=Seams(), popen=lambda argv, cwd: Slow(),
                               watch=NoWatch, limit_s=5)
    assert record["timed_out"] is True and record["exit"] == 77 and stop.is_file()


def test_later_steps_hold_the_card_to_the_first_reading_plus_a_margin(tmp_path: Path):
    relay_s1.preflight(tmp_path, "g", Seams(gpu={"used_mib": 3000, "workloads": []}), first=False, baseline_mib=2700)
    with pytest.raises(relay_s1.PreflightError, match="idle is at most 3212"):
        relay_s1.preflight(tmp_path, "g", Seams(gpu={"used_mib": 3300, "workloads": []}), first=False,
                           baseline_mib=2700)


def test_the_live_gpu_reader_names_only_compute_workloads(monkeypatch):
    def run(argv, **kwargs):
        if "--query-gpu=memory.used" in argv:
            return subprocess.CompletedProcess(argv, 0, "2751\n", "")
        rows = ("1, C:\\Windows\\explorer.exe\n2, C:\\Program Files\\Google\\chrome.exe\n"
                "3, E:\\AI\\envs\\py\\python.exe\n")
        return subprocess.CompletedProcess(argv, 0, rows, "")

    monkeypatch.setattr(relay_s1.subprocess, "run", run)
    assert relay_s1.Seams().gpu() == {"used_mib": 2751, "workloads": ["python.exe"]}

