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
            "gpu": {"used_mib": 2008, "compute_apps": 0}, "heartbeat": 5.0,
        }
        self.state.update(changes)
        self.status_args = []

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

    def listeners(self, port):
        return self.state["listeners"]

    def heartbeat_age_s(self):
        return self.state["heartbeat"]


def test_a_healthy_preflight_returns_what_it_read(tmp_path: Path):
    facts = relay_s1.preflight(tmp_path, "g-1", Seams())
    assert facts == {"heartbeat_age_s": 5.0, "nemo_relay": "0.10.1", "git_sha": "abc", "ollama_version": "0.35.1",
                     "idle_vram_mib": 2008}


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
        ("g", {"gpu": {"used_mib": 2008, "compute_apps": 1}}, "other process"),
        ("g", {"gpu": {"used_mib": 9000, "compute_apps": 0}}, "9000 MiB"),
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


def test_a_step_runs_the_spec_traced_and_records_the_count(tmp_path: Path):
    seen = []

    def main(argv):
        seen.append(argv)
        return 0

    record = relay_s1.run_step("decode", "p", "g-1", tmp_path, main=main)
    assert seen == [["run", str(Path("experiments/relay-s1/decode.yaml")), "--root", str(tmp_path), "--grant", "g-1",
                     "--trace", "relay", "--run-id", "p-decode"]]
    assert record["exit"] == 0
    assert json.loads((tmp_path / "logs" / "p-decode.http.json").read_text(encoding="utf-8"))["run_id"] == "p-decode"


def test_the_stop_step_writes_the_stop_file_and_expects_exit_77(tmp_path: Path):
    class Proc:
        def wait(self, timeout):
            assert (tmp_path / "receipts" / "p-stop.stop").is_file()
            return 77

    started = []
    record = relay_s1.run_stop("p", "g", tmp_path, after_s=5, popen=lambda argv, cwd: started.append(argv) or Proc(),
                               sleep=lambda s: None)
    assert record["stopped"] is True
    assert started[0][1:3] == ["-m", "interlock"] and "--trace" in started[0]


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


def test_a_failed_step_stops_the_run_and_the_record_is_still_written(tmp_path: Path):
    calls = []

    def main(argv):
        calls.append(argv[0])
        return 1 if argv[0] == "run" else 0

    with pytest.raises(relay_s1.PreflightError, match="decode exited 1"):
        relay_s1.drive(tmp_path, "g", "p", ("decode", "long"), seams=Seams(), main=main)
    assert calls == ["run", "check"]  # long never ran; the check ran in the finally
    driver = json.loads((tmp_path / "logs" / "p.driver.json").read_text(encoding="utf-8"))
    assert driver["check_passed"] is True and "decode" in driver["steps"] and "long" not in driver["steps"]
    assert set(driver["needles"]) == {"host", "user", "home"}


def test_a_refused_preflight_runs_nothing(tmp_path: Path):
    calls = []
    with pytest.raises(relay_s1.PreflightError):
        relay_s1.drive(tmp_path, "g", "p", ("decode",), seams=Seams(listeners=0), main=lambda a: calls.append(a) or 0)
    assert calls == []
