"""Ollama supervisor, against fakes.

Nothing here opens a socket, starts ollama, kills a process, or runs nvidia-smi.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest
import yaml

from switchyard.adapters.base import AdapterError, StepResult
from switchyard.adapters.ollama import OllamaAdapter
from switchyard.harness.fill import FillSeams, quiet_fill_seams
from switchyard.harness.runner import RunnerError, dry_run, run_spec
from switchyard.harness.watch import WatchHooks
from switchyard.ollama_supervisor import (
    URL_5090,
    URL_CPU,
    Health,
    OllamaSupervisor,
    Process,
    RestartRefused,
    ServerFingerprint,
    SupervisorConfig,
    SupervisorSeams,
    SupervisorStop,
    _load_psutil,
    card_mib_from_text,
    environment_mismatch,
    failure_kind,
    health,
    health_report,
    http_get,
    listening_pids,
    model_listed,
    pids_from_psutil,
    pids_from_ss,
    pids_from_windows,
    port_of,
    read_card_mib,
    read_cuda,
    real_seams,
    restart_policy_lines,
    retry_budget,
    serve_argv,
    spawn_detached,
    stop_pid,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = "alpha beta "


class World:
    """A process table, an HTTP health fake, and a card reading. No operating system."""

    def __init__(self):
        self.clock_s = 0.0
        self.up = False
        self.hold_up = False
        self.version = "0.35.1"
        self.cuda = "13.0"
        self.flash = "true"
        self.kv = "q8_0"
        self.baseline = 1500
        self.card = 1500
        self.ps_models = []
        self.ps_payload = None
        self.version_after_spawn = None
        self.cuda_after_spawn = None
        self.card_after_spawn = None
        self.keep_down_after_spawn = False
        self.spawn_pid = None
        self.listen_on_spawn = True
        self.argv_value = None
        self.config = SupervisorConfig()
        self.processes = [
            Process(pid=10, name="ollama", port=11434, listening=True),
            Process(pid=20, name="ollama", port=11490, listening=True),
            Process(pid=30, name="ollama app.exe", port=None, listening=False),
        ]
        self.stopped = []
        self.spawned = []
        self.next_pid = 100

    def fingerprint(self) -> ServerFingerprint:
        return ServerFingerprint(
            version="0.35.1",
            cuda_backend="13.0",
            flash_attention=self.flash,
            kv_cache_type=self.kv,
            baseline_mib=self.baseline,
            model="qwen3:8b",
        )

    def health_get(self, url, timeout):
        del url, timeout
        if self.up:
            if not self.hold_up:
                self.up = False
            return 200, {"version": self.version}, None
        return None, None, "connection refused"

    def process_rows(self):
        return list(self.processes)

    def stop(self, pid):
        self.stopped.append(pid)
        self.processes = [row for row in self.processes if row.pid != pid]

    def spawn(self, argv, env):
        self.spawned.append((list(argv), dict(env)))
        pid = self.next_pid if self.spawn_pid is None else self.spawn_pid
        if self.spawn_pid is None:
            self.next_pid += 1
        host = env.get("OLLAMA_HOST", "127.0.0.1:11434")
        port = int(str(host).rsplit(":", 1)[-1])
        if self.listen_on_spawn:
            self.processes.append(Process(pid=pid, name="ollama", port=port, listening=True))
        if self.version_after_spawn:
            self.version = self.version_after_spawn
        if self.cuda_after_spawn:
            self.cuda = self.cuda_after_spawn
        if self.card_after_spawn is not None:
            self.card = self.card_after_spawn
        if not self.keep_down_after_spawn:
            self.up = True
        return pid

    def card_mib(self):
        return self.card

    def ps(self, url):
        del url
        if self.ps_payload is not None:
            return self.ps_payload
        return {"models": [{"name": name} for name in self.ps_models]}

    def environment(self, url):
        del url
        return ServerFingerprint(
            version=self.version,
            cuda_backend=self.cuda,
            flash_attention=self.flash,
            kv_cache_type=self.kv,
            baseline_mib=self.baseline,
        )

    def argv(self):
        return list(self.argv_value or ["ollama", "serve"])

    def clock(self):
        return self.clock_s

    def sleep(self, seconds):
        self.clock_s += float(seconds)

    def now(self):
        return "2026-10-12T12:00:00Z"

    def seams(self) -> SupervisorSeams:
        return SupervisorSeams(
            health_get=self.health_get,
            processes=self.process_rows,
            stop=self.stop,
            spawn=self.spawn,
            card_mib=self.card_mib,
            ps=self.ps,
            environment=self.environment,
            argv=self.argv,
            clock=self.clock,
            sleep=self.sleep,
            now=self.now,
        )


def _supervisor(world: World, *, fingerprint=True, config=None) -> OllamaSupervisor:
    return OllamaSupervisor(
        world.seams(),
        fingerprint=world.fingerprint() if fingerprint else None,
        config=config or world.config,
    )


def _watch() -> WatchHooks:
    return WatchHooks(
        start_thread=False,
        ram_sampler=lambda: 1.0,
        rss_sampler=lambda: 0,
        used_sampler=lambda: 0,
        total_sampler=lambda: 1,
        state_sampler=lambda _uses: {},
        sleep=lambda _seconds: None,
    )


def _commit(repo: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "dev"], cwd=repo, check=True)
    subprocess.run(["git", "config", "core.autocrlf", "false"], cwd=repo, check=True)
    subprocess.run(["git", "config", "core.longpaths", "true"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=repo, check=True, capture_output=True)


def _bind(repo: Path, spec: dict) -> Path:
    path = repo / "experiments" / "t" / "spec.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8", newline="\n")
    _commit(repo)
    return path


def _arm(arm_id: str, device: str = "CPU") -> dict:
    return {
        "id": arm_id,
        "task": "generate",
        "prompt": "The switchyard counts the same cars on every pass.",
        "device": device,
        "engine": "ollama",
        "model": "qwen3:8b",
        "revision_or_digest": "abc",
        "quant": "unrecorded",
        "licence": "unrecorded",
        "engine_version": "0.35.1",
        "thinking_level": "off",
        "num_predict": 8,
        "num_ctx": None,
        "batch": None,
        "static_shape": None,
        "precision": None,
    }


def _interleaved(arms: list[dict], **extra) -> dict:
    spec = {
        "id": "T",
        "procedure": "interleaved",
        "question": "Does a hung Ollama come back between steps?",
        "decision": "The next step runs after a restart.",
        "needs": {"grant": False, "npu_session": False, "cpu_window": False},
        "stop_rules": ["a failed step stops the sequence"],
        "dataset": {"name": "fixture"},
        "metrics": ["latency_ms"],
        "warmups": 0,
        "repeats": 1,
        "band": 0.1,
        "compile_cache": "reuse",
        "arms": arms,
    }
    spec.update(extra)
    if any(arm["device"] == "5090" for arm in spec["arms"]):
        spec["needs"]["grant"] = True
    return spec


class Script:
    def __init__(self, failures):
        self.failures = list(failures)
        self.calls = []

    def run(self, request):
        self.calls.append(request.step_id)
        item = self.failures.pop(0) if self.failures else None
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, str):
            return StepResult(ok=False, error=item)
        latency = 20.0 if item is None else float(item)
        return StepResult(ok=True, latency_ms=latency, throughput=5.0, throughput_unit="tok/s")

    def unload(self, model, device=None):
        return None

    def loaded_models(self):
        return []


def _run(tmp_path: Path, script: Script, world: World, spec: dict, run_id: str = "run-1"):
    repo = tmp_path / run_id
    path = _bind(repo, spec)
    supervisor = _supervisor(world)
    rows = run_spec(
        path,
        root=repo,
        grant="grant-t" if spec["needs"]["grant"] else None,
        adapters={"ollama": script},
        date="2026-10-12",
        run_id=run_id,
        watch=_watch(),
        supervisor=supervisor,
        heartbeat=_fresh(tmp_path) if spec["needs"]["grant"] else None,
    )
    return repo, rows, supervisor


def _marker(repo: Path, run_id: str = "run-1") -> dict:
    return json.loads((repo / "receipts" / f"{run_id}.run.json").read_text(encoding="utf-8"))


def _fresh(tmp_path: Path) -> Path:
    path = tmp_path / "_watchdog_HEARTBEAT"
    path.write_text("pulse", encoding="utf-8")
    now = time.time()
    os.utime(path, (now, now))
    return path


def test_a_down_server_is_restarted_and_the_next_step_runs(tmp_path: Path):
    world = World()
    script = Script([AdapterError("connection refused")])
    spec = _interleaved([_arm("A"), _arm("B")])
    repo, rows, supervisor = _run(tmp_path, script, world, spec)
    assert script.calls[0] == "A"
    assert "B" in script.calls
    assert len(world.spawned) == 1
    assert world.stopped == [20]
    assert any(row.pid == 10 for row in world.processes)
    assert any(row.pid == 30 for row in world.processes)
    assert supervisor.restarts[0].why == "down"
    assert supervisor.restarts[0].port == 11490
    env = world.spawned[0][1]
    assert env["OLLAMA_HOST"] == "127.0.0.1:11490"
    assert env["OLLAMA_FLASH_ATTENTION"] == "true"
    assert env["OLLAMA_KV_CACHE_TYPE"] == "q8_0"
    assert env["OLLAMA_CUDA_BACKEND"] == "13.0"
    assert "signin" not in " ".join(world.spawned[0][0])
    assert rows
    assert all(row["settings"]["ollama_restarts"][0]["why"] == "down" for row in rows)
    failed = _marker(repo)["failed_steps"]
    assert failed[0]["step"] == "A cold"
    assert failed[0]["inconclusive"] is True
    _fill_down_then_next(tmp_path)


def test_a_slow_but_healthy_request_is_not_restarted(tmp_path: Path):
    world = World()
    world.up = True
    world.hold_up = True
    world.ps_models = ["qwen3:8b"]
    decision = _supervisor(world).after_failure(URL_5090, model="qwen3:8b", kind="timeout")
    assert decision.action == "leave"
    assert world.stopped == []
    assert world.spawned == []
    assert "healthy" in decision.note

    script = Script([AdapterError("the request timed out")])
    spec = _interleaved([_arm("A"), _arm("B")])
    with pytest.raises(RunnerError, match="not restarted"):
        _run(tmp_path, script, world, spec, run_id="slow-1")
    assert script.calls == ["A"]
    assert world.spawned == []


def test_a_hung_request_restarts_when_health_fails():
    world = World()
    decision = _supervisor(world).after_failure(URL_5090, model="qwen3:8b", kind="timeout")
    assert decision.action == "restarted"
    assert decision.record is not None
    assert decision.record.why == "hung"
    assert decision.record.port == 11434
    assert world.stopped == [10]


def test_a_hung_request_restarts_when_the_model_is_not_listed():
    world = World()
    world.up = True
    world.hold_up = True
    world.ps_models = ["other:7b"]
    decision = _supervisor(world).after_failure(URL_5090, model="qwen3:8b", kind="timeout")
    assert decision.action == "restarted"
    assert decision.record is not None
    assert decision.record.why == "hung"
    world.ps_payload = None
    assert world.stopped == [10]


def test_only_the_port_owner_is_stopped():
    world = World()
    decision = _supervisor(world).after_failure(URL_5090, model="qwen3:8b", kind="transport")
    assert decision.action == "restarted"
    assert world.stopped == [10]
    names = {row.pid: row.name for row in world.processes}
    assert names[20] == "ollama"
    assert names[30] == "ollama app.exe"
    assert 10 not in names


def test_a_version_or_backend_mismatch_after_restart_stops_the_run(tmp_path: Path):
    world = World()
    world.version_after_spawn = "9.9.9"
    supervisor = _supervisor(world)
    with pytest.raises(SupervisorStop, match="version changed"):
        supervisor.after_failure(URL_5090, model="qwen3:8b", kind="transport")
    assert len(world.spawned) == 1
    assert len(supervisor.restarts) == 1
    assert supervisor.restarts[0].version == "9.9.9"

    cuda = World()
    cuda.cuda_after_spawn = "12.0"
    with pytest.raises(SupervisorStop, match="cuda_backend changed"):
        _supervisor(cuda).after_failure(URL_CPU, model="qwen3:8b", kind="transport")

    script = Script([AdapterError("connection refused")])
    harness = World()
    harness.version_after_spawn = "9.9.9"
    with pytest.raises(RunnerError, match="version changed"):
        _run(tmp_path, script, harness, _interleaved([_arm("A"), _arm("B")]), run_id="mismatch-1")
    assert script.calls == ["A"]
    marker = _marker(tmp_path / "mismatch-1", "mismatch-1")
    assert marker["restarts"][0]["why"] == "down"
    assert marker["failed_steps"][0]["inconclusive"] is True


def test_the_restart_cap_stops_the_run(tmp_path: Path):
    world = World()
    supervisor = _supervisor(world)
    assert supervisor.after_failure(URL_5090, model="qwen3:8b", kind="transport").action == "restarted"
    assert supervisor.after_failure(URL_5090, model="qwen3:8b", kind="transport").action == "restarted"
    with pytest.raises(SupervisorStop, match="restart cap of 2"):
        supervisor.after_failure(URL_5090, model="qwen3:8b", kind="transport")
    assert len(world.spawned) == 2

    script = Script(
        [
            AdapterError("connection refused"),
            AdapterError("connection refused"),
            AdapterError("connection refused"),
        ]
    )
    spec = _interleaved([_arm("A"), _arm("B")])
    harness = World()
    with pytest.raises(RunnerError, match="restart cap of 2"):
        _run(tmp_path, script, harness, spec, run_id="cap-1")
    assert len(harness.spawned) == 2
    assert script.calls == ["A", "B", "A"]


def test_the_in_flight_step_is_marked_failed_and_no_measurement_crosses_a_restart(tmp_path: Path):
    world = World()
    script = Script([AdapterError("connection refused")])
    repo, rows, _supervisor_ran = _run(tmp_path, script, world, _interleaved([_arm("A"), _arm("B")]))
    marker = _marker(repo)
    assert marker["failed_steps"][0]["inconclusive"] is True
    assert rows
    assert all("connection refused" not in (row.get("notes") or "") for row in rows)
    assert all(row.get("aborted") is not True for row in rows)

    retry_world = World()
    retry_script = Script([AdapterError("connection refused")])
    spec = _interleaved([_arm("A")], retry=1)
    repo, rows, _again = _run(tmp_path, retry_script, retry_world, spec, run_id="retry-1")
    assert retry_script.calls == ["A", "A", "A"]
    assert rows[0]["latency"]["throughput"] == 5.0
    assert _marker(repo, "retry-1")["failed_steps"][0]["inconclusive"] is True
    assert len(retry_world.spawned) == 1

    supervisor = _supervisor(World())
    epoch = supervisor.begin_measurement()
    assert supervisor.keep_sample(epoch, {"latency_ms": 1}) == {"latency_ms": 1}
    supervisor.end_measurement()
    supervisor._epoch += 1
    assert supervisor.keep_sample(epoch, {"latency_ms": 1}) is None


def test_no_restart_happens_mid_measurement():
    world = World()
    supervisor = _supervisor(world)
    supervisor.begin_measurement()
    with pytest.raises(RestartRefused, match="measurement is open"):
        supervisor.after_failure(URL_5090, model="qwen3:8b", kind="transport")
    assert world.stopped == []
    assert world.spawned == []
    assert supervisor.restarts == []
    supervisor.end_measurement()
    decision = supervisor.after_failure(URL_5090, model="qwen3:8b", kind="transport")
    assert decision.action == "restarted"


def test_the_receipt_carries_the_restart_record(tmp_path: Path):
    world = World()
    script = Script([AdapterError("connection refused")])
    repo, rows, supervisor = _run(tmp_path, script, world, _interleaved([_arm("A"), _arm("B")]), run_id="receipt-1")
    record = supervisor.restarts[0].as_dict()
    assert record["when"] == "2026-10-12T12:00:00Z"
    assert record["port"] == 11490
    assert record["why"] == "down"
    assert record["old_pid"] == 20
    assert isinstance(record["new_pid"], int) and record["new_pid"] != 20
    assert record["seconds_to_healthy"] == 0
    assert record["version"] == "0.35.1"
    marker = _marker(repo, "receipt-1")
    assert marker["restarts"] == [record]
    written = [
        json.loads(line)
        for line in (repo / "receipts" / "receipt-1.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert written
    assert all(row["settings"]["ollama_restarts"] == [record] for row in written)
    assert rows[0]["settings"]["ollama_restarts"] == [record]


def test_cli_health_output(monkeypatch, capsys):
    def fake(url, timeout, connection=None):
        del timeout, connection
        if url.endswith("11434"):
            return 200, {"version": "0.35.1"}, None
        return None, None, "connection refused"

    monkeypatch.setattr("switchyard.ollama_supervisor.http_get_version", fake)
    from switchyard.cli import main

    code = main(["ollama-health"])
    out = capsys.readouterr().out
    assert out == (
        "http://127.0.0.1:11434 healthy 0.35.1\n"
        "http://127.0.0.1:11490 down connection refused\n"
    )
    assert code == 1

    def both(url, timeout, connection=None):
        del url, timeout, connection
        return 200, {"version": "0.35.1"}, None

    monkeypatch.setattr("switchyard.ollama_supervisor.http_get_version", both)
    assert main(["ollama-health"]) == 0
    assert "11490 healthy 0.35.1" in capsys.readouterr().out


def test_dry_run_states_the_restart_policy():
    text = dry_run(ROOT / "experiments" / "e5" / "spec.yaml", ROOT)
    assert "ollama restart: at most 2 per run" in text
    assert "between steps only, never during a measurement" in text
    assert "switchyard does not sign in" in text
    assert "heartbeat not read; adapters not called" in text
    interleaved = dry_run(ROOT / "experiments" / "e4" / "spec.yaml", ROOT)
    assert "at most 2 per run" in interleaved
    assert any("at most 2" in line for line in restart_policy_lines())


def test_a_healthy_5xx_is_left_alone_and_a_missing_fingerprint_stops():
    world = World()
    world.up = True
    world.hold_up = True
    decision = _supervisor(world).after_failure(URL_5090, model="qwen3:8b", kind="server")
    assert decision.action == "leave"
    assert world.stopped == []
    bare = World()
    with pytest.raises(SupervisorStop, match="no recorded Ollama environment"):
        _supervisor(bare, fingerprint=False).after_failure(URL_5090, model="qwen3:8b", kind="transport")
    assert bare.stopped == []


def test_a_signin_argv_stops_before_any_process_is_killed():
    world = World()
    world.argv_value = ["ollama", "signin"]
    with pytest.raises(SupervisorStop, match="does not sign in"):
        _supervisor(world).after_failure(URL_5090, model="qwen3:8b", kind="transport")
    assert world.stopped == []
    assert world.spawned == []


def test_restart_records_a_deadline_a_bad_pid_and_a_card_that_stays_high():
    deadline = World()
    deadline.keep_down_after_spawn = True
    deadline.config = SupervisorConfig(down_gap_s=0, restart_deadline_s=0)
    missed = _supervisor(deadline)
    with pytest.raises(SupervisorStop, match="did not come back"):
        missed.after_failure(URL_5090, model="qwen3:8b", kind="transport")
    assert missed.restarts[-1].seconds_to_healthy is None
    assert missed.restarts[-1].version is None
    assert "does not sign in" in (missed.restarts[-1].note or "")

    recorded = _supervisor(World())
    recorded.seams.spawn = lambda argv, env: None
    with pytest.raises(SupervisorStop, match="did not return a pid"):
        recorded.after_failure(URL_5090, model="qwen3:8b", kind="transport")
    assert recorded.restarts[-1].new_pid is None
    assert recorded.restarts[-1].seconds_to_healthy is None

    same = World()
    same.spawn_pid = 10
    with pytest.raises(SupervisorStop, match="still the one that came back"):
        _supervisor(same).after_failure(URL_5090, model="qwen3:8b", kind="transport")

    quiet = World()
    quiet.listen_on_spawn = False
    with pytest.raises(SupervisorStop, match="not the process listening"):
        _supervisor(quiet).after_failure(URL_5090, model="qwen3:8b", kind="transport")

    high = World()
    high.card_after_spawn = 9000
    with pytest.raises(SupervisorStop, match="did not return to its baseline"):
        stopped = _supervisor(high)
        stopped.after_failure(URL_5090, model="qwen3:8b", kind="transport")
    assert stopped.restarts[-1].note.startswith("the card did not return")


def test_health_parsers_and_process_lookup_use_fakes():
    assert health(URL_5090, get=lambda url, timeout: (200, {"version": "0.35.1"}, None)) == Health(
        True, "0.35.1", None
    )
    assert health(URL_5090, get=lambda url, timeout: (_ for _ in ()).throw(RuntimeError("closed"))).ok is False
    assert health(URL_5090, get=lambda url, timeout: (500, {"version": "0.35.1"}, None)).ok is False
    assert health(URL_5090, get=lambda url, timeout: (200, {}, None)).ok is False
    text, ok = health_report(get=lambda url, timeout: (200, {"version": "0.35.1"}, None))
    assert ok is True
    assert "11434 healthy 0.35.1" in text
    assert "11490 healthy 0.35.1" in text

    assert failure_kind("the request timed out") == "timeout"
    assert failure_kind("connection refused") == "transport"
    assert failure_kind("status 503") == "server"
    assert failure_kind("show failed") is None
    assert failure_kind("hang: still running") is None
    assert failure_kind(None, status=502) == "server"
    assert retry_budget({}) == 0
    assert retry_budget({"retry": True}) == 0
    assert retry_budget({"retry": -1}) == 0
    assert retry_budget({"retry": 2}) == 2
    assert model_listed({"models": [{"name": "qwen3:8b"}]}, "qwen3:8b") is True
    assert model_listed({"models": [{"model": "qwen3:8b:latest"}]}, "qwen3:8b") is True
    assert model_listed({"models": []}, "qwen3:8b") is False
    assert model_listed(None, "qwen3:8b") is False

    recorded = ServerFingerprint("0.35.1", "13.0", "true", "q8_0", 1500)
    assert environment_mismatch(recorded, recorded) is None
    assert "flash_attention" in environment_mismatch(
        recorded, ServerFingerprint("0.35.1", "13.0", "false", "q8_0", 1500)
    )
    assert recorded.serve_env(11434)["OLLAMA_HOST"] == "127.0.0.1:11434"
    blank = ServerFingerprint("0.35.1", "unrecorded", "unrecorded", "unrecorded")
    assert "OLLAMA_CUDA_BACKEND" not in blank.serve_env(11490)
    with pytest.raises(SupervisorStop):
        port_of("http://127.0.0.1:1")

    assert pids_from_windows("0\n\n4242\n4242\n") == [4242]
    assert pids_from_ss('users:(("ollama",pid=77,fd=3))') == [77]

    class _Conn:
        status = "LISTEN"
        pid = 88

        class laddr:
            port = 11434

    class _Module:
        @staticmethod
        def net_connections(kind="tcp"):
            assert kind == "tcp"
            return [_Conn(), _Conn()]

    assert pids_from_psutil(11434, _Module) == [88]

    def windows(argv, capture_output, text, timeout, check):
        assert "Get-NetTCPConnection" in argv[-1]
        assert "-LocalPort 11434" in argv[-1]
        assert "-State Listen" in argv[-1]
        return subprocess.CompletedProcess(argv, 0, stdout="4242\n", stderr="")

    assert listening_pids(11434, run=windows, platform="win32") == [4242]

    def linux(argv, capture_output, text, timeout, check):
        assert argv[0] == "ss"
        return subprocess.CompletedProcess(argv, 0, stdout='users:(("ollama",pid=77,fd=3))\n', stderr="")

    assert listening_pids(11490, run=linux, platform="linux") == [77]

    def empty_ss(argv, capture_output, text, timeout, check):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="")

    assert listening_pids(11434, run=empty_ss, platform="linux", psutil_module=_Module) == [88]

    killed = {}

    def killer(argv, capture_output, text, timeout, check):
        killed["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    stop_pid(15, run=killer, platform="win32")
    assert killed["argv"][:3] == ["taskkill", "/PID", "15"]
    stop_pid(15, run=killer, platform="linux")
    assert killed["argv"] == ["kill", "15"]
    with pytest.raises(SupervisorStop):
        stop_pid(True, run=killer, platform="linux")

    seen = {}

    class _Proc:
        pid = 55

    def popen(argv, **kwargs):
        seen["argv"] = argv
        seen["flags"] = kwargs
        return _Proc()

    assert spawn_detached(["ollama", "serve"], {"OLLAMA_HOST": "127.0.0.1:11434"}, popen=popen, platform="linux") == 55
    assert seen["flags"]["start_new_session"] is True
    assert spawn_detached(["ollama", "serve"], {"A": "b"}, popen=popen, platform="win32") == 55
    assert "creationflags" in seen["flags"]
    with pytest.raises(SupervisorStop, match="does not sign in"):
        spawn_detached(["ollama", "signin"], {}, popen=popen, platform="linux")

    windows_argv = " ".join(serve_argv("win32", localappdata="/opt/ollama-root"))
    assert "ollama-root" in windows_argv
    assert windows_argv.endswith("ollama.exe serve") or windows_argv.endswith("ollama.exe serve".replace("/", "\\"))
    assert "signin" not in windows_argv.lower()
    assert "Users" not in windows_argv
    assert serve_argv("linux") == ["ollama", "serve"]

    assert card_mib_from_text("1500\n") == 1500
    assert card_mib_from_text("1500\n1600\n") is None

    def nvidia(argv, capture_output, text, timeout, check):
        if any("memory.used" in str(part) for part in argv):
            return subprocess.CompletedProcess(argv, 0, stdout="1500\n", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="13.0\n", stderr="")

    assert read_card_mib(run=nvidia) == 1500
    assert read_cuda(run=nvidia) == "13.0"

    class _Response:
        status = 200

        def read(self):
            return b'{"version":"0.35.1"}'

    class _Connection:
        def __init__(self, host, port, timeout=None):
            assert host == "127.0.0.1"
            assert port == 11434

        def request(self, method, path):
            assert method == "GET"
            assert path == "/api/version"

        def getresponse(self):
            return _Response()

        def close(self):
            return None

    status, payload, error = http_get(URL_5090, "/api/version", 1, connection=_Connection)
    assert (status, payload, error) == (200, {"version": "0.35.1"}, None)
    refused, _payload, why = http_get("https://example.com", "/api/version", 1, connection=_Connection)
    assert refused is None
    assert why

    built = real_seams(
        platform="linux",
        popen=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("spawned")),
        connection=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("connected")),
    )
    assert built.argv() == ["ollama", "serve"]
    assert _load_psutil().net_connections


def test_a_wait_sleeps_until_the_server_answers():
    world = World()
    world.hold_up = True
    answers = {"n": 0}

    def health_get(url, timeout):
        del url, timeout
        answers["n"] += 1
        if answers["n"] >= 5:
            return 200, {"version": "0.35.1"}, None
        return None, None, "connection refused"

    world.health_get = health_get
    world.config = SupervisorConfig(down_failures=3, down_gap_s=2, restart_deadline_s=60)
    decision = _supervisor(world).after_failure(URL_5090, model="qwen3:8b", kind="transport")
    assert decision.record is not None
    assert decision.record.seconds_to_healthy == 2


# --- fill-card -----------------------------------------------------------


def _model(tag="qwen3:8b") -> dict:
    return {
        "model": tag,
        "revision_or_digest": "abc123",
        "quant": "unrecorded",
        "licence": "unrecorded",
        "engine": "ollama",
        "engine_version": "0.35.1",
        "device": "5090",
    }


def _fill_spec(models: list[dict], **extra) -> dict:
    spec = {
        "id": "E5",
        "procedure": "fill-card",
        "question": "What context stays on the card?",
        "decision": "The confirmed window.",
        "abort_mib": 30500,
        "ceiling_mib": 31200,
        "grain": 2048,
        "confirm_prompt_fraction": 0.9,
        "confirm_num_predict": 1,
        "sample_ms": 200,
        "spill_shared_mib": 256,
        "spill_prompt_ratio": 0.5,
        "max_ctx": 8192,
        "shift": False,
        "truncate": False,
        "needs": {"grant": True, "npu_session": False, "cpu_window": False},
        "dataset": {"name": "e5-confirm-prompt", "seed": SEED},
        "metrics": ["max_resident_ctx", "peak_mib"],
        "stop_rules": ["Abort the moment card VRAM reaches 30500 MiB."],
        "models": models,
    }
    spec.update(extra)
    return spec


class _Transport:
    def __init__(self, seams, *, show_errors=0):
        self.seams = seams
        self.show_errors = show_errors
        self.loaded = None
        self.unloads = []
        self.shown = None
        self.state = {"ctx": None, "confirm": False, "busy": False}

    def cancel(self):
        return None

    def card_mib(self):
        if not self.state["busy"]:
            return 1500
        return 20000

    def shared_mib(self):
        return 1000

    def dedicated_mib(self):
        return 12

    def __call__(self, method, url, body):
        del method
        if url.endswith("/api/ps"):
            if self.loaded is None:
                return {"models": []}
            return {"models": [self.loaded]}
        if url.endswith("/api/tags"):
            name = self.shown or "qwen3:8b"
            return {"models": [{"name": name, "size": 16_000 * 1024 * 1024}]}
        if url.endswith("/api/show"):
            if isinstance(body, dict) and isinstance(body.get("model"), str):
                self.shown = body["model"]
            if self.show_errors:
                self.show_errors -= 1
                return {"error": "connection refused"}
            return {
                "model_info": {"qwen3.context_length": 4096},
                "details": {"quantization_level": "Q4_K_M"},
                "parameters": "flash_attn true\nkv_cache_type q8_0\n",
            }
        if not isinstance(body, dict):
            return {}
        if body.get("keep_alive") == 0 and body.get("prompt") == "":
            self.unloads.append(body.get("model"))
            self.loaded = None
            self.state["busy"] = False
            return {}
        options = body.get("options") or {}
        self.state["ctx"] = options.get("num_ctx")
        self.state["confirm"] = (not body.get("raw")) and body.get("prompt") != SEED
        self.state["busy"] = True
        if self.seams is not None:
            self.seams.sample()
        if body.get("raw"):
            return {"prompt_eval_count": 100000}
        self.loaded = {"name": body["model"], "size": 1000, "size_vram": 1000}
        return {
            "eval_count": 1,
            "eval_duration": 1_000_000_000,
            "total_duration": 5_000_000,
            "prompt_eval_count": 10,
            "prompt_eval_duration": int(10 / 100 * 1_000_000_000),
        }


def _fill_seams(transport: _Transport) -> FillSeams:
    seams = quiet_fill_seams()
    seams.card_mib = transport.card_mib
    seams.shared_mib = transport.shared_mib
    seams.dedicated_mib = transport.dedicated_mib
    seams.ollama_version = lambda: "0.35.1"
    seams.driver = lambda: "581.29"
    seams.cuda_backend = lambda: "13.0"
    seams.sleep = lambda _seconds: None
    transport.seams = seams
    return seams


def _fill_run(tmp_path: Path, world: World, spec: dict, transport: _Transport, run_id: str):
    repo = tmp_path / run_id
    path = _bind(repo, spec)
    rows = run_spec(
        path,
        root=repo,
        grant="g",
        adapters={"ollama": OllamaAdapter(transport=transport)},
        heartbeat=_fresh(tmp_path),
        date="2026-10-12",
        run_id=run_id,
        seams=_fill_seams(transport),
        supervisor=_supervisor(world),
    )
    return repo, rows


def _fill_down_then_next(tmp_path: Path):
    world = World()
    transport = _Transport(None, show_errors=1)
    spec = _fill_spec([_model("qwen3:8b"), _model("qwen3:4b")])
    repo, rows = _fill_run(tmp_path, world, spec, transport, "fill-next")
    assert len(world.spawned) == 1
    assert world.stopped == [10]
    assert any(row.pid == 20 for row in world.processes)
    assert any(row.pid == 30 for row in world.processes)
    assert rows[0]["aborted"] is True
    assert rows[0]["settings"]["max_resident_ctx"] is None
    assert "inconclusive" in rows[0]["notes"]
    assert rows[1]["settings"]["max_resident_ctx"] == 8192
    assert rows[0]["settings"]["ollama_restarts"][0]["why"] == "down"
    assert _marker(repo, "fill-next")["restarts"][0]["port"] == 11434


def test_fill_card_retry_cap_and_mismatch_stop_between_models(tmp_path: Path):
    retry_world = World()
    transport = _Transport(None, show_errors=1)
    spec = _fill_spec([_model()], retry=1)
    _repo, rows = _fill_run(tmp_path, retry_world, spec, transport, "fill-retry")
    assert len(retry_world.spawned) == 1
    assert len(rows) == 1
    assert rows[0].get("aborted") is not True
    assert rows[0]["settings"]["max_resident_ctx"] == 8192

    cap_world = World()
    cap_transport = _Transport(None, show_errors=3)
    cap_spec = _fill_spec([_model()], retry=2)
    with pytest.raises(RunnerError, match="restart cap of 2"):
        _fill_run(tmp_path, cap_world, cap_spec, cap_transport, "fill-cap")
    assert len(cap_world.spawned) == 2

    mismatch = World()
    mismatch.version_after_spawn = "9.9.9"
    bad = _Transport(None, show_errors=1)
    with pytest.raises(RunnerError, match="version changed"):
        _fill_run(tmp_path, mismatch, _fill_spec([_model()]), bad, "fill-mismatch")
    marker = _marker(tmp_path / "fill-mismatch", "fill-mismatch")
    assert marker["restarts"]
    assert marker["failed_steps"][0]["inconclusive"] is True


def test_real_seams_and_the_remaining_fakes_do_not_touch_a_server():
    def run(argv, capture_output, text, timeout, check):
        del capture_output, text, timeout, check
        joined = " ".join(str(part) for part in argv)
        if argv[0] == "ss":
            return subprocess.CompletedProcess(argv, 0, stdout='users:(("ollama",pid=9,fd=3))\n', stderr="")
        if "memory.used" in joined:
            return subprocess.CompletedProcess(argv, 0, stdout="100\n", stderr="")
        if "cuda_version" in joined:
            return subprocess.CompletedProcess(argv, 0, stdout="13.0\n", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    class _Response:
        status = 200

        def read(self):
            return b'{"version":"0.35.1","models":[{"name":"qwen3:8b"}]}'

    class _Connection:
        def __init__(self, host, port, timeout=None):
            assert host in {"127.0.0.1", "localhost"}

        def request(self, method, path):
            assert method == "GET"
            assert path in {"/api/version", "/api/ps"}

        def getresponse(self):
            return _Response()

        def close(self):
            raise RuntimeError("close failed")

    class _Proc:
        pid = 44

    def popen(argv, **kwargs):
        assert kwargs["env"]["OLLAMA_HOST"] == "127.0.0.1:11434"
        return _Proc()

    seams = real_seams(run=run, platform="linux", popen=popen, connection=_Connection)
    status, payload, error = seams.health_get(URL_5090, 1)
    assert (status, payload["version"], error) == (200, "0.35.1", None)
    assert [row.pid for row in seams.processes()] == [9, 9]
    seams.stop(9)
    assert seams.spawn(["ollama", "serve"], {"OLLAMA_HOST": "127.0.0.1:11434"}) == 44
    found = seams.environment(URL_5090)
    assert found.version == "0.35.1"
    assert found.cuda_backend == "13.0"
    assert found.flash_attention == "unrecorded"
    assert seams.card_mib() == 100
    assert seams.ps(URL_5090)["models"][0]["name"] == "qwen3:8b"

    supervisor = OllamaSupervisor(seams)
    supervisor.prime(URL_5090)
    assert supervisor.recorded is not None
    assert supervisor.recorded.version == "0.35.1"
    supervisor.prime(URL_CPU)
    assert supervisor.recorded.version == "0.35.1"

    broken = World()
    bare = _supervisor(broken, fingerprint=False)
    bare.seams.environment = lambda url: (_ for _ in ()).throw(RuntimeError("down"))
    bare.prime(URL_5090)
    assert bare.recorded is None
    bare.seams.environment = lambda url: "nope"
    bare.prime(URL_5090)
    assert bare.recorded is None

    hung = World()
    hung.up = True
    hung.hold_up = True
    hung.ps = lambda url: (_ for _ in ()).throw(RuntimeError("ps down"))
    decision = _supervisor(hung).after_failure(URL_5090, model="qwen3:8b", kind="timeout")
    assert decision.record is not None
    assert decision.record.why == "hung"

    blind = World()
    blind.card_mib = lambda: (_ for _ in ()).throw(RuntimeError("no card"))
    with pytest.raises(SupervisorStop, match="did not return to its baseline"):
        card = _supervisor(blind)
        card.after_failure(URL_5090, model="qwen3:8b", kind="transport")
    assert card.restarts

    unreadable = World()
    unreadable.environment = lambda url: (_ for _ in ()).throw(RuntimeError("unread"))
    with pytest.raises(SupervisorStop, match="could not be read"):
        _supervisor(unreadable).after_failure(URL_5090, model="qwen3:8b", kind="transport")

    assert failure_kind(None) is None
    assert failure_kind("") is None
    assert failure_kind("server error") == "server"
    assert retry_budget(None) == 0
    assert card_mib_from_text(None) is None
    assert card_mib_from_text("nope") is None
    assert card_mib_from_text("-1") is None
    assert read_card_mib(run=lambda argv, **kwargs: subprocess.CompletedProcess(argv, 1, stdout="", stderr="")) is None
    assert read_cuda(run=lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, stdout="13.0\n12.0\n", stderr="")) is None

    class _Boom:
        def __init__(self, host, port, timeout=None):
            return None

        def request(self, method, path):
            raise OSError("closed")

        def close(self):
            return None

    status, payload, error = http_get("http://localhost:11434", "version", 1, connection=_Boom)
    assert error == "Ollama health only opens local http"
    status, payload, error = http_get(URL_5090, "/api/version", 1, connection=_Boom)
    assert "closed" in error

    class _BadJson:
        def __init__(self, host, port, timeout=None):
            return None

        def request(self, method, path):
            return None

        def getresponse(self):
            class _Body:
                status = 200

                def read(self):
                    return b"["

            return _Body()

        def close(self):
            return None

    status, payload, error = http_get(URL_5090, "/api/version", 1, connection=_BadJson)
    assert "not json" in error or "not an object" in error

    class _Zero:
        pid = 0

    with pytest.raises(SupervisorStop, match="did not return a pid"):
        spawn_detached(["ollama", "serve"], {}, popen=lambda argv, **kwargs: _Zero(), platform="linux")

    supervisor = OllamaSupervisor(World().seams())
    supervisor.begin_measurement()
    with pytest.raises(RestartRefused, match="already open"):
        supervisor.begin_measurement()
    assert _supervisor(World()).after_failure(URL_5090, model="qwen3:8b", kind="other").action == "leave"

    class _PsutilBoom:
        @staticmethod
        def net_connections(kind="tcp"):
            raise RuntimeError("no")

    assert pids_from_psutil(11434, _PsutilBoom) == []
    defaulted = SupervisorSeams(
        health_get=lambda url, timeout: (200, {"version": "0.35.1"}, None),
        processes=list,
        stop=lambda pid: None,
        spawn=lambda argv, env: 1,
        card_mib=lambda: 1,
        ps=lambda url: {},
        environment=lambda url: None,
        argv=lambda: ["ollama", "serve"],
    )
    assert defaulted.now().endswith("Z")
