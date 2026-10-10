"""The run command's device path: every guard refuses before an adapter is called.

Adapters are fakes and the spec's rest runs against a fake clock. No test in
this file touches a device.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest
import yaml

from switchyard.adapters.base import AdapterError, StepResult
from switchyard.cli import main
from switchyard.guards import GuardError
from switchyard.harness.runner import (
    RunnerError,
    devices_not_started,
    dry_run,
    heartbeat_path,
    real_adapters,
    run_spec,
)
from switchyard.harness.spec import SpecError
from switchyard.receipt import check_tree, load_jsonl

ROOT = Path(__file__).resolve().parents[1]
E5 = ROOT / "experiments" / "e5" / "spec.yaml"


def _arm(arm_id, device):
    return {
        "id": arm_id,
        "task": "generate",
        "device": device,
        "engine": "fake",
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


def _spec(arms, *, needs=None, rest_s=900, warmups=0, repeats=2):
    return {
        "id": "T",
        "procedure": "interleaved",
        "question": "Does the run path guard before it touches a device?",
        "decision": "A test verdict, computed after the run.",
        "needs": needs or {"grant": False, "npu_session": False, "cpu_window": False},
        "stop_rules": ["a failed step stops the sequence"],
        "dataset": {"name": "fixture"},
        "metrics": ["latency_ms"],
        "warmups": warmups,
        "repeats": repeats,
        "band": 0.1,
        "compile_cache": "reuse",
        "rest_s": rest_s,
        "arms": arms,
    }


def _commit(repo: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "dev"], cwd=repo, check=True)
    subprocess.run(["git", "config", "core.autocrlf", "false"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=repo, check=True, capture_output=True)


def _bind(repo: Path, spec: dict) -> Path:
    path = repo / "experiments" / "t" / "spec.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8", newline="\n")
    _commit(repo)
    return path


class Fake:
    """A run/unload adapter with a card probe. ``loaded=None`` makes the probe unexpected."""

    def __init__(self, samples=None, *, loaded=None):
        self.samples = samples or {}
        self.loaded = loaded
        self.calls = []
        self.unloaded = []

    def run(self, request):
        item = self.samples[request.step_id].pop(0)
        if isinstance(item, BaseException):
            raise item
        self.calls.append(request.step_id)
        return item

    def unload(self, model, device=None):
        self.unloaded.append((model, device))

    def loaded_models(self):
        if self.loaded is None:
            raise AssertionError("the card probe was not expected")
        return self.loaded


class FakeTime:
    """sleep advances the fake clock by ``advance`` seconds, not the requested time."""

    def __init__(self, advance=None):
        self.now = 1_000_000.0
        self.advance = advance

    def sleep(self, seconds: float) -> None:
        self.now += self.advance if self.advance is not None else seconds

    def monotonic(self) -> float:
        return self.now


def _no_rest_attempted(seconds: float) -> None:
    raise AssertionError(f"the rest was attempted ({seconds}s)")


def _result(latency=10):
    return StepResult(ok=True, latency_ms=latency)


def _fresh_heartbeat(tmp_path: Path) -> Path:
    path = tmp_path / "_watchdog_HEARTBEAT"
    path.write_text("pulse", encoding="utf-8")
    now = time.time()
    os.utime(path, (now, now))
    return path


NEED_GRANT = {"grant": True, "npu_session": False, "cpu_window": False}
NEED_SESSION = {"grant": False, "npu_session": True, "cpu_window": False}
NEED_WINDOW = {"grant": False, "npu_session": False, "cpu_window": True}


def test_a_missing_grant_refuses_before_any_adapter(tmp_path: Path):
    repo = tmp_path / "repo"
    spec_path = _bind(repo, _spec([_arm("A", "5090")], needs=NEED_GRANT))
    fake = Fake()
    with pytest.raises(GuardError, match="--grant"):
        run_spec(spec_path, root=repo, adapters={"fake": fake}, sleep=_no_rest_attempted)
    assert fake.calls == []
    assert fake.unloaded == []


def test_a_missing_npu_session_refuses_before_any_adapter(tmp_path: Path):
    repo = tmp_path / "repo"
    spec_path = _bind(repo, _spec([_arm("A", "NPU")], needs=NEED_SESSION))
    fake = Fake()
    with pytest.raises(GuardError, match="--npu-session"):
        run_spec(spec_path, root=repo, grant="g", adapters={"fake": fake}, sleep=_no_rest_attempted)
    assert fake.calls == []


def test_a_missing_cpu_window_refuses_before_any_adapter(tmp_path: Path):
    repo = tmp_path / "repo"
    spec_path = _bind(repo, _spec([_arm("A", "CPU")], needs=NEED_WINDOW))
    fake = Fake()
    with pytest.raises(GuardError, match="--cpu-window"):
        run_spec(spec_path, root=repo, adapters={"fake": fake}, sleep=_no_rest_attempted)
    assert fake.calls == []


def test_an_edited_spec_refuses_before_any_adapter(tmp_path: Path):
    repo = tmp_path / "repo"
    spec_path = _bind(repo, _spec([_arm("A", "CPU")]))
    spec_path.write_text(
        spec_path.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8", newline="\n"
    )
    fake = Fake()
    with pytest.raises(SpecError, match="edited after commit"):
        run_spec(spec_path, root=repo, adapters={"fake": fake}, sleep=_no_rest_attempted)
    assert fake.calls == []


def test_stale_heartbeat_refuses_before_the_probe(tmp_path: Path):
    repo = tmp_path / "repo"
    spec_path = _bind(repo, _spec([_arm("A", "5090")], needs=NEED_GRANT))
    heartbeat = tmp_path / "_watchdog_HEARTBEAT"
    heartbeat.write_text("pulse", encoding="utf-8")
    stale = time.time() - 120
    os.utime(heartbeat, (stale, stale))
    fake = Fake()  # loaded=None: a probe would fail the test
    with pytest.raises(GuardError, match="watchdog heartbeat"):
        run_spec(
            spec_path,
            root=repo,
            grant="g",
            adapters={"fake": fake},
            heartbeat=heartbeat,
            sleep=_no_rest_attempted,
        )
    assert fake.calls == []


def test_a_loaded_card_refuses(tmp_path: Path):
    repo = tmp_path / "repo"
    spec_path = _bind(repo, _spec([_arm("A", "5090")], needs=NEED_GRANT))
    fake = Fake(loaded=[{"name": "qwen3:8b"}])
    with pytest.raises(GuardError, match="still on the card"):
        run_spec(
            spec_path,
            root=repo,
            grant="g",
            adapters={"fake": fake},
            heartbeat=_fresh_heartbeat(tmp_path),
            sleep=_no_rest_attempted,
        )
    assert fake.calls == []


def test_a_short_rest_refuses(tmp_path: Path):
    repo = tmp_path / "repo"
    spec_path = _bind(repo, _spec([_arm("A", "5090")], needs=NEED_GRANT))
    clock = FakeTime(advance=100.0)  # the sleep lied: less than the spec's rest passed
    fake = Fake(loaded=[])
    with pytest.raises(GuardError, match="rested"):
        run_spec(
            spec_path,
            root=repo,
            grant="g",
            adapters={"fake": fake},
            heartbeat=_fresh_heartbeat(tmp_path),
            sleep=clock.sleep,
            clock=clock.monotonic,
        )
    assert fake.calls == []


def test_an_in_repo_interpreter_is_refused(tmp_path: Path):
    repo = tmp_path / "repo"
    spec_path = _bind(repo, _spec([_arm("A", "CPU")]))
    fake = Fake()
    bad = repo / ".venv" / "Scripts" / "python.exe"
    with pytest.raises(GuardError, match="UV_PROJECT_ENVIRONMENT"):
        run_spec(spec_path, root=repo, adapters={"fake": fake}, python_path=str(bad), sleep=_no_rest_attempted)
    assert fake.calls == []


def test_a_fill_card_spec_has_no_run_path(tmp_path: Path):
    repo = tmp_path / "repo"
    target = repo / "experiments" / "e5" / "spec.yaml"
    target.parent.mkdir(parents=True)
    target.write_text(E5.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    _commit(repo)
    with pytest.raises(GuardError, match="no run path"):
        run_spec(target, root=repo, grant="g", adapters={"fake": Fake()}, sleep=_no_rest_attempted)


def test_an_existing_receipt_name_is_refused(tmp_path: Path):
    repo = tmp_path / "repo"
    spec_path = _bind(repo, _spec([_arm("A", "CPU")]))
    receipts = repo / "receipts"
    receipts.mkdir()
    (receipts / "run-t.jsonl").write_text("", encoding="utf-8")
    fake = Fake()
    with pytest.raises(GuardError, match="already exists"):
        run_spec(
            spec_path,
            root=repo,
            adapters={"fake": fake},
            run_id="run-t",
            date="2026-10-10",
            sleep=_no_rest_attempted,
        )
    assert fake.calls == []


def test_a_clean_run_writes_a_receipt_check_accepts(tmp_path: Path):
    repo = tmp_path / "repo"
    spec_path = _bind(repo, _spec([_arm("A", "CPU")]))
    clock = FakeTime()
    fake = Fake({"A": [_result(10), _result(11), _result(12)]})
    rows = run_spec(
        spec_path,
        root=repo,
        adapters={"fake": fake},
        date="2026-10-10",
        sleep=clock.sleep,
        clock=clock.monotonic,
    )
    assert len(rows) == 1
    receipt = repo / "receipts" / "t-2026-10-10.jsonl"
    assert receipt.is_file()
    assert check_tree(repo) == []
    assert main(["check", "--root", str(repo)]) == 0
    # The bare repo has no table directory: the run leaves the table alone.
    assert not (repo / "table").exists()
    assert fake.unloaded == [("qwen3:8b", "CPU")]


def test_a_5090_run_proceeds_after_both_card_checks(tmp_path: Path):
    repo = tmp_path / "repo"
    spec_path = _bind(repo, _spec([_arm("A", "5090")], needs=NEED_GRANT))
    clock = FakeTime()
    fake = Fake({"A": [_result(50), _result(51), _result(52)]}, loaded=[])
    rows = run_spec(
        spec_path,
        root=repo,
        grant="g",
        adapters={"fake": fake},
        heartbeat=_fresh_heartbeat(tmp_path),
        sleep=clock.sleep,
        clock=clock.monotonic,
        date="2026-10-10",
        run_id="card-ok",
    )
    assert len(rows) == 1
    assert check_tree(repo) == []
    assert fake.calls == ["A", "A", "A"]
    assert fake.unloaded == [("qwen3:8b", "5090")]


def test_a_hang_stops_the_run_and_keeps_a_partial_receipt(tmp_path: Path):
    repo = tmp_path / "repo"
    spec_path = _bind(repo, _spec([_arm("A", "CPU"), _arm("B", "CPU")]))
    clock = FakeTime()
    adapter = Fake(
        {
            "A": [_result(10), _result(11), _result(12)],
            "B": [AdapterError("hang: no result within 60 s")],
        }
    )
    with pytest.raises(RunnerError, match="hang"):
        run_spec(
            spec_path,
            root=repo,
            adapters={"fake": adapter},
            date="2026-10-10",
            run_id="hang-1",
            sleep=clock.sleep,
            clock=clock.monotonic,
        )
    receipt = repo / "receipts" / "hang-1.jsonl"
    assert receipt.is_file()
    rows = load_jsonl(receipt)
    assert [row.get("outcome") for row in rows] == ["hang", None]
    assert rows[0]["partial"] is True
    assert rows[1]["aborted"] is True
    assert rows[1]["id"] == "T/A/hang-1"
    assert rows[1]["settings"]["abortion"]["stop_step"] == "B cold"
    assert check_tree(repo) == []


def test_the_command_runs_and_refuses_a_name_collision(tmp_path: Path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    spec_path = _bind(repo, _spec([_arm("A", "CPU")]))
    fake = Fake({"A": [_result(10), _result(11), _result(12)]})
    clock = FakeTime()
    monkeypatch.setattr("switchyard.harness.runner.for_engine", lambda engine: fake)
    monkeypatch.setattr("time.sleep", clock.sleep)
    monkeypatch.setattr("time.monotonic", clock.monotonic)
    assert main(["run", str(spec_path), "--root", str(repo), "--run-id", "cli-run"]) == 0
    assert (repo / "receipts" / "cli-run.jsonl").is_file()
    assert main(["run", str(spec_path), "--root", str(repo), "--run-id", "cli-run"]) == 1
    assert "already exists" in capsys.readouterr().err


def test_the_command_refuses_a_missing_token_before_building_adapters(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    spec_path = _bind(repo, _spec([_arm("A", "5090")], needs=NEED_GRANT))

    def explode(engine):
        raise AssertionError("an adapter was built before the guards")

    monkeypatch.setattr("switchyard.harness.runner.for_engine", explode)
    assert main(["run", str(spec_path), "--root", str(repo)]) == 1


def test_real_adapters_uses_the_registry_per_engine(monkeypatch):
    seen = []
    monkeypatch.setattr(
        "switchyard.harness.runner.for_engine", lambda engine: seen.append(engine) or engine
    )
    adapters = real_adapters({"arms": [{"engine": "ollama"}, {"engine": "openvino"}, {"engine": "ollama"}]})
    assert adapters == {"ollama": "ollama", "openvino": "openvino"}
    assert seen == ["ollama", "openvino"]


def test_heartbeat_path_prefers_the_env_override(monkeypatch, tmp_path: Path):
    target = tmp_path / "hb"
    monkeypatch.setenv("WATCHDOG_HEARTBEAT", str(target))
    assert heartbeat_path() == target
    monkeypatch.delenv("WATCHDOG_HEARTBEAT")
    assert heartbeat_path() == Path("E:/AI/training/_watchdog_HEARTBEAT")
    assert heartbeat_path(target).name == "hb"


def test_the_dry_run_closes_with_devices_not_started(tmp_path: Path):
    repo = tmp_path / "repo"
    spec_path = _bind(repo, _spec([_arm("A", "CPU")]))
    text = dry_run(spec_path, repo)
    assert "heartbeat not read" in text
    assert text.rstrip().endswith(devices_not_started())
