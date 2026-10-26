"""After the first NVML timeout the run's sampler is `nvidia-smi` in a subprocess (T4 r4's hang, card iz finding D1)."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from typing import Any

import pytest

from interlock.harness import linux_cuda
from interlock.harness.linux_cuda import GuardedSampler, LinuxShapeFailure
from interlock.receipt import validate_row
from tests.fake_ollama import FakeOllama
from tests.test_wall_cap_and_hang import IDENT, _ladder, _spec_repo


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    monkeypatch.setattr(linux_cuda, "SAMPLER_TIMEOUT_S", 0.3)


def _blocked() -> tuple[int, int]:
    threading.Event().wait()
    return (0, 0)


def test_the_first_timeout_is_unmeasured_and_the_run_then_samples_through_nvidia_smi(monkeypatch):
    asked: list[float] = []
    monkeypatch.setattr(linux_cuda, "_nvidia_smi_query", lambda timeout_s: asked.append(timeout_s) or "2048, 15360\n")
    calls = {"primary": 0}

    def primary() -> tuple[int, int]:
        calls["primary"] += 1
        return _blocked()

    sampler = GuardedSampler(primary)
    with pytest.raises(LinuxShapeFailure) as caught:
        sampler()
    assert caught.value.failure_class == "unmeasured" and "did not answer within 0.3 s" in caught.value.reason
    assert sampler.mode == "nvidia-smi-fallback" and "nvml sampler did not answer within 0.3 s" in sampler.switch_reason
    started = time.monotonic()
    for _ in range(5):
        assert sampler() == (2048, 15360)  # later samples pay no SAMPLER_TIMEOUT_S, and never call the wedged NVML
    assert time.monotonic() - started < 1.0
    assert calls["primary"] == 1 and asked == [linux_cuda.NVIDIA_SMI_TIMEOUT_S] * 5


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10.0), "nvidia-smi did not answer within 10 s"),
        (FileNotFoundError("nvidia-smi"), "nvidia-smi failed: nvidia-smi"),
        (subprocess.CalledProcessError(9, "nvidia-smi"), "nvidia-smi failed:"),
    ],
)
def test_a_fallback_that_hangs_or_fails_is_unmeasured_with_the_real_reason_every_time(monkeypatch, failure, expected):
    def query(timeout_s: float) -> str:
        raise failure

    monkeypatch.setattr(linux_cuda, "_nvidia_smi_query", query)
    sampler = GuardedSampler(_blocked)
    with pytest.raises(LinuxShapeFailure):
        sampler()  # the NVML timeout
    for _ in range(3):  # and it stays bounded, cell after cell
        with pytest.raises(LinuxShapeFailure) as caught:
            sampler()
        assert caught.value.failure_class == "unmeasured" and expected in caught.value.reason


def test_a_fallback_answer_that_is_not_used_and_total_is_unmeasured(monkeypatch):
    monkeypatch.setattr(linux_cuda, "_nvidia_smi_query", lambda timeout_s: "No devices were found\n")
    sampler = GuardedSampler(_blocked)
    with pytest.raises(LinuxShapeFailure):
        sampler()
    with pytest.raises(LinuxShapeFailure) as caught:
        sampler()
    assert "not 'used, total'" in caught.value.reason


def test_the_real_fallback_runs_nvidia_smi_as_a_subprocess_with_a_timeout(monkeypatch):
    seen: dict[str, Any] = {}

    def run(argv, **kwargs):
        seen.update(argv=argv, **kwargs)
        return subprocess.CompletedProcess(argv, 0, stdout="4096, 15360\n", stderr="")

    monkeypatch.setattr(linux_cuda.subprocess, "run", run)
    assert linux_cuda._nvidia_smi_query(10.0) == "4096, 15360\n"
    assert seen["argv"][:2] == ["nvidia-smi", "--query-gpu=memory.used,memory.total"]
    assert seen["timeout"] == 10.0 and seen["check"] is True


def test_a_cell_records_the_sampler_it_used_and_why_the_run_switched(monkeypatch):
    monkeypatch.setattr(linux_cuda, "_nvidia_smi_query", lambda timeout_s: "1000, 15360\n")
    sampler = GuardedSampler(_blocked)
    first = _ladder(FakeOllama(), (512,), sample=sampler)  # the NVML timeout
    later = _ladder(FakeOllama(), (512,), sample=sampler)
    assert first[0]["outcome"] == "unmeasured"
    assert first[0]["settings"]["vram_sampler"] == "nvidia-smi-fallback"
    assert "nvml sampler did not answer" in first[0]["settings"]["vram_sampler_reason"]
    assert "outcome" not in later[0]
    assert later[0]["settings"]["vram_sampler"] == "nvidia-smi-fallback"
    for row in (*first, *later):
        validate_row(row)


def test_a_run_that_never_timed_out_records_its_primary_sampler():
    sampler = GuardedSampler(lambda: (1000, 15360))
    (row,) = _ladder(FakeOllama(), (512,), sample=sampler)
    assert row["settings"]["vram_sampler"] == "nvml" and row["settings"]["vram_sampler_reason"] is None


def test_the_t4_r5_run_holds_nvml_wedged_after_a_host_ram_kill_the_first_cell_is_lost_and_later_cells_measure(
    tmp_path, monkeypatch
):
    from interlock.harness.runner import run_spec

    monkeypatch.setattr(linux_cuda, "_nvidia_smi_query", lambda timeout_s: "1000, 15360\n")
    spec = _spec_repo(tmp_path)
    server = FakeOllama(vram_mib=6000.0)  # c4 at 16k and 32k are SIGKILLed for host RAM
    seen = {"blocked": 0}

    def nvml() -> tuple[int, int]:
        if server.kills:  # once the OOM killer has taken a process, NVML never answers again
            seen["blocked"] += 1
            threading.Event().wait()
        return (1000, 15360)

    outcome: dict[str, Any] = {}

    def run() -> None:
        try:
            outcome["rows"] = run_spec(
                spec, root=tmp_path, run_id="c2-r5", date="2026-10-26",
                python_path=str(tmp_path.parent / "external-python"), models=["qwen3:8b"],
                seams={"server": server, "remote_device": IDENT, "sample_vram_mib": nvml},
            )
        except BaseException as exc:  # noqa: BLE001  # surfaced below
            outcome["error"] = exc

    started = time.monotonic()
    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(120)
    assert not worker.is_alive(), "the run hung"
    assert "error" not in outcome, outcome.get("error")
    rows = outcome["rows"]
    assert server.kills >= 2  # more than one host-RAM kill: without the switch each one is followed by its own hang
    assert seen["blocked"] == 1  # NVML was asked once after it wedged; the run then used nvidia-smi
    lost = [r for r in rows if "VRAM sampler did not answer" in r.get("notes", "")]
    assert len(lost) == 1 and lost[0]["outcome"] == "unmeasured"
    after = rows[rows.index(lost[0]) + 1 :]
    assert any("outcome" not in r for r in after), "no cell after the switch was measured"
    used = {r["settings"]["vram_sampler"] for r in after if "vram_sampler" in r["settings"]}
    assert used == {"nvidia-smi-fallback"}
    assert time.monotonic() - started < 100
    record = json.loads((tmp_path / "receipts" / "c2-r5.run.json").read_text(encoding="utf-8"))
    assert record["closed"] is True
