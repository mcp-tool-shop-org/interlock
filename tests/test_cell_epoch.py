"""1.0.9: a cell epoch with a server restart after a cell's outer deadline, and a strict nvidia-smi parse.

When `CELL_DEADLINE_S` fires, the cell's thread is abandoned, not stopped. Without an epoch its next request would land
in the following cell, sharing slots, KV and prefill with it.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from interlock.devices import RemoteDeviceIdentity
from interlock.harness import linux_cuda
from interlock.harness.linux_cuda import (
    _CELL_EPOCH,
    GuardedSampler,
    LinuxOllamaServer,
    LinuxShapeFailure,
    StaleCell,
    measure_linux_ollama_generate,
    run_linux_generate_ladder,
)
from tests.fake_ollama import FakeOllama

IDENT = RemoteDeviceIdentity(
    provider="colab", card="A100-80", vram_mib=81920, driver_version="550.54.15", cuda_version="12.4",
    pool="us-central1-a",
)


class _Proc:
    pid = 4242

    def poll(self):
        return None

    def terminate(self) -> None:
        return None

    def wait(self, timeout=None):
        return 0


def test_a_call_from_an_old_cell_is_refused_and_the_current_cell_and_unbound_callers_are_not(tmp_path):
    server = LinuxOllamaServer(
        log_path=tmp_path / "ollama.log", popen=lambda *a, **k: _Proc(), http_get=lambda _u: {"version": "x"},
        http_post=lambda _u, _p: {"ok": True},
    )
    server.start()
    server.begin_cell()
    first = server.epoch
    token = _CELL_EPOCH.set(first)
    try:
        assert server.post("/api/generate", {}) == {"ok": True}  # this cell
        server.begin_cell()  # the next cell begins
        with pytest.raises(StaleCell):
            server.post("/api/generate", {})  # the old cell's call
        with pytest.raises(StaleCell):
            server.get("/api/ps")
        _CELL_EPOCH.set(server.epoch)
        assert server.post("/api/generate", {}) == {"ok": True}
    finally:
        _CELL_EPOCH.reset(token)
    assert server.get("/api/ps")  # a caller that is in no cell (a spec's setup) is not refused
    server.stop()


def _two_cells_the_first_outliving_its_deadline(monkeypatch) -> tuple[FakeOllama, list[dict], threading.Event]:
    monkeypatch.setattr(linux_cuda, "CELL_DEADLINE_S", 0.4)
    server = FakeOllama()
    blocked = threading.Event()
    release = threading.Event()
    original = server.post
    state = {"calls": 0}

    def post(path: str, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        if _CELL_EPOCH.get() == 1:  # the first cell
            state["calls"] += 1
            if state["calls"] == 2:  # its second probe never comes back, until the next cell is under way
                blocked.set()
                release.wait(30)
        elif (_CELL_EPOCH.get() or 0) >= 2:  # the restart advanced the epoch, then the next cell began
            release.set()  # the next cell has begun: the old thread wakes inside it
        return original(path, payload, timeout)

    server.post = post
    rows = run_linux_generate_ladder(
        spec_id="C2-GENERATE", spec_rel="experiments/c2-generate/spec.yaml", run_id="a100-r1", date="2026-10-26",
        remote_device=IDENT, model="qwen3:30b", revision_or_digest="d" * 64,
        measure_shape=lambda length, conc: measure_linux_ollama_generate(
            server, lambda: (1000, 81920), model="m", prompt_tokens=length, concurrency=conc, repeats=1,
            clock=server.clock,
        ),
        prompt_lengths=(512, 1024), concurrencies=(1,), quant="Q4_K_M", licence="apache-2.0",
        recover=server.restart_after_deadline,
    )
    return server, rows, release


def test_after_a_cells_deadline_the_next_cell_sees_no_request_from_the_old_one_and_the_server_restarted(monkeypatch):
    server, rows, _ = _two_cells_the_first_outliving_its_deadline(monkeypatch)
    assert rows[0]["outcome"] == "unmeasured" and "outer deadline" in rows[0]["notes"]
    assert "outcome" not in rows[1]  # the next cell measured
    deadline = time.monotonic() + 10
    while server.stale_calls == 0 and time.monotonic() < deadline:
        time.sleep(0.05)  # the abandoned thread wakes and tries to carry on
    assert server.stale_calls >= 1  # ... and its request into the next cell was refused
    assert server.restarts_after_deadline == 1
    # every request the server accepted after the next cell began belongs to that cell
    accepted = [call["epoch"] for call in server.all_calls]
    newest = accepted[-1]
    assert newest > 1 and set(accepted[accepted.index(newest) :]) == {newest}
    assert rows[0]["settings"].get("server_restarted_after_deadline") is None
    assert rows[1]["settings"]["server_restarted_after_deadline"] is True  # the next row says so


def test_a_normal_run_does_not_restart_the_server(monkeypatch):
    monkeypatch.setattr(linux_cuda, "CELL_DEADLINE_S", 30)
    server = FakeOllama()
    rows = run_linux_generate_ladder(
        spec_id="C2-GENERATE", spec_rel="experiments/c2-generate/spec.yaml", run_id="a100-r1", date="2026-10-26",
        remote_device=IDENT, model="qwen3:30b", revision_or_digest="d" * 64,
        measure_shape=lambda length, conc: measure_linux_ollama_generate(
            server, lambda: (1000, 81920), model="m", prompt_tokens=length, concurrency=conc, repeats=1,
            clock=server.clock,
        ),
        prompt_lengths=(512, 1024), concurrencies=(1,), quant="Q4_K_M", licence="apache-2.0",
        recover=server.restart_after_deadline,
    )
    assert server.restarts_after_deadline == 0 and server.stale_calls == 0
    assert all("server_restarted_after_deadline" not in row["settings"] for row in rows)


def test_the_measurement_leaves_the_callers_epoch_alone():
    before = _CELL_EPOCH.get()
    server = FakeOllama()
    measure_linux_ollama_generate(
        server, lambda: (1000, 81920), model="m", prompt_tokens=512, concurrency=1, repeats=1, clock=server.clock
    )
    assert _CELL_EPOCH.get() == before


# -- hygiene ------------------------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    monkeypatch.setattr(linux_cuda, "SAMPLER_TIMEOUT_S", 0.3)


def _hangs() -> tuple[int, int]:
    threading.Event().wait()
    return (0, 0)


def test_the_first_sample_of_a_window_and_a_batch_cell_has_a_deadline():
    from tests.test_linux_cuda_sampler_error import Server

    started = time.monotonic()
    with pytest.raises(LinuxShapeFailure) as window:
        linux_cuda.measure_linux_ollama_window(Server(), _hangs, model="m", num_ctx=2048)
    with pytest.raises(LinuxShapeFailure) as batch:
        linux_cuda.measure_linux_ollama_batch(Server(), _hangs, model="m", task="embed", batch=1, token_length=128)
    for caught in (window, batch):
        assert caught.value.failure_class == "unmeasured" and "did not answer within 0.3 s" in caught.value.reason
    assert time.monotonic() - started < 5


@pytest.mark.parametrize(
    "answer",
    ["1, 2, 3\n", "5000\n", "", "-1, 15360\n", "100, 0\n", "100, -5\n", "inf, 15360\n", "100, inf\n", "nan, 15360\n",
     "lots, 15360\n"],
)
def test_the_nvidia_smi_fallback_parse_is_strict(monkeypatch, answer):
    monkeypatch.setattr(linux_cuda, "_nvidia_smi_query", lambda timeout_s: answer)
    sampler = GuardedSampler(_hangs)
    with pytest.raises(LinuxShapeFailure):
        sampler()  # the NVML timeout, and the switch
    with pytest.raises(LinuxShapeFailure) as caught:
        sampler()
    assert caught.value.failure_class == "unmeasured"
    assert "nvidia-smi answered" in caught.value.reason


def test_the_nvidia_smi_fallback_accepts_two_numbers(monkeypatch):
    monkeypatch.setattr(linux_cuda, "_nvidia_smi_query", lambda timeout_s: "0, 81920\n")
    sampler = GuardedSampler(_hangs)
    with pytest.raises(LinuxShapeFailure):
        sampler()
    assert sampler() == (0, 81920)  # used may be 0


# -- card jb's follow-ups ----------------------------------------------------------------------------------------------


def test_a_deadline_with_no_recover_hook_does_not_claim_a_restart(monkeypatch, capsys):
    monkeypatch.setattr(linux_cuda, "CELL_DEADLINE_S", 0.3)
    hung = threading.Event()
    calls = {"n": 0}

    def measure(length: int, concurrency: int) -> dict:
        calls["n"] += 1
        if calls["n"] == 1:
            hung.wait(10)  # the first cell never returns in time
        return {"ttft_ms": 100.0, "prefill_ms": 90.0, "decode_tok_s": 20.0, "peak_mib": 9000, "size": 5,
                "size_vram": 5, "prompt_tokens_evaluated": length}

    rows = run_linux_generate_ladder(
        spec_id="C2-GENERATE", spec_rel="experiments/c2-generate/spec.yaml", run_id="a100-r1", date="2026-10-26",
        remote_device=IDENT, model="qwen3:30b", revision_or_digest="d" * 64, measure_shape=measure,
        prompt_lengths=(512, 1024), concurrencies=(1,), quant="Q4_K_M", licence="apache-2.0",
    )  # recover is None: a direct caller
    hung.set()
    assert rows[0]["outcome"] == "unmeasured" and "outcome" not in rows[1]
    assert "server_restarted_after_deadline" not in rows[1]["settings"]  # nothing restarted, so nothing is claimed
    err = capsys.readouterr().err
    assert "outer deadline fired; no recover hook, server not restarted" in err
    assert "restarting the server" not in err


def test_a_call_from_the_abandoned_cell_after_the_restart_but_before_the_next_cell_is_refused(tmp_path):
    server = LinuxOllamaServer(
        log_path=tmp_path / "ollama.log", popen=lambda *a, **k: _Proc(), http_get=lambda _u: {"version": "x"},
        http_post=lambda _u, _p: {"ok": True},
    )
    server.start()
    server.begin_cell()
    token = _CELL_EPOCH.set(server.epoch)  # the abandoned cell's thread
    try:
        assert server.post("/api/generate", {}) == {"ok": True}
        server.restart_after_deadline()  # the ladder restarts after the deadline; the next begin_cell() has not run yet
        with pytest.raises(StaleCell):
            server.post("/api/generate", {})
        with pytest.raises(StaleCell):
            server.get("/api/ps")
    finally:
        _CELL_EPOCH.reset(token)
    server.stop()


@pytest.mark.parametrize(
    "answer",
    ["20000, 15360\n", "-0.5, 15360\n", "1000, 15360\n2000, 15360\n", "1000, 15360\n\n3, 4\n", "15361, 15360\n",
     "1000, 1e999\n"],
)
def test_the_nvidia_smi_fallback_bounds_used_against_total_and_takes_one_line(monkeypatch, answer):
    monkeypatch.setattr(linux_cuda, "_nvidia_smi_query", lambda timeout_s: answer)
    sampler = GuardedSampler(_hangs)
    with pytest.raises(LinuxShapeFailure):
        sampler()
    with pytest.raises(LinuxShapeFailure) as caught:
        sampler()
    assert caught.value.failure_class == "unmeasured" and "nvidia-smi answered" in caught.value.reason


def test_the_nvidia_smi_fallback_still_accepts_used_equal_to_total_and_a_blank_trailing_line(monkeypatch):
    monkeypatch.setattr(linux_cuda, "_nvidia_smi_query", lambda timeout_s: "15360, 15360\n\n")
    sampler = GuardedSampler(_hangs)
    with pytest.raises(LinuxShapeFailure):
        sampler()
    assert sampler() == (15360, 15360)
