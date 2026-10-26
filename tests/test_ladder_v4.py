"""Ladder v4 (T4 r3): a partial offload is oom before any timed call, a SIGKILL is host RAM only on the log's numbers,
and each cell's request timeout is scaled from its own verify call and recorded.

The log fixtures are the lines of T4 r3's ``linux-ollama.log`` that the parser reads (Ollama v0.40.11).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from interlock.devices import RemoteDeviceIdentity
from interlock.harness import linux_cuda
from interlock.harness.linux_cuda import (
    CALIBRATION_TIMEOUT_S,
    GENERATE_LADDER_VERSION,
    LinuxShapeFailure,
    measure_linux_ollama_generate,
    parse_load_report,
    request_timeout_s,
    run_linux_generate_ladder,
)
from interlock.receipt import validate_row
from interlock.router.policy import LADDER_MEASURED_FROM, Decision, Request, decide
from tests.fake_ollama import FakeOllama

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ollama_logs" / "0.40.11"
IDENT = RemoteDeviceIdentity(
    provider="colab", card="T4", vram_mib=15360, driver_version="550.54.15", cuda_version="12.4", pool="us-central1-a"
)


def _measure(server: FakeOllama, target: int, concurrency: int = 1, repeats: int = 2) -> dict:
    return measure_linux_ollama_generate(
        server,
        lambda: (1000, 15360),
        model="m",
        prompt_tokens=target,
        concurrency=concurrency,
        repeats=repeats,
        clock=server.clock,
    )


def _ladder(server: FakeOllama, lengths: tuple[int, ...], concurrencies: tuple[int, ...] = (1,)) -> list[dict]:
    return run_linux_generate_ladder(
        spec_id="C2-GENERATE", spec_rel="experiments/c2-generate/spec.yaml", run_id="t4-r4", date="2026-10-26",
        remote_device=IDENT, model="phi3.5:3.8b", revision_or_digest="d" * 64,
        measure_shape=lambda length, conc: _measure(server, length, conc),
        prompt_lengths=lengths, concurrencies=concurrencies, quant="Q4_K_M", licence="mit",
    )


def test_the_version_is_4_and_the_router_guard_stays_at_3():
    assert GENERATE_LADDER_VERSION == 4
    assert LADDER_MEASURED_FROM == 3


# -- the log parser, on lines from T4 r3 ------------------------------------------------------------------------------


def test_the_parser_reads_the_killed_p8192_c8_load_from_the_real_log():
    report = parse_load_report((FIXTURES / "r3_killed_p8192_c8.log").read_text(encoding="utf-8"))
    assert report["layers_on_gpu"] == 15 and report["layers_total"] == 33
    assert report["cpu_kv_mib"] == 14688.0
    assert report["cpu_mapped_mib"] == pytest.approx(731.45 + 70.34 + 461.41)  # the CPU_Mapped lines are summed
    assert report["system_total_mib"] == pytest.approx(12.7 * 1024)


def test_the_parser_reads_a_partial_offload_and_a_fully_resident_load():
    partial = parse_load_report((FIXTURES / "r3_partial_29_of_33.log").read_text(encoding="utf-8"))
    assert (partial["layers_on_gpu"], partial["layers_total"], partial["cpu_kv_mib"]) == (29, 33, 1584.0)
    resident = parse_load_report((FIXTURES / "r3_resident.log").read_text(encoding="utf-8"))
    assert (resident["layers_on_gpu"], resident["layers_total"]) == (33, 33)
    assert resident["cpu_kv_mib"] is None  # not present, so not inferred


def test_the_parser_returns_none_for_everything_it_does_not_find_and_uses_the_latest_load():
    assert parse_load_report("") == {
        "layers_on_gpu": None, "layers_total": None, "cpu_kv_mib": None, "cpu_mapped_mib": None,
        "system_total_mib": None,
    }
    two = (FIXTURES / "r3_killed_p8192_c8.log").read_text(encoding="utf-8") + (
        FIXTURES / "r3_resident.log"
    ).read_text(encoding="utf-8")
    assert parse_load_report(two)["layers_on_gpu"] == 33  # the latest load, not the first


# -- A: a partial offload is oom, before any verify or timed call ------------------------------------------------------


def _small_card(**extra: Any) -> FakeOllama:
    """A card that holds the weights and about 5.6 GB of KV: a c4 cell at 8k needs 12.8 GB."""

    return FakeOllama(vram_mib=8000.0, **extra)


def test_a_partial_offload_is_oom_with_its_evidence_and_no_verify_or_timed_call_is_sent():
    server = _small_card()
    with pytest.raises(LinuxShapeFailure) as caught:
        _measure(server, 8192, concurrency=4)
    failure = caught.value
    assert failure.failure_class == "oom"
    assert "partial offload at num_ctx" in failure.reason and "x 4 slots" in failure.reason
    assert "layers on the GPU" in failure.reason and "MiB of KV in host RAM" in failure.reason
    assert "size_vram" in failure.reason  # /api/ps said so as well
    assert len(server.all_calls) == 1  # the first probe only: no second probe, no verify, no timed call
    assert server.clock() < 1.0  # seconds of fake time, not a 120 s timeout
    offload = failure.evidence["offload"]
    assert offload["layers_on_gpu"] < offload["layers_total"] == 33
    assert offload["cpu_kv_mib"] > 0 and offload["source"] == "llama-server log and /api/ps"


def test_a_partial_offload_is_found_from_api_ps_alone_when_the_log_says_nothing():
    server = _small_card(log_lines=False)
    with pytest.raises(LinuxShapeFailure) as caught:
        _measure(server, 8192, concurrency=4)
    assert caught.value.failure_class == "oom"
    assert "layers on the GPU" not in caught.value.reason  # only what was read is in the reason
    assert caught.value.evidence["offload"]["source"] == "/api/ps"


def test_a_fully_resident_cell_is_not_called_a_partial_offload():
    server = FakeOllama(vram_mib=15360.0)
    assert _measure(server, 2048)["size"] > 0


def test_the_oom_row_carries_the_offload_evidence_and_the_budget_and_validates():
    (row,) = _ladder(_small_card(), (8192,), (4,))
    validate_row(row)
    assert row["outcome"] == "oom" and row["notes"].startswith("oom: partial offload")
    assert row["settings"]["offload"]["layers_total"] == 33
    assert row["settings"]["request_timeout_s"] == CALIBRATION_TIMEOUT_S  # the budget in force when it failed
    assert row["settings"]["ladder_version"] == 4


# -- B: a SIGKILL during a load is host RAM only on the log's numbers -------------------------------------------------


def test_a_kill_with_host_buffers_over_the_vm_ram_is_oom_host_ram():
    server = _small_card()  # 33k x 8 slots of KV: 99 GB, almost all of it in host RAM
    with pytest.raises(LinuxShapeFailure) as caught:
        _measure(server, 32768, concurrency=8)
    reason = caught.value.reason
    assert caught.value.failure_class == "oom"
    assert reason.startswith("host RAM: the load needed ") and "MiB in host RAM (CPU KV " in reason
    assert "mapped weights 70" in reason and "the VM has 13005 MiB (Ollama system memory)" in reason
    assert len(server.all_calls) == 1  # the kill came on the first probe


def test_a_kill_with_no_log_numbers_stays_unmeasured():
    server = _small_card(log_lines=False)
    with pytest.raises(LinuxShapeFailure) as caught:
        _measure(server, 32768, concurrency=8)
    assert caught.value.failure_class == "unmeasured"
    assert "signal: killed" in caught.value.reason and "no host-RAM numbers" in caught.value.reason


def test_a_kill_with_host_buffers_under_the_vm_ram_stays_unmeasured():
    class Killed(FakeOllama):
        def post(self, path, payload, timeout=None):  # a kill whose log shows 1 GB of host KV on a 12.7 GiB VM
            self.log_text = (FIXTURES / "r3_partial_29_of_33.log").read_text(encoding="utf-8")
            raise RuntimeError("llama-server process has terminated: signal: killed")

    with pytest.raises(LinuxShapeFailure) as caught:
        _measure(Killed(), 2048)
    assert caught.value.failure_class == "unmeasured"


def test_a_kill_with_no_system_memory_line_is_not_host_ram():
    class Killed(FakeOllama):
        def post(self, path, payload, timeout=None):
            self.log_text = "llama_kv_cache:        CPU KV buffer size = 99999.00 MiB\n"
            raise RuntimeError("llama-server process has terminated: signal: killed")

    with pytest.raises(LinuxShapeFailure) as caught:
        _measure(Killed(), 2048)
    assert caught.value.failure_class == "unmeasured"  # never infer host RAM without the numbers


# -- C: a request timeout scaled from the cell's own verify call, and recorded ----------------------------------------


def test_the_clamp_holds_at_both_ends():
    assert request_timeout_s(1, 0.5) == 120  # 62 s of budget is under the floor
    assert request_timeout_s(1, 130.0) == 3 * 130 + 60
    assert request_timeout_s(8, 200.0) == 900  # 4860 s is over the ceiling
    assert request_timeout_s(4, 25.0) == 360  # p16384 on a T4: ceil(3 x 4 x 25) + 60


def test_a_slow_fully_resident_cell_measures_under_the_scaled_budget():
    server = FakeOllama(overhead_s=130.0)  # a timed call takes over 130 s: past the old fixed 120 s
    result = _measure(server, 512)
    assert result["request_timeout_s"] > 120
    timed = [t for t, call in zip(server.timeouts, server.all_calls, strict=True) if call["num_predict"] > 1]
    assert set(timed) == {result["request_timeout_s"]}
    assert {t for t, call in zip(server.timeouts, server.all_calls, strict=True) if call["num_predict"] == 1} == {
        CALIBRATION_TIMEOUT_S
    }


def test_the_same_cell_fails_as_a_timeout_when_the_budget_is_forced_to_120(monkeypatch):
    monkeypatch.setattr(linux_cuda, "REQUEST_TIMEOUT_CEILING_S", 120)
    server = FakeOllama(overhead_s=130.0)
    with pytest.raises(LinuxShapeFailure) as caught:
        _measure(server, 512)
    assert caught.value.failure_class == "timeout"
    assert caught.value.reason.startswith("did not finish within 120 s (scaled from a 130.")
    assert "s verify call at concurrency 1)" in caught.value.reason
    assert caught.value.evidence["request_timeout_s"] == 120


def test_every_generate_row_records_the_request_timeout_measured_or_failed(monkeypatch):
    (ok,) = _ladder(FakeOllama(overhead_s=130.0), (512,))
    assert ok["settings"]["request_timeout_s"] > 120 and ok["settings"]["ladder_version"] == 4
    monkeypatch.setattr(linux_cuda, "REQUEST_TIMEOUT_CEILING_S", 120)
    (timed_out,) = _ladder(FakeOllama(overhead_s=130.0), (512,))
    assert timed_out["outcome"] == "timeout" and timed_out["settings"]["request_timeout_s"] == 120
    assert "did not finish within 120 s" in timed_out["notes"]
    for row in (ok, timed_out):
        validate_row(row)


def test_the_server_post_takes_a_timeout_and_passes_it_to_the_request():
    seen: list[float] = []

    def post(url, payload, timeout=120.0):
        seen.append(timeout)
        return {}

    server = linux_cuda.LinuxOllamaServer()
    original = linux_cuda._default_http_post
    linux_cuda._default_http_post = post
    try:
        server.post("/api/generate", {}, timeout=451)
    finally:
        linux_cuda._default_http_post = original
    assert seen[0] == 451


# -- F2: the router drops rows whose ttft was not measured before it compares prefix caching ---------------------------


def _v3_row(row_id: str, throughput: float, ttft: float, concurrency: int = 1) -> dict:
    return {
        "id": row_id, "task": "generate", "model": {"repo_or_tag": "phi3.5:3.8b", "revision_or_digest": "abc"},
        "engine": {"name": "ollama", "version": "0.40.11"}, "device": "colab:T4",
        "settings": {"thinking_level": "off", "num_predict": 128, "num_ctx": 2240, "concurrency": concurrency,
                     "ladder_version": 3, "prefix_caching": "off"},
        "date": "2026-10-26", "repeats": 3, "provisional": False,
        "artifacts": {"slots": [{"phase": "repeat", "throughput": throughput, "latency_ms": 100, "ttft_ms": ttft}] * 3},
        "provenance": {"experiment_spec": "experiments/c2-generate/spec.yaml", "run_id": "t"},
    }


def _fill_card_row() -> dict:
    row = _v3_row("fill-card", 71.6, 400.0)
    row["settings"] = {"thinking_level": "off", "num_predict": None, "num_ctx": 22528, "max_resident_ctx": 22528,
                       "ttft_ms": 400.0}  # unversioned, prefix_caching unset: as r3 produced it
    return row


def test_a_fill_card_row_beside_v3_rows_no_longer_makes_a_ttft_floor_refuse():
    rows = [_v3_row("v3-c1", 67.0, 900.0), _v3_row("v3-c4", 120.0, 4200.0, concurrency=4), _fill_card_row()]
    request = Request(
        task="generate", as_of=date(2026, 10, 26), model="phi3.5:3.8b", allow_remote=True, device="colab:T4",
        max_ttft_ms=5000.0,
    )
    result = decide(rows, request)
    assert isinstance(result, Decision)
    assert result.document["receipts"][0].startswith("v3-")
    assert any("without a measured ttft not considered" in note for note in result.document["notes"])


# -- F3: a missing total or eval duration is unmeasured ---------------------------------------------------------------


@pytest.mark.parametrize("missing", ["total_duration", "eval_duration"])
def test_a_reply_without_total_or_eval_duration_is_unmeasured_and_nothing_is_guessed(missing):
    class Bare(FakeOllama):
        def post(self, path, payload, timeout=None):
            reply = super().post(path, payload, timeout)
            if payload["options"]["num_predict"] > 1:
                reply.pop(missing)
            return reply

    with pytest.raises(LinuxShapeFailure) as caught:
        _measure(Bare(), 512)
    assert caught.value.failure_class == "unmeasured"
    assert "no total_duration or eval_duration" in caught.value.reason


def test_a_verify_call_without_a_total_duration_is_unmeasured():
    class Bare(FakeOllama):
        def post(self, path, payload, timeout=None):
            reply = super().post(path, payload, timeout)
            reply.pop("total_duration")
            return reply

    with pytest.raises(LinuxShapeFailure) as caught:
        _measure(Bare(), 512)
    assert caught.value.failure_class == "unmeasured"
    assert "no total_duration for the verify call" in caught.value.reason


def test_json_round_trip_of_an_oom_row_keeps_the_offload_evidence():
    (row,) = _ladder(_small_card(), (8192,), (4,))
    assert json.loads(json.dumps(row))["settings"]["offload"] == row["settings"]["offload"]
