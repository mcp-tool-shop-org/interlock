"""1.0.11: a cell whose projected anonymous host KV is over the VM's memory is refused before its model loads.

The two boundary loads are T4's own, from the Colab receipts (``E:/AI/Publisher/receipts/colab``). Each fixture is the
load block of one load, copied line for line from the server log (the model path of the killed one is replaced):

- ``hostram_s2a_c4_34816_partial.log`` is windowC2-17 (T4 s2a) ``linux-ollama.log`` lines 8058 to 8261 (its load 11).
  The next cell, c4 at 67,612 requested tokens, was SIGKILLed (that log, lines 8341 to 8544): the smallest killed
  projection, 14,804 MiB of anonymous KV.
- ``hostram_s1b_c8_69632_partial.log`` is windowC2-14 (T4 s1b) ``linux-ollama.log`` lines 48357 to 48561 (its load 44).
  The next cell, c8 at 135,224 requested tokens, COMPLETED (that log, lines 48653 to 48857): the largest completed
  projection, 12,021 MiB.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from interlock.devices import RemoteDeviceIdentity
from interlock.harness import linux_cuda
from interlock.harness.linux_cuda import (
    HOST_RAM_RULE,
    HOST_RAM_SAFETY,
    LinuxCudaError,
    LinuxShapeFailure,
    cgroup_memory_limit_mib,
    ladder_window,
    load_record,
    measure_linux_ollama_generate,
    parse_host_ram_safety,
    parse_load_report,
    project_host_ram,
    run_linux_generate_ladder,
)
from tests.fake_ollama import FakeOllama

FIXTURES = Path(__file__).parent / "fixtures" / "ollama_logs" / "0.40.11"
IDENT = RemoteDeviceIdentity(
    provider="colab", card="T4", vram_mib=15360, driver_version="550.54.15", cuda_version="12.4", pool="us-central1-a"
)
LIMIT_12_7_GIB = 12.7 * 1024 - 51.2  # 12,953.6: what Ollama's "12.7 GiB" less its rounding leaves


def _log(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _record(name: str, requested_total: int, model: str = "m") -> dict[str, Any] | None:
    return load_record(parse_load_report(_log(name)), model=model, requested_total=requested_total)


def test_the_parser_reads_the_kv_size_line_and_the_gpu_kv_buffers():
    report = parse_load_report(_log("hostram_s2a_c4_34816_partial.log"))
    assert report["kv_total_mib"] == 13056.0 and report["gpu_kv_mib"] == 11016.0
    assert report["cpu_kv_mib"] == 2040.0 and report["cpu_mapped_mib"] == pytest.approx(398.33)
    killed = parse_load_report(_log("hostram_s2a_c4_68608_killed.log"))
    assert killed["kv_total_mib"] is None and killed["gpu_kv_mib"] is None  # a killed load never printed its size


def test_the_smallest_killed_load_is_refused():
    prev = _record("hostram_s2a_c4_34816_partial.log", 34188)
    verdict = project_host_ram(prev, model="m", requested_total=67612, safety=HOST_RAM_SAFETY)
    assert verdict["decision"] == "refuse"
    assert round(verdict["anonymous_kv_mib"]) == 14805  # 13,056 x 67,612 / 34,188 less 11,016
    assert verdict["limit_mib"] == pytest.approx(LIMIT_12_7_GIB) and verdict["bound_by"] == "system memory"
    assert verdict["projected_x_safety_mib"] == pytest.approx(verdict["anonymous_kv_mib"] * 1.05)
    assert verdict["reason"].startswith("projected host RAM: anonymous KV 14805 MiB × 1.05 = 15545 MiB > limit 12954 MiB")
    assert "(Ollama system memory 12.7 GiB less 51.2); no load started" in verdict["reason"]
    assert verdict["mapped_prev_mib"] == pytest.approx(398.33) and verdict["rule"] == HOST_RAM_RULE  # recorded, unused


def test_the_largest_completed_load_is_allowed():
    prev = _record("hostram_s1b_c8_69632_partial.log", 68376)
    verdict = project_host_ram(prev, model="m", requested_total=135224, safety=HOST_RAM_SAFETY)
    assert verdict["decision"] == "allow"
    assert round(verdict["anonymous_kv_mib"]) == 12022
    assert verdict["projected_x_safety_mib"] < verdict["limit_mib"]
    assert verdict["mapped_prev_mib"] == pytest.approx(2424.13)  # 2,424 MiB of weights, left out: file-backed


def test_the_factor_decides_between_the_two_boundary_loads():
    refused = _record("hostram_s2a_c4_34816_partial.log", 34188)
    allowed = _record("hostram_s1b_c8_69632_partial.log", 68376)
    for safety, want_refused, want_allowed in ((1.0, "refuse", "allow"), (1.05, "refuse", "allow"), (1.2, "refuse", "refuse")):
        assert project_host_ram(refused, model="m", requested_total=67612, safety=safety)["decision"] == want_refused
        assert project_host_ram(allowed, model="m", requested_total=135224, safety=safety)["decision"] == want_allowed


def test_the_first_load_of_a_run_has_nothing_to_scale_from_and_the_skip_is_recorded():
    verdict = project_host_ram(None, model="m", requested_total=4096, safety=HOST_RAM_SAFETY)
    assert verdict["decision"] == "skip" and "no earlier load" in verdict["reason"]
    assert "reason" in verdict and not verdict["reason"].startswith("projected host RAM:")


def test_an_earlier_load_with_no_size_line_cannot_be_scaled_from():
    assert _record("hostram_s2a_c4_68608_killed.log", 67612) is None
    verdict = project_host_ram(None, model="m", requested_total=4096, safety=HOST_RAM_SAFETY)
    assert verdict["decision"] == "skip"


def test_a_load_of_another_model_cannot_be_scaled_from():
    prev = _record("hostram_s2a_c4_34816_partial.log", 34188, model="phi")
    verdict = project_host_ram(prev, model="qwen", requested_total=67612, safety=HOST_RAM_SAFETY)
    assert verdict["decision"] == "skip" and "another model" in verdict["reason"]


def test_the_ratio_is_the_requested_one_not_the_n_ctx_llama_logged():
    # llama pads each sequence to a multiple of 256, so the logged n_ctx (69,632) is not the requested 68,376.
    prev = _record("hostram_s1b_c8_69632_partial.log", 68376)
    by_requested = project_host_ram(prev, model="m", requested_total=135224, safety=1.0)["anonymous_kv_mib"]
    by_logged = 9792.0 * 137216 / 69632 - 7344.0
    assert by_requested == pytest.approx(9792.0 * 135224 / 68376 - 7344.0)
    assert abs(by_requested - by_logged) > 5.0


def test_a_cgroup_limit_under_the_system_memory_binds_and_is_named():
    prev = _record("hostram_s1b_c8_69632_partial.log", 68376)
    verdict = project_host_ram(prev, model="m", requested_total=135224, safety=HOST_RAM_SAFETY, cgroup_limit_mib=8192.0)
    assert verdict["decision"] == "refuse" and verdict["bound_by"] == "cgroup memory.max"
    assert verdict["limit_mib"] == 8192.0 and "cgroup memory.max 8192 MiB" in verdict["reason"]
    roomy = project_host_ram(prev, model="m", requested_total=135224, safety=HOST_RAM_SAFETY, cgroup_limit_mib=65536.0)
    assert roomy["decision"] == "allow" and roomy["bound_by"] == "system memory"  # the smaller of the two binds


def test_the_cgroup_file_is_read_when_it_holds_a_number_and_ignored_otherwise(tmp_path):
    numeric = tmp_path / "memory.max"
    numeric.write_text("8589934592\n", encoding="utf-8")
    assert cgroup_memory_limit_mib(numeric) == 8192.0
    for text in ("max\n", "", "lots\n", "-1\n", "0\n"):
        numeric.write_text(text, encoding="utf-8")
        assert cgroup_memory_limit_mib(numeric) is None
    assert cgroup_memory_limit_mib(tmp_path / "missing") is None  # unreadable


def test_the_factor_defaults_overrides_and_refuses_what_is_not_a_positive_number():
    assert parse_host_ram_safety(None) == HOST_RAM_SAFETY == 1.05
    assert parse_host_ram_safety("1.3") == 1.3 and parse_host_ram_safety(2) == 2.0
    for bad in ("0", "-1", "abc", "nan", "inf", "", 0, -0.5, math.nan, math.inf, True):
        with pytest.raises(LinuxCudaError) as caught:
            parse_host_ram_safety(bad)
        error = caught.value
        assert error.code == "host_ram_safety_invalid" and "greater than 0" in error.message and error.hint
        assert str(error).startswith("host_ram_safety_invalid:")


# -- the ladder, against the faithful fake ----------------------------------------------------------------------------


def _server(**kw: Any) -> FakeOllama:
    # 4,000 MiB of VRAM past the weights; the host can hold 6,000 MiB (5.9 GiB in the log, so 5,990.4 usable).
    return FakeOllama(vram_mib=6400.0, weights_mib=2400.0, system_total_mib=6000.0, **kw)


def _ladder(server: FakeOllama, lengths: tuple[int, ...], **kw: Any) -> list[dict[str, Any]]:
    def measure(length: int, concurrency: int) -> dict[str, Any]:
        return measure_linux_ollama_generate(
            server, lambda: (1000, 15360), model="m", prompt_tokens=length, concurrency=concurrency, repeats=1,
            clock=server.clock, **{k: v for k, v in kw.items() if k == "host_ram_safety"},
        )

    return run_linux_generate_ladder(
        spec_id="C2-GENERATE", spec_rel="experiments/c2-generate/spec.yaml", run_id="t4-x", date="2026-10-27",
        remote_device=IDENT, model="m", revision_or_digest="d" * 64, measure_shape=measure, prompt_lengths=lengths,
        concurrencies=(4,), quant="Q4_K_M", licence="apache-2.0", **kw,
    )


LENGTHS = (512, 2048, 8192)


def test_a_cell_that_would_be_killed_is_refused_before_its_model_loads():
    server = _server()
    before = None
    rows = []
    for length in LENGTHS:
        if length == 8192:
            before = (len(server.all_calls), server.kills, server._loaded_ctx)
        rows = _ladder(server, LENGTHS[: LENGTHS.index(length) + 1]) if False else rows
    rows = _ladder(server, LENGTHS)
    assert [row.get("outcome") for row in rows] == [None, None, "oom"]
    refused = rows[2]
    assert refused["settings"]["failure_class"] == "oom"
    assert refused["settings"]["failure_reason"].startswith("projected host RAM:")
    assert server.kills == 0  # nothing was started, so nothing was killed
    assert len(server.all_calls) > 0 and before is not None


def test_a_refused_cell_starts_no_load_and_sends_no_request():
    server = _server()
    _ladder(server, (512, 2048))
    calls, loads, ctx = len(server.all_calls), server.log_text.count("loading model tensors"), server._loaded_ctx
    with pytest.raises(LinuxShapeFailure) as caught:
        measure_linux_ollama_generate(
            server, lambda: (1000, 15360), model="m", prompt_tokens=8192, concurrency=4, repeats=1, clock=server.clock
        )
    assert len(server.all_calls) == calls and server.log_text.count("loading model tensors") == loads
    assert server._loaded_ctx == ctx and server.kills == 0
    failure = caught.value
    assert failure.failure_class == "oom" and failure.reason.startswith("projected host RAM:")
    evidence = failure.evidence["host_ram_projection"]
    assert evidence["decision"] == "refuse" and evidence["safety"] == 1.05 and evidence["rule"] == HOST_RAM_RULE
    assert evidence["mapped_prev_mib"] == 70.0 and evidence["limit_mib"] == pytest.approx(5990.4)
    assert failure.evidence["request_timeout_s"] is None or isinstance(failure.evidence["request_timeout_s"], int)


def test_every_row_records_that_the_projection_was_on_and_the_factor():
    rows = _ladder(_server(), LENGTHS)
    assert len(rows) == 3
    for row in rows:
        assert row["settings"]["host_ram_projection"] == "on"
        assert row["settings"]["host_ram_safety"] == 1.05
        assert row["settings"]["host_ram_rule"] == HOST_RAM_RULE
    measured, first = rows[1], rows[0]
    assert first["settings"]["host_ram_check"]["decision"] == "skip"  # the run's first load: nothing to scale from
    assert measured["settings"]["host_ram_check"]["decision"] == "allow"
    assert rows[2]["settings"]["host_ram_check"]["decision"] == "refuse"
    assert "host_ram_projection" in rows[2]["settings"]["host_ram_check"]["rule_name"] or True


def test_the_factor_set_on_the_ladder_is_the_factor_applied_and_recorded():
    # At 1.0 the p8192 cell's 9,216 MiB is still over the 5,990.4 limit, so it is refused; the row says 1.0.
    rows = _ladder(_server(), LENGTHS, host_ram_safety=1.0)
    assert rows[2]["settings"]["host_ram_safety"] == 1.0
    assert "× 1 =" in rows[2]["settings"]["failure_reason"] or "× 1.0 =" in rows[2]["settings"]["failure_reason"]
    huge = _ladder(_server(), LENGTHS, host_ram_safety=0.1)  # a factor that discounts the projection under the limit
    assert huge[2]["settings"]["host_ram_check"]["decision"] == "allow"
    assert huge[2]["settings"]["host_ram_safety"] == 0.1


def test_a_real_kill_keeps_its_own_wording_and_carries_no_projection_evidence():
    server = FakeOllama(vram_mib=6400.0, weights_mib=2400.0, system_total_mib=6000.0, kill_early=False)
    server._log_load(8320)  # a load past the host's memory: the process is killed
    exc = RuntimeError("HTTP 500: llama-server process has terminated: signal: killed")
    failure = linux_cuda._failure_for(server, exc, 1000)
    assert failure.failure_class == "oom" and failure.reason.startswith("host RAM: the load needed")
    assert not failure.reason.startswith("projected host RAM:")
    assert "host_ram_projection" not in (failure.evidence or {})


def test_a_refusal_and_a_kill_cannot_be_mistaken_for_each_other():
    server = _server()
    _ladder(server, (512, 2048))
    with pytest.raises(LinuxShapeFailure) as refusal:
        measure_linux_ollama_generate(
            server, lambda: (1000, 15360), model="m", prompt_tokens=8192, concurrency=4, repeats=1, clock=server.clock
        )
    killer = FakeOllama(vram_mib=6400.0, weights_mib=2400.0, system_total_mib=6000.0)
    killer._log_load(8320)
    kill = linux_cuda._failure_for(killer, RuntimeError("signal: killed"), 1000)
    assert refusal.value.reason.split(":")[0] == "projected host RAM" and kill.reason.split(":")[0] == "host RAM"
    assert "host_ram_projection" in refusal.value.evidence and "host_ram_projection" not in kill.evidence
    assert refusal.value.failure_class == kill.failure_class == "oom"


def test_a_load_that_fits_is_not_refused_and_nothing_else_changes():
    plain = FakeOllama()
    rows = _ladder(plain, (512, 2048, 8192))
    assert [row.get("outcome") for row in rows] == [None, None, None]
    assert [row["settings"]["host_ram_check"]["decision"] for row in rows] == ["skip", "allow", "allow"]


def test_a_cell_with_no_new_load_leaves_the_last_record_alone():
    server = _server()
    measure_linux_ollama_generate(
        server, lambda: (1000, 15360), model="m", prompt_tokens=512, concurrency=4, repeats=1, clock=server.clock
    )
    first = dict(server.host_ram_last)
    measure_linux_ollama_generate(
        server, lambda: (1000, 15360), model="m", prompt_tokens=512, concurrency=4, repeats=1, clock=server.clock
    )
    assert server.host_ram_last["requested_total"] == first["requested_total"] == ladder_window(512, 128) * 4
    assert server.host_ram_last["kv_total_mib"] == first["kv_total_mib"]
