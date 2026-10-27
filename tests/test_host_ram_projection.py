"""1.0.11: a cell whose projected anonymous host KV is over the VM's limit is refused before its model loads.

The numbers are T4's own, from the Colab receipts (``E:/AI/Publisher/receipts/colab``, windows s1a to s2a2):

- ``tests/fixtures/host_ram_replay.tsv`` is the corrected replay's 234 attributed generate-ladder loads (6 of the 7
  windows have any; s1a2 printed none), cut to the columns the projection reads.
- ``hostram_*.log`` under ``tests/fixtures/ollama_logs/0.40.11`` are four load blocks copied line for line from the
  server logs: windowC2-17 (s2a) lines 8058 to 8261 and 8341 to 8544 (loads 11 and 12), windowC2-14 (s1b) lines 48357 to
  48561 and 48653 to 48857 (loads 44 and 45). Load 12 of s2a was SIGKILLed; load 45 of s1b completed.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import pytest

from interlock.devices import RemoteDeviceIdentity
from interlock.harness import linux_cuda
from interlock.harness.linux_cuda import (
    CGROUP_MEMORY_MAX,
    HOST_RAM_PROJECTION_SAFETY,
    HostRamSafetyError,
    HostRamState,
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
    run_linux_spec,
)
from interlock.receipt import validate_row
from tests.fake_ollama import FakeOllama

FIXTURES = Path(__file__).parent / "fixtures"
LOGS = FIXTURES / "ollama_logs" / "0.40.11"
IDENT = RemoteDeviceIdentity(
    provider="colab", card="T4", vram_mib=15360, driver_version="550.54.15", cuda_version="12.4", pool="us-central1-a"
)
LIMIT = 12.7 * 1024 - 51.2  # 12,953.6: Ollama's "12.7 GiB" less its 0.1 GiB rounding


@pytest.fixture(autouse=True)
def _no_cgroup(monkeypatch, tmp_path):
    """The machine running the tests may be a container with a memory.max of its own."""

    monkeypatch.setattr(linux_cuda, "CGROUP_MEMORY_MAX", tmp_path / "no-such-memory.max")


def _log(name: str) -> str:
    return (LOGS / name).read_text(encoding="utf-8")


def _prev(name: str, *, cell: str = "p8192 c4", num_ctx: int = 8547, slots: int = 4) -> dict[str, Any]:
    prev = load_record(parse_load_report(_log(name)), cell=cell, num_ctx_req=num_ctx, slots_req=slots)
    assert prev is not None
    return prev


# -- the parser ------------------------------------------------------------------------------------------------------


def test_the_parser_keeps_its_keys_and_adds_n_ctx_the_kv_size_and_the_gpu_kv():
    report = parse_load_report(_log("hostram_s2a_c4_34816_partial.log"))
    assert report["n_ctx"] == 34816 and report["kv_total_mib"] == 13056.0 and report["gpu_kv_mib"] == 11016.0
    assert report["cpu_kv_mib"] == 2040.0 and report["layers_on_gpu"] == 28 and report["layers_total"] == 33
    assert report["cpu_mapped_mib"] == pytest.approx(398.33)
    assert report["system_total_mib"] == pytest.approx(12.7 * 1024)
    assert report["system_total_margin_mib"] == 51.2


def test_a_killed_load_printed_no_size_line_so_it_has_no_kv_total_and_no_gpu_kv():
    report = parse_load_report(_log("hostram_s2a_c4_68608_killed.log"))
    assert report["kv_total_mib"] is None and report["gpu_kv_mib"] is None
    assert report["cpu_kv_mib"] == 14472.0 and report["n_ctx"] == 68608


def test_a_load_with_a_size_line_and_no_cuda_buffer_has_zero_gpu_kv_and_a_load_without_either_has_none():
    cpu_only = (
        "load_tensors: loading model tensors\nllama_context: n_ctx = 512\n"
        "llama_kv_cache: size = 96.00 MiB (512 cells)\n"
    )
    assert parse_load_report(cpu_only)["gpu_kv_mib"] == 0.0
    assert parse_load_report("")["gpu_kv_mib"] is None and parse_load_report("")["kv_total_mib"] is None


def test_load_record_needs_a_kv_total_and_a_system_total():
    killed = parse_load_report(_log("hostram_s2a_c4_68608_killed.log"))
    assert load_record(killed, cell="c", num_ctx_req=1, slots_req=1) is None
    assert load_record(parse_load_report(""), cell="c", num_ctx_req=1, slots_req=1) is None
    record = _prev("hostram_s2a_c4_34816_partial.log")
    assert record["req_total"] == 8547 * 4 == 34188 and record["n_ctx"] == 34816
    assert record["kv_total_mib"] == 13056.0 and record["kv_gpu_mib"] == 11016.0
    assert record["system_total_mib"] == pytest.approx(13004.8) and record["margin_mib"] == 51.2


# -- the rule --------------------------------------------------------------------------------------------------------


def test_the_smallest_killed_load_is_refused_with_every_number():
    prev = _prev("hostram_s2a_c4_34816_partial.log")
    row, reason = project_host_ram(prev, num_ctx_next=16903, slots_next=4, factor=1.05)
    assert row["decision"] == "refuse" and row["mode"] == "on" and row["factor"] == 1.05
    assert row["projkv_mib"] == pytest.approx(14804.2, abs=0.1)
    assert row["kv_total_next"] == pytest.approx(25820.2, abs=0.1)
    assert row["host_projected_mib"] == pytest.approx(15544.4, abs=0.1)
    assert row["proj_mapped_mib"] == pytest.approx(15202.5, abs=0.2) and row["mapped_prev"] == pytest.approx(398.33)
    assert row["limit_mib"] == pytest.approx(LIMIT) and row["bound_by"] == "ollama_system_memory"
    assert row["source_load"] == "p8192 c4" and row["req_total_prev"] == 34188 and row["req_total_next"] == 67612
    assert (row["num_ctx_req_prev"], row["slots_req_prev"], row["n_ctx_prev"]) == (8547, 4, 34816)
    assert (row["num_ctx_req_next"], row["slots_req_next"]) == (16903, 4)
    assert (row["kv_total_prev"], row["kv_gpu_prev"], row["system_total_mib"], row["margin_mib"]) == (
        13056.0, 11016.0, pytest.approx(13004.8), 51.2,
    )
    assert row["cgroup_max_mib"] is None
    assert reason == (
        "projected host RAM: 15544 MiB (CPU KV 14804 MiB x 1.05, from p8192 c4 KV 13056 MiB at 34188 -> 67612 "
        "requested tokens less 11016 MiB GPU KV) > limit 12954 MiB (Ollama system memory 13005 MiB less 51.2 MiB "
        "margin)"
    )


def test_the_largest_completed_load_is_allowed_and_the_mapped_weights_do_not_decide():
    prev = _prev("hostram_s1b_c8_69632_partial.log", cell="p8192 c8", num_ctx=8547, slots=8)
    assert prev["req_total"] == 68376
    row, reason = project_host_ram(prev, num_ctx_next=16903, slots_next=8, factor=1.05)
    assert row["decision"] == "allow" and reason is None
    assert row["projkv_mib"] == pytest.approx(12021.2, abs=0.1)
    assert row["host_projected_mib"] == pytest.approx(12622.3, abs=0.1)
    # over the limit, and recorded for information only
    assert row["proj_mapped_mib"] == pytest.approx(14445.3, abs=0.2)
    assert row["proj_mapped_mib"] > row["limit_mib"] > row["host_projected_mib"]


def test_the_factor_decides_between_the_two_boundary_loads():
    refused = _prev("hostram_s2a_c4_34816_partial.log")
    allowed = _prev("hostram_s1b_c8_69632_partial.log", slots=8)
    for factor, a, b in ((1.0, "refuse", "allow"), (1.05, "refuse", "allow"), (1.15, "refuse", "refuse")):
        assert project_host_ram(refused, num_ctx_next=16903, slots_next=4, factor=factor)[0]["decision"] == a
        assert project_host_ram(allowed, num_ctx_next=16903, slots_next=8, factor=factor)[0]["decision"] == b


def test_the_ratio_is_the_requested_one_and_not_the_n_ctx_llama_logged():
    prev = _prev("hostram_s1b_c8_69632_partial.log", slots=8)  # llama logged 69,632; 8,547 x 8 = 68,376 was asked for
    row, _ = project_host_ram(prev, num_ctx_next=16903, slots_next=8, factor=1.0)
    assert row["kv_total_next"] == pytest.approx(9792.0 * 135224 / 68376, abs=0.1)
    assert abs(row["kv_total_next"] - 9792.0 * 135224 / 69632) > 100


def test_a_projection_under_the_gpu_kv_is_zero_not_negative():
    prev = _prev("hostram_s2a_c4_34816_partial.log")
    row, _ = project_host_ram(prev, num_ctx_next=714, slots_next=4, factor=1.05)
    assert row["projkv_mib"] == 0.0 and row["decision"] == "allow"


def test_a_cgroup_limit_under_ollamas_binds_and_is_named():
    prev = _prev("hostram_s1b_c8_69632_partial.log", slots=8)
    row, reason = project_host_ram(prev, num_ctx_next=16903, slots_next=8, factor=1.05, cgroup_max_mib=8192.0)
    assert row["decision"] == "refuse" and row["bound_by"] == "cgroup_memory_max"
    assert row["limit_mib"] == 8192.0 and row["cgroup_max_mib"] == 8192.0
    assert reason is not None and "cgroup memory.max 8192 MiB" in reason and reason.startswith("projected host RAM:")
    roomy, _ = project_host_ram(prev, num_ctx_next=16903, slots_next=8, factor=1.05, cgroup_max_mib=65536.0)
    assert roomy["bound_by"] == "ollama_system_memory" and roomy["limit_mib"] == pytest.approx(LIMIT)
    assert roomy["cgroup_max_mib"] == 65536.0


def test_the_cgroup_file_counts_only_when_it_holds_a_number(tmp_path):
    path = tmp_path / "memory.max"
    path.write_text("8589934592\n", encoding="utf-8")
    assert cgroup_memory_limit_mib(path) == 8192.0
    for text in ("max\n", "", "lots\n", "-1\n", "0\n"):
        path.write_text(text, encoding="utf-8")
        assert cgroup_memory_limit_mib(path) is None
    assert cgroup_memory_limit_mib(tmp_path / "missing") is None  # unreadable
    assert CGROUP_MEMORY_MAX.as_posix() == "/sys/fs/cgroup/memory.max"


# -- the factor and the mode -----------------------------------------------------------------------------------------


def test_the_factor_defaults_to_1_05_takes_an_override_and_refuses_what_is_not_a_positive_number():
    assert parse_host_ram_safety(None) == HOST_RAM_PROJECTION_SAFETY == 1.05
    assert parse_host_ram_safety("1.3") == 1.3 and parse_host_ram_safety(2) == 2.0
    for bad in ("0", "-1", "abc", "nan", "inf", "", 0, -0.5, math.nan, math.inf, True):
        with pytest.raises(HostRamSafetyError) as caught:
            parse_host_ram_safety(bad)
        error = caught.value
        assert isinstance(error, LinuxCudaError)
        assert error.code == "host_ram_safety_invalid" and "greater than 0" in error.message and error.hint
        assert str(error).startswith("host_ram_safety_invalid:") and "hint:" in str(error)


def test_the_state_refuses_an_unknown_mode():
    assert HostRamState().mode == "on" and HostRamState(mode="off").mode == "off"
    with pytest.raises(LinuxCudaError):
        HostRamState(mode="maybe")


# -- the replay over the seven T4 windows ----------------------------------------------------------------------------


def _replay() -> list[dict[str, Any]]:
    with (FIXTURES / "host_ram_replay.tsv").open(encoding="utf-8") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    rows = list(csv.DictReader(lines, delimiter="\t"))
    prev_by_run: dict[tuple[str, str], dict[str, Any]] = {}
    out = []
    for row in rows:
        slots, req_total = int(row["slots"]), int(row["req_total"])
        key = (row["window"], row["run"])
        prev = prev_by_run.get(key)
        verdict = None
        if prev is not None:
            verdict, _ = project_host_ram(prev, num_ctx_next=req_total // slots, slots_next=slots, factor=1.05)
        out.append({**row, "verdict": verdict})
        if row["kv"] != "None":  # only a load that printed its size line becomes the run's previous load
            limit = float(row["limit"])
            prev_by_run[key] = {
                "cell": row["cell"], "num_ctx_req": req_total // slots, "slots_req": slots, "req_total": req_total,
                "n_ctx": int(row["n_ctx"]), "kv_total_mib": float(row["kv"]), "kv_gpu_mib": float(row["gpu_kv"]),
                "mapped_mib": float(row["mapped"]), "system_total_mib": limit + 51.2, "margin_mib": 51.2,
            }
    return out


def test_the_replay_refuses_every_kill_and_no_completed_load():
    rows = _replay()
    completed = [r for r in rows if r["status"] != "killed" and r["verdict"] is not None]
    killed = [r for r in rows if r["status"] == "killed"]
    assert len(rows) == 234 and len(completed) == 188 and len(killed) == 34
    assert all(r["verdict"] is not None for r in killed)  # no kill is the first load of its run
    assert [r for r in completed if r["verdict"]["decision"] == "refuse"] == []
    assert [r for r in killed if r["verdict"]["decision"] != "refuse"] == []
    assert sum(1 for r in rows if r["verdict"] is None) == 12  # the first load of each run is never projected


def test_the_replay_boundary_loads_are_where_the_design_says():
    rows = {(r["window"], r["load"]): r for r in _replay()}
    top = rows[("s1b", "45")]
    assert top["run"].endswith("qwen3-8b-Q8_0") and top["cell"] == "p16384 c8" and top["status"] == "partial"
    assert top["verdict"]["projkv_mib"] == pytest.approx(12021.2, abs=0.1) and top["verdict"]["decision"] == "allow"
    second = rows[("s1b", "39")]
    assert second["cell"] == "p32768 c4" and second["verdict"]["projkv_mib"] == pytest.approx(11951.0, abs=0.1)
    assert second["verdict"]["decision"] == "allow"
    for window in ("s1a", "s2a", "s2a2"):
        low = rows[(window, "12")]
        assert low["status"] == "killed" and low["cell"] == "p16384 c4" and low["run"].endswith("phi3.5-3.8b-Q4_K_M")
        assert low["verdict"]["projkv_mib"] == pytest.approx(14804.2, abs=0.1)
        assert low["verdict"]["decision"] == "refuse"
    projected = [r["verdict"]["projkv_mib"] for r in rows.values() if r["verdict"] is not None]
    done = max(r["verdict"]["projkv_mib"] for r in rows.values() if r["verdict"] and r["status"] != "killed")
    kills = min(r["verdict"]["projkv_mib"] for r in rows.values() if r["verdict"] and r["status"] == "killed")
    assert (round(done, 1), round(kills, 1)) == (12021.2, 14804.2) and len(projected) == 222
    assert done * 1.05 < LIMIT < kills * 1.05  # the band: 12,337 to 12,954 MiB is where a false refusal would sit


# -- the ladder, against the faithful fake ---------------------------------------------------------------------------


def _fake(**kw: Any) -> FakeOllama:
    # 4,000 MiB of VRAM past the weights, and the T4's 12.7 GiB of host memory.
    return FakeOllama(vram_mib=6400.0, weights_mib=2400.0, system_total_mib=13004.8, **kw)


def _ladder(
    server: FakeOllama, state: HostRamState, lengths: tuple[int, ...], concurrencies: tuple[int, ...]
) -> list[dict]:
    def measure(length: int, concurrency: int) -> dict[str, Any]:
        return measure_linux_ollama_generate(
            server, lambda: (1000, 15360), model="m", prompt_tokens=length, concurrency=concurrency, repeats=1,
            clock=server.clock, host_ram=state,
        )

    return run_linux_generate_ladder(
        spec_id="C2-GENERATE", spec_rel="experiments/c2-generate/spec.yaml", run_id="t4-x", date="2026-10-27",
        remote_device=IDENT, model="m", revision_or_digest="d" * 64, measure_shape=measure, prompt_lengths=lengths,
        concurrencies=concurrencies, quant="Q4_K_M", licence="apache-2.0", host_ram=state,
    )


def test_a_refusal_sequence_starts_no_load_sends_no_request_and_survives_a_concurrency_change():
    server, state = _fake(), HostRamState()
    first = _ladder(server, state, (8192, 16384, 32768), (4,))
    # p8192 c4 spills to host RAM (a partial offload, as T4 s2a's load 11) and the next two are refused.
    assert [r.get("outcome") for r in first] == ["oom", "oom", "oom"]
    assert first[0]["settings"]["failure_reason"].startswith("partial offload")
    assert first[0]["settings"]["host_ram_projection"]["decision"] == "skipped"  # the run's first load
    kills, calls = server.kills, len(server.all_calls)
    for row, length in zip(first[1:], (16384, 32768), strict=True):
        settings = row["settings"]
        assert settings["failure_reason"].startswith("projected host RAM:") and "peak_mib" not in settings
        assert settings["num_ctx_source"] == "nominal: no request was sent"
        projection = settings["host_ram_projection"]
        assert projection["decision"] == "refuse" and projection["source_load"] == "p8192 c4"  # p32768 still from p8192
        assert projection["req_total_next"] == ladder_window(length, 128) * 4
    assert server.kills == kills == 0 and len(server.all_calls) == calls  # the refusals sent nothing
    assert state.prev_load["cell"] == "p8192 c4"  # a refused cell does not replace the last load
    restarts = server.restarts
    second = _ladder(server, state, (512,), (8,))
    assert server.restarts == restarts + 1  # ensure_parallel(8) ran for the cell that was allowed
    assert second[0].get("outcome") is None
    assert second[0]["settings"]["host_ram_projection"]["decision"] == "allow"
    assert second[0]["settings"]["host_ram_projection"]["source_load"] == "p8192 c4"
    for row in (*first, *second):
        validate_row(row)
        assert row["settings"]["host_ram_projection"]["mode"] == "on"
        assert row["settings"]["ladder_version"] == 4


def test_a_killed_cell_does_not_replace_the_last_load():
    server = _fake()
    state = HostRamState(factor=0.01)  # a factor that discounts every projection under the limit: nothing is refused
    rows = _ladder(server, state, (512, 8192, 65536), (4,))
    assert [r["settings"]["host_ram_projection"]["decision"] for r in rows] == ["skipped", "allow", "allow"]
    assert rows[2]["settings"]["failure_reason"].startswith("host RAM: the load needed")  # a real kill, 1.0.10's words
    assert server.kills == 1
    assert state.prev_load["cell"] == "p8192 c4"  # the killed cell printed no size line, so it is not the last load


def test_a_real_kill_and_a_refusal_say_different_things():
    server, state = _fake(), HostRamState(factor=0.01)
    killed = _ladder(server, state, (512, 65536), (4,))[1]
    refused = _ladder(_fake(), HostRamState(), (8192, 16384), (4,))[1]
    assert killed["settings"]["failure_reason"].split(":")[0] == "host RAM"
    assert refused["settings"]["failure_reason"].split(":")[0] == "projected host RAM"
    assert killed["settings"]["failure_class"] == refused["settings"]["failure_class"] == "oom"
    # the projection allowed it; the kill was real
    assert killed["settings"]["host_ram_projection"]["decision"] == "allow"
    assert "peak_mib" in killed["settings"] and "peak_mib" not in refused["settings"]


def test_a_second_model_starts_without_the_first_models_load():
    spec = {
        "id": "C2-GENERATE", "procedure": "generate-ladder", "platform": "linux-cuda", "prompt_lengths": [512, 1024],
        "concurrencies": [1], "models": [
            {"model": "a", "quant": "Q4_K_M", "device": "colab:T4"},
            {"model": "b", "quant": "Q4_K_M", "device": "colab:T4"},
        ],
    }
    rows = run_linux_spec(
        spec, spec_path=Path("experiments/c2-generate/spec.yaml"), spec_rel="experiments/c2-generate/spec.yaml",
        run_id="t4-x", date="2026-10-27", remote_device=IDENT, sample_vram_mib=lambda: (1000, 15360),
        server=_fake(),  # type: ignore[arg-type]
    )
    decisions = [(r["model"]["repo_or_tag"], r["settings"]["host_ram_projection"]["decision"]) for r in rows]
    assert decisions == [("a", "skipped"), ("a", "allow"), ("b", "skipped"), ("b", "allow")]
    assert rows[2]["settings"]["host_ram_projection"]["skipped"] == "first load of run"
    for row in rows:
        validate_row(row)


def test_a_pre_registered_refusal_row_says_no_load_was_attempted():
    spec = {
        "id": "C2-GENERATE", "procedure": "generate-ladder", "platform": "linux-cuda", "prompt_lengths": [512],
        "concurrencies": [1], "models": [
            {"model": "big", "quant": "Q8_0", "device": "colab:T4", "expected_outcome": "load_refused"}
        ],
    }
    (row,) = run_linux_spec(
        spec, spec_path=Path("experiments/c2-generate/spec.yaml"), spec_rel="experiments/c2-generate/spec.yaml",
        run_id="t4-x", date="2026-10-27", remote_device=IDENT, sample_vram_mib=lambda: (1000, 15360),
        server=_fake(),  # type: ignore[arg-type]
    )
    projection = row["settings"]["host_ram_projection"]
    assert projection["mode"] == "on" and projection["decision"] == "skipped" and "no load" in projection["skipped"]
    validate_row(row)


def test_off_matches_1_0_10_rows_except_for_the_mode_dict():
    on_rows = _ladder(_fake(), HostRamState(), (512, 2048), (1,))
    off_rows = _ladder(_fake(), HostRamState(mode="off"), (512, 2048), (1,))
    assert [r["settings"]["host_ram_projection"] for r in off_rows] == [{"mode": "off"}, {"mode": "off"}]
    for on, off in zip(on_rows, off_rows, strict=True):
        a, b = dict(on["settings"]), dict(off["settings"])
        a.pop("host_ram_projection"), b.pop("host_ram_projection")
        assert a == b


def test_off_does_not_refuse_a_cell_the_projection_would_have():
    on = _ladder(_fake(), HostRamState(), (8192, 16384), (4,))
    off = _ladder(_fake(), HostRamState(mode="off"), (8192, 16384), (4,))
    assert on[1]["settings"]["failure_reason"].startswith("projected host RAM:")
    assert not off[1]["settings"]["failure_reason"].startswith("projected host RAM:")


def test_an_override_of_the_factor_changes_a_decision_and_is_recorded():
    default = _ladder(_fake(), HostRamState(), (8192, 16384), (4,))[1]["settings"]["host_ram_projection"]
    lenient = _ladder(_fake(), HostRamState(factor=0.1), (8192, 16384), (4,))[1]["settings"]["host_ram_projection"]
    assert (default["decision"], default["factor"]) == ("refuse", 1.05)
    assert (lenient["decision"], lenient["factor"]) == ("allow", 0.1)


def test_a_cgroup_limit_read_from_the_file_binds_a_cell(tmp_path):
    path = tmp_path / "memory.max"
    path.write_text(str(6 * 1024 * 1024 * 1024), encoding="utf-8")
    server, state = _fake(), HostRamState()

    def cell(**kw: Any) -> None:
        measure_linux_ollama_generate(
            server, lambda: (1000, 15360), model="m", prompt_tokens=8192, concurrency=4, repeats=1,
            clock=server.clock, host_ram=state, **kw,
        )

    with pytest.raises(LinuxShapeFailure, match="partial offload"):
        cell()  # the first load: 8,820 MiB of KV in host RAM, which fits the VM
    with pytest.raises(LinuxShapeFailure) as caught:
        cell(cgroup_path=path)  # the same cell again, under a 6 GiB cgroup
    projection = caught.value.evidence["host_ram_projection"]
    assert projection["decision"] == "refuse" and projection["bound_by"] == "cgroup_memory_max"
    assert projection["cgroup_max_mib"] == 6144.0 and projection["limit_mib"] == 6144.0
    path.write_text("max", encoding="utf-8")
    with pytest.raises(LinuxShapeFailure, match="partial offload"):
        cell(cgroup_path=path)  # "max" is ignored: the projection is under Ollama's own limit


def test_the_factor_and_mode_reach_run_linux_spec():
    spec = {
        "id": "C2-GENERATE", "procedure": "generate-ladder", "platform": "linux-cuda", "prompt_lengths": [512, 2048],
        "concurrencies": [1], "models": [{"model": "a", "quant": "Q4_K_M", "device": "colab:T4"}],
    }

    def run(**kw: Any) -> list[dict]:
        return run_linux_spec(
            spec, spec_path=Path("experiments/c2-generate/spec.yaml"), spec_rel="experiments/c2-generate/spec.yaml",
            run_id="t4-x", date="2026-10-27", remote_device=IDENT, sample_vram_mib=lambda: (1000, 15360),
            server=_fake(), **kw,  # type: ignore[arg-type]
        )

    assert [r["settings"]["host_ram_projection"]["decision"] for r in run()] == ["skipped", "allow"]
    assert run(host_ram_safety=1.3)[1]["settings"]["host_ram_projection"]["factor"] == 1.3
    assert [r["settings"]["host_ram_projection"] for r in run(host_ram_projection="off")] == [{"mode": "off"}] * 2
    with pytest.raises(HostRamSafetyError):
        run(host_ram_safety="nan")
    with pytest.raises(LinuxCudaError):
        run(host_ram_projection="sometimes")


def test_the_module_still_exposes_the_failure_classifier_unchanged():
    assert callable(linux_cuda._failure_for)
