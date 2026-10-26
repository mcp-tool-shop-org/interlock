"""Ladder v3: a real prefill, real concurrency, and a router that never reads what earlier ladders could not measure.

The fake in ``tests/fake_ollama.py`` caches the longest common prompt prefix, queues requests unless it has the slots,
and truncates the way Ollama does (T4 r2, Ollama v0.40.11).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from interlock.cli import main
from interlock.devices import RemoteDeviceIdentity
from interlock.harness import linux_cuda
from interlock.harness.linux_cuda import (
    GENERATE_LADDER_VERSION,
    LinuxOllamaServer,
    LinuxShapeFailure,
    measure_linux_ollama_generate,
    run_linux_generate_ladder,
)
from interlock.receipt import validate_row
from interlock.router.policy import Decision, Refusal, Request, decide, throughput_measured, ttft_measured
from tests.fake_ollama import FakeOllama

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


def test_the_version_is_3():
    assert GENERATE_LADDER_VERSION == 3


# -- A: the window is never overshot --------------------------------------------------------------------------------


@pytest.mark.parametrize("tokens_per_word", [1.0, 1.5, 2.0])
@pytest.mark.parametrize("target", [16384, 32768])
def test_a_prompt_that_tokenises_over_the_probes_still_fits_its_window_at_16k_and_32k(target, tokens_per_word):
    server = FakeOllama(tokens_per_word=tokens_per_word, long_prompt_bias=0.012)  # v2's window was target + 192
    result = _measure(server, target)
    assert abs(result["prompt_tokens_evaluated"] - target) <= target * 0.02
    window = result["num_ctx_sent"]
    assert set(server.contexts) == {window}
    assert max(server.evaluated_counts) <= window - 2  # Ollama truncates at num_ctx - 1; nothing reached it
    assert window >= target + 0.02 * target + 128  # sized from the accepted prompt, not a fixed 192


# -- B: every timed prefill is real ----------------------------------------------------------------------------------


def test_the_fake_really_answers_a_repeated_prompt_from_its_cache():
    server = FakeOllama()
    payload = {"prompt": "river stone market " * 100, "options": {"num_ctx": 4096, "num_predict": 8}}
    first = server.post("/api/generate", payload)
    second = server.post("/api/generate", payload)
    assert first["prompt_eval_count"] == second["prompt_eval_count"] == 300
    assert second["prompt_eval_duration"] < first["prompt_eval_duration"] / 100


def test_every_timed_call_starts_with_a_prefix_of_its_own_so_the_cache_cannot_answer():
    server = FakeOllama()
    result = _measure(server, 2048, repeats=3)
    firsts = [prompt.split()[0] for prompt in server.prompts]
    assert len(firsts) == 4 and len(set(firsts)) == 4  # no two calls share even their first word
    expected_ms = 2048 / server.prefill_tok_s * 1000
    for slot in result["slots"]:
        assert slot["ttft_ms"] == pytest.approx(expected_ms, rel=0.05)  # a full prefill, warmup and repeats alike


def test_concurrent_workers_in_one_pass_get_different_prefixes():
    server = FakeOllama()
    _measure(server, 1024, concurrency=4, repeats=1)
    firsts = [prompt.split()[0] for prompt in server.prompts]
    assert len(firsts) == 8 and len(set(firsts)) == 8


def test_the_calibration_verify_call_differs_from_every_timed_call():
    server = FakeOllama()
    seen: list[str] = []
    original = server.post

    def post(path, payload):
        seen.append(payload["prompt"].split()[0])
        return original(path, payload)

    server.post = post
    _measure(server, 1024, repeats=2)
    assert len(seen) == len(set(seen))  # probes, the verify call and every timed call each start differently


def test_a_cell_whose_prefill_came_from_the_cache_is_unmeasured(monkeypatch):
    monkeypatch.setattr(linux_cuda, "ladder_prefix", lambda nonce, call: "0001abcd ")  # the v2 behaviour: identical
    server = FakeOllama()
    with pytest.raises(LinuxShapeFailure) as caught:
        _measure(server, 2048)
    assert caught.value.failure_class == "unmeasured"
    assert "prompt came from llama-server's cache" in caught.value.reason


def test_ttft_rises_with_prompt_length_across_the_ladder():
    server = FakeOllama()
    rows = run_linux_generate_ladder(
        spec_id="C2-GENERATE", spec_rel="experiments/c2-generate/spec.yaml", run_id="t4-r3", date="2026-10-26",
        remote_device=IDENT, model="phi3.5:3.8b", revision_or_digest="d" * 64,
        measure_shape=lambda length, conc: _measure(server, length, conc),
        prompt_lengths=(512, 1024, 2048, 4096), concurrencies=(1,), quant="Q4_K_M", licence="mit",
    )
    ttfts = [row["settings"]["ttft_ms"] for row in rows]
    assert all(row.get("outcome") is None for row in rows)
    assert ttfts == sorted(ttfts) and ttfts[-1] > 6 * ttfts[0]  # about 8x for 8x the tokens
    assert {row["settings"]["ladder_version"] for row in rows} == {3}
    assert {row["settings"]["prefix_caching"] for row in rows} == {"off"}
    for row in rows:
        validate_row(row)


def test_a_ladder_cell_whose_ttft_did_not_grow_with_the_prompt_is_unmeasured():
    def flat(length: int, concurrency: int) -> dict:  # what v2 recorded: a cache hit at every length
        return {"ttft_ms": 15.0, "decode_tok_s": 30.0, "peak_mib": 6000, "size": 5, "size_vram": 5,
                "prompt_tokens_evaluated": length}

    rows = run_linux_generate_ladder(
        spec_id="C2-GENERATE", spec_rel="experiments/c2-generate/spec.yaml", run_id="t4-r3", date="2026-10-26",
        remote_device=IDENT, model="phi3.5:3.8b", revision_or_digest="d" * 64, measure_shape=flat,
        prompt_lengths=(512, 2048, 8192), concurrencies=(1,), quant="Q4_K_M", licence="mit",
    )
    assert [row.get("outcome") for row in rows] == [None, "unmeasured", "unmeasured"]
    assert "did not scale with the prompt" in rows[1]["notes"]
    assert all(row["settings"]["ladder_version"] == 3 for row in rows)


# -- C: real concurrency, wall-clock throughput ----------------------------------------------------------------------


def test_with_one_slot_the_requests_queue_and_the_wall_clock_aggregate_is_about_one_stream():
    single = _measure(FakeOllama(), 512, concurrency=1)["decode_tok_s"]
    queued = FakeOllama(honour_parallel=False, report_slots=False)  # a server that ignores the slot count
    aggregate = _measure(queued, 512, concurrency=4)["decode_tok_s"]
    assert aggregate == pytest.approx(single, rel=0.1)  # v2 reported about 4x this


def test_with_a_slot_per_request_the_aggregate_is_about_the_concurrency_times_one_stream():
    single = _measure(FakeOllama(), 512, concurrency=1)["decode_tok_s"]
    result = _measure(FakeOllama(), 512, concurrency=4)
    assert result["decode_tok_s"] == pytest.approx(4 * single, rel=0.1)
    assert result["parallel_slots"] == 4
    assert result["parallel_slots_source"] == "llama-server log"
    assert result["throughput_method"] == "ollama_durations_decode_window"
    assert result["per_request_decode_tok_s"] == pytest.approx(50.0, rel=0.01)  # a separate field, per stream


@pytest.mark.parametrize("target", [512, 2048, 8192, 32768])
def test_at_concurrency_1_the_decode_aggregate_is_the_per_request_decode_rate_at_every_length(target):
    server = FakeOllama()
    result = _measure(server, target, concurrency=1)
    assert result["decode_tok_s"] == pytest.approx(server.decode_tok_s, rel=0.01)  # prefill is not in it
    assert result["per_request_decode_tok_s"] == pytest.approx(server.decode_tok_s, rel=0.01)


# Time that is neither prefill nor decode (Ollama's Go side; the rig measured 0.1 s at 512, 1.4 s at 8k, 2.3 s at 16k,
# 0.15 s at 32k) is in neither the decode rate nor the prefill, so the fake adds a non-monotonic amount of it.
def _go_side(length: int) -> float:
    return {512: 0.11, 2048: 0.5, 8192: 1.36, 16384: 2.27, 32768: 0.15}.get(length, 0.3)


@pytest.mark.parametrize("target", [512, 2048, 8192, 16384, 32768])
def test_the_decode_aggregate_at_c1_is_the_per_request_rate_whatever_time_the_go_side_adds(target):
    server = FakeOllama(overhead_s=_go_side(target))
    result = _measure(server, target)
    assert result["decode_tok_s"] == pytest.approx(server.decode_tok_s, rel=0.001)
    assert result["per_request_decode_tok_s"] == pytest.approx(server.decode_tok_s, rel=0.001)
    assert result["overhead_ms"] == pytest.approx(_go_side(target) * 1000, rel=0.01)  # recorded per row


@pytest.mark.parametrize("concurrency", [4, 8])
def test_the_decode_aggregate_with_slots_is_c_streams_whatever_the_go_side_adds(concurrency):
    server = FakeOllama(overhead_s=1.36)
    result = _measure(server, 2048, concurrency=concurrency)
    assert result["decode_tok_s"] == pytest.approx(concurrency * server.decode_tok_s, rel=0.01)


def test_a_queued_server_reads_about_one_stream_even_without_the_slot_log():
    single = _measure(FakeOllama(overhead_s=0.5), 512)["decode_tok_s"]
    queued = FakeOllama(overhead_s=0.5, honour_parallel=False, report_slots=False)
    assert _measure(queued, 512, concurrency=4)["decode_tok_s"] == pytest.approx(single, rel=0.2)


def test_ttft_is_the_time_to_the_first_token_and_the_prefill_is_its_own_field():
    server = FakeOllama(overhead_s=1.0)
    result = _measure(server, 2048)
    prefill_ms = 2048 / server.prefill_tok_s * 1000
    assert result["prefill_ms"] == pytest.approx(prefill_ms, rel=0.05)
    assert result["ttft_ms"] == pytest.approx(prefill_ms + 1000.0, rel=0.05)  # what a user waits through
    assert result["ttft_ms"] > result["prefill_ms"]


def test_a_ttft_floor_reads_the_time_to_the_first_token_and_the_cache_check_reads_the_prefill():
    server = FakeOllama(overhead_s=1.0)
    (row,) = run_linux_generate_ladder(
        spec_id="C2-GENERATE", spec_rel="experiments/c2-generate/spec.yaml", run_id="t4-r3", date="2026-10-26",
        remote_device=IDENT, model="phi3.5:3.8b", revision_or_digest="d" * 64,
        measure_shape=lambda length, conc: _measure(server, length, conc),
        prompt_lengths=(2048,), concurrencies=(1,), quant="Q4_K_M", licence="mit",
    )
    assert row["settings"]["prefill_ms"] < 600 < row["settings"]["ttft_ms"] < 1600
    assert row["settings"]["overhead_ms"] == pytest.approx(1000.0, rel=0.01)
    assert isinstance(decide([row], request(max_ttft_ms=2000.0)), Decision)
    assert isinstance(decide([row], request(max_ttft_ms=1000.0)), Refusal)  # the prefill (512 ms) would pass


def test_the_cache_check_still_reads_the_prefill_when_the_go_side_adds_time(monkeypatch):
    monkeypatch.setattr(linux_cuda, "ladder_prefix", lambda nonce, call: "0001abcd ")
    with pytest.raises(LinuxShapeFailure) as caught:
        _measure(FakeOllama(overhead_s=2.0), 2048)  # a long total hides nothing: the prefill is still a cache hit
    assert "prompt came from llama-server's cache" in caught.value.reason


def test_the_end_to_end_rate_is_its_own_field_and_falls_with_the_prompt_length():
    short = _measure(FakeOllama(), 512)
    long = _measure(FakeOllama(), 32768)
    assert long["end_to_end_tok_s"] < short["end_to_end_tok_s"] / 3  # prefill dominates a long prompt's wall clock
    assert short["end_to_end_tok_s"] < short["decode_tok_s"]
    assert long["decode_tok_s"] == pytest.approx(short["decode_tok_s"], rel=0.01)  # the router ranks on this one


def test_the_router_ranks_on_the_decode_aggregate_not_the_end_to_end_rate():
    server = FakeOllama()
    (long_row,) = run_linux_generate_ladder(
        spec_id="C2-GENERATE", spec_rel="experiments/c2-generate/spec.yaml", run_id="t4-r3", date="2026-10-26",
        remote_device=IDENT, model="phi3.5:3.8b", revision_or_digest="d" * 64,
        measure_shape=lambda length, conc: _measure(server, length, conc),
        prompt_lengths=(8192,), concurrencies=(1,), quant="Q4_K_M", licence="mit",
    )
    assert long_row["settings"]["end_to_end_tok_s"] < long_row["settings"]["decode_tok_s"]
    result = decide([long_row], request())
    assert result.document["margin"]["bound"] == pytest.approx(long_row["settings"]["decode_tok_s"], rel=0.001)


def test_a_server_that_logged_fewer_slots_than_the_concurrency_is_unmeasured():
    server = FakeOllama(honour_parallel=False)  # logs n_slots = 1 for a concurrency of 4
    with pytest.raises(LinuxShapeFailure) as caught:
        _measure(server, 512, concurrency=4)
    assert caught.value.failure_class == "unmeasured"
    assert "1 slot(s) for concurrency 4" in caught.value.reason


def test_the_slot_count_is_recorded_in_the_row_with_where_it_came_from():
    server = FakeOllama()
    (row,) = run_linux_generate_ladder(
        spec_id="C2-GENERATE", spec_rel="experiments/c2-generate/spec.yaml", run_id="t4-r3", date="2026-10-26",
        remote_device=IDENT, model="phi3.5:3.8b", revision_or_digest="d" * 64,
        measure_shape=lambda length, conc: _measure(server, length, conc),
        prompt_lengths=(512,), concurrencies=(4,), quant="Q4_K_M", licence="mit",
    )
    assert row["settings"]["parallel_slots"] == 4
    assert row["settings"]["parallel_slots_source"] == "llama-server log"
    assert row["settings"]["num_ctx"] > 512  # what was sent, not the nominal cell
    assert row["settings"]["cell_tokens"] == 512


def test_the_ladder_sets_each_slot_count_once_by_looping_concurrency_outermost():
    server = FakeOllama()
    rows = run_linux_generate_ladder(
        spec_id="C2-GENERATE", spec_rel="experiments/c2-generate/spec.yaml", run_id="t4-r3", date="2026-10-26",
        remote_device=IDENT, model="phi3.5:3.8b", revision_or_digest="d" * 64,
        measure_shape=lambda length, conc: _measure(server, length, conc),
        prompt_lengths=(512, 1024), concurrencies=(1, 4, 8), quant="Q4_K_M", licence="mit",
    )
    assert server.restarts == 2  # 1 -> 4 -> 8; the first count needs no restart
    assert [row["settings"]["concurrency"] for row in rows] == [1, 1, 4, 4, 8, 8]


def test_each_readiness_poll_is_short_so_one_hung_request_cannot_use_the_budget(monkeypatch):
    seen: list[float] = []

    class _Reply:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def read(self):
            return b'{"version": "0.40.11"}'

    def urlopen(_request, timeout):
        seen.append(timeout)
        return _Reply()

    monkeypatch.setattr(linux_cuda.urllib.request, "urlopen", urlopen)
    server = LinuxOllamaServer()  # the real http_get
    assert server._poll_get("http://127.0.0.1:11441/api/version") == {"version": "0.40.11"}
    assert seen == [linux_cuda.READY_POLL_TIMEOUT_S] and seen[0] <= 2.0


class _Proc:
    pid = 4242

    def __init__(self) -> None:
        self.terminated = False

    def poll(self):
        return 0 if self.terminated else None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout=None):
        return 0


def test_the_real_server_runs_with_ollama_num_parallel_and_restarts_to_change_it(tmp_path: Path):
    launched: list[dict] = []

    def popen(argv, env=None, **_kwargs):
        launched.append(dict(env))
        return _Proc()

    server = LinuxOllamaServer(
        log_path=tmp_path / "ollama.log", popen=popen, http_get=lambda _url: {"version": "0.40.11"}
    )
    server.start()
    server.ensure_parallel(1)
    assert len(launched) == 1 and launched[0]["OLLAMA_NUM_PARALLEL"] == "1"
    server.ensure_parallel(4)
    assert len(launched) == 2 and launched[1]["OLLAMA_NUM_PARALLEL"] == "4"
    server.ensure_parallel(4)
    assert len(launched) == 2  # already at 4
    server.stop()


def test_the_slot_count_is_read_from_this_processs_part_of_the_server_log(tmp_path: Path):
    log = tmp_path / "ollama.log"
    log.write_text("load_model: n_slots = 8\n", encoding="utf-8")  # an earlier process
    server = LinuxOllamaServer(log_path=log, popen=lambda *a, **k: _Proc(), http_get=lambda _u: {"version": "x"})
    server.start()
    assert server.effective_slots() is None  # nothing logged by this process yet
    with log.open("a", encoding="utf-8") as handle:
        handle.write("srv  load_model: n_slots = 4\n")
    assert server.effective_slots() == 4
    server.stop()


# -- Requirement 1: the router never reads what earlier ladders could not measure ------------------------------------


def row(row_id: str, device: str, throughput: float, *, version: int | None, concurrency: int = 1, ttft: float = 10.0):
    settings: dict = {"thinking_level": "off", "num_predict": 128, "num_ctx": 2048, "concurrency": concurrency}
    if version is not None:
        settings["ladder_version"] = version
        settings["prefix_caching"] = "off"
    slot = {"phase": "repeat", "throughput": throughput, "latency_ms": 100, "ttft_ms": ttft}
    return {
        "id": row_id,
        "task": "generate",
        "model": {"repo_or_tag": "phi3.5:3.8b", "revision_or_digest": "abc"},
        "engine": {"name": "ollama", "version": "0.40.11"},
        "device": device,
        "settings": settings,
        "date": "2026-10-26",
        "repeats": 3,
        "provisional": False,
        "artifacts": {"slots": [slot] * 3},
        "provenance": {"experiment_spec": "experiments/c2-generate/spec.yaml", "run_id": "t"},
    }


def request(**extra) -> Request:
    return Request(task="generate", as_of=date(2026, 10, 26), model="phi3.5:3.8b", allow_remote=True, **extra)


def winner(result) -> str:
    assert isinstance(result, Decision)
    return result.document["receipts"][0]


def test_pre_v3_throughput_above_one_slot_is_never_used_and_the_note_says_so():
    rows = [row("t4-v2-c1", "colab:T4", 50.0, version=2), row("t4-v2-c4", "colab:T4", 400.0, version=2, concurrency=4)]
    result = decide(rows, request())
    assert winner(result) == "t4-v2-c1"  # the inflated c4 figure did not win
    assert any("1 pre-v3 row(s) with concurrency > 1 not used" in note for note in result.document["notes"])
    assert not throughput_measured(rows[1]) and throughput_measured(rows[0])


def test_a_pre_v3_row_with_an_inflated_floor_figure_cannot_pass_a_decode_floor():
    rows = [row("t4-v2-c4", "colab:T4", 400.0, version=2, concurrency=4)]
    result = decide(rows, request(min_metrics={"decode_tok_s": 100.0}))
    assert isinstance(result, Refusal)


def test_v3_throughput_above_one_slot_is_used():
    rows = [row("t4-v3-c1", "colab:T4", 50.0, version=3), row("t4-v3-c4", "colab:T4", 180.0, version=3, concurrency=4)]
    assert winner(decide(rows, request())) == "t4-v3-c4"


def test_a_pre_v3_ttft_never_passes_a_ttft_floor_and_a_v3_ttft_does():
    pre = [row("t4-v2", "colab:T4", 50.0, version=2, ttft=15.0)]
    assert isinstance(decide(pre, request(max_ttft_ms=100.0)), Refusal)
    assert not ttft_measured(pre[0])
    fresh = [row("t4-v3", "colab:T4", 50.0, version=3, ttft=80.0)]
    assert winner(decide(fresh, request(max_ttft_ms=100.0))) == "t4-v3"
    assert ttft_measured(fresh[0])


def test_the_ttft_note_counts_the_rows_that_were_not_read():
    rig = row("rig", "5090", 200.0, version=None, ttft=3.7)
    rig["settings"]["prefix_caching"] = "off"
    rows = [rig, row("l4-v3", "colab:L4", 40.0, version=3, ttft=60.0)]
    result = decide(rows, request(max_ttft_ms=100.0))
    assert winner(result) == "l4-v3"
    assert "ttft of 1 pre-v3 row(s) not used: it was a prompt-cache hit, not a prefill" in result.document["notes"]


def test_a_rig_row_that_records_a_ttft_is_not_read_either():
    rig = row("rig", "5090", 200.0, version=None, ttft=3.7)  # the harness cannot be told apart from a cached one
    assert not ttft_measured(rig)
    assert isinstance(decide([rig], request(max_ttft_ms=100.0)), Refusal)


def test_the_table_marks_the_columns_that_were_not_measured(tmp_path: Path, capsys):
    table = tmp_path / "table"
    table.mkdir()
    rows = [
        row("a-v1-c1", "colab:T4", 50.0, version=None, ttft=14.0),
        row("b-v2-c4", "colab:T4", 400.0, version=2, concurrency=4),
        row("c-v3-c4", "colab:T4", 180.0, version=3, concurrency=4, ttft=80.0),
    ]
    rows[0]["device"] = "colab:T4"
    (table / "routing.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert main(["table", "--root", str(tmp_path)]) == 0
    lines = {line.split("\t")[0]: line for line in capsys.readouterr().out.splitlines()}
    assert "ttft: not measured" in lines["a-v1-c1"]
    assert "throughput: not measured" in lines["b-v2-c4"] and "ttft: not measured" in lines["b-v2-c4"]
    assert "not measured" not in lines["c-v3-c4"] and "ladder v3" in lines["c-v3-c4"]
