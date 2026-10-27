"""1.0.10: the TTFT scaling floor no longer gates a cell; the prefill ceiling is the cache check.

The numbers are T4 s1b's own (qwen3:8b Q4_K_M). At concurrency above 1 one pass's mean prefill spikes by 2 to 4x, so a
floor on the scaling of the prefill across cells measured noise: it lost 7 of 21 cells there.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from interlock.devices import RemoteDeviceIdentity
from interlock.harness.linux_cuda import (
    GENERATE_LADDER_VERSION,
    LinuxShapeFailure,
    measure_linux_ollama_generate,
    run_linux_generate_ladder,
)
from interlock.router.policy import Decision, Request, decide
from tests.fake_ollama import FakeOllama
from tests.test_ladder_versions import row as router_row

IDENT = RemoteDeviceIdentity(
    provider="colab", card="T4", vram_mib=15360, driver_version="550.54.15", cuda_version="12.4", pool="us-central1-a"
)

# prefill_ms by prompt length, from T4 s1b: the repeat-pass median that became the cell's prefill_ms.
C4 = {512: 3747.16, 1024: 1946.7, 2048: 3339.5, 4096: 5815.6, 8192: 11432.6}
C8 = {512: 2333.9, 1024: 2067.2, 2048: 3472.8, 4096: 6003.9}
# c4 512's own passes: a warm-up, then three repeats (the median of the repeats is 3,747.16).
C4_512_PASSES = [4529.0, 4052.247, 1099.072, 3747.16]


def _measure_shape(table: dict[int, dict[int, float]]):
    def measure(length: int, concurrency: int) -> dict[str, Any]:
        prefill = table[concurrency][length]
        passes = [prefill, prefill, prefill, prefill]
        if (length, concurrency) == (512, 4):
            passes = C4_512_PASSES
        slots = [
            {"phase": "warmup" if i == 0 else "repeat", "throughput": 20.0, "latency_ms": 5000.0,
             "ttft_ms": value + 40.0, "prefill_ms": value}
            for i, value in enumerate(passes)
        ]
        return {"ttft_ms": prefill + 40.0, "prefill_ms": prefill, "decode_tok_s": 20.0, "peak_mib": 9000, "size": 5,
                "size_vram": 5, "prompt_tokens_evaluated": length, "slots": slots}

    return measure


def _ladder(measure, lengths, concurrencies) -> list[dict]:
    return run_linux_generate_ladder(
        spec_id="C2-GENERATE", spec_rel="experiments/c2-generate/spec.yaml", run_id="t4-s1b", date="2026-10-27",
        remote_device=IDENT, model="qwen3:8b", revision_or_digest="d" * 64, measure_shape=measure,
        prompt_lengths=lengths, concurrencies=concurrencies, quant="Q4_K_M", licence="apache-2.0",
    )


def test_t4_s1bs_own_c4_and_c8_prefills_all_measure():
    rows = _ladder(_measure_shape({4: C4, 8: C8}), (512, 1024, 2048, 4096, 8192), (4,))
    assert [row.get("outcome") for row in rows] == [None] * 5  # 1.0.9 lost 1024 to 8192 at c4 to the floor
    rows = _ladder(_measure_shape({8: C8}), (512, 1024, 2048, 4096), (8,))
    assert [row.get("outcome") for row in rows] == [None] * 4  # and 1024 to 4096 at c8


def test_a_cell_whose_prefill_is_a_cache_hit_is_still_unmeasured_by_the_ceiling():
    # c4 2048 with a 2 ms prefill: over 1,000,000 tok/s, far over PREFILL_TOK_S_CEILING (250,000).
    server = FakeOllama(prefill_tok_s=1_100_000.0)

    def measure(length: int, concurrency: int) -> dict:
        return measure_linux_ollama_generate(
            server, lambda: (1000, 15360), model="m", prompt_tokens=length, concurrency=concurrency, repeats=2,
            clock=server.clock,
        )

    (row,) = _ladder(measure, (2048,), (4,))
    assert row["outcome"] == "unmeasured" and "llama-server's cache" in row["notes"]
    with pytest.raises(LinuxShapeFailure) as caught:
        measure(2048, 4)
    assert "prompt came from llama-server's cache" in caught.value.reason


def test_the_spread_of_the_repeat_passes_is_recorded_and_not_gated_on():
    rows = _ladder(_measure_shape({4: C4}), (512,), (4,))
    settings = rows[0]["settings"]
    assert settings["prefill_ms_passes"] == [4052.247, 1099.072, 3747.16]  # the repeats, in order, no warm-up
    assert settings["prefill_spread"] == 3.687  # 4052.247 / 1099.072
    assert "outcome" not in rows[0]


def test_a_flat_row_has_a_spread_of_one_and_a_row_with_no_slots_has_none():
    (flat,) = _ladder(_measure_shape({1: {512: 100.0}}), (512,), (1,))
    assert flat["settings"]["prefill_spread"] == 1.0

    def bare(length: int, concurrency: int) -> dict:  # a stub that returns no slots
        return {"ttft_ms": 140.0, "prefill_ms": 100.0, "decode_tok_s": 20.0, "peak_mib": 9000, "size": 5,
                "size_vram": 5, "prompt_tokens_evaluated": length}

    (row,) = _ladder(bare, (512,), (1,))
    assert row["settings"]["prefill_ms_passes"] == [] and row["settings"]["prefill_spread"] is None


def test_every_generate_row_is_version_4_and_says_the_floor_is_off():
    def refuse(length: int, concurrency: int) -> dict:
        raise LinuxShapeFailure("oom", "partial offload")

    measured = _ladder(_measure_shape({4: C4}), (512, 1024), (4,))
    failed = _ladder(refuse, (512,), (4,))
    for row in (*measured, *failed):
        assert row["settings"]["ladder_version"] == GENERATE_LADDER_VERSION == 4
        assert row["settings"]["ttft_floor"] == "off"


def test_the_router_keeps_a_v4_row_with_the_field_and_a_v4_row_without_it():
    with_field = router_row("t4-1.0.10", "colab:T4", 100.0, 4)
    with_field["settings"]["ttft_floor"] = "off"
    without = router_row("l4-1.0.9", "colab:L4", 90.0, 4)
    request = Request(task="generate", as_of=date(2026, 10, 27), model="phi3.5:3.8b", allow_remote=True)
    result = decide([with_field, without], request)
    assert isinstance(result, Decision)
    assert result.document["receipts"] == ["t4-1.0.10"]
    assert [item["id"] for item in result.document["runners_up"]] == ["l4-1.0.9"]  # compared, not filtered out
    assert not any("older-ladder" in note or "ladder version" in note for note in result.document["notes"])
    flipped = decide([router_row("t4-1.0.10", "colab:T4", 80.0, 4), {**without, "id": "l4-x"}], request)
    assert flipped.document["receipts"] == ["l4-x"]  # a row without the field still wins on its numbers
