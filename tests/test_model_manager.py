"""Tests for the model manager (Phase 1, card nu) on a fake engine (no GPU).

Covers Design Section 4:
1. Slow load (ready after N s, under the cap): the probe succeeds, and ``cold_load_s`` ≈ N.
2. A load that never finishes: ``load_failed`` at the cap (with ``load_cap_s`` and ``last_engine_status``), no
   measurement row.
3. Eviction mid-ladder (``/api/ps`` stops listing the model): reload, row flagged ``reloaded: True``, second
   ``cold_load_s`` recorded.
4. A partial offload (``size_vram < size``): recorded, ``offload_fraction < 1``.
5. The C2:86 case replayed: fill-card alone with a cold load of ~84 s (plus ~37.5 s probe work > 120 s total)
   produces a real resident row, not ``timeout``.
6. For Ollama, a context change that reloads: each reload recorded as warm (page cache) when resident, or cold
   (with ``reloaded: True``) when evicted.
7. Backward compatibility: old rows without ``cold_load_s``, ``warm``, ``offload_fraction``, or ``reloaded``
   validate against ``routing-row.schema.json`` and read cleanly via ``read_row_load_info``.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from interlock.devices import RemoteDeviceIdentity
from interlock.harness.linux_cuda import (
    LinuxShapeFailure,
    measure_linux_ollama_generate,
    measure_linux_ollama_window,
    run_linux_fill_card,
    run_linux_generate_ladder,
    run_linux_spec,
)
from interlock.model_manager import (
    LOAD_CAP_FLOOR_S,
    LoadFailedError,
    ModelManager,
    OllamaModelManager,
    load_cap_s,
)
from interlock.receipt import is_speed_row, read_row_load_info, validate_row
from interlock.router.policy import Refusal, Request, decide
from tests.fake_ollama import FakeOllama

IDENT_A100_80 = RemoteDeviceIdentity(
    provider="colab",
    card="A100-80",
    vram_mib=81920,
    driver_version="550.54.15",
    cuda_version="12.4",
    pool="us-central1-a",
)
IDENT_T4 = RemoteDeviceIdentity(
    provider="colab",
    card="T4",
    vram_mib=15360,
    driver_version="550.54.15",
    cuda_version="12.4",
    pool="us-central1-a",
)


def test_load_cap_s_scales_with_size_and_enforces_floor():
    # MISSING_TAG_BYTES (16 GiB) fallback: 16 * 20 * 2 = 640 s
    assert load_cap_s(None) == pytest.approx(640.0)
    # 31 GiB: 31 * 20 * 2 = 1240 s
    assert load_cap_s(31 * 1024**3) == pytest.approx(1240.0)
    # C2:86 gemma4:31b Q8_0 (33,836,414,838 B = 31.5126 GiB): ~1260.5 s
    assert load_cap_s(33_836_414_838) == pytest.approx((33_836_414_838 / (1024**3)) * 40.0)
    # Tiny 1 GiB model hits the 180 s floor
    assert load_cap_s(1024**3) == LOAD_CAP_FLOOR_S


def test_slow_load_under_cap_succeeds_and_records_cold_load_s():
    server = FakeOllama(cold_load_s=47.5, vram_mib=81920.0, weights_mib=32000.0)
    manager = OllamaModelManager(server)
    reading = measure_linux_ollama_window(
        server,
        lambda: (33000, 81920),
        model="gemma4:31b-q8_0",
        num_ctx=2048,
        quant="Q8_0",
        model_manager=manager,
        size_bytes=33 * 1024**3,
    )
    assert reading["cold_load_s"] == pytest.approx(47.5)
    assert reading["warm"] is False
    assert reading["reloaded"] is False
    assert reading["offload_fraction"] == 1.0
    assert reading["size"] == reading["size_vram"] > 0
    assert len(server.load_calls) == 1
    assert server.load_calls[0]["timeout"] == pytest.approx(1320.0)


def test_load_that_never_finishes_fails_with_load_failed_and_no_measurement_row():
    # Case A: /api/ps never lists the model before cap_s expires
    server = FakeOllama(cold_load_s=5.0, never_ready=True, poll_step_s=60.0)
    manager = OllamaModelManager(server, default_cap_s=180.0, sleep=lambda _s: None)

    with pytest.raises(LoadFailedError) as caught:
        manager.ensure_loaded("qwen3:30b-a3b", "Q4_K_M", 2048)
    err = caught.value
    assert isinstance(err, LinuxShapeFailure)
    assert err.failure_class == "load_failed"
    assert err.cap_s == 180.0
    assert "/api/ps models=[]" in err.last_status

    # Plugged into fill-card: returns a single load_failed row, not a measurement row
    fill_server = FakeOllama(cold_load_s=5.0, never_ready=True, poll_step_s=60.0)
    fill_mgr = OllamaModelManager(fill_server, default_cap_s=180.0, sleep=lambda _s: None)
    fill_row = run_linux_fill_card(
        spec_id="C2-FILL-CARD",
        spec_rel="experiments/c2-fill-card/spec.yaml",
        run_id="c2-never-ready",
        date="2026-10-29",
        remote_device=IDENT_A100_80,
        model="qwen3:30b-a3b",
        revision_or_digest="a" * 64,
        trained_ctx=32768,
        abort_fraction=0.90,
        grain=2048,
        quant="Q4_K_M",
        licence="apache-2.0",
        measure_window=lambda ctx: measure_linux_ollama_window(
            fill_server,
            lambda: (1000, 81920),
            model="qwen3:30b-a3b-q4_k_m",
            num_ctx=ctx,
            quant="Q4_K_M",
            model_manager=fill_mgr,
        ),
    )
    validate_row(fill_row)
    assert fill_row["outcome"] == "load_failed"
    assert is_speed_row(fill_row) is False
    assert fill_row["latency"]["throughput"] is None
    assert fill_row["settings"]["failure_class"] == "load_failed"
    assert fill_row["settings"]["load_cap_s"] == 180.0
    assert "/api/ps models=[]" in fill_row["settings"]["last_engine_status"]
    assert len(fill_server.all_calls) == 0  # no probe was sent after ensure_loaded failed

    # Plugged into generate-ladder: fails the step as load_failed with 1 row, no measurement row
    gen_server = FakeOllama(cold_load_s=250.0)
    gen_mgr = OllamaModelManager(gen_server, default_cap_s=180.0, sleep=lambda _s: None)
    gen_rows = run_linux_generate_ladder(
        spec_id="C2-GENERATE",
        spec_rel="experiments/c2-generate/spec.yaml",
        run_id="c2-gen-load-failed",
        date="2026-10-29",
        remote_device=IDENT_A100_80,
        model="qwen3:30b-a3b",
        revision_or_digest="a" * 64,
        prompt_lengths=(512, 1024, 2048),
        concurrencies=(1, 4),
        quant="Q4_K_M",
        licence="apache-2.0",
        model_manager=gen_mgr,
        measure_shape=lambda plen, conc: measure_linux_ollama_generate(
            gen_server,
            lambda: (1000, 81920),
            model="qwen3:30b-a3b-q4_k_m",
            prompt_tokens=plen,
            concurrency=conc,
            repeats=1,
            clock=gen_server.clock,
        ),
    )
    assert len(gen_rows) == 1
    validate_row(gen_rows[0])
    assert gen_rows[0]["outcome"] == "load_failed"
    assert is_speed_row(gen_rows[0]) is False
    assert gen_rows[0]["settings"]["load_cap_s"] == 180.0
    assert "timed out" in gen_rows[0]["settings"]["last_engine_status"]
    assert len(gen_server.all_calls) == 0

    # Router excludes load_failed as unmeasured, never as a capability refusal
    decision = decide(
        [fill_row],
        Request(
            task="generate",
            as_of=date(2026, 10, 29),
            model="qwen3:30b-a3b",
            allow_remote=True,
            device="colab:A100-80",
        ),
    )
    assert isinstance(decision, Refusal)
    assert [item["code"] for item in decision.document["excluded"]] == ["load_failed"]
    assert "Measured failure reasons" not in json.dumps(decision.document)


def test_eviction_mid_ladder_reloads_and_flags_reloaded_with_second_cold_load_s():
    server = FakeOllama(cold_load_s=31.0, warm_load_s=2.0, vram_mib=81920.0, weights_mib=16000.0)
    manager = OllamaModelManager(server)
    lengths = (512, 1024, 2048)

    def measure_cell(plen: int, conc: int) -> dict[str, Any]:
        if plen == 1024:
            # Simulate Ollama evicting the model between p512 and p1024
            server.evict()
            server.cold_load_s = 28.5
        return measure_linux_ollama_generate(
            server,
            lambda: (18000, 81920),
            model="qwen3:14b-q8_0",
            prompt_tokens=plen,
            concurrency=conc,
            repeats=1,
            clock=server.clock,
        )

    rows = run_linux_generate_ladder(
        spec_id="C2-GENERATE",
        spec_rel="experiments/c2-generate/spec.yaml",
        run_id="c2-evict-mid-ladder",
        date="2026-10-29",
        remote_device=IDENT_A100_80,
        model="qwen3:14b",
        revision_or_digest="b" * 64,
        prompt_lengths=lengths,
        concurrencies=(1,),
        quant="Q8_0",
        licence="apache-2.0",
        model_manager=manager,
        measure_shape=measure_cell,
    )
    assert len(rows) == 3
    for row in rows:
        validate_row(row)
        assert is_speed_row(row)

    # Cell 0 (p512): initial cold load (31.0 s), warm=False, reloaded=False
    assert rows[0]["cold_load_s"] == pytest.approx(31.0)
    assert rows[0]["warm"] is False
    assert rows[0]["reloaded"] is False
    assert rows[0]["settings"]["cold_load_s"] == pytest.approx(31.0)
    assert rows[0]["settings"]["warm"] is False
    assert rows[0]["settings"]["reloaded"] is False

    # Cell 1 (p1024): evicted before cell -> reloaded=True, warm=False, second cold_load_s=28.5 s
    assert rows[1]["cold_load_s"] == pytest.approx(28.5)
    assert rows[1]["warm"] is False
    assert rows[1]["reloaded"] is True
    assert rows[1]["settings"]["cold_load_s"] == pytest.approx(28.5)
    assert rows[1]["settings"]["warm"] is False
    assert rows[1]["settings"]["reloaded"] is True
    assert rows[1]["model"]["reloaded"] is True

    # Cell 2 (p2048): still resident -> warm=True, reloaded=True (preserved from step or False on current_record)
    assert rows[2]["warm"] is True


def test_partial_offload_records_size_size_vram_and_offload_fraction_below_one():
    # Card has 8000 MiB VRAM; weights are 2400 MiB; at num_ctx 8384 x 4 slots KV is 12576 MiB -> partial offload
    server = FakeOllama(cold_load_s=12.0, vram_mib=8000.0, weights_mib=2400.0)
    server.ensure_parallel(4)
    manager = OllamaModelManager(server)
    rec = manager.ensure_loaded("phi3.5:3.8b-q4_k_m", "Q4_K_M", 8384)
    assert rec.size is not None and rec.size_vram is not None
    assert rec.size_vram < rec.size
    assert rec.offload_fraction is not None and 0.0 < rec.offload_fraction < 1.0
    assert rec.offload_fraction == pytest.approx(rec.size_vram / rec.size, abs=1e-6)

    # Also verify a partial-offload failure row from fill-card carries size, size_vram, and offload_fraction < 1
    fill_server = FakeOllama(cold_load_s=12.0, vram_mib=2500.0, weights_mib=2400.0)
    fill_mgr = OllamaModelManager(fill_server)
    row = run_linux_fill_card(
        spec_id="C2-FILL-CARD",
        spec_rel="experiments/c2-fill-card/spec.yaml",
        run_id="c2-partial-offload",
        date="2026-10-29",
        remote_device=IDENT_T4,
        model="phi3.5:3.8b",
        revision_or_digest="c" * 64,
        trained_ctx=8192,
        abort_fraction=0.90,
        grain=2048,
        quant="Q4_K_M",
        licence="mit",
        measure_window=lambda ctx: measure_linux_ollama_window(
            fill_server,
            lambda: (2500, 15360),
            model="phi3.5:3.8b-q4_k_m",
            num_ctx=ctx,
            quant="Q4_K_M",
            model_manager=fill_mgr,
        ),
    )
    validate_row(row)
    assert row["outcome"] == "oom"
    assert row["size_vram"] < row["size"]
    assert 0.0 < row["offload_fraction"] < 1.0
    assert row["gpu_resident"] is False
    assert row["model"]["offload_fraction"] == row["settings"]["offload_fraction"] == row["offload_fraction"]
    assert row["cold_load_s"] == pytest.approx(12.0)


def test_c2_86_replay_fill_card_alone_with_84s_cold_load_produces_real_row_not_timeout():
    """Replay C2:86 (A100-80 s3, gemma4:31b Q8_0, 33.8 GB):
    - Cold load takes ~80.54 s (we test 84.0 s as specified in Design Section 4).
    - First probe's prefill + decode takes ~37.5 s (overhead_s=32.2, decode_tok_s=3.0 for 16 tokens -> 5.33 s).
    - Without ensure_loaded, 84.0 + 37.5 = 121.5 s > 120.0 s urlopen timeout -> failed with outcome='timeout'.
    - With ensure_loaded plugged into fill-card, the 84.0 s cold load runs under cap_s (~1260.5 s), and the
      probe's 120 s clock starts after ensure_loaded returns (taking 37.5 s <= 120 s), producing a real row.
    """

    server = FakeOllama(
        cold_load_s=84.0,
        warm_load_s=3.5,
        overhead_s=32.2,
        prefill_tok_s=1000.0,
        decode_tok_s=3.0,
        vram_mib=81920.0,
        weights_mib=32269.0,
        kv_mib_per_token=0.25,
    )
    manager = OllamaModelManager(server)
    gemma_q8_bytes = 33_836_414_838

    row = run_linux_fill_card(
        spec_id="C2-FILL-CARD",
        spec_rel="experiments/c2-fill-card/spec.yaml",
        run_id="windowC2-86-a10080-s3k-replay",
        date="2026-10-29",
        remote_device=IDENT_A100_80,
        model="sha256:6316f0629137b022feff2eff9fb9cdecf3be6ac4ce7fa2fc1d63ff9fc2f5a67b",
        revision_or_digest="6316f0629137b022feff2eff9fb9cdecf3be6ac4ce7fa2fc1d63ff9fc2f5a67b",
        trained_ctx=32768,
        abort_fraction=0.90,
        grain=2048,
        quant="Q8_0",
        licence="gemma",
        measure_window=lambda ctx: measure_linux_ollama_window(
            server,
            lambda: (35000, 81920),
            model="c2-sha256-6316f0629137-q8_0",
            num_ctx=ctx,
            quant="Q8_0",
            model_manager=manager,
            size_bytes=gemma_q8_bytes,
        ),
    )
    validate_row(row)
    assert "outcome" not in row, f"expected a real measured row, got outcome={row.get('outcome')!r}"
    assert is_speed_row(row) is True
    assert row["gpu_resident"] is True
    assert row["settings"]["max_resident_ctx"] == 32768
    assert row["cold_load_s"] == pytest.approx(84.0)
    assert row["settings"]["cold_load_s"] == pytest.approx(84.0)
    assert row["model"]["cold_load_s"] == pytest.approx(84.0)
    assert row["offload_fraction"] == 1.0
    assert row["reloaded"] is False
    assert row["size"] == row["size_vram"] > 0


def test_ollama_context_change_reloads_and_records_warm_or_cold():
    server = FakeOllama(cold_load_s=42.0, warm_load_s=3.25, vram_mib=81920.0, weights_mib=16000.0)
    manager = OllamaModelManager(server)

    # 1. First load at ctx=2048 is cold (warm=False, reloaded=False, cold_load_s=42.0)
    r1 = manager.ensure_loaded("qwen3:14b-q8_0", "Q8_0", 2048)
    assert r1.warm is False
    assert r1.reloaded is False
    assert r1.cold_load_s == pytest.approx(42.0)
    assert len(server.load_calls) == 1

    # 2. Same ctx=2048 while resident issues no POST and records warm=True, cold_load_s=0.0
    r_same = manager.ensure_loaded("qwen3:14b-q8_0", "Q8_0", 2048)
    assert r_same.warm is True
    assert r_same.reloaded is False
    assert r_same.cold_load_s == pytest.approx(0.0)
    assert len(server.load_calls) == 1  # no new load call

    # 3. Context change to ctx=4096 while resident reloads from warm page cache (warm=True, cold_load_s=3.25)
    r2 = manager.ensure_loaded("qwen3:14b-q8_0", "Q8_0", 4096)
    assert r2.warm is True
    assert r2.reloaded is False
    assert r2.cold_load_s == pytest.approx(3.25)
    assert len(server.load_calls) == 2
    assert server.load_calls[-1]["num_ctx"] == 4096

    # 4. Evict and change context to ctx=8192 -> cold reload (warm=False, reloaded=True, cold_load_s=42.0)
    server.evict()
    r3 = manager.ensure_loaded("qwen3:14b-q8_0", "Q8_0", 8192)
    assert r3.warm is False
    assert r3.reloaded is True
    assert r3.cold_load_s == pytest.approx(42.0)
    assert len(server.load_calls) == 3

    # 5. keep() and unload() exercise the full ModelManager interface
    assert isinstance(manager, ModelManager) or hasattr(manager, "keep")
    manager.keep("qwen3:14b-q8_0", keep_alive="15m")
    assert server.load_calls[-1]["keep_alive"] == "15m"
    assert manager.still_loaded("qwen3:14b-q8_0") is True
    manager.unload("qwen3:14b-q8_0")
    assert manager.still_loaded("qwen3:14b-q8_0") is False


def test_old_rows_without_new_load_fields_validate_and_read_cleanly():
    old_row = {
        "id": "C2-GENERATE/qwen3:8b/Q4_K_M/L4/p512-c1/windowC2-01",
        "task": "generate",
        "model": {
            "repo_or_tag": "qwen3:8b",
            "revision_or_digest": "500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41",
            "quant": "Q4_K_M",
            "licence": "apache-2.0",
            "size": 5200000000,
            "size_vram": 5200000000,
        },
        "model_family": "qwen",
        "engine": {"name": "ollama", "version": "0.6.2"},
        "device": "colab:L4",
        "remote_device": IDENT_T4.to_dict() | {"card": "L4", "vram_mib": 23034},
        "gpu_resident": True,
        "settings": {
            "thinking_level": "off",
            "num_predict": 128,
            "num_ctx": 704,
            "batch": None,
            "static_shape": None,
            "precision": None,
            "prompt_tokens": 512,
            "concurrency": 1,
            "peak_mib": 6800,
            "abort_mib": 20730,
        },
        "quality": [],
        "latency": {
            "cold_ms": None,
            "warm_p50_ms": 120.0,
            "warm_p95_ms": 125.0,
            "throughput": 82.5,
            "throughput_unit": "tok/s",
            "compile_ms": None,
        },
        "artifacts": {
            "slots": [
                {"phase": "warmup", "throughput": 80.0, "latency_ms": 122.0, "ttft_ms": 40.0},
                {"phase": "repeat", "throughput": 82.5, "latency_ms": 120.0, "ttft_ms": 38.0},
            ]
        },
        "date": "2026-10-25",
        "repeats": 3,
        "provisional": False,
        "provenance": {"experiment_spec": "experiments/c2-generate/spec.yaml", "run_id": "windowC2-01"},
    }
    validate_row(old_row)
    info = read_row_load_info(old_row)
    assert info == {
        "cold_load_s": None,
        "warm": None,
        "size": 5200000000,
        "size_vram": 5200000000,
        "offload_fraction": 1.0,
        "reloaded": None,
        "load_signal": None,
        "expires_at": None,
    }


def test_run_linux_spec_fill_card_and_generate_stamp_load_fields():
    server = FakeOllama(cold_load_s=9.5, warm_load_s=1.25, vram_mib=23034.0, weights_mib=5000.0)
    spec = {
        "id": "C2-FILL-CARD",
        "procedure": "fill-card",
        "platform": "linux-cuda",
        "grain": 2048,
        "models": [
            {
                "model": "qwen3:8b",
                "quant": "Q4_K_M",
                "device": "colab:T4",
                "revision_or_digest": "500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41",
                "trained_ctx": 4096,
                "licence": "apache-2.0",
            }
        ],
    }
    (row,) = run_linux_spec(
        spec,
        spec_path=Path("experiments/c2-fill-card/spec.yaml"),
        spec_rel="experiments/c2-fill-card/spec.yaml",
        run_id="t4-fill-spec",
        date="2026-10-29",
        remote_device=IDENT_T4,
        sample_vram_mib=lambda: (6000, 15360),
        server=server,  # type: ignore[arg-type]
    )
    validate_row(row)
    info = read_row_load_info(row)
    assert info["cold_load_s"] == pytest.approx(9.5)
    assert info["offload_fraction"] == 1.0
    assert info["reloaded"] is False


def test_model_manager_edge_cases_and_mapping_protocol():
    from interlock.harness.linux_cuda import LinuxShapeFailure

    class _MinimalServer:
        version = "0.40.11"

        def __init__(self) -> None:
            self.posts: list[tuple[str, dict]] = []
            self.ps_mode = "ready"
            self.tags_error = True

        def get(self, path: str) -> object:
            if path == "/api/tags":
                if self.tags_error:
                    raise RuntimeError("tags unavailable")
                return {"models": [{"name": "m1:latest", "size": 4_000_000_000}]}
            if path == "/api/ps":
                if self.ps_mode == "error":
                    raise RuntimeError("ps error")
                if self.ps_mode == "non_dict":
                    return ["bad"]
                if self.ps_mode == "zero_size":
                    return {"models": ["skip", {"name": "m1", "size": 0, "size_vram": 0}]}
                if self.ps_mode == "other_model":
                    return {
                        "models": [
                            {"name": "other", "size": 100, "size_vram": 100},
                            {"name": "third", "size": 100, "size_vram": 100},
                        ]
                    }
                return {
                    "models": [
                        {
                            "name": "m1:latest",
                            "size": 4_000_000_000,
                            "size_vram": 4_000_000_000,
                            "expires_at": "2026-10-29T09:00:00Z",
                        }
                    ]
                }
            return {}

        def post(self, path: str, payload: dict) -> dict:
            self.posts.append((path, payload))
            if payload.get("model") == "err_resp":
                return {"error": "load error from response"}
            if payload.get("model") == "oom_model":
                raise RuntimeError("CUDA out of memory allocating KV cache")
            return {"done": True, "done_reason": "load"}

    srv = _MinimalServer()
    mgr = OllamaModelManager(
        srv,
        engine_version="0.40.11",
        size_bytes_by_model={"m1": 4_000_000_000, "exact:tag": 2_000_000_000},
    )
    assert mgr.engine_version == "0.40.11"
    assert mgr._lookup_tag_size_bytes("exact:tag") == 2_000_000_000
    assert mgr._lookup_tag_size_bytes("m1:latest") == 4_000_000_000
    assert mgr._lookup_tag_size_bytes("unknown") is None

    rec = mgr.ensure_loaded("m1:latest", "Q4_K_M", num_ctx=2048, cap_s=300.0)
    assert rec.ready_signal == "ollama:/api/ps"
    assert rec["signal"] == "ollama:/api/ps"
    assert list(rec) == list(rec.to_dict().keys())
    assert len(rec) == 11
    assert mgr.last_record("m1") == rec
    assert mgr.current_record("m1") is not None

    srv.ps_mode = "error"
    assert mgr.current_record("m1") is None
    srv.ps_mode = "non_dict"
    assert mgr.still_loaded("m1") is False
    srv.ps_mode = "zero_size"
    assert mgr.still_loaded("m1") is False
    srv.ps_mode = "other_model"
    assert mgr.still_loaded("m1") is False

    mgr.unload("m1")
    assert srv.posts[-1][1]["keep_alive"] == 0

    with pytest.raises(LoadFailedError):
        mgr.ensure_loaded("err_resp", "Q4_K_M", ctx=1024, cap_s=10.0)

    with pytest.raises(LinuxShapeFailure) as exc_info:
        mgr.ensure_loaded("oom_model", "Q4_K_M", ctx=1024, cap_s=10.0)
    assert exc_info.value.failure_class == "oom"

