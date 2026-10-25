"""E4b with fakes. Nothing here binds a port, starts Ollama, or touches a GPU."""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from interlock.adapters.base import StepRequest
from interlock.adapters.ollama import OllamaAdapter
from interlock.adapters.record_fixtures import read_card_mib
from interlock.guards import NO_WATCHDOG, NO_WATCHDOG_NOTE, GuardError
from interlock.harness.e4b import (
    GENERATE_TIMEOUT_S,
    PILOT_CALLS,
    PILOT_CLOCK_RULE,
    PILOT_MIN_GOOD,
    PROBE_TIMEOUT_S,
    CpuBurn,
    E4bError,
    IntelLane,
    LiveGpuSample,
    RepeatLimit,
    _header,
    build_live_seams,
    call_time_spent,
    card_models_at_ceiling,
    collect_series,
    counting_prompt,
    dry_text,
    generate_loop,
    is_steady,
    live_sample_cpu,
    live_sample_gpu,
    note_vram,
    ollama_pin,
    owned_generate,
    parse_nvidia_smi,
    parse_typeperf,
    perform_slot,
    pilot_destination,
    pilot_summary,
    prepare_prompt,
    prove_owned_server,
    read_git,
    read_nvml,
    refusal_reason,
    round_up_mhz,
    run_block,
    run_burn,
    run_from_cli,
    run_pilot,
    sample_gpu,
    schedule_e4b,
    smi_query,
    spawn_ollama,
    threaded_sample,
    typeperf_once,
    unload_owned,
    urllib_exchange,
    warm_5090,
)
from interlock.harness.e4b_stats import (
    StatsError,
    bootstrap_median,
    decide,
    effect,
    exact_interval,
    hodges_lehmann,
    signed_rank_cdf,
    tost,
)
from interlock.harness.ollama_serve import (
    LISTENER_SCRIPT,
    READY_INTERVAL_S,
    READY_TIMEOUT_S,
    ServeError,
    Server,
    assert_port_free,
    get_json,
    pin_install,
    read_log_segment,
    receipt_environment,
    resolve_pin,
    start_line,
    stop_process_tree,
)
from interlock.harness.spec import SpecError, load_spec, validate_spec
from interlock.harness.timing import percentile

ROOT = Path(__file__).resolve().parents[1]
E4B = ROOT / "experiments" / "e4b" / "spec.yaml"
E4 = ROOT / "experiments" / "e4" / "spec.yaml"
E5 = ROOT / "experiments" / "e5" / "spec.yaml"
PIN = "500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41"
CUDA_LINE = "load_backend: loaded CUDA backend from cuda_v13\n"
# A first load the default fake can pass. The flash and KV values are the load, not the env.
LOAD_LINES = (
    "starting llama-server --flash-attn auto\n"
    "llama_kv_cache: K (q8_0): 1.00 MiB, V (q8_0): 1.00 MiB\n"
    "Flash Attention enabled\n"
    "CUDA_Host model buffer size = 1.00 MiB\n"
    "CUDA_Host compute buffer size = 1.00 MiB\n"
    "CUDA_Host output buffer size = 1.00 MiB\n"
)


def _measurement(**overrides):
    base = {
        "clock_step_mhz": 7.5,
        "clock_tolerance_mhz": 7.5,
        "steady_band_mhz": 7.5,
        "steady_samples": 10,
        "steady_sample_ms": 200,
        "warmup_min_s": 3,
        "warmup_num_predict": 768,
        "warmup_max_attempts": 2,
        "load_refuse_s": 1.0,
        "call_gap_max_s": 1,
    }
    base.update(overrides)
    return base


def _arm(arm_id, device, *, companion=None, companion_device=None):
    arm = {
        "id": arm_id,
        "device": device,
        "engine": "ollama" if device == "5090" else "openvino",
        "model": "qwen3:8b" if device == "5090" else arm_id,
        "revision_or_digest": PIN if device == "5090" else "unrecorded",
        "task": "generate" if device == "5090" else "embed",
        "num_predict": 1024 if device == "5090" else None,
        "measure": "decode_tok_s" if device == "5090" else "latency_ms",
    }
    if companion:
        arm["companion"] = companion
        arm["companion_device"] = companion_device
    return arm


def _spec(pairs, *, repeats=1, probes=None, **measurement):
    seen = set()
    for pair in pairs:
        for key in ("alone", "paired"):
            seen.add(pair[key])
    catalogue = {
        "5090-alone": _arm("5090-alone", "5090"),
        "npu-alone": _arm("npu-alone", "NPU"),
        "igpu-alone": _arm("igpu-alone", "iGPU"),
        "5090-with-npu": _arm("5090-with-npu", "5090", companion="npu-alone", companion_device="NPU"),
        "5090-with-igpu": _arm("5090-with-igpu", "5090", companion="igpu-alone", companion_device="iGPU"),
        "npu-with-5090": _arm("npu-with-5090", "NPU", companion="5090-alone", companion_device="5090"),
        "igpu-with-5090": _arm("igpu-with-5090", "iGPU", companion="5090-alone", companion_device="5090"),
    }
    pending = list(seen)
    while pending:
        companion = catalogue[pending.pop()].get("companion")
        if companion and companion not in seen:
            seen.add(companion)
            pending.append(companion)
    arms = []
    for arm_id in (
        "5090-alone",
        "npu-alone",
        "igpu-alone",
        "5090-with-npu",
        "5090-with-igpu",
        "npu-with-5090",
        "igpu-with-5090",
    ):
        if arm_id in seen:
            arms.append(catalogue[arm_id])
    return {
        "id": "E4b",
        "band": 0.10,
        "repeats": repeats,
        "extension_repeats": 15,
        "pairs": pairs,
        "arms": arms,
        "probes": probes or [],
        "measurement": _measurement(**measurement),
        "dataset": {"seed": "The switchyard counts the same cars on every pass. ", "target_tokens": 4},
    }


PAIRS = [
    {"id": "5090-npu", "alone": "5090-alone", "paired": "5090-with-npu"},
    {"id": "5090-igpu", "alone": "5090-alone", "paired": "5090-with-igpu"},
    {"id": "npu-5090", "alone": "npu-alone", "paired": "npu-with-5090"},
    {"id": "igpu-5090", "alone": "igpu-alone", "paired": "igpu-with-5090"},
]
PROBE = {
    "id": "igpu-after-cpu-burn",
    "device": "iGPU",
    "engine": "openvino",
    "task": "nli",
    "model": "cross-encoder/nli-deberta-v3-base",
    "num_predict": None,
    "measure": "latency_ms",
}


def _clean(arm, *, num_predict, kind, value=None):
    device = arm["device"]
    predict = num_predict if isinstance(num_predict, int) else 1
    if value is None:
        value = 100.0 if device == "5090" else 10.0
    return {
        "value": value,
        "load_duration_s": 0.01 if device == "5090" else None,
        "eval_count": predict,
        "elapsed_s": 3.2 if kind == "warm" else 1.5,
        "clocks_mhz": [2000.0] * 16,
        "power_w": [400.0] * 16,
        "truncated_tokens": 0 if device in {"NPU", "iGPU"} else None,
    }


class _Server:
    def __init__(self):
        self.events = []
        self.stopped = False
        self.running = SimpleNamespace(
            argv=["ollama", "serve"],
            environment={
                "OLLAMA_HOST": "127.0.0.1:11441",
                "OLLAMA_MODELS": "models",
                "OLLAMA_DEBUG": "1",
                "PATH": r"C:\Windows\System32",
                "SYSTEMROOT": r"C:\Windows",
            },
            version_start={"version": "0.35.1"},
            version_stop=None,
        )

    def __enter__(self):
        self.events.append("start")
        return self.running

    def stop(self):
        if not self.stopped:
            self.stopped = True
            self.events.append("stop")
            self.running.version_stop = {"version": "0.35.1"}
        return self.running

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False


def _seams(server, call, *, companion=None, resamples=20):
    seams = SimpleNamespace()
    seams.server = server
    seams.trace = []
    seams.calls = []
    seams.probes = []
    seams.stops = []
    seams.starts = []
    seams.loads = []
    seams.sleeps = []
    seams.threads = []
    seams.resamples = resamples
    seams.phase = None

    def wrapped(arm, *, num_predict, measured, kind, prompt, role=None):
        seams.calls.append((kind, arm["id"], num_predict, measured, role, seams.phase))
        return call(arm, num_predict=num_predict, measured=measured, kind=kind, prompt=prompt)

    seams.call = wrapped
    seams.start_companion = lambda arm, prompt="": seams.starts.append(arm["id"])
    seams.stop_companion = lambda arm, *, kind: seams.stops.append((arm["id"], kind))
    seams.companion_report = companion or (lambda arm: _clean(arm, num_predict=arm.get("num_predict"), kind="measure"))
    seams.sample_cpu = lambda name: 110.0 if name == "performance" else 40.0
    seams.sample_gpu = lambda: (2000.0, 400.0)
    seams.sleep = lambda seconds: seams.sleeps.append(seconds)
    seams.now = lambda: 0.0
    seams.start_threads = lambda count: seams.threads.append(count)
    heartbeat = Path(tempfile.mkdtemp(prefix="e4b-heartbeat-")) / "heartbeat"
    heartbeat.write_text("ok", encoding="utf-8")
    seams.heartbeat = heartbeat
    seams.wall = time.time
    seams.card_models = lambda: []
    seams.log_segment = lambda: CUDA_LINE + LOAD_LINES

    def probe(method, path):
        seams.probes.append((method, path))
        if path == "/api/ps":
            return {
                "models": [
                    {
                        "name": "qwen3:8b",
                        "model": "qwen3:8b",
                        "size": 100,
                        "size_vram": 100,
                        "digest": PIN,
                    }
                ]
            }
        return {"version": "0.35.1"}

    seams.probe = probe
    seams.load_resident = lambda: seams.loads.append("load")
    seams.unload_resident = lambda: seams.loads.append("unload")
    seams.count_tokens = lambda text: len(text.split())
    seams.git = lambda repo: {"git_sha": "abc123", "clean_tree": True}
    seams.vram = lambda: 128
    return seams


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_the_e4b_spec_loads_with_the_pilot_tolerances():
    # R&D's amendment from pilot e4b-pilot.json at 0ea9506: the p95 within-call range 15.35 rounds up to 22.5.
    spec = load_spec(E4B)
    assert spec["id"] == "E4b"
    assert spec["measurement"]["clock_tolerance_mhz"] == 22.5
    assert spec["measurement"]["steady_band_mhz"] == 22.5
    assert spec["repeats"] == 10
    assert spec["extension_repeats"] == 15
    load_spec(E4)


def test_e4b_validation_rejects_a_changed_contract():
    data = json.loads(json.dumps(load_spec(E4B), default=str))
    data["repeats"] = 5
    with pytest.raises(SpecError, match="repeats"):
        validate_spec(data)
    data = load_spec(E4B)
    data["measurement"]["clock_tolerance_mhz"] = -1
    with pytest.raises(SpecError, match="clock_tolerance"):
        validate_spec(data)
    amended = load_spec(E4B)
    amended["measurement"]["clock_tolerance_mhz"] = 15
    amended["measurement"]["steady_band_mhz"] = 15
    validate_spec(amended)


def test_dry_run_names_the_e4b_rules_and_the_pilot(capsys):
    from interlock.cli import main
    from interlock.harness.runner import dry_run

    text = dry_run(E4B, ROOT)
    for phrase in (
        "own Ollama 127.0.0.1:11441",
        "11434 and 11490 not used",
        "GET /api/version",
        "GET /api/ps",
        "keep_alive 0",
        "until steady",
        "5090 companion is sustained before the Intel warm-up",
        "two refusals",
        "Hodges-Lehmann",
        "exact Wilcoxon",
        "TOST +/-0.10",
        "extension to 15",
        "50000",
        "20261012",
        "not a gate",
        "clock_tolerance_mhz: 22.5",
        "steady_band_mhz: 22.5",
        "interlock run experiments/e4b/spec.yaml --pilot",
        "heartbeat not read; adapters not called",
        "devices not started (--dry-run)",
        "counting sequence, shortest prefix at or above 256 tokens",
    ):
        assert phrase in text
    pilot = dry_run(E4B, ROOT, pilot=True)
    for phrase in (
        "5090-alone only",
        f"measured calls: {PILOT_CALLS}",
        "num_predict: 1024",
        "768 tokens, at least 3 s",
        "steady gate off",
        "rounded up to 7.5 MHz",
        "eval_count against num_predict",
        "not receipts/",
        "not the table",
        "devices not started (--dry-run)",
        "counting sequence, shortest prefix at or above 256 tokens",
    ):
        assert phrase in pilot
    assert main(["run", str(E4B), "--dry-run", "--root", str(ROOT)]) == 0
    assert main(["run", str(E4B), "--pilot", "--dry-run", "--root", str(ROOT)]) == 0
    printed = capsys.readouterr().out
    assert "devices not started (--dry-run)" in printed
    assert main(["run", str(E5), "--pilot", "--dry-run", "--root", str(ROOT)]) == 1
    assert "--pilot is only for E4b" in capsys.readouterr().err
    subset = dry_run(E4B, ROOT, pairs=["npu-5090", "igpu-5090"])
    assert "pairs: npu-5090, igpu-5090" in subset
    assert "5090 companion is sustained before the Intel warm-up" in subset
    assert subset.strip().endswith("devices not started (--dry-run)")
    assert main(["run", str(E4B), "--dry-run", "--root", str(ROOT), "--pairs", "not-a-pair"]) == 1
    assert "unknown pair" in capsys.readouterr().err


def test_a_full_block_refuses_null_tolerances_before_heartbeat(monkeypatch):
    # The committed spec now carries the pilot's values, so the refusal is checked on a copy with them nulled.
    from interlock.harness import e4b

    def explode(*args, **kwargs):
        raise AssertionError("live seams were built")

    monkeypatch.setattr("interlock.harness.e4b.build_live_seams", explode)
    monkeypatch.setattr(
        "interlock.harness.runner.require_heartbeat",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("heartbeat")),
    )
    for field in ("clock_tolerance_mhz", "steady_band_mhz"):
        spec = load_spec(E4B)
        spec["measurement"][field] = None
        with pytest.raises(E4bError, match="null until the pilot amendment"):
            e4b._require_amended(spec)
    e4b._require_amended(load_spec(E4B))


def test_schedule_rotates_and_the_probe_follows_igpu_5090():
    spec = _spec(PAIRS, probes=[PROBE])
    items = schedule_e4b(spec, 2)
    cold = [item["arm"] for item in items if item["kind"] == "cold"]
    assert cold[0] == "5090-alone"
    assert len(cold) == len(set(cold))
    first = [item["pair"] for item in items if item.get("repeat") == 0 and item["kind"] == "triplet"]
    second = [item["pair"] for item in items if item.get("repeat") == 1 and item["kind"] == "triplet"]
    assert first == ["5090-npu", "5090-igpu", "npu-5090", "igpu-5090"]
    assert second == ["5090-igpu", "npu-5090", "igpu-5090", "5090-npu"]
    for index, item in enumerate(items):
        if item["kind"] == "triplet" and item["pair"] == "igpu-5090":
            assert items[index + 1] == {"kind": "probe", "probe": "igpu-after-cpu-burn", "repeat": item["repeat"]}


def test_block_records_the_estimate_and_keeps_the_burn_out_of_the_verdict(tmp_path: Path):
    spec = _spec(PAIRS, probes=[PROBE])
    server = _Server()

    def call(arm, *, num_predict, measured, kind, prompt):
        payload = _clean(arm, num_predict=num_predict, kind=kind)
        if arm["id"] == "igpu-after-cpu-burn" and kind == "measure":
            payload["value"] = 999.0
        return payload

    seams = _seams(server, call)
    run_block(spec, seams, root=tmp_path, run_id="block-1", repo=tmp_path)
    rows = _rows(tmp_path / "receipts" / "block-1.jsonl")
    header = rows[0]
    assert header["kind"] == "e4b-header"
    assert header["git_sha"] == "abc123"
    assert header["clean_tree"] is True
    assert header["environment"]["OLLAMA_HOST"] == "127.0.0.1:11441"
    assert set(header["environment"]) == {"OLLAMA_HOST", "OLLAMA_MODELS", "OLLAMA_DEBUG"}
    assert "11434" not in header["environment"]["OLLAMA_HOST"]
    assert "11490" not in "".join(header["environment"].values())
    assert header["version_start"]["version"] == "0.35.1"
    assert header["version_stop"]["version"] == "0.35.1"
    assert header["vram_mib_at_start"] == 128
    assert "not gated" in header["vram_note"]
    assert header["aborted"] is False
    assert seams.loads == ["load", "unload"]
    assert seams.probes == [
        ("GET", "/api/version"),
        ("GET", "/api/ps"),
        ("GET", "/api/ps"),
        ("GET", "/api/version"),
    ]
    assert server.stopped is True
    assert not (tmp_path / "table").exists()
    for pair in (row for row in rows if row["kind"] == "e4b-pair"):
        assert pair["verdict"] == "free"
        assert pair["extension_used"] is False
        assert pair["d_i"] == [0.0]
        assert pair["bootstrap"]["decides"] is False
        assert pair["bootstrap"]["seed"] == 20261012
    probes = [row for row in rows if row["kind"] == "e4b-probe"]
    assert len(probes) == 1
    assert probes[0]["gate"] is False
    assert probes[0]["gpu_calls"] == 0
    assert probes[0]["igpu_latency_ms"] == 999.0
    assert probes[0]["utility_target"] == 40.0
    assert probes[0]["samples"]
    assert "performance" in probes[0]["samples"][0]
    assert seams.sleeps
    paired = [item for item in seams.trace if item[0] == "start_companion"]
    assert paired
    for item in paired:
        start = seams.trace.index(item)
        warm = next(index for index, later in enumerate(seams.trace) if index > start and later[0] == "warm")
        assert warm > start
    assert ("npu-alone", "unload") in seams.stops
    assert ("5090-alone", "end-generate") in seams.stops
    series = next(slot["series"] for row in rows if row["kind"] == "e4b-pair" for slot in row["slots"] if slot.get(
        "series"))
    assert series.startswith("receipts/")
    assert "sm_clock_mhz" in (tmp_path / series).read_text(encoding="utf-8")
    assert not (tmp_path / "compile-cache" / "block-1" / "5090-alone").exists()
    assert "compile-cache/block-1/5090-alone" in header["caches"]


def test_companion_starts_before_warmup_and_steady_takes_a_second_attempt(tmp_path: Path):
    spec = _spec([PAIRS[0]])
    server = _Server()
    warms = {"n": 0}

    def call(arm, *, num_predict, measured, kind, prompt):
        payload = _clean(arm, num_predict=num_predict, kind=kind)
        if kind == "warm" and arm["id"] == "5090-with-npu" and seams.phase != "cold" and warms["n"] == 0:
            warms["n"] += 1
            payload["clocks_mhz"] = [1000.0 + index * 50 for index in range(16)]
        return payload

    seams = _seams(server, call)
    run_block(spec, seams, root=tmp_path, run_id="steady", repo=tmp_path)
    warm_ids = [item for item in seams.trace if item[0] == "warm" and item[1] == "5090-with-npu"]
    assert len(warm_ids) >= 2
    start = seams.trace.index(("start_companion", "npu-alone"))
    warm = next(
        index
        for index, item in enumerate(seams.trace)
        if index > start and item[0] == "warm" and item[1] == "5090-with-npu"
    )
    second = next(
        index
        for index, item in enumerate(seams.trace)
        if index > warm and item[0] == "warm" and item[1] == "5090-with-npu"
    )
    measure = next(
        index
        for index, item in enumerate(seams.trace)
        if index > second and item[0] == "measure-start" and item[1] == "5090-with-npu"
    )
    assert start < warm < second < measure


def test_each_refusal_rule():
    flat = {"clocks_mhz": [2000.0, 2000.0], "power_w": [1.0, 1.0], "eval_count": 1024, "load_duration_s": 0.2}
    assert refusal_reason("5090", flat, num_predict=1024, clock_tolerance=7.5, steady_ok=True) is None
    assert refusal_reason("5090", {**flat, "load_duration_s": 1.0}, num_predict=1024, clock_tolerance=7.5,
        steady_ok=True) is None
    assert refusal_reason("5090", {**flat, "load_duration_s": 1.01}, num_predict=1024, clock_tolerance=7.5,
        steady_ok=True) == "load_duration_s"
    assert refusal_reason("5090", {**flat, "clocks_mhz": [2000.0, 2007.5]}, num_predict=1024, clock_tolerance=7.5,
        steady_ok=True) is None
    assert refusal_reason("5090", {**flat, "clocks_mhz": [2000.0, 2008.0]}, num_predict=1024, clock_tolerance=7.5,
        steady_ok=True) == "clock_range"
    assert refusal_reason("5090", {**flat, "clocks_mhz": [2000.0, None]}, num_predict=1024, clock_tolerance=7.5,
        steady_ok=True) == "missing_clock"
    assert refusal_reason("5090", {**flat, "power_w": [None]}, num_predict=1024, clock_tolerance=7.5,
        steady_ok=True) == "missing_power"
    assert refusal_reason("5090", flat, num_predict=1024, clock_tolerance=7.5, steady_ok=False) == "clock_not_steady"
    assert refusal_reason("5090", {**flat, "eval_count": 1023}, num_predict=1024, clock_tolerance=7.5,
        steady_ok=True) == "short_decode"
    assert refusal_reason("NPU", {"truncated_tokens": 0}, num_predict=None, clock_tolerance=None,
        steady_ok=None) is None
    assert refusal_reason("iGPU", {"truncated_tokens": 1}, num_predict=None, clock_tolerance=None,
        steady_ok=None) is None
    assert refusal_reason("NPU", {"truncated_tokens": 2}, num_predict=None, clock_tolerance=None,
        steady_ok=None) == "truncated_tokens"
    assert refusal_reason("iGPU", {}, num_predict=None, clock_tolerance=None, steady_ok=None) == "truncated_tokens"


def _run_mutated(tmp_path, mutate, *, limit, arm_id="5090-alone", phase_ok=None):
    spec = _spec([PAIRS[0]])
    state = {"n": 0}

    def call(arm, *, num_predict, measured, kind, prompt):
        payload = _clean(arm, num_predict=num_predict, kind=kind)
        if kind == "measure" and arm["id"] == arm_id and seams.phase != "cold" and (phase_ok is None or phase_ok(
            seams)):
            if state["n"] < limit:
                state["n"] += 1
                payload.update(mutate)
        return payload

    server = _Server()
    seams = _seams(server, call)
    run_block(spec, seams, root=tmp_path, run_id="mut", repo=tmp_path)
    rows = _rows(tmp_path / "receipts" / "mut.jsonl")
    return rows, seams


def test_a_refused_slot_is_rerun_once_and_two_refusals_are_inconclusive(tmp_path: Path):
    rows, seams = _run_mutated(tmp_path, {"eval_count": 3}, limit=1)
    pair = next(row for row in rows if row["kind"] == "e4b-pair")
    assert pair["verdict"] == "free"
    assert pair["refusal_events"] == 1
    assert pair["d_i"] == [0.0]
    other = tmp_path / "two"
    spec = _spec([PAIRS[0]])
    state = {"n": 0}

    def call(arm, *, num_predict, measured, kind, prompt):
        payload = _clean(arm, num_predict=num_predict, kind=kind)
        if kind == "measure" and arm["id"] == "5090-alone" and seams2.phase != "cold" and state["n"] < 2:
            state["n"] += 1
            payload["eval_count"] = 3
        return payload

    seams2 = _seams(_Server(), call)
    run_block(spec, seams2, root=other, run_id="two", repo=other)
    pair = next(row for row in _rows(other / "receipts" / "two.jsonl") if row["kind"] == "e4b-pair")
    assert pair["verdict"] == "inconclusive"
    assert pair["refusal_events"] == 2
    assert all(slot.get("role") != "paired" or slot.get("skipped") for slot in pair["slots"])


def test_truncated_tokens_and_a_companion_reload_refuse_the_slot(tmp_path: Path):
    spec = _spec([PAIRS[2]])
    state = {"n": 0}

    def call(arm, *, num_predict, measured, kind, prompt):
        payload = _clean(arm, num_predict=num_predict, kind=kind)
        if kind == "measure" and arm["id"] == "npu-alone" and seams.phase != "cold" and state["n"] < 2:
            state["n"] += 1
            payload["truncated_tokens"] = 4
        return payload

    seams = _seams(_Server(), call)
    run_block(spec, seams, root=tmp_path, run_id="trunc", repo=tmp_path)
    pair = next(row for row in _rows(tmp_path / "receipts" / "trunc.jsonl") if row["kind"] == "e4b-pair")
    assert pair["verdict"] == "inconclusive"
    assert "truncated_tokens" in pair["slots"][0]["attempt_reasons"]

    spec = _spec([PAIRS[0]])
    bad = {"left": 1}

    def report(arm):
        payload = _clean(arm, num_predict=arm.get("num_predict"), kind="measure")
        if bad["left"]:
            bad["left"] -= 1
            payload["load_duration_s"] = 5
        return payload

    def call(arm, *, num_predict, measured, kind, prompt):
        return _clean(arm, num_predict=num_predict, kind=kind)

    seams = _seams(_Server(), call, companion=report)
    root = tmp_path / "companion"
    run_block(spec, seams, root=root, run_id="comp", repo=root)
    pair = next(row for row in _rows(root / "receipts" / "comp.jsonl") if row["kind"] == "e4b-pair")
    assert pair["refusal_events"] == 1
    assert pair["verdict"] == "free"
    assert any("companion_load_duration_s" in slot.get("attempt_reasons", []) for slot in pair["slots"])


def test_a_5090_companion_is_sustained_before_the_intel_warm(tmp_path: Path):
    spec = _spec([PAIRS[2]])

    def call(arm, *, num_predict, measured, kind, prompt):
        return _clean(arm, num_predict=num_predict, kind=kind)

    seams = _seams(_Server(), call)
    run_block(spec, seams, root=tmp_path, run_id="sustained", repo=tmp_path)
    start = seams.trace.index(("start_companion", "5090-alone"))
    done = seams.trace.index(("companion-warm-done", "5090-alone"), start)
    decode = seams.trace.index(("companion-decode-start", "5090-alone"), done)
    intel = next(
        index
        for index, item in enumerate(seams.trace)
        if index > decode and item[:2] == ("warm", "npu-with-5090")
    )
    warm = next(
        index
        for index, item in enumerate(seams.trace)
        if start < index < done and item == ("warm", "5090-alone", 768)
    )
    assert warm < done < decode < intel


def test_the_judged_companion_call_spans_the_intel_measurement(tmp_path: Path):
    spec = _spec([PAIRS[2]])
    clock = {"t": 0.0}
    spans = []

    def call(arm, *, num_predict, measured, kind, prompt):
        start = clock["t"]
        payload = _clean(arm, num_predict=num_predict, kind=kind)
        if kind == "warm" and arm["id"] == "5090-alone":
            payload["clocks_mhz"] = [1000.0] * 6 + [2000.0] * 10
            payload["elapsed_s"] = 3.2
            clock["t"] += 3.2
        elif kind == "companion" and arm["id"] == "5090-alone":
            payload["clocks_mhz"] = [2100.0] * 16
            clock["t"] += 0.01
        else:
            clock["t"] += 0.04
        spans.append((kind, arm["id"], start, clock["t"]))
        return payload

    seams = _seams(_Server(), call)
    seams.now = lambda: clock["t"]
    run_block(spec, seams, root=tmp_path, run_id="judged", repo=tmp_path)
    pair = next(row for row in _rows(tmp_path / "receipts" / "judged.jsonl") if row["kind"] == "e4b-pair")
    paired = next(slot for slot in pair["slots"] if slot["role"] == "paired")
    measure = [item for item in spans if item[0] == "measure" and item[1] == "npu-with-5090"][-1]
    assert paired["refusal"] is None
    assert "companion_clock_range" not in paired["attempt_reasons"]
    assert paired["companion_call_part"] == "decode"
    assert paired["companion_clock_range_mhz"] == 0
    assert paired["companion_call_started_s"] <= measure[2]
    assert paired["companion_call_ended_s"] >= measure[3]
    warm_calls = [item for item in seams.calls if item[0] == "warm" and item[1] == "5090-alone"]
    assert warm_calls
    assert warm_calls[0][2] == 768


def test_no_covering_companion_call_refuses_companion_not_covering(tmp_path: Path, monkeypatch):
    # R&D on #92: falling back to the latest companion call would judge block 2's ramp again.
    import interlock.harness.e4b as e4b_module

    spec = _spec([PAIRS[2]])

    def call(arm, *, num_predict, measured, kind, prompt):
        payload = _clean(arm, num_predict=num_predict, kind=kind)
        if kind == "warm" and arm["id"] == "5090-alone":
            payload["clocks_mhz"] = [2000.0] * 16
            payload["elapsed_s"] = 3.2
        return payload

    monkeypatch.setattr(e4b_module, "_covering_call", lambda calls, span: None)
    seams = _seams(_Server(), call)
    reported = []
    original = seams.companion_report
    seams.companion_report = lambda companion: reported.append(companion) or original(companion)
    run_block(spec, seams, root=tmp_path, run_id="not-covering", repo=tmp_path)
    pair = next(row for row in _rows(tmp_path / "receipts" / "not-covering.jsonl") if row["kind"] == "e4b-pair")
    paired = next(slot for slot in pair["slots"] if slot["role"] == "paired")
    assert "companion_not_covering" in paired["attempt_reasons"]
    assert paired["refusal"] == "companion_not_covering"
    assert not reported


def test_an_unsteady_5090_companion_refuses_clock_not_steady(tmp_path: Path):
    spec = _spec([PAIRS[2]])

    def call(arm, *, num_predict, measured, kind, prompt):
        payload = _clean(arm, num_predict=num_predict, kind=kind)
        if kind == "warm" and arm["id"] == "5090-alone":
            payload["clocks_mhz"] = [1000.0 + index * 50 for index in range(16)]
            payload["elapsed_s"] = 3.2
        return payload

    server = _Server()
    seams = _seams(server, call)
    run_block(spec, seams, root=tmp_path, run_id="companion-unsteady", repo=tmp_path)
    pair = next(row for row in _rows(tmp_path / "receipts" / "companion-unsteady.jsonl") if row["kind"] == "e4b-pair")
    paired = next(slot for slot in pair["slots"] if slot["role"] == "paired")
    assert paired["attempt_reasons"] == ["clock_not_steady", "clock_not_steady"]
    assert paired["refusal"] == "clock_not_steady"
    assert "companion_clock_not_steady" not in paired["attempt_reasons"]
    assert not any(item[0] == "companion-decode-start" for item in seams.trace)
    assert not any(item[0] == "measure-start" and item[2] == "paired" for item in seams.trace)
    assert sum(1 for item in seams.trace if item[:2] == ("warm", "npu-with-5090")) == 1


def test_pairs_runs_only_the_named_pairs_and_records_the_subset(tmp_path: Path):
    spec = _spec(PAIRS, probes=[PROBE])

    def call(arm, *, num_predict, measured, kind, prompt):
        return _clean(arm, num_predict=num_predict, kind=kind)

    seams = _seams(_Server(), call)
    run_block(
        spec,
        seams,
        root=tmp_path,
        run_id="subset",
        repo=tmp_path,
        pairs=["igpu-5090", "npu-5090"],
    )
    rows = _rows(tmp_path / "receipts" / "subset.jsonl")
    assert rows[0]["pairs"] == ["npu-5090", "igpu-5090"]
    kinds = {row["pair"] for row in rows if row["kind"] == "e4b-pair"}
    assert kinds == {"npu-5090", "igpu-5090"}
    cold = [row for row in rows if row["kind"] == "e4b-cold"]
    assert cold[0]["arm"] == "npu-alone"
    assert cold[0]["role"] == "cold"
    assert "5090-with-npu" not in {row["arm"] for row in cold}
    for pair in (row for row in rows if row["kind"] == "e4b-pair"):
        roles = [slot["role"] for slot in pair["slots"]]
        assert "alone-before" in roles
        assert "alone-after" in roles
    assert any(row["kind"] == "e4b-probe" for row in rows)
    server = _Server()
    with pytest.raises(E4bError, match="unknown pair"):
        run_block(spec, _seams(server, call), root=tmp_path / "bad", run_id="bad", repo=tmp_path, pairs=["npu-5090",
            "no-such"])
    assert server.events == []
    with pytest.raises(E4bError, match="more than once"):
        run_block(spec, _seams(_Server(), call), root=tmp_path / "dup", run_id="dup", repo=tmp_path, pairs=["npu-5090",
            "npu-5090"])


def test_the_live_5090_companion_waits_until_the_next_decode_is_underway(tmp_path: Path):
    exe, models = _pinned_install(tmp_path)
    release = threading.Event()
    decode_entered = threading.Event()
    held = {"in": False}
    bodies = []

    def transport(method, url, body=None):
        assert ":11434" not in url and ":11490" not in url
        predict = None
        if isinstance(body, dict):
            predict = (body.get("options") or {}).get("num_predict")
            if body.get("keep_alive") == -1:
                bodies.append(predict)
                if predict != 768:
                    held["in"] = True
                    decode_entered.set()
                    assert release.wait(2)
                    held["in"] = False
        return {
            "version": "0.35.1",
            "eval_count": predict or 1,
            "eval_duration": 4_000_000_000,
            "load_duration": 10,
            "prompt_eval_count": 4,
        }

    def run(cmd):
        return json.dumps({"exe": str(exe), "models_dir": str(models)})

    spec = _spec([PAIRS[2]])
    seams = build_live_seams(
        tmp_path,
        "sustained-live",
        spec,
        pilot=False,
        run=run,
        transport=transport,
        opener=lambda method, url: transport(method, url),
        bind=lambda host, port: None,
        sampler=lambda invoke, sample, every: {**invoke(), "clocks_mhz": [1800.0] * 16, "power_w": [300.0] * 16},
        sample=lambda: (1800.0, 300.0),
    )
    arm = next(item for item in spec["arms"] if item["id"] == "5090-alone")
    seams.start_companion(arm, prompt="cars")
    try:
        assert decode_entered.wait(2)
        sustained = seams.await_sustained_companion(arm)
        assert sustained["ok"] is True
        assert held["in"] is True
        start = seams.trace.index(("warm", "5090-alone", 768))
        done = seams.trace.index(("companion-warm-done", "5090-alone"), start)
        decode = seams.trace.index(("companion-decode-start", "5090-alone"), done)
        assert start < done < decode
        assert bodies == [768, 1024]
    finally:
        release.set()
        seams.stop_companion(arm, kind="end-generate")


def test_unsteady_warmup_refuses_clock_not_steady(tmp_path: Path):
    spec = _spec([PAIRS[0]])

    def call(arm, *, num_predict, measured, kind, prompt):
        payload = _clean(arm, num_predict=num_predict, kind=kind)
        if kind == "warm" and arm["device"] == "5090" and seams.phase != "cold":
            payload["clocks_mhz"] = [1000.0 + index * 50 for index in range(16)]
        return payload

    seams = _seams(_Server(), call)
    run_block(spec, seams, root=tmp_path, run_id="unsteady", repo=tmp_path)
    pair = next(row for row in _rows(tmp_path / "receipts" / "unsteady.jsonl") if row["kind"] == "e4b-pair")
    assert pair["verdict"] == "inconclusive"
    assert "clock_not_steady" in pair["slots"][0]["attempt_reasons"]
    assert not any(item[0] == "measure-start" and item[2] == "alone-before" for item in seams.trace)


def test_extension_runs_five_more_triplets_once(tmp_path: Path):
    spec = _spec([PAIRS[0]], repeats=10)
    paired_n = {"n": 0}

    def call(arm, *, num_predict, measured, kind, prompt):
        payload = _clean(arm, num_predict=num_predict, kind=kind)
        if kind == "measure" and arm["id"] == "5090-with-npu" and seams.phase != "cold":
            sign = 1 if paired_n["n"] % 2 == 0 else -1
            paired_n["n"] += 1
            payload["value"] = 100.0 * math.exp(0.12 * sign)
        return payload

    seams = _seams(_Server(), call, resamples=30)
    run_block(spec, seams, root=tmp_path, run_id="extend", repo=tmp_path)
    pair = next(row for row in _rows(tmp_path / "receipts" / "extend.jsonl") if row["kind"] == "e4b-pair")
    assert pair["extension_used"] is True
    assert len(pair["d_i"]) == 15
    assert pair["verdict"] == "inconclusive"
    assert pair["bootstrap"]["decides"] is False


def test_one_inconclusive_pair_does_not_decide_another(tmp_path: Path):
    spec = _spec([PAIRS[2], PAIRS[3]], probes=[PROBE])
    state = {"n": 0}

    def call(arm, *, num_predict, measured, kind, prompt):
        payload = _clean(arm, num_predict=num_predict, kind=kind)
        if kind == "measure" and arm["id"] == "npu-alone" and seams.phase != "cold" and state["n"] < 2:
            state["n"] += 1
            payload["truncated_tokens"] = 9
        return payload

    seams = _seams(_Server(), call)
    run_block(spec, seams, root=tmp_path, run_id="split", repo=tmp_path)
    pairs = {row["pair"]: row for row in _rows(tmp_path / "receipts" / "split.jsonl") if row["kind"] == "e4b-pair"}
    assert pairs["npu-5090"]["verdict"] == "inconclusive"
    assert pairs["igpu-5090"]["verdict"] == "free"


def test_pilot_reports_the_rounded_range_and_stays_out_of_receipts(tmp_path: Path):
    spec = _spec([PAIRS[0]])
    ranges = []

    def call(arm, *, num_predict, measured, kind, prompt):
        payload = _clean(arm, num_predict=num_predict, kind=kind)
        if kind == "measure":
            width = float(len(ranges))
            ranges.append(width)
            payload["clocks_mhz"] = [2000.0, 2000.0 + width]
            payload["eval_count"] = 1024
        return payload

    seams = _seams(_Server(), call)
    document = run_pilot(spec, seams, root=tmp_path, run_id="pilot-1", repo=tmp_path, calls=PILOT_CALLS)
    assert document["calls"] == PILOT_CALLS
    assert document["every_decode_full_length"] is True
    expected = round_up_mhz(percentile(ranges, 95))
    assert document["clock_tolerance_mhz"] == expected
    assert document["steady_band_mhz"] == expected
    assert document["git_sha"] == "abc123"
    assert not (tmp_path / "receipts").exists()
    assert not (tmp_path / "table").exists()
    assert (tmp_path / "pilot-e4b" / "pilot-1.json").is_file()
    short = tmp_path / "short"

    def short_call(arm, *, num_predict, measured, kind, prompt):
        payload = _clean(arm, num_predict=num_predict, kind=kind)
        if kind == "measure":
            payload["eval_count"] = 4
            payload["clocks_mhz"] = [1.0, 1.0]
        return payload

    run_pilot(_spec([PAIRS[0]]), _seams(_Server(), short_call), root=short, run_id="pilot-2", repo=short, calls=3)
    body = json.loads((short / "pilot-e4b" / "pilot-2.json").read_text(encoding="utf-8"))
    assert body["every_decode_full_length"] is False
    with pytest.raises(E4bError, match="receipts"):
        pilot_destination(tmp_path / "receipts", "nope")


def test_abort_stops_the_server_and_deletes_the_cache(tmp_path: Path):
    spec = _spec([PAIRS[0]])

    def call(arm, *, num_predict, measured, kind, prompt):
        raise RuntimeError("boom")

    server = _Server()
    seams = _seams(server, call)
    with pytest.raises(RuntimeError, match="boom"):
        run_block(spec, seams, root=tmp_path, run_id="abort", repo=tmp_path)
    assert server.stopped is True
    assert "unload" in seams.loads
    assert not (tmp_path / "compile-cache" / "abort" / "5090-alone").exists()
    assert not (tmp_path / "receipts" / "abort.jsonl").exists()
    parked = next((tmp_path / "aborted").glob("abort-*"))
    header = _rows(parked / "abort.jsonl")[0]
    assert header["aborted"] is True
    assert header["run_id"] == "abort"


def test_stop_session_unloads_with_keep_alive_0():
    seen = []

    def transport(method, url, body, timeout=None):
        seen.append(body)
        return {}

    adapter = OllamaAdapter(transport=transport, device="5090")
    adapter.stop_session(StepRequest("s", "generate", "qwen3:8b", "5090", "ollama", {}))
    assert seen[0]["keep_alive"] == 0
    assert seen[0]["prompt"] == ""
    assert "companion" not in seen[0]
    fixture = json.loads((ROOT / "tests" / "fixtures" / "ollama" / "0.35.1" / "companion_stop.json").read_text(
        encoding="utf-8"))
    unload = json.loads((ROOT / "tests" / "fixtures" / "ollama" / "0.35.1" / "keep_alive_0.json").read_text(
        encoding="utf-8"))
    assert fixture["response"]["done_reason"] == "load"
    assert unload["response"]["done_reason"] == "unload"


def test_owned_server_never_posts_to_a_shared_port():
    seen = []

    def transport(method, url, body):
        seen.append((method, url, body))
        if ":11434" in url or ":11490" in url:
            raise AssertionError(url)
        return {"eval_count": 4, "eval_duration": 1_000_000_000, "load_duration": 2_000_000_000, "prompt_eval_count": 8}

    arm = _arm("5090-alone", "5090")
    with pytest.raises(E4bError, match="11434"):
        owned_generate(transport, "http://127.0.0.1:11434", arm, "prompt", 1024)
    with pytest.raises(E4bError, match="11490"):
        unload_owned(transport, "http://127.0.0.1:11490", "qwen3:8b")
    owned_generate(transport, "http://127.0.0.1:11441", arm, "prompt", 1024)
    unload_owned(transport, "http://127.0.0.1:11441", "qwen3:8b")
    assert seen[0][2]["options"]["num_predict"] == 1024
    assert seen[0][2]["keep_alive"] == -1
    assert "companion" not in seen[0][2]
    assert seen[1][2] == {"model": "qwen3:8b", "prompt": "", "stream": False, "keep_alive": 0}


def test_stats_match_the_hand_counted_cases():
    assert effect(100, 100, 90) < 0
    assert effect(10, 10, 12) > 0
    assert hodges_lehmann([1, 2, 3]) == 2
    small = exact_interval([1, 2, 3])
    assert small == {"low": 1.0, "high": 3.0, "achieved": 1.0, "k": 0}
    assert signed_rank_cdf(3, 0) == 0.125
    five = exact_interval([1, 2, 3, 4, 5])
    assert five["k"] == 1
    assert five["low"] == 1
    assert five["high"] == 5
    assert five["achieved"] == pytest.approx(0.9375)
    sums = [0]
    for rank in range(1, 11):
        sums += [item + rank for item in sums]
    level = 0.90
    best = 0
    for k in range(1, 30):
        covered = 1 - 2 * (sum(item <= k - 1 for item in sums) / 1024)
        if covered + 1e-12 >= level:
            best = k
        else:
            break
    flat = exact_interval([0.0] * 10)
    assert flat["k"] == best
    assert flat["low"] == 0
    assert flat["high"] == 0
    assert tost(-0.10, 0.10, band=0.10)["verdict"] == "free"
    assert tost(0.11, 0.2, band=0.10) == {"verdict": "not_free", "side": "above", "straddles": False}
    assert tost(-0.3, -0.11, band=0.10)["side"] == "below"
    touch = tost(0.10, 0.20, band=0.10)
    assert touch["verdict"] == "inconclusive"
    assert touch["straddles"] is False
    assert tost(-0.05, 0.15, band=0.10)["straddles"] is True
    wide = [0.12 if index % 2 == 0 else -0.12 for index in range(10)]
    called = decide(wide)
    assert called["verdict"] == "extend"
    assert called["extension_used"] is False
    again = decide(wide + wide[:5], allow_extension=False)
    assert again["verdict"] == "inconclusive"
    assert decide([0.2] * 10)["verdict"] == "not_free"
    assert decide([-0.2] * 10)["side"] == "below"
    assert decide([0.0] * 10)["verdict"] == "free"
    one = bootstrap_median([1, 2, 3, 4], resamples=200, seed=20261012)
    two = bootstrap_median([1, 2, 3, 4], resamples=200, seed=20261012)
    assert one == two
    assert one["decides"] is False
    with pytest.raises(StatsError):
        effect(0, 1, 1)
    with pytest.raises(StatsError):
        exact_interval([1], level=0)
    with pytest.raises(StatsError):
        tost(1, 0, band=0.1)
    assert signed_rank_cdf(4, -1) == 0


def test_sampling_parsers_and_the_burn_match_utility_without_a_gpu():
    assert parse_nvidia_smi("1830, 412.50 W\n") == (1830.0, 412.50)
    assert parse_nvidia_smi("[N/A], [N/A]") == (None, None)
    assert parse_typeperf(
        '(PDH-CSV 4.0),"\\\\HOST\\Processor Information(_Total)\\% Processor Performance"\n"t","123.5"\n') == 123.5
    assert sample_gpu(nvml=lambda: (_ for _ in ()).throw(RuntimeError("nvml")), smi=lambda: "10, 2") == (10.0, 2.0)
    assert sample_gpu() == (None, None)

    class Library:
        NVML_CLOCK_SM = 1

        def nvmlInit(self):
            return None

        def nvmlDeviceGetHandleByIndex(self, index):
            return index

        def nvmlDeviceGetClockInfo(self, handle, kind):
            return 1800

        def nvmlDeviceGetPowerUsage(self, handle):
            return 250000

    assert read_nvml(Library()) == (1800.0, 250.0)
    assert smi_query(lambda cmd: "9, 3\n") == "9, 3\n"
    assert typeperf_once("counter", lambda cmd: '"x","4.5"\n') == 4.5
    seen = []
    series = collect_series(sample=lambda: (1, 2), sleep=lambda seconds: seen.append(seconds), every_s=0.2,
        duration_s=0.4)
    assert series == [(1, 2), (1, 2), (1, 2)]
    assert seen == [0.2, 0.2]
    assert round_up_mhz(0) == 0
    assert round_up_mhz(7.5) == 7.5
    assert round_up_mhz(7.5 + 1e-6) == 15
    assert round_up_mhz(10) == 15
    assert round_up_mhz(15) == 15
    assert is_steady([2000.0] * 10, band=7.5, count=10) is True
    assert is_steady([1000.0 + index * 50 for index in range(16)], band=7.5, count=10) is False
    assert is_steady([1.0], band=7.5, count=10) is False
    threads = {"n": 0}
    slept = []
    burn = run_burn(
        target_utility=30,
        duration_s=2.5,
        sample_utility=lambda: 10.0 * threads["n"],
        sample_performance=lambda: 80.0,
        sleep=lambda seconds: slept.append(seconds),
        start_threads=lambda count: threads.__setitem__("n", threads["n"] + count),
    )
    assert burn["threads"] == 3
    assert burn["matched"] is True
    assert burn["gpu_calls"] == 0
    assert sum(slept) == pytest.approx(2.5)
    assert note_vram(lambda: 4)["vram_mib_at_start"] == 4
    assert note_vram(lambda: (_ for _ in ()).throw(RuntimeError("vram")))["vram_mib_at_start"] is None
    summary = pilot_summary([0, 10, 20], [True, True, False])
    assert summary["every_decode_full_length"] is False
    assert summary["clock_tolerance_mhz"] == round_up_mhz(percentile([0, 10, 20], 95))


def test_run_owned_server_lifecycle(tmp_path: Path):
    exe = tmp_path / "ollama.exe"
    exe.write_bytes(b"exe")
    models = tmp_path / "models"
    models.mkdir()
    library = exe.resolve().parent / "lib" / "ollama" / "cuda_v13" / "ggml-cuda.dll"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"cuda")
    pin = pin_install(exe, models)
    assert len(pin.cuda_sha256) == 64
    with pytest.raises(ServeError, match="executable"):
        pin_install(tmp_path / "missing.exe", models)
    argv, env = start_line(pin, base={})
    assert argv == [str(exe), "serve"]
    assert env["OLLAMA_HOST"] == "127.0.0.1:11441"
    assert env["OLLAMA_DEBUG"] == "1"
    assert set(env) == {"OLLAMA_HOST", "OLLAMA_MODELS", "OLLAMA_DEBUG"}
    assert "LocalPort 11434" in LISTENER_SCRIPT
    assert "11490" not in LISTENER_SCRIPT
    assert "Invoke-WebRequest" not in LISTENER_SCRIPT
    opened = {"n": 0}

    def refuse_open(method, url):
        opened["n"] += 1
        return {}

    with pytest.raises(ServeError, match="11434"):
        get_json("http://127.0.0.1:11434/api/version", refuse_open)
    with pytest.raises(ServeError, match="11490"):
        get_json("http://127.0.0.1:11490/api/ps", refuse_open)
    with pytest.raises(ServeError, match="probes only"):
        get_json("http://127.0.0.1:11441/api/generate", refuse_open)
    assert opened["n"] == 0

    class Sock:
        def __init__(self):
            self.closed = False

        def bind(self, address):
            raise OSError("busy")

        def close(self):
            self.closed = True

    sock = Sock()
    with pytest.raises(ServeError, match="will not bind"):
        assert_port_free("127.0.0.1", 11441, opener=lambda: sock)
    assert sock.closed is True
    with pytest.raises(ServeError, match="never used"):
        assert_port_free("127.0.0.1", 11434, opener=lambda: (_ for _ in ()).throw(AssertionError("bound")))

    killed = []
    spawned = {}

    def spawn(argv, environment, log_path):
        spawned["argv"] = argv
        spawned["env"] = environment
        return SimpleNamespace(pid=42)

    def opener(method, url):
        assert method == "GET"
        assert url.startswith("http://127.0.0.1:11441/")
        if url.endswith("/boom"):
            raise ServeError("unread")
        return {"version": "0.35.1"}

    server = Server(
        pin,
        opener=opener,
        spawn=spawn,
        stop_tree=lambda pid: killed.append(pid),
        log_path=tmp_path / "logs" / "ollama.log",
        bind=lambda host, port: None,
    )
    with pytest.raises(RuntimeError, match="abort"):
        with server:
            assert server.probe_ps()["version"] == "0.35.1"
            raise RuntimeError("abort")
    assert killed == [42]
    assert server.running.version_stop == {"version": "0.35.1"}
    assert spawned["env"]["OLLAMA_HOST"] == "127.0.0.1:11441"

    def fail_open(method, url):
        raise ServeError("down")

    def spawn_dead(argv, environment, log_path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as handle:
            handle.write(b"child exited before bind\n")
        return SimpleNamespace(pid=42, poll=lambda: 1)

    dead = Server(
        pin,
        opener=fail_open,
        spawn=spawn_dead,
        stop_tree=lambda pid: killed.append(pid),
        log_path=tmp_path / "logs" / "dead.log",
        bind=lambda host, port: None,
    )
    with pytest.raises(ServeError, match="exited before") as exited:
        dead.start()
    assert "child exited before bind" in str(exited.value)
    assert killed == [42, 42]
    commands = []
    stop_process_tree(9, system="nt", run=lambda cmd, check=False: commands.append(cmd))
    stop_process_tree(9, system="posix", run=lambda cmd, check=False: commands.append(cmd))
    assert commands[0] == ["taskkill", "/PID", "9", "/T", "/F"]
    assert commands[1] == ["kill", "--", "-9"]
    report = json.dumps({"exe": str(exe), "models_dir": str(models)})

    def listener(cmd):
        assert "11434" in cmd[-1]
        assert "11490" not in cmd[-1]
        return SimpleNamespace(stdout=report)

    resolved = resolve_pin(listener)
    assert resolved.exe == str(exe)
    flags = []
    spawn_ollama(["exe", "serve"], {"A": "1"}, tmp_path / "child.log", popen=lambda argv,
        **kwargs: flags.append(kwargs) or SimpleNamespace(pid=1), system="posix")
    spawn_ollama(["exe", "serve"], {"A": "1"}, tmp_path / "child-nt.log", popen=lambda argv,
        **kwargs: flags.append(kwargs) or SimpleNamespace(pid=1), system="nt")
    assert "creationflags" not in flags[0]
    assert "creationflags" in flags[1]


def test_live_seams_post_only_to_11441(tmp_path: Path, monkeypatch):
    exe = tmp_path / "ollama.exe"
    exe.write_bytes(b"exe")
    models = tmp_path / "models"
    models.mkdir()
    library = exe.parent / "lib" / "ollama" / "cuda_v13" / "ggml-cuda.dll"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"cuda")
    seen = []

    def transport(method, url, body=None):
        seen.append((method, url, body))
        assert ":11434" not in url and ":11490" not in url
        return {"version": "0.35.1", "eval_count": 2, "eval_duration": 1_000_000_000, "load_duration": 10,
            "prompt_eval_count": 3}

    def run(cmd):
        return json.dumps({"exe": str(exe), "models_dir": str(models)})

    spec = _spec([PAIRS[0]])
    seams = build_live_seams(
        tmp_path,
        "live",
        spec,
        pilot=True,
        run=run,
        transport=transport,
        opener=lambda method, url: transport(method, url),
        bind=lambda host, port: None,
        sampler=lambda invoke, sample, every: {**invoke(), "clocks_mhz": [1.0], "power_w": [2.0]},
        sample=lambda: (1.0, 2.0),
    )
    result = seams.call(spec["arms"][0], num_predict=1024, measured=True, kind="measure", prompt="cars")
    assert result["clocks_mhz"] == [1.0]
    assert seen[0][0] == "POST"
    assert seen[0][1] == "http://127.0.0.1:11441/api/generate"
    assert seen[0][2]["options"]["num_predict"] == 1024
    seams.probe("GET", "/api/version")
    seams.load_resident()
    assert seen[-1][2]["keep_alive"] == -1
    assert seen[-1][2]["options"]["num_predict"] == 1
    seams.unload_resident()
    assert seen[-1][2]["keep_alive"] == 0
    assert "companion" not in seen[-1][2]
    stop = threading.Event()
    posts = []

    def post(arm):
        posts.append(arm["id"])
        if len(posts) == 2:
            stop.set()
        return {"n": len(posts)}

    assert generate_loop({"id": "5090-alone"}, post, stop)["n"] == 2
    dirty = read_git(tmp_path, lambda args: "a" * 40 if "rev-parse" in args else " M spec.yaml\n")
    assert dirty["clean_tree"] is False
    clean = read_git(tmp_path, lambda args: "a" * 40 + "\n" if "rev-parse" in args else "")
    assert clean["clean_tree"] is True

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"version": "0.35.1"}'

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout=None: Response())
    assert urllib_exchange("GET", "http://127.0.0.1:11441/api/version")["version"] == "0.35.1"


def _pinned_install(tmp_path: Path):
    exe = tmp_path / "ollama.exe"
    exe.write_bytes(b"exe")
    models = tmp_path / "models"
    models.mkdir()
    library = exe.parent / "lib" / "ollama" / "cuda_v13" / "ggml-cuda.dll"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"cuda")
    return exe, models


def test_the_5090_companion_report_is_the_generate_that_finished(tmp_path: Path):
    exe, models = _pinned_install(tmp_path)
    entered = threading.Event()
    release = threading.Event()

    def transport(method, url, body=None):
        assert ":11434" not in url and ":11490" not in url
        if isinstance(body, dict) and body.get("keep_alive") == -1:
            entered.set()
            assert release.wait(2)
        return {
            "version": "0.35.1",
            "eval_count": 1024,
            "eval_duration": 1_000_000_000,
            "load_duration": 10,
            "prompt_eval_count": 4,
        }

    def run(cmd):
        return json.dumps({"exe": str(exe), "models_dir": str(models)})

    spec = _spec([PAIRS[0]])
    seams = build_live_seams(
        tmp_path,
        "companion",
        spec,
        pilot=False,
        run=run,
        transport=transport,
        opener=lambda method, url: transport(method, url),
        bind=lambda host, port: None,
        sampler=lambda invoke, sample, every: {**invoke(), "clocks_mhz": [1800.0], "power_w": [300.0]},
        sample=lambda: (1800.0, 300.0),
    )
    arm = next(item for item in spec["arms"] if item["id"] == "5090-alone")
    seams.start_companion(arm, prompt="cars")
    assert entered.wait(2)
    release.set()
    seams.stop_companion(arm, kind="end-generate")
    report = seams.companion_report(arm)
    assert report["eval_count"] == 1024
    assert report["clocks_mhz"] == [1800.0]
    assert report["power_w"] == [300.0]
    assert report["load_duration_s"] == pytest.approx(1e-8)


def test_intel_lane_reports_the_companion_truncation(tmp_path: Path):
    class Adapter:
        def __init__(self):
            self._session = None
            self.unloaded = None

        def run(self, request):
            assert request.settings["prompt"] == "cars"
            assert request.settings["static_shape"] == "1x256"
            assert request.device == "NPU"
            return SimpleNamespace(ok=True, latency_ms=12.5, resolved={"truncated_tokens": 1}, error=None)

        def start_session(self, request):
            self._session = {"box": {"truncated_tokens": 0, "call_s": 0.02}, "model": request.model}

        def stop_session(self, request):
            self._session = None

        def unload(self, model, device=None):
            self.unloaded = (model, device)

    lane = IntelLane(tmp_path, "intel", adapter=Adapter())
    arm = _arm("npu-alone", "NPU")
    arm["task"] = "embed"
    arm["static_shape"] = "1x256"
    got = lane.call(arm, num_predict=None, measured=True, kind="measure", prompt="cars")
    assert got["value"] == 12.5
    assert got["truncated_tokens"] == 1
    assert got["load_duration_s"] is None
    lane.start(arm, prompt="cars")
    lane.stop(arm)
    report = lane.report(arm)
    assert report["truncated_tokens"] == 0
    assert report["value"] == pytest.approx(20)
    lane.unload(arm)
    assert lane.adapter().unloaded == ("npu-alone", "NPU")
    bad = IntelLane(tmp_path, "bad", adapter=SimpleNamespace(run=lambda request: SimpleNamespace(ok=False,
        error="nope")))
    with pytest.raises(E4bError, match="nope"):
        bad.call(arm, num_predict=None, measured=True, kind="measure", prompt="cars")


def test_threaded_sample_keeps_readings_from_during_the_call():
    def invoke():
        time.sleep(0.03)
        return {"value": 1}

    result = threaded_sample(invoke, lambda: (9.0, 8.0), 0.005)
    assert result["value"] == 1
    assert result["clocks_mhz"]
    assert 9.0 in result["clocks_mhz"]
    assert result["power_w"][0] == 8.0


def test_warm_until_steady_and_a_short_warm_is_not_steady():
    spec = _spec([PAIRS[0]])
    attempts = {"n": 0}

    def call(arm, *, num_predict, measured, kind, prompt):
        payload = _clean(arm, num_predict=num_predict, kind=kind)
        attempts["n"] += 1
        if attempts["n"] == 1:
            payload["elapsed_s"] = 0.2
        return payload

    seams = _seams(_Server(), call)
    done = warm_5090(seams, spec["arms"][0], "prompt", spec["measurement"], enforce=True)
    assert done["ok"] is True
    assert len(done["attempts"]) == 2
    attempts["n"] = 0

    def short(arm, *, num_predict, measured, kind, prompt):
        payload = _clean(arm, num_predict=num_predict, kind=kind)
        payload["elapsed_s"] = 0.2
        return payload

    failed = warm_5090(_seams(_Server(), short), spec["arms"][0], "prompt", spec["measurement"], enforce=True)
    assert failed["ok"] is False
    assert len(failed["attempts"]) == 2


def test_pilot_requires_a_grant_and_not_an_npu_session(tmp_path: Path):
    from interlock.harness.e4b import run_from_cli

    spec = load_spec(E4B)
    with pytest.raises(GuardError, match="grant"):
        run_from_cli(
            spec,
            spec_path=E4B,
            root=tmp_path,
            repo=ROOT,
            grant=None,
            npu_session=None,
            run_id="p",
            pilot=True,
            seams=_seams(_Server(), lambda **kwargs: _clean(_arm("5090-alone", "5090"), num_predict=1024,
                kind="measure")),
        )


def test_dry_text_with_an_amended_tolerance_names_the_number():
    spec = _spec(PAIRS)
    text = dry_text(spec, "blob", pilot=False)
    assert "clock_tolerance_mhz: 7.5" in text
    assert "steady_band_mhz: 7.5" in text
    assert "heartbeat not read" in text


def _fresh_pin(tmp_path: Path):
    exe = tmp_path / "ollama.exe"
    exe.write_bytes(b"exe")
    models = tmp_path / "models"
    models.mkdir()
    library = exe.parent / "lib" / "ollama" / "cuda_v13" / "ggml-cuda.dll"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"cuda")
    return pin_install(exe, models)


def _stale_heartbeat(path: Path) -> Path:
    path.write_text("old", encoding="utf-8")
    stamp = time.time() - 10_000
    os.utime(path, (stamp, stamp))
    return path


def _resident_ps(*, size=100, vram=100, digest=PIN):
    return {
        "models": [
            {"name": "qwen3:8b", "model": "qwen3:8b", "size": size, "size_vram": vram, "digest": digest}
        ]
    }


def test_the_spawned_environment_is_inherited_without_ollama_or_llama_arg(tmp_path: Path):
    pin = _fresh_pin(tmp_path)
    base = {
        "SYSTEMROOT": r"C:\Windows",
        "PATH": r"C:\Windows\System32",
        "LOCALAPPDATA": r"C:\AppData\Local",
        "TEMP": r"C:\Temp",
        "OLLAMA_HOST": "127.0.0.1:11434",
        "OLLAMA_MODELS": r"D:\other-models",
        "OLLAMA_DEBUG": "0",
        "OLLAMA_KEEP_ALIVE": "0",
        "LLAMA_ARG_N_GPU_LAYERS": "0",
    }
    argv, env = start_line(pin, base=base)
    assert argv == [pin.exe, "serve"]
    assert env["SYSTEMROOT"] == base["SYSTEMROOT"]
    assert env["PATH"] == base["PATH"]
    assert env["LOCALAPPDATA"] == base["LOCALAPPDATA"]
    assert env["TEMP"] == base["TEMP"]
    assert env["OLLAMA_HOST"] == "127.0.0.1:11441"
    assert env["OLLAMA_MODELS"] == pin.models_dir
    assert env["OLLAMA_DEBUG"] == "1"
    assert "OLLAMA_KEEP_ALIVE" not in env
    assert "LLAMA_ARG_N_GPU_LAYERS" not in env
    assert not any(key.upper().startswith("LLAMA_ARG_") for key in env)
    recorded = {"OLLAMA_HOST", "OLLAMA_MODELS", "OLLAMA_DEBUG"}
    assert not any(key.upper().startswith("OLLAMA_") and key not in recorded for key in env)
    seen = {}

    def popen(argv, **kwargs):
        seen["env"] = kwargs["env"]
        return SimpleNamespace(pid=1)

    spawn_ollama(argv, env, tmp_path / "child.log", popen=popen, system="posix")
    assert seen["env"]["PATH"] == base["PATH"]
    assert seen["env"]["SYSTEMROOT"] == base["SYSTEMROOT"]
    assert "LLAMA_ARG_N_GPU_LAYERS" not in seen["env"]
    assert set(receipt_environment(env)) == recorded
    folded = start_line(pin, base={"ollama_host": "127.0.0.1:11434", "PATH": "kept"})
    assert folded[1]["OLLAMA_HOST"] == "127.0.0.1:11441"
    assert "ollama_host" not in folded[1]


def test_a_cpu_backend_line_refuses_the_block_and_the_pilot(tmp_path: Path):
    def call(arm, *, num_predict, measured, kind, prompt):
        raise AssertionError("the block ran after a bad backend")

    for pilot in (False, True):
        root = tmp_path / ("pilot" if pilot else "block")
        root.mkdir()
        log = root / "server.log"
        old = "load_backend: loaded CUDA backend from cuda_v13\n"
        log.write_text(old, encoding="utf-8")
        spawned = {"n": 0}

        def spawn(argv, environment, log_path, _spawned=spawned):
            _spawned["n"] += 1
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write("load_backend: loaded CPU backend from cpu\n")
            return SimpleNamespace(pid=7)

        pin = _fresh_pin(root)
        server = Server(
            pin,
            opener=lambda method, url: {"version": "0.35.1"},
            spawn=spawn,
            stop_tree=lambda pid: None,
            log_path=log,
            bind=lambda host, port: None,
        )
        seams = _seams(server, call)
        del seams.log_segment
        spec = _spec([PAIRS[0]])
        with pytest.raises(E4bError, match="cuda_v13"):
            if pilot:
                run_pilot(spec, seams, root=root, run_id="cpu", repo=root, calls=1)
            else:
                run_block(spec, seams, root=root, run_id="cpu", repo=root)
        assert spawned["n"] == 1
        segment = read_log_segment(log, len(old.encode("utf-8")))
        assert "cuda_v13" not in segment
        assert "CPU" in segment
        assert read_log_segment(root / "missing.log", 0) == ""
        assert read_log_segment(log, 10**9) == ""
        assert seams.calls == []


def test_partial_residency_refuses_the_block_and_the_pilot(tmp_path: Path):
    def call(arm, *, num_predict, measured, kind, prompt):
        raise AssertionError("the block ran on a partial load")

    for pilot in (False, True):
        root = tmp_path / ("pilot" if pilot else "block")
        server = _Server()
        seams = _seams(server, call)

        def probe(method, path, _seams=seams):
            _seams.probes.append((method, path))
            if path == "/api/ps":
                return _resident_ps(size=100, vram=40)
            return {"version": "0.35.1"}

        seams.probe = probe
        spec = _spec([PAIRS[0]])
        with pytest.raises(E4bError, match="size_vram"):
            if pilot:
                run_pilot(spec, seams, root=root, run_id="partial", repo=root, calls=1)
            else:
                run_block(spec, seams, root=root, run_id="partial", repo=root)
        assert server.events[0] == "start"
        assert seams.calls == []


def test_a_digest_mismatch_refuses_the_block_and_the_pilot(tmp_path: Path):
    def call(arm, *, num_predict, measured, kind, prompt):
        raise AssertionError("the block ran on the wrong digest")

    for pilot in (False, True):
        root = tmp_path / ("pilot" if pilot else "block")
        server = _Server()
        seams = _seams(server, call)

        def probe(method, path, _seams=seams):
            _seams.probes.append((method, path))
            if path == "/api/ps":
                return _resident_ps(digest="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")
            return {"version": "0.35.1"}

        seams.probe = probe
        spec = _spec([PAIRS[0]])
        with pytest.raises(E4bError, match="digest"):
            if pilot:
                run_pilot(spec, seams, root=root, run_id="digest", repo=root, calls=1)
            else:
                run_block(spec, seams, root=root, run_id="digest", repo=root)
        assert server.events[0] == "start"
        assert seams.calls == []
    spec = _spec(PAIRS)
    prove_owned_server(spec, segment=CUDA_LINE, ps=_resident_ps(digest="sha256:" + PIN))
    with pytest.raises(E4bError, match="size_vram"):
        prove_owned_server(spec, segment=CUDA_LINE, ps={"models": []})
    with pytest.raises(E4bError, match="cuda_v13"):
        prove_owned_server(spec, segment="weights came from cuda_v13\n", ps=_resident_ps())
    with pytest.raises(E4bError, match="size_vram"):
        prove_owned_server(spec, segment=CUDA_LINE, ps=_resident_ps(size=True, vram=True))
    lone = {"id": "other-5090", "device": "5090", "model": "qwen3:8b", "revision_or_digest": PIN}
    prove_owned_server({"arms": [lone]}, segment=CUDA_LINE, ps=_resident_ps())
    with pytest.raises(E4bError, match="no 5090"):
        ollama_pin({"arms": [{"device": "NPU"}]})
    with pytest.raises(E4bError, match="no model"):
        ollama_pin({"arms": [{"id": "5090-alone", "device": "5090", "model": ""}]})
    with pytest.raises(E4bError, match="no digest"):
        ollama_pin({"arms": [{"id": "5090-alone", "device": "5090", "model": "qwen3:8b",
            "revision_or_digest": "unrecorded"}]})
    prove_owned_server(
        spec,
        segment=CUDA_LINE,
        ps={"models": [{"model": "qwen3:8b", "size": 100, "size_vram": 100, "digest": PIN}]},
    )


def test_a_dead_heartbeat_refuses_the_block_and_the_pilot_before_spawn(tmp_path: Path, monkeypatch):
    def call(arm, *, num_predict, measured, kind, prompt):
        raise AssertionError("a dead heartbeat still ran")

    for pilot in (False, True):
        root = tmp_path / ("pilot" if pilot else "block")
        server = _Server()
        seams = _seams(server, call)
        seams.heartbeat = _stale_heartbeat(tmp_path / f"stale-{pilot}")
        seams.wall = time.time
        spec = _spec([PAIRS[0]])
        with pytest.raises(GuardError, match="heartbeat"):
            if pilot:
                run_pilot(spec, seams, root=root, run_id="stale", repo=root, calls=1)
            else:
                run_block(spec, seams, root=root, run_id="stale", repo=root)
        assert server.events == []

    fresh = tmp_path / "fresh-heartbeat"
    fresh.write_text("ok", encoding="utf-8")
    monkeypatch.setattr("interlock.harness.runner.heartbeat_path", lambda: fresh)

    def ok_call(arm, *, num_predict, measured, kind, prompt):
        return _clean(arm, num_predict=num_predict, kind=kind)

    server = _Server()
    seams = _seams(server, ok_call)
    del seams.heartbeat
    del seams.wall
    run_block(_spec([PAIRS[0]]), seams, root=tmp_path / "default-heartbeat", run_id="default-hb", repo=tmp_path)
    assert server.events[0] == "start"


def test_an_unset_heartbeat_refuses_the_block_and_the_pilot_before_spawn(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("WATCHDOG_HEARTBEAT", raising=False)
    monkeypatch.chdir(tmp_path)

    def call(arm, *, num_predict, measured, kind, prompt):
        raise AssertionError("an unset heartbeat still ran")

    for pilot in (False, True):
        root = tmp_path / ("pilot" if pilot else "block")
        server = _Server()
        seams = _seams(server, call)
        seams.heartbeat = None
        spec = _spec([PAIRS[0]])
        with pytest.raises(GuardError, match="WATCHDOG_HEARTBEAT"):
            if pilot:
                run_pilot(spec, seams, root=root, run_id="unset", repo=root, calls=1)
            else:
                run_block(spec, seams, root=root, run_id="unset", repo=root)
        assert server.events == []


def test_no_watchdog_runs_the_pilot_and_both_pilot_documents_record_it(tmp_path: Path):
    def ok_call(arm, *, num_predict, measured, kind, prompt):
        payload = _clean(arm, num_predict=num_predict, kind=kind)
        if kind == "measure":
            payload["clocks_mhz"] = [2000.0, 2001.0]
            payload["eval_count"] = 1024
        return payload

    seams = _seams(_Server(), ok_call)
    assert "watchdog" not in run_pilot(
        _spec([PAIRS[0]]), seams, root=tmp_path / "with", run_id="with", repo=tmp_path, calls=PILOT_CALLS
    )
    seams = _seams(_Server(), ok_call)
    seams.heartbeat = NO_WATCHDOG
    document = run_pilot(
        _spec([PAIRS[0]]), seams, root=tmp_path / "without", run_id="without", repo=tmp_path, calls=PILOT_CALLS
    )
    assert document["watchdog"] == NO_WATCHDOG_NOTE

    def boom(arm, *, num_predict, measured, kind, prompt):
        raise RuntimeError("boom")

    seams = _seams(_Server(), boom)
    seams.heartbeat = NO_WATCHDOG
    root = tmp_path / "aborted-run"
    with pytest.raises(RuntimeError, match="boom"):
        run_pilot(_spec([PAIRS[0]]), seams, root=root, run_id="parked", repo=root, calls=1)
    parked = next((root / "aborted").glob("parked-*")) / "parked.json"
    body = json.loads(parked.read_text(encoding="utf-8"))
    assert body["aborted"] is True
    assert body["watchdog"] == NO_WATCHDOG_NOTE


def test_a_card_that_is_not_empty_refuses_the_block_and_the_pilot_before_spawn(tmp_path: Path):
    def call(arm, *, num_predict, measured, kind, prompt):
        raise AssertionError("a loaded card still ran")

    for pilot in (False, True):
        root = tmp_path / ("pilot" if pilot else "block")
        server = _Server()
        seams = _seams(server, call)
        seams.card_models = lambda: [{"name": "qwen3:8b"}]
        spec = _spec([PAIRS[0]])
        with pytest.raises(GuardError, match="still on the card"):
            if pilot:
                run_pilot(spec, seams, root=root, run_id="card", repo=root, calls=1)
            else:
                run_block(spec, seams, root=root, run_id="card", repo=root)
        assert server.events == []
    seams = _seams(_Server(), call)
    seams.card_models = lambda: "not-a-list"
    with pytest.raises(GuardError, match="still on the card"):
        run_block(_spec([PAIRS[0]]), seams, root=tmp_path / "unreadable", run_id="unreadable", repo=tmp_path)
    missing = _seams(_Server(), call)
    del missing.card_models
    with pytest.raises(GuardError, match="still on the card"):
        run_pilot(_spec([PAIRS[0]]), missing, root=tmp_path / "no-card", run_id="no-card", repo=tmp_path, calls=1)
    assert card_models_at_ceiling(lambda: 4096, ceiling=4096) == []
    assert card_models_at_ceiling(lambda: 4097, ceiling=4096) == [4097]
    assert card_models_at_ceiling(lambda: None, ceiling=4096) == [None]
    assert card_models_at_ceiling(lambda: True, ceiling=4096) == [True]
    exe_root = tmp_path / "live"
    exe_root.mkdir()
    pin_root = _fresh_pin(exe_root)

    def listener(cmd):
        return json.dumps({"exe": pin_root.exe, "models_dir": pin_root.models_dir})

    spec = _spec([PAIRS[0]])
    seams = build_live_seams(
        exe_root,
        "ceiling",
        spec,
        pilot=True,
        run=listener,
        bind=lambda host, port: None,
        card_mib=lambda: 100,
        transport=lambda method, url, body=None: {"version": "0.35.1"},
        opener=lambda method, url: {"version": "0.35.1"},
    )
    assert seams.card_models() == []
    busy = build_live_seams(
        exe_root,
        "busy",
        spec,
        pilot=True,
        run=listener,
        bind=lambda host, port: None,
        card_mib=lambda: 5000,
        transport=lambda method, url, body=None: {"version": "0.35.1"},
        opener=lambda method, url: {"version": "0.35.1"},
    )
    assert busy.card_models() == [5000]


def test_the_pilot_counts_its_prompt_only_once_the_owned_server_is_up(tmp_path: Path):
    # Live 2026-10-12: the pilot counted on 11441 before starting it, and the count was refused (WinError 10061).
    spec = _spec([PAIRS[0]])
    server = _Server()

    def call(arm, *, num_predict, measured, kind, prompt):
        payload = _clean(arm, num_predict=num_predict, kind=kind)
        if kind == "measure":
            payload["clocks_mhz"] = [2000.0, 2000.0]
            payload["eval_count"] = 1024
        return payload

    seams = _seams(server, call)
    counted = []

    def count(text):
        assert "start" in server.events, "the prompt was counted before the run-owned server started"
        counted.append(text)
        return len(text.split())

    seams.count_tokens = count
    run_pilot(spec, seams, root=tmp_path, run_id="pilot-order", repo=tmp_path, calls=PILOT_CALLS)
    assert counted


def test_the_owned_server_waits_out_the_first_refused_version_calls(tmp_path: Path):
    pin = _fresh_pin(tmp_path)
    tries = {"n": 0}
    slept = []

    def opener(method, url):
        assert method == "GET"
        assert url == "http://127.0.0.1:11441/api/version"
        tries["n"] += 1
        if tries["n"] < 3:
            raise OSError("10061")
        return {"version": "0.35.1"}

    clock = {"t": 0.0}

    def now():
        return clock["t"]

    def sleep(seconds):
        slept.append(seconds)
        clock["t"] += seconds

    killed = []
    server = Server(
        pin,
        opener=opener,
        spawn=lambda argv, environment, log_path: SimpleNamespace(pid=7),
        stop_tree=lambda pid: killed.append(pid),
        log_path=tmp_path / "wait.log",
        bind=lambda host, port: None,
        sleep=sleep,
        now=now,
    )
    running = server.start()
    assert running.version_start == {"version": "0.35.1"}
    assert tries["n"] == 3
    assert slept == [READY_INTERVAL_S, READY_INTERVAL_S]
    assert READY_TIMEOUT_S == 30
    assert killed == []


def test_the_owned_server_stops_when_version_never_answers(tmp_path: Path):
    pin = _fresh_pin(tmp_path)
    clock = {"t": 0.0}

    def now():
        return clock["t"]

    def sleep(seconds):
        clock["t"] += seconds

    killed = []
    server = Server(
        pin,
        opener=lambda method, url: (_ for _ in ()).throw(OSError("10061")),
        spawn=lambda argv, environment, log_path: SimpleNamespace(pid=8),
        stop_tree=lambda pid: killed.append(pid),
        log_path=tmp_path / "never.log",
        bind=lambda host, port: None,
        sleep=sleep,
        now=now,
    )
    with pytest.raises(ServeError, match="within 30"):
        server.start()
    assert killed == [8]
    assert clock["t"] >= READY_TIMEOUT_S


def test_one_missing_pilot_clock_is_recorded_and_seventeen_good_calls_refuse(tmp_path: Path):
    spec = _spec([PAIRS[0]])
    state = {"n": 0}

    def call(arm, *, num_predict, measured, kind, prompt):
        payload = _clean(arm, num_predict=num_predict, kind=kind)
        if kind != "measure":
            return payload
        state["n"] += 1
        payload["eval_count"] = 1024
        payload["sample_interval_s"] = 0.2
        if state["n"] == 1:
            payload["clocks_mhz"] = [2000.0, None]
        else:
            payload["clocks_mhz"] = [2000.0, 2010.0]
        return payload

    document = run_pilot(spec, _seams(_Server(), call), root=tmp_path, run_id="one-miss", repo=tmp_path,
        calls=PILOT_CALLS)
    assert document["missing_calls"] == 1
    assert document["good_calls"] == PILOT_CALLS - 1
    assert document["calls"] == PILOT_CALLS
    assert document["pilot_clock_rule"] == PILOT_CLOCK_RULE
    assert document["sample_interval_s"] == pytest.approx(0.2)
    assert len(document["ranges_mhz"]) == PILOT_CALLS - 1

    short = tmp_path / "few"
    state["n"] = 0

    def too_few(arm, *, num_predict, measured, kind, prompt):
        payload = _clean(arm, num_predict=num_predict, kind=kind)
        if kind == "measure":
            state["n"] += 1
            if state["n"] <= PILOT_CALLS - PILOT_MIN_GOOD + 1:
                payload["clocks_mhz"] = [None]
        return payload

    with pytest.raises(E4bError, match="at least 18"):
        run_pilot(spec, _seams(_Server(), too_few), root=short, run_id="few", repo=short, calls=PILOT_CALLS)
    assert not (short / "pilot-e4b" / "few.json").exists()


def test_a_generate_timeout_and_a_probe_timeout_are_refusals(monkeypatch):
    def slow(request, timeout=None):
        raise TimeoutError(timeout)

    monkeypatch.setattr("urllib.request.urlopen", slow)
    with pytest.raises(E4bError, match="GET /api/version timed out after 10"):
        urllib_exchange("GET", "http://127.0.0.1:11441/api/version")
    with pytest.raises(E4bError, match="POST /api/generate timed out after 300"):
        urllib_exchange("POST", "http://127.0.0.1:11441/api/generate", {"prompt": "x"})
    assert urllib_exchange.probe_timeout_s == PROBE_TIMEOUT_S
    assert urllib_exchange.generate_timeout_s == GENERATE_TIMEOUT_S


def test_the_burn_stops_when_the_slot_ends_and_when_it_raises():
    burn = CpuBurn()
    burn.start(1)
    try:
        assert burn._threads and burn._threads[0].is_alive()
    finally:
        burn.stop()
    assert burn._threads == []

    stopped = {"n": 0}

    def start(count):
        start.calls.append(count)

    start.calls = []

    def stop():
        stopped["n"] += 1

    start.stop = stop
    with pytest.raises(E4bError, match="cannot be negative"):
        run_burn(
            target_utility=-1,
            duration_s=1,
            sample_utility=lambda: 0,
            sample_performance=lambda: 0,
            sleep=lambda seconds: None,
            start_threads=start,
        )
    assert stopped["n"] == 1
    assert live_sample_cpu.live is True


def test_live_sample_gpu_records_nvml_then_smi(monkeypatch):
    record_nvml = type("R", (), {"probe": None, "intervals": []})()
    monkeypatch.setattr("interlock.harness.e4b.read_nvml", lambda library: (1500.0, 80.0))
    assert live_sample_gpu(record_nvml) == (1500.0, 80.0)
    assert record_nvml.probe == "nvml"

    record_smi = type("R", (), {"probe": None, "intervals": []})()

    def no_nvml(library):
        raise RuntimeError("nvml down")

    monkeypatch.setattr("interlock.harness.e4b.read_nvml", no_nvml)
    monkeypatch.setattr("interlock.harness.e4b.smi_query", lambda run: "1800, 40")
    assert live_sample_gpu(record_smi) == (1800.0, 40.0)
    assert record_smi.probe == "smi"


def test_the_block_receipt_carries_the_server_config_and_the_probe(tmp_path: Path):
    line = 'time=t msg="server config" env="map[OLLAMA_FLASH_ATTENTION:false OLLAMA_KV_CACHE_TYPE:q8_0]"'
    running = SimpleNamespace(argv=["ollama", "serve"], environment={}, version_start={}, version_stop=None)
    record = type("R", (), {"probe": "smi", "intervals": [0.21, 0.25]})()
    seams = SimpleNamespace(
        git=lambda repo: {"git_sha": "a" * 40, "clean_tree": True},
        vram=lambda: 12,
        log_segment=lambda: line + "\n" + CUDA_LINE + LOAD_LINES,
        sample_record=record,
    )
    header = _header(_spec([PAIRS[0]]), seams, tmp_path, running, [], "prompt", aborted=False)
    assert header["flash_attention"] == "true"
    assert header["config_flash_attention"] == "false"
    assert header["kv_cache_type"] == "q8_0"
    assert header["server_config_line"] == line
    assert header["gpu_probe"] == "smi"
    assert header["sample_interval_s"] == pytest.approx(0.23)


def test_live_seams_are_not_stubs(tmp_path: Path, monkeypatch):
    pin = _fresh_pin(tmp_path)
    monkeypatch.setattr("interlock.harness.e4b.resolve_pin", lambda run: pin)

    class FakeServer:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr("interlock.harness.ollama_serve.Server", FakeServer)
    seams = build_live_seams(tmp_path, "live-defaults", _spec([PAIRS[0]]), pilot=True)
    assert isinstance(seams.sample_gpu, LiveGpuSample)
    assert seams.sample_gpu.live is True
    assert seams.sample_cpu is live_sample_cpu
    assert isinstance(getattr(seams.start_threads, "__self__", None), CpuBurn)
    assert seams.vram is read_card_mib
    assert seams.probe_timeout_s == PROBE_TIMEOUT_S == 10
    assert seams.generate_timeout_s == GENERATE_TIMEOUT_S == 300
    assert seams.transport is urllib_exchange
    assert seams.transport.probe_timeout_s == 10
    assert seams.transport.generate_timeout_s == 300


def _line_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if line)


def test_the_counting_prompt_is_the_shortest_prefix_at_the_token_target():
    text = counting_prompt(256, _line_count)
    assert text == "".join(f"{n}\n" for n in range(1, 257))
    assert _line_count(text) >= 256
    shorter = "".join(f"{n}\n" for n in range(1, 256))
    assert text.startswith(shorter)
    assert _line_count(shorter) < 256
    assert "interlock" not in text
    spec = _spec([PAIRS[0]])
    spec["dataset"]["target_tokens"] = 256
    spec["dataset"]["seed"] = "The switchyard counts the same cars on every pass. "
    assert prepare_prompt(spec, _line_count) == text


def test_every_arm_uses_the_same_counting_prompt(tmp_path: Path):
    prompts = []

    def call(arm, *, num_predict, measured, kind, prompt):
        prompts.append((arm["id"], prompt))
        return _clean(arm, num_predict=num_predict, kind=kind)

    seams = _seams(_Server(), call)

    def start(arm, prompt=""):
        prompts.append((arm["id"], prompt))
        seams.starts.append(arm["id"])

    seams.start_companion = start
    run_block(_spec([PAIRS[0]]), seams, root=tmp_path, run_id="count-prompt", repo=tmp_path)
    assert len(prompts) > 1
    assert len({prompt for _arm_id, prompt in prompts}) == 1
    assert {arm_id for arm_id, _prompt in prompts} >= {"5090-alone", "5090-with-npu", "npu-alone"}
    prompt = prompts[0][1]
    assert prompt == "1\n2\n3\n4\n"
    assert _line_count(prompt) >= 4
    assert _line_count("1\n2\n3\n") < 4
    assert "cars" not in prompt


def test_a_repeat_limit_in_the_pilot_is_a_missing_call_and_keeps_its_time(tmp_path: Path):
    spec = _spec([PAIRS[0]])
    state = {"n": 0}

    def call(arm, *, num_predict, measured, kind, prompt):
        state["n"] += 1
        if kind == "warm" and state["n"] == 1:
            raise RepeatLimit(
                "prediction aborted, token repeat limit reached",
                call_time_spent(
                    {"eval_count": 619, "eval_duration": 2_000_000_000, "load_duration": 10_000_000,
                        "prompt_eval_duration": 50_000_000, "prompt_eval_count": 256},
                    kind="warm",
                    reason="token repeat limit",
                ),
            )
        payload = _clean(arm, num_predict=num_predict, kind=kind)
        payload["prompt_eval_count"] = 4
        payload["prompt_eval_s"] = 0.05
        payload["total_s"] = payload["load_duration_s"] + payload["prompt_eval_s"] + payload["elapsed_s"]
        return payload

    document = run_pilot(spec, _seams(_Server(), call), root=tmp_path, run_id="repeat-pilot", repo=tmp_path,
        calls=PILOT_CALLS)
    assert document["missing_calls"] == 1
    assert document["good_calls"] == PILOT_CALLS - 1
    assert document["calls"] == PILOT_CALLS
    failed = document["time_spent"][0]
    assert failed["missing"] is True
    assert failed["phases"][0]["kind"] == "warm"
    assert failed["phases"][0]["reason"] == "token repeat limit"
    assert failed["phases"][0]["eval_count"] == 619
    assert failed["phases"][0]["eval_s"] == pytest.approx(2.0)
    assert failed["phases"][0]["load_s"] == pytest.approx(0.01)
    assert failed["phases"][0]["prompt_eval_s"] == pytest.approx(0.05)
    kept = document["time_spent"][1]
    assert kept["missing"] is False
    measure = kept["phases"][-1]
    assert measure["kind"] == "measure"
    assert measure["load_s"] == pytest.approx(0.01)
    assert measure["prompt_eval_s"] == pytest.approx(0.05)
    assert measure["eval_s"] == pytest.approx(1.5)
    assert (tmp_path / "pilot-e4b" / "repeat-pilot.json").is_file()


def test_a_repeat_limit_in_the_block_is_a_short_decode_rerun(tmp_path: Path):
    spec = _spec([PAIRS[0]])
    state = {"n": 0}

    def call(arm, *, num_predict, measured, kind, prompt):
        if kind == "measure":
            state["n"] += 1
            if state["n"] == 1:
                raise RepeatLimit("HTTP 500 token repeat limit reached")
        return _clean(arm, num_predict=num_predict, kind=kind)

    arm = next(item for item in spec["arms"] if item["id"] == "5090-alone")
    slot = perform_slot(
        _seams(_Server(), call),
        arm,
        arms={item["id"]: item for item in spec["arms"]},
        prompt="1\n2\n",
        measurement=spec["measurement"],
        role="alone-before",
        pair="5090-npu",
        repeat=0,
        root=tmp_path,
        run_id="rerun",
        attempts=2,
        enforce_steady=False,
        phase="repeat",
    )
    assert state["n"] == 2
    assert slot["attempt_reasons"] == ["short_decode"]
    assert slot["refusal"] is None
    assert slot["refusal_events"] == 1
    assert slot["time_spent"][0]["kind"] == "warm"
    failed = next(item for item in slot["time_spent"] if item.get("reason") == "token repeat limit")
    assert failed["kind"] == "measure"
    assert slot["time_spent"][-1]["kind"] == "measure"
    assert "reason" not in slot["time_spent"][-1]


def test_any_other_500_stays_fatal(tmp_path: Path, monkeypatch):
    import urllib.error

    def crash(request, timeout=None):
        body = b'{"error":"model runner crashed"}'
        raise urllib.error.HTTPError(request.full_url, 500, "Internal Server Error", hdrs=None,
            fp=__import__("io").BytesIO(body))

    monkeypatch.setattr("urllib.request.urlopen", crash)
    with pytest.raises(urllib.error.HTTPError):
        urllib_exchange("POST", "http://127.0.0.1:11441/api/generate", {"prompt": "x"})

    def repeat(request, timeout=None):
        body = b'{"error":"prediction aborted, token repeat limit reached","eval_count":619,"eval_duration":2000000000}'
        raise urllib.error.HTTPError(request.full_url, 500, "Internal Server Error", hdrs=None,
            fp=__import__("io").BytesIO(body))

    monkeypatch.setattr("urllib.request.urlopen", repeat)
    with pytest.raises(RepeatLimit, match="token repeat limit") as caught:
        urllib_exchange("POST", "http://127.0.0.1:11441/api/generate", {"prompt": "x"})
    assert caught.value.time_spent["eval_count"] == 619
    assert caught.value.time_spent["eval_s"] == pytest.approx(2.0)

    def call(arm, *, num_predict, measured, kind, prompt):
        raise RuntimeError("HTTP 500: model runner crashed")

    with pytest.raises(RuntimeError, match="model runner crashed"):
        run_pilot(_spec([PAIRS[0]]), _seams(_Server(), call), root=tmp_path, run_id="other-500", repo=tmp_path, calls=1)
    assert not (tmp_path / "pilot-e4b" / "other-500.json").exists()


def _writes_during_the_call(root: Path, calls: dict):
    def call(arm, *, num_predict, measured, kind, prompt):
        calls["n"] += 1
        (root / "pilot-e4b").mkdir(parents=True, exist_ok=True)
        (root / "pilot-e4b" / "during.json").write_text("{}\n", encoding="utf-8")
        (root / "logs").mkdir(parents=True, exist_ok=True)
        (root / "logs" / "during.log").write_text("log\n", encoding="utf-8")
        return _clean(arm, num_predict=num_predict, kind=kind)

    return call


def _git_clean_once_then_dirty(root: Path, calls: dict):
    """A second read would see the run's own output and report a dirty tree."""

    seen = []

    def git(repo):
        seen.append(
            {
                "calls": calls["n"],
                "pilot": (root / "pilot-e4b").exists(),
                "logs": (root / "logs").exists(),
                "cache": (root / "compile-cache").exists(),
            }
        )
        return {"git_sha": "abc123", "clean_tree": len(seen) == 1}

    return git, seen


def test_a_run_that_writes_output_keeps_the_clean_tree_from_the_start(tmp_path: Path):
    ignored = {line.strip() for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()}
    assert "pilot-e4b/" in ignored
    assert "logs/" in ignored
    assert "receipts/" not in ignored
    assert "receipts/*.jsonl" not in ignored

    spec = _spec([PAIRS[0]])
    for kind in ("pilot", "block"):
        root = tmp_path / kind
        root.mkdir()
        calls = {"n": 0}
        git, seen = _git_clean_once_then_dirty(root, calls)
        seams = _seams(_Server(), _writes_during_the_call(root, calls))
        seams.git = git
        if kind == "pilot":
            document = run_pilot(spec, seams, root=root, run_id="kept-clean", repo=root, calls=1)
            written = json.loads((root / "pilot-e4b" / "kept-clean.json").read_text(encoding="utf-8"))
            assert document["clean_tree"] is True
            assert written["clean_tree"] is True
            assert written["git_sha"] == "abc123"
        else:
            run_block(spec, seams, root=root, run_id="kept-clean", repo=root)
            header = _rows(root / "receipts" / "kept-clean.jsonl")[0]
            assert header["clean_tree"] is True
            assert header["git_sha"] == "abc123"
        assert seen == [{"calls": 0, "pilot": False, "logs": False, "cache": False}]
        assert (root / "pilot-e4b" / "during.json").is_file()
        assert (root / "logs" / "during.log").is_file()


def test_the_cli_reads_git_once_before_the_first_device_call(tmp_path: Path):
    def once(root: Path, *, pilot: bool):
        calls = {"n": 0}
        git, seen = _git_clean_once_then_dirty(root, calls)
        seams = _seams(_Server(), _writes_during_the_call(root, calls))
        seams.git = git
        spec = load_spec(E4B) if pilot else _spec([PAIRS[0]])
        result = run_from_cli(
            spec,
            spec_path=E4B,
            root=root,
            repo=ROOT,
            grant="grant",
            npu_session=None if pilot else "session",
            run_id="cli-once",
            pilot=pilot,
            seams=seams,
        )
        assert seen == [{"calls": 0, "pilot": False, "logs": False, "cache": False}]
        if pilot:
            assert result["clean_tree"] is True
            assert result["git_sha"] == "abc123"
            written = json.loads((root / "pilot-e4b" / "cli-once.json").read_text(encoding="utf-8"))
            assert written["clean_tree"] is True
        else:
            header = _rows(root / "receipts" / "cli-once.jsonl")[0]
            assert header["clean_tree"] is True
            assert header["git_sha"] == "abc123"

    once(tmp_path / "pilot", pilot=True)
    once(tmp_path / "block", pilot=False)


def test_a_dirty_tree_at_the_start_is_recorded_false(tmp_path: Path):
    def git(repo):
        return {"git_sha": "abc123", "clean_tree": False}

    spec = _spec([PAIRS[0]])
    for pilot in (False, True):
        root = tmp_path / ("pilot" if pilot else "block")
        root.mkdir()
        server = _Server()
        seams = _seams(
            server,
            lambda arm, **kwargs: _clean(arm, num_predict=kwargs["num_predict"], kind=kwargs["kind"]),
        )
        seams.git = git
        with pytest.raises(E4bError, match="porcelain"):
            if pilot:
                run_pilot(spec, seams, root=root, run_id="dirty-start", repo=root, calls=1)
            else:
                run_block(spec, seams, root=root, run_id="dirty-start", repo=root)
        assert server.events == []
        assert not (root / "receipts").exists()
        assert not (root / "pilot-e4b").exists()


def _git_completed(stdout: str):
    return SimpleNamespace(returncode=0, stdout=stdout)


def _porcelain_reader(porcelain: str):
    def run(args):
        if "rev-parse" in args:
            return _git_completed("a" * 40 + "\n")
        return _git_completed(porcelain)

    return run


def test_a_dirty_start_refuses_before_any_device_or_server(tmp_path: Path):
    home = Path.home()
    lines = [
        " M src/interlock/harness/e4b.py",
        "?? " + (home / "notes.txt").as_posix(),
        " M experiments/e4b/spec.yaml",
        "?? FOURTH-LINE-NOT-IN-REASON",
    ]
    recorded = read_git(tmp_path, _porcelain_reader("\n".join(lines) + "\n"))
    spec = _spec([PAIRS[0]])

    def call(arm, *, num_predict, measured, kind, prompt):
        raise AssertionError("a device call ran")

    for pilot in (False, True):
        root = tmp_path / ("pilot-dirty" if pilot else "block-dirty")
        root.mkdir()
        server = _Server()
        seams = _seams(server, call)
        seams.git = lambda repo, _recorded=recorded: dict(_recorded)
        with pytest.raises(E4bError, match="e4b.py") as caught:
            if pilot:
                run_pilot(spec, seams, root=root, run_id="dirty", repo=root, calls=1)
            else:
                run_block(spec, seams, root=root, run_id="dirty", repo=root)
        assert server.events == []
        assert seams.loads == []
        assert seams.calls == []
        text = str(caught.value)
        assert "spec.yaml" in text
        assert "FOURTH-LINE-NOT-IN-REASON" not in text
        assert str(home) not in text
        assert not (root / "compile-cache").exists()
        assert not (root / "pilot-e4b").exists()
        assert not (root / "receipts").exists()


def test_a_clean_start_runs_the_block_and_the_pilot(tmp_path: Path):
    recorded = read_git(tmp_path, _porcelain_reader(""))
    assert recorded["clean_tree"] is True
    spec = _spec([PAIRS[0]])

    def call(arm, *, num_predict, measured, kind, prompt):
        return _clean(arm, num_predict=num_predict, kind=kind)

    for pilot in (False, True):
        root = tmp_path / ("pilot-clean" if pilot else "block-clean")
        root.mkdir()
        server = _Server()
        seams = _seams(server, call)
        seams.git = lambda repo, _recorded=recorded: dict(_recorded)
        if pilot:
            document = run_pilot(spec, seams, root=root, run_id="clean", repo=root, calls=1)
            assert document["clean_tree"] is True
        else:
            run_block(spec, seams, root=root, run_id="clean", repo=root)
            assert _rows(root / "receipts" / "clean.jsonl")[0]["clean_tree"] is True
        assert "start" in server.events


def _e4b_log(*parts: str) -> str:
    return "\n".join(part.strip("\n") for part in parts if part) + "\n"


def test_flash_attention_disabled_on_the_load_line_is_false():
    from interlock.harness.e4b import _flash_flag

    assert _flash_flag("enabled") == "true"
    assert _flash_flag("disabled") == "false"
    assert _flash_flag(None) is None


def test_a_recorded_load_line_sets_flash_attention_true(tmp_path: Path):
    config = (ROOT / "tests" / "fixtures" / "e5" / "server-config-11434.txt").read_text(encoding="utf-8")
    load = (ROOT / "tests" / "fixtures" / "e5" / "load-lines-11434.txt").read_text(encoding="utf-8")
    assert "OLLAMA_FLASH_ATTENTION:false" in config
    assert "Flash Attention enabled" in load
    segment = _e4b_log(config, CUDA_LINE, load)
    server = _Server()
    seams = _seams(server, lambda arm, **kwargs: _clean(arm, num_predict=kwargs["num_predict"], kind=kwargs["kind"]))
    seams.log_segment = lambda: segment
    run_block(_spec([PAIRS[0]]), seams, root=tmp_path, run_id="flash-on", repo=tmp_path)
    header = _rows(tmp_path / "receipts" / "flash-on.jsonl")[0]
    assert header["flash_attention"] == "true"
    assert header["config_flash_attention"] == "false"
    assert header["kv_cache_type"] == "f16"


def test_a_missing_load_line_refuses_before_the_first_measured_slot(tmp_path: Path):
    config = (ROOT / "tests" / "fixtures" / "e5" / "server-config-11434.txt").read_text(encoding="utf-8")
    segment = _e4b_log(config, CUDA_LINE)
    assert "flash-attn" not in segment
    spec = _spec([PAIRS[0]])
    measured_slots = []

    def call(arm, *, num_predict, measured, kind, prompt):
        if kind == "measure":
            measured_slots.append(arm["id"])
        return _clean(arm, num_predict=num_predict, kind=kind)

    for pilot in (False, True):
        root = tmp_path / ("pilot-missing" if pilot else "block-missing")
        root.mkdir()
        before = len(measured_slots)
        server = _Server()
        seams = _seams(server, call)
        seams.log_segment = lambda: segment
        with pytest.raises(E4bError, match="first load"):
            if pilot:
                run_pilot(spec, seams, root=root, run_id="missing-load", repo=root, calls=1)
            else:
                run_block(spec, seams, root=root, run_id="missing-load", repo=root)
        assert measured_slots[before:] == []
