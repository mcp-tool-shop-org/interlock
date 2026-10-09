"""Harness, adapters, and the E4 and E5 specs.

Device runs are fakes. Dry-run does not stat the heartbeat or call an adapter.
"""

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from switchyard.adapters.base import AdapterError, StepRequest, StepResult
from switchyard.adapters.offrig_calibrate import OffrigCalibrateAdapter, plan_argv, scratch_project
from switchyard.adapters.ollama import (
    OllamaAdapter,
    context_length,
    default_transport,
    generate_body,
    quantization_level,
    refuse_cloud,
    think_value,
    url_for,
)
from switchyard.adapters.registry import for_engine
from switchyard.guards import GuardError
from switchyard.harness.e5 import (
    PROMPT_RATIO,
    SHARED_GROWTH_MIB,
    Observation,
    SearchError,
    fit_prompt,
    is_spill,
    over_margin,
    search,
)
from switchyard.harness.lock import require_unmodified
from switchyard.harness.receipts import write_run
from switchyard.harness.runner import RunnerError, check_flags, dry_run, run_interleaved
from switchyard.harness.spec import SpecError, load_spec, validate_spec
from switchyard.harness.timing import (
    TimingError,
    aba_call,
    percentile,
    relative_spread,
    schedule,
    schedule_aba,
    summarize,
    within_band,
)
from switchyard.receipt import load_jsonl

ROOT = Path(__file__).resolve().parents[1]
E4 = ROOT / "experiments" / "e4" / "spec.yaml"
E5 = ROOT / "experiments" / "e5" / "spec.yaml"


class Scripted:
    def __init__(self, samples, *, sessions=False):
        self.samples = samples
        self.calls = []
        self.requests = []
        self.unloaded = []
        self.sessions = []
        self._sessions = sessions

    def run(self, request):
        self.calls.append(request.step_id)
        self.requests.append(request)
        return self.samples[request.step_id].pop(0)

    def unload(self, model, device=None):
        self.unloaded.append((model, device))

    def start_session(self, request):
        if not self._sessions:
            raise AssertionError("start was not expected")
        self.sessions.append(("start", request.step_id, request.device))

    def stop_session(self, request):
        self.sessions.append(("stop", request.step_id, request.device))


class MeasureOnly:
    """A run/unload adapter with no companion session methods."""

    def __init__(self, samples):
        self.samples = samples
        self.calls = []

    def run(self, request):
        self.calls.append(request.step_id)
        return self.samples[request.step_id].pop(0)

    def unload(self, model, device=None):
        return None


def _result(latency, throughput=None, ok=True, error=None, **kwargs):
    unit = kwargs.pop("unit", "tok/s" if throughput is not None else None)
    return StepResult(
        ok=ok,
        latency_ms=latency,
        throughput=throughput,
        throughput_unit=unit if throughput is not None else None,
        error=error,
        **kwargs,
    )


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


def _spec():
    return {
        "id": "T",
        "procedure": "interleaved",
        "question": "Does interleaving keep cold out of the warm percentile?",
        "decision": "The warm percentile ignores cold and warm-up samples.",
        "needs": {"grant": True, "npu_session": True, "cpu_window": False},
        "stop_rules": ["a failed step stops the sequence"],
        "dataset": {"name": "fixture"},
        "metrics": ["latency_ms"],
        "warmups": 1,
        "repeats": 2,
        "band": 0.1,
        "compile_cache": "reuse",
        "rest_s": 900,
        "arms": [_arm("A", "5090"), _arm("B", "NPU")],
    }


def _bind(repo: Path, spec: dict) -> Path:
    path = repo / "experiments" / "t" / "spec.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8", newline="\n")
    _commit(repo)
    return path


def _invoke(repo: Path, spec_path: Path, adapters: dict, **kwargs):
    return run_interleaved(
        adapters,
        grant=kwargs["grant"] if "grant" in kwargs else "grant-t",
        npu_session=kwargs["npu_session"] if "npu_session" in kwargs else "session-t",
        cpu_window=kwargs.get("cpu_window"),
        root=kwargs.get("root", repo),
        repo=repo,
        spec_path=spec_path,
        run_id=kwargs.get("run_id", "run-1"),
        date="2026-10-09",
    )


def _resident(peak, *, failed=False, shared=1000, prompt=20.0, tok_s=10, ttft=100):
    return Observation(
        resident=not failed,
        peak_mib=31000 if failed else peak,
        tok_s=tok_s,
        ttft_ms=ttft,
        shared_mib=shared,
        prompt_tok_s=prompt,
    )


def test_schedule_interleaves_and_percentiles_ignore_warmups():
    slots = schedule(["A", "B"], warmups=1, repeats=2)
    assert [(slot.phase, slot.step_id) for slot in slots] == [
        ("cold", "A"),
        ("cold", "B"),
        ("warmup", "A"),
        ("warmup", "B"),
        ("repeat", "A"),
        ("repeat", "B"),
        ("repeat", "A"),
        ("repeat", "B"),
    ]
    assert percentile([10, 30], 50) == 20
    assert percentile([10, 30], 95) == 29
    assert within_band(100, 110, band=0.1) is True
    assert within_band(100, 111, band=0.1) is False
    assert within_band(100, 90, band=0.1) is True
    assert within_band(0, 0, band=0.1) is True
    assert within_band(0, 1, band=0.1) is False


def test_a_failed_step_stops_and_writes_nothing(tmp_path: Path):
    repo = tmp_path / "repo"
    spec_path = _bind(repo, _spec())
    script = Scripted(
        {
            "A": [_result(100)],
            "B": [_result(1, ok=False, error="stopped")],
        }
    )
    with pytest.raises(RunnerError, match="stopped"):
        _invoke(repo, spec_path, {"fake": script})
    assert script.calls == ["A", "B"]
    assert script.unloaded == [("qwen3:8b", "5090"), ("qwen3:8b", "NPU")]
    assert not (repo / "receipts").exists()


def test_repeats_build_a_receipt_and_keep_cold_separate(tmp_path: Path):
    repo = tmp_path / "repo"
    spec_path = _bind(repo, _spec())
    script = Scripted(
        {
            "A": [_result(100, 1), _result(80, 1), _result(10, 4), _result(30, 8)],
            "B": [_result(5, 1), _result(4, 1), _result(2, 3), _result(4, 5)],
        }
    )
    (repo / "table").mkdir()
    rows = _invoke(repo, spec_path, {"fake": script})
    by_id = {row["id"]: row for row in rows}
    arm_a = by_id["T/A/run-1"]
    assert arm_a["latency"]["cold_ms"] == 100
    assert arm_a["latency"]["warm_p50_ms"] == 20
    assert arm_a["latency"]["warm_p95_ms"] == 29
    assert arm_a["latency"]["throughput_unit"] == "tok/s"
    assert arm_a["provisional"] is False
    assert arm_a["settings"]["grant"] == "grant-t"
    assert arm_a["settings"]["static_shape"] is None
    assert arm_a["settings"]["compile_cache"] == "reuse"
    assert arm_a["settings"]["rest_s"] == 900
    assert "keep_alive" not in arm_a["settings"]
    assert arm_a["settings"]["environment"]["nvidia_driver"] == "unrecorded"
    assert arm_a["provenance"] == {"experiment_spec": "experiments/t/spec.yaml", "run_id": "run-1"}
    assert arm_a["model_family"] == "qwen"
    stored = load_jsonl(repo / "table" / "routing.jsonl")
    assert {row["id"] for row in stored} == {row["id"] for row in rows}
    assert script.calls[:4] == ["A", "B", "A", "B"]
    with pytest.raises(GuardError):
        _invoke(repo, spec_path, {"fake": script}, grant=None, run_id="run-2")


def test_e4_and_e5_specs_pin_the_rules():
    e4 = load_spec(E4)
    e5 = load_spec(E5)
    assert e4["band"] == 0.10
    assert e4["repeats"] == 5 and e4["warmups"] == 1
    assert e4["compile_cache"] == "fresh"
    assert e4["order"] == "aba"
    assert e4["rest_s"] == 900
    assert "choice_rule" not in e4 and "choice_rule" not in e5
    assert [pair["id"] for pair in e4["pairs"]] == ["5090-npu", "5090-igpu", "npu-5090", "igpu-5090"]
    arms = {arm["id"]: arm for arm in e4["arms"]}
    assert arms["5090-with-npu"]["companion"] == "npu-alone"
    assert arms["npu-with-5090"]["companion"] == "5090-alone"
    assert arms["5090-with-igpu"]["companion"] == "igpu-alone"
    assert arms["igpu-with-5090"]["companion"] == "5090-alone"
    for arm in e4["arms"]:
        companion = arm.get("companion")
        if companion:
            assert arms[companion]["device"] == arm["companion_device"]
    slots = schedule_aba(e4["pairs"], warmups=e4["warmups"], repeats=e4["repeats"])
    assert len(slots) == 79
    assert {slot.step_id for slot in slots if slot.phase == "cold"} == set(arms)
    assert "half" in e4["decision"]
    assert "eval_count / eval_duration" in " ".join(e4["stop_rules"])
    nli = [arm for arm in e4["arms"] if arm["task"] == "nli"]
    assert nli and all(arm["batch"] == 1 for arm in nli)
    assert any(arm["model"] == "BAAI/bge-base-en-v1.5" and arm["device"] == "NPU" for arm in e4["arms"])
    bge = next(arm for arm in e4["arms"] if arm["model"] == "BAAI/bge-base-en-v1.5")
    assert bge["revision_or_digest"] == "unrecorded"
    assert e5["abort_mib"] == 30500
    assert e5["ceiling_mib"] == 31200
    assert e5["grain"] == 2048
    assert e5["spill_shared_mib"] == SHARED_GROWTH_MIB == 256
    assert e5["spill_prompt_ratio"] == PROMPT_RATIO == 0.5
    assert "spill_tok_ratio" not in e5 and "spill_ttft_ratio" not in e5
    assert e5["rest_s"] == 900
    assert len(e5["models"]) == 6
    assert e5["shift"] is False and e5["truncate"] is False
    rules = " ".join(e5["stop_rules"])
    assert r"\GPU Process Memory(*)\Shared Usage" in rules
    assert "/api/show" in rules
    assert "prompt_eval_count / prompt_eval_duration" in rules
    broken = dict(e4)
    broken["arms"] = [dict(arm) for arm in e4["arms"]]
    broken["arms"][2] = dict(broken["arms"][2], batch=8)
    with pytest.raises(SpecError, match="batch"):
        validate_spec(broken)
    with pytest.raises(SpecError, match="30500"):
        validate_spec({**e5, "abort_mib": 31200})


def test_an_edited_spec_is_refused_and_dry_run_does_not_touch_devices(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    for name in ("e4", "e5"):
        source = ROOT / "experiments" / name / "spec.yaml"
        target = repo / "experiments" / name / "spec.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    _commit(repo)
    monkeypatch.setattr(
        "switchyard.guards.require_heartbeat",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("heartbeat was read")),
    )
    e4_text = dry_run(repo / "experiments" / "e4" / "spec.yaml", repo)
    e5_text = dry_run(repo / "experiments" / "e5" / "spec.yaml", repo)
    assert "heartbeat not read" in e4_text
    assert "slots: 79" in e4_text
    assert "order: aba" in e4_text
    assert "compile_cache: fresh" in e4_text
    assert "rest_s: 900" in e4_text
    assert "abort_mib: 30500" in e5_text
    assert "spill_shared_mib: 256" in e5_text
    assert "spill_prompt_ratio: 0.5" in e5_text
    assert "quant is read from /api/show at run time" in e5_text
    assert "not launched" not in e4_text
    spec = repo / "experiments" / "e4" / "spec.yaml"
    spec.write_text(spec.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8", newline="\n")
    with pytest.raises(SpecError, match="edited after commit"):
        require_unmodified(repo, spec)


def test_the_command_dry_runs_and_refuses_to_start_a_device(tmp_path: Path):
    from switchyard.cli import main

    repo = tmp_path / "repo"
    target = repo / "experiments" / "e5" / "spec.yaml"
    target.parent.mkdir(parents=True)
    target.write_text(E5.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    _commit(repo)
    assert main(["run", str(target), "--dry-run", "--root", str(repo)]) == 0
    assert main(["run", str(target), "--root", str(repo), "--grant", "g"]) == 1
    assert main(["run", str(target), "--root", str(repo)]) == 1


def test_ollama_stays_on_loopback_and_refuses_cloud():
    assert url_for("5090").endswith(":11434")
    assert url_for("CPU").endswith(":11490")
    with pytest.raises(AdapterError):
        url_for("NPU")
    with pytest.raises(AdapterError):
        refuse_cloud("qwen3:8b-cloud")
    request = StepRequest("s", "generate", "qwen3:8b", "5090", "ollama", {"num_predict": 1, "shift": False, "truncate": False, "thinking_level": "off"})
    body = generate_body(request)
    assert body["shift"] is False and body["truncate"] is False and body["think"] is False
    seen = {}

    def transport(method, url, payload):
        seen["url"] = url
        seen["body"] = payload
        if url.endswith("/api/ps"):
            return {"models": []}
        if payload and payload.get("keep_alive") == 0:
            return {}
        return {"eval_count": 4, "eval_duration": 2_000_000_000, "total_duration": 3_000_000}
        # 3 ms total, 2 tok/s

    adapter = OllamaAdapter(transport=transport)
    result = adapter.run(request)
    assert result.ok and result.throughput == 2
    assert seen["url"].startswith("http://127.0.0.1:11434")
    adapter.unload("qwen3:8b")
    assert seen["body"]["keep_alive"] == 0
    assert adapter.loaded_models() == []
    bad = OllamaAdapter(transport=lambda *args: {"error": "nope"})
    assert bad.run(request).ok is False


def test_offrig_calibrate_is_planned_and_not_launched():
    spec = {
        "id": "cal",
        "model": "qwen3:8b",
        "url": "http://127.0.0.1:11434",
        "gold": "experiments/gold.jsonl",
        "think": "off",
        "num_predict": 32,
        "gpu_cost_hr": 0,
    }
    argv = plan_argv(spec)
    assert argv[:3] == ["offrig", "verify", "calibrate"]
    assert "E:/AI/switchyard-scratch/cal" in argv
    assert "0" in argv
    assert "pod" not in " ".join(argv)
    with pytest.raises(AdapterError):
        plan_argv({**spec, "url": "https://example.test"})
    with pytest.raises(AdapterError):
        plan_argv({**spec, "gpu_cost_hr": 1})
    with pytest.raises(AdapterError):
        plan_argv({**spec, "model": "foo:cloud"})
    launched = []
    adapter = OffrigCalibrateAdapter()
    request = StepRequest("cal", "verify", "qwen3:8b", "5090", "offrig-calibrate", spec)
    with pytest.raises(AdapterError, match="not launched"):
        adapter.run(request)
    OffrigCalibrateAdapter(executor=lambda argv: launched.append(argv) or 0).run(request)
    assert launched and launched[0][0] == "offrig"


def test_openvino_is_a_plug_in_and_not_a_fallback():
    # The plug-in landed with PR #5 (Intel lane). The no-fallback half is unchanged.
    from switchyard.adapters.openvino import adapter as ov_adapter

    assert isinstance(for_engine("openvino"), type(ov_adapter()))
    with pytest.raises(AdapterError, match="unknown engine"):
        for_engine("cuda")


def test_fill_card_doubles_bisects_and_steps_back():
    def climb(ctx, kind):
        return _resident(30000, failed=ctx >= 16384)

    found = search(default_ctx=4096, probe=climb, abort_mib=30500)
    assert found.max_ctx == 14336
    assert found.limit == "margin"
    assert ("load", 16384) in found.calls
    assert all(ctx <= 16384 for _, ctx in found.calls)
    assert ("confirm", 14336) in found.calls
    assert over_margin(Observation(True, 30500, 1, 1), 30500) is False
    assert over_margin(Observation(True, 30501, 1, 1), 30500) is True

    def spill_top(ctx, kind):
        shared = 1000 + 257 if kind == "confirm" and ctx >= 8192 else 1000
        return _resident(20000, shared=shared)

    slowed = search(default_ctx=4096, probe=spill_top, abort_mib=30500, max_ctx=8192)
    assert slowed.max_ctx == 6144
    assert slowed.limit == "max_ctx"
    assert ("confirm", 6144) in slowed.calls

    def huge_ttft(ctx, kind):
        return _resident(20000, tok_s=1, ttft=1_000_000)

    held = search(default_ctx=4096, probe=huge_ttft, abort_mib=30500, max_ctx=8192)
    assert held.max_ctx == 8192
    assert held.ttft_ms == 1_000_000
    assert held.tok_s == 1

    def mid_fail(ctx, kind):
        return _resident(30000, failed=ctx >= 12288)

    bisected = search(default_ctx=4096, probe=mid_fail, abort_mib=30500)
    assert bisected.max_ctx == 10240
    assert ("load", 12288) in bisected.calls

    def odd_grid(ctx, kind):
        return _resident(20000, failed=ctx >= 8192)

    snapped = search(default_ctx=4096, probe=odd_grid, abort_mib=30500, grain=3000, max_ctx=8192)
    assert snapped.max_ctx == 7096

    nowhere = search(default_ctx=4096, probe=lambda ctx, kind: Observation(False, 1), abort_mib=30500)
    assert nowhere.max_ctx is None
    assert nowhere.reason == "default context is not resident"
    assert nowhere.limit == "margin"
    assert fit_prompt("ab", 4, len) == "abab"
    with pytest.raises(SearchError):
        fit_prompt("", 1, len)
    with pytest.raises(SearchError):
        fit_prompt("ab", 0, len)
    with pytest.raises(SearchError, match="never reaches"):
        fit_prompt("x" * 600_000, 1, lambda _text: 0)


def test_search_spill_and_schedule_refuse_bad_inputs():
    reference = _resident(1, shared=1000, prompt=20)
    assert is_spill(Observation(True, 1), reference, shared_growth_mib=256, prompt_ratio=0.5) is True
    assert is_spill(Observation(True, 1, shared_mib=1000), reference, shared_growth_mib=256, prompt_ratio=0.5) is True
    assert is_spill(_resident(1, shared=1256, prompt=20, tok_s=1, ttft=9_999), reference, shared_growth_mib=256, prompt_ratio=0.5) is False
    assert is_spill(_resident(1, shared=1257, prompt=20), reference, shared_growth_mib=256, prompt_ratio=0.5) is True
    assert is_spill(_resident(1, shared=1000, prompt=10), reference, shared_growth_mib=256, prompt_ratio=0.5) is False
    assert is_spill(_resident(1, shared=1000, prompt=9), reference, shared_growth_mib=256, prompt_ratio=0.5) is True
    assert over_margin(Observation(True, None, error="lost"), 30500) is True
    assert over_margin(Observation(True, None), 30500) is True

    def confirm_fails(ctx, kind):
        if kind == "confirm":
            return _resident(20000, shared=2000, prompt=1)
        return _resident(20000)

    refused = search(default_ctx=4096, probe=confirm_fails, abort_mib=30500, max_ctx=4096)
    assert refused.max_ctx is None
    assert refused.reason == "no confirmed window"
    with pytest.raises(SearchError):
        search(default_ctx=0, probe=confirm_fails, abort_mib=30500)
    with pytest.raises(SearchError):
        search(default_ctx=4096, probe=confirm_fails, abort_mib=0)
    with pytest.raises(SearchError):
        search(default_ctx=4096, probe=confirm_fails, abort_mib=30500, shared_growth_mib=-1)
    with pytest.raises(SearchError):
        search(default_ctx=4096, probe=confirm_fails, abort_mib=30500, prompt_ratio=0)
    with pytest.raises(SearchError):
        search(default_ctx=4096, probe=confirm_fails, abort_mib=30500, prompt_ratio=1.1)

    with pytest.raises(TimingError):
        schedule([], warmups=0, repeats=1)
    with pytest.raises(TimingError):
        schedule(["A"], warmups=-1, repeats=1)
    with pytest.raises(TimingError):
        percentile([], 50)
    with pytest.raises(TimingError):
        percentile([1], 101)
    assert percentile([7], 50) == 7
    with pytest.raises(TimingError):
        within_band(1, 1, band=-0.1)
    empty = summarize(None, [], [])
    assert empty.warm_p50_ms is None and empty.throughput_unit is None
    items = summarize(1, [2, 4], [8, 10], throughput_unit="items/s")
    assert items.throughput == 9 and items.throughput_unit == "items/s"
    quiet = summarize(1, [], [], throughput_unit="tok/s")
    assert quiet.throughput is None and quiet.throughput_unit is None
    within = aba_call([100, 100], [100, 100], [110, 110], band=0.1)
    assert within["call"] == "within"
    outside = aba_call([100, 100], [100, 100], [120, 120], band=0.1)
    assert outside["call"] == "outside"
    noisy = aba_call([97, 103], [97, 103], [100, 100], band=0.1)
    assert noisy["call"] == "inconclusive"
    exact = aba_call([95, 105], [95, 105], [100, 100], band=0.2)
    assert exact["call"] == "within"
    assert exact["alone_spread"] <= 0.2 / 2
    assert aba_call([], [1], [1], band=0.1)["call"] == "inconclusive"
    blown = aba_call([0, 0], [0, 1], [0], band=0.1)
    assert blown["call"] == "inconclusive" and blown["alone_spread"] is None
    assert relative_spread([0, 0]) == 0
    with pytest.raises(TimingError):
        relative_spread([])
    with pytest.raises(TimingError):
        schedule_aba([], warmups=0, repeats=1)


def test_spec_rejects_a_contract_that_is_not_pre_registered(tmp_path: Path):
    good = _spec()
    assert validate_spec(good)["id"] == "T"
    with pytest.raises(SpecError, match="mapping"):
        validate_spec([])
    missing = dict(good)
    del missing["question"]
    with pytest.raises(SpecError, match="question"):
        validate_spec(missing)
    with pytest.raises(SpecError, match="unknown procedure"):
        validate_spec({**good, "procedure": "live"})
    with pytest.raises(SpecError, match="needs"):
        validate_spec({**good, "needs": []})
    with pytest.raises(SpecError, match="cpu_window"):
        validate_spec({**good, "needs": {"grant": True, "npu_session": False, "cpu_window": "yes"}})
    with pytest.raises(SpecError, match="stop_rules"):
        validate_spec({**good, "stop_rules": []})
    with pytest.raises(SpecError, match="warmups"):
        validate_spec({**good, "warmups": -1})
    with pytest.raises(SpecError, match="repeats"):
        validate_spec({**good, "repeats": 0})
    with pytest.raises(SpecError, match="band"):
        validate_spec({**good, "band": True})
    with pytest.raises(SpecError, match="arms"):
        validate_spec({**good, "arms": []})
    with pytest.raises(SpecError, match="id"):
        validate_spec({**good, "arms": [{}]})
    duplicated = [_arm("A", "5090"), _arm("A", "CPU")]
    with pytest.raises(SpecError, match="duplicate"):
        validate_spec({**good, "arms": duplicated})
    nameless = _arm("A", "5090")
    del nameless["model"]
    with pytest.raises(SpecError, match="model"):
        validate_spec({**good, "arms": [nameless]})
    with pytest.raises(SpecError, match="device"):
        validate_spec({**good, "arms": [_arm("A", "GPU.1")]})
    legacy = _arm("A", "5090")
    legacy["task"] = "verify"
    with pytest.raises(SpecError, match="exact routing task"):
        validate_spec({**good, "arms": [legacy]})

    card = load_spec(E5)
    with pytest.raises(SpecError, match="2048"):
        validate_spec({**card, "grain": 1024})
    with pytest.raises(SpecError, match="confirm_prompt_fraction"):
        validate_spec({**card, "confirm_prompt_fraction": 1})
    with pytest.raises(SpecError, match="spill_shared_mib"):
        validate_spec({**card, "spill_shared_mib": 512})
    with pytest.raises(SpecError, match="spill_prompt_ratio"):
        validate_spec({**card, "spill_prompt_ratio": 0.7})
    with pytest.raises(SpecError, match="rest_s"):
        validate_spec({**card, "rest_s": 899})
    with pytest.raises(SpecError, match="rest_s"):
        validate_spec({**good, "rest_s": True})
    with pytest.raises(SpecError, match="compile_cache"):
        validate_spec({key: value for key, value in good.items() if key != "compile_cache"})
    with pytest.raises(SpecError, match="unknown order"):
        validate_spec({**good, "order": "paired"})
    paired = _arm("P", "NPU")
    paired["companion"] = "missing"
    paired["companion_device"] = "5090"
    aba = {**good, "order": "aba", "arms": [_arm("A", "5090"), paired], "pairs": [{"id": "p", "alone": "A", "paired": "P"}]}
    with pytest.raises(SpecError, match="not an arm"):
        validate_spec(aba)
    paired["companion"] = "A"
    paired["companion_device"] = "NPU"
    with pytest.raises(SpecError, match="companion_device"):
        validate_spec({**aba, "arms": [_arm("A", "5090"), paired]})
    bare = dict(paired)
    del bare["companion_device"]
    with pytest.raises(SpecError, match="companion_device"):
        validate_spec({**aba, "arms": [_arm("A", "5090"), bare], "pairs": [{"id": "p", "alone": "A", "paired": "P"}]})
    with pytest.raises(SpecError, match="pairs"):
        validate_spec({**good, "order": "aba", "pairs": []})
    with pytest.raises(SpecError, match="roster"):
        validate_spec({**card, "models": []})
    broken_model = dict(card["models"][0])
    del broken_model["revision_or_digest"]
    with pytest.raises(SpecError, match="revision_or_digest"):
        validate_spec({**card, "models": [broken_model]})
    with pytest.raises(SpecError, match="5090"):
        validate_spec({**card, "models": [{**card["models"][0], "device": "CPU"}]})
    with pytest.raises(SpecError, match="grant"):
        validate_spec({**card, "needs": {**card["needs"], "grant": False}})

    calibrate = {
        "id": "cal",
        "question": "Does a scratch calibrate stay local?",
        "decision": "The adapter plans argv and does not launch.",
        "procedure": "offrig-calibrate",
        "needs": {"grant": True, "npu_session": False, "cpu_window": True},
        "stop_rules": ["a non-zero offrig exit is a failed step"],
        "dataset": {"name": "gold"},
        "metrics": ["accuracy"],
        "model": "qwen3:8b",
        "url": "http://127.0.0.1:11434",
        "gold": "experiments/gold.jsonl",
    }
    assert validate_spec(calibrate)["procedure"] == "offrig-calibrate"
    with pytest.raises(SpecError, match="gold"):
        validate_spec({k: v for k, v in calibrate.items() if k != "gold"})
    with pytest.raises(SpecError, match="grant"):
        validate_spec({**calibrate, "needs": {**calibrate["needs"], "grant": False}})
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(good), encoding="utf-8", newline="\n")
    assert load_spec(path)["id"] == "T"
    empty = tmp_path / "empty.yaml"
    empty.write_text("null\n", encoding="utf-8", newline="\n")
    with pytest.raises(SpecError, match="mapping"):
        load_spec(empty)


def test_dry_run_plans_calibrate_and_names_the_cpu_window(tmp_path: Path):
    repo = tmp_path / "repo"
    spec_path = repo / "experiments" / "cal" / "spec.yaml"
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text(
        "\n".join(
            [
                "id: cal",
                "question: Does a scratch calibrate stay local?",
                "decision: The adapter plans argv and does not launch.",
                "procedure: offrig-calibrate",
                "needs:",
                "  grant: true",
                "  npu_session: true",
                "  cpu_window: true",
                "stop_rules:",
                "  - a non-zero offrig exit is a failed step",
                "dataset:",
                "  name: gold",
                "metrics:",
                "  - accuracy",
                "model: qwen3:8b",
                "url: http://127.0.0.1:11434",
                "gold: experiments/gold.jsonl",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    _commit(repo)
    text = dry_run(spec_path, repo)
    assert "not launched" in text
    assert "E:/AI/switchyard-scratch/cal" in text
    assert "--cpu-window" in text
    loaded = load_spec(spec_path)
    check_flags(loaded, "grant-t", "session-t", "window-t")
    with pytest.raises(GuardError):
        check_flags(loaded, "grant-t", None, "window-t")
    check_flags(_spec(), "grant-t", "session-t", None)
    outside = ROOT / "experiments" / "e4" / "spec.yaml"
    with pytest.raises(SpecError, match="outside"):
        require_unmodified(repo, outside)
    loose = tmp_path / "loose"
    loose.mkdir()
    (loose / "spec.yaml").write_text("id: x\n", encoding="utf-8", newline="\n")
    with pytest.raises(SpecError, match="not a committed"):
        require_unmodified(loose, loose / "spec.yaml")


def test_duplicate_receipt_ids_are_refused(tmp_path: Path):
    repo = tmp_path / "repo"
    spec_path = _bind(repo, _spec())
    script = Scripted(
        {
            "A": [_result(1, 1), _result(1, 1), _result(1, 1), _result(1, 1)],
            "B": [_result(1, 1), _result(1, 1), _result(1, 1), _result(1, 1)],
        }
    )
    (repo / "table").mkdir()
    rows = _invoke(repo, spec_path, {"fake": script})
    with pytest.raises(ValueError, match="duplicate"):
        write_run(repo, "run-1", rows)


def test_adapter_edges_stay_local(monkeypatch):
    request = StepRequest(
        "s",
        "generate",
        "qwen3:8b",
        "5090",
        "ollama",
        {"num_predict": 1, "num_ctx": 2048, "thinking_level": "on"},
    )
    body = generate_body(request)
    assert body["options"]["num_ctx"] == 2048
    assert body["think"] is True
    assert OllamaAdapter(transport=lambda *_args: ["nope"]).run(request).ok is False
    assert OllamaAdapter(transport=lambda *_args: "nope").loaded_models() == []
    assert OllamaAdapter(transport=lambda *_args: {"models": "nope"}).loaded_models() == []

    captured = {}

    class _Body:
        def read(self):
            return b'{"ok": true}'

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def _urlopen(req, timeout=None):
        captured["data"] = req.data
        captured["timeout"] = timeout
        return _Body()

    monkeypatch.setattr("switchyard.adapters.ollama.urlopen", _urlopen)
    assert default_transport("POST", "http://127.0.0.1:11434/api/ps", {"a": 1}) == {"ok": True}
    assert captured["timeout"] is None
    assert default_transport("GET", "http://127.0.0.1:11434/api/ps", None) == {"ok": True}
    assert captured["data"] is None
    assert default_transport("GET", "http://127.0.0.1:11434/api/ps", None, timeout=12) == {"ok": True}
    assert captured["timeout"] == 12
    OllamaAdapter(timeout=7).transport("GET", "http://127.0.0.1:11434/api/ps", None)
    assert captured["timeout"] == 7

    with pytest.raises(AdapterError):
        scratch_project(" ")
    with pytest.raises(AdapterError):
        scratch_project("../cal")
    spec = {
        "id": "cal",
        "model": "qwen3:8b",
        "url": "http://127.0.0.1:11434",
        "gold": "experiments/gold.jsonl",
        "think": "off",
        "num_predict": 32,
    }
    with pytest.raises(AdapterError):
        plan_argv({**spec, "model": ""})
    with pytest.raises(AdapterError):
        plan_argv({**spec, "gold": "experiments/../gold.jsonl"})
    with pytest.raises(AdapterError):
        plan_argv({**spec, "think": "maybe"})
    with pytest.raises(AdapterError):
        plan_argv({**spec, "num_predict": 0})
    failed = OffrigCalibrateAdapter(executor=lambda _argv: 3).run(
        StepRequest("cal", "verify", "qwen3:8b", "5090", "offrig-calibrate", spec)
    )
    assert failed.ok is False and "exited 3" in failed.error
    OffrigCalibrateAdapter().unload("qwen3:8b")

    transport = lambda *_args: {}
    assert for_engine("ollama", transport=transport).transport is transport
    wired = for_engine("offrig-calibrate", executor=lambda _argv: 0)
    assert wired.executor is not None

    import switchyard.adapters.registry as registry

    monkeypatch.setattr(registry.importlib, "import_module", lambda _name: object())
    with pytest.raises(AdapterError, match="adapter"):
        for_engine("openvino")
    marker = object()
    monkeypatch.setattr(
        registry.importlib,
        "import_module",
        lambda _name: type("Mod", (), {"adapter": staticmethod(lambda: marker)})(),
    )
    assert for_engine("openvino") is marker


def test_the_run_path_refuses_an_edited_spec_before_any_adapter_call(tmp_path: Path):
    repo = tmp_path / "repo"
    spec_path = _bind(repo, _spec())
    spec_path.write_text(spec_path.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8", newline="\n")
    script = Scripted({"A": [], "B": []})
    with pytest.raises(SpecError, match="edited after commit"):
        _invoke(repo, spec_path, {"fake": script})
    assert script.calls == []


def test_a_bad_run_id_and_a_fill_card_spec_are_refused(tmp_path: Path):
    with pytest.raises(RunnerError, match="single name"):
        run_interleaved(
            {},
            grant=None,
            npu_session=None,
            cpu_window=None,
            root=tmp_path,
            repo=tmp_path,
            spec_path=tmp_path / "missing.yaml",
            run_id="a/b",
            date="2026-10-09",
        )
    repo = tmp_path / "repo"
    target = repo / "experiments" / "e5" / "spec.yaml"
    target.parent.mkdir(parents=True)
    target.write_text(E5.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    _commit(repo)
    with pytest.raises(RunnerError, match="interleaved"):
        _invoke(repo, target, {})


def test_companion_sessions_wrap_only_the_paired_slot(tmp_path: Path):
    alone = _arm("A", "5090")
    paired = _arm("P", "NPU")
    for arm in (alone, paired):
        arm["static_shape"] = "1,256"
        arm["precision"] = "fp16"
        arm["keep_alive"] = "10m"
    paired["companion"] = "A"
    paired["companion_device"] = "5090"
    paired["measure"] = "decode_tok_s"
    spec = _spec()
    spec.update(
        {
            "compile_cache": "fresh",
            "order": "aba",
            "warmups": 0,
            "repeats": 1,
            "arms": [alone, paired],
            "pairs": [{"id": "p1", "alone": "A", "paired": "P"}],
        }
    )
    script = Scripted(
        {
            "A": [
                _result(
                    100,
                    100,
                    compile_ms=12,
                    first_call_ms=3,
                    output="cold-a",
                    sensors={"gpu_temp_c": 50, "sm_clock_mhz": 1800, "power_w": 200, "secret": "no"},
                ),
                _result(40, 100, output="before"),
                _result(40, 100, output=[0.25, 0.5]),
            ],
            "P": [
                _result(8, 100, compile_ms=4, first_call_ms=1, resolved={"num_ctx": 4096, "quant": "Q4_K_M"}),
                _result(9, 100, output="paired-text"),
            ],
        },
        sessions=True,
    )
    repo = tmp_path / "repo"
    rows = _invoke(repo, _bind(repo, spec), {"fake": script})
    assert script.calls == ["A", "P", "A", "P", "A"]
    assert script.sessions == [("start", "A", "5090"), ("stop", "A", "5090")]
    by_id = {row["id"]: row for row in rows}
    arm_a = by_id["T/A/run-1"]
    assert arm_a["latency"]["cold_ms"] == 100
    assert arm_a["latency"]["compile_ms"] == 12
    assert arm_a["latency"]["first_call_ms"] == 3
    assert arm_a["latency"]["warm_p50_ms"] == 40
    assert arm_a["settings"]["static_shape"] == "1,256"
    assert arm_a["settings"]["precision"] == "fp16"
    assert arm_a["settings"]["keep_alive"] == "10m"
    assert arm_a["settings"]["compile_cache_dir"] == "compile-cache/run-1/A"
    assert arm_a["settings"]["compile_cache_deleted"] is True
    assert "aba" not in arm_a["settings"]
    sensors = arm_a["artifacts"]["slots"][0]["sensors"]
    assert sensors == {"gpu_temp_c": 50, "sm_clock_mhz": 1800, "power_w": 200}
    assert [0.25, 0.5] in [slot.get("output") for slot in arm_a["artifacts"]["slots"]]
    arm_p = by_id["T/P/run-1"]
    assert arm_p["model"]["quant"] == "Q4_K_M"
    assert arm_p["settings"]["num_ctx"] == 4096
    assert arm_p["settings"]["aba"][0]["call"] == "within"
    assert arm_p["settings"]["aba"][0]["pair"] == "p1"
    sent = script.requests[0].settings
    assert sent["static_shape"] == "1,256"
    assert sent["keep_alive"] == "10m"
    assert sent["compile_cache_dir"] != arm_a["settings"]["compile_cache_dir"]
    assert not Path(sent["compile_cache_dir"]).exists()
    dumped = json.dumps(rows)
    assert "Users" not in dumped
    assert sent["compile_cache_dir"] not in dumped

    bare = MeasureOnly(
        {
            "A": [_result(1, 1), _result(1, 1), _result(1, 1)],
            "P": [_result(1, 1)],
        }
    )
    repo2 = tmp_path / "bare"
    with pytest.raises(RunnerError, match="start_session"):
        _invoke(repo2, _bind(repo2, spec), {"fake": bare}, run_id="run-2")
    assert not (repo2 / "receipts").exists()


def test_an_interleaved_companion_starts_on_measure_slots_only(tmp_path: Path):
    spec = _spec()
    spec["warmups"] = 0
    spec["repeats"] = 1
    spec["arms"][1]["companion"] = "A"
    spec["arms"][1]["companion_device"] = "5090"
    script = Scripted(
        {
            "A": [_result(1, 1), _result(2, 2)],
            "B": [_result(1, 1, unit="items/s"), _result(3, 4, unit="items/s")],
        },
        sessions=True,
    )
    repo = tmp_path / "repo"
    rows = _invoke(repo, _bind(repo, spec), {"fake": script})
    assert script.sessions == [("start", "A", "5090"), ("stop", "A", "5090")]
    arm_b = next(row for row in rows if row["id"] == "T/B/run-1")
    assert arm_b["latency"]["throughput_unit"] == "items/s"
    assert arm_b["latency"]["cold_ms"] == 1

    mixed = _spec()
    mixed["warmups"] = 0
    script = Scripted(
        {
            "A": [_result(1, 1, unit="tok/s"), _result(1, 1, unit="tok/s"), _result(1, 1, unit="items/s")],
            "B": [_result(1, 1), _result(1, 1), _result(1, 1)],
        }
    )
    repo3 = tmp_path / "units"
    with pytest.raises(TimingError, match="throughput unit"):
        _invoke(repo3, _bind(repo3, mixed), {"fake": script})
    assert not (repo3 / "receipts").exists()


def test_companion_device_is_guarded_even_when_no_arm_measures_it():
    ghost = _arm("ghost", "CPU")
    measured = _arm("P", "NPU")
    measured["companion"] = "ghost"
    measured["companion_device"] = "5090"
    spec = _spec()
    spec["needs"] = {"grant": False, "npu_session": True, "cpu_window": False}
    spec["arms"] = [ghost, measured]
    assert validate_spec(spec)["id"] == "T"
    with pytest.raises(GuardError, match="grant"):
        check_flags(spec, None, "session-t", None)
    check_flags(spec, "grant-t", "session-t", None)


def test_ollama_resolves_show_and_does_not_fake_a_companion():
    assert think_value("off") is False
    assert think_value("on") is True
    assert think_value("low") == "low"
    assert think_value(None) is None
    plain = StepRequest("s", "generate", "qwen3:8b", "5090", "ollama", {"num_predict": 1})
    assert "think" not in generate_body(plain)
    low = StepRequest(
        "s",
        "generate",
        "qwen3:8b",
        "5090",
        "ollama",
        {"num_predict": 1, "thinking_level": "low", "keep_alive": None, "temperature": None, "prompt": None},
    )
    body = generate_body(low)
    assert body["think"] == "low"
    assert body["keep_alive"] == "5m"
    assert body["options"]["temperature"] == 0
    assert body["prompt"] == ""

    shown = {
        "details": {"quantization_level": "Q4_K_M"},
        "model_info": {"qwen3.context_length": 4096, "qwen3.embedding_length": 32},
    }
    calls = []

    def transport(method, url, payload):
        calls.append((url, payload))
        if url.endswith("/api/show"):
            return shown
        return {
            "eval_count": 8,
            "eval_duration": 1_000_000_000,
            "prompt_eval_count": 4,
            "prompt_eval_duration": 500_000_000,
            "total_duration": 2_000_000_000,
            "response": "ok",
        }

    adapter = OllamaAdapter(transport=transport)
    request = StepRequest(
        "s",
        "generate",
        "qwen3:8b",
        "5090",
        "ollama",
        {"num_predict": 1, "num_ctx": None, "quant": "unrecorded", "thinking_level": "off"},
    )
    result = adapter.run(request)
    assert result.resolved == {"num_ctx": 4096, "quant": "Q4_K_M"}
    assert result.throughput == 8
    assert result.prompt_tok_s == 8
    assert result.latency_ms == 2000
    assert result.output == "ok"
    assert result.compile_ms is None
    generate = next(payload for url, payload in calls if url.endswith("/api/generate"))
    assert generate["options"]["num_ctx"] == 4096
    calls.clear()
    ready = StepRequest(
        "s",
        "generate",
        "qwen3:8b",
        "5090",
        "ollama",
        {"num_predict": 1, "num_ctx": 2048, "quant": "Q4_K_M"},
    )
    assert adapter.run(ready).resolved is None
    assert all(not url.endswith("/api/show") for url, _payload in calls)

    with pytest.raises(AdapterError, match="one context"):
        context_length({"model_info": {}})
    with pytest.raises(AdapterError, match="one context"):
        context_length({"model_info": {"a.context_length": 1, "b.context_length": 2}})
    with pytest.raises(AdapterError, match="model_info"):
        context_length({})
    assert context_length({"model_info": {"qwen3.context_length": 4096, "other": 1}}) == 4096
    assert quantization_level({}) == "unrecorded"
    assert quantization_level({"details": {"quantization_level": ""}}) == "unrecorded"
    assert quantization_level({"details": "nope"}) == "unrecorded"

    seen = []

    def mark(method, url, payload):
        seen.append((url, payload))
        if url.endswith("/api/show"):
            return shown
        return {}

    marker = OllamaAdapter(transport=mark, device="CPU")
    start = StepRequest(
        "c",
        "generate",
        "qwen3:8b",
        "CPU",
        "ollama",
        {"num_predict": 1, "num_ctx": None, "quant": "unrecorded", "thinking_level": "off"},
    )
    marker.start_session(start)
    marker.stop_session(start)
    assert seen[0][0].endswith("/api/show")
    assert seen[1][1]["companion"] == "start"
    assert seen[1][1]["options"]["num_ctx"] == 4096
    assert seen[1][0].startswith("http://127.0.0.1:11490/")
    assert seen[2][1]["companion"] == "stop"
    assert seen[2][1].get("keep_alive") != 0
    marker.unload("qwen3:8b", device="5090")
    assert seen[-1][0].startswith("http://127.0.0.1:11434/")
    assert seen[-1][1]["keep_alive"] == 0
    with pytest.raises(AdapterError, match="missing"):
        OllamaAdapter(transport=lambda *_args: {"error": "missing"}).show("qwen3:8b")
    with pytest.raises(AdapterError, match="non-object"):
        OllamaAdapter(transport=lambda *_args: ["nope"]).show("qwen3:8b")


def _commit(repo: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "dev"], cwd=repo, check=True)
    subprocess.run(["git", "config", "core.autocrlf", "false"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=repo, check=True, capture_output=True)
