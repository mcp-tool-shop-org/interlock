"""Harness, adapters, and the E4 and E5 specs.

Device runs are fakes. Dry-run does not stat the heartbeat or call an adapter.
"""

import json
import subprocess
from pathlib import Path

import pytest

from switchyard.adapters.base import AdapterError, StepRequest, StepResult
from switchyard.adapters.offrig_calibrate import OffrigCalibrateAdapter, plan_argv, scratch_project
from switchyard.adapters.ollama import OllamaAdapter, default_transport, generate_body, refuse_cloud, url_for
from switchyard.adapters.registry import for_engine
from switchyard.guards import GuardError
from switchyard.harness.e5 import Observation, SearchError, fit_prompt, is_slow, over_margin, search
from switchyard.harness.lock import require_unmodified
from switchyard.harness.receipts import write_run
from switchyard.harness.runner import check_flags, dry_run, run_interleaved
from switchyard.harness.spec import SpecError, load_spec, validate_spec
from switchyard.harness.timing import TimingError, percentile, schedule, summarize, within_band
from switchyard.receipt import load_jsonl

ROOT = Path(__file__).resolve().parents[1]
E4 = ROOT / "experiments" / "e4" / "spec.yaml"
E5 = ROOT / "experiments" / "e5" / "spec.yaml"


class Scripted:
    def __init__(self, samples):
        self.samples = samples
        self.calls = []
        self.unloaded = []

    def run(self, request):
        self.calls.append(request.step_id)
        return self.samples[request.step_id].pop(0)

    def unload(self, model):
        self.unloaded.append(model)


def _result(latency, throughput=None, ok=True, error=None):
    return StepResult(ok=ok, latency_ms=latency, throughput=throughput, throughput_unit="tok/s", error=error)


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
        "arms": [_arm("A", "5090"), _arm("B", "NPU")],
    }


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
    spec = _spec()
    script = Scripted(
        {
            "A": [_result(100)],
            "B": [_result(1, ok=False, error="stopped")],
        }
    )
    with pytest.raises(Exception, match="stopped"):
        run_interleaved(
            spec,
            {"fake": script},
            grant="grant-t",
            npu_session="session-t",
            cpu_window=None,
            root=tmp_path,
            spec_rel="experiments/t/spec.yaml",
            run_id="run-1",
            date="2026-10-09",
        )
    assert script.calls == ["A", "B"]
    assert script.unloaded == ["qwen3:8b"]
    assert not (tmp_path / "receipts").exists()


def test_repeats_build_a_receipt_and_keep_cold_separate(tmp_path: Path):
    spec = _spec()
    script = Scripted(
        {
            "A": [_result(100, 1), _result(80, 1), _result(10, 4), _result(30, 8)],
            "B": [_result(5, 1), _result(4, 1), _result(2, 3), _result(4, 5)],
        }
    )
    (tmp_path / "table").mkdir()
    rows = run_interleaved(
        spec,
        {"fake": script},
        grant="grant-t",
        npu_session="session-t",
        cpu_window=None,
        root=tmp_path,
        spec_rel="experiments/t/spec.yaml",
        run_id="run-1",
        date="2026-10-09",
    )
    by_id = {row["id"]: row for row in rows}
    arm_a = by_id["T/A/run-1"]
    assert arm_a["latency"]["cold_ms"] == 100
    assert arm_a["latency"]["warm_p50_ms"] == 20
    assert arm_a["latency"]["warm_p95_ms"] == 29
    assert arm_a["provisional"] is False
    assert arm_a["settings"]["grant"] == "grant-t"
    assert arm_a["settings"]["environment"]["nvidia_driver"] == "unrecorded"
    assert arm_a["provenance"] == {"experiment_spec": "experiments/t/spec.yaml", "run_id": "run-1"}
    stored = load_jsonl(tmp_path / "table" / "routing.jsonl")
    assert {row["id"] for row in stored} == {row["id"] for row in rows}
    assert script.calls[:4] == ["A", "B", "A", "B"]
    with pytest.raises(GuardError):
        run_interleaved(
            spec,
            {"fake": script},
            grant=None,
            npu_session="session-t",
            cpu_window=None,
            root=tmp_path,
            spec_rel="experiments/t/spec.yaml",
            run_id="run-2",
            date="2026-10-09",
        )


def test_e4_and_e5_specs_pin_the_rules():
    e4 = load_spec(E4)
    e5 = load_spec(E5)
    assert e4["band"] == 0.10
    assert e4["repeats"] == 5 and e4["warmups"] == 1
    nli = [arm for arm in e4["arms"] if arm["task"] == "nli"]
    assert nli and all(arm["batch"] == 1 for arm in nli)
    assert any(arm["model"] == "BAAI/bge-base-en-v1.5" and arm["device"] == "NPU" for arm in e4["arms"])
    assert e5["abort_mib"] == 30500
    assert e5["ceiling_mib"] == 31200
    assert e5["grain"] == 2048
    assert len(e5["models"]) == 6
    assert e5["shift"] is False and e5["truncate"] is False
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
    assert "slots:" in e4_text
    assert "abort_mib: 30500" in e5_text
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
    with pytest.raises(AdapterError, match="Intel lane"):
        for_engine("openvino")
    with pytest.raises(AdapterError, match="unknown engine"):
        for_engine("cuda")


def test_fill_card_doubles_bisects_and_steps_back():
    def climb(ctx, kind):
        failed = ctx >= 16384
        return Observation(
            resident=not failed,
            peak_mib=30000 if not failed else 31000,
            tok_s=10,
            ttft_ms=100,
        )

    found = search(default_ctx=4096, probe=climb, abort_mib=30500)
    assert found.max_ctx == 14336
    assert ("load", 16384) in found.calls
    assert all(ctx <= 16384 for _, ctx in found.calls)
    assert ("confirm", 14336) in found.calls
    assert over_margin(Observation(True, 30500, 1, 1), 30500) is False
    assert over_margin(Observation(True, 30501, 1, 1), 30500) is True

    def slow_top(ctx, kind):
        if kind == "confirm" and ctx >= 8192:
            return Observation(True, 20000, tok_s=1, ttft_ms=100)
        return Observation(True, 20000, tok_s=10, ttft_ms=100)

    slowed = search(default_ctx=4096, probe=slow_top, abort_mib=30500, max_ctx=8192)
    assert slowed.max_ctx == 6144
    assert ("confirm", 6144) in slowed.calls

    def mid_fail(ctx, kind):
        failed = ctx >= 12288
        return Observation(not failed, 30000 if not failed else 31000, tok_s=10, ttft_ms=100)

    bisected = search(default_ctx=4096, probe=mid_fail, abort_mib=30500)
    assert bisected.max_ctx == 10240
    assert ("load", 12288) in bisected.calls

    def odd_grid(ctx, kind):
        failed = ctx >= 8192
        return Observation(not failed, 20000 if not failed else 31000, tok_s=10, ttft_ms=100)

    snapped = search(default_ctx=4096, probe=odd_grid, abort_mib=30500, grain=3000, max_ctx=8192)
    assert snapped.max_ctx == 7096

    nowhere = search(default_ctx=4096, probe=lambda ctx, kind: Observation(False, 1), abort_mib=30500)
    assert nowhere.max_ctx is None
    assert fit_prompt("ab", 4, len) == "abab"
    with pytest.raises(SearchError):
        fit_prompt("", 1, len)
    with pytest.raises(SearchError):
        fit_prompt("ab", 0, len)
    with pytest.raises(SearchError, match="never reaches"):
        fit_prompt("x" * 600_000, 1, lambda _text: 0)


def test_search_spill_and_schedule_refuse_bad_inputs():
    reference = Observation(True, 1, tok_s=10, ttft_ms=100)
    assert is_slow(Observation(True, 1), reference, tok_ratio=0.5, ttft_ratio=2) is True
    assert is_slow(reference, Observation(True, 1), tok_ratio=0.5, ttft_ratio=2) is True
    assert is_slow(Observation(True, 1, tok_s=10, ttft_ms=201), reference, tok_ratio=0.5, ttft_ratio=2) is True
    assert is_slow(Observation(True, 1, tok_s=5, ttft_ms=200), reference, tok_ratio=0.5, ttft_ratio=2) is False
    assert over_margin(Observation(True, None, error="lost"), 30500) is True
    assert over_margin(Observation(True, None), 30500) is True

    def confirm_fails(ctx, kind):
        if kind == "confirm":
            return Observation(True, 20000, tok_s=1, ttft_ms=100)
        return Observation(True, 20000, tok_s=10, ttft_ms=100)

    refused = search(default_ctx=4096, probe=confirm_fails, abort_mib=30500, max_ctx=4096)
    assert refused.max_ctx is None
    assert refused.reason == "no confirmed window"
    with pytest.raises(SearchError):
        search(default_ctx=0, probe=confirm_fails, abort_mib=30500)
    with pytest.raises(SearchError):
        search(default_ctx=4096, probe=confirm_fails, abort_mib=0)

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

    card = load_spec(E5)
    with pytest.raises(SpecError, match="2048"):
        validate_spec({**card, "grain": 1024})
    with pytest.raises(SpecError, match="confirm_prompt_fraction"):
        validate_spec({**card, "confirm_prompt_fraction": 1})
    with pytest.raises(SpecError, match="spill_tok_ratio"):
        validate_spec({**card, "spill_tok_ratio": "half"})
    with pytest.raises(SpecError, match="spill_tok_ratio"):
        validate_spec({**card, "spill_tok_ratio": 2})
    with pytest.raises(SpecError, match="spill_ttft_ratio"):
        validate_spec({**card, "spill_ttft_ratio": 0.5})
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
    spec = _spec()
    script = Scripted(
        {
            "A": [_result(1, 1), _result(1, 1), _result(1, 1), _result(1, 1)],
            "B": [_result(1, 1), _result(1, 1), _result(1, 1), _result(1, 1)],
        }
    )
    (tmp_path / "table").mkdir()
    rows = run_interleaved(
        spec,
        {"fake": script},
        grant="grant-t",
        npu_session="session-t",
        cpu_window=None,
        root=tmp_path,
        spec_rel="experiments/t/spec.yaml",
        run_id="run-1",
        date="2026-10-09",
    )
    with pytest.raises(ValueError, match="duplicate"):
        write_run(tmp_path, "run-1", rows)


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

    def _urlopen(req, timeout):
        captured["data"] = req.data
        captured["timeout"] = timeout
        return _Body()

    monkeypatch.setattr("switchyard.adapters.ollama.urlopen", _urlopen)
    assert default_transport("POST", "http://127.0.0.1:11434/api/ps", {"a": 1}) == {"ok": True}
    assert default_transport("GET", "http://127.0.0.1:11434/api/ps", None) == {"ok": True}
    assert captured["timeout"] == 30
    assert captured["data"] is None

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


def _commit(repo: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "dev"], cwd=repo, check=True)
    subprocess.run(["git", "config", "core.autocrlf", "false"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=repo, check=True, capture_output=True)
