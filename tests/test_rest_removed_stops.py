"""Run-start device state, the decision-metric andon, the operator stop file.

Rest is the Publisher's scheduling decision, not a harness rule (maintainer, 2026-10-11).
"""

from __future__ import annotations

import json
import threading

import pytest

from interlock.adapters.base import AdapterError
from interlock.harness.marker import marker_path
from interlock.harness.runner import RunnerError
from interlock.harness.spec import SpecError, validate_spec
from interlock.harness.watch import EXIT_OPERATOR_STOP, WatchHooks, read_start_state
from interlock.receipt import load_jsonl
from tests.test_harness import _result, _spec as _harness_spec
from tests.test_run_guard import BlockingAdapter, Clock, Exits, _hooks, _invoke, _spec


class Plain:
    def __init__(self, throughput=None):
        self.throughput = throughput
        self.calls = []

    def run(self, request):
        self.calls.append(request.step_id)
        return _result(5, throughput=self.throughput)

    def unload(self, model, device=None):
        return None


def _plain_hooks(**kwargs):
    return WatchHooks(ram_sampler=None, start_thread=False, **kwargs)


# --- start_state -------------------------------------------------------------


def test_start_state_is_on_every_row_and_taken_once(tmp_path):
    repo, spec_path = _spec(tmp_path)
    taken = []

    def sampler(uses_5090):
        taken.append(uses_5090)
        return {"cpu_percent": 3.0, "cpu_freq_mhz": 3700.0, "ram_pct": 41.0}

    rows = _invoke(repo, spec_path, Plain(), _plain_hooks(state_sampler=sampler))
    assert taken == [False]
    assert all(row["settings"]["start_state"]["ram_pct"] == 41.0 for row in rows)


def test_a_failing_start_state_is_unrecorded_and_never_fails_the_run(tmp_path):
    repo, spec_path = _spec(tmp_path)

    def broken(uses_5090):
        raise OSError("no counters")

    rows = _invoke(repo, spec_path, Plain(), _plain_hooks(state_sampler=broken))
    assert rows[0]["settings"]["start_state"] == "unrecorded"


def test_read_start_state_reads_the_card_only_for_a_5090_run(monkeypatch):
    import interlock.harness.watch as watch

    calls = []

    class Done:
        returncode = 0
        stdout = "54, 2100, 61.5, 1024\n"

    monkeypatch.setattr(watch.subprocess, "run", lambda *a, **k: calls.append(a) or Done())
    plain = read_start_state(False)
    assert calls == [] and "gpu_temp_c" not in plain
    card = read_start_state(True)
    assert card["gpu_temp_c"] == 54.0 and card["sm_clock_mhz"] == 2100.0
    assert card["power_w"] == 61.5 and card["memory_used_mib"] == 1024.0
    for key in ("cpu_percent", "cpu_freq_mhz", "ram_pct"):
        assert key in plain and key in card


def test_read_start_state_marks_missing_values_unrecorded(monkeypatch):
    import interlock.harness.watch as watch

    def missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(watch.subprocess, "run", missing)
    monkeypatch.setattr(watch, "psutil", None)
    state = read_start_state(True)
    assert set(state.values()) == {"unrecorded"}


def test_read_start_state_survives_an_odd_nvidia_smi(monkeypatch):
    import interlock.harness.watch as watch

    class Odd:
        returncode = 0
        stdout = "54, [N/A], x\n"

    monkeypatch.setattr(watch.subprocess, "run", lambda *a, **k: Odd())
    state = read_start_state(True)
    assert state["gpu_temp_c"] == 54.0 and state["sm_clock_mhz"] == "unrecorded"


# --- the decision-metric andon -------------------------------------------------


def test_decision_metric_is_validated():
    good = _harness_spec()
    assert validate_spec({**good, "decision_metric": "latency.throughput"})["decision_metric"] == "latency.throughput"
    for bad in ("throughput", "latency.nope", "quality.x", 5):
        with pytest.raises(SpecError, match="decision_metric"):
            validate_spec({**good, "decision_metric": bad})
    validate_spec(good)


def test_an_arm_without_the_decision_metric_stops_the_run_after_the_first_such_arm(tmp_path):
    repo, spec_path = _spec(tmp_path, decision_metric="latency.throughput")
    with pytest.raises(RunnerError, match=r"decision metric latency\.throughput missing on "):
        _invoke(repo, spec_path, Plain(throughput=None), _plain_hooks())
    rows = load_jsonl(repo / "receipts" / "run-1.jsonl")
    assert any(row.get("aborted") and "decision metric" in row["settings"]["abortion"]["error"] for row in rows)
    assert json.loads(marker_path(repo, "run-1").read_text(encoding="utf-8"))["closed"] is True


def test_a_present_decision_metric_does_nothing(tmp_path):
    repo, spec_path = _spec(tmp_path, decision_metric="latency.throughput")
    rows = _invoke(repo, spec_path, Plain(throughput=42.0), _plain_hooks())
    assert len(rows) == 2 and not any(row.get("aborted") for row in rows)


def test_a_spec_without_decision_metric_is_unaffected(tmp_path):
    repo, spec_path = _spec(tmp_path)
    rows = _invoke(repo, spec_path, Plain(throughput=None), _plain_hooks())
    assert len(rows) == 2


# --- the operator stop file -----------------------------------------------------


class StopAdapter(BlockingAdapter):
    """The NPU arm writes the stop file, then sits in a native call until the watcher ends the run."""

    def __init__(self, *args, stop_file=None, text="", **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_file = stop_file
        self.text = text

    def run(self, request):
        if request.step_id == "npu" and self.stop_file is not None:
            self.stop_file.write_text(self.text, encoding="utf-8")
        return super().run(request)


def test_the_stop_file_flushes_closes_the_marker_and_exits_77_while_the_main_thread_is_blocked(tmp_path):
    repo, spec_path = _spec(tmp_path)
    clock, gate = Clock(), threading.Event()
    exits = Exits(repo / "receipts" / "run-1.jsonl", gate)
    hooks, _ram = _hooks(clock, exits)
    stop = repo / "receipts" / "run-1.stop"
    adapter = StopAdapter(
        clock, gate, compile_spent_s=1, in_compile=False, stop_file=stop, text="  Mike needs the card\nsecond line\n"
    )
    with pytest.raises(AdapterError, match="released"):
        _invoke(repo, spec_path, adapter, hooks)
    assert exits.codes == [EXIT_OPERATOR_STOP] == [77]
    assert exits.blocked_when_called is True
    by_id = {row["id"]: row for row in exits.seen}
    error = "stopped by operator: Mike needs the card"
    assert by_id["T/cpu/run-1"]["settings"]["abortion"]["error"] == error
    assert by_id["T/npu/run-1"]["aborted"] is True
    assert exits.marker["closed"] is True and exits.marker["reason"] == error
    assert not stop.exists() and stop.with_name("run-1.stop.done").is_file()


def test_an_empty_stop_file_still_stops_with_a_default_reason(tmp_path):
    repo, spec_path = _spec(tmp_path)
    clock, gate = Clock(), threading.Event()
    exits = Exits(repo / "receipts" / "run-1.jsonl", gate)
    hooks, _ram = _hooks(clock, exits)
    adapter = StopAdapter(
        clock, gate, compile_spent_s=1, in_compile=False, stop_file=repo / "receipts" / "run-1.stop", text=""
    )
    with pytest.raises(AdapterError):
        _invoke(repo, spec_path, adapter, hooks)
    assert exits.codes == [77]
    assert exits.marker["reason"] == "stopped by operator: no reason given"


def test_no_stop_file_does_nothing(tmp_path):
    repo, spec_path = _spec(tmp_path)
    clock, gate = Clock(), threading.Event()
    exits = Exits(repo / "receipts" / "run-1.jsonl", gate)
    hooks, _ram = _hooks(clock, exits)
    rows = _invoke(repo, spec_path, Plain(), hooks)
    assert exits.codes == [] and len(rows) == 2


def test_prefix_caching_is_validated_on_the_spec_and_on_arms():
    good = _harness_spec()
    for value in ("on", "off", "default"):
        validate_spec({**good, "prefix_caching": value})
    with pytest.raises(SpecError, match="prefix_caching"):
        validate_spec({**good, "prefix_caching": "maybe"})
    arms = [dict(good["arms"][0], prefix_caching="sometimes"), good["arms"][1]]
    with pytest.raises(SpecError, match="prefix_caching"):
        validate_spec({**good, "arms": arms})


def test_generate_rows_record_the_effective_prefix_caching(tmp_path):
    repo, spec_path = _spec(tmp_path)
    import yaml

    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    for arm in spec["arms"]:
        arm.update(task="generate", engine="openvino", device="CPU")
        arm.pop("batch", None)
        arm.pop("static_shape", None)
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8", newline="")
    from tests.test_harness import _commit

    _commit(repo)

    class Gen(Plain):
        def run(self, request):
            return _result(5, throughput=9.0, resolved={"prefix_caching": "library default (uncontrolled)"})

    from interlock.harness.runner import run_interleaved

    rows = run_interleaved(
        {"openvino": Gen()}, grant=None, npu_session=None, cpu_window=None, root=repo, repo=repo,
        spec_path=spec_path, run_id="run-pc", date="2026-10-11", watch=_plain_hooks(),
    )
    assert all(row["settings"]["prefix_caching"] == "library default (uncontrolled)" for row in rows)


def test_a_generate_arm_with_no_prompt_is_refused():
    # e3-ref 2026-10-11: no arm carried a prompt, so 14 calls sent "" and measured nothing.
    import copy
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    e3ref = yaml.safe_load((root / "experiments/e3-ref/spec.yaml").read_text(encoding="utf-8"))
    for arm in e3ref["arms"]:
        arm.pop("prompt", None)  # the shape of e3-ref's first run
    e3ref["dataset"] = {"name": "e3-ref-fixed-prompt", "rule": "anchors named, never referenced"}
    with pytest.raises(SpecError, match="has no prompt"):
        validate_spec(e3ref)
    fixed = copy.deepcopy(e3ref)
    for arm in fixed["arms"]:
        arm["prompt"] = "The switchyard counts the same cars on every pass. "
    validate_spec(fixed)
    built = copy.deepcopy(e3ref)
    built["dataset"] = {"name": "x", "rule": "built", "seed": "a b c ", "target_tokens": 256}
    validate_spec(built)  # the harness builds the prompt from the seed
    validate_spec(yaml.safe_load((root / "experiments/e3/spec.yaml").read_text(encoding="utf-8")))
