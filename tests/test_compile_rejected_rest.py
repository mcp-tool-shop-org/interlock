"""The compile_rejected outcome (ruling on #36), the one-line gaps summary, and the absence of any harness rest.

Adapters, clocks and receipts are fakes; no test touches a device or the wall clock.
"""

from __future__ import annotations

import json

import pytest

import interlock.adapters.openvino as ovm
from interlock.adapters.base import AdapterError, CompileRejected, NativeBinaryRejected, compile_rejected_error
from interlock.harness.receipts import device_lost_row, for_table
from interlock.harness.runner import compiler_message, dry_run, run_spec
from interlock.importers.gaps import aborted_run_lines, summarize_error
from interlock.receipt import ReceiptError, check_tree, is_speed_row, load_jsonl, validate_row
from tests.support import row as fixture_row
from tests.test_harness import ScriptedRaise, _emb_arm, _invoke, _result
from tests.test_harness import _bind as _bind_harness
from tests.test_harness import _spec as _harness_spec
from tests.test_openvino_adapter import FakeCompiled, FakeCore, local_model, make_adapter, req
from tests.test_run_path import Fake, _arm, _bind, _spec

# A real NPU compiler refusal (issue #36): the error leads with a stack of
# "Exception from" lines and the compiler's own message comes last.
COMPILE_REJECTED = (
    "Exception from src\\inference\\src\\cpp\\core.cpp:120:\n"
    "Exception from src\\inference\\src\\dev\\plugin.cpp:54:\n"
    "Exception from src\\plugins\\intel_npu\\src\\plugin\\src\\plugin.cpp:560:\n"
    "Exception from src\\plugins\\intel_npu\\src\\compiler_adapter\\src\\ze_graph_ext_wrappers.cpp:433:\n"
    "Compilation failed. Level0 pfnCreate2 result: ZE_RESULT_ERROR_INVALID_NULL_POINTER, "
    "code 0x78000007 - pointer argument may not be nullptr . [NPU_VCL] Compiler returned msg:\n"
    "Target buffer size '2048' does not match source buffer size '256'"
)


# --- the classifier --------------------------------------------------------


def test_compile_rejected_error_needs_both_markers():
    assert compile_rejected_error(COMPILE_REJECTED) is True
    assert compile_rejected_error("[NPU_VCL] Compiler returned msg: x") is False  # no code
    assert compile_rejected_error("code 0x78000007 - pointer argument may not be nullptr") is False  # no marker
    assert compile_rejected_error("") is False
    assert compile_rejected_error(None) is False


def test_the_compiler_message_is_the_text_after_the_marker_on_one_line():
    assert compiler_message(COMPILE_REJECTED) == "Target buffer size '2048' does not match source buffer size '256'"
    assert compiler_message("no marker here") is None


# --- the adapter marks a compile-site refusal -------------------------------


class CompileRefusingCore(FakeCore):
    def __init__(self, message):
        super().__init__()
        self.message = message

    def compile_model(self, graph, device, config):
        self.compiled.append((device, dict(config)))
        raise RuntimeError(self.message)


def _embed(tmp_path, monkeypatch, message, device):
    m = local_model(tmp_path)
    monkeypatch.setitem(ovm._POOLING, m, "cls")
    adapter = make_adapter(core=CompileRefusingCore(message))
    return lambda: adapter.run(req(model=m, device=device, static_shape="1x16"))


def test_an_npu_compile_refusal_is_raised_as_compile_rejected_with_the_time_to_failure(tmp_path, monkeypatch):
    run = _embed(tmp_path, monkeypatch, COMPILE_REJECTED, "NPU")
    with pytest.raises(CompileRejected, match="0x78000007") as caught:
        run()
    assert caught.value.failed_s is not None and caught.value.failed_s >= 0
    assert not isinstance(caught.value, NativeBinaryRejected)


def test_the_same_error_on_the_cpu_is_not_a_compile_rejection(tmp_path, monkeypatch):
    result = _embed(tmp_path, monkeypatch, COMPILE_REJECTED, "CPU")()
    assert result.ok is False and "0x78000007" in result.error  # the old failed-step shape


@pytest.mark.parametrize(
    "message",
    ["[NPU_VCL] Compiler returned msg: bad graph", "ZE_RESULT_ERROR_INVALID_NULL_POINTER, code 0x78000007"],
)
def test_a_missing_marker_is_not_a_compile_rejection(tmp_path, monkeypatch, message):
    result = _embed(tmp_path, monkeypatch, message, "NPU")()
    assert result.ok is False  # not marked: the old failed-step shape


def test_the_markers_at_execute_are_not_a_compile_rejection(tmp_path, monkeypatch):
    class ExecuteRefusingCore(FakeCore):
        def compile_model(self, graph, device, config):
            self.compiled.append((device, dict(config)))
            return FakeCompiled(fail=True, msg=COMPILE_REJECTED)

    m = local_model(tmp_path)
    monkeypatch.setitem(ovm._POOLING, m, "cls")
    adapter = make_adapter(core=ExecuteRefusingCore())
    result = adapter.run(req(model=m, device="NPU", static_shape="1x16"))
    assert result.ok is False  # raised at execute, not at compile: not marked


# --- the runner records it -------------------------------------------------


def _rejecting_spec(tmp_path):
    repo = tmp_path / "repo"
    spec = _harness_spec()
    spec["warmups"] = 0
    spec["repeats"] = 1
    spec["arms"] = [
        _emb_arm("npu512", "NPU", "1x512"),
        _emb_arm("npu128", "NPU", "1x128"),
        _emb_arm("cpu512", "CPU", "1x512"),
    ]
    return repo, _bind_harness(repo, spec)


def test_a_compile_rejection_is_a_capability_row_scoped_to_the_shape(tmp_path, capsys):
    repo, spec_path = _rejecting_spec(tmp_path)
    script = ScriptedRaise(
        {
            "npu512": [CompileRejected(COMPILE_REJECTED, failed_s=3.5)],
            "npu128": [_result(30), _result(5)],
            "cpu512": [_result(50), _result(10)],
        }
    )
    (repo / "table").mkdir()
    rows = _invoke(repo, spec_path, {"fake": script})
    # No retry; the NPU is not off; only the identical shape is skipped.
    assert script.calls == ["npu512", "npu128", "cpu512", "npu128", "cpu512"]
    by_id = {row["id"]: row for row in rows}
    rejected = by_id["T/npu512/run-1"]
    assert rejected["outcome"] == "compile_rejected"
    assert rejected["provisional"] is True
    assert "partial" not in rejected and "aborted" not in rejected
    assert all(value is None for value in rejected["latency"].values())
    evidence = rejected["settings"]["rejection"]
    assert evidence["compiler_message"] == "Target buffer size '2048' does not match source buffer size '256'"
    assert evidence["code"] == "0x78000007"
    assert "npu_driver" in evidence and "openvino" in evidence
    assert evidence["time_to_failure_s"] == 3.5
    assert evidence["run_id"] == "run-1"
    assert evidence["shape"] == {"batch": 1, "static_shape": "1x512"}
    assert rejected["engine"]["version"]
    assert "not run: npu512 repeat 0" in rejected["notes"]
    assert "outcome" not in by_id["T/npu128/run-1"]
    assert "outcome" not in by_id["T/cpu512/run-1"]
    assert capsys.readouterr().out == "npu512 NPU compile_rejected\ndone\n"
    assert check_tree(repo) == []
    kept = {row["id"]: row for row in load_jsonl(repo / "table" / "routing.jsonl")}["T/npu512/run-1"]
    assert kept["outcome"] == "compile_rejected"
    assert kept["notes"].startswith("this device can't do this setting")


def test_a_compile_rejection_skips_only_the_identical_shape(tmp_path):
    repo = tmp_path / "repo"
    spec = _harness_spec()
    spec["warmups"] = 0
    spec["repeats"] = 1
    spec["arms"] = [_emb_arm("a", "NPU", "1x512"), _emb_arm("b", "NPU", "1x512"), _emb_arm("c", "NPU", "1x128")]
    spec_path = _bind_harness(repo, spec)
    script = ScriptedRaise(
        {"a": [CompileRejected(COMPILE_REJECTED, failed_s=1.0)], "b": [], "c": [_result(5), _result(2)]}
    )
    (repo / "table").mkdir()
    _invoke(repo, spec_path, {"fake": script})
    assert "b" not in script.calls
    assert script.calls.count("c") == 2


def test_a_compile_rejection_on_the_cpu_aborts(tmp_path):
    repo = tmp_path / "repo"
    spec = _harness_spec()
    spec["warmups"] = 0
    spec["repeats"] = 1
    spec["arms"] = [_emb_arm("cpu", "CPU", "1x128")]
    spec_path = _bind_harness(repo, spec)
    script = ScriptedRaise({"cpu": [CompileRejected(COMPILE_REJECTED, failed_s=1.0)]})
    (repo / "table").mkdir()
    with pytest.raises(CompileRejected):
        _invoke(repo, spec_path, {"fake": script})


@pytest.mark.parametrize("error", ["[NPU_VCL] Compiler returned msg: x", "code 0x78000007"])
def test_a_compile_rejection_missing_a_marker_aborts(tmp_path, error):
    repo, spec_path = _rejecting_spec(tmp_path)
    script = ScriptedRaise({"npu512": [CompileRejected(error, failed_s=1.0)]})
    (repo / "table").mkdir()
    with pytest.raises(CompileRejected):
        _invoke(repo, spec_path, {"fake": script})
    assert script.calls == ["npu512"]


def test_a_plain_adapter_error_at_the_npu_still_aborts(tmp_path):
    repo, spec_path = _rejecting_spec(tmp_path)
    script = ScriptedRaise({"npu512": [AdapterError(COMPILE_REJECTED)]})
    (repo / "table").mkdir()
    with pytest.raises(AdapterError):
        _invoke(repo, spec_path, {"fake": script})


def test_the_compile_rejected_row_validates_and_carries_no_timing():
    row = device_lost_row(
        spec_id="T",
        arm=_emb_arm("npu512", "NPU", "1x512"),
        repeats=1,
        date="2026-10-10",
        spec_rel="experiments/t/spec.yaml",
        run_id="run-1",
        grant=None,
        env={"npu_firmware": "32.0.100.5540", "nvidia_driver": "unrecorded"},
        outcome="compile_rejected",
    )
    assert row["outcome"] == "compile_rejected" and row["provisional"] is True
    assert is_speed_row(row) is False
    tabled = for_table(row)
    assert tabled["notes"] == "this device can't do this setting"
    bad = json.loads(json.dumps(row))
    bad["latency"]["compile_ms"] = 12.0
    with pytest.raises(ReceiptError):
        validate_row(bad)
    bad = json.loads(json.dumps(row))
    bad["outcome"] = "compile_exploded"
    with pytest.raises(ReceiptError):
        validate_row(bad)


# --- the gaps line ---------------------------------------------------------


def test_the_error_summary_prefers_the_compiler_message():
    line = summarize_error(COMPILE_REJECTED)
    assert "\n" not in line
    assert line == (
        "Compiler returned msg: Target buffer size '2048' does not match source buffer size '256' "
        "(ZE_RESULT_ERROR_INVALID_NULL_POINTER, 0x78000007)"
    )


def test_the_error_summary_leads_with_the_compiler_message_so_the_cut_keeps_the_cause():
    padded = "Exception from " + "x" * 400 + ":\nCompiler returned msg:\nTarget buffer size '2048' does not match"
    assert summarize_error(padded).startswith("Compiler returned msg: Target buffer size '2048'")
    same_line = "ZE_RESULT_ERROR_INVALID_NULL_POINTER 0x78000007\n[NPU_VCL] Compiler returned msg: bad graph"
    assert summarize_error(same_line) == (
        "Compiler returned msg: bad graph (ZE_RESULT_ERROR_INVALID_NULL_POINTER, 0x78000007)"
    )
    assert summarize_error("Compiler returned msg: only words") == "Compiler returned msg: only words"
    assert summarize_error("Compiler returned msg:") == "Compiler returned msg:"
    [line] = aborted_run_lines(
        [
            fixture_row(
                aborted=True,
                provenance={"experiment_spec": "x", "run_id": "r1"},
                settings={"abortion": {"stop_step": "npu cold", "error": padded}},
            )
        ]
    )
    assert "Target buffer size '2048'" in line


def test_the_error_summary_falls_back_to_the_level_zero_line_then_the_last_line():
    assert summarize_error("Exception from a:\nL0 failed ZE_RESULT_ERROR_DEVICE_LOST x\nmore") == (
        "L0 failed ZE_RESULT_ERROR_DEVICE_LOST x"
    )
    assert summarize_error("first\n\nlast line  \n\n") == "last line"
    assert summarize_error("one line only") == "one line only"
    assert summarize_error(None) == ""


def test_the_aborted_run_line_is_a_single_line():
    def aborted(error):
        return fixture_row(
            aborted=True,
            provenance={"experiment_spec": "x", "run_id": "r1"},
            settings={"abortion": {"stop_step": "npu cold", "error": error}},
        )

    [line] = aborted_run_lines([aborted(COMPILE_REJECTED)])
    assert "\n" not in line and "Exception from" not in line
    assert "Target buffer size '2048'" in line
    [plain] = aborted_run_lines([aborted("boom")])
    assert plain == "Aborted run `r1` stopped at `npu cold`: boom"


# --- no rest in the harness (maintainer, 2026-10-11) -----------------------

NOW = 1_800_000_000.0


def _run(tmp_path, *, spec=None):
    repo = tmp_path / "repo"
    spec_path = _bind(repo, spec or _spec([_arm("A", "CPU")]))
    fake = Fake({"A": [_result(10), _result(11), _result(12)]})
    rows = run_spec(spec_path, root=repo, adapters={"fake": fake}, date="2026-10-10", wall=lambda: NOW)
    return repo, spec_path, rows


def test_a_run_records_no_rest_and_never_waits(tmp_path, monkeypatch):
    import time

    monkeypatch.setattr(time, "sleep", lambda s: (_ for _ in ()).throw(AssertionError("the run slept")))
    _repo, _path, rows = _run(tmp_path)
    assert "rest" not in rows[0]["settings"] and "rest_s" not in rows[0]["settings"]


def test_an_old_spec_with_rest_s_runs_and_a_spec_without_it_is_valid(tmp_path):
    _repo, _path, rows = _run(tmp_path)  # the fixture spec still carries rest_s: 900, ignored
    assert rows
    spec = _spec([_arm("A", "CPU")])
    spec.pop("rest_s")
    _repo, _path, rows = _run(tmp_path / "b", spec=spec)
    assert rows


def test_the_dry_run_has_no_rest_line(tmp_path):
    repo = tmp_path / "repo"
    spec_path = _bind(repo, _spec([_arm("A", "CPU")]))
    assert "rest_s" not in dry_run(spec_path, repo)
