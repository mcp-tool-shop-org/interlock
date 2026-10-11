"""Harness, adapters, and the E4 and E5 specs.

Device runs are fakes. Dry-run does not stat the heartbeat or call an adapter.
"""

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from switchyard.adapters.base import AdapterError, NativeBinaryRejected, StepRequest, StepResult
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
from switchyard.harness.environment import driver_version_from_lines, package_versions
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
from switchyard.harness.receipts import apply_parity, device_lost_row, environment, for_table, import_receipt, write_run
from switchyard.harness.runner import (
    RunnerError,
    check_flags,
    device_lost_error,
    dry_run,
    hang_error,
    native_binary_rejected_error,
    run_interleaved,
)
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
from switchyard.importers.gaps import open_gaps
from switchyard.receipt import ReceiptError, check_tree, is_speed_row, load_jsonl, validate_row

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
        "prompt": "The switchyard counts the same cars on every pass.",
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
        notes=kwargs.get("notes", ""),
    )


# The E2 signature (driver 32.0.100.5540): the l512 compile wrote a complete
# blob and the NPU refused the native binary at execute (issue #33).
NATIVE_BINARY_REJECTED = (
    "L0 pfnAppendGraphExecute: ZE_RESULT_ERROR_INVALID_NATIVE_BINARY, "
    "code 0x7800000f - native binary is not supported by the device"
)


class ScriptedRaise(Scripted):
    """A sample that is an exception instance is raised, not returned."""

    def run(self, request):
        self.calls.append(request.step_id)
        self.requests.append(request)
        item = self.samples[request.step_id].pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _emb_arm(arm_id, device, shape):
    """An arm with the static shape the NPU lane needs."""

    arm = _arm(arm_id, device)
    arm["task"] = "embed"
    arm["batch"] = 1
    arm["static_shape"] = shape
    return arm


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


def test_a_failed_step_stops_and_flushes_the_measured_slots(tmp_path: Path):
    """Ruling #33: an abort keeps the slots already measured as aborted rows."""

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
    stored = load_jsonl(repo / "receipts" / "run-1.jsonl")
    assert [row["id"] for row in stored] == ["T/A/run-1"]
    row = stored[0]
    assert row["aborted"] is True
    assert "outcome" not in row
    assert row["latency"]["cold_ms"] == 100
    assert row["settings"]["abortion"] == {
        "stop_step": "B cold",
        "error": "stopped",
        "cold_done": True,
        "repeats_done": 0,
        "repeats_required": 2,
    }


def test_a_hang_is_not_a_device_loss():
    assert hang_error("hang: no result within 60 s") is True
    assert device_lost_error("hang: no result within 60 s") is False
    assert device_lost_error("ZE_RESULT_ERROR_DEVICE_LOST") is True
    assert device_lost_error("device_lost") is True
    assert device_lost_error("device lost: NPU disappeared") is True
    sentence = (
        "L0 zeFenceHostSynchronize result: ZE_RESULT_ERROR_DEVICE_LOST, "
        "code 0x70000001 - device hung, reset, was removed, or driver update occurred"
    )
    assert device_lost_error(sentence) is True
    assert device_lost_error("device hung") is False
    assert device_lost_error("device hung, reset, was removed") is False
    assert device_lost_error("") is False
    assert device_lost_error(None) is False


def test_a_native_binary_rejection_is_not_a_hang_or_a_device_loss():
    assert native_binary_rejected_error(NATIVE_BINARY_REJECTED) is True
    assert native_binary_rejected_error("ZE_RESULT_ERROR_INVALID_NATIVE_BINARY") is True
    assert native_binary_rejected_error("code 0x7800000f") is True
    assert native_binary_rejected_error("hang: no result within 60 s") is False
    assert native_binary_rejected_error("ZE_RESULT_ERROR_DEVICE_LOST") is False
    assert native_binary_rejected_error("ze_result_error_invalid_extension") is False
    assert native_binary_rejected_error("") is False
    assert native_binary_rejected_error(None) is False


def test_a_hang_keeps_finished_arms_and_stops_without_retry(tmp_path: Path, capsys):
    repo = tmp_path / "repo"
    spec = _spec()
    spec["warmups"] = 0
    spec["repeats"] = 1
    spec["arms"] = [_arm("A", "5090"), _arm("B", "NPU"), _arm("C", "iGPU")]
    spec_path = _bind(repo, spec)
    seen = {}

    class Watch(Scripted):
        def run(self, request):
            if request.step_id == "B" and len(self.samples["B"]) == 1:
                stored = load_jsonl(repo / "receipts" / "run-1.jsonl")
                seen["ids"] = [row["id"] for row in stored]
                seen["partial"] = stored[0]["partial"]
                seen["cold"] = stored[0]["latency"]["cold_ms"]
            return super().run(request)

    script = Watch(
        {
            "A": [_result(100, 4), _result(20, 8)],
            "B": [_result(5, 1), _result(999, ok=False, error="hang: no result within 60 s")],
            "C": [_result(7, 1)],
        }
    )
    (repo / "table").mkdir()
    with pytest.raises(RunnerError, match="hang:"):
        _invoke(repo, spec_path, {"fake": script})
    assert script.calls == ["A", "B", "C", "A", "B"]
    assert seen == {"ids": ["T/A/run-1"], "partial": True, "cold": 100}
    stored = {row["id"]: row for row in load_jsonl(repo / "receipts" / "run-1.jsonl")}
    assert set(stored) == {"T/A/run-1", "T/B/run-1", "T/C/run-1"}
    assert stored["T/A/run-1"]["partial"] is True
    assert stored["T/A/run-1"]["latency"]["cold_ms"] == 100
    assert "outcome" not in stored["T/A/run-1"]
    hung = stored["T/B/run-1"]
    assert hung["outcome"] == "hang"
    assert hung["partial"] is True
    assert hung["notes"] == "this device can't do this setting"
    assert all(value is None for value in hung["latency"].values())
    assert "artifacts" not in hung
    cut = stored["T/C/run-1"]
    assert cut["aborted"] is True
    assert "outcome" not in cut
    assert cut["latency"]["cold_ms"] == 7
    assert cut["settings"]["abortion"]["stop_step"] == "B repeat 0"
    assert cut["settings"]["abortion"]["repeats_done"] == 0
    assert cut["settings"]["abortion"]["repeats_required"] == 1
    table = {row["id"]: row for row in load_jsonl(repo / "table" / "routing.jsonl")}
    assert table["T/B/run-1"]["outcome"] == "hang"
    assert table["T/B/run-1"]["latency"]["cold_ms"] is None
    assert table["T/B/run-1"]["notes"] == "this device can't do this setting"
    assert "T/A/run-1" in table
    assert "T/C/run-1" not in table
    assert script.unloaded == [("qwen3:8b", "5090"), ("qwen3:8b", "NPU"), ("qwen3:8b", "iGPU")]
    assert capsys.readouterr().out == "B NPU hang\n"


def test_a_device_loss_keeps_a_finished_arm_and_the_other_device_goes_on(tmp_path: Path, capsys):
    repo = tmp_path / "repo"
    spec = _spec()
    spec["warmups"] = 0
    spec["repeats"] = 1
    spec["arms"] = [_arm("B", "NPU"), _arm("D", "NPU"), _arm("A", "5090")]
    spec_path = _bind(repo, spec)
    script = Scripted(
        {
            "B": [_result(100, 4), _result(20, 8)],
            "D": [_result(5, 1), _result(999, ok=True, resolved={"outcome": "device_lost"})],
            "A": [_result(50, 2), _result(40, 3)],
        }
    )
    (repo / "table").mkdir()
    rows = _invoke(repo, spec_path, {"fake": script})
    assert script.calls == ["B", "D", "A", "B", "D", "A"]
    by_id = {row["id"]: row for row in rows}
    assert set(by_id) == {"T/B/run-1", "T/D/run-1", "T/A/run-1"}
    assert "outcome" not in by_id["T/B/run-1"]
    assert by_id["T/B/run-1"]["latency"]["cold_ms"] == 100
    assert "partial" not in by_id["T/B/run-1"]
    assert by_id["T/A/run-1"]["latency"]["cold_ms"] == 50
    lost = by_id["T/D/run-1"]
    assert lost["outcome"] == "device_lost"
    assert "partial" not in lost
    assert lost["notes"] == "this device can't do this setting"
    assert all(value is None for value in lost["latency"].values())
    assert "artifacts" not in lost
    table = {row["id"]: row for row in load_jsonl(repo / "table" / "routing.jsonl")}
    assert table["T/D/run-1"]["latency"]["cold_ms"] is None
    assert table["T/D/run-1"]["notes"] == "this device can't do this setting"
    assert capsys.readouterr().out == "D NPU device_lost\ndone\n"


def test_a_not_run_step_writes_no_row_and_is_listed_in_the_notes(tmp_path: Path):
    repo = tmp_path / "repo"
    spec = _spec()
    spec["warmups"] = 0
    spec["repeats"] = 1
    spec_path = _bind(repo, spec)
    script = Scripted(
        {
            "A": [_result(100, 4), _result(20, 8)],
            "B": [_result(None, ok=True, resolved={"outcome": "not run"})],
        }
    )
    rows = _invoke(repo, spec_path, {"fake": script}, notes="kept")
    assert script.calls == ["A", "B", "A"]
    assert [row["id"] for row in rows] == ["T/A/run-1"]
    assert "outcome" not in rows[0]
    assert rows[0]["latency"]["cold_ms"] == 100
    assert rows[0]["notes"] == "kept. not run: B cold, B repeat 0"


def test_a_failed_device_loss_marks_the_device_off_and_continues(tmp_path: Path):
    repo = tmp_path / "repo"
    spec = _spec()
    spec["warmups"] = 0
    spec["repeats"] = 1
    spec_path = _bind(repo, spec)
    script = Scripted(
        {
            "A": [_result(100, 4), _result(20, 8)],
            "B": [_result(999, ok=False, error="device lost: NPU disappeared")],
        }
    )
    rows = _invoke(repo, spec_path, {"fake": script})
    assert script.calls == ["A", "B", "A"]
    by_id = {row["id"]: row for row in rows}
    assert by_id["T/B/run-1"]["outcome"] == "device_lost"
    assert all(value is None for value in by_id["T/B/run-1"]["latency"].values())
    assert by_id["T/A/run-1"]["latency"]["cold_ms"] == 100


def test_a_raised_hang_stops_the_run(tmp_path: Path):
    repo = tmp_path / "repo"
    spec = _spec()
    spec["warmups"] = 0
    spec["repeats"] = 1
    spec_path = _bind(repo, spec)

    class Hang:
        def __init__(self):
            self.calls = []

        def run(self, request):
            self.calls.append(request.step_id)
            if request.step_id == "B":
                raise AdapterError("hang: no result within 60 s")
            return _result(10, 1)

        def unload(self, model, device=None):
            return None

    script = Hang()
    with pytest.raises(RunnerError, match="hang:"):
        _invoke(repo, spec_path, {"fake": script})
    assert script.calls == ["A", "B"]
    stored = load_jsonl(repo / "receipts" / "run-1.jsonl")
    assert [row["id"] for row in stored] == ["T/B/run-1", "T/A/run-1"]
    assert stored[0]["outcome"] == "hang"
    assert all(value is None for value in stored[0]["latency"].values())
    assert stored[1]["aborted"] is True
    assert stored[1]["latency"]["cold_ms"] == 10
    assert stored[1]["settings"]["abortion"]["stop_step"] == "B cold"


def test_a_native_binary_rejection_is_a_capability_row_scoped_to_shape(tmp_path: Path, capsys):
    """Ruling #33.1: the l512 NPU arm records the outcome; every other shape runs."""

    repo = tmp_path / "repo"
    spec = _spec()
    spec["warmups"] = 0
    spec["repeats"] = 1
    spec["compile_cache"] = "fresh"
    spec["arms"] = [
        _emb_arm("npu512", "NPU", "1x512"),
        _emb_arm("npu128", "NPU", "1x128"),
        _emb_arm("cpu512", "CPU", "1x512"),
        _emb_arm("igpu512", "iGPU", "1x512"),
    ]
    spec_path = _bind(repo, spec)

    class Rejecting(ScriptedRaise):
        def run(self, request):
            item = self.samples[request.step_id][0]
            if isinstance(item, NativeBinaryRejected):
                blob = Path(request.settings["compile_cache_dir"]) / "compiled.blob"
                blob.write_bytes(b"x" * 4096)
            return super().run(request)

    script = Rejecting(
        {
            "npu512": [NativeBinaryRejected(NATIVE_BINARY_REJECTED, compile_s=41.25)],
            "npu128": [_result(30), _result(5)],
            "cpu512": [_result(50), _result(10)],
            "igpu512": [_result(70), _result(12)],
        }
    )
    (repo / "table").mkdir()
    rows = _invoke(repo, spec_path, {"fake": script})
    # The NPU is not marked off: after the l512 cold, l128, CPU and iGPU all run.
    # Only the rejected shape's repeat is skipped, and it is listed in the notes.
    assert script.calls == ["npu512", "npu128", "cpu512", "igpu512", "npu128", "cpu512", "igpu512"]
    by_id = {row["id"]: row for row in rows}
    assert set(by_id) == {"T/npu512/run-1", "T/npu128/run-1", "T/cpu512/run-1", "T/igpu512/run-1"}
    rejected = by_id["T/npu512/run-1"]
    assert rejected["outcome"] == "native_binary_rejected"
    assert rejected["provisional"] is True
    assert "partial" not in rejected
    assert "aborted" not in rejected
    latency = rejected["latency"]
    assert latency["compile_ms"] == pytest.approx(41250.0)
    for key in ("cold_ms", "warm_p50_ms", "warm_p95_ms", "throughput", "first_call_ms", "compile_warm_reload_ms"):
        assert latency[key] is None
    assert latency["throughput_unit"] is None
    assert "artifacts" not in rejected
    evidence = rejected["settings"]["rejection"]
    assert "ZE_RESULT_ERROR_INVALID_NATIVE_BINARY" in evidence["error"]
    assert evidence["code"] == "0x7800000f"
    assert evidence["shape"] == {"batch": 1, "static_shape": "1x512"}
    assert evidence["compile_cache_dir"] == "compile-cache/run-1/npu512"
    assert evidence["blob_bytes"] == 4096
    assert "npu_driver" in evidence
    assert rejected["notes"].startswith("this device can't do this setting")
    assert "not run: npu512 repeat 0" in rejected["notes"]
    for done in ("npu128", "cpu512", "igpu512"):
        row = by_id[f"T/{done}/run-1"]
        assert "outcome" not in row and "aborted" not in row and "partial" not in row
    table = {row["id"]: row for row in load_jsonl(repo / "table" / "routing.jsonl")}
    kept = table["T/npu512/run-1"]
    assert kept["outcome"] == "native_binary_rejected"
    assert kept["latency"]["compile_ms"] == pytest.approx(41250.0)
    assert kept["provisional"] is True
    assert capsys.readouterr().out == "npu512 NPU native_binary_rejected\ndone\n"
    assert check_tree(repo) == []


def test_a_rejection_after_a_success_is_a_state_change_and_aborts(tmp_path: Path):
    """Ruling #33.1: a shape that already ran, then rejected, is not a capability."""

    repo = tmp_path / "repo"
    spec = _spec()
    spec["warmups"] = 0
    spec["repeats"] = 2
    spec["arms"] = [_emb_arm("a", "NPU", "1x128"), _emb_arm("b", "NPU", "1x128")]
    spec_path = _bind(repo, spec)
    script = ScriptedRaise(
        {
            "a": [_result(10), _result(4), _result(4)],
            "b": [NativeBinaryRejected(NATIVE_BINARY_REJECTED, compile_s=30.0)],
        }
    )
    (repo / "table").mkdir()
    with pytest.raises(NativeBinaryRejected, match="0x7800000f"):
        _invoke(repo, spec_path, {"fake": script})
    assert script.calls == ["a", "b"]
    stored = load_jsonl(repo / "receipts" / "run-1.jsonl")
    assert [row["id"] for row in stored] == ["T/a/run-1"]
    row = stored[0]
    assert row["aborted"] is True
    assert "outcome" not in row
    assert row["settings"]["abortion"]["stop_step"] == "b cold"
    assert "0x7800000f" in row["settings"]["abortion"]["error"]
    table_rows = load_jsonl(repo / "table" / "routing.jsonl")
    assert all(row["aborted"] is not True for row in table_rows)
    assert "T/a/run-1" not in {row["id"] for row in table_rows}


def test_a_rejection_off_the_npu_aborts_the_run(tmp_path: Path):
    """Ruling #33.1: the capability bound holds on the NPU only."""

    repo = tmp_path / "repo"
    spec = _spec()
    spec["warmups"] = 0
    spec["repeats"] = 1
    spec["arms"] = [_emb_arm("npu128", "NPU", "1x128"), _emb_arm("cpu128", "CPU", "1x128")]
    spec_path = _bind(repo, spec)
    script = ScriptedRaise(
        {
            "npu128": [_result(10), _result(4)],
            "cpu128": [NativeBinaryRejected(NATIVE_BINARY_REJECTED, compile_s=0.5), _result(2)],
        }
    )
    with pytest.raises(NativeBinaryRejected, match="0x7800000f"):
        _invoke(repo, spec_path, {"fake": script})
    assert script.calls == ["npu128", "cpu128"]
    stored = load_jsonl(repo / "receipts" / "run-1.jsonl")
    assert [row["id"] for row in stored] == ["T/npu128/run-1"]
    assert stored[0]["aborted"] is True
    assert "outcome" not in stored[0]


def test_an_unknown_error_flushes_aborted_rows_and_gaps_gets_the_line(tmp_path: Path):
    """Ruling #33.1d: on abort every completed slot lands as an aborted row."""

    repo = tmp_path / "repo"
    spec = _spec()
    spec["warmups"] = 0
    spec["repeats"] = 2
    spec["arms"] = [_arm("A", "5090"), _arm("B", "NPU"), _arm("C", "iGPU")]
    spec_path = _bind(repo, spec)
    script = Scripted(
        {
            "A": [_result(100, 4), _result(20, 8), _result(21, 8)],
            "B": [_result(5, 1), _result(1, ok=False, error="some driver timeout weirdness")],
            "C": [_result(7, 1), _result(3, 1), _result(3, 1)],
        }
    )
    table = repo / "table"
    table.mkdir()
    (table / "gaps.md").write_text("# Gap report\n", encoding="utf-8")
    with pytest.raises(RunnerError, match="weirdness"):
        _invoke(repo, spec_path, {"fake": script})
    assert script.calls == ["A", "B", "C", "A", "B"]
    stored = {row["id"]: row for row in load_jsonl(repo / "receipts" / "run-1.jsonl")}
    assert set(stored) == {"T/A/run-1", "T/B/run-1", "T/C/run-1"}
    for row in stored.values():
        assert row["aborted"] is True
        assert "outcome" not in row
    assert stored["T/A/run-1"]["settings"]["abortion"]["repeats_done"] == 1
    assert stored["T/A/run-1"]["settings"]["abortion"]["repeats_required"] == 2
    assert stored["T/A/run-1"]["latency"]["warm_p50_ms"] == 20
    assert stored["T/B/run-1"]["settings"]["abortion"]["repeats_done"] == 0
    assert stored["T/B/run-1"]["settings"]["abortion"]["cold_done"] is True
    assert stored["T/B/run-1"]["latency"]["cold_ms"] == 5
    assert stored["T/C/run-1"]["settings"]["abortion"]["repeats_done"] == 0
    # The routing-table import refuses the aborted rows; the receipt keeps them.
    assert load_jsonl(table / "routing.jsonl") == []
    gaps = (table / "gaps.md").read_text(encoding="utf-8")
    line = "Aborted run `run-1` stopped at `B repeat 0`: some driver timeout weirdness"
    assert gaps.count(line) == 1
    assert check_tree(repo) == []


def test_the_rejected_row_validates_and_the_table_sees_pending_replication():
    """Ruling #33.2b: one occurrence is provisional "not supported", n=1."""

    row = device_lost_row(
        spec_id="T",
        arm=_emb_arm("npu512", "NPU", "1x512"),
        repeats=1,
        date="2026-10-10",
        spec_rel="experiments/t/spec.yaml",
        run_id="run-1",
        grant=None,
        env={"npu_firmware": "32.0.100.5540", "nvidia_driver": "unrecorded"},
        extra_settings={
            "rejection": {
                "error": NATIVE_BINARY_REJECTED,
                "code": "0x7800000f",
                "npu_driver": "32.0.100.5540",
                "shape": {"batch": 1, "static_shape": "1x512"},
                "compile_cache_dir": "compile-cache/run-1/npu512",
                "blob_bytes": 1212121212,
            }
        },
        outcome="native_binary_rejected",
        compile_ms=41250.0,
    )
    assert row["outcome"] == "native_binary_rejected"
    assert row["provisional"] is True
    assert row["latency"]["compile_ms"] == 41250.0
    assert is_speed_row(row) is False
    tabled = for_table(row)
    assert tabled["latency"]["compile_ms"] == 41250.0
    assert "artifacts" not in tabled
    assert tabled["notes"] == "this device can't do this setting"
    assert tabled["provisional"] is True
    # Timing beyond the compile refuses validation.
    bad = json.loads(json.dumps(row))
    bad["latency"]["cold_ms"] = 12.0
    with pytest.raises(ReceiptError):
        validate_row(bad)
    # The gap stays open: a capability row is not a speed.
    candidate = {
        "id": "e2-npu-l512",
        "experiment": "E2",
        "why": "needs a speed row at the shape",
        "match": {"model": "qwen3:8b", "task": "generate", "device": "NPU"},
        "needs": [],
    }
    assert open_gaps([tabled], [candidate])[0]["id"] == "e2-npu-l512"


def test_a_raised_device_loss_marks_the_device_off_and_continues(tmp_path: Path):
    repo = tmp_path / "repo"
    spec = _spec()
    spec["warmups"] = 0
    spec["repeats"] = 1
    spec_path = _bind(repo, spec)

    class Loss:
        def __init__(self):
            self.calls = []

        def run(self, request):
            self.calls.append(request.step_id)
            if request.step_id == "B":
                raise AdapterError("ZE_RESULT_ERROR_DEVICE_LOST")
            return _result(100, 4)

        def unload(self, model, device=None):
            return None

    script = Loss()
    rows = _invoke(repo, spec_path, {"fake": script})
    assert script.calls == ["A", "B", "A"]
    by_id = {row["id"]: row for row in rows}
    assert set(by_id) == {"T/A/run-1", "T/B/run-1"}
    assert by_id["T/A/run-1"]["latency"]["cold_ms"] == 100
    assert by_id["T/B/run-1"]["outcome"] == "device_lost"
    assert all(value is None for value in by_id["T/B/run-1"]["latency"].values())
    assert by_id["T/B/run-1"]["notes"] == "this device can't do this setting. not run: B repeat 0"


def test_an_unsupported_length_stays_a_measured_row(tmp_path: Path):
    repo = tmp_path / "repo"
    spec = _spec()
    spec["warmups"] = 0
    spec["repeats"] = 1
    spec["arms"] = [_arm("A", "NPU")]
    spec_path = _bind(repo, spec)
    script = Scripted(
        {
            "A": [
                _result(10, 1, resolved={"outcome": "unsupported length"}),
                _result(12, 1, resolved={"outcome": "unsupported length"}),
            ],
        }
    )
    rows = _invoke(repo, spec_path, {"fake": script})
    assert len(rows) == 1
    assert "outcome" not in rows[0]
    assert rows[0]["latency"]["cold_ms"] == 10


def test_the_importer_drops_a_speed_figure_from_device_lost(tmp_path: Path):
    repo = tmp_path / "repo"
    arm = _arm("B", "NPU")
    lost = device_lost_row(
        spec_id="T",
        arm=arm,
        repeats=2,
        date="2026-10-09",
        spec_rel="experiments/t/spec.yaml",
        run_id="run-1",
        grant="grant-t",
        env={"nvidia_driver": "unrecorded"},
    )
    lost["latency"]["cold_ms"] = 12
    lost["latency"]["throughput"] = 9
    lost["artifacts"] = {"slots": [{"latency_ms": 12}]}
    lost["notes"] = "arm B"
    receipts = repo / "receipts"
    receipts.mkdir(parents=True)
    (receipts / "run-1.jsonl").write_text(json.dumps(lost) + "\n", encoding="utf-8")
    (repo / "table").mkdir()
    imported = import_receipt(repo, "run-1")
    assert imported[0]["outcome"] == "device_lost"
    assert imported[0]["notes"] == "this device can't do this setting. arm B"
    assert all(value is None for value in imported[0]["latency"].values())
    assert "artifacts" not in imported[0]
    stored = load_jsonl(repo / "table" / "routing.jsonl")
    assert stored[0]["latency"]["throughput"] is None
    assert "artifacts" not in load_jsonl(repo / "receipts" / "run-1.jsonl")[0]
    hung_row = device_lost_row(
        spec_id="T",
        arm=arm,
        repeats=2,
        date="2026-10-09",
        spec_rel="experiments/t/spec.yaml",
        run_id="run-2",
        grant="grant-t",
        env={"nvidia_driver": "unrecorded"},
        outcome="hang",
    )
    hung_row["latency"]["cold_ms"] = 12
    hung_row["notes"] = "arm B"
    (receipts / "run-2.jsonl").write_text(json.dumps(hung_row) + "\n", encoding="utf-8")
    hung = import_receipt(repo, "run-2")
    assert hung[0]["outcome"] == "hang"
    assert hung[0]["latency"]["cold_ms"] is None
    assert hung[0]["notes"] == "this device can't do this setting. arm B"


class _Counting:
    def __init__(self, samples):
        self.script = Scripted(samples)
        self.counts = 0

    def count_tokens(self, model, text):
        self.counts += 1
        return len(text)

    def run(self, request):
        return self.script.run(request)

    def unload(self, model, device=None):
        self.script.unload(model, device)


def test_a_dataset_target_is_one_prefix_per_model_and_the_counts_land_on_the_row(tmp_path: Path):
    repo = tmp_path / "repo"
    spec = _spec()
    spec["warmups"] = 0
    spec["repeats"] = 1
    arm_a = _arm("A", "CPU")
    arm_b = _arm("B", "NPU")
    arm_a["static_shape"] = "1x4"
    arm_a["batch"] = 1
    arm_b["static_shape"] = "1x2"
    arm_b["batch"] = 1
    arm_b["model"] = "other-model"
    arm_c = _arm("C", "iGPU")
    arm_c["static_shape"] = "1x?"
    arm_c["batch"] = 1
    arm_d = _arm("D", "CPU")
    arm_d["static_shape"] = None
    arm_d["batch"] = 1
    for arm in (arm_a, arm_b, arm_c, arm_d):
        arm.pop("prompt")  # the dataset builds the prompt
    spec["arms"] = [arm_a, arm_b, arm_c, arm_d]
    spec["dataset"] = {"name": "e1-fixed-prompt", "seed": "ab", "target_tokens": 4}
    counter = _Counting(
        {
            "A": [_result(10, 1), _result(8, 1)],
            "B": [_result(9, 1), _result(7, 1)],
            "C": [_result(6, 1), _result(5, 1)],
            "D": [_result(4, 1), _result(3, 1)],
        }
    )
    rows = _invoke(repo, _bind(repo, spec), {"fake": counter})
    assert counter.counts == 10
    assert [request.settings["prompt"] for request in counter.script.requests] == ["abab"] * 8
    by_id = {row["id"]: row for row in rows}
    assert by_id["T/A/run-1"]["settings"]["prompt_tokens"] == 4
    assert by_id["T/A/run-1"]["settings"]["truncated_tokens"] == 0
    assert by_id["T/B/run-1"]["settings"]["prompt_tokens"] == 4
    assert by_id["T/B/run-1"]["settings"]["truncated_tokens"] == 2
    assert by_id["T/C/run-1"]["settings"]["prompt_tokens"] == 4
    assert "truncated_tokens" not in by_id["T/C/run-1"]["settings"]
    assert "compile_warm_reload_ms" not in by_id["T/A/run-1"]["latency"]


def test_the_adapter_token_counts_and_warm_reload_overwrite_the_prefix_estimate(tmp_path: Path):
    repo = tmp_path / "repo"
    spec = _spec()
    spec["warmups"] = 0
    spec["repeats"] = 1
    arm = _arm("A", "NPU")
    arm["static_shape"] = "1x4"
    arm["batch"] = 1
    arm.pop("prompt")  # the dataset builds the prompt
    spec["arms"] = [arm]
    spec["dataset"] = {"name": "e1-fixed-prompt", "seed": "ab", "target_tokens": 4}
    counter = _Counting(
        {
            "A": [
                _result(
                    10,
                    1,
                    compile_ms=4,
                    first_call_ms=3,
                    resolved={"compile_warm_reload_ms": 80.5, "prompt_tokens": 2048, "truncated_tokens": 1920},
                ),
                _result(8, 1),
            ]
        }
    )
    rows = _invoke(repo, _bind(repo, spec), {"fake": counter})
    assert rows[0]["latency"]["compile_warm_reload_ms"] == 80.5
    assert rows[0]["latency"]["compile_ms"] == 4
    assert rows[0]["settings"]["prompt_tokens"] == 2048
    assert rows[0]["settings"]["truncated_tokens"] == 1920


def test_a_receipt_row_carries_token_type_ids_fed(tmp_path: Path):
    repo = tmp_path / "repo"
    spec = _spec()
    spec["warmups"] = 0
    spec["repeats"] = 1
    fed = _arm("fed", "CPU")
    omitted = _arm("omitted", "NPU")
    absent = _arm("absent", "iGPU")
    spec["arms"] = [fed, omitted, absent]
    counter = _Counting(
        {
            "fed": [_result(4, 1, resolved={"token_type_ids_fed": True}), _result(2, 1)],
            "omitted": [_result(4, 1, resolved={"token_type_ids_fed": False}), _result(2, 1)],
            "absent": [_result(4, 1, resolved={}), _result(2, 1)],
        }
    )
    rows = _invoke(repo, _bind(repo, spec), {"fake": counter})
    by_id = {row["id"]: row for row in rows}
    assert by_id["T/fed/run-1"]["settings"]["token_type_ids_fed"] is True
    assert by_id["T/omitted/run-1"]["settings"]["token_type_ids_fed"] is False
    assert "token_type_ids_fed" not in by_id["T/absent/run-1"]["settings"]


def test_a_target_without_a_tokenizer_is_refused_before_a_step(tmp_path: Path):
    repo = tmp_path / "repo"
    spec = _spec()
    spec["warmups"] = 0
    spec["repeats"] = 1
    arm = _arm("A", "NPU")
    arm.pop("prompt")  # the dataset builds the prompt
    spec["arms"] = [arm]
    spec["dataset"] = {"name": "e1-fixed-prompt", "seed": "ab", "target_tokens": 4}
    script = Scripted({"A": [_result(1, 1)]})
    with pytest.raises(RunnerError, match="cannot count tokens"):
        _invoke(repo, _bind(repo, spec), {"fake": script})
    assert script.calls == []


def test_an_empty_seed_and_a_dataset_that_is_not_a_map_are_refused_or_skipped(tmp_path: Path):
    repo = tmp_path / "repo"
    spec = _spec()
    spec["warmups"] = 0
    spec["repeats"] = 1
    arm = _arm("A", "CPU")
    arm["static_shape"] = "1x?"
    arm.pop("prompt")
    spec["arms"] = [arm]
    spec["dataset"] = {"name": "e1-fixed-prompt", "seed": "", "target_tokens": 4}
    counter = _Counting({"A": [_result(1, 1)]})
    # an empty seed is no prompt source, so the generate arm is refused at load
    with pytest.raises(SpecError, match="has no prompt"):
        _invoke(repo, _bind(repo, spec), {"fake": counter})
    assert counter.script.calls == []

    repo2 = tmp_path / "plain"
    plain = _spec()
    plain["warmups"] = 0
    plain["repeats"] = 1
    plain["arms"] = [_arm("A", "CPU")]
    plain["dataset"] = "fixture"
    script = Scripted({"A": [_result(10, 1), _result(8, 1)]})
    rows = _invoke(repo2, _bind(repo2, plain), {"fake": script})
    # a dataset that is not a map is skipped; the arm's own prompt is sent
    assert script.requests[0].settings["prompt"] == plain["arms"][0]["prompt"]
    assert "prompt_tokens" not in rows[0]["settings"]


def test_a_seed_without_a_target_is_passed_through(tmp_path: Path):
    repo = tmp_path / "repo"
    spec = _spec()
    spec["warmups"] = 0
    spec["repeats"] = 1
    arm = _arm("A", "CPU")
    arm.pop("prompt")  # the seed is passed through as the prompt
    spec["arms"] = [arm]
    spec["dataset"] = {"name": "fixture", "seed": "hello"}
    script = Scripted({"A": [_result(10, 1), _result(8, 1)]})
    rows = _invoke(repo, _bind(repo, spec), {"fake": script})
    assert script.requests[0].settings["prompt"] == "hello"
    assert "prompt_tokens" not in rows[0]["settings"]
    assert "truncated_tokens" not in rows[0]["settings"]


def _guard_row(device, shape, vector, **extra):
    batch, _length = shape.split("x")
    row = {
        "id": extra.pop("id", f"{device}-{shape}"),
        "device": device,
        "model": {"repo_or_tag": "BAAI/bge-base-en-v1.5", "revision_or_digest": "rev"},
        "settings": {"batch": int(batch), "static_shape": shape},
        "artifacts": {"slots": [{"phase": "repeat", "output": vector}]},
        "latency": {"warm_p50_ms": 5, "compile_warm_reload_ms": 10},
    }
    row.update(extra)
    return row


def test_parity_marks_the_arm_that_missed_and_leaves_the_other_lengths():
    rows = [
        _guard_row("CPU", "1x128", [1.0, 0.0], id="cpu-128"),
        _guard_row("CPU", "1x512", [1.0, 0.0], id="cpu-512"),
        _guard_row("NPU", "1x128", [1.0, 0.0], id="npu-128"),
        _guard_row("NPU", "1x512", [0.0, 1.0], id="npu-512"),
        _guard_row("CPU", "8x512", [0.0, 1.0], id="cpu-b8"),
        _guard_row("NPU", "1x512", [1.0, 0.0, 0.0], id="dim"),
        _guard_row("CPU", "1x32", [0.0, 0.0], id="cpu-zero"),
        _guard_row("NPU", "1x32", [0.0, 0.0], id="zero"),
        _guard_row("NPU", "1x8", [1.0, 0.0], id="bad-shape", settings={"static_shape": "1xno"}),
        _guard_row("NPU", "1x8", [1.0, 0.0], id="bad-batch", settings={"static_shape": "nx8"}),
        _guard_row("NPU", "1x8", [1.0, 0.0], id="zero-len", settings={"batch": 1, "static_shape": "1x0"}),
        _guard_row("NPU", "1x8", [1.0, 0.0], id="from-shape", settings={"static_shape": "2x128"}),
        _guard_row("NPU", "1x8", [1.0, 0.0], id="cold-only", artifacts={"slots": [{"phase": "cold", "output": [1.0, 0.0]}]}),
        _guard_row("NPU", "1x8", "nope", id="text-out"),
        _guard_row("NPU", "1x512", [1.0, True], id="bool-out"),
        _guard_row("NPU", "1x64", [1.0, 0.0], id="no-ref"),
    ]
    rows.append({"id": "lost", "outcome": "device_lost", "device": "NPU", "parity_valid": True})
    rows.append({"id": "preset", "device": "iGPU", "settings": {}, "parity_valid": False})
    apply_parity(rows)
    by_id = {row["id"]: row for row in rows}
    assert by_id["cpu-128"]["parity_valid"] is True
    assert by_id["npu-128"]["parity_valid"] is True
    assert by_id["npu-512"]["parity_valid"] is False
    assert by_id["npu-512"]["latency"]["compile_warm_reload_ms"] == 10
    assert by_id["cpu-b8"]["parity_valid"] is False
    assert "parity_valid" not in by_id["dim"]
    assert "parity_valid" not in by_id["zero"]
    assert "parity_valid" not in by_id["bad-shape"]
    assert "parity_valid" not in by_id["bad-batch"]
    assert "parity_valid" not in by_id["zero-len"]
    assert "parity_valid" not in by_id["cold-only"]
    assert "parity_valid" not in by_id["text-out"]
    assert "parity_valid" not in by_id["bool-out"]
    assert "parity_valid" not in by_id["no-ref"]
    assert by_id["from-shape"]["parity_valid"] is True
    assert by_id["lost"]["parity_valid"] is True
    assert by_id["preset"]["parity_valid"] is False
    shorter_miss = [
        _guard_row("CPU", "1x128", [1.0, 0.0], id="cpu-128"),
        _guard_row("CPU", "1x512", [1.0, 0.0], id="cpu-512"),
        _guard_row("NPU", "1x128", [0.0, 1.0], id="npu-128"),
        _guard_row("NPU", "1x512", [1.0, 0.0], id="npu-512"),
    ]
    apply_parity(shorter_miss)
    marked = {row["id"]: row["parity_valid"] for row in shorter_miss}
    assert marked == {"cpu-128": True, "cpu-512": True, "npu-128": False, "npu-512": True}


def test_import_judges_a_later_receipt_against_the_table_and_keeps_the_timing(tmp_path: Path):
    from tests.support import row

    repo = tmp_path / "repo"
    (repo / "table").mkdir(parents=True)
    (repo / "receipts").mkdir(parents=True)
    settings = {
        "thinking_level": "off",
        "num_predict": None,
        "num_ctx": None,
        "batch": 1,
        "static_shape": "1x128",
        "precision": None,
    }
    cpu = row(
        id="E1/cpu/main",
        device="CPU",
        quality=[],
        settings=settings,
        provenance={"experiment_spec": "experiments/e1/spec.yaml", "run_id": "main"},
        artifacts={"slots": [{"phase": "repeat", "output": [1.0, 0.0]}]},
    )
    (repo / "table" / "routing.jsonl").write_text(json.dumps(cpu) + "\n", encoding="utf-8")
    npu = row(
        id="E1/npu/tail",
        device="NPU",
        quality=[],
        settings=settings,
        provenance={"experiment_spec": "experiments/e1/spec.yaml", "run_id": "tail"},
        artifacts={"slots": [{"phase": "cold", "output": [1.0, 0.0]}, {"phase": "repeat", "output": [0.0, 1.0]}]},
    )
    npu["latency"]["warm_p50_ms"] = 14
    npu["latency"]["compile_warm_reload_ms"] = 40
    (repo / "receipts" / "tail.jsonl").write_text(json.dumps(npu) + "\n", encoding="utf-8")
    imported = import_receipt(repo, "tail")
    assert imported[0]["parity_valid"] is False
    assert imported[0]["latency"]["warm_p50_ms"] == 14
    assert imported[0]["latency"]["compile_warm_reload_ms"] == 40
    stored = load_jsonl(repo / "table" / "routing.jsonl")
    assert [row["id"] for row in stored] == ["E1/cpu/main", "E1/npu/tail"]
    assert "parity_valid" not in stored[0]
    assert stored[1]["parity_valid"] is False
    assert stored[1]["latency"]["compile_warm_reload_ms"] == 40


def test_a_failed_parity_guard_is_recorded_on_the_run_and_not_served(tmp_path: Path):
    from switchyard.scope import select

    repo = tmp_path / "repo"
    spec = _spec()
    spec["warmups"] = 0
    spec["repeats"] = 1
    cpu = _arm("CPU", "CPU")
    npu = _arm("NPU", "NPU")
    for arm in (cpu, npu):
        arm["batch"] = 1
        arm["static_shape"] = "1x8"
        arm["model"] = "BAAI/bge-base-en-v1.5"
    spec["arms"] = [cpu, npu]
    script = Scripted(
        {
            "CPU": [_result(10, 1, output=[9.0, 0.0]), _result(4, 1, output=[1.0, 0.0])],
            "NPU": [
                _result(
                    12,
                    1,
                    compile_ms=3,
                    first_call_ms=2,
                    resolved={"compile_warm_reload_ms": 30.0},
                    output=[0.0, 1.0],
                ),
                _result(5, 1, output=[0.0, 1.0]),
            ],
        }
    )
    spec_path = _bind(repo, spec)
    (repo / "table").mkdir()
    rows = _invoke(repo, spec_path, {"fake": script})
    by_id = {row["id"]: row for row in rows}
    assert by_id["T/CPU/run-1"]["parity_valid"] is True
    assert by_id["T/NPU/run-1"]["parity_valid"] is False
    assert by_id["T/NPU/run-1"]["latency"]["warm_p50_ms"] == 5
    assert by_id["T/NPU/run-1"]["latency"]["compile_warm_reload_ms"] == 30.0
    assert "compile_warm_reload_ms" not in by_id["T/CPU/run-1"]["latency"]
    answer = select(rows, "generate")
    assert [row["id"] for row in answer.rows] == ["T/CPU/run-1"]
    stored = load_jsonl(repo / "table" / "routing.jsonl")
    npu_row = next(row for row in stored if row["id"] == "T/NPU/run-1")
    assert npu_row["parity_valid"] is False
    assert npu_row["latency"]["compile_warm_reload_ms"] == 30.0


def test_a_raised_spec_error_keeps_a_finished_arm(tmp_path: Path):
    repo = tmp_path / "repo"
    spec = _spec()
    spec["warmups"] = 0
    spec["repeats"] = 1
    spec_path = _bind(repo, spec)

    class Boom:
        def __init__(self):
            self.calls = []
            self.unloaded = []

        def run(self, request):
            self.calls.append(request.step_id)
            if len(self.calls) == 4:
                raise AdapterError("the NPU needs a static_shape")
            return _result(10, 1)

        def unload(self, model, device=None):
            self.unloaded.append((model, device))

    script = Boom()
    with pytest.raises(AdapterError, match="static_shape"):
        _invoke(repo, spec_path, {"fake": script})
    assert script.calls == ["A", "B", "A", "B"]
    stored = load_jsonl(repo / "receipts" / "run-1.jsonl")
    assert [row["id"] for row in stored] == ["T/A/run-1", "T/B/run-1"]
    assert stored[0]["partial"] is True
    assert "outcome" not in stored[0]
    assert stored[1]["aborted"] is True
    assert stored[1]["settings"]["abortion"]["stop_step"] == "B repeat 0"


def test_a_later_failure_keeps_a_finished_arm_without_calling_it_device_lost(tmp_path: Path):
    repo = tmp_path / "repo"
    spec = _spec()
    spec["warmups"] = 0
    spec["repeats"] = 1
    spec_path = _bind(repo, spec)
    script = Scripted(
        {
            "A": [_result(100, 4), _result(20, 8)],
            "B": [_result(5, 1), _result(1, ok=False, error="stopped")],
        }
    )
    with pytest.raises(RunnerError, match="stopped"):
        _invoke(repo, spec_path, {"fake": script})
    assert script.calls == ["A", "B", "A", "B"]
    stored = load_jsonl(repo / "receipts" / "run-1.jsonl")
    assert [row["id"] for row in stored] == ["T/A/run-1", "T/B/run-1"]
    assert stored[0]["partial"] is True
    assert stored[0]["latency"]["cold_ms"] == 100
    assert "outcome" not in stored[0]
    assert stored[1]["aborted"] is True
    assert stored[1]["latency"]["cold_ms"] == 5
    assert stored[1]["settings"]["abortion"] == {
        "stop_step": "B repeat 0",
        "error": "stopped",
        "cold_done": True,
        "repeats_done": 0,
        "repeats_required": 1,
    }


def test_repeats_build_a_receipt_and_keep_cold_separate(tmp_path: Path, capsys):
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
    assert capsys.readouterr().out == "done\n"
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
    assert "rest_s" not in arm_a["settings"] and "rest" not in arm_a["settings"]
    assert "keep_alive" not in arm_a["settings"]
    assert arm_a["settings"]["environment"]["nvidia_driver"] == "unrecorded"
    assert arm_a["provenance"] == {"experiment_spec": "experiments/t/spec.yaml", "run_id": "run-1"}
    assert "partial" not in arm_a and "outcome" not in arm_a
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
    assert "prompt built at run time by prompt_eval_count" in e4_text
    assert "the final prefix is counted again from a cold load" in e4_text
    assert "prompt built at run time by prompt_eval_count" not in e5_text
    assert "slots: 79" in e4_text
    assert "order: aba" in e4_text
    assert "compile_cache: fresh" in e4_text
    assert "rest_s" not in e4_text
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
    # rest_s is no longer read: any value, or none, is accepted
    validate_spec({**card, "rest_s": 899})
    validate_spec({**good, "rest_s": True})
    validate_spec({key: value for key, value in good.items() if key != "rest_s"})
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
    stored = load_jsonl(repo2 / "receipts" / "run-2.jsonl")
    assert [row["id"] for row in stored] == ["T/A/run-2", "T/P/run-2"]
    assert all(row["aborted"] is True for row in stored)
    assert "start_session" in stored[0]["settings"]["abortion"]["error"]


def test_a_companion_session_gets_its_own_arms_bound(tmp_path: Path):
    class BoundSessions(Scripted):
        def start_session(self, request):
            super().start_session(request)
            self.session_bounds.append(request.settings.get("call_timeout_s"))

    alone = _arm("A", "5090")
    paired = _arm("P", "NPU")
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
    script = BoundSessions(
        {
            # A's cold call takes 50 s, so A's warm bound is 150 s; P's cold is 8 ms, so P's is 60 s.
            "A": [_result(50_000, 100, compile_ms=12), _result(40, 100), _result(40, 100)],
            "P": [_result(8, 100, compile_ms=4), _result(9, 100)],
        },
        sessions=True,
    )
    script.session_bounds = []
    repo = tmp_path / "repo"
    _invoke(repo, _bind(repo, spec), {"fake": script})
    assert script.sessions[0] == ("start", "A", "5090")
    assert script.session_bounds == [150.0]
    measured = [r.settings.get("call_timeout_s") for r in script.requests if r.step_id == "P"]
    assert measured[-1] == 60.0


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
    assert not list((repo3 / "receipts").glob("*.jsonl"))  # the run-start marker may exist; no receipt does


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


def test_environment_pins_openvino_the_npu_driver_and_packages(monkeypatch):
    from switchyard.harness import environment as env_capture

    versions = {"openvino": "2026.4.1", "openvino-tokenizers": "2026.4.1.0"}

    def fake_dist(name: str) -> str:
        return versions.get(name.lower().replace("_", "-"), "unrecorded")

    monkeypatch.setattr(env_capture, "dist_version", fake_dist)
    monkeypatch.setattr(env_capture, "npu_driver_version", lambda: "32.0.100.5540")
    captured = environment()
    assert captured["openvino"] == "2026.4.1"
    assert captured["npu_firmware"] == "32.0.100.5540"
    assert captured["packages"] == "openvino==2026.4.1 openvino-tokenizers==2026.4.1.0"
    assert captured["power_plan"] == "unrecorded"
    assert captured["ollama_ps"] == "unrecorded"
    assert captured["nvidia_driver"] == "unrecorded"
    assert environment(openvino="kept")["openvino"] == "kept"
    assert driver_version_from_lines(["32.0.100.5540", "32.0.100.5540"]) == "32.0.100.5540"
    assert driver_version_from_lines(["32.0.100.5540", "32.0.100.1"]) == "unrecorded"
    assert driver_version_from_lines(["32.0.100.5540 extra"]) == "unrecorded"
    assert driver_version_from_lines([]) == "unrecorded"
    assert package_versions(()) == "unrecorded"


def test_dist_version_reads_metadata_and_a_missing_package_stays_unrecorded(monkeypatch):
    from switchyard.harness import environment as env_capture

    assert env_capture.dist_version("this-package-is-not-installed") == "unrecorded"
    assert env_capture.dist_version("pytest") != "unrecorded"

    def blank(_name):
        return "  "

    monkeypatch.setattr(env_capture, "version", blank)
    assert env_capture.dist_version("pytest") == "unrecorded"

    def broken(_name):
        raise RuntimeError("metadata failed")

    monkeypatch.setattr(env_capture, "version", broken)
    assert env_capture.dist_version("pytest") == "unrecorded"


def test_the_npu_driver_query_keeps_one_version_and_does_not_guess(monkeypatch):
    from switchyard.harness import environment as env_capture

    calls = []

    class Completed:
        def __init__(self, code, out):
            self.returncode = code
            self.stdout = out

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        return Completed(0, "32.0.100.5540\n")

    monkeypatch.setattr(env_capture.sys, "platform", "win32")
    monkeypatch.setattr(env_capture.subprocess, "run", fake_run)
    env_capture.clear_driver_cache()
    assert env_capture.npu_driver_version() == "32.0.100.5540"
    assert env_capture.npu_driver_version() == "32.0.100.5540"
    assert len(calls) == 1

    def failed(cmd, **_kwargs):
        return Completed(1, "")

    monkeypatch.setattr(env_capture.subprocess, "run", failed)
    env_capture.clear_driver_cache()
    assert env_capture.npu_driver_version() == "unrecorded"

    def explode(cmd, **_kwargs):
        raise env_capture.subprocess.TimeoutExpired(cmd, 30)

    monkeypatch.setattr(env_capture.subprocess, "run", explode)
    env_capture.clear_driver_cache()
    assert env_capture.npu_driver_version() == "unrecorded"

    monkeypatch.setattr(env_capture.sys, "platform", "linux")
    env_capture.clear_driver_cache()
    assert env_capture.npu_driver_version() == "unrecorded"


def test_a_run_records_the_captured_environment_on_the_row(tmp_path, monkeypatch):
    from switchyard.harness import environment as env_capture

    monkeypatch.setattr(env_capture, "dist_version", lambda name: "2026.4.1" if name == "openvino" else "unrecorded")
    monkeypatch.setattr(env_capture, "npu_driver_version", lambda: "32.0.100.5540")
    monkeypatch.setattr(env_capture, "package_versions", lambda: "openvino==2026.4.1")
    repo = tmp_path / "repo"
    spec = _spec()
    spec["warmups"] = 0
    spec["repeats"] = 1
    spec["needs"] = {"grant": False, "npu_session": False, "cpu_window": False}
    spec["arms"] = [_arm("A", "CPU")]
    adapter = MeasureOnly({"A": [_result(4.0), _result(2.0)]})
    rows = _invoke(repo, _bind(repo, spec), {"fake": adapter}, grant=None, npu_session=None)
    env = rows[0]["settings"]["environment"]
    assert env["openvino"] == "2026.4.1"
    assert env["npu_firmware"] == "32.0.100.5540"
    assert env["packages"] == "openvino==2026.4.1"
    assert env["ollama_ps"] == "unrecorded"


def _commit(repo: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "dev"], cwd=repo, check=True)
    subprocess.run(["git", "config", "core.autocrlf", "false"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=repo, check=True, capture_output=True)


def test_a_generate_row_carries_genai_counter_medians_from_repeats(tmp_path: Path):
    repo = tmp_path / "repo"
    spec = _spec()
    spec["warmups"] = 0
    spec["repeats"] = 3
    gen = _arm("gen", "NPU")
    plain = _arm("plain", "CPU")
    spec["arms"] = [gen, plain]

    def g(ttft, tpot):
        return {"genai": {"ttft_ms": ttft, "tpot_ms": tpot, "input_tokens": 1024, "generated_tokens": 128}}

    counter = _Counting(
        {
            # the cold call's counters must not count; repeats give medians 30 / 5
            "gen": [_result(9, 1, resolved=g(999, 99)), _result(4, 1, resolved=g(30, 5)),
                    _result(4, 1, resolved=g(10, 4)), _result(4, 1, resolved=g(50, 6))],
            "plain": [_result(4, 1), _result(2, 1), _result(2, 1), _result(2, 1)],
        }
    )
    rows = _invoke(repo, _bind(repo, spec), {"fake": counter})
    by_id = {row["id"]: row for row in rows}
    genai = by_id["T/gen/run-1"]["settings"]["genai"]
    assert genai["ttft_ms_p50"] == 30 and genai["tpot_ms_p50"] == 5
    assert genai["input_tokens"] == 1024 and genai["generated_tokens"] == 128
    assert genai["repeats_counted"] == 3
    assert "genai" not in by_id["T/plain/run-1"]["settings"]


# ---------------------------------------------------------------------------
# Ruling #36: the per-arm per-call bound, and the CPU-noise addendum.
# ---------------------------------------------------------------------------


from types import SimpleNamespace


class Timed:
    """A fake adapter whose calls take planned seconds, honouring the bound.

    The runner hands each request its per-arm bound under
    ``settings["call_timeout_s"]``; a call planned above it answers the
    adapter's hang verdict, exactly as the OpenVINO adapter would.
    """

    def __init__(self, plan: dict[str, list[float]]):
        self.plan = plan
        self.calls = []
        self.bounds = []
        self.unloaded = []

    def run(self, request):
        self.calls.append(request.step_id)
        bound = request.settings.get("call_timeout_s", 60.0)
        self.bounds.append(bound)
        seconds = self.plan[request.step_id].pop(0)
        if seconds > bound:
            return StepResult(
                ok=False,
                error=f"hang: no result within {bound:g} s; arm inconclusive, device pair poisoned for the run",
            )
        return _result(seconds * 1000, 4)

    def unload(self, model, device=None):
        self.unloaded.append((model, device))


def _timed_spec():
    spec = _spec()
    spec["warmups"] = 1
    spec["repeats"] = 1
    spec["arms"] = [_arm("A", "CPU")]
    return spec


def test_a_cold_call_above_60s_passes_under_the_ceiling(tmp_path: Path):
    """Ruling #36a: the cold call has no reference, so it gets the ceiling."""

    repo = tmp_path / "repo"
    spec = _timed_spec()
    script = Timed({"A": [90.0, 120.0, 200.0]})
    rows = _invoke(repo, _bind(repo, spec), {"fake": script})
    assert script.bounds == [600.0, 270.0, 270.0]
    row = rows[0]
    assert row["settings"]["bound_s"] == {"cold": 600.0, "warm": 270.0}
    assert row["settings"]["cold_call_s"] == 90.0
    slots = {s["phase"]: s for s in row["artifacts"]["slots"]}
    assert slots["cold"]["bound_s"] == 600.0
    assert slots["warmup"]["bound_s"] == 270.0


def test_a_warm_call_between_60s_and_3x_cold_passes(tmp_path: Path):
    """Ruling #36a: max(60 s, 3 x the arm's own cold call) is the warm bound."""

    repo = tmp_path / "repo"
    spec = _timed_spec()
    script = Timed({"A": [30.0, 75.0, 80.0]})
    rows = _invoke(repo, _bind(repo, spec), {"fake": script})
    assert script.bounds == [600.0, 90.0, 90.0]
    assert rows[0]["latency"]["warm_p50_ms"] == 80000


def test_a_warm_call_above_3x_cold_is_a_hang(tmp_path: Path):
    """Ruling #36a: exceeding the arm's own bound is still the hang verdict."""

    repo = tmp_path / "repo"
    spec = _timed_spec()
    script = Timed({"A": [10.0, 61.0]})
    with pytest.raises(RunnerError, match="hang:"):
        _invoke(repo, _bind(repo, spec), {"fake": script})
    assert script.bounds == [600.0, 60.0]
    stored = load_jsonl(repo / "receipts" / "run-1.jsonl")
    assert stored[0]["outcome"] == "hang"
    assert stored[0]["settings"]["bound_s"] == {"cold": 600.0, "warm": 60.0}


def test_nothing_exceeds_the_ceiling(tmp_path: Path):
    """Ruling #36a: C bounds every call, the cold call included."""

    repo = tmp_path / "repo"
    spec = _timed_spec()
    spec["call_ceiling_s"] = 100
    script = Timed({"A": [150.0]})
    with pytest.raises(RunnerError, match="hang: no result within 100"):
        _invoke(repo, _bind(repo, spec), {"fake": script})
    assert script.bounds == [100.0]


def test_call_ceiling_s_is_validated(tmp_path: Path):
    spec = _timed_spec()
    assert validate_spec({**spec, "call_ceiling_s": 120})["call_ceiling_s"] == 120
    with pytest.raises(SpecError, match="call_ceiling_s"):
        validate_spec({**spec, "call_ceiling_s": 59})
    with pytest.raises(SpecError, match="call_ceiling_s"):
        validate_spec({**spec, "call_ceiling_s": True})
    with pytest.raises(SpecError, match="call_ceiling_s"):
        validate_spec({**spec, "call_ceiling_s": "600"})


def test_an_arm_the_spec_says_not_run_is_declined(tmp_path: Path):
    """Ruling #36b: the amendment excludes an arm with ``not_run: <reason>``."""

    repo = tmp_path / "repo"
    spec = _timed_spec()
    l512 = _emb_arm("l512", "NPU", "1x512")
    l512["not_run"] = "not supported at this shape on driver 32.0.100.5540"
    spec["arms"] = [_arm("A", "CPU"), l512]
    script = Timed({"A": [5.0, 5.0, 5.0], "l512": [1.0, 1.0, 1.0]})
    rows = _invoke(repo, _bind(repo, spec), {"fake": script})
    assert script.calls == ["A", "A", "A"]
    assert [row["id"] for row in rows] == ["T/A/run-1"]
    assert rows[0]["notes"] == "not run: l512 (the spec says not run: not supported at this shape on driver 32.0.100.5540)"


def test_not_run_needs_a_reason():
    arm = _arm("A", "NPU")
    arm["not_run"] = ""
    with pytest.raises(SpecError, match="not_run"):
        validate_spec({**_timed_spec(), "arms": [arm]})
    arm["not_run"] = True
    with pytest.raises(SpecError, match="not_run"):
        validate_spec({**_timed_spec(), "arms": [arm]})


class _FakePsutil:
    """The addendum's counter source: fixed percent and frequency readings."""

    def __init__(self, percent=42.0, mhz=3500.0):
        self.percent = percent
        self.mhz = mhz

    def cpu_percent(self, interval=None):
        return self.percent

    def cpu_freq(self):
        return SimpleNamespace(current=self.mhz)


def test_a_slot_records_cpu_load_and_frequency(monkeypatch, tmp_path: Path):
    """The #36 addendum: psutil readings at each slot edge, summarized max."""

    import switchyard.harness.runner as mod

    monkeypatch.setattr(mod, "psutil", _FakePsutil())
    repo = tmp_path / "repo"
    spec = _timed_spec()
    script = Timed({"A": [5.0, 5.0, 5.0]})
    rows = _invoke(repo, _bind(repo, spec), {"fake": script})
    row = rows[0]
    assert row["settings"]["cpu_percent_max"] == 42.0
    cold = row["artifacts"]["slots"][0]
    assert cold["cpu"]["slot_start"] == {"cpu_percent": 42.0, "cpu_freq_mhz": 3500.0}
    assert cold["cpu"]["slot_end"] == {"cpu_percent": 42.0, "cpu_freq_mhz": 3500.0}


def test_cpu_noise_is_unrecorded_and_never_a_failure(monkeypatch, tmp_path: Path):
    """The #36 addendum: without psutil the field says unrecorded and the run passes."""

    import switchyard.harness.runner as mod

    monkeypatch.setattr(mod, "psutil", None)
    repo = tmp_path / "repo"
    spec = _timed_spec()
    script = Timed({"A": [5.0, 5.0, 5.0]})
    rows = _invoke(repo, _bind(repo, spec), {"fake": script})
    row = rows[0]
    assert row["settings"]["cpu_percent_max"] == "unrecorded"
    cold = row["artifacts"]["slots"][0]
    assert cold["cpu"] == {"slot_start": "unrecorded", "slot_end": "unrecorded"}
