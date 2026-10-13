"""Hot cache, mmap, cold copy and warm-up until steady. Fakes only.

Nothing here starts a device, runs robocopy, queries a physical disk, or
downloads a model.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest
import yaml

from interlock.adapters import openvino as ov_mod
from interlock.adapters.base import AdapterError, StepRequest
from interlock.harness import watch as watch_mod
from interlock.harness.runner import RunnerError, dry_run, run_interleaved
from interlock.harness.spec import SpecError, validate_spec
from interlock.harness.steady import BlockPlan, SlotStream, UntilSteady
from interlock.harness.storage import (
    describe_drive,
    drive_letter,
    hot_snapshot,
    parse_disk_json,
    physical_disk_script,
)
from interlock.harness.watch import WatchHooks
from tests.test_harness import Scripted, _result
from tests.test_openvino_adapter import FakeGenai, FakePipeline, make_adapter

REPO = "OpenVINO/Phi-3.5-mini-instruct-int4-cw-ov"
REV = "dc27b4e451118b1e82c159a839476df02797837a"
MODEL = f"{REPO}@{REV}"
ENV = {
    "nvidia_driver": "unrecorded",
    "npu_firmware": "unrecorded",
    "openvino": "unrecorded",
    "power_plan": "unrecorded",
    "ollama_ps": "unrecorded",
    "packages": "unrecorded",
    "cuda_backend": "unrecorded",
    "igpu_shared_memory_override": "unrecorded",
}
HOOKS = WatchHooks(state_sampler=lambda _uses_5090: {}, start_thread=False, ram_sampler=lambda: 1.0)


def _commit(repo: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    for key, value in (
        ("user.email", "dev@example.com"),
        ("user.name", "dev"),
        ("core.autocrlf", "false"),
        ("core.longpaths", "true"),
    ):
        subprocess.run(["git", "config", key, value], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=repo, check=True, capture_output=True)


def _bind(repo: Path, spec: dict, name: str = "phi") -> Path:
    path = repo / "experiments" / name / "spec.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8", newline="\n")
    _commit(repo)
    return path


def _run(repo: Path, spec: dict, adapters: dict, *, name: str = "phi", run_id: str = "phi-1", watch=HOOKS):
    return run_interleaved(
        adapters,
        grant=None,
        npu_session=None,
        cpu_window=None,
        root=repo,
        repo=repo,
        spec_path=_bind(repo, spec, name),
        run_id=run_id,
        date="2026-10-11",
        env=ENV,
        watch=watch,
    )


def _arm(arm_id: str = "phi-35-mini-cpu-l1024", **over) -> dict:
    arm = {
        "id": arm_id,
        "task": "generate",
        "device": "CPU",
        "engine": "openvino",
        "model": MODEL,
        "revision_or_digest": REV,
        "quant": "int4",
        "licence": "mit",
        "engine_version": "2026.4.1",
        "thinking_level": "off",
        "num_predict": 128,
        "num_ctx": None,
        "batch": 1,
        "static_shape": None,
        "precision": None,
        "prompt": "The switchyard counts the same cars on every pass.",
    }
    arm.update(over)
    return arm


def _spec(arms: list[dict] | None = None, **over) -> dict:
    spec = {
        "id": "PHI-RERUN",
        "procedure": "interleaved",
        "question": "Does Phi stay steady once the weights are on the internal drive?",
        "decision": "Warm until two calls agree, then measure the repeats.",
        "needs": {"grant": False, "npu_session": False, "cpu_window": True},
        "stop_rules": ["a failed step stops the sequence"],
        "dataset": {"name": "phi-rerun", "seed": "The switchyard counts the same cars on every pass. "},
        "metrics": ["latency_ms", "throughput"],
        "decision_metric": "latency.throughput",
        "warmups": {"until_steady": True, "max": 6, "tolerance": 0.05},
        "repeats": 2,
        "band": 0.1,
        "compile_cache": "reuse",
        "arms": arms or [_arm()],
    }
    spec.update(over)
    return spec


def _plant(root: Path, revision: str, *, ref: str | None = None) -> Path:
    folder = root / "models--OpenVINO--Phi-3.5-mini-instruct-int4-cw-ov"
    snap = folder / "snapshots" / revision
    snap.mkdir(parents=True)
    (snap / "openvino_model.xml").write_text("<xml/>", encoding="utf-8")
    if ref is not None:
        refs = folder / "refs"
        refs.mkdir()
        (refs / ref).write_text(revision + "\n", encoding="utf-8")
    return snap


def _req(model: str, **settings) -> StepRequest:
    return StepRequest(
        step_id="phi",
        task="generate",
        model=model,
        device="CPU",
        engine="openvino",
        settings={"prompt": "hi", "num_predict": 8, **settings},
    )


def _facts(letter: str) -> dict:
    return {"drive": letter, "bus": "NVMe", "media": "SSD"}


class _Completed:
    def __init__(self, code: int) -> None:
        self.returncode = code


def _order(stream: SlotStream, values: dict[str, list[float | None]]) -> list[tuple]:
    order = []
    while True:
        item = stream.pop()
        if item is None:
            break
        slot, _block = item
        order.append((slot.phase, slot.step_id, slot.round_index))
        if slot.phase == "warmup":
            stream.note(slot, values[slot.step_id].pop(0))
        else:
            stream.note(slot, None)
    return order


def test_a_hot_copy_resolves_and_a_missing_copy_is_refused(tmp_path: Path):
    root = tmp_path / "hot"
    snap = _plant(root, REV, ref="main")
    downloads: list[tuple] = []

    def download(repo, revision):
        downloads.append((repo, revision))
        raise AssertionError("hot must not download")

    adapter = make_adapter(snapshot_download=download, hot_root=root, disk_facts=_facts)
    result = adapter.run(_req(MODEL, hf_cache="hot"))
    assert result.ok
    assert Path(next(iter(adapter._pipelines.values()))["path"]) == snap
    assert result.resolved["storage"]["bus"] == "NVMe"
    assert downloads == []

    # a ref name follows the commit file once
    assert hot_snapshot(root, REPO, "main") == snap

    archive = tmp_path / "archive"
    _plant(archive, REV)
    missing = make_adapter(snapshot_download=download, hot_root=tmp_path / "empty", disk_facts=_facts)
    with pytest.raises(AdapterError, match="refusing instead of downloading"):
        missing.run(_req(MODEL, hf_cache="hot"))
    assert downloads == []

    local = tmp_path / "local"
    local.mkdir()
    local_adapter = make_adapter(snapshot_download=download, hot_root=root, disk_facts=_facts)
    with pytest.raises(AdapterError, match="does not fall back"):
        local_adapter.run(_req(str(local), hf_cache="hot"))
    assert downloads == []


def test_archive_fetches_with_no_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple] = []

    def download(repo, revision):
        calls.append((repo, revision))
        dest = tmp_path / "snap"
        dest.mkdir()
        (dest / "openvino_model.xml").write_text("<xml/>", encoding="utf-8")
        return str(dest)

    adapter = make_adapter(snapshot_download=download, disk_facts=_facts)
    result = adapter.run(_req(MODEL, hf_cache="archive"))
    assert result.ok
    assert calls == [(REPO, REV)]
    assert Path(next(iter(adapter._pipelines.values()))["path"]) == tmp_path / "snap"

    module = types.ModuleType("huggingface_hub")

    def snapshot_download(repo, revision=None, **kwargs):
        module.calls.append((repo, revision, kwargs))
        return str(tmp_path / "snap")

    module.calls = []
    module.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    from interlock.harness.storage import default_snapshot_download

    assert default_snapshot_download(REPO, REV) == str(tmp_path / "snap")
    assert module.calls == [(REPO, REV, {})]


def test_enable_mmap_reaches_the_pipeline_and_splits_the_cache(tmp_path: Path):
    local = tmp_path / "model"
    local.mkdir()
    (local / "openvino_model.xml").write_text("<xml/>", encoding="utf-8")
    adapter = make_adapter(disk_facts=_facts)
    adapter.run(_req(str(local), compile_config={"ENABLE_MMAP": True}))
    adapter.run(_req(str(local), compile_config={"ENABLE_MMAP": False}))
    assert len(adapter._pipelines) == 2
    flags = sorted(entry["pipe"].config["ENABLE_MMAP"] for entry in adapter._pipelines.values())
    assert flags == [False, True]
    third = adapter.run(_req(str(local), compile_config={"ENABLE_MMAP": True}))
    assert third.ok
    assert "storage" not in (third.resolved or {})
    assert len(adapter._pipelines) == 2


def test_until_steady_stops_when_two_calls_agree():
    stream = SlotStream([], UntilSteady([BlockPlan(None, ["phi"], False, 2)], 6, 0.05))
    order = _order(stream, {"phi": [10.0, 10.0]})
    assert order == [
        ("cold", "phi", 0),
        ("warmup", "phi", 0),
        ("warmup", "phi", 1),
        ("repeat", "phi", 0),
        ("repeat", "phi", 1),
    ]
    assert stream.is_steady("phi")


def test_until_steady_at_max_is_unsteady():
    stream = SlotStream([], UntilSteady([BlockPlan(None, ["phi"], False, 2)], 2, 0.05))
    order = _order(stream, {"phi": [10.0, 20.0]})
    assert [item[0] for item in order].count("warmup") == 2
    assert order[-2:] == [("repeat", "phi", 0), ("repeat", "phi", 1)]
    assert stream.is_steady("phi") is False


def test_until_steady_waits_until_every_arm_in_the_block_settles():
    stream = SlotStream([], UntilSteady([BlockPlan("phi-cpu", ["A", "B"], False, 1)], 4, 0.05))
    order = _order(stream, {"A": [10.0, 10.0], "B": [10.0, 20.0, 20.0]})
    warm = [item for item in order if item[0] == "warmup"]
    assert ("warmup", "A", 2) not in warm
    assert ("warmup", "B", 2) in warm
    repeats = [item for item in order if item[0] == "repeat"]
    assert repeats == [("repeat", "A", 0), ("repeat", "B", 0)]
    assert order.index(("repeat", "A", 0)) > order.index(("warmup", "B", 2))


def test_a_drift_block_has_no_cold_slot():
    stream = SlotStream([], UntilSteady([BlockPlan("drift", ["phi-drift"], True, 2)], 2, 0.05))
    order = _order(stream, {"phi-drift": [10.0, 10.0]})
    assert [item[0] for item in order] == ["warmup", "warmup", "repeat", "repeat"]


def test_until_steady_converges_and_keeps_warmups_out_of_the_percentile(tmp_path: Path):
    spec = _spec(
        warmups={"until_steady": True, "max": 6, "tolerance": 0.05},
        repeats=2,
        needs={"grant": False, "npu_session": False, "cpu_window": False},
    )
    spec["arms"] = [_arm(engine="fake", model="phi-fake", hf_cache=None)]
    # hf_cache absent: pop the key so the arm is an ordinary generate arm
    spec["arms"][0].pop("hf_cache", None)
    samples = [
        _result(1.0, 10.0),  # cold
        _result(1.0, 10.0),  # warmup 0
        _result(1.0, 10.0),  # warmup 1 agrees
        _result(9.0, 30.0),
        _result(9.0, 30.0),
    ]
    adapter = Scripted({"phi-35-mini-cpu-l1024": samples})
    rows = _run(tmp_path / "repo", spec, {"fake": adapter})
    assert len(adapter.calls) == 5
    row = rows[0]
    assert row["latency"]["warm_p50_ms"] == 9.0
    assert row["latency"]["throughput"] == 30.0
    assert row["settings"]["unsteady"] is False
    assert row["settings"]["warmup_trace"] == [
        {"round": 0, "latency_ms": 1.0, "throughput": 10.0},
        {"round": 1, "latency_ms": 1.0, "throughput": 10.0},
    ]
    assert row["settings"]["hf_cache"] == "archive"
    assert row["settings"]["cold_copy"] is False


def test_until_steady_runs_one_block_then_the_next(tmp_path: Path):
    spec = _spec(
        arms=[_arm("A", engine="fake", model="a"), _arm("B", engine="fake", model="b")],
        order="blocked",
        repeats=1,
        warmups={"until_steady": True, "max": 2, "tolerance": 0.05},
        needs={"grant": False, "npu_session": False, "cpu_window": False},
        blocks=[{"id": "one", "arms": ["A"]}, {"id": "two", "arms": ["B"]}],
        drift_block={"id": "drift", "repeat_of": "one", "arms": ["A"], "repeats": 1},
    )
    samples = {
        "A": [_result(1.0, 10.0), _result(1.0, 10.0), _result(1.0, 10.0), _result(9.0, 30.0)],
        "B": [_result(1.0, 10.0), _result(1.0, 10.0), _result(1.0, 10.0), _result(9.0, 30.0)],
        "A-drift": [_result(1.0, 10.0), _result(1.0, 10.0), _result(9.0, 30.0)],
    }
    adapter = Scripted(samples)
    hooks = WatchHooks(
        state_sampler=lambda _uses_5090: {},
        start_thread=False,
        ram_sampler=lambda: 1.0,
        used_sampler=lambda: 1000,
        total_sampler=lambda: 10**12,
        sleep=lambda _seconds: (_ for _ in ()).throw(AssertionError("rest is not taken here")),
    )
    rows = _run(tmp_path / "repo", spec, {"fake": adapter}, name="blocks", run_id="phi-blocks", watch=hooks)
    assert adapter.calls == ["A", "A", "A", "A", "B", "B", "B", "B", "A-drift", "A-drift", "A-drift"]
    by_id = {row["id"].split("/")[1]: row for row in rows}
    assert by_id["A"]["settings"]["unsteady"] is False
    assert by_id["A"]["settings"]["blocked"]["block_id"] == "one"
    assert by_id["B"]["settings"]["blocked"]["block_id"] == "two"
    assert by_id["A-drift"]["descriptive"] is True
    assert by_id["A-drift"]["settings"]["blocked"]["block_id"] == "drift"
    unloaded = [model for model, _device in adapter.unloaded]
    assert "a" in unloaded and "b" in unloaded


def test_until_steady_hits_max_as_unsteady(tmp_path: Path):
    spec = _spec(
        warmups={"until_steady": True, "max": 2, "tolerance": 0.05},
        repeats=2,
        needs={"grant": False, "npu_session": False, "cpu_window": False},
    )
    spec["arms"] = [_arm(engine="fake", model="phi-fake")]
    samples = [
        _result(1.0, 10.0),
        _result(1.0, 10.0),
        _result(2.0, 20.0),
        _result(9.0, 30.0),
        _result(9.0, 30.0),
    ]
    adapter = Scripted({"phi-35-mini-cpu-l1024": samples})
    rows = _run(tmp_path / "repo", spec, {"fake": adapter}, name="unsteady", run_id="phi-unsteady")
    assert len(adapter.calls) == 5
    assert rows[0]["settings"]["unsteady"] is True
    assert len(rows[0]["settings"]["warmup_trace"]) == 2
    assert rows[0]["latency"]["throughput"] == 30.0


def test_a_failed_load_deletes_the_cold_copy(tmp_path: Path):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "weights.bin").write_text("weights", encoding="utf-8")
    copied: dict = {}

    def copy_tree(argv: list[str]):
        copied["argv"] = list(argv)
        copied["dst"] = Path(argv[2])
        return _Completed(0)

    class Boom(FakePipeline):
        def generate(self, prompt, config=None):
            raise RuntimeError("generate failed after the copy")

    class Genai(FakeGenai):
        LLMPipeline = Boom

    adapter = make_adapter(genai_module=Genai, copy_tree=copy_tree, disk_facts=_facts, snapshot_download=None)
    spec = _spec(
        arms=[_arm(model=str(snap), revision_or_digest="local", cold_copy=True, hf_cache="archive")],
        warmups=0,
        repeats=1,
        needs={"grant": False, "npu_session": False, "cpu_window": False},
    )
    spec.pop("decision_metric", None)
    with pytest.raises(RunnerError, match="generate failed after the copy"):
        _run(tmp_path / "repo", spec, {"openvino": adapter}, name="cold", run_id="phi-cold")
    assert "/J" in copied["argv"]
    assert drive_letter(copied["dst"]) == drive_letter(snap)
    assert not copied["dst"].exists()
    assert (snap / "weights.bin").is_file()
    assert adapter.unloaded if hasattr(adapter, "unloaded") else adapter._cold_copies == []


def test_the_receipt_records_storage_without_a_path(tmp_path: Path):
    calls: list[tuple] = []

    def download(repo, revision):
        calls.append((repo, revision))
        dest = tmp_path / "snap"
        dest.mkdir()
        (dest / "openvino_model.xml").write_text("<xml/>", encoding="utf-8")
        return str(dest)

    copied: dict = {}

    def copy_tree(argv: list[str]):
        copied["dst"] = Path(argv[2])
        copied["argv"] = list(argv)
        return _Completed(0)

    adapter = make_adapter(snapshot_download=download, copy_tree=copy_tree, disk_facts=_facts)
    spec = _spec(
        arms=[
            _arm(
                hf_cache="archive",
                cold_copy=True,
                compile_config={"ENABLE_MMAP": False},
            )
        ],
        warmups=0,
        repeats=1,
        needs={"grant": False, "npu_session": False, "cpu_window": False},
    )
    spec.pop("decision_metric", None)
    rows = _run(tmp_path / "repo", spec, {"openvino": adapter}, name="store", run_id="phi-store")
    settings = rows[0]["settings"]
    assert settings["hf_cache"] == "archive"
    assert settings["cold_copy"] is True
    assert settings["ENABLE_MMAP"] is False
    assert settings["storage"] == {"drive": drive_letter(tmp_path), "bus": "NVMe", "media": "SSD"}
    assert settings["cold_copy_drive"] == drive_letter(tmp_path)
    blob = json.dumps(rows[0])
    assert "Users" not in blob
    assert str(copied["dst"]) not in blob
    assert "/J" in copied["argv"]
    assert not copied["dst"].exists()
    assert calls == [(REPO, REV)]


def test_dry_run_names_the_phi_arm_options(tmp_path: Path):
    repo = tmp_path / "repo"
    spec = _spec(
        arms=[
            _arm(hf_cache="hot", cold_copy=True, compile_config={"ENABLE_MMAP": False}),
            _arm("phi-archive", hf_cache="archive", compile_config={"ENABLE_MMAP": True}),
        ],
    )
    text = dry_run(_bind(repo, spec), repo)
    assert "slots:" not in text
    assert (
        "warmups: until two consecutive calls agree within 0.05 on latency.throughput, "
        "at most 6; warm-up calls stay out of the percentiles"
    ) in text
    assert "phi-35-mini-cpu-l1024: hf_cache hot (C:/AI-Models/hf-cache-hot); a missing copy is a refusal" in text
    assert "phi-35-mini-cpu-l1024: cold copy on the snapshot drive before the first load, deleted after unload" in text
    assert "phi-35-mini-cpu-l1024: ENABLE_MMAP false" in text
    assert "phi-archive: hf_cache archive" in text
    assert "phi-archive: ENABLE_MMAP true" in text
    assert "phi-archive: cold copy" not in text
    assert "prompt built at run time by prompt_eval_count" not in text
    assert "a device run would require: --cpu-window" in text
    assert "devices not started (--dry-run)" in text
    assert "heartbeat not read" in text


def test_the_physical_disk_script_is_parsed_and_not_run():
    script = physical_disk_script("c")
    assert "Get-Partition -DriveLetter C" in script
    assert "Get-Disk" in script
    assert "Get-PhysicalDisk" in script
    assert "BusType" in script
    assert "MediaType" in script
    assert "ConvertTo-Json -Compress" in script
    assert parse_disk_json('{"BusType":"NVMe","MediaType":"SSD"}') == ("NVMe", "SSD")
    assert parse_disk_json('[{"BusType":"USB","MediaType":"Unspecified"}]') == ("USB", "Unspecified")
    assert describe_drive("C", query=lambda _letter: (_ for _ in ()).throw(RuntimeError("no disk"))) == {
        "drive": "C",
        "bus": "unrecorded",
        "media": "unrecorded",
    }


def test_until_steady_refuses_aba_and_a_non_openvino_cache():
    aba = _spec(order="aba")
    aba["pairs"] = [{"id": "p", "alone": "phi-35-mini-cpu-l1024", "paired": "phi-35-mini-cpu-l1024"}]
    with pytest.raises(SpecError, match="until_steady does not run on aba"):
        validate_spec(aba)
    ollama = _spec(arms=[_arm(engine="ollama", device="5090", hf_cache="hot")])
    with pytest.raises(SpecError, match="OpenVINO generate"):
        validate_spec(ollama)
    with pytest.raises(SpecError, match="until_steady true"):
        validate_spec(_spec(warmups={"until_steady": False, "max": 6, "tolerance": 0.05}))


def _measured(arm: dict) -> dict:
    spec = _spec(
        arms=[arm],
        warmups=0,
        repeats=1,
        needs={"grant": False, "npu_session": False, "cpu_window": False},
    )
    spec.pop("decision_metric", None)
    return spec


def _reader_hooks(reader):
    return WatchHooks(
        state_sampler=lambda _uses_5090: {},
        start_thread=False,
        ram_sampler=lambda: 1.0,
        slot_reader=reader,
    )


def _samples(row: dict) -> list[dict]:
    return [slot["sample"] for slot in row["artifacts"]["slots"]]


def test_a_contended_slot_records_top_after_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    walks = {"n": 0}

    def reader(_drive):
        return {
            "cpu_pct": 72.0,
            "ram_pct": 40.0,
            "disk_read_mb_s": None,
            "top": ["in-window", "should-not-appear"],
        }

    def rank(limit=3, **_kwargs):
        walks["n"] += 1
        assert limit == 3
        return ["busy-neighbor", "second", "third"]

    monkeypatch.setattr(watch_mod, "_top_other", rank)
    adapter = Scripted({"phi": [_result(1.0, 10.0), _result(1.0, 10.0)]})
    rows = _run(
        tmp_path / "repo",
        _measured(_arm("phi", engine="fake", model="phi-fake")),
        {"fake": adapter},
        name="busy",
        run_id="phi-busy",
        watch=_reader_hooks(reader),
    )
    samples = _samples(rows[0])
    assert len(samples) == 2
    assert all(sample["cpu_pct"]["max"] > 30 for sample in samples)
    assert all(sample["top_after_call"] == ["busy-neighbor", "second", "third"] for sample in samples)
    assert all("top_processes" not in sample for sample in samples)
    assert all("in-window" not in json.dumps(sample) for sample in samples)
    # One rank per slot. The runner's second close must not walk again.
    assert walks["n"] == 2


def test_a_short_call_still_has_two_readings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def rank(limit=3, **_kwargs):
        raise AssertionError("an idle slot walked processes")

    monkeypatch.setattr(watch_mod, "_top_other", rank)

    def reader(_drive):
        return {"cpu_pct": 5.0, "ram_pct": 11.0, "disk_read_mb_s": None, "top": ["quiet"]}

    adapter = Scripted({"phi": [_result(1.0, 10.0), _result(1.0, 10.0)]})
    rows = _run(
        tmp_path / "repo",
        _measured(_arm("phi", engine="fake", model="phi-fake")),
        {"fake": adapter},
        name="short",
        run_id="phi-short",
        watch=_reader_hooks(reader),
    )
    samples = _samples(rows[0])
    assert [sample["samples"] for sample in samples] == [2, 2]
    assert all(sample["cpu_pct"] == {"min": 5.0, "max": 5.0, "mean": 5.0} for sample in samples)
    assert all("top_after_call" not in sample for sample in samples)
    assert all("top_processes" not in sample for sample in samples)
    assert all("quiet" not in json.dumps(sample) for sample in samples)


def test_the_disk_counter_follows_the_load_drive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "openvino_model.xml").write_text("<xml/>", encoding="utf-8")
    # CI has no Windows drive letter. The adapter must pass drive_letter's
    # answer through, so a stand-in letter fails the test if it is ignored.
    letter = "D"
    monkeypatch.setattr(ov_mod, "drive_letter", lambda _path: letter)
    seen: list = []

    def download(repo, revision):
        assert (repo, revision) == (REPO, REV)
        return str(snap)

    def reader(drive):
        seen.append(drive)
        return {
            "cpu_pct": 4.0,
            "ram_pct": 8.0,
            "disk_read_mb_s": 7.25 if drive == letter else None,
            "top": [],
        }

    def copy_tree(_argv):
        return _Completed(0)

    adapter = make_adapter(snapshot_download=download, copy_tree=copy_tree, disk_facts=_facts)
    rows = _run(
        tmp_path / "repo",
        _measured(_arm(hf_cache="archive", cold_copy=True)),
        {"openvino": adapter},
        name="disk",
        run_id="phi-disk",
        watch=_reader_hooks(reader),
    )
    samples = _samples(rows[0])
    assert len(samples) == 2
    assert all(sample["load_drive"] == letter for sample in samples)
    assert all(sample["disk_read_mb_s"]["mean"] == pytest.approx(7.25) for sample in samples)
    assert set(seen) <= {None, letter}
    assert letter in seen


def test_two_arms_that_differ_only_in_hf_cache_get_two_pipelines(tmp_path: Path):
    hot = tmp_path / "hot"
    hot_snap = _plant(hot, REV, ref="main")
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "openvino_model.xml").write_text("<xml/>", encoding="utf-8")
    built: list[Path] = []

    class Recording(FakePipeline):
        def __init__(self, path, device, config):
            built.append(Path(path))
            super().__init__(path, device, config)

    class Genai(FakeGenai):
        LLMPipeline = Recording

    adapter = make_adapter(
        genai_module=Genai,
        snapshot_download=lambda _repo, _revision: str(archive),
        hot_root=hot,
        disk_facts=_facts,
    )

    def _one(step_id: str, cache: str) -> None:
        result = adapter.run(
            StepRequest(
                step_id=step_id,
                task="generate",
                model=MODEL,
                device="CPU",
                engine="openvino",
                settings={"prompt": "hi", "num_predict": 8, "hf_cache": cache},
            )
        )
        assert result.ok

    _one("phi-hot", "hot")
    _one("phi-archive", "archive")
    assert built == [hot_snap, archive]
    assert len(adapter._pipelines) == 2
    assert {key[3] for key in adapter._pipelines} == {"hot", "archive"}
    assert {Path(key[5]) for key in adapter._pipelines} == {hot_snap, archive}


def test_a_hot_arm_whose_tokenizer_is_missing_is_refused(tmp_path: Path):
    root = tmp_path / "hot"
    snap = _plant(root, REV, ref="main")
    asked: list = []
    downloads: list = []

    def download(*_args, **_kwargs):
        downloads.append(True)
        raise AssertionError("hot must not download")

    def loader(model, revision):
        asked.append((model, revision, os.environ.get("HF_HUB_OFFLINE")))
        raise AdapterError("tokenizer missing")

    previous = os.environ.get("HF_HUB_OFFLINE")
    os.environ.pop("HF_HUB_OFFLINE", None)
    try:
        adapter = make_adapter(
            snapshot_download=download,
            hot_root=root,
            disk_facts=_facts,
            tokenizer_loader=loader,
        )
        with pytest.raises(AdapterError, match="tokenizer missing"):
            adapter.run(_req(MODEL, hf_cache="hot"))
        assert downloads == []
        assert asked == [(str(snap), None, "1")]
        assert os.environ.get("HF_HUB_OFFLINE") is None

        asked.clear()
        missing = make_adapter(
            snapshot_download=download,
            hot_root=tmp_path / "empty",
            disk_facts=_facts,
            tokenizer_loader=loader,
        )
        with pytest.raises(AdapterError, match="refusing instead of downloading"):
            missing.run(_req(MODEL, hf_cache="hot"))
        assert asked == []
        assert downloads == []
    finally:
        if previous is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous


def test_a_directory_tokenizer_refuses_a_hub_fetch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list = []

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model, local_files_only=False, **kwargs):
            calls.append((model, local_files_only, kwargs))
            raise OSError("missing")

    module = types.ModuleType("transformers")
    module.AutoTokenizer = AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", module)
    with pytest.raises(AdapterError, match="refusing instead of fetching"):
        ov_mod._directory_tokenizer_loader(str(tmp_path / "snap"), "ignored-revision")
    assert calls == [(str(tmp_path / "snap"), True, {})]
    assert "snap" not in "tokenizer is missing under the resolved directory; refusing instead of fetching it"


def test_the_slot_reader_records_cpu_ram_and_a_missing_disk(monkeypatch: pytest.MonkeyPatch):
    class Counter:
        def __init__(self, read_bytes):
            self.read_bytes = read_bytes

    class FakePsutil:
        def __init__(self):
            self.bytes = 0
            self.walks = 0

        def cpu_percent(self, interval=None):
            return 15.0

        def virtual_memory(self):
            return type("Mem", (), {"percent": 22.0})()

        def disk_io_counters(self, perdisk=True):
            return {"PhysicalDrive2": Counter(self.bytes)}

        def process_iter(self):
            self.walks += 1
            raise AssertionError("read_slot walked processes")

    fake = FakePsutil()
    ticks = iter([0.0, 1.0])

    def disk_number(letter: str) -> int:
        if letter == "C":
            return 2
        raise OSError(f"no partition for {letter}")

    monkeypatch.setattr(watch_mod, "psutil", fake)
    monkeypatch.setattr(watch_mod, "_DISK", watch_mod._DiskRate(clock=lambda: next(ticks), disk_number=disk_number))
    first = watch_mod.read_slot("C")
    assert first["cpu_pct"] == 15.0 and first["ram_pct"] == 22.0
    assert first["disk_read_mb_s"] is None and first["disk_reason"] is None
    assert "top" not in first
    assert fake.walks == 0
    fake.bytes = 2 * 1024 * 1024
    second = watch_mod.read_slot("C")
    assert second["disk_read_mb_s"] == pytest.approx(2.0)
    assert "top" not in second
    missing = watch_mod.read_slot("Z")
    assert missing["disk_read_mb_s"] is None
    assert missing["disk_reason"] == "no partition for Z"
    assert fake.walks == 0

    class Boom:
        def cpu_percent(self, interval=None):
            raise RuntimeError("cpu")

        def virtual_memory(self):
            raise RuntimeError("ram")

        def disk_io_counters(self, perdisk=True):
            raise RuntimeError("disk")

    monkeypatch.setattr(watch_mod, "psutil", Boom())
    blank = watch_mod.read_slot("C")
    assert blank["cpu_pct"] is None and blank["ram_pct"] is None
    assert blank["disk_read_mb_s"] is None
    assert "top" not in blank
    probe = watch_mod.slot_probe(WatchHooks(start_thread=True, slot_reader=None))
    assert probe is not None and probe._reader is watch_mod.read_slot
    quiet = watch_mod.SlotProbe(lambda _drive: (_ for _ in ()).throw(RuntimeError("counter")), start_thread=False)
    quiet.open()
    summary = quiet.close()
    assert summary["samples"] == 2 and summary["cpu_pct"] is None


def test_a_physical_drive_counter_records_the_disk_rate(monkeypatch: pytest.MonkeyPatch):
    class Counter:
        def __init__(self, read_bytes):
            self.read_bytes = read_bytes

    class Disks:
        def __init__(self):
            self.bytes = 0

        def disk_io_counters(self, perdisk=True):
            return {"PhysicalDrive1": Counter(self.bytes), "C:": Counter(9_999_999)}

    disks = Disks()
    ticks = iter([0.0, 1.0, 2.0])
    mapped: list[str] = []

    def disk_number(letter: str) -> int:
        mapped.append(letter)
        assert letter == "E"
        return 1

    monkeypatch.setattr(watch_mod, "psutil", disks)
    rate = watch_mod._DiskRate(clock=lambda: next(ticks), disk_number=disk_number)
    monkeypatch.setattr(watch_mod, "_DISK", rate)
    baseline, baseline_reason = rate.mb_s("E")
    disks.bytes = 3 * 1024 * 1024
    value, reason = rate.mb_s("E")
    assert baseline is None and baseline_reason is None
    assert value == pytest.approx(3.0) and reason is None
    assert mapped == ["E"]
    assert watch_mod._disk_key({"C:": object(), "PhysicalDrive1": object()}, 1) == "PhysicalDrive1"
    assert watch_mod._disk_key({"C:": object()}, 1) is None

    failed = watch_mod._DiskRate(disk_number=lambda _letter: (_ for _ in ()).throw(OSError("no disk number")))
    nothing, why = failed.mb_s("E")
    assert nothing is None and why == "no disk number"
    probe = watch_mod.SlotProbe(
        lambda drive: {"cpu_pct": 1.0, "ram_pct": 1.0, "disk_read_mb_s": None, "disk_reason": why, "top": []},
        start_thread=False,
    )
    probe.set_drive("E")
    probe.open()
    summary = probe.close()
    assert summary["disk_read_mb_s"] == "unrecorded"
    assert summary["disk_reason"] == "no disk number"


def test_the_disk_number_query_parses_a_physical_disk_number():
    seen: dict[str, str] = {}

    def run(script: str):
        seen["script"] = script
        return type("Completed", (), {"returncode": 0, "stdout": "2\n", "stderr": ""})()

    assert watch_mod.windows_disk_number("e", run=run) == 2
    assert "Get-Partition -DriveLetter E" in seen["script"]
    assert "Get-Disk" in seen["script"]
    assert "Get-PhysicalDisk" not in seen["script"]

    def fail(_script: str):
        return type("Completed", (), {"returncode": 1, "stdout": "", "stderr": "no partition\nmore"})()

    with pytest.raises(OSError, match="no partition"):
        watch_mod.windows_disk_number("E", run=fail)
    with pytest.raises(OSError, match="not a drive letter"):
        watch_mod.windows_disk_number("EE", run=run)


def test_a_slot_probe_samples_during_a_call():
    seen = {"n": 0}

    def reader(_drive):
        seen["n"] += 1
        return {"cpu_pct": 1.0, "ram_pct": 1.0, "disk_read_mb_s": None, "top": []}

    probe = watch_mod.SlotProbe(reader, start_thread=True, interval_s=0.02)
    probe.open()
    time.sleep(0.12)
    summary = probe.close()
    assert summary["samples"] >= 3
    assert seen["n"] == summary["samples"]


def test_in_window_samples_do_not_walk_processes(monkeypatch: pytest.MonkeyPatch):
    """open() and the loop read system counters. They do not walk processes."""

    class Counting:
        def __init__(self):
            self.walks = 0

        def cpu_percent(self, interval=None):
            return 12.0

        def virtual_memory(self):
            return type("Mem", (), {"percent": 20.0})()

        def disk_io_counters(self, perdisk=True):
            return {}

        def Process(self):
            return type("Me", (), {"pid": 10, "children": lambda self, recursive=False: []})()

        def process_iter(self):
            self.walks += 1
            return []

    fake = Counting()
    monkeypatch.setattr(watch_mod, "psutil", fake)
    probe = watch_mod.SlotProbe(watch_mod.read_slot, start_thread=False)
    probe.open()
    assert fake.walks == 0
    probe.sample()
    assert fake.walks == 0
    assert all("top" not in row for row in probe._readings)

    threaded = watch_mod.SlotProbe(watch_mod.read_slot, start_thread=True, interval_s=0.02)
    threaded.open()
    time.sleep(0.08)
    threaded._stop.set()
    assert threaded._thread is not None
    threaded._thread.join(timeout=2)
    assert fake.walks == 0
    assert all("top" not in row for row in threaded._readings)


def test_top_after_call_appears_only_when_cpu_max_is_over_30(monkeypatch: pytest.MonkeyPatch):
    walks = {"n": 0}

    def rank(limit=3, **_kwargs):
        walks["n"] += 1
        return ["busy-neighbor"]

    monkeypatch.setattr(watch_mod, "_top_other", rank)

    def finish(cpu):
        probe = watch_mod.SlotProbe(
            lambda _drive: {"cpu_pct": cpu, "ram_pct": 1.0, "disk_read_mb_s": None, "top": ["in-window"]},
            start_thread=False,
        )
        probe.open()
        summary = probe.close()
        assert probe.close() is summary
        assert all("top" not in row for row in probe._readings)
        return summary

    hot = finish(30.1)
    assert hot["top_after_call"] == ["busy-neighbor"]
    assert "top_processes" not in hot
    assert "top_after_call" not in finish(30)
    assert "top_after_call" not in finish(4.0)
    missing = watch_mod.SlotProbe(lambda _drive: {}, start_thread=False)
    missing.open()
    assert "top_after_call" not in missing.close()
    assert walks["n"] == 1


class _Times:
    def __init__(self, user, system):
        self.user = user
        self.system = system


class _Proc:
    def __init__(self, pid, name, samples, percent=0.0, boom=False):
        self.pid = pid
        self._name = name
        self._samples = list(samples)
        self._i = 0
        self._percent = percent
        self._boom = boom

    def name(self):
        return self._name

    def cpu_percent(self, interval=None):
        return self._percent

    def cpu_times(self):
        if self._boom:
            raise RuntimeError("cpu_times")
        user, system = self._samples[min(self._i, len(self._samples) - 1)]
        self._i += 1
        return _Times(user, system)


def _ranked(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], list[float]]:
    """A world whose instant percents and CPU-time deltas disagree."""

    procs = [
        _Proc(0, "System Idle Process", [(100, 0), (400, 0)], percent=99),
        _Proc(4, "System", [(1, 1), (2, 1)], percent=5),
        _Proc(10, "interlock", [(0, 0), (50, 50)], percent=90),
        _Proc(11, "child", [(0, 0), (40, 40)], percent=80),
        _Proc(12, "percent-decoy", [(10, 0), (10.125, 0)], percent=95),
        _Proc(13, "delta-worker", [(1, 1), (4, 1)], percent=1),
        _Proc(14, "middle", [(2, 0), (2.5, 0.25)], percent=40),
        _Proc(16, "went-backwards", [(8, 0), (4, 0)], percent=70),
        _Proc(17, "unreadable", [], boom=True),
        _Proc(18, "no-user", [(None, 1), (None, 1)]),
    ]
    late = _Proc(15, "late", [(80, 80)], percent=90)
    calls = {"n": 0}

    class Psutil:
        def Process(self):
            class Me:
                pid = 10

                def children(self, recursive=False):
                    return [type("Child", (), {"pid": 11})()]

            return Me()

        def process_iter(self):
            calls["n"] += 1
            rows = list(procs)
            if calls["n"] > 1:
                rows.append(late)
            return rows

    monkeypatch.setattr(watch_mod, "psutil", Psutil())
    slept: list[float] = []
    names = watch_mod._top_other(3, sleep=lambda seconds: slept.append(seconds))
    return names, slept


def test_the_after_call_rank_skips_pid_0_and_this_tree_and_keeps_system(monkeypatch: pytest.MonkeyPatch):
    names, _slept = _ranked(monkeypatch)
    assert "System Idle Process" not in names
    assert "interlock" not in names
    assert "child" not in names
    assert "System" in names


def test_the_after_call_rank_uses_cpu_time_between_snapshots(monkeypatch: pytest.MonkeyPatch):
    names, slept = _ranked(monkeypatch)
    # delta-worker gained 3 s, System 1 s, middle 0.75 s. percent-decoy gained
    # 0.125 s. late is absent from the first snapshot, so it has no delta.
    assert names == ["delta-worker", "System", "middle"]
    assert "percent-decoy" not in names
    assert "late" not in names
    assert "went-backwards" not in names
    assert "unreadable" not in names
    assert "no-user" not in names
    assert slept == [1.0]


def test_the_after_call_rank_returns_nothing_when_the_tree_cannot_be_read(monkeypatch: pytest.MonkeyPatch):
    walks = {"n": 0}

    class NoTree:
        def Process(self):
            raise RuntimeError("proc")

        def process_iter(self):
            walks["n"] += 1
            return []

    monkeypatch.setattr(watch_mod, "psutil", NoTree())
    assert watch_mod._top_other(sleep=lambda _seconds: None) == []
    assert walks["n"] == 0

    class NoChildren:
        def Process(self):
            class Me:
                pid = 10

                def children(self, recursive=False):
                    raise RuntimeError("children")

            return Me()

        def process_iter(self):
            walks["n"] += 1
            return []

    monkeypatch.setattr(watch_mod, "psutil", NoChildren())
    assert watch_mod._top_other(sleep=lambda _seconds: None) == []
    assert walks["n"] == 0

    class NoIter:
        def Process(self):
            class Me:
                pid = 10

                def children(self, recursive=False):
                    return []

            return Me()

        def process_iter(self):
            raise RuntimeError("iter")

    monkeypatch.setattr(watch_mod, "psutil", NoIter())
    assert watch_mod._top_other(sleep=lambda _seconds: None) == []

    monkeypatch.setattr(watch_mod, "psutil", None)
    assert watch_mod._top_other() == []
