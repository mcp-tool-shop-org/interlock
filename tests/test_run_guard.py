"""The run guard: the compile bound, the RAM stop, the run-start marker, the cache drop.

Nothing here touches a device or ends the test process. The compile is a fake
that blocks on an Event; the watcher's clock, samplers and exit function are
injected, so the receipt is read WHILE the compile is still blocked.
"""

from __future__ import annotations

import json
import os
import stat
import threading
import time
from pathlib import Path

import pytest

from switchyard.adapters.base import AdapterError, CompileTracker
from switchyard.cli import main, run_report
from switchyard.guards import GuardError
from switchyard.harness import environment as env_capture
from switchyard.harness import runner as runner_module
from switchyard.harness.marker import RunMarker, marker_path, read_markers
from switchyard.harness.receipts import device_lost_row, environment, for_table
from switchyard.harness.runner import (
    _cache_dirs,
    _drop_caches,
    _rmtree_clearing_readonly,
    plan_rest,
    run_interleaved,
    run_spec,
)
from switchyard.harness.spec import SpecError, validate_spec
from switchyard.harness.watch import (
    COMPILE_CEILING_S,
    EXIT_COMPILE_TIMEOUT,
    EXIT_RAM_STOP,
    RunWatch,
    WatchHooks,
)
from switchyard.receipt import ReceiptError, check_tree, load_jsonl, receipt_paths, validate_row
from tests.test_harness import _bind, _emb_arm, _result, _spec as _harness_spec
from tests.test_run_path import Fake, FakeTime, _arm as _plain_arm, _bind as _bind_plain, _spec as _plain_spec

GIB = 2**30


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _spec(tmp_path: Path, **extra):
    repo = tmp_path / "repo"
    spec = _harness_spec()
    spec["warmups"] = 0
    spec["repeats"] = 1
    spec["needs"] = {"grant": False, "npu_session": False, "cpu_window": False}
    spec["arms"] = [_emb_arm("cpu", "CPU", "1x128"), _emb_arm("npu", "NPU", "1x512")]
    spec.update(extra)
    return repo, _bind(repo, spec)


class Exits:
    """The injected exit function: records the code and what the receipt held at that moment."""

    def __init__(self, receipt: Path, gate: threading.Event) -> None:
        self.receipt = receipt
        self.gate = gate
        self.codes: list[int] = []
        self.seen: list[dict] = []
        self.blocked_when_called = None

    def __call__(self, code: int) -> None:
        self.codes.append(code)
        self.blocked_when_called = not self.gate.is_set()
        if self.receipt.is_file():
            self.seen = load_jsonl(self.receipt)
        self.gate.set()  # lets the fake "native call" return so the test can finish


class BlockingAdapter:
    """CPU arm answers; the NPU arm's cold call sits inside a compile that never returns."""

    def __init__(self, clock: Clock, gate: threading.Event, *, compile_spent_s: float, ram: dict | None = None,
                 in_compile: bool = True) -> None:
        self.clock = clock
        self.gate = gate
        self.spent = compile_spent_s
        self.ram = ram
        self.in_compile = in_compile
        self.tracker = None
        self.calls: list[str] = []

    def attach_compile_tracker(self, tracker) -> None:
        self.tracker = tracker

    def run(self, request):
        self.calls.append(request.step_id)
        if request.step_id != "npu":
            return _result(5)
        if self.ram is not None:
            self.ram["v"] = 91.0
        if self.in_compile:
            with self.tracker.compiling("compile_model NPU"):
                self.clock.now += self.spent
                assert self.gate.wait(timeout=30), "the watcher never ended the run"
        else:
            assert self.gate.wait(timeout=30), "the watcher never ended the run"
        raise AdapterError("released after the watcher fired")

    def unload(self, model, device=None):
        return None


def _hooks(clock, gate_exit, *, ram=None, rss=5 * GIB, interval=0.01):
    ram = ram if ram is not None else {"v": 40.0}
    return WatchHooks(
        clock=clock,
        ram_sampler=lambda: ram["v"],
        rss_sampler=lambda: rss,
        exit_fn=gate_exit,
        interval_s=interval,
    ), ram


def _invoke(repo, spec_path, adapter, hooks, **kwargs):
    return run_interleaved(
        {"fake": adapter},
        grant=None,
        npu_session="session-t",
        cpu_window=None,
        root=repo,
        repo=repo,
        spec_path=spec_path,
        run_id="run-1",
        date="2026-10-10",
        watch=hooks,
        **kwargs,
    )


# --- A: the compile bound ----------------------------------------------------


def test_a_compile_past_the_ceiling_writes_the_receipt_while_still_blocked_and_ends_the_process(tmp_path):
    repo, spec_path = _spec(tmp_path)
    clock, gate = Clock(), threading.Event()
    exits = Exits(repo / "receipts" / "run-1.jsonl", gate)
    hooks, _ram = _hooks(clock, exits)
    adapter = BlockingAdapter(clock, gate, compile_spent_s=901)
    with pytest.raises(AdapterError, match="released"):
        _invoke(repo, spec_path, adapter, hooks)
    assert exits.codes == [EXIT_COMPILE_TIMEOUT]
    assert exits.blocked_when_called is True  # the receipt was written before the compile returned
    by_id = {row["id"]: row for row in exits.seen}
    timed_out = by_id["T/npu/run-1"]
    assert timed_out["outcome"] == "compile_timeout"
    assert timed_out["provisional"] is True
    assert all(value is None for value in timed_out["latency"].values())
    evidence = timed_out["settings"]["compile_timeout"]
    assert evidence["elapsed_s"] == 901
    assert evidence["ceiling_s"] == COMPILE_CEILING_S == 900
    assert evidence["peak_working_set_gb"] == 5.0
    assert evidence["npu_driver"] and evidence["openvino"]
    assert evidence["run_id"] == "run-1"
    assert evidence["shape"] == {"batch": 1, "static_shape": "1x512"}
    # The arm that had a cold sample is kept as an aborted row, with why.
    aborted = by_id["T/cpu/run-1"]
    assert aborted["aborted"] is True
    assert "compile ceiling" in aborted["settings"]["abortion"]["error"]
    assert aborted["settings"]["abortion"]["stop_step"] == "npu cold"
    assert timed_out["settings"]["ram_guard"]["stop_pct"] == 88
    for row in exits.seen:
        validate_row(row)


def test_a_compile_under_the_ceiling_does_nothing(tmp_path):
    repo, spec_path = _spec(tmp_path)
    clock, gate = Clock(), threading.Event()
    exits = Exits(repo / "receipts" / "run-1.jsonl", gate)
    hooks, _ram = _hooks(clock, exits)

    class Quick(BlockingAdapter):
        def run(self, request):
            with self.tracker.compiling("compile_model NPU"):
                self.clock.now += 899
                time.sleep(0.05)  # several watcher ticks inside the compile
            return _result(5)

    rows = _invoke(repo, spec_path, Quick(clock, gate, compile_spent_s=0), hooks)
    assert exits.codes == []
    assert all("outcome" not in row for row in rows)


def test_the_spec_can_move_the_ceiling(tmp_path):
    repo, spec_path = _spec(tmp_path, compile_ceiling_s=120)
    clock, gate = Clock(), threading.Event()
    exits = Exits(repo / "receipts" / "run-1.jsonl", gate)
    hooks, _ram = _hooks(clock, exits)
    with pytest.raises(AdapterError):
        _invoke(repo, spec_path, BlockingAdapter(clock, gate, compile_spent_s=130), hooks)
    row = next(row for row in exits.seen if row.get("outcome") == "compile_timeout")
    assert row["settings"]["compile_timeout"]["ceiling_s"] == 120


def test_compile_ceiling_s_is_validated():
    spec = _harness_spec()
    assert validate_spec({**spec, "compile_ceiling_s": 60})["compile_ceiling_s"] == 60
    for bad in (59, True, "900"):
        with pytest.raises(SpecError, match="compile_ceiling_s"):
            validate_spec({**spec, "compile_ceiling_s": bad})


def test_the_compile_tracker_brackets_a_compile_and_clears_on_error():
    clock = Clock()
    tracker = CompileTracker(clock)
    assert tracker.active() is None
    clock.now = 3.0
    with pytest.raises(RuntimeError):
        with tracker.compiling("compile_model NPU"):
            assert tracker.active() == ("compile_model NPU", 3.0)
            raise RuntimeError("compile failed")
    assert tracker.active() is None
    assert CompileTracker().clock is time.monotonic


def test_the_openvino_adapter_signals_each_compile_site(tmp_path, monkeypatch):
    from tests.test_openvino_adapter import local_model, make_adapter, req
    import switchyard.adapters.openvino as ovm

    adapter = make_adapter()
    tracker = CompileTracker()
    seen = []

    class Spy(CompileTracker):
        def compiling(self, label):
            seen.append(label)
            return tracker.compiling(label)

    adapter.attach_compile_tracker(Spy())
    model = local_model(tmp_path)
    monkeypatch.setitem(ovm._POOLING, model, "cls")
    assert adapter.run(req(model=model, device="NPU", static_shape="1x128")).ok
    assert seen == ["compile_model NPU"]
    seen.clear()
    adapter.run(req(task="generate", model=str(tmp_path), device="CPU", num_predict=4))
    assert seen == ["LLMPipeline CPU"]
    # No tracker attached: the compile still runs.
    plain = make_adapter()
    assert plain.compile_tracker is None
    assert plain.run(req(model=model, device="CPU")).ok


def test_the_watch_fires_once_and_survives_a_failing_recorder(capsys):
    clock = Clock()
    tracker = CompileTracker(clock)
    codes = []

    def broken(_info):
        raise OSError("disk full")

    watch = RunWatch(
        tracker,
        WatchHooks(clock=clock, ram_sampler=lambda: 10.0, rss_sampler=lambda: 0, exit_fn=codes.append, start_thread=False),
        ceiling_s=900,
        on_compile_timeout=broken,
        on_ram_stop=lambda reason: None,
    )
    assert watch.tick() is False
    with tracker.compiling("x"):
        clock.now = 901
        assert watch.tick() is True
        assert watch.tick() is True  # already fired: nothing again
    assert codes == [EXIT_COMPILE_TIMEOUT]
    assert "could not write the receipt" in capsys.readouterr().err


def test_a_failing_sampler_is_not_a_reading():
    def boom():
        raise OSError("counter gone")

    watch = RunWatch(
        CompileTracker(),
        WatchHooks(ram_sampler=boom, rss_sampler=boom, exit_fn=lambda code: None, start_thread=False),
        ceiling_s=900,
        on_compile_timeout=lambda info: None,
        on_ram_stop=lambda reason: None,
    )
    assert watch.sample_ram() is None
    assert watch.tick() is False
    assert watch.ram_guard() == {"stop_pct": 88, "peak_pct": None, "samples": 0}
    watch.start()  # start_thread False: nothing runs
    watch.stop()


# --- B: the RAM stop -----------------------------------------------------------


def test_ram_at_the_stop_flushes_and_ends_the_run_even_outside_a_compile(tmp_path):
    repo, spec_path = _spec(tmp_path)
    clock, gate = Clock(), threading.Event()
    exits = Exits(repo / "receipts" / "run-1.jsonl", gate)
    hooks, ram = _hooks(clock, exits)
    adapter = BlockingAdapter(clock, gate, compile_spent_s=0, ram=ram, in_compile=False)
    with pytest.raises(AdapterError):
        _invoke(repo, spec_path, adapter, hooks)
    assert exits.codes == [EXIT_RAM_STOP]
    assert exits.blocked_when_called is True
    by_id = {row["id"]: row for row in exits.seen}
    reason = "system RAM reached 91.0% (stop at 88%)"
    for arm in ("cpu", "npu"):  # the arm in progress is recorded too, with the reason
        row = by_id[f"T/{arm}/run-1"]
        assert row["aborted"] is True
        assert row["settings"]["abortion"]["error"] == reason
    assert by_id["T/npu/run-1"]["settings"]["ram_guard"]["peak_pct"] == 91.0
    assert not any(row.get("outcome") for row in exits.seen)


def test_ram_below_the_stop_does_nothing_and_every_row_records_the_guard(tmp_path):
    repo, spec_path = _spec(tmp_path)
    clock, gate = Clock(), threading.Event()
    exits = Exits(repo / "receipts" / "run-1.jsonl", gate)
    hooks, ram = _hooks(clock, exits)
    ram["v"] = 87.9

    class Slow(BlockingAdapter):
        def run(self, request):
            time.sleep(0.05)
            return _result(5)

    rows = _invoke(repo, spec_path, Slow(clock, gate, compile_spent_s=0), hooks)
    assert exits.codes == []
    for row in rows:
        guard = row["settings"]["ram_guard"]
        assert guard["stop_pct"] == 88
        assert guard["peak_pct"] == 87.9
        assert guard["samples"] >= 1


def test_without_psutil_the_guard_records_unrecorded_and_the_run_goes_on(tmp_path):
    repo, spec_path = _spec(tmp_path)
    hooks = WatchHooks(ram_sampler=None, rss_sampler=None, exit_fn=lambda code: pytest.fail("must not exit"))

    class Plain(BlockingAdapter):
        def run(self, request):
            return _result(5)

    rows = _invoke(repo, spec_path, Plain(Clock(), threading.Event(), compile_spent_s=0), hooks)
    assert [row["settings"]["ram_guard"] for row in rows] == ["unrecorded", "unrecorded"]


def test_the_ram_stop_reason_names_the_percent_and_the_stop():
    codes, reasons = [], []
    ram = {"v": 88.0}
    watch = RunWatch(
        CompileTracker(),
        WatchHooks(ram_sampler=lambda: ram["v"], rss_sampler=None, exit_fn=codes.append, start_thread=False),
        ceiling_s=900,
        on_compile_timeout=lambda info: None,
        on_ram_stop=reasons.append,
    )
    ram["v"] = 87.99
    assert watch.tick() is False
    ram["v"] = 88.0
    assert watch.tick() is True
    assert reasons == ["system RAM reached 88.0% (stop at 88%)"]
    assert codes == [EXIT_RAM_STOP]


# --- C: the run-start marker ---------------------------------------------------


NOW = 1_800_000_000.0


def _iso(epoch):
    import datetime

    return datetime.datetime.fromtimestamp(epoch, tz=datetime.timezone.utc).isoformat(timespec="seconds")


def _marker(repo: Path, run_id, *, ended_min_ago, duration_min=None, closed=False, devices=("CPU",)):
    data = {
        "run_id": run_id,
        "spec": "experiments/t/spec.yaml",
        "spec_blob": "abc",
        "devices": list(devices),
        "start": _iso(NOW - ((duration_min or 1) + ended_min_ago) * 60),
        "end": _iso(NOW - ended_min_ago * 60) if closed else None,
        "closed": closed,
    }
    folder = repo / "receipts"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{run_id}.run.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    written = NOW - ended_min_ago * 60
    os.utime(path, (written, written))
    return path


def _cpu_spec():
    return _plain_spec([_plain_arm("A", "CPU")])


def test_a_killed_run_leaves_a_marker_that_owes_rest_from_its_file_time(tmp_path):
    _marker(tmp_path, "killed", ended_min_ago=10)
    plan = plan_rest(_cpu_spec(), tmp_path, NOW)
    assert plan["ran"] is True
    assert plan["prior_run_id"] == "killed"
    assert plan["rest_s_applied"] == pytest.approx(900 - 10 * 60)
    assert plan["prior_run_min"] is None
    assert "unrecorded" in plan["reason"]


def test_a_closed_marker_of_a_twenty_minute_run_owes_nothing(tmp_path):
    _marker(tmp_path, "short", ended_min_ago=1, duration_min=20, closed=True)
    plan = plan_rest(_cpu_spec(), tmp_path, NOW)
    assert plan["ran"] is False
    assert plan["rest_s_applied"] == 0
    assert plan["prior_run_min"] == pytest.approx(20)


def test_a_closed_marker_of_a_long_run_owes_the_remainder(tmp_path):
    _marker(tmp_path, "long", ended_min_ago=10, duration_min=40, closed=True)
    plan = plan_rest(_cpu_spec(), tmp_path, NOW)
    assert plan["rest_s_applied"] == pytest.approx(300)
    assert plan["prior_run_min"] == pytest.approx(40)


def test_a_marker_on_other_devices_or_the_asking_run_itself_is_ignored(tmp_path):
    _marker(tmp_path, "npu-run", ended_min_ago=1, devices=("NPU",))
    _marker(tmp_path, "me", ended_min_ago=0)
    plan = plan_rest(_cpu_spec(), tmp_path, NOW, frozenset({"me"}))
    assert plan["ran"] is False and plan["prior_run_id"] is None


def test_an_unreadable_marker_counts_as_a_long_run(tmp_path):
    folder = tmp_path / "receipts"
    folder.mkdir()
    bad = folder / "broken.run.json"
    bad.write_text("{not json", encoding="utf-8")
    os.utime(bad, (NOW - 60, NOW - 60))
    plan = plan_rest(_cpu_spec(), tmp_path, NOW)
    assert plan["prior_run_id"] == "broken"
    assert plan["rest_s_applied"] == pytest.approx(900 - 60)
    assert read_markers(tmp_path) == [(bad, None)]


def test_a_run_with_a_receipt_and_a_marker_owes_the_larger_rest(tmp_path):
    from tests.test_compile_rejected_rest import _prior

    _prior(tmp_path, "both", duration_min=20, ended_min_ago=1)  # a short run: owes nothing
    _marker(tmp_path, "both", ended_min_ago=5)  # never closed: treated as long
    plan = plan_rest(_cpu_spec(), tmp_path, NOW)
    assert plan["prior_run_id"] == "both"
    assert plan["rest_s_applied"] == pytest.approx(900 - 5 * 60)


def test_a_marker_is_not_a_receipt(tmp_path):
    _marker(tmp_path, "killed", ended_min_ago=1)
    assert receipt_paths(tmp_path) == []
    assert check_tree(tmp_path) == []
    assert main(["check", "--root", str(tmp_path)]) == 0


def test_the_marker_is_written_before_any_adapter_call_and_updated_at_every_flush(tmp_path):
    repo, spec_path = _spec(tmp_path)
    path = marker_path(repo, "run-1")
    snapshots = []
    ticks = iter(range(1_000, 1_000_000))

    class Watching:
        def __init__(self):
            self.calls = 0

        def run(self, request):
            self.calls += 1
            snapshots.append(json.loads(path.read_text(encoding="utf-8")))
            return _result(5)

        def unload(self, model, device=None):
            return None

    rows = _invoke(repo, spec_path, Watching(), WatchHooks(ram_sampler=None, start_thread=False), wall=lambda: float(next(ticks)))
    first = snapshots[0]
    assert first["run_id"] == "run-1"
    assert first["spec"] == "experiments/t/spec.yaml"
    assert len(first["spec_blob"]) == 40
    assert first["devices"] == ["CPU", "NPU"]
    assert first["start"].endswith("+00:00")
    assert first["end"] is None and first["closed"] is False
    # cpu finishes on its repeat slot; its flush moved the end before the next call
    assert snapshots[-1]["end"] is not None and snapshots[-1]["closed"] is False
    final = json.loads(path.read_text(encoding="utf-8"))
    assert final["closed"] is True and final["end"] >= final["start"]
    assert len(rows) == 2


def test_the_marker_is_closed_when_the_run_fails_in_order(tmp_path):
    repo, spec_path = _spec(tmp_path)

    class Failing:
        def run(self, request):
            raise AdapterError("boom")

        def unload(self, model, device=None):
            return None

    with pytest.raises(AdapterError):
        _invoke(repo, spec_path, Failing(), WatchHooks(ram_sampler=None, start_thread=False))
    assert json.loads(marker_path(repo, "run-1").read_text(encoding="utf-8"))["closed"] is True


def test_run_spec_writes_the_marker_before_the_card_probe_and_not_its_own_rest(tmp_path):
    repo = tmp_path / "repo"
    spec_path = _bind_plain(repo, _plain_spec([_plain_arm("A", "5090")], needs={"grant": True, "npu_session": False, "cpu_window": False}))
    seen = []

    class Probing(Fake):
        def loaded_models(self):
            seen.append(marker_path(repo, "run-1").is_file())
            return []

    heartbeat = repo / "hb"
    heartbeat.write_text("pulse", encoding="utf-8")
    rested = FakeTime()
    run_spec(
        spec_path,
        root=repo,
        grant="g",
        run_id="run-1",
        date="2026-10-10",
        adapters={"fake": Probing({"A": [_result_plain(), _result_plain(), _result_plain()]})},
        heartbeat=heartbeat,
        sleep=rested.sleep,
        clock=rested.monotonic,
        watch=WatchHooks(ram_sampler=None, start_thread=False),
    )
    assert seen == [True, True]


def _result_plain():
    from switchyard.adapters.base import StepResult

    return StepResult(ok=True, latency_ms=10)


def test_a_refusal_before_the_run_starts_leaves_no_marker(tmp_path):
    repo = tmp_path / "repo"
    spec_path = _bind_plain(repo, _plain_spec([_plain_arm("A", "5090")], needs={"grant": True, "npu_session": False, "cpu_window": False}))
    fake = Fake(loaded=["something-on-the-card"])
    heartbeat = repo / "hb"
    heartbeat.write_text("pulse", encoding="utf-8")
    with pytest.raises(GuardError):
        run_spec(spec_path, root=repo, grant="g", run_id="run-1", adapters={"fake": fake}, heartbeat=heartbeat)
    assert not marker_path(repo, "run-1").exists()


def test_marker_writes_after_the_start_never_fail_the_run(tmp_path):
    marker = RunMarker(tmp_path, "r", spec_rel="s", spec_blob="b", devices={"CPU"}, start=_iso(NOW))
    marker.write()
    marker.path.unlink()
    marker.path.parent.rmdir()
    (tmp_path / "receipts").write_text("a file where the folder was", encoding="utf-8")
    marker.touch(_iso(NOW))  # OSError swallowed
    marker.close(_iso(NOW))
    marker.discard()


# --- D: the cache drop ----------------------------------------------------------


def _deny(_target):
    raise PermissionError(13, "Access is denied")


def _fresh_spec():
    return {"compile_cache": "fresh"}


def test_a_read_only_cache_blob_is_deleted(tmp_path):
    cache = tmp_path / "cache"
    (cache / "sub").mkdir(parents=True)
    blob = cache / "sub" / "model.blob"
    blob.write_bytes(b"x" * 10)
    os.chmod(blob, stat.S_IREAD)
    assert _rmtree_clearing_readonly(cache) == []
    assert not cache.exists()
    gone, failures = _drop_caches({"A": cache})  # already gone
    assert gone == {"A"} and failures == {}


def test_a_delete_that_fails_is_reported_not_swallowed(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "x.blob").write_bytes(b"x")

    def refusing(path, onexc=None, **_kwargs):
        for index in range(12):
            onexc(os.remove, str(Path(path) / f"x{index}.blob"), PermissionError(13, "Access is denied"))

    monkeypatch.setattr(runner_module.shutil, "rmtree", refusing)
    gone, failures = _drop_caches({"A": cache})
    assert gone == set()
    assert len(failures["A"]) == 11 and failures["A"][-1] == "and 2 more"
    assert failures["A"][0].startswith("x0.blob: FileNotFoundError")
    assert str(tmp_path) not in " ".join(failures["A"])

    monkeypatch.setattr(runner_module.shutil, "rmtree", lambda path, onexc=None, **kw: None)
    gone, failures = _drop_caches({"A": cache})
    assert failures == {"A": ["the directory is still there after the delete"]}
    # An error on a path outside the tree still reports a name, not a crash.
    monkeypatch.setattr(
        runner_module.shutil,
        "rmtree",
        lambda path, onexc=None, **kw: onexc(os.remove, str(tmp_path / "elsewhere.blob"), PermissionError(13, "x")),
    )
    assert _rmtree_clearing_readonly(cache)[0].startswith("elsewhere.blob")


def test_a_failed_delete_lands_in_the_row_settings(tmp_path, monkeypatch):
    repo, spec_path = _spec(tmp_path, compile_cache="fresh")
    real = runner_module.shutil.rmtree

    def refusing(path, onexc=None, **kwargs):
        if Path(path).name == "cpu":
            return onexc(os.remove, str(Path(path) / "left.blob"), PermissionError(13, "Access is denied"))
        return real(path, onexc=onexc, **kwargs)

    class Plain:
        def run(self, request):
            return _result(5)

        def unload(self, model, device=None):
            return None

    monkeypatch.setattr(runner_module.shutil, "rmtree", refusing)
    rows = _invoke(repo, spec_path, Plain(), WatchHooks(ram_sampler=None, start_thread=False))
    cpu = next(row for row in rows if row["id"].endswith("/cpu/run-1"))["settings"]
    npu = next(row for row in rows if row["id"].endswith("/npu/run-1"))["settings"]
    assert cpu["compile_cache_deleted"] is False
    assert cpu["compile_cache_delete_failures"][0].startswith("left.blob")
    assert npu["compile_cache_deleted"] is True and "compile_cache_delete_failures" not in npu
    assert cpu["compile_cache_cleared_at_start"] is False


def test_a_fresh_arm_starts_on_an_empty_directory(tmp_path):
    repo, spec_path = _spec(tmp_path, compile_cache="fresh")
    stale = repo / "compile-cache" / "run-1" / "npu"
    stale.mkdir(parents=True)
    blob = stale / "old.blob"
    blob.write_bytes(b"warm")
    os.chmod(blob, stat.S_IREAD)
    seen = {}

    class Looking:
        def run(self, request):
            seen[request.step_id] = sorted(p.name for p in Path(request.settings["compile_cache_dir"]).iterdir())
            return _result(5)

        def unload(self, model, device=None):
            return None

    rows = _invoke(repo, spec_path, Looking(), WatchHooks(ram_sampler=None, start_thread=False))
    assert seen == {"cpu": [], "npu": []}
    npu = next(row for row in rows if row["id"].endswith("/npu/run-1"))["settings"]
    cpu = next(row for row in rows if row["id"].endswith("/cpu/run-1"))["settings"]
    assert npu["compile_cache_cleared_at_start"] is True
    assert cpu["compile_cache_cleared_at_start"] is False


def test_a_fresh_cache_that_cannot_be_cleared_refuses_before_any_device_contact(tmp_path, monkeypatch):
    repo, spec_path = _spec(tmp_path, compile_cache="fresh")
    stale = repo / "compile-cache" / "run-1" / "cpu"
    stale.mkdir(parents=True)
    (stale / "old.blob").write_bytes(b"warm")
    monkeypatch.setattr(runner_module.shutil, "rmtree", lambda path, onexc=None, **kw: onexc(_deny, str(Path(path) / "old.blob"), PermissionError(13, "Access is denied")))

    class Untouched:
        def run(self, request):
            pytest.fail("a device was contacted")

        def unload(self, model, device=None):
            pytest.fail("a device was contacted")

    with pytest.raises(GuardError, match="not empty and could not be cleared"):
        _invoke(repo, spec_path, Untouched(), WatchHooks(ram_sampler=None, start_thread=False))
    assert not marker_path(repo, "run-1").exists()
    # A reuse spec never looks at the directory.
    assert _cache_dirs({"compile_cache": "reuse"}, repo, "run-1", {"cpu": {}}) == {}


# --- the capability row and the report ------------------------------------------


def test_a_compile_timeout_row_is_an_untimed_capability_row(tmp_path):
    arm = _emb_arm("npu", "NPU", "1x512")
    arm.update(revision_or_digest="abc", licence="x")
    row = device_lost_row(
        spec_id="T", arm=arm, repeats=1, date="2026-10-10", spec_rel="s.yaml", run_id="r", grant=None,
        env=environment(), outcome="compile_timeout", partial=False,
    )
    assert row["outcome"] == "compile_timeout" and row["provisional"] is True
    validate_row(row)
    row["latency"]["cold_ms"] = 5
    with pytest.raises(ReceiptError, match="no timing"):
        validate_row(row)
    row["latency"]["cold_ms"] = None
    assert for_table(row)["notes"].startswith("this device can't do this setting")
    assert run_report([row], finished=False) == f"{_arm_of(row)} NPU compile_timeout\n"


def _arm_of(row):
    return row["id"].split("/")[-2]


# --- F: the shared-memory override --------------------------------------------------


def test_the_shared_memory_override_reads_a_percent_and_falls_back_to_unrecorded():
    assert env_capture.igpu_shared_memory_override(lambda: 57) == "57%"
    assert env_capture.igpu_shared_memory_override(lambda: 57.5) == "57.5%"
    assert env_capture.igpu_shared_memory_override(lambda: None) == "unrecorded"
    assert env_capture.igpu_shared_memory_override(lambda: 0) == "unrecorded"
    assert env_capture.igpu_shared_memory_override(lambda: 140) == "unrecorded"
    assert env_capture.igpu_shared_memory_override(lambda: True) == "unrecorded"
    assert env_capture.igpu_shared_memory_override(lambda: "57") == "unrecorded"

    def broken():
        raise PermissionError("registry closed")

    assert env_capture.igpu_shared_memory_override(broken) == "unrecorded"


def test_the_default_reader_never_fails_and_is_unrecorded_off_windows(monkeypatch):
    value = env_capture.igpu_shared_memory_override()
    assert value == "unrecorded" or value.endswith("%")
    monkeypatch.setattr(env_capture.sys, "platform", "linux")
    assert env_capture.igpu_shared_memory_override() == "unrecorded"


def test_the_environment_carries_the_override(monkeypatch):
    monkeypatch.setattr(env_capture, "igpu_shared_memory_override", lambda reader=None: "57%")
    assert environment()["igpu_shared_memory_override"] == "57%"
    assert environment(igpu_shared_memory_override="kept")["igpu_shared_memory_override"] == "kept"
