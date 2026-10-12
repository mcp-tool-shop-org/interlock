"""Blocked order: one block at a time, memory proven back between blocks.

Everything here is a fake. The fake adapter mirrors the OpenVINO adapter's pipeline
cache (one entry per model and device) and its build-guard hook; RAM is a dict the
test moves; the clock and the sleep are injected, so the 60 s release wait costs
nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from interlock.adapters.base import AdapterError, BuildRefused
from interlock.cli import announce_run
from interlock.harness import runner as runner_module
from interlock.harness.marker import marker_path
from interlock.harness.receipts import import_receipt
from interlock.harness.runner import RunnerError, dry_run, run_interleaved
from interlock.harness.spec import SpecError, validate_spec
from interlock.harness.timing import schedule
from interlock.harness.watch import (
    BUILD_GUARD_PCT,
    BUILD_STEP_8B_BYTES,
    BUILD_STEP_DEFAULT_BYTES,
    BuildGuard,
    WatchHooks,
)
from interlock.receipt import is_speed_row, load_jsonl
from tests.test_harness import _arm, _bind, _result, _spec as _harness_spec

GIB = 2**30
QWEN = "OpenVINO/Qwen3-8B-int4-cw-ov"
PHI = "OpenVINO/Phi-3.5-mini-instruct-int4-cw-ov"


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _gen(arm_id, device, model):
    arm = _arm(arm_id, device)
    arm["model"] = model
    return arm


def _blocked_spec(tmp_path: Path, **extra):
    repo = tmp_path / "repo"
    spec = _harness_spec()
    spec["warmups"] = 1
    spec["repeats"] = 2
    spec["order"] = "blocked"
    spec["needs"] = {"grant": False, "npu_session": False, "cpu_window": False}
    spec["arms"] = [
        _gen("q-cpu-a", "CPU", QWEN),
        _gen("q-cpu-b", "CPU", QWEN),
        _gen("p-igpu-a", "iGPU", PHI),
        _gen("p-igpu-b", "iGPU", PHI),
    ]
    spec["blocks"] = [
        {"id": "PHI", "arms": ["p-igpu-a", "p-igpu-b"]},  # declared before the Qwen block on purpose
        {"id": "QWEN", "arms": ["q-cpu-a", "q-cpu-b"]},
    ]
    spec.update(extra)
    return repo, _bind(repo, spec), spec


class Ram:
    """Used and total RAM, in bytes, for the injected samplers."""

    def __init__(self, used_gib=10.0, total_gib=64.0) -> None:
        self.used = used_gib * GIB
        self.total = total_gib * GIB


def _hooks(clock: Clock, ram: Ram, used=None) -> WatchHooks:
    return WatchHooks(
        clock=clock,
        ram_sampler=None,
        rss_sampler=None,
        used_sampler=used or (lambda: ram.used),
        total_sampler=lambda: ram.total,
        sleep=clock.sleep,
        start_thread=False,
    )


class ResidentAdapter:
    """The OpenVINO adapter's pipeline cache, its build-guard hook and its unload."""

    def __init__(self, ram: Ram, *, build_gib=10.0, leak_gib=0.0, throughput=lambda step_id, n: 10.0, fail_on=None):
        self.ram = ram
        self.build_gib = build_gib
        self.leak_gib = leak_gib
        self.throughput = throughput
        self.fail_on = fail_on
        self._pipelines: dict = {}
        self.build_guard = None
        self.calls: list[str] = []
        self.peak_resident = 0
        self.builds: list[tuple] = []
        self.unloads: list[tuple] = []
        self.counts: dict[str, int] = {}

    def attach_build_guard(self, guard) -> None:
        self.build_guard = guard

    def run(self, request):
        key = (request.model, request.device)
        if key not in self._pipelines:
            if self._pipelines:
                raise AdapterError("a second generate pipeline would be resident")
            with self.build_guard.building(request.model):
                self.ram.used += self.build_gib * GIB
            self._pipelines[key] = {"warmed": False}
            self.builds.append(key)
            self.peak_resident = max(self.peak_resident, len(self._pipelines))
        self.calls.append(request.step_id)
        n = self.counts[request.step_id] = self.counts.get(request.step_id, 0) + 1
        if self.fail_on is not None and self.fail_on(request.step_id, n):
            raise AdapterError("drift boom")
        return _result(5, self.throughput(request.step_id, n))

    def unload(self, model, device=None):
        self.unloads.append((model, device))
        gone = [key for key in self._pipelines if key[0] == model]
        for key in gone:
            del self._pipelines[key]
            self.ram.used -= (self.build_gib - self.leak_gib) * GIB


def _run(repo, spec_path, adapter, hooks, **kwargs):
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


def _ids(slots):
    return [(slot.phase, slot.step_id, slot.round_index) for slot in slots]


# --- order -------------------------------------------------------------------


def test_blocks_run_in_the_declared_order_and_each_block_uses_the_old_scheduler(tmp_path):
    repo, spec_path, spec = _blocked_spec(tmp_path)
    clock, ram = Clock(), Ram()
    adapter = ResidentAdapter(ram)
    rows = _run(repo, spec_path, adapter, _hooks(clock, ram))
    expected = schedule(["p-igpu-a", "p-igpu-b"], warmups=1, repeats=2) + schedule(
        ["q-cpu-a", "q-cpu-b"], warmups=1, repeats=2
    )
    assert adapter.calls == [slot.step_id for slot in expected]
    assert runner_module._slots(spec) == expected
    assert runner_module._slots(spec, ["q-cpu-a", "q-cpu-b"]) == schedule(["q-cpu-a", "q-cpu-b"], warmups=1, repeats=2)
    assert {row["settings"]["blocked"]["block_id"] for row in rows} == {"PHI", "QWEN"}
    assert all(row["settings"]["blocked"]["build_guard_pct"] == 85 for row in rows)
    assert BUILD_GUARD_PCT == 85


def test_at_most_one_generate_pipeline_is_resident_and_each_block_unloads(tmp_path):
    repo, spec_path, _spec = _blocked_spec(tmp_path)
    clock, ram = Clock(), Ram()
    adapter = ResidentAdapter(ram)
    _run(repo, spec_path, adapter, _hooks(clock, ram))
    assert adapter.peak_resident == 1
    assert adapter.builds == [(PHI, "iGPU"), (QWEN, "CPU")]
    # the PHI block was unloaded before the QWEN block built (a second pipeline would have raised)
    assert adapter.unloads[0] == (PHI, "iGPU")
    assert not adapter._pipelines  # and the end of the run unloads the last block too
    assert ram.used == 10 * GIB


def test_a_second_pipeline_inside_a_block_is_refused_by_the_openvino_adapter(tmp_path):
    from tests.test_openvino_adapter import make_adapter, req

    adapter = make_adapter()
    adapter.attach_build_guard(BuildGuard(_hooks(Clock(), Ram())))
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    assert adapter.run(req(task="generate", model=str(tmp_path / "a"), device="CPU", num_predict=4)).ok
    with pytest.raises(AdapterError, match="blocked order keeps at most one"):
        adapter.run(req(task="generate", model=str(tmp_path / "b"), device="CPU", num_predict=4))
    adapter.unload(str(tmp_path / "a"), "CPU")
    assert adapter.run(req(task="generate", model=str(tmp_path / "b"), device="CPU", num_predict=4)).ok


# --- the release check ---------------------------------------------------------


def test_memory_not_released_stops_the_run_with_rows_a_flush_and_a_closed_marker(tmp_path):
    repo, spec_path, _spec = _blocked_spec(tmp_path)
    clock, ram = Clock(), Ram()
    adapter = ResidentAdapter(ram, leak_gib=5.0)  # 5 GiB stays: past the 3 GiB tolerance
    with pytest.raises(RunnerError, match="memory not released after block PHI"):
        _run(repo, spec_path, adapter, _hooks(clock, ram))
    assert clock.sleeps and set(clock.sleeps) == {2.0} and sum(clock.sleeps) >= 60
    assert adapter.calls == [slot.step_id for slot in schedule(["p-igpu-a", "p-igpu-b"], warmups=1, repeats=2)]
    rows = load_jsonl(repo / "receipts" / "run-1.jsonl")
    done = [row for row in rows if not row.get("aborted")]
    aborted = [row for row in rows if row.get("aborted")]
    assert {row["id"].split("/")[1] for row in done} == {"p-igpu-a", "p-igpu-b"}  # the block's rows were flushed
    assert {row["id"].split("/")[1] for row in aborted} == {"q-cpu-a", "q-cpu-b"}  # the unfinished block, never started
    for row in aborted:
        assert row["settings"]["abortion"]["error"] == "memory not released after block PHI"
        assert row["settings"]["blocked"]["block_id"] == "QWEN"
    marker = json.loads(marker_path(repo, "run-1").read_text(encoding="utf-8"))
    assert marker["closed"] is True


def test_memory_that_comes_back_inside_the_wait_lets_the_run_go_on(tmp_path):
    repo, spec_path, _spec = _blocked_spec(tmp_path)
    clock, ram = Clock(), Ram(used_gib=10.0)
    adapter = ResidentAdapter(ram, leak_gib=0.0)
    hot = {"extra": 0.0}

    def used():
        # after the PHI unload the OS gives the pages back slowly: 6 GiB over baseline until t = 7 s
        return ram.used + (hot["extra"] * GIB if clock.now < 7.0 else 0.0)

    original = adapter.unload

    def unload(model, device=None):
        original(model, device)
        if model == PHI:
            hot["extra"] = 6.0
            clock.now = 0.0

    adapter.unload = unload
    rows = _run(repo, spec_path, adapter, _hooks(clock, ram, used))
    assert len(rows) == 4
    assert clock.sleeps == [2.0, 2.0, 2.0, 2.0]  # polled every 2 s until t = 8 s


def test_without_a_ram_reading_the_release_check_stands_down(tmp_path):
    repo, spec_path, _spec = _blocked_spec(tmp_path)
    clock, ram = Clock(), Ram()
    hooks = _hooks(clock, ram)
    hooks.used_sampler = None
    hooks.total_sampler = None
    rows = _run(repo, spec_path, ResidentAdapter(ram, leak_gib=9.0), hooks)
    assert len(rows) == 4 and clock.sleeps == []


# --- the pre-build guard -----------------------------------------------------------


def test_a_build_that_would_cross_85_percent_is_not_run_and_the_run_goes_on(tmp_path):
    repo, spec_path, _spec = _blocked_spec(tmp_path)
    clock = Clock()
    # 64 GiB total, limit 54.4. Used 45: the 8B default step (12) gives 57 > 54.4 refused;
    # the Phi default step (6) gives 51 <= 54.4 allowed.
    ram = Ram(used_gib=45.0)
    adapter = ResidentAdapter(ram, build_gib=2.0)
    rows = _run(repo, spec_path, adapter, _hooks(clock, ram))
    assert adapter.builds == [(PHI, "iGPU")]  # PHI first, then QWEN was refused
    assert {row["id"].split("/")[1] for row in rows} == {"p-igpu-a", "p-igpu-b"}
    assert not any(step.startswith("q-") for step in adapter.calls)
    receipt = load_jsonl(repo / "receipts" / "run-1.jsonl")
    assert len(receipt) == 2
    notes = receipt[0]["notes"]
    assert "q-cpu-a (memory headroom)" in notes and "q-cpu-b (memory headroom)" in notes
    assert all(row.get("outcome") is None for row in receipt)  # not a capability outcome


def test_a_refusal_in_the_first_block_still_lets_the_next_block_run(tmp_path):
    clock = Clock()
    ram = Ram(used_gib=45.0)
    blocks = [{"id": "QWEN", "arms": ["q-cpu-a", "q-cpu-b"]}, {"id": "PHI", "arms": ["p-igpu-a", "p-igpu-b"]}]
    repo, spec_path, _spec = _blocked_spec(tmp_path, blocks=blocks)
    adapter = ResidentAdapter(ram, build_gib=2.0)
    rows = _run(repo, spec_path, adapter, _hooks(clock, ram))
    assert adapter.builds == [(PHI, "iGPU")]  # the refused first block built nothing
    assert {row["id"].split("/")[1] for row in rows} == {"p-igpu-a", "p-igpu-b"}
    assert ram.used == 45 * GIB  # the PHI build was unloaded again


def test_the_build_guard_learns_each_models_step_and_refuses_on_the_largest_seen():
    clock, ram = Clock(), Ram(used_gib=40.0)
    rss = {"v": 5 * GIB}
    hooks = _hooks(clock, ram)
    hooks.rss_sampler = lambda: rss["v"]
    guard = BuildGuard(hooks)
    assert guard.step_for("x/Qwen3-8B-int4") == BUILD_STEP_8B_BYTES
    assert guard.step_for("x/qwen3-8b") == BUILD_STEP_8B_BYTES
    assert guard.step_for("x/Phi-3.5-mini") == BUILD_STEP_DEFAULT_BYTES
    # 40 + 12 = 52 <= 54.4: the first 8B build is allowed; it measures a 3 GiB step
    with guard.building("x/Qwen3-8B-int4"):
        rss["v"] += 3 * GIB
    assert guard.steps == {"x/Qwen3-8B-int4": 3 * GIB}
    ram.used = 50.0 * GIB  # 50 + 3 <= 54.4 passes where the 12 GiB default (62) would not
    guard.check("x/Qwen3-8B-int4")
    with guard.building("x/Qwen3-8B-int4"):
        rss["v"] += 7 * GIB  # a larger step is now the one remembered
    assert guard.steps["x/Qwen3-8B-int4"] == 7 * GIB
    with pytest.raises(BuildRefused, match="memory headroom"):
        guard.check("x/Qwen3-8B-int4")  # 50 + 7 = 57 > 54.4
    assert guard.refused[-1]["model"] == "x/Qwen3-8B-int4"
    # no reading, no refusal
    quiet = BuildGuard(WatchHooks(ram_sampler=None, rss_sampler=None, used_sampler=None, total_sampler=None, start_thread=False))
    quiet.check("x/Qwen3-8B-int4")
    with quiet.building("x/Qwen3-8B-int4"):
        pass
    assert quiet.steps == {}


def test_the_openvino_adapter_checks_before_building_and_a_refusal_builds_nothing(tmp_path):
    from tests.test_openvino_adapter import make_adapter, req

    adapter = make_adapter()
    ram = Ram(used_gib=60.0)
    adapter.attach_build_guard(BuildGuard(_hooks(Clock(), ram)))
    with pytest.raises(BuildRefused):
        adapter.run(req(task="generate", model=str(tmp_path), device="CPU", num_predict=4))
    assert adapter._pipelines == {}


# --- the drift block -----------------------------------------------------------------


def _drift_spec(tmp_path):
    return _blocked_spec(
        tmp_path,
        drift_block={"id": "DRIFT", "repeat_of": "PHI", "arms": ["p-igpu-a"], "repeats": 2},
    )


def test_the_drift_block_is_reported_and_kept_out_of_the_table(tmp_path):
    repo, spec_path, spec = _drift_spec(tmp_path)
    (repo / "table").mkdir()
    clock, ram = Clock(), Ram()
    # the original arm measures 10 tok/s; the same arm after the other block measures 8
    adapter = ResidentAdapter(ram, throughput=lambda step_id, n: 8.0 if step_id == "p-igpu-a-drift" else 10.0)
    rows = _run(repo, spec_path, adapter, _hooks(clock, ram))
    drift_slots = [slot for slot in runner_module._slots(spec) if slot.step_id == "p-igpu-a-drift"]
    assert [slot.phase for slot in drift_slots] == ["warmup", "repeat", "repeat"]  # no cold slot
    assert adapter.calls[-3:] == ["p-igpu-a-drift"] * 3
    drift = [row for row in rows if row["settings"].get("drift_of")]
    assert len(drift) == 1
    row = drift[0]
    assert row["descriptive"] is True
    assert row["id"].split("/")[1] == "p-igpu-a-drift"
    assert row["settings"]["blocked"]["block_id"] == "DRIFT"
    assert row["settings"]["drift"] == {
        "of": "p-igpu-a",
        "warm_throughput": 8.0,
        "original_warm_throughput": 10.0,
        "ratio": 0.8,
    }
    assert is_speed_row(row) is False
    received = load_jsonl(repo / "receipts" / "run-1.jsonl")
    assert any(item.get("descriptive") for item in received)  # the receipt reports it
    table = load_jsonl(repo / "table" / "routing.jsonl")
    assert len(table) == 4 and not any(item.get("descriptive") or "drift" in item["id"] for item in table)
    # the drift rebuild came after an unload: the original pipeline was gone first
    assert adapter.builds == [(PHI, "iGPU"), (QWEN, "CPU"), (PHI, "iGPU")]


def test_a_drift_row_left_in_a_receipt_never_imports_into_the_table(tmp_path):
    repo, spec_path, _spec = _drift_spec(tmp_path)
    (repo / "table").mkdir()
    clock, ram = Clock(), Ram()
    adapter = ResidentAdapter(ram, fail_on=lambda step_id, n: step_id == "p-igpu-a-drift" and n == 3)
    with pytest.raises(AdapterError, match="drift boom"):
        _run(repo, spec_path, adapter, _hooks(clock, ram))
    receipt = load_jsonl(repo / "receipts" / "run-1.jsonl")
    assert any(item.get("descriptive") for item in receipt)
    table = (repo / "table" / "routing.jsonl")
    kept = load_jsonl(table)
    assert kept and not any(item.get("descriptive") or "drift" in item["id"] for item in kept)
    again = [item for item in receipt if item.get("descriptive")]
    assert all(is_speed_row(item) is False for item in again)
    # importing the receipt again (the aborted-run path) leaves the descriptive rows out of the table
    (repo / "table" / "routing.jsonl").write_text("", encoding="utf-8")
    import_receipt(repo, "run-1")
    assert not any(item.get("descriptive") for item in load_jsonl(repo / "table" / "routing.jsonl"))


def test_a_drift_arm_whose_original_did_not_finish_is_noted_not_run(tmp_path):
    repo, spec_path, _spec = _drift_spec(tmp_path)
    clock = Clock()
    ram = Ram(used_gib=50.0)  # 50 + 6 and 50 + 12 both pass 54.4: every block is refused
    adapter = ResidentAdapter(ram, build_gib=1.0)
    rows = _run(repo, spec_path, adapter, _hooks(clock, ram))
    assert rows == []
    assert adapter.calls == []


# --- the spec --------------------------------------------------------------------------


def test_a_blocked_spec_is_validated(tmp_path):
    _repo, _path, spec = _blocked_spec(tmp_path)
    validate_spec(spec)

    def broken(**change):
        data = json.loads(json.dumps(spec))
        for key, value in change.items():
            if value is None:
                data.pop(key, None)
            else:
                data[key] = value
        return data

    with pytest.raises(SpecError, match="at least one block"):
        validate_spec(broken(blocks=None))
    with pytest.raises(SpecError, match="at least one block"):
        validate_spec(broken(blocks=[]))
    with pytest.raises(SpecError, match="not in the spec"):
        validate_spec(broken(blocks=[{"id": "X", "arms": ["nope", "q-cpu-a", "q-cpu-b", "p-igpu-a", "p-igpu-b"]}]))
    with pytest.raises(SpecError, match="arms in no block"):
        validate_spec(broken(blocks=[{"id": "X", "arms": ["q-cpu-a"]}]))
    with pytest.raises(SpecError, match="exactly one block"):
        validate_spec(
            broken(
                blocks=[
                    {"id": "X", "arms": ["q-cpu-a", "q-cpu-b", "p-igpu-a"]},
                    {"id": "Y", "arms": ["p-igpu-a", "p-igpu-b"]},
                ]
            )
        )
    with pytest.raises(SpecError, match="duplicate block"):
        validate_spec(
            broken(
                blocks=[{"id": "X", "arms": ["q-cpu-a", "q-cpu-b"]}, {"id": "X", "arms": ["p-igpu-a", "p-igpu-b"]}]
            )
        )
    with pytest.raises(SpecError, match="every block needs an id"):
        validate_spec(broken(blocks=[{"arms": ["q-cpu-a"]}]))
    with pytest.raises(SpecError, match="needs arms"):
        validate_spec(broken(blocks=[{"id": "X", "arms": []}]))
    with pytest.raises(SpecError, match="no pairs"):
        validate_spec(broken(pairs=[{"id": "p"}]))
    with pytest.raises(SpecError, match="belong to order: blocked"):
        validate_spec(broken(order="interleaved"))
    drift = {"id": "D", "repeat_of": "PHI", "arms": ["p-igpu-a"], "repeats": 1}
    validate_spec(broken(drift_block=drift))
    for change, message in (
        ({"id": ""}, "needs an id"),
        ({"id": "PHI"}, "repeats a block id"),
        ({"repeat_of": "NOPE"}, "must name a block"),
        ({"arms": []}, "needs arms"),
        ({"arms": ["q-cpu-a"]}, "is not in block PHI"),
        ({"arms": ["p-igpu-a", "p-igpu-a"]}, "twice"),
        ({"repeats": 0}, "integer >= 1"),
        ({"repeats": True}, "integer >= 1"),
    ):
        with pytest.raises(SpecError, match=message):
            validate_spec(broken(drift_block={**drift, **change}))
    with pytest.raises(SpecError, match="drift_block needs an id"):
        validate_spec(broken(drift_block="x"))


def test_an_interleaved_spec_is_unaffected(tmp_path):
    repo = tmp_path / "repo"
    spec = _harness_spec()
    spec["needs"] = {"grant": False, "npu_session": False, "cpu_window": False}
    spec["arms"] = [_gen("a", "CPU", QWEN), _gen("b", "iGPU", PHI)]
    path = _bind(repo, spec)
    clock, ram = Clock(), Ram()
    attached = []

    class Plain(ResidentAdapter):
        def attach_build_guard(self, guard):
            attached.append(guard)

        def run(self, request):
            self.calls.append(request.step_id)
            return _result(5, 10.0)

    adapter = Plain(ram)
    rows = _run(repo, path, adapter, _hooks(clock, ram))
    assert attached == [] and adapter.unloads == [(QWEN, "CPU"), (PHI, "iGPU")]
    assert adapter.calls == [slot.step_id for slot in schedule(["a", "b"], warmups=1, repeats=2)]
    assert all("blocked" not in row["settings"] and "descriptive" not in row for row in rows)
    assert clock.sleeps == []
    assert "blocks" not in dry_run(path, repo)


def test_the_dry_run_lists_the_blocks(tmp_path):
    repo, spec_path, _spec = _drift_spec(tmp_path)
    text = dry_run(spec_path, repo)
    assert "order: blocked" in text
    assert "blocks: PHI (2 arms), QWEN (2 arms)" in text
    assert "within 3 GiB" in text and "85%" in text
    assert "drift block DRIFT" in text
    assert "slots: 19" in text  # 2 blocks x (2 cold + 2 warm-up + 4 repeat) + 1 warm-up + 2 repeats of the drift arm


def test_announce_is_not_confused_by_a_descriptive_row(tmp_path):
    repo, spec_path, _spec = _drift_spec(tmp_path)
    clock, ram = Clock(), Ram()
    rows = _run(repo, spec_path, ResidentAdapter(ram), _hooks(clock, ram))
    announce_run(rows, finished=True)
