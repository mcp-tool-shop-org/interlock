"""Dry-run a committed spec, or run one against adapters the caller supplies.

The command-line dry-run does not stat the heartbeat, does not call an adapter,
and does not launch offrig. A library run with fake adapters is how the timing
math and the receipt shape are tested. Phase 2's command refuses to start a
device: that waits for a Publisher window in Phase 3.
"""

from __future__ import annotations

from pathlib import Path

from switchyard.adapters.base import StepRequest
from switchyard.adapters.offrig_calibrate import plan_argv
from switchyard.guards import (
    assess_step,
)
from switchyard.harness.lock import require_unmodified
from switchyard.harness.receipts import environment, measured_row, write_run
from switchyard.harness.spec import load_spec
from switchyard.harness.timing import Slot, schedule, summarize


class RunnerError(Exception):
    """A step failed. Later steps do not run."""


def dry_run(spec_path: Path, repo: Path) -> str:
    spec = load_spec(spec_path)
    blob = require_unmodified(repo, spec_path)
    lines = [
        f"switchyard dry-run: {spec['id']}",
        f"spec blob: {blob}",
        f"procedure: {spec['procedure']}",
    ]
    if spec["procedure"] == "interleaved":
        slots = schedule([arm["id"] for arm in spec["arms"]], warmups=spec["warmups"], repeats=spec["repeats"])
        lines.append(f"slots: {len(slots)}")
        lines.append("cold is measured once per arm and kept out of the warm percentiles")
        lines.append(f"band: {spec['band']}")
    elif spec["procedure"] == "fill-card":
        lines.append(f"abort_mib: {spec['abort_mib']}")
        lines.append(f"ceiling_mib: {spec['ceiling_mib']}")
        lines.append(f"models: {len(spec['models'])}")
        lines.append("the sampler aborts at abort_mib and does not wait for the watchdog")
    else:
        lines.append("argv: " + " ".join(plan_argv(spec)))
        lines.append("not launched")
    needs = spec["needs"]
    flags = []
    if needs["grant"]:
        flags.append("--grant")
    if needs["npu_session"]:
        flags.append("--npu-session")
    if needs["cpu_window"]:
        flags.append("--cpu-window")
    lines.append("a device run would require: " + (", ".join(flags) if flags else "no extra flag"))
    lines.append("heartbeat not read; adapters not called")
    return "\n".join(lines) + "\n"


def run_interleaved(
    spec: dict,
    adapters: dict,
    *,
    grant: str | None,
    npu_session: str | None,
    cpu_window: str | None,
    root: Path,
    spec_rel: str,
    run_id: str,
    date: str,
    env: dict | None = None,
    notes: str = "",
) -> list[dict]:
    """Run every slot. A failed step stops the sequence and writes no receipt."""

    _guard_flags(spec, grant, npu_session, cpu_window)
    arms = {arm["id"]: arm for arm in spec["arms"]}
    slots = schedule(list(arms), warmups=spec["warmups"], repeats=spec["repeats"])
    cold: dict[str, float | None] = {}
    warm_ms: dict[str, list[float]] = {arm_id: [] for arm_id in arms}
    warm_tp: dict[str, list[float]] = {arm_id: [] for arm_id in arms}
    try:
        for slot in slots:
            arm = arms[slot.step_id]
            result = adapters[arm["engine"]].run(_request(arm, spec_rel))
            if not result.ok:
                raise RunnerError(f"step {slot.step_id} failed: {result.error}")
            _record(slot, result, cold, warm_ms, warm_tp)
    finally:
        _unload(adapters, arms)
    rows = []
    captured = env if env is not None else environment()
    for arm_id, arm in arms.items():
        summary = summarize(cold.get(arm_id), warm_ms[arm_id], warm_tp[arm_id])
        rows.append(
            measured_row(
                spec_id=spec["id"],
                arm=arm,
                summary=summary,
                repeats=spec["repeats"],
                date=date,
                spec_rel=spec_rel,
                run_id=run_id,
                grant=grant,
                env=captured,
                notes=notes,
            )
        )
    write_run(root, run_id, rows)
    return rows


def _request(arm: dict, spec_rel: str) -> StepRequest:
    settings = {
        "thinking_level": arm.get("thinking_level"),
        "num_predict": arm.get("num_predict"),
        "num_ctx": arm.get("num_ctx"),
        "batch": arm.get("batch"),
        "prompt": arm.get("prompt", ""),
        "spec": spec_rel,
    }
    return StepRequest(
        step_id=arm["id"],
        task=arm["task"],
        model=arm["model"],
        device=arm["device"],
        engine=arm["engine"],
        settings=settings,
    )


def _record(slot: Slot, result, cold, warm_ms, warm_tp) -> None:
    if slot.phase == "cold":
        cold[slot.step_id] = result.latency_ms
        return
    if slot.phase == "warmup":
        return
    if result.latency_ms is not None:
        warm_ms[slot.step_id].append(result.latency_ms)
    if result.throughput is not None:
        warm_tp[slot.step_id].append(result.throughput)


def _unload(adapters: dict, arms: dict) -> None:
    seen = set()
    for arm in arms.values():
        key = (arm["engine"], arm["model"])
        if key in seen:
            continue
        seen.add(key)
        adapters[arm["engine"]].unload(arm["model"])


def _guard_flags(spec: dict, grant: str | None, npu_session: str | None, cpu_window: str | None) -> None:
    devices = {arm["device"] for arm in spec["arms"]}
    needs = spec["needs"]
    for device in sorted(devices):
        assess_step(
            device=device,
            grant=grant,
            npu_session=npu_session,
            cpu_window=cpu_window,
            cpu_heavy=needs["cpu_window"] and device == "CPU",
        )


def devices_not_started() -> str:
    return "REFUSED: Phase 2 does not start a device. Re-run with --dry-run."


def check_flags(spec: dict, grant: str | None, npu_session: str | None, cpu_window: str | None) -> None:
    """Flag guards only. This does not stat the heartbeat or call a device."""

    if spec["procedure"] == "interleaved":
        _guard_flags(spec, grant, npu_session, cpu_window)
        return
    needs = spec["needs"]
    if needs["grant"]:
        assess_step(device="5090", grant=grant)
    if needs["npu_session"]:
        assess_step(device="NPU", npu_session=npu_session)
    if needs["cpu_window"]:
        assess_step(device="CPU", cpu_window=cpu_window, cpu_heavy=True)
