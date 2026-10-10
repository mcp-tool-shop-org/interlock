"""Dry-run a committed spec, or run one against adapters the caller supplies.

The command-line dry-run does not stat the heartbeat, does not call an adapter,
and does not launch offrig. A library run with fake adapters is how the timing
math and the receipt shape are tested. Phase 2's command refuses to start a
device: that waits for a Publisher window in Phase 3.

The library run loads the spec only after the committed blob matches the
working tree. That is the same lock the dry-run uses, on the path Phase 3 runs.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from switchyard.adapters.base import AdapterError, StepRequest
from switchyard.adapters.offrig_calibrate import plan_argv
from switchyard.cli import announce_run
from switchyard.guards import assess_step
from switchyard.harness.e5 import SearchError, fit_prompt
from switchyard.harness.lock import require_unmodified
from switchyard.harness.receipts import (
    device_lost_row,
    environment,
    import_receipt,
    measured_row,
    write_receipt,
    write_run,
)
from switchyard.harness.spec import load_spec
from switchyard.receipt import validate_row
from switchyard.harness.timing import Slot, TimingError, aba_call, schedule, schedule_aba, summarize

PASSTHROUGH = (
    "thinking_level",
    "num_predict",
    "num_ctx",
    "batch",
    "static_shape",
    "precision",
    "keep_alive",
    "prompt",
    "shift",
    "truncate",
    "temperature",
    "quant",
)
SENSOR_KEYS = ("gpu_temp_c", "sm_clock_mhz", "power_w", "shared_mib", "dedicated_mib")


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
        slots = _slots(spec)
        lines.append(f"slots: {len(slots)}")
        lines.append(f"order: {spec.get('order', 'interleaved')}")
        lines.append(f"compile_cache: {spec['compile_cache']}")
        lines.append(f"rest_s: {spec['rest_s']}")
        lines.append("cold is measured once per arm and kept out of the warm percentiles")
        lines.append(f"band: {spec['band']}")
    elif spec["procedure"] == "fill-card":
        lines.append(f"abort_mib: {spec['abort_mib']}")
        lines.append(f"ceiling_mib: {spec['ceiling_mib']}")
        lines.append(f"models: {len(spec['models'])}")
        lines.append(f"spill_shared_mib: {spec['spill_shared_mib']}")
        lines.append(f"spill_prompt_ratio: {spec['spill_prompt_ratio']}")
        lines.append(f"rest_s: {spec['rest_s']}")
        lines.append("the sampler aborts at abort_mib and does not wait for the watchdog")
        lines.append("quant is read from /api/show at run time")
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
    adapters: dict,
    *,
    grant: str | None,
    npu_session: str | None,
    cpu_window: str | None,
    root: Path,
    repo: Path,
    spec_path: Path,
    run_id: str,
    date: str,
    env: dict | None = None,
    notes: str = "",
) -> list[dict]:
    """Run a committed spec.

    Each arm is written into the receipt when its slots finish. The receipt
    stays partial until the run ends. A 60 s hang becomes one ``hang`` row
    and stops the run. A device loss becomes one ``device_lost`` row, that
    device is marked off, and the other devices carry on. A step reported as
    not run writes no row; it is listed in the receipt notes. Neither outcome
    is retried.
    """

    _check_run_id(run_id)
    require_unmodified(repo, spec_path)
    spec = load_spec(spec_path)
    if spec["procedure"] != "interleaved":
        raise RunnerError("run_interleaved only runs an interleaved spec")
    spec_rel = spec_path.resolve().relative_to(repo.resolve()).as_posix()
    _guard_flags(spec, grant, npu_session, cpu_window)
    arms = {arm["id"]: arm for arm in spec["arms"]}
    slots = _slots(spec)
    needed: dict[str, int] = {}
    for slot in slots:
        needed[slot.step_id] = needed.get(slot.step_id, 0) + 1
    records: list[dict] = []
    seen: dict[str, int] = {}
    finished: list[str] = []
    capability: dict[str, str] = {}
    off_devices: set[str] = set()
    skipped: list[str] = []
    captured = env if env is not None else environment()
    cache_dirs = _cache_dirs(spec, root, run_id, arms)
    prompt_cache: dict[str, tuple[str, int]] = {}
    removed: set[str] = set()
    imported = False

    def _run_notes() -> str:
        extra = "not run: " + ", ".join(skipped) if skipped else ""
        if notes and extra:
            return f"{notes}. {extra}"
        return notes or extra

    def _finished_row(arm_id: str, *, partial: bool, gone: set[str]) -> dict:
        extra = _extra_settings(spec, arms[arm_id], records, run_id, gone)
        extra.update(_prompt_counts(arms[arm_id], prompt_cache))
        row = measured_row(
            spec_id=spec["id"],
            arm=arms[arm_id],
            summary=_summary(records, arm_id),
            repeats=spec["repeats"],
            date=date,
            spec_rel=spec_rel,
            run_id=run_id,
            grant=grant,
            env=captured,
            notes=_run_notes(),
            resolved=_resolved(records, arm_id),
            artifacts=_artifacts(records, arm_id),
            extra_settings=extra,
        )
        if partial:
            row["partial"] = True
            validate_row(row)
        return row

    def _capability_row(arm_id: str, *, partial: bool, gone: set[str]) -> dict:
        extra = _extra_settings(spec, arms[arm_id], [], run_id, gone)
        extra.update(_prompt_counts(arms[arm_id], prompt_cache))
        return device_lost_row(
            spec_id=spec["id"],
            arm=arms[arm_id],
            repeats=spec["repeats"],
            date=date,
            spec_rel=spec_rel,
            run_id=run_id,
            grant=grant,
            env=captured,
            notes=_run_notes(),
            extra_settings=extra,
            outcome=capability[arm_id],
            partial=partial,
        )

    def _rows(*, partial: bool, gone: set[str]) -> list[dict]:
        rows = [_finished_row(arm_id, partial=partial, gone=gone) for arm_id in finished]
        rows.extend(_capability_row(arm_id, partial=partial, gone=gone) for arm_id in capability)
        return rows

    def flush(*, gone: set[str] | None = None) -> None:
        used = removed if gone is None else gone
        rows = _rows(partial=True, gone=used)
        if rows:
            write_receipt(root, run_id, rows)

    def _remember(arm: dict, outcome: str) -> None:
        """Record a hang or a device loss. Partial samples from this arm are discarded."""

        arm_id = arm["id"]
        records[:] = [row for row in records if row["step_id"] != arm_id]
        capability[arm_id] = outcome
        if outcome == "device_lost":
            off_devices.add(arm["device"])

    def _note(slot: Slot) -> None:
        if slot.phase == "cold":
            label = f"{slot.step_id} cold"
        else:
            label = f"{slot.step_id} {slot.phase} {slot.round_index}"
        skipped.append(label)

    try:
        try:
            for slot in slots:
                arm = arms[slot.step_id]
                companion = _companion(arm, arms, slot)
                if arm["device"] in off_devices or (
                    companion is not None and companion["device"] in off_devices
                ):
                    _note(slot)
                    continue
                started = False
                if companion is not None:
                    _session(
                        adapters,
                        companion,
                        "start",
                        spec,
                        spec_rel,
                        cache_dirs.get(companion["id"]),
                        prompt_cache,
                    )
                    started = True
                try:
                    kind = "ok"
                    error = ""
                    try:
                        result = adapters[arm["engine"]].run(
                            _request(
                                arm,
                                spec,
                                spec_rel,
                                cache_dirs.get(arm["id"]),
                                adapters[arm["engine"]],
                                prompt_cache,
                            )
                        )
                    except AdapterError as exc:
                        error = str(exc)
                        if hang_error(error):
                            kind = "hang"
                        elif device_lost_error(error):
                            kind = "device_lost"
                        else:
                            flush()
                            raise
                    else:
                        reported = _reported_outcome(result)
                        if reported == "device_lost":
                            kind = "device_lost"
                        elif reported == "not run":
                            kind = "not run"
                        elif not result.ok:
                            error = result.error or ""
                            if hang_error(error):
                                kind = "hang"
                            elif device_lost_error(error):
                                kind = "device_lost"
                            else:
                                kind = "fail"
                    if kind == "hang":
                        _remember(arm, "hang")
                        flush()
                        raise RunnerError(f"step {slot.step_id} failed: {error}")
                    if kind == "device_lost":
                        _remember(arm, "device_lost")
                        flush()
                    elif kind == "not run":
                        off_devices.add(arm["device"])
                        _note(slot)
                    elif kind == "fail":
                        flush()
                        raise RunnerError(f"step {slot.step_id} failed: {error}")
                    else:
                        records.append(_record(slot, result))
                        seen[slot.step_id] = seen.get(slot.step_id, 0) + 1
                        if seen[slot.step_id] == needed[slot.step_id]:
                            finished.append(slot.step_id)
                            flush()
                finally:
                    if started and companion is not None:
                        _session(
                            adapters,
                            companion,
                            "stop",
                            spec,
                            spec_rel,
                            cache_dirs.get(companion["id"]),
                            prompt_cache,
                        )
        finally:
            _unload(adapters, arms)
            removed.update(_drop_caches(cache_dirs))
        rows = _rows(partial=False, gone=removed)
        if rows:
            write_run(root, run_id, rows)
        imported = True
        announce_run(rows, finished=True)
        return rows
    finally:
        receipt = root / "receipts" / f"{run_id}.jsonl"
        if not imported and receipt.is_file():
            flush(gone=removed)
            written = import_receipt(root, run_id)
            announce_run(written, finished=False)


def hang_error(error: str | None) -> bool:
    """A 60 s hang. The call was stuck and the run stopped watching."""

    return isinstance(error, str) and error.lower().startswith("hang:")


def device_lost_error(error: str | None) -> bool:
    """A driver reset. A hang is not a device loss.

    The match is the Level Zero code, ``ZE_RESULT_ERROR_DEVICE_LOST`` or
    ``device_lost``. The driver's own sentence also says "device hung"; that
    phrase has not been seen without the code, so it is not a match.
    """

    if not isinstance(error, str) or not error or hang_error(error):
        return False
    text = error.lower()
    return "ze_result_error_device_lost" in text or "device_lost" in text or "device lost" in text


def _reported_outcome(result) -> str | None:
    """The adapter's own outcome, when it finished the step instead of failing it."""

    resolved = result.resolved
    if not isinstance(resolved, dict):
        return None
    outcome = resolved.get("outcome")
    if outcome in ("device_lost", "not run"):
        return outcome
    return None


def _slots(spec: dict) -> list[Slot]:
    order = spec.get("order", "interleaved")
    if order == "aba":
        return schedule_aba(spec["pairs"], warmups=spec["warmups"], repeats=spec["repeats"])
    return schedule([arm["id"] for arm in spec["arms"]], warmups=spec["warmups"], repeats=spec["repeats"])


def _request(
    arm: dict,
    spec: dict,
    spec_rel: str,
    cache_dir: Path | None,
    adapter,
    prompt_cache: dict[str, tuple[str, int]],
) -> StepRequest:
    settings = {key: arm.get(key) for key in PASSTHROUGH}
    if not settings.get("prompt"):
        prompt = _dataset_prompt(arm, spec, adapter, prompt_cache)
        if prompt is not None:
            settings["prompt"] = prompt
    settings["spec"] = spec_rel
    settings["compile_cache"] = spec.get("compile_cache")
    if cache_dir is not None:
        settings["compile_cache_dir"] = str(cache_dir)
    return StepRequest(
        step_id=arm["id"],
        task=arm["task"],
        model=arm["model"],
        device=arm["device"],
        engine=arm["engine"],
        settings=settings,
    )


def _record(slot: Slot, result) -> dict:
    sensors = None
    if isinstance(result.sensors, dict):
        sensors = {key: result.sensors.get(key) for key in SENSOR_KEYS if key in result.sensors}
    return {
        "step_id": slot.step_id,
        "phase": slot.phase,
        "round": slot.round_index,
        "role": slot.role,
        "pair": slot.pair,
        "latency_ms": result.latency_ms,
        "throughput": result.throughput,
        "throughput_unit": result.throughput_unit,
        "prompt_tok_s": result.prompt_tok_s,
        "compile_ms": result.compile_ms,
        "first_call_ms": result.first_call_ms,
        "output": result.output,
        "resolved": result.resolved,
        "sensors": sensors,
    }


def _summary(records: list[dict], arm_id: str):
    cold = [row for row in records if row["step_id"] == arm_id and row["phase"] == "cold"]
    repeats = [row for row in records if row["step_id"] == arm_id and row["phase"] == "repeat"]
    cold_row = cold[-1] if cold else None
    units = [
        row["throughput_unit"]
        for row in repeats
        if row["throughput"] is not None
    ]
    present = [unit for unit in units if unit]
    if present and (len(set(present)) != 1 or len(present) != len(units)):
        raise TimingError(f"arm {arm_id} reported more than one throughput unit")
    unit = present[0] if present else None
    return summarize(
        None if cold_row is None else cold_row["latency_ms"],
        [row["latency_ms"] for row in repeats if row["latency_ms"] is not None],
        [row["throughput"] for row in repeats if row["throughput"] is not None],
        throughput_unit=unit,
        compile_ms=None if cold_row is None else cold_row["compile_ms"],
        first_call_ms=None if cold_row is None else cold_row["first_call_ms"],
    )


def _resolved(records: list[dict], arm_id: str) -> dict:
    learned: dict = {}
    for row in records:
        if row["step_id"] != arm_id or not isinstance(row["resolved"], dict):
            continue
        for key, value in row["resolved"].items():
            if value is not None:
                learned[key] = value
    return learned


def _artifacts(records: list[dict], arm_id: str) -> dict | None:
    slots = []
    for row in records:
        if row["step_id"] != arm_id:
            continue
        item = {
            "phase": row["phase"],
            "round": row["round"],
            "role": row["role"],
            "pair": row["pair"],
            "latency_ms": row["latency_ms"],
            "throughput": row["throughput"],
            "throughput_unit": row["throughput_unit"],
            "prompt_tok_s": row["prompt_tok_s"],
        }
        if row["output"] is not None:
            item["output"] = row["output"]
        if row["sensors"]:
            item["sensors"] = row["sensors"]
        slots.append(item)
    if not slots:
        return None
    return {"slots": slots}


def _extra_settings(spec: dict, arm: dict, records: list[dict], run_id: str, removed: set[str]) -> dict:
    settings = {
        "compile_cache": spec.get("compile_cache"),
        "rest_s": spec.get("rest_s"),
    }
    if spec.get("compile_cache") == "fresh":
        settings["compile_cache_dir"] = f"compile-cache/{run_id}/{arm['id']}"
        settings["compile_cache_deleted"] = arm["id"] in removed
    if spec.get("order") == "aba":
        calls = []
        for pair in spec.get("pairs", []):
            if pair["paired"] != arm["id"]:
                continue
            measure = arm.get("measure", "latency_ms")
            calls.append(
                {
                    "pair": pair["id"],
                    **aba_call(
                        _series(records, pair["alone"], pair["id"], "alone-before", measure),
                        _series(records, pair["alone"], pair["id"], "alone-after", measure),
                        _series(records, pair["paired"], pair["id"], "paired", measure),
                        band=spec["band"],
                    ),
                }
            )
        if calls:
            settings["aba"] = calls
    return settings


def _series(records: list[dict], arm_id: str, pair: str, role: str, measure: str) -> list[float]:
    key = "throughput" if measure == "decode_tok_s" else "latency_ms"
    values = []
    for row in records:
        if row["phase"] != "repeat" or row["step_id"] != arm_id or row["pair"] != pair or row["role"] != role:
            continue
        if row[key] is not None:
            values.append(row[key])
    return values


def _companion(arm: dict, arms: dict, slot: Slot) -> dict | None:
    """The sustained load for a paired slot, or for an interleaved measure slot.

    Alone-before, alone-after, and cold do not start a companion. A missing
    companion id never does.
    """

    companion_id = arm.get("companion")
    if not companion_id:
        return None
    if slot.role in {"alone-before", "alone-after", "cold"}:
        return None
    if slot.role in {"paired", "measure"}:
        return arms[companion_id]
    return None


def _session(
    adapters: dict,
    arm: dict,
    action: str,
    spec: dict,
    spec_rel: str,
    cache_dir: Path | None,
    prompt_cache: dict[str, tuple[str, int]],
) -> None:
    adapter = adapters[arm["engine"]]
    method = getattr(adapter, f"{action}_session", None)
    if not callable(method):
        raise RunnerError(
            f"{arm['engine']} has no {action}_session. "
            "The harness owns companion sessions and does not fake one."
        )
    method(_request(arm, spec, spec_rel, cache_dir, adapter, prompt_cache))


def _dataset_prompt(arm: dict, spec: dict, adapter, prompt_cache: dict[str, tuple[str, int]]) -> str | None:
    """The spec's prefix: repeat the seed until the model tokenizer reaches the target.

    One prefix per model, shared by every device of that model. A spec with a
    seed and no ``target_tokens`` still passes the seed through. A target with
    no tokenizer count is refused, so the raw seed is not measured by mistake.
    """

    dataset = spec.get("dataset") if isinstance(spec.get("dataset"), dict) else None
    if not isinstance(dataset, dict):
        return None
    seed = dataset.get("seed")
    if not isinstance(seed, str):
        return None
    target = dataset.get("target_tokens")
    if isinstance(target, bool) or not isinstance(target, int):
        return seed
    model = arm["model"]
    cached = prompt_cache.get(model)
    if cached is None:
        counter = getattr(adapter, "count_tokens", None)
        if not callable(counter):
            raise RunnerError(
                f"{arm['id']}: dataset.target_tokens is {target} and {arm['engine']} cannot count tokens"
            )

        def count(text: str, model: str = model) -> int:
            return counter(model, text)

        try:
            text = fit_prompt(seed, target, count)
        except SearchError as exc:
            raise RunnerError(str(exc)) from exc
        cached = (text, count(text))
        prompt_cache[model] = cached
    return cached[0]


def _prompt_counts(arm: dict, prompt_cache: dict[str, tuple[str, int]]) -> dict:
    cached = prompt_cache.get(arm["model"])
    if cached is None:
        return {}
    _text, prompt_tokens = cached
    counts = {"prompt_tokens": prompt_tokens}
    length = _static_length(arm.get("static_shape"))
    if length is not None:
        counts["truncated_tokens"] = max(0, prompt_tokens - length)
    return counts


def _static_length(shape: object) -> int | None:
    if not isinstance(shape, str) or "x" not in shape.lower():
        return None
    length = shape.lower().split("x", 1)[1]
    if not length.isdigit():
        return None
    value = int(length)
    return value if value > 0 else None


def _cache_dirs(spec: dict, root: Path, run_id: str, arms: dict) -> dict[str, Path]:
    if spec.get("compile_cache") != "fresh":
        return {}
    dirs = {}
    for arm_id in arms:
        path = root / "compile-cache" / run_id / arm_id
        path.mkdir(parents=True, exist_ok=True)
        dirs[arm_id] = path
    return dirs


def _drop_caches(cache_dirs: dict[str, Path]) -> set[str]:
    gone = set()
    for arm_id, path in cache_dirs.items():
        shutil.rmtree(path, ignore_errors=True)
        if not path.exists():
            gone.add(arm_id)
    return gone


def _check_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or not run_id or any(part in run_id for part in ("/", "\\", "..")):
        raise RunnerError("run_id must be a single name")


def _unload(adapters: dict, arms: dict) -> None:
    seen = set()
    for arm in arms.values():
        key = (arm["engine"], arm["model"], arm["device"])
        if key in seen:
            continue
        seen.add(key)
        adapters[arm["engine"]].unload(arm["model"], arm["device"])


def _guard_flags(spec: dict, grant: str | None, npu_session: str | None, cpu_window: str | None) -> None:
    devices = set()
    for arm in spec["arms"]:
        devices.add(arm["device"])
        if arm.get("companion_device"):
            devices.add(arm["companion_device"])
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
