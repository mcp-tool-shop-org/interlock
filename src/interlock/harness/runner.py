"""Dry-run a committed spec, or run one against adapters the caller supplies.

The command-line dry-run does not stat the heartbeat, does not call an adapter,
and does not launch offrig. An Ollama dataset target is named there as a
run-time prompt_eval_count and is not counted. A library run with fake adapters is how the timing
math and the receipt shape are tested. The command's real path is run_spec:
every guard (interpreter, procedure, flag, committed blob, card state) refuses before an adapter is built or called; then run_interleaved
measures.

The library run loads the spec only after the committed blob matches the
working tree. That is the same lock the dry-run uses, on the path Phase 3 runs.
"""

from __future__ import annotations

import datetime
import gc
import os
import re
import shutil
import stat
import sys
import threading
import time
from pathlib import Path

from interlock.adapters.base import (
    COMPILE_REJECTED_CODE,
    COMPILER_MESSAGE_MARKER,
    AdapterError,
    BuildRefused,
    CompileRejected,
    NativeBinaryRejected,
    CompileTracker,
    StepRequest,
    compile_rejected_error,
)
from interlock.adapters.ollama import COUNT_TIMEOUT_NOTE
from interlock.adapters.offrig_calibrate import plan_argv
from interlock.adapters.registry import for_engine
from interlock.cli import announce_run
from interlock.devices import CARD_5090
from interlock.guards import (
    GuardError,
    assess_step,
    require_card_empty,
    require_heartbeat,
)
from interlock.harness.count_budget import roster_timeout
from interlock.harness.e5 import SearchError, fit_prompt, fit_prompt_words
from interlock.harness.fill import quiet_fill_seams, real_fill_seams, run_fill_card
from interlock.ollama_supervisor import (
    SupervisorStop,
    failure_kind,
    restart_policy_lines,
    retry_budget,
)
from interlock.adapters.ollama import url_for
from interlock.harness.steady import BlockPlan, SlotStream, UntilSteady, call_metric, warmup_policy, warmup_trace
from interlock.harness.lock import require_unmodified
from interlock.harness.marker import RunMarker
from interlock.harness.receipts import (
    device_lost_row,
    environment,
    import_receipt,
    measured_row,
    write_receipt,
    write_run,
)
from interlock.harness.spec import load_spec
from interlock.harness.watch import (
    BUILD_GUARD_PCT,
    COMPILE_CEILING_S,
    RAM_STOP_PCT,
    RELEASE_POLL_S,
    RELEASE_TOLERANCE_BYTES,
    RELEASE_WAIT_S,
    BuildGuard,
    RunWatch,
    WatchHooks,
    read_bytes,
    slot_probe,
)
from interlock.receipt import ReceiptError, load_jsonl, validate_row
from interlock.harness.timing import Slot, TimingError, aba_call, schedule, schedule_aba, summarize

try:  # the CPU-window evidence of ruling #36's addendum; never a step failure
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

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
    "dynamic_shape",
    "temperature",
    "quant",
    "hf_cache",
    "compile_config",
    "cold_copy",
)
SENSOR_KEYS = ("gpu_temp_c", "sm_clock_mhz", "power_w", "shared_mib", "dedicated_mib")


class RunnerError(Exception):
    """A step failed. Later steps do not run."""


def _storage_lines(spec: dict) -> list[str]:
    """Dry-run lines for the cache, the cold copy and ENABLE_MMAP. Absent options add nothing."""

    lines = []
    for arm in spec.get("arms") or []:
        cache = arm.get("hf_cache")
        if cache == "hot":
            lines.append(
                f"{arm['id']}: hf_cache hot (C:/AI-Models/hf-cache-hot); a missing copy is a refusal"
            )
        elif cache == "archive":
            lines.append(f"{arm['id']}: hf_cache archive")
        if arm.get("cold_copy") is True:
            lines.append(
                f"{arm['id']}: cold copy on the snapshot drive before the first load, deleted after unload"
            )
        config = arm.get("compile_config") if isinstance(arm.get("compile_config"), dict) else {}
        if "ENABLE_MMAP" in config:
            lines.append(f"{arm['id']}: ENABLE_MMAP {str(bool(config['ENABLE_MMAP'])).lower()}")
    return lines


def parse_models(text: str | None) -> list[str] | None:
    """Comma-separated fill-card model names. None means the whole roster."""

    if text is None:
        return None
    names = [part.strip() for part in str(text).split(",")]
    if not names or any(not name for name in names):
        raise RunnerError("--models names at least one model")
    return names


def resolve_models(spec: dict, models: list[str] | None) -> tuple[list[dict], list[str] | None]:
    """The roster entries this run measures, in spec order. None records no subset."""

    roster = [entry for entry in spec.get("models") or [] if isinstance(entry, dict)]
    if models is None:
        return roster, None
    if not isinstance(models, (list, tuple)) or not models:
        raise RunnerError("--models names at least one model")
    known = [entry.get("model") for entry in roster]
    unknown = [name for name in models if name not in known]
    if unknown:
        raise RunnerError(f"unknown model: {unknown[0]}")
    if len(list(models)) != len(set(models)):
        raise RunnerError("a model is named more than once")
    chosen = [entry for entry in roster if entry.get("model") in set(models)]
    return chosen, [entry["model"] for entry in chosen]


def dry_run(
    spec_path: Path,
    repo: Path,
    pilot: bool = False,
    pairs: list[str] | None = None,
    models: list[str] | None = None,
) -> str:
    spec = load_spec(spec_path)
    if models is not None and spec.get("procedure") != "fill-card":
        raise RunnerError("--models is for a fill-card spec")
    recorded_models = None
    if models is not None:
        _chosen, recorded_models = resolve_models(spec, models)
    if pilot and spec.get("id") != "E4b":
        from interlock.harness.e4b import E4bError

        raise E4bError("--pilot is only for E4b")
    blob = require_unmodified(repo, spec_path)
    if spec.get("id") == "E4b":
        from interlock.harness.e4b import E4bError, dry_text, resolve_pairs

        recorded = None
        if pairs is not None:
            if pilot:
                raise E4bError("--pairs is for the block, not the pilot")
            _chosen, recorded = resolve_pairs(spec, pairs)
        return dry_text(spec, blob, pilot=pilot, pairs=recorded)
    lines = [
        f"interlock dry-run: {spec['id']}",
        f"spec blob: {blob}",
        f"procedure: {spec['procedure']}",
    ]
    if spec["procedure"] == "interleaved":
        policy = warmup_policy(spec)
        if policy is None:
            slots = _slots(spec)
            lines.append(f"slots: {len(slots)}")
        else:
            lines.append(
                f"warmups: until two consecutive calls agree within {policy['tolerance']} "
                f"on {spec['decision_metric']}, at most {policy['max']}; "
                "warm-up calls stay out of the percentiles"
            )
        lines.append(f"order: {spec.get('order', 'interleaved')}")
        if spec.get("order") == "blocked":
            lines.append("blocks: " + ", ".join(f"{block['id']} ({len(block['arms'])} arms)" for block in spec["blocks"]))
            lines.append(
                f"between blocks: unload, then used RAM within {RELEASE_TOLERANCE_BYTES // 2**30} GiB of the "
                f"run's start (waits up to {RELEASE_WAIT_S:g} s, else the run stops)"
            )
            lines.append(f"a build is refused when used RAM plus its build step would pass {BUILD_GUARD_PCT}% of total")
            drift = spec.get("drift_block")
            if drift:
                lines.append(
                    f"drift block {drift['id']}: {len(drift['arms'])} arms of {drift['repeat_of']} repeated "
                    f"{drift['repeats']} times, descriptive rows kept out of the table"
                )
        lines.append(f"compile_cache: {spec['compile_cache']}")
        lines.append(
            f"compile ceiling: {_compile_ceiling(spec):g} s; the run stops at {RAM_STOP_PCT:g}% system RAM"
        )
        lines.append("cold is measured once per arm and kept out of the warm percentiles")
        lines.append(f"band: {spec['band']}")
        lines.extend(_storage_lines(spec))
    elif spec["procedure"] == "fill-card":
        lines.append(f"abort_mib: {spec['abort_mib']}")
        lines.append(f"ceiling_mib: {spec['ceiling_mib']}")
        lines.append(f"models: {len(spec['models'])}")
        if recorded_models is not None:
            lines.append("models measured: " + ", ".join(recorded_models))
        lines.append(f"spill_shared_mib: {spec['spill_shared_mib']}")
        lines.append(f"spill_prompt_ratio: {spec['spill_prompt_ratio']}")
        lines.append("the sampler aborts at abort_mib and does not wait for the watchdog")
        lines.append("quant is read from /api/show at run time")
        lines.append("the search cap is the trained context length; no window above it is loaded")
        lines.append("a trained window that is not resident is halved to one grain, then bisected upward")
        lines.append("the reference is the floor window, one grain, loaded and confirmed first for every model")
        lines.append("the lowest resident window is superseded")
        lines.append(f"adapter_match_tolerance_mib: {spec['adapter_match_tolerance_mib']}")
        lines.append("the tolerance is a matching allowance and is not the spill threshold")
        lines.append("the 5090 adapter is pinned once at run start within that tolerance of nvidia-smi memory.used")
        lines.append("zero or several instances within tolerance refuses before any load")
        lines.append("later shared and dedicated reads use the pinned instance")
        lines.append("that floor confirm is the reference for the shared gate and the prompt-rate gate")
        lines.append("a window spills when its shared reading exceeds that floor confirm by more than spill_shared_mib")
        lines.append("that sentence is superseded when the peak dedicated is below shared_gate_from_mib")
        lines.append(f"shared_gate_from_mib: {spec['shared_gate_from_mib']}")
        lines.append("the shared gate counts only under capacity pressure")
        lines.append("the shared gate applies at or above shared_gate_from_mib")
        lines.append("below shared_gate_from_mib, shared growth is recorded and is not a spill")
        lines.append("the prompt-rate gate is a cliff against the next lower confirmed window")
        lines.append("a confirm spills when its prompt rate is under spill_prompt_ratio times that neighbour")
        lines.append("the floor confirm stays the reference for the CUDA_Host check")
        lines.append("a missing shared reading during a window is still a spill")
        lines.append("the floor confirm is checked against the idle shared reading and the CUDA_Host buffer sum")
        lines.append("a baseline that exceeds that sum by more than 256 MiB is inconclusive and is not confirmed")
        lines.append("CUDA_Host model, compute, and output sizes come from the first-load lines")
        lines.append("a missing CUDA_Host line refuses after the first load")
        lines.append("the dedicated delta is recorded and is not a gate")
        lines.append("preflight samples nvidia-smi, both counters, the server log, the ggml-cuda sha, git, and driver CUDA before any load")
        lines.append("driver CUDA accepts CUDA Version and CUDA UMD Version")
        lines.append("a missing log or dll refuses at preflight")
        lines.append("the log is INTERLOCK_OLLAMA_LOG, else the 11434 listener parent's redirect")
        lines.append("the tray app log is never a fallback")
        lines.append("an unresolved log refuses at preflight")
        lines.append("the latest server-config line must name OLLAMA_HOST 127.0.0.1:11434")
        lines.append("the exe is INTERLOCK_OLLAMA_EXE, else the 11434 listener")
        lines.append("the server-config line is recorded as it is, and an empty KV cache type is unset")
        lines.append("flash attention and the KV cache type are gated on the first-load lines")
        lines.append("a missing load line refuses before any measured window")
        lines.append("the runner pid is llama-server.exe and stays record-only")
        lines.append("default context and the backend library come from the server log")
        lines.append("driver CUDA is the nvidia-smi header")
        lines.append("confirmation prefix counted at run time with prompt_eval_count")
        lines.append("the final prefix is counted again from a cold load")
        lines.append("an abort closes the request, then unloads, then reads the card again")
        lines.append("the next load waits until the abort has finished")
        lines.append(
            "an unload that does not finish within 60 seconds stops the run; the model may still be resident"
        )
        lines.append("closing the request does not return the memory a load already allocated")
        lines.append("a window predicted to reach abort_mib is predicted_over and is not loaded")
        lines.append("one measured window bounds the next context from that window's share")
        lines.append("after two resident loads the next peak is predicted from their measured slope")
        lines.append("predicted_over applies when that slope reaches abort_mib")
        lines.append("limit is predicted when predicted_over stopped the search")
        lines.append("a measured abort on the confirm walk is margin")
        lines.append(f"settle_tolerance_mib: {spec['settle_tolerance_mib']}")
        lines.append(f"settle_timeout_s: {spec['settle_timeout_s']}")
        lines.append(
            "before a measured load or confirm, the pinned adapter's dedicated reading "
            "settles within settle_tolerance_mib of the expected pre-call level"
        )
        lines.append("between models and after an unload that level is the idle baseline")
        lines.append("within a model the model stays resident, so that level is the resident floor")
        lines.append(
            "a card that does not settle within settle_timeout_s refuses that call and the measurement is not kept"
        )
        lines.append("a second window with no bound stops the row")
        lines.append("one model at a time; unloaded between models; rest is not taken here")
        lines.append("the ceiling is never loaded")
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
    if _prompt_built_at_runtime(spec):
        lines.append(
            "prompt built at run time by prompt_eval_count; "
            "the final prefix is counted again from a cold load"
        )
    lines.append("a device run would require: " + (", ".join(flags) if flags else "no extra flag"))
    lines.extend(restart_policy_lines())
    lines.append("heartbeat not read; adapters not called")
    lines.append(devices_not_started())
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
    wall=None,
    watch: WatchHooks | None = None,
    marker: RunMarker | None = None,
    supervisor=None,
) -> list[dict]:
    """Run a committed spec.

    Each arm is written into the receipt when its slots finish. The receipt
    stays partial until the run ends. A 60 s hang becomes one ``hang`` row
    and stops the run. A device loss becomes one ``device_lost`` row, that
    device is marked off, and the other devices carry on. A step reported as
    not run writes no row; it is listed in the receipt notes. Neither outcome
    is retried.

    An NPU that refuses its compiled binary at execute becomes one
    ``native_binary_rejected`` row scoped to model x device x static shape
    (ruling #33). The shape keeps its compile time as evidence, arms with the
    identical key are marked not run, and every other arm carries on. On any
    abort the slots already measured are flushed as rows marked
    ``aborted: true`` with the counts actually completed; the table import
    refuses them and ``gaps.md`` records the aborted run.

    An NPU compiler that refuses the graph inside ``compile_model`` with
    ``[NPU_VCL] Compiler returned msg`` and ``0x78000007`` becomes one
    ``compile_rejected`` row (ruling on #36), scoped the same way and never
    retried. Every row records ``settings.run_window`` (the run's start and
    latest end, from ``wall``) so a later run can read how long this one
    lasted. The first kept sample after an Ollama restart is marked
    ``cold_after_restart`` and left out of the warm percentiles. A sample
    kept on a retry is marked ``retried``.

    A watcher thread guards the run from outside the main thread, which a
    native call can hold for an hour. A compile that runs past
    ``compile_ceiling_s`` (default 15 min) becomes one ``compile_timeout`` row;
    system RAM at 88% aborts the run; either way the receipt is flushed and the
    process ends with a distinctive exit code (see ``interlock.harness.watch``).
    A run-start marker, ``receipts/<run_id>.run.json``, makes a run killed from
    outside visible: unclosed means killed.
    """

    _check_run_id(run_id)
    blob = require_unmodified(repo, spec_path)
    spec = load_spec(spec_path)
    if spec["procedure"] != "interleaved":
        raise RunnerError("run_interleaved only runs an interleaved spec")
    spec_rel = spec_path.resolve().relative_to(repo.resolve()).as_posix()
    _guard_flags(spec, grant, npu_session, cpu_window)
    arms = {arm["id"]: arm for arm in spec["arms"]}
    blocked = spec.get("order") == "blocked"
    policy = warmup_policy(spec)
    if blocked:
        arms.update(_drift_arms(spec))
    if policy is None:
        if blocked:
            plan = _block_plan(spec)
        else:
            plan = [(None, [], _slots(spec))]
        items = [(slot, block_id) for block_id, _ids, block_slots in plan for slot in block_slots]
        stream = SlotStream(items)
        block_arms = {block_id: ids for block_id, ids, _slots_of in plan}
    else:
        if blocked:
            blocks = [
                BlockPlan(block["id"], list(block["arms"]), False, spec["repeats"]) for block in spec["blocks"]
            ]
            drift = spec.get("drift_block")
            if drift:
                drift_ids = [_drift_id(arm_id) for arm_id in drift["arms"]]
                blocks.append(BlockPlan(drift["id"], drift_ids, True, int(drift["repeats"])))
        else:
            blocks = [BlockPlan(None, [arm["id"] for arm in spec["arms"]], False, spec["repeats"])]
        stream = SlotStream([], UntilSteady(blocks, policy["max"], policy["tolerance"]))
        block_arms = {block.id: list(block.arms) for block in blocks}
    block_of = {arm_id: block_id for block_id, ids in block_arms.items() for arm_id in ids}
    needed = stream.needed
    refused: dict[str, str] = {}
    stop_all = False
    baseline_used: int | None = None
    records: list[dict] = []
    seen: dict[str, int] = {}
    retry_counts: dict[tuple, int] = {}
    retried_keys: set[tuple] = set()
    finished: list[str] = []
    capability: dict[str, str] = {}
    off_devices: set[str] = set()
    skipped: list[str] = []
    executed_ok: set[tuple] = set()
    rejected_keys: set[tuple] = set()
    rejections: dict[str, dict] = {}
    aborted: dict | None = None
    current: Slot | None = None
    ceiling = _call_ceiling(spec)
    cold_calls: dict[str, float] = {}
    declined_ids: set[str] = set()
    captured = env if env is not None else environment()
    cache_report: dict = {"cleared": set(), "failures": {}}
    cache_dirs = _cache_dirs(spec, root, run_id, arms, cache_report["cleared"])
    prompt_cache: dict[str, tuple[str, int, str | None, int | None, int | None]] = {}
    removed: set[str] = set()
    imported = False
    wall = wall or time.time
    started_at = wall()
    hooks = watch or WatchHooks()
    tracker = CompileTracker(hooks.clock)
    state_lock = threading.RLock()  # the main loop's mutations and the watcher's flush
    halted_by_watch = False
    if marker is None:
        marker = RunMarker(
            root, run_id, spec_rel=spec_rel, spec_blob=blob, devices=_spec_devices(spec), start=_iso(started_at)
        )
    marker.entered = True
    marker.restart(_iso(started_at))  # before any adapter is touched
    start_state = _start_state(hooks, _uses_5090(spec))  # recorded once, before the first device contact
    decision_metric = spec.get("decision_metric")
    build_guard = BuildGuard(hooks) if blocked else None
    if blocked:
        baseline_used = read_bytes(hooks.used_sampler)

    def _run_notes() -> str:
        extra = "not run: " + ", ".join(skipped) if skipped else ""
        if notes and extra:
            return f"{notes}. {extra}"
        return notes or extra

    def _bound_settings(arm_id: str) -> dict:
        """The per-arm call bound actually applied (ruling #36a), on every row."""

        cold_call = cold_calls.get(arm_id)
        warm = min(ceiling, max(60.0, 3.0 * cold_call)) if cold_call is not None else ceiling
        return {"bound_s": {"cold": ceiling, "warm": warm}, "cold_call_s": cold_call}

    def _blocked_extra(arm_id: str) -> dict:
        """What a blocked run records on every row: the block and the build guard; a drift row also its ratio."""

        if not blocked:
            return {}
        extra: dict = {"blocked": {"block_id": block_of[arm_id], "build_guard_pct": BUILD_GUARD_PCT}}
        origin = arms[arm_id].get("drift_of")
        if origin is not None:
            now = _summary(records, arm_id).throughput
            before = _summary(records, origin).throughput if origin in finished else None
            extra["drift_of"] = origin
            extra["drift"] = {
                "of": origin,
                "warm_throughput": now,
                "original_warm_throughput": before,
                "ratio": now / before if now is not None and before else None,
            }
        return extra

    def _mark_descriptive(row: dict, arm_id: str) -> dict:
        if arms[arm_id].get("drift_of") is not None:
            row["descriptive"] = True
            validate_row(row)
        return row

    def _restart_marks(extra: dict, arm_id: str) -> None:
        """The kept sample after a restart is cold, and a retried sample says so."""

        samples = [row for row in records if row["step_id"] == arm_id]
        if any(row.get("cold_after_restart") for row in samples):
            extra["cold_after_restart"] = True
        if any(row.get("retried") for row in samples):
            extra["retried"] = True

    def _finished_row(arm_id: str, *, partial: bool, gone: set[str]) -> dict:
        extra = _extra_settings(spec, arms[arm_id], records, run_id, gone, cache_report)
        extra.update(_prompt_counts(arms[arm_id], prompt_cache))
        extra.update(_bound_settings(arm_id))
        extra.update(_blocked_extra(arm_id))
        _rerun_settings(extra, spec, arms[arm_id], records, stream)
        extra["cpu_percent_max"] = _cpu_percent_max(records, arm_id)
        _restart_marks(extra, arm_id)
        genai = _genai_summary(records, arm_id)
        if genai:
            extra["genai"] = genai
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
        return _mark_descriptive(row, arm_id)

    def _capability_row(arm_id: str, *, partial: bool, gone: set[str]) -> dict:
        extra = _extra_settings(spec, arms[arm_id], [], run_id, gone, cache_report)
        extra.update(_prompt_counts(arms[arm_id], prompt_cache))
        extra.update(_bound_settings(arm_id))
        extra.update(_blocked_extra(arm_id))
        extra["cpu_percent_max"] = _cpu_percent_max(records, arm_id)
        outcome = capability[arm_id]
        evidence = rejections.get(arm_id)
        compile_ms = None
        if evidence is not None and outcome == "compile_timeout":
            extra["compile_timeout"] = dict(evidence)
        elif evidence is not None:
            compile_ms = evidence.get("compile_ms")
            extra["rejection"] = {key: value for key, value in evidence.items() if key != "compile_ms"}
        row = device_lost_row(
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
            outcome=outcome,
            partial=partial,
            compile_ms=compile_ms,
        )
        return _mark_descriptive(row, arm_id)

    def _aborted_row(arm_id: str, *, gone: set[str]) -> dict:
        """The partial record of an arm the abort stopped mid-schedule."""

        arm = arms[arm_id]
        extra = _extra_settings(spec, arm, records, run_id, gone, cache_report)
        extra.update(_prompt_counts(arm, prompt_cache))
        extra.update(_bound_settings(arm_id))
        extra.update(_blocked_extra(arm_id))
        _rerun_settings(extra, spec, arm, records, stream)
        extra["cpu_percent_max"] = _cpu_percent_max(records, arm_id)
        _restart_marks(extra, arm_id)
        genai = _genai_summary(records, arm_id)
        if genai:
            extra["genai"] = genai
        cold_done = any(row["step_id"] == arm_id and row["phase"] == "cold" for row in records)
        warm_done = sum(1 for row in records if row["step_id"] == arm_id and row["phase"] == "repeat")
        extra["abortion"] = {
            "stop_step": aborted["step"],
            "error": aborted["error"],
            "cold_done": cold_done,
            "repeats_done": warm_done,
            "repeats_required": spec["repeats"],
        }
        progress = f"aborted: cold {'done' if cold_done else 'not done'}, warm repeats {warm_done} of {spec['repeats']}"
        base = _run_notes()
        row = measured_row(
            spec_id=spec["id"],
            arm=arm,
            summary=_summary(records, arm_id),
            repeats=spec["repeats"],
            date=date,
            spec_rel=spec_rel,
            run_id=run_id,
            grant=grant,
            env=captured,
            notes=f"{progress}. {base}" if base else progress,
            resolved=_resolved(records, arm_id),
            artifacts=_artifacts(records, arm_id),
            extra_settings=extra,
        )
        row["aborted"] = True
        validate_row(row)
        return _mark_descriptive(row, arm_id)

    def _rows(*, partial: bool, gone: set[str]) -> list[dict]:
        rows = [_finished_row(arm_id, partial=partial, gone=gone) for arm_id in finished]
        rows.extend(_capability_row(arm_id, partial=partial, gone=gone) for arm_id in capability)
        if aborted is not None and partial:
            for arm_id, arm in arms.items():
                if arm_id in finished or arm_id in capability or arm_id in refused:
                    continue
                in_progress = halted_by_watch and current is not None and current.step_id == arm_id
                # a stop between blocks leaves the later blocks' arms unfinished: they get rows too
                unreached = stop_all and arm.get("drift_of") is None
                if not in_progress and not unreached and not any(record["step_id"] == arm_id for record in records):
                    continue
                rows.append(_aborted_row(arm_id, gone=gone))
        window = {"start": _iso(started_at), "end": _iso(wall())}
        restarts = marker.restart_records()
        for row in rows:
            row["settings"]["run_window"] = window
            row["settings"]["start_state"] = start_state
            row["settings"]["ram_guard"] = guard.ram_guard()
            row["settings"]["compile_ceiling_s"] = guard.ceiling_s
            if restarts:
                row["settings"]["ollama_restarts"] = restarts
                validate_row(row)
        return rows

    def _flush_locked(*, gone: set[str] | None = None) -> None:
        """The flush body. The caller holds ``state_lock``, or is the watcher that could not get it."""

        used = removed if gone is None else gone
        guard.sample_ram()  # the peak is current at every flush
        rows = _rows(partial=True, gone=used)
        if rows:
            write_receipt(root, run_id, rows)
        marker.touch(_iso(wall()))

    def flush(*, gone: set[str] | None = None) -> None:
        with state_lock:
            _flush_locked(gone=gone)

    def _halt(error: str, outcome: str | None = None, *, operator: bool = False) -> None:
        """The watcher ends the run: record why, flush every row, bring the receipt into the table.

        Runs on the watcher thread while the main thread may be inside a
        native call. It waits for the state lock a few seconds, then goes on
        without it: a receipt that may be a row short beats none.
        """

        nonlocal aborted, halted_by_watch, stop_all
        held = state_lock.acquire(timeout=hooks.lock_wait_s)
        try:
            halted_by_watch = True
            if operator:
                stop_all = True  # every unfinished arm, started or not, gets an aborted row
            aborted = {"step": _label(current) if current is not None else "setup", "error": error}
            if outcome is not None and current is not None:
                _remember(arms[current.step_id], outcome)
            _flush_locked()  # never blocks on the lock: the watcher must reach exit_fn
            marker.close(_iso(wall()), error if operator else None)  # unclosed means only "killed from outside"
            if (root / "receipts" / f"{run_id}.jsonl").is_file():
                try:
                    announce_run(import_receipt(root, run_id), finished=False)
                except Exception:
                    pass  # the receipt is written; the table import is the later step
        finally:
            if held:
                state_lock.release()

    def _on_compile_timeout(info: dict) -> None:
        arm = arms[current.step_id] if current is not None else None
        if arm is not None:
            rejections[arm["id"]] = _compile_timeout_evidence(arm, info, run_id, captured)
        error = f"{info['site']} ran {info['elapsed_s']:.0f} s, over the {info['ceiling_s']:.0f} s compile ceiling"
        _halt(error, "compile_timeout")

    def _on_operator_stop(reason: str) -> None:
        _halt(f"stopped by operator: {reason}", operator=True)

    guard = RunWatch(
        tracker,
        hooks,
        ceiling_s=_compile_ceiling(spec),
        on_compile_timeout=_on_compile_timeout,
        on_ram_stop=_halt,
        stop_pct=RAM_STOP_PCT,
        stop_file=root / "receipts" / f"{run_id}{STOP_SUFFIX}",
        on_operator_stop=_on_operator_stop,
    )

    def _remember(arm: dict, outcome: str) -> None:
        """Record a hang or a device loss. Partial samples from this arm are discarded."""

        arm_id = arm["id"]
        records[:] = [row for row in records if row["step_id"] != arm_id]
        capability[arm_id] = outcome
        if outcome == "device_lost":
            off_devices.add(arm["device"])

    def _note(slot: Slot) -> None:
        skipped.append(_label(slot))

    def _refuse_block(block_id: str) -> None:
        """A build was refused for memory headroom: the block's unfinished arms are not run."""

        gone_ids = [
            arm_id
            for arm_id in block_arms[block_id]
            if arm_id not in finished and arm_id not in capability and arm_id not in refused
        ]
        for arm_id in gone_ids:
            refused[arm_id] = "memory headroom"
            skipped.append(f"{arm_id} (memory headroom)")
        records[:] = [row for row in records if row["step_id"] not in gone_ids]

    def _released() -> bool:
        """Used RAM back within the tolerance of the run's start, waiting up to RELEASE_WAIT_S."""

        if baseline_used is None:
            return True
        deadline = hooks.clock() + RELEASE_WAIT_S
        while True:
            used = read_bytes(hooks.used_sampler)
            if used is None or used <= baseline_used + RELEASE_TOLERANCE_BYTES:
                return True
            if hooks.clock() >= deadline:
                return False
            hooks.sleep(RELEASE_POLL_S)

    def _end_block(block_id: str) -> None:
        """Flush the block's rows, unload what it used, and prove the memory came back or stop the run."""

        nonlocal aborted, stop_all
        flush()
        used_arms: dict[str, dict] = {}
        for arm_id in block_arms[block_id]:
            used_arms[arm_id] = arms[arm_id]
            companion_id = arms[arm_id].get("companion")
            if companion_id:
                used_arms[companion_id] = arms[companion_id]
        _unload(adapters, used_arms)
        gc.collect()
        if _released():
            return
        error = f"memory not released after block {block_id}"
        with state_lock:
            aborted = {"step": f"block {block_id} end", "error": error}
            stop_all = True
            flush()
        raise RunnerError(error)

    def _record_restarts() -> None:
        """Copy committed restarts onto the marker, including one that then stops the run."""

        if supervisor is None:
            return
        reason = supervisor.restarts_disabled
        if reason:
            marker.note_restart_disabled(reason)
        have = marker.restart_records()
        for record in supervisor.restarts:
            data = record.as_dict()
            if data not in have:
                marker.note_restart(data)
                have.append(data)

    def _resume_after_repair(slot, block_id, arm, kind_name: str, error: str) -> None:
        """The step has ended. Restart, or stop without one. Never during the measurement."""

        nonlocal aborted
        try:
            decision = supervisor.after_failure(url_for(arm["device"]), model=arm["model"], kind=kind_name)
        except SupervisorStop as exc:
            _record_restarts()
            marker.note_failure(_label(slot), error)
            with state_lock:
                aborted = {"step": _label(slot), "error": str(exc)}
                flush()
            raise RunnerError(str(exc)) from exc
        _record_restarts()
        marker.note_failure(_label(slot), error)
        if decision.action != "restarted" or decision.record is None:
            with state_lock:
                aborted = {"step": _label(slot), "error": decision.note or error}
                flush()
            raise RunnerError(decision.note or error)
        key = (slot.step_id, slot.phase, slot.round_index)
        used = retry_counts.get(key, 0)
        if used < retry_budget(spec):
            retry_counts[key] = used + 1
            retried_keys.add(key)
            stream.queue.appendleft((slot, block_id))
            return
        stream.note(slot, None)

    for adapter in adapters.values():
        attach = getattr(adapter, "attach_compile_tracker", None)
        if callable(attach):
            attach(tracker)
        attach_build_guard = getattr(adapter, "attach_build_guard", None)
        if build_guard is not None and callable(attach_build_guard):
            attach_build_guard(build_guard)
    guard.start()
    active_block = None

    def _drop(slot: Slot) -> None:
        """A slot that did not run. A warm-up drop lets the rest of the round finish."""

        stream.note(slot, None)

    try:
        try:
            while True:
                item = stream.pop()
                if item is None:
                    break
                slot, block_id = item
                if block_id != active_block:
                    if active_block is not None:
                        _end_block(active_block)
                    active_block = block_id
                current = slot
                arm = arms[slot.step_id]
                if arm["id"] in refused:
                    _drop(slot)
                    continue
                if arm.get("drift_of") is not None and arm["drift_of"] not in finished:
                    if arm["id"] not in declined_ids:
                        declined_ids.add(arm["id"])
                        skipped.append(f"{arm['id']} (drift of {arm['drift_of']}, which did not finish)")
                    _drop(slot)
                    continue
                companion = _companion(arm, arms, slot)
                if arm.get("not_run"):
                    if arm["id"] not in declined_ids:
                        declined_ids.add(arm["id"])
                        skipped.append(f"{arm['id']} (the spec says not run: {arm['not_run']})")
                    _drop(slot)
                    continue
                if _shape_key(arm) in rejected_keys or (
                    companion is not None and _shape_key(companion) in rejected_keys
                ):
                    _note(slot)
                    _drop(slot)
                    continue
                if arm["device"] in off_devices or (
                    companion is not None and companion["device"] in off_devices
                ):
                    _note(slot)
                    _drop(slot)
                    continue
                started = False
                compile_s: float | None = None
                pending: tuple[str, str] | None = None
                epoch: int | None = None
                bound = _call_bound(slot, cold_calls.get(slot.step_id), ceiling)
                if supervisor is not None and arm.get("engine") == "ollama":
                    try:
                        ollama_url = url_for(arm["device"])
                    except AdapterError:
                        ollama_url = None
                    if ollama_url is not None:
                        supervisor.prime(ollama_url)
                        reason = supervisor.restarts_disabled
                        if reason:
                            marker.note_restart_disabled(reason)
                        epoch = supervisor.begin_measurement()
                if companion is not None:
                    _session(
                        adapters,
                        companion,
                        "start",
                        spec,
                        spec_rel,
                        cache_dirs.get(companion["id"]),
                        prompt_cache,
                        bound_s=_call_bound(slot, cold_calls.get(companion["id"]), ceiling),
                    )
                    started = True
                cpu_start = _cpu_sample()
                adapter = adapters[arm["engine"]]
                probe = slot_probe(hooks)
                if probe is not None:
                    adapter._slot_probe = probe
                    probe.open()
                try:
                    kind = "ok"
                    error = ""
                    try:
                        result = adapter.run(
                            _request(
                                arm,
                                spec,
                                spec_rel,
                                cache_dirs.get(arm["id"]),
                                adapters[arm["engine"]],
                                prompt_cache,
                                bound_s=bound,
                            )
                        )
                    except AdapterError as exc:
                        cpu_end = _cpu_sample()
                        error = str(exc)
                        if hang_error(error):
                            kind = "hang"
                        elif device_lost_error(error):
                            kind = "device_lost"
                        elif isinstance(exc, BuildRefused):
                            kind = "memory_headroom"
                        elif _is_compile_rejection(exc, error, arm):
                            kind = "compile_rejected"
                            compile_s = exc.failed_s
                        elif _is_rejection(exc, error, arm, executed_ok):
                            kind = "native_binary_rejected"
                            compile_s = exc.compile_s
                        else:
                            repair = _ollama_repair_kind(supervisor, arm, error)
                            if repair is None:
                                aborted = {"step": _label(slot), "error": error}
                                flush()
                                raise
                            kind = "repair"
                            pending = (repair, error)
                    else:
                        cpu_end = _cpu_sample()
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
                                repair = _ollama_repair_kind(supervisor, arm, error)
                                if repair is None:
                                    kind = "fail"
                                else:
                                    kind = "repair"
                                    pending = (repair, error)
                    with state_lock:
                        if kind != "ok" and kind != "repair":
                            stream.note(slot, None)
                        if kind == "hang":
                            aborted = {"step": _label(slot), "error": error}
                            _remember(arm, "hang")
                            flush()
                            raise RunnerError(f"step {slot.step_id} failed: {error}")
                        if kind == "device_lost":
                            _remember(arm, "device_lost")
                            flush()
                        elif kind == "memory_headroom":
                            _refuse_block(block_id)
                            flush()
                        elif kind == "compile_rejected":
                            rejected_keys.add(_shape_key(arm))
                            rejections[arm["id"]] = _compile_rejection_evidence(
                                arm, error, compile_s, run_id, captured
                            )
                            _remember(arm, "compile_rejected")
                            flush()
                        elif kind == "native_binary_rejected":
                            rejected_keys.add(_shape_key(arm))
                            rejections[arm["id"]] = _rejection_evidence(
                                arm, error, compile_s, cache_dirs, run_id, captured
                            )
                            _remember(arm, "native_binary_rejected")
                            flush()
                        elif kind == "not run":
                            off_devices.add(arm["device"])
                            _note(slot)
                        elif kind == "repair":
                            pass
                        elif kind == "fail":
                            aborted = {"step": _label(slot), "error": error}
                            flush()
                            raise RunnerError(f"step {slot.step_id} failed: {error}")
                        else:
                            sample = _record(
                                slot,
                                result,
                                bound,
                                cold_calls.get(arm["id"]),
                                {"slot_start": cpu_start, "slot_end": cpu_end},
                                _finish_slot_probe(probe, adapter),
                            )
                            if epoch is not None:
                                try:
                                    sample = supervisor.keep_sample(epoch, sample)
                                except SupervisorStop as exc:
                                    marker.note_failure(_label(slot), str(exc))
                                    aborted = {"step": _label(slot), "error": str(exc)}
                                    flush()
                                    raise RunnerError(str(exc)) from exc
                            if sample is None:
                                marker.note_failure(_label(slot), "a measurement crossed a restart")
                                stream.note(slot, None)
                                pending = None
                            else:
                                if supervisor is not None and supervisor.consume_cold_after_restart():
                                    sample["cold_after_restart"] = True
                                if (slot.step_id, slot.phase, slot.round_index) in retried_keys:
                                    sample["retried"] = True
                                records.append(sample)
                                if slot.phase == "cold" and result.latency_ms is not None:
                                    cold_calls[arm["id"]] = result.latency_ms / 1000
                                executed_ok.add(_shape_key(arm))
                                seen[slot.step_id] = seen.get(slot.step_id, 0) + 1
                                if slot.phase == "warmup":
                                    metric = call_metric(result, decision_metric)
                                    # A missing reading does not agree, and it does not drop the arm.
                                    stream.note(slot, metric if metric is not None else float("nan"))
                                else:
                                    stream.note(slot, None)
                                if (
                                    slot.step_id not in finished
                                    and seen[slot.step_id] == needed[slot.step_id]
                                    and not stream.accepts_more(slot.step_id)
                                    and not stream.was_dropped(slot.step_id)
                                ):
                                    finished.append(slot.step_id)
                                    flush()
                                    if decision_metric and _row_path(
                                        _finished_row(slot.step_id, partial=True, gone=removed), decision_metric
                                    ) is None:
                                        error = f"decision metric {decision_metric} missing on {slot.step_id}"
                                        aborted = {"step": _label(slot), "error": error}
                                        stop_all = True
                                        flush()
                                        raise RunnerError(error)
                finally:
                    if epoch is not None:
                        supervisor.end_measurement()
                    _finish_slot_probe(probe, adapter)
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
                if pending is not None:
                    _resume_after_repair(slot, block_id, arm, pending[0], pending[1])
        finally:
            _unload(adapters, arms)
            gone, failures = _drop_caches(cache_dirs)
            removed.update(gone)
            cache_report["failures"].update(failures)
        rows = _rows(partial=False, gone=removed)
        if rows:
            write_run(root, run_id, rows)
        imported = True
        announce_run(rows, finished=True)
        return rows
    except BaseException as exc:
        if aborted is None:
            aborted = {
                "step": _label(current) if current is not None else "setup",
                "error": str(exc) or type(exc).__name__,
            }
        raise
    finally:
        guard.stop()
        try:
            receipt = root / "receipts" / f"{run_id}.jsonl"
            if not imported and (receipt.is_file() or aborted is not None):
                flush(gone=removed)
                if receipt.is_file():
                    written = import_receipt(root, run_id)
                    announce_run(written, finished=False)
        finally:
            marker.close(_iso(wall()))


def _ollama_repair_kind(supervisor, arm: dict, error: str) -> str | None:
    """A transport, timeout, or 5xx on an Ollama arm. Other engines are unchanged."""

    if supervisor is None or arm.get("engine") != "ollama":
        return None
    return failure_kind(error)


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


def native_binary_rejected_error(error: str | None) -> bool:
    """The device refused the compiled binary at execute (ruling #33).

    The match is the Level Zero code ``ZE_RESULT_ERROR_INVALID_NATIVE_BINARY``
    or its number, ``0x7800000f``. A hang is not a rejection. Distinct from a
    ``device_lost`` driver reset: the device is healthy and other shapes run.
    """

    if not isinstance(error, str) or not error or hang_error(error):
        return False
    text = error.lower()
    return "ze_result_error_invalid_native_binary" in text or "0x7800000f" in text


CALL_CEILING_DEFAULT = 600.0


def _compile_ceiling(spec: dict) -> float:
    """The bound on one native compile (R&D ruling 2026-10-10): 15 minutes unless the spec says."""

    value = spec.get("compile_ceiling_s")
    return COMPILE_CEILING_S if value is None else float(value)


def _call_ceiling(spec: dict) -> float:
    """The per-call ceiling C (ruling #36a). The cold call gets the ceiling."""

    value = spec.get("call_ceiling_s")
    if value is None:
        return CALL_CEILING_DEFAULT
    return float(value)


def _call_bound(slot: Slot, cold_call_s: float | None, ceiling: float) -> float:
    """The slot's per-call bound: the ceiling for a cold, else
    ``max(60 s, 3 x the arm's own cold call)``, capped at the ceiling."""

    if slot.phase == "cold" or cold_call_s is None:
        return ceiling
    return min(ceiling, max(60.0, 3.0 * cold_call_s))


def _finish_slot_probe(probe, adapter) -> dict | None:
    """The slot's min/max/mean, after the final reading. A second call does not sample again."""

    if probe is None:
        return None
    summary = probe.close()
    if getattr(adapter, "_slot_probe", None) is probe:
        adapter._slot_probe = None
    return summary


def _cpu_sample():
    """The system CPU load and frequency at a slot edge. ``unrecorded``,
    never a failure, when psutil or a counter is missing."""

    if psutil is None:
        return "unrecorded"
    try:
        percent = psutil.cpu_percent(interval=0.2)
        freq = psutil.cpu_freq()
        mhz = getattr(freq, "current", None)
        return {
            "cpu_percent": percent,
            "cpu_freq_mhz": mhz if mhz is not None else "unrecorded",
        }
    except Exception:  # pragma: no cover
        return "unrecorded"


def _cpu_percent_max(records: list[dict], arm_id: str):
    """The max sampled CPU percent across an arm's slots, or ``unrecorded``."""

    values = []
    for row in records:
        if row["step_id"] != arm_id:
            continue
        cpu = row.get("cpu") or {}
        for side in ("slot_start", "slot_end"):
            sample = cpu.get(side)
            if isinstance(sample, dict) and isinstance(sample.get("cpu_percent"), (int, float)):
                values.append(sample["cpu_percent"])
    return max(values) if values else "unrecorded"


def _label(slot: Slot) -> str:
    if slot.phase == "cold":
        return f"{slot.step_id} cold"
    return f"{slot.step_id} {slot.phase} {slot.round_index}"


def _shape_key(arm: dict) -> tuple:
    """The capability scope: model x device x static shape (batch and length)."""

    return (arm["model"], arm["device"], arm.get("batch"), arm.get("static_shape"))


def _is_rejection(exc: BaseException, error: str, arm: dict, executed_ok: set[tuple]) -> bool:
    """The native-binary capability bound. All four conditions, or the error aborts.

    The device is the NPU; the adapter raised the execute-site subtype (the
    compile succeeded first); the error carries the signature; and this run
    has not already executed the same shape — a shape that ran and is then
    refused means the device changed state, which is an unknown error.
    """

    if arm["device"] != "NPU":
        return False
    if not isinstance(exc, NativeBinaryRejected):
        return False
    if not native_binary_rejected_error(error):
        return False
    return _shape_key(arm) not in executed_ok


def _is_compile_rejection(exc: BaseException, error: str, arm: dict) -> bool:
    """The compile-rejected capability bound (ruling on #36). All three, or the error aborts.

    The device is the NPU; the adapter raised the compile-site subtype (the
    error came out of ``compile_model``, not out of an execute); and the error
    carries both the compiler-message marker and ``0x78000007``.
    """

    if arm["device"] != "NPU":
        return False
    if not isinstance(exc, CompileRejected):
        return False
    return compile_rejected_error(error)


def compiler_message(error: str) -> str | None:
    """The compiler's own words: the text after ``Compiler returned msg:``, on one line."""

    _, marker, tail = error.partition(COMPILER_MESSAGE_MARKER)
    if not marker:
        return None
    parts = [line.strip() for line in tail.lstrip(": 	").splitlines()]
    text = " ".join(part for part in parts if part)
    return text or None


def _compile_rejection_evidence(
    arm: dict,
    error: str,
    failed_s: float | None,
    run_id: str,
    env: dict,
) -> dict:
    """What the compile-rejected row records: the compiler's message and code, the
    drivers, the time to failure and the run. There is no blob and no compile time."""

    return {
        "compiler_message": compiler_message(error),
        "code": COMPILE_REJECTED_CODE,
        "npu_driver": env.get("npu_firmware") or "unrecorded",
        "openvino": env.get("openvino") or "unrecorded",
        "time_to_failure_s": failed_s,
        "run_id": run_id,
        "shape": {"batch": arm.get("batch"), "static_shape": arm.get("static_shape")},
    }


def _compile_timeout_evidence(arm: dict, info: dict, run_id: str, env: dict) -> dict:
    """What the compile-timeout row records: how long, against what ceiling, how big the process grew."""

    peak = info.get("peak_working_set_gb")
    return {
        "elapsed_s": info["elapsed_s"],
        "ceiling_s": info["ceiling_s"],
        "peak_working_set_gb": None if peak is None else round(peak, 2),
        "npu_driver": env.get("npu_firmware") or "unrecorded",
        "openvino": env.get("openvino") or "unrecorded",
        "run_id": run_id,
        "shape": {"batch": arm.get("batch"), "static_shape": arm.get("static_shape")},
    }


def _rejection_evidence(
    arm: dict,
    error: str,
    compile_s: float | None,
    cache_dirs: dict[str, Path],
    run_id: str,
    env: dict,
) -> dict:
    """What the rejected row records, beyond the shape the arm already carries."""

    cache_dir = cache_dirs.get(arm["id"])
    blob_bytes = None
    cache_rel = None
    if cache_dir is not None and cache_dir.is_dir():
        blob_bytes = sum(path.stat().st_size for path in cache_dir.rglob("*") if path.is_file())
        cache_rel = f"compile-cache/{run_id}/{arm['id']}"
    match = re.search(r"0x[0-9a-fA-F]+", error)
    return {
        "error": error,
        "code": match.group(0) if match else None,
        "npu_driver": env.get("npu_firmware") or "unrecorded",
        "shape": {"batch": arm.get("batch"), "static_shape": arm.get("static_shape")},
        "compile_cache_dir": cache_rel,
        "blob_bytes": blob_bytes,
        "compile_ms": None if compile_s is None else compile_s * 1000,
    }


def _reported_outcome(result) -> str | None:
    """The adapter's own outcome, when it finished the step instead of failing it."""

    resolved = result.resolved
    if not isinstance(resolved, dict):
        return None
    outcome = resolved.get("outcome")
    if outcome in ("device_lost", "not run"):
        return outcome
    return None


def _slots(spec: dict, arm_ids: list[str] | None = None) -> list[Slot]:
    """The schedule. ``arm_ids`` restricts the interleave to a subset: a block's arms.

    A blocked spec's whole schedule is its blocks' schedules one after another.
    """

    order = spec.get("order", "interleaved")
    if order == "blocked" and arm_ids is None:
        return [slot for _block_id, _ids, block_slots in _block_plan(spec) for slot in block_slots]
    if order == "aba":
        return schedule_aba(spec["pairs"], warmups=spec["warmups"], repeats=spec["repeats"])
    ids = arm_ids if arm_ids is not None else [arm["id"] for arm in spec["arms"]]
    return schedule(ids, warmups=spec["warmups"], repeats=spec["repeats"])


def _drift_id(arm_id: str) -> str:
    return f"{arm_id}-drift"


def _drift_arms(spec: dict) -> dict[str, dict]:
    """The drift block's arms: the originals again under their own ids, each knowing what it repeats."""

    drift = spec.get("drift_block")
    if not drift:
        return {}
    by_id = {arm["id"]: arm for arm in spec["arms"]}
    return {
        _drift_id(arm_id): {**by_id[arm_id], "id": _drift_id(arm_id), "drift_of": arm_id}
        for arm_id in drift["arms"]
    }


def _block_plan(spec: dict) -> list[tuple[str, list[str], list[Slot]]]:
    """``(block id, arm ids, slots)`` for each block, then the drift block.

    A block is the ordinary schedule over its own arms (cold once per arm,
    warm-ups, repeats). The drift block runs the chosen arms' warm slots only: no
    cold slot, at least one warm-up (the first call after a rebuild is not a
    warm sample), ``repeats`` repeats.
    """

    plan = [(block["id"], list(block["arms"]), _slots(spec, list(block["arms"]))) for block in spec["blocks"]]
    drift = spec.get("drift_block")
    if drift:
        ids = [_drift_id(arm_id) for arm_id in drift["arms"]]
        warm = schedule(ids, warmups=max(1, spec["warmups"]), repeats=drift["repeats"])
        plan.append((drift["id"], ids, [slot for slot in warm if slot.phase != "cold"]))
    return plan


def _request(
    arm: dict,
    spec: dict,
    spec_rel: str,
    cache_dir: Path | None,
    adapter,
    prompt_cache: dict[str, tuple[str, int, str | None, int | None, int | None]],
    bound_s: float | None = None,
) -> StepRequest:
    settings = {key: arm.get(key) for key in PASSTHROUGH}
    if not settings.get("prompt"):
        prompt = _dataset_prompt(arm, spec, adapter, prompt_cache)
        if prompt is not None:
            settings["prompt"] = prompt
    settings["prefix_caching"] = arm.get("prefix_caching", spec.get("prefix_caching"))
    settings["spec"] = spec_rel
    settings["compile_cache"] = spec.get("compile_cache")
    if cache_dir is not None:
        settings["compile_cache_dir"] = str(cache_dir)
    if bound_s is not None:
        settings["call_timeout_s"] = bound_s
    return StepRequest(
        step_id=arm["id"],
        task=arm["task"],
        model=arm["model"],
        device=arm["device"],
        engine=arm["engine"],
        settings=settings,
    )


def _record(
    slot: Slot,
    result,
    bound_s: float | None = None,
    cold_call_s: float | None = None,
    cpu: dict | None = None,
    sample: dict | None = None,
) -> dict:
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
        "bound_s": bound_s,
        "cold_call_s": cold_call_s,
        "cpu": cpu,
        "sample": sample,
        "output": result.output,
        "resolved": result.resolved,
        "sensors": sensors,
    }


def _summary(records: list[dict], arm_id: str):
    cold = [row for row in records if row["step_id"] == arm_id and row["phase"] == "cold"]
    repeats = [
        row
        for row in records
        if row["step_id"] == arm_id and row["phase"] == "repeat" and not row.get("cold_after_restart")
    ]
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
        if row.get("bound_s") is not None:
            item["bound_s"] = row["bound_s"]
        if row.get("cpu") is not None:
            item["cpu"] = row["cpu"]
        if row.get("sample"):
            item["sample"] = row["sample"]
        if row["output"] is not None:
            item["output"] = row["output"]
        if row["sensors"]:
            item["sensors"] = row["sensors"]
        slots.append(item)
    if not slots:
        return None
    return {"slots": slots}


def _genai_summary(records: list[dict], arm_id: str) -> dict:
    """Medians of openvino-genai's own counters over an arm's repeat slots.

    A generate step reports ttft_ms, tpot_ms, input_tokens and generated_tokens
    under resolved["genai"] on every call. Without this they never reach the
    receipt (E3's TTFT and R_speed read them). Empty for any other task.
    """

    values: dict[str, list[float]] = {}
    for row in records:
        if row["step_id"] != arm_id or row["phase"] != "repeat" or not isinstance(row.get("resolved"), dict):
            continue
        genai = row["resolved"].get("genai")
        if not isinstance(genai, dict):
            continue
        for key in ("ttft_ms", "tpot_ms", "input_tokens", "generated_tokens"):
            value = genai.get(key)
            if isinstance(value, (int, float)):
                values.setdefault(key, []).append(float(value))
    out: dict = {}
    for key, series in values.items():
        series.sort()
        mid = len(series) // 2
        median = series[mid] if len(series) % 2 else (series[mid - 1] + series[mid]) / 2
        out[f"{key}_p50" if key.endswith("_ms") else key] = median
    if out:
        out["repeats_counted"] = max(len(series) for series in values.values())
    return out


def _rerun_settings(extra: dict, spec: dict, arm: dict, records: list[dict], stream: SlotStream) -> None:
    """Cache, mmap, cold copy and the warm-up trace. Arms that use none of these stay as they were."""

    uses = (
        isinstance(spec.get("warmups"), dict)
        or bool(arm.get("hf_cache"))
        or bool(arm.get("compile_config"))
        or arm.get("cold_copy") is True
    )
    if not uses:
        return
    extra["hf_cache"] = arm.get("hf_cache") or "archive"
    extra["cold_copy"] = arm.get("cold_copy") is True
    mmap = (arm.get("compile_config") or {}).get("ENABLE_MMAP") if isinstance(arm.get("compile_config"), dict) else None
    if isinstance(mmap, bool):
        extra["ENABLE_MMAP"] = mmap
    learned = _resolved(records, arm["id"])
    storage = learned.get("storage")
    if isinstance(storage, dict):
        extra["storage"] = {
            "drive": storage.get("drive") or "unrecorded",
            "bus": storage.get("bus") or "unrecorded",
            "media": storage.get("media") or "unrecorded",
        }
    drive = learned.get("cold_copy_drive")
    if isinstance(drive, str) and drive:
        extra["cold_copy_drive"] = drive
    if isinstance(spec.get("warmups"), dict):
        extra["warmup_trace"] = warmup_trace(records, arm["id"])
        extra["unsteady"] = not stream.is_steady(arm["id"])


def _extra_settings(
    spec: dict, arm: dict, records: list[dict], run_id: str, removed: set[str], cache_report: dict | None = None
) -> dict:
    settings = {
        "compile_cache": spec.get("compile_cache"),
    }
    if arm.get("task") == "generate" and arm.get("engine") == "openvino":
        learned = _resolved(records, arm["id"]).get("prefix_caching")
        settings["prefix_caching"] = learned if learned is not None else "library default (unknown)"
    if spec.get("compile_cache") == "fresh":
        settings["compile_cache_dir"] = f"compile-cache/{run_id}/{arm['id']}"
        settings["compile_cache_deleted"] = arm["id"] in removed
        if cache_report is not None:
            settings["compile_cache_cleared_at_start"] = arm["id"] in cache_report["cleared"]
            failures = cache_report["failures"].get(arm["id"])
            if failures:
                settings["compile_cache_delete_failures"] = failures
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
    prompt_cache: dict[str, tuple[str, int, str | None, int | None, int | None]],
    bound_s: float | None = None,
) -> None:
    adapter = adapters[arm["engine"]]
    method = getattr(adapter, f"{action}_session", None)
    if not callable(method):
        raise RunnerError(
            f"{arm['engine']} has no {action}_session. "
            "The harness owns companion sessions and does not fake one."
        )
    method(_request(arm, spec, spec_rel, cache_dir, adapter, prompt_cache, bound_s=bound_s))


def _count_settings(arm: dict) -> dict:
    """Load options the token count shares with the measurement.

    ``num_predict`` is not copied: the count asks for a single completion token
    (``COUNT_NUM_PREDICT``), never 0, which Ollama reads as no limit.
    ``num_ctx`` is copied even when it is null, so both calls resolve the same
    context from ``/api/show``.
    """

    return {
        key: arm[key]
        for key in ("thinking_level", "num_ctx", "keep_alive", "temperature", "shift", "truncate")
        if key in arm
    }


def _keep_alive_unloads(value: object) -> bool:
    if value == 0:
        return True
    return isinstance(value, str) and value.strip().lower() in {"0", "0s", "0m", "0h"}


def _count_leaves_loaded(arm: dict) -> bool:
    """True when the count's own keep_alive would leave the model resident.

    The Ollama default is five minutes. An explicit 0 unloads at the end of
    each count call, and the runner does not send a second unload.
    """

    if arm.get("keep_alive") is None:
        return True
    return not _keep_alive_unloads(arm.get("keep_alive"))


def _ollama_prompt_method(fit: str) -> str:
    if fit == "exact":
        return "prompt_eval_count"
    return "prompt_eval_count (shortest prefix at or above)"


def _dataset_prompt(arm: dict, spec: dict, adapter, prompt_cache: dict[str, tuple[str, int, str | None, int | None, int | None]]) -> str | None:
    """The spec's prefix: repeat the seed until the model tokenizer reaches the target.

    One prefix per model, shared by every device of that model. A spec with a
    seed and no ``target_tokens`` still passes the seed through. A target with
    no tokenizer count is refused, so the raw seed is not measured by mistake.

    An Ollama arm counts with the served model's ``prompt_eval_count`` and
    trims on word boundaries. That count is device contact. It runs only from
    a granted run, after the heartbeat and the empty-card check, because this
    function is reached from the measurement loop. The dry-run never calls it.
    A resident model can report only the tokens it evaluated, so a later
    prefix of an earlier one can undercount. After the prefix is chosen, the
    model is unloaded when the search would leave it loaded, and the final
    prefix is counted once more from that cold load. A cold count that differs
    from the search refuses the arm before any measurement. Both counts are
    recorded. The measurement load stays the spec's own: a count that would
    leave the model resident is unloaded again before it.
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
        try:
            if arm["engine"] == "ollama":
                count_timeout = roster_timeout(adapter, spec)

                def count(text: str, model: str = model) -> int:
                    return counter(
                        model,
                        text,
                        settings=_count_settings(arm),
                        device=arm["device"],
                        timeout=count_timeout,
                    )

                text, tokens, fit = fit_prompt_words(seed, target, count)
                if _count_leaves_loaded(arm):
                    adapter.unload(model, arm["device"])
                cold_tokens = count(text)
                if cold_tokens != tokens:
                    if _count_leaves_loaded(arm):
                        adapter.unload(model, arm["device"])
                    raise RunnerError(
                        f"{arm['id']}: Ollama token count refused: "
                        f"the search counted {tokens} and a cold load counted {cold_tokens}"
                    )
                cached = (text, cold_tokens, _ollama_prompt_method(fit), tokens, cold_tokens)
                prompt_cache[model] = cached
                if _count_leaves_loaded(arm):
                    adapter.unload(model, arm["device"])
            else:

                def count(text: str, model: str = model) -> int:
                    return counter(model, text)

                text = fit_prompt(seed, target, count)
                cached = (text, count(text), None, None, None)
                prompt_cache[model] = cached
        except SearchError as exc:
            raise RunnerError(str(exc)) from exc
        except AdapterError as exc:
            if str(exc) == COUNT_TIMEOUT_NOTE:
                raise RunnerError(f"{arm['id']}: {COUNT_TIMEOUT_NOTE}") from exc
            raise RunnerError(f"{arm['id']}: {arm['engine']} token count failed: {exc}") from exc
    return cached[0]


def _prompt_counts(arm: dict, prompt_cache: dict[str, tuple[str, int, str | None, int | None, int | None]]) -> dict:
    cached = prompt_cache.get(arm["model"])
    if cached is None:
        return {}
    _text, prompt_tokens, method, search_tokens, cold_tokens = cached
    counts = {"prompt_tokens": prompt_tokens}
    if method:
        counts["prompt_token_method"] = method
    if search_tokens is not None:
        counts["prompt_token_search"] = search_tokens
    if cold_tokens is not None:
        counts["prompt_token_cold"] = cold_tokens
    length = _static_length(arm.get("static_shape"))
    if length is not None:
        counts["truncated_tokens"] = max(0, prompt_tokens - length)
    return counts


def _prompt_built_at_runtime(spec: dict) -> bool:
    """True when an Ollama arm's prefix is counted on the device during the run."""

    dataset = spec.get("dataset")
    if not isinstance(dataset, dict):
        return False
    target = dataset.get("target_tokens")
    if isinstance(target, bool) or not isinstance(target, int):
        return False
    arms = spec.get("arms")
    if not isinstance(arms, list):
        return False
    return any(isinstance(arm, dict) and arm.get("engine") == "ollama" for arm in arms)


def _static_length(shape: object) -> int | None:
    if not isinstance(shape, str) or "x" not in shape.lower():
        return None
    length = shape.lower().split("x", 1)[1]
    if not length.isdigit():
        return None
    value = int(length)
    return value if value > 0 else None


def _cache_dirs(
    spec: dict, root: Path, run_id: str, arms: dict, cleared: set[str] | None = None
) -> dict[str, Path]:
    """The per-arm cache directories of a ``compile_cache: fresh`` spec, each EMPTY.

    A directory left from an earlier try of the same run id is deleted first
    and the arm is added to ``cleared``. One that cannot be emptied refuses the
    run here, before any device is touched: a "fresh" arm on a warm cache would
    measure a reload.
    """

    if spec.get("compile_cache") != "fresh":
        return {}
    dirs = {}
    for arm_id in arms:
        path = root / "compile-cache" / run_id / arm_id
        if path.is_dir() and any(path.iterdir()):
            failures = _rmtree_clearing_readonly(path)
            if path.is_dir() and any(path.iterdir()):
                detail = f": {failures[0]}" if failures else ""
                raise GuardError(
                    f"REFUSED: compile-cache/{run_id}/{arm_id} is not empty and could not be cleared{detail}. "
                    "A fresh-cache arm must start on an empty directory; delete it and run again."
                )
            if cleared is not None:
                cleared.add(arm_id)
        path.mkdir(parents=True, exist_ok=True)
        dirs[arm_id] = path
    return dirs


DELETE_FAILURES_KEPT = 10


def _rmtree_clearing_readonly(path: Path) -> list[str]:
    """Delete a directory tree. A read-only file (every OpenVINO cache blob on
    Windows is one) has its attribute cleared and the delete retried. What still
    cannot be deleted is returned as short lines, relative to ``path``, and never raised."""

    failures: list[str] = []

    def onexc(func, target, exc) -> None:
        try:
            os.chmod(target, stat.S_IWRITE)
            func(target)
        except OSError as err:
            try:
                name = Path(target).relative_to(path).as_posix()
            except ValueError:
                name = Path(target).name
            failures.append(f"{name}: {type(err).__name__}: {err.strerror or 'failed'}")

    shutil.rmtree(path, onexc=onexc)
    if len(failures) > DELETE_FAILURES_KEPT:
        extra = len(failures) - DELETE_FAILURES_KEPT
        failures = failures[:DELETE_FAILURES_KEPT] + [f"and {extra} more"]
    return failures


def _drop_caches(cache_dirs: dict[str, Path]) -> tuple[set[str], dict[str, list[str]]]:
    """Delete each arm's cache. Returns the arms now gone and, per arm, why a delete failed."""

    gone = set()
    failures: dict[str, list[str]] = {}
    for arm_id, path in cache_dirs.items():
        problems = _rmtree_clearing_readonly(path) if path.exists() else []
        if not path.exists():
            gone.add(arm_id)
        elif not problems:
            problems = ["the directory is still there after the delete"]
        if problems:
            failures[arm_id] = problems
    return gone, failures


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
    """The dry-run's closing line. The real run path is :func:`run_spec`."""

    return "devices not started (--dry-run)"


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

HEARTBEAT_DEFAULT = Path("E:/AI/training/_watchdog_HEARTBEAT")
HEARTBEAT_ENV = "WATCHDOG_HEARTBEAT"


def heartbeat_path(override: Path | None = None) -> Path:
    """The VRAM watchdog heartbeat file. The WATCHDOG_HEARTBEAT environment
    variable overrides the rig default, the same variable chain_step.sh honours."""

    if override is not None:
        return override
    return Path(os.environ.get(HEARTBEAT_ENV, str(HEARTBEAT_DEFAULT)))


def real_adapters(spec: dict) -> dict:
    """One registry adapter per engine the spec names. Construction touches no device."""

    return {engine: for_engine(engine) for engine in sorted({arm["engine"] for arm in spec["arms"]})}


def _refuse_repo_interpreter(root: Path, python_path: Path) -> None:
    """Refuse an interpreter inside the checkout.

    uv's fallback is an in-repo ``.venv``. The pinned environment lives outside
    the repo; a run from the shadow .venv is exactly where unpinned packages
    would reach a receipt.
    """

    try:
        python_path.resolve().relative_to(root.resolve())
    except ValueError:
        return
    raise GuardError(
        "REFUSED: the interpreter is inside the repository checkout. Device "
        "runs use the pinned environment outside the repo; run with "
        "UV_PROJECT_ENVIRONMENT set, not the in-repo .venv."
    )


def _uses_5090(spec: dict) -> bool:
    return any(
        arm["device"] == CARD_5090 or arm.get("companion_device") == CARD_5090
        for arm in spec["arms"]
    )


def _card_probe(adapters: dict, spec: dict):
    """The loaded-models probe of the adapter serving the 5090 arms."""

    engine = next(
        arm["engine"]
        for arm in spec["arms"]
        if arm["device"] == CARD_5090 or arm.get("companion_device") == CARD_5090
    )
    probe = getattr(adapters[engine], "loaded_models", None)
    if not callable(probe):
        raise GuardError(f"REFUSED: the {engine} adapter cannot report the card state")
    return probe


def _card_guards(probe, path: Path, now: float) -> None:
    """Heartbeat, then empty card. Both are read fresh on every call."""

    require_heartbeat(path, now)
    require_card_empty(probe())


def _iso(epoch: float) -> str:
    return datetime.datetime.fromtimestamp(epoch, tz=datetime.timezone.utc).isoformat(timespec="seconds")


STOP_SUFFIX = ".stop"


def _start_state(hooks, uses_5090: bool) -> dict:
    """Device state at run start for every row's settings. Recorded, never gated."""

    try:
        state = hooks.state_sampler(uses_5090)
        return state if isinstance(state, dict) else "unrecorded"
    except Exception:
        return "unrecorded"


def _row_path(row: dict, path: str):
    """A dotted field path in a row ("latency.throughput"), None when absent or null."""

    value = row
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _spec_devices(spec: dict) -> set[str]:
    devices = {arm["device"] for arm in spec["arms"]}
    devices.update(arm["companion_device"] for arm in spec["arms"] if arm.get("companion_device"))
    return devices


def run_spec(
    spec_path: Path,
    *,
    root: Path,
    grant: str | None = None,
    npu_session: str | None = None,
    cpu_window: str | None = None,
    run_id: str | None = None,
    date: str | None = None,
    notes: str = "",
    adapters: dict | None = None,
    heartbeat: Path | None = None,
    wall=None,
    python_path: str | None = None,
    watch: WatchHooks | None = None,
    seams=None,
    supervisor=None,
    pilot: bool = False,
    e4b_seams=None,
    quarantine: Path | None = None,
    pairs: list[str] | None = None,
    models: list[str] | None = None,
) -> list[dict]:
    """Run a committed spec. Every refusal lands before device contact.

    The order is: interpreter, spec load, token flags, the committed blob,
    the receipt-name collision, then the procedure. An interleaved spec builds
    adapters and checks the card once. A fill-card spec checks the grant, the
    heartbeat and an empty card before each model, and it does not read
    ``arms``. ``run_interleaved`` re-checks the blob and flags and owns the
    measurement; a completed interleaved run appends its rows to the routing
    table. A fill-card run writes its receipt and does not import it.

    The run-start marker is written right after the receipt-name check, before
    the first adapter or card probe; a refusal before the run takes over
    removes it, so a run that never started cannot owe a rest.
    """

    _refuse_repo_interpreter(root, Path(python_path or sys.executable))
    spec = load_spec(spec_path)
    recorded_models = None
    if models is not None:
        if spec.get("procedure") != "fill-card":
            raise RunnerError("--models is for a fill-card spec")
        _chosen, recorded_models = resolve_models(spec, models)
    if spec.get("id") == "E4b":
        from interlock.harness.e4b import run_from_cli

        return run_from_cli(
            spec,
            spec_path=spec_path,
            root=root,
            repo=root,
            grant=grant,
            npu_session=npu_session,
            run_id=run_id,
            pilot=pilot,
            seams=e4b_seams,
            notes=notes,
            quarantine=quarantine,
            pairs=pairs,
        )
    if pilot:
        from interlock.harness.e4b import E4bError

        raise E4bError("--pilot is only for E4b")
    check_flags(spec, grant, npu_session, cpu_window)
    blob = require_unmodified(root, spec_path)
    stamp = date or datetime.date.today().isoformat()
    name = run_id or f"{spec['id'].lower()}-{stamp}"
    receipt = root / "receipts" / f"{name}.jsonl"
    if receipt.exists():
        raise GuardError(f"REFUSED: receipts/{name}.jsonl already exists; pass --run-id for another name")
    wall = wall or time.time
    if supervisor is not None:
        if grant and not getattr(supervisor, "grant", None):
            supervisor.grant = grant
        if getattr(supervisor, "heartbeat", None) is None and heartbeat is not None:
            supervisor.heartbeat = heartbeat
    spec_rel = spec_path.resolve().relative_to(root.resolve()).as_posix()
    devices = {"5090"} if spec["procedure"] == "fill-card" else _spec_devices(spec)
    marker = RunMarker(
        root,
        name,
        spec_rel=spec_rel,
        spec_blob=blob,
        devices=devices,
        start=_iso(wall()),
    )
    marker.write()
    try:
        if spec["procedure"] == "fill-card":
            injected = adapters is not None
            if adapters is None:
                adapters = {"ollama": for_engine("ollama")}
            if seams is not None:
                chosen = seams
            elif injected:
                chosen = quiet_fill_seams()
            else:
                chosen = real_fill_seams(adapters["ollama"])
            return run_fill_card(
                spec,
                adapters=adapters,
                seams=chosen,
                grant=grant,
                root=root,
                spec_rel=spec_rel,
                run_id=name,
                date=stamp,
                heartbeat=heartbeat_path(heartbeat),
                wall=wall,
                marker=marker,
                notes=notes,
                supervisor=supervisor,
                models=recorded_models,
            )
        if spec["procedure"] != "interleaved":
            raise GuardError(
                f"REFUSED: a {spec['procedure']!r} spec has no run path; this command runs interleaved specs"
            )
        if adapters is None:
            adapters = real_adapters(spec)
        if _uses_5090(spec):
            probe = _card_probe(adapters, spec)
            path = heartbeat_path(heartbeat)
            _card_guards(probe, path, wall())
        return run_interleaved(
            adapters,
            grant=grant,
            npu_session=npu_session,
            cpu_window=cpu_window,
            root=root,
            repo=root,
            spec_path=spec_path,
            run_id=name,
            date=stamp,
            notes=notes,
            wall=wall,
            watch=watch,
            marker=marker,
            supervisor=supervisor,
        )
    finally:
        if not marker.entered:
            marker.discard()
