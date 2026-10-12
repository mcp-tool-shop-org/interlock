"""Turn measured samples into routing rows.

Numbers come from the samples. Missing environment fields stay ``unrecorded``.
The row is validated before it is written.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from interlock.harness import environment as env_capture
from interlock.harness.environment import UNRECORDED
from interlock.harness.timing import Summary
from interlock.families import stamp_family
from interlock.importers.gaps import append_aborted_runs
from interlock.receipt import CAPABILITY_OUTCOMES, load_jsonl, validate_row

# The pre-registered parity guard. The first repeat must reach this cosine
# against the same model's CPU batch-1 arm at the same length.
PARITY_MIN = 0.999

DEVICE_LOST_NOTE = "this device can't do this setting"


def _blank_latency() -> dict:
    return {
        "cold_ms": None,
        "warm_p50_ms": None,
        "warm_p95_ms": None,
        "throughput": None,
        "throughput_unit": None,
        "compile_ms": None,
        "first_call_ms": None,
        "compile_warm_reload_ms": None,
    }


def environment(**overrides) -> dict:
    """The receipt environment. A passed field replaces the captured one.

    OpenVINO, the Intel package stack, the NPU driver and the Intel shared
    memory override are read when this process can see them. Power plan and ``ollama_ps`` stay ``unrecorded``.
    """

    base = {
        "nvidia_driver": UNRECORDED,
        "npu_firmware": env_capture.npu_driver_version(),
        "openvino": env_capture.dist_version("openvino"),
        "power_plan": UNRECORDED,
        "ollama_ps": UNRECORDED,
        "packages": env_capture.package_versions(),
        "cuda_backend": UNRECORDED,
        "igpu_shared_memory_override": env_capture.igpu_shared_memory_override(),
    }
    base.update(overrides)
    return base


def measured_row(
    *,
    spec_id: str,
    arm: dict,
    summary: Summary,
    repeats: int,
    date: str,
    spec_rel: str,
    run_id: str,
    grant: str | None,
    env: dict,
    notes: str,
    resolved: dict | None = None,
    artifacts: dict | None = None,
    extra_settings: dict | None = None,
) -> dict:
    learned = resolved or {}
    quant = learned.get("quant") or arm.get("quant") or UNRECORDED
    settings = {
        "thinking_level": arm.get("thinking_level"),
        "num_predict": arm.get("num_predict"),
        "num_ctx": learned.get("num_ctx", arm.get("num_ctx")),
        "batch": arm.get("batch"),
        "static_shape": arm.get("static_shape"),
        "precision": arm.get("precision"),
        "grant": grant,
        "environment": env,
    }
    if arm.get("keep_alive") is not None:
        settings["keep_alive"] = arm["keep_alive"]
    if extra_settings:
        settings.update(extra_settings)
    for key in ("prompt_tokens", "truncated_tokens", "token_type_ids_fed"):
        if learned.get(key) is not None:
            settings[key] = learned[key]
    if learned.get("dynamic_shape") is True or arm.get("dynamic_shape") is True:
        settings["dynamic_shape"] = True
    latency = {
        "cold_ms": summary.cold_ms,
        "warm_p50_ms": summary.warm_p50_ms,
        "warm_p95_ms": summary.warm_p95_ms,
        "throughput": summary.throughput,
        "throughput_unit": summary.throughput_unit,
        "compile_ms": summary.compile_ms,
        "first_call_ms": summary.first_call_ms,
    }
    reload_ms = learned.get("compile_warm_reload_ms")
    if reload_ms is not None:
        latency["compile_warm_reload_ms"] = reload_ms
    row = {
        "id": f"{spec_id}/{arm['id']}/{run_id}",
        "task": arm["task"],
        "model": {
            "repo_or_tag": arm["model"],
            "revision_or_digest": arm["revision_or_digest"],
            "quant": quant,
            "licence": arm["licence"],
        },
        "engine": {"name": arm["engine"], "version": arm.get("engine_version") or UNRECORDED},
        "device": arm["device"],
        "settings": settings,
        "quality": [],
        "latency": latency,
        "date": date,
        "repeats": repeats,
        "provisional": repeats < 2,
        "provenance": {"experiment_spec": spec_rel, "run_id": run_id},
        "notes": notes,
    }
    if artifacts:
        row["artifacts"] = artifacts
    stamp_family(row)
    validate_row(row)
    return row


def device_lost_row(
    *,
    spec_id: str,
    arm: dict,
    repeats: int,
    date: str,
    spec_rel: str,
    run_id: str,
    grant: str | None,
    env: dict,
    notes: str = "",
    extra_settings: dict | None = None,
    outcome: str = "device_lost",
    partial: bool = True,
    compile_ms: float | None = None,
) -> dict:
    """A capability row: a hang, a device loss, or a refused graph or binary.

    No timing, and no retry. ``outcome`` is ``device_lost``, ``hang``,
    ``native_binary_rejected``, ``compile_rejected`` or ``compile_timeout``. All
    five tell the table that this device cannot do the setting. ``partial`` stays on while the run is still open.
    A rejection carries ``compile_ms``: the compile finished, the execute did
    not. The other outcomes have no timing; a ``compile_rejected`` row's time
    to failure is under ``settings.rejection``.
    """

    text = DEVICE_LOST_NOTE if not notes else f"{DEVICE_LOST_NOTE}. {notes}"
    row = measured_row(
        spec_id=spec_id,
        arm=arm,
        summary=Summary(None, None, None, None, None),
        repeats=repeats,
        date=date,
        spec_rel=spec_rel,
        run_id=run_id,
        grant=grant,
        env=env,
        notes=text,
        extra_settings=extra_settings,
    )
    row["outcome"] = outcome
    if partial:
        row["partial"] = True
    row["provisional"] = True
    row["latency"] = _blank_latency()
    if compile_ms is not None:
        row["latency"]["compile_ms"] = compile_ms
    row.pop("artifacts", None)
    validate_row(row)
    return row


def for_table(row: dict) -> dict:
    """The table copy of a receipt row.

    A capability row (``device_lost``, ``hang``, ``native_binary_rejected``,
    ``compile_rejected``, ``compile_timeout``) cannot carry a speed. Timing and artifacts are stripped, and the note says
    the device cannot do the setting. A refused native binary keeps its
    compile time: the compile is evidence, the execute never happened.
    """

    if row.get("outcome") not in CAPABILITY_OUTCOMES:
        validate_row(row)
        return row
    cleaned = dict(row)
    cleaned.pop("artifacts", None)
    cleaned["latency"] = _blank_latency()
    if row.get("outcome") == "native_binary_rejected":
        cleaned["latency"]["compile_ms"] = (row.get("latency") or {}).get("compile_ms")
    notes = cleaned.get("notes") or ""
    if DEVICE_LOST_NOTE not in notes:
        cleaned["notes"] = DEVICE_LOST_NOTE if not notes else f"{DEVICE_LOST_NOTE}. {notes}"
    validate_row(cleaned)
    return cleaned


def write_receipt(root: Path, run_id: str, rows: list[dict]) -> Path:
    """Write ``receipts/<run_id>.jsonl`` without touching the table."""

    for row in rows:
        validate_row(row)
    receipts = root / "receipts"
    receipts.mkdir(exist_ok=True)
    path = receipts / f"{run_id}.jsonl"
    _dump(path, rows)
    return path


def import_receipt(root: Path, run_id: str) -> list[dict]:
    """Bring a run receipt into the routing table.

    ``device_lost``, ``hang``, ``native_binary_rejected`` and
    ``compile_rejected`` land as "this device can't do this setting", with the
    timing removed (a native-binary rejection keeps its compile time). A failed parity guard is recorded with its timing and
    is not a speed figure. CPU batch-1 references already in the table count,
    so a later receipt can be judged against them.

    Rows marked ``aborted: true`` never route: they stay in the receipt as
    the record, and each aborted run gets one line in ``table/gaps.md``. Rows
    marked ``descriptive: true`` (a drift block's repeats) stay in the receipt
    and never reach the table.
    """

    from interlock.harness.lifecycle import is_quarantined

    if is_quarantined(root, run_id):
        return []
    path = root / "receipts" / f"{run_id}.jsonl"
    loaded = load_jsonl(path)
    aborted = [row for row in loaded if row.get("aborted") is True]
    descriptive = [row for row in loaded if row.get("descriptive") is True and row.get("aborted") is not True]
    routable = [row for row in loaded if row.get("aborted") is not True and row.get("descriptive") is not True]
    rows = apply_parity(routable, references=_table_rows(root))
    rows = [for_table(row) for row in rows]
    _dump(path, rows + descriptive + aborted)
    _append_table(root, rows)
    if aborted:
        append_aborted_runs(root, aborted)
    return rows


def write_run(root: Path, run_id: str, rows: list[dict]) -> Path:
    """Write ``receipts/<run_id>.jsonl`` and append the rows to the table.

    A ``descriptive`` row is written to the receipt and left out of the table.
    """

    stamped = apply_parity(rows, references=_table_rows(root))
    path = write_receipt(root, run_id, [for_table(row) for row in stamped])
    _append_table(root, [row for row in load_jsonl(path) if row.get("descriptive") is not True])
    return path


def apply_parity(rows: list[dict], *, references: list[dict] | None = None) -> list[dict]:
    """Stamp ``parity_valid`` from the pre-registered guard.

    The first repeat vector is compared with the same model's CPU batch-1 arm
    at the same length. Cosine below 0.999 marks that arm only. A miss at the
    model's longest length does not mark a shorter length of the same device
    and model, and a miss at a shorter length does not mark a longer one that
    passed. Timing stays on the row. A row the guard cannot judge keeps any
    mark it already has.
    """

    pool = list(references or []) + list(rows)
    cpu: dict[tuple, list[float]] = {}
    for row in pool:
        if row.get("outcome") in CAPABILITY_OUTCOMES or row.get("device") != "CPU":
            continue
        parsed = _length_batch(row)
        vector = _first_repeat_vector(row)
        if parsed is None or parsed[0] != 1 or vector is None:
            continue
        cpu.setdefault((_model_key(row), parsed[1]), vector)
    for row in rows:
        if row.get("outcome") in CAPABILITY_OUTCOMES:
            continue
        parsed = _length_batch(row)
        vector = _first_repeat_vector(row)
        if parsed is None or vector is None:
            continue
        reference = cpu.get((_model_key(row), parsed[1]))
        if reference is None:
            continue
        score = _cosine(vector, reference)
        if score is None:
            continue
        row["parity_valid"] = score >= PARITY_MIN
    return rows


def _table_rows(root: Path) -> list[dict]:
    table = root / "table" / "routing.jsonl"
    if not table.is_file():
        return []
    return load_jsonl(table)


def _model_key(row: dict) -> tuple:
    model = row.get("model") or {}
    return (model.get("repo_or_tag"), model.get("revision_or_digest"))


def _length_batch(row: dict) -> tuple[int, int] | None:
    settings = row.get("settings") or {}
    shape = settings.get("static_shape")
    batch = settings.get("batch")
    if not isinstance(shape, str) or "x" not in shape.lower():
        return None
    left, right = shape.lower().split("x", 1)
    if not right.isdigit():
        return None
    if isinstance(batch, bool) or not isinstance(batch, int):
        if not left.isdigit():
            return None
        batch = int(left)
    length = int(right)
    if length < 1 or batch < 1:
        return None
    return batch, length


def _first_repeat_vector(row: dict) -> list[float] | None:
    artifacts = row.get("artifacts")
    slots = artifacts.get("slots") if isinstance(artifacts, dict) else None
    if not isinstance(slots, list):
        return None
    for slot in slots:
        if isinstance(slot, dict) and slot.get("phase") == "repeat":
            return _vector(slot.get("output"))
    return None


def _vector(value: object) -> list[float] | None:
    if not isinstance(value, list) or not value:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return None
    return [float(item) for item in value]


def _cosine(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right):
        return None
    dot = left_norm = right_norm = 0.0
    for one, other in zip(left, right):
        dot += one * other
        left_norm += one * one
        right_norm += other * other
    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    return dot / ((left_norm ** 0.5) * (right_norm ** 0.5))


def _append_table(root: Path, rows: list[dict]) -> None:
    table = root / "table" / "routing.jsonl"
    if not (table.is_file() or (root / "table").is_dir()):
        return
    table.parent.mkdir(exist_ok=True)
    existing = load_jsonl(table) if table.is_file() else []
    ids = {row["id"] for row in existing}
    for row in rows:
        if row["id"] in ids:
            raise ValueError(f"duplicate receipt id {row['id']}")
    merged = sorted(existing + rows, key=lambda item: item["id"])
    _dump(table, merged)


def _dump(path: Path, rows: list[dict]) -> None:
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    write_atomic(path, text)


def write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` all-or-nothing.

    The watcher can end the process mid-write, and the main thread may be
    writing the same receipt. Each writer fills its own temp file in the same
    folder (``<name>.<pid>.<thread>.tmp``) and ``os.replace`` swaps it in, so a
    reader sees the old file or the new one, never a mixture.
    """

    temporary = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
