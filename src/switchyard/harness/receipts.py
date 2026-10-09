"""Turn measured samples into routing rows.

Numbers come from the samples. Missing environment fields stay ``unrecorded``.
The row is validated before it is written.
"""

from __future__ import annotations

import json
from pathlib import Path

from switchyard.harness.timing import Summary
from switchyard.families import stamp_family
from switchyard.receipt import load_jsonl, validate_row

UNRECORDED = "unrecorded"
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
    }


def environment(**overrides) -> dict:
    base = {
        "nvidia_driver": UNRECORDED,
        "npu_firmware": UNRECORDED,
        "openvino": UNRECORDED,
        "power_plan": UNRECORDED,
        "ollama_ps": UNRECORDED,
        "packages": UNRECORDED,
        "cuda_backend": UNRECORDED,
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
        "latency": {
            "cold_ms": summary.cold_ms,
            "warm_p50_ms": summary.warm_p50_ms,
            "warm_p95_ms": summary.warm_p95_ms,
            "throughput": summary.throughput,
            "throughput_unit": summary.throughput_unit,
            "compile_ms": summary.compile_ms,
            "first_call_ms": summary.first_call_ms,
        },
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
) -> dict:
    """A hang or a device loss. No timing, and no retry.

    ``outcome`` is ``device_lost`` or ``hang``. Both tell the table that this
    device cannot do the setting. ``partial`` stays on while the run is still
    open.
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
    row.pop("artifacts", None)
    validate_row(row)
    return row


def for_table(row: dict) -> dict:
    """The table copy of a receipt row.

    A ``device_lost`` or ``hang`` row cannot carry a speed. Timing and
    artifacts are stripped, and the note says the device cannot do the setting.
    """

    if row.get("outcome") not in ("device_lost", "hang"):
        validate_row(row)
        return row
    cleaned = dict(row)
    cleaned.pop("artifacts", None)
    cleaned["latency"] = _blank_latency()
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

    ``device_lost`` and ``hang`` land as "this device can't do this setting",
    with the timing removed. Neither is a speed figure.
    """

    path = root / "receipts" / f"{run_id}.jsonl"
    rows = [for_table(row) for row in load_jsonl(path)]
    _dump(path, rows)
    _append_table(root, rows)
    return rows


def write_run(root: Path, run_id: str, rows: list[dict]) -> Path:
    """Write ``receipts/<run_id>.jsonl`` and append the rows to the table."""

    path = write_receipt(root, run_id, [for_table(row) for row in rows])
    _append_table(root, load_jsonl(path))
    return path


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
    path.write_text(text, encoding="utf-8")
