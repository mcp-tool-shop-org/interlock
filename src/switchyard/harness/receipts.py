"""Turn measured samples into routing rows.

Numbers come from the samples. Missing environment fields stay ``unrecorded``.
The row is validated before it is written.
"""

from __future__ import annotations

import json
from pathlib import Path

from switchyard.harness.timing import Summary
from switchyard.receipt import load_jsonl, validate_row

UNRECORDED = "unrecorded"


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
) -> dict:
    settings = {
        "thinking_level": arm.get("thinking_level"),
        "num_predict": arm.get("num_predict"),
        "num_ctx": arm.get("num_ctx"),
        "batch": arm.get("batch"),
        "static_shape": arm.get("static_shape"),
        "precision": arm.get("precision"),
        "grant": grant,
        "environment": env,
    }
    row = {
        "id": f"{spec_id}/{arm['id']}/{run_id}",
        "task": arm["task"],
        "model": {
            "repo_or_tag": arm["model"],
            "revision_or_digest": arm["revision_or_digest"],
            "quant": arm.get("quant") or UNRECORDED,
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
            "compile_ms": None,
        },
        "date": date,
        "repeats": repeats,
        "provisional": repeats < 2,
        "provenance": {"experiment_spec": spec_rel, "run_id": run_id},
        "notes": notes,
    }
    validate_row(row)
    return row


def write_run(root: Path, run_id: str, rows: list[dict]) -> Path:
    """Write ``receipts/<run_id>.jsonl`` and append the rows to the table."""

    for row in rows:
        validate_row(row)
    receipts = root / "receipts"
    receipts.mkdir(exist_ok=True)
    path = receipts / f"{run_id}.jsonl"
    _dump(path, rows)
    table = root / "table" / "routing.jsonl"
    if table.is_file() or (root / "table").is_dir():
        table.parent.mkdir(exist_ok=True)
        existing = load_jsonl(table) if table.is_file() else []
        ids = {row["id"] for row in existing}
        for row in rows:
            if row["id"] in ids:
                raise ValueError(f"duplicate receipt id {row['id']}")
        merged = sorted(existing + rows, key=lambda item: item["id"])
        _dump(table, merged)
    return path


def _dump(path: Path, rows: list[dict]) -> None:
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")
