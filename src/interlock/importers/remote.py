"""Importer for remote Colab measurement window receipts.

Turns a remote window's receipt payload into validated remote routing rows
and keeps remote rows in their own pool, separate from local rig rows.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from interlock.devices import (
    DeviceRefused,
    RemoteDeviceIdentity,
    partition_rows_by_pool,
)
from interlock.families import stamp_family
from interlock.harness.linux_cuda import FAILURE_CLASSES
from interlock.importers.common import UNRECORDED, ImportError_, blank_latency, blank_settings, source_provenance
from interlock.receipt import ReceiptError, validate_row


def import_remote_window(
    raw: bytes,
    *,
    source_path: str,
    commit_sha: str | None = None,
) -> list[dict[str, Any]]:
    """Parse a remote window JSON receipt into validated remote routing rows."""

    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ImportError_(f"{source_path}: invalid UTF-8 JSON: {exc}") from exc

    if not isinstance(doc, Mapping):
        raise ImportError_(f"{source_path}: remote window receipt must be a JSON object")

    run_id = str(doc.get("run_id") or "").strip()
    date = str(doc.get("date") or "").strip()
    remote_raw = doc.get("remote_device")
    records = doc.get("records")
    if not run_id or not date or not isinstance(remote_raw, Mapping) or not isinstance(records, list) or not records:
        raise ImportError_(f"{source_path}: remote window requires run_id, date, remote_device, and non-empty records")

    try:
        ident = RemoteDeviceIdentity.from_dict(remote_raw)
    except DeviceRefused as exc:
        raise ImportError_(f"{source_path}: {exc}") from exc

    provenance = source_provenance(source_path, commit_sha, raw)
    doc_engine_version = str(doc.get("engine_version") or "").strip()
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(records):
        if not isinstance(entry, Mapping):
            raise ImportError_(f"{source_path}[{index}]: record must be a mapping")
        rows.append(
            _record_to_row(
                entry,
                index=index,
                run_id=run_id,
                date=date,
                ident=ident,
                provenance=provenance,
                doc_engine_version=doc_engine_version,
            )
        )
    return rows


def _record_to_row(
    entry: Mapping[str, Any],
    *,
    index: int,
    run_id: str,
    date: str,
    ident: RemoteDeviceIdentity,
    provenance: dict[str, Any],
    doc_engine_version: str = "",
) -> dict[str, Any]:
    task = str(entry.get("task") or "generate")
    model_name = str(entry.get("model") or "").strip()
    digest = str(entry.get("revision_or_digest") or UNRECORDED).strip()
    if not model_name:
        raise ImportError_(f"{run_id}[{index}]: record is missing model")

    engine_version = str(entry.get("engine_version") or doc_engine_version or UNRECORDED)
    quant = str(entry.get("quant") or UNRECORDED)
    licence = str(entry.get("licence") or UNRECORDED)
    num_ctx = entry.get("num_ctx")
    num_predict = entry.get("num_predict")
    batch = entry.get("batch")
    prompt_tokens = entry.get("prompt_tokens")
    concurrency = entry.get("concurrency")
    failure_class = entry.get("failure_class")

    row_id = str(
        entry.get("id")
        or f"C2/{task}/{model_name}/{ident.card}/{index}/{run_id}"
    )

    if failure_class is not None:
        if failure_class not in FAILURE_CLASSES:
            raise ImportError_(f"{row_id}: unsupported failure_class {failure_class!r}")
        reason = str(entry.get("failure_reason") or "").strip()
        if not reason:
            raise ImportError_(f"{row_id}: failure record requires failure_reason")
        settings = blank_settings(
            thinking_level="off" if task == "generate" else None,
            num_ctx=num_ctx,
            num_predict=num_predict,
            batch=batch,
            failure_class=failure_class,
            failure_reason=reason,
        )
        if prompt_tokens is not None:
            settings["prompt_tokens"] = prompt_tokens
        if concurrency is not None:
            settings["concurrency"] = concurrency
        row: dict[str, Any] = {
            "id": row_id,
            "task": task,
            "model": {
                "repo_or_tag": model_name,
                "revision_or_digest": digest,
                "quant": quant,
                "licence": licence,
            },
            "engine": {"name": "ollama", "version": engine_version},
            "device": ident.logical_device,
            "remote_device": ident.to_dict(),
            "settings": settings,
            "quality": [],
            "latency": blank_latency(),
            "date": date,
            "repeats": 1,
            "provisional": False,
            "provenance": dict(provenance),
            "outcome": str(failure_class),
            "notes": f"{failure_class}: {reason}",
        }
        stamp_family(row)
        try:
            validate_row(row)
        except ReceiptError as exc:
            raise ImportError_(str(exc)) from exc
        return row

    ttft_ms = entry.get("ttft_ms")
    decode_tok_s = entry.get("decode_tok_s")
    throughput = float(decode_tok_s if decode_tok_s is not None else entry["throughput"])
    unit = "tok/s" if task == "generate" else "items/s"
    latency_ms = float(entry.get("latency_ms") or entry.get("warm_p50_ms") or ttft_ms or 0.0)
    p95_ms = float(entry.get("warm_p95_ms") or latency_ms)
    settings = blank_settings(
        thinking_level="off" if task == "generate" else None,
        num_ctx=num_ctx,
        num_predict=num_predict,
        batch=batch,
    )
    for opt_key in ("prompt_tokens", "concurrency", "ttft_ms", "decode_tok_s", "peak_mib", "max_resident_ctx"):
        if entry.get(opt_key) is not None:
            settings[opt_key] = entry[opt_key]

    model_obj: dict[str, Any] = {
        "repo_or_tag": model_name,
        "revision_or_digest": digest,
        "quant": quant,
        "licence": licence,
    }
    size = entry.get("size")
    size_vram = entry.get("size_vram")
    if size is not None and size_vram is not None:
        model_obj["size"] = int(size)
        model_obj["size_vram"] = int(size_vram)

    raw_slots = entry.get("slots")
    if isinstance(raw_slots, list) and raw_slots:
        slots: list[dict[str, Any]] = []
        for s in raw_slots:
            if not isinstance(s, Mapping):
                raise ImportError_(f"{row_id}: slot must be a mapping")
            phase = str(s.get("phase") or "repeat")
            slot_tput = float(
                s.get("throughput")
                if s.get("throughput") is not None
                else (s.get("decode_tok_s") if s.get("decode_tok_s") is not None else throughput)
            )
            slot_lat = float(s.get("latency_ms") if s.get("latency_ms") is not None else latency_ms)
            slot_item: dict[str, Any] = {
                "phase": phase,
                "throughput": slot_tput,
                "latency_ms": slot_lat,
            }
            slot_ttft = s.get("ttft_ms", ttft_ms)
            if slot_ttft is not None:
                slot_item["ttft_ms"] = float(slot_ttft)
            slots.append(slot_item)
        repeats = sum(1 for s in slots if s.get("phase") == "repeat")
        if repeats < 1:
            raise ImportError_(f"{row_id}: slots must include at least one repeat slot")
    else:
        slot: dict[str, Any] = {"phase": "repeat", "throughput": throughput, "latency_ms": latency_ms}
        if ttft_ms is not None:
            slot["ttft_ms"] = float(ttft_ms)
        slots = [slot]
        repeats = 1

    row = {
        "id": row_id,
        "task": task,
        "model": model_obj,
        "engine": {"name": "ollama", "version": engine_version},
        "device": ident.logical_device,
        "remote_device": ident.to_dict(),
        "settings": settings,
        "quality": [],
        "latency": blank_latency(
            warm_p50_ms=latency_ms,
            warm_p95_ms=p95_ms,
            throughput=throughput,
            throughput_unit=unit,
        ),
        "artifacts": {"slots": slots},
        "date": date,
        "repeats": repeats,
        "provisional": False,
        "provenance": dict(provenance),
    }
    if size is not None and size_vram is not None:
        row["gpu_resident"] = int(size_vram) == int(size)
    stamp_family(row)
    try:
        validate_row(row)
    except ReceiptError as exc:
        raise ImportError_(str(exc)) from exc
    return row


def split_pools(rows: Iterable[Mapping[str, object]]) -> dict[str, list[Mapping[str, object]]]:
    """Partition routing rows into disjoint ``rig`` and ``remote`` pools."""

    try:
        return partition_rows_by_pool(rows)
    except DeviceRefused as exc:
        raise ImportError_(str(exc)) from exc
