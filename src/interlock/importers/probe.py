"""npu-probe speed and parity rows.

A GPU id becomes iGPU only when the same file's device map gives it a full
name containing "Intel". GPU.1 is never imported. A file that says GPU.0 and
does not name the device is skipped, not guessed. Error rows are skipped.
Cold and warm are not in these receipts, so those fields stay empty. That is
why E1 exists.
"""

from __future__ import annotations

from interlock.families import stamp_family
from interlock.importers.common import UNRECORDED, blank_latency, blank_settings, sha256_bytes, source_provenance
from interlock.receipt import validate_row


def probe_rows(document: dict, *, source_path: str, commit_sha: str, file_date: str) -> tuple[list[dict], list[str]]:
    names = _intel_gpu_ids(document.get("devices") or {})
    dataset = sha256_bytes(_canonical(document).encode("utf-8"))
    engine_version = str(document.get("openvino") or UNRECORDED)
    rows: list[dict] = []
    skipped: list[str] = []
    cases = document.get("cases") or document.get("rows") or []
    contended = "contended" in str(document.get("note") or "").lower() or "before treating" in str(document.get(
        "note") or "").lower()
    for index, case in enumerate(cases):
        if case.get("error"):
            skipped.append(f"{source_path}#{index}: error row, not a routing row")
            continue
        device = _device(case.get("device"), names, source_path, index, skipped)
        if device is None:
            continue
        kind = case.get("kind") or "engine"
        task = {"embed": "embed", "nli": "nli"}.get(kind, "engine")
        model = case.get("model") or f"openvino-bench/{case.get('case', 'unknown')}"
        quality = []
        if "min_cosine_vs_cpu" in case:
            cosine = case["min_cosine_vs_cpu"]
            quality.append(
                {
                    "name": "min_cosine_vs_cpu",
                    "value": cosine,
                    "n": 0,
                    "interval": {"low": cosine, "high": cosine, "method": "source-minimum"},
                    "dataset_hash": dataset,
                }
            )
        batch = case.get("batch")
        precision = case.get("precision")
        row_id = _row_id(source_path, model, device, case, index)
        notes = "Single receipt, provisional. The source does not split cold and warm."
        if quality and quality[0]["n"] == 0 and "min_cosine_vs_cpu" in case:
            notes += " The source reports a minimum cosine and not the comparison count."
        if contended:
            notes += " The source marks this run contended."
        row = {
            "id": row_id,
            "task": task,
            "model": {
                "repo_or_tag": model,
                "revision_or_digest": UNRECORDED,
                "quant": UNRECORDED,
                "licence": UNRECORDED,
            },
            "engine": {"name": "openvino", "version": engine_version},
            "device": device,
            "settings": blank_settings(
                batch=batch,
                precision=precision,
                static_shape=str(document["seq"]) if document.get("seq") else None,
                case=case.get("case"),
            ),
            "quality": quality,
            "latency": blank_latency(
                throughput=_throughput(case),
                throughput_unit=_unit(case),
                compile_ms=_ms(case.get("compile_s")),
                source_median_ms=_ms(case.get("median_s")) if "median_s" in case else _number(case.get("ms_per_call")),
                source_best_ms=_ms(case.get("best_s")) if "best_s" in case else None,
            ),
            "date": file_date,
            "repeats": 1,
            "provisional": True,
            "provenance": source_provenance(source_path, commit_sha),
            "notes": notes,
        }
        stamp_family(row)
        validate_row(row)
        rows.append(row)
    return rows, skipped


def _intel_gpu_ids(devices: dict) -> dict[str, str]:
    found = {}
    for ov_id, full_name in devices.items():
        if isinstance(ov_id, str) and ov_id.upper().startswith("GPU") and "intel" in str(full_name).lower():
            found[ov_id.upper()] = str(full_name)
    return found


def _device(raw, names: dict[str, str], source: str, index: int, skipped: list[str]) -> str | None:
    if not isinstance(raw, str):
        skipped.append(f"{source}#{index}: no device")
        return None
    token = raw.strip()
    upper = token.upper()
    if upper in {"NPU", "CPU"}:
        return upper
    if upper == "GPU.1" or upper == "GPU":
        skipped.append(f"{source}#{index}: refused {token}")
        return None
    if upper.startswith("GPU"):
        full_name = names.get(upper)
        if full_name is None:
            skipped.append(f"{source}#{index}: {token} has no Intel full name in this file")
            return None
        return "iGPU"
    skipped.append(f"{source}#{index}: unknown device {token}")
    return None


def _throughput(case: dict):
    if "items_per_s" in case:
        return case["items_per_s"]
    if "tops" in case:
        return case["tops"]
    return None


def _unit(case: dict) -> str | None:
    if "items_per_s" in case:
        return "items/s"
    if "tops" in case:
        return "TOPS"
    return None


def _ms(seconds) -> float | None:
    if seconds is None:
        return None
    return float(seconds) * 1000.0


def _number(value):
    return None if value is None else float(value)


def _row_id(source: str, model: str, device: str, case: dict, index: int) -> str:
    stem = source.rsplit("/", 1)[-1].removesuffix(".json")
    batch = case.get("batch")
    precision = case.get("precision") or case.get("kind") or case.get("case") or index
    return f"probe/{stem}/{model}/{device}/b{batch}/{precision}"


def _canonical(document: dict) -> str:
    import json

    return json.dumps(document, sort_keys=True, separators=(",", ":"))
