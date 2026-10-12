"""Shared pieces for the importers. They read receipts. They do not score gold themselves
except where a published tally rule is the only way to turn verdict lines into a cell.
"""

from __future__ import annotations

import hashlib
import statistics
from datetime import datetime, timezone

# The z used by offrig's calibrate intervals and by the ladder and difficulty reports.
WILSON_Z = 1.959963984540054

UNRECORDED = "unrecorded"


class ImportError_(Exception):
    """An import refused to guess. Named with a trailing underscore so it is not the builtin."""


def wilson(k: int, n: int) -> tuple[float, float]:
    """Wilson interval with the same edge rule as the published reports."""

    if n == 0:
        return (0.0, 1.0)
    p = k / n
    z2 = WILSON_Z * WILSON_Z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    margin = WILSON_Z * ((p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5) / (1 + z2 / n)
    low = 0.0 if k == 0 else max(0.0, centre - margin)
    high = 1.0 if k == n else min(1.0, centre + margin)
    return (low, high)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def gold_fingerprint(manifest: dict) -> str:
    lines = [f"{item['file']} {item['sha256']}" for item in manifest.get("gold", [])]
    return sha256_text("\n".join(lines))


def device_for_ollama(url: str) -> str:
    if url.startswith("http://127.0.0.1:11434"):
        return "5090"
    if url.startswith("http://127.0.0.1:11490"):
        return "CPU"
    raise ImportError_(f"refusing Ollama url {url!r}; expected 11434 or 11490")


def date_from_unix(timestamp: int | float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d")


def spread(values: list[float]) -> dict | None:
    if not values:
        return None
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "n": len(values),
    }


def blank_settings(**extra) -> dict:
    settings = {
        "thinking_level": None,
        "num_predict": None,
        "num_ctx": None,
        "batch": None,
        "static_shape": None,
        "precision": None,
    }
    settings.update(extra)
    return settings


def blank_latency(**extra) -> dict:
    latency = {
        "cold_ms": None,
        "warm_p50_ms": None,
        "warm_p95_ms": None,
        "throughput": None,
        "throughput_unit": None,
        "compile_ms": None,
    }
    latency.update(extra)
    return latency


def interval_metric(name: str, block: dict, dataset_hash: str) -> dict:
    """Copy a Wilson interval that the source receipt already computed."""

    return {
        "name": name,
        "value": block.get("rate"),
        "n": int(block.get("of") or 0),
        "interval": {
            "low": block["low"],
            "high": block["high"],
            "method": "wilson-95",
        },
        "dataset_hash": dataset_hash,
    }


def counted_metric(name: str, k: int, n: int, dataset_hash: str) -> dict:
    low, high = wilson(k, n)
    value = None if n == 0 else k / n
    return {
        "name": name,
        "value": value,
        "n": n,
        "interval": {"low": low, "high": high, "method": "wilson-95"},
        "dataset_hash": dataset_hash,
    }


def point_metric(name: str, value: float | None, n: int, dataset_hash: str) -> dict:
    number = value if value is not None else 0.0
    return {
        "name": name,
        "value": value,
        "n": n,
        "interval": {"low": number, "high": number, "method": "source-point"},
        "dataset_hash": dataset_hash,
    }


def source_provenance(path: str, commit_sha: str | None, content: bytes | None = None) -> dict:
    provenance = {"source_receipt_path": path, "commit_sha": commit_sha}
    if content is not None:
        provenance["content_sha256"] = sha256_bytes(content)
    if commit_sha is None and content is None:
        raise ImportError_("a receipt with no commit sha needs bytes to hash")
    return provenance
