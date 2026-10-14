"""Kimi's NLI floor, one row per check type and split.

The results file records the OpenVINO id ``GPU.0``. The pre-registration in
the same commit says the device is the Intel iGPU, found by name. The row
says iGPU and keeps the OpenVINO id in settings. Quality intervals are copied
from the receipt.
"""

from __future__ import annotations

from interlock.families import stamp_family
from interlock.importers.common import (
    UNRECORDED,
    blank_latency,
    blank_settings,
    interval_metric,
    point_metric,
    sha256_bytes,
    source_provenance,
)
from interlock.receipt import validate_row

_BLOCKS = ("false_accept", "false_accept_unsupported_only", "abstain")


def nli_rows(document: dict, *, source_path: str, commit_sha: str, prereg_says_intel_igpu: bool) -> list[dict]:
    if not prereg_says_intel_igpu:
        raise ValueError(
            "the NLI receipt's GPU id is not imported unless the same commit's prereg names the Intel iGPU")
    model = document["model"]
    if model.get("device") != "GPU.0":
        raise ValueError(f"unexpected NLI device {model.get('device')!r}")
    dataset = sha256_bytes(_canonical(document).encode("utf-8"))
    rows = []
    for leg_name, leg in document["legs"].items():
        check, _, split = leg_name.partition("/")
        metrics = leg["metrics"]
        quality = [interval_metric(name, metrics[name], dataset) for name in _BLOCKS]
        quality.append(
            point_metric(
                "decided_balanced_accuracy",
                metrics.get("decided_balanced_accuracy"),
                int(metrics.get("n") or 0),
                dataset,
            )
        )
        per_claim = leg.get("sec_per_claim") or {}
        row = {
            "id": f"nli/{model['repo']}/{check}/{split}",
            "task": "nli",
            "model": {
                "repo_or_tag": model["repo"],
                "revision_or_digest": model["revision"],
                "quant": UNRECORDED,
                "licence": UNRECORDED,
            },
            "engine": {"name": "openvino", "version": document["openvino"]},
            "device": "iGPU",
            "settings": blank_settings(
                batch=model.get("batch"),
                static_shape=str(model.get("seq")) if model.get("seq") else None,
                check_type=check,
                split=split,
                ov_device_id=model["device"],
                ir_digest=model.get("ir_digest"),
            ),
            "quality": quality,
            "latency": blank_latency(
                source_mean_ms=_ms(per_claim.get("mean")),
                source_max_ms=_ms(per_claim.get("max")),
            ),
            "date": document["date"],
            "repeats": 1,
            "provisional": True,
            "provenance": source_provenance(source_path, commit_sha),
            "notes": (
                "Single run, provisional. Cold and warm are not in this receipt. "
                "OpenVINO id GPU.0 is the Intel iGPU: the prereg at this commit says the device was found by name. "
                "dataset_hash is the sha256 of this results file, not of the gold files. "
                f"Truncated items in this leg: {leg.get('truncated_n')}."
            ),
        }
        stamp_family(row)
        validate_row(row)
        rows.append(row)
    return rows


def prereg_names_intel_igpu(text: str) -> bool:
    lowered = text.lower()
    return "intel igpu" in lowered and "found by name" in lowered


def _ms(seconds):
    if seconds is None:
        return None
    return float(seconds) * 1000.0


def _canonical(document: dict) -> str:
    import json

    return json.dumps(document, sort_keys=True, separators=(",", ":"))
