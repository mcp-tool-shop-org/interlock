"""Dated RunPod offer and budget snapshots.

offrig's CLI prints text for `offrig gpus` and `offrig budget --show` until the
Publisher's JSON flags land. This module reads a version-1 snapshot file only.
It does not parse that text, and it does not open offrig's database.
"""

from __future__ import annotations

import re

from interlock.importers.common import UNRECORDED, ImportError_, blank_latency, blank_settings, source_provenance
from interlock.families import stamp_family
from interlock.receipt import validate_row

SNAPSHOT_VERSION = 1
_DATE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_BUDGET_FIELDS = ("cap", "cap_source", "committed", "spent", "remaining", "balance", "warning")


def offer_rows(snapshot: dict, *, source_path: str, content: bytes) -> list[dict]:
    version = snapshot.get("version")
    if version != SNAPSHOT_VERSION:
        raise ImportError_(f"offer snapshot version {version!r} is not {SNAPSHOT_VERSION}")
    date = snapshot.get("date")
    if not isinstance(date, str) or _DATE.fullmatch(date) is None:
        raise ImportError_("offer snapshot needs a date YYYY-MM-DD")
    rows = [_gpu_row(item, date, source_path, content) for item in snapshot.get("gpus") or []]
    budget = snapshot.get("budget")
    if budget is not None:
        rows.extend(_budget_rows(budget, date, source_path, content))
    if not rows:
        raise ImportError_("offer snapshot has no gpus and no budget")
    return rows


def remote_option_allowed(worst_case_usd: float | None, remaining_usd: float | None) -> tuple[bool, str]:
    """Whether a rented route is still inside the remaining budget.

    Null remaining is not a pass. This function does not rent anything.
    """

    if worst_case_usd is None:
        return False, "worst case is unrecorded"
    if remaining_usd is None:
        return False, "remaining budget is unrecorded"
    if worst_case_usd > remaining_usd:
        return False, f"worst case {worst_case_usd} exceeds remaining {remaining_usd}"
    return True, "within remaining budget"


def _gpu_row(item: dict, date: str, source_path: str, content: bytes) -> dict:
    gpu = item.get("gpu_type")
    if not isinstance(gpu, str) or not gpu.strip():
        raise ImportError_("an offer needs gpu_type")
    row = {
        "id": f"offer/runpod/{gpu}/{date}",
        "task": "offer",
        "model": {
            "repo_or_tag": gpu,
            "revision_or_digest": UNRECORDED,
            "quant": UNRECORDED,
            "licence": UNRECORDED,
        },
        "engine": {"name": "runpod", "version": str(item.get("offrig_commit") or UNRECORDED)},
        "device": f"runpod:{gpu}",
        "settings": blank_settings(
            price_snapshot=date,
            usd_per_hr=item.get("usd_per_hr"),
            vram_gb=item.get("vram_gb"),
            boot_s=item.get("boot_s"),
            pull_s=item.get("pull_s"),
            sec_per_claim=item.get("sec_per_claim"),
        ),
        "quality": [],
        "latency": blank_latency(),
        "date": date,
        "repeats": 1,
        "provisional": True,
        "provenance": source_provenance(source_path, None, content),
        "notes": (
            f"price_snapshot: {date}. Recorded snapshot, version 1. "
            "Not parsed from offrig's text, and not read from its database."
        ),
    }
    stamp_family(row)
    validate_row(row)
    return row


def _budget_rows(budget: dict, date: str, source_path: str, content: bytes) -> list[dict]:
    if budget.get("version") != SNAPSHOT_VERSION:
        raise ImportError_(f"budget snapshot version {budget.get('version')!r} is not {SNAPSHOT_VERSION}")
    providers = budget.get("providers")
    if not isinstance(providers, dict) or not providers:
        raise ImportError_("budget snapshot needs providers")
    rows = []
    for name, block in providers.items():
        if not isinstance(block, dict):
            raise ImportError_(f"budget provider {name} is not an object")
        settings = blank_settings(price_snapshot=date, provider=name)
        for field in _BUDGET_FIELDS:
            settings[field] = block.get(field)
        settings["ceiling"] = budget.get("ceiling")
        row = {
            "id": f"budget/{name}/{date}",
            "task": "offer",
            "model": {
                "repo_or_tag": f"budget:{name}",
                "revision_or_digest": UNRECORDED,
                "quant": UNRECORDED,
                "licence": UNRECORDED,
            },
            "engine": {"name": "offrig-budget", "version": str(SNAPSHOT_VERSION)},
            "device": f"runpod:{name}",
            "settings": settings,
            "quality": [],
            "latency": blank_latency(),
            "date": date,
            "repeats": 1,
            "provisional": True,
            "provenance": source_provenance(source_path, None, content),
            "notes": (
                f"price_snapshot: {date}. Budget version 1 for {name}. "
                "Read-only snapshot. interlock does not set a cap."
            ),
        }
        stamp_family(row)
        validate_row(row)
        rows.append(row)
    return rows
