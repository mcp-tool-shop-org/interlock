"""The gap report is the experiment backlog.

A candidate is open when no row matches it, or when a row matches the model
and device but is missing a field the candidate needs. E1 stays open while
embed rows have speed but no cold/warm split.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_candidates(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def open_gaps(rows: list[dict], candidates: list[dict]) -> list[dict]:
    report = []
    for candidate in candidates:
        partial = []
        filled = False
        for row in rows:
            if row.get("outcome") in ("device_lost", "hang"):
                continue
            if not _matches(row, candidate["match"]):
                continue
            missing = [field for field in candidate.get("needs", []) if not _has_need(row, field)]
            if not missing:
                filled = True
                break
            partial.append({"id": row["id"], "missing": missing})
        if not filled:
            report.append({**candidate, "partial": partial})
    return report


def render_gaps(
    rows: list[dict],
    candidates: list[dict],
    *,
    ladder_note: str,
    skipped: list[str] | None = None,
) -> str:
    open_items = open_gaps(rows, candidates)
    by_experiment: dict[str, list[dict]] = {}
    for item in open_items:
        by_experiment.setdefault(item.get("experiment") or "other", []).append(item)
    lines = [
        "# Gap report",
        "",
        "This is the experiment backlog. A cell is open when the table has no row that fills it.",
        f"Imported rows: {len(rows)}.",
        "",
        ladder_note,
        "",
        "## Why E1–E4 are open",
        "",
    ]
    for experiment in ("E1", "E2", "E3", "E4"):
        lines.append(f"### {experiment}")
        lines.append("")
        items = by_experiment.get(experiment, [])
        if not items:
            lines.append("No open cell. This experiment's candidates are filled.")
            lines.append("")
            continue
        for item in items:
            lines.append(f"- `{item['id']}`. {item['why']}")
            for partial in item["partial"]:
                missing = ", ".join(partial["missing"])
                lines.append(f"  - Partial `{partial['id']}` is missing {missing}.")
        lines.append("")
    others = by_experiment.get("other", [])
    if others:
        lines.append("## Other open cells")
        lines.append("")
        for item in others:
            lines.append(f"- `{item['id']}`. {item['why']}")
        lines.append("")
    counts: dict[str, int] = {}
    for row in rows:
        key = f"{row['task']} / {row['device']}"
        counts[key] = counts.get(key, 0) + 1
    lines.append("## What the table does hold")
    lines.append("")
    for key in sorted(counts):
        lines.append(f"- {key}: {counts[key]}")
    lines.append("")
    skipped = skipped or []
    lines.append("## Skipped source rows")
    lines.append("")
    if not skipped:
        lines.append("None.")
    else:
        lines.append(f"{len(skipped)} source row(s) were not imported.")
        for line in skipped[:40]:
            lines.append(f"- {line}")
        if len(skipped) > 40:
            lines.append(f"- … {len(skipped) - 40} more.")
    lines.append("")
    lines.append("")
    return "\n".join(lines)


def _matches(row: dict, match: dict) -> bool:
    for key, expected in match.items():
        if key == "model" and row["model"]["repo_or_tag"] != expected:
            return False
        if key == "task" and row["task"] != expected:
            return False
        if key == "device" and row["device"] != expected:
            return False
        if key == "split" and row["settings"].get("split") != expected:
            return False
        if key not in {"model", "task", "device", "split"}:
            if row["settings"].get(key) != expected:
                return False
    return True


def _has_need(row: dict, field: str) -> bool:
    if field in row["latency"] and row["latency"][field] is not None:
        return True
    if field in row["settings"] and row["settings"][field] not in (None, ""):
        return True
    return any(metric["name"] == field and metric["value"] is not None for metric in row["quality"])
