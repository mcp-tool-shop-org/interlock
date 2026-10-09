"""Math-ladder rows: one per model, cell and verdict view.

The cell tally follows the ladder report. ``final_verdict`` is one view.
``model_verdict`` is the other, labelled post-hoc. A commit sha is recorded
when the run is in the pinned tree. A content hash is always recorded.
"""

from __future__ import annotations

from switchyard.importers.calibration import quote_rule_of
from switchyard.importers.common import (
    UNRECORDED,
    blank_latency,
    blank_settings,
    counted_metric,
    date_from_unix,
    device_for_ollama,
    sha256_bytes,
    source_provenance,
)
from switchyard.receipt import validate_row

VIEWS = ("final_verdict", "model_verdict")


def ladder_rows(
    gold_lines: list[dict],
    manifest: dict,
    verdicts: list[dict],
    *,
    source_dir: str,
    gold_bytes: bytes,
    verdict_bytes: bytes,
    commit_sha: str | None = None,
) -> list[dict]:
    gold = {line["id"]: line for line in gold_lines}
    settings_src = manifest["settings"]
    device = device_for_ollama(settings_src["url"])
    dataset = sha256_bytes(gold_bytes)
    content = gold_bytes + b"\n" + verdict_bytes
    date = date_from_unix(manifest["started_at"])
    model = manifest["model"]
    rule, rule_source = quote_rule_of(manifest, verdicts)
    rows = []
    for view in VIEWS:
        counts = _tally(gold, verdicts, view)
        for cell, count in sorted(counts.items()):
            if count["n"] == 0:
                continue
            where = "final verdict" if view == "final_verdict" else "model verdict, post-hoc"
            note = f"Single run, provisional. View: {where}. quote_rule={rule}. quote_rule_source: {rule_source}."
            if count["n"] < 30:
                note += " n<30, tentative."
            row = {
                "id": f"ladder/{model}/{cell}/q-{rule}/{view}",
                "task": "verify",
                "model": {
                    "repo_or_tag": model,
                    "revision_or_digest": manifest["model_digest"],
                    "quant": UNRECORDED,
                    "licence": UNRECORDED,
                },
                "engine": {"name": "ollama", "version": manifest["ollama_version"]},
                "device": device,
                "settings": blank_settings(
                    thinking_level=settings_src.get("think"),
                    num_predict=settings_src.get("num_predict"),
                    num_ctx=settings_src.get("num_ctx"),
                    split=settings_src.get("split"),
                    cell=cell,
                    verdict_view=view,
                    quote_rule=rule,
                    quote_rule_source=rule_source,
                ),
                "quality": [
                    counted_metric("accuracy", count["right"], count["n"], dataset),
                    counted_metric("false_accept", count["fa"], count["neg"], dataset),
                ],
                "latency": blank_latency(),
                "date": date,
                "repeats": 1,
                "provisional": True,
                "provenance": source_provenance(f"{source_dir}/verdicts.jsonl", commit_sha, content),
                "notes": note,
            }
            validate_row(row)
            rows.append(row)
    return rows


def _tally(gold: dict, verdicts: list[dict], field: str) -> dict[str, dict]:
    cells: dict[str, dict] = {}
    for line in verdicts:
        item = gold.get(line.get("claim_id"))
        if item is None:
            continue
        cell = cells.setdefault(item["ladder"]["cell"], {"n": 0, "right": 0, "fa": 0, "neg": 0})
        if line.get("status") == "unusable":
            continue
        verdict = line.get(field)
        cell["n"] += 1
        cell["right"] += int(verdict == item["label"])
        if item["label"] == "unsupported":
            cell["neg"] += 1
            cell["fa"] += int(verdict == "supported")
    return cells
