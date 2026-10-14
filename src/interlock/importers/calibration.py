"""Calibration-chain rows: one per model, check type and split, plus difficulty tiers.

Default-rule intervals are copied from metrics.json. They are not recomputed.
Tier rows recount the verdict lines with the rule published in the difficulty
report: a cannot_tell gold is correct when the verdict is unsupported or
cannot_tell, and a false accept is a supported verdict on gold that is not
supported. Latency is the spread of per-claim wall_seconds. These runs do not
separate cold and warm.
"""

from __future__ import annotations

from interlock.families import stamp_family
from interlock.importers.common import (
    UNRECORDED,
    ImportError_,
    blank_latency,
    blank_settings,
    counted_metric,
    date_from_unix,
    device_for_ollama,
    gold_fingerprint,
    interval_metric,
    point_metric,
    source_provenance,
    spread,
)
from interlock.receipt import validate_row
from interlock.tasks import CHECK_TO_TASK

_DEFAULT_BLOCKS = ("false_accept", "false_accept_unsupported_only", "abstain")

# Pre-#47 binaries had only rule 1, and they did not write pins.quote_rule.
# This is the inference R&D named. It is not offrig's serde default.
INFERRED_QUOTE_RULE = 1
INFERRED_QUOTE_RULE_SOURCE = "pins lack quote_rule; offrig#47 (8114f07) pins it on every verdict"


def calibration_rows(
    manifest: dict,
    metrics: dict,
    verdicts: list[dict],
    tiers: dict,
    *,
    source_dir: str,
    commit_sha: str,
) -> list[dict]:
    settings_src = manifest["settings"]
    dataset = gold_fingerprint(manifest)
    device = device_for_ollama(settings_src["url"])
    model = manifest["model"]
    check = settings_src["check_type"]
    task = task_for_check(check)
    split = settings_src["split"]
    date = date_from_unix(manifest["started_at"])
    rule, rule_source = quote_rule_of(manifest, verdicts)
    rule_token = str(rule)
    think = settings_src.get("think")
    num_predict = settings_src.get("num_predict")
    offrig_version = str(manifest.get("offrig_version") or UNRECORDED)
    engine_name = "ollama"
    digest = manifest["model_digest"]
    parent_id = (
        f"verify/{model}/{digest[:12]}/{check}/{split}"
        f"/q-{rule_token}/np-{num_predict}/think-{think}"
        f"/eng-{engine_name}-{manifest['ollama_version']}/offrig-{offrig_version}"
    )
    walls = [
        line["wall_seconds"]
        for line in verdicts
        if line.get("status") == "ok" and isinstance(line.get("wall_seconds"), (int, float))
    ]
    provenance = source_provenance(f"{source_dir}/metrics.json", commit_sha)
    parent = _row(
        row_id=parent_id,
        task=task,
        model=model,
        digest=manifest["model_digest"],
        engine_version=manifest["ollama_version"],
        device=device,
        settings=blank_settings(
            thinking_level=think,
            num_predict=num_predict,
            num_ctx=settings_src.get("num_ctx"),
            check_type=check,
            split=split,
            structured=settings_src.get("structured"),
            quote_rule=rule,
            quote_rule_source=rule_source,
            offrig_version=offrig_version,
        ),
        quality=_default_quality(metrics, dataset),
        latency=blank_latency(claim_wall_s=spread(walls)),
        date=date,
        provenance=provenance,
        notes=(
            "Single chain run, so the row is provisional. "
            "Cold and warm are not in this receipt; claim_wall_s is the spread of per-claim wall_seconds. "
            f"passes_default_rule={_passes(metrics)}."
            + _quote_note(rule, rule_source)
        ),
    )
    rows = [parent]
    for tier, counts in _tier_counts(verdicts, tiers, check, split).items():
        note = "Timing stays on the parent row."
        if counts["n"] < 30:
            note += " n<30, tentative."
        rows.append(
            _row(
                row_id=f"{parent_id}/tier/{tier}",
                task=task,
                model=model,
                digest=manifest["model_digest"],
                engine_version=manifest["ollama_version"],
                device=device,
                settings=blank_settings(
                    thinking_level=think,
                    num_predict=num_predict,
                    num_ctx=settings_src.get("num_ctx"),
                    check_type=check,
                    split=split,
                    tier=tier,
                    parent_id=parent_id,
                    verdict_view="final_verdict",
                    quote_rule=rule,
                    quote_rule_source=rule_source,
                    offrig_version=offrig_version,
                ),
                quality=[
                    counted_metric("accuracy", counts["right"], counts["n"], dataset),
                    counted_metric("false_accept", counts["fa"], counts["neg"], dataset),
                ],
                latency=blank_latency(),
                date=date,
                provenance=source_provenance(f"{source_dir}/verdicts.jsonl", commit_sha),
                notes=note,
            )
        )
    return rows


def _default_quality(metrics: dict, dataset: str) -> list[dict]:
    quality = []
    for block_row in metrics.get("rows", []):
        for name in _DEFAULT_BLOCKS:
            quality.append(interval_metric(name, block_row[name], dataset))
        quality.append(
            point_metric(
                "decided_balanced_accuracy",
                block_row.get("decided_balanced_accuracy"),
                int(block_row.get("n") or 0),
                dataset,
            )
        )
    return quality


def quote_rule_of(manifest: dict, verdicts: list[dict]) -> tuple[int, str]:
    """The rule this run was scored under, and where that number came from.

    A pin written on the manifest or the verdicts wins. When no line carries
    ``pins.quote_rule``, the run was judged by a pre-#47 binary, which had
    only rule 1. That inference is recorded. offrig's serde default is not.
    """

    found: list[int] = []
    settings = manifest.get("settings") or {}
    if "quote_rule" in settings:
        found.append(_as_rule(settings["quote_rule"], "manifest"))
    for line in verdicts:
        if "quote_rule" in line:
            found.append(_as_rule(line["quote_rule"], "verdict"))
        pins = line.get("pins") or {}
        if "quote_rule" in pins:
            found.append(_as_rule(pins["quote_rule"], "verdict pin"))
    if not found:
        return INFERRED_QUOTE_RULE, INFERRED_QUOTE_RULE_SOURCE
    distinct = set(found)
    if len(distinct) != 1:
        raise ImportError_(f"quote_rule disagrees inside one run: {sorted(distinct)}")
    return found[0], "receipt"


def _as_rule(value, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ImportError_(f"quote_rule in {where} is {value!r}, not an integer")
    if value < 1:
        raise ImportError_(f"quote_rule in {where} is {value}")
    return value


def _quote_note(rule: int, source: str) -> str:
    if source == INFERRED_QUOTE_RULE_SOURCE:
        return f" quote_rule={rule}. quote_rule_source: {source}."
    return f" quote_rule={rule}, taken from the receipt and kept as its own row."


def _passes(metrics: dict) -> bool | None:
    flags = [row.get("passes_default_rule") for row in metrics.get("rows", [])]
    if not flags:
        return None
    return all(flags)


def _tier_counts(verdicts: list[dict], tiers: dict, check: str, split: str) -> dict[str, dict]:
    cells: dict[str, dict] = {}
    for line in verdicts:
        if line.get("status") == "unusable":
            continue
        info = tiers.get(line.get("claim_id"))
        if info is None or info.get("check_type") != check or info.get("split") != split:
            continue
        cell = cells.setdefault(info["tier"], {"n": 0, "right": 0, "fa": 0, "neg": 0})
        label = info["label"]
        verdict = line.get("final_verdict")
        cell["n"] += 1
        correct = verdict == label or (label == "cannot_tell" and verdict in ("unsupported", "cannot_tell"))
        cell["right"] += int(correct)
        if label != "supported":
            cell["neg"] += 1
            cell["fa"] += int(verdict == "supported")
    return cells


def task_for_check(check: str) -> str:
    """The exact task for a calibration check. An unknown check is not ``verify``."""

    try:
        return CHECK_TO_TASK[check]
    except KeyError:
        raise ImportError_(f"check_type {check!r} has no task; refusing to call it verify") from None


def _row(**kwargs) -> dict:
    row = {
        "id": kwargs["row_id"],
        "task": kwargs["task"],
        "model": {
            "repo_or_tag": kwargs["model"],
            "revision_or_digest": kwargs["digest"],
            "quant": UNRECORDED,
            "licence": UNRECORDED,
        },
        "engine": {"name": "ollama", "version": kwargs["engine_version"]},
        "device": kwargs["device"],
        "settings": kwargs["settings"],
        "quality": kwargs["quality"],
        "latency": kwargs["latency"],
        "date": kwargs["date"],
        "repeats": 1,
        "provisional": True,
        "provenance": kwargs["provenance"],
        "notes": kwargs["notes"],
    }
    stamp_family(row)
    validate_row(row)
    return row
