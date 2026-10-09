"""Mechanical readout of a spec's choice rule.

ASPIRE's ``judge_control_run.choose()`` was not in the tree this was written
against, so this follows the brief's four fields: an eligibility floor, a
ranking key, a tie broken toward the committed choice, and a fallback. It
does not invent a fifth field and it does not re-score the rows.
"""

from __future__ import annotations


def choose(rows: list[dict], rule: dict, *, selection_sha: str) -> dict:
    """Pick one row id, or none, and record why.

    Higher ranking value wins. A tie prefers the committed choice (the row
    id, or a provenance sha, matching ``tie_rule.toward`` or ``selection_sha``).
    The fallback wins a tie only when the committed choice is not among the
    tied rows. A row below the floor is never chosen, including the fallback.
    """

    floor = rule["eligibility_floor"]
    floor_metric = floor["metric"]
    minimum = floor["minimum"]
    ranking_key = rule["ranking_key"]
    toward = rule["tie_rule"]["toward"]
    fallback = rule["fallback"]
    eligible = [row for row in rows if _eligible(row, floor_metric, minimum, ranking_key)]
    best = _top(eligible, ranking_key)
    chosen = None
    reason = "none"
    if len(best) == 1:
        chosen = best[0]["id"]
        reason = "ranked"
    elif best:
        committed = [row for row in best if _is_committed(row, toward, selection_sha)]
        if committed:
            chosen = committed[0]["id"]
            reason = "tie"
        else:
            fallen = [row for row in best if row["id"] == fallback]
            if fallen:
                chosen = fallen[0]["id"]
                reason = "fallback"
    return {
        "selection_sha": selection_sha,
        "eligible": [row["id"] for row in eligible],
        "chosen": chosen,
        "reason": reason,
        "ranking_key": ranking_key,
        "fallback": fallback,
    }


def _eligible(row: dict, floor_metric: str, minimum: float, ranking_key: str) -> bool:
    floor_value = _metric_value(row, floor_metric)
    rank_value = _metric_value(row, ranking_key)
    if floor_value is None or rank_value is None:
        return False
    return floor_value >= minimum


def _top(rows: list[dict], ranking_key: str) -> list[dict]:
    if not rows:
        return []
    best = max(_metric_value(row, ranking_key) for row in rows)
    return [row for row in rows if _metric_value(row, ranking_key) == best]


def _is_committed(row: dict, toward: str, selection_sha: str) -> bool:
    if row["id"] in {toward, selection_sha}:
        return True
    provenance = row.get("provenance") or {}
    shas = {provenance.get("commit_sha"), provenance.get("content_sha256")}
    return toward in shas or selection_sha in shas


def _metric_value(row: dict, name: str):
    for metric in row.get("quality") or []:
        if metric.get("name") != name:
            continue
        value = metric.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return value
    return None
