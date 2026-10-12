"""The gap report is the experiment backlog.

A candidate is open when no row matches it, or when a row matches the model
and device but is missing a field the candidate needs. A row pinned as
``repo@revision`` matches a candidate that names the repo. E1 stays open
while embed rows have speed but no cold/warm split.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from interlock.receipt import is_speed_row


def summarize_error(error: object) -> str:
    """One line for a possibly multi-line exception.

    OpenVINO errors lead with a stack of "Exception from src" lines and
    carry the cause late. Prefer the compiler's message, led by its own words
    so the 200-character cut keeps the cause, then the code:
    ``Compiler returned msg: <message> (ZE_RESULT_..., 0x...)``. Else the
    Level Zero result line (``ZE_RESULT_``), else the last non-empty line.
    """

    lines = [line.strip() for line in str(error or "").splitlines() if line.strip()]
    if not lines:
        return ""
    for index, line in enumerate(lines):
        if "Compiler returned msg" in line:
            following = lines[index + 1] if index + 1 < len(lines) else ""
            return _compiler_line(line, following, str(error))
    for line in lines:
        if "ZE_RESULT_" in line:
            return line
    return lines[-1]


def _compiler_line(line: str, following: str, error: str) -> str:
    """``Compiler returned msg: <message> (<ZE_RESULT_...>, <0x code>)``.

    The message is the text after the marker on its line, else the next line.
    The Level Zero name and hex code are taken from anywhere in the error, so
    the cause leads and the 200-character cut cannot take it.
    """

    _, _, tail = line.partition("Compiler returned msg")
    message = tail.lstrip(": 	").strip() or following.strip()
    if not message:
        return line
    ze = re.search(r"ZE_RESULT_\w+", error)
    code = re.search(r"0x[0-9a-fA-F]+", error)
    codes = ", ".join(match.group(0) for match in (ze, code) if match)
    suffix = f" ({codes})" if codes else ""
    return f"Compiler returned msg: {message}{suffix}"


def aborted_run_lines(rows: list[dict]) -> list[str]:
    """One line per aborted run: the run id, the step that stopped it, and the error.

    Rows marked ``aborted: true`` carry the abort context under
    ``settings.abortion``. Repeated rows from the same run collapse into one
    line on the first row seen.
    """

    lines: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if row.get("aborted") is not True:
            continue
        run_id = (row.get("provenance") or {}).get("run_id")
        if not run_id or run_id in seen:
            continue
        seen.add(run_id)
        abortion = (row.get("settings") or {}).get("abortion")
        if not isinstance(abortion, dict):
            abortion = {}
        step = abortion.get("stop_step") or "unknown step"
        error = summarize_error(abortion.get("error"))
        if len(error) > 200:
            error = error[:199] + "…"
        lines.append(f"Aborted run `{run_id}` stopped at `{step}`: {error}")
    return lines


def append_aborted_runs(root: Path, rows: list[dict]) -> list[str]:
    """Append the aborted-run lines to ``table/gaps.md``. Returns the lines added.

    ``gaps.md`` is a generated file; the Aborted runs section lives at its end
    so a regeneration can carry the same lines. A run already listed is not
    repeated. No gaps file yet means nothing to append to; the receipt keeps
    the rows either way.
    """

    lines = aborted_run_lines(rows)
    if not lines:
        return []
    path = root / "table" / "gaps.md"
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    new = [line for line in lines if line not in text]
    if not new:
        return []
    text = text.rstrip("\n") + "\n"
    if "## Aborted runs" not in text:
        text += "\n## Aborted runs\n\n"
    text += "".join(f"- {line}\n" for line in new) + "\n"
    from interlock.harness.receipts import write_atomic

    write_atomic(path, text)
    return new


def load_candidates(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def open_gaps(rows: list[dict], candidates: list[dict]) -> list[dict]:
    report = []
    for candidate in candidates:
        partial = []
        filled = False
        for row in rows:
            if not is_speed_row(row):
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
    aborted: list[str] | None = None,
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
    if aborted is not None:
        lines.append("")
        lines.append("## Aborted runs")
        lines.append("")
        if aborted:
            for line in aborted:
                lines.append(f"- {line}")
        else:
            lines.append("None.")
    lines.append("")
    lines.append("")
    return "\n".join(lines)


def _matches(row: dict, match: dict) -> bool:
    for key, expected in match.items():
        if key == "model":
            actual = row["model"]["repo_or_tag"]
            if actual != expected and actual.split("@", 1)[0] != expected:
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
