"""Run names, stray series, and the aborted-run quarantine.

A measurement run ends as one receipt, or as a directory under the quarantine
root. It does not leave a series directory with no receipt, and it does not
reuse a name. Quarantine is not a result: nothing here is imported into the
routing table.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


class LifecycleError(Exception):
    """A run id is taken, or receipts/ holds a series with no receipt."""


_SHA_SUFFIX = re.compile(r"^[0-9A-Za-z]+(?:-\d+)?$")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def quarantine_root_for(root: Path, setting: Path | None) -> Path:
    """Default ``aborted/`` under the run root. An absolute setting is kept."""

    if setting is None:
        return root / "aborted"
    if setting.is_absolute():
        return setting
    return root / setting


def _check_name(run_id: str) -> None:
    if not isinstance(run_id, str) or not run_id or any(part in run_id for part in ("/", "\\", "..")):
        raise LifecycleError("run_id must be a single name")


def _rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def matching_quarantine(root: Path, run_id: str, setting: Path | None = None) -> list[Path]:
    """Directories ``aborted/{run_id}-{sha}``. A longer id is not a match."""

    folder = quarantine_root_for(root, setting)
    if not folder.is_dir():
        return []
    prefix = f"{run_id}-"
    found = []
    for path in folder.iterdir():
        if not path.is_dir() or not path.name.startswith(prefix):
            continue
        if _SHA_SUFFIX.fullmatch(path.name[len(prefix):]):
            found.append(path)
    return sorted(found)


def artefacts(root: Path, run_id: str, setting: Path | None = None) -> list[str]:
    """Paths that mean this run id is already in use. Empty when the name is free."""

    _check_name(run_id)
    candidates = [
        root / "receipts" / f"{run_id}.jsonl",
        root / "receipts" / f"{run_id}-series",
        root / "logs" / f"{run_id}.log",
        root / "pilot-e4b" / f"{run_id}.log",
        root / "pilot-e4b" / f"{run_id}.json",
        root / "compile-cache" / run_id,
    ]
    found = [_rel(root, path) for path in candidates if path.exists()]
    found.extend(_rel(root, path) for path in matching_quarantine(root, run_id, setting))
    return found


def require_free(root: Path, run_id: str, setting: Path | None = None) -> None:
    found = artefacts(root, run_id, setting)
    if found:
        raise LifecycleError(f"run id {run_id} is in use: {', '.join(found)}")


def stray_series(root: Path) -> list[tuple[str, Path]]:
    """``receipts/{id}-series/`` directories that have no ``receipts/{id}.jsonl``."""

    receipts = root / "receipts"
    if not receipts.is_dir():
        return []
    found = []
    for path in sorted(receipts.iterdir()):
        if not path.is_dir() or not path.name.endswith("-series"):
            continue
        run_id = path.name[: -len("-series")]
        if not (receipts / f"{run_id}.jsonl").is_file():
            found.append((run_id, path))
    return found


def refuse_strays(root: Path) -> None:
    """Before any device. Names each stray and the command that sets it aside."""

    rows = stray_series(root)
    if not rows:
        return
    lines = [
        f"{_rel(root, path)}/ has no receipt. Quarantine it with: switchyard runs quarantine {run_id}"
        for run_id, path in rows
    ]
    raise LifecycleError("\n".join(lines))


def choose_run_id(root: Path, base: str, explicit: str | None, setting: Path | None = None) -> str:
    """``base``, then ``base-2``, ``base-3``. An explicit id that is taken is refused."""

    if explicit is not None:
        _check_name(explicit)
        require_free(root, explicit, setting)
        return explicit
    _check_name(base)
    candidate = base
    number = 2
    while artefacts(root, candidate, setting):
        candidate = f"{base}-{number}"
        number += 1
        if number > 1000:
            raise LifecycleError(f"no free run id after {base}-999")
        _check_name(candidate)
    return candidate


def resolve_run_id(root: Path, *, base: str, explicit: str | None, setting: Path | None = None) -> str:
    """Stray check, then the explicit id or the next free one. No device is touched."""

    if explicit is not None:
        _check_name(explicit)
        require_free(root, explicit, setting)
        refuse_strays(root)
        return explicit
    refuse_strays(root)
    return choose_run_id(root, base, None, setting)


def is_quarantined(root: Path, run_id: str, setting: Path | None = None) -> bool:
    return bool(matching_quarantine(root, run_id, setting))


def run_id_from_quarantine(path: Path) -> str:
    note = path / "NOTE.json"
    if note.is_file():
        try:
            document = json.loads(note.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            document = None
        if isinstance(document, dict) and isinstance(document.get("run_id"), str):
            return document["run_id"]
    if "-" in path.name:
        return path.name.rsplit("-", 1)[0]
    return path.name


def state_of(root: Path, run_id: str, setting: Path | None = None) -> str:
    if is_quarantined(root, run_id, setting):
        return "aborted"
    if (root / "receipts" / f"{run_id}.jsonl").is_file() or (root / "pilot-e4b" / f"{run_id}.json").is_file():
        return "complete"
    return "stray"


def _ids_under(root: Path, setting: Path | None) -> set[str]:
    found: set[str] = set()
    receipts = root / "receipts"
    if receipts.is_dir():
        for path in receipts.iterdir():
            if path.is_file() and path.suffix == ".jsonl":
                found.add(path.stem)
            elif path.is_dir() and path.name.endswith("-series"):
                found.add(path.name[: -len("-series")])
            elif path.is_file() and path.name.endswith(".run.json"):
                found.add(path.name[: -len(".run.json")])
    for folder, suffix in (("logs", ".log"), ("pilot-e4b", ".log"), ("pilot-e4b", ".json")):
        directory = root / folder
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if path.is_file() and path.name.endswith(suffix):
                found.add(path.name[: -len(suffix)])
    cache = root / "compile-cache"
    if cache.is_dir():
        for path in cache.iterdir():
            if path.is_dir():
                found.add(path.name)
    folder = quarantine_root_for(root, setting)
    if folder.is_dir():
        for path in folder.iterdir():
            if path.is_dir():
                found.add(run_id_from_quarantine(path))
    return found


def list_runs(root: Path, setting: Path | None = None) -> list[tuple[str, str]]:
    return [(run_id, state_of(root, run_id, setting)) for run_id in sorted(_ids_under(root, setting))]


def format_runs(rows: list[tuple[str, str]]) -> str:
    if not rows:
        return "no runs\n"
    return "".join(f"{run_id}\t{state}\n" for run_id, state in rows)


def _candidates(root: Path, run_id: str, *, include_receipt: bool) -> list[Path]:
    paths = []
    if include_receipt:
        paths.append(root / "receipts" / f"{run_id}.jsonl")
        paths.append(root / "pilot-e4b" / f"{run_id}.json")
    paths.extend(
        [
            root / "receipts" / f"{run_id}-series",
            root / "receipts" / f"{run_id}.run.json",
            root / "logs" / f"{run_id}.log",
            root / "pilot-e4b" / f"{run_id}.log",
        ]
    )
    return paths


def move_aside(
    root: Path,
    run_id: str,
    *,
    sha: str | None,
    started: str | None,
    ended: str | None,
    exc_type: str | None,
    message: str,
    traceback_text: str | None = None,
    by_hand: bool = False,
    include_receipt: bool = False,
    setting: Path | None = None,
) -> Path | None:
    """Move this run's partial outputs under the quarantine root and write NOTE.json.

    Returns None when there is nothing to move. Does not touch a device.
    """

    _check_name(run_id)
    sources = [path for path in _candidates(root, run_id, include_receipt=include_receipt) if path.exists()]
    if not sources:
        return None
    folder = quarantine_root_for(root, setting)
    short = (sha or "unknown")[:7] or "unknown"
    dest = folder / f"{run_id}-{short}"
    extra = 2
    while dest.exists():
        dest = folder / f"{run_id}-{short}-{extra}"
        extra += 1
    dest.mkdir(parents=True)
    moved = []
    for src in sources:
        shutil.move(src, dest / src.name)
        moved.append(_rel(root, src))
    note = {
        "run_id": run_id,
        "harness_sha": sha,
        "started": started,
        "ended": ended,
        "exception_type": exc_type,
        "message": message,
        "files": moved,
    }
    if traceback_text:
        note["traceback"] = traceback_text
    if by_hand:
        note["quarantined_by"] = "hand"
    (dest / "NOTE.json").write_text(json.dumps(note, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dest


def quarantine_stray(root: Path, run_id: str, *, sha: str | None, setting: Path | None = None) -> Path:
    """Set a stray aside by hand. A finished receipt is not a stray."""

    _check_name(run_id)
    if (root / "receipts" / f"{run_id}.jsonl").is_file() or (root / "pilot-e4b" / f"{run_id}.json").is_file():
        raise LifecycleError(f"{run_id} has a receipt; it is not a stray")
    if is_quarantined(root, run_id, setting):
        raise LifecycleError(f"{run_id} is already quarantined")
    dest = move_aside(
        root,
        run_id,
        sha=sha,
        started=None,
        ended=utc_now(),
        exc_type=None,
        message="quarantined by hand",
        by_hand=True,
        include_receipt=False,
        setting=setting,
    )
    if dest is None:
        raise LifecycleError(f"{run_id} has nothing to quarantine")
    return dest
