"""The run-start marker: ``receipts/<run_id>.run.json``.

A run killed from outside (a hard kill while the main thread is stuck in a
native call) never reaches its flush, so it leaves no receipt and the next
run cannot tell it happened. The marker is written before any device contact,
carries the run's start, and is updated at every flush. A closed marker is a
run that ended in order; an unclosed one is a run of unknown end. ``plan_rest``
reads both. A marker is local state and is not a receipt: ``switchyard check``
skips it and ``.gitignore`` keeps it out of commits.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from switchyard.receipt import RUN_MARKER_SUFFIX


def marker_path(root: Path, run_id: str) -> Path:
    return root / "receipts" / f"{run_id}{RUN_MARKER_SUFFIX}"


class RunMarker:
    """One run's marker. Writes are whole-file replacements, so a kill cannot leave half a file."""

    def __init__(self, root: Path, run_id: str, *, spec_rel: str, spec_blob: str, devices, start: str) -> None:
        self.path = marker_path(root, run_id)
        self.entered = False  # set by run_interleaved once it owns the marker
        self._data = {
            "run_id": run_id,
            "spec": spec_rel,
            "spec_blob": spec_blob,
            "devices": sorted(devices),
            "start": start,
            "end": None,
            "closed": False,
        }

    def write(self) -> None:
        self.path.parent.mkdir(exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(self._data, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)

    def restart(self, start: str) -> None:
        """The run proper begins: the start moves past any rest that was slept."""

        self._data["start"] = start
        self.write()

    def touch(self, end: str) -> None:
        """A flush happened: the run was alive at ``end``. Never a failure of the run."""

        self._data["end"] = end
        try:
            self.write()
        except OSError:
            pass

    def close(self, end: str) -> None:
        self._data["end"] = end
        self._data["closed"] = True
        try:
            self.write()
        except OSError:
            pass

    def discard(self) -> None:
        """The run was refused before it touched a device: it never happened."""

        try:
            self.path.unlink()
        except OSError:
            pass


def read_markers(root: Path) -> list[tuple[Path, dict | None]]:
    """Every marker under ``receipts/`` as (path, data). Unreadable data is None."""

    folder = root / "receipts"
    found = []
    for path in sorted(folder.glob(f"*{RUN_MARKER_SUFFIX}")) if folder.is_dir() else []:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        found.append((path, data if isinstance(data, dict) else None))
    return found
