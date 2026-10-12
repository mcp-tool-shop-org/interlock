"""The run-start marker: ``receipts/<run_id>.run.json``.

A run killed from outside (a hard kill while the main thread is stuck in a
native call) never reaches its flush, so it leaves no receipt and the next
run cannot tell it happened. The marker is written before any device contact,
carries the run's start, and is updated at every flush. A closed marker is a
run that ended in order; an unclosed one is a run killed from outside. It feeds no
scheduling logic. A marker is local state and is not a receipt: ``interlock check``
skips it and ``.gitignore`` keeps it out of commits.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from interlock.receipt import RUN_MARKER_SUFFIX


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
        temporary = self.path.with_name(f"{self.path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        temporary.write_text(json.dumps(self._data, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)

    def restart(self, start: str) -> None:
        """The run proper begins: the start is re-stamped."""

        self._data["start"] = start
        self.write()

    def touch(self, end: str) -> None:
        """A flush happened: the run was alive at ``end``. Never a failure of the run."""

        self._data["end"] = end
        try:
            self.write()
        except OSError:
            pass

    def close(self, end: str, reason: str | None = None) -> None:
        self._data["end"] = end
        self._data["closed"] = True
        if reason is not None:
            self._data["reason"] = reason
        try:
            self.write()
        except OSError:
            pass

    def note_restart(self, record: dict) -> None:
        """One Ollama restart. This is run metadata, not a routing row."""

        self._data.setdefault("restarts", []).append(record)
        try:
            self.write()
        except OSError:
            pass

    def note_restart_disabled(self, reason: str) -> None:
        """The start record does not match the live server, so this run does not restart."""

        if self._data.get("ollama_restart_disabled") == reason:
            return
        self._data["ollama_restart_disabled"] = reason
        try:
            self.write()
        except OSError:
            pass

    def note_failure(self, step: str, error: str) -> None:
        """The step that was in flight. It is inconclusive, never a pass."""

        self._data.setdefault("failed_steps", []).append(
            {"step": step, "error": error, "inconclusive": True}
        )
        try:
            self.write()
        except OSError:
            pass

    def restart_records(self) -> list:
        return list(self._data.get("restarts") or [])

    def discard(self) -> None:
        """The run was refused before it touched a device: it never happened."""

        try:
            self.path.unlink()
        except OSError:
            pass
