"""Refuse a spec whose working tree bytes differ from the committed blob.

``git hash-object`` is the worktree file. ``git rev-parse HEAD:path`` is the
committed blob. Dry-run and a real run both require them to match.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from switchyard.harness.spec import SpecError


def require_unmodified(repo: Path, spec_path: Path) -> str:
    rel = _relative(repo, spec_path)
    working = _git(repo, "hash-object", "--", spec_path.as_posix())
    committed = _git(repo, "rev-parse", f"HEAD:{rel}")
    if working != committed:
        raise SpecError(f"spec {rel} was edited after commit ({working} != {committed})")
    return committed


def _relative(repo: Path, spec_path: Path) -> str:
    try:
        return spec_path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError as exc:
        raise SpecError("the spec is outside the repository") from exc


def _git(repo: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", repo.as_posix(), *args],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise SpecError(f"spec is not a committed blob ({detail})") from exc
    return result.stdout.decode("utf-8").strip()
