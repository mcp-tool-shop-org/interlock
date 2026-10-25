"""Where an OpenVINO arm reads its weights.

``archive`` is the default and keeps today's snapshot fetch: no cache directory
is forced, so the process cache is the one already in use. ``hot`` resolves the
same repo and revision under the hot cache (``hot_root()``) and refuses when that
copy is absent. It does not download, and it does not fall back to the archive.

A cold copy is a fresh directory on the snapshot's own drive, filled with
unbuffered I/O (``robocopy /J``), so the first load does not reuse the
operating system's file cache. The copy is deleted when the arm unloads.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from interlock.paths import PathsConfigError, configured


class StorageError(Exception):
    """The weights could not be resolved or copied. The caller refuses the arm."""


def hot_root() -> Path:
    """The hot weights cache an arm with ``hf_cache: hot`` reads: ``INTERLOCK_WEIGHTS_HOT``, else ``[paths]
    weights_hot`` in ``interlock.toml``, else ``hot/`` under ``HF_HOME`` (or the user's Hugging Face cache)."""

    try:
        found = configured("INTERLOCK_WEIGHTS_HOT", "weights_hot")
    except PathsConfigError as exc:
        raise StorageError(str(exc)) from exc
    if found is not None:
        return Path(found)
    base = os.environ.get("HF_HOME")
    return (Path(base) if base else Path.home() / ".cache" / "huggingface") / "hot"


def repo_folder(repo: str) -> str:
    """The Hugging Face cache folder for a repo id. Slashes become ``--``."""

    return "models--" + repo.replace("/", "--")


def drive_letter(path: Path) -> str:
    """The Windows drive letter, or ``unrecorded`` when the path has none."""

    letter = path.drive[:1]
    if not letter.isalpha():
        return "unrecorded"
    return letter.upper()


def hot_snapshot(root: Path, repo: str, revision: str) -> Path:
    """The snapshot directory under ``root``, or ``StorageError`` when it is missing.

    A revision that is a commit hash is the snapshot folder name. A ref file
    (the cache's ``refs/<name>``) is followed once to that commit. Nothing is
    downloaded.
    """

    folder = root / repo_folder(repo)
    exact = folder / "snapshots" / revision
    if exact.is_dir():
        return exact
    ref = folder / "refs" / revision
    if ref.is_file():
        commit = ref.read_text(encoding="utf-8").strip()
        snapped = folder / "snapshots" / commit
        if commit and snapped.is_dir():
            return snapped
    raise StorageError(f"hot cache has no copy of {repo}@{revision} under {root}")


def fresh_copy_dir(snapshot: Path) -> Path:
    """An empty directory beside the snapshot, on the same drive."""

    base = snapshot.parent / ".interlock-cold"
    base.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="copy-", dir=base))


def robocopy_argv(src: Path, dst: Path) -> list[str]:
    """Unbuffered copy of the snapshot tree. ``/J`` is the unbuffered switch."""

    return [
        "robocopy",
        str(src),
        str(dst),
        "/E",
        "/J",
        "/NFL",
        "/NDL",
        "/NJH",
        "/NJS",
        "/NC",
        "/NS",
        "/NP",
    ]


def copy_unbuffered(src: Path, dst: Path, run) -> None:
    """Copy ``src`` onto ``dst``. Robocopy's own codes 0 through 7 are success."""

    argv = robocopy_argv(src, dst)
    completed = run(argv)
    code = getattr(completed, "returncode", None)
    if not isinstance(code, int) or code >= 8:
        raise StorageError(f"cold copy failed ({code})")


def physical_disk_script(letter: str) -> str:
    """PowerShell that reads bus and media for one drive letter from Get-PhysicalDisk."""

    if len(letter) != 1 or not letter.isalpha():
        raise StorageError(f"not a drive letter: {letter!r}")
    safe = letter.upper()
    return (
        f"$p = Get-Partition -DriveLetter {safe} -ErrorAction Stop; "
        f"$d = Get-Disk -Number $p.DiskNumber; "
        "$phy = @(Get-PhysicalDisk | Where-Object "
            "{ $_.DeviceId -eq [string]$d.Number -or $_.DeviceId -eq $d.Number }); "
        "if (-not $phy) { throw 'no physical disk' }; "
        "[pscustomobject]@{BusType=[string]$phy[0].BusType; MediaType=[string]$phy[0].MediaType} "
        "| ConvertTo-Json -Compress"
    )


def parse_disk_json(text: str) -> tuple[str, str]:
    """``(bus, media)`` from the script's JSON. Missing fields stay ``unrecorded``."""

    data = json.loads(text)
    if isinstance(data, list):
        data = data[0] if data else {}
    if not isinstance(data, dict):
        raise StorageError("physical disk query did not return an object")
    bus = str(data.get("BusType") or "unrecorded")
    media = str(data.get("MediaType") or "unrecorded")
    return bus, media


def query_physical_disk(letter: str) -> tuple[str, str]:
    """Run the physical-disk script. Tests pass a fake ``run`` and do not call this."""

    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", physical_disk_script(letter)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise StorageError(completed.stderr.strip() or "physical disk query failed")
    return parse_disk_json(completed.stdout)


def describe_drive(letter: str, query=None) -> dict:
    """Drive letter plus bus and media. A failed query is recorded, not fatal."""

    if not letter or letter == "unrecorded":
        return {"drive": "unrecorded", "bus": "unrecorded", "media": "unrecorded"}
    try:
        bus, media = (query or query_physical_disk)(letter)
    except Exception:
        return {"drive": letter, "bus": "unrecorded", "media": "unrecorded"}
    return {"drive": letter, "bus": bus, "media": media}


def default_snapshot_download(repo: str, revision: str | None):
    """Today's fetch. No cache directory is passed, so the current cache stays in use."""

    from huggingface_hub import snapshot_download

    if revision:
        return snapshot_download(repo, revision=revision)
    return snapshot_download(repo)
