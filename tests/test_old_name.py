"""The old project name stays only where a record or a one-release fallback needs it.

Line rules, besides the path allow-list: ``SWITCHYARD_`` is the one-release
environment fallback. ``mcp-tool-shop-org/switchyard`` stays until the GitHub
rename. ``switchyard-scratch`` is the Publisher's directory outside this repo.
``The switchyard counts``, ``fills`` and ``parity`` are pinned measurement
prompts. A dated ``Amendment`` or ``Amended`` comment in an experiment spec
stays as history.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEEDLE = "switchyard"

# A path under one of these is not a current command. The reason is the allow-list.
ALLOW_PREFIXES = (
    ("receipts/", "receipts are records of what ran"),
    ("table/", "the routing table is a record of what ran"),
    ("docs/handoff/", "the 2026-10-09 build handoff records the name chosen that day"),
    (
        "atlas/",
        "the generated map records that the old command is gone, and the GitHub repo is still the old name",
    ),
)
ALLOW_FILES = (
    ("CHANGELOG.md", "changelog history, and the Unreleased line that records the rename"),
    (
        "experiments/e1/incident-2026-10-09.md",
        "the incident note is a record of what ran",
    ),
    (
        "experiments/e3/run-notes-2026-10-10-11.md",
        "the run notes are a record of what ran",
    ),
    ("tests/test_old_name.py", "this test names the string it searches for"),
)


def _allowed_path(rel: str) -> bool:
    if any(rel == path for path, _reason in ALLOW_FILES):
        return True
    return any(rel.startswith(prefix) for prefix, _reason in ALLOW_PREFIXES)


def _allowed_line(rel: str, line: str) -> bool:
    """True when every remaining hit is a pinned prompt, a dated amendment, or a fallback."""

    residual = line.replace("SWITCHYARD_", "")
    lowered = residual.lower()
    lowered = lowered.replace("mcp-tool-shop-org/switchyard", "")
    lowered = lowered.replace("switchyard-scratch", "")
    if NEEDLE not in lowered:
        return True
    if (
        "The switchyard counts" in line
        or "The switchyard fills" in line
        or "The switchyard parity" in line
    ):
        return True
    if rel.startswith("experiments/") and rel.endswith((".yaml", ".yml")):
        stripped = line.lstrip()
        if stripped.startswith("#") and ("Amendment" in line or "Amended" in line):
            return True
    return False


def test_the_old_name_is_only_on_the_allow_list():
    listed = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    hits = []
    for rel in listed.decode("utf-8").split("\0"):
        if not rel or _allowed_path(rel):
            continue
        if NEEDLE in Path(rel).name.lower():
            hits.append(f"{rel}: filename")
        data = (ROOT / rel).read_bytes()
        if NEEDLE.encode("ascii") not in data.lower():
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            hits.append(f"{rel}: binary")
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if NEEDLE not in line.lower():
                continue
            if _allowed_line(rel, line):
                continue
            hits.append(f"{rel}:{number}: {line.strip()[:180]}")
    assert hits == []
