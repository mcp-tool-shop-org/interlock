"""Keep personal desktop app names out of committed source, test, docs and script files.

Committed historical receipts and routing tables stay as they are; this gate
checks every tracked source, test, docs and script file at HEAD.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Split literals so this test file does not match the patterns it builds.
DESKTOP_NAME_PARTS: tuple[tuple[str, ...], ...] = (
    ("Anti", "gravity", r"\.exe"),
    ("clau", "de", r"\.exe"),
    ("chro", "me", r"\.exe"),
    ("steam", "web", "helper"),
    ("ste", "am", r"\.exe"),
    ("Omen", "Command"),
    ("Omen", " ", "Command"),
    ("Intel", "Graphics", "Software"),
    ("Intel", " ", "Graphics"),
    ("Snipping", "Tool"),
    ("Note", "pad", r"\.exe"),
    ("gr", "ok", r"\.exe"),
)

# Explicit exclusions for historical receipts, routing tables and experiment data.
EXCLUDED_PREFIXES: tuple[tuple[str, str], ...] = (
    ("receipts/", "historical receipts cited by sha in studies"),
    ("table/", "historical routing table rows"),
    ("experiments/", "historical experiment specs and quiet-window logs"),
    ("atlas/", "generated repository map"),
    ("assets/", "binary logo assets"),
    ("site/", "static Astro site bundle"),
)

# Scope: source, test, docs, and script files (plus workflows and root markdown/config).
CHECKED_PREFIXES: tuple[str, ...] = (
    "src/",
    "tests/",
    "docs/",
    "scripts/",
    ".github/",
)


def desktop_name_pattern() -> re.Pattern[str]:
    joined = "|".join("".join(parts) for parts in DESKTOP_NAME_PARTS)
    return re.compile(f"(?:{joined})")


def is_excluded(rel: str) -> bool:
    return any(rel.startswith(prefix) for prefix, _reason in EXCLUDED_PREFIXES)


def is_checked(rel: str) -> bool:
    if is_excluded(rel):
        return False
    if any(rel.startswith(prefix) for prefix in CHECKED_PREFIXES):
        return True
    return "/" not in rel


def scan_tree(root: Path = ROOT) -> list[str]:
    pattern = desktop_name_pattern()
    listed = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)  # noqa: S603, S607
    hits: list[str] = []
    for rel in listed.decode("utf-8").split("\0"):
        if not rel or not is_checked(rel):
            continue
        path = root / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            match = pattern.search(line)
            if match is not None:
                hits.append(f"{rel}:{number}: matched forbidden desktop app pattern")
    return hits


def test_no_desktop_app_names_in_source_tests_docs_or_scripts():
    assert scan_tree(ROOT) == []
