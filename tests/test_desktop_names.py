"""Keep personal desktop app names out of committed source, test, docs, script and experiment files.

Committed historical receipts, routing tables and one historical quiet-window CSV stay as they are;
this gate checks every tracked source, test, docs, script, experiment, workflow and root file at HEAD.
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

# Explicit prefix exclusions for historical receipts and routing tables.
EXCLUDED_PREFIXES: tuple[tuple[str, str], ...] = (
    ("receipts/", "historical receipts cited by sha in studies"),
    ("table/", "historical routing table rows"),
)

# Exact-file exclusion for the one historical quiet-window log under experiments/.
EXCLUDED_FILES: tuple[tuple[str, str], ...] = (
    ("experiments/e2/quiet-window-cpu-e2-2026-10-10c.csv", "historical quiet-window log"),
)

# Scope: source, test, docs, script, experiment and workflow files (plus root markdown/config).
# atlas/, assets/ and site/ are outside CHECKED_PREFIXES and are omitted from EXCLUDED_PREFIXES.
CHECKED_PREFIXES: tuple[str, ...] = (
    "src/",
    "tests/",
    "docs/",
    "scripts/",
    "experiments/",
    ".github/",
)

# 90% floor of the 213 checked files at HEAD (213 * 0.9 = 191.7 -> 192).
MIN_CHECKED_FILES = 192


def desktop_name_pattern() -> re.Pattern[str]:
    joined = "|".join("".join(parts) for parts in DESKTOP_NAME_PARTS)
    return re.compile(f"(?:{joined})")


def is_excluded(rel: str) -> bool:
    if any(rel == path for path, _reason in EXCLUDED_FILES):
        return True
    return any(rel.startswith(prefix) for prefix, _reason in EXCLUDED_PREFIXES)


def is_checked(rel: str) -> bool:
    if is_excluded(rel):
        return False
    if any(rel.startswith(prefix) for prefix in CHECKED_PREFIXES):
        return True
    return "/" not in rel


def scan_tree(root: Path = ROOT) -> tuple[list[str], int]:
    pattern = desktop_name_pattern()
    listed = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)  # noqa: S603, S607
    hits: list[str] = []
    read_count = 0
    for rel in listed.decode("utf-8").split("\0"):
        if not rel or not is_checked(rel):
            continue
        path = root / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            hits.append(f"{rel}: failed to decode as utf-8: {exc}")
            continue
        read_count += 1
        for number, line in enumerate(text.splitlines(), 1):
            match = pattern.search(line)
            if match is not None:
                hits.append(f"{rel}:{number}: matched forbidden desktop app pattern")
    return hits, read_count


def _commit_repo(repo: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)  # noqa: S603, S607
    for key, value in (
        ("user.email", "dev@example.com"),
        ("user.name", "dev"),
        ("core.autocrlf", "false"),
        ("core.longpaths", "true"),
    ):
        subprocess.run(["git", "config", key, value], cwd=repo, check=True)  # noqa: S603, S607
    subprocess.run(["git", "add", "."], cwd=repo, check=True)  # noqa: S603, S607
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=repo, check=True, capture_output=True)  # noqa: S603, S607


def test_no_desktop_app_names_in_source_tests_docs_or_scripts():
    hits, read_count = scan_tree(ROOT)
    assert hits == []
    assert read_count >= MIN_CHECKED_FILES


def test_scan_tree_reports_planted_name_in_temp_git_repo(tmp_path: Path):
    planted = "".join(DESKTOP_NAME_PARTS[10]).replace(r"\.", ".")
    sample = tmp_path / "src" / "sample.py"
    sample.parent.mkdir(parents=True)
    sample.write_text(f"# planted {planted}\n", encoding="utf-8")
    spec = tmp_path / "experiments" / "e2" / "spec.yaml"
    spec.parent.mkdir(parents=True)
    spec.write_text("task: generate\n", encoding="utf-8")
    excluded_csv = tmp_path / "experiments" / "e2" / "quiet-window-cpu-e2-2026-10-10c.csv"
    excluded_csv.write_text(f"0,{planted}\n", encoding="utf-8")
    _commit_repo(tmp_path)

    hits, read_count = scan_tree(tmp_path)
    assert hits == ["src/sample.py:1: matched forbidden desktop app pattern"]
    assert read_count == 2


def test_scan_tree_reports_decode_failure_in_checked_scope(tmp_path: Path):
    broken = tmp_path / "docs" / "broken.md"
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b"\xff\xfe\x00\x00not-utf8")
    _commit_repo(tmp_path)

    hits, read_count = scan_tree(tmp_path)
    assert read_count == 0
    assert len(hits) == 1
    assert hits[0].startswith("docs/broken.md: failed to decode as utf-8:")

