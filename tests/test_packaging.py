"""The wheel is what users install. Nothing the code reads at run time may live only in the repository."""

# ruff: noqa: S603  (every subprocess here runs uv or the venv python built from this repo)

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "interlock"
FAMILIES_TABLE = ROOT / "table" / "model-families.json"
FAMILIES_RESOURCE = "interlock/data/model-families.json"

# Climbing out of the package from `__file__` is only allowed in a module that first asks importlib.resources.
PACKAGED_FIRST = "resources.files("


def _build_wheel(out_dir: Path) -> Path:
    uv = shutil.which("uv")
    assert uv, "uv is required to build the wheel for this test"
    subprocess.run([uv, "build", "--wheel", "--out-dir", str(out_dir)], cwd=ROOT, check=True, capture_output=True)
    (wheel,) = out_dir.glob("*.whl")
    return wheel


def test_the_wheel_carries_the_family_table_byte_for_byte(tmp_path):
    wheel = _build_wheel(tmp_path)
    with zipfile.ZipFile(wheel) as archive:
        assert archive.read(FAMILIES_RESOURCE) == FAMILIES_TABLE.read_bytes()


def test_family_of_and_stamp_family_work_from_a_clean_venv_outside_the_repository(tmp_path):
    """A wheel install on this leg's Python, run with cwd outside the repo and no PYTHONPATH."""

    wheel = _build_wheel(tmp_path / "wheel")
    venv = tmp_path / "venv"
    uv = shutil.which("uv")
    subprocess.run([uv, "venv", "--python", sys.executable, str(venv)], check=True, capture_output=True)
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    install = [uv, "pip", "install", "--python", str(python), "--no-deps", str(wheel)]
    subprocess.run(install, check=True, capture_output=True)
    table = json.loads(FAMILIES_TABLE.read_text(encoding="utf-8"))
    tag, family = next(iter(table.items()))
    work = tmp_path / "elsewhere"
    work.mkdir()
    script = (
        "from interlock.families import family_of, stamp_family\n"
        f"assert family_of({tag!r}) == {family!r}\n"
        f"row = {{'model': {{'repo_or_tag': {tag!r}}}}}\n"
        "stamp_family(row)\n"
        f"assert row['model_family'] == {family!r}\n"
        "assert family_of('no-such-tag:1b') is None\n"
        "print('ok')\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}}
    done = subprocess.run([str(python), "-I", "-c", script], cwd=work, env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"


def _levels_up(tree: ast.AST) -> list[tuple[int, int]]:
    """(line, directories above the file's own directory) for every expression built from `__file__`."""

    parent = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    found = []
    for name in ast.walk(tree):
        if not (isinstance(name, ast.Name) and name.id == "__file__"):
            continue
        node, up = name, 0
        while node in parent:
            above = parent[node]
            if isinstance(above, ast.Attribute) and above.value is node:
                if above.attr == "parent":
                    up += 1
                elif above.attr == "parents":
                    sub = parent.get(above)
                    if isinstance(sub, ast.Subscript) and isinstance(sub.slice, ast.Constant):
                        up += sub.slice.value + 1
                        node = sub
                        continue
            elif not isinstance(above, (ast.Call, ast.Subscript)):
                break
            node = above
        found.append((name.lineno, up))
    return found


def test_no_file_relative_read_leaves_the_package_without_a_packaged_copy():
    checked = 0
    for path in sorted(PACKAGE.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        inside = len(path.relative_to(PACKAGE).parts)  # parents[inside - 1] is the package root
        for line, up in _levels_up(ast.parse(source)):
            checked += 1
            if up > inside:
                assert PACKAGED_FIRST in source, (
                    f"{path.relative_to(ROOT)}:{line} reads {up} directories up from __file__, outside the package, "
                    "and the module has no importlib.resources packaged copy"
                )
    assert checked


def test_the_sweep_catches_the_1_0_3_families_read():
    (line, up), = _levels_up(ast.parse('p = Path(__file__).resolve().parents[2] / "table" / "x.json"\n'))
    assert up == 3 and line == 1
    assert _levels_up(ast.parse('p = Path(__file__).resolve().parent.joinpath("schema", "a.json")\n')) == [(1, 1)]


# The SLSA generator is a reusable workflow that must be called by tag (it refuses a sha ref).
TAG_PINNED = {"slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v2.1.0"}


def test_every_action_in_the_workflows_is_pinned_by_commit_sha_with_a_version_comment():
    found = 0
    for workflow in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        for ref, rest in re.findall(r"uses: (\S+)(.*)", workflow.read_text(encoding="utf-8")):
            found += 1
            if ref in TAG_PINNED:
                continue
            assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", ref), f"{workflow.name}: {ref} is not pinned by commit sha"
            assert re.search(r"# v\d", rest), f"{workflow.name}: {ref} has no version comment"
    assert found
