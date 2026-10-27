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

# A `__file__`-relative read may leave the package only where the module first asks importlib.resources for the
# packaged copy. Each such line is named exactly; a new one, or a moved one, fails until it is reviewed and listed.
ESCAPES_ALLOWED = {
    ("src/interlock/families.py", 23),
    ("src/interlock/harness/linux_cuda.py", 84),
}


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
    install = [uv, "pip", "install", "--python", str(python), str(wheel)]
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
        "from interlock.devices import RemoteDeviceIdentity\n"
        "from interlock.harness.linux_cuda import THROUGHPUT_INPUTS_SHA256, run_linux_throughput_ladder\n"
        "card = RemoteDeviceIdentity(provider='colab', card='L4', vram_mib=23034, driver_version='550.54.15',\n"
        "                            cuda_version='12.4', pool='us-central1-a')\n"
        "models = {'embed': ('BAAI/bge-base-en-v1.5', 'd1'), 'rerank': ('BAAI/bge-reranker-v2-m3', 'd2'),\n"
        "          'nli': ('cross-encoder/nli-deberta-v3-base', 'd3')}\n"
        "stub = lambda task, batch, token_length=510: {'throughput': 10.0 * batch, 'warm_p50_ms': 5.0 * batch,\n"
        "                                              'peak_mib': 2000, 'size': 1800, 'size_vram': 1800}\n"
        "rows = run_linux_throughput_ladder(spec_id='C2-THROUGHPUT', spec_rel='experiments/c2-throughput/spec.yaml',\n"
        "                                   run_id='wheel-stub', date='2026-10-26', remote_device=card,\n"
        "                                   models_by_task=models, measure_batch=stub)\n"
        "assert len(rows) == 9, len(rows)\n"
        "assert {r['settings']['input_fixture_sha256'] for r in rows} == {THROUGHPUT_INPUTS_SHA256}\n"
        "print('ok')\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}}
    done = subprocess.run([str(python), "-I", "-c", script], cwd=work, env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"


def test_the_host_ram_projection_works_from_a_clean_venv_outside_the_repository(tmp_path):
    """1.0.11: the installed wheel parses a real load block and refuses the T4 load that was killed.

    The two blocks are copied from the receipts (tests/fixtures/ollama_logs/0.40.11/hostram_*.log) and handed to the
    installed code as text, so the wheel needs no data file for this; its listing is checked for the module.
    """

    wheel = _build_wheel(tmp_path / "wheel")
    with zipfile.ZipFile(wheel) as archive:
        assert "interlock/harness/linux_cuda.py" in archive.namelist()
    venv = tmp_path / "venv"
    uv = shutil.which("uv")
    subprocess.run([uv, "venv", "--python", sys.executable, str(venv)], check=True, capture_output=True)
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run([uv, "pip", "install", "--python", str(python), str(wheel)], check=True, capture_output=True)
    logs = ROOT / "tests" / "fixtures" / "ollama_logs" / "0.40.11"
    refused_prev = (logs / "hostram_s2a_c4_34816_partial.log").read_text(encoding="utf-8")
    allowed_prev = (logs / "hostram_s1b_c8_69632_partial.log").read_text(encoding="utf-8")
    work = tmp_path / "elsewhere"
    work.mkdir()
    lines = [
        "from interlock.harness.linux_cuda import load_record, parse_load_report, project_host_ram",
        f"refused = load_record(parse_load_report({refused_prev!r}), cell='p8192 c4', num_ctx_req=8547, slots_req=4)",
        "row, reason = project_host_ram(refused, num_ctx_next=16903, slots_next=4, factor=1.05)",
        "assert row['decision'] == 'refuse' and reason.startswith('projected host RAM: 15544 MiB'), (row, reason)",
        f"allowed = load_record(parse_load_report({allowed_prev!r}), cell='p8192 c8', num_ctx_req=8547, slots_req=8)",
        "row, reason = project_host_ram(allowed, num_ctx_next=16903, slots_next=8, factor=1.05)",
        "assert row['decision'] == 'allow' and reason is None and row['projkv_mib'] == 12021.2, row",
        "print('ok')",
    ]
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}}
    done = subprocess.run(
        [str(python), "-I", "-c", "\n".join(lines)], cwd=work, env=env, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"


def _climb(parent, node, up):
    """Follow the expression built from `node` upward. Returns (directories climbed, problems, last node)."""

    problems = []
    while node in parent:
        above = parent[node]
        if isinstance(above, ast.Attribute) and above.value is node:
            if above.attr == "parent":
                up += 1
            elif above.attr == "parents":
                sub = parent.get(above)
                if isinstance(sub, ast.Subscript) and isinstance(sub.slice, ast.Constant):
                    up += sub.slice.value + 1
                else:
                    problems.append("parents[<variable>]")
                node = sub if isinstance(sub, ast.Subscript) else above
                continue
            node = above
        elif isinstance(above, ast.BinOp) and isinstance(above.op, ast.Div) and above.left is node:
            if isinstance(above.right, ast.Constant) and above.right.value == "..":
                up += 1
            node = above
        elif isinstance(above, ast.Call):
            func = above.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if node is not func and name == "dirname":
                problems.append("os.path.dirname(__file__)")
            up += sum(1 for arg in above.args if isinstance(arg, ast.Constant) and arg.value == "..")
            node = above
        elif isinstance(above, ast.Subscript) and above.value is node:
            node = above
        else:
            break
    return up, problems, node


def _file_reads(source: str) -> list[tuple[int, int, list[str]]]:
    """(line, directories above the file's own directory, problems) for every expression built from `__file__`.

    A name bound to such an expression is followed, so `here = Path(__file__).parent; here.parent` is counted.
    """

    tree = ast.parse(source)
    parent = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    tracked: dict[str, int] = {}
    found: list[tuple[int, int, list[str]]] = []
    for _ in range(4):
        found = []
        for name in ast.walk(tree):
            if not isinstance(name, ast.Name):
                continue
            if name.id == "__file__":
                base = 0
            elif name.id in tracked and isinstance(name.ctx, ast.Load):
                base = tracked[name.id] + 1  # the bound path is `base` directories up; +1 makes 0 mean the file itself
            else:
                continue
            start = base - 1 if name.id != "__file__" else 0
            up, problems, last = _climb(parent, name, start)
            if name.id == "__file__" or up != start or problems:  # a bound path merely passed along is not a climb
                found.append((name.lineno, up, problems))
            bound = parent.get(last)
            if isinstance(bound, (ast.Assign, ast.AnnAssign)) and bound.value is last:
                targets = bound.targets if isinstance(bound, ast.Assign) else [bound.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        tracked[target.id] = max(tracked.get(target.id, 0), up)
    return found


def package_escapes(package: Path, root: Path) -> tuple[list[str], set[tuple[str, int]]]:
    """Violations of the `__file__` rule in a package directory, and the allow-listed (file, line) pairs it met."""

    violations, met = [], set()
    for path in sorted(package.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        inside = len(path.relative_to(package).parts)  # parents[inside - 1] is the package root
        for line, up, problems in _file_reads(path.read_text(encoding="utf-8")):
            escapes = up > inside
            if not (escapes or problems):
                continue
            if (rel, line) in ESCAPES_ALLOWED:
                met.add((rel, line))
            else:
                what = problems or [f"{up} directories up from __file__, outside the package"]
                violations.append(f"{rel}:{line} {', '.join(what)}")
    return violations, met


def test_no_file_relative_read_leaves_the_package_except_the_named_lines():
    violations, met = package_escapes(PACKAGE, ROOT)
    assert not violations, chr(10).join(violations)
    assert met == ESCAPES_ALLOWED, f"allow-listed lines that no longer escape (remove them): {ESCAPES_ALLOWED - met}"


PATH = "from pathlib import Path"
PLANTED = {
    "parents index": [PATH, 'X = Path(__file__).resolve().parents[2] / "t.json"'],
    "parent chain": [PATH, 'X = Path(__file__).parent.parent.parent / "t.json"'],
    "dirname": ["import os", "X = os.path.dirname(__file__)"],
    "variable index": [PATH, "N = 1", "X = Path(__file__).parents[N]"],
    "dotdot division": [PATH, 'X = Path(__file__).parent / ".." / ".." / "t.json"'],
    "dotdot join": ["import os", 'X = os.path.join(os.fspath(__file__), "..", "..", "..")'],
    "bound name": [PATH, "here = Path(__file__).parent", 'X = here.parent.parent / "t.json"'],
}


def test_the_sweep_fails_on_each_planted_climb(tmp_path):
    package = tmp_path / "src" / "pkg"
    package.mkdir(parents=True)
    for label, source in PLANTED.items():
        (package / "planted.py").write_text(chr(10).join(source) + chr(10), encoding="utf-8")
        violations, _ = package_escapes(package, tmp_path)
        assert violations, f"the sweep missed: {label}"
    clean = chr(10).join([PATH, 'X = Path(__file__).parent / "schema"'])
    (package / "planted.py").write_text(clean, encoding="utf-8")
    assert package_escapes(package, tmp_path)[0] == []


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


def test_family_table_lookup_names_both_places_it_tried(tmp_path, monkeypatch):
    import pytest

    from interlock import families

    monkeypatch.setattr(families, "_FAMILIES_SOURCE_PATH", tmp_path / "table" / "model-families.json")
    monkeypatch.setattr(families.resources, "files", lambda _package: tmp_path / "installed")
    with pytest.raises(families.FamilyError) as error:
        families.family_of("anything")
    message = str(error.value)
    assert str(tmp_path / "installed" / "data" / "model-families.json") in message
    assert str(tmp_path / "table" / "model-families.json") in message


def test_family_table_lookup_prefers_the_packaged_copy_then_the_source_table(tmp_path, monkeypatch):
    from interlock import families

    source = tmp_path / "table" / "model-families.json"
    packaged = tmp_path / "installed" / "data" / "model-families.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(families, "_FAMILIES_SOURCE_PATH", source)
    monkeypatch.setattr(families.resources, "files", lambda _package: tmp_path / "installed")
    assert families.families_path() == source

    packaged.parent.mkdir(parents=True)
    packaged.write_text("{}", encoding="utf-8")
    assert families.families_path() == packaged
