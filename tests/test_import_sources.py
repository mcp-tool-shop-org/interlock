"""The full `interlock import` rebuild reads its source repos from configuration, not from rig paths in code.

The commit each source must be at stays pinned (importers/pins.py); only where the clone lives is configurable:
`--source name=path` (repeatable) or `[import.sources]` in `<root>/interlock.toml`. The flag beats the file.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from interlock.cli import main
from interlock.importers.assemble import import_receipts
from interlock.importers.common import ImportError_
from interlock.importers.pins import ASPIRE_COMMIT, NPU_BUILDOUT_COMMIT, RND_COMMIT

ROOT = Path(__file__).resolve().parents[1]
PINS = {"rnd": RND_COMMIT, "npu": NPU_BUILDOUT_COMMIT, "aspire": ASPIRE_COMMIT}


def _git_repo(path: Path) -> str:
    """A fixture repo with one commit; returns that commit's sha."""

    path.mkdir(parents=True)
    for cmd in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "dev@example.com"],
        ["git", "config", "user.name", "dev"],
    ):
        subprocess.run(cmd, cwd=path, check=True, capture_output=True)  # noqa: S603, S607
    (path / "f.txt").write_text("x", encoding="utf-8")
    for cmd in (["git", "add", "."], ["git", "commit", "-q", "-m", "c"]):
        subprocess.run(cmd, cwd=path, check=True, capture_output=True)  # noqa: S603, S607
    out = subprocess.run(  # noqa: S603, S607
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    )
    return out.stdout.strip()


@pytest.fixture
def capture(monkeypatch):
    """Replace the rebuild with a recorder; the CLI still resolves sources and writes the table."""

    seen: dict = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return [], []

    monkeypatch.setattr("interlock.importers.assemble.import_receipts", fake)
    return seen


def _table(root: Path) -> None:
    (root / "table").mkdir(exist_ok=True)
    (root / "table" / "candidates.json").write_text("[]", encoding="utf-8")


def _flags(**paths: str) -> list[str]:
    out: list[str] = []
    for name, path in paths.items():
        out += ["--source", f"{name}={path}"]
    return out


def test_a_rebuild_with_no_sources_names_each_one_its_pin_and_how_to_set_it(tmp_path, capsys):
    _table(tmp_path)
    assert main(["import", "--root", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err
    for name, pin in PINS.items():
        assert name in err
        assert pin in err
    assert "--source" in err
    assert "[import.sources]" in err
    assert "interlock.toml" in err


def test_a_rebuild_with_one_source_names_only_the_missing_ones(tmp_path, capsys):
    _table(tmp_path)
    assert main(["import", "--root", str(tmp_path), *_flags(rnd="/clones/rnd")]) == 1
    err = capsys.readouterr().err
    assert NPU_BUILDOUT_COMMIT in err
    assert ASPIRE_COMMIT in err
    assert RND_COMMIT not in err


def test_the_remote_window_merge_needs_no_sources(tmp_path, capsys):
    _table(tmp_path)
    window = tmp_path / "w.jsonl"
    window.write_bytes(b"")
    # an empty file is refused by the importer, not by a missing source
    assert main(["import", "--root", str(tmp_path), "--remote-window", str(window)]) == 1
    err = capsys.readouterr().err
    assert "--source" not in err
    assert "[import.sources]" not in err


def test_source_flags_reach_the_rebuild_and_are_recorded_as_given(tmp_path, capture):
    _table(tmp_path)
    given = {"rnd": "clones/rnd", "npu": "clones/rnd", "aspire": "clones/aspire-si"}
    assert main(["import", "--root", str(tmp_path), *_flags(**given)]) == 0
    assert capture["rnd_repo"] == Path("clones/rnd")
    assert capture["npu_repo"] == Path("clones/rnd")
    assert capture["aspire_repo"] == Path("clones/aspire-si")
    gaps = (tmp_path / "table" / "gaps.md").read_text(encoding="utf-8")
    for name, path in given.items():
        assert f"{name} @ {PINS[name]}: {path}" in gaps


def test_the_flag_beats_the_config_file(tmp_path, capture):
    _table(tmp_path)
    (tmp_path / "interlock.toml").write_text(
        '[import.sources]\nrnd = "cfg/rnd"\nnpu = "cfg/npu"\naspire = "cfg/aspire"\n', encoding="utf-8"
    )
    assert main(["import", "--root", str(tmp_path), *_flags(rnd="flag/rnd")]) == 0
    assert capture["rnd_repo"] == Path("flag/rnd")
    assert capture["npu_repo"] == Path("cfg/npu")
    assert capture["aspire_repo"] == Path("cfg/aspire")


def test_a_config_file_alone_is_enough(tmp_path, capture):
    _table(tmp_path)
    (tmp_path / "interlock.toml").write_text(
        '[import.sources]\nrnd = "cfg/rnd"\nnpu = "cfg/npu"\naspire = "cfg/aspire"\n', encoding="utf-8"
    )
    assert main(["import", "--root", str(tmp_path)]) == 0
    assert capture["rnd_repo"] == Path("cfg/rnd")


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--source", "bogus=/x"], "unknown source 'bogus'"),
        (["--source", "rnd"], "name=path"),
        (["--source", "rnd="], "name=path"),
    ],
    ids=["unknown-name", "no-equals", "empty-path"],
)
def test_a_malformed_source_flag_is_a_message_not_a_traceback(tmp_path, capsys, argv, message):
    _table(tmp_path)
    assert main(["import", "--root", str(tmp_path), *argv]) == 1
    err = capsys.readouterr().err
    assert message in err
    assert "Traceback" not in err


def test_a_malformed_config_file_is_a_message_not_a_traceback(tmp_path, capsys):
    _table(tmp_path)
    (tmp_path / "interlock.toml").write_text("[import.sources\n", encoding="utf-8")
    assert main(["import", "--root", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "interlock.toml" in err
    assert "Traceback" not in err


def test_a_source_not_at_its_pinned_commit_is_refused_naming_the_commit(tmp_path):
    rnd, aspire, npu = tmp_path / "rnd", tmp_path / "aspire", tmp_path / "npu"
    rnd_sha, aspire_sha = _git_repo(rnd), _git_repo(aspire)
    _git_repo(npu)  # a real repo, but without the pinned NPU commit
    with pytest.raises(ImportError_, match=NPU_BUILDOUT_COMMIT[:12]):
        import_receipts(rnd_repo=rnd, npu_repo=npu, aspire_repo=aspire, rnd_pin=rnd_sha, aspire_pin=aspire_sha)


def test_the_importers_name_no_rig_path():
    offenders = [
        f"{path.relative_to(ROOT).as_posix()}:{number}"
        for path in sorted((ROOT / "src" / "interlock" / "importers").rglob("*.py"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if "E:/AI" in line or "E:\\AI" in line
    ]
    assert offenders == []


def test_the_npu_pin_names_its_tag():
    text = (ROOT / "src" / "interlock" / "importers" / "pins.py").read_text(encoding="utf-8")
    assert "data/npu-buildout-275e511" in text
