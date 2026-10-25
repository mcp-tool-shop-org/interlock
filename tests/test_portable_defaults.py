"""No rig path is hard-coded in shipped code: each location is configuration with a portable default.

Env first, then `[paths]` in `interlock.toml` in the working directory; the default is portable (a temp or user
cache location, or off). A guard scans `src/` for drive-letter, `/home/` and `/Users/` paths, with an empty allow-list.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVE = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")
ALLOWED: dict[str, str] = {}  # path:line -> justification. Empty unless a PR justifies an entry.


def _scan() -> list[str]:
    hits = []
    for path in sorted((ROOT / "src").rglob("*")):
        if not path.is_file() or path.suffix not in {".py", ".json", ".yaml", ".toml", ".md"}:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if DRIVE.search(line) or "/home/" in line or "/Users/" in line or "\\Users\\" in line:
                where = f"{path.relative_to(ROOT).as_posix()}:{number}"
                if where not in ALLOWED:
                    hits.append(f"{where}: {line.strip()[:80]}")
    return hits


def test_no_shipped_file_names_a_drive_letter_home_or_users_path():
    assert _scan() == []


def test_the_allow_list_starts_empty():
    assert ALLOWED == {}


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    for name in ("INTERLOCK_SCRATCH_ROOT", "WATCHDOG_HEARTBEAT", "INTERLOCK_WEIGHTS_HOT", "HF_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


def _toml(tmp_path: Path, body: str) -> None:
    (tmp_path / "interlock.toml").write_text(f"[paths]\n{body}\n", encoding="utf-8")


# ── scratch root (offrig calibrate --project) ────────────────────────────────────────────────────


def test_the_scratch_default_is_under_the_system_temp_directory():
    from interlock.adapters.offrig_calibrate import scratch_project

    expected = (Path(tempfile.gettempdir()) / "interlock-scratch").as_posix()
    assert scratch_project("cal") == f"{expected}/cal"


def test_the_scratch_root_comes_from_env_then_the_config_file(monkeypatch, tmp_path):
    from interlock.adapters.offrig_calibrate import scratch_project

    _toml(tmp_path, 'scratch_root = "cfg-scratch"')
    assert scratch_project("cal") == "cfg-scratch/cal"
    monkeypatch.setenv("INTERLOCK_SCRATCH_ROOT", "env-scratch")
    assert scratch_project("cal") == "env-scratch/cal"


# ── watchdog heartbeat (read by the VRAM watchdog guard; unset refuses, --no-watchdog is explicit) ─


def test_an_unset_heartbeat_is_none_not_a_default_path():
    from interlock.harness.runner import heartbeat_path

    assert heartbeat_path() is None


def test_the_heartbeat_comes_from_the_argument_env_or_config_file(monkeypatch, tmp_path):
    from interlock.harness.runner import heartbeat_path

    _toml(tmp_path, 'heartbeat = "cfg-hb"')
    assert heartbeat_path() == Path("cfg-hb")
    monkeypatch.setenv("WATCHDOG_HEARTBEAT", "env-hb")
    assert heartbeat_path() == Path("env-hb")
    assert heartbeat_path(Path("arg-hb")) == Path("arg-hb")


def test_an_unset_heartbeat_refuses_and_names_the_variable():
    from interlock.guards import GuardError, require_heartbeat

    with pytest.raises(GuardError, match="WATCHDOG_HEARTBEAT") as caught:
        require_heartbeat(None, 0.0)
    assert "--no-watchdog" in str(caught.value)


def test_the_card_guards_refuse_an_unset_heartbeat_before_the_card_probe():
    from interlock.guards import GuardError
    from interlock.harness.runner import _card_guards

    def probe():
        raise AssertionError("the card was probed")

    with pytest.raises(GuardError, match="WATCHDOG_HEARTBEAT"):
        _card_guards(probe, None, 0.0)


def test_only_the_no_watchdog_marker_skips_the_heartbeat_guard_and_the_card_guard_still_runs():
    from interlock.guards import NO_WATCHDOG, GuardError, require_heartbeat
    from interlock.harness.runner import _card_guards, heartbeat_path

    assert require_heartbeat(NO_WATCHDOG, 0.0) == 0.0
    assert heartbeat_path(NO_WATCHDOG) is NO_WATCHDOG
    _card_guards(lambda: [], NO_WATCHDOG, 0.0)
    with pytest.raises(GuardError, match="still on the card"):
        _card_guards(lambda: ["some-model"], NO_WATCHDOG, 0.0)


def test_a_configured_heartbeat_is_still_checked(tmp_path):
    from interlock.guards import GuardError
    from interlock.harness.runner import _card_guards

    with pytest.raises(GuardError, match="heartbeat"):
        _card_guards(lambda: [], tmp_path / "missing-heartbeat", 0.0)


def _captured_run(monkeypatch, extra: list[str]):
    from interlock.cli import main

    seen: dict = {}

    def fake_run_spec(*args, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr("interlock.harness.runner.run_spec", fake_run_spec)
    spec = ROOT / "experiments" / "e5" / "spec.yaml"
    assert main(["run", str(spec), "--root", str(ROOT), "--grant", "g", *extra]) == 0
    return seen


def test_the_run_command_passes_no_watchdog_only_when_the_flag_is_given(monkeypatch):
    from interlock.guards import NO_WATCHDOG

    without = _captured_run(monkeypatch, [])
    assert without["heartbeat"] is None  # unset stays unset, and the guard refuses it
    assert without["supervisor"].heartbeat is None
    given = _captured_run(monkeypatch, ["--no-watchdog"])
    assert given["heartbeat"] is NO_WATCHDOG
    assert given["supervisor"].heartbeat is NO_WATCHDOG


# ── the hot weights cache (an arm with hf_cache: hot) ────────────────────────────────────────────


def test_the_hot_cache_default_is_under_the_user_hugging_face_cache_and_the_dead_constants_are_gone():
    from interlock.harness import storage

    assert storage.hot_root() == Path.home() / ".cache" / "huggingface" / "hot"
    assert not hasattr(storage, "ARCHIVE_ROOT")
    assert not hasattr(storage, "HOT_ROOT")


def test_the_hot_cache_comes_from_env_then_the_config_file(monkeypatch, tmp_path):
    from interlock.harness import storage

    _toml(tmp_path, 'weights_hot = "cfg-hot"')
    assert storage.hot_root() == Path("cfg-hot")
    monkeypatch.setenv("INTERLOCK_WEIGHTS_HOT", "env-hot")
    assert storage.hot_root() == Path("env-hot")


def test_the_default_hot_cache_follows_hf_home(monkeypatch, tmp_path):
    from interlock.harness import storage

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert storage.hot_root() == tmp_path / "hf" / "hot"


def test_the_dry_run_line_for_a_hot_arm_names_the_setting_not_a_drive():
    from interlock.harness.runner import _storage_lines

    lines = _storage_lines({"arms": [{"id": "a", "hf_cache": "hot"}]})
    assert lines and "INTERLOCK_WEIGHTS_HOT" in lines[0]
    assert not DRIVE.search(lines[0])


def test_a_malformed_config_file_is_a_structured_error(tmp_path):
    from interlock.harness import storage

    (tmp_path / "interlock.toml").write_text("[paths\n", encoding="utf-8")
    with pytest.raises(storage.StorageError, match="interlock.toml"):
        storage.hot_root()


def test_a_paths_table_that_is_not_a_table_or_a_value_that_is_not_a_path_is_a_structured_error(tmp_path):
    from interlock.paths import PathsConfigError, configured

    (tmp_path / "interlock.toml").write_text('paths = "x"\n', encoding="utf-8")
    with pytest.raises(PathsConfigError, match="must be a table"):
        configured("NOT_SET_ANYWHERE", "heartbeat")
    for body in ('heartbeat = 3', 'heartbeat = "  "'):
        _toml(tmp_path, body)
        with pytest.raises(PathsConfigError, match="must be a path string"):
            configured("NOT_SET_ANYWHERE", "heartbeat")
    _toml(tmp_path, 'scratch_root = "x"')
    assert configured("NOT_SET_ANYWHERE", "heartbeat") is None


def test_the_run_command_reports_a_malformed_paths_file_on_stderr_and_exits_1(tmp_path, capsys):
    from interlock.cli import main

    (tmp_path / "interlock.toml").write_text("[paths\n", encoding="utf-8")
    spec = ROOT / "experiments" / "e5" / "spec.yaml"
    assert main(["run", str(spec), "--root", str(ROOT), "--grant", "g"]) == 1
    assert "interlock.toml" in capsys.readouterr().err
