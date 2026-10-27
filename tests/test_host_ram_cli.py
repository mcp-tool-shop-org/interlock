"""1.0.11: the two ``interlock run`` flags, ``--host-ram-projection`` and ``--host-ram-safety``.

They live in ``cli.py`` (the ``run`` parser, handed to ``run_spec``) and ``harness/runner.py`` (checked before any
device contact, then handed to ``run_linux_spec``). No GPU and no Colab session.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from interlock.cli import main
from interlock.harness.runner import RunnerError, run_spec
from tests import test_linux_cuda_harness_fixes as fixes
from tests.test_linux_cuda_harness_fixes import Ollama, _seams

GENERATE = "experiments/c2-generate/spec.yaml"
repo = fixes.repo  # the temporary git checkout of the C2 specs, as the 1.0.2 tests use


def _spec(repo: Path) -> str:
    return str(repo / GENERATE)


def test_help_documents_both_flags_with_their_defaults_and_what_off_means(capsys):
    with pytest.raises(SystemExit):
        main(["run", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "--host-ram-projection {on,off}" in text and "--host-ram-safety FACTOR" in text
    assert "on (the default)" in text and "off behaves as 1.0.10 did" in text and "mode off" in text
    assert "default 1.05" in text and "a finite number above 0" in text


def test_the_flags_default_to_none_and_a_given_value_is_passed_on(repo, monkeypatch):
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr("interlock.harness.runner.run_spec", lambda *args, **kwargs: seen.append(kwargs) or [])
    assert main(["run", _spec(repo), "--root", str(repo)]) == 0
    given = ["--host-ram-projection", "off", "--host-ram-safety", "1.3"]
    assert main(["run", _spec(repo), "--root", str(repo), *given]) == 0
    assert (seen[0]["host_ram_projection"], seen[0]["host_ram_safety"]) == (None, None)
    assert (seen[1]["host_ram_projection"], seen[1]["host_ram_safety"]) == ("off", "1.3")


@pytest.mark.parametrize("bad", ["0", "-1", "nan", "abc", "inf"])
def test_a_bad_factor_is_refused_with_the_structured_error_and_no_receipt(repo, capsys, bad):
    code = main(["run", _spec(repo), "--root", str(repo), "--host-ram-safety", bad, "--run-id", "bad"])
    err = capsys.readouterr().err
    assert code == 1
    assert err.startswith("host_ram_safety_invalid: ") and "greater than 0" in err and "\nhint: " in err
    assert not (repo / "receipts" / "bad.jsonl").exists()


def test_the_flags_reach_run_linux_spec(repo, monkeypatch):
    seen: list[dict[str, Any]] = []

    def fake(spec: Any, **kwargs: Any) -> list[dict]:
        seen.append(kwargs)
        return []

    monkeypatch.setattr("interlock.harness.linux_cuda.run_linux_spec", fake)
    for run_id, extra in (("a", {}), ("b", {"host_ram_projection": "off", "host_ram_safety": "1.3"})):
        run_spec(
            repo / GENERATE, root=repo, run_id=run_id, date="2026-10-27",
            python_path=str(repo.parent / "external-python"), seams=_seams(Ollama()), **extra,
        )
    assert (seen[0]["host_ram_projection"], seen[0]["host_ram_safety"]) == ("on", None)
    assert (seen[1]["host_ram_projection"], seen[1]["host_ram_safety"]) == ("off", "1.3")


def test_a_flag_on_a_spec_that_is_not_linux_cuda_is_refused(tmp_path):
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(RunnerError, match="linux-cuda"):
        run_spec(
            root / "experiments" / "e4b" / "spec.yaml", root=root, host_ram_safety="1.1",
            python_path=str(tmp_path / "external-python"),
        )
