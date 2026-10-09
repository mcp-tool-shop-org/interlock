import json
from pathlib import Path

from switchyard.cli import main
from tests.support import row


def test_check_is_green_on_an_empty_root(tmp_path: Path, capsys):
    assert main(["check", "--root", str(tmp_path)]) == 0
    assert "0 receipt(s), ok" in capsys.readouterr().out


def test_check_fails_on_a_bad_receipt(tmp_path: Path, capsys):
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    (receipts / "bad.json").write_text(json.dumps(row(device="AUTO")), encoding="utf-8")
    assert main(["check", "--root", str(tmp_path)]) == 1
    assert "problem" in capsys.readouterr().err


def test_module_entrypoint_parses_check(tmp_path: Path, monkeypatch):
    import runpy
    import sys

    import pytest

    monkeypatch.setattr(sys, "argv", ["switchyard", "check", "--root", str(tmp_path)])
    with pytest.raises(SystemExit) as caught:
        runpy.run_module("switchyard", run_name="__main__")
    assert caught.value.code == 0
