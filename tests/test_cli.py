import json
from pathlib import Path

from switchyard.cli import main, run_report
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


def test_a_finished_run_names_a_loss_before_done():
    text = run_report(
        [
            {"id": "T/A/run-1", "device": "5090"},
            {"id": "T/D/run-1", "device": "NPU", "outcome": "device_lost"},
            {"id": "T/B/run-1", "device": "iGPU", "outcome": "hang"},
        ],
        finished=True,
    )
    assert text == "D NPU device_lost\nB iGPU hang\ndone\n"
    assert run_report([{"id": "T/A/run-1", "device": "5090"}], finished=True) == "done\n"
    stopped = run_report([{"id": "T/B/run-1", "device": "NPU", "outcome": "hang"}], finished=False)
    assert stopped == "B NPU hang\n"
    assert "done" not in stopped
    assert run_report([{"id": "bare", "device": "NPU", "outcome": "device_lost"}], finished=False) == "bare NPU device_lost\n"
    assert run_report([{"id": "T/A/run-1", "device": "5090"}], finished=False) == ""


def test_module_entrypoint_parses_check(tmp_path: Path, monkeypatch):
    import runpy
    import sys

    import pytest

    monkeypatch.setattr(sys, "argv", ["switchyard", "check", "--root", str(tmp_path)])
    with pytest.raises(SystemExit) as caught:
        runpy.run_module("switchyard", run_name="__main__")
    assert caught.value.code == 0
