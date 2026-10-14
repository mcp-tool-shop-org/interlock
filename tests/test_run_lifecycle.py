"""Run names, a stopped run's receipt, and the quarantine. Fakes only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_e4b import E4B, PAIRS, ROOT, _clean, _rows, _seams, _Server, _spec

from interlock.cli import main
from interlock.harness.e4b import run_block, run_from_cli, run_pilot
from interlock.harness.lifecycle import (
    LifecycleError,
    format_runs,
    list_runs,
    move_aside,
)
from interlock.harness.receipts import import_receipt
from interlock.receipt import receipt_paths


def _call(counter):
    def call(arm, *, num_predict, measured, kind, prompt):
        counter["n"] += 1
        return _clean(arm, num_predict=num_predict, kind=kind)

    return call


def _place(root: Path, kind: str, run_id: str) -> str:
    if kind == "receipt":
        path = root / "receipts" / f"{run_id}.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("{}\n", encoding="utf-8")
        return f"receipts/{run_id}.jsonl"
    if kind == "series":
        path = root / "receipts" / f"{run_id}-series"
        path.mkdir(parents=True)
        return f"receipts/{run_id}-series"
    if kind == "log":
        path = root / "logs" / f"{run_id}.log"
        path.parent.mkdir(parents=True)
        path.write_text("log\n", encoding="utf-8")
        return f"logs/{run_id}.log"
    if kind == "pilot-log":
        path = root / "pilot-e4b" / f"{run_id}.log"
        path.parent.mkdir(parents=True)
        path.write_text("log\n", encoding="utf-8")
        return f"pilot-e4b/{run_id}.log"
    if kind == "cache":
        path = root / "compile-cache" / run_id
        path.mkdir(parents=True)
        return f"compile-cache/{run_id}"
    if kind == "quarantine":
        path = root / "aborted" / f"{run_id}-abc1234"
        path.mkdir(parents=True)
        return f"aborted/{run_id}-abc1234"
    if kind == "pilot-json":
        path = root / "pilot-e4b" / f"{run_id}.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}\n", encoding="utf-8")
        return f"pilot-e4b/{run_id}.json"
    raise AssertionError(kind)


@pytest.mark.parametrize(
    "kind",
    ["receipt", "series", "log", "pilot-log", "cache", "quarantine", "pilot-json"],
)
def test_an_explicit_run_id_in_use_is_refused(tmp_path: Path, kind: str):
    needle = _place(tmp_path, kind, "taken")
    counter = {"n": 0}
    with pytest.raises(LifecycleError, match="in use") as caught:
        run_from_cli(
            _spec([PAIRS[0]]),
            spec_path=E4B,
            root=tmp_path,
            repo=ROOT,
            grant="grant",
            npu_session="session",
            run_id="taken",
            pilot=False,
            seams=_seams(_Server(), _call(counter)),
        )
    assert needle in str(caught.value)
    assert counter["n"] == 0


def test_an_omitted_run_id_picks_the_next_free_one(tmp_path: Path):
    spec = _spec([PAIRS[0]])
    first = run_from_cli(
        spec,
        spec_path=E4B,
        root=tmp_path,
        repo=ROOT,
        grant="grant",
        npu_session="session",
        run_id=None,
        pilot=False,
        seams=_seams(_Server(), _call({"n": 0})),
    )
    assert first
    header = _rows(tmp_path / "receipts" / "e4b-block.jsonl")[0]
    assert header["run_id"] == "e4b-block"
    second = run_from_cli(
        spec,
        spec_path=E4B,
        root=tmp_path,
        repo=ROOT,
        grant="grant",
        npu_session="session",
        run_id=None,
        pilot=False,
        seams=_seams(_Server(), _call({"n": 0})),
    )
    assert second
    again = _rows(tmp_path / "receipts" / "e4b-block-2.jsonl")[0]
    assert again["run_id"] == "e4b-block-2"


def test_an_omitted_run_id_is_the_first_line(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr("interlock.ollama_supervisor.real_seams", lambda **kwargs: object())
    monkeypatch.setattr("interlock.ollama_supervisor.OllamaSupervisor", lambda *args, **kwargs: object())
    monkeypatch.setattr("interlock.harness.runner.run_spec", lambda *args, **kwargs: [])
    assert main(["run", str(E4B), "--root", str(tmp_path), "--grant", "g", "--npu-session", "s"]) == 0
    assert capsys.readouterr().out == "run id: e4b-block\n"


def test_a_run_whose_cleanup_fails_still_writes_its_receipt(tmp_path: Path, monkeypatch):
    def refuse(path, onexc=None):
        raise PermissionError(5, "Access is denied", str(path))

    monkeypatch.setattr("interlock.harness.e4b.shutil.rmtree", refuse)
    run_block(
        _spec([PAIRS[0]]),
        _seams(_Server(), _call({"n": 0})),
        root=tmp_path,
        run_id="kept",
        repo=tmp_path,
    )
    receipt = tmp_path / "receipts" / "kept.jsonl"
    assert receipt.is_file()
    header = _rows(receipt)[0]
    assert header["aborted"] is False
    assert header["compile_cache_cleanup_failed"]
    assert header["compile_cache_cleanup_failed"][0].startswith("compile-cache/kept/")


def test_a_stopped_run_keeps_finished_rows_in_quarantine(tmp_path: Path):
    run_id = "stopped"

    def call(arm, *, num_predict, measured, kind, prompt):
        if arm["id"] == "5090-alone":
            (tmp_path / "logs").mkdir(exist_ok=True)
            (tmp_path / "logs" / f"{run_id}.log").write_text("log\n", encoding="utf-8")
            (tmp_path / "receipts").mkdir(exist_ok=True)
            (tmp_path / "receipts" / f"{run_id}.run.json").write_text("{}\n", encoding="utf-8")
            return _clean(arm, num_predict=num_predict, kind=kind)
        raise RuntimeError("stopped after the cold slot")

    with pytest.raises(RuntimeError, match="stopped after the cold slot"):
        run_block(_spec([PAIRS[0]]), _seams(_Server(), call), root=tmp_path, run_id=run_id, repo=tmp_path)
    assert not (tmp_path / "receipts" / f"{run_id}.jsonl").exists()
    assert not (tmp_path / "receipts" / f"{run_id}-series").exists()
    assert not (tmp_path / "receipts" / f"{run_id}.run.json").exists()
    assert not (tmp_path / "logs" / f"{run_id}.log").exists()
    parked = next((tmp_path / "aborted").glob(f"{run_id}-*"))
    rows = _rows(parked / f"{run_id}.jsonl")
    assert rows[0]["aborted"] is True
    assert rows[0]["run_id"] == run_id
    assert any(row.get("kind") == "e4b-cold" and row.get("arm") == "5090-alone" for row in rows)
    note = json.loads((parked / "NOTE.json").read_text(encoding="utf-8"))
    assert note["run_id"] == run_id
    assert note["exception_type"] == "RuntimeError"
    assert "stopped after the cold slot" in note["message"]
    assert note["harness_sha"] == "abc123"
    assert note["started"] and note["ended"]
    assert any(item.startswith("receipts/") for item in note["files"])
    assert (parked / f"{run_id}-series").is_dir()


def test_a_keyboard_interrupt_is_a_quarantined_abort(tmp_path: Path):
    run_id = "ctrlc"

    def call(arm, *, num_predict, measured, kind, prompt):
        if arm["id"] == "5090-alone":
            return _clean(arm, num_predict=num_predict, kind=kind)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_block(_spec([PAIRS[0]]), _seams(_Server(), call), root=tmp_path, run_id=run_id, repo=tmp_path)
    assert not (tmp_path / "receipts" / f"{run_id}.jsonl").exists()
    parked = next((tmp_path / "aborted").glob(f"{run_id}-*"))
    rows = _rows(parked / f"{run_id}.jsonl")
    assert rows[0]["aborted"] is True
    assert rows[0]["run_id"] == run_id
    assert "cleanup_errors" not in rows[0]
    assert any(row.get("kind") == "e4b-cold" and row.get("arm") == "5090-alone" for row in rows)
    assert list_runs(tmp_path) == [(run_id, "aborted")]
    note = json.loads((parked / "NOTE.json").read_text(encoding="utf-8"))
    assert note["exception_type"] == "KeyboardInterrupt"


def test_an_abort_keeps_its_error_when_unload_also_raises(tmp_path: Path):
    run_id = "unload-raises"

    def call(arm, *, num_predict, measured, kind, prompt):
        if arm["id"] == "5090-alone":
            return _clean(arm, num_predict=num_predict, kind=kind)
        raise RuntimeError("stopped after the cold slot")

    seams = _seams(_Server(), call)

    def unload():
        raise RuntimeError("unload refused")

    seams.unload_resident = unload
    original = seams.probe

    def probe(method, path):
        if path == "/api/version" and ("GET", "/api/version") in seams.probes:
            raise RuntimeError("version unread")
        return original(method, path)

    seams.probe = probe
    with pytest.raises(RuntimeError, match="stopped after the cold slot"):
        run_block(_spec([PAIRS[0]]), seams, root=tmp_path, run_id=run_id, repo=tmp_path)
    assert not (tmp_path / "receipts" / f"{run_id}.jsonl").exists()
    parked = next((tmp_path / "aborted").glob(f"{run_id}-*"))
    rows = _rows(parked / f"{run_id}.jsonl")
    assert rows[0]["aborted"] is True
    assert any("unload refused" in item for item in rows[0]["cleanup_errors"])
    assert any("version unread" in item for item in rows[0]["cleanup_errors"])
    assert any(row.get("kind") == "e4b-cold" and row.get("arm") == "5090-alone" for row in rows)
    note = json.loads((parked / "NOTE.json").read_text(encoding="utf-8"))
    assert note["exception_type"] == "RuntimeError"
    assert "stopped after the cold slot" in note["message"]


def test_a_stopped_pilot_is_quarantined(tmp_path: Path):
    def call(arm, *, num_predict, measured, kind, prompt):
        raise RuntimeError("pilot stopped")

    with pytest.raises(RuntimeError, match="pilot stopped"):
        run_pilot(
            _spec([PAIRS[0]]),
            _seams(_Server(), call),
            root=tmp_path,
            run_id="pilot-stop",
            repo=tmp_path,
            calls=1,
        )
    assert not (tmp_path / "pilot-e4b" / "pilot-stop.json").exists()
    parked = next((tmp_path / "aborted").glob("pilot-stop-*"))
    document = json.loads((parked / "pilot-stop.json").read_text(encoding="utf-8"))
    assert document["aborted"] is True
    assert document["run_id"] == "pilot-stop"
    note = json.loads((parked / "NOTE.json").read_text(encoding="utf-8"))
    assert note["exception_type"] == "RuntimeError"


def test_a_stray_series_refuses_before_any_device(tmp_path: Path):
    stray = tmp_path / "receipts" / "left-series"
    stray.mkdir(parents=True)
    (stray / "a.jsonl").write_text("{}\n", encoding="utf-8")
    counter = {"n": 0}
    with pytest.raises(LifecycleError, match="left-series") as caught:
        run_from_cli(
            _spec([PAIRS[0]]),
            spec_path=E4B,
            root=tmp_path,
            repo=ROOT,
            grant="grant",
            npu_session="session",
            run_id="fresh-name",
            pilot=False,
            seams=_seams(_Server(), _call(counter)),
        )
    assert "interlock runs quarantine left" in str(caught.value)
    assert counter["n"] == 0


def test_runs_list_reports_complete_aborted_and_stray(tmp_path: Path, capsys):
    (tmp_path / "receipts").mkdir()
    (tmp_path / "receipts" / "e4b-block.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "receipts" / "left-series").mkdir()
    move_aside(
        tmp_path,
        "e4b-pilot",
        sha="abc1234",
        started="2026-10-12T00:00:00+00:00",
        ended="2026-10-12T00:01:00+00:00",
        exc_type="RuntimeError",
        message="stopped",
        include_receipt=False,
    )
    # move_aside returns None when nothing exists. Plant a log, then move it.
    log = tmp_path / "logs" / "e4b-pilot.log"
    log.parent.mkdir()
    log.write_text("log\n", encoding="utf-8")
    move_aside(
        tmp_path,
        "e4b-pilot",
        sha="abc1234",
        started="2026-10-12T00:00:00+00:00",
        ended="2026-10-12T00:01:00+00:00",
        exc_type="RuntimeError",
        message="stopped",
        include_receipt=False,
    )
    text = format_runs(list_runs(tmp_path))
    assert text == "e4b-block\tcomplete\ne4b-pilot\taborted\nleft\tstray\n"
    assert main(["runs", "list", "--root", str(tmp_path)]) == 0
    assert capsys.readouterr().out == text


def test_runs_quarantine_moves_a_stray_and_writes_the_hand_note(tmp_path: Path):
    series = tmp_path / "receipts" / "left-series"
    series.mkdir(parents=True)
    (series / "a.jsonl").write_text("{}\n", encoding="utf-8")
    assert main(["runs", "quarantine", "left", "--root", str(tmp_path)]) == 0
    assert not series.exists()
    note_path = next((tmp_path / "aborted").glob("left-*/NOTE.json"))
    note = json.loads(note_path.read_text(encoding="utf-8"))
    assert note["message"] == "quarantined by hand"
    assert note["quarantined_by"] == "hand"
    assert note["run_id"] == "left"
    assert "receipts/left-series" in note["files"]
    assert not (tmp_path / "receipts" / "left-series").exists()


def test_a_quarantined_run_is_excluded_from_the_table(tmp_path: Path):
    receipt = tmp_path / "receipts" / "aside.jsonl"
    receipt.parent.mkdir()
    receipt.write_text('{"id":"should-not-route"}\n', encoding="utf-8")
    series = tmp_path / "receipts" / "aside-series"
    series.mkdir()
    (series / "s.jsonl").write_text("{}\n", encoding="utf-8")
    move_aside(
        tmp_path,
        "aside",
        sha="abc1234",
        started="2026-10-12T00:00:00+00:00",
        ended="2026-10-12T00:01:00+00:00",
        exc_type="RuntimeError",
        message="stopped",
        include_receipt=True,
    )
    receipt.write_text('{"not":"a row"}\n', encoding="utf-8")
    table = tmp_path / "table" / "routing.jsonl"
    table.parent.mkdir()
    table.write_text("", encoding="utf-8")
    assert import_receipt(tmp_path, "aside") == []
    assert table.read_text(encoding="utf-8") == ""
    parked = next(tmp_path.glob("aborted/aside-*"))
    (parked / "decoy.json").write_text("{}\n", encoding="utf-8")
    assert all("aborted" not in path.as_posix() for path in receipt_paths(tmp_path))
