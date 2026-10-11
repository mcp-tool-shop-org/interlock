"""The route command: help, human output, and the input flags."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from switchyard.cli import main

ROOT = Path(__file__).resolve().parents[1]
DECISION = json.loads((ROOT / "src" / "switchyard" / "schema" / "route-decision.v1.json").read_text(encoding="utf-8"))


def write_table(tmp_path: Path, rows: list[dict]) -> None:
    table = tmp_path / "table"
    table.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row) for row in rows)
    (table / "routing.jsonl").write_text(text + "\n", encoding="utf-8")


def row(row_id, device, throughput, **extra):
    body = {
        "id": row_id,
        "task": "generate",
        "model": {"repo_or_tag": "phi:mini", "revision_or_digest": "abc"},
        "engine": {"name": "ollama", "version": "1"},
        "device": device,
        "settings": {"thinking_level": "off"},
        "date": "2026-01-01",
        "repeats": 1,
        "provisional": True,
        "artifacts": {"slots": [{"phase": "repeat", "throughput": throughput, "latency_ms": 100}]},
        "provenance": {"experiment_spec": "experiments/e3/spec.yaml", "run_id": "t"},
    }
    body.update(extra)
    return body


def test_route_help_names_the_constraints(capsys):
    with pytest.raises(SystemExit) as caught:
        main(["route", "--help"])
    assert caught.value.code == 0
    text = capsys.readouterr().out
    for needle in (
        "--task",
        "--model",
        "--min-<metric>",
        "--min-tok-s",
        "--max-latency-ms",
        "--max-ttft-ms",
        "--ctx",
        "--thinking",
        "--busy",
        "--budget-left-usd",
        "--json",
        "--as-of",
        "prefix_caching",
        "budget unchecked",
        "native_binary_rejected",
        "compile_rejected",
        "compile_timeout",
        "hang",
        "device_lost",
        "parity_invalid",
        "any recorded thinking level",
        "unintended",
        "excluded",
    ):
        assert needle in text


def test_route_prints_a_decision_for_a_person(tmp_path: Path, capsys):
    remote_settings = {
        "thinking_level": "off",
        "worst_case_usd": 4,
        "offrig_plan": {"profile": "small", "max_hours": 1},
    }
    write_table(
        tmp_path,
        [
            row("local/slow", "CPU", 5),
            row("local/lost", "iGPU", 1, outcome="hang"),
            row("remote/fast", "runpod:A40", 30, settings=remote_settings),
        ],
    )
    assert main(["route", "--root", str(tmp_path), "--task", "generate", "--as-of", "2026-10-11"]) == 0
    text = capsys.readouterr().out
    assert "runpod:A40" in text
    assert "phi:mini@abc" in text
    assert "margin decode_tok_s" in text
    assert "delta" in text
    assert "receipts: remote/fast" in text
    assert "flags: stale, provisional" in text
    assert "note: budget unchecked" in text
    assert "offrig_plan:" in text
    assert "budget: unchecked" in text
    assert "runner-up: local/slow" in text
    assert "excluded: local/lost\thang" in text

    write_table(tmp_path, [row("only/one", "iGPU", 9)])
    assert main(["route", "--root", str(tmp_path), "--task", "generate", "--as-of", "2026-10-11"]) == 0
    alone = capsys.readouterr().out
    assert "no runner-up" in alone
    assert "note: no 5090 row" in alone


def test_route_json_validates_and_a_floor_refuses(tmp_path: Path, capsys):
    write_table(tmp_path, [row("only/one", "CPU", 9), row("other/one", "iGPU", 4)])
    assert main(
        ["route", "--root", str(tmp_path), "--task", "generate", "--as-of", "2026-10-11", "--json", "--min-tok-s", "1"]
    ) == 0
    document = json.loads(capsys.readouterr().out)
    Draft202012Validator(DECISION).validate(document)
    assert document["runners_up"][0]["constraint"] == "decode_tok_s"

    assert main(
        ["route", "--root", str(tmp_path), "--task", "generate", "--as-of", "2026-10-11", "--json", "--min-tok-s", "100"]
    ) == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    refused = json.loads(captured.out)
    assert refused["constraint"] == "min_tok_s"
    assert refused["code"] == "no_candidate"


def test_route_refuses_on_stderr_when_a_person_is_reading(tmp_path: Path, capsys):
    write_table(tmp_path, [row("only/one", "CPU", 9)])
    assert main(["route", "--root", str(tmp_path), "--task", "generate", "--min-tok-s", "100"]) == 1
    err = capsys.readouterr().err
    assert "no row meets the constraints" in err
    assert "constraint: min_tok_s" in err
    assert "experiment: E3" in err
    assert "closest:" in err

    (tmp_path / "table" / "routing.jsonl").write_text("", encoding="utf-8")
    assert main(["route", "--root", str(tmp_path), "--task", "rerank"]) == 1
    empty = capsys.readouterr().err
    assert "Nothing was close" in empty
    assert "closest:" not in empty


def test_route_refuses_a_missing_table(tmp_path: Path, capsys):
    assert main(["route", "--root", str(tmp_path), "--task", "generate"]) == 1
    err = capsys.readouterr().err
    assert "no routing table" in err
    assert "experiment: unregistered" in err

    assert main(["route", "--root", str(tmp_path), "--task", "generate", "--json"]) == 1
    document = json.loads(capsys.readouterr().out)
    assert document["code"] == "no_table"
    assert document["experiment"] == "unregistered"
    assert document["closest"] is None


def test_route_rejects_a_bad_date_and_a_bad_floor(tmp_path: Path, capsys):
    write_table(tmp_path, [row("only/one", "CPU", 9)])
    assert main(["route", "--root", str(tmp_path), "--task", "generate", "--as-of", "yesterday"]) == 1
    assert "--as-of must be YYYY-MM-DD" in capsys.readouterr().err
    assert main(["route", "--root", str(tmp_path), "--task", "generate", "--min-tok-s", "nope"]) == 1
    assert "needs a number" in capsys.readouterr().err
    assert main(["route", "--root", str(tmp_path), "--task", "generate", "--min-tok-s"]) == 1
    assert "needs a number" in capsys.readouterr().err


def test_route_busy_does_not_leak_into_the_next_call(tmp_path: Path, capsys):
    write_table(tmp_path, [row("g/cpu", "CPU", 30), row("g/igpu", "iGPU", 10)])
    assert main(["route", "--root", str(tmp_path), "--task", "generate", "--busy", "CPU", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["device"] == "iGPU"
    assert main(["route", "--root", str(tmp_path), "--task", "generate", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["device"] == "CPU"


def test_an_unknown_task_is_a_usage_error():
    with pytest.raises(SystemExit) as caught:
        main(["route", "--task", "verify"])
    assert caught.value.code == 2
