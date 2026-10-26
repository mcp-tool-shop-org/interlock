"""Speeds from different generate-ladder versions are different measurements and are never ranked together."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from interlock.cli import main
from interlock.router.policy import Decision, Request, decide, ladder_version


def row(row_id: str, device: str, throughput: float, version: int | None = None) -> dict:
    settings: dict = {"thinking_level": "off", "num_predict": 128, "num_ctx": 2048}
    if version is not None:
        settings["ladder_version"] = version
    return {
        "id": row_id,
        "task": "generate",
        "model": {"repo_or_tag": "phi3.5:3.8b", "revision_or_digest": "abc"},
        "engine": {"name": "ollama", "version": "0.6.2"},
        "device": device,
        "settings": settings,
        "date": "2026-10-26",
        "repeats": 3,
        "provisional": False,
        "artifacts": {"slots": [{"phase": "repeat", "throughput": throughput, "latency_ms": 100}] * 3},
        "provenance": {"experiment_spec": "experiments/c2-generate/spec.yaml", "run_id": "t"},
    }


def request(**extra) -> Request:
    return Request(task="generate", as_of=date(2026, 10, 26), model="phi3.5:3.8b", allow_remote=True, **extra)


def winner(result) -> str:
    assert isinstance(result, Decision)
    return result.document["receipts"][0]


def test_a_row_with_no_ladder_version_on_a_colab_device_is_version_1_and_a_rig_row_has_none():
    assert ladder_version(row("a", "colab:T4", 1.0)) == 1
    assert ladder_version(row("a", "colab:T4", 1.0, 2)) == 2
    assert ladder_version(row("a", "5090", 1.0)) is None


def test_on_one_device_and_model_only_the_newest_version_is_a_candidate_even_when_the_old_one_is_faster():
    rows = [row("t4-v1", "colab:T4", 300.0), row("t4-v2", "colab:T4", 100.0, 2)]
    result = decide(rows, request())
    assert winner(result) == "t4-v2"
    assert "1 older-ladder row(s) not compared" in result.document["notes"]
    assert "t4-v1" not in {r["id"] for r in result.document["runners_up"]}


def test_v1_and_v2_rows_on_different_devices_are_never_ranked_against_each_other():
    rows = [row("l4-v1", "colab:L4", 300.0), row("t4-v2", "colab:T4", 100.0, 2)]
    result = decide(rows, request())
    assert winner(result) == "t4-v2"
    assert any("ranked on ladder version 2 only; version(s) 1 not compared" in n for n in result.document["notes"])
    assert "l4-v1" not in {r["id"] for r in result.document["runners_up"]}


def test_a_device_that_only_has_version_1_still_routes_when_it_is_the_one_asked_for():
    rows = [row("l4-v1", "colab:L4", 300.0), row("t4-v2", "colab:T4", 100.0, 2)]
    assert winner(decide(rows, request(device="colab:L4"))) == "l4-v1"


def test_a_rig_row_with_no_ladder_is_still_compared_with_the_newest_ladder_rows():
    rows = [row("rig", "5090", 150.0), row("t4-v1", "colab:T4", 400.0), row("t4-v2", "colab:T4", 100.0, 2)]
    result = decide(rows, request())
    assert winner(result) == "rig"
    assert "t4-v1" not in {r["id"] for r in result.document["runners_up"]}


def test_the_table_groups_ladder_rows_and_shows_the_version(tmp_path: Path, capsys):
    table = tmp_path / "table"
    table.mkdir()
    rows = [
        row("t4-v2", "colab:T4", 100.0, 2),
        row("rig", "5090", 150.0),
        row("t4-v1", "colab:T4", 300.0),
        row("l4-v2", "colab:L4", 90.0, 2),
    ]
    (table / "routing.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert main(["table", "--root", str(tmp_path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line.split("\t")[0] for line in lines[:-1]] == ["rig", "l4-v2", "t4-v1", "t4-v2"]
    assert lines[0].count("ladder") == 0
    assert lines[1].endswith("ladder v2") and lines[2].endswith("ladder v1") and lines[3].endswith("ladder v2")
