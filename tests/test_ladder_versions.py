"""Speeds from different generate-ladder versions are different measurements and are never ranked together."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from interlock.cli import main
from interlock.devices import RemoteDeviceIdentity
from interlock.router.policy import Decision, Refusal, Request, decide, ladder_version


def row(row_id: str, device: str, throughput: float, version: int | None = None) -> dict:
    settings: dict = {"thinking_level": "off", "num_predict": 128, "num_ctx": 2048, "concurrency": 1}
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


def fill_card_row(row_id: str, device: str, max_ctx: int) -> dict:
    body = row(row_id, device, 0.0)
    body["settings"] = {"thinking_level": "off", "num_predict": None, "num_ctx": max_ctx, "max_resident_ctx": max_ctx}
    return body


def test_a_fill_card_row_is_not_a_ladder_row_and_keeps_routing_beside_v2_rows():
    fill = fill_card_row("t4-fill", "colab:T4", 22528)
    assert ladder_version(fill) is None
    rows = [fill, row("t4-v2", "colab:T4", 100.0, 2)]
    result = decide(rows, request(ctx=16384))
    assert winner(result) == "t4-fill"


def test_a_v1_row_cannot_win_because_the_v2_row_failed_a_floor():
    # The v2 row is slower than the floor; the v1 row (faster) is an older ladder and must not take its place.
    v1, v2 = row("t4-v1", "colab:T4", 300.0), row("t4-v2", "colab:T4", 100.0, 2)
    result = decide([v1, v2], request(min_metrics={"decode_tok_s": 200.0}))
    assert isinstance(result, Refusal)
    named = json.dumps(result.document)
    assert "t4-v1" not in named


def test_a_v1_row_is_not_listed_as_a_runner_up_when_the_v2_row_wins():
    other = row("l4-v2", "colab:L4", 50.0, 2)
    result = decide([row("t4-v1", "colab:T4", 300.0), row("t4-v2", "colab:T4", 100.0, 2), other], request())
    assert winner(result) == "t4-v2"
    assert "t4-v1" not in json.dumps(result.document)


def test_the_table_copy_of_every_unmeasured_row_does_not_say_the_device_cannot():
    from interlock.harness.linux_cuda import build_linux_failure_row
    from interlock.harness.receipts import for_table

    ident = RemoteDeviceIdentity(
        provider="colab", card="T4", vram_mib=15360, driver_version="550.54.15", cuda_version="12.4",
        pool="us-central1-a",
    )
    for kind in ("sampler_error", "generation_aborted", "unmeasured"):
        failure = build_linux_failure_row(
            spec_id="C2-GENERATE", spec_rel="experiments/c2-generate/spec.yaml", run_id="r", date="2026-10-26",
            task="generate", model="phi3.5:3.8b", revision_or_digest="d" * 64, remote_device=ident,
            failure_class=kind, reason="not measured", quant="Q4_K_M", licence="mit", num_ctx=2048,
            prompt_tokens=2048, concurrency=1,
        )
        assert "can't do this setting" not in for_table(failure)["notes"], kind


def test_the_legacy_envelope_importer_keeps_the_ladder_version_and_the_prompt_method():
    from interlock.importers.remote import import_remote_window

    entry = {
        "task": "generate", "model": "phi3.5:3.8b", "digest": "d" * 64, "quant": "Q4_K_M", "licence": "mit",
        "num_ctx": 2048, "prompt_tokens": 2051, "prompt_tokens_target": 2048,
        "prompt_token_method": "ollama_prompt_eval_count", "ladder_version": 2, "concurrency": 1,
        "ttft_ms": 10.0, "decode_tok_s": 30.0, "latency_ms": 100.0, "peak_mib": 6000, "num_predict": 128,
        "size": 5, "size_vram": 5,
    }
    envelope = {
        "run_id": "t4-r2", "date": "2026-10-26", "records": [entry],
        "remote_device": {"provider": "colab", "card": "T4", "vram_mib": 15360, "driver_version": "550.54.15",
                          "cuda_version": "12.4", "pool": "us-central1-a"},
    }
    try:
        rows = import_remote_window(json.dumps(envelope).encode("utf-8"), source_path="receipts/w.json")
    except Exception as exc:  # the envelope shape is the importer's; show it rather than guess
        raise AssertionError(f"envelope not accepted: {exc}") from exc
    assert rows[0]["settings"]["ladder_version"] == 2
    assert rows[0]["settings"]["prompt_token_method"] == "ollama_prompt_eval_count"  # noqa: S105
