"""Router item D: `route --concurrency`, a margin between distinct candidates, and fill-card rows out of generate.

The table is T4 r3's own (`tests/fixtures/route/r3_table.jsonl`): before this change `route` picked phi3.5's p512 c8 row
(decode bound 115.67, the c8 aggregate), put a margin between phi's c8 row and phi's own c4 row, and let the fill-card
row compete.
"""

from __future__ import annotations

import copy
import json
from datetime import date
from pathlib import Path

import pytest

from interlock.cli import main
from interlock.router.policy import Decision, Refusal, Request, decide

TABLE = Path(__file__).resolve().parent / "fixtures" / "route" / "r3_table.jsonl"


def _rows() -> list[dict]:
    return [json.loads(line) for line in TABLE.read_text(encoding="utf-8").splitlines() if line.strip()]


def _route(tmp_path: Path, capsys, *extra: str, rows: list[dict] | None = None) -> tuple[int, dict]:
    (tmp_path / "table").mkdir(exist_ok=True)
    table = tmp_path / "table" / "routing.jsonl"
    table.write_text("\n".join(json.dumps(r) for r in (rows if rows is not None else _rows())) + "\n", encoding="utf-8")
    code = main(
        ["route", "--task", "generate", "--device", "colab:T4", "--allow-remote", "--as-of", "2026-10-26", "--json",
         "--root", str(tmp_path), *extra]
    )
    return code, json.loads(capsys.readouterr().out)


def _shape(document: dict) -> str:
    return document["receipts"][0].rsplit("/", 2)[-2]  # e.g. p512-c8


def test_the_default_is_one_stream_and_picks_a_c1_row(tmp_path, capsys):
    code, document = _route(tmp_path, capsys)
    assert code == 0
    assert _shape(document).endswith("-c1")
    assert document["settings"]["concurrency"] == 1
    assert document["concurrency"] == 1 and document["ranked_field"] == "decode_tok_s"
    assert Request(task="generate", as_of=date(2026, 10, 26)).concurrency == 1


def test_concurrency_8_picks_the_c8_aggregate_and_says_what_it_ranked(tmp_path, capsys):
    code, document = _route(tmp_path, capsys, "--concurrency", "8")
    assert code == 0
    assert _shape(document) == "p512-c8"
    assert document["margin"]["bound"] == pytest.approx(115.7, abs=0.2)
    assert document["concurrency"] == 8
    assert any(
        note == "ranked at concurrency 8 on decode_tok_s, the pass aggregate across 8 concurrent request(s)"
        for note in document["notes"]
    )


def test_concurrency_4_picks_a_c4_row(tmp_path, capsys):
    code, document = _route(tmp_path, capsys, "--concurrency", "4")
    assert code == 0 and _shape(document) == "p512-c4"


def test_a_concurrency_with_no_rows_is_a_structured_refusal_naming_the_concurrencies_that_exist(tmp_path, capsys):
    code, document = _route(tmp_path, capsys, "--concurrency", "2")
    assert code == 1
    assert document["code"] == "no_candidate" and document["constraint"] == "concurrency"
    assert "Rows exist at concurrency 1, 4, 8; none at 2." in document["hint"]


def test_one_model_has_no_margin_between_its_own_shapes(tmp_path, capsys):
    # Before: "delta 18.343 over 97.325", phi's c8 row against phi's own c4 row.
    code, document = _route(tmp_path, capsys, "--concurrency", "8")
    margin = document["margin"]
    assert margin["next_bound"] is None and margin["delta"] is None and margin["next_id"] is None
    assert "no other candidate" in document["notes"]
    code, document = _route(tmp_path, capsys)  # c1: five other p-lengths of the same model are shapes, not rivals
    assert document["margin"]["next_id"] is None and "no other candidate" in document["notes"]


def _second_model(rows: list[dict], factor: float) -> list[dict]:
    """The same table's measured generate rows again as another model, slower by ``factor``."""

    copies = []
    for row in rows:
        if row["task"] != "generate" or row.get("outcome") or row["settings"].get("concurrency") is None:
            continue
        twin = copy.deepcopy(row)
        twin["id"] = row["id"].replace("phi3.5:3.8b", "qwen3:8b")
        twin["model"]["repo_or_tag"] = "qwen3:8b"
        twin["model"]["revision_or_digest"] = "9" * 64
        for slot in twin["artifacts"]["slots"]:
            if "throughput" in slot:
                slot["throughput"] *= factor
        copies.append(twin)
    return copies


def test_with_two_models_the_margin_is_between_them_not_between_two_shapes_of_one(tmp_path, capsys):
    rows = _rows()
    code, document = _route(tmp_path, capsys, "--concurrency", "8", rows=rows + _second_model(rows, 0.5))
    assert code == 0
    assert "phi3.5" in document["model"]
    margin = document["margin"]
    assert margin["next_id"] is not None and "qwen3" in margin["next_id"]  # the other candidate, never a phi shape
    assert margin["delta"] == pytest.approx(margin["bound"] - margin["next_bound"])
    assert "no other candidate" not in document["notes"]
    assert margin["next_bound"] == pytest.approx(0.5 * margin["bound"], rel=0.02)  # p512-c8 against qwen's p512-c8


def test_the_margin_uses_each_candidates_best_row_at_the_requested_shape(tmp_path, capsys):
    rows = _rows()
    twins = _second_model(rows, 0.9)  # the twin's p512 row is 0.9x phi's; its other lengths are far lower
    code, document = _route(tmp_path, capsys, "--concurrency", "1", rows=rows + twins)
    margin = document["margin"]
    assert "qwen3" in margin["next_id"] and "p512-c1" in margin["next_id"]
    assert margin["delta"] == pytest.approx(0.1 * margin["bound"], rel=0.05)


# -- fill-card rows ------------------------------------------------------------------------------------------------


def test_a_fill_card_row_never_wins_generate_and_is_listed_as_excluded(tmp_path, capsys):
    rows = _rows()
    fill = next(r for r in rows if "fill-card" in r["id"].lower())
    for row in rows:
        if row is not fill and row["task"] == "generate" and "artifacts" in row:
            for slot in row["artifacts"]["slots"]:
                if "throughput" in slot:
                    slot["throughput"] = 1.0  # make the fill-card row (71.6) the fastest thing in the table
    code, document = _route(tmp_path, capsys, rows=rows)
    assert code == 0
    assert "fill-card" not in document["receipts"][0].lower()
    assert {"id": fill["id"], "code": "fill_card"} in document["excluded"]


def test_a_table_with_only_a_fill_card_row_has_no_generate_candidate(tmp_path, capsys):
    rows = [r for r in _rows() if "fill-card" in r["id"].lower()]
    code, document = _route(tmp_path, capsys, rows=rows)
    assert code == 1 and document["code"] == "no_candidate"
    assert {"id": rows[0]["id"], "code": "fill_card"} in document["excluded"]


# -- the pre-v3 guard still holds, and other tasks are untouched ------------------------------------------------------


def test_pre_v3_rows_keep_their_guard_under_a_requested_concurrency():
    rows = _rows()
    old = [
        r for r in rows
        if r["task"] == "generate" and not r.get("outcome") and r["settings"].get("concurrency") == 8
    ]
    for row in old:
        row["settings"]["ladder_version"] = 2  # as r2 wrote them
    request = Request(
        task="generate", as_of=date(2026, 10, 26), model="phi3.5", allow_remote=True, device="colab:T4", concurrency=8
    )
    assert isinstance(decide(old, request), Refusal)  # a pre-v3 c8 throughput is never used


def test_other_tasks_ignore_the_concurrency(tmp_path, capsys):
    rows = _rows()
    request = Request(task="embed", as_of=date(2026, 10, 26), allow_remote=True, device="colab:T4", concurrency=8)
    result = decide(rows, request)
    assert isinstance(result, Decision) and "concurrency" not in result.document


def test_the_json_decision_validates_against_the_schema(tmp_path, capsys):
    from jsonschema import Draft202012Validator

    schema = json.loads(
        (Path(__file__).resolve().parents[1] / "src" / "interlock" / "schema" / "route-decision.v1.json").read_text(
            encoding="utf-8"
        )
    )
    for extra in ((), ("--concurrency", "8")):
        _, document = _route(tmp_path, capsys, *extra)
        Draft202012Validator(schema).validate(document)
    rows = _rows()
    _, document = _route(tmp_path, capsys, rows=rows + _second_model(rows, 0.5))
    Draft202012Validator(schema).validate(document)


def test_other_shapes_of_the_winners_own_candidate_are_not_listed_as_runners_up(tmp_path, capsys):
    _, document = _route(tmp_path, capsys, "--concurrency", "8")
    assert document["runners_up"] == []  # phi's p1024/p2048 c8 rows are shapes of the winner, not alternatives
    rows = _rows()
    _, with_rival = _route(tmp_path, capsys, "--concurrency", "8", rows=rows + _second_model(rows, 0.5))
    assert with_rival["runners_up"]
    assert all("qwen3" in item["id"] for item in with_rival["runners_up"])


def test_a_refusal_gives_each_measured_failure_reason_once_with_a_count(tmp_path, capsys):
    _, document = _route(tmp_path, capsys, "--concurrency", "2")
    hint = document["hint"]
    measured = hint.split("Measured failure reasons: ", 1)[1]
    reasons = measured.rstrip(".").split("; ")
    assert len(reasons) == len(set(reasons))  # no reason repeated
    assert any("(x" in reason for reason in reasons)  # the repeated Gemma reason, counted
