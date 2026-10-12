"""Task scope, the family table, and the choice-rule readout.

The numbers in here are fixtures. They are not ASPIRE's judge results.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from switchyard.families import FamilyError, family_of, stamp_family
from switchyard.harness.choice import choose
from switchyard.harness.spec import SpecError, validate_spec
from switchyard.importers.gaps import open_gaps
from switchyard.receipt import ReceiptError, is_speed_row, schema, validate_row
from switchyard.scope import ScopeError, select
from switchyard.tasks import JUDGE_RATES, TASKS
from tests.support import row

ROOT = Path(__file__).resolve().parents[1]


def _rate(name: str, value: float) -> dict:
    return {
        "name": name,
        "value": value,
        "n": 10,
        "interval": {"low": 0.0, "high": 1.0, "method": "fixture"},
        "dataset_hash": "abc",
    }


def _spec(**overrides):
    data = {
        "id": "T",
        "procedure": "interleaved",
        "question": "Does the choice rule stay data?",
        "decision": "The harness reads the rule and does not re-score.",
        "needs": {"grant": True, "npu_session": False, "cpu_window": False},
        "stop_rules": ["a failed step stops the sequence"],
        "dataset": {"name": "fixture"},
        "metrics": ["consensus_agreement"],
        "warmups": 0,
        "repeats": 1,
        "band": 0.1,
        "compile_cache": "reuse",
        "rest_s": 900,
        "arms": [
            {
                "id": "A",
                "task": "judge:planted-error",
                "device": "5090",
                "engine": "ollama",
                "model": "qwen3:8b",
                "revision_or_digest": "abc",
                "licence": "unrecorded",
            }
        ],
    }
    data.update(overrides)
    return data


def _rule(**overrides):
    rule = {
        "eligibility_floor": {"metric": "consensus_agreement", "minimum": 0.5},
        "ranking_key": "consensus_agreement",
        "tie_rule": {"toward": "row-committed"},
        "fallback": "row-fallback",
    }
    rule.update(overrides)
    return rule


def _choice_row(row_id: str, agreement: float, *, sha: str | None = None) -> dict:
    item = {"id": row_id, "quality": [_rate("consensus_agreement", agreement), _rate("false_reject_rate", 0.0)]}
    if sha is not None:
        item["provenance"] = {"commit_sha": sha}
    return item


def test_the_schema_task_list_is_the_code_task_list():
    assert set(schema()["properties"]["task"]["enum"]) == TASKS
    assert "verify" not in TASKS


def test_a_task_is_not_answered_from_another_task():
    grounded = {"id": "g", "task": "verify:grounded", "model_family": "granite"}
    ladder = {"id": "m", "task": "verify:math-ladder", "model_family": "granite"}
    direct = select([grounded, ladder], "verify:grounded", allow_proxy=True)
    assert direct.evidence == "direct"
    assert direct.rows == (grounded,)
    assert direct.satisfies_hard_constraint is True
    refused = select([ladder], "verify:grounded")
    assert refused.evidence == "none"
    assert refused.rows == ()
    assert refused.satisfies_hard_constraint is False
    proxy = select([ladder], "verify:grounded", allow_proxy=True)
    assert proxy.evidence == "proxy"
    assert proxy.source_tasks == ("verify:math-ladder",)
    assert proxy.satisfies_hard_constraint is False
    assert proxy.rows == (ladder,)


def test_device_lost_is_not_a_speed_answer():
    lost = {"id": "e", "task": "embed", "outcome": "device_lost", "model_family": "qwen"}
    hung = {"id": "h", "task": "embed", "outcome": "hang", "model_family": "qwen"}
    real = {"id": "r", "task": "embed", "model_family": "qwen"}
    answer = select([lost, hung, real], "embed")
    assert answer.rows == (real,)
    assert answer.satisfies_hard_constraint is True
    only = select([lost, hung], "embed", allow_proxy=True)
    assert only.evidence == "none"
    assert only.rows == ()


def test_a_failed_parity_guard_is_not_a_speed_answer():
    bad = {"id": "b", "task": "embed", "parity_valid": False, "model_family": "qwen"}
    good = {"id": "g", "task": "embed", "parity_valid": True, "model_family": "qwen"}
    plain = {"id": "p", "task": "embed", "model_family": "qwen"}
    answer = select([bad, good, plain], "embed")
    assert [row["id"] for row in answer.rows] == ["g", "p"]
    assert answer.satisfies_hard_constraint is True
    only = select([bad], "embed", allow_proxy=True)
    assert only.evidence == "none"
    assert only.rows == ()


def test_an_unknown_task_and_a_family_string_are_refused():
    with pytest.raises(ScopeError, match="unknown task"):
        select([], "verify")
    with pytest.raises(ScopeError, match="exclude_families"):
        select([], "embed", exclude_families="qwen")


def test_excluding_a_family_drops_that_lineage_and_keeps_an_unlisted_tag():
    qwen = {"id": "q", "task": "judge:planted-error", "model_family": "qwen"}
    gemma = {"id": "g", "task": "judge:planted-error", "model_family": "gemma"}
    bge = {"id": "b", "task": "embed"}
    answer = select([qwen, gemma, bge], "judge:planted-error", exclude_families=("qwen",))
    assert answer.evidence == "direct"
    assert answer.rows == (gemma,)
    embed = select([qwen, bge], "embed", exclude_families=("qwen",))
    assert embed.rows == (bge,)
    assert embed.satisfies_hard_constraint is True
    empty = select([qwen], "judge:planted-error", exclude_families=("qwen",), allow_proxy=True)
    assert empty.evidence == "none"


def test_family_lookup_is_the_exact_tag():
    assert family_of("qwen3:8b") == "qwen"
    assert family_of("qwen3:14b") == "qwen"
    assert family_of("gemma4:31b") == "gemma"
    assert family_of("granite4.1:30b") == "granite"
    assert family_of("llama3.1:8b") == "llama"
    assert family_of("mistral-small:24b") == "mistral"
    assert family_of("qwen3") is None
    assert family_of("qwen3:8b-extra") is None
    assert family_of("BAAI/bge-base-en-v1.5") is None
    with pytest.raises(FamilyError, match="string"):
        family_of(None)
    stamped = {"model": {"repo_or_tag": "qwen3:8b"}, "model_family": "gemma"}
    stamp_family(stamped)
    assert stamped["model_family"] == "qwen"
    plain = {"model": {"repo_or_tag": "BAAI/bge-base-en-v1.5"}, "model_family": "qwen"}
    stamp_family(plain)
    assert "model_family" not in plain


def test_a_row_must_agree_with_the_family_table():
    missing = row()
    missing["model"]["repo_or_tag"] = "qwen3:8b"
    with pytest.raises(ReceiptError, match="from the family table"):
        validate_row(missing)
    missing["model_family"] = "gemma"
    with pytest.raises(ReceiptError, match="from the family table"):
        validate_row(missing)
    missing["model_family"] = "qwen"
    validate_row(missing)
    invented = row()
    invented["model_family"] = "qwen"
    with pytest.raises(ReceiptError, match="must be absent"):
        validate_row(invented)


def test_a_broken_family_table_refuses_the_row(monkeypatch, tmp_path: Path):
    target = tmp_path / "model-families.json"
    monkeypatch.setattr("switchyard.families.families_path", lambda: target)
    with pytest.raises(ReceiptError, match="family table"):
        validate_row(row())
    target.write_text("[]", encoding="utf-8")
    with pytest.raises(ReceiptError, match="family table"):
        validate_row(row())
    target.write_text("{}", encoding="utf-8")
    with pytest.raises(ReceiptError, match="family table"):
        validate_row(row())
    target.write_text('{"tag": 1}', encoding="utf-8")
    with pytest.raises(ReceiptError, match="family table"):
        validate_row(row())


def test_judge_fields_are_optional_and_residency_follows_the_two_sizes():
    judged = row()
    judged["quality"] = [_rate(name, 0.1) for name in JUDGE_RATES]
    judged["per_type"] = {"quoted": _rate("false_reject_rate", 0.0)}
    judged["cost"] = {"seconds": 1.25, "tokens": 30}
    validate_row(judged)
    names = [metric["name"] for metric in judged["quality"]]
    assert "false_reject_rate" in names and "false_accept_rate" in names
    broken = row()
    broken["per_type"] = {"quoted": _rate("false_reject_rate", 0.2)}
    broken["per_type"]["quoted"]["interval"] = {"low": 0.9, "high": 0.1, "method": "fixture"}
    with pytest.raises(ReceiptError, match="interval"):
        validate_row(broken)
    resident = row()
    resident["model"]["size"] = 10
    resident["model"]["size_vram"] = 10
    with pytest.raises(ReceiptError, match="must record"):
        validate_row(resident)
    resident["gpu_resident"] = False
    with pytest.raises(ReceiptError, match="equals size"):
        validate_row(resident)
    resident["gpu_resident"] = True
    validate_row(resident)
    spilled = row()
    spilled["model"]["size"] = 10
    spilled["model"]["size_vram"] = 11
    spilled["gpu_resident"] = False
    validate_row(spilled)
    claimed = row()
    claimed["gpu_resident"] = True
    with pytest.raises(ReceiptError, match="needs model.size"):
        validate_row(claimed)
    with pytest.raises(ReceiptError):
        validate_row(row(task="verify"))


def test_the_choice_rule_is_four_fields_and_a_frozen_sha():
    sha = "a" * 40
    accepted = validate_spec({**_spec(), "choice_rule": _rule(), "selection_sha": sha})
    assert accepted["selection_sha"] == sha
    with pytest.raises(SpecError, match="belongs with"):
        validate_spec({**_spec(), "selection_sha": sha})
    with pytest.raises(SpecError, match="mapping"):
        validate_spec({**_spec(), "choice_rule": [], "selection_sha": sha})
    with pytest.raises(SpecError, match="eligibility floor"):
        validate_spec({**_spec(), "choice_rule": {"ranking_key": "x"}, "selection_sha": sha})
    with pytest.raises(SpecError, match="metric and a minimum"):
        validate_spec({**_spec(), "choice_rule": _rule(eligibility_floor={"metric": "x"}), "selection_sha": sha})
    with pytest.raises(SpecError, match="must name a quality metric"):
        validate_spec(
            {**_spec(), "choice_rule": _rule(eligibility_floor={"metric": "", "minimum": 0}), "selection_sha": sha}
        )
    with pytest.raises(SpecError, match="must be a number"):
        validate_spec(
            {**_spec(), "choice_rule": _rule(eligibility_floor={"metric": "x", "minimum": True}), "selection_sha": sha}
        )
    with pytest.raises(SpecError, match="ranking_key"):
        validate_spec({**_spec(), "choice_rule": _rule(ranking_key=""), "selection_sha": sha})
    with pytest.raises(SpecError, match="holds toward"):
        validate_spec({**_spec(), "choice_rule": _rule(tie_rule={"toward": "a", "else": "b"}), "selection_sha": sha})
    with pytest.raises(SpecError, match="committed choice"):
        validate_spec({**_spec(), "choice_rule": _rule(tie_rule={"toward": ""}), "selection_sha": sha})
    with pytest.raises(SpecError, match="must name a row"):
        validate_spec({**_spec(), "choice_rule": _rule(fallback=""), "selection_sha": sha})
    with pytest.raises(SpecError, match="frozen selection_sha"):
        validate_spec({**_spec(), "choice_rule": _rule(), "selection_sha": ""})


def test_choose_ranks_breaks_ties_toward_the_committed_choice_and_falls_back():
    rule = _rule()
    sha = "c" * 40
    ranked = choose([_choice_row("low", 0.6), _choice_row("high", 0.9)], rule, selection_sha=sha)
    assert ranked["chosen"] == "high" and ranked["reason"] == "ranked"
    tied = choose(
        [_choice_row("other", 0.9), _choice_row("row-committed", 0.9)],
        rule,
        selection_sha=sha,
    )
    assert tied["chosen"] == "row-committed" and tied["reason"] == "tie"
    by_sha = choose(
        [_choice_row("other", 0.9), _choice_row("pinned", 0.9, sha=sha)],
        rule,
        selection_sha=sha,
    )
    assert by_sha["chosen"] == "pinned" and by_sha["reason"] == "tie"
    fallen = choose(
        [_choice_row("other", 0.9), _choice_row("row-fallback", 0.9)],
        rule,
        selection_sha=sha,
    )
    assert fallen["chosen"] == "row-fallback" and fallen["reason"] == "fallback"
    unresolved = choose([_choice_row("one", 0.9), _choice_row("two", 0.9)], rule, selection_sha=sha)
    assert unresolved["chosen"] is None and unresolved["reason"] == "none"
    below = choose([_choice_row("low", 0.4)], rule, selection_sha=sha)
    assert below["eligible"] == [] and below["reason"] == "none"
    exact = choose([_choice_row("edge", 0.5)], rule, selection_sha=sha)
    assert exact["chosen"] == "edge"
    flagged = {"id": "flag", "quality": [{**_rate("consensus_agreement", 0.9), "value": True}]}
    assert choose([flagged], rule, selection_sha=sha)["eligible"] == []
    unranked = {"id": "bare", "quality": [_rate("consensus_agreement", 0.9)]}
    skipped = choose([unranked, _choice_row("kept", 0.6)], _rule(ranking_key="false_reject_rate"), selection_sha=sha)
    assert skipped["eligible"] == ["kept"]


def test_one_tasks_quote_rule_does_not_fill_anothers_gap():
    rows = [
        {
            "id": "g",
            "task": "verify:grounded",
            "device": "5090",
            "model": {"repo_or_tag": "qwen3:8b"},
            "settings": {"quote_rule": 2},
            "quality": [],
            "latency": {},
        }
    ]
    candidates = [
        {
            "id": "quote-rule-2-grounded",
            "match": {"task": "verify:grounded", "quote_rule": 2},
            "needs": ["quote_rule"],
        },
        {
            "id": "quote-rule-2-math-ladder",
            "match": {"task": "verify:math-ladder", "quote_rule": 2},
            "needs": ["quote_rule"],
        },
    ]
    assert {item["id"] for item in open_gaps(rows, candidates)} == {"quote-rule-2-math-ladder"}


def test_the_committed_table_is_rekeyed_and_does_not_invent_judge_rates():
    rows = [
        json.loads(line)
        for line in (ROOT / "table" / "routing.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    families = json.loads((ROOT / "table" / "model-families.json").read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for item in rows:
        counts[item["task"]] = counts.get(item["task"], 0) + 1
        tag = item["model"]["repo_or_tag"]
        names = {metric["name"] for metric in item["quality"]}
        assert "false_reject_rate" not in names
        assert "false_accept_rate" not in names
        if item["id"].startswith("aspire/B/"):
            assert names == {"consensus_agreement", "abstain_rate"}
            assert "per_type" in item and "cost" in item
            assert "gpu_resident" not in item
            provenance = item["provenance"]
            assert provenance["commit_sha"] == "3b05eab928894311993ffa3af2075fef39cccce6"
            assert provenance["source_receipt_path"] == "docs/runs/receipts/2026-10-08-auditor/B-readout.json"
            assert provenance["content_sha256"] == "d61915ebf4106c3067b6ea08d83c3aace9555113afb8193fe480c7013cb47eb5"
        else:
            assert "gpu_resident" not in item and "cost" not in item and "per_type" not in item
        if tag in families:
            assert item["model_family"] == families[tag]
        else:
            assert "model_family" not in item
    assert counts["verify:grounded"] == 24
    assert counts["verify:reasoning"] == 30
    assert counts["verify:math-ladder"] == 312
    assert "verify" not in counts
    assert counts["judge:planted-error"] == 1
    assert "judge:same-meaning-rewrite" not in counts
    assert len(rows) == 509  # +7: e4-2026-10-12 (all four pairs inconclusive by the noise rule, R&D readout on #76); +1: e2-2026-10-10b l512 native_binary_rejected (the l2048-b8 hang row superseded by #36); +14: e2-2026-10-10c; +1: the tail l512-b8 compile_rejected; +1: the tail l128-b8 (2026-10-11); +27: e3-2026-10-11b; +2: e3-ref-2026-10-11b
    assert sum(1 for item in rows if item["id"].startswith("ladder/")) == 312
    assert sum(1 for item in rows if item["id"].startswith("verify/")) == 54
    e1 = [item for item in rows if item["id"].startswith("E1/")]
    assert len(e1) == 47
    assert all("artifacts" not in item for item in e1)
    assert all(item["settings"]["environment"]["openvino"] == "unrecorded" for item in e1)
    miss_id = "E1/nomic-npu-l2048-b1/e1-intel-2026-10-09-rerun2"
    miss = next(item for item in e1 if item["id"] == miss_id)
    assert miss["parity_valid"] is False
    assert miss["latency"]["cold_ms"] is not None
    assert miss["latency"]["warm_p50_ms"] is not None
    assert is_speed_row(miss) is False
    assert miss_id not in {item["id"] for item in select(rows, "embed").rows}
    lost_ids = {
        "E1/nomic-npu-l2048-b8/e1-intel-2026-10-09-rerun2-b8tail",
        "E1/nomic-npu-l1024-b8/e1-intel-2026-10-09-rerun2-nomic-diag",
    }
    lost = [item for item in e1 if item["id"] in lost_ids]
    assert {item["id"] for item in lost} == lost_ids
    for item in lost:
        assert item["outcome"] == "device_lost"
        assert item["latency"]["cold_ms"] is None
        assert item["latency"]["warm_p50_ms"] is None
        assert "this device can't do this setting" in item["notes"]
        assert is_speed_row(item) is False
