"""Goldens on the committed routing table. These tests do not invent rows."""

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from interlock.cli import main
from interlock.router.policy import decode_bound, prefix_mode

ROOT = Path(__file__).resolve().parents[1]
DECISION = json.loads((ROOT / "src" / "interlock" / "schema" / "route-decision.v1.json").read_text(encoding="utf-8"))
PLAN = json.loads((ROOT / "tests" / "fixtures" / "offrig-plan-input.8f2a94a.json").read_text(encoding="utf-8"))


def load_table():
    path = ROOT / "table" / "routing.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def invoke(capsys, *args):
    code = main(["route", "--root", str(ROOT), "--as-of", "2026-10-11", "--json", *args])
    captured = capsys.readouterr()
    return code, json.loads(captured.out), captured.err


def assert_decision(document):
    errors = sorted(Draft202012Validator(DECISION).iter_errors(document), key=lambda item: list(item.path))
    assert errors == []


def test_phi_l1024_busy_5090_picks_igpu_from_bounds(capsys):
    """CPU Phi l1024, thinking off: the worst repeat is about 7.04 tok/s and the median is about 20.4.

    That spread is a contention outlier. v0 ranks by the minimum. There is no
    outlier trimming. CPU Phi needs a quiet-window rerun before its bound is
    trusted. On this cell the runner-up by the bound is the NPU (about 15.13),
    not the CPU, because the CPU bound collapses to about 7.04. The iGPU
    minimum is about 21.46.
    """

    code, document, err = invoke(
        capsys,
        "--task", "generate",
        "--model", "Phi",
        "--thinking", "off",
        "--ctx", "1024",
        "--busy", "5090",
    )
    assert err == ""
    assert code == 0
    assert document["device"] == "iGPU"
    assert document["task"] == "generate"
    assert "no 5090 row" not in document["notes"]
    assert document["budget"] == "not_applicable"
    assert document["flags"] == []
    assert_decision(document)

    rows = load_table()
    by_id = {row["id"]: row for row in rows}
    winner = by_id[document["receipts"][0]]
    assert "l1024" in winner["id"]
    assert "igpu" in winner["id"]
    bound = decode_bound(winner)
    assert document["margin"]["metric"] == "decode_tok_s"
    assert document["margin"]["direction"] == "higher"
    assert document["margin"]["bound"] == bound
    assert document["margin"]["bound"] != winner["latency"]["throughput"]
    assert 20 < bound < 23

    cpu = next(row for row in rows if row["id"].startswith("E3/phi-35-mini-cpu-l1024"))
    cpu_min = decode_bound(cpu)
    assert cpu_min < 8
    assert cpu["latency"]["throughput"] > 20
    nxt = by_id[document["margin"]["next_id"]]
    assert nxt["device"] == "NPU"
    assert "l1024" in nxt["id"]
    assert document["margin"]["next_bound"] == decode_bound(nxt)
    assert cpu_min < document["margin"]["next_bound"] < bound
    assert document["margin"]["delta"] == bound - document["margin"]["next_bound"]
    assert 14 < document["margin"]["next_bound"] < 17


def test_phi_without_busy_picks_igpu_and_says_no_5090_row(capsys):
    code, document, err = invoke(
        capsys,
        "--task", "generate",
        "--model", "Phi",
        "--thinking", "off",
        "--ctx", "1024",
    )
    assert err == ""
    assert code == 0
    assert document["device"] == "iGPU"
    assert document["notes"] == [
        "no 5090 row",
        "ranked at concurrency 1 on decode_tok_s, the pass aggregate across 1 concurrent request(s)",
    ]
    assert document["receipts"][0].startswith("E3/phi-35-mini-igpu-l1024")
    assert_decision(document)
    phi_5090 = [
        row
        for row in load_table()
        if row["task"] == "generate"
        and row["device"] == "5090"
        and "phi" in row["model"]["repo_or_tag"].casefold()
    ]
    assert phi_5090 == []


def test_qwen_floor_15_busy_5090_refuses_naming_e3(capsys):
    code, document, err = invoke(
        capsys,
        "--task", "generate",
        "--model", "Qwen3-8B",
        "--busy", "5090",
        "--min-tok-s", "15",
    )
    assert err == ""
    assert code == 1
    assert document["code"] == "no_candidate"
    assert document["constraint"] == "min_tok_s"
    assert document["experiment"] == "E3"
    assert document["closest"]["id"].startswith("E3/")
    assert not document["closest"]["id"].startswith("E3-REF")
    rows = load_table()
    by_id = {row["id"]: row for row in rows}
    closest = by_id[document["closest"]["id"]]
    assert closest["device"] != "5090"
    intel = []
    card = []
    for row in rows:
        if row.get("task") != "generate" or row.get("outcome"):
            continue
        repo = row["model"]["repo_or_tag"]
        folded = repo.casefold()
        if "qwen3-8b" not in folded and "qwen3:8b" not in folded:
            continue
        bound = decode_bound(row)
        if bound is None:
            continue
        if row["device"] == "5090":
            card.append(bound)
        else:
            intel.append(bound)
    best_intel = max(intel)
    assert best_intel < 15
    assert document["closest"]["bound"] == best_intel
    assert document["closest"]["gap"] == 15 - best_intel
    assert max(card) >= 15

    open_code, open_doc, open_err = invoke(
        capsys,
        "--task", "generate",
        "--model", "Qwen3-8B",
        "--min-tok-s", "15",
    )
    assert open_err == ""
    assert open_code == 0
    assert open_doc["device"] == "5090"
    assert_decision(open_doc)


def test_qwen3_8b_thinking_on_keeps_e3_low_and_the_e3_ref_on_arm(capsys):
    rows = [
        row
        for row in load_table()
        if row.get("task") == "generate"
        and row.get("outcome") is None
        and (
            "qwen3-8b" in row["model"]["repo_or_tag"].casefold()
            or "qwen3:8b" in row["model"]["repo_or_tag"].casefold()
        )
    ]
    by_id = {row["id"]: row for row in rows}
    low = [
        "E3/qwen3-8b-cpu-l256-think/e3-2026-10-11b",
        "E3/qwen3-8b-cpu-l1024-think/e3-2026-10-11b",
        "E3/qwen3-8b-cpu-l4096-think/e3-2026-10-11b",
        "E3/qwen3-8b-igpu-l256-think/e3-2026-10-11b",
        "E3/qwen3-8b-igpu-l1024-think/e3-2026-10-11b",
        "E3/qwen3-8b-igpu-l4096-think/e3-2026-10-11b",
        "E3/qwen3-8b-npu-l256-think/e3-2026-10-11b",
        "E3/qwen3-8b-npu-l1024-think/e3-2026-10-11b",
        "E3/qwen3-8b-npu-l4096-think/e3-2026-10-11b",
    ]
    on_arm = "E3-REF/ref-5090-qwen3-8b-thinkon/e3-ref-2026-10-11b"
    found_low = {row_id for row_id, row in by_id.items() if row["settings"]["thinking_level"] == "low"}
    assert found_low == set(low)
    assert by_id[on_arm]["settings"]["thinking_level"] == "on"

    code, document, err = invoke(capsys, "--task", "generate", "--model", "Qwen3-8B", "--thinking", "on")
    assert err == ""
    assert code == 0
    assert_decision(document)
    seen = set(document["receipts"]) | {item["id"] for item in document["runners_up"]}
    assert set(low) <= seen
    assert on_arm in seen
    for item in document["runners_up"]:
        if item["id"] in low or item["id"] == on_arm:
            assert item["constraint"] != "thinking"
        if by_id.get(item["id"], {}).get("settings", {}).get("thinking_level") == "off":
            assert item["constraint"] == "thinking"
    winner = by_id[document["receipts"][0]]
    assert document["settings"]["thinking_level"] == winner["settings"]["thinking_level"]

    busy_code, busy, busy_err = invoke(
        capsys,
        "--task", "generate",
        "--model", "Qwen3-8B",
        "--thinking", "on",
        "--busy", "5090",
    )
    assert busy_err == ""
    assert busy_code == 0
    assert_decision(busy)
    assert busy["receipts"][0] in low
    assert busy["settings"]["thinking_level"] == "low"
    assert busy["settings"]["thinking_level"] == by_id[busy["receipts"][0]]["settings"]["thinking_level"]
    busy_reasons = {item["id"]: item["constraint"] for item in busy["runners_up"]}
    assert busy_reasons[on_arm] == "busy"


def test_nomic_l2048_parity_invalid_is_excluded_not_ranked(capsys):
    missed = "E1/nomic-npu-l2048-b1/e1-intel-2026-10-09-rerun2"
    code, document, err = invoke(capsys, "--task", "embed", "--model", "nomic-embed-text")
    assert err == ""
    assert code == 1
    assert document["code"] == "no_candidate"
    excluded = {item["id"]: item["code"] for item in document["excluded"]}
    assert excluded[missed] == "parity_invalid"
    assert sum(item_code == "parity_invalid" for item_code in excluded.values()) == 1
    assert document["closest"]["id"] != missed

    code, document, err = invoke(capsys, "--task", "embed")
    assert err == ""
    assert code == 0
    assert_decision(document)
    excluded = {item["id"]: item["code"] for item in document["excluded"]}
    assert excluded[missed] == "parity_invalid"
    ranked = set(document["receipts"]) | {item["id"] for item in document["runners_up"]}
    assert missed not in ranked


def test_qwen3_embedding_npu_512_is_excluded_not_ranked(capsys):
    code, document, err = invoke(
        capsys,
        "--task", "embed",
        "--model", "Qwen3-Embedding",
        "--ctx", "512",
    )
    assert err == ""
    assert code == 0
    assert_decision(document)
    excluded = {item["id"]: item["code"] for item in document["excluded"]}
    native = [row_id for row_id in excluded if "qwen3emb-npu-l512-b1" in row_id]
    compiled = [row_id for row_id in excluded if "qwen3emb-npu-l512-b8" in row_id]
    assert len(native) == 1
    assert excluded[native[0]] == "native_binary_rejected"
    assert len(compiled) == 1
    assert excluded[compiled[0]] == "compile_rejected"
    assert "l512" in document["receipts"][0]
    ranked = set(document["receipts"]) | {item["id"] for item in document["runners_up"]}
    assert native[0] not in ranked
    assert compiled[0] not in ranked
    passed = [
        row["id"]
        for row in load_table()
        if "qwen3emb" in row["id"] and "l128" in row["id"] and row.get("outcome") is None
    ]
    assert passed
    assert all(row_id not in excluded for row_id in passed)


def test_ttft_on_the_committed_table_refuses_unset_prefix_caching(capsys):
    rows = load_table()
    assert all(prefix_mode(row) is None for row in rows)
    code, document, err = invoke(
        capsys,
        "--task", "generate",
        "--model", "Phi",
        "--thinking", "off",
        "--ctx", "1024",
        "--max-ttft-ms", "1000",
    )
    assert err == ""
    assert code == 1
    assert document["code"] == "prefix_caching_unknown"
    assert "settings.prefix_caching" in document["hint"]
    assert document["experiment"] == "E3"
    assert document["constraint"] == "max_ttft_ms"


def test_verify_quote_rule_2_refuses_unregistered(capsys):
    code, document, err = invoke(capsys, "--task", "verify:grounded", "--quote-rule", "2")
    assert err == ""
    assert code == 1
    assert document["code"] == "no_candidate"
    assert document["constraint"] == "quote_rule"
    assert document["experiment"] == "unregistered"
    assert "unregistered" in document["hint"]


def test_golden_decision_validates_against_route_decision_schema(capsys):
    code, document, err = invoke(
        capsys,
        "--task", "generate",
        "--model", "Phi",
        "--thinking", "off",
        "--ctx", "1024",
        "--busy", "5090",
    )
    assert code == 0 and err == ""
    assert_decision(document)
    code, document, err = invoke(capsys, "--task", "embed", "--model", "Qwen3-Embedding", "--ctx", "512")
    assert code == 0 and err == ""
    assert_decision(document)
    Draft202012Validator.check_schema(DECISION)
    Draft202012Validator.check_schema(PLAN)
