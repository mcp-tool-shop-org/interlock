"""ASPIRE judge-control rows, re-derived from small raw fixtures.

Expected numbers are counted off the fixtures here, not taken from the importer.
"""

from __future__ import annotations

import json

import pytest

from interlock.importers.aspire_judges import AspireImportError, b_rows, judge_control_rows
from interlock.importers.common import ImportError_, sha256_bytes, wilson
from interlock.receipt import validate_row

DIGEST = "8039dd90c1138d772437a0779a33b7349efd5d9cca71edcd26e4dd463f90439d"


def _b_fixture():
    selection = {
        "seed": 0,
        "splits": ["s1", "s2"],
        "control_both_not_wrong": ["n1", "n2", "n3", "n4"],
        "control_both_wrong": ["w1", "w2", "w3"],
        "order": ["n1", "s1", "w1", "n2", "w2", "s2", "n3", "w3", "n4"],
    }
    verdicts = {
        "s1": {"outcome": "wrong", "grammatical": True, "seconds": 2.0},
        "s2": {"outcome": "unsure", "seconds": 4.0},
        "n1": {"outcome": "not wrong", "seconds": 2.0},
        "n2": {"outcome": "wrong", "grammatical": True, "seconds": 2.0},
        "n3": {"outcome": "unsure", "seconds": 2.0},
        "n4": {"outcome": "not wrong", "seconds": 2.0},
        "w1": {"outcome": "wrong", "grammatical": True, "seconds": 2.0},
        "w2": {"outcome": "wrong", "grammatical": False, "seconds": 2.0},  # not agreement: ungrammatical
        "w3": {"outcome": "wrong", "grammatical": True, "seconds": 2.0},
    }
    manifest = {
        "judge": "mistral-small:24b",
        "digest": DIGEST,
        "prompt_sha256": "p" * 64,
        "schema_sha256": "s" * 64,
        "options": {"num_predict": 2048, "num_ctx": 8192},
        "think": False,
        "splits": ["s1", "s2"],
        "control": sorted(["n1", "n2", "n3", "n4", "w1", "w2", "w3"]),
        "order": selection["order"],
    }
    readout = {
        "control": {"both wrong": {"agrees": 2}, "both not wrong": {"agrees": 2}},
        "reading": {"control_passed": False},
    }
    return {k: json.dumps(v).encode("utf-8") for k, v in
            {"readout": readout, "manifest": manifest, "verdicts": verdicts, "selection": selection}.items()}


def _b(**override):
    files = _b_fixture() | override
    return b_rows(**files, source_dir="aspire-si-runs/run/matched", date="2026-10-09")


def test_b_row_counts_consensus_per_side_and_never_claims_gold_rates():
    (row,) = _b()
    validate_row(row)
    assert row["task"] == "judge:planted-error"
    assert row["model_family"] == "mistral"
    assert row["device"] == "5090"
    assert row["settings"]["thinking_level"] == "off"
    names = {m["name"]: m for m in row["quality"]}
    assert set(names) == {"consensus_agreement", "abstain_rate"}  # no false_reject / false_accept from consensus
    assert (names["consensus_agreement"]["value"], names["consensus_agreement"]["n"]) == (4 / 7, 7)
    assert (names["abstain_rate"]["value"], names["abstain_rate"]["n"]) == (2 / 9, 9)
    wrong_side = row["per_type"]["consensus:both-wrong"]
    assert (wrong_side["value"], wrong_side["n"]) == (2 / 3, 3)
    low, high = wilson(2, 3)
    assert (wrong_side["interval"]["low"], wrong_side["interval"]["high"]) == (low, high)
    assert row["per_type"]["consensus:both-not-wrong"]["value"] == 2 / 4
    assert row["cost"] == {"seconds": 20 / 9, "tokens": None}
    assert row["provenance"]["content_sha256"] == sha256_bytes(_b_fixture()["readout"])
    assert row["provenance"]["commit_sha"] is None
    assert row["model"]["quant"] == "unrecorded" and row["engine"]["version"] == "unrecorded"
    assert "gpu_resident" not in row
    assert "failed" in row["notes"]


def test_b_refuses_a_readout_that_disagrees_with_its_verdicts():
    bad = json.loads(_b_fixture()["readout"])
    bad["control"]["both wrong"]["agrees"] = 3  # the ungrammatical "wrong" must not count
    with pytest.raises(AspireImportError, match="both wrong"):
        _b(readout=json.dumps(bad).encode("utf-8"))


def test_b_refuses_missing_verdicts_and_a_mismatched_selection():
    verdicts = json.loads(_b_fixture()["verdicts"])
    del verdicts["n4"]
    with pytest.raises(AspireImportError, match="lack 1"):
        _b(verdicts=json.dumps(verdicts).encode("utf-8"))
    manifest = json.loads(_b_fixture()["manifest"])
    manifest["control"] = manifest["control"][:-1]
    with pytest.raises(AspireImportError, match="different items"):
        _b(manifest=json.dumps(manifest).encode("utf-8"))


def _control_fixture(size=1000, size_vram=1000):
    items = (
        [{"id": f"h-num-{i}", "side": "harmless", "kind": "number_words"} for i in range(2)]
        + [{"id": f"h-con-{i}", "side": "harmless", "kind": "contraction"} for i in range(2)]
        + [{"id": f"x-neg-{i}", "side": "harmful", "kind": "negation"} for i in range(2)]
        + [{"id": f"x-ant-{i}", "side": "harmful", "kind": "antonym"} for i in range(2)]
        + [{"id": f"i-{i}", "side": "identity", "kind": "identity"} for i in range(2)]
    )
    selection = {"items": items, "order": [r["id"] for r in reversed(items)]}
    sel_bytes = json.dumps(selection).encode("utf-8")
    verdicts = {
        "h-num-0": {"outcome": "same", "claim_changed": False, "seconds": 1.0},
        "h-num-1": {"outcome": "changed", "claim_changed": True, "seconds": 1.0},
        "h-con-0": {"outcome": "same", "claim_changed": True, "seconds": 1.0},  # not strict agreement
        "h-con-1": {"outcome": "unsure", "claim_changed": False, "seconds": 1.0},
        "x-neg-0": {"outcome": "changed", "claim_changed": True, "seconds": 3.0},
        "x-neg-1": {"outcome": "changed", "claim_changed": True, "seconds": 3.0},
        "x-ant-0": {"outcome": "same", "claim_changed": False, "seconds": 3.0},
        "x-ant-1": {"outcome": "truncated", "seconds": 3.0},
        "i-0": {"outcome": "same", "claim_changed": False, "seconds": 1.0},
        "i-1": {"outcome": "changed", "claim_changed": True, "seconds": 1.0},
    }
    manifest = {
        "selection_sha256": sha256_bytes(sel_bytes),
        "prompt_sha256": "p" * 64,
        "schema_sha256": "s" * 64,
        "ollama_url": "http://127.0.0.1:11434",
        "ollama_version": "0.35.1",
        "judges": [{"model": "qwen3:14b", "digest": "bdbd181c33f2" + "0" * 52, "quant": "Q4_K_M", "think": True,
                    "options": {"num_predict": 16000, "num_ctx": 24576}}],
        "order": selection["order"],
    }
    readout = {
        "scores": {"qwen3:14b": {"harmless_agree": 1, "harmful_agree": 2}},
        "residency": {"qwen3:14b": {"size": size, "size_vram": size_vram, "on_gpu": size_vram >= size}},
        "result": {"choice": "mistral-small:24b"},
    }
    return {
        "readout": json.dumps(readout).encode("utf-8"),
        "manifest": json.dumps(manifest).encode("utf-8"),
        "verdicts": {"qwen3_14b": json.dumps(verdicts).encode("utf-8")},
        "selection": sel_bytes,
    }


def _control(**override):
    files = _control_fixture() | override
    return judge_control_rows(**files, source_dir="aspire-si-runs/run/matched", date="2026-10-09")


def test_control_row_carries_true_rates_per_type_residency_and_cost():
    (row,) = _control()
    validate_row(row)
    assert row["task"] == "judge:same-meaning-rewrite"
    assert row["model_family"] == "qwen"
    assert row["settings"]["thinking_level"] == "on"
    assert (row["settings"]["num_predict"], row["settings"]["num_ctx"]) == (16000, 24576)
    assert row["engine"]["version"] == "0.35.1" and row["model"]["quant"] == "Q4_K_M"
    q = {m["name"]: (m["value"], m["n"]) for m in row["quality"]}
    assert q["false_reject_rate"] == (1 / 4, 4)  # one harmless called "changed"
    assert q["false_accept_rate"] == (1 / 4, 4)  # one harmful called "same"
    assert q["abstain_rate"] == (1 / 8, 8)  # one "unsure" among the decided sides
    pt = {k: (v["name"], v["value"], v["n"]) for k, v in row["per_type"].items()}
    assert pt["harmless:number_words"] == ("false_reject_rate", 0.5, 2)
    assert pt["harmless:contraction"] == ("false_reject_rate", 0.0, 2)
    assert pt["harmful:antonym"] == ("false_accept_rate", 0.5, 2)
    assert pt["harmful:negation"] == ("false_accept_rate", 0.0, 2)
    assert pt["identity"] == ("false_reject_rate", 0.5, 2)
    assert row["gpu_resident"] is True and row["model"]["size_vram"] == 1000
    assert row["cost"]["seconds"] == pytest.approx(18 / 10)
    assert "Unusable replies: 1" in row["notes"] and "Chosen by the pre-registered rule: False" in row["notes"]


def test_control_records_an_offloaded_model_as_not_resident():
    (row,) = _control(**{"readout": _control_fixture(size=1000, size_vram=900)["readout"]})
    assert row["gpu_resident"] is False


def test_control_refuses_a_readout_that_disagrees_with_its_verdicts_or_a_foreign_selection():
    readout = json.loads(_control_fixture()["readout"])
    readout["scores"]["qwen3:14b"]["harmless_agree"] = 2  # "same" with claim_changed true isn't agreement
    with pytest.raises(AspireImportError, match="agreement counts"):
        _control(readout=json.dumps(readout).encode("utf-8"))
    selection = json.loads(_control_fixture()["selection"])
    selection["order"] = selection["order"][::-1]
    with pytest.raises(AspireImportError, match="different selection"):
        _control(selection=json.dumps(selection).encode("utf-8"))
    with pytest.raises(AspireImportError, match="no verdict file"):
        _control(verdicts={})


def test_the_refusal_is_still_an_importer_error():
    assert issubclass(AspireImportError, ImportError_)
