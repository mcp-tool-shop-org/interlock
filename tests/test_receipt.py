import json
from pathlib import Path

import pytest

from switchyard.receipt import ReceiptError, check_tree, validate_row
from tests.support import row


def test_a_complete_row_validates():
    validate_row(row())


def test_schema_refuses_gpu_1_and_a_broken_interval():
    bad_device = row(device="GPU.1")
    with pytest.raises(ReceiptError, match="device"):
        validate_row(bad_device)
    bad_interval = row()
    bad_interval["quality"][0]["interval"] = {"low": 0.9, "high": 0.1, "method": "wilson-95"}
    with pytest.raises(ReceiptError, match="interval"):
        validate_row(bad_interval)


def test_home_directory_paths_and_unpinned_bytes_are_refused():
    homed = row()
    # Built in pieces so the test source itself is not a home-directory path.
    homed["notes"] = "C:" + "\\Users\\" + "someone\\receipt.json"
    with pytest.raises(ReceiptError, match="home-directory"):
        validate_row(homed)
    unpinned = row(provenance={"source_receipt_path": "E:/AI/rnd-ladder/ladder-qwen3_8b/metrics.json", "commit_sha": None})
    with pytest.raises(ReceiptError, match="content_sha256"):
        validate_row(unpinned)
    unpinned["provenance"]["content_sha256"] = "deadbeef"
    validate_row(unpinned)


def test_device_lost_cannot_carry_timing():
    lost = row()
    lost["outcome"] = "device_lost"
    lost["latency"] = {
        "cold_ms": None,
        "warm_p50_ms": None,
        "warm_p95_ms": None,
        "throughput": None,
        "throughput_unit": None,
        "compile_ms": None,
        "first_call_ms": None,
    }
    lost["quality"] = []
    validate_row(lost)
    lost["latency"]["cold_ms"] = 12
    with pytest.raises(ReceiptError, match="no timing"):
        validate_row(lost)
    lost["latency"]["cold_ms"] = None
    lost["artifacts"] = {"slots": [{"latency_ms": 12}]}
    with pytest.raises(ReceiptError, match="no timing"):
        validate_row(lost)


def test_experiment_provenance_is_the_other_legal_shape():
    validate_row(row(provenance={"experiment_spec": "experiments/e1-cold-warm/spec.yaml", "run_id": "dry-run"}))


def test_check_is_green_on_an_empty_tree_and_red_on_a_bad_receipt(tmp_path: Path):
    assert check_tree(tmp_path) == []
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    (receipts / "ok.json").write_text(json.dumps(row()), encoding="utf-8")
    assert check_tree(tmp_path) == []
    (receipts / "bad.json").write_text("{", encoding="utf-8")
    problems = check_tree(tmp_path)
    assert len(problems) == 1
    assert "bad.json" in problems[0]


def test_check_reads_a_jsonl_table(tmp_path: Path):
    table = tmp_path / "table"
    table.mkdir()
    (table / "routing.jsonl").write_text(json.dumps(row()) + "\n" + json.dumps(row(id="fixture/two")) + "\n", encoding="utf-8")
    assert check_tree(tmp_path) == []
    (table / "routing.jsonl").write_text("{not json}\n", encoding="utf-8")
    assert check_tree(tmp_path)


def test_a_list_of_rows_and_a_non_object_are_both_handled(tmp_path: Path):
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    (receipts / "many.json").write_text(json.dumps([row(), row(id="fixture/two")]), encoding="utf-8")
    assert check_tree(tmp_path) == []
    (receipts / "nope.json").write_text("1", encoding="utf-8")
    assert any("nope.json" in problem for problem in check_tree(tmp_path))
