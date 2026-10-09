"""Re-derive imported rows from raw receipt fixtures.

The expected numbers are read off the fixture, not off the importer's own
intermediate values.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from switchyard.importers.calibration import calibration_rows, quote_rule_of
from switchyard.importers.common import ImportError_
from switchyard.importers.ladder import ladder_rows
from switchyard.importers.nli import nli_rows
from switchyard.importers.offers import offer_rows, remote_option_allowed
from switchyard.importers.probe import probe_rows
from switchyard.receipt import validate_row

_METRICS = {
    "rows": [
        {
            "n": 4,
            "passes_default_rule": False,
            "false_accept": {"rate": 0.25, "of": 4, "low": 0.01, "high": 0.8},
            "false_accept_unsupported_only": {"rate": 0.0, "of": 2, "low": 0.0, "high": 0.5},
            "abstain": {"rate": 0.0, "of": 4, "low": 0.0, "high": 0.5},
            "decided_balanced_accuracy": 0.5,
        }
    ]
}


def _manifest(rule=None):
    settings = {
        "url": "http://127.0.0.1:11434",
        "think": "on",
        "structured": "on",
        "split": "tune",
        "check_type": "grounded",
        "num_ctx": 16384,
        "num_predict": 4096,
    }
    if rule is not None:
        settings["quote_rule"] = rule
    return {
        "offrig_version": "1.0.0",
        "started_at": 1791527981,
        "model": "qwen3:8b",
        "model_digest": "500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41",
        "ollama_version": "0.35.1",
        "settings": settings,
        "gold": [{"file": "grounded.jsonl", "sha256": "ab"}],
    }


def _verdicts():
    labels = ["supported", "unsupported", "cannot_tell", "unsupported"]
    verdicts = ["supported", "supported", "cannot_tell", "unsupported"]
    rows = []
    for index, (label, verdict) in enumerate(zip(labels, verdicts)):
        rows.append(
            {
                "claim_id": f"c{index}",
                "status": "ok",
                "final_verdict": verdict,
                "wall_seconds": 1.0 + index,
            }
        )
    return rows


def _tiers():
    labels = ["supported", "unsupported", "cannot_tell", "unsupported"]
    return {
        f"c{index}": {"check_type": "grounded", "split": "tune", "label": label, "tier": "T1"}
        for index, label in enumerate(labels)
    }


def test_calibration_copies_the_published_interval_and_infers_quote_rule_1():
    from switchyard.importers.calibration import INFERRED_QUOTE_RULE_SOURCE

    manifest = _manifest()
    rows = calibration_rows(
        manifest, _METRICS, _verdicts(), _tiers(), source_dir="runs/cal", commit_sha="a" * 40
    )
    parent = rows[0]
    block = _METRICS["rows"][0]["false_accept"]
    metric = next(item for item in parent["quality"] if item["name"] == "false_accept")
    assert metric["value"] == block["rate"]
    assert metric["interval"]["low"] == block["low"]
    assert metric["interval"]["high"] == block["high"]
    assert metric["n"] == block["of"]
    assert parent["settings"]["quote_rule"] == 1
    assert parent["settings"]["quote_rule_source"] == INFERRED_QUOTE_RULE_SOURCE
    assert "/q-1/" in parent["id"]
    assert parent["date"] == "2026-10-09"
    assert parent["task"] == "verify:grounded"
    assert parent["model_family"] == "qwen"
    assert quote_rule_of(manifest, _verdicts()) == (1, INFERRED_QUOTE_RULE_SOURCE)


def test_quote_rule_1_and_rule_2_are_different_rows():
    kwargs = {"metrics": _METRICS, "verdicts": _verdicts(), "tiers": _tiers(), "source_dir": "runs/cal", "commit_sha": "b" * 40}
    rule_1 = calibration_rows(_manifest(1), **kwargs)[0]["id"]
    rule_2 = calibration_rows(_manifest(2), **kwargs)[0]["id"]
    assert rule_1 != rule_2
    assert "/q-1/" in rule_1
    assert "/q-2/" in rule_2
    explicit = calibration_rows(_manifest(2), **kwargs)[0]
    assert explicit["settings"]["quote_rule_source"] == "receipt"


def test_a_tier_row_is_recounted_from_the_verdict_lines():
    rows = calibration_rows(
        _manifest(), _METRICS, _verdicts(), _tiers(), source_dir="runs/cal", commit_sha="c" * 40
    )
    tier = next(row for row in rows if row["id"].endswith("/tier/T1") or "/tier/T1" in row["id"])
    accuracy = next(item for item in tier["quality"] if item["name"] == "accuracy")
    false_accept = next(item for item in tier["quality"] if item["name"] == "false_accept")
    # supported->supported, unsupported->supported, cannot_tell->cannot_tell, unsupported->unsupported.
    # Correct on the difficulty rule: 1, 0, 1, 1. False accepts: the two unsupported golds, one supported verdict.
    assert accuracy["n"] == 4
    assert accuracy["value"] == 0.75
    assert false_accept["n"] == 3
    assert false_accept["value"] == pytest.approx(1 / 3)


def test_ladder_cell_is_tallied_on_both_views():
    gold = [{"id": "a", "label": "unsupported", "ladder": {"cell": "L1"}}]
    manifest = _manifest()
    verdicts = [{"claim_id": "a", "status": "ok", "final_verdict": "supported", "model_verdict": "unsupported"}]
    gold_bytes = b'{"id":"a"}\n'
    verdict_bytes = b'{"claim_id":"a"}\n'
    rows = ladder_rows(
        gold, manifest, verdicts, source_dir="ladder-qwen3_8b", gold_bytes=gold_bytes, verdict_bytes=verdict_bytes
    )
    final = next(row for row in rows if row["id"].endswith("/final_verdict"))
    model = next(row for row in rows if row["id"].endswith("/model_verdict"))
    final_fa = next(item for item in final["quality"] if item["name"] == "false_accept")
    model_acc = next(item for item in model["quality"] if item["name"] == "accuracy")
    assert final_fa["value"] == 1.0
    assert model_acc["value"] == 1.0
    assert final["settings"]["quote_rule"] == 1
    assert "/q-1/" in final["id"]
    assert final["provenance"]["commit_sha"] is None
    assert len(final["provenance"]["content_sha256"]) == 64
    assert final["task"] == "verify:math-ladder"
    assert final["model_family"] == "qwen"


def test_an_unknown_check_type_is_not_called_verify():
    manifest = _manifest()
    manifest["settings"]["check_type"] = "planted-error"
    with pytest.raises(ImportError_, match="planted-error"):
        calibration_rows(
            manifest, _METRICS, _verdicts(), _tiers(), source_dir="runs/cal", commit_sha="e" * 40
        )


def test_probe_keeps_a_named_intel_gpu_and_skips_an_unnamed_one():
    named = {
        "openvino": "2026.4.1",
        "devices": {"GPU.0": "Intel(R) Graphics (iGPU)", "CPU": "Intel CPU"},
        "rows": [{"device": "GPU.0", "kind": "embed", "model": "BAAI/bge-base-en-v1.5", "batch": 1, "ms_per_call": 3.5}],
    }
    rows, skipped = probe_rows(named, source_path="results/2026-10-09-mac.json", commit_sha="d" * 40, file_date="2026-10-09")
    assert skipped == []
    assert rows[0]["device"] == "iGPU"
    assert rows[0]["latency"]["cold_ms"] is None
    assert rows[0]["latency"]["source_median_ms"] == 3.5
    unnamed = {"devices": {}, "cases": [{"device": "GPU.0", "kind": "embed", "model": "m", "batch": 1}]}
    kept, skipped = probe_rows(unnamed, source_path="results/chain.json", commit_sha="d" * 40, file_date="2026-10-09")
    assert kept == []
    assert skipped


def test_nli_row_uses_the_prereg_name_and_copies_the_interval():
    document = {
        "date": "2026-10-09",
        "openvino": "2026.4.1",
        "model": {"repo": "cross-encoder/nli-deberta-v3-base", "revision": "abc", "device": "GPU.0", "batch": 1, "seq": 512},
        "legs": {
            "grounded/tune": {
                "metrics": {
                    "n": 10,
                    "false_accept": {"rate": 0.1, "of": 10, "low": 0.0, "high": 0.4},
                    "false_accept_unsupported_only": {"rate": 0.0, "of": 4, "low": 0.0, "high": 0.5},
                    "abstain": {"rate": 0.2, "of": 10, "low": 0.0, "high": 0.5},
                    "decided_balanced_accuracy": 0.8,
                },
                "sec_per_claim": {"mean": 0.01, "max": 0.02},
                "truncated_n": 0,
            }
        },
    }
    rows = nli_rows(document, source_path="nli.json", commit_sha="e" * 40, prereg_says_intel_igpu=True)
    block = document["legs"]["grounded/tune"]["metrics"]["false_accept"]
    metric = next(item for item in rows[0]["quality"] if item["name"] == "false_accept")
    assert rows[0]["device"] == "iGPU"
    assert rows[0]["settings"]["ov_device_id"] == "GPU.0"
    assert metric["value"] == block["rate"]
    assert metric["interval"]["high"] == block["high"]
    with pytest.raises(ValueError):
        nli_rows(document, source_path="nli.json", commit_sha="e" * 40, prereg_says_intel_igpu=False)


def test_offer_snapshot_is_version_locked_and_does_not_invent_a_budget_pass():
    payload = {
        "version": 1,
        "date": "2026-10-09",
        "gpus": [{"gpu_type": "RTX 4090", "usd_per_hr": 0.44, "vram_gb": 24}],
        "budget": {
            "version": 1,
            "providers": {"runpod": {"cap": 20, "remaining": 12, "spent": 1, "committed": 0, "balance": None, "cap_source": "fixture", "warning": None}},
            "ceiling": 20,
        },
    }
    raw = json.dumps(payload).encode("utf-8")
    rows = offer_rows(payload, source_path="fixtures/offers.json", content=raw)
    gpu = next(row for row in rows if row["id"] == "offer/runpod/RTX 4090/2026-10-09")
    budget = next(row for row in rows if row["id"] == "budget/runpod/2026-10-09")
    assert gpu["device"] == "runpod:RTX 4090"
    assert gpu["settings"]["price_snapshot"] == "2026-10-09"
    assert "price_snapshot: 2026-10-09" in gpu["notes"]
    assert budget["settings"]["remaining"] == 12
    for row in rows:
        validate_row(row)
    payload["version"] = 2
    with pytest.raises(ImportError_, match="version"):
        offer_rows(payload, source_path="fixtures/offers.json", content=raw)
    assert remote_option_allowed(13, 12) == (False, "worst case 13 exceeds remaining 12")
    assert remote_option_allowed(1.5, None)[0] is False
    assert remote_option_allowed(1.5, 12)[0] is True


def test_quote_rules_that_disagree_are_refused():
    verdicts = _verdicts()
    verdicts[0]["quote_rule"] = 2
    with pytest.raises(ImportError_, match="disagrees"):
        calibration_rows(_manifest(1), _METRICS, verdicts, _tiers(), source_dir="runs/cal", commit_sha="f" * 40)


def test_probe_skips_error_rows_and_gpu_1():
    document = {
        "devices": {"GPU.0": "Intel(R) Graphics (iGPU)"},
        "cases": [
            {"device": "GPU.1", "kind": "embed", "model": "m", "batch": 1},
            {"device": "NPU", "kind": "embed", "model": "m", "batch": 8, "error": "DEVICE_LOST"},
        ],
    }
    rows, skipped = probe_rows(document, source_path="results/probe.json", commit_sha="d" * 40, file_date="2026-10-09")
    assert rows == []
    assert len(skipped) == 2


def test_gap_report_names_the_missing_field_on_a_partial_row():
    from switchyard.importers.gaps import render_gaps

    rows = [
        {
            "id": "probe/bge/NPU",
            "task": "embed",
            "device": "NPU",
            "model": {"repo_or_tag": "BAAI/bge-base-en-v1.5"},
            "settings": {},
            "latency": {"cold_ms": None, "warm_p50_ms": None},
            "quality": [],
        }
    ]
    candidates = [
        {
            "id": "E1-bge-NPU",
            "experiment": "E1",
            "match": {"task": "embed", "model": "BAAI/bge-base-en-v1.5", "device": "NPU"},
            "needs": ["cold_ms", "warm_p50_ms"],
            "why": "speed without a cold split",
        }
    ]
    text = render_gaps(rows, candidates, ladder_note="Ladder note.", skipped=["chain#1: unnamed GPU"])
    assert "probe/bge/NPU" in text
    assert "cold_ms" in text
    assert "unnamed GPU" in text


def test_import_command_writes_jsonl_and_the_gap_report(tmp_path: Path, monkeypatch, capsys):
    from switchyard.cli import main
    from tests.support import row

    table = tmp_path / "table"
    table.mkdir()
    (table / "candidates.json").write_text(
        json.dumps(
            [
                {
                    "id": "E1-bge-NPU",
                    "experiment": "E1",
                    "match": {"task": "embed", "model": "other", "device": "CPU"},
                    "needs": ["cold_ms"],
                    "why": "open",
                }
            ]
        ),
        encoding="utf-8",
    )

    def fake_import(**kwargs):
        return [row()], ["one skip"]

    monkeypatch.setattr("switchyard.importers.assemble.import_receipts", fake_import)
    assert main(["import", "--root", str(tmp_path)]) == 0
    written = (table / "routing.jsonl").read_text(encoding="utf-8")
    assert "BAAI/bge-base-en-v1.5" in written
    assert "Why E1" in (table / "gaps.md").read_text(encoding="utf-8")
    assert "1 row(s)" in capsys.readouterr().out


def test_table_filters_by_task(tmp_path: Path, capsys):
    from switchyard.cli import main

    table = tmp_path / "table"
    table.mkdir()
    rows = [
        {"id": "a", "task": "embed", "device": "NPU", "model": {"repo_or_tag": "bge"}},
        {"id": "b", "task": "verify:math-ladder", "device": "5090", "model": {"repo_or_tag": "qwen3:8b"}},
    ]
    (table / "routing.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    assert main(["table", "--root", str(tmp_path), "--task", "embed"]) == 0
    out = capsys.readouterr().out
    assert "\ta\tembed\t" in out or out.splitlines()[0].startswith("a\tembed")
    assert "verify" not in out.splitlines()[0]
    assert out.splitlines()[-1] == "1 row(s)"


def _commit(repo: Path) -> str:
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "dev"], cwd=repo, check=True)
    subprocess.run(["git", "config", "core.autocrlf", "false"], cwd=repo, check=True)
    # The ladder paths exceed MAX_PATH when pytest's temp dir is already long.
    subprocess.run(["git", "config", "core.longpaths", "true"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=repo, check=True, capture_output=True)
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def _b_receipts(*, written: str | None = "2026-10-09T03:57:28") -> dict[str, bytes]:
    selection = {
        "seed": 0,
        "splits": ["s1"],
        "control_both_not_wrong": ["n1"],
        "control_both_wrong": ["w1"],
        "order": ["n1", "s1", "w1"],
    }
    if written is not None:
        selection["written"] = written
    verdicts = {
        "s1": {"outcome": "unsure", "seconds": 1.0},
        "n1": {"outcome": "not wrong", "seconds": 1.0},
        "w1": {"outcome": "wrong", "grammatical": True, "seconds": 1.0},
    }
    manifest = {
        "judge": "mistral-small:24b",
        "digest": "8039dd90c1138d772437a0779a33b7349efd5d9cca71edcd26e4dd463f90439d",
        "prompt_sha256": "p" * 64,
        "schema_sha256": "s" * 64,
        "options": {"num_predict": 2048, "num_ctx": 8192},
        "think": False,
        "splits": ["s1"],
        "control": ["n1", "w1"],
        "order": ["n1", "s1", "w1"],
    }
    readout = {
        "control": {"both wrong": {"agrees": 1}, "both not wrong": {"agrees": 1}},
        "reading": {"control_passed": False},
    }
    return {
        "B-selection.json": json.dumps(selection).encode("utf-8"),
        "B-manifest.json": json.dumps(manifest).encode("utf-8"),
        "B-readout.json": json.dumps(readout).encode("utf-8"),
        "verdicts-error-B-mistral-small_24b.json": json.dumps(verdicts).encode("utf-8"),
    }


def _commit_auditor(repo: Path, files: dict[str, bytes], *, extra: dict[str, bytes] | None = None) -> str:
    base = repo / "docs/runs/receipts/2026-10-08-auditor"
    base.mkdir(parents=True)
    for name, data in {**files, **(extra or {})}.items():
        (base / name).write_bytes(data)
    return _commit(repo)


def test_aspire_import_records_the_pin_path_and_content_sha256(tmp_path: Path):
    from switchyard.importers.assemble import _aspire_b
    from switchyard.importers.common import sha256_bytes

    repo = tmp_path / "aspire"
    files = _b_receipts()
    pin = _commit_auditor(
        repo,
        files,
        extra={
            "judge-control-selection.json": b'{"items":[],"order":[]}',
            "judge-control-readout.json": b'{"scores":{}}',
        },
    )
    readout = repo / "docs/runs/receipts/2026-10-08-auditor/B-readout.json"
    committed = readout.read_bytes()
    readout.write_bytes(b'{"control":{}}')
    (row,) = _aspire_b(repo, pin)
    assert row["task"] == "judge:planted-error"
    assert row["date"] == "2026-10-09"
    assert row["provenance"]["commit_sha"] == pin
    assert row["provenance"]["source_receipt_path"] == "docs/runs/receipts/2026-10-08-auditor/B-readout.json"
    assert row["provenance"]["content_sha256"] == sha256_bytes(committed)
    assert row["provenance"]["content_sha256"] != sha256_bytes(readout.read_bytes())


def test_aspire_import_refuses_a_missing_readout_and_an_undated_selection(tmp_path: Path):
    from switchyard.importers.assemble import _aspire_b

    missing = tmp_path / "missing"
    files = _b_receipts()
    del files["B-readout.json"]
    pin = _commit_auditor(missing, files)
    with pytest.raises(ImportError_, match="B-readout.json"):
        _aspire_b(missing, pin)
    undated = tmp_path / "undated"
    pin = _commit_auditor(undated, _b_receipts(written=None))
    with pytest.raises(ImportError_, match="written date"):
        _aspire_b(undated, pin)
    short = tmp_path / "short"
    pin = _commit_auditor(short, _b_receipts(written="yesterday"))
    with pytest.raises(ImportError_, match="written date"):
        _aspire_b(short, pin)
    malformed = tmp_path / "malformed"
    pin = _commit_auditor(malformed, _b_receipts(written="not-a-date"))
    with pytest.raises(ImportError_, match="written date"):
        _aspire_b(malformed, pin)
    nodash = tmp_path / "nodash"
    pin = _commit_auditor(nodash, _b_receipts(written="abcdefghij"))
    with pytest.raises(ImportError_, match="written date"):
        _aspire_b(nodash, pin)


def test_import_reads_a_pinned_tree_and_refuses_a_missing_ladder(tmp_path: Path):
    from switchyard.importers.assemble import import_receipts
    from switchyard.importers.common import sha256_bytes
    from switchyard.importers.pins import LADDER_RUNS

    rnd = tmp_path / "rnd"
    chain = rnd / "experiments/verifier-gold/calibration/results/2026-10-09-chain"
    for index in range(12):
        run = chain / f"cal-{index}"
        run.mkdir(parents=True)
        manifest = _manifest()
        manifest["model"] = f"model-{index}"
        (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8", newline="\n")
        (run / "metrics.json").write_text(json.dumps(_METRICS), encoding="utf-8", newline="\n")
        (run / "verdicts.jsonl").write_text("", encoding="utf-8", newline="\n")
    tiers = rnd / "experiments/verifier-gold/calibration/tiers.json"
    tiers.parent.mkdir(parents=True, exist_ok=True)
    tiers.write_text("{}", encoding="utf-8", newline="\n")
    gold = b'{"id":"a","label":"unsupported","ladder":{"cell":"L1"}}\n'
    gold_path = rnd / "experiments/verifier-gold/math-ladder/ladder.jsonl"
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    gold_path.write_bytes(gold)
    probes = rnd / "experiments/npu-probe/results"
    probes.mkdir(parents=True)
    (probes / "2026-10-09-probe.json").write_text(
        json.dumps({"openvino": "2026.4.1", "devices": {"CPU": "Intel"}, "cases": []}),
        encoding="utf-8",
        newline="\n",
    )
    (probes / "2026-10-09-mac.json").write_text(
        json.dumps({"openvino": "2026.4.1", "devices": {"GPU.0": "Intel(R) Graphics (iGPU)"}, "rows": []}),
        encoding="utf-8",
        newline="\n",
    )
    (probes / "2026-10-09-mac-chain.json").write_text(
        json.dumps({"note": "Contended. Rerun before treating this as the receipt.", "rows": [{"device": "GPU.0"}]}),
        encoding="utf-8",
        newline="\n",
    )
    npu = tmp_path / "npu"
    nli_dir = npu / "experiments/verifier-gold/calibration"
    (nli_dir / "results").mkdir(parents=True)
    (nli_dir / "nli-floor.md").write_text("Intel iGPU found by name.\n", encoding="utf-8", newline="\n")
    nli = {
        "date": "2026-10-09",
        "openvino": "2026.4.1",
        "model": {"repo": "cross-encoder/nli-deberta-v3-base", "revision": "abc", "device": "GPU.0", "batch": 1},
        "legs": {
            "grounded/tune": {
                "metrics": {
                    "n": 1,
                    "false_accept": {"rate": 0.0, "of": 1, "low": 0.0, "high": 1.0},
                    "false_accept_unsupported_only": {"rate": 0.0, "of": 1, "low": 0.0, "high": 1.0},
                    "abstain": {"rate": 0.0, "of": 1, "low": 0.0, "high": 1.0},
                }
            }
        },
    }
    (nli_dir / "results/2026-10-09-nli-floor.json").write_text(json.dumps(nli), encoding="utf-8", newline="\n")
    rnd_pin = _commit(rnd)
    npu_pin = _commit(npu)
    aspire = tmp_path / "aspire"
    aspire_pin = _commit_auditor(
        aspire,
        _b_receipts(),
        extra={"judge-control-readout.json": b'{"scores":{}}'},
    )
    with pytest.raises(ImportError_, match="missing ladder"):
        import_receipts(
            rnd_repo=rnd,
            npu_repo=npu,
            aspire_repo=aspire,
            rnd_pin=rnd_pin,
            npu_pin=npu_pin,
            aspire_pin=aspire_pin,
        )
    digest = sha256_bytes(gold)
    ladder_root = rnd / "experiments/verifier-gold/math-ladder/results/2026-10-09-ladder"
    for index, name in enumerate(LADDER_RUNS):
        run = ladder_root / name
        run.mkdir(parents=True)
        manifest = _manifest()
        manifest["model"] = f"ladder-{index}"
        manifest["gold"] = [{"file": "ladder.jsonl", "sha256": digest}]
        (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8", newline="\n")
        (run / "verdicts.jsonl").write_text(
            '{"claim_id":"a","status":"ok","final_verdict":"unsupported","model_verdict":"unsupported"}\n',
            encoding="utf-8",
            newline="\n",
        )
    rnd_pin = _commit_more(rnd)
    offers = tmp_path / "offers.json"
    offers.write_text(
        json.dumps({"version": 1, "date": "2026-10-09", "gpus": [{"gpu_type": "L4", "usd_per_hr": 0.2}]}),
        encoding="utf-8",
    )
    rows, skipped = import_receipts(
        rnd_repo=rnd,
        npu_repo=npu,
        aspire_repo=aspire,
        offers_path=offers,
        rnd_pin=rnd_pin,
        npu_pin=npu_pin,
        aspire_pin=aspire_pin,
    )
    planted = [row for row in rows if row["task"] == "judge:planted-error"]
    assert len(planted) == 1
    assert planted[0]["provenance"]["commit_sha"] == aspire_pin
    assert not any(row["task"] == "judge:same-meaning-rewrite" for row in rows)
    assert any(row["task"] == "offer" for row in rows)
    assert any(row["task"] == "nli" for row in rows)
    assert any(row["settings"].get("quote_rule") == 1 and "8114f07" in row["settings"].get("quote_rule_source", "") for row in rows)
    assert skipped


def _commit_more(repo: Path) -> str:
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "more"], cwd=repo, check=True, capture_output=True)
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def test_a_missing_pin_is_refused(tmp_path: Path):
    from switchyard.importers.assemble import require_pin

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "dev"], cwd=tmp_path, check=True)
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    subprocess.run(["git", "add", "a.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "a"], cwd=tmp_path, check=True, capture_output=True)
    with pytest.raises(ImportError_):
        require_pin(tmp_path, "0123456789abcdef0123456789abcdef01234567")
