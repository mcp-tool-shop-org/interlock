"""`interlock import --remote-window` takes the routing-row JSONL that `interlock run` writes on Linux.

The old envelope ({run_id, date, remote_device, records}) stays; its tests live in test_importers.py. Here: the
receipt a Colab window writes is imported as it is, keeps its provenance, and merges into the table instead of
replacing it. No GPU and no Colab session: run_linux_spec on fakes writes the rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from interlock.cli import main
from interlock.devices import RemoteDeviceIdentity
from interlock.harness.linux_cuda import build_linux_failure_row, run_linux_spec
from interlock.harness.spec import load_spec
from interlock.importers.common import ImportError_
from interlock.importers.remote import import_remote_window
from interlock.router.policy import experiment_of

ROOT = Path(__file__).resolve().parents[1]
IDENT = RemoteDeviceIdentity(
    provider="colab",
    card="L4",
    vram_mib=23034,
    driver_version="550.54.15",
    cuda_version="12.4",
    pool="us-central1-a",
)


class FakeServer:
    version = "0.6.2"

    def get(self, path: str) -> dict[str, Any]:
        return {"models": [{"name": "m", "size": 5200000000, "size_vram": 5200000000}]}

    def post(self, _path: str, payload: dict[str, Any]) -> dict[str, Any]:
        ctx = int((payload.get("options") or {}).get("num_ctx") or 2048)
        if ctx > 8192:
            return {"error": f"request ({ctx} tokens) exceeds the available context size (8192 tokens)"}
        return {
            "load_duration": 20_000_000,
            "prompt_eval_count": int(ctx * 0.9),
            "prompt_eval_duration": 100_000_000,
            "eval_count": 16,
            "eval_duration": 250_000_000,
            "total_duration": 25_000_000,
        }

    def unload(self, _model: str) -> None:
        return None


def _rows_for(spec_rel: str, model: str, run_id: str, **narrow: Any) -> list[dict]:
    spec = load_spec(ROOT / spec_rel)
    spec["models"] = [
        entry
        for entry in spec["models"]
        if entry.get("model") == model
        and entry.get("device") == "colab:L4"
        and not entry.get("expected_outcome")
        and all(entry.get(k) == v for k, v in narrow.items())
    ][:1]
    assert spec["models"], (spec_rel, model)
    return run_linux_spec(
        spec,
        spec_path=ROOT / spec_rel,
        spec_rel=spec_rel,
        run_id=run_id,
        date="2026-10-25",
        remote_device=IDENT,
        sample_vram_mib=lambda: (10240, 23034),
        server=FakeServer(),  # type: ignore[arg-type]
    )


@pytest.fixture(scope="module")
def window_rows() -> list[dict]:
    fill = _rows_for("experiments/c2-fill-card/spec.yaml", "qwen3:8b", "w1")
    generate = _rows_for("experiments/c2-generate/spec.yaml", "qwen3:8b", "w1")[:1]
    throughput = _rows_for("experiments/c2-throughput/spec.yaml", "BAAI/bge-base-en-v1.5", "w1")[:1]
    refused = build_linux_failure_row(
        spec_id="C2-FILL-CARD",
        spec_rel="experiments/c2-fill-card/spec.yaml",
        run_id="w1",
        date="2026-10-25",
        task="generate",
        model="qwen3:8b",
        revision_or_digest="500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41",
        remote_device=IDENT,
        failure_class="load_refused",
        reason="pre-registered refusal",
        quant="BF16",
        licence="apache-2.0",
    )
    return [*fill, *generate, *throughput, refused]


def _jsonl(rows: list[dict]) -> bytes:
    return ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode("utf-8")


def test_a_linux_receipt_imports_as_it_is(window_rows):
    assert len(window_rows) == 4
    imported = import_remote_window(_jsonl(window_rows), source_path="receipts/w1.jsonl")
    assert len(imported) == 4
    assert [r["provenance"] for r in imported] == [r["provenance"] for r in window_rows]
    assert [experiment_of(r) for r in imported] == ["C2-FILL-CARD", "C2-GENERATE", "C2-THROUGHPUT", "C2-FILL-CARD"]
    for original, row in zip(window_rows, imported, strict=True):
        assert json.dumps(row["settings"], sort_keys=True) == json.dumps(original["settings"], sort_keys=True)
        assert row.get("notes") == original.get("notes")
    assert "input_fixture_sha256" in imported[2]["settings"]


def test_a_single_row_object_imports(window_rows):
    one = json.dumps(window_rows[0]).encode("utf-8")
    assert len(import_remote_window(one, source_path="receipts/w1.jsonl")) == 1


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: {**r, "device": "5090"}, "not a remote"),
        (lambda r: {**r, "descriptive": True}, "descriptive"),
        (lambda r: {**r, "aborted": True}, "aborted"),
    ],
    ids=["local-device", "descriptive", "aborted"],
)
def test_a_jsonl_row_that_cannot_route_is_refused_naming_the_file(window_rows, mutate, message):
    raw = _jsonl([mutate(window_rows[0])])
    with pytest.raises(ImportError_, match=rf"receipts/bad.jsonl.*{message}"):
        import_remote_window(raw, source_path="receipts/bad.jsonl")


def test_a_jsonl_row_that_fails_the_schema_is_refused_naming_the_file(window_rows):
    broken = {k: v for k, v in window_rows[0].items() if k != "latency"}
    with pytest.raises(ImportError_, match="receipts/bad.jsonl"):
        import_remote_window(_jsonl([broken]), source_path="receipts/bad.jsonl")


def test_a_jsonl_line_that_is_not_json_is_refused_naming_the_file(window_rows):
    raw = _jsonl(window_rows[:1]) + b"{not json\n"
    with pytest.raises(ImportError_, match="receipts/bad.jsonl.*invalid"):
        import_remote_window(raw, source_path="receipts/bad.jsonl")


def _seeded_table(root: Path) -> list[str]:
    table = root / "table"
    table.mkdir()
    old = [
        {"id": "E2/z-old", "task": "generate", "device": "5090"},
        {"id": "E1/a-old", "task": "embed", "device": "NPU"},
    ]
    # Rows only need to be rows here: the import does not validate what it keeps.
    lines = [json.dumps(r, sort_keys=True) for r in sorted(old, key=lambda r: r["id"])]
    (table / "routing.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def test_import_remote_window_merges_into_the_table_without_import_receipts(tmp_path, window_rows, capsys):
    old_lines = _seeded_table(tmp_path)
    window = tmp_path / "w1.jsonl"
    window.write_bytes(_jsonl(window_rows))

    assert main(["import", "--root", str(tmp_path), "--remote-window", str(window)]) == 0
    out = capsys.readouterr()
    assert "Traceback" not in out.err

    lines = (tmp_path / "table" / "routing.jsonl").read_text(encoding="utf-8").splitlines()
    ids = [json.loads(line)["id"] for line in lines]
    assert ids == sorted(ids)
    assert len(lines) == 2 + 4
    for old in old_lines:
        assert old in lines
    assert not list((tmp_path / "table").glob("*.tmp"))

    # the second import of the same file is refused and leaves the table as it was
    before = (tmp_path / "table" / "routing.jsonl").read_bytes()
    assert main(["import", "--root", str(tmp_path), "--remote-window", str(window)]) == 1
    err = capsys.readouterr().err
    assert "duplicate row id" in err
    assert "Traceback" not in err
    assert (tmp_path / "table" / "routing.jsonl").read_bytes() == before


def test_import_remote_window_reports_a_bad_file_without_a_traceback(tmp_path, capsys):
    _seeded_table(tmp_path)
    bad = tmp_path / "bad.jsonl"
    bad.write_bytes(b"{not json\n")
    assert main(["import", "--root", str(tmp_path), "--remote-window", str(bad)]) == 1
    err = capsys.readouterr().err
    assert "bad.jsonl" in err
    assert "Traceback" not in err


def test_import_remote_window_builds_the_table_when_there_is_none(tmp_path, window_rows):
    window = tmp_path / "w1.jsonl"
    window.write_bytes(_jsonl(window_rows))
    assert main(["import", "--root", str(tmp_path), "--remote-window", str(window)]) == 0
    assert len((tmp_path / "table" / "routing.jsonl").read_text(encoding="utf-8").splitlines()) == 4


def test_a_full_rebuild_that_would_drop_a_table_row_is_refused(tmp_path, monkeypatch, capsys):
    old_lines = _seeded_table(tmp_path)
    table_file = tmp_path / "table" / "routing.jsonl"
    # a row the rebuild will not produce, as E2-E4 rows are not produced by import_receipts
    kept = json.loads(old_lines[0])
    monkeypatch.setattr("interlock.importers.assemble.import_receipts", lambda **_kw: ([kept], []))
    before = table_file.read_bytes()
    offers = tmp_path / "offers.json"
    offers.write_text("[]", encoding="utf-8")

    assert main(["import", "--root", str(tmp_path), "--offers", str(offers)]) == 1
    err = capsys.readouterr().err
    assert "REFUSED" in err
    assert "would write 1 row(s) and drop 1 of the 2" in err
    assert json.loads(old_lines[1])["id"] in err
    assert "Traceback" not in err
    assert table_file.read_bytes() == before
