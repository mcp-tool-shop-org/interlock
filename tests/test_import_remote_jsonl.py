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


@pytest.mark.parametrize("drop_key", ["latency", "model", "id"])
def test_a_jsonl_row_that_fails_the_schema_is_refused_naming_the_file_and_row(window_rows, drop_key):
    broken = {k: v for k, v in window_rows[0].items() if k != drop_key}
    with pytest.raises(ImportError_, match=rf"receipts/bad\.jsonl\[1\].*{drop_key}"):
        import_remote_window(_jsonl([window_rows[0], broken]), source_path="receipts/bad.jsonl")


@pytest.mark.parametrize(
    "bad_model", [None, ["not", "a", "mapping"], {"repo_or_tag": 7}], ids=["none", "list", "non-string-tag"]
)
def test_a_jsonl_row_with_a_malformed_model_is_an_import_error_not_a_crash(window_rows, bad_model):
    broken = {**window_rows[0], "model": bad_model}
    with pytest.raises(ImportError_, match=r"receipts/bad\.jsonl\[1\]"):
        import_remote_window(_jsonl([window_rows[0], broken]), source_path="receipts/bad.jsonl")


def test_a_jsonl_line_that_is_not_json_is_refused_naming_the_file_and_line(window_rows):
    raw = _jsonl(window_rows[:1]) + b"{not json\n"
    with pytest.raises(ImportError_, match=r"receipts/bad\.jsonl:2: invalid UTF-8 JSON"):
        import_remote_window(raw, source_path="receipts/bad.jsonl")


def _owned(row_id: str) -> dict:
    """A row a full rebuild owns: a pinned import (branch-1 provenance)."""

    return {
        "id": row_id,
        "task": "generate",
        "device": "5090",
        "provenance": {"source_receipt_path": "receipts/x.json", "commit_sha": "c0ffee"},
    }


def _harness(row_id: str) -> dict:
    """A row the rebuild does not own: a harness row (branch-2 provenance), as E2-E4 are."""

    return {
        "id": row_id,
        "task": "generate",
        "device": "5090",
        "provenance": {"experiment_spec": "experiments/e2/spec.yaml", "run_id": "r"},
    }


def _write_table(root: Path, rows: list[dict]) -> bytes:
    table = root / "table"
    table.mkdir(exist_ok=True)
    (table / "candidates.json").write_text("[]", encoding="utf-8")
    data = "".join(json.dumps(r, sort_keys=True) + "\n" for r in sorted(rows, key=lambda r: r["id"]))
    (table / "routing.jsonl").write_text(data, encoding="utf-8")
    return data.encode("utf-8")


def _seeded_table(root: Path) -> list[str]:
    _write_table(root, [_harness("E2/z-old"), _harness("E1x/a-old")])
    return (root / "table" / "routing.jsonl").read_text(encoding="utf-8").splitlines()


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


def _rebuild_returns(monkeypatch, rows: list[dict]) -> None:
    monkeypatch.setattr("interlock.importers.assemble.import_receipts", lambda **_kw: (rows, []))


def _offers(root: Path) -> list[str]:
    path = root / "offers.json"
    path.write_text("[]", encoding="utf-8")
    return [
        "--offers",
        str(path),
        "--source",
        "rnd=clones/rnd",
        "--source",
        "npu=clones/rnd",
        "--source",
        "aspire=clones/aspire",
    ]


def test_a_rebuild_keeps_every_row_it_does_not_own(tmp_path, monkeypatch, capsys, window_rows):
    harness = [_harness(f"E2/n{i:02d}") for i in range(53)]
    remote = window_rows[0]
    _write_table(tmp_path, [_owned("ladder/a"), *harness, remote])
    before = {
        json.loads(line)["id"]: line
        for line in (tmp_path / "table" / "routing.jsonl").read_text(encoding="utf-8").splitlines()
    }
    _rebuild_returns(monkeypatch, [{**_owned("ladder/a"), "task": "embed"}, _owned("ladder/b")])

    assert main(["import", "--root", str(tmp_path), *_offers(tmp_path)]) == 0
    out = capsys.readouterr()
    assert "Traceback" not in out.err
    lines = (tmp_path / "table" / "routing.jsonl").read_text(encoding="utf-8").splitlines()
    after = {json.loads(line)["id"]: line for line in lines}
    assert len(lines) == 53 + 1 + 2
    assert list(after) == sorted(after)
    for row_id in [*(h["id"] for h in harness), remote["id"]]:
        assert after[row_id] == before[row_id]
    assert json.loads(after["ladder/a"])["task"] == "embed"
    assert "ladder/b" in after


def test_a_rebuild_that_no_longer_produces_a_row_it_owns_is_refused_unless_allowed(tmp_path, monkeypatch, capsys):
    _write_table(tmp_path, [_owned("ladder/a"), _owned("ladder/b"), _harness("E2/keep")])
    table_file = tmp_path / "table" / "routing.jsonl"
    before = table_file.read_bytes()
    # more rows than the table holds, so a row-count check alone would pass; ladder/b is still gone
    _rebuild_returns(monkeypatch, [_owned("ladder/a"), _owned("ladder/c"), _owned("ladder/d"), _owned("ladder/e")])

    assert main(["import", "--root", str(tmp_path), *_offers(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("REFUSED: the rebuild no longer produces 1 of the 2 row(s) it owns")
    assert "ladder/b" in err
    assert "--allow-shrink" in err
    assert "Traceback" not in err
    assert table_file.read_bytes() == before
    assert not (tmp_path / "table" / "import-dropped.txt").exists()

    assert main(["import", "--root", str(tmp_path), *_offers(tmp_path), "--allow-shrink"]) == 0
    out = capsys.readouterr().out
    assert "dropped ladder/b" in out
    assert (tmp_path / "table" / "import-dropped.txt").read_text(encoding="utf-8").splitlines() == ["ladder/b"]
    ids = [json.loads(line)["id"] for line in table_file.read_text(encoding="utf-8").splitlines()]
    assert "ladder/b" not in ids
    assert "E2/keep" in ids


@pytest.mark.parametrize("path", ["merge", "rebuild"])
def test_a_table_with_duplicate_ids_is_refused_in_both_import_paths(tmp_path, monkeypatch, capsys, window_rows, path):
    table = tmp_path / "table"
    table.mkdir()
    row = json.dumps(_harness("E2/dup"), sort_keys=True)
    (table / "routing.jsonl").write_text(f"{row}\n{row}\n", encoding="utf-8")
    before = (table / "routing.jsonl").read_bytes()
    if path == "merge":
        window = tmp_path / "w1.jsonl"
        window.write_bytes(_jsonl(window_rows))
        argv = ["import", "--root", str(tmp_path), "--remote-window", str(window)]
    else:
        _rebuild_returns(monkeypatch, [_owned("ladder/a")])
        argv = ["import", "--root", str(tmp_path), *_offers(tmp_path)]

    assert main(argv) == 1
    err = capsys.readouterr().err
    assert "duplicate row id(s): E2/dup" in err
    assert "Traceback" not in err
    assert (table / "routing.jsonl").read_bytes() == before


def test_a_missing_window_file_and_an_unreadable_table_are_messages_not_tracebacks(tmp_path, capsys):
    assert main(["import", "--root", str(tmp_path), "--remote-window", str(tmp_path / "nope.jsonl")]) == 1
    err = capsys.readouterr().err
    assert "nope.jsonl" in err
    assert "Traceback" not in err

    table = tmp_path / "table"
    table.mkdir()
    (table / "routing.jsonl").write_bytes(b"\xff\xfe not utf-8\n")
    window = tmp_path / "w.jsonl"
    window.write_bytes(b"{}\n")
    assert main(["import", "--root", str(tmp_path), "--remote-window", str(window)]) == 1
    assert "Traceback" not in capsys.readouterr().err


@pytest.mark.parametrize("candidates", [None, "{not json"], ids=["missing", "invalid"])
def test_a_rebuild_whose_candidates_file_is_bad_writes_nothing(tmp_path, monkeypatch, capsys, candidates):
    _write_table(tmp_path, [_owned("ladder/a"), _owned("ladder/b")])
    table = tmp_path / "table"
    if candidates is None:
        (table / "candidates.json").unlink()
    else:
        (table / "candidates.json").write_text(candidates, encoding="utf-8")
    (table / "import-dropped.txt").write_text("earlier\n", encoding="utf-8")
    before = {name: (table / name).read_bytes() for name in ("routing.jsonl", "import-dropped.txt")}
    _rebuild_returns(monkeypatch, [_owned("ladder/a")])

    assert main(["import", "--root", str(tmp_path), *_offers(tmp_path), "--allow-shrink"]) == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert {name: (table / name).read_bytes() for name in before} == before
    assert not (table / "gaps.md").exists()
    assert not list(table.glob("*.tmp"))
