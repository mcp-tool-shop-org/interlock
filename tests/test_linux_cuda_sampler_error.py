"""A VRAM sampler that fails mid-cell is a failed measurement, not a refusal; plus the gaps review hc named.

`sampler_error` is its own outcome. It says the cell was not measured. It never says the card cannot load the
quant, so it must not read as `load_refused` in the schema, the router, the table copy or the importer.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from interlock.devices import RemoteDeviceIdentity
from interlock.harness.linux_cuda import (
    LinuxShapeFailure,
    _query_ps_entry,
    build_linux_failure_row,
    measure_linux_ollama_batch,
    measure_linux_ollama_generate,
    measure_linux_ollama_window,
    run_linux_spec,
)
from interlock.harness.receipts import for_table
from interlock.harness.spec import load_spec
from interlock.importers.remote import import_remote_window
from interlock.receipt import is_speed_row, validate_row
from interlock.router.policy import Refusal, Request, decide

ROOT = Path(__file__).resolve().parents[1]
IDENT = RemoteDeviceIdentity(
    provider="colab",
    card="L4",
    vram_mib=23034,
    driver_version="550.54.15",
    cuda_version="12.4",
    pool="us-central1-a",
)


class Server:
    """A fake Ollama server that records every call in order."""

    version = "0.6.2"

    def __init__(self, unload_raises: bool = False) -> None:
        self.events: list[tuple[str, str]] = []
        self.unload_raises = unload_raises

    def get(self, _path: str) -> dict[str, Any]:
        return {"models": [{"name": "m", "size": 5200000000, "size_vram": 5200000000}]}

    def post(self, _path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.events.append(("post", str(payload.get("model"))))
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

    def unload(self, model: str) -> None:
        self.events.append(("unload", model))
        if self.unload_raises:
            raise OSError("unload failed")


def _sampler_failing_after(n_ok: int):
    calls = {"n": 0}

    def sample():
        calls["n"] += 1
        if calls["n"] > n_ok:
            raise RuntimeError("nvidia-smi went away")
        return (10240, 23034)

    return sample


# ── 1. sampler_error ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "call",
    [
        lambda s, sampler: measure_linux_ollama_window(s, sampler, model="m", num_ctx=2048),
        lambda s, sampler: measure_linux_ollama_generate(s, sampler, model="m", prompt_tokens=512, concurrency=1),
        lambda s, sampler: measure_linux_ollama_batch(s, sampler, model="m", task="embed", batch=1, token_length=128),
    ],
    ids=["window", "generate", "batch"],
)
def test_a_sampler_that_fails_after_the_call_is_not_a_load_refusal(call):
    with pytest.raises(LinuxShapeFailure) as caught:
        call(Server(), _sampler_failing_after(1))
    assert caught.value.failure_class == "sampler_error"
    assert "nvidia-smi went away" in caught.value.reason


def _failure_row(**kw: Any) -> dict:
    return build_linux_failure_row(
        spec_id="C2-FILL-CARD",
        spec_rel="experiments/c2-fill-card/spec.yaml",
        run_id="w",
        date="2026-10-25",
        task="generate",
        model="qwen3:8b",
        revision_or_digest="500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41",
        remote_device=IDENT,
        failure_class="sampler_error",
        reason="VRAM sampler failed: nvidia-smi went away",
        quant="Q4_K_M",
        licence="apache-2.0",
        **kw,
    )


def test_a_sampler_error_row_validates_and_is_not_a_speed_row():
    row = _failure_row()
    validate_row(row)
    assert row["outcome"] == "sampler_error"
    assert row["notes"].startswith("sampler_error:")
    assert is_speed_row(row) is False
    assert row["latency"]["throughput"] is None


def test_the_table_copy_of_a_sampler_error_row_does_not_say_the_device_cannot():
    copied = for_table(_failure_row())
    assert copied["outcome"] == "sampler_error"
    assert "can't do this setting" not in copied["notes"]


def test_a_sampler_error_row_imports_from_a_window_receipt():
    raw = (json.dumps(_failure_row(), sort_keys=True) + "\n").encode("utf-8")
    rows = import_remote_window(raw, source_path="receipts/w.jsonl")
    assert rows[0]["outcome"] == "sampler_error"


def test_the_router_lists_a_sampler_error_row_as_excluded_and_never_as_a_refusal_reason():
    result = decide(
        [_failure_row()],
        Request(task="generate", as_of=date(2026, 10, 26), model="qwen3:8b", allow_remote=True, device="colab:L4"),
    )
    assert isinstance(result, Refusal)
    document = result.document
    assert [item["code"] for item in document["excluded"]] == ["sampler_error"]
    assert "Measured failure reasons" not in json.dumps(document)


def test_a_fill_card_cell_whose_sampler_fails_is_one_sampler_error_row_not_a_measured_ceiling():
    spec = load_spec(ROOT / "experiments/c2-fill-card/spec.yaml")
    spec["models"] = [
        e for e in spec["models"] if e["model"] == "qwen3:8b" and e["device"] == "colab:L4"
    ][:1]
    rows = run_linux_spec(
        spec,
        spec_path=ROOT / "experiments/c2-fill-card/spec.yaml",
        spec_rel="experiments/c2-fill-card/spec.yaml",
        run_id="s",
        date="2026-10-25",
        remote_device=IDENT,
        sample_vram_mib=_sampler_failing_after(3),
        server=Server(),  # type: ignore[arg-type]
    )
    assert [row["outcome"] for row in rows] == ["sampler_error"]
    assert rows[0]["settings"]["failure_class"] == "sampler_error"


# ── 2. the gaps review hc named ──────────────────────────────────────────────────────────────────


def _narrowed(spec_rel: str, model: str, quants: int = 2, **match: Any) -> dict:
    spec = load_spec(ROOT / spec_rel)
    spec["models"] = [
        e
        for e in spec["models"]
        if e["model"] == model
        and e["device"] == "colab:L4"
        and not e.get("expected_outcome")
        and all(e.get(k) == v for k, v in match.items())
    ][:quants]
    assert len(spec["models"]) == quants, (spec_rel, model)
    return spec


def _run(spec: dict, spec_rel: str, server: Server, on_rows=None, sampler=None):
    return run_linux_spec(
        spec,
        spec_path=ROOT / spec_rel,
        spec_rel=spec_rel,
        run_id="g",
        date="2026-10-25",
        remote_device=IDENT,
        sample_vram_mib=sampler or (lambda: (10240, 23034)),
        server=server,  # type: ignore[arg-type]
        on_rows=on_rows,
    )


@pytest.mark.parametrize(
    ("spec_rel", "model", "extra"),
    [
        ("experiments/c2-generate/spec.yaml", "qwen3:8b", {}),
        ("experiments/c2-throughput/spec.yaml", "BAAI/bge-base-en-v1.5", {}),
    ],
    ids=["generate-ladder", "batch-ladder"],
)
def test_ladders_unload_each_quant_and_flush_after_each_cell(spec_rel, model, extra):
    quants = 2 if "generate" in spec_rel else 1
    spec = _narrowed(spec_rel, model, quants=quants, **extra)
    server = Server()
    seen: list[int] = []
    rows = _run(spec, spec_rel, server, on_rows=lambda done: seen.append(len(done)))
    unloaded = [tag for kind, tag in server.events if kind == "unload"]
    assert len(unloaded) == quants
    assert len(set(unloaded)) == quants
    assert seen[-1] == len(rows)
    assert len(rows) > quants
    assert seen == sorted(seen)
    assert len(set(seen)) >= len(rows)  # a flush for every new row, not one at the end


def test_each_quant_is_unloaded_before_the_next_quants_first_cell():
    spec_rel = "experiments/c2-fill-card/spec.yaml"
    spec = _narrowed(spec_rel, "qwen3:8b", quants=3)
    server = Server()
    _run(spec, spec_rel, server)
    kinds: list[tuple[str, str]] = []
    for event in server.events:
        if not kinds or kinds[-1] != event:
            kinds.append(event)
    tags = list(dict.fromkeys(tag for kind, tag in kinds if kind == "post"))
    assert len(tags) == 3
    expected: list[tuple[str, str]] = []
    for tag in tags:
        expected += [("post", tag), ("unload", tag)]
    assert kinds == expected


def test_an_unload_that_raises_does_not_replace_the_cells_own_error():
    spec_rel = "experiments/c2-fill-card/spec.yaml"
    spec = _narrowed(spec_rel, "qwen3:8b", quants=1)
    server = Server(unload_raises=True)

    def sampler():
        raise RuntimeError("cell boom")

    with pytest.raises(RuntimeError, match="cell boom"):
        _run(spec, spec_rel, server, sampler=sampler)
    assert [kind for kind, _tag in server.events] == ["unload"]


def test_an_unload_that_raises_after_a_good_cell_leaves_the_row():
    spec_rel = "experiments/c2-fill-card/spec.yaml"
    spec = _narrowed(spec_rel, "qwen3:8b", quants=1)
    rows = _run(spec, spec_rel, Server(unload_raises=True))
    assert [row["model"]["quant"] for row in rows] == [spec["models"][0]["quant"]]


@pytest.mark.parametrize(("resident", "asked"), [("foo", "foo:latest"), ("foo:latest", "foo")])
def test_the_ps_query_matches_latest_in_both_directions(resident, asked):
    class Ps:
        def get(self, _path):
            return {
                "models": [
                    {"name": "other:q4", "model": "other:q4", "size": 1, "size_vram": 1},
                    {"name": resident, "model": resident, "size": 700, "size_vram": 600},
                ]
            }

    assert _query_ps_entry(Ps(), asked) == (700, 600)  # type: ignore[arg-type]
