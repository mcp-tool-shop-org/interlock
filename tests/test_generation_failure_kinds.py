"""A looped generation and a request timeout are not load refusals; the generate ladder says which text it sent."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from interlock.devices import RemoteDeviceIdentity
from interlock.harness.linux_cuda import (
    GENERATE_LADDER_VERSION,
    LinuxShapeFailure,
    build_linux_failure_row,
    classify_linux_error,
    ladder_prompt,
    measure_linux_ollama_generate,
    run_linux_generate_ladder,
)
from interlock.receipt import is_speed_row, validate_row
from interlock.router.policy import EXCLUDED_OUTCOMES, Refusal, Request, decide

IDENT = RemoteDeviceIdentity(
    provider="colab", card="T4", vram_mib=15360, driver_version="550.54.15", cuda_version="12.4", pool="us-central1-a"
)
LOOP = "prediction aborted, token repeat limit reached"


@pytest.mark.parametrize(
    ("message", "kind"),
    [
        (LOOP, "generation_aborted"),
        ("HTTP 500: " + LOOP, "generation_aborted"),
        ("HTTP 500 after 2m0s", "timeout"),
        ("Post http://127.0.0.1:11434/api/generate: context deadline exceeded", "timeout"),
        ("the read timed out", "timeout"),
        ("model failed to load: resource limitations", "load_refused"),
        ("something else broke", "load_refused"),
        ("CUDA error: out of memory", "oom"),
    ],
)
def test_each_generation_error_gets_its_own_kind(message, kind):
    assert classify_linux_error(RuntimeError(message)).failure_class == kind


class _Server:
    """A stub Ollama that returns whatever the test says."""

    def __init__(self, reply: dict[str, Any]) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def get(self, _path: str) -> dict[str, Any]:
        return {"models": [{"name": "m", "size": 5, "size_vram": 5}]}

    def post(self, _path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.prompts.append(payload["prompt"])
        return self.reply


def _measure(reply: dict[str, Any]) -> dict[str, Any]:
    return measure_linux_ollama_generate(
        _Server(reply), lambda: (1000, 15360), model="m", prompt_tokens=512, concurrency=1, repeats=1
    )


def test_a_stub_backend_that_aborts_a_loop_yields_generation_aborted_not_load_refused():
    with pytest.raises(LinuxShapeFailure) as caught:
        _measure({"error": LOOP})
    assert caught.value.failure_class == "generation_aborted"
    assert LOOP in caught.value.reason


def test_a_stub_backend_that_times_out_yields_timeout_not_load_refused():
    with pytest.raises(LinuxShapeFailure) as caught:
        _measure({"error": "HTTP 500 after 2m0s"})
    assert caught.value.failure_class == "timeout"


def test_a_stub_backend_that_cannot_load_the_model_still_yields_load_refused():
    with pytest.raises(LinuxShapeFailure) as caught:
        _measure({"error": "model failed to load, this may be due to resource limitations"})
    assert caught.value.failure_class == "load_refused"


def _row(kind: str) -> dict:
    return build_linux_failure_row(
        spec_id="C2-GENERATE",
        spec_rel="experiments/c2-generate/spec.yaml",
        run_id="t4-r2",
        date="2026-10-26",
        task="generate",
        model="phi3.5:3.8b",
        revision_or_digest="d" * 64,
        remote_device=IDENT,
        failure_class=kind,
        reason=LOOP if kind == "generation_aborted" else "HTTP 500 after 2m0s",
        num_ctx=2048,
        prompt_tokens=2048,
        concurrency=1,
        quant="Q4_K_M",
        licence="mit",
    )


@pytest.mark.parametrize("kind", ["generation_aborted", "timeout"])
def test_the_failure_rows_validate_carry_the_ladder_version_and_are_not_speeds(kind):
    row = _row(kind)
    validate_row(row)
    assert row["outcome"] == kind
    assert row["settings"]["ladder_version"] == GENERATE_LADDER_VERSION == 2
    assert is_speed_row(row) is False


def test_the_router_excludes_the_new_kinds_as_it_excludes_load_refused():
    assert {"generation_aborted", "timeout", "load_refused"} <= set(EXCLUDED_OUTCOMES)
    for kind in ("generation_aborted", "timeout", "load_refused"):
        row = _row(kind if kind != "load_refused" else "load_refused")
        request = Request(task="generate", as_of=date(2026, 10, 26), model="phi3.5:3.8b", allow_remote=True)
        result = decide([row], request)
        assert isinstance(result, Refusal)
        assert {"id": row["id"], "code": kind} in result.document["excluded"]


def test_a_generate_ladder_row_records_the_ladder_version_and_an_aborted_cell_is_its_own_row():
    def measure_shape(prompt_len: int, concurrency: int) -> dict[str, Any]:
        if prompt_len == 2048:
            raise LinuxShapeFailure("generation_aborted", LOOP)
        return {"ttft_ms": 100.0, "decode_tok_s": 30.0, "peak_mib": 6000, "size": 5, "size_vram": 5}

    rows = run_linux_generate_ladder(
        spec_id="C2-GENERATE",
        spec_rel="experiments/c2-generate/spec.yaml",
        run_id="t4-r2",
        date="2026-10-26",
        remote_device=IDENT,
        model="phi3.5:3.8b",
        revision_or_digest="d" * 64,
        measure_shape=measure_shape,
        prompt_lengths=(1024, 2048),
        concurrencies=(1,),
        quant="Q4_K_M",
        licence="mit",
    )
    assert [r.get("outcome") for r in rows] == [None, "generation_aborted"]
    assert {r["settings"]["ladder_version"] for r in rows} == {GENERATE_LADDER_VERSION}


def test_the_ladder_prompt_is_varied_deterministic_and_does_not_repeat_the_old_token():
    text = ladder_prompt(2048)
    assert text == ladder_prompt(2048)
    body = text.split("\n\n")[0].split()
    assert len(body) == 2048
    assert "token" not in set(body)
    assert len(set(body)) > 40
    # no short period: no window of 8 words repeats back to back anywhere
    assert all(body[i : i + 8] != body[i + 8 : i + 16] for i in range(0, len(body) - 16))
    assert len(ladder_prompt(1)) > 0 and ladder_prompt(0).split("\n\n")[0].count(" ") == 0


def test_the_generate_cell_sends_the_ladder_prompt():
    server = _Server({"prompt_eval_duration": 1, "eval_count": 8, "eval_duration": 1})
    measure_linux_ollama_generate(server, lambda: (1000, 15360), model="m", prompt_tokens=256, concurrency=1, repeats=1)
    assert set(server.prompts) == {ladder_prompt(256)}
