"""A looped generation and a request timeout are not load refusals; the generate ladder says which text it sent."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from interlock.devices import RemoteDeviceIdentity
from interlock.harness.linux_cuda import (
    _LADDER_WORDS,
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
        ("error loading model: unsupported architecture", "load_refused"),
        ("unable to load model", "load_refused"),
        ("model is too large for system memory", "load_refused"),
        ("the prompt is longer than the context length currently available to the model", "ctx_exceeded"),
        ("something else broke", "unmeasured"),
        ("the room was a bloom", "unmeasured"),
        ("oom-killer fired", "oom"),
        ("CUDA error: out of memory", "oom"),
    ],
)
def test_each_generation_error_gets_its_own_kind(message, kind):
    assert classify_linux_error(RuntimeError(message)).failure_class == kind


INSTRUCTION_TOKENS = 14


class _Server:
    """A stub Ollama that returns whatever the test says."""

    def __init__(self, reply: dict[str, Any], tokens_per_word: float = 1.0) -> None:
        self.reply = reply
        self.tokens_per_word = tokens_per_word
        self.prompts: list[str] = []

    def get(self, _path: str) -> dict[str, Any]:
        return {"models": [{"name": "m", "size": 5, "size_vram": 5}]}

    def post(self, _path: str, payload: dict[str, Any]) -> dict[str, Any]:
        options = payload["options"]
        if options["num_predict"] != 1:
            self.prompts.append(payload["prompt"])
        if "error" in self.reply:
            return self.reply
        return {"prompt_eval_count": self.evaluated(payload), **self.reply}

    def evaluated(self, payload: dict[str, Any]) -> int:
        """What real Ollama reports: the prompt's tokens, or about half the window when the prompt overflows it."""

        body = payload["prompt"].split(chr(10) * 2)[0]
        tokens = int(len(body.split()) * self.tokens_per_word) + INSTRUCTION_TOKENS
        window = payload["options"]["num_ctx"]
        if tokens <= window - 1:
            return tokens
        # Ollama's contextShiftPromptLimit (llama_server.go, v0.40.9 and v0.40.11): num_keep is 4.
        return window - max((window - 4) // 2, 1)


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
        _measure({"error": "unable to load full model on GPU"})
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
    assert len(_LADDER_WORDS) == len(set(_LADDER_WORDS))  # no word listed twice
    # no short period: no window of 8 words repeats back to back anywhere
    assert all(body[i : i + 8] != body[i + 8 : i + 16] for i in range(0, len(body) - 16))
    assert len(ladder_prompt(1)) > 0 and ladder_prompt(0).split("\n\n")[0].count(" ") == 0


def test_the_generate_cell_sends_the_ladder_prompt():
    server = _Server({"prompt_eval_duration": 1, "eval_count": 8, "eval_duration": 1})
    result = measure_linux_ollama_generate(
        server, lambda: (1000, 15360), model="m", prompt_tokens=256, concurrency=1, repeats=1
    )
    (prompt,) = set(server.prompts)
    assert prompt == ladder_prompt(len(prompt.split()) - 12)  # the calibrated word count plus the instruction's words
    assert abs(result["prompt_tokens_evaluated"] - 256) <= 5
    assert result["prompt_tokens_evaluated"] == server.evaluated({"prompt": prompt, "options": {"num_ctx": 10**6}})


@pytest.mark.parametrize("tokens_per_word", [1.0, 1.5, 2.0])
@pytest.mark.parametrize("target", [512, 2048, 8192])
def test_the_recorded_prompt_length_is_what_ollama_evaluated_and_num_ctx_has_room_for_it(target, tokens_per_word):
    server = _Server({"prompt_eval_duration": 1, "eval_count": 8, "eval_duration": 1}, tokens_per_word=tokens_per_word)
    result = measure_linux_ollama_generate(
        server, lambda: (1000, 15360), model="m", prompt_tokens=target, concurrency=1, repeats=1, num_predict=128
    )
    assert abs(result["prompt_tokens_evaluated"] - target) <= target * 0.02
    assert result["prompt_tokens_target"] == target


def test_the_fake_server_truncates_the_way_ollama_does():
    server = _Server({})
    window = {"num_ctx": 2240}
    body = " ".join(["w"] * 3000) + chr(10) * 2 + "x"
    assert server.evaluated({"prompt": body, "options": window}) == 1122  # measured on Ollama 0.40.9
    assert server.evaluated({"prompt": "w " * 100, "options": window}) == 114


def test_the_calibration_never_sends_an_overflowing_prompt_to_size_the_prompt():
    server = _Server({"prompt_eval_duration": 1, "eval_count": 8, "eval_duration": 1}, tokens_per_word=1.5)
    sent: list[int] = []
    original = server.post

    def post(path, payload):
        sent.append(server.evaluated(payload))
        return original(path, payload)

    server.post = post
    measure_linux_ollama_generate(
        server, lambda: (1000, 15360), model="m", prompt_tokens=2048, concurrency=1, repeats=1
    )
    window = 2048 + 128 + 64
    assert all(count <= window for count in sent)  # a reading of window // 2 would mean a truncated prompt


def test_a_phrase_list_load_refusal_is_not_the_runner_crash_or_a_pull_error():
    for text, kind in (
        ("model runner has unexpectedly stopped, this may be due to resource limitations", "unmeasured"),
        ("failed to load manifest for registry.ollama.ai/library/x", "unmeasured"),
        ("error loading model: tensor mismatch", "load_refused"),
        ("failed to load model: out of space", "load_refused"),
        ("insufficient memory", "load_refused"),
    ):
        assert classify_linux_error(RuntimeError(text)).failure_class == kind, text


def test_a_prompt_that_ollama_truncates_below_the_target_is_unmeasured_not_a_short_row():
    class Truncating(_Server):
        def post(self, _path, payload):  # reads only half the window, whatever it is sent
            return {"prompt_eval_count": payload["options"]["num_ctx"] // 2, "prompt_eval_duration": 1,
                    "eval_count": 8, "eval_duration": 1}

        evaluated = None

    with pytest.raises(LinuxShapeFailure) as caught:
        measure_linux_ollama_generate(
            Truncating({}), lambda: (1000, 15360), model="m", prompt_tokens=2048, concurrency=1, repeats=1
        )
    assert caught.value.failure_class == "unmeasured"
    assert "could not size the prompt" in caught.value.reason


def test_a_reply_with_no_prompt_eval_count_is_unmeasured():
    class Silent(_Server):
        def post(self, _path, payload):
            return {"eval_count": 8}

    with pytest.raises(LinuxShapeFailure) as caught:
        measure_linux_ollama_generate(Silent({}), lambda: (1000, 15360), model="m", prompt_tokens=512, concurrency=1)
    assert caught.value.failure_class == "unmeasured"


def test_a_ladder_row_records_the_evaluated_length_and_how_it_was_counted():
    measured = {"ttft_ms": 10.0, "decode_tok_s": 30.0, "peak_mib": 6000, "size": 5, "size_vram": 5,
                "prompt_tokens_evaluated": 2051, "prompt_tokens_target": 2048}
    (row,) = run_linux_generate_ladder(
        spec_id="C2-GENERATE", spec_rel="experiments/c2-generate/spec.yaml", run_id="t4-r2", date="2026-10-26",
        remote_device=IDENT, model="phi3.5:3.8b", revision_or_digest="d" * 64, measure_shape=lambda _p, _c: measured,
        prompt_lengths=(2048,), concurrencies=(1,), quant="Q4_K_M", licence="mit",
    )
    assert row["settings"]["prompt_tokens"] == 2051
    assert row["settings"]["prompt_token_method"] == "ollama_prompt_eval_count"  # noqa: S105
    validate_row(row)


def test_a_failure_row_with_a_requested_length_says_it_did_not_verify_it():
    row = _row("timeout")
    assert row["settings"]["prompt_tokens"] == 2048
    assert row["settings"]["prompt_tokens_target"] == 2048
    assert row["settings"]["prompt_token_method"] == "requested_not_verified"  # noqa: S105
    validate_row(row)


def test_a_fill_card_failure_row_is_not_stamped_as_a_ladder_row():
    row = build_linux_failure_row(
        spec_id="C2-FILL-CARD", spec_rel="experiments/c2-fill-card/spec.yaml", run_id="r", date="2026-10-26",
        task="generate", model="phi3.5:3.8b", revision_or_digest="d" * 64, remote_device=IDENT,
        failure_class="oom", reason="peak over the abort line", quant="Q4_K_M", licence="mit",
    )
    assert "ladder_version" not in row["settings"]


def test_the_post_measurement_oom_row_records_the_evaluated_count_and_its_method():
    measured = {"ttft_ms": 10.0, "decode_tok_s": 30.0, "peak_mib": 15000, "size": 5, "size_vram": 5,
                "prompt_tokens_evaluated": 2051, "prompt_tokens_target": 2048}
    (row,) = run_linux_generate_ladder(
        spec_id="C2-GENERATE", spec_rel="experiments/c2-generate/spec.yaml", run_id="t4-r2", date="2026-10-26",
        remote_device=IDENT, model="phi3.5:3.8b", revision_or_digest="d" * 64, measure_shape=lambda _p, _c: measured,
        prompt_lengths=(2048,), concurrencies=(1,), quant="Q4_K_M", licence="mit",
    )
    assert row["outcome"] == "oom"
    assert row["settings"]["prompt_tokens"] == 2051
    assert row["settings"]["prompt_tokens_target"] == 2048
    assert row["settings"]["prompt_token_method"] == "ollama_prompt_eval_count"  # noqa: S105


def test_load_refused_needs_the_whole_ollama_message():
    for text, kind in (
        ("insufficient memory", "load_refused"),
        ("model is too large for system memory", "load_refused"),
        ("insufficient privileges to open the socket", "unmeasured"),
        ("the file is too large for the upload", "unmeasured"),
    ):
        assert classify_linux_error(RuntimeError(text)).failure_class == kind, text
