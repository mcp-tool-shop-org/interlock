"""Ollama counts a dataset prefix with the served model's own tokenizer.

The transport is a fake. Nothing here opens a socket or starts a device.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest
import yaml

from switchyard.adapters.base import AdapterError, StepRequest
from switchyard.adapters.ollama import OllamaAdapter, generate_body
from switchyard.guards import GuardError
from switchyard.harness.e5 import SearchError, fit_prompt_words
from switchyard.harness.runner import RunnerError, dry_run, run_spec
from switchyard.harness.watch import WatchHooks

ROOT = Path(__file__).resolve().parents[1]
E4 = ROOT / "experiments" / "e4" / "spec.yaml"
E5 = ROOT / "experiments" / "e5" / "spec.yaml"


def _arm(**extra):
    arm = {
        "id": "5090-alone",
        "task": "generate",
        "device": "5090",
        "engine": "ollama",
        "model": "qwen3:8b",
        "revision_or_digest": "abc",
        "quant": "unrecorded",
        "licence": "unrecorded",
        "engine_version": "0.35.1",
        "thinking_level": "off",
        "num_predict": 8,
        "num_ctx": None,
        "batch": None,
        "static_shape": None,
        "precision": None,
    }
    arm.update(extra)
    return arm


def _spec(arm, *, seed="aa bb ", target=3):
    return {
        "id": "T",
        "procedure": "interleaved",
        "question": "Does the served model count the prefix?",
        "decision": "The receipt records the count.",
        "needs": {"grant": True, "npu_session": False, "cpu_window": False},
        "stop_rules": ["a failed step stops the sequence"],
        "dataset": {"name": "counted", "seed": seed, "target_tokens": target},
        "metrics": ["latency_ms"],
        "warmups": 0,
        "repeats": 1,
        "band": 0.1,
        "compile_cache": "reuse",
        "rest_s": 900,
        "arms": [arm],
    }


def _commit(repo: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "dev"], cwd=repo, check=True)
    subprocess.run(["git", "config", "core.autocrlf", "false"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=repo, check=True, capture_output=True)


def _bind(repo: Path, spec: dict) -> Path:
    path = repo / "experiments" / "t" / "spec.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8", newline="\n")
    _commit(repo)
    return path


def _fresh_heartbeat(tmp_path: Path) -> Path:
    path = tmp_path / "_watchdog_HEARTBEAT"
    path.write_text("pulse", encoding="utf-8")
    now = time.time()
    os.utime(path, (now, now))
    return path


def _watch() -> WatchHooks:
    """No watcher thread and no GPU sample. The card probe is the fake transport."""

    return WatchHooks(state_sampler=lambda _uses_5090: {}, start_thread=False)


class _Transport:
    """Ollama's three calls: show, an empty card, and generate."""

    def __init__(self, count_of, *, generate_error=None):
        self.count_of = count_of
        self.generate_error = generate_error
        self.calls: list[tuple] = []

    def __call__(self, method, url, body):
        self.calls.append((method, url, body))
        if url.endswith("/api/ps"):
            return {"models": []}
        if url.endswith("/api/show"):
            return {
                "model_info": {"qwen3.context_length": 4096},
                "details": {"quantization_level": "Q4_K_M"},
            }
        if not url.endswith("/api/generate"):
            raise AssertionError(url)
        if isinstance(body, dict) and body.get("options", {}).get("num_predict") == 0:
            raise AssertionError("num_predict 0 reached the server")
        if self.generate_error and isinstance(body, dict) and body.get("raw") is True:
            return {"error": self.generate_error}
        if isinstance(body, dict) and body.get("keep_alive") == 0 and body.get("prompt") == "":
            return {"done": True}
        prompt = body.get("prompt") if isinstance(body, dict) else ""
        return {
            "done": True,
            "response": "ok",
            "prompt_eval_count": self.count_of(prompt or ""),
            "prompt_eval_duration": 1_000_000,
            "eval_count": 4,
            "eval_duration": 1_000_000,
            "total_duration": 2_000_000,
        }


class _CachingTransport(_Transport):
    """A resident model reports one token fewer for a prompt it has already seen."""

    def __init__(self):
        self.seen: list[str] = []
        self.resident = False
        super().__init__(self._cached_count)

    def __call__(self, method, url, body):
        if isinstance(body, dict) and body.get("keep_alive") == 0 and body.get("prompt") == "":
            self.resident = False
            self.seen.clear()
        result = super().__call__(method, url, body)
        if isinstance(body, dict) and body.get("raw") is True:
            if body.get("keep_alive") == 0:
                self.resident = False
                self.seen.clear()
            else:
                self.seen.append(str(body.get("prompt") or ""))
                self.resident = True
        return result

    def _cached_count(self, prompt: str) -> int:
        true = len(prompt.split())
        if self.resident and any(prompt.startswith(earlier) for earlier in self.seen):
            return max(0, true - 1)
        return true


def _kinds(calls):
    kinds = []
    for _method, url, body in calls:
        if url.endswith("/api/ps"):
            kinds.append("ps")
        elif url.endswith("/api/show"):
            kinds.append("show")
        elif isinstance(body, dict) and body.get("raw") is True:
            kinds.append("count")
        elif isinstance(body, dict) and body.get("prompt") == "" and body.get("keep_alive") == 0:
            kinds.append("unload")
        else:
            kinds.append("measure")
    return kinds


def _run(tmp_path: Path, spec: dict, transport: _Transport, monkeypatch=None):
    repo = tmp_path / "repo"
    spec_path = _bind(repo, spec)
    events: list[str] = []
    adapter_transport = transport
    if monkeypatch is not None:
        from switchyard.harness import runner as runner_mod

        real_heartbeat = runner_mod.require_heartbeat

        def _heartbeat(path, now, max_age_s=60):
            events.append("heartbeat")
            return real_heartbeat(path, now, max_age_s=max_age_s)

        monkeypatch.setattr(runner_mod, "require_heartbeat", _heartbeat)

        class _Recording:
            """Special methods are looked up on the class, so this is a real wrapper."""

            def __call__(self, method, url, body):
                if str(url).endswith("/api/ps"):
                    events.append("ps")
                elif isinstance(body, dict) and body.get("raw") is True:
                    events.append("count")
                return transport(method, url, body)

        adapter_transport = _Recording()
    rows = run_spec(
        spec_path,
        root=repo,
        grant="grant-t",
        adapters={"ollama": OllamaAdapter(transport=adapter_transport, device="5090")},
        heartbeat=_fresh_heartbeat(tmp_path),
        date="2026-10-11",
        run_id="run-1",
        watch=_watch(),
    )
    return rows, events


def test_word_trim_hits_the_target_and_keeps_the_shortest_prefix_above_it():
    text, count, fit = fit_prompt_words("aa bb ", 3, lambda prompt: len(prompt.split()))
    assert (text, count, fit) == ("aa bb aa", 3, "exact")

    def two_per_word(prompt: str) -> int:
        return 2 * len(prompt.split())

    text, count, fit = fit_prompt_words("aa bb ", 5, two_per_word)
    assert (text, count, fit) == ("aa bb aa", 6, "shortest_at_or_above")
    with pytest.raises(SearchError, match="seed"):
        fit_prompt_words("   ", 1, len)
    with pytest.raises(SearchError, match="at least one"):
        fit_prompt_words("aa", 0, len)
    with pytest.raises(SearchError, match="non-negative"):
        fit_prompt_words("aa", 1, lambda _prompt: True)
    with pytest.raises(SearchError, match="never reaches"):
        fit_prompt_words("x" * 600_000, 1, lambda _prompt: 0)


def test_the_count_call_is_raw_and_asks_for_one_token_never_zero():
    # Ollama 0.35.1 reads num_predict 0 as no limit and generates until the context fills (E4, 2026-10-11).
    seen = []

    def transport(_method, url, body):
        seen.append((url, body))
        if url.endswith("/api/show"):
            return {"error": "missing"}
        return {"prompt_eval_count": 4}

    adapter = OllamaAdapter(transport=transport, device="5090")
    with pytest.raises(AdapterError, match="missing"):
        adapter.count_tokens("qwen3:8b", "hello", settings={"num_ctx": None, "thinking_level": "off"})
    assert adapter.count_tokens(
        "qwen3:8b",
        "hello",
        settings={"num_ctx": 2048, "thinking_level": "off", "keep_alive": "5m"},
    ) == 4
    count = seen[-1][1]
    assert count["raw"] is True
    assert count["truncate"] is False
    assert count["options"]["num_predict"] == 1
    assert count["options"]["num_ctx"] == 2048
    assert count["keep_alive"] == "5m"
    assert all(body["options"]["num_predict"] != 0 for url, body in seen if isinstance(body, dict) and "options" in body)
    with pytest.raises(AdapterError, match="invalid num_predict"):
        OllamaAdapter(transport=lambda *_args: {"error": "invalid num_predict"}, device="5090").count_tokens(
            "qwen3:8b", "hello", settings={"num_ctx": 32}
        )
    with pytest.raises(AdapterError, match="prompt_eval_count"):
        OllamaAdapter(transport=lambda *_args: {"done": True}, device="5090").count_tokens(
            "qwen3:8b", "hello", settings={"num_ctx": 32}
        )
    with pytest.raises(AdapterError, match="non-object"):
        OllamaAdapter(transport=lambda *_args: ["nope"], device="5090").count_tokens(
            "qwen3:8b", "hello", settings={"num_ctx": 32}
        )
    with pytest.raises(AdapterError, match="cloud"):
        adapter.count_tokens("qwen3:8b-cloud", "hello", settings={"num_ctx": 32})


def test_an_exact_ollama_count_lands_on_the_receipt_after_the_guards(tmp_path: Path, monkeypatch):
    transport = _Transport(lambda prompt: len(prompt.split()))
    rows, events = _run(tmp_path, _spec(_arm()), transport, monkeypatch)
    assert events.index("heartbeat") < events.index("ps") < events.index("count")
    kinds = _kinds(transport.calls)
    assert kinds.index("count") < kinds.index("unload") < kinds.index("measure")
    counts = [body for _method, url, body in transport.calls if isinstance(body, dict) and body.get("raw") is True]
    assert counts
    assert all(body["options"]["num_predict"] == 1 for body in counts)
    assert all(body["truncate"] is False for body in counts)
    assert all(body["options"]["num_ctx"] == 4096 for body in counts)
    assert all(body["think"] is False for body in counts)
    assert all(body["keep_alive"] == "5m" for body in counts)
    assert counts[-1]["prompt"] == "aa bb aa"
    measured = [
        body
        for _method, _url, body in transport.calls
        if isinstance(body, dict) and body.get("raw") is not True and body.get("prompt")
    ]
    assert measured
    assert all(body["options"]["num_predict"] == 8 for body in measured)
    assert all(body["options"]["num_ctx"] == 4096 for body in measured)
    assert all(body["keep_alive"] == "5m" for body in measured)
    assert all(body["prompt"] == "aa bb aa" for body in measured)
    assert "raw" not in measured[0]
    assert rows[0]["settings"]["prompt_tokens"] == 3
    assert rows[0]["settings"]["prompt_token_search"] == 3
    assert rows[0]["settings"]["prompt_token_cold"] == 3
    assert rows[0]["settings"]["prompt_token_method"] == "prompt_eval_count"


def test_the_shortest_prefix_at_or_above_is_recorded(tmp_path: Path):
    transport = _Transport(lambda prompt: 2 * len(prompt.split()))
    rows, _events = _run(tmp_path, _spec(_arm(), target=5), transport)
    assert rows[0]["settings"]["prompt_tokens"] == 6
    assert rows[0]["settings"]["prompt_token_method"] == "prompt_eval_count (shortest prefix at or above)"
    counts = [body for _method, _url, body in transport.calls if isinstance(body, dict) and body.get("raw") is True]
    assert counts[-1]["prompt"] == "aa bb aa"


def test_a_cached_count_lower_than_the_cold_load_refuses_the_arm(tmp_path: Path):
    """A resident prompt cache reports fewer tokens. The cold re-count disagrees."""

    transport = _CachingTransport()
    repo = tmp_path / "repo"
    with pytest.raises(RunnerError, match="search counted 3 and a cold load counted 4"):
        run_spec(
            _bind(repo, _spec(_arm())),
            root=repo,
            grant="grant-t",
            adapters={"ollama": OllamaAdapter(transport=transport, device="5090")},
            heartbeat=_fresh_heartbeat(tmp_path),
            date="2026-10-11",
            run_id="run-1",
            watch=_watch(),
        )
    raw_at = [
        index
        for index, (_method, _url, body) in enumerate(transport.calls)
        if isinstance(body, dict) and body.get("raw") is True
    ]
    unloads = [
        index
        for index, (_method, _url, body) in enumerate(transport.calls)
        if isinstance(body, dict) and body.get("prompt") == "" and body.get("keep_alive") == 0
    ]
    assert raw_at and unloads and unloads[0] < raw_at[-1]
    assert transport.calls[raw_at[-1]][2]["prompt"] == "aa bb aa bb"
    assert not any(
        isinstance(body, dict) and body.get("options", {}).get("num_predict") == 8
        for _method, _url, body in transport.calls
    )


def test_a_count_error_refuses_the_arm_before_a_measurement(tmp_path: Path):
    transport = _Transport(lambda _prompt: 1, generate_error="count failed")
    repo = tmp_path / "repo"
    with pytest.raises(RunnerError, match="token count failed: count failed"):
        run_spec(
            _bind(repo, _spec(_arm())),
            root=repo,
            grant="grant-t",
            adapters={"ollama": OllamaAdapter(transport=transport, device="5090")},
            heartbeat=_fresh_heartbeat(tmp_path),
            date="2026-10-11",
            run_id="run-1",
            watch=_watch(),
        )
    assert not any(
        isinstance(body, dict) and body.get("options", {}).get("num_predict") == 8
        for _method, _url, body in transport.calls
    )


def test_keep_alive_zero_is_not_unloaded_a_second_time(tmp_path: Path):
    transport = _Transport(lambda prompt: len(prompt.split()))
    _run(tmp_path, _spec(_arm(keep_alive=0)), transport)
    kinds = _kinds(transport.calls)
    assert "count" in kinds and "measure" in kinds
    assert kinds.index("measure") < kinds.index("unload")
    counts = [body for _method, _url, body in transport.calls if isinstance(body, dict) and body.get("raw") is True]
    assert counts and all(body["keep_alive"] == 0 for body in counts)


def test_a_missing_grant_and_a_stale_heartbeat_never_count(tmp_path: Path):
    transport = _Transport(lambda prompt: len(prompt.split()))
    repo = tmp_path / "repo"
    spec_path = _bind(repo, _spec(_arm()))
    adapter = OllamaAdapter(transport=transport, device="5090")
    with pytest.raises(GuardError, match="--grant"):
        run_spec(
            spec_path,
            root=repo,
            adapters={"ollama": adapter},
            heartbeat=_fresh_heartbeat(tmp_path),
            watch=_watch(),
        )
    assert transport.calls == []
    heartbeat = tmp_path / "stale"
    heartbeat.write_text("pulse", encoding="utf-8")
    stale = time.time() - 120
    os.utime(heartbeat, (stale, stale))
    with pytest.raises(GuardError, match="watchdog heartbeat"):
        run_spec(
            spec_path,
            root=repo,
            grant="grant-t",
            adapters={"ollama": adapter},
            heartbeat=heartbeat,
            run_id="stale",
            watch=_watch(),
        )
    assert transport.calls == []


def test_dry_run_names_the_count_and_does_not_call_it(tmp_path: Path, monkeypatch):
    called = []
    monkeypatch.setattr(
        OllamaAdapter,
        "count_tokens",
        lambda *_args, **_kwargs: called.append("count"),
    )
    monkeypatch.setattr(
        "switchyard.adapters.ollama.default_transport",
        lambda *_args, **_kwargs: called.append("transport"),
    )
    repo = tmp_path / "repo"
    for name, source in (("e4", E4), ("e5", E5)):
        target = repo / "experiments" / name / "spec.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    _commit(repo)
    e4_text = dry_run(repo / "experiments" / "e4" / "spec.yaml", repo)
    e5_text = dry_run(repo / "experiments" / "e5" / "spec.yaml", repo)
    assert "prompt built at run time by prompt_eval_count" in e4_text
    assert "the final prefix is counted again from a cold load" in e4_text
    assert "prompt built at run time by prompt_eval_count" not in e5_text
    assert "heartbeat not read; adapters not called" in e4_text
    assert "devices not started (--dry-run)" in e4_text
    assert called == []


@pytest.mark.parametrize("value", [0, None, True, -1, "1"])
def test_a_generate_with_no_completion_limit_is_refused_before_any_request(value):
    # Ollama reads num_predict 0, null or missing as no limit (E4, 2026-10-11).
    calls = []
    adapter = OllamaAdapter(transport=lambda *args: calls.append(args) or {}, device="5090")
    request = StepRequest(step_id="s", task="generate", model="qwen3:8b", device="5090", engine="ollama",
                          settings={"prompt": "hi", "num_predict": value})
    with pytest.raises(AdapterError, match="num_predict must be"):
        generate_body(request)
    missing = StepRequest(step_id="s", task="generate", model="qwen3:8b", device="5090", engine="ollama",
                          settings={"prompt": "hi"})
    with pytest.raises(AdapterError, match="num_predict must be"):
        generate_body(missing)
    assert calls == []


def test_a_long_target_takes_a_few_counts_not_one_per_repeat():
    # E5's first live run spent over 30 minutes adding one seed at a time toward 118k tokens (2026-10-12).
    seed = "The switchyard counts the same cars on every pass. "
    calls = []

    def count(prompt: str) -> int:
        calls.append(len(prompt))
        return len(prompt.split()) * 13 // 9  # about 13 tokens per 9 words, like the real seed

    text, number, fit = fit_prompt_words(seed, 118_000, count)
    assert number >= 118_000 and fit in ("exact", "shortest_at_or_above")
    assert count(text[: text.rstrip().rfind(" ")]) < 118_000 or number == 118_000
    assert len(calls) <= 25, len(calls)
