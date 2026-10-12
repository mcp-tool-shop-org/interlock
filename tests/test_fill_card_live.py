"""The fill-card live path, against fakes.

Nothing here opens a socket, runs nvidia-smi, or reads a GPU counter.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import threading
import time
from pathlib import Path

import pytest
import yaml

from switchyard.adapters.base import AdapterError
from switchyard.adapters.ollama import OllamaAdapter
from switchyard.adapters.ollama_cases import require_recorded_behaviour
from switchyard.harness.spec import SpecError, validate_spec
from switchyard.guards import GuardError, require_card_empty, require_grant, require_heartbeat
from switchyard.harness.fill import (
    ADAPTER_DEDICATED,
    ADAPTER_SHARED,
    DRIVER_ARGV,
    MEMORY_ARGV,
    NVIDIA_SMI_ARGV,
    CardSampler,
    FillSeams,
    adapter_usage,
    driver_cuda,
    facts_from_log,
    first_line,
    ggml_cuda_sha256,
    match_adapter_instance,
    parse_counter_text,
    parse_mib,
    quiet_fill_seams,
    real_fill_seams,
    run_fill_card,
    runner_pid,
    window_delta,
)
from switchyard.harness.runner import RunnerError, dry_run, run_spec
from switchyard.receipt import check_tree, load_jsonl

ROOT = Path(__file__).resolve().parents[1]
E5 = ROOT / "experiments" / "e5" / "spec.yaml"
SEED = "alpha beta "


def _model(tag="qwen3:8b"):
    return {
        "model": tag,
        "revision_or_digest": "abc123",
        "quant": "unrecorded",
        "licence": "unrecorded",
        "engine": "ollama",
        "engine_version": "0.35.1",
        "device": "5090",
    }


def _spec(models, *, max_ctx=8192):
    return {
        "id": "E5",
        "procedure": "fill-card",
        "question": "What context stays on the card?",
        "decision": "The confirmed window.",
        "abort_mib": 30500,
        "ceiling_mib": 31200,
        "grain": 2048,
        "confirm_prompt_fraction": 0.9,
        "confirm_num_predict": 1,
        "sample_ms": 200,
        "spill_shared_mib": 256,
        "spill_prompt_ratio": 0.5,
        "max_ctx": max_ctx,
        "shift": False,
        "truncate": False,
        "needs": {"grant": True, "npu_session": False, "cpu_window": False},
        "dataset": {"name": "e5-confirm-prompt", "seed": SEED},
        "metrics": [
            "max_resident_ctx",
            "peak_mib",
            "ttft_ms",
            "decode_tok_s",
            "prompt_tok_s",
            "shared_mib",
            "idle_baseline_mib",
        ],
        "stop_rules": ["Abort the moment card VRAM reaches 30500 MiB."],
        "models": models,
    }


def _commit(repo: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "dev"], cwd=repo, check=True)
    subprocess.run(["git", "config", "core.autocrlf", "false"], cwd=repo, check=True)
    subprocess.run(["git", "config", "core.longpaths", "true"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=repo, check=True, capture_output=True)


def _bind(repo: Path, spec: dict) -> Path:
    path = repo / "experiments" / "e5" / "spec.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8", newline="\n")
    _commit(repo)
    return path


def _fresh(tmp_path: Path) -> Path:
    path = tmp_path / "_watchdog_HEARTBEAT"
    path.write_text("pulse", encoding="utf-8")
    now = time.time()
    os.utime(path, (now, now))
    return path


class _Transport:
    def __init__(self, seams, *, show_error=None, explode=False, spill_shared=False, spill_prompt=False, missing_shared=False, missing_prompt=False, abort_at=None, stick=False, linear=None, cold_disagrees=False, omit_tags=False, tag_bytes=None, busy_mib=20000, hide_baseline=False, settle_high=0, self_sample=True, split_vram=False, hold_abort=False, slow_unload=0, hang_unload=False, trained=4096, unresident_at=None):
        self.seams = seams
        self.show_error = show_error
        self.explode = explode
        self.spill_shared = spill_shared
        self.spill_prompt = spill_prompt
        self.missing_shared = missing_shared
        self.missing_prompt = missing_prompt
        self.abort_at = abort_at
        self.stick = stick
        self.linear = linear
        self.cold_disagrees = cold_disagrees
        self.omit_tags = omit_tags
        self.tag_bytes = 16_000 * 1024 * 1024 if tag_bytes is None else tag_bytes
        self.busy_mib = busy_mib
        self.hide_baseline = hide_baseline
        self.settle_high = settle_high
        self.self_sample = self_sample
        self.split_vram = split_vram
        self.hold_abort = hold_abort
        self.slow_unload = slow_unload
        self.hang_unload = hang_unload
        self.trained = trained
        self.unresident_at = unresident_at
        self.block_count = False
        self.released = threading.Event()
        self._hang = threading.Event()
        self.settle_reads = 0
        self.events = []
        self._release = threading.Event()
        self.shown = None
        self.calls = []
        self.unloads = []
        self.abort_unloads = []
        self.order = []
        self.loaded = None
        self.trace = None
        self.state = {"ctx": None, "confirm": False, "busy": False}

    def cancel(self):
        self.order.append("cancel")
        self._release.set()

    def card_mib(self):
        if self.stick and self.seams is not None and self.seams.over:
            return 30500
        if not self.state["busy"]:
            if self.hide_baseline:
                return None
            if self.settle_high and self.unloads and self.settle_reads < self.settle_high:
                self.settle_reads += 1
                return 30500
            if self.unloads:
                self.events.append("settle")
            return 1500
        ctx = self.state["ctx"]
        if self.abort_at is not None and isinstance(ctx, int) and ctx >= self.abort_at:
            return 30500
        if self.linear is not None and isinstance(ctx, int):
            base, per_ctx = self.linear
            return base + per_ctx * ctx
        return self.busy_mib

    def shared_mib(self):
        if self.missing_shared and self.state["confirm"]:
            return None
        ctx = self.state["ctx"]
        if self.spill_shared and self.state["confirm"] and isinstance(ctx, int) and ctx >= 8192:
            return 1257
        return 1000

    def dedicated_mib(self):
        return 12

    def __call__(self, method, url, body):
        require_recorded_behaviour(method, url, body)
        self.calls.append((method, url, body))
        if url.endswith("/api/ps"):
            if self.loaded is None:
                return {"models": []}
            return {"models": [self.loaded]}
        if url.endswith("/api/tags"):
            if self.omit_tags:
                return {"models": []}
            name = self.shown or "qwen3:8b"
            return {"models": [{"name": name, "size": self.tag_bytes}]}
        if url.endswith("/api/show"):
            if isinstance(body, dict) and isinstance(body.get("model"), str):
                self.shown = body["model"]
            if self.trace is not None:
                self.trace.append("show")
            if self.show_error:
                return {"error": self.show_error}
            return {
                "model_info": {"qwen3.context_length": self.trained},
                "details": {"quantization_level": "Q4_K_M"},
                "parameters": "flash_attn true\nkv_cache_type q8_0\n",
            }
        if not isinstance(body, dict):
            return {}
        if body.get("keep_alive") == 0 and body.get("prompt") == "":
            if self.hang_unload and self.seams is not None and self.seams.over:
                self.order.append("unload-hang")
                self._hang.wait()
                return {}
            if self.slow_unload:
                time.sleep(self.slow_unload)
            if self.seams.over and self.state["busy"]:
                self.abort_unloads.append(self.state["ctx"])
            self.order.append("unload")
            self.unloads.append(body["model"])
            self.loaded = None
            self.state["busy"] = False
            return {}
        options = body.get("options") or {}
        self.state["ctx"] = options.get("num_ctx")
        self.state["confirm"] = (not body.get("raw")) and body.get("prompt") != SEED
        self.state["busy"] = True
        self.events.append(("load", self.state["ctx"]))
        if self.explode and not body.get("raw"):
            raise RuntimeError("generate exploded")
        holding = (
            self.hold_abort
            and not body.get("raw")
            and self.abort_at is not None
            and isinstance(self.state["ctx"], int)
            and self.state["ctx"] >= self.abort_at
        )
        if self.self_sample or not holding:
            self.seams.sample()
        if holding:
            self._release.wait(5)
        if body.get("raw") and self.block_count:
            self.released.wait(2)
        if body.get("raw"):
            count = 100000
            if self.cold_disagrees and self.loaded is None:
                count = 100001
            return {"prompt_eval_count": count}
        above = (
            self.unresident_at is not None
            and isinstance(self.state["ctx"], int)
            and self.state["ctx"] >= self.unresident_at
        )
        vram = 999 if self.split_vram or above else 1000
        self.loaded = {"name": body["model"], "size": 1000, "size_vram": vram}
        rate = self._rate()
        payload = {"eval_count": 1, "eval_duration": 1_000_000_000, "total_duration": 5_000_000}
        if rate is not None:
            payload["prompt_eval_count"] = 10
            payload["prompt_eval_duration"] = int(10 / rate * 1_000_000_000)
        return payload

    def _rate(self):
        if self.missing_prompt and self.state["confirm"]:
            return None
        ctx = self.state["ctx"]
        if self.spill_prompt and self.state["confirm"] and isinstance(ctx, int) and ctx >= 8192:
            return 40
        return 100


def _seams(transport: _Transport) -> FillSeams:
    seams = quiet_fill_seams()
    seams.card_mib = transport.card_mib
    seams.shared_mib = transport.shared_mib
    seams.dedicated_mib = transport.dedicated_mib
    seams.ollama_version = lambda: "0.35.1"
    seams.driver = lambda: "581.29"
    seams.cuda_backend = lambda: "13.0"
    transport.seams = seams
    slept = []
    seams.sleep = lambda seconds: slept.append(seconds)
    seams.slept = slept
    return seams


def _run(tmp_path, transport, *, models=None, max_ctx=8192, grant="g", seams=None, folder="repo", unload_timeout=None):
    repo = tmp_path / folder
    spec_path = _bind(repo, _spec(models or [_model()], max_ctx=max_ctx))
    adapter_kwargs = {}
    if unload_timeout is not None:
        adapter_kwargs["unload_timeout"] = unload_timeout
    adapter = OllamaAdapter(transport=transport, **adapter_kwargs)
    rows = run_spec(
        spec_path,
        root=repo,
        grant=grant,
        adapters={"ollama": adapter},
        heartbeat=_fresh(tmp_path),
        date="2026-10-11",
        run_id="fill-1",
        seams=seams,
    )
    return repo, rows


def _ctxs(transport: _Transport) -> list[int]:
    found = []
    for _method, _url, body in transport.calls:
        if isinstance(body, dict) and isinstance(body.get("options"), dict):
            ctx = body["options"].get("num_ctx")
            if isinstance(ctx, int):
                found.append(ctx)
    return found


def test_a_clean_search_records_the_confirmed_window(tmp_path: Path):
    transport = _Transport(None)
    seams = _seams(transport)
    repo, rows = _run(tmp_path, transport, seams=seams)
    row = rows[0]
    assert row["settings"]["max_resident_ctx"] == 4096
    assert row["settings"]["num_ctx"] == 4096
    assert row["settings"]["peak_mib"] == 20000
    assert row["settings"]["shared_mib"] == 1000
    assert row["settings"]["prompt_tok_s"] == 100
    assert row["settings"]["decode_tok_s"] == 1
    assert row["settings"]["ttft_ms"] == 100
    assert row["settings"]["idle_baseline_mib"] == 1500
    assert row["settings"]["ollama_version"] == "0.35.1"
    assert row["settings"]["cuda_backend"] == "13.0"
    assert row["settings"]["nvidia_driver"] == "581.29"
    assert row["settings"]["digest"] == "abc123"
    assert row["settings"]["quantization"] == "Q4_K_M"
    assert row["model"]["quant"] == "Q4_K_M"
    assert row["settings"]["flash_attention"] == "unrecorded"
    assert row["settings"]["kv_cache_type"] == "unrecorded"
    assert row["settings"]["default_context"] == "unrecorded"
    assert row["settings"]["backend_library"] == "unrecorded"
    assert row["settings"]["server_facts_reason"] == "server log is missing"
    assert row["settings"]["ggml_cuda_sha"] == "unrecorded"
    assert "ggml-cuda.dll" in row["settings"]["ggml_cuda_reason"]
    assert row["settings"]["search_calls"] == [
        ["load", 4096, 20000, 1000, 100, "resident"],
        ["confirm", 4096, 20000, 1000, 100, "confirmed"],
    ]
    assert row["settings"]["dedicated_delta_mib"] == 0
    assert row["settings"]["dedicated_mib"] == 12
    assert row["settings"]["prompt_token_method"].startswith("prompt_eval_count")
    assert row["settings"]["prompt_tokens"] == 100000
    assert row["settings"]["prompt_token_search"] == 100000
    assert row["settings"]["prompt_token_cold"] == 100000
    assert row["settings"]["weights_mib"] == 16000
    assert "one_window_bound" not in row["settings"]
    assert row["gpu_resident"] is True
    assert row["model"]["size"] == row["model"]["size_vram"] == 1000
    assert row["model_family"] == "qwen"
    assert row["provisional"] is True
    assert "aborted" not in row
    assert 31200 not in _ctxs(transport)
    assert all(ctx <= 4096 for ctx in _ctxs(transport))
    confirms = [
        body
        for _method, _url, body in transport.calls
        if isinstance(body, dict) and body.get("raw") is not True and body.get("prompt") not in (None, "", SEED)
    ]
    assert confirms and confirms[-1]["options"]["num_ctx"] == 4096
    assert confirms[-1]["shift"] is False and confirms[-1]["truncate"] is False
    assert confirms[-1]["options"]["num_predict"] == 1
    assert any(body.get("raw") is True and body["options"]["num_predict"] == 1 for _m, _u, body in transport.calls if isinstance(body, dict))
    assert seams.slept == []
    assert not (repo / "table").exists()
    assert check_tree(repo) == []
    assert transport.unloads


def test_the_sampler_unloads_when_the_window_reaches_abort_mib(tmp_path: Path):
    transport = _Transport(None, abort_at=8192, trained=8192)
    seams = _seams(transport)
    _repo, rows = _run(tmp_path, transport, seams=seams)
    row = rows[0]
    assert row["settings"]["max_resident_ctx"] == 6144
    assert 8192 in row["settings"]["aborted_windows"]
    assert 31200 not in _ctxs(transport)
    assert 31200 not in row["settings"]["aborted_windows"]
    assert transport.abort_unloads == [8192]
    assert transport.order[:2] == ["cancel", "unload"]
    assert row["settings"]["after_unload_mib"] == [{"ctx": 8192, "mib": 1500}]
    assert "after the unload" not in row["notes"]
    assert "predicted_over" not in row["settings"]
    assert transport.unloads


def test_a_window_whose_card_does_not_fall_after_unload_is_noted(tmp_path: Path):
    transport = _Transport(None, abort_at=8192, trained=8192, stick=True)
    seams = _seams(transport)
    _repo, rows = _run(tmp_path, transport, seams=seams)
    row = rows[0]
    assert transport.order[:2] == ["cancel", "unload"]
    assert {"ctx": 8192, "mib": 30500} in row["settings"]["after_unload_mib"]
    assert "8192 stayed at 30500 MiB after the unload" in row["notes"]
    assert row["settings"]["max_resident_ctx"] != 8192


def test_a_bisect_predicted_over_is_skipped(tmp_path: Path):
    transport = _Transport(None, linear=(10000, 2), trained=32768)
    seams = _seams(transport)
    _repo, rows = _run(tmp_path, transport, seams=seams, max_ctx=32768)
    row = rows[0]
    ctxs = _ctxs(transport)
    assert ctxs[0] == 32768
    assert 8192 in ctxs
    assert 12288 in row["settings"]["predicted_over"]
    assert 12288 not in ctxs
    assert 31200 not in ctxs
    assert row["settings"]["max_resident_ctx"] not in row["settings"]["predicted_over"]
    assert row["settings"]["max_resident_ctx"] == 10240


def test_one_measured_window_skips_a_bisect_with_less_than_one_cache_of_room(tmp_path: Path):
    """The resident window leaves less than one cache of room, so the bisect is predicted_over."""

    weights = 1000 * 1024 * 1024
    transport = _Transport(None, busy_mib=28000, tag_bytes=weights, trained=16384, abort_at=16384)
    seams = _seams(transport)
    _repo, rows = _run(tmp_path, transport, seams=seams, max_ctx=16384)
    row = rows[0]
    ctxs = _ctxs(transport)
    reading, baseline, ctx_1 = 28000, 1500, 8192
    share = reading - baseline - 1000

    def bound(ctx: int) -> int:
        extra = share * (ctx - ctx_1)
        return reading + (extra + ctx_1 - 1) // ctx_1

    assert row["settings"]["weights_mib"] == 1000
    assert row["settings"]["max_resident_ctx"] == 8192
    assert {"ctx": 12288, "mib": bound(12288)} in row["settings"]["one_window_bound"]
    assert bound(12288) >= 30500
    assert 12288 in row["settings"]["predicted_over"]
    assert 10240 in row["settings"]["predicted_over"]
    assert 12288 not in ctxs
    assert 10240 not in ctxs
    assert 31200 not in ctxs


def test_a_missing_weight_size_stops_before_the_larger_window(tmp_path: Path):
    transport = _Transport(None, omit_tags=True, trained=16384, abort_at=16384)
    seams = _seams(transport)
    with pytest.raises(RunnerError, match="the second window had no bound"):
        _run(tmp_path, transport, seams=seams, max_ctx=16384)
    ctxs = _ctxs(transport)
    assert 16384 in ctxs
    assert 8192 in ctxs
    assert 12288 not in ctxs


def test_a_missing_baseline_stops_before_the_larger_window(tmp_path: Path):
    transport = _Transport(None, hide_baseline=True, trained=16384, abort_at=16384)
    seams = _seams(transport)
    with pytest.raises(RunnerError, match="the second window had no bound"):
        _run(tmp_path, transport, seams=seams, max_ctx=16384)
    ctxs = _ctxs(transport)
    assert 16384 in ctxs
    assert 8192 in ctxs
    assert 12288 not in ctxs


def test_a_non_positive_share_stops_before_the_larger_window(tmp_path: Path):
    """The tag is larger than the card above the baseline, so the share is not a bound."""

    weights = 19000 * 1024 * 1024
    transport = _Transport(None, tag_bytes=weights, trained=16384, abort_at=16384)
    seams = _seams(transport)
    with pytest.raises(RunnerError, match="the second window had no bound"):
        _run(tmp_path, transport, seams=seams, max_ctx=16384)
    ctxs = _ctxs(transport)
    assert 16384 in ctxs
    assert 8192 in ctxs
    assert 12288 not in ctxs


def test_a_window_that_is_not_fully_resident_searches_down(tmp_path: Path):
    """A non-resident trained window is not a reason to load a larger one."""

    transport = _Transport(None, split_vram=True, trained=8192)
    seams = _seams(transport)
    _repo, rows = _run(tmp_path, transport, seams=seams, max_ctx=8192)
    ctxs = _ctxs(transport)
    assert ctxs[0] == 8192
    assert 4096 in ctxs
    assert all(ctx <= 8192 for ctx in ctxs)
    assert rows[0]["settings"]["max_resident_ctx"] is None
    assert rows[0]["notes"] == "no resident window"


def test_the_next_load_waits_until_the_abort_has_settled(tmp_path: Path):
    """A slow unload stays on the sampler thread. The next load starts after the settle read."""

    transport = _Transport(None, abort_at=8192, trained=8192, self_sample=False, hold_abort=True, slow_unload=0.4)
    seams = _seams(transport)
    seams.threaded = True
    _repo, rows = _run(tmp_path, transport, seams=seams)
    settle_at = transport.events.index("settle")
    next_load = transport.events.index(("load", 6144))
    assert transport.events.index(("load", 8192)) < settle_at < next_load
    assert rows[0]["settings"]["max_resident_ctx"] == 6144
    assert {"ctx": 8192, "mib": 1500} in rows[0]["settings"]["after_unload_mib"]


def test_a_hanging_unload_stops_before_the_next_load(tmp_path: Path):
    """The abort's unload never returns. The run stops, and the next size does not load."""

    transport = _Transport(None, abort_at=8192, trained=8192, self_sample=False, hold_abort=True, hang_unload=True)
    seams = _seams(transport)
    seams.threaded = True
    started = time.monotonic()
    with pytest.raises(RunnerError, match="the model may still be resident"):
        _run(tmp_path, transport, seams=seams, unload_timeout=0.3)
    assert time.monotonic() - started < 3
    ctxs = _ctxs(transport)
    assert 8192 in ctxs
    assert 6144 not in ctxs
    assert "unload-hang" in transport.order


def test_a_card_that_falls_while_the_unload_settles_is_not_noted(tmp_path: Path):
    transport = _Transport(None, abort_at=8192, trained=8192, settle_high=3)
    seams = _seams(transport)
    _repo, rows = _run(tmp_path, transport, seams=seams)
    row = rows[0]
    assert {"ctx": 8192, "mib": 1500} in row["settings"]["after_unload_mib"]
    assert "after the unload" not in row["notes"]
    assert transport.settle_reads == 3


def test_a_cold_recount_mismatch_refuses_the_confirmation(tmp_path: Path):
    transport = _Transport(None, cold_disagrees=True)
    seams = _seams(transport)
    with pytest.raises(RunnerError, match="the search counted 100000 and a cold load counted 100001"):
        _run(tmp_path, transport, seams=seams)
    raw = [
        index
        for index, (_method, _url, body) in enumerate(transport.calls)
        if isinstance(body, dict) and body.get("raw") is True
    ]
    unloads = [
        index
        for index, (_method, _url, body) in enumerate(transport.calls)
        if isinstance(body, dict) and body.get("prompt") == "" and body.get("keep_alive") == 0
    ]
    assert raw and unloads and unloads[0] < raw[-1]
    assert not any(
        isinstance(body, dict) and body.get("raw") is not True and body.get("prompt") not in (None, "", SEED)
        for _method, _url, body in transport.calls
    )


def test_shared_growth_steps_the_candidate_back(tmp_path: Path):
    transport = _Transport(None, spill_shared=True, trained=8192)
    seams = _seams(transport)
    _repo, rows = _run(tmp_path, transport, seams=seams)
    assert rows[0]["settings"]["max_resident_ctx"] == 6144
    assert rows[0]["settings"]["shared_mib"] == 1000
    calls = rows[0]["settings"]["search_calls"]
    spilled = [call for call in calls if call[0] == "confirm" and call[1] == 8192]
    assert spilled == [["confirm", 8192, 20000, 1257, 100, "spill"]]
    assert calls[-1] == ["confirm", 6144, 20000, 1000, 100, "confirmed"]


def test_a_fallen_prompt_rate_steps_the_candidate_back(tmp_path: Path):
    transport = _Transport(None, spill_prompt=True, trained=8192)
    seams = _seams(transport)
    _repo, rows = _run(tmp_path, transport, seams=seams)
    assert rows[0]["settings"]["max_resident_ctx"] == 6144
    assert rows[0]["settings"]["prompt_tok_s"] == 100


def test_a_missing_reading_is_a_spill(tmp_path: Path):
    transport = _Transport(None, missing_shared=True)
    seams = _seams(transport)
    _repo, rows = _run(tmp_path, transport, seams=seams)
    assert rows[0]["settings"]["max_resident_ctx"] is None
    assert rows[0]["notes"] == "no confirmed window"
    assert "aborted" not in rows[0]

    transport = _Transport(None, missing_prompt=True)
    seams = _seams(transport)
    _repo, rows = _run(tmp_path, transport, seams=seams, folder="repo-prompt")
    assert rows[0]["settings"]["max_resident_ctx"] is None


def test_a_failed_show_is_a_failed_step_and_unloads(tmp_path: Path):
    transport = _Transport(None, show_error="show failed")
    seams = _seams(transport)
    with pytest.raises(RunnerError, match="show failed"):
        _run(tmp_path, transport, seams=seams)
    assert transport.unloads == ["qwen3:8b"]
    assert not any(
        isinstance(body, dict) and body.get("options", {}).get("num_predict") == 1
        for _method, _url, body in transport.calls
    )
    repo = tmp_path / "repo"
    row = load_jsonl(repo / "receipts" / "fill-1.jsonl")[0]
    assert row["aborted"] is True
    assert "show failed" in row["notes"]
    assert not (repo / "table").exists()


def test_an_exception_still_unloads(tmp_path: Path):
    transport = _Transport(None, explode=True)
    seams = _seams(transport)
    with pytest.raises(RuntimeError, match="generate exploded"):
        _run(tmp_path, transport, seams=seams)
    assert transport.unloads == ["qwen3:8b"]


def test_guards_run_per_model_and_rest_is_not_taken(tmp_path: Path, monkeypatch):
    order = []
    real_grant = require_grant
    real_heartbeat = require_heartbeat
    real_empty = require_card_empty

    def grant(value, *, uses_5090):
        order.append("grant")
        return real_grant(value, uses_5090=uses_5090)

    def heartbeat(path, now, **kwargs):
        order.append("heartbeat")
        return real_heartbeat(path, now, **kwargs)

    def empty(models):
        order.append("ps")
        return real_empty(models)

    monkeypatch.setattr("switchyard.harness.fill.require_grant", grant)
    monkeypatch.setattr("switchyard.harness.fill.require_heartbeat", heartbeat)
    monkeypatch.setattr("switchyard.harness.fill.require_card_empty", empty)
    transport = _Transport(None)
    transport.trace = order
    seams = _seams(transport)
    _repo, rows = _run(
        tmp_path,
        transport,
        models=[_model("qwen3:8b"), _model("llama3.1:8b")],
        max_ctx=4096,
        seams=seams,
    )
    assert [row["model"]["repo_or_tag"] for row in rows] == ["qwen3:8b", "llama3.1:8b"]
    markers = [item for item in order if item in {"grant", "heartbeat", "ps", "show"}]
    assert markers == [
        "grant",
        "heartbeat",
        "ps",
        "show",
        "grant",
        "heartbeat",
        "ps",
        "show",
    ]
    assert seams.slept == []
    assert 900 not in seams.slept


def test_a_fake_adapter_without_seams_does_not_call_nvidia(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "switchyard.harness.fill._capture",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("a device command ran")),
    )
    transport = _Transport(quiet_fill_seams())
    _repo, rows = _run(tmp_path, transport, seams=None)
    assert rows[0]["settings"]["max_resident_ctx"] is None
    assert rows[0]["notes"] == "no resident window"


def test_a_missing_grant_never_calls_the_adapter(tmp_path: Path):
    transport = _Transport(quiet_fill_seams())
    with pytest.raises(GuardError, match="--grant"):
        _run(tmp_path, transport, grant=None)
    assert transport.calls == []


def test_dry_run_names_the_confirmation_count(monkeypatch):
    monkeypatch.setattr(
        "switchyard.guards.require_heartbeat",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("heartbeat was read")),
    )
    text = dry_run(E5, ROOT)
    assert "confirmation prefix counted at run time with prompt_eval_count" in text
    assert "one model at a time; unloaded between models; rest is not taken here" in text
    assert "the ceiling is never loaded" in text
    assert "prompt built at run time by prompt_eval_count" not in text
    assert "abort_mib: 30500" in text
    assert "ceiling_mib: 31200" in text
    assert "quant is read from /api/show at run time" in text
    assert "closing the request does not return the memory a load already allocated" in text
    assert "one measured window bounds the next context from that window's share" in text
    assert "a second window with no bound stops the row" in text
    assert "the next load waits until the abort has finished" in text
    assert "an unload that does not finish within 60 seconds stops the run; the model may still be resident" in text
    assert "heartbeat not read; adapters not called" in text
    assert "devices not started (--dry-run)" in text
    assert "rest_s" not in text
    source = (ROOT / "src" / "switchyard" / "harness" / "fill.py").read_text(encoding="utf-8")
    assert "31200" not in source


_MIB = 1024 * 1024
_GPU = "luid_5090_phys_0"
_IGPU = "luid_igpu_phys_0"


def _bytes(mib: int) -> int:
    return mib * _MIB


def test_the_search_does_not_load_above_the_trained_length(tmp_path: Path):
    transport = _Transport(None, trained=4096)
    seams = _seams(transport)
    _repo, rows = _run(tmp_path, transport, seams=seams, max_ctx=1_048_576)
    ctxs = _ctxs(transport)
    assert ctxs[0] == 4096
    assert all(ctx <= 4096 for ctx in ctxs)
    assert rows[0]["settings"]["max_resident_ctx"] == 4096
    assert rows[0]["settings"]["num_ctx"] == 4096


def test_a_trained_window_that_is_not_resident_searches_down_to_the_grain(tmp_path: Path):
    transport = _Transport(None, trained=16384, unresident_at=8192)
    seams = _seams(transport)
    _repo, rows = _run(tmp_path, transport, seams=seams, max_ctx=16384)
    ctxs = _ctxs(transport)
    row = rows[0]
    assert ctxs[0] == 16384
    assert all(ctx <= 16384 for ctx in ctxs)
    assert 8192 in ctxs
    assert row["settings"]["max_resident_ctx"] == 6144
    assert 6144 % 2048 == 0
    calls = row["settings"]["search_calls"]
    assert calls[0] == ["load", 16384, 20000, 1000, 100, "not resident"]
    assert ["load", 6144, 20000, 1000, 100, "resident"] in calls
    assert calls[-1] == ["confirm", 6144, 20000, 1000, 100, "confirmed"]


def test_shared_growth_uses_the_pre_window_baseline(tmp_path: Path):
    """A high absolute shared reading is not a spill when this window barely grew.

    The load's shared reading stays at 100. The confirmation sits near 2200.
    Gating on the load would call that a spill. The baseline taken just before
    the confirmation is 2000, so 200 MiB of growth is under the 256 MiB line.
    """

    transport = _Transport(None, trained=8192)
    seams = _seams(transport)
    script = [
        (_GPU, 100, 12),
        (_GPU, 100, 12),
        (_GPU, 2000, 12),
        (_GPU, 2200, 20),
    ]

    def adapter_reading():
        if not script:
            raise AssertionError("the window asked for another adapter reading")
        return script.pop(0)

    seams.adapter_reading = adapter_reading
    _repo, rows = _run(tmp_path, transport, seams=seams, max_ctx=8192)
    assert script == []
    assert rows[0]["settings"]["max_resident_ctx"] == 8192
    assert rows[0]["settings"]["shared_mib"] == 2200
    assert rows[0]["settings"]["dedicated_delta_mib"] == 8
    assert rows[0]["settings"]["dedicated_mib"] == 20
    assert rows[0]["settings"]["search_calls"][-1] == ["confirm", 8192, 20000, 2200, 100, "confirmed"]


def test_the_5090_is_the_dedicated_match_and_a_bad_match_is_a_spill(tmp_path: Path):
    mib = _MIB
    igpu = (_IGPU, _bytes(512))
    gpu = (_GPU, _bytes(20000))
    assert match_adapter_instance([igpu, gpu], 20000) == _GPU
    assert match_adapter_instance([gpu, igpu], 20000) == _GPU
    assert match_adapter_instance([igpu, (_GPU, _bytes(20000)), ("luid_other_phys_0", _bytes(20000))], 20000) is None
    assert match_adapter_instance([igpu], 20000) is None
    assert adapter_usage(
        [igpu, gpu],
        [(_IGPU, _bytes(10)), (_GPU, _bytes(30))],
        20000,
    ) == (_GPU, 30, 20000)
    assert window_delta((_GPU, 1000, 20), (_IGPU, 1001, 21)) == (None, None, None)
    assert window_delta((_GPU, 1000, 12), (_GPU, 1257, 20)) == (1257, 1000, 8)
    assert window_delta(None, (_GPU, 1, 1)) == (None, None, None)

    transport = _Transport(None, trained=8192)
    seams = _seams(transport)

    def matched():
        grown = transport.state["confirm"] and transport.state["ctx"] == 8192
        dedicated = [igpu, gpu]
        shared = [(_IGPU, 10 * mib), (_GPU, _bytes(1257 if grown else 1000))]
        return adapter_usage(dedicated, shared, 20000)

    seams.adapter_reading = matched
    _repo, rows = _run(tmp_path, transport, seams=seams, max_ctx=8192)
    calls = rows[0]["settings"]["search_calls"]
    assert rows[0]["settings"]["max_resident_ctx"] == 6144
    assert ["confirm", 8192, 20000, 1257, 100, "spill"] in calls
    assert calls[-1][3] == 1000

    transport = _Transport(None, trained=4096)
    seams = _seams(transport)

    def ambiguous():
        dedicated = [igpu, (_GPU, _bytes(20000)), ("luid_other_phys_0", _bytes(20000))]
        shared = [(_IGPU, _bytes(10)), (_GPU, _bytes(1000))]
        return adapter_usage(dedicated, shared, 20000)

    seams.adapter_reading = ambiguous
    _repo, rows = _run(tmp_path, transport, seams=seams, max_ctx=4096, folder="ambiguous")
    assert rows[0]["settings"]["max_resident_ctx"] is None
    assert rows[0]["notes"] == "no confirmed window"
    assert rows[0]["settings"]["search_calls"]
    assert all(call[3] is None for call in rows[0]["settings"]["search_calls"])
    assert any(call[0] == "confirm" and call[5] == "spill" for call in rows[0]["settings"]["search_calls"])


def test_the_server_log_fills_the_four_fields(tmp_path: Path):
    transport = _Transport(None, trained=4096)
    seams = _seams(transport)
    dll = tmp_path / "ggml-cuda.dll"
    dll.write_bytes(b"cuda-bytes")
    library = "C:/ollama/lib/ollama/cuda_v13/ggml-cuda.dll"
    log = (
        'msg="server config" env="map[OLLAMA_FLASH_ATTENTION:false OLLAMA_KV_CACHE_TYPE:f16 OLLAMA_CONTEXT_LENGTH:2048]"\n'
        'msg="server config" env="map[OLLAMA_FLASH_ATTENTION:true OLLAMA_KV_CACHE_TYPE:q8_0 OLLAMA_CONTEXT_LENGTH:8192]"\n'
        "load_backend: loaded CPU backend from C:/ollama/lib/ollama/cpu/ggml-cpu.dll\n"
        f"load_backend: loaded CUDA backend from {library}\n"
    )
    seams.server_log = lambda: log
    seams.ggml_dll = lambda: str(dll)
    seams.runner_pid = lambda: 4242
    _repo, rows = _run(tmp_path, transport, seams=seams)
    settings = rows[0]["settings"]
    assert settings["flash_attention"] == "true"
    assert settings["kv_cache_type"] == "q8_0"
    assert settings["default_context"] == "8192"
    assert settings["backend_library"] == library
    assert "server_facts_reason" not in settings
    assert settings["ggml_cuda_sha"] == hashlib.sha256(b"cuda-bytes").hexdigest()
    assert "ggml_cuda_reason" not in settings
    assert settings["runner_pid"] == 4242


def test_readers_parse_fake_output_and_do_not_guess(tmp_path: Path):
    assert parse_mib("20000\n") == 20000
    assert parse_mib("20000\n100\n") is None
    assert first_line("581.29\n") == "581.29"
    header = "NVIDIA-SMI 617.14    Driver Version: 617.14    CUDA Version: 13.2\n"
    assert driver_cuda(header) == "driver 13.2"
    assert driver_cuda("no cuda header\n") is None
    assert runner_pid(["42 7 ollama.exe", "9 7 ollama.exe"]) is None
    assert runner_pid(["42 7 OLLAMA.EXE"]) == 42
    assert runner_pid([]) is None
    missing = facts_from_log(None)
    assert missing["flash_attention"] == "unrecorded"
    assert missing["reason"] == "server log is missing"
    partial = facts_from_log('msg="server config" env="map[OLLAMA_FLASH_ATTENTION:true]"\n')
    assert partial["flash_attention"] == "true"
    assert partial["kv_cache_type"] == "unrecorded"
    assert "kv_cache_type" in partial["reason"]
    assert ggml_cuda_sha256(None) == ("unrecorded", "ggml-cuda.dll path is not configured")
    missing_dll = tmp_path / "absent.dll"
    assert ggml_cuda_sha256(str(missing_dll))[0] == "unrecorded"

    class Completed:
        def __init__(self, stdout):
            self.returncode = 0
            self.stdout = stdout

    counters = {"mode": "match"}

    def read_counter(counter):
        seen_counters.append(counter)
        if counters["mode"] == "ambiguous" and counter == ADAPTER_DEDICATED:
            return f"{_IGPU}={_bytes(20000)}\n{_GPU}={_bytes(20000)}\n"
        if counter == ADAPTER_DEDICATED:
            return f"{_IGPU}={_bytes(512)}\n{_GPU}={_bytes(20000)}\n"
        if counter == ADAPTER_SHARED:
            return f"{_IGPU}={_bytes(10)}\n{_GPU}={_bytes(30)}\n"
        raise AssertionError(counter)

    def run(argv, **kwargs):
        seen.append(tuple(argv))
        if tuple(argv) == MEMORY_ARGV:
            return Completed("20000\n")
        if tuple(argv) == DRIVER_ARGV:
            return Completed("581.29\n")
        if tuple(argv) == NVIDIA_SMI_ARGV:
            return Completed(header)
        script = argv[-1] if argv else ""
        if "cuda_version" in script or "cuda_version" in " ".join(argv):
            raise AssertionError(argv)
        if "11434" in script:
            return Completed("42 100 ollama.exe\n")
        raise AssertionError(argv)

    seen = []
    seen_counters = []

    class Adapter:
        def transport(self, method, url, body):
            assert method == "GET" and url.endswith("/api/version")
            return {"version": "0.35.1"}

    seams = real_fill_seams(Adapter(), run=run, read_counter=read_counter)
    assert seams.card_mib() == 20000
    assert seams.shared_mib() == 30
    assert seams.dedicated_mib() == 20000
    assert seams.driver() == "581.29"
    assert seams.cuda_backend() == "driver 13.2"
    assert seams.ollama_version() == "0.35.1"
    assert seams.runner_pid() == 42
    assert MEMORY_ARGV in seen
    assert NVIDIA_SMI_ARGV in seen
    assert ADAPTER_DEDICATED in seen_counters
    assert ADAPTER_SHARED in seen_counters
    assert not any("cuda_version" in " ".join(argv) for argv in seen)
    assert seams.threaded is True
    counters["mode"] = "ambiguous"
    assert seams.shared_mib() is None
    assert seams.dedicated_mib() is None


def test_the_sampler_loop_stops_at_abort_without_a_device():
    seams = quiet_fill_seams()
    seams.card_mib = lambda: 30500
    unloaded = []
    sampler = CardSampler(seams, 30500, 200, lambda: unloaded.append("unload"))
    sampler._loop()
    assert unloaded == ["unload"]
    assert sampler.over is True
    assert sampler.peak == 30500

    seams.card_mib = lambda: 10
    quiet = CardSampler(seams, 30500, 200, lambda: None)
    quiet._stop.set()
    quiet.sample()
    quiet._loop()
    assert quiet.peak == 10


@pytest.mark.parametrize("value", [0, None, True])
def test_a_confirmation_with_no_completion_limit_is_refused(tmp_path: Path, value):
    repo = tmp_path / "repo"
    spec = _spec([_model()])
    spec["confirm_num_predict"] = value
    spec_path = _bind(repo, spec)
    transport = _Transport(None)
    seams = _seams(transport)
    with pytest.raises((SpecError, ValueError, AdapterError, RunnerError)):
        run_spec(spec_path, root=repo, grant="g", adapters={"ollama": OllamaAdapter(transport=transport)},
                 heartbeat=_fresh(tmp_path), date="2026-10-11", run_id="fill-np", seams=seams)
    assert not any(isinstance(body, dict) and body.get("prompt") not in (None, "") for _m, _u, body in transport.calls)


def test_spec_validation_refuses_confirm_num_predict_0():
    spec = _spec([_model()])
    spec["confirm_num_predict"] = 0
    with pytest.raises(SpecError, match="confirm_num_predict"):
        validate_spec(spec)


def test_the_runtime_check_refuses_confirm_num_predict_before_the_card(tmp_path: Path):
    spec = _spec([_model()])
    spec["confirm_num_predict"] = 0

    class Boom:
        def loaded_models(self):
            raise AssertionError("device contact")

    class Marker:
        entered = False

        def close(self, _end):
            return None

    with pytest.raises(ValueError, match="confirm_num_predict"):
        run_fill_card(
            spec,
            adapters={"ollama": Boom()},
            seams=quiet_fill_seams(),
            grant="g",
            root=tmp_path,
            spec_rel="experiments/e5/spec.yaml",
            run_id="rt",
            date="2026-10-12",
            heartbeat=tmp_path / "beat",
            wall=lambda: 0.0,
            marker=Marker(),
        )
