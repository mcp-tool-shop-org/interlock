"""The fill-card live path, against fakes.

Nothing here opens a socket, runs nvidia-smi, or reads a GPU counter.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import pytest
import yaml

from interlock.adapters.base import AdapterError
from interlock.adapters.ollama import OllamaAdapter
from interlock.adapters.ollama_cases import require_recorded_behaviour
from interlock.harness.spec import SpecError, validate_spec
from interlock.guards import GuardError, require_card_empty, require_grant, require_heartbeat
from interlock.harness.fill import (
    install_relative,
    ADAPTER_DEDICATED,
    ADAPTER_SHARED,
    DRIVER_ARGV,
    MEMORY_ARGV,
    NVIDIA_SMI_ARGV,
    CardSampler,
    FillPreflightError,
    FillSeams,
    adapter_usage,
    driver_cuda,
    facts_from_log,
    first_line,
    ggml_cuda_sha256,
    match_adapter_instance,
    parse_counter_text,
    parse_mib,
    UNSETTLED,
    _attach_pre_calls,
    quiet_fill_seams,
    real_fill_seams,
    run_fill_card,
    wait_until_stable,
    runner_pid,
    window_delta,
)
from interlock.harness.e5 import SHARED_GATE_FROM_MIB
from interlock.harness.runner import RunnerError, dry_run, run_spec
from interlock.receipt import check_tree, load_jsonl

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


def _spec(models, *, max_ctx=8192, confirm_num_predict=1):
    return {
        "id": "E5",
        "procedure": "fill-card",
        "question": "What context stays on the card?",
        "decision": "The confirmed window.",
        "abort_mib": 30500,
        "ceiling_mib": 31200,
        "grain": 2048,
        "confirm_prompt_fraction": 0.9,
        "confirm_num_predict": confirm_num_predict,
        "sample_ms": 200,
        "spill_shared_mib": 256,
        "adapter_match_tolerance_mib": 512,
        "spill_prompt_ratio": 0.5,
        "shared_gate_from_mib": 28452,
        "settle_tolerance_mib": 512,
        "settle_timeout_s": 30,
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
    def __init__(self, seams, *, show_error=None, explode=False, spill_shared=False, spill_prompt=False, missing_shared=False, missing_prompt=False, abort_at=None, stick=False, linear=None, cold_disagrees=False, omit_tags=False, tag_bytes=None, busy_mib=20000, hide_baseline=False, settle_high=0, self_sample=True, split_vram=False, hold_abort=False, slow_unload=0, hang_unload=False, trained=4096, unresident_at=None, idle_mib=1500, peaks=None, fallback_peak=None, gate_peak_at=None, dedicated_script=None, dedicated_tracks_resident=False, resident_level=28000, idle_level=100, stuck_after_unload=False):
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
        self.idle_mib = idle_mib
        self.peaks = peaks
        self.fallback_peak = fallback_peak
        self.gate_peak_at = gate_peak_at
        self.dedicated_script = dedicated_script
        self.dedicated_tracks_resident = dedicated_tracks_resident
        self.resident_level = resident_level
        self.idle_level = idle_level
        self.stuck_after_unload = stuck_after_unload
        self._stuck = False
        self.timeline = []
        self._dedicated_index = 0
        self.block_count = False
        self.released = threading.Event()
        self._hang = threading.Event()
        self.settle_reads = 0
        self.events = []
        self._release = threading.Event()
        self.shown = None
        self.calls = []
        self.before_generate = None
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
            return self.idle_mib
        ctx = self.state["ctx"]
        if self.abort_at is not None and isinstance(ctx, int) and ctx >= self.abort_at:
            return 30500
        if self.linear is not None and isinstance(ctx, int):
            base, per_ctx = self.linear
            return base + per_ctx * ctx
        pressured = self.spill_shared or (
            self.gate_peak_at is not None and isinstance(ctx, int) and ctx >= self.gate_peak_at
        )
        if pressured and self.state["confirm"] and isinstance(ctx, int) and ctx >= 8192:
            return SHARED_GATE_FROM_MIB
        if self.peaks is not None and isinstance(ctx, int):
            if ctx in self.peaks:
                return self.peaks[ctx]
            if self.fallback_peak is not None:
                return self.fallback_peak
        return self.busy_mib

    def shared_mib(self):
        if self.missing_shared and self.state["confirm"]:
            return None
        ctx = self.state["ctx"]
        if self.spill_shared and self.state["confirm"] and isinstance(ctx, int) and ctx >= 8192:
            return 1257
        return 1000

    def dedicated_mib(self):
        if self.dedicated_tracks_resident:
            # High while a model is resident. Idle only after unload clears it,
            # unless the card is stuck and the unload does not free the reading.
            if self._stuck or self.loaded is not None:
                value = self.resident_level
            else:
                value = self.idle_level
            self.timeline.append(("dedicated", value))
            return value
        script = self.dedicated_script
        if script is None:
            return 12
        index = self._dedicated_index
        self._dedicated_index += 1
        if index < len(script):
            return script[index]
        return script[-1]

    def __call__(self, method, url, body):
        require_recorded_behaviour(method, url, body)
        self.calls.append((method, url, body))
        if url.endswith("/api/ps"):
            if self.loaded is None:
                return {"models": []}
            return {"models": [self.loaded]}
        if url.endswith("/api/version"):
            return {"version": "0.35.1"}
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
            if self.stuck_after_unload:
                self._stuck = True
            if self.dedicated_tracks_resident:
                self.timeline.append(("unload",))
            return {}
        options = body.get("options") or {}
        self.state["ctx"] = options.get("num_ctx")
        self.state["confirm"] = (not body.get("raw")) and body.get("prompt") != SEED
        self.state["busy"] = True
        if self.dedicated_tracks_resident:
            self.timeline.append(("request", self.state["ctx"], body.get("raw") is True))
        self.events.append(("load", self.state["ctx"]))
        if callable(self.before_generate) and not body.get("raw") and body.get("prompt") != "":
            self.before_generate()
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


def _run(tmp_path, transport, *, models=None, max_ctx=8192, grant="g", seams=None, folder="repo", unload_timeout=None, confirm_num_predict=1):
    repo = tmp_path / folder
    spec_path = _bind(repo, _spec(models or [_model()], max_ctx=max_ctx, confirm_num_predict=confirm_num_predict))
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
    assert row["settings"]["decode_tok_s"] is None
    assert row["settings"]["decode_reason"] == "decode not measured: num_predict 1"
    assert row["latency"]["throughput"] is None
    assert row["latency"]["throughput_unit"] is None
    assert row["latency"]["throughput_reason"] == "decode not measured: num_predict 1"
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
        ["load", 2048, 20000, 1000, 100, "resident"],
        ["confirm", 2048, 20000, 1000, 100, "confirmed"],
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
    assert {"ctx": 4096, "mib": 22500} in row["settings"]["one_window_bound"]
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


def _advance(seams):
    """A clock that moves only when the harness sleeps, so a timeout cannot spin."""

    clock = {"t": 0.0}
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        clock["t"] += float(seconds)

    seams.sleep = sleep
    seams.slept = slept
    seams.clock = lambda: clock["t"]
    return slept


def test_a_card_still_freeing_memory_waits_then_measures(tmp_path: Path):
    # Idle is captured first. The next readings are still well above it, then the card returns.
    transport = _Transport(None, dedicated_script=[100, 8000, 4000, 1000, 100])
    seams = _seams(transport)
    _repo, rows = _run(tmp_path, transport, seams=seams)
    row = rows[0]
    assert seams.slept
    assert all(pause == 0.2 for pause in seams.slept)
    assert _ctxs(transport)
    assert row["settings"]["max_resident_ctx"] == 4096
    pre = row["settings"]["pre_call_dedicated_mib"]
    assert pre[0] == {
        "kind": "load",
        "ctx": 2048,
        "mib": 100,
        "resident_ctx": None,
        "unloaded": False,
    }
    assert all(item["mib"] == 100 for item in pre)
    assert all("resident_ctx" in item and "unloaded" in item for item in pre)
    assert 8000 not in [item["mib"] for item in pre]


def test_a_card_that_never_settles_refuses_the_call(tmp_path: Path):
    transport = _Transport(None, dedicated_script=[100, 9000])
    seams = _seams(transport)
    _advance(seams)
    _repo, rows = _run(tmp_path, transport, seams=seams)
    row = rows[0]
    assert row["settings"]["max_resident_ctx"] is None
    assert row["settings"]["peak_mib"] is None
    assert row["settings"]["search_calls"] == [
        ["load", 2048, None, None, None, UNSETTLED],
    ]
    assert row["settings"]["pre_call_dedicated_mib"] == [
        {"kind": "load", "ctx": 2048, "mib": 9000, "resident_ctx": None, "unloaded": False},
    ]
    assert _ctxs(transport) == []
    assert seams.slept
    # Resident floor: a card that keeps moving is the same refusal, and the reading is not kept.
    jump = {"n": 0}

    def read():
        jump["n"] += 1000
        return jump["n"]

    clock = {"t": 0.0}
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        clock["t"] += float(seconds)

    found, reason = wait_until_stable(
        read,
        sleep,
        lambda: clock["t"],
        tolerance_mib=512,
        timeout_s=30,
        first=27329,
    )
    assert reason == UNSETTLED
    assert slept
    assert found != 27329


def test_the_pre_call_reading_is_on_the_receipt(tmp_path: Path):
    transport = _Transport(None, dedicated_script=[250])
    seams = _seams(transport)
    repo, rows = _run(tmp_path, transport, seams=seams)
    row = rows[0]
    pre = row["settings"]["pre_call_dedicated_mib"]
    measured = [call for call in row["settings"]["search_calls"] if call[5] != "predicted_over"]
    assert [(item["kind"], item["ctx"], item["mib"]) for item in pre] == [
        (call[0], call[1], 250) for call in measured
    ]
    assert [(item["resident_ctx"], item["unloaded"]) for item in pre] == [
        (None, False),
        (2048, False),
        (2048, True),
        (4096, False),
    ]
    assert len(pre) == len(measured)
    stored = load_jsonl(repo / "receipts" / "fill-1.jsonl")[0]
    assert stored["settings"]["pre_call_dedicated_mib"] == pre
    # A predicted_over of the same context is not a measured call, so it does not take the next reading.
    paired = {
        "settings": {
            "search_calls": [
                ["load", 8192, 100, None, None, "predicted_over"],
                ["load", 8192, 20000, 1000, 100, "resident"],
                ["confirm", 8192, 20000, 1000, 100, "confirmed"],
            ]
        }
    }
    _attach_pre_calls(
        paired,
        [
            {"kind": "load", "ctx": 8192, "mib": 400, "resident_ctx": None, "unloaded": False},
            {"kind": "confirm", "ctx": 8192, "mib": 410, "resident_ctx": 8192, "unloaded": False},
        ],
    )
    assert paired["settings"]["pre_call_dedicated_mib"] == [
        {"kind": "load", "ctx": 8192, "mib": 400, "resident_ctx": None, "unloaded": False},
        {"kind": "confirm", "ctx": 8192, "mib": 410, "resident_ctx": 8192, "unloaded": False},
    ]


def _at(timeline, event, start=0):
    for index in range(start, len(timeline)):
        if timeline[index] == event:
            return index
    raise AssertionError(event)


def test_a_context_change_unloads_and_settles_to_idle_before_the_call(tmp_path: Path):
    """Resident at 43008, then the confirm walk starts at 4096.

    Dedicated stays at the resident level until unload clears the loaded model.
    No falling script is handed in. A share of 1 MiB keeps the one-window
    prediction of 43008 under abort_mib, so that load stays resident.
    """

    weights = 18499 * 1024 * 1024
    transport = _Transport(
        None,
        trained=43008,
        tag_bytes=weights,
        dedicated_tracks_resident=True,
        resident_level=28000,
        idle_level=100,
    )
    assert transport.dedicated_script is None
    seams = _seams(transport)
    _repo, rows = _run(tmp_path, transport, seams=seams, max_ctx=43008)
    row = rows[0]
    assert ["load", 43008, 20000, 1000, 100, "resident"] in row["settings"]["search_calls"]
    pre = row["settings"]["pre_call_dedicated_mib"]
    assert all("resident_ctx" in item and "unloaded" in item for item in pre)
    confirm = next(item for item in pre if item["kind"] == "confirm" and item["ctx"] == 4096)
    assert confirm == {
        "kind": "confirm",
        "ctx": 4096,
        "mib": 100,
        "resident_ctx": 43008,
        "unloaded": True,
    }
    load_at = _at(transport.timeline, ("request", 43008, False))
    confirm_at = _at(transport.timeline, ("request", 4096, True), load_at + 1)
    gap = transport.timeline[load_at + 1:confirm_at]
    unload_at = gap.index(("unload",))
    assert ("dedicated", 28000) in gap[:unload_at]
    assert ("dedicated", 100) not in gap[:unload_at]
    assert ("dedicated", 100) in gap[unload_at + 1:]
    assert transport.timeline[confirm_at] == ("request", 4096, True)


def test_the_same_context_keeps_the_resident_floor_wait(tmp_path: Path):
    transport = _Transport(
        None,
        trained=4096,
        dedicated_tracks_resident=True,
        resident_level=28000,
        idle_level=100,
    )
    assert transport.dedicated_script is None
    seams = _seams(transport)
    _repo, rows = _run(tmp_path, transport, seams=seams, max_ctx=4096)
    pre = rows[0]["settings"]["pre_call_dedicated_mib"]
    assert pre[1] == {
        "kind": "confirm",
        "ctx": 2048,
        "mib": 28000,
        "resident_ctx": 2048,
        "unloaded": False,
    }
    assert pre[3] == {
        "kind": "confirm",
        "ctx": 4096,
        "mib": 28000,
        "resident_ctx": 4096,
        "unloaded": False,
    }
    load_at = _at(transport.timeline, ("request", 2048, False))
    confirm_at = _at(transport.timeline, ("request", 2048, True), load_at + 1)
    gap = transport.timeline[load_at + 1:confirm_at]
    assert ("unload",) not in gap
    assert ("dedicated", 28000) in gap
    later = _at(transport.timeline, ("request", 4096, False), confirm_at + 1)
    again = _at(transport.timeline, ("request", 4096, True), later + 1)
    assert ("unload",) not in transport.timeline[later + 1:again]


def test_a_card_that_never_returns_to_idle_after_unload_refuses_the_call(tmp_path: Path):
    transport = _Transport(
        None,
        trained=4096,
        dedicated_tracks_resident=True,
        resident_level=28000,
        idle_level=100,
        stuck_after_unload=True,
    )
    assert transport.dedicated_script is None
    seams = _seams(transport)
    _advance(seams)
    _repo, rows = _run(tmp_path, transport, seams=seams, max_ctx=4096)
    row = rows[0]
    assert ["load", 4096, None, None, None, UNSETTLED] in row["settings"]["search_calls"]
    refused = next(
        item
        for item in row["settings"]["pre_call_dedicated_mib"]
        if item["kind"] == "load" and item["ctx"] == 4096
    )
    assert refused == {
        "kind": "load",
        "ctx": 4096,
        "mib": 28000,
        "resident_ctx": 2048,
        "unloaded": True,
    }
    assert 4096 not in _ctxs(transport)
    assert seams.slept
    assert row["settings"]["max_resident_ctx"] == 2048


def test_num_predict_1_does_not_publish_a_decode_rate(tmp_path: Path):
    transport = _Transport(None)
    seams = _seams(transport)
    _repo, rows = _run(tmp_path, transport, seams=seams)
    row = rows[0]
    assert row["settings"]["num_predict"] == 1
    assert row["settings"]["decode_tok_s"] is None
    assert row["latency"]["throughput"] is None
    assert row["settings"]["decode_reason"] == "decode not measured: num_predict 1"
    assert row["latency"]["throughput_reason"] == "decode not measured: num_predict 1"
    confirms = [
        body
        for _method, _url, body in transport.calls
        if isinstance(body, dict) and body.get("raw") is not True and body.get("prompt") not in (None, "", SEED)
    ]
    assert confirms[-1]["options"]["num_predict"] == 1


def test_num_predict_above_1_keeps_the_measured_rate(tmp_path: Path):
    transport = _Transport(None)
    seams = _seams(transport)
    _repo, rows = _run(tmp_path, transport, seams=seams, confirm_num_predict=8)
    row = rows[0]
    assert row["settings"]["num_predict"] == 8
    assert row["settings"]["decode_tok_s"] == 1.0
    assert row["latency"]["throughput"] == 1.0
    assert row["latency"]["throughput_unit"] == "tok/s"
    assert "decode_reason" not in row["settings"]
    assert "throughput_reason" not in row["latency"]
    confirms = [
        body
        for _method, _url, body in transport.calls
        if isinstance(body, dict) and body.get("raw") is not True and body.get("prompt") not in (None, "", SEED)
    ]
    assert confirms[-1]["options"]["num_predict"] == 8


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
    cancel_at = transport.order.index("cancel")
    assert transport.order[cancel_at:cancel_at + 2] == ["cancel", "unload"]
    assert row["settings"]["after_unload_mib"] == [{"ctx": 8192, "mib": 1500}]
    assert "after the unload" not in row["notes"]
    assert "predicted_over" not in row["settings"]
    assert transport.unloads


def test_a_window_whose_card_does_not_fall_after_unload_is_noted(tmp_path: Path):
    transport = _Transport(None, abort_at=8192, trained=8192, stick=True)
    seams = _seams(transport)
    _repo, rows = _run(tmp_path, transport, seams=seams)
    row = rows[0]
    cancel_at = transport.order.index("cancel")
    assert transport.order[cancel_at:cancel_at + 2] == ["cancel", "unload"]
    assert {"ctx": 8192, "mib": 30500} in row["settings"]["after_unload_mib"]
    assert "8192 stayed at 30500 MiB after the unload" in row["notes"]
    assert row["settings"]["max_resident_ctx"] != 8192


def test_a_bisect_predicted_over_is_skipped(tmp_path: Path):
    # The floor is measured first. This weight keeps that floor's share positive
    # and lets 8192 load, while the slope from the floor still predicts 12288 over.
    weights = 8596 * 1024 * 1024
    transport = _Transport(None, linear=(10000, 2), trained=32768, tag_bytes=weights)
    seams = _seams(transport)
    _repo, rows = _run(tmp_path, transport, seams=seams, max_ctx=32768)
    row = rows[0]
    ctxs = _ctxs(transport)
    assert ctxs[0] == 2048
    assert 8192 in ctxs
    assert 12288 in row["settings"]["predicted_over"]
    assert 12288 not in ctxs
    assert 31200 not in ctxs
    assert row["settings"]["weights_mib"] == 8596
    assert row["settings"]["max_resident_ctx"] not in row["settings"]["predicted_over"]
    assert row["settings"]["max_resident_ctx"] == 10240


def test_one_measured_window_skips_a_bisect_with_less_than_one_cache_of_room(tmp_path: Path):
    """The floor is the one measured window. Its share leaves no room for the next grain."""

    weights = 1000 * 1024 * 1024
    transport = _Transport(None, busy_mib=28000, tag_bytes=weights, trained=16384, abort_at=16384)
    seams = _seams(transport)
    _repo, rows = _run(tmp_path, transport, seams=seams, max_ctx=16384)
    row = rows[0]
    ctxs = _ctxs(transport)
    reading, baseline, ctx_1 = 28000, 1500, 2048
    share = reading - baseline - 1000

    def bound(ctx: int) -> int:
        extra = share * (ctx - ctx_1)
        return reading + (extra + ctx_1 - 1) // ctx_1

    assert row["settings"]["weights_mib"] == 1000
    assert row["settings"]["max_resident_ctx"] == 2048
    assert {"ctx": 4096, "mib": bound(4096)} in row["settings"]["one_window_bound"]
    assert bound(4096) >= 30500
    assert 4096 in row["settings"]["predicted_over"]
    assert 8192 in row["settings"]["predicted_over"]
    assert 16384 in row["settings"]["predicted_over"]
    assert 4096 not in ctxs
    assert 8192 not in ctxs
    assert 16384 not in ctxs
    assert 31200 not in ctxs


def test_a_missing_weight_size_stops_before_the_larger_window(tmp_path: Path):
    transport = _Transport(None, omit_tags=True, trained=16384, abort_at=16384)
    seams = _seams(transport)
    with pytest.raises(RunnerError, match="the second window had no bound"):
        _run(tmp_path, transport, seams=seams, max_ctx=16384)
    ctxs = _ctxs(transport)
    assert 2048 in ctxs
    assert 16384 not in ctxs
    assert 8192 not in ctxs


def test_a_missing_baseline_stops_before_the_larger_window(tmp_path: Path):
    transport = _Transport(None, hide_baseline=True, trained=16384, abort_at=16384)
    seams = _seams(transport)
    with pytest.raises(RunnerError, match="the second window had no bound"):
        _run(tmp_path, transport, seams=seams, max_ctx=16384)
    ctxs = _ctxs(transport)
    assert 2048 in ctxs
    assert 16384 not in ctxs
    assert 8192 not in ctxs


def test_a_non_positive_share_stops_before_the_larger_window(tmp_path: Path):
    """The tag is larger than the card above the baseline, so the share is not a bound."""

    weights = 19000 * 1024 * 1024
    transport = _Transport(None, tag_bytes=weights, trained=16384, abort_at=16384)
    seams = _seams(transport)
    with pytest.raises(RunnerError, match="the second window had no bound"):
        _run(tmp_path, transport, seams=seams, max_ctx=16384)
    ctxs = _ctxs(transport)
    assert 2048 in ctxs
    assert 16384 not in ctxs
    assert 8192 not in ctxs


def test_a_window_that_is_not_fully_resident_searches_down(tmp_path: Path):
    """A floor that is not resident is not a reason to load the cap."""

    transport = _Transport(None, split_vram=True, trained=8192)
    seams = _seams(transport)
    _repo, rows = _run(tmp_path, transport, seams=seams, max_ctx=8192)
    ctxs = _ctxs(transport)
    assert ctxs[0] == 2048
    assert 8192 not in ctxs
    assert 4096 not in ctxs
    assert all(ctx <= 2048 for ctx in ctxs)
    assert rows[0]["settings"]["max_resident_ctx"] is None
    assert rows[0]["notes"] == "no resident window"


def test_the_next_load_waits_until_the_abort_has_settled(tmp_path: Path):
    """A slow unload stays on the sampler thread. The next load starts after the settle read."""

    transport = _Transport(None, abort_at=8192, trained=8192, self_sample=False, hold_abort=True, slow_unload=0.4)
    seams = _seams(transport)
    seams.threaded = True
    _repo, rows = _run(tmp_path, transport, seams=seams)
    abort_load = transport.events.index(("load", 8192))
    next_load = next(
        index for index, event in enumerate(transport.events) if event == ("load", 6144) and index > abort_load
    )
    assert any(event == "settle" for event in transport.events[abort_load + 1:next_load])
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
    assert ["confirm", 6144, 20000, 1000, 100, "confirmed"] in calls
    assert calls[-1] == ["confirm", 8192, SHARED_GATE_FROM_MIB, 1257, 100, "spill"]


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

    monkeypatch.setattr("interlock.harness.fill.require_grant", grant)
    monkeypatch.setattr("interlock.harness.fill.require_heartbeat", heartbeat)
    monkeypatch.setattr("interlock.harness.fill.require_card_empty", empty)
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
        "interlock.harness.fill._capture",
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
        "interlock.guards.require_heartbeat",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("heartbeat was read")),
    )
    text = dry_run(E5, ROOT)
    assert "confirmation prefix counted at run time with prompt_eval_count" in text
    assert "one model at a time; unloaded between models; rest is not taken here" in text
    assert "the ceiling is never loaded" in text
    assert "prompt built at run time by prompt_eval_count" not in text
    assert "abort_mib: 30500" in text
    assert "ceiling_mib: 31200" in text
    assert "adapter_match_tolerance_mib: 512" in text
    assert "the tolerance is a matching allowance and is not the spill threshold" in text
    assert "pinned once at run start" in text
    assert "zero or several instances within tolerance refuses before any load" in text
    assert "later shared and dedicated reads use the pinned instance" in text
    assert "the reference is the floor window, one grain, loaded and confirmed first for every model" in text
    assert "the lowest resident window is superseded" in text
    assert "that floor confirm is the reference for the shared gate and the prompt-rate gate" in text
    assert "a window spills when its shared reading exceeds that floor confirm by more than spill_shared_mib" in text
    assert "a missing shared reading during a window is still a spill" in text
    assert "the floor confirm is checked against the idle shared reading and the CUDA_Host buffer sum" in text
    assert "the spill baseline is the model's lowest resident window" not in text
    assert "the prompt-rate rule is unchanged" not in text
    assert "a window spills when its shared reading exceeds that baseline by more than spill_shared_mib" not in text
    assert "a baseline that exceeds that sum by more than 256 MiB is inconclusive and is not confirmed" in text
    assert "CUDA_Host model, compute, and output sizes come from the first-load lines" in text
    assert "a missing CUDA_Host line refuses after the first load" in text
    assert "the shared gate is the delta from the baseline taken before the window" not in text
    assert "preflight samples nvidia-smi, both counters, the server log, the ggml-cuda sha, git, and driver CUDA before any load" in text
    assert "driver CUDA accepts CUDA Version and CUDA UMD Version" in text
    assert "a missing log or dll refuses at preflight" in text
    assert "the log is INTERLOCK_OLLAMA_LOG, else the 11434 listener parent's redirect" in text
    assert "the tray app log is never a fallback" in text
    assert "an unresolved log refuses at preflight" in text
    assert "OLLAMA_HOST 127.0.0.1:11434" in text
    assert "an empty KV cache type is unset" in text
    assert "gated on the first-load lines" in text
    assert "a missing load line refuses before any measured window" in text
    assert "the runner pid is llama-server.exe and stays record-only" in text
    assert "the exe is INTERLOCK_OLLAMA_EXE, else the 11434 listener" in text
    assert "quant is read from /api/show at run time" in text
    assert "closing the request does not return the memory a load already allocated" in text
    assert "one measured window bounds the next context from that window's share" in text
    assert "shared_gate_from_mib: 28452" in text
    assert "the shared gate counts only under capacity pressure" in text
    assert "below shared_gate_from_mib, shared growth is recorded and is not a spill" in text
    assert "the prompt-rate gate is a cliff against the next lower confirmed window" in text
    assert "after two resident loads the next peak is predicted from their measured slope" in text
    assert "limit is predicted when predicted_over stopped the search" in text
    assert "a measured abort on the confirm walk is margin" in text
    assert "settle_tolerance_mib: 512" in text
    assert "settle_timeout_s: 30" in text
    assert "a call at the same num_ctx as the resident runner keeps the resident floor" in text
    assert "a num_ctx change unloads first and settles to the idle baseline, then the call loads at the new num_ctx" in text
    assert "between models and after an unload that level is the idle baseline" in text
    assert "a card that does not settle within settle_timeout_s refuses that call and the measurement is not kept" in text
    smoke = dry_run(ROOT / "experiments" / "e5-smoke" / "spec.yaml", ROOT)
    assert "settle_tolerance_mib: 512" in smoke
    assert "settle_timeout_s: 30" in smoke
    assert "a call at the same num_ctx as the resident runner keeps the resident floor" in smoke
    assert "a num_ctx change unloads first and settles to the idle baseline, then the call loads at the new num_ctx" in smoke
    assert "a measured abort on the confirm walk is margin" in smoke
    assert "a second window with no bound stops the row" in text
    assert "the next load waits until the abort has finished" in text
    assert "an unload that does not finish within 60 seconds stops the run; the model may still be resident" in text
    assert "heartbeat not read; adapters not called" in text
    assert "devices not started (--dry-run)" in text
    assert "rest_s" not in text
    source = (ROOT / "src" / "interlock" / "harness" / "fill.py").read_text(encoding="utf-8")
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
    assert ctxs[0] == 2048
    assert 4096 in ctxs
    assert all(ctx <= 4096 for ctx in ctxs)
    assert rows[0]["settings"]["max_resident_ctx"] == 4096
    assert rows[0]["settings"]["num_ctx"] == 4096


def test_a_trained_window_that_is_not_resident_searches_down_to_the_grain(tmp_path: Path):
    # 17200 MiB of weights leaves a share the floor can bound under abort, so the
    # cap is loaded and found not resident instead of being predicted over.
    weights = 17200 * 1024 * 1024
    transport = _Transport(None, trained=16384, unresident_at=8192, tag_bytes=weights)
    seams = _seams(transport)
    _repo, rows = _run(tmp_path, transport, seams=seams, max_ctx=16384)
    ctxs = _ctxs(transport)
    row = rows[0]
    assert ctxs[0] == 2048
    assert all(ctx <= 16384 for ctx in ctxs)
    assert 8192 in ctxs
    assert 16384 in ctxs
    assert row["settings"]["max_resident_ctx"] == 6144
    assert 6144 % 2048 == 0
    calls = row["settings"]["search_calls"]
    assert calls[0] == ["load", 2048, 20000, 1000, 100, "resident"]
    assert ["load", 16384, 20000, 1000, 100, "not resident"] in calls
    assert ["load", 6144, 20000, 1000, 100, "resident"] in calls
    assert calls[-1] == ["confirm", 6144, 20000, 1000, 100, "confirmed"]


def test_shared_growth_is_measured_from_the_floor_confirm(tmp_path: Path):
    """The 8192 confirm spills against the floor confirm, not against its own load.

    The floor confirm sits at 100. The cap's load also sits at 100, so a
    comparison with that load would be 0. The 8192 confirm reads 2200, which
    is 2100 over the floor, and its peak is at the shared gate. The confirmed
    window below it is 6144 at 200.
    """

    transport = _Transport(None, trained=8192, gate_peak_at=8192)
    seams = _seams(transport)
    script = [
        (_GPU, 100, 12),
        (_GPU, 100, 12),
        (_GPU, 100, 12),
        (_GPU, 100, 12),
        (_GPU, 100, 12),
        (_GPU, 100, 12),
        (_GPU, 100, 12),
        (_GPU, 150, 12),
        (_GPU, 150, 12),
        (_GPU, 200, 14),
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
    calls = rows[0]["settings"]["search_calls"]
    assert rows[0]["settings"]["max_resident_ctx"] == 6144
    assert rows[0]["settings"]["shared_mib"] == 200
    assert ["confirm", 6144, 20000, 200, 100, "confirmed"] in calls
    assert calls[-1] == ["confirm", 8192, SHARED_GATE_FROM_MIB, 2200, 100, "spill"]


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

    transport = _Transport(None, trained=8192, gate_peak_at=8192)
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
    assert rows[0]["settings"]["shared_mib"] == 1000
    assert ["confirm", 6144, 20000, 1000, 100, "confirmed"] in calls
    assert ["confirm", 8192, SHARED_GATE_FROM_MIB, 1257, 100, "spill"] in calls

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
    assert settings["backend_library"] == install_relative(library)
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
    assert runner_pid(["42 7 ollama.exe"]) is None
    assert runner_pid(["42 7 llama-server.exe"]) == 42
    assert runner_pid(["42 7 LLAMA-SERVER.EXE"]) == 42
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
            assert "llama-server.exe" in script
            return Completed("42 100 llama-server.exe\n")
        raise AssertionError(argv)

    seen = []
    seen_counters = []

    class Adapter:
        def transport(self, method, url, body):
            assert method == "GET" and url.endswith("/api/version")
            return {"version": "0.35.1"}

    seams = real_fill_seams(Adapter(), run=run, read_counter=read_counter)
    assert seams.card_mib() == 20000
    assert seams.shared_mib() is None
    seams.bind_pin(_GPU)
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
    assert seams.shared_mib() == 30
    assert seams.dedicated_mib() == 20000


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


_SERVER_LOG = (
    'msg="server config" env="map[OLLAMA_FLASH_ATTENTION:true OLLAMA_KV_CACHE_TYPE:q8_0 '
    'OLLAMA_HOST:http://127.0.0.1:11434 OLLAMA_CONTEXT_LENGTH:8192]"\n'
    "load_backend: loaded CUDA backend from C:/ollama/lib/ollama/cuda_v13/ggml-cuda.dll\n"
)
_DLL_BYTES = b"cuda-bytes"
_GPU_LUID = "luid_0x00000000_0x000140e1_phys_0"


def _clear_fill_env(monkeypatch) -> None:
    for name in (
        "INTERLOCK_OLLAMA_LOG",
        "INTERLOCK_OLLAMA_EXE",
        "SWITCHYARD_OLLAMA_LOG",
        "SWITCHYARD_OLLAMA_EXE",
    ):
        monkeypatch.delenv(name, raising=False)


def _fixture_text(name: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current = None
    for line in (ROOT / "tests" / "fixtures" / "e5" / name).read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections[current] = []
            continue
        if current is not None:
            sections[current].append(line)
    return {key: "\n".join(lines) + "\n" for key, lines in sections.items()}


class _Completed:
    def __init__(self, stdout: str, returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode


def _install_tree(tmp_path: Path, *, log: bool = True, dll: bool = True) -> dict:
    root = tmp_path / "dist"
    exe = root / "ollama.exe"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_bytes(b"exe")
    dll_path = root / "lib" / "ollama" / "cuda_v13" / "ggml-cuda.dll"
    if dll:
        dll_path.parent.mkdir(parents=True, exist_ok=True)
        dll_path.write_bytes(_DLL_BYTES)
    app = tmp_path / "appdata"
    log_path = app / "Ollama" / "server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log:
        log_path.write_text(_SERVER_LOG, encoding="utf-8", newline="\n")
    return {"exe": exe, "dll": dll_path, "log": log_path, "app": app}


def _zero_dedicated(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if "=" not in line:
            continue
        name, _value = line.split("=", 1)
        lines.append(f"{name}=0")
    return "\n".join(lines) + "\n"


def _live_commands(install: dict, fault: str):
    idle = _fixture_text("idle-card-pin.txt")
    header = (ROOT / "tests" / "fixtures" / "e5" / "nvidia-smi-617.42.txt").read_text(encoding="utf-8")
    if fault == "several":
        dedicated = _fixture_text("two-adapters-within-tolerance.txt")["dedicated"]
    elif fault == "zero":
        dedicated = _zero_dedicated(idle["dedicated"])
    else:
        dedicated = idle["dedicated"]
    shared = "luid_0x00000000_0x00011111_phys_0=0\n" if fault == "shared" else idle["shared"]
    nvidia = "nope\n" if fault == "nvidia" else idle["nvidia-smi"]
    if fault == "cuda":
        header = "NVIDIA-SMI 617.42    Driver Version: 617.42    KMD Version: 617.42\n"
    seen = []

    def run(argv, **kwargs):
        argv = tuple(argv)
        seen.append(argv)
        if argv == MEMORY_ARGV:
            return _Completed(nvidia)
        if argv == NVIDIA_SMI_ARGV:
            return _Completed(header)
        if argv == DRIVER_ARGV:
            return _Completed("617.42\n")
        if argv and argv[0] == "powershell":
            script = argv[-1]
            if "llama-server.exe" in script:
                return _Completed("77 5 llama-server.exe\n")
            if "CommandLine" in script:
                return _Completed(f'cmd.exe /c ollama serve >> "{install["log"]}" 2>&1')
            if "ConvertTo-Json" in script:
                return _Completed(json.dumps({"exe": str(install["exe"])}))
        raise AssertionError(argv)

    def read_counter(counter):
        if counter == ADAPTER_DEDICATED:
            return dedicated
        if counter == ADAPTER_SHARED:
            return shared
        raise AssertionError(counter)

    return run, read_counter, seen


def _committed_repo(tmp_path: Path, spec: dict, *, dirty: bool = False, folder: str = "repo"):
    repo = tmp_path / folder
    repo.mkdir()
    (repo / ".gitignore").write_text(
        "receipts/*.run.json\nreceipts/*.tmp\nreceipts/*.stop\nreceipts/*.stop.done\n",
        encoding="utf-8",
        newline="\n",
    )
    spec_path = _bind(repo, spec)
    if dirty:
        (repo / "DIRTY").write_text("dirty\n", encoding="utf-8", newline="\n")
    return repo, spec_path


class _Probe:
    """Accepts the empty-card check and refuses anything that would load."""

    def __init__(self) -> None:
        self.calls = []

    def __call__(self, method, url, body):
        self.calls.append((method, url, body))
        if str(url).endswith("/api/ps"):
            return {"models": []}
        raise AssertionError(f"load before refusal: {method} {url}")


def _no_load(calls) -> None:
    assert not any(str(url).endswith("/api/show") or str(url).endswith("/api/generate") for _m, url, _b in calls)


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("nvidia", "nvidia-smi memory.used returned no number"),
        ("zero", "zero instances within adapter_match_tolerance_mib"),
        ("several", "several instances within adapter_match_tolerance_mib"),
        ("shared", "Shared Usage returned no number"),
        ("log", "server log"),
        ("dll", "ggml-cuda.dll"),
        ("cuda", "driver CUDA did not parse"),
        ("git", "40-hex"),
        ("git-down", "exit code"),
        ("tolerance", "matching allowance"),
    ],
)
def test_a_preflight_failure_refuses_before_any_load(tmp_path: Path, monkeypatch, fault, message):
    _clear_fill_env(monkeypatch)
    install = _install_tree(tmp_path, log=fault != "log", dll=fault != "dll")
    monkeypatch.setenv("LOCALAPPDATA", str(install["app"]))
    if fault == "git":
        monkeypatch.setattr(
            "interlock.harness.e4b.read_git",
            lambda repo, run: {"git_sha": "abc", "clean_tree": True},
        )
    if fault == "git-down":
        def _down(repo, run):
            from interlock.harness.e4b import E4bError

            raise E4bError("git rev-parse HEAD failed with exit code 1")

        monkeypatch.setattr("interlock.harness.e4b.read_git", _down)
    spec = _spec([_model()])
    if fault == "tolerance":
        spec["adapter_match_tolerance_mib"] = 256
    repo, spec_path = _committed_repo(tmp_path, spec, folder=f"refuse-{fault}")
    run, read_counter, _seen = _live_commands(install, fault)
    transport = _Probe()
    adapter = OllamaAdapter(transport=transport)
    seams = real_fill_seams(adapter, run=run, read_counter=read_counter)
    with pytest.raises((RunnerError, SpecError), match=message):
        run_spec(
            spec_path,
            root=repo,
            grant="g",
            adapters={"ollama": adapter},
            heartbeat=_fresh(tmp_path),
            date="2026-10-12",
            run_id=f"fill-{fault}",
            seams=seams,
        )
    _no_load(transport.calls)
    assert seams.shared_mib() is None
    assert not (repo / "receipts" / f"fill-{fault}.jsonl").exists()


def test_preflight_refuses_a_wrong_tolerance_before_any_sample(tmp_path: Path):
    seen = []

    def run(argv, **kwargs):
        seen.append(tuple(argv))
        raise AssertionError(argv)

    seams = real_fill_seams(object(), run=run, read_counter=lambda counter: (_ for _ in ()).throw(AssertionError(counter)))
    spec = _spec([_model()])
    spec["adapter_match_tolerance_mib"] = 256
    with pytest.raises(FillPreflightError, match="matching allowance"):
        seams.preflight(spec, tmp_path)
    assert seen == []
    assert seams.shared_mib() is None


def test_a_dirty_start_is_recorded_false_on_the_aborted_receipt(tmp_path: Path, monkeypatch):
    _clear_fill_env(monkeypatch)
    install = _install_tree(tmp_path)
    monkeypatch.setenv("INTERLOCK_OLLAMA_LOG", str(install["log"]))
    monkeypatch.setenv("INTERLOCK_OLLAMA_EXE", str(install["exe"]))
    repo, spec_path = _committed_repo(tmp_path, _spec([_model()]), dirty=True, folder="dirty")
    run, read_counter, seen = _live_commands(install, "ok")
    transport = _Probe()
    adapter = OllamaAdapter(transport=transport)
    seams = real_fill_seams(adapter, run=run, read_counter=read_counter)
    with pytest.raises(RunnerError, match="git porcelain is not empty"):
        run_spec(
            spec_path,
            root=repo,
            grant="g",
            adapters={"ollama": adapter},
            heartbeat=_fresh(tmp_path),
            date="2026-10-12",
            run_id="fill-dirty",
            seams=seams,
        )
    _no_load(transport.calls)
    assert not any(argv and argv[0] == "powershell" for argv in seen)
    receipt = repo / "receipts" / "fill-dirty.jsonl"
    rows = load_jsonl(receipt)
    row = rows[0]
    sha = subprocess.run(
        ["git", "-C", repo.as_posix(), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert row["aborted"] is True
    assert row["provenance"]["git_sha"] == sha
    assert row["provenance"]["clean_tree"] is False
    assert row["settings"]["adapter_instance"] == _GPU_LUID
    assert row["settings"]["adapter_match_dedicated_mib"] == 2508
    assert row["settings"]["adapter_match_nvidia_mib"] == 2490
    assert row["settings"]["preflight_shared_mib"] == 74
    assert row["settings"]["preflight_driver_cuda"] == "driver 13.4"
    assert row["settings"]["log_source"] == "INTERLOCK_OLLAMA_LOG"
    assert row["settings"]["exe_source"] == "INTERLOCK_OLLAMA_EXE"
    text = receipt.read_text(encoding="utf-8")
    assert "Users" not in text
    assert str(install["log"]) not in text
    assert str(install["exe"]) not in text


def _recorded_log(name: str) -> str:
    return (ROOT / "tests" / "fixtures" / "e5" / name).read_text(encoding="utf-8")


def _decoy_tray(tmp_path: Path, monkeypatch) -> None:
    tray = tmp_path / "tray"
    (tray / "Ollama").mkdir(parents=True)
    (tray / "Ollama" / "server.log").write_text(
        'msg="server config" env="map[OLLAMA_HOST:http://127.0.0.1:11490 OLLAMA_FLASH_ATTENTION:true]"\n',
        encoding="utf-8",
        newline="\n",
    )
    monkeypatch.setenv("LOCALAPPDATA", str(tray))


def test_a_clean_preflight_records_the_sample_the_sources_and_git(tmp_path: Path, monkeypatch, capsys):
    _clear_fill_env(monkeypatch)
    install = _install_tree(tmp_path)
    _decoy_tray(tmp_path, monkeypatch)
    install["log"].write_text(_recorded_log("server-config-11434.txt"), encoding="utf-8", newline="\n")
    load_lines = _recorded_log("load-lines-11434.txt")

    def append() -> None:
        with install["log"].open("a", encoding="utf-8", newline="") as handle:
            handle.write(load_lines if load_lines.endswith("\n") else load_lines + "\n")

    repo, spec_path = _committed_repo(tmp_path, _spec([_model()]), folder="clean")
    run, read_counter, seen = _live_commands(install, "ok")
    transport = _Transport(None)
    transport.before_generate = append

    def rising(argv, **kwargs):
        # The idle pin stays 2490. While a generate is in flight the card rises,
        # so the floor has a positive share and can bound the cap.
        if tuple(argv) == MEMORY_ARGV and transport.state["busy"]:
            return _Completed("20000\n")
        return run(argv, **kwargs)

    adapter = OllamaAdapter(transport=transport)
    seams = real_fill_seams(adapter, run=rising, read_counter=read_counter)
    seams.threaded = False
    seams.sleep = lambda _seconds: None
    transport.seams = seams
    rows = run_spec(
        spec_path,
        root=repo,
        grant="g",
        adapters={"ollama": adapter},
        heartbeat=_fresh(tmp_path),
        date="2026-10-12",
        run_id="fill-clean",
        seams=seams,
    )
    printed = capsys.readouterr().out
    assert "driver_cuda=driver 13.4" in printed
    assert "config_flash_attention=false" in printed
    assert "config_kv_cache_type=unset" in printed
    assert "log_source=listener-parent" in printed
    assert "LOCALAPPDATA" not in printed
    assert "exe_source=listener" in printed
    assert _GPU_LUID in printed
    assert any("CommandLine" in (argv[-1] if argv else "") for argv in seen)
    assert any("ConvertTo-Json" in (argv[-1] if argv else "") for argv in seen)
    row = rows[0]
    sha = subprocess.run(
        ["git", "-C", repo.as_posix(), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    settings = row["settings"]
    assert row["provenance"]["git_sha"] == sha
    assert row["provenance"]["clean_tree"] is True
    assert settings["adapter_instance"] == _GPU_LUID
    assert settings["adapter_match_dedicated_mib"] == 2508
    assert settings["adapter_match_nvidia_mib"] == 2490
    assert settings["preflight_shared_mib"] == 74
    assert settings["shared_mib"] == 74
    assert settings["dedicated_mib"] == 2508
    assert settings["preflight_driver_cuda"] == "driver 13.4"
    assert settings["cuda_backend"] == "driver 13.4"
    assert settings["config_flash_attention"] == "false"
    assert settings["config_kv_cache_type"] == "unset"
    assert settings["flash_attention"] == "enabled"
    assert settings["flash_attn"] == "auto"
    assert settings["kv_cache_type"] == "f16"
    assert settings["ggml_cuda_sha"] == hashlib.sha256(_DLL_BYTES).hexdigest()
    assert settings["log_source"] == "listener-parent"
    assert settings["exe_source"] == "listener"
    assert settings["runner_pid"] == 77
    assert settings["max_resident_ctx"] == 4096
    assert settings["baseline_ctx"] == 2048
    assert settings["baseline_shared_mib"] == 74
    assert settings["idle_shared_mib"] == 74
    assert settings["baseline_check"] == "ok"
    assert settings["cuda_host_model_mib"] == 333.84
    assert settings["cuda_host_compute_mib"] == 18.01
    assert settings["cuda_host_output_mib"] == 0.58
    assert settings["cuda_host_sum_mib"] == 352.43
    assert isinstance(settings["baseline_ctx"], int)
    assert isinstance(settings["baseline_shared_mib"], int)
    assert all(isinstance(item["delta_mib"], int) for item in settings["shared_delta_mib"])
    urls = [url for _method, url, _body in transport.calls]
    assert urls[0].endswith("/api/ps")
    assert any(url.endswith("/api/show") for url in urls)
    assert any(url.endswith("/api/generate") for url in urls)
    text = (repo / "receipts" / "fill-clean.jsonl").read_text(encoding="utf-8")
    assert "Users" not in text
    assert str(install["log"]) not in text
    assert str(install["exe"]) not in text


def test_a_missing_load_line_refuses_before_a_measured_window(tmp_path: Path, monkeypatch):
    _clear_fill_env(monkeypatch)
    install = _install_tree(tmp_path)
    _decoy_tray(tmp_path, monkeypatch)
    # The lines are already in the file, from an earlier model. This load writes nothing.
    install["log"].write_text(
        _recorded_log("server-config-11434.txt") + _recorded_log("load-lines-11434.txt"),
        encoding="utf-8",
        newline="\n",
    )
    repo, spec_path = _committed_repo(tmp_path, _spec([_model()]), folder="stale-load")
    run, read_counter, _seen = _live_commands(install, "ok")
    transport = _Transport(None)
    adapter = OllamaAdapter(transport=transport)
    seams = real_fill_seams(adapter, run=run, read_counter=read_counter)
    seams.threaded = False
    seams.sleep = lambda _seconds: None
    transport.seams = seams
    with pytest.raises(RunnerError, match="first load"):
        run_spec(
            spec_path,
            root=repo,
            grant="g",
            adapters={"ollama": adapter},
            heartbeat=_fresh(tmp_path),
            date="2026-10-12",
            run_id="fill-stale",
            seams=seams,
        )
    row = load_jsonl(repo / "receipts" / "fill-stale.jsonl")[0]
    assert row["aborted"] is True
    assert row["settings"]["max_resident_ctx"] is None
    assert "search_calls" not in row["settings"]
    assert row["settings"]["flash_attention"] == "unrecorded"
    assert row["settings"]["kv_cache_type"] == "unrecorded"
    assert "flash_attn" not in row["settings"]
    assert row["settings"]["config_flash_attention"] == "false"
    assert row["settings"]["config_kv_cache_type"] == "unset"
    assert row["settings"]["runner_pid"] == 77
    assert row["settings"]["log_source"] == "listener-parent"
    prompts = [
        body.get("prompt")
        for _method, url, body in transport.calls
        if str(url).endswith("/api/generate") and isinstance(body, dict) and body.get("prompt")
    ]
    assert len(prompts) == 1
    text = (repo / "receipts" / "fill-stale.jsonl").read_text(encoding="utf-8")
    assert "Users" not in text


def test_missing_cuda_host_lines_refuse_after_the_first_load(tmp_path: Path, monkeypatch):
    _clear_fill_env(monkeypatch)
    install = _install_tree(tmp_path)
    _decoy_tray(tmp_path, monkeypatch)
    install["log"].write_text(_recorded_log("server-config-11434.txt"), encoding="utf-8", newline="\n")
    full = _recorded_log("load-lines-11434.txt")
    kept = "\n".join(line for line in full.splitlines() if "CUDA_Host" not in line) + "\n"
    assert "flash-attn" in kept and "llama_kv_cache" in kept
    assert "CUDA_Host" not in kept

    def append() -> None:
        with install["log"].open("a", encoding="utf-8", newline="") as handle:
            handle.write(kept)

    repo, spec_path = _committed_repo(tmp_path, _spec([_model()]), folder="missing-host")
    run, read_counter, _seen = _live_commands(install, "ok")
    transport = _Transport(None)
    transport.before_generate = append
    adapter = OllamaAdapter(transport=transport)
    seams = real_fill_seams(adapter, run=run, read_counter=read_counter)
    seams.threaded = False
    seams.sleep = lambda _seconds: None
    transport.seams = seams
    with pytest.raises(RunnerError, match="CUDA_Host"):
        run_spec(
            spec_path,
            root=repo,
            grant="g",
            adapters={"ollama": adapter},
            heartbeat=_fresh(tmp_path),
            date="2026-10-13",
            run_id="fill-missing-host",
            seams=seams,
        )
    row = load_jsonl(repo / "receipts" / "fill-missing-host.jsonl")[0]
    assert row["aborted"] is True
    assert row["settings"]["max_resident_ctx"] is None
    assert "search_calls" not in row["settings"]
    assert "CUDA_Host model" in row["notes"]
    assert "CUDA_Host compute" in row["notes"]
    assert "CUDA_Host output" in row["notes"]
    assert "Users" not in (repo / "receipts" / "fill-missing-host.jsonl").read_text(encoding="utf-8")
