"""Recorded Ollama exchanges, replayed by fakes. Nothing here starts a server model."""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from interlock.adapters.base import AdapterError
from interlock.adapters.ollama import COUNT_TIMEOUT_NOTE, OllamaAdapter
from interlock.adapters.ollama_cases import (
    CASES,
    MODEL,
    NO_LIMIT_NOTE,
    RecordedNoReply,
    case_by_name,
    covered_parameters,
    fixture_document,
    request_body,
    sent_parameters,
)
from interlock.adapters.record_fixtures import (
    SETTLE_STEP_S,
    RecordError,
    _observed,
    http_exchange,
    read_card_mib,
    record_fixtures,
)
from interlock.harness.count_budget import PREFILL_TOKENS_PER_S, count_timeout_s, roster_timeout
from tests.ollama_replay import ReplayTransport, replay

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "ollama"


def _beat(tmp_path: Path) -> Path:
    path = tmp_path / "_watchdog_HEARTBEAT"
    path.write_text("pulse", encoding="utf-8")
    now = time.time()
    os.utime(path, (now, now))
    return path


def _record(tmp_path: Path, **kwargs):
    beat = kwargs.pop("heartbeat", None)
    if beat is None:
        beat = _beat(tmp_path)
    values = {
        "model": "qwen3:8b",
        "host": "http://127.0.0.1:11434",
        "out_dir": tmp_path,
        "timeout": 15,
        "ceiling": 4096,
        "today": "2026-10-12",
        "grant": "grant-t",
        "heartbeat": beat,
        "now": time.time(),
    }
    values.update(kwargs)
    return record_fixtures(**values)


def _document(case):
    if case.behaviour == "no_reply":
        response = {"done": True, "eval_count": 64}
    elif case.behaviour == "error":
        response = {"error": "the prompt exceeds the context length"}
    elif case.name == "raw_true":
        response = {"done": True, "prompt_eval_count": 2}
    elif case.path == "/api/version":
        response = {"version": "0.0.0"}
    else:
        response = {"done": True}
    return fixture_document(
        case,
        model=MODEL,
        version="0.0.0",
        digest="sha256:abc",
        date="2026-10-12",
        behaviour=case.behaviour,
        response=response,
    )


def test_replay_of_each_recorded_fixture(tmp_path: Path):
    documents = [_document(case) for case in CASES]
    for document in documents:
        if document["behaviour"] == "no_reply":
            assert document["response"]["eval_count"] > 1
            body = document["request"]["body"]
            assert body["options"]["num_ctx"] == 64
            assert body["shift"] is False
            with pytest.raises(RecordedNoReply, match="no reply"):
                replay(document)
            continue
        assert replay(document) == document["response"]
    folder = tmp_path / "0.0.0"
    folder.mkdir()
    for document in documents:
        (folder / f"{document['case']}.json").write_text(json.dumps(document), encoding="utf-8")
    transport = ReplayTransport.load(folder)
    version = transport("GET", "http://127.0.0.1:11434/api/version", None)
    assert version == {"version": "0.0.0"}
    raw = next(document for document in documents if document["case"] == "raw_true")
    assert transport("POST", "http://127.0.0.1:11434/api/generate", raw["request"]["body"])["prompt_eval_count"] == 2
    silent = next(document for document in documents if document["case"] == "num_predict_0")
    with pytest.raises(RecordedNoReply, match="no reply"):
        transport("POST", "http://127.0.0.1:11434/api/generate", silent["request"]["body"])
    if FIXTURE_ROOT.is_dir():
        for version_dir in sorted(path for path in FIXTURE_ROOT.iterdir() if path.is_dir()):
            recorded = ReplayTransport.load(version_dir)
            assert {document["case"] for document in recorded.documents} == {case.name for case in CASES}
            for document in recorded.documents:
                catalog = case_by_name(document["case"])
                if catalog.behaviour == "no_reply":
                    assert document["behaviour"] == "no_reply"
                    assert document["response"]["eval_count"] > 1
                    body = document["request"]["body"]
                    assert body["options"]["num_ctx"] == 64
                    assert body["shift"] is False
                else:
                    replay(document)


def test_the_catalogue_agrees_with_the_recorded_fixture():
    assert FIXTURE_ROOT.is_dir()
    version_dirs = [path for path in FIXTURE_ROOT.iterdir() if path.is_dir()]
    assert version_dirs
    for version_dir in version_dirs:
        for case in CASES:
            path = version_dir / f"{case.name}.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            assert document["behaviour"] == case.behaviour, case.name
            assert document["request"]["method"] == case.method, case.name
            assert document["request"]["path"] == case.path, case.name
            assert document["request"]["body"] == request_body(case, document["model"]), case.name
    for name in ("num_predict_0", "num_predict_null", "num_predict_missing"):
        case = case_by_name(name)
        assert case.note == NO_LIMIT_NOTE
        assert "11 prompt tokens and 144 eval tokens" in case.note
        assert "num_ctx 64" in case.note
        document = json.loads((FIXTURE_ROOT / "0.35.1" / f"{name}.json").read_text(encoding="utf-8"))
        assert document["response"]["prompt_eval_count"] == 11
        assert document["response"]["eval_count"] == 144
        assert document["response"]["done_reason"] == "stop"
        assert document["request"]["body"]["options"]["num_ctx"] == 64


def test_only_records_truncate_at_the_override_window(tmp_path: Path):
    import importlib.util

    path = ROOT / "scripts" / "record_ollama_fixtures.py"
    spec = importlib.util.spec_from_file_location("record_ollama_fixtures_only", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    beat = _beat(tmp_path)
    bodies = []

    def transport(method, url, body, timeout):
        bodies.append((method, url, body))
        assert timeout == 15
        if url.endswith("/api/ps"):
            return {"models": []}
        if url.endswith("/api/version"):
            return {"version": "0.35.1"}
        if url.endswith("/api/tags"):
            return {"models": [{"name": "qwen3:8b", "digest": "sha256:abc", "size": 5}]}
        return {"done": True, "done_reason": "length"}

    code = module.main(
        [
            "--model", "qwen3:8b",
            "--host", "http://127.0.0.1:11434",
            "--timeout", "15",
            "--out", str(tmp_path / "out"),
            "--grant", "grant-t",
            "--only", "truncate_over_num_ctx",
            "--num-ctx", "2048",
            "--prompt-repeats", "2500",
        ],
        transport=transport,
        card_mib=lambda: 100,
        today="2026-10-12",
        heartbeat=beat,
        now=time.time(),
    )
    assert code == 0
    written = list((tmp_path / "out").rglob("*.json"))
    assert [item.name for item in written] == ["truncate_over_num_ctx.json"]
    document = json.loads(written[0].read_text(encoding="utf-8"))
    assert document["request"]["body"]["options"]["num_ctx"] == 2048
    assert document["request"]["body"]["prompt"] == "word " * 2500
    assert document["request"]["body"]["truncate"] is False
    assert document["behaviour"] == "reply"
    assert any(isinstance(body, dict) and body.get("keep_alive") == 0 for _method, _url, body in bodies)
    catalog = case_by_name("truncate_over_num_ctx")
    assert catalog.behaviour == "error"
    assert catalog.body["options"]["num_ctx"] == 2048
    assert catalog.body["prompt"] == "word " * 2500
    called = []
    refused = module.main(
        ["--model", "qwen3:8b", "--grant", "grant-t", "--only", "not-a-case"],
        transport=lambda *_args: called.append("called"),
        card_mib=lambda: 100,
        today="2026-10-12",
        heartbeat=beat,
        now=time.time(),
    )
    assert refused == 2
    assert called == []


def test_every_ollama_option_the_adapter_can_send_has_a_fixture():
    missing = sent_parameters() - covered_parameters()
    assert missing == frozenset()


def test_the_recorder_command_refuses_a_cloud_model(capsys, tmp_path: Path):
    import importlib.util

    path = ROOT / "scripts" / "record_ollama_fixtures.py"
    spec = importlib.util.spec_from_file_location("record_ollama_fixtures_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    beat = tmp_path / "_watchdog_HEARTBEAT"
    beat.write_text("pulse", encoding="utf-8")
    code = module.main(
        ["--model", "qwen3:8b:cloud", "--grant", "grant-t"],
        transport=lambda *_args: {},
        card_mib=lambda: 100,
        today="2026-10-12",
        heartbeat=beat,
        now=time.time(),
    )
    assert code == 2
    assert "cloud" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        module.main(["--model", "qwen3:8b"])


def test_the_recorder_refuses_a_cloud_model(tmp_path: Path):
    called = []

    def transport(*_args):
        called.append("called")
        return {}

    with pytest.raises(RecordError, match="cloud"):
        _record(tmp_path, model="qwen3:8b:cloud", transport=transport, card_mib=lambda: 100)
    assert called == []


def test_the_recorder_refuses_to_run_without_a_grant(tmp_path: Path):
    called = []

    def transport(*_args):
        called.append("called")
        return {}

    with pytest.raises(RecordError, match="--grant"):
        _record(tmp_path, grant="", transport=transport, card_mib=lambda: 100)
    stale = tmp_path / "stale"
    stale.write_text("old", encoding="utf-8")
    os.utime(stale, (time.time() - 120, time.time() - 120))
    with pytest.raises(RecordError, match="heartbeat"):
        _record(
            tmp_path,
            grant="grant-t",
            heartbeat=stale,
            now=time.time(),
            transport=transport,
            card_mib=lambda: 100,
        )
    assert called == []


def test_the_recorder_refuses_a_card_that_is_not_at_baseline(tmp_path: Path):
    calls = []

    def loaded(method, url, _body, _timeout):
        calls.append(url)
        if url.endswith("/api/ps"):
            return {"models": [{"name": "qwen3:8b"}]}
        return {}

    with pytest.raises(RecordError, match="still on the card"):
        _record(tmp_path, transport=loaded, card_mib=lambda: 100)
    assert calls == ["http://127.0.0.1:11434/api/ps"]
    with pytest.raises(RecordError, match="above 4096"):
        _record(tmp_path, transport=lambda *_args: {"models": []}, card_mib=lambda: 9000)


def _recording_transport(timeouts: list):
    def transport(method, url, body, timeout):
        timeouts.append(timeout)
        if url.endswith("/api/ps"):
            return {"models": []}
        if url.endswith("/api/version"):
            return {"version": "0.35.1"}
        if url.endswith("/api/tags"):
            return {"models": [{"name": "qwen3:8b", "digest": "sha256:abc", "size": 5}]}
        if url.endswith("/api/show"):
            return {"model_info": {"qwen3.context_length": 4096}}
        options = body.get("options") if isinstance(body, dict) else None
        if isinstance(options, dict) and ("num_predict" not in options or options.get("num_predict") in (0, None)):
            return {"done": True, "eval_count": 64}
        if isinstance(options, dict) and options.get("num_ctx") == 2048 and len(str(body.get("prompt",
            "")).split()) > 2048:
            return {"error": "the prompt exceeds the context length"}
        return {"done": True, "prompt_eval_count": 1}

    return transport


def test_num_predict_0_is_recorded_as_no_reply_within_the_timeout(tmp_path: Path):
    timeouts: list = []
    written = _record(
        tmp_path,
        transport=_recording_transport(timeouts),
        card_mib=lambda: 100,
    )
    assert len(written) == len(CASES)
    assert set(timeouts) == {15}
    saved = json.loads((tmp_path / "0.35.1" / "num_predict_0.json").read_text(encoding="utf-8"))
    assert saved["behaviour"] == "no_reply"
    assert saved["response"]["eval_count"] > 1
    assert saved["request"]["body"]["options"]["num_ctx"] == 64
    assert saved["request"]["body"]["shift"] is False
    assert saved["ollama_version"] == "0.35.1"
    assert saved["digest"] == "sha256:abc"
    assert saved["date"] == "2026-10-12"
    assert saved["request"]["body"]["options"]["num_predict"] == 0
    cold = json.loads((tmp_path / "0.35.1" / "num_predict_1.json").read_text(encoding="utf-8"))
    assert isinstance(cold["cold_load_s"], float) and cold["cold_load_s"] >= 0
    for name in ("num_predict_null", "num_predict_missing"):
        document = json.loads((tmp_path / "0.35.1" / f"{name}.json").read_text(encoding="utf-8"))
        assert document["behaviour"] == "no_reply"
        assert document["response"]["eval_count"] > 1
        assert document["request"]["body"]["options"]["num_ctx"] == 64
        assert document["request"]["body"]["shift"] is False
    truncate = json.loads((tmp_path / "0.35.1" / "truncate_over_num_ctx.json").read_text(encoding="utf-8"))
    assert truncate["behaviour"] == "error"
    assert "exceeds" in truncate["response"]["error"]
    raw = json.loads((tmp_path / "0.35.1" / "raw_true.json").read_text(encoding="utf-8"))
    assert raw["response"]["prompt_eval_count"] == 1


def test_http_exchange_stops_at_the_timeout_and_reads_a_json_object():
    import socket

    listening = socket.socket()
    listening.bind(("127.0.0.1", 0))
    listening.listen(1)
    port = listening.getsockname()[1]
    try:
        assert http_exchange("GET", f"http://127.0.0.1:{port}/api/version", None, 0.2) is None
    finally:
        listening.close()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            raw = b'{"version":"0"}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, _format, *_args):
            return None

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    try:
        answered = http_exchange("GET", f"http://127.0.0.1:{server.server_address[1]}/api/version", None, 2)
    finally:
        thread.join(2)
        server.server_close()
    assert answered == {"version": "0"}


def test_read_card_mib_accepts_one_line_and_a_failed_read():
    class Completed:
        stdout = "1500\n"

    assert read_card_mib(run=lambda *_args, **_kwargs: Completed()) == 1500

    def broken(*_args, **_kwargs):
        raise OSError("nvidia-smi is not available")

    assert read_card_mib(run=broken) is None


def test_a_count_exchange_that_times_out_refuses_with_the_note():
    def transport(_method, _url, _body):
        raise AdapterError("the request timed out")

    adapter = OllamaAdapter(transport=transport, device="5090")
    with pytest.raises(AdapterError, match="the token count timed out") as raised:
        adapter.count_tokens("qwen3:8b", "hi", settings={"num_ctx": 32}, timeout=2)
    assert str(raised.value) == COUNT_TIMEOUT_NOTE


def test_a_no_limit_reply_stays_no_reply_and_keeps_eval_count():
    behaviour, response = _observed(case_by_name("num_predict_0"), None)
    assert behaviour == "no_reply" and response is None
    behaviour, response = _observed(case_by_name("num_predict_0"), {"done": True, "eval_count": 64})
    assert behaviour == "no_reply"
    assert response["eval_count"] > 1


def test_the_count_budget_names_a_prompt_length_term():
    bare = count_timeout_s([1024**3])
    long = count_timeout_s([1024**3], prompt_tokens=118_000)
    assert long - bare == pytest.approx(118_000 / PREFILL_TOKENS_PER_S)
    assert count_timeout_s([1024**3], prompt_tokens=True) == bare

    class Adapter:
        def tag_size(self, _model, _device):
            return 1024**3

    interleaved = {
        "procedure": "interleaved",
        "arms": [{"engine": "ollama", "model": "qwen3:8b", "device": "5090"}],
        "dataset": {"target_tokens": 6000},
    }
    assert roster_timeout(Adapter(), interleaved) == pytest.approx(bare + 1.0)
    fill = {
        "procedure": "fill-card",
        "models": [{"model": "qwen3:8b"}],
        "confirm_prompt_fraction": 0.5,
        "max_ctx": 12_000,
    }
    assert roster_timeout(Adapter(), fill) == pytest.approx(bare + 1.0)


def _is_recorded_load(body) -> bool:
    options = body.get("options") if isinstance(body, dict) else None
    return (
        isinstance(body, dict)
        and isinstance(options, dict)
        and options.get("num_predict") == 1
        and "num_ctx" not in options
        and "shift" not in body
        and "raw" not in body
        and "think" not in body
        and "companion" not in body
    )


def test_an_unload_waits_for_the_card_to_fall(tmp_path: Path):
    seen_load = {"yes": False}
    highs = {"n": 0}
    sleeps: list[float] = []

    def card():
        if not seen_load["yes"]:
            return 100
        highs["n"] += 1
        return 9000 if highs["n"] <= 3 else 100

    base = _recording_transport([])

    def transport(method, url, body, timeout):
        if _is_recorded_load(body):
            seen_load["yes"] = True
        return base(method, url, body, timeout)

    written = _record(tmp_path, transport=transport, card_mib=card, sleep=lambda seconds: sleeps.append(seconds))
    assert len(written) == len(CASES)
    assert highs["n"] >= 4
    assert sleeps[:3] == [SETTLE_STEP_S, SETTLE_STEP_S, SETTLE_STEP_S]


def test_a_card_that_never_falls_refuses_after_an_unload(tmp_path: Path):
    seen_load = {"yes": False}
    calls = []
    base = _recording_transport([])

    def transport(method, url, body, timeout):
        calls.append(body)
        if _is_recorded_load(body):
            seen_load["yes"] = True
        return base(method, url, body, timeout)

    with pytest.raises(RecordError, match="above 4096"):
        _record(
            tmp_path,
            transport=transport,
            card_mib=lambda: 9000 if seen_load["yes"] else 100,
            sleep=lambda _seconds: None,
        )
    load_at = next(index for index, body in enumerate(calls) if _is_recorded_load(body))
    assert any(isinstance(body, dict) and body.get("keep_alive") == 0 for body in calls[load_at + 1 :])


def test_an_error_after_a_load_still_unloads(tmp_path: Path):
    calls = []
    base = _recording_transport([])

    def transport(method, url, body, timeout):
        calls.append(body)
        if isinstance(body, dict) and body.get("raw") is True:
            raise RecordError("boom")
        return base(method, url, body, timeout)

    with pytest.raises(RecordError, match="boom"):
        _record(tmp_path, transport=transport, card_mib=lambda: 100, sleep=lambda _seconds: None)
    assert isinstance(calls[-1], dict) and calls[-1].get("keep_alive") == 0


def _is_companion_stop(body) -> bool:
    return isinstance(body, dict) and body.get("companion") == "stop"


def _is_unload(body) -> bool:
    return (
        isinstance(body, dict)
        and body.get("keep_alive") == 0
        and "companion" not in body
        and "options" not in body
    )


def test_a_companion_stop_that_loads_waits_for_the_card_to_fall(tmp_path: Path):
    seen = {"stop": False}
    highs = {"n": 0}
    sleeps: list[float] = []
    calls = []
    base = _recording_transport([])

    def card():
        if not seen["stop"]:
            return 100
        highs["n"] += 1
        return 11908 if highs["n"] <= 3 else 100

    def transport(method, url, body, timeout):
        calls.append(body)
        if _is_companion_stop(body):
            seen["stop"] = True
        return base(method, url, body, timeout)

    written = _record(tmp_path, transport=transport, card_mib=card, sleep=lambda seconds: sleeps.append(seconds))
    assert len(written) == len(CASES)
    assert highs["n"] >= 4
    assert sleeps == [SETTLE_STEP_S, SETTLE_STEP_S, SETTLE_STEP_S]
    stop_at = next(index for index, body in enumerate(calls) if _is_companion_stop(body))
    assert any(_is_unload(body) for body in calls[stop_at + 1 :])


def test_a_companion_stop_that_loads_still_unloads_when_the_card_never_falls(tmp_path: Path, capsys):
    seen = {"stop": False}
    calls = []
    base = _recording_transport([])

    def transport(method, url, body, timeout):
        calls.append(body)
        if _is_companion_stop(body):
            seen["stop"] = True
        return base(method, url, body, timeout)

    with pytest.raises(RecordError, match="above 4096"):
        _record(
            tmp_path,
            transport=transport,
            card_mib=lambda: 11908 if seen["stop"] else 100,
            sleep=lambda _seconds: None,
        )
    stop_at = next(index for index, body in enumerate(calls) if _is_companion_stop(body))
    assert any(_is_unload(body) for body in calls[stop_at + 1 :])
    err = capsys.readouterr().err
    stop_line = err.find("companion=stop")
    assert stop_line != -1
    assert "POST /api/generate" in err
    assert "keep_alive=0" in err[stop_line:]
