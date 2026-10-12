"""Recorded Ollama exchanges, replayed by fakes. Nothing here starts a server model."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from switchyard.adapters.base import AdapterError
from switchyard.adapters.ollama import COUNT_TIMEOUT_NOTE, OllamaAdapter
from switchyard.adapters.ollama_cases import (
    CASES,
    MODEL,
    RecordedNoReply,
    case_by_name,
    covered_parameters,
    fixture_document,
    sent_parameters,
)
from switchyard.adapters.record_fixtures import (
    RecordError,
    http_exchange,
    read_card_mib,
    record_fixtures,
)
from tests.ollama_replay import ReplayTransport, replay

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "ollama"


def _document(case):
    if case.behaviour == "no_reply":
        response = None
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
                    assert document["response"] is None
                else:
                    replay(document)


def test_every_ollama_option_the_adapter_can_send_has_a_fixture():
    missing = sent_parameters() - covered_parameters()
    assert missing == frozenset()


def test_the_recorder_command_refuses_a_cloud_model(capsys):
    import importlib.util

    path = ROOT / "scripts" / "record_ollama_fixtures.py"
    spec = importlib.util.spec_from_file_location("record_ollama_fixtures_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    code = module.main(
        ["--model", "qwen3:8b:cloud"],
        transport=lambda *_args: {},
        card_mib=lambda: 100,
        today="2026-10-12",
    )
    assert code == 2
    assert "cloud" in capsys.readouterr().err


def test_the_recorder_refuses_a_cloud_model(tmp_path: Path):
    called = []

    def transport(*_args):
        called.append("called")
        return {}

    with pytest.raises(RecordError, match="cloud"):
        record_fixtures(
            model="qwen3:8b:cloud",
            host="http://127.0.0.1:11434",
            out_dir=tmp_path,
            timeout=15,
            ceiling=4096,
            today="2026-10-12",
            transport=transport,
            card_mib=lambda: 100,
        )
    assert called == []


def test_the_recorder_refuses_a_card_that_is_not_at_baseline(tmp_path: Path):
    def loaded(method, url, _body, _timeout):
        if url.endswith("/api/ps"):
            return {"models": [{"name": "qwen3:8b"}]}
        return {}

    with pytest.raises(RecordError, match="not at baseline"):
        record_fixtures(
            model="qwen3:8b",
            host="http://127.0.0.1:11434",
            out_dir=tmp_path,
            timeout=15,
            ceiling=4096,
            today="2026-10-12",
            transport=loaded,
            card_mib=lambda: 100,
        )
    with pytest.raises(RecordError, match="above 4096"):
        record_fixtures(
            model="qwen3:8b",
            host="http://127.0.0.1:11434",
            out_dir=tmp_path,
            timeout=15,
            ceiling=4096,
            today="2026-10-12",
            transport=lambda *_args: {"models": []},
            card_mib=lambda: 9000,
        )


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
            return None
        if isinstance(options, dict) and options.get("num_ctx") == 8:
            return {"error": "the prompt exceeds the context length"}
        return {"done": True, "prompt_eval_count": 1}

    return transport


def test_num_predict_0_is_recorded_as_no_reply_within_the_timeout(tmp_path: Path):
    timeouts: list = []
    written = record_fixtures(
        model="qwen3:8b",
        host="http://127.0.0.1:11434",
        out_dir=tmp_path,
        timeout=15,
        ceiling=4096,
        today="2026-10-12",
        transport=_recording_transport(timeouts),
        card_mib=lambda: 100,
    )
    assert len(written) == len(CASES)
    assert set(timeouts) == {15}
    saved = json.loads((tmp_path / "0.35.1" / "num_predict_0.json").read_text(encoding="utf-8"))
    assert saved["behaviour"] == "no_reply"
    assert saved["response"] is None
    assert saved["ollama_version"] == "0.35.1"
    assert saved["digest"] == "sha256:abc"
    assert saved["date"] == "2026-10-12"
    assert saved["request"]["body"]["options"]["num_predict"] == 0
    for name in ("num_predict_null", "num_predict_missing"):
        document = json.loads((tmp_path / "0.35.1" / f"{name}.json").read_text(encoding="utf-8"))
        assert document["behaviour"] == "no_reply"
        assert document["response"] is None
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
