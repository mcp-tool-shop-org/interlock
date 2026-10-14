"""The call counter and the ordered call log (``--trace``). No device and no Relay: Relay's sink has its own tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from interlock.calltrace import (
    Tracer,
    TraceError,
    classify,
    instrument,
    release,
    traced_transport,
)

URL = "http://127.0.0.1:11434"


@pytest.mark.parametrize(
    ("path", "body", "kind"),
    [
        ("/api/tags", None, "tags"),
        ("/api/ps", None, "ps"),
        ("/api/show", {"model": "m"}, "show"),
        ("/api/generate", {"model": "m", "prompt": "hi"}, "generate"),
        ("/api/generate", {"model": "m", "keep_alive": 0, "prompt": ""}, "unload"),
        ("/api/generate", {"model": "m", "raw": True, "prompt": "x"}, "count"),
        ("/api/generate", {"model": "m", "companion": "start"}, "companion"),
        ("/api/generate", None, "other"),
        ("/api/version", None, "other"),
    ],
)
def test_a_request_is_classified_from_its_endpoint_and_body(path, body, kind):
    assert classify("POST", URL + path, body) == kind


def test_the_counter_is_on_with_tracing_off_and_writes_nothing(tmp_path: Path):
    tracer = Tracer(root=tmp_path, run_id="r", mode="off")
    with tracer:
        tracer.begin_slot(arm="A", index="measure:0:A")
        with tracer.call("generate", "/api/generate", model="m", device="5090") as box:
            box["status"] = "ok"
            box["payload"] = {"prompt_eval_count": 12, "eval_count": 7}
        ids = tracer.end_slot()["call_ids"]
    assert ids == [1]
    art = tracer.artifacts("A")
    assert art["call_counts"]["generate"]["n"] == 1
    assert art["call_counts"]["generate"]["errors"] == 0
    assert "trace" not in art
    assert not (tmp_path / "logs").exists()


def test_the_events_log_writes_one_parsable_line_per_call_and_marks(tmp_path: Path):
    tracer = Tracer(root=tmp_path, run_id="r", mode="events")
    with tracer:
        tracer.begin_slot(arm="A", index="measure:0:A")
        record = tracer.start("count", "/api/generate", model="m", device="5090", request_bytes=40)
        tracer.end(record, "timeout")
        record = tracer.start("generate", "/api/generate", model="m", device="5090")
        tracer.end(record, "ok", {"prompt_eval_count": 3, "eval_count": 2})
        tracer.end_slot()
        tracer.mark("stop_file", reason="test")
        tracer.flush()
    lines = [json.loads(line) for line in (tmp_path / "logs" / "r.events.jsonl").read_text(encoding="utf-8").splitlines()]
    calls = [line for line in lines if "call_id" in line]
    assert [c["kind"] for c in calls] == ["count", "generate"]
    assert calls[0]["status"] == "timeout"
    assert calls[1]["completion_tokens"] == 2
    assert calls[0]["arm"] == "A"
    assert any(line.get("mark") == "stop_file" for line in lines)
    art = tracer.artifacts("A")
    assert art["trace"] == {"file": "logs/r.events.jsonl"}
    assert art["call_counts"]["count"]["timeouts"] == 1


def test_a_call_left_open_is_closed_as_abandoned(tmp_path: Path):
    tracer = Tracer(root=tmp_path, run_id="r", mode="events")
    with tracer:
        tracer.start("generate", "/api/generate", model="m", device="CPU")
    line = json.loads((tmp_path / "logs" / "r.events.jsonl").read_text(encoding="utf-8"))
    assert line["status"] == "abandoned"
    assert tracer.artifacts("_run")["call_counts"]["generate"]["errors"] == 1


def test_a_call_ended_twice_is_counted_once(tmp_path: Path):
    tracer = Tracer(root=tmp_path, run_id="r")
    with tracer:
        record = tracer.start("tags", "/api/tags", model=None, device=None)
        tracer.end(record, "ok")
        tracer.end(record, "ok")
    assert tracer.counts_for("_run")["tags"]["n"] == 1


def test_an_otel_variable_refuses_tracing(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://example.invalid")
    with pytest.raises(TraceError, match="OTEL_"):
        Tracer(root=tmp_path, run_id="r", mode="events").open()
    Tracer(root=tmp_path, run_id="r", mode="off").open().close()  # the counter alone is allowed


def test_an_unknown_mode_is_refused(tmp_path: Path):
    with pytest.raises(TraceError, match="unknown trace mode"):
        Tracer(root=tmp_path, run_id="r", mode="loud")


def test_the_relay_mode_uses_the_factory_it_is_given(tmp_path: Path):
    made = []

    class Sink:
        scope_id = "scope-1"

        def __init__(self, logs, name, start_ns, run_id):
            made.append((logs, name, run_id))
            self.events = []

        def start(self, record):
            self.events.append(("start", record.call_id))

        def end(self, record):
            self.events.append(("end", record.call_id))

        def mark(self, name, t_ns, data):
            self.events.append(("mark", name))

        def flush(self):
            self.events.append(("flush",))

        def close(self, end_ns=None):
            self.events.append(("close",))

    tracer = Tracer(root=tmp_path, run_id="r", mode="relay", relay_factory=Sink)
    with tracer:
        sink = tracer._sink
        with tracer.call("generate", "/api/generate", model="m", device="5090") as box:
            box["status"] = "ok"
        tracer.mark("grant")
        tracer.flush()
    assert made == [(tmp_path / "logs", "r.atof.jsonl", "r")]
    assert sink.events == [("start", 1), ("end", 1), ("mark", "grant"), ("flush",), ("close",)]
    assert tracer.artifacts("_run")["trace"] == {"file": "logs/r.atof.jsonl", "span": "scope-1"}


def test_the_traced_transport_records_ok_timeout_and_error(tmp_path: Path):
    def transport(method, url, body, **kwargs):
        if body and body.get("prompt") == "slow":
            raise TimeoutError("timed out")
        if body and body.get("prompt") == "bad":
            raise RuntimeError("HTTP 500")
        return {"eval_count": 4, "prompt_eval_count": 9}

    tracer = Tracer(root=tmp_path, run_id="r")
    with tracer:
        wrapped = traced_transport(transport, tracer, device="5090")
        assert wrapped.__wrapped__ is transport
        assert wrapped("POST", URL + "/api/generate", {"model": "m", "prompt": "ok"})["eval_count"] == 4
        with pytest.raises(TimeoutError):
            wrapped("POST", URL + "/api/generate", {"model": "m", "prompt": "slow"})
        with pytest.raises(RuntimeError):
            wrapped("POST", URL + "/api/generate", {"model": "m", "prompt": "bad"})
    counts = tracer.counts_for("_run")["generate"]
    assert (counts["n"], counts["timeouts"], counts["errors"]) == (3, 1, 1)


class _Ollama:
    def __init__(self):
        self.device = "5090"
        self.transport = lambda method, url, body=None: {"ok": True}


def test_instrument_wraps_ollama_once_and_sets_the_openvino_tracer(tmp_path: Path):
    from interlock.adapters import openvino

    tracer = Tracer(root=tmp_path, run_id="r")
    adapter = _Ollama()
    adapters = {"ollama": adapter, "openvino": object()}
    instrument(adapters, tracer)
    first = adapter.transport
    instrument(adapters, tracer)
    assert adapter.transport is first  # not wrapped twice
    assert openvino._TRACER is tracer
    release(adapters)
    assert openvino._TRACER is None
