"""The call counter and the ordered call log (``--trace``). No device and no Relay: Relay's sink has its own tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from interlock.calltrace import (
    TraceError,
    Tracer,
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
        ("/api/embed", {"model": "m", "input": "x"}, "embed"),
        ("/api/embeddings", {"model": "m", "prompt": "x"}, "embed"),
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
    text = (tmp_path / "logs" / "r.events.jsonl").read_text(encoding="utf-8")
    lines = [json.loads(line) for line in text.splitlines()]
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


def test_a_timeout_with_no_message_is_still_a_timeout(tmp_path: Path):
    def transport(method, url, body, **kwargs):
        raise TimeoutError()

    tracer = Tracer(root=tmp_path, run_id="r")
    with tracer:
        with pytest.raises(TimeoutError):
            traced_transport(transport, tracer)("POST", URL + "/api/tags", None)
    assert tracer.counts_for("_run")["tags"]["timeouts"] == 1


def test_the_request_size_is_measured_only_for_relay(tmp_path: Path):
    seen = []

    class Sink:
        scope_id = None

        def __init__(self, *args):
            pass

        def start(self, record):
            seen.append(record.request_bytes)

        def end(self, record):
            pass

        def mark(self, *args):
            pass

        def flush(self):
            pass

        def close(self, end_ns=None):
            pass

    body = {"model": "m", "prompt": "x" * 100}
    for mode in ("events", "relay"):
        tracer = Tracer(root=tmp_path, run_id=mode, mode=mode, relay_factory=Sink)
        with tracer:
            traced_transport(lambda *a, **k: {}, tracer)("POST", URL + "/api/generate", body)
    assert seen == [len(json.dumps(body))]


def test_an_openvino_worker_fault_is_an_exception_not_ok(tmp_path: Path):
    from interlock.adapters import openvino

    def boom():
        raise RuntimeError("device said no")

    tracer = Tracer(root=tmp_path, run_id="ov")
    openvino.set_tracer(tracer)
    try:
        with tracer:
            with pytest.raises(RuntimeError):
                openvino._timed_call(boom, 1.0, kind="embed", model="m", device="NPU")
            status, _ = openvino._timed_call(lambda: 1, 1.0, kind="embed", model="m", device="NPU")
            assert status == "ok"
    finally:
        openvino.set_tracer(None)
    counts = tracer.counts_for("_run")["embed"]
    assert (counts["n"], counts["errors"], counts["timeouts"]) == (2, 1, 0)


def test_the_traced_transport_keeps_the_transports_own_methods(tmp_path: Path):
    class Transport:
        cancelled = 0

        def __call__(self, method, url, body=None, **kwargs):
            return {}

        def cancel(self):
            self.cancelled += 1

    inner = Transport()
    wrapped = traced_transport(inner, Tracer(root=tmp_path, run_id="r"))
    wrapped.cancel()
    assert inner.cancelled == 1
    assert not hasattr(traced_transport(lambda *a: {}, Tracer(root=tmp_path, run_id="s")), "cancel")


def test_after_the_close_a_call_is_neither_counted_nor_logged(tmp_path: Path):
    tracer = Tracer(root=tmp_path, run_id="r", mode="events")
    tracer.open()
    in_flight = tracer.start("generate", "/api/generate", model="m", device="5090")
    tracer.close()  # the watcher's close before os._exit
    late = tracer.start("generate", "/api/generate", model="m", device="5090")  # a thread still calling in
    tracer.end(late, "ok")
    tracer.end(in_flight, "ok")
    tracer.mark("late")
    tracer.flush()
    tracer.close()  # the run's own close afterwards does nothing
    lines = [json.loads(x) for x in (tmp_path / "logs" / "r.events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [x["status"] for x in lines] == ["abandoned"]
    assert tracer.counts_for("_run")["generate"] == {"n": 1, "ms": tracer.counts_for("_run")["generate"]["ms"],
                                                     "errors": 1, "timeouts": 0}


def test_a_close_racing_starts_leaves_every_logged_start_with_its_end(tmp_path: Path):
    import threading

    class Sink:
        scope_id = None

        def __init__(self, *args):
            self.events = []
            self.closed = False

        def start(self, record):
            assert not self.closed
            self.events.append(("start", record.call_id))

        def end(self, record):
            assert not self.closed
            self.events.append(("end", record.call_id))

        def mark(self, *args):
            pass

        def flush(self):
            pass

        def close(self, end_ns=None):
            self.closed = True

    tracer = Tracer(root=tmp_path, run_id="race", mode="relay", relay_factory=Sink)
    tracer.open()
    sink = tracer._sink
    stop = threading.Event()

    def caller():
        while not stop.is_set():
            record = tracer.start("generate", "/api/generate", model="m", device="5090")
            tracer.end(record, "ok")

    workers = [threading.Thread(target=caller) for _ in range(3)]
    for w in workers:
        w.start()
    tracer.close()
    stop.set()
    for w in workers:
        w.join()
    starts = {i for kind, i in sink.events if kind == "start"}
    ends = {i for kind, i in sink.events if kind == "end"}
    assert starts == ends


def test_release_unwraps_the_transport_so_a_reused_adapter_counts_in_its_next_run(tmp_path: Path):
    adapter = _Ollama()
    inner = adapter.transport
    first, second = Tracer(root=tmp_path, run_id="a"), Tracer(root=tmp_path, run_id="b")
    for tracer in (first, second):
        with tracer:
            instrument({"ollama": adapter}, tracer)
            adapter.transport("GET", URL + "/api/tags", None)
            release({"ollama": adapter})
        assert adapter.transport is inner
    assert first.counts_for("_run")["tags"]["n"] == second.counts_for("_run")["tags"]["n"] == 1
