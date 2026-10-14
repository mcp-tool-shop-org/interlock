"""Stage 0 of the interlock-relay design, on fakes: H1 (complete capture) and H3 (safe with interlock's rules).

The Ollama ground truth is the fake transport's own request log, below the hook. The Relay tests need the
``relay`` extra and skip without it; the native event log runs everywhere.
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import pytest

from interlock.adapters.ollama import OllamaAdapter
from interlock.calltrace import TraceError, Tracer, instrument, release

CANARY = "AMU_CANARY_PROMPT_7c1e"
MODES = ["events", pytest.param("relay", marks=pytest.mark.relay)]


def _need(mode):
    if mode == "relay":
        pytest.importorskip("nemo_relay")


class Recorder:
    """A fake Ollama transport that logs every request it receives: the ground truth for H1."""

    def __init__(self, timeouts=0):
        self.log = []
        self.timeouts = timeouts

    def __call__(self, method, url, body=None, **kwargs):
        self.log.append((url.rsplit("/api/", 1)[-1], dict(body or {})))
        if body and body.get("raw") and self.timeouts:
            self.timeouts -= 1
            raise TimeoutError("timed out")
        return {"response": CANARY + " reply", "prompt_eval_count": 9, "eval_count": 1}


def _trace_calls(tmp_path: Path, mode: str, run_id: str) -> list[dict]:
    if mode == "events":
        lines = (tmp_path / "logs" / f"{run_id}.events.jsonl").read_text(encoding="utf-8").splitlines()
        return [json.loads(x) for x in lines if "call_id" in json.loads(x)]
    calls = {}
    for line in (tmp_path / "logs" / f"{run_id}.atof.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("category") == "llm":
            calls.setdefault(event["uuid"], {}).update(event.get("metadata") or {})
            calls[event["uuid"]][event["scope_category"]] = True
    return list(calls.values())


@pytest.mark.parametrize("mode", MODES)
def test_h1_every_request_the_fake_received_is_in_the_trace_once(tmp_path: Path, mode):
    _need(mode)
    fake = Recorder(timeouts=1)
    adapter = OllamaAdapter(transport=fake)
    tracer = Tracer(root=tmp_path, run_id="h1", mode=mode)
    with tracer:
        instrument({"ollama": adapter}, tracer)
        tracer.begin_slot(arm="B", index="measure:0:B")
        with pytest.raises(TimeoutError):
            adapter.count_tokens("qwen3:8b", CANARY)
        assert adapter.count_tokens("qwen3:8b", CANARY) == 9
        adapter.transport("POST", "http://127.0.0.1:11434/api/generate", {"model": "qwen3:8b", "prompt": CANARY})
        adapter.unload("qwen3:8b")
        tracer.end_slot()
        release({"ollama": adapter})
    calls = _trace_calls(tmp_path, mode, "h1")
    assert len(calls) == len(fake.log) == 4
    assert [c["kind"] for c in sorted(calls, key=lambda c: c["call_id"])] == ["count", "count", "generate", "unload"]
    statuses = sorted(c["status"] for c in calls)
    assert statuses == ["ok", "ok", "ok", "timeout"]
    if mode == "relay":
        assert all(c.get("start") and c.get("end") for c in calls)  # no unclosed span


@pytest.mark.parametrize("mode", MODES)
def test_h3_the_trace_holds_no_prompt_host_user_or_home(tmp_path: Path, mode):
    _need(mode)
    adapter = OllamaAdapter(transport=Recorder())
    tracer = Tracer(root=tmp_path, run_id="h3", mode=mode)
    with tracer:
        instrument({"ollama": adapter}, tracer)
        adapter.transport("POST", "http://127.0.0.1:11434/api/generate", {"model": "m", "prompt": CANARY})
        tracer.mark("grant", grant="g")
    text = next((tmp_path / "logs").glob("h3.*.jsonl")).read_text(encoding="utf-8")
    assert CANARY not in text
    for needle in (socket.gethostname(), os.environ.get("USERNAME") or os.environ.get("USER"), str(Path.home())):
        if needle:
            assert needle.lower() not in text.lower()


@pytest.mark.parametrize("mode", MODES)
def test_h3_an_interrupted_run_leaves_a_closed_parsable_trace(tmp_path: Path, mode):
    _need(mode)
    tracer = Tracer(root=tmp_path, run_id="stop", mode=mode)
    with pytest.raises(KeyboardInterrupt):
        with tracer:
            done = tracer.start("generate", "/api/generate", model="m", device="5090")
            tracer.end(done, "ok")
            tracer.start("generate", "/api/generate", model="m", device="5090")  # in flight at the stop
            tracer.mark("stop_file")
            raise KeyboardInterrupt
    text = next((tmp_path / "logs").glob("stop.*.jsonl")).read_text(encoding="utf-8")
    lines = [json.loads(x) for x in text.splitlines()]
    if mode == "events":
        assert [x["status"] for x in lines if "call_id" in x] == ["ok", "abandoned"]
    else:
        llm = [x for x in lines if x.get("category") == "llm"]
        assert sum(x["scope_category"] == "start" for x in llm) == sum(x["scope_category"] == "end" for x in llm) == 2
        assert [x["scope_category"] for x in lines if x.get("category") == "agent"] == ["start", "end"]


@pytest.mark.parametrize("mode", MODES)
def test_h3_tracing_opens_no_socket(tmp_path: Path, mode, monkeypatch):
    _need(mode)
    opened = []
    real = socket.socket

    # On Windows, asyncio's event loop makes its self-pipe from a loopback socket pair (127.0.0.1, port 0). That is
    # Python's, not Relay's, and never leaves the machine. Anything else is refused.
    def loopback(address):
        return isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1")

    class Guard(real):
        def connect(self, address):
            opened.append(("connect", address))
            if not loopback(address):
                raise AssertionError(f"connect to {address}")
            return super().connect(address)

        def bind(self, address):
            opened.append(("bind", address))
            if not (loopback(address) and address[1] == 0):
                raise AssertionError(f"bind to {address}")
            return super().bind(address)

        def connect_ex(self, address):
            opened.append(("connect_ex", address))
            if not loopback(address):
                raise AssertionError(f"connect_ex to {address}")
            return super().connect_ex(address)

        def sendto(self, data, *args):
            address = args[-1]
            opened.append(("sendto", address))
            if not loopback(address):
                raise AssertionError(f"sendto {address}")
            return super().sendto(data, *args)

    monkeypatch.setattr(socket, "socket", Guard)
    tracer = Tracer(root=tmp_path, run_id="net", mode=mode)
    with tracer:
        record = tracer.start("tags", "/api/tags", model=None, device=None)
        tracer.end(record, "ok")
        tracer.flush()
    assert all(loopback(address) for _, address in opened)
    if mode == "events":
        assert opened == []


@pytest.mark.relay
def test_the_relay_sink_refuses_layered_plugin_configs(tmp_path: Path, monkeypatch):
    pytest.importorskip("nemo_relay")
    from nemo_relay import plugin

    from interlock import relay_sink

    monkeypatch.setattr(plugin, "validate", lambda config: {"config_paths": ["elsewhere/plugins.toml"]})
    with pytest.raises(TraceError, match="plugins.toml"):
        relay_sink.RelaySink(tmp_path, "x.atof.jsonl", 0, "r")


def test_otel_refuses_before_the_sink_opens(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OTEL_SERVICE_NAME", "x")
    with pytest.raises(TraceError):
        Tracer(root=tmp_path, run_id="o", mode="relay", relay_factory=lambda *a: pytest.fail("sink opened")).open()


@pytest.mark.parametrize("mode", ["off", *MODES])
def test_a_traced_run_writes_receipts_check_accepts(tmp_path: Path, mode):
    _need(mode if isinstance(mode, str) else "relay")
    from interlock.harness.runner import run_spec
    from interlock.receipt import check_tree, load_jsonl
    from tests.test_run_path import Fake, _arm, _bind, _result, _spec

    repo = tmp_path / "repo"
    spec_path = _bind(repo, _spec([_arm("A", "CPU")]))
    rows = run_spec(spec_path, root=repo, adapters={"fake": Fake({"A": [_result(10), _result(11), _result(12)]})},
                    date="2026-10-10", trace=mode)
    assert len(rows) == 1
    assert check_tree(repo) == []
    row = load_jsonl(repo / "receipts" / "t-2026-10-10.jsonl")[0]
    assert "call_counts" in row["artifacts"]
    if mode == "off":
        assert "trace" not in row["artifacts"]
    else:
        suffix = "atof" if mode == "relay" else "events"
        assert row["artifacts"]["trace"]["file"] == f"logs/t-2026-10-10.{suffix}.jsonl"
        assert (repo / row["artifacts"]["trace"]["file"]).is_file()


def test_trace_is_refused_where_it_is_not_wired(tmp_path: Path):
    from interlock.harness.runner import RunnerError, run_spec

    # E4b's calls go through its own seams. The refusal comes before any adapter, seam or run marker.
    spec = Path(__file__).resolve().parents[1] / "experiments" / "e4b" / "spec.yaml"
    with pytest.raises(RunnerError, match="not wired for E4b"):
        run_spec(spec, root=tmp_path, adapters={}, date="2026-10-10", trace="events")
    assert not (tmp_path / "receipts").exists()


@pytest.mark.parametrize("mode", MODES)
def test_h1_the_fill_card_path_traces_every_call_its_transport_received(tmp_path: Path, mode):
    _need(mode)
    from interlock.adapters.ollama import OllamaAdapter
    from interlock.harness.runner import run_spec
    from interlock.receipt import check_tree, load_jsonl
    from tests.test_fill_card_live import _bind, _fresh, _model, _seams, _spec, _Transport

    transport = _Transport(None)
    seams = _seams(transport)
    repo = tmp_path / "repo"
    spec_path = _bind(repo, _spec([_model()], max_ctx=8192, confirm_num_predict=1))
    rows = run_spec(spec_path, root=repo, grant="g", adapters={"ollama": OllamaAdapter(transport=transport)},
                    heartbeat=_fresh(tmp_path), date="2026-10-11", run_id="fill-1", seams=seams, trace=mode)
    assert rows and check_tree(repo) == []
    calls = _trace_calls(repo, mode, "fill-1")
    assert len(calls) == len(transport.calls) > 0
    row = load_jsonl(repo / "receipts" / "fill-1.jsonl")[0]
    counts = row["artifacts"]["call_counts"]
    # The row counts its own slot. Run-level calls, such as the card-empty check before the model, are not its.
    in_slot = [c for c in calls if (c.get("slot") or "").startswith("fill:")]
    assert sum(v["n"] for v in counts.values()) == len(in_slot) > 0
    assert all(c["kind"] in ("ps", "tags") for c in calls if c not in in_slot)
    assert row["artifacts"]["trace"]["file"].startswith("logs/fill-1.")



@pytest.mark.relay
def test_relay_keeps_every_span_whole_under_two_threads(tmp_path: Path):
    pytest.importorskip("nemo_relay")
    import threading

    tracer = Tracer(root=tmp_path, run_id="threads", mode="relay")

    def loop(kind):
        for _ in range(50):
            record = tracer.start(kind, "/api/generate", model="m", device="5090")
            tracer.end(record, "ok")

    with tracer:
        tracer.begin_slot(arm="A", index="measure:0:A")
        workers = [threading.Thread(target=loop, args=(k,)) for k in ("generate", "companion")]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        tracer.end_slot()
    text = (tmp_path / "logs" / "threads.atof.jsonl").read_text(encoding="utf-8")
    events = [json.loads(x) for x in text.splitlines()]
    llm = [e for e in events if e.get("category") == "llm"]
    starts = {e["uuid"] for e in llm if e["scope_category"] == "start"}
    ends = {e["uuid"] for e in llm if e["scope_category"] == "end"}
    assert len(starts) == len(ends) == 100 and starts == ends
    ids = {e["metadata"]["call_id"] for e in llm if e["scope_category"] == "end"}
    assert ids == set(range(1, 101))


@pytest.mark.relay
def test_a_second_live_relay_sink_is_refused_and_a_failed_start_releases(tmp_path: Path, monkeypatch):
    pytest.importorskip("nemo_relay")
    from nemo_relay import scope

    from interlock.relay_sink import RelaySink

    first = RelaySink(tmp_path, "a.atof.jsonl", 0, "a")
    try:
        with pytest.raises(TraceError, match="already open"):
            RelaySink(tmp_path, "b.atof.jsonl", 0, "b")
    finally:
        first.close()

    def broken(*args, **kwargs):
        raise RuntimeError("push failed")

    with monkeypatch.context() as patch:
        patch.setattr(scope, "push", broken)
        with pytest.raises(RuntimeError, match="push failed"):
            RelaySink(tmp_path, "c.atof.jsonl", 0, "c")
    # Nothing leaked: the lock, the sanitizers and the activation were all released.
    RelaySink(tmp_path, "d.atof.jsonl", 0, "d").close()


def test_a_failed_slot_still_ends_its_tracer_slot(tmp_path: Path, monkeypatch):
    from interlock.adapters.base import AdapterError
    from interlock.calltrace import Tracer as T
    from interlock.harness.runner import RunnerError, run_spec
    from tests.test_run_path import Fake, _arm, _bind, _result, _spec

    begun, ended = [], []
    real_begin, real_end = T.begin_slot, T.end_slot
    monkeypatch.setattr(T, "begin_slot", lambda self, **s: (begun.append(s), real_begin(self, **s))[1])
    monkeypatch.setattr(T, "end_slot", lambda self: (ended.append(1), real_end(self))[1])
    repo = tmp_path / "repo"
    spec_path = _bind(repo, _spec([_arm("A", "CPU"), _arm("B", "CPU")]))
    adapter = Fake({"A": [_result(10), _result(11), _result(12)], "B": [AdapterError("hang: no result within 60 s")]})
    with pytest.raises(RunnerError, match="hang"):
        run_spec(spec_path, root=repo, adapters={"fake": adapter}, date="2026-10-10", run_id="slot-1", trace="events")
    assert begun and len(ended) == len(begun)


@pytest.mark.parametrize("mode", MODES)
def test_h3_the_real_stop_file_leaves_a_closed_parsable_trace(tmp_path: Path, mode):
    """interlock's operator stop: the watcher writes the receipt, then ends the process with os._exit. The halt
    must close the trace first, with the call in flight closed as abandoned."""

    _need(mode)
    import threading

    from interlock.adapters.base import AdapterError
    from tests.test_rest_removed_stops import StopAdapter
    from tests.test_run_guard import Clock, Exits, _hooks, _invoke, _spec

    repo, spec_path = _spec(tmp_path)
    clock, gate = Clock(), threading.Event()
    exits = Exits(repo / "receipts" / "run-1.jsonl", gate)
    hooks, _ram = _hooks(clock, exits)
    tracer = Tracer(root=repo, run_id="run-1", mode=mode)

    class Traced(StopAdapter):
        def run(self, request):
            record = tracer.start("generate", "/api/generate", model="m", device=request.step_id)
            if request.step_id != "npu":
                tracer.end(record, "ok")
            return super().run(request)  # the npu call stays in flight until the watcher ends the run

    adapter = Traced(clock, gate, compile_spent_s=1, in_compile=False, stop_file=repo / "receipts" / "run-1.stop",
                     text="stop for the test")
    with tracer:
        with pytest.raises(AdapterError, match="released"):
            _invoke(repo, spec_path, adapter, hooks, tracer=tracer)
    assert exits.codes == [77]
    text = next((repo / "logs").glob("run-1.*.jsonl")).read_text(encoding="utf-8")
    lines = [json.loads(x) for x in text.splitlines()]
    if mode == "events":
        calls = [x for x in lines if "call_id" in x]
        assert [c["status"] for c in calls] == ["ok", "abandoned"]
        assert any(x.get("mark") == "halt" for x in lines)
    else:
        llm = [x for x in lines if x.get("category") == "llm"]
        assert sum(x["scope_category"] == "start" for x in llm) == sum(x["scope_category"] == "end" for x in llm) == 2
        assert [x["scope_category"] for x in lines if x.get("category") == "agent"] == ["start", "end"]
        assert any(x.get("kind") == "mark" and x.get("name") == "halt" for x in lines)
