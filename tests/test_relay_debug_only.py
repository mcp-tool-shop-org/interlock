"""Design amendment v3.2 (2026-10-14): NeMo Relay as a debug-only tracer.

Stage 1 on the 5090 failed H1 by exactly one request per step (the card guard's /api/ps ran before the tracer was
attached) and showed three companion spans outside any slot (the companion's stop ran after end_slot). These tests
drive the real Ollama adapter, its real ``http.client`` transport and the runner's 5090 guard against an in-process
fake Ollama on a loopback port, and check H1 and H3 deterministically: no device, no sleep. They also check the
ruling: a traced run is a debugging run, so none of its rows routes.
"""

from __future__ import annotations

import getpass
import http.client
import json
import socket
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from interlock.adapters.base import StepResult
from interlock.adapters.ollama import OllamaAdapter
from interlock.calltrace import Tracer, instrument, release, traced_transport
from interlock.guards import GuardError
from interlock.harness import runner as runner_module
from interlock.harness.receipts import import_receipt, write_run
from interlock.harness.runner import run_spec
from interlock.harness.watch import WatchHooks
from interlock.receipt import check_tree, is_speed_row, load_jsonl
from tests.test_relay_stage1 import MODES, _need
from tests.test_relay_stage1 import _run as _traced_cpu_run
from tests.test_run_path import Fake, _bind
from tests.test_run_path import _arm as _plain_arm
from tests.test_run_path import _spec as _plain_spec

CANARY = "AMU_CANARY_PROMPT_7c1e"
GENERATE_REPLY = {
    "response": "ok", "done": True, "total_duration": 450_000_000, "eval_count": 64, "eval_duration": 320_000_000,
    "prompt_eval_count": 256, "prompt_eval_duration": 30_000_000, "load_duration": 1_000_000,
}


class FakeOllama:
    """An Ollama on a loopback port: ps, tags, show, generate, a companion's start and an unload, over real HTTP.
    Every request it serves is kept, which is H1's ground truth from the server's side."""

    def __init__(self):
        self.served: list[tuple[str, dict]] = []
        served, lock = self.served, threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _reply(self, payload: dict) -> None:
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:  # noqa: N802  (http.server's name)
                self.close_connection = True
                path = urlsplit(self.path).path
                with lock:
                    served.append((path, {}))
                if path.endswith("/api/ps"):
                    self._reply({"models": []})
                elif path.endswith("/api/tags"):
                    self._reply({"models": [{"name": "qwen3:8b", "size": 5_000_000_000}]})
                else:
                    self._reply({})

            def do_POST(self) -> None:  # noqa: N802
                self.close_connection = True
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                path = urlsplit(self.path).path
                with lock:
                    served.append((path, body))
                if path.endswith("/api/show"):
                    self._reply({"model_info": {"qwen3.context_length": 40960},
                                 "details": {"quantization_level": "Q4_K_M"}})
                elif body.get("keep_alive") == 0 and body.get("prompt", "") == "":
                    self._reply({"done": True, "done_reason": "unload"})
                elif body.get("companion") == "start":
                    self._reply({"done": True, "done_reason": "load"})
                else:
                    self._reply(GENERATE_REPLY)

            def log_message(self, *args) -> None:
                return None

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, name="fake-ollama", daemon=True)

    def __enter__(self) -> FakeOllama:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class RequestCounter:
    """Every ``http.client`` request sent from the adapter's module, by path: the count below the hook, taken the way
    the stage 1 driver took it, over the whole of run_spec (the card guard included)."""

    def __init__(self, monkeypatch):
        self.by_path: dict[str, int] = {}
        counts, lock, original = self.by_path, threading.Lock(), http.client.HTTPConnection.request

        def counted(conn, method, url, *args, **kwargs):
            frames = traceback.extract_stack()
            if any(frame.filename.replace("\\", "/").endswith("interlock/adapters/ollama.py") for frame in frames):
                path = urlsplit(url).path
                with lock:
                    counts[path] = counts.get(path, 0) + 1
            return original(conn, method, url, *args, **kwargs)

        monkeypatch.setattr(http.client.HTTPConnection, "request", counted)


def _ollama_arm(arm_id: str, *, trace: bool, companion: str | None = None, model: str = "qwen3:8b") -> dict:
    arm = {
        "id": arm_id, "task": "generate", "device": "5090", "engine": "ollama", "model": model,
        "revision_or_digest": "abc", "quant": "unrecorded", "licence": "unrecorded", "engine_version": "0.35.1",
        "thinking_level": "off", "num_predict": 8, "num_ctx": 4096, "batch": None,
        "prompt": f"The switchyard counts the same cars on every pass. {CANARY}", "measure": "decode_tok_s",
        "trace": trace,
    }
    if companion is not None:
        arm["companion"] = companion
        arm["companion_device"] = "5090"
    return arm


def _companion_spec() -> dict:
    """Stage 1's companion spec in miniature: solo, the same arm paired with a companion traced and untraced, and
    the companion itself. Per-arm tracing, on the 5090, so the card guard runs."""

    return {
        "id": "RELAY-V32", "procedure": "interleaved", "order": "aba",
        "question": "Is every request counted, with the card guard and the companion's stop inside the window?",
        "decision": "A test verdict, computed after the run.",
        "needs": {"grant": True, "npu_session": False, "cpu_window": False},
        "stop_rules": ["a failed step stops the sequence"], "dataset": {"name": "fixture"},
        "metrics": ["decode_tok_s", "latency_ms"], "warmups": 0, "repeats": 2, "band": 0.1,
        "compile_cache": "reuse", "rest_s": 0,
        "arms": [
            _ollama_arm("solo", trace=False),
            _ollama_arm("with-companion-on", trace=True, companion="companion"),
            _ollama_arm("with-companion-off", trace=False, companion="companion"),
            _ollama_arm("companion", trace=False, model="llama3.1:8b"),
        ],
        "pairs": [
            {"id": "companion-traced", "alone": "solo", "paired": "with-companion-on"},
            {"id": "companion-untraced", "alone": "solo", "paired": "with-companion-off"},
        ],
    }


def _run_5090(tmp_path: Path, monkeypatch, mode: str, spec: dict, run_id: str):
    repo = tmp_path / "repo"
    spec_path = _bind(repo, spec)
    (repo / "table").mkdir()
    heartbeat = repo / "hb"
    heartbeat.write_text("pulse", encoding="utf-8")
    monkeypatch.setattr(runner_module, "psutil", None)  # no 0.2 s CPU samples: the fake answers in microseconds
    with FakeOllama() as server:
        monkeypatch.setattr("interlock.adapters.ollama.URL_5090", server.url)
        counter = RequestCounter(monkeypatch)
        rows = run_spec(spec_path, root=repo, grant="g", run_id=run_id, date="2026-10-14", heartbeat=heartbeat,
                        adapters={"ollama": OllamaAdapter(device="5090")}, trace=mode,
                        watch=WatchHooks(ram_sampler=None, start_thread=False, sleep=lambda s: None))
    return repo, rows, server, counter


def _spans(repo: Path, run_id: str, mode: str):
    """The trace's call ids as started and ended, each call's kind, its end record, and the raw text."""

    name = f"{run_id}.{'events' if mode == 'events' else 'atof'}.jsonl"
    text = (repo / "logs" / name).read_text(encoding="utf-8")
    lines = [json.loads(x) for x in text.splitlines() if x.strip()]
    if mode == "events":
        calls = [x for x in lines if "call_id" in x]
        ids = [x["call_id"] for x in calls]
        return ids, ids, {x["call_id"]: x["kind"] for x in calls}, {x["call_id"]: x for x in calls}, text
    llm = [x for x in lines if x.get("category") == "llm"]
    starts = [x["metadata"]["call_id"] for x in llm if x["scope_category"] == "start"]
    ends = [x["metadata"]["call_id"] for x in llm if x["scope_category"] == "end"]
    kinds = {x["metadata"]["call_id"]: x["metadata"]["kind"] for x in llm if x["scope_category"] == "start"}
    end_records = {x["metadata"]["call_id"]: x["metadata"] for x in llm if x["scope_category"] == "end"}
    return starts, ends, kinds, end_records, text


def _by_kind(table: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for kinds in table.values():
        for kind, entry in kinds.items():
            out[kind] = out.get(kind, 0) + entry["n"]
    return out


@pytest.mark.parametrize("mode", MODES)
def test_every_request_is_counted_with_the_guard_and_the_companion_stop_inside_the_window(tmp_path, monkeypatch, mode):
    _need(mode)
    repo, rows, server, counter = _run_5090(tmp_path, monkeypatch, mode, _companion_spec(), "v32")
    assert check_tree(repo) == []
    table = json.loads((repo / "logs" / "v32.calls.json").read_text(encoding="utf-8"))["counts"]
    by_kind = _by_kind(table)

    # H1 (a): the count below the hook equals the call table, path by path. The card guard's /api/ps is the one
    # request stage 1 missed; it is now counted, in _run, and so are the companion's starts between slots.
    assert table["_run"]["ps"]["n"] == 1 == counter.by_path["/api/ps"] == by_kind["ps"]
    assert table["_run"]["companion"]["n"] == 4
    assert counter.by_path["/api/show"] == by_kind["show"]
    assert counter.by_path["/api/generate"] == sum(by_kind.get(k, 0) for k in ("generate", "unload", "companion"))
    assert sum(counter.by_path.values()) == sum(by_kind.values()) == len(server.served)

    # H1 (b): the companion's stop is now the slot's last call, in both arms alike: show, generate, unload.
    slots = {row["id"].split("/")[1]: row["artifacts"]["slots"] for row in rows}
    for arm_id in ("with-companion-on", "with-companion-off"):
        paired = [s for s in slots[arm_id] if s["role"] == "paired"]
        assert len(paired) == 2 and all(len(s["call_ids"]) == 3 for s in paired)
        assert table[arm_id]["unload"]["n"] == 2  # attributed to the arm, not to _run as in stage 1
    assert all(len(s["call_ids"]) == 2 for s in slots["solo"])
    assert all(s["traced"] is True for s in slots["with-companion-on"])
    assert all(s["traced"] is False for arm_id in ("solo", "with-companion-off") for s in slots[arm_id])

    # H1 (b) and (c): the spans are exactly the traced slots' calls, each started once and ended once.
    starts, ends, kinds, end_records, text = _spans(repo, "v32", mode)
    traced_ids = [i for s in slots["with-companion-on"] for i in s["call_ids"]]
    assert sorted(starts) == sorted(ends) == sorted(traced_ids) and len(set(starts)) == len(starts)
    for s in slots["with-companion-on"]:
        if s["role"] == "paired":
            assert [kinds[i] for i in s["call_ids"]] == ["show", "generate", "unload"]
    # The request size is the transport's own byte count, on every span's end (amendment v3.2, item 3).
    assert all(isinstance(end_records[i]["request_bytes"], int) and end_records[i]["request_bytes"] > 0
               for i in traced_ids)

    # H3: no prompt text, home path, host or user in the trace or the call table.
    calls_text = (repo / "logs" / "v32.calls.json").read_text(encoding="utf-8")
    home = str(Path.home())
    for needle in (CANARY, home, json.dumps(home)[1:-1]):
        assert needle not in text and needle not in calls_text
    host, user = socket.gethostname(), getpass.getuser()
    if len(host) >= 4:
        assert host not in text
    if len(user) >= 4:
        assert user not in text

    # Debug only: every row of a traced run is descriptive, never a speed, and never in the table.
    receipt = load_jsonl(repo / "receipts" / "v32.jsonl")
    assert {row["id"].split("/")[1] for row in receipt} == {"solo", "with-companion-on", "with-companion-off"}
    assert all(row["descriptive"] is True and is_speed_row(row) is False for row in receipt)
    table_path = repo / "table" / "routing.jsonl"
    assert not table_path.is_file() or load_jsonl(table_path) == []
    assert import_receipt(repo, "v32") == []
    assert not table_path.is_file() or load_jsonl(table_path) == []
    assert not (repo / "table" / "gaps.md").exists()


def test_run_wide_tracing_emits_the_guard_and_the_final_unload_into_run(tmp_path, monkeypatch):
    """Without per-arm tracing, calls outside a slot are traced too: the guard's ps first, with no arm."""

    arm = {**_plain_arm("A", "5090"), "engine": "ollama", "num_ctx": 4096}
    spec = _plain_spec([arm], needs={"grant": True, "npu_session": False, "cpu_window": False}, rest_s=0, repeats=1)
    repo, rows, server, counter = _run_5090(tmp_path, monkeypatch, "events", spec, "wide")
    starts, _ends, kinds, _records, _text = _spans(repo, "wide", "events")
    events = [json.loads(x) for x in (repo / "logs" / "wide.events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(starts) == len(server.served) == sum(counter.by_path.values())
    assert kinds[starts[0]] == "ps" and "arm" not in events[0]
    assert kinds[starts[-1]] == "unload" and "arm" not in events[-1]
    assert [kinds[i] for i in starts] == ["ps", "show", "generate", "show", "generate", "unload"]
    assert all(row["descriptive"] is True for row in rows)


def test_the_adapters_transport_sizes_the_request_once_and_a_bare_transport_leaves_it_unknown(tmp_path, monkeypatch):
    dumped: list[str] = []
    real_dumps = json.dumps

    def counting_dumps(obj, *args, **kwargs):
        out = real_dumps(obj, *args, **kwargs)
        dumped.append(out)
        return out

    with FakeOllama() as server:
        monkeypatch.setattr("interlock.adapters.ollama.URL_5090", server.url)
        adapter = OllamaAdapter(device="5090")
        monkeypatch.setattr(json, "dumps", counting_dumps)
        with Tracer(root=tmp_path, run_id="size", mode="events") as tracer:
            instrument({"ollama": adapter}, tracer)
            try:
                adapter.show("qwen3:8b")
                adapter.loaded_models()
            finally:
                release({"ollama": adapter})
    lines = [json.loads(x) for x in (tmp_path / "logs" / "size.events.jsonl").read_text(encoding="utf-8").splitlines()]
    body = '{"model": "qwen3:8b"}'
    assert [(x["kind"], x["request_bytes"]) for x in lines] == [("show", len(body)), ("ps", 0)]
    assert dumped.count(body) == 1  # serialised once, by the transport; the tracer never does it again

    with Tracer(root=tmp_path, run_id="bare", mode="events") as tracer:
        traced_transport(lambda *a, **k: {}, tracer)("POST", "http://127.0.0.1:1/api/show", {"model": "m"})
    line = json.loads((tmp_path / "logs" / "bare.events.jsonl").read_text(encoding="utf-8").strip())
    assert line["request_bytes"] is None


def test_a_traced_runs_rows_never_route_and_an_untraced_run_still_does(tmp_path):
    # A per-arm traced run on fakes (stage 1's own test harness): every row is descriptive, traced arm or not.
    repo, rows, _server = _traced_cpu_run(tmp_path, "events")
    assert [row["descriptive"] for row in rows] == [True, True]
    (repo / "table").mkdir()
    assert import_receipt(repo, "s1") == []
    assert load_jsonl(repo / "table" / "routing.jsonl") == []
    # An aborted traced receipt is not a routing gap either.
    aborted = dict(load_jsonl(repo / "receipts" / "s1.jsonl")[0])
    aborted["id"] = aborted["id"].replace("/s1", "/s1-aborted")
    aborted["aborted"] = True
    write_run(repo, "s1-aborted", [aborted])
    assert import_receipt(repo, "s1-aborted") == []
    assert load_jsonl(repo / "table" / "routing.jsonl") == []
    assert not (repo / "table" / "gaps.md").exists()

    # The same path untraced is untouched: the row routes.
    plain = tmp_path / "plain"
    spec_path = _bind(plain, _plain_spec([_plain_arm("A", "CPU")], rest_s=0))
    (plain / "table").mkdir()
    fake = Fake({"A": [StepResult(ok=True, latency_ms=10) for _ in range(3)]})
    rows = run_spec(spec_path, root=plain, run_id="off", date="2026-10-14", adapters={"fake": fake}, trace="off",
                    watch=WatchHooks(ram_sampler=None, start_thread=False, sleep=lambda s: None))
    assert "descriptive" not in rows[0]
    assert [row["id"] for row in load_jsonl(plain / "table" / "routing.jsonl")] == ["T/A/off"]


def test_a_trace_log_left_by_a_refused_traced_attempt_refuses_the_name_to_a_later_run(tmp_path):
    """Argon's review: a traced run refused at the card guard leaves logs/<id>.atof.jsonl with no receipt. An untraced
    run under the same name would then be marked descriptive and kept out of the table, so the name is refused."""

    repo = tmp_path / "repo"
    spec_path = _bind(repo, _plain_spec([_plain_arm("A", "CPU")], rest_s=0))
    (repo / "logs").mkdir()
    (repo / "logs" / "x.atof.jsonl").write_text("", encoding="utf-8")
    fake = Fake({"A": [StepResult(ok=True, latency_ms=10) for _ in range(3)]})
    with pytest.raises(GuardError, match="logs/x.atof.jsonl already exists"):
        run_spec(spec_path, root=repo, run_id="x", date="2026-10-14", adapters={"fake": fake}, trace="off",
                 watch=WatchHooks(ram_sampler=None, start_thread=False, sleep=lambda s: None))
    assert not (repo / "receipts").exists()

