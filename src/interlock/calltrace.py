"""Every model call interlock makes, counted per kind and optionally logged in order.

The counter is always on. Each receipt row carries ``artifacts.call_counts``:
per call kind, how many calls, their client-side time, errors and timeouts.
That is a summary, not a log.

``--trace`` adds an ordered log of the same calls under ``logs/``:

- ``events``: one JSON line per finished call, written by this module
  (``logs/<run_id>.events.jsonl``);
- ``relay``: NVIDIA NeMo Relay's ATOF events, through its Python API
  (``logs/<run_id>.atof.jsonl``). Relay is imported only on this path.

Calls are classified from the request itself, so counts and unloads made on
worker threads need no context. A call is attributed to the slot whose window
it starts in, whatever its thread: a companion session is started and stopped
inside its slot, so its calls belong to that slot by construction, under their
own kind (``companion``). Every timestamp comes from one clock: an epoch
offset taken once when the run opens, plus ``perf_counter_ns``. Prompts and
responses are never recorded, only their byte lengths.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

KINDS = ("generate", "embed", "count", "show", "tags", "ps", "unload", "companion")
MODES = ("off", "events", "relay")


class TraceError(RuntimeError):
    """A tracer that cannot start safely refuses the run."""


def classify(method: str, url: str, body: dict | None) -> str:
    """The call kind of one Ollama request, from its endpoint and body."""

    path = urlsplit(url).path
    if path.endswith("/api/tags"):
        return "tags"
    if path.endswith("/api/ps"):
        return "ps"
    if path.endswith("/api/show"):
        return "show"
    if path.endswith(("/api/embed", "/api/embeddings")):
        return "embed"
    if path.endswith("/api/generate") and isinstance(body, dict):
        if body.get("companion") == "start":
            return "companion"
        if body.get("keep_alive") == 0 and body.get("prompt", "") == "":
            return "unload"
        if body.get("raw") is True:
            return "count"
        return "generate"
    return "other"


def _usage(payload: object) -> tuple[int | None, int | None]:
    if not isinstance(payload, dict):
        return None, None
    prompt = payload.get("prompt_eval_count")
    completion = payload.get("eval_count")
    return (prompt if isinstance(prompt, int) else None, completion if isinstance(completion, int) else None)


@dataclass
class CallRecord:
    call_id: int
    kind: str
    endpoint: str
    model: str | None
    device: str | None
    thread: str
    start_ns: int
    end_ns: int | None = None
    status: str = "open"
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    request_bytes: int | None = None  # the bytes the transport sent; None when the transport doesn't say
    extra: dict = field(default_factory=dict)
    emitted: bool = False

    @property
    def ms(self) -> float | None:
        return None if self.end_ns is None else (self.end_ns - self.start_ns) / 1e6


class _EventSink:
    """The native ordered log: one JSON line per finished call."""

    def __init__(self, path: Path):
        self.path = path
        self._file = path.open("w", encoding="utf-8")
        self._lock = threading.Lock()
        # The watcher may close the log while a worker or the run thread still calls in; later calls are dropped.
        self._closed = False

    def start(self, record: CallRecord) -> None:
        return None

    def end(self, record: CallRecord) -> None:
        line = {
            "call_id": record.call_id,
            "t_ns": record.start_ns,
            "end_ns": record.end_ns,
            "thread": record.thread,
            "kind": record.kind,
            "endpoint": record.endpoint,
            "model": record.model,
            "device": record.device,
            "ms": record.ms,
            "status": record.status,
            "request_bytes": record.request_bytes,
            "prompt_tokens": record.prompt_tokens,
            "completion_tokens": record.completion_tokens,
            **record.extra,
        }
        text = json.dumps(line, sort_keys=True) + "\n"
        with self._lock:
            if not self._closed:
                self._file.write(text)

    def mark(self, name: str, t_ns: int, data: dict) -> None:
        with self._lock:
            if not self._closed:
                self._file.write(json.dumps({"mark": name, "t_ns": t_ns, **data}, sort_keys=True) + "\n")

    def flush(self) -> None:
        with self._lock:
            if not self._closed:
                self._file.flush()

    def close(self, end_ns: int | None = None) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._file.flush()
                self._file.close()


class Tracer:
    """One run's call counter, and its ordered log when tracing is on."""

    def __init__(self, *, root: Path, run_id: str, mode: str = "off", relay_factory=None, per_arm: bool = False):
        if mode not in MODES:
            raise TraceError(f"unknown trace mode {mode}")
        self.root = Path(root)
        self.run_id = run_id
        self.mode = mode
        self._relay_factory = relay_factory
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._counts: dict[str, dict] = {}
        self._slot: dict = {}
        self._slot_calls: list[int] = []
        self._open: dict[int, CallRecord] = {}
        self._sink = None
        # Every sink call happens under self._lock (the lock order is always tracer, then sink), so a watcher
        # close can't land between a call's count and its sink event (Argon round 2, M-New-1).
        self._closed = False
        # Stage 1's off-on-off triples switch the sink's events per slot. The counter stays on (design v3.1).
        # Under per-arm tracing, emission is off outside slots: begin_slot and set_emitting turn it on for a traced
        # arm's slot, and end_slot turns it off (amendment v3.2). A run-wide trace emits everything.
        self._per_arm = bool(per_arm)
        self._emit = not self._per_arm
        self._slot_overhead_ns = 0
        self._offset_ns = 0
        self.trace_file: str | None = None
        self.scope_id: str | None = None

    # -- the run --------------------------------------------------------------

    def open(self) -> Tracer:
        """Take the one epoch offset and open the sink. Refuses before any device."""

        self._offset_ns = time.time_ns() - time.perf_counter_ns()
        if self.mode == "off":
            return self
        if any(key.startswith("OTEL_") for key in os.environ):
            raise TraceError("REFUSED: an OTEL_* variable is set; tracing must stay on this machine")
        logs = self.root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        if self.mode == "events":
            name = f"{self.run_id}.events.jsonl"
            self._sink = _EventSink(logs / name)
        else:
            name = f"{self.run_id}.atof.jsonl"
            factory = self._relay_factory
            if factory is None:
                from interlock.relay_sink import RelaySink as factory
            self._sink = factory(logs, name, self.now_ns(), self.run_id)
        self.trace_file = f"logs/{name}"
        self.scope_id = getattr(self._sink, "scope_id", None)
        return self

    def close(self) -> None:
        """Close any call left open, then flush and close the sink. Runs on every exit path, and on the watcher's
        thread before a halt's os._exit. A second close does nothing; calls that arrive after it are not recorded."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            for record in list(self._open.values()):
                record.end_ns = self.now_ns()
                record.status = "abandoned"
                self._finish_locked(record)
            self._open.clear()
            if self._sink is not None:
                sink, self._sink = self._sink, None
                sink.close(self.now_ns())
                self._write_calls()

    def _write_calls(self) -> None:
        """Every count, run-level calls included, beside the trace: ``logs/<run_id>.calls.json``. A receipt row
        carries only its own arm's counts, and stage 1's H1 check compares the whole table with an independent
        count of requests."""

        try:
            path = self.root / "logs" / f"{self.run_id}.calls.json"
            path.write_text(json.dumps({"run_id": self.run_id, "mode": self.mode, "counts": self._counts},
                                       indent=1, sort_keys=True) + "\n", encoding="utf-8")
        except OSError as exc:  # the trace is already closed; a missing summary is reported, not fatal
            print(f"interlock: could not write the call summary: {exc}", file=sys.stderr)

    def __enter__(self) -> Tracer:
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    def now_ns(self) -> int:
        return self._offset_ns + time.perf_counter_ns()

    def mark(self, name: str, **data) -> None:
        with self._lock:
            if self._sink is not None:
                self._sink.mark(name, self.now_ns(), data)

    def flush(self) -> None:
        with self._lock:
            if self._sink is not None:
                self._sink.flush()

    # -- slots and rows ---------------------------------------------------------

    @property
    def emitting(self) -> bool:
        return self._emit and self._sink is not None

    def set_emitting(self, on: bool) -> None:
        """Turn the sink's call events on or off for the calls that start from now. A call keeps the choice it
        started with, so no start is ever left without its end."""

        with self._lock:
            self._emit = bool(on)

    def set_per_arm(self, on: bool) -> None:
        """Scope the sink's events to traced slots: calls outside a slot are counted and not emitted."""

        with self._lock:
            self._per_arm = bool(on)
            if self._per_arm and not self._slot:
                self._emit = False

    def begin_slot(self, **slot) -> None:
        with self._lock:
            self._slot = dict(slot)
            self._slot_calls = []
            self._slot_overhead_ns = 0

    def end_slot(self) -> dict:
        """The slot's calls, and the client time the hooks spent on them (the tracer's own bookkeeping and its
        sink, in microseconds): stage 1's client-time gate compares this between traced and untraced slots.

        Under per-arm tracing this is the one place emission goes off, so nothing after the slot (a companion's
        stop, a recovery unload) can leave a span outside it (amendment v3.2)."""

        with self._lock:
            calls = list(self._slot_calls)
            overhead_us = round(self._slot_overhead_ns / 1000, 3)
            self._slot = {}
            self._slot_calls = []
            self._slot_overhead_ns = 0
            if self._per_arm:
                self._emit = False
        return {"call_ids": calls, "hook_us": overhead_us}

    def counts_for(self, arm_id: str) -> dict:
        """``artifacts.call_counts`` for one arm's row."""

        with self._lock:
            return {kind: dict(value) for kind, value in sorted(self._counts.get(arm_id, {}).items())}

    # -- calls ------------------------------------------------------------------

    def start(
        self, kind: str, endpoint: str, *, model: str | None, device: str | None, request_bytes: int | None = None
    ) -> CallRecord:
        record = CallRecord(
            call_id=next(self._ids),
            kind=kind,
            endpoint=endpoint,
            model=model,
            device=device,
            thread=threading.current_thread().name,
            start_ns=self.now_ns(),
            request_bytes=request_bytes,
        )
        with self._lock:
            if self._closed:
                return record  # after the close: not counted, not logged
            self._open[record.call_id] = record
            if self._slot:
                self._slot_calls.append(record.call_id)
                record.extra = {"arm": self._slot.get("arm"), "slot": self._slot.get("index")}
            if self._sink is not None and self._emit:
                record.emitted = True
                self._sink.start(record)
        return record

    def charge(self, record: CallRecord, hook_ns: int) -> None:
        """Add a call's hook time (everything its hook did outside the model call) to the slot it started in."""

        with self._lock:
            if self._slot and record.extra.get("slot") == self._slot.get("index"):
                self._slot_overhead_ns += hook_ns

    def end(self, record: CallRecord, status: str, payload: object = None) -> None:
        record.end_ns = self.now_ns()
        record.status = status
        record.prompt_tokens, record.completion_tokens = _usage(payload)
        with self._lock:
            if self._open.pop(record.call_id, None) is None:
                return
            self._finish_locked(record)

    def _finish_locked(self, record: CallRecord) -> None:
        """Count a finished call and log its end. The caller holds self._lock."""

        arm = record.extra.get("arm") or "_run"
        fresh = {"n": 0, "ms": 0.0, "errors": 0, "timeouts": 0}
        entry = self._counts.setdefault(arm, {}).setdefault(record.kind, fresh)
        entry["n"] += 1
        entry["ms"] = round(entry["ms"] + (record.ms or 0.0), 3)
        if record.status == "timeout":
            entry["timeouts"] += 1
        elif record.status != "ok":
            entry["errors"] += 1
        if self._sink is not None and record.emitted:
            self._sink.end(record)

    @contextmanager
    def call(self, kind: str, endpoint: str, *, model: str | None, device: str | None):
        record = self.start(kind, endpoint, model=model, device=device)
        box: dict = {"status": "exception", "payload": None}
        try:
            yield box
        finally:
            self.end(record, box["status"], box["payload"])

    def artifacts(self, arm_id: str) -> dict:
        """What a receipt row carries for one arm: the counts, and the trace pointer when tracing."""

        out: dict = {"call_counts": self.counts_for(arm_id)}
        if self.trace_file is not None:
            out["trace"] = {"file": self.trace_file}
            if self.scope_id:
                out["trace"]["span"] = self.scope_id
        return out


class _TracedTransport:
    """An Ollama transport, classified, timed and counted on the calling thread.

    Every other attribute goes to the wrapped transport, so methods the adapter looks for on its transport (such
    as ``cancel``, which ends an in-flight request before an unload) still reach it."""

    def __init__(self, transport, tracer: Tracer, device: str | None):
        self.__wrapped__ = transport
        self._tracer = tracer
        self._device = device

    def __getattr__(self, name):
        return getattr(self.__wrapped__, name)

    def __call__(self, method: str, url: str, body: dict | None, *args, **kwargs):
        entered = time.perf_counter_ns()
        tracer = self._tracer
        kind = classify(method, url, body)
        model = body.get("model") if isinstance(body, dict) else None
        record = tracer.start(kind, urlsplit(url).path, model=model, device=self._device)
        if getattr(self.__wrapped__, "reports_request_bytes", False):
            # The transport reports the bytes it sends, so the body is never serialised a second time here
            # (amendment v3.2). A transport that doesn't report leaves the size unknown.

            def note_sent(count: int) -> None:
                record.request_bytes = count

            kwargs = {**kwargs, "on_send": note_sent}
        status, payload = "exception", None
        sent = time.perf_counter_ns()
        returned = sent
        try:
            payload = self.__wrapped__(method, url, body, *args, **kwargs)
            returned = time.perf_counter_ns()
            status = "ok"
            return payload
        except Exception as exc:
            returned = time.perf_counter_ns()
            status = "timeout" if isinstance(exc, TimeoutError) or "timed out" in str(exc).lower() else "http_error"
            raise
        finally:
            tracer.end(record, status, payload)
            # Hook time: before the request went out, and after it came back.
            tracer.charge(record, (sent - entered) + (time.perf_counter_ns() - returned))


def traced_transport(transport, tracer: Tracer, *, device: str | None = None):
    """Wrap an Ollama transport: every request is classified, timed and counted on the calling thread."""

    return _TracedTransport(transport, tracer, device)


def instrument(adapters: dict, tracer: Tracer) -> None:
    """Count every model call the run's adapters make: Ollama through its
    transport, OpenVINO through its bounded call."""

    for engine, adapter in adapters.items():
        if engine == "ollama" and hasattr(adapter, "transport"):
            if getattr(adapter.transport, "__wrapped__", None) is None:
                adapter.transport = traced_transport(adapter.transport, tracer, device=getattr(adapter, "device", None))
        elif engine == "openvino":
            import importlib

            module = importlib.import_module("interlock.adapters.openvino")
            module.set_tracer(tracer)


def release(adapters: dict) -> None:
    """Undo ``instrument``: the Ollama transport is unwrapped, so a reused adapter is counted by its next run's
    tracer, and the OpenVINO module's tracer is cleared (Argon round 2, M-New-2)."""

    for engine, adapter in adapters.items():
        if engine == "ollama" and hasattr(adapter, "transport"):
            inner = getattr(adapter.transport, "__wrapped__", None)
            if inner is not None:
                adapter.transport = inner
        elif engine == "openvino":
            import importlib

            module = importlib.import_module("interlock.adapters.openvino")
            module.set_tracer(None)
