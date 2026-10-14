"""NeMo Relay as the ordered call log (``--trace relay``). Imported only on that path.

Relay is driven through its Python API on a private event loop: an
observability plugin with one ATOF file sink and nothing else, one run scope,
and one ``llm.call``/``llm.call_end`` span per model call. Every timestamp is
interlock's own, so spans and the run scope share one clock. Payload
sanitizers replace prompts and responses with their byte lengths. Token counts
go in the span's metadata, which the ATOF file keeps.

The sink refuses to start if Relay would layer in any configuration file
(user or system ``plugins.toml``), so the only sink is the one named here.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import threading
from datetime import UTC, datetime
from pathlib import Path

from interlock.calltrace import CallRecord, TraceError


def _when(t_ns: int) -> datetime:
    return datetime.fromtimestamp(t_ns / 1e9, tz=UTC)


def _length_only(payload, _context=None) -> dict:
    """A payload as its byte length only. A record already of that shape passes through: the request record is
    built as ``{"bytes": n}`` before the sanitizer sees it, and re-measuring it recorded the length of that dict
    (13 or 15 in stage 1) instead of ``n`` (amendment v3.2)."""

    if isinstance(payload, dict) and set(payload) == {"bytes"} and (
        payload["bytes"] is None or (isinstance(payload["bytes"], int) and not isinstance(payload["bytes"], bool))
    ):
        return payload
    return {"bytes": len(json.dumps(payload, default=str))}


def _request_length_only(request, _context=None):
    """Relay's request sanitizer must hand back an ``LLMRequest``: one whose content is only its byte length."""

    import nemo_relay

    content = getattr(request, "content", request)
    return nemo_relay.LLMRequest({}, _length_only(content))


_REQUEST = "interlock-length-only-request"
_RESPONSE = "interlock-length-only-response"
# Relay's plugin activation and its sanitizers are process-wide, so one sink is live at a time.
_LIVE = threading.Lock()


class RelaySink:
    """One run's Relay trace. Calls arrive from the run thread and from worker threads (unloads, OpenVINO
    companions), so every Relay call goes through one lock (Argon B1)."""

    def __init__(self, logs: Path, filename: str, start_ns: int, run_id: str):
        import nemo_relay
        from nemo_relay import guardrails, plugin, scope

        self._relay = nemo_relay
        self._scope = scope
        self._guardrails = guardrails
        self._lock = threading.Lock()
        self._spans: dict[int, object] = {}
        # The watcher may close the sink while a thread still calls in; later calls are dropped.
        self._closed = False
        config = {
            "version": 1,
            "components": [{
                "kind": "observability",
                "enabled": True,
                "config": {
                    "version": 4,
                    "enable_full_payloads": False,
                    "atof": {"enabled": True, "sinks": [
                        {"type": "file", "output_directory": str(logs), "filename": filename, "mode": "overwrite"}
                    ]},
                },
            }],
        }
        report = plugin.validate(config)
        if report.get("config_paths"):
            raise TraceError("REFUSED: Relay would layer in other plugins.toml files; remove them before tracing")
        if not _LIVE.acquire(blocking=False):
            raise TraceError("REFUSED: a Relay trace is already open in this process; one run traces at a time")
        # Anything acquired below is released if a later step fails (Argon B3).
        self._loop = None
        self._activation = None
        registered: list[str] = []
        try:
            self._loop = asyncio.new_event_loop()
            self._activation = self._loop.run_until_complete(plugin.initialize(config))
            guardrails.register_llm_sanitize_request(_REQUEST, 0, _request_length_only)
            registered.append("request")
            guardrails.register_llm_sanitize_response(_RESPONSE, 0, _length_only)
            registered.append("response")
            self._run = scope.push(f"interlock-run {run_id}", scope.ScopeType.Agent, timestamp=_when(start_ns))
            self.scope_id = self._run.uuid
            # Relay keeps its scope stack in a context variable. The watcher closes the trace from its own
            # thread on a halt, so the pop runs in the context the scope was pushed in.
            self._context = contextvars.copy_context()
        except BaseException:
            self._release(registered)
            raise

    def _release(self, registered=("request", "response")) -> None:
        try:
            if "request" in registered:
                self._guardrails.deregister_llm_sanitize_request(_REQUEST)
            if "response" in registered:
                self._guardrails.deregister_llm_sanitize_response(_RESPONSE)
            if self._activation is not None:
                self._loop.run_until_complete(self._activation.close())
        finally:
            if self._loop is not None:
                self._loop.close()
            _LIVE.release()

    def start(self, record: CallRecord) -> None:
        request = self._relay.LLMRequest({}, {"bytes": record.request_bytes})
        with self._lock:
            if self._closed:
                return
            handle = self._relay.llm.call(
                "ollama" if record.endpoint.startswith("/api/") else "openvino",
                request,
                handle=self._run,
                model_name=record.model,
                metadata={"call_id": record.call_id, "kind": record.kind, "endpoint": record.endpoint,
                          "device": record.device, "thread": record.thread, **record.extra},
                timestamp=_when(record.start_ns),
            )
            self._spans[record.call_id] = handle

    def end(self, record: CallRecord) -> None:
        with self._lock:
            handle = self._spans.pop(record.call_id, None)
            if handle is None or self._closed:
                return
            self._relay.llm.call_end(
                handle,
                {"status": record.status},
                # extra carries what is only known at the end, such as an OpenVINO timeout's worker_still_running.
                metadata={**record.extra, "call_id": record.call_id, "status": record.status, "ms": record.ms,
                          "request_bytes": record.request_bytes, "prompt_tokens": record.prompt_tokens,
                          "completion_tokens": record.completion_tokens},
                timestamp=_when(record.end_ns),
            )

    def mark(self, name: str, t_ns: int, data: dict) -> None:
        with self._lock:
            if self._closed:
                return
            self._scope.event(name, handle=self._run, data=data, timestamp=_when(t_ns))

    def flush(self) -> None:
        with self._lock:
            if not self._closed:
                self._relay.subscribers.flush()

    def close(self, end_ns: int | None = None) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._context.run(
                    self._scope.pop, self._run, timestamp=_when(end_ns) if end_ns is not None else None
                )
            finally:
                try:
                    self._relay.subscribers.flush()
                finally:
                    self._release()
