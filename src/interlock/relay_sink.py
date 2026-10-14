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
import json
from datetime import datetime, timezone
from pathlib import Path

from interlock.calltrace import CallRecord, TraceError


def _when(t_ns: int) -> datetime:
    return datetime.fromtimestamp(t_ns / 1e9, tz=timezone.utc)


def _length_only(payload, _context=None) -> dict:
    return {"bytes": len(json.dumps(payload, default=str))}


class RelaySink:
    def __init__(self, logs: Path, filename: str, start_ns: int, run_id: str):
        import nemo_relay
        from nemo_relay import guardrails, plugin, scope

        self._relay = nemo_relay
        self._scope = scope
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
        self._loop = asyncio.new_event_loop()
        self._activation = self._loop.run_until_complete(plugin.initialize(config))
        guardrails.register_llm_sanitize_request("interlock-length-only-request", 0, _length_only)
        guardrails.register_llm_sanitize_response("interlock-length-only-response", 0, _length_only)
        self._run = scope.push(f"interlock-run {run_id}", scope.ScopeType.Agent, timestamp=_when(start_ns))
        self.scope_id = self._run.uuid
        self._spans: dict[int, object] = {}

    def start(self, record: CallRecord) -> None:
        request = self._relay.LLMRequest({}, {"bytes": record.request_bytes})
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
        handle = self._spans.pop(record.call_id, None)
        if handle is None:
            return
        self._relay.llm.call_end(
            handle,
            {"status": record.status},
            metadata={"call_id": record.call_id, "status": record.status, "ms": record.ms,
                      "prompt_tokens": record.prompt_tokens, "completion_tokens": record.completion_tokens},
            timestamp=_when(record.end_ns),
        )

    def mark(self, name: str, t_ns: int, data: dict) -> None:
        self._scope.event(name, handle=self._run, data=data, timestamp=_when(t_ns))

    def flush(self) -> None:
        self._relay.subscribers.flush()

    def close(self, end_ns: int | None = None) -> None:
        try:
            self._scope.pop(self._run, timestamp=_when(end_ns) if end_ns is not None else None)
        finally:
            self._relay.subscribers.flush()
            self._loop.run_until_complete(self._activation.close())
            self._loop.close()
