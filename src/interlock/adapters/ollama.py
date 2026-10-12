"""Ollama over local HTTP.

The 5090 is ``127.0.0.1:11434``. CPU-only Ollama is ``127.0.0.1:11490``.
Cloud tags are refused. This module does not open a socket unless a transport
is given, or the default transport is actually called.

Decode tokens per second are ``eval_count / eval_duration`` from the server.
Prompt-processing tokens per second are ``prompt_eval_count / prompt_eval_duration``.
Neither one is wall clock. ``think`` keeps the level string when it is not
``off`` or ``on``, so a later model can use low, medium, or high.

The default transport sets no timeout. A caller that wants one passes
``timeout`` on the adapter. ``cancel`` shuts the open socket down before the
response is read, so a load can be closed while it is still running. Closing
the socket does not free memory the server has already allocated. The unload
call has its own timeout, about 60 seconds. If that expires, the model may
still be resident.
"""

from __future__ import annotations

import http.client
import json
import select
import socket
import threading
import time
from urllib.parse import urlsplit

# The token count asks for one completion token. Ollama reads num_predict 0 as no limit.
COUNT_NUM_PREDICT = 1

_CANCEL_POLL_S = 0.2
UNLOAD_TIMEOUT_S = 60.0
UNLOAD_TIMEOUT_NOTE = "the unload timed out; the model may still be resident"
COUNT_TIMEOUT_NOTE = "the token count timed out; the arm was refused"

from interlock.adapters.base import AdapterError, StepRequest, StepResult

URL_5090 = "http://127.0.0.1:11434"
URL_CPU = "http://127.0.0.1:11490"


def url_for(device: str) -> str:
    if device == "5090":
        return URL_5090
    if device == "CPU":
        return URL_CPU
    raise AdapterError(f"Ollama has no URL for device {device}")


def refuse_cloud(model: str) -> None:
    lowered = model.lower()
    if ":cloud" in lowered or lowered.endswith("-cloud"):
        raise AdapterError(f"cloud model tags are refused ({model})")


def think_value(level: object) -> object:
    """Map a thinking level onto Ollama's ``think`` field.

    ``off`` is false and ``on`` is true. Any other string is passed through,
    which is the seam for low, medium, and high.
    """

    if level == "off":
        return False
    if level == "on":
        return True
    return level


def _given(settings: dict, key: str, default: object) -> object:
    """A missing key and an explicit null both mean the adapter default."""

    value = settings.get(key)
    return default if value is None else value


def _require_num_predict(value) -> None:
    """Refuse a generate with no completion limit before any request.

    Ollama reads ``num_predict`` 0, null or missing as no limit and generates
    until the context fills, which is what stalled E4's first run.
    """

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AdapterError(f"num_predict must be a whole number of at least 1, not {value!r}")


def generate_body(request: StepRequest) -> dict:
    refuse_cloud(request.model)
    settings = request.settings
    _require_num_predict(settings.get("num_predict"))
    options = {
        "num_predict": settings.get("num_predict"),
        "temperature": _given(settings, "temperature", 0),
    }
    if settings.get("num_ctx") is not None:
        options["num_ctx"] = settings["num_ctx"]
    body = {
        "model": request.model,
        "prompt": _given(settings, "prompt", ""),
        "stream": False,
        "options": options,
        "keep_alive": _given(settings, "keep_alive", "5m"),
    }
    if settings.get("shift") is False:
        body["shift"] = False
    if settings.get("truncate") is False:
        body["truncate"] = False
    think = settings.get("thinking_level")
    if think is not None:
        body["think"] = think_value(think)
    return body




def context_length(payload: dict) -> int:
    """The one ``*.context_length`` value in ``/api/show`` ``model_info``."""

    info = payload.get("model_info")
    if not isinstance(info, dict):
        raise AdapterError("Ollama show has no model_info")
    found = [
        value
        for key, value in info.items()
        if isinstance(key, str) and key.endswith(".context_length") and isinstance(value, int) and value > 0
    ]
    if len(found) != 1:
        raise AdapterError("Ollama show did not name one context length")
    return found[0]


def quantization_level(payload: dict) -> str:
    """``details.quantization_level`` from ``/api/show``, or ``unrecorded``."""

    details = payload.get("details")
    if not isinstance(details, dict):
        return "unrecorded"
    level = details.get("quantization_level")
    if not isinstance(level, str) or not level:
        return "unrecorded"
    return level


class _Ticket:
    """One request's cancel switch. A later request does not clear it."""

    def __init__(self, close) -> None:
        self.close = close
        self.cancelled = threading.Event()


class _Flight:
    """The request in flight. Each ``arm`` installs that request's own ticket."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: _Ticket | None = None

    def arm(self, close) -> _Ticket:
        ticket = _Ticket(close)
        with self._lock:
            self._current = ticket
        return ticket

    def disarm(self, ticket: _Ticket) -> None:
        with self._lock:
            if self._current is ticket:
                self._current = None

    def cancel(self) -> None:
        with self._lock:
            ticket = self._current
        if ticket is None:
            return
        ticket.cancelled.set()
        try:
            ticket.close()
        except Exception:
            return


def _shutdown(conn: http.client.HTTPConnection) -> None:
    """Shut the socket down, then close it. A bare close does not unblock a read."""

    sock = getattr(conn, "sock", None)
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass
    try:
        conn.close()
    except OSError:
        pass


def default_transport(
    method: str,
    url: str,
    body: dict | None,
    timeout: float | None = None,
    flight: _Flight | None = None,
) -> dict:
    """POST or GET over ``http.client``, armed before the response is read.

    The closer is armed before ``getresponse``. During a non-streaming
    generate that is the whole load: the server sends no bytes until it is
    finished, so a closer armed only after the headers arrive never fires.
    The closer shuts the socket down. Windows leaves a blocked ``recv``
    running after that, so the wait is sliced and stops when cancel was asked.
    """

    parts = urlsplit(url)
    if parts.scheme != "http" or not parts.hostname:
        raise AdapterError("Ollama transport only opens local http")
    path = parts.path or "/"
    if parts.query:
        path = f"{path}?{parts.query}"
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data is not None else {}
    conn = http.client.HTTPConnection(parts.hostname, parts.port or 80, timeout=timeout)
    ticket = None
    try:
        if flight is not None:
            ticket = flight.arm(lambda: _shutdown(conn))
        conn.request(method, path, body=data, headers=headers)
        if ticket is not None and getattr(conn, "sock", None) is not None:
            _wait_for_reply(conn, ticket, timeout)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
    except AdapterError:
        raise
    except (OSError, http.client.HTTPException, json.JSONDecodeError):
        raise AdapterError("the request was closed")
    finally:
        if ticket is not None and flight is not None:
            flight.disarm(ticket)
        try:
            conn.close()
        except OSError:
            pass
    return payload


def _wait_for_reply(conn: http.client.HTTPConnection, ticket: _Ticket, timeout: float | None) -> None:
    """Return when the socket is readable, or raise once this request was cancelled."""

    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        if ticket.cancelled.is_set():
            raise AdapterError("the request was closed")
        if deadline is not None and time.monotonic() >= deadline:
            raise AdapterError("the request timed out")
        sock = getattr(conn, "sock", None)
        if sock is None:
            raise AdapterError("the request was closed")
        slice_s = _CANCEL_POLL_S
        if deadline is not None:
            slice_s = min(slice_s, max(0.0, deadline - time.monotonic()))
        try:
            readable, _, _ = select.select([sock], [], [], slice_s)
        except (OSError, ValueError):
            raise AdapterError("the request was closed")
        if readable:
            return


def _rate(count: object, duration_ns: object) -> float | None:
    if isinstance(count, int) and isinstance(duration_ns, int) and duration_ns > 0:
        return count / (duration_ns / 1_000_000_000)
    return None


class OllamaAdapter:
    def __init__(
        self,
        *,
        transport=None,
        device: str = "5090",
        timeout: float | None = None,
        unload_timeout: float = UNLOAD_TIMEOUT_S,
    ):
        self.timeout = timeout
        self.unload_timeout = unload_timeout
        self._unload_gave_up = False
        self._default_transport = transport is None
        self._flight = _Flight()
        if transport is None:
            def bound(method: str, url: str, body: dict | None, timeout: float | None = None) -> dict:
                chosen = self.timeout if timeout is None else timeout
                return default_transport(method, url, body, timeout=chosen, flight=self._flight)

            self.transport = bound
        else:
            self.transport = transport
        self.device = device

    def cancel(self) -> None:
        """Close the in-flight request so the unload can take effect.

        This does not free memory the load has already allocated. The unload
        is what asks the server to drop the model.
        """

        self._flight.cancel()
        closer = getattr(self.transport, "cancel", None)
        if callable(closer):
            closer()

    def count_tokens(
        self,
        model: str,
        text: str,
        *,
        settings: dict | None = None,
        device: str | None = None,
        timeout: float | None = None,
    ) -> int:
        """How many tokens this served model assigns to ``text``.

        One ``/api/generate`` with ``raw`` true, so the template is not counted,
        and ``num_predict`` 1, so the completion is a single token. The count is
        ``prompt_eval_count``. ``num_predict`` 0 is never sent: Ollama 0.35.1
        reads 0 as no limit and generates until the context fills (seen live on
        2026-10-11, E4's first run). Load options in ``settings``
        (context, thinking, keep_alive) are the measurement's, so the count
        and the measurement name the same load. This does not unload. The
        harness unloads afterwards when the count would leave the model resident.
        """

        refuse_cloud(model)
        merged = dict(settings or {})
        merged["prompt"] = text
        merged["num_predict"] = COUNT_NUM_PREDICT
        request = StepRequest(
            step_id="count",
            task="generate",
            model=model,
            device=device or self.device,
            engine="ollama",
            settings=merged,
        )
        request, _resolved = self._resolve(request)
        url = url_for(request.device)
        body = generate_body(request)
        body["raw"] = True
        body["truncate"] = False
        body["options"]["num_predict"] = COUNT_NUM_PREDICT
        payload = self._count_payload(url, body, timeout)
        count = payload.get("prompt_eval_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise AdapterError("Ollama did not report prompt_eval_count")
        return count

    def _count_payload(self, url: str, body: dict, timeout: float | None = None) -> dict:
        payload = self._count_exchange(url, body, timeout)
        if not isinstance(payload, dict):
            raise AdapterError("Ollama returned a non-object")
        error = payload.get("error")
        if error:
            raise AdapterError(str(error))
        return payload

    def _count_exchange(self, url: str, body: dict, timeout: float | None) -> dict:
        """POST the count. ``timeout`` bounds the call even when the fake ignores it."""

        if timeout is None:
            return self._post_count(url, body, None)
        box: dict = {}

        def invoke() -> None:
            try:
                box["payload"] = self._post_count(url, body, timeout)
            except BaseException as exc:
                box["error"] = exc

        worker = threading.Thread(target=invoke, name="ollama-count", daemon=True)
        worker.start()
        worker.join(timeout)
        if worker.is_alive():
            self.cancel()
            raise AdapterError(COUNT_TIMEOUT_NOTE)
        if "error" in box:
            exc = box["error"]
            if isinstance(exc, AdapterError) and str(exc) == "the request timed out":
                raise AdapterError(COUNT_TIMEOUT_NOTE) from exc
            raise exc
        if "payload" not in box:
            raise AdapterError(COUNT_TIMEOUT_NOTE)
        return box["payload"]

    def _post_count(self, url: str, body: dict, timeout: float | None) -> dict:
        if timeout is not None and self._default_transport:
            return self.transport("POST", f"{url}/api/generate", body, timeout=timeout)
        return self.transport("POST", f"{url}/api/generate", body)

    def tag_size(self, model: str, device: str | None = None) -> int | None:
        """The model's ``size`` from ``/api/tags``, in bytes. Missing is None."""

        refuse_cloud(model)
        url = url_for(device or self.device)
        payload = self.transport("GET", f"{url}/api/tags", None)
        if not isinstance(payload, dict):
            return None
        models = payload.get("models")
        if not isinstance(models, list):
            return None
        for item in models:
            if not isinstance(item, dict):
                continue
            label = item.get("name") or item.get("model")
            if not isinstance(label, str) or label != model:
                continue
            size = item.get("size")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                return None
            return size
        return None

    def show(self, model: str, device: str | None = None) -> dict:
        refuse_cloud(model)
        url = url_for(device or self.device)
        payload = self.transport("POST", f"{url}/api/show", {"model": model})
        if not isinstance(payload, dict):
            raise AdapterError("Ollama show returned a non-object")
        if payload.get("error"):
            raise AdapterError(str(payload["error"]))
        return payload

    def _resolve(self, request: StepRequest) -> tuple[StepRequest, dict]:
        settings = dict(request.settings)
        need_ctx = "num_ctx" in settings and settings["num_ctx"] is None
        need_quant = "quant" in settings and settings.get("quant") in (None, "unrecorded")
        if not need_ctx and not need_quant:
            return request, {}
        shown = self.show(request.model, request.device)
        resolved: dict = {}
        if need_ctx:
            settings["num_ctx"] = context_length(shown)
            resolved["num_ctx"] = settings["num_ctx"]
        if need_ctx or need_quant:
            resolved["quant"] = quantization_level(shown)
        return (
            StepRequest(
                step_id=request.step_id,
                task=request.task,
                model=request.model,
                device=request.device,
                engine=request.engine,
                settings=settings,
            ),
            resolved,
        )

    def run(self, request: StepRequest) -> StepResult:
        refuse_cloud(request.model)
        request, resolved = self._resolve(request)
        url = url_for(request.device)
        payload = self.transport("POST", f"{url}/api/generate", generate_body(request))
        if not isinstance(payload, dict):
            return StepResult(ok=False, error="Ollama returned a non-object")
        if payload.get("error"):
            return StepResult(ok=False, error=str(payload["error"]))
        total_ns = payload.get("total_duration")
        latency = (total_ns / 1_000_000) if isinstance(total_ns, int) else None
        prompt_ns = payload.get("prompt_eval_duration")
        return StepResult(
            ok=True,
            latency_ms=latency,
            throughput=_rate(payload.get("eval_count"), payload.get("eval_duration")),
            throughput_unit="tok/s",
            prompt_tok_s=_rate(payload.get("prompt_eval_count"), prompt_ns),
            output=payload.get("response"),
            resolved=resolved or None,
            prompt_eval_duration_ns=prompt_ns if isinstance(prompt_ns, int) and not isinstance(prompt_ns, bool) else None,
        )

    def unload(self, model: str, device: str | None = None) -> None:
        """Ask the server to drop the model.

        This call has its own deadline, separate from a generate. A generate
        can sit for as long as the load takes. An unload that never returns
        would leave the model resident and the caller waiting. When the
        deadline passes, the error says the model may still be resident.
        A later unload does not wait out the deadline again.
        """

        refuse_cloud(model)
        if self._unload_gave_up:
            raise AdapterError(UNLOAD_TIMEOUT_NOTE)
        url = url_for(device or self.device)
        body = {"model": model, "prompt": "", "stream": False, "keep_alive": 0}
        limit = self.unload_timeout
        error: list[BaseException] = []

        def invoke() -> None:
            try:
                if self._default_transport:
                    self.transport("POST", f"{url}/api/generate", body, timeout=limit)
                else:
                    self.transport("POST", f"{url}/api/generate", body)
            except Exception as exc:
                error.append(exc)

        worker = threading.Thread(target=invoke, name="ollama-unload", daemon=True)
        worker.start()
        worker.join(limit)
        if worker.is_alive():
            self._unload_gave_up = True
            self.cancel()
            raise AdapterError(UNLOAD_TIMEOUT_NOTE)
        if error:
            exc = error[0]
            if isinstance(exc, AdapterError) and str(exc) == "the request timed out":
                self._unload_gave_up = True
                raise AdapterError(UNLOAD_TIMEOUT_NOTE) from exc
            raise exc

    def start_session(self, request: StepRequest) -> None:
        """Begin a companion load.

        The transport must return while that load is still running. This method
        does not wait out the paired measurement, and it does not unload.
        ``num_ctx`` null is resolved here the same way ``run`` resolves it, so
        the companion load and the measured step name the same context.
        """

        refuse_cloud(request.model)
        request, _resolved = self._resolve(request)
        url = url_for(request.device)
        body = generate_body(request)
        body["companion"] = "start"
        self.transport("POST", f"{url}/api/generate", body)

    def stop_session(self, request: StepRequest) -> None:
        """End the companion by unloading it.

        An empty generate is not an unload. Ollama 0.35.1 answers that shape
        with done_reason load (tests/fixtures/ollama/0.35.1/companion_stop.json).
        The unload shape is keep_alive 0.
        """

        self.unload(request.model, request.device)

    def loaded_models(self) -> list:
        url = url_for(self.device)
        payload = self.transport("GET", f"{url}/api/ps", None)
        if not isinstance(payload, dict):
            return []
        models = payload.get("models", [])
        return models if isinstance(models, list) else []
