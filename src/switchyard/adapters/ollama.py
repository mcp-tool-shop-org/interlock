"""Ollama over local HTTP.

The 5090 is ``127.0.0.1:11434``. CPU-only Ollama is ``127.0.0.1:11490``.
Cloud tags are refused. This module does not open a socket unless a transport
is given, or the default transport is actually called.
"""

from __future__ import annotations

import json
from urllib.request import Request, urlopen

from switchyard.adapters.base import AdapterError, StepRequest, StepResult

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


def generate_body(request: StepRequest) -> dict:
    refuse_cloud(request.model)
    settings = request.settings
    options = {
        "num_predict": settings.get("num_predict"),
        "temperature": settings.get("temperature", 0),
    }
    if settings.get("num_ctx") is not None:
        options["num_ctx"] = settings["num_ctx"]
    body = {
        "model": request.model,
        "prompt": settings.get("prompt", ""),
        "stream": False,
        "options": options,
        "keep_alive": settings.get("keep_alive", "5m"),
    }
    if settings.get("shift") is False:
        body["shift"] = False
    if settings.get("truncate") is False:
        body["truncate"] = False
    think = settings.get("thinking_level")
    if think is not None:
        body["think"] = think != "off"
    return body


def default_transport(method: str, url: str, body: dict | None) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


class OllamaAdapter:
    def __init__(self, *, transport=None, device: str = "5090"):
        self.transport = transport or default_transport
        self.device = device

    def run(self, request: StepRequest) -> StepResult:
        refuse_cloud(request.model)
        url = url_for(request.device)
        payload = self.transport("POST", f"{url}/api/generate", generate_body(request))
        if not isinstance(payload, dict):
            return StepResult(ok=False, error="Ollama returned a non-object")
        if payload.get("error"):
            return StepResult(ok=False, error=str(payload["error"]))
        eval_count = payload.get("eval_count")
        eval_ns = payload.get("eval_duration")
        throughput = None
        if isinstance(eval_count, int) and isinstance(eval_ns, int) and eval_ns > 0:
            throughput = eval_count / (eval_ns / 1_000_000_000)
        total_ns = payload.get("total_duration")
        latency = (total_ns / 1_000_000) if isinstance(total_ns, int) else None
        return StepResult(ok=True, latency_ms=latency, throughput=throughput, throughput_unit="tok/s")

    def unload(self, model: str) -> None:
        refuse_cloud(model)
        url = url_for(self.device)
        self.transport(
            "POST",
            f"{url}/api/generate",
            {"model": model, "prompt": "", "stream": False, "keep_alive": 0},
        )

    def loaded_models(self) -> list:
        url = url_for(self.device)
        payload = self.transport("GET", f"{url}/api/ps", None)
        if not isinstance(payload, dict):
            return []
        models = payload.get("models", [])
        return models if isinstance(models, list) else []
