"""The Ollama exchanges the fakes are allowed to imitate.

Each case is one fixture file the recorder writes. ``no_reply`` is a call that
must not be waited out: Ollama 0.35.1 reads ``num_predict`` 0, null, or missing
as no limit. A parameter the adapter can send and no case covers is a missing
fixture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.parse import urlsplit

from switchyard.adapters.base import StepRequest
from switchyard.adapters.ollama import generate_body

MODEL = "qwen3:8b"
SHORT_PROMPT = "hi"
LONG_PROMPT = "word " * 64


@dataclass(frozen=True)
class Case:
    name: str
    method: str
    path: str
    body: dict | None
    behaviour: str
    covers: frozenset[str]


def _generate(options: dict, **extra) -> dict:
    body = {
        "model": MODEL,
        "prompt": SHORT_PROMPT,
        "stream": False,
        "keep_alive": "5m",
        "options": options,
    }
    body.update(extra)
    return body


CASES: tuple[Case, ...] = (
    Case(
        "num_predict_1",
        "POST",
        "/api/generate",
        _generate({"num_predict": 1, "temperature": 0}),
        "reply",
        frozenset({"num_predict", "temperature", "stream", "keep_alive"}),
    ),
    Case(
        "num_predict_0",
        "POST",
        "/api/generate",
        _generate({"num_predict": 0, "temperature": 0, "num_ctx": 64}, shift=False),
        "no_reply",
        frozenset({"num_predict", "num_ctx", "shift"}),
    ),
    Case(
        "num_predict_null",
        "POST",
        "/api/generate",
        _generate({"num_predict": None, "temperature": 0, "num_ctx": 64}, shift=False),
        "no_reply",
        frozenset({"num_predict", "num_ctx", "shift"}),
    ),
    Case(
        "num_predict_missing",
        "POST",
        "/api/generate",
        _generate({"temperature": 0, "num_ctx": 64}, shift=False),
        "no_reply",
        frozenset({"num_predict", "num_ctx", "shift"}),
    ),
    Case(
        "raw_true",
        "POST",
        "/api/generate",
        _generate({"num_predict": 1, "temperature": 0, "num_ctx": 2048}, raw=True, truncate=False),
        "reply",
        frozenset({"raw", "truncate", "num_ctx", "num_predict"}),
    ),
    Case(
        "keep_alive_0",
        "POST",
        "/api/generate",
        {"model": MODEL, "prompt": "", "stream": False, "keep_alive": 0},
        "reply",
        frozenset({"keep_alive"}),
    ),
    Case("ps_loaded", "GET", "/api/ps", None, "reply", frozenset({"/api/ps"})),
    Case("ps_empty", "GET", "/api/ps", None, "reply", frozenset({"/api/ps"})),
    Case(
        "show_qwen3_8b",
        "POST",
        "/api/show",
        {"model": MODEL},
        "reply",
        frozenset({"/api/show"}),
    ),
    Case("tags", "GET", "/api/tags", None, "reply", frozenset({"/api/tags"})),
    Case("version", "GET", "/api/version", None, "reply", frozenset({"/api/version"})),
    Case(
        "truncate_over_num_ctx",
        "POST",
        "/api/generate",
        _generate(
            {"num_predict": 1, "temperature": 0, "num_ctx": 8},
            truncate=False,
            prompt=LONG_PROMPT,
        ),
        "error",
        frozenset({"truncate", "num_ctx"}),
    ),
    Case(
        "think_off",
        "POST",
        "/api/generate",
        _generate({"num_predict": 1, "temperature": 0}, think=False),
        "reply",
        frozenset({"think"}),
    ),
    Case(
        "shift_false",
        "POST",
        "/api/generate",
        _generate({"num_predict": 1, "temperature": 0}, shift=False),
        "reply",
        frozenset({"shift"}),
    ),
    Case(
        "companion_start",
        "POST",
        "/api/generate",
        _generate({"num_predict": 1, "temperature": 0}, companion="start"),
        "reply",
        frozenset({"companion"}),
    ),
    Case(
        "companion_stop",
        "POST",
        "/api/generate",
        {"model": MODEL, "prompt": "", "stream": False, "companion": "stop"},
        "reply",
        frozenset({"companion"}),
    ),
)

_BY_NAME = {case.name: case for case in CASES}


def case_by_name(name: str) -> Case:
    return _BY_NAME[name]


def covered_parameters() -> frozenset[str]:
    names: set[str] = set()
    for case in CASES:
        names.update(case.covers)
    return frozenset(names)


def sent_parameters() -> frozenset[str]:
    """Every option, body flag, and path the Ollama adapter can put on the wire.

    One body per settings branch. An option that only one branch adds still
    needs a fixture.
    """

    def names_of(settings: dict, **extra) -> set[str]:
        request = StepRequest("s", "generate", MODEL, "5090", "ollama", settings)
        body = generate_body(request)
        body.update(extra)
        found = set(body.get("options") or {})
        found.update(key for key in body if key not in {"model", "prompt", "options"})
        return found

    base = {
        "num_predict": 1,
        "temperature": 0,
        "num_ctx": 2048,
        "keep_alive": "5m",
        "prompt": SHORT_PROMPT,
    }
    names: set[str] = set()
    names.update(names_of({**base, "thinking_level": "off"}))
    names.update(names_of({**base, "thinking_level": "on"}))
    names.update(names_of({**base, "shift": False}))
    names.update(names_of({**base, "truncate": False}))
    names.update(names_of(dict(base)))
    names.update(names_of(dict(base), raw=True, truncate=False))
    names.update(names_of(dict(base), companion="start"))
    names.update({"stream", "companion"})
    names.update({"/api/ps", "/api/show", "/api/tags", "/api/version"})
    return frozenset(names)


def request_path(url: str) -> str:
    path = urlsplit(url).path
    return path or url


class RecordedNoReply(AssertionError):
    """The recorded exchange for this parameter does not answer."""


def _covered(name: str) -> bool:
    return name in covered_parameters()


def _no_reply(detail: str) -> None:
    raise RecordedNoReply(f"recorded behaviour is no reply ({detail})")


def require_recorded_behaviour(method: str, url: str, body) -> None:
    """Fail when this request's parameter has no fixture, or the fixture does not answer.

    An unload (``keep_alive`` 0, empty prompt, no options) and a companion stop
    are their own cases. They are not a generate with a missing ``num_predict``.
    """

    path = request_path(url)
    if path in {"/api/ps", "/api/tags", "/api/version", "/api/show"}:
        if not _covered(path):
            raise AssertionError(f"no fixture for {path}")
        return
    if method != "POST" or path != "/api/generate":
        raise AssertionError(f"no fixture for {method} {path}")
    if not isinstance(body, dict):
        raise AssertionError("no fixture for a generate without a body")
    if body.get("companion") == "stop" and "options" not in body:
        if not _covered("companion"):
            raise AssertionError("no fixture for companion")
        return
    if (
        body.get("keep_alive") == 0
        and body.get("prompt") == ""
        and "options" not in body
        and "companion" not in body
    ):
        if not _covered("keep_alive"):
            raise AssertionError("no fixture for keep_alive")
        return
    options = body.get("options")
    if not isinstance(options, dict) or "num_predict" not in options or options.get("num_predict") is None:
        _no_reply("num_predict missing or null")
    value = options["num_predict"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        _no_reply("num_predict 0")
    for key in options:
        if not _covered(key):
            raise AssertionError(f"no fixture for option {key}")
    for key in ("raw", "truncate", "shift", "think", "companion"):
        if key in body and not _covered(key):
            raise AssertionError(f"no fixture for {key}")


def request_body(case: Case, model: str) -> dict | None:
    """The case body, with the recorded model name filled in."""

    if case.body is None:
        return None
    body = json.loads(json.dumps(case.body))
    if isinstance(body, dict) and "model" in body:
        body["model"] = model
    return body


def fixture_document(
    case: Case,
    *,
    model: str,
    version: str,
    digest: str,
    date: str,
    behaviour: str,
    response,
) -> dict:
    """One fixture file. ``response`` is null when the call did not answer."""

    return {
        "ollama_version": version,
        "model": model,
        "digest": digest,
        "date": date,
        "case": case.name,
        "request": {"method": case.method, "path": case.path, "body": request_body(case, model)},
        "behaviour": behaviour,
        "response": response,
    }
