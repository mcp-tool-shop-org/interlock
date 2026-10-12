"""Record one Ollama exchange per case.

The Publisher runs this against the local server. Seat B does not. A call that
does not answer within the timeout is stored as ``no_reply`` and is not waited
out. ``-cloud`` models are refused. The card must be idle before the first
call and after every call that can load a model. Each request is written to
stderr as it is sent. The prompt is not.
"""

from __future__ import annotations

import http.client
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

from switchyard.adapters.base import AdapterError
from switchyard.adapters.ollama import refuse_cloud
from switchyard.adapters.ollama_cases import CASES, Case, fixture_document, request_body
from switchyard.guards import GuardError, require_card_empty, require_grant, require_heartbeat
from switchyard.harness.fill import MEMORY_ARGV, parse_mib

RECORD_TIMEOUT_S = 15.0
IDLE_CEILING_MIB = 4096
# Same bound as the fill-card settle: the card does not fall the instant keep_alive is 0.
SETTLE_S = 10.0
SETTLE_STEP_S = 0.2

# Captured while num_predict_1 is still resident, before that unload.
_WHILE_LOADED = "ps_loaded"


def _leaves_resident(case: Case) -> bool:
    """A generate can load a model. The keep_alive 0 unload shape does not.

    A companion stop is a generate. It has no options and an empty prompt, and
    it is not the unload: on the rig it loaded the model.
    """

    if case.method != "POST" or case.path != "/api/generate":
        return False
    body = case.body if isinstance(case.body, dict) else {}
    return not (
        body.get("keep_alive") == 0
        and body.get("prompt") in ("", None)
        and "companion" not in body
        and "options" not in body
    )


def _request_line(method: str, url: str, body) -> str:
    """One stderr line for a request. The prompt is not included."""

    path = urlsplit(url).path or url
    parts: list[str] = []
    if isinstance(body, dict):
        if body.get("keep_alive") == 0:
            parts.append("keep_alive=0")
        if "companion" in body:
            parts.append(f"companion={body.get('companion')}")
        options = body.get("options")
        if isinstance(options, dict) and "num_predict" in options:
            parts.append(f"num_predict={options.get('num_predict')!r}")
        if body.get("raw") is True:
            parts.append("raw")
        if "think" in body:
            parts.append(f"think={body.get('think')!r}")
        if "shift" in body:
            parts.append(f"shift={body.get('shift')!r}")
        if body.get("truncate") is False:
            parts.append("truncate=false")
    suffix = (" " + " ".join(parts)) if parts else ""
    return f"{method} {path}{suffix}"


def _log_request(method: str, url: str, body) -> None:
    sys.stderr.write(_request_line(method, url, body) + "\n")
    sys.stderr.flush()


class RecordError(Exception):
    """The recorder will not write a fixture."""


def refuse_model(model: str) -> None:
    try:
        refuse_cloud(model)
    except AdapterError as exc:
        raise RecordError(str(exc)) from exc


def refuse_unless_baseline(models, card_mib: int | None, *, ceiling: int = IDLE_CEILING_MIB) -> None:
    """Refuse when a model is loaded or the card is above the idle ceiling."""

    if isinstance(models, list) and models:
        raise RecordError("the card is not at baseline: a model is loaded")
    if isinstance(card_mib, bool) or not isinstance(card_mib, int):
        raise RecordError("the card is not at baseline: memory could not be read")
    if card_mib > ceiling:
        raise RecordError(f"the card is not at baseline: {card_mib} MiB is above {ceiling} MiB")


def read_card_mib(run=None) -> int | None:
    """One nvidia-smi memory reading, in MiB. A failed read is None."""

    import subprocess

    launch = subprocess.run if run is None else run
    try:
        completed = launch(list(MEMORY_ARGV), capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return parse_mib(getattr(completed, "stdout", None))


def http_exchange(method: str, url: str, body: dict | None, timeout: float):
    """One local HTTP call. None means the server did not answer within ``timeout``."""

    parts = urlsplit(url)
    if parts.scheme != "http" or not parts.hostname:
        raise RecordError("the recorder only opens local http")
    path = parts.path or "/"
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data is not None else {}
    conn = http.client.HTTPConnection(parts.hostname, parts.port or 80, timeout=timeout)
    try:
        conn.request(method, path, body=data, headers=headers)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
    except TimeoutError:
        return None
    except (OSError, http.client.HTTPException, json.JSONDecodeError, UnicodeError) as exc:
        raise RecordError("the request failed") from exc
    finally:
        try:
            conn.close()
        except OSError:
            pass
    if not isinstance(payload, dict):
        raise RecordError("Ollama returned a non-object")
    return payload


def _observed(case: Case, payload) -> tuple[str, dict | None]:
    """What this call did.

    A timeout is ``no_reply`` with no body. A catalogued no-limit generate that
    answers inside the timeout stays ``no_reply``, so a fake still refuses it,
    and the reply is kept so ``eval_count`` is on the fixture.
    """

    if payload is None:
        return "no_reply", None
    if case.behaviour == "no_reply":
        return "no_reply", payload if isinstance(payload, dict) else None
    if isinstance(payload, dict) and payload.get("error"):
        return "error", payload
    return "reply", payload if isinstance(payload, dict) else None


def _unload_body(model: str) -> dict:
    return {"model": model, "prompt": "", "stream": False, "keep_alive": 0}


def _guards(grant: str | None, heartbeat: Path, now: float) -> None:
    """Grant and a fresh heartbeat. Neither call touches Ollama."""

    try:
        require_grant(grant, uses_5090=True)
        require_heartbeat(heartbeat, now)
    except GuardError as exc:
        raise RecordError(str(exc)) from exc


def _require_empty(models) -> None:
    loaded = models if isinstance(models, list) else ["unreadable"]
    try:
        require_card_empty(loaded)
    except GuardError as exc:
        raise RecordError(str(exc)) from exc


def record_fixtures(
    *,
    model: str,
    host: str,
    out_dir: Path,
    timeout: float,
    ceiling: int,
    today: str,
    transport,
    card_mib,
    grant: str | None,
    heartbeat: Path,
    now: float,
    sleep=None,
) -> list[Path]:
    """Write one JSON file per case under ``out_dir/<ollama-version>/``.

    ``transport(method, url, body, timeout)`` returns a dict, or None when the
    call does not answer. ``card_mib()`` is the card reading in MiB. The grant
    and the heartbeat are checked before any request. The first request is the
    empty-card probe. A generate other than the unload shape, including a
    companion stop, is a load: the unload is sent and the card is polled before
    the next idle check. A refusal after that load still sends the unload.
    Every request is written to stderr.
    """

    if timeout <= 0:
        raise RecordError("timeout must be a positive number of seconds")
    _guards(grant, heartbeat, now)
    refuse_model(model)
    host = host.rstrip("/")
    refuse_unless_baseline([], card_mib(), ceiling=ceiling)

    def call(method, url, body, timeout):
        _log_request(method, url, body)
        return transport(method, url, body, timeout)

    ps = call("GET", f"{host}/api/ps", None, timeout)
    models = ps.get("models") if isinstance(ps, dict) else None
    _require_empty(models)

    version_payload = call("GET", f"{host}/api/version", None, timeout)
    version = version_payload.get("version") if isinstance(version_payload, dict) else None
    if not isinstance(version, str) or not version.strip():
        raise RecordError("Ollama did not report a version")
    version = version.strip()

    tags = call("GET", f"{host}/api/tags", None, timeout)
    digest = _digest(tags, model)
    folder = out_dir / version
    folder.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    def capture(case: Case) -> None:
        started = time.perf_counter()
        payload = call(case.method, f"{host}{case.path}", request_body(case, model), timeout)
        elapsed = time.perf_counter() - started
        behaviour, response = _observed(case, payload)
        document = fixture_document(
            case,
            model=model,
            version=version,
            digest=digest,
            date=today,
            behaviour=behaviour,
            response=response,
        )
        if case.name == "num_predict_1":
            document["cold_load_s"] = round(elapsed, 3)
        path = folder / f"{case.name}.json"
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
        written[case.name] = path

    pause = time.sleep if sleep is None else sleep
    resident = False

    def baseline_again() -> None:
        """Immediate check. Used only before the first load, when the card should already be idle."""

        refuse_unless_baseline([], card_mib(), ceiling=ceiling)
        again = call("GET", f"{host}/api/ps", None, timeout)
        loaded = again.get("models") if isinstance(again, dict) else ["unreadable"]
        _require_empty(loaded)

    def send_unload() -> None:
        payload = call("POST", f"{host}/api/generate", _unload_body(model), timeout)
        if payload is None:
            raise RecordError("the unload timed out; the model may still be resident")

    def settle() -> None:
        """Poll after the unload. Refuse only once the settle bound has passed."""

        waited = 0.0
        while True:
            reading = card_mib()
            again = call("GET", f"{host}/api/ps", None, timeout)
            models = again.get("models") if isinstance(again, dict) else None
            card_ok = isinstance(reading, int) and not isinstance(reading, bool) and reading <= ceiling
            empty = isinstance(models, list) and len(models) == 0
            if card_ok and empty:
                return
            if waited >= SETTLE_S:
                refuse_unless_baseline([], reading, ceiling=ceiling)
                _require_empty(models)
                return
            pause(SETTLE_STEP_S)
            waited += SETTLE_STEP_S

    def unload() -> None:
        """Send the unload first, then wait for the card and an empty `/api/ps`."""

        nonlocal resident
        send_unload()
        # The unload is on the wire. A later refusal must not depend on settle
        # finishing before release() will skip a second send.
        resident = False
        settle()

    def release() -> None:
        """Best-effort unload so a refusal does not leave the model resident."""

        nonlocal resident
        if not resident:
            return
        resident = False
        try:
            call("POST", f"{host}/api/generate", _unload_body(model), timeout)
        except Exception:
            return

    def capture_loaded(case: Case) -> None:
        nonlocal resident
        resident = True
        capture(case)

    quiet = [case for case in CASES if not _leaves_resident(case) and case.name != _WHILE_LOADED]
    try:
        for case in quiet:
            capture(case)
        baseline_again()
        capture_loaded(case_by_loaded())
        capture(case_named(_WHILE_LOADED))
        unload()
        for case in CASES:
            if case.name in written:
                continue
            if _leaves_resident(case):
                capture_loaded(case)
                unload()
            else:
                capture(case)
        missing = [case.name for case in CASES if case.name not in written]
        if missing:
            raise RecordError(f"fixtures were not written: {', '.join(missing)}")
        return [written[case.name] for case in CASES]
    except BaseException:
        release()
        raise


def case_named(name: str) -> Case:
    for case in CASES:
        if case.name == name:
            return case
    raise RecordError(f"unknown case {name}")


def case_by_loaded() -> Case:
    return case_named("num_predict_1")


def _digest(payload, model: str) -> str:
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise RecordError("Ollama tags did not list models")
    for item in payload["models"]:
        if not isinstance(item, dict):
            continue
        label = item.get("name") or item.get("model")
        if label != model:
            continue
        digest = item.get("digest")
        if isinstance(digest, str) and digest.strip():
            return digest.strip()
        raise RecordError(f"{model} has no digest in /api/tags")
    raise RecordError(f"{model} is not in /api/tags")


