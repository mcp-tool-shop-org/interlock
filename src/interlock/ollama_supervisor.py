"""Restart a hung local Ollama between steps.

Health is ``GET /api/version`` on ``127.0.0.1:11434`` (the 5090) and
``127.0.0.1:11490`` (CPU only). A restart runs only after a step has
ended, and only when that step failed with a transport error, a timeout
that is a real hang, or a 5xx while the server is down. A long request
whose version still answers, and whose model is still listed, is left
alone.

Only port 11434 is restarted. Port 11490 is offrig's CPU indexer, and no
grant covers it. Health still reports it. A down 11490 stops the run.

The process that is stopped is the one listening on 11434, and only when
that port has exactly one listener whose executable is the Ollama install
``serve`` starts. Zero, or two or more, or a path that is missing or is
not that install, refuses and stops nothing. Nothing is killed by name.
The other port's server is not touched, and 11490 is not queried on the
way to that decision. The tray app is not touched unless it is the
listener. Immediately before that stop, the restart re-checks the grant,
the watchdog heartbeat, and that ``/api/ps`` lists no model from another
session. The record keeps the pid and the path that was checked.

``ollama serve`` is replayed from the start line recorded at run start:
the argv, every ``OLLAMA_*`` and ``LLAMA_ARG_*`` variable, and the log
file. Stdout is appended to that log. interlock does not sign in. There
is no ``OLLAMA_CUDA_BACKEND`` variable. The CUDA check is the sha256 of
the loaded ``ggml-cuda`` library and the ``load_backend`` line written
after the spawn. At prime, the record's ``OLLAMA_*`` and ``LLAMA_ARG_*``
set is compared with the live process environment, or with the server
config line when that environment cannot be read. A mismatch means this
run does not restart. After a restart the same set is read from the new
log segment, not from the two fields alone.

Tests pass fakes. The command's seams are the only ones that can spawn
a process, and a test never calls them.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urlsplit

from interlock.guards import GuardError, require_grant, require_heartbeat

URL_5090 = "http://127.0.0.1:11434"
URL_CPU = "http://127.0.0.1:11490"
LOCAL_PORTS = (11434, 11490)
SIGNIN_NOTE = "interlock does not sign in. A person has to start Ollama."
UNRECORDED = "unrecorded"

_PID_IN_SS = re.compile(r"pid=(\d+)")
_STATUS_5XX = re.compile(r"\b(?:status|http)\s+5\d\d\b")
_BACKEND_LINE = re.compile(r"load_backend: loaded CUDA backend from \S+")
_ENV_IN_LOG = re.compile(r"\b(OLLAMA_[A-Z0-9_]+|LLAMA_ARG_[A-Z0-9_]+)[=:]([^\s,\]]+)")
SERVE_RECORD_ENV = "INTERLOCK_OLLAMA_SERVE"
_OLD_ENV_WARNED: set[str] = set()


def read_project_env(name: str) -> str | None:
    """Read an ``INTERLOCK_*`` variable. ``SWITCHYARD_*`` still works for one release."""

    value = os.environ.get(name)
    if isinstance(value, str) and value.strip():
        return value
    if not name.startswith("INTERLOCK_"):
        return None
    old = "SWITCHYARD_" + name[len("INTERLOCK_") :]
    value = os.environ.get(old)
    if not isinstance(value, str) or not value.strip():
        return None
    if old not in _OLD_ENV_WARNED:
        _OLD_ENV_WARNED.add(old)
        print(f"warning: {old} is deprecated; set {name}", file=sys.stderr)
    return value


class SupervisorStop(Exception):
    """The run stops. The note is for a person."""


class RestartRefused(Exception):
    """A restart was asked for during a measurement, so nothing was stopped."""


@dataclass(frozen=True)
class SupervisorConfig:
    """The bounds. A run may pass its own; the defaults are the brief's."""

    health_timeout_s: float = 5.0
    down_failures: int = 3
    down_gap_s: float = 2.0
    restart_deadline_s: float = 60.0
    max_restarts: int = 2


@dataclass(frozen=True)
class Health:
    ok: bool
    version: str | None
    error: str | None


@dataclass(frozen=True)
class ServeStart:
    """The start line recorded at run start and replayed exactly.

    ``env`` keeps ``OLLAMA_*`` and ``LLAMA_ARG_*`` only. ``library_path`` is
    the loaded ``ggml-cuda`` library whose sha256 is compared after a restart.
    """

    argv: tuple[str, ...]
    env: dict[str, str]
    log_path: str
    library_path: str = ""
    ggml_cuda_sha256: str = ""
    backend_line: str = ""


@dataclass(frozen=True)
class ServerFingerprint:
    """What the run recorded, and what a restart must still match.

    ``ggml_cuda_sha256`` is the loaded CUDA library, not the driver version
    from nvidia-smi. ``backend_line`` is the server log's ``load_backend``
    line. Flash attention and the KV cache type are the start line's values,
    read again from the log after the restart.
    """

    version: str
    flash_attention: str
    kv_cache_type: str
    ggml_cuda_sha256: str
    backend_line: str
    baseline_mib: int | None = None
    model: str | None = None

    def serve_env(self, port: int) -> dict[str, str]:
        """Flash attention and the KV cache type, when the run recorded them.

        The restart itself replays :class:`ServeStart`. This env is the
        recorded pair, not a stand-in for the start line. An unrecorded
        value is left unset. There is no ``OLLAMA_CUDA_BACKEND``.
        """

        env = {"OLLAMA_HOST": f"127.0.0.1:{int(port)}"}
        if _recorded(self.flash_attention):
            env["OLLAMA_FLASH_ATTENTION"] = self.flash_attention
        if _recorded(self.kv_cache_type):
            env["OLLAMA_KV_CACHE_TYPE"] = self.kv_cache_type
        return env


@dataclass(frozen=True)
class RestartRecord:
    when: str
    port: int
    why: str
    old_pid: int | None
    new_pid: int | None
    seconds_to_healthy: float | None
    version: str | None
    note: str | None = None
    checked_path: str | None = None

    def as_dict(self) -> dict:
        return {
            "when": self.when,
            "port": self.port,
            "why": self.why,
            "old_pid": self.old_pid,
            "new_pid": self.new_pid,
            "seconds_to_healthy": self.seconds_to_healthy,
            "version": self.version,
            "note": self.note,
            "checked_path": _receipt_path(self.checked_path),
        }


@dataclass(frozen=True)
class Decision:
    """``restarted`` carries a record. ``leave`` does not stop or start anything."""

    action: str
    record: RestartRecord | None = None
    note: str = ""


@dataclass(frozen=True)
class Process:
    """One row of a process table. Only a listening owner of the port is stopped."""

    pid: int
    name: str
    port: int | None
    listening: bool = False


@dataclass
class SupervisorSeams:
    """Every effect. Tests fill these in. The command uses :func:`real_seams`."""

    health_get: object
    processes: object
    stop: object
    spawn: object
    card_mib: object
    ps: object
    environment: object
    argv: object
    clock: object = time.monotonic
    sleep: object = time.sleep
    now: object = None
    wall: object = time.time
    load_start: object = None
    process_env: object = None
    log_size: object = None
    log_segment: object = None
    listening: object = None
    executable: object = None
    localappdata: object = None
    platform: object = None

    def __post_init__(self) -> None:
        if self.now is None:
            self.now = _now


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _recorded(value: str | None) -> bool:
    return isinstance(value, str) and bool(value) and value != UNRECORDED


def retry_budget(spec: dict) -> int:
    """How many times a failed step may be tried again. Unset means none."""

    value = spec.get("retry") if isinstance(spec, dict) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def failure_kind(error: str | None, *, status: int | None = None) -> str | None:
    """``transport``, ``timeout``, or ``server`` (a 5xx). Anything else is not a restart."""

    if isinstance(status, int) and not isinstance(status, bool) and 500 <= status <= 599:
        return "server"
    if not isinstance(error, str) or not error:
        return None
    text = error.lower()
    if "timed out" in text or "timeout" in text:
        return "timeout"
    if _STATUS_5XX.search(text) or "server error" in text:
        return "server"
    if (
        "the request was closed" in text
        or "connection refused" in text
        or "connection reset" in text
        or "unreachable" in text
        or "failed to establish" in text
    ):
        return "transport"
    return None


def port_of(url: str) -> int:
    parts = urlsplit(url)
    if parts.port not in LOCAL_PORTS:
        raise SupervisorStop(f"Ollama restart refuses {url}")
    return int(parts.port)


def owners(processes: list[Process], port: int) -> list[Process]:
    """Listeners on ``port``. A matching name that is not listening is not an owner."""

    found = []
    for row in processes:
        if row.listening and row.port == port and isinstance(row.pid, int) and row.pid > 0:
            found.append(row)
    return found


def ollama_env(raw: dict | None) -> dict[str, str]:
    """``OLLAMA_*`` and ``LLAMA_ARG_*`` only. Other variables are not replayed."""

    if not isinstance(raw, dict):
        return {}
    kept: dict[str, str] = {}
    for key, value in raw.items():
        name = str(key)
        if name.startswith("OLLAMA_") or name.startswith("LLAMA_ARG_"):
            kept[name] = "" if value is None else str(value)
    return kept


def environment_mismatch(
    recorded: ServerFingerprint,
    current: ServerFingerprint,
    *,
    check_backend: bool = True,
) -> str | None:
    """The field that moved, or None when the restart is still the same server.

    The driver CUDA version is not compared. A restart cannot change it, so
    that check would always pass. ``check_backend`` is false when the new
    log segment has no ``load_backend`` line yet: the line from before the
    spawn is not a match, and the check waits for the first model load.
    """

    pairs = [
        ("version", recorded.version, current.version),
        ("flash_attention", recorded.flash_attention, current.flash_attention),
        ("kv_cache_type", recorded.kv_cache_type, current.kv_cache_type),
        ("CUDA library sha", recorded.ggml_cuda_sha256, current.ggml_cuda_sha256),
    ]
    if check_backend:
        pairs.append(("backend line", recorded.backend_line, current.backend_line))
    for name, before, after in pairs:
        if before != after:
            return (
                f"the Ollama {name} changed across the restart ({before} to {after}); "
                "the run stops. A restart that changes the backend is not measurement-neutral."
            )
    return None


def foreign_models(payload: object, model: str | None) -> list[str]:
    """Models ``/api/ps`` lists that are not this run's. An unreadable payload lists none.

    A down server does not answer, and that is not another session. A payload
    that answers with a different model is another session.
    """

    if not isinstance(payload, dict):
        return []
    models = payload.get("models")
    if not isinstance(models, list):
        return []
    foreign: list[str] = []
    for item in models:
        name = ""
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            raw = item.get("name") or item.get("model") or ""
            name = raw if isinstance(raw, str) else ""
        if not name:
            continue
        if model and model_listed({"models": [item if isinstance(item, dict) else {"name": name}]}, model):
            continue
        foreign.append(name)
    return foreign


def backend_line_from_log(text: str | None) -> str:
    """The last ``load_backend`` line, or empty when the log does not have one."""

    if not isinstance(text, str) or not text:
        return ""
    found = _BACKEND_LINE.findall(text)
    return found[-1] if found else ""


def env_from_log(text: str | None) -> dict[str, str]:
    """``OLLAMA_*`` and ``LLAMA_ARG_*`` values written in a server log."""

    if not isinstance(text, str) or not text:
        return {}
    return {key: value for key, value in _ENV_IN_LOG.findall(text)}


def log_byte_size(path: str) -> int:
    """Bytes in the server log. A missing log is empty, and the offset is zero.

    Any other read error propagates. The restart must not treat it as an empty log.
    """

    if not isinstance(path, str) or not path.strip():
        return 0
    try:
        return Path(path).stat().st_size
    except FileNotFoundError:
        return 0


def read_log_from(path: str, offset: int) -> str:
    """The log text written at and after ``offset``. The offset is a byte count."""

    if not isinstance(path, str) or not path.strip():
        return ""
    start = offset if isinstance(offset, int) and not isinstance(offset, bool) and offset > 0 else 0
    try:
        with open(path, "rb") as handle:
            handle.seek(start)
            data = handle.read()
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")


def _unescape_log_value(value: str) -> str:
    """Go's quoted server-config line writes a backslash as two."""

    return value.replace("\\\\", "\\").replace('\\"', '"')


def _parse_go_map(body: str) -> dict[str, str]:
    """One Go ``map[k:v]`` body. A bracketed value may contain spaces."""

    found: dict[str, str] = {}
    index = 0
    length = len(body)
    while index < length:
        while index < length and body[index].isspace():
            index += 1
        if index >= length:
            break
        colon = body.find(":", index)
        if colon < 0:
            break
        key = body[index:colon]
        if not key or any(char.isspace() for char in key):
            break
        index = colon + 1
        if index < length and body[index] == "[":
            depth = 1
            index += 1
            start = index
            while index < length and depth:
                if body[index] == "[":
                    depth += 1
                elif body[index] == "]":
                    depth -= 1
                    if depth == 0:
                        break
                index += 1
            found[key] = _unescape_log_value(body[start:index])
            if index < length and body[index] == "]":
                index += 1
            continue
        start = index
        while index < length:
            if body[index].isspace():
                nxt = index + 1
                while nxt < length and body[nxt].isspace():
                    nxt += 1
                end = nxt
                while end < length and (body[end].isalnum() or body[end] == "_"):
                    end += 1
                if end > nxt and end < length and body[end] == ":":
                    break
            index += 1
        found[key] = _unescape_log_value(body[start:index].rstrip())
    return found


def server_config_env(text: str | None) -> dict[str, str]:
    """``OLLAMA_*`` and ``LLAMA_ARG_*`` from the last ``server config`` line.

    The line is Ollama's ``env="map[...]"`` log. An older line in the same
    file is not this server. A segment that has not logged one yet is empty.
    """

    if not isinstance(text, str) or not text:
        return {}
    body = ""
    for line in text.splitlines():
        if 'msg="server config"' not in line:
            continue
        mark = 'env="map['
        start = line.find(mark)
        if start < 0:
            continue
        start += len(mark)
        end = line.rfind(']"')
        if end < start:
            continue
        body = line[start:end]
    if not body:
        return {}
    return ollama_env(_parse_go_map(body))


def _env_token(key: str, value: str | None) -> str | None:
    """Host values match with or without the ``http://`` scheme."""

    if value is None:
        return None
    if key == "OLLAMA_HOST":
        for prefix in ("http://", "https://"):
            if value.startswith(prefix):
                return value[len(prefix):]
    return value


def ollama_env_mismatch(expected: dict, actual: dict, *, allow_extra: bool, across_restart: bool) -> str | None:
    """The first ``OLLAMA_*`` or ``LLAMA_ARG_*`` key that differs, or None.

    ``allow_extra`` is for a process environment compared with a server
    config line: Ollama logs defaults the process did not set. A key the
    run was measured under still has to match. Extra keys are not ignored
    when both sides are the same kind of record.
    """

    left = ollama_env(expected)
    right = ollama_env(actual)
    for key in sorted(set(left) | set(right)):
        if allow_extra and key not in left:
            continue
        if _env_token(key, left.get(key)) == _env_token(key, right.get(key)):
            continue
        before = left.get(key)
        after = right.get(key)
        if across_restart:
            return (
                f"the Ollama environment changed across the restart ({key}: {before!r} to {after!r}); "
                "the run stops. A restart that changes the environment is not measurement-neutral."
            )
        return (
            f"the Ollama start record does not match the live server ({key}: {before!r} to {after!r}). "
            "This run makes no restarts."
        )
    return None


def file_sha256(path: str | Path) -> str:
    """Sha256 of a file. The caller catches a missing library."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _log_text(path: str) -> str:
    """The server log, or empty when it cannot be read. The path is local."""

    if not isinstance(path, str) or not path.strip():
        return ""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _overlay_start(found: ServerFingerprint, start: ServeStart | None) -> ServerFingerprint:
    """Fill empty before-side fields from the start line. Never a stand-in for a later read."""

    if start is None:
        return found
    flash = found.flash_attention
    kv = found.kv_cache_type
    sha = found.ggml_cuda_sha256
    line = found.backend_line
    if not _recorded(flash):
        flash = start.env.get("OLLAMA_FLASH_ATTENTION") or flash
    if not _recorded(kv):
        kv = start.env.get("OLLAMA_KV_CACHE_TYPE") or kv
    if not _recorded(sha):
        sha = start.ggml_cuda_sha256 or sha
    if not _recorded(line):
        line = start.backend_line or line
    return replace(
        found,
        flash_attention=flash or "",
        kv_cache_type=kv or "",
        ggml_cuda_sha256=sha or "",
        backend_line=line or "",
    )


def load_serve_start(path: str | Path | None = None) -> ServeStart | None:
    """The Publisher's start-line record. Unset means no record, and no restart.

    The path is ``INTERLOCK_OLLAMA_SERVE`` when ``path`` is omitted.
    ``SWITCHYARD_OLLAMA_SERVE`` still works for one release and warns.
    Nothing is fetched. A missing file is no record.
    """

    chosen = path if path is not None else read_project_env(SERVE_RECORD_ENV)
    if not chosen:
        return None
    file = Path(chosen)
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    argv = data.get("argv")
    if not isinstance(argv, list) or not argv:
        return None
    return ServeStart(
        argv=tuple(str(part) for part in argv),
        env=ollama_env(data.get("env") if isinstance(data.get("env"), dict) else {}),
        log_path=str(data.get("log_path") or ""),
        library_path=str(data.get("library_path") or ""),
        ggml_cuda_sha256=str(data.get("ggml_cuda_sha256") or ""),
        backend_line=str(data.get("backend_line") or ""),
    )


def health(url: str, *, timeout: float = 5.0, get=None) -> Health:
    """Healthy when ``GET /api/version`` answers with a version inside ``timeout``."""

    getter = get or http_get_version
    try:
        status, payload, error = getter(url, timeout)
    except Exception as exc:
        return Health(False, None, str(exc) or "the version request failed")
    if error:
        return Health(False, None, error)
    if status != 200 or not isinstance(payload, dict):
        return Health(False, None, f"status {status}")
    version = payload.get("version")
    if not isinstance(version, str) or not version.strip():
        return Health(False, None, "the version response had no version")
    return Health(True, version.strip(), None)


def health_report(get=None, *, timeout: float = 5.0) -> tuple[str, bool]:
    """One line per local port. The bool is true when both answered."""

    lines = []
    ok = True
    for url in (URL_5090, URL_CPU):
        found = health(url, timeout=timeout, get=get)
        if found.ok:
            lines.append(f"{url} healthy {found.version}")
        else:
            ok = False
            lines.append(f"{url} down {found.error}")
    return "\n".join(lines) + "\n", ok


def restart_policy_lines(config: SupervisorConfig | None = None) -> list[str]:
    """What a dry-run says about restarts. Nothing is contacted."""

    chosen = config or SupervisorConfig()
    return [
        "ollama restart: between steps only, never during a measurement",
        f"ollama restart: at most {chosen.max_restarts} per run",
        f"ollama restart: health is GET /api/version within {chosen.health_timeout_s:g} s",
        (
            f"ollama restart: down is {chosen.down_failures} failed health checks, "
            f"{chosen.down_gap_s:g} s apart"
        ),
        (
            "ollama restart: a request past its deadline restarts only when health then fails "
            "or /api/ps shows no progress"
        ),
        "ollama restart: a slow request whose version answers and whose model is listed is left alone",
        "ollama restart: only 127.0.0.1:11434 is restarted; 11490 is reported and never stopped",
        "ollama restart: the grant, the heartbeat, and /api/ps are re-checked before anything is stopped",
        "ollama restart: the recorded start line is replayed, and the log is appended",
        "ollama restart: the CUDA library sha, the backend line written after the spawn, and the full OLLAMA_* set "
            "must match",
        "ollama restart: the first measured step after a restart stays out of the warm percentiles",
        "ollama restart: the tray app is not stopped unless it owns the port",
        f"ollama restart: wait up to {chosen.restart_deadline_s:g} s to become healthy, then the card must be back at "
            f"baseline",
        "ollama restart: interlock does not sign in",
    ]


def model_listed(payload: object, model: str) -> bool:
    """True when ``/api/ps`` lists ``model``. An unreadable payload is not a listing."""

    if not isinstance(payload, dict):
        return False
    models = payload.get("models")
    if not isinstance(models, list):
        return False
    for item in models:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("model")
        if isinstance(name, str) and (name == model or name.startswith(model + ":")):
            return True
    return False


def pids_from_windows(text: str) -> list[int]:
    """OwningProcess lines from ``Get-NetTCPConnection ... | Select-Object -ExpandProperty``."""

    found = []
    for line in text.splitlines():
        token = line.strip()
        if token.isdigit():
            pid = int(token)
            if pid > 0 and pid not in found:
                found.append(pid)
    return found


def pids_from_ss(text: str) -> list[int]:
    """``pid=`` fields from ``ss -ltnp``."""

    found = []
    for match in _PID_IN_SS.finditer(text):
        pid = int(match.group(1))
        if pid > 0 and pid not in found:
            found.append(pid)
    return found


def pids_from_psutil(port: int, module) -> list[int]:
    """Listen PIDs from a psutil-shaped module. The real module is not required."""

    found = []
    try:
        connections = module.net_connections(kind="tcp")
    except Exception:
        return []
    for conn in connections:
        status = getattr(conn, "status", None)
        address = getattr(conn, "laddr", None)
        pid = getattr(conn, "pid", None)
        if status != "LISTEN" or address is None or not isinstance(pid, int) or pid <= 0:
            continue
        if getattr(address, "port", None) != port:
            continue
        if pid not in found:
            found.append(pid)
    return found


def listening_pids(port: int, *, run=None, platform: str | None = None, psutil_module=None) -> list[int]:
    """PIDs in Listen on ``port``.

    Windows asks ``Get-NetTCPConnection -LocalPort <port>`` for the socket with no remote port.
    It doesn't filter on ``-State Listen``: on Windows build 26340 a listening socket reports
    state 0, so that filter finds nothing.
    Linux asks ``ss -ltnp`` and, when that names nobody, psutil.
    """

    system = platform or sys.platform
    number = int(port)
    if system == "win32":
        script = (
            "Get-NetTCPConnection -LocalPort "
            + str(number)
            + " | Where-Object { $_.RemotePort -eq 0 } | Select-Object -ExpandProperty OwningProcess"
        )
        text = _capture(run, ["powershell", "-NoProfile", "-Command", script])
        return pids_from_windows(text or "")
    text = _capture(run, ["ss", "-ltnp", f"sport = :{number}"])
    found = pids_from_ss(text or "")
    if found:
        return found
    module = psutil_module if psutil_module is not None else _load_psutil()
    return pids_from_psutil(number, module)


def stop_pid(pid: int, *, run=None, platform: str | None = None) -> None:
    """Stop this pid. There is no name-based kill."""

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise SupervisorStop("refusing to stop a process that is not a listening pid")
    system = platform or sys.platform
    if system == "win32":
        argv = ["taskkill", "/PID", str(pid), "/F"]
    else:
        argv = ["kill", str(pid)]
    _capture(run, argv)


def serve_argv(platform: str | None = None, *, localappdata: str | None = None) -> list[str]:
    """``ollama serve``. On Windows that is the per-user install. Never ``signin``."""

    system = platform or sys.platform
    if system == "win32":
        root = localappdata if localappdata is not None else os.environ.get("LOCALAPPDATA", "")
        exe = str(Path(root) / "Programs" / "Ollama" / "ollama.exe")
    else:
        exe = "ollama"
    return [exe, "serve"]


# A home directory, not a particular account. Receipts refuse the prefix.
_HOME_PREFIX = re.compile(
    r"(?i)(?:[A-Za-z]:[\\/]+Users[\\/]+[^\\/]+|[\\/]Users[\\/]+[^\\/]+|[\\/]home[\\/]+[^\\/]+)"
)


def _receipt_path(path: str | None) -> str | None:
    """The path a receipt may store. A home prefix becomes ``~``."""

    if not isinstance(path, str):
        return None
    return _HOME_PREFIX.sub("~", path)


def executables_match(found: str, expected: str, *, platform: str) -> bool:
    """Whether ``found`` is the executable ``expected`` names.

    Windows compares the resolved paths without regard to case. Elsewhere a
    bare expected name matches the file name, and is not resolved against
    the working directory. An absolute expected path has to match exactly.
    """

    if platform != "win32" and expected == Path(expected).name:
        return Path(found).name == expected
    left = str(Path(found).resolve())
    right = str(Path(expected).resolve())
    if platform == "win32":
        return left.replace("\\", "/").casefold() == right.replace("\\", "/").casefold()
    return left == right


def pid_executable(pid: int, psutil_module=None) -> str | None:
    """The executable path of ``pid``. A path that cannot be read is None."""

    module = psutil_module if psutil_module is not None else _load_psutil()
    try:
        path = module.Process(pid).exe()
    except Exception:
        return None
    if not isinstance(path, str) or not path.strip():
        return None
    return path


def spawn_detached(
    argv: list[str],
    env: dict,
    log_path: str,
    *,
    popen=None,
    platform: str | None = None,
) -> int:
    """Start ``argv`` in its own session, appending stdout and stderr to ``log_path``.

    The caller supplies ``popen`` in tests. The log is never discarded.
    """

    if any("signin" in str(part).lower() for part in argv):
        raise SupervisorStop(SIGNIN_NOTE)
    if not isinstance(log_path, str) or not log_path.strip():
        raise SupervisorStop("the recorded Ollama start line has no log file. " + SIGNIN_NOTE)
    try:
        handle = open(log_path, "a", encoding="utf-8")
    except OSError as exc:
        raise SupervisorStop(
            f"the Ollama log could not be opened ({exc}). " + SIGNIN_NOTE
        ) from exc
    system = platform or sys.platform
    flags: dict = {"stdout": handle, "stderr": handle, "env": env}
    if system == "win32":
        flags["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0x00000008) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200
        )
    else:
        flags["start_new_session"] = True
    launcher = popen or subprocess.Popen
    try:
        proc = launcher(list(argv), **flags)
    except Exception:
        handle.close()
        raise
    pid = getattr(proc, "pid", None)
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        handle.close()
        raise SupervisorStop("ollama serve did not return a pid. " + SIGNIN_NOTE)
    return pid


def card_mib_from_text(text: str | None) -> int | None:
    """One nvidia-smi memory figure, in MiB."""

    if not isinstance(text, str):
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        return None
    token = lines[0].replace("MiB", "").replace(",", "").split()[0]
    try:
        value = float(token)
    except ValueError:
        return None
    if value < 0:
        return None
    return int(value)


def read_card_mib(*, run=None) -> int | None:
    completed = _run(run, ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    if completed is None or getattr(completed, "returncode", 1) != 0:
        return None
    return card_mib_from_text(getattr(completed, "stdout", None))


def read_cuda(*, run=None) -> str | None:
    completed = _run(run, ["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader,nounits"])
    if completed is None or getattr(completed, "returncode", 1) != 0:
        return None
    text = getattr(completed, "stdout", None)
    if not isinstance(text, str):
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        return None
    return lines[0]


def http_get(url: str, path: str, timeout: float, *, connection=None):
    """GET one local path. A test passes ``connection`` and opens nothing."""

    parts = urlsplit(url)
    if parts.scheme != "http" or parts.hostname not in {"127.0.0.1", "localhost"}:
        return None, None, "Ollama health only opens local http"
    if not path.startswith("/"):
        return None, None, "Ollama health only opens local http"
    factory = connection or http.client.HTTPConnection
    conn = factory(parts.hostname, parts.port or 80, timeout=timeout)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        status = getattr(response, "status", None)
        raw = response.read()
    except Exception as exc:
        return None, None, str(exc) or "the version request failed"
    finally:
        closer = getattr(conn, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:  # noqa: S110  # a failed close still returns the response
                pass
    if not isinstance(raw, (bytes, bytearray)):
        return status, None, "the version response was not json"
    try:
        payload = json.loads(bytes(raw).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return status, None, "the version response was not json"
    if not isinstance(payload, dict):
        return status, None, "the version response was not an object"
    return status, payload, None


def http_get_version(url: str, timeout: float, *, connection=None):
    """``GET /api/version``. Only loopback."""

    return http_get(url, "/api/version", timeout, connection=connection)


def _load_psutil():
    import psutil

    return psutil


def environ_of(pid: int, module) -> dict | None:
    """``OLLAMA_*`` and ``LLAMA_ARG_*`` from a process table. A refused read is None."""

    try:
        env = module.Process(pid).environ()
    except Exception:
        return None
    if not isinstance(env, dict):
        return None
    return ollama_env(env)


def _capture(run, argv: list[str]) -> str | None:
    completed = _run(run, argv)
    if completed is None or getattr(completed, "returncode", 1) != 0:
        return None
    stdout = getattr(completed, "stdout", None)
    return stdout if isinstance(stdout, str) else None


def _run(run, argv: list[str]):
    runner = run or subprocess.run
    try:
        return runner(argv, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None


class OllamaSupervisor:
    """Hang detection and the restart. Effects go through :class:`SupervisorSeams`."""

    def __init__(
        self,
        seams: SupervisorSeams,
        fingerprint: ServerFingerprint | None = None,
        config: SupervisorConfig | None = None,
        *,
        grant: str | None = None,
        heartbeat: Path | str | None = None,
        start: ServeStart | None = None,
    ) -> None:
        self.seams = seams
        self.config = config or SupervisorConfig()
        self._recorded = fingerprint
        self.restarts: list[RestartRecord] = []
        self._open = False
        self._epoch = 0
        self._cold_after = False
        self.grant = grant
        self.heartbeat = None if heartbeat is None else Path(heartbeat)
        self.start = start
        self._env_checked = False
        self._restarts_disabled: str | None = None
        self._measured_env: dict | None = None
        self._measured_from_process = False
        self.log_offset: int | None = None
        self._backend_pending = False
        self._segment_env: dict | None = None
        self._checked_path: str | None = None

    @property
    def recorded(self) -> ServerFingerprint | None:
        return self._recorded

    def remember(self, fingerprint: ServerFingerprint, *, replace: bool = False) -> None:
        """Keep the first fingerprint, unless the caller has a fuller one."""

        if self._recorded is None or replace:
            self._recorded = fingerprint

    @property
    def restarts_disabled(self) -> str | None:
        """Why this run must not restart, or None when the record matches the live server."""

        return self._restarts_disabled

    @property
    def backend_pending(self) -> bool:
        """True when the new log segment has no ``load_backend`` line yet."""

        return self._backend_pending

    def prime(self, url: str) -> None:
        """Record the server once, before a step. A down server stays unrecorded.

        An empty flash-attention, KV-cache, library-sha, or backend-line read
        is filled from the start line. The read after a restart is not.
        The start record is compared with the live server here. A mismatch
        disables restarts for the run.
        """

        self._ensure_start()
        self.check_start_record(url)
        if self._recorded is not None:
            return
        try:
            found = self.seams.environment(url)
        except Exception:
            return
        if not isinstance(found, ServerFingerprint) or not found.version:
            return
        self._recorded = _overlay_start(found, self.start)

    def consume_cold_after_restart(self) -> bool:
        """True once, for the next kept sample after a restart that came back."""

        if not self._cold_after:
            return False
        self._cold_after = False
        return True

    def _ensure_start(self) -> ServeStart | None:
        """Load the start line the first time a restart or a prime needs it."""

        if self.start is not None:
            return self.start
        loader = getattr(self.seams, "load_start", None)
        if not callable(loader):
            return None
        try:
            found = loader()
        except Exception:
            return None
        if isinstance(found, ServeStart):
            self.start = found
        return self.start

    def check_start_record(self, url: str) -> str | None:
        """Compare the start record with the live server. Once per run.

        The live set is ``process_env`` when that seam can read it, and the
        server config line otherwise. Any difference disables restarts. The
        caller writes the reason on the marker. This does not record a
        fingerprint and it does not stop a process.
        """

        self._compare_record(url)
        return self._restarts_disabled

    def confirm_backend(self) -> None:
        """The ``load_backend`` line in the log segment written after the spawn.

        Called after the first model load and before that sample is kept.
        The line from before the offset is not a match. A segment that still
        has no line stops the run.
        """

        if not self._backend_pending:
            return
        segment = self._segment_text()
        line = backend_line_from_log(segment or "")
        expected = "" if self._recorded is None else self._recorded.backend_line
        self._backend_pending = False
        if not line:
            raise SupervisorStop(
                "the new server log has no load_backend line after the first model load; "
                "the run stops. The line from before the restart is not a match."
            )
        if line != expected:
            raise SupervisorStop(
                f"the Ollama backend line changed across the restart ({expected} to {line}); "
                "the run stops. A restart that changes the backend is not measurement-neutral."
            )

    def _compare_record(self, url: str) -> None:
        if self._env_checked:
            return
        self._ensure_start()
        if self.start is None:
            return
        self._env_checked = True
        live, from_process = self._live_env(url)
        if live is None:
            self._restarts_disabled = (
                "the live Ollama environment could not be read, so this run makes no restarts. "
                + SIGNIN_NOTE
            )
            return
        self._measured_env = live
        self._measured_from_process = from_process
        mismatch = ollama_env_mismatch(
            self.start.env,
            live,
            allow_extra=False,
            across_restart=False,
        )
        if mismatch:
            self._restarts_disabled = mismatch

    def _live_env(self, url: str) -> tuple[dict | None, bool]:
        """The live ``OLLAMA_*`` set, and whether it came from the process."""

        reader = getattr(self.seams, "process_env", None)
        if callable(reader):
            try:
                found = reader(url)
            except Exception:
                found = None
            if isinstance(found, dict):
                return ollama_env(found), True
        text = self._whole_log()
        if text:
            logged = server_config_env(text)
            if logged:
                return logged, False
        return None, False

    def _whole_log(self) -> str | None:
        if self.start is None:
            return None
        return self._read_segment(0)

    def _segment_text(self) -> str | None:
        if self.log_offset is None:
            return None
        return self._read_segment(self.log_offset)

    def _read_segment(self, offset: int) -> str | None:
        """None when this seam has no log reader. An empty string is an empty segment."""

        if self.start is None or not isinstance(self.start.log_path, str) or not self.start.log_path.strip():
            return None
        reader = getattr(self.seams, "log_segment", None)
        if not callable(reader):
            return None
        try:
            text = reader(self.start.log_path, offset)
        except Exception:
            return None
        if not isinstance(text, str):
            return None
        return text

    def _note_offset(self) -> int | None:
        """Bytes in the log before the spawn. None when there is no log seam.

        A seam that raises, or that does not return a count, is not an empty
        log. Zero is an empty log. The caller stops rather than read from zero.
        """

        sizer = getattr(self.seams, "log_size", None)
        if not callable(sizer) or self.start is None:
            return None
        try:
            size = sizer(self.start.log_path)
        except Exception:
            return None
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            return None
        return size

    def _post_restart_fingerprint(self, url: str) -> ServerFingerprint | None:
        """The server after the restart. The log read is the new segment only."""

        try:
            current = self.seams.environment(url)
        except Exception:
            return None
        if not isinstance(current, ServerFingerprint):
            return None
        self._segment_env = None
        self._backend_pending = False
        if self.log_offset is None:
            if callable(getattr(self.seams, "log_size", None)):
                return None
            return current
        segment = self._segment_text()
        if segment is None:
            return current
        logged = server_config_env(segment)
        self._segment_env = logged
        line = backend_line_from_log(segment)
        flash = logged.get("OLLAMA_FLASH_ATTENTION", "")
        kv = logged.get("OLLAMA_KV_CACHE_TYPE", "")
        if not line:
            self._backend_pending = True
            line = ""
        return replace(
            current,
            flash_attention=flash,
            kv_cache_type=kv,
            backend_line=line,
        )

    def _admit(self, url: str, model: str | None, *, hung: bool) -> None:
        """Grant, heartbeat, and ``/api/ps``, the same checks as admission.

        An unreadable ``/api/ps`` is not another session when the server is
        down. During a hang, health can still answer. An unreadable ``/api/ps``
        then refuses the restart, because another session's model would be
        dropped without being seen. A payload that names a different model
        is refused before anything stops.
        """

        try:
            require_grant(self.grant, uses_5090=True)
        except GuardError as exc:
            raise SupervisorStop(str(exc)) from exc
        if self.heartbeat is None:
            raise SupervisorStop("REFUSED: watchdog heartbeat is missing")
        wall = self.seams.wall if callable(getattr(self.seams, "wall", None)) else time.time
        try:
            require_heartbeat(Path(self.heartbeat), float(wall()))
        except GuardError as exc:
            raise SupervisorStop(str(exc)) from exc
        readable = True
        try:
            payload = self.seams.ps(url)
        except Exception:
            payload = None
            readable = False
        if payload is None:
            readable = False
        if hung and not readable:
            found = self._once(url)
            if found.ok:
                raise SupervisorStop(
                    "/api/ps is unreadable while the server still answers; "
                    "a restart during a hang is refused"
                )
        foreign = foreign_models(payload, model)
        if foreign:
            listed = ", ".join(foreign)
            raise SupervisorStop(
                f"/api/ps lists {listed}, which is not this run's model; "
                "a restart that would drop another session is refused"
            )

    def begin_measurement(self) -> int:
        if self._open:
            raise RestartRefused("a measurement is already open")
        self._open = True
        return self._epoch

    def end_measurement(self) -> None:
        self._open = False

    def sample_stands(self, epoch: int) -> bool:
        """A sample from ``epoch`` stands only when no restart has happened since."""

        return epoch == self._epoch

    def keep_sample(self, epoch: int, sample: dict) -> dict | None:
        """None when a restart has happened since ``epoch``. The caller has ended the step.

        A backend line that was missing at the restart is read here, after the
        model has loaded and before this sample is kept. A missing or different
        line stops the run, and the sample is not returned.
        """

        if not self.sample_stands(epoch):
            return None
        self.confirm_backend()
        return sample

    def server_is_down(self, url: str) -> bool:
        """True when health fails ``down_failures`` times in a row."""

        misses = 0
        while misses < self.config.down_failures:
            if self._once(url).ok:
                return False
            misses += 1
            if misses < self.config.down_failures:
                self.seams.sleep(self.config.down_gap_s)
        return True

    def request_is_hung(self, url: str, model: str) -> bool:
        """The generate already passed its deadline.

        Healthy means the version answers and ``/api/ps`` lists the model.
        Health failing, or the model missing, is a hang. A missing ``/api/ps``
        is no progress, and that is a hang too.
        """

        found = self._once(url)
        if found.ok and self._listed(url, model):
            return False
        return True

    def after_failure(self, url: str, *, model: str, kind: str) -> Decision:
        """Between steps. Raises :class:`RestartRefused` if a measurement is open.

        ``leave`` means the server stays up. A restart that cannot be shown
        to be the same server raises :class:`SupervisorStop`.
        """

        if self._open:
            raise RestartRefused("a measurement is open; restart waits until the step has ended")
        if kind == "timeout":
            if not self.request_is_hung(url, model):
                return Decision(
                    "leave",
                    note="the request is slow and the server is healthy; it is not restarted",
                )
            why = "hung"
        elif kind in ("transport", "server"):
            if not self.server_is_down(url):
                return Decision("leave", note="the server answered /api/version; it is not restarted")
            why = "down"
        else:
            return Decision("leave", note="the failure is not a transport, a timeout, or a 5xx")
        record = self._restart(url, why, model)
        return Decision("restarted", record=record)

    def _once(self, url: str) -> Health:
        return health(url, timeout=self.config.health_timeout_s, get=self.seams.health_get)

    def _listed(self, url: str, model: str) -> bool:
        try:
            payload = self.seams.ps(url)
        except Exception:
            return False
        return model_listed(payload, model)

    def _listening_ids(self, port: int) -> list[int]:
        """Distinct listening pids on ``port``. A seam asks that port only."""

        ask = getattr(self.seams, "listening", None)
        if callable(ask):
            raw = ask(port)
        else:
            raw = [row.pid for row in owners(list(self.seams.processes()), port)]
        found: list[int] = []
        for pid in raw or []:
            if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 or pid in found:
                continue
            found.append(pid)
        return found

    def _install_exe(self) -> str:
        platform = getattr(self.seams, "platform", None) or sys.platform
        local = getattr(self.seams, "localappdata", None)
        if platform == "win32":
            return serve_argv("win32", localappdata=local if isinstance(local, str) else None)[0]
        return serve_argv(str(platform))[0]

    def _one_listener(self, port: int) -> tuple[int, str]:
        """The one pid this restart may stop, and the path that was checked.

        ``processes()`` asks every local port, including 11490. This does not.
        """

        pids = self._listening_ids(port)
        if len(pids) != 1:
            listed = ", ".join(str(pid) for pid in pids) if pids else "none"
            raise SupervisorStop(
                f"refusing to stop: port {port} has {len(pids)} listening pids ({listed})"
            )
        pid = pids[0]
        read = getattr(self.seams, "executable", None)
        path = None
        if callable(read):
            try:
                path = read(pid)
            except Exception:
                path = None
        if not isinstance(path, str) or not path.strip():
            raise SupervisorStop(f"refusing to stop pid {pid}: its executable path could not be read")
        platform = getattr(self.seams, "platform", None) or sys.platform
        if not executables_match(path, self._install_exe(), platform=str(platform)):
            raise SupervisorStop(
                f"refusing to stop pid {pid}: {_receipt_path(path)} is not the Ollama install"
            )
        return pid, path

    def _restart(self, url: str, why: str, model: str | None = None) -> RestartRecord:
        self._checked_path = None
        port = port_of(url)
        if port == 11490:
            raise SupervisorStop(
                "Ollama on port 11490 is not restarted. It is the CPU indexer, and no grant covers it."
            )
        if self._recorded is None:
            raise SupervisorStop(
                "the run has no recorded Ollama environment, so a restart would not be "
                "measurement-neutral. " + SIGNIN_NOTE
            )
        self._ensure_start()
        if self.start is None:
            raise SupervisorStop(
                "the run has no recorded Ollama start line, so a restart would not replay it. "
                + SIGNIN_NOTE
            )
        self._compare_record(url)
        if self._restarts_disabled:
            raise SupervisorStop(self._restarts_disabled)
        self._admit(url, model, hung=(why == "hung"))
        argv = list(self.start.argv)
        if any("signin" in str(part).lower() for part in argv):
            raise SupervisorStop(SIGNIN_NOTE)
        if not isinstance(self.start.log_path, str) or not self.start.log_path.strip():
            raise SupervisorStop("the recorded Ollama start line has no log file. " + SIGNIN_NOTE)
        if len(self.restarts) >= self.config.max_restarts:
            raise SupervisorStop(
                f"restart cap of {self.config.max_restarts} reached; the run stops"
            )
        pid, checked = self._one_listener(port)
        self._checked_path = checked
        self.seams.stop(pid)
        old_pids = [pid]
        started = float(self.seams.clock())
        self.log_offset = self._note_offset()
        try:
            new_pid = self.seams.spawn(argv, dict(self.start.env), self.start.log_path)
        except SupervisorStop as exc:
            self._fail(url, why, old_pids, None, started, None, str(exc))
        if not isinstance(new_pid, int) or isinstance(new_pid, bool) or new_pid <= 0:
            self._fail(
                url,
                why,
                old_pids,
                None,
                started,
                None,
                "ollama serve did not return a pid. " + SIGNIN_NOTE,
            )
        if new_pid in old_pids:
            self._fail(
                url,
                why,
                old_pids,
                new_pid,
                started,
                None,
                "the process that owned the port is still the one that came back; the run stops",
            )
        version, healthy_at = self._wait_until_healthy(url, why, old_pids, new_pid, started)
        if new_pid not in self._listening_ids(port):
            self._fail(
                url,
                why,
                old_pids,
                new_pid,
                started,
                version,
                "the new server is not the process listening on the port; the run stops",
            )
        baseline_note = self._baseline_note()
        if baseline_note:
            self._commit(port, why, old_pids, new_pid, started, healthy_at, version, baseline_note)
            raise SupervisorStop(baseline_note)
        current = self._post_restart_fingerprint(url)
        if not isinstance(current, ServerFingerprint):
            unread = callable(getattr(self.seams, "log_size", None)) and self.log_offset is None
            self._fail(
                url,
                why,
                old_pids,
                new_pid,
                started,
                version,
                (
                    "the Ollama log size could not be read, so the new segment is unknown; "
                    "the run stops. The log from before the restart is not a match."
                    if unread
                    else "the Ollama environment could not be read after the restart; the run stops"
                ),
            )
        mismatch = environment_mismatch(
            self._recorded,
            current,
            check_backend=not self._backend_pending,
        )
        if mismatch is None and self._segment_env is not None and self._measured_env is not None:
            mismatch = ollama_env_mismatch(
                self._measured_env,
                self._segment_env,
                allow_extra=self._measured_from_process,
                across_restart=True,
            )
        record = self._commit(
            port,
            why,
            old_pids,
            new_pid,
            started,
            healthy_at,
            version,
            mismatch,
        )
        if mismatch:
            raise SupervisorStop(mismatch)
        self._cold_after = True
        return record

    def _wait_until_healthy(self, url, why, old_pids, new_pid, started):
        deadline = started + self.config.restart_deadline_s
        while True:
            found = self._once(url)
            if found.ok:
                return found.version, float(self.seams.clock())
            if float(self.seams.clock()) >= deadline:
                self._fail(
                    url,
                    why,
                    old_pids,
                    new_pid,
                    started,
                    None,
                    (
                        f"Ollama on port {port_of(url)} did not come back within "
                        f"{self.config.restart_deadline_s:g} s. {SIGNIN_NOTE}"
                    ),
                )
            self.seams.sleep(self.config.down_gap_s)

    def _baseline_note(self) -> str | None:
        baseline = None if self._recorded is None else self._recorded.baseline_mib
        try:
            reading = self.seams.card_mib()
        except Exception:
            reading = None
        if baseline is None or not isinstance(reading, int) or isinstance(reading, bool) or reading > baseline:
            return "the card did not return to its baseline; the run stops before the next load"
        return None

    def _commit(self, port, why, old_pids, new_pid, started, healthy_at, version, note) -> RestartRecord:
        seconds = None if healthy_at is None else healthy_at - started
        record = RestartRecord(
            when=self.seams.now(),
            port=port,
            why=why,
            old_pid=old_pids[0] if old_pids else None,
            new_pid=new_pid,
            seconds_to_healthy=seconds,
            version=version,
            note=note,
            checked_path=self._checked_path,
        )
        self.restarts.append(record)
        self._epoch += 1
        return record

    def _fail(self, url, why, old_pids, new_pid, started, version, note) -> None:
        self._commit(port_of(url), why, old_pids, new_pid, started, None, version, note)
        raise SupervisorStop(note)


def real_processes(*, run=None, platform: str | None = None, psutil_module=None) -> list[Process]:
    """Listeners on the two local ports. No other process is included."""

    rows = []
    for port in LOCAL_PORTS:
        for pid in listening_pids(port, run=run, platform=platform, psutil_module=psutil_module):
            rows.append(Process(pid=pid, name="ollama", port=port, listening=True))
    return rows


def real_seams(*, run=None, platform: str | None = None, popen=None, connection=None) -> SupervisorSeams:
    """The command's seams. Building them does not spawn, kill, or open a socket."""

    system = platform or sys.platform

    def health_get(url: str, timeout: float):
        return http_get_version(url, timeout, connection=connection)

    def processes():
        return real_processes(run=run, platform=system)

    def stop(pid: int) -> None:
        stop_pid(pid, run=run, platform=system)

    def spawn(argv: list[str], env: dict, log_path: str) -> int:
        base = {
            key: value
            for key, value in os.environ.items()
            if not str(key).startswith("OLLAMA_") and not str(key).startswith("LLAMA_ARG_")
        }
        base.update(env)
        return spawn_detached(argv, base, log_path, popen=popen, platform=system)

    def environment(url: str) -> ServerFingerprint:
        found = health(url, get=health_get)
        start = None
        try:
            start = load_serve_start()
        except Exception:
            start = None
        sha = ""
        flash = ""
        kv = ""
        line = ""
        if isinstance(start, ServeStart):
            if start.library_path:
                try:
                    sha = file_sha256(start.library_path)
                except OSError:
                    sha = ""
            text = _log_text(start.log_path)
            line = backend_line_from_log(text)
            logged = env_from_log(text)
            flash = logged.get("OLLAMA_FLASH_ATTENTION", "")
            kv = logged.get("OLLAMA_KV_CACHE_TYPE", "")
        return ServerFingerprint(
            version=found.version or "",
            flash_attention=flash,
            kv_cache_type=kv,
            ggml_cuda_sha256=sha,
            backend_line=line,
            baseline_mib=read_card_mib(run=run),
        )

    def ps(url: str):
        status, payload, error = http_get(url, "/api/ps", 5.0, connection=connection)
        if error or status != 200:
            return None
        return payload

    def listening(port: int) -> list[int]:
        # 11490 is the CPU indexer. Refuse before any query, including this seam.
        if int(port) == 11490:
            raise SupervisorStop(
                "Ollama on port 11490 is not restarted. It is the CPU indexer, and no grant covers it."
            )
        return listening_pids(int(port), run=run, platform=system)

    def executable(pid: int) -> str | None:
        return pid_executable(pid)

    def process_env(url: str):
        """The listener's environment. A test that injects ``run`` has no live process."""

        if run is not None:
            return None
        parts = urlsplit(url)
        if parts.port is None:
            return None
        pids = listening_pids(parts.port, platform=system)
        if not pids:
            return None
        return environ_of(pids[0], _load_psutil())

    return SupervisorSeams(
        health_get=health_get,
        processes=processes,
        stop=stop,
        spawn=spawn,
        card_mib=lambda: read_card_mib(run=run),
        ps=ps,
        environment=environment,
        argv=lambda: serve_argv(system),
        load_start=load_serve_start,
        process_env=process_env,
        log_size=log_byte_size,
        log_segment=read_log_from,
        listening=listening,
        executable=executable,
        localappdata=os.environ.get("LOCALAPPDATA"),
        platform=system,
    )
