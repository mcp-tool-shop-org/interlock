"""Restart a hung local Ollama between steps.

Health is ``GET /api/version`` on ``127.0.0.1:11434`` (the 5090) and
``127.0.0.1:11490`` (CPU only). A restart runs only after a step has
ended, and only when that step failed with a transport error, a timeout
that is a real hang, or a 5xx while the server is down. A long request
whose version still answers, and whose model is still listed, is left
alone.

The process that is stopped is the one listening on that port. Nothing
is killed by name. The other port's server is not touched. The tray app
is not touched unless it is the listener. ``ollama serve`` is started
detached, with ``OLLAMA_HOST`` on that port and the flash-attention and
KV-cache values the run recorded. switchyard does not sign in.

The 2026-10-09 design draft waited on a grant ledger, refused the CPU
port, and capped restarts at one. The 2026-10-12 brief is the rule this
module implements: both ports, at most two restarts a run, on a run the
harness has already admitted. The harness still requires the grant.

Tests pass fakes. The command's seams are the only ones that can spawn
a process, and a test never calls them.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

URL_5090 = "http://127.0.0.1:11434"
URL_CPU = "http://127.0.0.1:11490"
LOCAL_PORTS = (11434, 11490)
SIGNIN_NOTE = "switchyard does not sign in. A person has to start Ollama."
UNRECORDED = "unrecorded"

_PID_IN_SS = re.compile(r"pid=(\d+)")
_STATUS_5XX = re.compile(r"\b(?:status|http)\s+5\d\d\b")


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
class ServerFingerprint:
    """What the run recorded, and what a restart must still match."""

    version: str
    cuda_backend: str
    flash_attention: str
    kv_cache_type: str
    baseline_mib: int | None = None
    model: str | None = None

    def serve_env(self, port: int) -> dict[str, str]:
        """The environment of ``ollama serve``. Host is this port.

        Flash attention and the KV cache type are passed through when the
        run recorded them. An unrecorded value is left unset. The CUDA
        backend is checked after the restart; it is also passed through
        when it was recorded, under ``OLLAMA_CUDA_BACKEND``.
        """

        env = {"OLLAMA_HOST": f"127.0.0.1:{int(port)}"}
        if _recorded(self.flash_attention):
            env["OLLAMA_FLASH_ATTENTION"] = self.flash_attention
        if _recorded(self.kv_cache_type):
            env["OLLAMA_KV_CACHE_TYPE"] = self.kv_cache_type
        if _recorded(self.cuda_backend):
            env["OLLAMA_CUDA_BACKEND"] = self.cuda_backend
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


def environment_mismatch(recorded: ServerFingerprint, current: ServerFingerprint) -> str | None:
    """The field that moved, or None when the restart is still the same server."""

    pairs = (
        ("version", recorded.version, current.version),
        ("cuda_backend", recorded.cuda_backend, current.cuda_backend),
        ("flash_attention", recorded.flash_attention, current.flash_attention),
        ("kv_cache_type", recorded.kv_cache_type, current.kv_cache_type),
    )
    for name, before, after in pairs:
        if before != after:
            return (
                f"the Ollama {name} changed across the restart ({before} to {after}); "
                "the run stops. A restart that changes the backend is not measurement-neutral."
            )
    return None


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
        "ollama restart: ports 127.0.0.1:11434 and 127.0.0.1:11490; only the listener on that port is stopped",
        "ollama restart: the tray app is not stopped unless it owns the port",
        f"ollama restart: wait up to {chosen.restart_deadline_s:g} s to become healthy, then the card must be back at baseline",
        "ollama restart: switchyard does not sign in",
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

    Windows asks ``Get-NetTCPConnection -LocalPort <port> -State Listen``.
    Linux asks ``ss -ltnp`` and, when that names nobody, psutil.
    """

    system = platform or sys.platform
    number = int(port)
    if system == "win32":
        script = (
            "Get-NetTCPConnection -LocalPort "
            + str(number)
            + " -State Listen | Select-Object -ExpandProperty OwningProcess"
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


def spawn_detached(argv: list[str], env: dict, *, popen=None, platform: str | None = None) -> int:
    """Start ``argv`` in its own session and return the pid. The caller supplies ``popen`` in tests."""

    if any("signin" in str(part).lower() for part in argv):
        raise SupervisorStop(SIGNIN_NOTE)
    system = platform or sys.platform
    flags: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL, "env": env}
    if system == "win32":
        flags["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0x00000008) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200
        )
    else:
        flags["start_new_session"] = True
    launcher = popen or subprocess.Popen
    proc = launcher(list(argv), **flags)
    pid = getattr(proc, "pid", None)
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
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
            except Exception:
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
    ) -> None:
        self.seams = seams
        self.config = config or SupervisorConfig()
        self._recorded = fingerprint
        self.restarts: list[RestartRecord] = []
        self._open = False
        self._epoch = 0

    @property
    def recorded(self) -> ServerFingerprint | None:
        return self._recorded

    def remember(self, fingerprint: ServerFingerprint, *, replace: bool = False) -> None:
        """Keep the first fingerprint, unless the caller has a fuller one."""

        if self._recorded is None or replace:
            self._recorded = fingerprint

    def prime(self, url: str) -> None:
        """Record the server once, before a step. A down server stays unrecorded."""

        if self._recorded is not None:
            return
        try:
            found = self.seams.environment(url)
        except Exception:
            return
        if not isinstance(found, ServerFingerprint) or not found.version:
            return
        self._recorded = found

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
        """None when a restart has happened since ``epoch``. The caller has ended the step."""

        if not self.sample_stands(epoch):
            return None
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
        record = self._restart(url, why)
        return Decision("restarted", record=record)

    def _once(self, url: str) -> Health:
        return health(url, timeout=self.config.health_timeout_s, get=self.seams.health_get)

    def _listed(self, url: str, model: str) -> bool:
        try:
            payload = self.seams.ps(url)
        except Exception:
            return False
        return model_listed(payload, model)

    def _restart(self, url: str, why: str) -> RestartRecord:
        if self._recorded is None:
            raise SupervisorStop(
                "the run has no recorded Ollama environment, so a restart would not be "
                "measurement-neutral. " + SIGNIN_NOTE
            )
        if len(self.restarts) >= self.config.max_restarts:
            raise SupervisorStop(
                f"restart cap of {self.config.max_restarts} reached; the run stops"
            )
        port = port_of(url)
        argv = list(self.seams.argv())
        if any("signin" in str(part).lower() for part in argv):
            raise SupervisorStop(SIGNIN_NOTE)
        owned = owners(list(self.seams.processes()), port)
        old_pids = [row.pid for row in owned]
        for pid in old_pids:
            self.seams.stop(pid)
        started = float(self.seams.clock())
        new_pid = self.seams.spawn(argv, self._recorded.serve_env(port))
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
        listening = [row.pid for row in owners(list(self.seams.processes()), port)]
        if new_pid not in listening:
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
        try:
            current = self.seams.environment(url)
        except Exception:
            current = None
        if not isinstance(current, ServerFingerprint):
            self._fail(
                url,
                why,
                old_pids,
                new_pid,
                started,
                version,
                "the Ollama environment could not be read after the restart; the run stops",
            )
        mismatch = environment_mismatch(self._recorded, current)
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

    def spawn(argv: list[str], env: dict) -> int:
        base = dict(os.environ)
        base.update(env)
        return spawn_detached(argv, base, popen=popen, platform=system)

    def environment(url: str) -> ServerFingerprint:
        found = health(url, get=health_get)
        cuda = read_cuda(run=run) or UNRECORDED
        return ServerFingerprint(
            version=found.version or "",
            cuda_backend=cuda,
            flash_attention=UNRECORDED,
            kv_cache_type=UNRECORDED,
            baseline_mib=read_card_mib(run=run),
        )

    def ps(url: str):
        status, payload, error = http_get(url, "/api/ps", 5.0, connection=connection)
        if error or status != 200:
            return None
        return payload

    return SupervisorSeams(
        health_get=health_get,
        processes=processes,
        stop=stop,
        spawn=spawn,
        card_mib=lambda: read_card_mib(run=run),
        ps=ps,
        environment=environment,
        argv=lambda: serve_argv(system),
    )
