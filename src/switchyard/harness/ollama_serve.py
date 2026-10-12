"""An Ollama the harness starts, pins, and stops.

The measurement server is the harness's own unelevated child on
127.0.0.1:11441. It is pinned to the install that is already listening on
11434: that process's executable, its ``OLLAMA_MODELS`` directory, and the
sha256 of ``lib/ollama/cuda_v13/ggml-cuda.dll`` beside the executable.
11434 and 11490 are never opened. A port that will not bind refuses the run.
The process tree is stopped at block end and on abort.

Nothing here contacts a GPU. The live listener command is data until a caller
runs it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


HOST = "127.0.0.1"
PORT = 11441
SHARED_PORTS = frozenset({11434, 11490})
CUDA_RELATIVE = Path("lib") / "ollama" / "cuda_v13" / "ggml-cuda.dll"


class ServeError(Exception):
    """The run-owned server cannot start, or it was asked to touch a shared port."""


@dataclass(frozen=True)
class Pin:
    exe: str
    models_dir: str
    cuda_sha256: str


@dataclass
class Running:
    pin: Pin
    argv: list[str]
    environment: dict[str, str]
    version_start: dict
    version_stop: dict | None = None
    process: object | None = None
    log_path: Path | None = None
    log_offset: int = 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cuda_dll(exe: Path) -> Path:
    return Path(exe).resolve().parent / CUDA_RELATIVE


def pin_install(exe: Path, models_dir: Path, *, read_sha=sha256_file) -> Pin:
    """Pin one install. A missing executable, models directory, or CUDA library refuses."""

    if not Path(exe).is_file():
        raise ServeError(f"Ollama executable is missing ({exe})")
    if not Path(models_dir).is_dir():
        raise ServeError(f"OLLAMA_MODELS is missing ({models_dir})")
    library = cuda_dll(Path(exe))
    if not library.is_file():
        raise ServeError(f"CUDA backend is missing ({library})")
    sha = read_sha(library)
    if not (isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha)):
        raise ServeError(f"CUDA backend sha is not a 64-hex digest ({sha!r})")
    return Pin(str(Path(exe)), str(Path(models_dir)), sha)


def assert_port_free(host: str, port: int, *, opener=socket.socket) -> None:
    """Refuse when the port will not bind. The socket is closed before the child starts."""

    if port in SHARED_PORTS:
        raise ServeError("11434 and 11490 are never used")
    sock = opener()
    try:
        sock.bind((host, port))
    except OSError as exc:
        raise ServeError(f"port {port} will not bind") from exc
    finally:
        close = getattr(sock, "close", None)
        if callable(close):
            close()


RECORDED_ENV = ("OLLAMA_HOST", "OLLAMA_MODELS", "OLLAMA_DEBUG")


def _stripped_key(key: str) -> bool:
    """``OLLAMA_*`` and ``LLAMA_ARG_*`` come from this start, not the parent."""

    folded = key.upper()
    return folded.startswith("OLLAMA_") or folded.startswith("LLAMA_ARG_")


def start_line(
    pin: Pin,
    *,
    host: str = HOST,
    port: int = PORT,
    base: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """The child command and the environment it actually receives.

    ``base`` is the process environment when omitted. Every ``OLLAMA_*`` and
    ``LLAMA_ARG_*`` key is removed, then the recorded host, models directory
    and debug flag are set. The receipt keeps only those three.
    """

    source = os.environ if base is None else base
    environment = {key: value for key, value in source.items() if not _stripped_key(key)}
    environment["OLLAMA_HOST"] = f"{host}:{port}"
    environment["OLLAMA_MODELS"] = pin.models_dir
    environment["OLLAMA_DEBUG"] = "1"
    return [pin.exe, "serve"], environment


def receipt_environment(environment: dict) -> dict[str, str]:
    """The start line's recorded variables. Inherited keys stay on the child."""

    return {key: environment[key] for key in RECORDED_ENV if key in environment}


def read_log_segment(path: Path, offset: int) -> str:
    """Text written after ``offset``. Lines from before this start are excluded."""

    if not path.is_file():
        return ""
    data = path.read_bytes()
    if offset < 0 or offset > len(data):
        return ""
    return data[offset:].decode("utf-8", errors="replace")


def refuse_shared_url(url: str) -> None:
    port = urlsplit(url).port
    if port in SHARED_PORTS:
        raise ServeError(f"refusing {port}")


def get_json(url: str, opener) -> dict:
    """GET one URL. A generate, and any URL on 11434 or 11490, is refused."""

    refuse_shared_url(url)
    if urlsplit(url).path not in {"/api/version", "/api/ps"}:
        raise ServeError(f"probes only GET /api/version and /api/ps, not {url}")
    payload = opener("GET", url)
    if not isinstance(payload, dict):
        raise ServeError(f"GET {url} did not return an object")
    return payload


def stop_process_tree(pid: int, *, system: str | None = None, run=None) -> None:
    """Stop the child and every process it started. ``run`` is injectable."""

    system = os.name if system is None else system
    if run is None:
        run = subprocess.run
    if system == "nt":
        run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False)
        return
    run(["kill", "--", f"-{pid}"], check=False)


def parse_pin_json(text: str) -> dict:
    """The listener report is one JSON object with ``exe`` and ``models_dir``."""

    try:
        report = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ServeError("the 11434 listener report is not JSON") from exc
    if not isinstance(report, dict) or not report.get("exe") or not report.get("models_dir"):
        raise ServeError("the 11434 listener report needs exe and models_dir")
    return report


LISTENER_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$conn = @(Get-NetTCPConnection -LocalPort 11434 | Where-Object { $_.RemotePort -eq 0 })[0]
$proc = Get-CimInstance Win32_Process -Filter "ProcessId=$($conn.OwningProcess)"
$models = [Environment]::GetEnvironmentVariable('OLLAMA_MODELS', 'User')
if (-not $models) { $models = [Environment]::GetEnvironmentVariable('OLLAMA_MODELS', 'Process') }
@{ exe = $proc.ExecutablePath; models_dir = $models } | ConvertTo-Json -Compress
""".strip()


def resolve_pin(run, *, read_sha=sha256_file) -> Pin:
    """Read 11434's install without opening that port.

    ``run`` executes the listener script and returns an object with ``stdout``.
    """

    completed = run(["powershell", "-NoProfile", "-Command", LISTENER_SCRIPT])
    stdout = completed if isinstance(completed, str) else getattr(completed, "stdout", "")
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8")
    report = parse_pin_json(stdout or "")
    return pin_install(Path(report["exe"]), Path(report["models_dir"]), read_sha=read_sha)


class Server:
    """Start, record, and stop one child. Transports and the process are injected in tests.

    ``stop_tree`` is the injected killer. The method that stops the child is
    ``stop``; the two names stay apart so the method can call the killer.
    """

    def __init__(self, pin: Pin, *, host: str = HOST, port: int = PORT, opener, spawn, stop_tree, log_path: Path, bind=assert_port_free):
        self.pin = pin
        self.host = host
        self.port = port
        self.opener = opener
        self.spawn = spawn
        self._stop_tree = stop_tree
        self._bind = bind
        self.log_path = log_path
        self.running: Running | None = None
        self._stopped = False

    def _url(self, path: str) -> str:
        return f"http://{self.host}:{self.port}{path}"

    def start(self) -> Running:
        self._bind(self.host, self.port)
        argv, environment = start_line(self.pin, host=self.host, port=self.port)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        offset = self.log_path.stat().st_size if self.log_path.is_file() else 0
        process = self.spawn(argv, environment, self.log_path)
        self.running = Running(
            pin=self.pin,
            argv=argv,
            environment=environment,
            version_start={},
            process=process,
            log_path=self.log_path,
            log_offset=offset,
        )
        try:
            self.running.version_start = get_json(self._url("/api/version"), self.opener)
        except Exception:
            try:
                self.stop()
            except Exception:
                pass
            raise
        return self.running

    def probe_ps(self) -> dict:
        if self.running is None:
            raise ServeError("the server is not started")
        return get_json(self._url("/api/ps"), self.opener)

    def stop(self) -> Running:
        """GET /api/version, then stop the process tree. A failed GET still kills the tree."""

        if self.running is None:
            raise ServeError("the server is not started")
        error: Exception | None = None
        if not self.running.version_stop:
            try:
                self.running.version_stop = get_json(self._url("/api/version"), self.opener)
            except Exception as exc:
                self.running.version_stop = {}
                error = exc
        if not self._stopped:
            self._stopped = True
            pid = getattr(self.running.process, "pid", None)
            if pid is not None:
                self._stop_tree(pid)
        if error is not None:
            raise error
        return self.running

    def __enter__(self) -> Running:
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.running is not None and not self._stopped:
            try:
                self.stop()
            except Exception:
                if exc_type is None:
                    raise
