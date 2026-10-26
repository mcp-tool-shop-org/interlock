"""Linux CUDA measuring path for remote Colab cards alongside the Windows rig path.

Unlike E5 on Windows (which reads ``\\GPU Adapter Memory(*)\\Shared Usage`` via
PowerShell ``Get-Counter``), Linux CUDA has no WDDM shared-memory counter.
Residency comes from Ollama ``/api/ps`` (``size_vram == size``) and peak VRAM,
total VRAM, driver version, and CUDA version come from NVML (``pynvml`` /
``nvidia-ml-py``) or ``nvidia-smi --query-gpu``.

Failed shapes (``oom``, ``load_refused``, ``timeout``, ``ctx_exceeded``) are
recorded as validated routing rows, never silently skipped. A cell that was not
measured (``sampler_error``, ``generation_aborted``, ``unmeasured``) is recorded the same way and
says nothing about what the card can load.
"""

from __future__ import annotations

import hashlib
import inspect
import ipaddress
import itertools
import json
import math
import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

from interlock import __version__
from interlock.devices import COLAB_CARDS, RemoteDeviceIdentity
from interlock.families import stamp_family
from interlock.receipt import validate_row

UNRECORDED = "unrecorded"
FAILURE_CLASSES = ("oom", "load_refused", "timeout", "ctx_exceeded")
# A cell that was not measured. It says nothing about what the card can load, so it is not a capability outcome and
# a spec cannot pre-register it (spec.py checks FAILURE_CLASSES, which stays the capability set).
MEASUREMENT_FAILURE_CLASSES = ("sampler_error", "generation_aborted", "unmeasured")
ROW_FAILURE_CLASSES = FAILURE_CLASSES + MEASUREMENT_FAILURE_CLASSES
GENERATE_PROMPT_LENGTHS = (512, 1024, 2048, 4096, 8192, 16384, 32768)
GENERATE_CONCURRENCIES = (1, 4, 8)
# The generate ladder's version, recorded in every generate-ladder row's settings (`ladder_version`). A row with no
# field is version 1. Version 1 sent "token " repeated: some models loop on it until Ollama aborts the prediction
# (T4 r1). Version 2 sent varied text, but (T4 r2) its window could be overshot (the prompt was truncated to half),
# every timed call resent an identical prompt so llama-server answered from its prompt cache (the ttft is a cache
# hit), and the concurrency cells ran through one slot while the aggregate multiplied by the concurrency. Version 3
# sizes the window from the accepted prompt, gives every timed call a unique prefix, runs `concurrency` slots, and
# takes the aggregate from the pass's wall clock. Speeds from different versions are never compared; a version-1 or
# version-2 ttft or concurrency throughput is never used (see the router; the guard stays at version 3, whose measured
# ttft and throughput are still valid). Version 4 (T4 r3) checks that the model loads fully on the GPU before any
# timed call (a partial offload is `oom`, where v3 recorded `timeout`), reads a SIGKILL during a load as host RAM only
# when the log's numbers say so (`oom`; v3 recorded `unmeasured`), and scales each cell's request timeout from its own
# verify call and records it.
GENERATE_LADDER_VERSION = 4
THROUGHPUT_TASKS = ("embed", "rerank", "nli")
THROUGHPUT_BATCH_SIZES = (1, 8, 32)
THROUGHPUT_TOKEN_LENGTHS = (128, 510)
THROUGHPUT_SEED = "The switchyard parity check measures every batch on the same text. "
THROUGHPUT_INPUTS_REL = "experiments/c2-throughput/inputs.json"
THROUGHPUT_INPUTS_SHA256 = "48f5905ddcb24f66d9a6cd59ff2376e8766db3f8ebd4aa1e141f2fb290332966"
# The wheel carries the file as package data (pyproject force-include); a source checkout or an
# editable install has only the committed copy under experiments/.
THROUGHPUT_INPUTS_RESOURCE = "data/c2-throughput-inputs.json"
THROUGHPUT_INPUTS_SOURCE_PATH = Path(__file__).resolve().parents[3] / "experiments" / "c2-throughput" / "inputs.json"


def _throughput_inputs_path() -> Path | Traversable:
    packaged = resources.files("interlock.harness").joinpath(THROUGHPUT_INPUTS_RESOURCE)
    return packaged if packaged.is_file() else THROUGHPUT_INPUTS_SOURCE_PATH


THROUGHPUT_INPUTS_PATH = _throughput_inputs_path()
_THROUGHPUT_MODEL_ALIASES: dict[str, str] = {
    "nomic-ai/nomic-embed-text-v1.5": "nomic-ai/nomic-embed-text-v1.5",
    "nomic-embed-text:137m-v1.5-fp16": "nomic-ai/nomic-embed-text-v1.5",
    "nomic-ai-nomic-embed-text-v1.5-fp16": "nomic-ai/nomic-embed-text-v1.5",
    "BAAI/bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    "baai-bge-base-en-v1.5-fp16": "BAAI/bge-base-en-v1.5",
    "baai-bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen/Qwen3-Embedding-0.6B",
    "qwen3-embedding:0.6b-fp16": "Qwen/Qwen3-Embedding-0.6B",
    "qwen-qwen3-embedding-0.6b-fp16": "Qwen/Qwen3-Embedding-0.6B",
    "BAAI/bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
    "baai-bge-reranker-v2-m3-fp16": "BAAI/bge-reranker-v2-m3",
    "cross-encoder/nli-deberta-v3-base": "cross-encoder/nli-deberta-v3-base",
    "cross-encoder-nli-deberta-v3-base-fp16": "cross-encoder/nli-deberta-v3-base",
}
_CACHED_THROUGHPUT_FIXTURE: dict[str, Any] | None = None


def load_throughput_inputs_fixture(fixture_path: Path | None = None) -> dict[str, Any]:
    """Load and verify the committed per-model throughput inputs fixture."""

    global _CACHED_THROUGHPUT_FIXTURE
    target = fixture_path or THROUGHPUT_INPUTS_PATH
    if fixture_path is None and _CACHED_THROUGHPUT_FIXTURE is not None:
        return _CACHED_THROUGHPUT_FIXTURE
    raw = target.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if fixture_path is None and actual_sha != THROUGHPUT_INPUTS_SHA256:
        raise LinuxCudaError(
            f"throughput inputs fixture {target} sha256 mismatch: expected {THROUGHPUT_INPUTS_SHA256}, got {actual_sha}"
        )
    parsed = json.loads(raw.decode("utf-8"))
    if fixture_path is None:
        _CACHED_THROUGHPUT_FIXTURE = parsed
    return parsed


def get_throughput_input_spec(model: str, token_length: int) -> dict[str, Any] | None:
    """Return the exact tokenized input record for ``(model, token_length)`` when present in the committed fixture."""

    fixture = load_throughput_inputs_fixture()
    canonical = _THROUGHPUT_MODEL_ALIASES.get(model, model)
    model_entry = (fixture.get("models") or {}).get(canonical)
    if not isinstance(model_entry, Mapping):
        return None
    inp = (model_entry.get("inputs") or {}).get(str(token_length))
    if not isinstance(inp, Mapping):
        return None
    return {
        "text": str(inp["text"]),
        "prompt_tokens": int(inp["tokens"]),
        "prompt_tokens_with_special": int(inp["tokens_with_special"]),
        "prompt_token_method": "tokenizer_exact",
        "input_fixture_sha256": THROUGHPUT_INPUTS_SHA256,
    }
FORBIDDEN_SHARED_PORTS = frozenset({11434, 11490})
COLAB_CARD_VRAM_MIB_RANGES: dict[str, tuple[int, int]] = {
    "T4": (14000, 17000),
    "L4": (21000, 26000),
    "A100-40": (38000, 44000),
    "A100-80": (76000, 86000),
    "G4": (90000, 104000),
}

_CUDA_HEADER = re.compile(r"CUDA\s+(?:UMD\s+)?Version\s*:\s*([0-9]+\.[0-9]+)")


class LinuxCudaError(Exception):
    """Refusal or probe error in the Linux CUDA measuring path."""


class LinuxShapeFailure(Exception):
    """A measured shape failed on the remote card and must be recorded as a failure row."""

    def __init__(
        self,
        failure_class: str,
        reason: str,
        *,
        peak_mib: int | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        if failure_class not in ROW_FAILURE_CLASSES:
            raise LinuxCudaError(
                f"unknown Linux CUDA failure class {failure_class!r}; expected one of {ROW_FAILURE_CLASSES}"
            )
        if not isinstance(reason, str) or not reason.strip():
            raise LinuxCudaError("LinuxShapeFailure requires a non-empty reason")
        super().__init__(f"{failure_class}: {reason.strip()}")
        self.failure_class = failure_class
        self.reason = reason.strip()
        self.peak_mib = peak_mib
        self.evidence = evidence  # what the failure was read from, put in the row (offload, request_timeout_s)


@dataclass(frozen=True)
class LinuxGpuReading:
    card: str
    used_mib: int
    total_mib: int
    driver_version: str
    cuda_version: str
    pool: str

    def to_remote_identity(self) -> RemoteDeviceIdentity:
        return RemoteDeviceIdentity(
            provider="colab",
            card=self.card,
            vram_mib=self.total_mib,
            driver_version=self.driver_version,
            cuda_version=self.cuda_version,
            pool=self.pool,
        )


def map_card_name(raw_name: str, total_mib: int) -> str:
    """Map an NVML or ``nvidia-smi`` GPU name and VRAM total to a canonical Colab card token."""

    upper = raw_name.strip().upper()
    card: str | None = None
    if upper in COLAB_CARDS:
        card = upper
    elif "A100" in upper:
        if "40GB" in upper:
            card = "A100-40"
        elif "80GB" in upper:
            card = "A100-80"
        else:
            card = "A100-80" if total_mib >= 60000 else "A100-40"
    elif "L4" in upper and "L40" not in upper:
        card = "L4"
    elif "T4" in upper:
        card = "T4"
    elif "G4" in upper or "RTX PRO 6000" in upper or "BLACKWELL" in upper:
        card = "G4"
    if card is None:
        raise LinuxCudaError(f"unrecognized Colab GPU name {raw_name!r} ({total_mib} MiB)")
    low_mib, high_mib = COLAB_CARD_VRAM_MIB_RANGES[card]
    if isinstance(total_mib, bool) or not isinstance(total_mib, int) or not (low_mib <= total_mib <= high_mib):
        raise LinuxCudaError(
            f"VRAM {total_mib} MiB for {raw_name!r} does not match {card} range [{low_mib}, {high_mib}] MiB"
        )
    return card


def compute_abort_mib(total_vram_mib: int, abort_fraction: float) -> int:
    """Compute the card-specific abort line as ``floor(total_vram_mib * abort_fraction)``."""

    if isinstance(total_vram_mib, bool) or not isinstance(total_vram_mib, int) or total_vram_mib <= 0:
        raise LinuxCudaError("total_vram_mib must be a positive integer")
    if (
        isinstance(abort_fraction, bool)
        or not isinstance(abort_fraction, (int, float))
        or not (0.50 <= float(abort_fraction) < 1.0)
    ):
        raise LinuxCudaError("abort_fraction must be in [0.50, 1.0)")
    return int(math.floor(total_vram_mib * float(abort_fraction)))


def probe_nvidia_smi(
    *,
    runner: Callable[[list[str]], str],
    pool: str = "us-central1-a",
) -> LinuxGpuReading:
    """Read GPU identity, residency/used MiB, total MiB, driver and CUDA version via ``nvidia-smi``."""

    raw_runner = runner

    def runner(argv: list[str]) -> str:  # noqa: F811 - every call below goes through the refusal
        try:
            return raw_runner(argv)
        except (OSError, subprocess.SubprocessError) as exc:
            detail = (getattr(exc, "stderr", None) or "").strip() or str(exc)
            raise LinuxCudaError(f"REFUSED: no NVIDIA GPU available via nvidia-smi: {detail}") from exc

    query_out = runner(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    lines = [line.strip() for line in query_out.splitlines() if line.strip()]
    if len(lines) != 1:
        raise LinuxCudaError(f"expected 1 GPU row from nvidia-smi, got {len(lines)}")
    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) != 4:
        raise LinuxCudaError(f"malformed nvidia-smi CSV row: {lines[0]!r}")
    raw_name, used_str, total_str, driver_version = parts
    try:
        used_mib = int(float(used_str))
        total_mib = int(float(total_str))
    except ValueError as exc:
        raise LinuxCudaError(f"non-numeric VRAM reading in {lines[0]!r}") from exc

    header_out = runner(["nvidia-smi"])
    match = _CUDA_HEADER.search(header_out)
    if match is None:
        raise LinuxCudaError("nvidia-smi header did not report CUDA Version")
    cuda_version = match.group(1)
    card = map_card_name(raw_name, total_mib)
    return LinuxGpuReading(
        card=card,
        used_mib=used_mib,
        total_mib=total_mib,
        driver_version=driver_version,
        cuda_version=cuda_version,
        pool=pool,
    )


def probe_nvml(*, nvml: Any, pool: str = "us-central1-a", index: int = 0) -> LinuxGpuReading:
    """Read GPU identity, used/total MiB, driver and CUDA version from an NVML module."""

    nvml.nvmlInit()
    try:
        handle = nvml.nvmlDeviceGetHandleByIndex(index)
        raw_name = nvml.nvmlDeviceGetName(handle)
        if isinstance(raw_name, bytes):
            raw_name = raw_name.decode("utf-8")
        mem = nvml.nvmlDeviceGetMemoryInfo(handle)
        used_mib = int(mem.used) // (1024 * 1024)
        total_mib = int(mem.total) // (1024 * 1024)
        driver = nvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode("utf-8")
        cuda_raw = int(nvml.nvmlSystemGetCudaDriverVersion_v2())
        cuda_major = cuda_raw // 1000
        cuda_minor = (cuda_raw % 1000) // 10
        cuda_version = f"{cuda_major}.{cuda_minor}"
    finally:
        nvml.nvmlShutdown()
    card = map_card_name(str(raw_name), total_mib)
    return LinuxGpuReading(
        card=card,
        used_mib=used_mib,
        total_mib=total_mib,
        driver_version=str(driver),
        cuda_version=cuda_version,
        pool=pool,
    )


def _localhost_resolves_only_to_loopback() -> bool:
    try:
        infos = socket.getaddrinfo("localhost", None)
    except OSError:
        return False
    if not infos:
        return False
    for info in infos:
        sockaddr = info[4]
        raw_ip = str(sockaddr[0]).split("%", 1)[0]
        try:
            addr = ipaddress.ip_address(raw_ip)
        except ValueError:
            return False
        if addr not in (ipaddress.IPv4Address("127.0.0.1"), ipaddress.IPv6Address("::1")):
            return False
    return True


def validate_loopback_ollama_host(host: str) -> tuple[str, int]:
    """Validate that ``host`` is ``127.0.0.1:<port>``, ``[::1]:<port>``, or loopback ``localhost:<port>``."""

    if (
        not isinstance(host, str)
        or not host
        or host != host.strip()
        or any(ch in host for ch in ("/", "\\", "@", "?", "#", " ", "\t", "\r", "\n"))
    ):
        raise LinuxCudaError(f"refusing non-loopback Ollama host {host!r}")
    if host.startswith("["):
        if "]:" not in host:
            raise LinuxCudaError(f"host must include port: {host!r}")
        bracket_host, port_text = host.rsplit(":", 1)
        if not (bracket_host.startswith("[") and bracket_host.endswith("]")):
            raise LinuxCudaError(f"refusing non-loopback Ollama host {host!r}")
        hostname = bracket_host[1:-1]
        is_bracketed_v6 = True
    else:
        if ":" not in host:
            raise LinuxCudaError(f"host must include port: {host!r}")
        if host.count(":") != 1 or "[" in host or "]" in host:
            raise LinuxCudaError(
                f"refusing non-loopback Ollama host {host!r}; only 127.0.0.1, [::1], or loopback localhost is allowed"
            )
        hostname, port_text = host.rsplit(":", 1)
        is_bracketed_v6 = False

    if not (port_text.isascii() and port_text.isdigit()):
        raise LinuxCudaError(f"invalid port in Ollama host {host!r}")
    port = int(port_text)
    if port < 1 or port > 65535:
        raise LinuxCudaError(f"invalid port in Ollama host {host!r}")
    if port in FORBIDDEN_SHARED_PORTS:
        raise LinuxCudaError(
            f"refusing shared port {port}; Linux harness must launch its own Ollama on a dedicated port"
        )

    if is_bracketed_v6:
        if hostname != "::1":
            raise LinuxCudaError(
                f"refusing non-loopback Ollama host {host!r}; only [::1] is allowed for IPv6 loopback"
            )
        return hostname, port

    if hostname == "127.0.0.1":
        return hostname, port
    if hostname == "localhost":
        if not _localhost_resolves_only_to_loopback():
            raise LinuxCudaError(
                f"refusing non-loopback Ollama host {host!r}; localhost did not resolve exclusively to loopback"
            )
        return hostname, port

    raise LinuxCudaError(
        f"refusing non-loopback Ollama host {host!r}; only 127.0.0.1, [::1], or loopback localhost is allowed"
    )


def _validate_loopback_ollama_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "http" or parsed.username is not None or parsed.password is not None or not parsed.netloc:
        raise LinuxCudaError(f"refusing non-loopback Ollama URL {url!r}")
    try:
        validate_loopback_ollama_host(parsed.netloc)
    except LinuxCudaError as exc:
        raise LinuxCudaError(f"refusing non-loopback Ollama URL {url!r}: {exc}") from exc


READY_POLL_TIMEOUT_S = 2.0  # one readiness poll; the whole wait is ``ready_timeout_s``


def _default_http_get(url: str, timeout: float = 30.0) -> dict[str, Any]:
    _validate_loopback_ollama_url(url)
    req = urllib.request.Request(url, method="GET")  # noqa: S310
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def _default_http_post(url: str, payload: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
    _validate_loopback_ollama_url(url)
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and parsed.get("error"):
                return parsed
        except json.JSONDecodeError:
            pass
        return {"error": f"HTTP {exc.code}: {raw.strip() or exc.reason}"}


_OFFLOADED = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers")
_CPU_KV = re.compile(r"CPU KV buffer size\s*=\s*([\d.]+)\s*MiB")
_CPU_MAPPED = re.compile(r"CPU_Mapped model buffer size\s*=\s*([\d.]+)\s*MiB")
_SYSTEM_MEMORY = re.compile(r'msg="system memory"[^\n]*?total="([\d.]+)\s*(GiB|MiB)"')


def parse_load_report(text: str) -> dict[str, Any]:
    """The latest model load in an Ollama server log: layers on the GPU, host KV and mapped weights, system memory.

    The load starts at the last ``load_tensors: loading model tensors`` line (else the last ``offloaded`` line). The
    system memory is the last ``msg="system memory"`` line before it. Every value is None when it is not in the log;
    nothing is inferred.
    """

    starts = [m.start() for m in re.finditer(r"load_tensors: loading model tensors", text)] or [
        m.start() for m in _OFFLOADED.finditer(text)
    ]
    start = starts[-1] if starts else len(text)
    block = text[start:]
    layers = _OFFLOADED.findall(block)
    cpu_kv = [float(v) for v in _CPU_KV.findall(block)]
    mapped = [float(v) for v in _CPU_MAPPED.findall(block)]
    system = _SYSTEM_MEMORY.findall(text[:start] if starts else text)
    system_mib = None
    if system:
        amount, unit = system[-1]
        system_mib = float(amount) * (1024.0 if unit == "GiB" else 1.0)
    return {
        "layers_on_gpu": int(layers[-1][0]) if layers else None,
        "layers_total": int(layers[-1][1]) if layers else None,
        "cpu_kv_mib": sum(cpu_kv) if cpu_kv else None,
        "cpu_mapped_mib": sum(mapped) if mapped else None,
        "system_total_mib": system_mib,
    }


class LinuxOllamaServer:
    """Run-owned Ollama child process on a Linux VM, started and stopped by the harness."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1:11441",
        log_path: Path = Path("logs/linux-ollama.log"),
        exe: str = "ollama",
        popen: Callable[..., Any] | None = None,
        http_get: Callable[[str], dict[str, Any]] | None = None,
        http_post: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        ready_timeout_s: float = 30.0,
        poll_interval_s: float = 0.25,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        validate_loopback_ollama_host(host)
        self.host = host
        self.base_url = f"http://{host}"
        self.log_path = log_path
        self.exe = exe
        self.ready_timeout_s = ready_timeout_s
        self.poll_interval_s = poll_interval_s
        self._custom_popen = popen is not None
        self._popen = popen or subprocess.Popen
        self._http_get = http_get or _default_http_get
        # A hung poll must not use up the readiness budget: each poll is short, the loop retries inside the budget.
        self._poll_get = http_get or (lambda url: _default_http_get(url, timeout=READY_POLL_TIMEOUT_S))
        self._http_post = http_post or _default_http_post
        self._custom_post = http_post is not None
        self._sleep = sleep
        self._proc: Any = None
        self._log_handle: Any = None
        self.version: str = UNRECORDED
        self.num_parallel = 1
        self._log_offset = 0

    @property
    def pid(self) -> int | None:
        return None if self._proc is None else int(self._proc.pid)

    def get(self, path: str) -> dict[str, Any]:
        return self._http_get(f"{self.base_url}{path}")

    def post(self, path: str, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        if timeout is not None and not self._custom_post:
            return _default_http_post(f"{self.base_url}{path}", payload, timeout=timeout)
        return self._http_post(f"{self.base_url}{path}", payload)

    def __enter__(self) -> LinuxOllamaServer:
        validate_loopback_ollama_host(self.host)
        env = dict(os.environ)
        env["OLLAMA_HOST"] = self.host
        env["OLLAMA_NUM_PARALLEL"] = str(self.num_parallel)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_offset = self.log_path.stat().st_size if self.log_path.is_file() else 0
        if self._custom_popen:
            self._proc = self._popen([self.exe, "serve"], env=env)
        else:
            self._log_handle = self.log_path.open("ab")
            self._proc = self._popen(
                [self.exe, "serve"],
                env=env,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
            )
        deadline = time.monotonic() + self.ready_timeout_s
        last_exc: BaseException | None = None
        while True:
            if self._proc is not None and self._proc.poll() is not None:
                raise LinuxCudaError(
                    f"Ollama server exited before becoming ready (exit code {self._proc.poll()})"
                )
            try:
                reply = self._poll_get(f"{self.base_url}/api/version")
                self.version = str(reply.get("version") or UNRECORDED)
                return self
            except (OSError, RuntimeError, ValueError) as exc:
                last_exc = exc
                if self._proc is not None and self._proc.poll() is not None:
                    raise LinuxCudaError(
                        f"Ollama server exited before becoming ready (exit code {self._proc.poll()}): {exc}"
                    ) from exc
                if time.monotonic() >= deadline:
                    raise LinuxCudaError(
                        f"Ollama server on {self.host} did not become ready within {self.ready_timeout_s}s: {last_exc}"
                    ) from last_exc
                (self._sleep or time.sleep)(self.poll_interval_s)

    def unload(self, model: str) -> None:
        self._http_post(f"{self.base_url}/api/generate", {"model": model, "keep_alive": 0})

    def ensure_parallel(self, slots: int) -> None:
        """Run the server with exactly ``slots`` parallel slots (``OLLAMA_NUM_PARALLEL``), restarting it if needed.

        Ollama's default is 1, so concurrent requests queue. The count is exact, not a floor: KV memory is
        ``num_ctx`` times the slots, and a 1-slot cell measured on a 8-slot server would hold 8x the memory.
        """

        if slots < 1:
            raise LinuxCudaError(f"parallel slots must be at least 1, got {slots}")
        if slots == self.num_parallel and self._proc is not None and self._proc.poll() is None:
            return
        running = self._proc is not None
        if running:
            self.stop()
        self.num_parallel = slots
        if running:
            self.start()

    def load_report(self) -> dict[str, Any]:
        """The latest load's layers, host buffers and system memory, from this process's part of the server log."""

        try:
            with self.log_path.open("rb") as handle:
                handle.seek(self._log_offset)
                text = handle.read().decode("utf-8", errors="replace")
        except OSError:
            return parse_load_report("")
        return parse_load_report(text)

    def effective_slots(self) -> int | None:
        """The slot count llama-server logged for this process (``n_slots`` or ``--parallel``), or None."""

        try:
            with self.log_path.open("rb") as handle:
                handle.seek(self._log_offset)
                text = handle.read().decode("utf-8", errors="replace")
        except OSError:
            return None
        found = re.findall(r"n_slots\s*[=:]\s*(\d+)|(?:--parallel|-np)\s+(\d+)", text)
        if not found:
            return None
        last = found[-1]
        return int(last[0] or last[1])

    def start(self) -> LinuxOllamaServer:
        return self.__enter__()

    def stop(self) -> None:
        self.__exit__(None, None, None)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait()
        finally:
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None


def build_linux_failure_row(
    *,
    spec_id: str,
    spec_rel: str,
    run_id: str,
    date: str,
    task: str,
    model: str,
    revision_or_digest: str,
    remote_device: RemoteDeviceIdentity,
    failure_class: str,
    reason: str,
    engine_version: str = UNRECORDED,
    quant: str = UNRECORDED,
    licence: str = UNRECORDED,
    num_ctx: int | None = None,
    num_predict: int | None = None,
    batch: int | None = None,
    prompt_tokens: int | None = None,
    prompt_tokens_target: int | None = None,
    prompt_token_method: str | None = None,
    token_length: int | None = None,
    concurrency: int | None = None,
    peak_mib: int | None = None,
    abort_mib: int | None = None,
    offload: dict[str, Any] | None = None,
    request_timeout_s: int | None = None,
) -> dict[str, Any]:
    """Build a schema-valid capability/failure row for a failed remote shape."""

    if failure_class not in ROW_FAILURE_CLASSES:
        raise LinuxCudaError(f"invalid failure_class {failure_class!r}")
    if not isinstance(reason, str) or not reason.strip():
        raise LinuxCudaError("failure row requires a non-empty measured reason")
    clean_reason = reason.strip()
    note = clean_reason if clean_reason.startswith(f"{failure_class}:") else f"{failure_class}: {clean_reason}"
    shape_tag = _shape_suffix(
        num_ctx=num_ctx,
        prompt_tokens=prompt_tokens,
        token_length=token_length,
        concurrency=concurrency,
        batch=batch,
    )
    settings: dict[str, Any] = {
        "thinking_level": "off" if task == "generate" else None,
        "num_predict": num_predict,
        "num_ctx": num_ctx,
        "batch": batch,
        "static_shape": f"{batch}x{token_length}" if (batch is not None and token_length is not None) else None,
        "precision": None,
        "failure_class": failure_class,
        "failure_reason": clean_reason,
    }
    if task == "generate" and concurrency is not None:  # a ladder cell; a fill-card row is not one
        settings["ladder_version"] = GENERATE_LADDER_VERSION
    if prompt_tokens is not None:
        settings["prompt_tokens"] = prompt_tokens
        # A failure row that carries a requested length did not verify it; one that carries a count says how it was got.
        settings["prompt_token_method"] = prompt_token_method or "requested_not_verified"
        settings["prompt_tokens_target"] = prompt_tokens_target if prompt_tokens_target is not None else prompt_tokens
    elif token_length is not None:
        input_spec = get_throughput_input_spec(model, token_length)
        if input_spec is not None:
            settings["prompt_tokens"] = input_spec["prompt_tokens"]
            settings["prompt_tokens_with_special"] = input_spec["prompt_tokens_with_special"]
            settings["prompt_token_method"] = input_spec["prompt_token_method"]
            settings["input_fixture_sha256"] = input_spec["input_fixture_sha256"]
        else:
            settings["prompt_tokens"] = token_length
            settings["prompt_token_method"] = "word_approx_1to1"  # noqa: S105
    if concurrency is not None:
        settings["concurrency"] = concurrency
        settings["request_timeout_s"] = request_timeout_s  # the budget in force (None before one existed)
    if offload is not None:
        settings["offload"] = offload
    if peak_mib is not None:
        settings["peak_mib"] = peak_mib
    if abort_mib is not None:
        settings["abort_mib"] = abort_mib
    row_prefix = f"{spec_id}/{task}/{model}" if task != "generate" else f"{spec_id}/{model}"
    row: dict[str, Any] = {
        "id": f"{row_prefix}/{quant}/{remote_device.card}{shape_tag}/{run_id}",
        "task": task,
        "model": {
            "repo_or_tag": model,
            "revision_or_digest": revision_or_digest,
            "quant": quant,
            "licence": licence,
        },
        "engine": {"name": "ollama", "version": engine_version},
        "device": remote_device.logical_device,
        "remote_device": remote_device.to_dict(),
        "settings": settings,
        "quality": [],
        "latency": {
            "cold_ms": None,
            "warm_p50_ms": None,
            "warm_p95_ms": None,
            "throughput": None,
            "throughput_unit": None,
            "compile_ms": None,
        },
        "date": date,
        "repeats": 1,
        "provisional": False,
        "provenance": {"experiment_spec": spec_rel, "run_id": run_id},
        "outcome": failure_class,
        "notes": note,
    }
    stamp_family(row)
    validate_row(row)
    return row


def _shape_suffix(
    *,
    num_ctx: int | None,
    prompt_tokens: int | None,
    concurrency: int | None,
    batch: int | None,
    token_length: int | None = None,
) -> str:
    parts: list[str] = []
    if num_ctx is not None:
        parts.append(f"ctx{num_ctx}")
    if prompt_tokens is not None:
        parts.append(f"p{prompt_tokens}")
    if token_length is not None:
        parts.append(f"l{token_length}")
    if concurrency is not None:
        parts.append(f"c{concurrency}")
    if batch is not None:
        parts.append(f"b{batch}")
    return ("/" + "-".join(parts)) if parts else ""


def run_linux_fill_card(
    *,
    spec_id: str,
    spec_rel: str,
    run_id: str,
    date: str,
    remote_device: RemoteDeviceIdentity,
    model: str,
    revision_or_digest: str,
    trained_ctx: int,
    abort_fraction: float,
    grain: int,
    measure_window: Callable[[int], Mapping[str, Any]],
    spill_prompt_ratio: float = 0.5,
    engine_version: str = UNRECORDED,
    quant: str = UNRECORDED,
    licence: str = UNRECORDED,
) -> dict[str, Any]:
    """Find the largest resident ``num_ctx`` under ``floor(vram_mib * abort_fraction)`` on Linux CUDA.

    Doubles from ``grain`` up to ``trained_ctx`` and bisects on the ``grain`` grid between the
    highest passing window and the first failing window. No Windows shared-memory counter is read.
    A window passes when ``size_vram == size``, ``peak_mib < abort_mib``, and
    ``prompt_tok_s >= spill_prompt_ratio * prev_prompt_tok_s``.
    If even the floor grain fails, a failure row is returned.
    """

    abort_mib = compute_abort_mib(remote_device.vram_mib, abort_fraction)
    if grain <= 0 or trained_ctx < grain:
        return build_linux_failure_row(
            spec_id=spec_id,
            spec_rel=spec_rel,
            run_id=run_id,
            date=date,
            task="generate",
            model=model,
            revision_or_digest=revision_or_digest,
            remote_device=remote_device,
            failure_class="ctx_exceeded",
            reason=f"trained_ctx {trained_ctx} is below grain {grain}",
            num_ctx=trained_ctx,
            abort_mib=abort_mib,
            engine_version=engine_version,
            quant=quant,
            licence=licence,
        )

    candidates: list[int] = [grain]
    cursor = grain * 2
    while cursor <= trained_ctx:
        candidates.append(cursor)
        cursor *= 2
    top_grain = (trained_ctx // grain) * grain
    if candidates[-1] != top_grain and top_grain > candidates[-1]:
        candidates.append(top_grain)

    def _eval_window(
        candidate: int, ref_prompt_rate: float | None
    ) -> tuple[Mapping[str, Any] | None, tuple[str, str, int | None] | None]:
        try:
            reading = measure_window(candidate)
        except LinuxShapeFailure as exc:
            return None, (exc.failure_class, exc.reason, exc.peak_mib)
        size = int(reading["size"])
        size_vram = int(reading["size_vram"])
        peak_mib = int(reading["peak_mib"])
        prompt_rate = float(reading["prompt_tok_s"])
        if size_vram < size:
            return None, (
                "oom",
                f"partial layer offload at num_ctx {candidate} (size_vram {size_vram} < size {size})",
                peak_mib,
            )
        if peak_mib >= abort_mib:
            return None, (
                "oom",
                f"peak {peak_mib} MiB reached abort_mib {abort_mib} MiB at num_ctx {candidate}",
                peak_mib,
            )
        if ref_prompt_rate is not None and prompt_rate < spill_prompt_ratio * ref_prompt_rate:
            return None, (
                "oom",
                f"prompt_tok_s {prompt_rate:.1f} fell below {spill_prompt_ratio}x previous {ref_prompt_rate:.1f}",
                peak_mib,
            )
        return reading, None

    best_ctx: int | None = None
    best_reading: Mapping[str, Any] | None = None
    prev_prompt_rate: float | None = None
    last_failure: tuple[str, str, int | None] | None = None
    failed_ctx: int | None = None
    measurement_failed = False

    for candidate in candidates:
        reading, failure = _eval_window(candidate, prev_prompt_rate)
        if failure is not None or reading is None:
            last_failure = failure
            failed_ctx = candidate
            measurement_failed = failure is not None and failure[0] in MEASUREMENT_FAILURE_CLASSES
            break
        best_ctx = candidate
        best_reading = reading
        prev_prompt_rate = float(reading["prompt_tok_s"])

    if not measurement_failed and best_ctx is not None and failed_ctx is not None and failed_ctx - best_ctx > grain:
        low_steps = best_ctx // grain
        high_steps = failed_ctx // grain
        while high_steps - low_steps > 1:
            mid_steps = (low_steps + high_steps) // 2
            mid_ctx = mid_steps * grain
            reading, failure = _eval_window(mid_ctx, prev_prompt_rate)
            if failure is None and reading is not None:
                best_ctx = mid_ctx
                best_reading = reading
                prev_prompt_rate = float(reading["prompt_tok_s"])
                low_steps = mid_steps
            else:
                last_failure = failure
                high_steps = mid_steps
                if failure is not None and failure[0] in MEASUREMENT_FAILURE_CLASSES:
                    measurement_failed = True
                    break

    if measurement_failed or best_ctx is None or best_reading is None:
        f_class, f_reason, f_peak = last_failure or ("load_refused", "no resident window found", None)
        return build_linux_failure_row(
            spec_id=spec_id,
            spec_rel=spec_rel,
            run_id=run_id,
            date=date,
            task="generate",
            model=model,
            revision_or_digest=revision_or_digest,
            remote_device=remote_device,
            failure_class=f_class,
            reason=f_reason,
            num_ctx=grain,
            peak_mib=f_peak,
            abort_mib=abort_mib,
            engine_version=engine_version,
            quant=quant,
            licence=licence,
        )

    decode_tok_s = float(best_reading["decode_tok_s"])
    ttft_ms = float(best_reading["ttft_ms"])
    prompt_tok_s = float(best_reading["prompt_tok_s"])
    peak_mib = int(best_reading["peak_mib"])
    size = int(best_reading["size"])
    size_vram = int(best_reading["size_vram"])
    raw_slots = best_reading.get("slots")
    slots = (
        list(raw_slots)
        if isinstance(raw_slots, list) and len(raw_slots) >= 2
        else [
            {"phase": "warmup", "throughput": decode_tok_s, "latency_ms": ttft_ms, "ttft_ms": ttft_ms},
            {"phase": "repeat", "throughput": decode_tok_s, "latency_ms": ttft_ms, "ttft_ms": ttft_ms},
        ]
    )
    repeats = sum(1 for s in slots if isinstance(s, Mapping) and s.get("phase") == "repeat") or 1
    row: dict[str, Any] = {
        "id": f"{spec_id}/{model}/{quant}/{remote_device.card}-fill/{run_id}",
        "task": "generate",
        "model": {
            "repo_or_tag": model,
            "revision_or_digest": revision_or_digest,
            "quant": quant,
            "licence": licence,
            "size": size,
            "size_vram": size_vram,
        },
        "engine": {"name": "ollama", "version": engine_version},
        "device": remote_device.logical_device,
        "remote_device": remote_device.to_dict(),
        "gpu_resident": True,
        "settings": {
            "thinking_level": "off",
            "num_predict": 1,
            "num_ctx": best_ctx,
            "batch": None,
            "static_shape": None,
            "precision": None,
            "max_resident_ctx": best_ctx,
            "peak_mib": peak_mib,
            "abort_mib": abort_mib,
            "abort_fraction": float(abort_fraction),
            "ttft_ms": ttft_ms,
            "decode_tok_s": decode_tok_s,
            "prompt_tok_s": prompt_tok_s,
        },
        "quality": [],
        "latency": {
            "cold_ms": None,
            "warm_p50_ms": ttft_ms,
            "warm_p95_ms": ttft_ms,
            "throughput": decode_tok_s,
            "throughput_unit": "tok/s",
            "compile_ms": None,
        },
        "artifacts": {"slots": slots},
        "date": date,
        "repeats": repeats,
        "provisional": False,
        "provenance": {"experiment_spec": spec_rel, "run_id": run_id},
        "notes": f"resident up to num_ctx {best_ctx} (peak {peak_mib} MiB < abort {abort_mib} MiB)",
    }
    stamp_family(row)
    validate_row(row)
    return row


class _ProgressRows(list):
    """A list that reports itself to ``notify`` each time a row is added, so a lost run keeps its finished cells."""

    def __init__(self, notify: Callable[[list[dict[str, Any]]], None] | None) -> None:
        super().__init__()
        self._notify = notify

    def append(self, item: Any) -> None:
        super().append(item)
        if self._notify is not None:
            self._notify(self)

    def extend(self, items: Any) -> None:
        super().extend(items)
        if self._notify is not None:
            self._notify(self)


def run_linux_generate_ladder(
    *,
    spec_id: str,
    spec_rel: str,
    run_id: str,
    date: str,
    remote_device: RemoteDeviceIdentity,
    model: str,
    revision_or_digest: str,
    measure_shape: Callable[[int, int], Mapping[str, Any]],
    prompt_lengths: tuple[int, ...] = GENERATE_PROMPT_LENGTHS,
    concurrencies: tuple[int, ...] = GENERATE_CONCURRENCIES,
    abort_fraction: float = 0.90,
    num_predict: int = 128,
    engine_version: str = UNRECORDED,
    quant: str = UNRECORDED,
    licence: str = UNRECORDED,
    on_progress: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    """Sweep generate TTFT and decode tok/s across prompt lengths 512..32k at concurrencies 1, 4, 8."""

    abort_mib = compute_abort_mib(remote_device.vram_mib, abort_fraction)
    rows: list[dict[str, Any]] = _ProgressRows(on_progress)
    # Concurrency is the outer loop: the server restarts to change its slot count, so each count is set once.
    for concurrency in concurrencies:
        baseline: tuple[int, float] | None = None  # (target tokens, ttft ms) of the first measured cell at this count
        for prompt_len in sorted(prompt_lengths):
            try:
                measured = measure_shape(prompt_len, concurrency)
            except LinuxShapeFailure as exc:
                rows.append(
                    build_linux_failure_row(
                        spec_id=spec_id,
                        spec_rel=spec_rel,
                        run_id=run_id,
                        date=date,
                        task="generate",
                        model=model,
                        revision_or_digest=revision_or_digest,
                        remote_device=remote_device,
                        failure_class=exc.failure_class,
                        reason=exc.reason,
                        num_ctx=prompt_len,
                        num_predict=num_predict,
                        prompt_tokens=prompt_len,
                        concurrency=concurrency,
                        offload=(exc.evidence or {}).get("offload"),
                        request_timeout_s=(exc.evidence or {}).get("request_timeout_s"),
                        peak_mib=exc.peak_mib,
                        abort_mib=abort_mib,
                        engine_version=engine_version,
                        quant=quant,
                        licence=licence,
                    )
                )
                continue

            ttft_ms = float(measured["ttft_ms"])
            decode_tok_s = float(measured["decode_tok_s"])
            latency_ms = float(measured.get("latency_ms", ttft_ms))
            peak_mib = int(measured["peak_mib"])
            size = int(measured["size"])
            size_vram = int(measured["size_vram"])
            if size_vram < size or peak_mib >= abort_mib:
                reason = (
                    f"partial offload ({size_vram} < {size})"
                    if size_vram < size
                    else f"peak {peak_mib} MiB >= abort_mib {abort_mib} MiB"
                )
                rows.append(
                    build_linux_failure_row(
                        spec_id=spec_id,
                        spec_rel=spec_rel,
                        run_id=run_id,
                        date=date,
                        task="generate",
                        model=model,
                        revision_or_digest=revision_or_digest,
                        remote_device=remote_device,
                        failure_class="oom",
                        reason=reason,
                        num_ctx=prompt_len,
                        num_predict=num_predict,
                        prompt_tokens=_evaluated_or_requested(measured, prompt_len)[0],
                        prompt_tokens_target=prompt_len,
                        prompt_token_method=_evaluated_or_requested(measured, prompt_len)[1],
                        concurrency=concurrency,
                        request_timeout_s=measured.get("request_timeout_s"),
                        peak_mib=peak_mib,
                        abort_mib=abort_mib,
                        engine_version=engine_version,
                        quant=quant,
                        licence=licence,
                    )
                )
                continue

            raw_slots = measured.get("slots")
            slots = (
                list(raw_slots)
                if isinstance(raw_slots, list) and len(raw_slots) >= 2
                else [
                    {"phase": "warmup", "throughput": decode_tok_s, "latency_ms": latency_ms, "ttft_ms": ttft_ms},
                    {"phase": "repeat", "throughput": decode_tok_s, "latency_ms": latency_ms, "ttft_ms": ttft_ms},
                ]
            )
            repeats = sum(1 for s in slots if isinstance(s, Mapping) and s.get("phase") == "repeat") or 1
            evaluated_tokens, token_method = _evaluated_or_requested(measured, prompt_len)
            prefill_ms = float(measured.get("prefill_ms", ttft_ms))
            reason = _ttft_does_not_scale(baseline, prompt_len, prefill_ms)
            if reason is None and baseline is None:
                baseline = (prompt_len, prefill_ms)
            if reason is not None:
                rows.append(
                    build_linux_failure_row(
                        spec_id=spec_id,
                        spec_rel=spec_rel,
                        run_id=run_id,
                        date=date,
                        task="generate",
                        model=model,
                        revision_or_digest=revision_or_digest,
                        remote_device=remote_device,
                        failure_class="unmeasured",
                        reason=reason,
                        num_ctx=prompt_len,
                        num_predict=num_predict,
                        prompt_tokens=evaluated_tokens,
                        prompt_tokens_target=prompt_len,
                        prompt_token_method=token_method,
                        concurrency=concurrency,
                        request_timeout_s=measured.get("request_timeout_s"),
                        peak_mib=peak_mib,
                        abort_mib=abort_mib,
                        engine_version=engine_version,
                        quant=quant,
                        licence=licence,
                    )
                )
                continue
            row: dict[str, Any] = {
                "id": f"{spec_id}/{model}/{quant}/{remote_device.card}/p{prompt_len}-c{concurrency}/{run_id}",
                "task": "generate",
                "model": {
                    "repo_or_tag": model,
                    "revision_or_digest": revision_or_digest,
                    "quant": quant,
                    "licence": licence,
                    "size": size,
                    "size_vram": size_vram,
                },
                "engine": {"name": "ollama", "version": engine_version},
                "device": remote_device.logical_device,
                "remote_device": remote_device.to_dict(),
                "gpu_resident": True,
                "settings": {
                    "thinking_level": "off",
                    "num_predict": num_predict,
                    "num_ctx": int(measured.get("num_ctx_sent") or prompt_len),  # what the request sent
                    "cell_tokens": prompt_len,  # the nominal cell size
                    "batch": None,
                    "static_shape": None,
                    "precision": None,
                    "prompt_tokens": evaluated_tokens,
                    "prompt_tokens_target": prompt_len,
                    "prompt_token_method": token_method,
                    "ladder_version": GENERATE_LADDER_VERSION,
                    "concurrency": concurrency,
                    "prefix_caching": "off",  # every timed call has its own prefix, so no prefix is cached
                    "request_timeout_s": measured.get("request_timeout_s"),
                    "parallel_slots": measured.get("parallel_slots", concurrency),
                    "parallel_slots_source": measured.get("parallel_slots_source", "requested"),
                    "throughput_method": measured.get("throughput_method", "ollama_durations_decode_window"),
                    "end_to_end_tok_s": measured.get("end_to_end_tok_s"),
                    "prefill_ms": prefill_ms,
                    "overhead_ms": measured.get("overhead_ms"),
                    "per_request_decode_tok_s": measured.get("per_request_decode_tok_s"),
                    "ttft_ms": ttft_ms,
                    "decode_tok_s": decode_tok_s,
                    "peak_mib": peak_mib,
                    "abort_mib": abort_mib,
                },
                "quality": [],
                "latency": {
                    "cold_ms": None,
                    "warm_p50_ms": latency_ms,
                    "warm_p95_ms": latency_ms,
                    "throughput": decode_tok_s,
                    "throughput_unit": "tok/s",
                    "compile_ms": None,
                },
                "artifacts": {"slots": slots},
                "date": date,
                "repeats": repeats,
                "provisional": False,
                "provenance": {"experiment_spec": spec_rel, "run_id": run_id},
            }
            stamp_family(row)
            validate_row(row)
            rows.append(row)
    return rows


def run_linux_throughput_ladder(
    *,
    spec_id: str,
    spec_rel: str,
    run_id: str,
    date: str,
    remote_device: RemoteDeviceIdentity,
    models_by_task: Mapping[str, tuple[str, str]],
    measure_batch: Callable[..., Mapping[str, Any]],
    tasks: tuple[str, ...] = THROUGHPUT_TASKS,
    batch_sizes: tuple[int, ...] = THROUGHPUT_BATCH_SIZES,
    token_lengths: tuple[int, ...] = (510,),
    abort_fraction: float = 0.90,
    engine_version: str = UNRECORDED,
    quant: str = UNRECORDED,
    licence: str = UNRECORDED,
    on_progress: Callable[[list[dict[str, Any]]], None] | None = None,
    meta_by_task: Mapping[str, tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Sweep embed, rerank, and NLI throughput across token lengths and batch sizes 1, 8, and 32 on Linux CUDA."""

    abort_mib = compute_abort_mib(remote_device.vram_mib, abort_fraction)
    rows: list[dict[str, Any]] = _ProgressRows(on_progress)
    for task in tasks:
        model, digest = models_by_task[task]
        task_quant, task_licence = meta_by_task.get(task, (quant, licence)) if meta_by_task else (quant, licence)
        for token_len in token_lengths:
            for batch in batch_sizes:
                try:
                    try:
                        measured = measure_batch(task, batch, token_length=token_len)
                    except TypeError:
                        measured = measure_batch(task, batch)
                except LinuxShapeFailure as exc:
                    rows.append(
                        build_linux_failure_row(
                            spec_id=spec_id,
                            spec_rel=spec_rel,
                            run_id=run_id,
                            date=date,
                            task=task,
                            model=model,
                            revision_or_digest=digest,
                            remote_device=remote_device,
                            failure_class=exc.failure_class,
                            reason=exc.reason,
                            batch=batch,
                            token_length=token_len,
                            peak_mib=exc.peak_mib,
                            abort_mib=abort_mib,
                            engine_version=engine_version,
                            quant=task_quant,
                            licence=task_licence,
                        )
                    )
                    continue

                tput = float(measured["throughput"])
                p50 = float(measured["warm_p50_ms"])
                p95 = float(measured.get("warm_p95_ms", p50))
                peak_mib = int(measured["peak_mib"])
                size = int(measured["size"])
                size_vram = int(measured["size_vram"])
                if size_vram < size or peak_mib >= abort_mib:
                    reason = (
                        f"partial offload ({size_vram} < {size})"
                        if size_vram < size
                        else f"peak {peak_mib} MiB >= abort_mib {abort_mib} MiB"
                    )
                    rows.append(
                        build_linux_failure_row(
                            spec_id=spec_id,
                            spec_rel=spec_rel,
                            run_id=run_id,
                            date=date,
                            task=task,
                            model=model,
                            revision_or_digest=digest,
                            remote_device=remote_device,
                            failure_class="oom",
                            reason=reason,
                            batch=batch,
                            token_length=token_len,
                            peak_mib=peak_mib,
                            abort_mib=abort_mib,
                            engine_version=engine_version,
                            quant=task_quant,
                            licence=task_licence,
                        )
                    )
                    continue

                raw_slots = measured.get("slots")
                slots = (
                    list(raw_slots)
                    if isinstance(raw_slots, list) and len(raw_slots) >= 2
                    else [
                        {"phase": "warmup", "throughput": tput, "latency_ms": p50},
                        {"phase": "repeat", "throughput": tput, "latency_ms": p50},
                    ]
                )
                repeats = sum(1 for s in slots if isinstance(s, Mapping) and s.get("phase") == "repeat") or 1
                input_spec = get_throughput_input_spec(model, token_len)
                row_settings: dict[str, Any] = {
                    "thinking_level": None,
                    "num_predict": None,
                    "num_ctx": None,
                    "batch": batch,
                    "static_shape": f"{batch}x{token_len}",
                    "precision": "fp16",
                    "prompt_tokens": input_spec["prompt_tokens"] if input_spec is not None else token_len,
                    "prompt_token_method": (
                        input_spec["prompt_token_method"]
                        if input_spec is not None
                        else "word_approx_1to1"  # noqa: S105
                    ),
                    "peak_mib": peak_mib,
                    "abort_mib": abort_mib,
                }
                if input_spec is not None:
                    row_settings["prompt_tokens_with_special"] = input_spec["prompt_tokens_with_special"]
                    row_settings["input_fixture_sha256"] = input_spec["input_fixture_sha256"]
                row: dict[str, Any] = {
                    "id": f"{spec_id}/{task}/{model}/{task_quant}/{remote_device.card}/l{token_len}-b{batch}/{run_id}",
                    "task": task,
                    "model": {
                        "repo_or_tag": model,
                        "revision_or_digest": digest,
                        "quant": task_quant,
                        "licence": task_licence,
                        "size": size,
                        "size_vram": size_vram,
                    },
                    "engine": {"name": "ollama", "version": engine_version},
                    "device": remote_device.logical_device,
                    "remote_device": remote_device.to_dict(),
                    "gpu_resident": True,
                    "settings": row_settings,
                    "quality": [],
                    "latency": {
                        "cold_ms": None,
                        "warm_p50_ms": p50,
                        "warm_p95_ms": p95,
                        "throughput": tput,
                        "throughput_unit": "items/s",
                        "compile_ms": None,
                    },
                    "artifacts": {"slots": slots},
                    "date": date,
                    "repeats": repeats,
                    "provisional": False,
                    "provenance": {"experiment_spec": spec_rel, "run_id": run_id},
                }
                stamp_family(row)
                validate_row(row)
                rows.append(row)
    return rows


# Ollama's own wording for a model it will not load, or cannot fit (checked against the 0.40.9 binary). "resource
# limitations" is the runner-crash message, which can come mid-generation, so it is not here.
_LOAD_REFUSAL_PHRASES = (
    "unable to load",
    "error loading model",
    "failed to load model",  # not "failed to load manifest": that is a pull or registry error
    "insufficient memory",
    "not resident",
    "too large for system memory",
)


def classify_linux_error(exc: BaseException, peak_mib: int | None = None) -> LinuxShapeFailure:
    """Classify an HTTP or runtime error. `load_refused` is for a recognised load or VRAM refusal only."""

    if isinstance(exc, LinuxShapeFailure):
        return exc
    msg = str(exc).strip() or exc.__class__.__name__
    low = msg.lower()
    # The model loaded and generated; Ollama stopped a prediction that looped. Not a refusal, and not a speed.
    if "repeat limit" in low or "prediction aborted" in low:
        return LinuxShapeFailure("generation_aborted", msg, peak_mib=peak_mib)
    if (
        isinstance(exc, TimeoutError)
        or "timed out" in low
        or "timeout" in low
        or "deadline exceeded" in low
        or re.search(r"http \d{3}.*after \d+(?:\.\d+)?(?:ms|s|m|h)", low)  # Go's "HTTP 500 after 2m0s"
    ):
        return LinuxShapeFailure("timeout", msg, peak_mib=peak_mib)
    if "longer than the context length" in low or (
        "context" in low and ("exceed" in low or "limit" in low or "too long" in low)
    ):
        return LinuxShapeFailure("ctx_exceeded", msg, peak_mib=peak_mib)
    if "out of memory" in low or "cuda error" in low or re.search(r"\boom\b", low) or "alloc" in low:
        return LinuxShapeFailure("oom", msg, peak_mib=peak_mib)
    if any(phrase in low for phrase in _LOAD_REFUSAL_PHRASES):
        return LinuxShapeFailure("load_refused", msg, peak_mib=peak_mib)
    # Anything else says nothing about what the card can load. It is a cell that was not measured.
    return LinuxShapeFailure("unmeasured", msg, peak_mib=peak_mib)


def _default_nvidia_smi_runner(argv: list[str]) -> str:
    proc = subprocess.run(argv, check=True, capture_output=True, text=True)  # noqa: S603
    return proc.stdout


def probe_linux_gpu(
    *,
    nvml_module: Any | None = None,
    runner: Callable[[list[str]], str] | None = None,
    pool: str | None = None,
) -> tuple[RemoteDeviceIdentity, Callable[[], tuple[int, int]]]:
    """Probe the active Linux CUDA GPU via NVML or ``nvidia-smi`` and return ``(identity, sample_fn)``."""

    resolved_pool = pool or "us-central1-a"
    if nvml_module is not None:
        reading = probe_nvml(nvml=nvml_module, pool=resolved_pool)
        return reading.to_remote_identity(), (
            lambda: (
                probe_nvml(nvml=nvml_module, pool=resolved_pool).used_mib,
                reading.total_mib,
            )
        )
    try:
        import pynvml  # type: ignore[import-not-found]

        reading = probe_nvml(nvml=pynvml, pool=resolved_pool)
        return reading.to_remote_identity(), (
            lambda: (
                probe_nvml(nvml=pynvml, pool=resolved_pool).used_mib,
                reading.total_mib,
            )
        )
    except Exception:
        active_runner = runner or _default_nvidia_smi_runner
        reading = probe_nvidia_smi(runner=active_runner, pool=resolved_pool)
        return reading.to_remote_identity(), (
            lambda: (
                probe_nvidia_smi(runner=active_runner, pool=resolved_pool).used_mib,
                reading.total_mib,
            )
        )


def _sample_during_cell(sample_vram_mib: Callable[[], tuple[int, int]]) -> tuple[int, int]:
    """A VRAM reading taken inside a cell. A sampler failure is a failed measurement, never a load refusal."""

    try:
        return sample_vram_mib()
    except Exception as exc:
        raise LinuxShapeFailure("sampler_error", f"VRAM sampler failed: {exc}") from exc


def _bare_tag(name: object) -> str:
    """A model name as ``/api/ps`` and ``/api/generate`` may each spell it: ``foo`` and ``foo:latest`` match."""

    text = name if isinstance(name, str) else ""
    return text[: -len(":latest")] if text.endswith(":latest") else text


def _query_ps_entry(server: LinuxOllamaServer, model: str) -> tuple[int, int]:
    """Read ``(size, size_vram)`` for ``model`` from ``/api/ps``."""

    ps_doc = server.get("/api/ps")
    models = ps_doc.get("models") if isinstance(ps_doc, Mapping) else None
    if isinstance(models, list):
        wanted = _bare_tag(model)
        for entry in models:
            if isinstance(entry, Mapping) and (
                _bare_tag(entry.get("name")) == wanted or _bare_tag(entry.get("model")) == wanted or len(models) == 1
            ):
                size = int(entry.get("size") or 0)
                size_vram = int(entry.get("size_vram") or 0)
                if size > 0:
                    return size, size_vram
    raise LinuxShapeFailure("load_refused", f"model {model} not resident in /api/ps")


def measure_linux_ollama_window(
    server: LinuxOllamaServer,
    sample_vram_mib: Callable[[], tuple[int, int]],
    *,
    model: str,
    num_ctx: int,
) -> dict[str, Any]:
    """Measure one ``num_ctx`` window against a running ``LinuxOllamaServer`` and VRAM sampler."""

    used_before, _ = sample_vram_mib()
    peak_mib = used_before
    try:
        resp = server.post(
            "/api/generate",
            {
                "model": model,
                "prompt": "interlock window probe " * max(1, min(64, num_ctx // 64)),
                "stream": False,
                "options": {"num_ctx": num_ctx, "num_predict": 16, "temperature": 0.0},
            },
        )
        if isinstance(resp, Mapping) and resp.get("error"):
            raise RuntimeError(str(resp["error"]))
        used_after, _ = _sample_during_cell(sample_vram_mib)
        peak_mib = max(peak_mib, used_after)
        size, size_vram = _query_ps_entry(server, model)
    except BaseException as exc:
        raise classify_linux_error(exc, peak_mib=peak_mib) from exc

    prompt_eval_count = max(1, int(resp.get("prompt_eval_count") or 16))
    prompt_eval_ns = max(1, int(resp.get("prompt_eval_duration") or 10_000_000))
    eval_count = max(1, int(resp.get("eval_count") or 16))
    eval_ns = max(1, int(resp.get("eval_duration") or 50_000_000))
    ttft_ms = round(prompt_eval_ns / 1_000_000.0, 3)
    prompt_tok_s = round(prompt_eval_count / (prompt_eval_ns / 1_000_000_000.0), 3)
    decode_tok_s = round(eval_count / (eval_ns / 1_000_000_000.0), 3)
    return {
        "size": size,
        "size_vram": size_vram,
        "peak_mib": peak_mib,
        "prompt_tok_s": prompt_tok_s,
        "decode_tok_s": decode_tok_s,
        "ttft_ms": ttft_ms,
    }


_LADDER_WORDS = (
    "river stone market lantern copper garden window harbor bridge forest letter engine ribbon valley candle "
    "mirror anchor meadow thread summit orchard compass velvet harvest signal marble island kettle pepper "
    "falcon ladder quartz pasture saddle timber violet walnut yarrow zephyr amber basket cobalt dune ember "
    "fjord glacier hollow indigo jasper kiln lagoon mosaic nectar opal prairie quiver reef sparrow tundra"
).split()


LADDER_CTX_MARGIN = 64
LADDER_PROBE_WORDS = (64, 192)  # small enough never to overflow a window
LADDER_TOKEN_TOLERANCE = 0.02  # the evaluated prompt must be within 2% of the target
# A prefill faster than this cannot be real on any card this harness targets; it means llama-server answered from its
# prompt cache (it reuses the longest common prefix). A cell above it is unmeasured.
PREFILL_TOK_S_CEILING = 250_000.0
# Prefill time grows with the prompt: from the first measured cell at a concurrency, a longer cell's ttft must be at
# least this fraction of the proportional growth. A cached cell stays flat, far under it.
TTFT_SCALING_FLOOR = 0.6
# One request's budget. Calibration calls are single requests (a partial offload is caught before they matter). A timed
# call gets three times what `concurrency` verify-sized requests need, plus decode, clamped; the budget is recorded in
# every generate row. After the residency check a timeout can only be a fully resident cell that is slow.
CALIBRATION_TIMEOUT_S = 900
REQUEST_TIMEOUT_FACTOR = 3
REQUEST_TIMEOUT_DECODE_ALLOWANCE_S = 60
REQUEST_TIMEOUT_FLOOR_S = 120
REQUEST_TIMEOUT_CEILING_S = 900
_LADDER_INSTRUCTION = "In two or three sentences, describe a quiet town by a river."
_CELL_COUNTER = itertools.count(1)


def ladder_window(prompt_tokens: int, num_predict: int) -> int:
    """The ``num_ctx`` a cell sends: room for the largest prompt the 2% tolerance accepts, the reply and a margin.

    Ollama truncates a prompt of ``num_ctx - 1`` tokens or more to about half the window, so the window is sized from
    the accepted prompt and never from a fixed margin the calibration can overshoot.
    """

    tolerance = max(1, int(prompt_tokens * LADDER_TOKEN_TOLERANCE))
    return prompt_tokens + tolerance + num_predict + LADDER_CTX_MARGIN


def ladder_prefix(nonce: str, call: int) -> str:
    """A fixed-width prefix that is unique to one call and differs from the first character.

    llama-server reuses the longest common token prefix of consecutive prompts, and Ollama cannot switch that off, so
    every timed call (and the calibration's verify call) starts with its own call number and the cell's nonce.
    """

    return f"{call:04d}{nonce} "


def _cell_nonce(model: str, prompt_tokens: int, concurrency: int) -> str:
    seed = f"{model}|{prompt_tokens}|{concurrency}|{next(_CELL_COUNTER)}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8]


def ladder_prompt(prompt_tokens: int, prefix: str = "") -> str:
    """``prefix``, then varied, deterministic text of ``prompt_tokens`` words, ending in an instruction.

    Version 1 sent ``"token " * n``. Some models answer that by repeating it until Ollama aborts the prediction, and the
    cell then timed a loop. Here the word order shifts every pass over the list, so no short period repeats, and the
    closing line asks for prose. ``GENERATE_LADDER_VERSION`` names this text.
    """

    count = max(1, prompt_tokens)
    size = len(_LADDER_WORDS)
    words = [_LADDER_WORDS[(i * 7 + (i // size) * 13 + i // (size * size)) % size] for i in range(count)]
    return prefix + " ".join(words) + "\n\n" + _LADDER_INSTRUCTION


def _post(server: Any, path: str, payload: dict[str, Any], timeout: float) -> Any:
    """``server.post`` with a request timeout, for servers that take one (a fake or an older server may not)."""

    if "timeout" in inspect.signature(server.post).parameters:
        return server.post(path, payload, timeout=timeout)
    return server.post(path, payload)


def request_timeout_s(concurrency: int, verify_s: float) -> int:
    """The budget for one timed call: three times what ``concurrency`` verify-sized requests need, plus decode."""

    scaled = math.ceil(REQUEST_TIMEOUT_FACTOR * max(1, concurrency) * verify_s) + REQUEST_TIMEOUT_DECODE_ALLOWANCE_S
    return int(min(max(scaled, REQUEST_TIMEOUT_FLOOR_S), REQUEST_TIMEOUT_CEILING_S))


def _evaluated_tokens(
    server: LinuxOllamaServer, model: str, prompt: str, num_ctx: int
) -> tuple[int, float | None]:
    """Ollama's ``prompt_eval_count`` for ``prompt`` and the request's own seconds (``total - load``), if reported.

    A one-token reply keeps this cheap. These are single requests, so they get the long calibration timeout.
    """

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_ctx": num_ctx, "num_predict": 1, "temperature": 0.0},
    }
    resp = _post(server, "/api/generate", payload, CALIBRATION_TIMEOUT_S)
    if isinstance(resp, Mapping) and resp.get("error"):
        raise RuntimeError(str(resp["error"]))
    evaluated = int(resp.get("prompt_eval_count") or 0) if isinstance(resp, Mapping) else 0
    if evaluated <= 0:
        raise LinuxShapeFailure("unmeasured", "Ollama reported no prompt_eval_count; the prompt length is unknown")
    total = int(resp.get("total_duration") or 0)
    seconds = (total - int(resp.get("load_duration") or 0)) / 1e9 if total > 0 else None
    return evaluated, seconds


def _evaluated_or_requested(measured: Mapping[str, Any], requested: int) -> tuple[int, str]:
    """The prompt length a ladder cell can vouch for, and how it knows."""

    reported = measured.get("prompt_tokens_evaluated")
    if isinstance(reported, int) and not isinstance(reported, bool):
        return reported, "ollama_prompt_eval_count"
    return requested, "requested_not_verified"


def _load_report(server: Any) -> dict[str, Any]:
    """The latest load's layers, host buffers and system memory from the server log; an old server has none."""

    reader = getattr(server, "load_report", None)
    report = reader() if callable(reader) else {}
    return report if isinstance(report, dict) else {}


def _check_fully_resident(
    server: LinuxOllamaServer, model: str, num_ctx: int, concurrency: int, peak_mib: int | None
) -> None:
    """Right after the first probe has loaded the model at the cell's window and slots, it must be fully on the GPU.

    A model that is partly in host RAM runs on the CPU for the rest of the cell and times out (T4 r3: 3 cells), and
    its verify call would be the thing that failed. Residency is read from ``/api/ps`` and from the load's log lines,
    and either one is enough. Only what was read is put in the reason.
    """

    try:
        size, size_vram = _query_ps_entry(server, model)
    except LinuxShapeFailure:
        size = size_vram = None
    report = _load_report(server)
    layers, total = report.get("layers_on_gpu"), report.get("layers_total")
    cpu_kv = report.get("cpu_kv_mib")
    by_layers = layers is not None and total is not None and layers < total
    by_kv = cpu_kv is not None and cpu_kv > 0
    by_ps = size is not None and size_vram is not None and size_vram < size
    if not (by_layers or by_kv or by_ps):
        return
    parts = []
    if by_layers:
        parts.append(f"{layers}/{total} layers on the GPU")
    if by_kv:
        parts.append(f"{cpu_kv:.0f} MiB of KV in host RAM")
    if by_ps:
        parts.append(f"size_vram {size_vram} < size {size}")
    offload = {
        "layers_on_gpu": layers,
        "layers_total": total,
        "cpu_kv_mib": cpu_kv,
        "size": size,
        "size_vram": size_vram,
        "source": " and ".join(
            name for name, hit in (("llama-server log", by_layers or by_kv), ("/api/ps", by_ps)) if hit
        ),
    }
    raise LinuxShapeFailure(
        "oom",
        f"partial offload at num_ctx {num_ctx} x {concurrency} slots: " + ", ".join(parts),
        peak_mib=peak_mib,
        evidence={"offload": offload, "request_timeout_s": CALIBRATION_TIMEOUT_S},
    )


def _failure_for(
    server: Any, exc: BaseException, peak_mib: int | None, evidence: dict[str, Any] | None = None
) -> LinuxShapeFailure:
    """A failure for ``exc``. A SIGKILL during a load is host RAM only when the log's numbers say so."""

    if isinstance(exc, LinuxShapeFailure):
        failure = exc
    elif "signal: killed" in str(exc).lower():
        report = _load_report(server)
        cpu_kv, mapped = report.get("cpu_kv_mib"), report.get("cpu_mapped_mib") or 0
        system = report.get("system_total_mib")
        if cpu_kv is not None and system is not None and cpu_kv + mapped > system:
            failure = LinuxShapeFailure(
                "oom",
                f"host RAM: the load needed {cpu_kv + mapped:.0f} MiB in host RAM (CPU KV {cpu_kv:.0f} + mapped "
                f"weights {mapped:.0f}); the VM has {system:.0f} MiB (Ollama system memory)",
                peak_mib=peak_mib,
                evidence={"offload": {**report, "source": "llama-server log"}},
            )
        else:
            failure = LinuxShapeFailure(
                "unmeasured", f"{exc} (the server log has no host-RAM numbers that explain it)", peak_mib=peak_mib
            )
    else:
        failure = classify_linux_error(exc, peak_mib=peak_mib)
    if evidence:
        failure.evidence = {**evidence, **(failure.evidence or {})}
    return failure


def _calibrate_ladder_prompt(
    server: LinuxOllamaServer, model: str, target: int, num_ctx: int, nonce: str, concurrency: int = 1
) -> tuple[int, int, float]:
    """The word count whose prompt (prefix included) Ollama evaluates to ``target`` tokens, the count it evaluated,
    and the seconds the accepted verify request took (``total - load``, a single full-length request).

    The first probe loads the model at the cell's window and slots, and must find it fully on the GPU. Ollama
    truncates a prompt that overflows ``num_ctx`` to about half of it, and reports the count it kept, so a reading
    from an overflowing prompt says nothing about its length. Two small prompts that cannot overflow give the
    tokens-per-word ratio and the fixed tokens (prefix and instruction); the word count follows, and is then verified
    with the real ``num_ctx``. A reading well under what the ratio predicts means truncation: the prompt is cut back to
    the target estimate and tried again. A prompt that cannot be brought within tolerance of the target is a cell that
    was not measured, never a row that records a length it did not time.
    """

    small, large = LADDER_PROBE_WORDS
    low, _ = _evaluated_tokens(server, model, ladder_prompt(small, ladder_prefix(nonce, 9001)), num_ctx)
    _check_fully_resident(server, model, num_ctx, concurrency, None)
    high, _ = _evaluated_tokens(server, model, ladder_prompt(large, ladder_prefix(nonce, 9002)), num_ctx)
    ratio = (high - low) / (large - small)
    if ratio <= 0 or high >= num_ctx * 0.9:
        raise LinuxShapeFailure(
            "unmeasured",
            f"could not size the prompt: {low} tokens for {small} words and {high} for {large} (num_ctx {num_ctx})",
        )
    fixed = max(0.0, low - ratio * small)
    tolerance = max(1, int(target * LADDER_TOKEN_TOLERANCE))
    estimate = max(1, round((target - fixed) / ratio))
    words = estimate
    evaluated = 0
    for attempt in range(4):
        prompt = ladder_prompt(words, ladder_prefix(nonce, 9100 + attempt))
        evaluated, verify_s = _evaluated_tokens(server, model, prompt, num_ctx)
        expected = fixed + ratio * words
        if evaluated < expected * 0.9:
            if words > estimate:
                words = estimate  # over the target and truncated: cut back to the estimate and look again
                continue
            raise LinuxShapeFailure(
                "unmeasured",
                f"the prompt was truncated: {words} words should evaluate to about {round(expected)} tokens "
                f"and evaluated to {evaluated} (num_ctx {num_ctx})",
            )
        if abs(evaluated - target) <= tolerance:
            if verify_s is None or verify_s <= 0:
                raise LinuxShapeFailure("unmeasured", "Ollama reported no total_duration for the verify call")
            return words, evaluated, verify_s
        words = max(1, round(words + (target - evaluated) / ratio))
    raise LinuxShapeFailure(
        "unmeasured",
        f"the prompt evaluated to {evaluated} tokens, not the target {target} (num_ctx {num_ctx}); "
        "the cell would time a different length than it records",
    )


def _ttft_does_not_scale(baseline: tuple[int, float] | None, prompt_len: int, prefill_ms: float) -> str | None:
    """A reason when a longer cell's prefill did not grow with its prompt, which means the prefill was cached."""

    if baseline is None or prompt_len <= baseline[0]:
        return None
    base_len, base_prefill = baseline
    required = TTFT_SCALING_FLOOR * base_prefill * (prompt_len / base_len)
    if prefill_ms >= required:
        return None
    return (
        f"prefill {prefill_ms:.1f} ms at {prompt_len} tokens is under {required:.1f} ms, {TTFT_SCALING_FLOOR:.0%} of "
        f"the growth from {base_prefill:.1f} ms at {base_len} tokens: the prefill did not scale with the prompt "
        "(a prompt-cache hit)"
    )


def measure_linux_ollama_generate(
    server: LinuxOllamaServer,
    sample_vram_mib: Callable[[], tuple[int, int]],
    *,
    model: str,
    prompt_tokens: int,
    concurrency: int,
    num_predict: int = 128,
    repeats: int = 3,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Measure one ``(prompt_tokens, concurrency)`` cell on a running ``LinuxOllamaServer``.

    The server runs ``concurrency`` slots; the model must load fully on the GPU; every timed call has its own prompt
    prefix and a budget scaled from the cell's own verify call; a prefill rate no card reaches is an unmeasured cell.
    """

    ensure_parallel = getattr(server, "ensure_parallel", None)
    if callable(ensure_parallel):
        ensure_parallel(max(1, concurrency))
    used_before, _ = sample_vram_mib()
    peak_mib = used_before
    num_ctx = ladder_window(prompt_tokens, num_predict)
    nonce = _cell_nonce(model, prompt_tokens, concurrency)
    calibration_budget = {"request_timeout_s": CALIBRATION_TIMEOUT_S}
    try:
        words, evaluated, verify_s = _calibrate_ladder_prompt(server, model, prompt_tokens, num_ctx, nonce, concurrency)
    except BaseException as exc:
        raise _failure_for(server, exc, peak_mib, calibration_budget) from exc
    budget_s = request_timeout_s(concurrency, verify_s)
    budget = {"request_timeout_s": budget_s}
    reported_slots = getattr(server, "effective_slots", lambda: None)()
    if isinstance(reported_slots, int) and not isinstance(reported_slots, bool):
        slot_count, slot_source = reported_slots, "llama-server log"
        if slot_count < concurrency:
            raise LinuxShapeFailure(
                "unmeasured",
                f"the server ran {slot_count} slot(s) for concurrency {concurrency}; the requests would queue",
                peak_mib=peak_mib,
                evidence=budget,
            )
    else:
        slot_count, slot_source = max(1, concurrency), "requested OLLAMA_NUM_PARALLEL (not read from the server log)"
    workers = max(1, concurrency)
    passes = ("warmup", *("repeat" for _ in range(max(1, repeats))))
    slots: list[dict[str, Any]] = []
    per_request: list[float] = []

    def _call_once(index: int) -> dict[str, Any]:
        payload = {
            "model": model,
            "prompt": ladder_prompt(words, ladder_prefix(nonce, index)),
            "stream": False,
            "options": {"num_ctx": num_ctx, "num_predict": num_predict, "temperature": 0.0},
        }
        resp = _post(server, "/api/generate", payload, budget_s)
        if isinstance(resp, Mapping) and resp.get("error"):
            raise RuntimeError(str(resp["error"]))
        return dict(resp) if isinstance(resp, Mapping) else {}

    try:
        for pass_no, phase in enumerate(passes):
            first = pass_no * workers + 1  # every call in the cell has its own number, hence its own prefix
            started = clock()
            if workers == 1:
                pass_resps = [_call_once(first)]
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    pass_resps = list(pool.map(_call_once, range(first, first + workers)))
            wall_s = max(clock() - started, 1e-9)
            used_now, _ = _sample_during_cell(sample_vram_mib)
            peak_mib = max(peak_mib, used_now)
            prefill_ns = [int(r.get("prompt_eval_duration") or 0) for r in pass_resps]
            if min(prefill_ns) <= 0:
                raise LinuxShapeFailure("unmeasured", "Ollama reported no prompt_eval_duration; the prefill is unknown")
            prefill_tok_s = evaluated / (min(prefill_ns) / 1_000_000_000.0)
            if prefill_tok_s > PREFILL_TOK_S_CEILING:
                raise LinuxShapeFailure(
                    "unmeasured",
                    f"a prefill of {evaluated} tokens in {min(prefill_ns) / 1e6:.2f} ms is {prefill_tok_s:,.0f} tok/s, "
                    f"over the {PREFILL_TOK_S_CEILING:,.0f} tok/s ceiling: the prompt came from llama-server's cache",
                    peak_mib=peak_mib,
                )
            eval_counts = [max(1, int(r.get("eval_count") or num_predict)) for r in pass_resps]
            eval_durations_ns = [int(r.get("eval_duration") or 0) for r in pass_resps]
            total_ns = [int(r.get("total_duration") or 0) for r in pass_resps]
            # Without Ollama's own clock for each request the decode window cannot be told from the queue wait and
            # the overhead, and the concurrency inflation (T4 r2) would come back. So no figure is guessed.
            if min(total_ns) <= 0 or min(eval_durations_ns) <= 0:
                raise LinuxShapeFailure(
                    "unmeasured", "Ollama reported no total_duration or eval_duration; the decode window is unknown"
                )
            load_ns = [int(r.get("load_duration") or 0) for r in pass_resps]
            rates = [
                count / (duration / 1_000_000_000.0)
                for count, duration in zip(eval_counts, eval_durations_ns, strict=True)
            ]
            if phase == "repeat":
                per_request.extend(rates)
            # Between the request and the first token there is time that is neither prefill nor decode (T4 r2, the
            # rig: up to 2.3 s at 16k), so the client's wall clock cannot give a clean decode rate; the replies' own
            # durations can.
            decode_window_s = max(
                (max(total_ns) - min(t - e for t, e in zip(total_ns, eval_durations_ns, strict=True))) / 1e9, 1e-9
            )
            ttft_ns = [t - e - ld for t, e, ld in zip(total_ns, eval_durations_ns, load_ns, strict=True)]
            overhead_ns = [
                t - ld - p - e for t, ld, p, e in zip(total_ns, load_ns, prefill_ns, eval_durations_ns, strict=True)
            ]
            slots.append(
                {
                    "phase": phase,
                    # The aggregate decode rate: the pass's generated tokens over its decode window, from the earliest
                    # decode start to the latest end, all on Ollama's own durations. At one request it is exactly
                    # eval_count / eval_duration.
                    "throughput": round(sum(eval_counts) / decode_window_s, 3),
                    "end_to_end_tok_s": round(sum(eval_counts) / wall_s, 3),  # the same tokens over the client's pass
                    "latency_ms": round(wall_s * 1000.0, 3),
                    # The time to the first token the user waits for: Ollama's total less the decode and the load. It
                    # includes the time that is neither prefill nor decode. The prefill alone is `prefill_ms`, and the
                    # cache check reads that one.
                    "ttft_ms": round((sum(ttft_ns) / len(ttft_ns)) / 1_000_000.0, 3),
                    "prefill_ms": round((sum(prefill_ns) / len(prefill_ns)) / 1_000_000.0, 3),
                    "overhead_ms": round((sum(overhead_ns) / len(overhead_ns)) / 1_000_000.0, 3),
                    "tokens": sum(eval_counts),
                    "requests": workers,
                }
            )
        size, size_vram = _query_ps_entry(server, model)
    except BaseException as exc:
        failure = _failure_for(server, exc, peak_mib, budget)
        if failure.failure_class == "timeout":
            failure.reason = (
                f"did not finish within {budget_s} s (scaled from a {verify_s:.1f} s verify call at concurrency "
                f"{concurrency})"
            )
            failure.args = (f"timeout: {failure.reason}",)
        raise failure from exc

    repeat_slots = [s for s in slots if s["phase"] == "repeat"]
    mid = len(repeat_slots) // 2
    ttft_ms = float(sorted(float(s["ttft_ms"]) for s in repeat_slots)[mid])
    prefill_ms = float(sorted(float(s["prefill_ms"]) for s in repeat_slots)[mid])
    overhead_ms = float(sorted(float(s["overhead_ms"]) for s in repeat_slots)[mid])
    decode_tok_s = float(sorted(float(s["throughput"]) for s in repeat_slots)[mid])
    total_ms = float(sorted(float(s["latency_ms"]) for s in repeat_slots)[mid])
    return {
        "ttft_ms": ttft_ms,
        "prefill_ms": prefill_ms,
        "overhead_ms": overhead_ms,
        "decode_tok_s": decode_tok_s,
        "latency_ms": total_ms,
        "peak_mib": peak_mib,
        "size": size,
        "size_vram": size_vram,
        "slots": slots,
        "prompt_tokens_evaluated": evaluated,
        "prompt_tokens_target": prompt_tokens,
        "num_ctx_sent": num_ctx,
        "parallel_slots": slot_count,
        "parallel_slots_source": slot_source,
        "throughput_method": "ollama_durations_decode_window",
        "end_to_end_tok_s": round(float(sorted(float(s["end_to_end_tok_s"]) for s in repeat_slots)[mid]), 3),
        "request_timeout_s": budget_s,
        "per_request_decode_tok_s": round(sum(per_request) / len(per_request), 3) if per_request else None,
    }


def _build_fixed_token_item(
    token_length: int,
    item_index: int = 0,
    seed: str = THROUGHPUT_SEED,
    *,
    model: str | None = None,
) -> str:
    if isinstance(token_length, bool) or not isinstance(token_length, int) or token_length < 1:
        raise LinuxCudaError(f"token_length must be a positive integer, got {token_length!r}")
    if model is not None:
        input_spec = get_throughput_input_spec(model, token_length)
        if input_spec is not None:
            return str(input_spec["text"])
    words = [w for w in seed.split() if w] or ["token"]
    chosen = [words[(item_index + i) % len(words)] for i in range(token_length)]
    return " ".join(chosen)


def _batch_endpoint_and_payload(
    model: str,
    task: str,
    batch: int,
    *,
    token_length: int = 510,
) -> tuple[str, dict[str, Any]]:
    if task == "embed":
        items = [_build_fixed_token_item(token_length, idx, model=model) for idx in range(batch)]
        return "/api/embed", {"model": model, "input": items}
    if task in ("rerank", "nli"):
        raise LinuxCudaError(
            f"out of scope: stock Ollama does not serve {task!r} (/api/{task}); "
            "declare the entry as a pre-registered load_refused row"
        )
    raise LinuxCudaError(f"unsupported batch task: {task!r}")


def measure_linux_ollama_batch(
    server: LinuxOllamaServer,
    sample_vram_mib: Callable[[], tuple[int, int]],
    *,
    model: str,
    task: str,
    batch: int,
    token_length: int = 510,
    repeats: int = 3,
) -> dict[str, Any]:
    """Measure one ``(task, token_length, batch)`` cell on a running ``LinuxOllamaServer``."""

    endpoint, payload = _batch_endpoint_and_payload(model, task, batch, token_length=token_length)
    used_before, _ = sample_vram_mib()
    peak_mib = used_before
    passes = ("warmup", *("repeat" for _ in range(max(1, repeats))))
    slots: list[dict[str, Any]] = []
    try:
        for phase in passes:
            resp = server.post(endpoint, payload)
            if isinstance(resp, Mapping) and resp.get("error"):
                raise RuntimeError(str(resp["error"]))
            used_after, _ = _sample_during_cell(sample_vram_mib)
            peak_mib = max(peak_mib, used_after)
            total_ns = max(1, int(resp.get("total_duration") or 10_000_000))
            pass_lat_ms = round(total_ns / 1_000_000.0, 3)
            pass_tput = round(batch / (total_ns / 1_000_000_000.0), 3)
            slots.append({"phase": phase, "throughput": pass_tput, "latency_ms": pass_lat_ms})
        size, size_vram = _query_ps_entry(server, model)
    except BaseException as exc:
        raise classify_linux_error(exc, peak_mib=peak_mib) from exc

    repeat_slots = [s for s in slots if s["phase"] == "repeat"]
    latencies = sorted(float(s["latency_ms"]) for s in repeat_slots)
    tputs = sorted(float(s["throughput"]) for s in repeat_slots)
    mid = len(repeat_slots) // 2
    return {
        "throughput": tputs[mid],
        "warm_p50_ms": latencies[mid],
        "warm_p95_ms": latencies[-1],
        "peak_mib": peak_mib,
        "size": size,
        "size_vram": size_vram,
        "slots": slots,
    }


def _ollama_model_tag(model_tag: str, quant: str, explicit_tag: Any = None) -> str:
    if isinstance(explicit_tag, str) and explicit_tag.strip():
        return explicit_tag.strip()
    base = model_tag
    if base.startswith("sha256:") and len(base) >= len("sha256:") + 12:
        base = f"c2-sha256-{base.removeprefix('sha256:')[:12]}"
    elif "/" in base:
        base = re.sub(r"[^a-zA-Z0-9._-]+", "-", base).strip("-").lower()
    if quant and quant != UNRECORDED:
        return f"{base}-{quant.lower()}"
    return base


def _relative_spec_posix(spec_path: Path, spec_rel: str | None) -> str:
    if spec_rel:
        return spec_rel
    parts = spec_path.parts
    if "experiments" in parts:
        idx = parts.index("experiments")
        return Path(*parts[idx:]).as_posix()
    return spec_path.name if spec_path.is_absolute() else spec_path.as_posix()


@contextmanager
def _unloaded_after(server: Any, tag: str) -> Iterator[None]:
    """Unload ``tag`` when its cells end, however they end, so the next quant's baseline is an empty card."""

    try:
        yield
    finally:
        try:
            server.unload(tag)
        except (OSError, RuntimeError, ValueError):
            pass


def run_linux_spec(
    spec: Mapping[str, Any],
    *,
    spec_path: Path,
    run_id: str,
    date: str,
    spec_rel: str | None = None,
    remote_device: RemoteDeviceIdentity | None = None,
    sample_vram_mib: Callable[[], tuple[int, int]] | None = None,
    server: LinuxOllamaServer | None = None,
    on_rows: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    """Execute a ``platform: linux-cuda`` experiment spec and return validated routing rows."""

    if remote_device is None or sample_vram_mib is None:
        probed_device, probed_sampler = probe_linux_gpu()
        remote_device = remote_device or probed_device
        sample_vram_mib = sample_vram_mib or probed_sampler

    owns_server = server is None
    active_server = server or LinuxOllamaServer()
    if owns_server:
        active_server.start()

    try:
        engine_version = active_server.version or UNRECORDED
        spec_id = str(spec["id"])
        resolved_spec_rel = _relative_spec_posix(spec_path, spec_rel)
        mode = str(spec.get("mode") or spec.get("procedure"))
        abort_fraction = float(spec.get("abort_fraction", 0.90))
        abort_mib = compute_abort_mib(remote_device.vram_mib, abort_fraction)
        models = list(spec.get("models") or [])
        rows: list[dict[str, Any]] = _ProgressRows(on_rows)

        def progress(part: list[dict[str, Any]]) -> None:
            if on_rows is not None:
                on_rows([*rows, *part])


        if mode == "fill-card":
            grain = int(spec.get("grain", 2048))
            spill_ratio = float(spec.get("spill_prompt_ratio", 0.5))
            for entry in models:
                card_filter = entry.get("card") or entry.get("device")
                if card_filter and card_filter != remote_device.logical_device:
                    continue
                model_tag = str(entry.get("repo_or_tag") or entry.get("model"))
                digest = str(entry.get("revision_or_digest", UNRECORDED))
                trained_ctx = int(entry.get("trained_ctx", 32768))
                quant = str(entry.get("quant", UNRECORDED))
                licence = str(entry.get("licence", UNRECORDED))
                expected_outcome = entry.get("expected_outcome")
                if isinstance(expected_outcome, str) and expected_outcome in FAILURE_CLASSES:
                    dev_desc = f"{remote_device.logical_device} ({remote_device.vram_mib} MiB)"
                    reason = str(
                        entry.get("failure_reason")
                        or f"pre-registered {expected_outcome}: {model_tag} ({quant}) exceeds {dev_desc}"
                    )
                    rows.append(
                        build_linux_failure_row(
                            spec_id=spec_id,
                            spec_rel=resolved_spec_rel,
                            run_id=run_id,
                            date=date,
                            task="generate",
                            model=model_tag,
                            revision_or_digest=digest,
                            remote_device=remote_device,
                            failure_class=expected_outcome,
                            reason=reason,
                            num_ctx=grain,
                            abort_mib=abort_mib,
                            engine_version=engine_version,
                            quant=quant,
                            licence=licence,
                        )
                    )
                    continue
                ollama_tag = _ollama_model_tag(model_tag, quant, entry.get("ollama_tag"))
                with _unloaded_after(active_server, ollama_tag):
                    rows.append(
                        run_linux_fill_card(
                            spec_id=spec_id,
                            spec_rel=resolved_spec_rel,
                            run_id=run_id,
                            date=date,
                            remote_device=remote_device,
                            model=model_tag,
                            revision_or_digest=digest,
                            trained_ctx=trained_ctx,
                            abort_fraction=abort_fraction,
                            grain=grain,
                            spill_prompt_ratio=spill_ratio,
                            engine_version=engine_version,
                            quant=quant,
                            licence=licence,
                            measure_window=lambda ctx, m=ollama_tag: measure_linux_ollama_window(
                                active_server, sample_vram_mib, model=m, num_ctx=ctx
                            ),
                        )
                    )
            return rows

        if mode == "generate-ladder":
            prompt_lengths = tuple(int(x) for x in (spec.get("prompt_lengths") or GENERATE_PROMPT_LENGTHS))
            concurrencies = tuple(int(x) for x in (spec.get("concurrencies") or GENERATE_CONCURRENCIES))
            num_predict = int(spec.get("num_predict", 128))
            for entry in models:
                card_filter = entry.get("card") or entry.get("device")
                if card_filter and card_filter != remote_device.logical_device:
                    continue
                model_tag = str(entry.get("repo_or_tag") or entry.get("model"))
                digest = str(entry.get("revision_or_digest", UNRECORDED))
                quant = str(entry.get("quant", UNRECORDED))
                licence = str(entry.get("licence", UNRECORDED))
                expected_outcome = entry.get("expected_outcome")
                if isinstance(expected_outcome, str) and expected_outcome in FAILURE_CLASSES:
                    dev_desc = f"{remote_device.logical_device} ({remote_device.vram_mib} MiB)"
                    reason = str(
                        entry.get("failure_reason")
                        or f"pre-registered {expected_outcome}: {model_tag} ({quant}) exceeds {dev_desc}"
                    )
                    for prompt_len in prompt_lengths:
                        for concurrency in concurrencies:
                            rows.append(
                                build_linux_failure_row(
                                    spec_id=spec_id,
                                    spec_rel=resolved_spec_rel,
                                    run_id=run_id,
                                    date=date,
                                    task="generate",
                                    model=model_tag,
                                    revision_or_digest=digest,
                                    remote_device=remote_device,
                                    failure_class=expected_outcome,
                                    reason=reason,
                                    num_ctx=prompt_len,
                                    num_predict=num_predict,
                                    prompt_tokens=prompt_len,
                                    concurrency=concurrency,
                                    abort_mib=abort_mib,
                                    engine_version=engine_version,
                                    quant=quant,
                                    licence=licence,
                                )
                            )
                    continue
                ollama_tag = _ollama_model_tag(model_tag, quant, entry.get("ollama_tag"))
                with _unloaded_after(active_server, ollama_tag):
                    rows.extend(
                        run_linux_generate_ladder(
                            spec_id=spec_id,
                            spec_rel=resolved_spec_rel,
                            run_id=run_id,
                            date=date,
                            remote_device=remote_device,
                            model=model_tag,
                            revision_or_digest=digest,
                            prompt_lengths=prompt_lengths,
                            concurrencies=concurrencies,
                            abort_fraction=abort_fraction,
                            num_predict=num_predict,
                            on_progress=progress,
                            engine_version=engine_version,
                            quant=quant,
                            licence=licence,
                            measure_shape=lambda plen, conc, m=ollama_tag: measure_linux_ollama_generate(
                                active_server,
                                sample_vram_mib,
                                model=m,
                                prompt_tokens=plen,
                                concurrency=conc,
                                num_predict=num_predict,
                            ),
                        )
                    )
            return rows

        if mode == "batch-ladder":
            tasks = tuple(str(t) for t in (spec.get("tasks") or THROUGHPUT_TASKS))
            batch_sizes = tuple(int(b) for b in (spec.get("batch_sizes") or THROUGHPUT_BATCH_SIZES))
            token_lengths = tuple(int(tl) for tl in (spec.get("token_lengths") or THROUGHPUT_TOKEN_LENGTHS))
            for task in tasks:
                task_entries = [
                    e
                    for e in models
                    if str(e.get("task")) == task
                    and (
                        not (e.get("card") or e.get("device"))
                        or (e.get("card") or e.get("device")) == remote_device.logical_device
                    )
                ]
                for entry in task_entries:
                    model_tag = str(entry.get("repo_or_tag") or entry.get("model"))
                    digest = str(entry.get("revision_or_digest", UNRECORDED))
                    quant = str(entry.get("quant", UNRECORDED))
                    licence = str(entry.get("licence", UNRECORDED))
                    expected_outcome = entry.get("expected_outcome")
                    if isinstance(expected_outcome, str) and expected_outcome in FAILURE_CLASSES:
                        dev_desc = f"{remote_device.logical_device} ({remote_device.vram_mib} MiB)"
                        reason = str(
                            entry.get("failure_reason")
                            or f"pre-registered {expected_outcome}: {model_tag} ({quant}) on {dev_desc}"
                        )
                        for token_len in token_lengths:
                            for batch in batch_sizes:
                                rows.append(
                                    build_linux_failure_row(
                                        spec_id=spec_id,
                                        spec_rel=resolved_spec_rel,
                                        run_id=run_id,
                                        date=date,
                                        task=task,
                                        model=model_tag,
                                        revision_or_digest=digest,
                                        remote_device=remote_device,
                                        failure_class=expected_outcome,
                                        reason=reason,
                                        batch=batch,
                                        token_length=token_len,
                                        abort_mib=abort_mib,
                                        engine_version=engine_version,
                                        quant=quant,
                                        licence=licence,
                                    )
                                )
                        continue
                    ollama_tag = _ollama_model_tag(model_tag, quant, entry.get("ollama_tag"))
                    with _unloaded_after(active_server, ollama_tag):
                        rows.extend(
                            run_linux_throughput_ladder(
                                spec_id=spec_id,
                                spec_rel=resolved_spec_rel,
                                run_id=run_id,
                                date=date,
                                remote_device=remote_device,
                                models_by_task={task: (model_tag, digest)},
                                tasks=(task,),
                                batch_sizes=batch_sizes,
                                token_lengths=token_lengths,
                                abort_fraction=abort_fraction,
                                on_progress=progress,
                                engine_version=engine_version,
                                quant=quant,
                                licence=licence,
                                measure_batch=lambda t, b, token_length=510, m=ollama_tag: measure_linux_ollama_batch(
                                    active_server,
                                    sample_vram_mib,
                                    model=m,
                                    task=t,
                                    batch=b,
                                    token_length=token_length,
                                ),
                            )
                        )
            return rows

        raise LinuxCudaError(f"unsupported linux-cuda mode: {mode!r}")
    finally:
        if owns_server:
            active_server.stop()


def resolve_c2_sources(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Resolve every model entry in a C2 ``linux-cuda`` spec to its single source, served tag, digest, and command."""

    if spec.get("platform") != "linux-cuda":
        raise LinuxCudaError("resolve_c2_sources requires a spec with platform: linux-cuda")
    models = spec.get("models")
    if not isinstance(models, list) or not models:
        raise LinuxCudaError("resolve_c2_sources requires a non-empty models roster")

    resolved: list[dict[str, Any]] = []
    for entry in models:
        device = str(entry.get("device") or entry.get("card") or UNRECORDED)
        model = str(entry.get("repo_or_tag") or entry.get("model") or UNRECORDED)
        quant = str(entry.get("quant") or UNRECORDED)
        task = str(entry.get("task") or "generate")
        digest = str(entry.get("revision_or_digest") or UNRECORDED)
        expected_outcome = entry.get("expected_outcome")
        failure_reason = str(entry.get("failure_reason") or "").strip() or None

        has_tag = bool(entry.get("ollama_tag"))
        has_gguf = entry.get("gguf") is not None
        is_declared_refusal = expected_outcome == "load_refused" and bool(
            failure_reason
            and (failure_reason.startswith("no published ") or "out of scope" in failure_reason.lower())
        )

        if has_tag and has_gguf:
            raise LinuxCudaError(f"entry {model!r} ({quant!r}) on {device} cannot name both ollama_tag and gguf")
        if has_tag:
            served_tag = str(entry["ollama_tag"]).strip()
            tag_lower = served_tag.lower()
            if tag_lower.endswith(("-fp16", "-f16")) and quant != "FP16":
                raise LinuxCudaError(
                    f"entry {model!r} on {device} pulls fp16 tag {served_tag!r} so quant must be 'FP16', not {quant!r}"
                )
            if tag_lower.endswith("-bf16") and quant != "BF16":
                raise LinuxCudaError(
                    f"entry {model!r} on {device} pulls bf16 tag {served_tag!r} so quant must be 'BF16', not {quant!r}"
                )
            resolved.append(
                {
                    "device": device,
                    "task": task,
                    "model": model,
                    "quant": quant,
                    "source_kind": "ollama_tag",
                    "served_tag": served_tag,
                    "expected_digest": digest,
                    "command": f"ollama pull {served_tag}",
                    "expected_outcome": expected_outcome,
                    "failure_reason": failure_reason,
                }
            )
            continue
        if has_gguf:
            gguf = entry["gguf"]
            served_tag = _ollama_model_tag(model, quant)
            repo = str(gguf["repo"]).strip()
            rev = str(gguf["revision"]).strip()
            file = str(gguf["file"]).strip()
            sha256 = str(gguf["sha256"]).strip()
            file_lower = file.lower()
            if file_lower.endswith(("-f16.gguf", "-fp16.gguf")) and quant != "FP16":
                raise LinuxCudaError(
                    f"entry {model!r} on {device} pulls fp16 GGUF {file!r} so quant must be 'FP16', not {quant!r}"
                )
            if file_lower.endswith("-bf16.gguf") and quant != "BF16":
                raise LinuxCudaError(
                    f"entry {model!r} on {device} pulls bf16 GGUF {file!r} so quant must be 'BF16', not {quant!r}"
                )
            resolved.append(
                {
                    "device": device,
                    "task": task,
                    "model": model,
                    "quant": quant,
                    "source_kind": "gguf",
                    "served_tag": served_tag,
                    "repo": repo,
                    "revision": rev,
                    "file": file,
                    "expected_digest": sha256,
                    "command": f"hf download {repo} {file} --revision {rev} && ollama create {served_tag} -f Modelfile",
                    "expected_outcome": expected_outcome,
                    "failure_reason": failure_reason,
                }
            )
            continue
        if is_declared_refusal:
            resolved.append(
                {
                    "device": device,
                    "task": task,
                    "model": model,
                    "quant": quant,
                    "source_kind": "load_refused",
                    "served_tag": None,
                    "expected_digest": digest,
                    "command": f"skip-pull ({failure_reason})",
                    "expected_outcome": expected_outcome,
                    "failure_reason": failure_reason,
                }
            )
            continue
        raise LinuxCudaError(f"entry {model!r} ({quant!r}) on {device} must name exactly one source")

    return resolved


def format_c2_sources(spec: Mapping[str, Any]) -> str:
    """Format resolved C2 model sources as tab-separated lines for the window launcher."""

    items = resolve_c2_sources(spec)
    lines = [
        f"{item['device']}\t{item['model']}\t{item['quant']}\t{item['source_kind']}\t"
        f"{item['served_tag'] or '-'}\t{item['expected_digest']}\t{item['command']}"
        for item in items
    ]
    return "\n".join(lines) + ("\n" if lines else "")


_CANONICAL_CARD_VRAM_MIB: dict[str, int] = {
    "colab:T4": 15360,
    "colab:L4": 23034,
    "colab:A100-40": 40960,
    "colab:A100-80": 81920,
    "colab:G4": 98304,
}


def c2_sources_user_agent() -> str:
    """The User-Agent the live registry and Hugging Face reads send. It follows the package version."""

    return f"interlock-c2-sources/{__version__}"


def _default_live_fetch(url: str) -> tuple[int, bytes]:
    if not url.startswith("https://"):
        raise LinuxCudaError(f"verify_c2_sources_live only permits https:// URLs, got {url!r}")
    headers = {"User-Agent": c2_sources_user_agent()}
    if "registry.ollama.ai" in url and "/manifests/" in url:
        headers["Accept"] = "application/vnd.docker.distribution.manifest.v2+json"
    req = urllib.request.Request(url, headers=headers)  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=30.0) as resp:  # noqa: S310
            return int(resp.status), resp.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), b""
    except OSError as exc:
        raise LinuxCudaError(f"network error fetching {url}: {exc}") from exc


def _canonical_quant_from_file_type(file_type: str) -> str:
    upper = file_type.strip().upper()
    if upper in ("F16", "FP16"):
        return "FP16"
    if upper == "BF16":
        return "BF16"
    return upper


def verify_c2_sources_live(
    spec: Mapping[str, Any],
    *,
    fetch: Callable[[str], tuple[int, bytes]] | None = None,
) -> list[dict[str, Any]]:
    """Query the live Ollama registry and Hugging Face for every entry in ``spec``."""

    items = resolve_c2_sources(spec)
    raw_fetch = fetch or _default_live_fetch
    cache: dict[str, tuple[int, bytes]] = {}

    def get_url(url: str) -> tuple[int, bytes]:
        if url not in cache:
            cache[url] = raw_fetch(url)
        return cache[url]

    verified: list[dict[str, Any]] = []
    for item in items:
        source_kind = item["source_kind"]
        quant = item["quant"]
        device = item["device"]
        model = item["model"]
        expected_digest = item["expected_digest"]
        expected_outcome = item.get("expected_outcome")
        failure_reason = str(item.get("failure_reason") or "")

        if source_kind == "ollama_tag":
            served_tag = str(item["served_tag"])
            repo, tag = served_tag.split(":", 1) if ":" in served_tag else ("library", served_tag)
            manifest_url = f"https://registry.ollama.ai/v2/library/{repo}/manifests/{tag}"
            status, manifest_bytes = get_url(manifest_url)
            if status != 200:
                raise LinuxCudaError(f"live check failed for {served_tag} ({manifest_url}): HTTP {status}")
            actual_digest = hashlib.sha256(manifest_bytes).hexdigest()
            if actual_digest != expected_digest:
                raise LinuxCudaError(
                    f"live digest mismatch for {served_tag}: expected {expected_digest}, got {actual_digest}"
                )
            manifest_doc = json.loads(manifest_bytes.decode("utf-8"))
            config_digest = str((manifest_doc.get("config") or {}).get("digest") or "")
            blob_url = f"https://registry.ollama.ai/v2/library/{repo}/blobs/{config_digest}"
            c_status, config_bytes = get_url(blob_url)
            if c_status != 200:
                raise LinuxCudaError(f"live config check failed for {served_tag} ({blob_url}): HTTP {c_status}")
            config_doc = json.loads(config_bytes.decode("utf-8"))
            file_type = str(config_doc.get("file_type") or "")
            expected_quant = _canonical_quant_from_file_type(file_type)
            if expected_quant != quant:
                raise LinuxCudaError(
                    f"live label mismatch for {served_tag} on {device}: "
                    f"config file_type is {file_type!r} ({expected_quant}), but spec declares quant={quant!r}"
                )
            model_bytes = 0
            for layer in manifest_doc.get("layers") or []:
                if isinstance(layer, Mapping) and layer.get("mediaType") == "application/vnd.ollama.image.model":
                    model_bytes = int(layer.get("size") or 0)
                    break
            model_mib = round(model_bytes / (1024 * 1024)) if model_bytes else 0
            card_vram = _CANONICAL_CARD_VRAM_MIB.get(device)
            if expected_outcome == "load_refused" and card_vram is not None and "MiB" in failure_reason:
                abort_ceiling = compute_abort_mib(card_vram, float(spec.get("abort_fraction", 0.90)))
                if model_mib < abort_ceiling:
                    raise LinuxCudaError(
                        f"live size check failed for {served_tag} on {device}: "
                        f"model layer is {model_mib} MiB < abort ceiling {abort_ceiling} MiB, so refusal is stale"
                    )
            verified.append(
                {
                    **item,
                    "live_status": "PASS",
                    "live_digest": actual_digest,
                    "live_file_type": file_type,
                    "live_model_bytes": model_bytes,
                    "live_model_mib": model_mib,
                }
            )
            continue

        if source_kind == "gguf":
            repo = str(item["repo"])
            rev = str(item["revision"])
            file = str(item["file"])
            tree_url = f"https://huggingface.co/api/models/{repo}/tree/{rev}"
            status, tree_bytes = get_url(tree_url)
            if status != 200:
                raise LinuxCudaError(f"live check failed for HF GGUF {repo}@{rev} ({tree_url}): HTTP {status}")
            tree_doc = json.loads(tree_bytes.decode("utf-8"))
            matched_entry = next(
                (e for e in tree_doc if isinstance(e, Mapping) and e.get("path") == file),
                None,
            )
            if matched_entry is None:
                raise LinuxCudaError(f"live check failed for {repo}@{rev}: file {file!r} not found in tree")
            lfs = matched_entry.get("lfs") if isinstance(matched_entry.get("lfs"), Mapping) else {}
            actual_sha = str(lfs.get("oid") or matched_entry.get("oid") or "")
            if actual_sha != expected_digest:
                raise LinuxCudaError(
                    f"live digest mismatch for {repo}/{file}@{rev}: expected {expected_digest}, got {actual_sha}"
                )
            file_size = int(lfs.get("size") or matched_entry.get("size") or 0)
            verified.append(
                {
                    **item,
                    "live_status": "PASS",
                    "live_digest": actual_sha,
                    "live_file_type": quant,
                    "live_model_bytes": file_size,
                    "live_model_mib": round(file_size / (1024 * 1024)) if file_size else 0,
                }
            )
            continue

        if source_kind == "load_refused":
            if failure_reason == "no published Q6_K build":
                if model.startswith("sha256:"):
                    ollama_repo, base_tag = "gemma4", "31b"
                    cand_tags = ("31b-it-q6_K", "31b-q6_K")
                else:
                    ollama_repo, base_tag = model.split(":", 1) if ":" in model else (model, "latest")
                    cand_tags = (f"{base_tag}-q6_K", f"{base_tag}-it-q6_K")
                base_url = f"https://registry.ollama.ai/v2/library/{ollama_repo}/manifests/{base_tag}"
                b_status, b_bytes = get_url(base_url)
                if b_status != 200:
                    raise LinuxCudaError(f"live check failed for base tag {ollama_repo}:{base_tag}: HTTP {b_status}")
                b_digest = hashlib.sha256(b_bytes).hexdigest()
                if b_digest != expected_digest:
                    raise LinuxCudaError(
                        f"live base digest mismatch for {ollama_repo}:{base_tag}: "
                        f"expected {expected_digest}, got {b_digest}"
                    )
                for c_tag in cand_tags:
                    c_url = f"https://registry.ollama.ai/v2/library/{ollama_repo}/manifests/{c_tag}"
                    c_status, _ = get_url(c_url)
                    if c_status == 200:
                        raise LinuxCudaError(
                            f"live check failed: {ollama_repo}:{c_tag} now resolves (HTTP 200), "
                            "so 'no published Q6_K build' is stale"
                        )
                verified.append(
                    {
                        **item,
                        "live_status": "PASS",
                        "live_digest": b_digest,
                        "live_file_type": "unpublished_404_verified",
                        "live_model_bytes": 0,
                        "live_model_mib": 0,
                    }
                )
                continue

            if "out of scope" in failure_reason.lower():
                tree_url = f"https://huggingface.co/api/models/{model}/tree/{expected_digest}"
                status, _ = get_url(tree_url)
                if status != 200:
                    raise LinuxCudaError(
                        f"live check failed for HF revision {model}@{expected_digest} ({tree_url}): HTTP {status}"
                    )
                verified.append(
                    {
                        **item,
                        "live_status": "PASS",
                        "live_digest": expected_digest,
                        "live_file_type": "hf_revision_verified",
                        "live_model_bytes": 0,
                        "live_model_mib": 0,
                    }
                )
                continue

    if str(spec.get("procedure") or "") == "batch-ladder":
        fixture = load_throughput_inputs_fixture()
        seen_models: set[str] = set()
        for item in verified:
            m_name = item["model"]
            if m_name in seen_models:
                continue
            seen_models.add(m_name)
            m_fix = (fixture.get("models") or {})[m_name]
            hf_repo = str(m_fix["hf_repo"])
            hf_rev = str(m_fix["hf_revision"])
            tok_url = f"https://huggingface.co/{hf_repo}/resolve/{hf_rev}/tokenizer.json"
            t_status, t_bytes = get_url(tok_url)
            if t_status != 200:
                raise LinuxCudaError(f"live tokenizer check failed for {hf_repo}@{hf_rev}: HTTP {t_status}")
            actual_tok_sha = hashlib.sha256(t_bytes).hexdigest()
            if actual_tok_sha != m_fix["tokenizer_sha256"]:
                raise LinuxCudaError(
                    f"live tokenizer.json sha256 mismatch for {hf_repo}@{hf_rev}: "
                    f"expected {m_fix['tokenizer_sha256']}, got {actual_tok_sha}"
                )

    return verified


def format_c2_sources_live(
    spec: Mapping[str, Any],
    *,
    fetch: Callable[[str], tuple[int, bytes]] | None = None,
) -> str:
    """Verify every C2 spec entry live and format the evidence table."""

    verified = verify_c2_sources_live(spec, fetch=fetch)
    lines = [
        f"{item['device']}\t{item['model']}\t{item['quant']}\t{item['source_kind']}\t"
        f"{item['served_tag'] or '-'}\t{item['live_digest']}\t{item['live_file_type']}\t"
        f"{item['live_model_mib']}MiB\t{item['live_status']}"
        for item in verified
    ]
    lines.append(
        f"VERIFY-LIVE PASS: {len(verified)}/{len(verified)} entries verified "
        "against registry.ollama.ai and huggingface.co"
    )
    return "\n".join(lines) + "\n"



