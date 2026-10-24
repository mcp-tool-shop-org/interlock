"""Linux CUDA measuring path for remote Colab cards alongside the Windows rig path.

Unlike E5 on Windows (which reads ``\\GPU Adapter Memory(*)\\Shared Usage`` via
PowerShell ``Get-Counter``), Linux CUDA has no WDDM shared-memory counter.
Residency comes from Ollama ``/api/ps`` (``size_vram == size``) and peak VRAM,
total VRAM, driver version, and CUDA version come from NVML (``pynvml`` /
``nvidia-ml-py``) or ``nvidia-smi --query-gpu``.

Failed shapes (``oom``, ``load_refused``, ``timeout``, ``ctx_exceeded``) are
recorded as validated routing rows, never silently skipped.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from interlock.devices import COLAB_CARDS, RemoteDeviceIdentity
from interlock.families import stamp_family
from interlock.receipt import validate_row

UNRECORDED = "unrecorded"
FAILURE_CLASSES = ("oom", "load_refused", "timeout", "ctx_exceeded")
GENERATE_PROMPT_LENGTHS = (512, 1024, 2048, 4096, 8192, 16384, 32768)
GENERATE_CONCURRENCIES = (1, 4, 8)
THROUGHPUT_TASKS = ("embed", "rerank", "nli")
THROUGHPUT_BATCH_SIZES = (1, 8, 32)
FORBIDDEN_SHARED_PORTS = frozenset({11434, 11490})

_CUDA_HEADER = re.compile(r"CUDA\s+(?:UMD\s+)?Version\s*:\s*([0-9]+\.[0-9]+)")


class LinuxCudaError(Exception):
    """Refusal or probe error in the Linux CUDA measuring path."""


class LinuxShapeFailure(Exception):
    """A measured shape failed on the remote card and must be recorded as a failure row."""

    def __init__(self, failure_class: str, reason: str, *, peak_mib: int | None = None) -> None:
        if failure_class not in FAILURE_CLASSES:
            raise LinuxCudaError(
                f"unknown Linux CUDA failure class {failure_class!r}; expected one of {FAILURE_CLASSES}"
            )
        if not isinstance(reason, str) or not reason.strip():
            raise LinuxCudaError("LinuxShapeFailure requires a non-empty reason")
        super().__init__(f"{failure_class}: {reason.strip()}")
        self.failure_class = failure_class
        self.reason = reason.strip()
        self.peak_mib = peak_mib


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
    if upper in COLAB_CARDS:
        return upper
    if "A100" in upper:
        return "A100-80" if total_mib >= 60000 or "80GB" in upper else "A100-40"
    if "L4" in upper and "L40" not in upper:
        return "L4"
    if "T4" in upper:
        return "T4"
    if "G4" in upper or "RTX PRO 6000" in upper or "BLACKWELL" in upper:
        return "G4"
    raise LinuxCudaError(f"unrecognized Colab GPU name {raw_name!r} ({total_mib} MiB)")


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


class LinuxOllamaServer:
    """Run-owned Ollama child process on a Linux VM, started and stopped by the harness."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1:11441",
        log_path: Path,
        exe: str = "ollama",
        popen: Callable[..., Any] | None = None,
        http_get: Callable[[str], dict[str, Any]] | None = None,
        http_post: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        if ":" not in host:
            raise LinuxCudaError(f"host must include port: {host!r}")
        _, port_text = host.rsplit(":", 1)
        port = int(port_text)
        if port in FORBIDDEN_SHARED_PORTS:
            raise LinuxCudaError(
                f"refusing shared port {port}; Linux harness must launch its own Ollama on a dedicated port"
            )
        self.host = host
        self.base_url = f"http://{host}"
        self.log_path = log_path
        self.exe = exe
        self._popen = popen or subprocess.Popen
        self._http_get = http_get
        self._http_post = http_post
        self._proc: Any = None
        self.version: str = UNRECORDED

    @property
    def pid(self) -> int | None:
        return None if self._proc is None else int(self._proc.pid)

    def __enter__(self) -> LinuxOllamaServer:
        env = dict(os.environ)
        env["OLLAMA_HOST"] = self.host
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._proc = self._popen([self.exe, "serve"], env=env)
        if self._http_get is not None:
            reply = self._http_get(f"{self.base_url}/api/version")
            self.version = str(reply.get("version") or UNRECORDED)
        return self

    def unload(self, model: str) -> None:
        if self._http_post is not None:
            self._http_post(f"{self.base_url}/api/generate", {"model": model, "keep_alive": 0})

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            self._proc.wait(timeout=5.0)


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
    engine_version: str = "0.35.1",
    quant: str = UNRECORDED,
    licence: str = UNRECORDED,
    num_ctx: int | None = None,
    num_predict: int | None = None,
    batch: int | None = None,
    prompt_tokens: int | None = None,
    concurrency: int | None = None,
    peak_mib: int | None = None,
    abort_mib: int | None = None,
) -> dict[str, Any]:
    """Build a schema-valid capability/failure row for a failed remote shape."""

    if failure_class not in FAILURE_CLASSES:
        raise LinuxCudaError(f"invalid failure_class {failure_class!r}")
    if not isinstance(reason, str) or not reason.strip():
        raise LinuxCudaError("failure row requires a non-empty measured reason")
    clean_reason = reason.strip()
    note = clean_reason if clean_reason.startswith(f"{failure_class}:") else f"{failure_class}: {clean_reason}"
    shape_tag = _shape_suffix(num_ctx=num_ctx, prompt_tokens=prompt_tokens, concurrency=concurrency, batch=batch)
    settings: dict[str, Any] = {
        "thinking_level": "off",
        "num_predict": num_predict,
        "num_ctx": num_ctx,
        "batch": batch,
        "static_shape": None,
        "precision": None,
        "failure_class": failure_class,
        "failure_reason": clean_reason,
    }
    if prompt_tokens is not None:
        settings["prompt_tokens"] = prompt_tokens
    if concurrency is not None:
        settings["concurrency"] = concurrency
    if peak_mib is not None:
        settings["peak_mib"] = peak_mib
    if abort_mib is not None:
        settings["abort_mib"] = abort_mib
    row: dict[str, Any] = {
        "id": f"{spec_id}/{model}/{remote_device.card}{shape_tag}/{run_id}",
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
) -> str:
    parts: list[str] = []
    if num_ctx is not None:
        parts.append(f"ctx{num_ctx}")
    if prompt_tokens is not None:
        parts.append(f"p{prompt_tokens}")
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
    engine_version: str = "0.35.1",
    quant: str = UNRECORDED,
    licence: str = UNRECORDED,
) -> dict[str, Any]:
    """Find the largest resident ``num_ctx`` under ``floor(vram_mib * abort_fraction)`` on Linux CUDA.

    No Windows shared-memory counter is read. A window passes when ``size_vram == size``,
    ``peak_mib < abort_mib``, and ``prompt_tok_s >= spill_prompt_ratio * prev_prompt_tok_s``.
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
        )

    candidates: list[int] = [grain]
    cursor = grain * 2
    while cursor <= trained_ctx:
        candidates.append(cursor)
        cursor *= 2
    if candidates[-1] != trained_ctx and trained_ctx > grain:
        candidates.append((trained_ctx // grain) * grain)

    best_ctx: int | None = None
    best_reading: Mapping[str, Any] | None = None
    prev_prompt_rate: float | None = None
    last_failure: tuple[str, str, int | None] | None = None

    for candidate in candidates:
        try:
            reading = measure_window(candidate)
        except LinuxShapeFailure as exc:
            last_failure = (exc.failure_class, exc.reason, exc.peak_mib)
            break

        size = int(reading["size"])
        size_vram = int(reading["size_vram"])
        peak_mib = int(reading["peak_mib"])
        prompt_rate = float(reading["prompt_tok_s"])

        if size_vram < size:
            last_failure = (
                "oom",
                f"partial layer offload at num_ctx {candidate} (size_vram {size_vram} < size {size})",
                peak_mib,
            )
            break
        if peak_mib >= abort_mib:
            last_failure = (
                "oom",
                f"peak {peak_mib} MiB reached abort_mib {abort_mib} MiB at num_ctx {candidate}",
                peak_mib,
            )
            break
        if prev_prompt_rate is not None and prompt_rate < spill_prompt_ratio * prev_prompt_rate:
            last_failure = (
                "oom",
                f"prompt_tok_s {prompt_rate:.1f} fell below {spill_prompt_ratio}x previous {prev_prompt_rate:.1f}",
                peak_mib,
            )
            break

        best_ctx = candidate
        best_reading = reading
        prev_prompt_rate = prompt_rate

    if best_ctx is None or best_reading is None:
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
    row: dict[str, Any] = {
        "id": f"{spec_id}/{model}/{remote_device.card}-fill/{run_id}",
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
        "artifacts": {
            "slots": [
                {"phase": "warmup", "throughput": decode_tok_s, "latency_ms": ttft_ms, "ttft_ms": ttft_ms},
                {"phase": "repeat", "throughput": decode_tok_s, "latency_ms": ttft_ms, "ttft_ms": ttft_ms},
            ]
        },
        "date": date,
        "repeats": 1,
        "provisional": False,
        "provenance": {"experiment_spec": spec_rel, "run_id": run_id},
        "notes": f"resident up to num_ctx {best_ctx} (peak {peak_mib} MiB < abort {abort_mib} MiB)",
    }
    stamp_family(row)
    validate_row(row)
    return row


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
    engine_version: str = "0.35.1",
    quant: str = UNRECORDED,
    licence: str = UNRECORDED,
) -> list[dict[str, Any]]:
    """Sweep generate TTFT and decode tok/s across prompt lengths 512..32k at concurrencies 1, 4, 8."""

    abort_mib = compute_abort_mib(remote_device.vram_mib, abort_fraction)
    rows: list[dict[str, Any]] = []
    for prompt_len in prompt_lengths:
        for concurrency in concurrencies:
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
                        prompt_tokens=prompt_len,
                        concurrency=concurrency,
                        peak_mib=peak_mib,
                        abort_mib=abort_mib,
                        engine_version=engine_version,
                        quant=quant,
                        licence=licence,
                    )
                )
                continue

            row: dict[str, Any] = {
                "id": f"{spec_id}/{model}/{remote_device.card}/p{prompt_len}-c{concurrency}/{run_id}",
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
                    "num_ctx": prompt_len,
                    "batch": None,
                    "static_shape": None,
                    "precision": None,
                    "prompt_tokens": prompt_len,
                    "concurrency": concurrency,
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
                "artifacts": {
                    "slots": [
                        {"phase": "warmup", "throughput": decode_tok_s, "latency_ms": latency_ms, "ttft_ms": ttft_ms},
                        {"phase": "repeat", "throughput": decode_tok_s, "latency_ms": latency_ms, "ttft_ms": ttft_ms},
                    ]
                },
                "date": date,
                "repeats": 3,
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
    measure_batch: Callable[[str, int], Mapping[str, Any]],
    tasks: tuple[str, ...] = THROUGHPUT_TASKS,
    batch_sizes: tuple[int, ...] = THROUGHPUT_BATCH_SIZES,
    abort_fraction: float = 0.90,
    engine_version: str = "0.35.1",
) -> list[dict[str, Any]]:
    """Sweep embed, rerank, and NLI throughput across batch sizes 1, 8, and 32 on Linux CUDA."""

    abort_mib = compute_abort_mib(remote_device.vram_mib, abort_fraction)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        model, digest = models_by_task[task]
        for batch in batch_sizes:
            try:
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
                        peak_mib=exc.peak_mib,
                        abort_mib=abort_mib,
                        engine_version=engine_version,
                    )
                )
                continue

            tput = float(measured["throughput"])
            p50 = float(measured["warm_p50_ms"])
            p95 = float(measured.get("warm_p95_ms", p50))
            peak_mib = int(measured["peak_mib"])
            size = int(measured["size"])
            size_vram = int(measured["size_vram"])
            row: dict[str, Any] = {
                "id": f"{spec_id}/{task}/{model}/{remote_device.card}/b{batch}/{run_id}",
                "task": task,
                "model": {
                    "repo_or_tag": model,
                    "revision_or_digest": digest,
                    "quant": UNRECORDED,
                    "licence": UNRECORDED,
                    "size": size,
                    "size_vram": size_vram,
                },
                "engine": {"name": "ollama", "version": engine_version},
                "device": remote_device.logical_device,
                "remote_device": remote_device.to_dict(),
                "gpu_resident": size_vram == size,
                "settings": {
                    "thinking_level": None,
                    "num_predict": None,
                    "num_ctx": None,
                    "batch": batch,
                    "static_shape": f"{batch}x512",
                    "precision": "fp16",
                    "peak_mib": peak_mib,
                    "abort_mib": abort_mib,
                },
                "quality": [],
                "latency": {
                    "cold_ms": None,
                    "warm_p50_ms": p50,
                    "warm_p95_ms": p95,
                    "throughput": tput,
                    "throughput_unit": "items/s",
                    "compile_ms": None,
                },
                "artifacts": {
                    "slots": [
                        {"phase": "warmup", "throughput": tput, "latency_ms": p50},
                        {"phase": "repeat", "throughput": tput, "latency_ms": p50},
                    ]
                },
                "date": date,
                "repeats": 3,
                "provisional": False,
                "provenance": {"experiment_spec": spec_rel, "run_id": run_id},
            }
            stamp_family(row)
            validate_row(row)
            rows.append(row)
    return rows
