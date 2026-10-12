"""Live fill-card run. The search in :mod:`interlock.harness.e5` decides.

One model at a time. The grant, the heartbeat and an empty card are checked
before that model's ``/api/show``. A failed show is a failed step. The load
uses ``num_predict`` 1. Confirmation counts a prefix of about 90 percent of
the candidate with the served model's ``prompt_eval_count``, then generates
it with shift and truncate false.

The sampler aborts at ``abort_mib``. It closes the in-flight request, then
unloads with ``keep_alive`` 0, then reads the card again until it falls
below ``abort_mib`` or about 10 seconds pass. A window whose card does not
fall is recorded with that note. Closing the request lets the unload take
effect and stops the decode. It does not take back memory the load has
already allocated.

Before a load, two measured windows extrapolate the next card reading. With
one measured window, the next context is bounded by that window's share of
the card above the idle baseline and the weight size from ``/api/tags``.
A share of 0 or less is not a bound, and a window that is not fully
resident (``size_vram`` equal to ``size``) is not a bound either. A window
predicted to reach ``abort_mib`` is ``predicted_over`` and is not loaded.
The first window still loads. If the weight size or the baseline cannot be
read, or the share is not positive, or that window is not fully resident,
the second window is not loaded and the row stops. An aborted window is
not a measurement. The next load waits until that abort's unload and card
re-read have finished. The unload call itself stops after about 60 seconds.
If it expires, the run stops with a note that the model may still be
resident. A token count stops after about twice a cold load of the largest
roster model, and that timeout refuses the model. The spec ceiling is not a probe.
Nothing here rests between models. A missing shared-memory reading or a
missing prompt rate is a spill, which the pure search already treats as a
step back. A confirmation prefix is counted again from a cold load, and a
disagreement refuses the model.

Tests pass fakes. The real nvidia-smi and counter readers run only when a
caller builds :func:`real_fill_seams` and then calls them.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from interlock.adapters.base import AdapterError, StepRequest
from interlock.ollama_supervisor import (
    ServerFingerprint,
    SupervisorStop,
    failure_kind,
    read_project_env,
    retry_budget,
)
from interlock.adapters.ollama import (
    COUNT_TIMEOUT_NOTE,
    UNLOAD_TIMEOUT_NOTE,
    context_length,
    quantization_level,
    url_for,
)
from interlock.harness.count_budget import roster_timeout
from interlock.cli import announce_run
from interlock.families import stamp_family
from interlock.guards import GuardError, require_card_empty, require_grant, require_heartbeat
from interlock.harness.e5 import (
    SearchError,
    Observation,
    fit_prompt_words,
    search,
)
from interlock.harness.receipts import write_receipt
from interlock.receipt import validate_row

UNRECORDED = "unrecorded"
INCONCLUSIVE = "the step is inconclusive"
ABORT_ERROR = "card memory reached abort_mib"
PREDICTED_OVER = "predicted_over"
UNLOAD_SETTLE_S = 10.0
_SETTLE_STEP_S = 0.2
MEMORY_ARGV = ("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits")
DRIVER_ARGV = ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits")
NVIDIA_SMI_ARGV = ("nvidia-smi",)
ADAPTER_SHARED = r"\GPU Adapter Memory(*)\Shared Usage"
ADAPTER_DEDICATED = r"\GPU Adapter Memory(*)\Dedicated Usage"
LOG_ENV = "INTERLOCK_OLLAMA_LOG"
EXE_ENV = "INTERLOCK_OLLAMA_EXE"
_MIB = 1024 * 1024
_ENV_IN_LOG = re.compile(r"\b(OLLAMA_[A-Z0-9_]+)[=:]([^\s,\]]+)")
_BACKEND_LINE = re.compile(r"load_backend: loaded \S+ backend from (\S+)")
_CUDA_HEADER = re.compile(r"CUDA Version:\s*([0-9.]+)")


class _WindowOver(Exception):
    """The sampler reached abort_mib while this window was in flight."""


class _CountRefused(Exception):
    """The cold re-count of a confirmation prefix disagreed with the search."""


class _NoBound(Exception):
    """The next window has no share that can be trusted, so it is not loaded."""


class _UnloadTimeout(Exception):
    """The unload did not return. The model may still be resident."""


class _Repairable(Exception):
    """A transport, timeout, or 5xx. The supervisor hears about it between steps."""

    def __init__(self, kind: str, error: str) -> None:
        super().__init__(error)
        self.kind = kind
        self.error = error


@dataclass
class FillSeams:
    """Readings the executor does not know how to take.

    A library run passes these. The command builds :func:`real_fill_seams`
    only when the caller did not pass adapters. A fake adapter therefore
    cannot reach nvidia-smi by forgetting the seams.
    """

    card_mib: Callable[[], int | None]
    shared_mib: Callable[[], int | None]
    dedicated_mib: Callable[[], int | None]
    ollama_version: Callable[[], str | None]
    driver: Callable[[], str | None]
    cuda_backend: Callable[[], str | None]
    sleep: Callable[[float], None]
    clock: Callable[[], float]
    threaded: bool = True

    def __post_init__(self) -> None:
        self.sample: Callable[[], int | None] = lambda: None
        self.over = False


def quiet_fill_seams() -> FillSeams:
    """Readings that stay absent. A missing shared reading is a spill, not a pass."""

    return FillSeams(
        card_mib=lambda: None,
        shared_mib=lambda: None,
        dedicated_mib=lambda: None,
        ollama_version=lambda: None,
        driver=lambda: None,
        cuda_backend=lambda: None,
        sleep=lambda _seconds: None,
        clock=time.monotonic,
        threaded=False,
    )


def real_fill_seams(adapter, *, run=None, read_counter=None) -> FillSeams:
    """The command's seams. ``run`` and ``read_counter`` are test fakes.

    Shared and dedicated usage come from the 5090's adapter instance. The
    runner pid is recorded when one child of the listener can be named. It
    is not a gate. The server log and the ggml library come from the
    environment, not from ``/api/show``.
    """

    run = run or subprocess.run
    read_counter = read_counter or (lambda counter: _powershell_counter(counter, run))

    def read_adapter():
        nvidia = parse_mib(_capture(run, MEMORY_ARGV))
        if nvidia is None:
            return None
        dedicated = parse_counter_text(read_counter(ADAPTER_DEDICATED) or "")
        shared = parse_counter_text(read_counter(ADAPTER_SHARED) or "")
        return adapter_usage(dedicated, shared, nvidia)

    def shared_mib() -> int | None:
        found = read_adapter()
        return None if found is None else found[1]

    def dedicated_mib() -> int | None:
        found = read_adapter()
        return None if found is None else found[2]

    def driver() -> str | None:
        return first_line(_capture(run, DRIVER_ARGV))

    def cuda() -> str | None:
        return driver_cuda(_capture(run, NVIDIA_SMI_ARGV))

    seams = FillSeams(
        card_mib=lambda: parse_mib(_capture(run, MEMORY_ARGV)),
        shared_mib=shared_mib,
        dedicated_mib=dedicated_mib,
        ollama_version=lambda: _ollama_version(adapter),
        driver=driver,
        cuda_backend=cuda,
        sleep=time.sleep,
        clock=time.monotonic,
        threaded=True,
    )
    seams.adapter_reading = read_adapter
    seams.server_log = lambda: _read_configured_log()
    seams.ggml_dll = lambda: _configured_dll()
    seams.runner_pid = lambda: _runner_pid(run)
    return seams


class CardSampler:
    """Peak card MiB while a window is in flight.

    ``threaded`` false means the transport calls :meth:`sample` itself, so a
    test can abort on one deterministic reading. The production sampler waits
    ``sample_ms`` between readings. Reaching ``abort_mib`` runs ``on_abort``,
    which closes the request before the unload. ``on_abort`` sets an event
    when it finishes, and the probe waits for that event with no timeout
    before it returns. The reading is not compared with the spec ceiling.
    """

    def __init__(self, seams: FillSeams, abort_mib: int, sample_ms: int, on_abort) -> None:
        self.seams = seams
        self.abort_mib = abort_mib
        self.sample_ms = sample_ms
        self.on_abort = on_abort
        self.peak: int | None = None
        self.over = False
        self._stop = threading.Event()
        self._abort_done = threading.Event()
        self._abort_done.set()
        self._lock = threading.Lock()
        self._firing = False
        seams.sample = self.sample

    def reset(self) -> None:
        with self._lock:
            self.peak = None
            self.over = False
            self.seams.over = False

    def sample(self) -> int | None:
        try:
            value = self.seams.card_mib()
        except Exception:
            return self.peak
        if isinstance(value, bool) or not isinstance(value, int):
            return self.peak
        fire = False
        with self._lock:
            self.peak = value if self.peak is None else max(self.peak, value)
            if value >= self.abort_mib and not self.over and not self._firing:
                self.over = True
                self.seams.over = True
                self._firing = True
                fire = True
        if fire:
            self._abort_done.clear()
            try:
                self.on_abort()
            except Exception:
                pass
            finally:
                self._firing = False
                self._abort_done.set()
        return value

    def wait_for_abort(self) -> None:
        """Block until an in-flight abort has finished unloading and re-reading.

        There is no timeout on this wait. The unload call stops after about
        60 seconds, and the settle bound is about 10 seconds on top of that.
        A shorter wait would return while the card is still settling.
        """

        self._abort_done.wait()

    def cover(self, call):
        if not self.seams.threaded:
            return call()
        self._stop.clear()
        thread = threading.Thread(target=self._loop, name="fill-card-vram", daemon=True)
        thread.start()
        try:
            return call()
        finally:
            self._stop.set()
            self._abort_done.wait()
            thread.join()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sample()
            if self.over or self._stop.is_set():
                return
            self.seams.sleep(self.sample_ms / 1000)


def parse_mib(text: str | None) -> int | None:
    """One nvidia-smi memory line, in MiB. Two lines is not a total."""

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


def first_line(text: str | None) -> str | None:
    if not isinstance(text, str):
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        return None
    return lines[0]


def parse_counter_text(text: str) -> list[tuple[str, float]]:
    rows = []
    for line in text.splitlines():
        if "=" in line:
            name, raw = line.split("=", 1)
        elif ":" in line:
            name, raw = line.rsplit(":", 1)
        else:
            continue
        try:
            rows.append((name.strip(), float(raw.strip().replace(",", ""))))
        except ValueError:
            continue
    return rows


def _cooked_mib(cooked: object) -> int | None:
    if isinstance(cooked, bool) or not isinstance(cooked, (int, float)) or cooked < 0:
        return None
    return int(cooked / _MIB)


def match_adapter_instance(dedicated: list[tuple[str, float]], nvidia_mib: int) -> str | None:
    """The one adapter whose dedicated MiB equals ``nvidia_mib``.

    The iGPU is an adapter too. Zero matches or more than one is missing.
    Order is not a match.
    """

    if isinstance(nvidia_mib, bool) or not isinstance(nvidia_mib, int):
        return None
    matches = []
    for name, cooked in dedicated:
        if not isinstance(name, str):
            continue
        if _cooked_mib(cooked) == nvidia_mib:
            matches.append(name)
    if len(matches) != 1:
        return None
    return matches[0]


def _instance_mib(samples: list[tuple[str, float]], instance: str) -> int | None:
    found = [_cooked_mib(cooked) for name, cooked in samples if name == instance and _cooked_mib(cooked) is not None]
    if len(found) != 1:
        return None
    return found[0]


def adapter_usage(
    dedicated: list[tuple[str, float]],
    shared: list[tuple[str, float]],
    nvidia_mib: int,
) -> tuple[str, int, int] | None:
    """``(instance, shared MiB, dedicated MiB)`` for the one matching adapter."""

    instance = match_adapter_instance(dedicated, nvidia_mib)
    if instance is None:
        return None
    shared_mib = _instance_mib(shared, instance)
    dedicated_mib = _instance_mib(dedicated, instance)
    if shared_mib is None or dedicated_mib is None:
        return None
    return instance, shared_mib, dedicated_mib


def window_delta(before, after) -> tuple[int | None, int | None, int | None]:
    """Shared after the window, the pre-window shared baseline, and the dedicated delta.

    A missing match, or two matches that name different instances, is a missing
    shared reading. The dedicated delta is recorded and is not a gate, so a
    missing dedicated value does not by itself blank the shared reading.
    """

    if before is None or after is None or before[0] != after[0]:
        return None, None, None
    dedicated_delta = None
    if before[2] is not None and after[2] is not None:
        dedicated_delta = after[2] - before[2]
    return after[1], before[1], dedicated_delta


def driver_cuda(text: str | None) -> str | None:
    """CUDA from the nvidia-smi header, labelled as the driver's.

    ``nvidia-smi --query-gpu=cuda_version`` is not a field on driver 617.14.
    """

    if not isinstance(text, str):
        return None
    match = _CUDA_HEADER.search(text)
    if match is None:
        return None
    return f"driver {match.group(1)}"


def runner_pid(lines: list[str]) -> int | None:
    """The one ``ollama.exe`` child of the listener. Several, or none, is unrecorded.

    Each line is ``pid parent name``. The caller already limited the lines to
    children of the process that listens on 11434. This value never gates.
    """

    found: list[int] = []
    for line in lines:
        if not isinstance(line, str):
            continue
        parts = line.split()
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        if parts[2].lower() != "ollama.exe":
            continue
        pid = int(parts[0])
        if pid not in found:
            found.append(pid)
    if len(found) != 1:
        return None
    return found[0]


def facts_from_log(text: str | None) -> dict[str, str | None]:
    """Flash attention, KV cache type, default context, and the backend library.

    The four values come from the latest server-config line and the latest
    ``load_backend`` line. A missing log, or a log with neither line, leaves
    them unrecorded and names the reason. ``/api/show`` is not a source.
    """

    blank = {
        "flash_attention": UNRECORDED,
        "kv_cache_type": UNRECORDED,
        "default_context": UNRECORDED,
        "backend_library": UNRECORDED,
        "reason": "server log is missing",
    }
    if not isinstance(text, str) or not text.strip():
        return blank
    config_lines = []
    libraries = []
    for line in text.splitlines():
        if "OLLAMA_" in line and ("server config" in line.lower() or "env=" in line.lower()):
            config_lines.append(line)
        match = _BACKEND_LINE.search(line)
        if match is not None:
            libraries.append(match.group(1))
    if not config_lines and not libraries:
        return {**blank, "reason": "server log has no config line"}
    env = {key: value for key, value in _ENV_IN_LOG.findall(config_lines[-1])} if config_lines else {}

    def take(key: str) -> str:
        value = env.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        return UNRECORDED

    facts = {
        "flash_attention": take("OLLAMA_FLASH_ATTENTION"),
        "kv_cache_type": take("OLLAMA_KV_CACHE_TYPE"),
        "default_context": take("OLLAMA_CONTEXT_LENGTH"),
        "backend_library": libraries[-1] if libraries else UNRECORDED,
    }
    missing = [name for name, value in facts.items() if value == UNRECORDED]
    facts["reason"] = None if not missing else "server log did not record " + ", ".join(missing)
    return facts


def ggml_cuda_sha256(path: str | None) -> tuple[str, str | None]:
    """Sha256 of ``ggml-cuda.dll``. A missing file is unrecorded, with a reason."""

    if not isinstance(path, str) or not path.strip():
        return UNRECORDED, "ggml-cuda.dll path is not configured"
    file = Path(path)
    try:
        digest = hashlib.sha256()
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 16), b""):
                digest.update(chunk)
    except OSError:
        return UNRECORDED, "ggml-cuda.dll is missing"
    return digest.hexdigest(), None


def _capture(run, argv) -> str | None:
    try:
        completed = run(argv, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if getattr(completed, "returncode", 1) != 0:
        return None
    stdout = getattr(completed, "stdout", None)
    return stdout if isinstance(stdout, str) else None


def _powershell_counter(counter: str, run) -> str | None:
    script = (
        f"(Get-Counter '{counter}').CounterSamples | "
        "ForEach-Object { $_.InstanceName + '=' + $_.CookedValue }"
    )
    return _capture(run, ["powershell", "-NoProfile", "-Command", script])


def _read_configured_log() -> str | None:
    path = read_project_env(LOG_ENV)
    if not isinstance(path, str) or not path.strip():
        return None
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _configured_dll() -> str | None:
    """``<exe>\\lib\\ollama\\cuda_v13\\ggml-cuda.dll``. The exe comes from config."""

    exe = read_project_env(EXE_ENV)
    if not isinstance(exe, str) or not exe.strip():
        return None
    return str(Path(exe).parent / "lib" / "ollama" / "cuda_v13" / "ggml-cuda.dll")


def _runner_pid(run) -> int | None:
    """The ollama.exe whose parent is the 11434 listener. A miss is unrecorded."""

    script = (
        "$listener = Get-NetTCPConnection -LocalPort 11434 -ErrorAction SilentlyContinue | "
        "Where-Object { $_.RemotePort -eq 0 } | "
        "Select-Object -First 1 -ExpandProperty OwningProcess; "
        "if (-not $listener) { return }; "
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.ParentProcessId -eq $listener -and $_.Name -eq 'ollama.exe' } | "
        "ForEach-Object { $_.ProcessId.ToString() + ' ' + $_.ParentProcessId.ToString() + ' ollama.exe' }"
    )
    text = _capture(run, ["powershell", "-NoProfile", "-Command", script])
    if text is None:
        return None
    return runner_pid(text.splitlines())


def _optional_reader(seams, name: str):
    reader = getattr(seams, name, None)
    if not callable(reader):
        return None
    try:
        return reader()
    except Exception:
        return None


def _window_readings(seams):
    reader = getattr(seams, "adapter_reading", None)
    if callable(reader):
        try:
            return reader()
        except Exception:
            return None
    shared = _optional_int(seams.shared_mib())
    dedicated = _optional_int(seams.dedicated_mib())
    if shared is None:
        return None
    return ("seam", shared, dedicated)


def _ollama_version(adapter) -> str | None:
    try:
        payload = adapter.transport("GET", f"{url_for('5090')}/api/version", None)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    version = payload.get("version")
    if not isinstance(version, str) or not version.strip():
        return None
    return version.strip()


def _refuse_confirm_num_predict(value: object) -> int:
    """The run-time line. Spec validation already refused this before contact."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"confirm_num_predict must be a whole number of at least 1, not {value!r}")
    return value


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _text(value: object) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return UNRECORDED


def _count_leaves_loaded(keep_alive: object) -> bool:
    if keep_alive is None:
        return True
    if keep_alive == 0:
        return False
    return not (isinstance(keep_alive, str) and keep_alive.strip().lower() in {"0", "0s", "0m", "0h"})


def _prompt_method(fit: str) -> str:
    if fit == "exact":
        return "prompt_eval_count"
    return "prompt_eval_count (shortest prefix at or above)"


def _ollama(adapters: dict):
    try:
        adapter = adapters["ollama"]
    except (KeyError, TypeError) as exc:
        raise GuardError("REFUSED: fill-card needs an ollama adapter") from exc
    return adapter


def _blank_latency(throughput: float | None) -> dict:
    return {
        "cold_ms": None,
        "warm_p50_ms": None,
        "warm_p95_ms": None,
        "throughput": throughput,
        "throughput_unit": "tok/s" if throughput is not None else None,
        "compile_ms": None,
    }


def _residency(models: object, name: str) -> tuple[bool, int | None, int | None]:
    if not isinstance(models, list):
        return False, None, None
    for item in models:
        if not isinstance(item, dict):
            continue
        label = item.get("name") or item.get("model")
        if not isinstance(label, str) or (label != name and not label.startswith(name + ":")):
            continue
        size = item.get("size")
        vram = item.get("size_vram")
        if isinstance(size, bool) or isinstance(vram, bool):
            return False, None, None
        if not isinstance(size, int) or not isinstance(vram, int):
            return False, None, None
        return size == vram, size, vram
    return False, None, None


def _over(peak: int | None) -> Observation:
    return Observation(False, peak, error=ABORT_ERROR)


def _measured_points(measured: list[tuple[int, int]]) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for seen, mib in measured:
        if points and points[-1][0] == seen:
            points[-1] = (seen, mib)
        else:
            points.append((seen, mib))
    return points


def _one_window_mib(reading: int, ctx_1: int, baseline: int, weights_mib: int, ctx: int) -> int | None:
    """Bound ``ctx`` from one measured window.

    The share is the reading above the idle baseline and the weight file.
    A share of 0 or less is not a bound: the inputs disagree, and flooring
    it would predict no growth. A positive share includes compute buffers,
    so it overstates the cache. The bound is
    ``reading + ceil(share * (ctx - ctx_1) / ctx_1)``.
    """

    if ctx_1 <= 0:
        return None
    share = reading - baseline - weights_mib
    if share <= 0:
        return None
    extra = share * (ctx - ctx_1)
    return reading + (extra + ctx_1 - 1) // ctx_1


def _weights_mib(adapter, model: str) -> int | None:
    """``/api/tags`` size, rounded up to MiB. A missing tag is None."""

    reader = getattr(adapter, "tag_size", None)
    if not callable(reader):
        return None
    size = reader(model, "5090")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        return None
    return (size + _MIB - 1) // _MIB


def _settled_card(seams: FillSeams, abort_mib: int) -> int | None:
    """Re-read until the card falls below ``abort_mib`` or about 10 s pass."""

    waited = 0.0
    reading: int | None = None
    while True:
        try:
            reading = _optional_int(seams.card_mib())
        except Exception:
            reading = None
        if reading is not None and reading < abort_mib:
            return reading
        if waited >= UNLOAD_SETTLE_S:
            return reading
        seams.sleep(_SETTLE_STEP_S)
        waited += _SETTLE_STEP_S


def _predict_mib(measured: list[tuple[int, int]], ctx: int) -> int | None:
    """Card MiB for ``ctx`` from the last two measured windows of this model.

    One window is not a line. The caller uses :func:`_one_window_mib` until a
    second window has been measured. Two windows give MiB per context token.
    The next step rounds up when the slope climbs, so a fraction of a MiB
    still counts as reaching ``abort_mib``.
    """

    points = _measured_points(measured)
    if len(points) < 2:
        return None
    (left_ctx, left_mib), (right_ctx, right_mib) = points[-2], points[-1]
    span = right_ctx - left_ctx
    if span == 0:
        return None
    rise = right_mib - left_mib
    steps = ctx - right_ctx
    if steps == 0:
        return right_mib
    if rise > 0 and steps > 0:
        return right_mib + (rise * steps + span - 1) // span
    return right_mib + (rise * steps) // span


def _stuck_note(ctx: int | None, reading: int | None) -> str:
    window = "the window" if ctx is None else str(ctx)
    if reading is None:
        return f"{window} had no card reading after the unload"
    return f"{window} stayed at {reading} MiB after the unload"


def _sync_restarts(supervisor, marker) -> None:
    """Every committed restart lands on the run marker, including one that stops the run."""

    if supervisor is None or marker is None:
        return
    reason = getattr(supervisor, "restarts_disabled", None)
    if reason:
        marker.note_restart_disabled(reason)
    have = marker.restart_records()
    for record in supervisor.restarts:
        data = record.as_dict()
        if data not in have:
            marker.note_restart(data)
            have.append(data)


def _strip_measurement(row: dict) -> None:
    """A step that failed across a restart keeps no window, peak, or rate."""

    settings = row.setdefault("settings", {})
    for key in (
        "limit",
        "search_calls",
        "prompt_tokens",
        "prompt_token_method",
        "prompt_token_search",
        "prompt_token_cold",
        "dedicated_mib",
        "aborted_windows",
        "after_unload_mib",
        "predicted_over",
        "weights_mib",
        "one_window_bound",
    ):
        settings.pop(key, None)
    for key in ("max_resident_ctx", "peak_mib", "ttft_ms", "decode_tok_s", "prompt_tok_s", "shared_mib"):
        settings[key] = None
    row.pop("gpu_resident", None)
    row["latency"] = _blank_latency(None)
    row["aborted"] = True


def _stamp_restarts(rows: list[dict], marker) -> None:
    records = marker.restart_records() if marker is not None else []
    if not records:
        return
    for row in rows:
        row.setdefault("settings", {})["ollama_restarts"] = records
        validate_row(row)


def _resume_fill(supervisor, marker, entry, repair, spec, retries, row) -> str | None:
    """Between models. None keeps the old abort path. A restart does not keep this row's window."""

    if supervisor is None or not repair:
        return None
    kind, error = repair[-1]
    model = entry["model"]
    try:
        decision = supervisor.after_failure(url_for("5090"), model=model, kind=kind)
    except SupervisorStop as exc:
        _sync_restarts(supervisor, marker)
        marker.note_failure(model, error)
        row["notes"] = _join(str(exc), row.get("notes") or "")
        _strip_measurement(row)
        return "stop"
    _sync_restarts(supervisor, marker)
    marker.note_failure(model, error)
    if decision.action != "restarted" or decision.record is None:
        row["notes"] = _join(decision.note or error, row.get("notes") or "")
        _strip_measurement(row)
        return "stop"
    used = retries.get(model, 0)
    if used < retry_budget(spec):
        retries[model] = used + 1
        return "retry"
    if INCONCLUSIVE not in (row.get("notes") or ""):
        row["notes"] = _join(INCONCLUSIVE, row.get("notes") or "")
    _strip_measurement(row)
    return "next"


def run_fill_card(
    spec: dict,
    *,
    adapters: dict,
    seams: FillSeams,
    grant: str | None,
    root: Path,
    spec_rel: str,
    run_id: str,
    date: str,
    heartbeat: Path,
    wall,
    marker,
    notes: str = "",
    supervisor=None,
) -> list[dict]:
    """Measure each roster model. Unload at the end of the model and on any exit."""

    _refuse_confirm_num_predict(spec.get("confirm_num_predict", 1))
    rows: list[dict] = []
    stop_note = None
    retries: dict[str, int] = {}
    pending_retry: set[str] = set()
    index = 0
    models = list(spec["models"])
    record_checked = False
    try:
        while index < len(models):
            entry = models[index]
            if supervisor is not None and not record_checked:
                supervisor.check_start_record(url_for("5090"))
                reason = supervisor.restarts_disabled
                if reason:
                    marker.note_restart_disabled(reason)
                record_checked = True
            require_grant(grant, uses_5090=True)
            require_heartbeat(heartbeat, wall())
            adapter = _ollama(adapters)
            require_card_empty(adapter.loaded_models())
            marker.entered = True
            repair: list[tuple[str, str]] = []
            if supervisor is not None:
                supervisor.begin_measurement()
            try:
                row = _measure_model(
                    spec,
                    entry,
                    adapter,
                    seams,
                    grant=grant,
                    spec_rel=spec_rel,
                    run_id=run_id,
                    date=date,
                    notes=notes,
                    supervisor=supervisor,
                    repair=repair,
                )
            finally:
                extra = _unload(adapter, entry["model"])
                if supervisor is not None:
                    supervisor.end_measurement()
            if extra and extra not in (row.get("notes") or ""):
                row["notes"] = _join(row.get("notes") or "", extra)
                row["aborted"] = True
            if extra and supervisor is not None:
                unload_kind = failure_kind(extra)
                if unload_kind:
                    repair.append((unload_kind, extra))
            resumed = _resume_fill(supervisor, marker, entry, repair, spec, retries, row)
            if resumed == "stop":
                rows.append(row)
                stop_note = row.get("notes") or "fill-card step failed"
                break
            if resumed == "retry":
                pending_retry.add(entry["model"])
                continue
            if resumed == "next":
                rows.append(row)
                index += 1
                continue
            if resumed is None and not row.get("aborted"):
                if supervisor is not None:
                    try:
                        supervisor.confirm_backend()
                    except SupervisorStop as exc:
                        _sync_restarts(supervisor, marker)
                        marker.note_failure(entry["model"], str(exc))
                        row["notes"] = _join(str(exc), row.get("notes") or "")
                        _strip_measurement(row)
                        rows.append(row)
                        stop_note = str(exc)
                        break
                if entry["model"] in pending_retry:
                    row["settings"]["retried"] = True
                    pending_retry.discard(entry["model"])
                if supervisor is not None and supervisor.consume_cold_after_restart():
                    row["settings"]["cold_after_restart"] = True
            rows.append(row)
            if row.get("aborted"):
                stop_note = row.get("notes") or "fill-card step failed"
                break
            index += 1
    finally:
        try:
            if rows:
                _stamp_restarts(rows, marker)
                write_receipt(root, run_id, rows)
        finally:
            if marker.entered:
                marker.close(_iso(wall()))
    if stop_note:
        from interlock.harness.runner import RunnerError

        raise RunnerError(stop_note)
    announce_run(rows, finished=True)
    return rows


def _unload(adapter, model: str, *, quiet: bool = True) -> str | None:
    """Unload one model.

    The end-of-model unload stays quiet, as it always has, so a missing
    server does not hide the row. A timeout is not quiet: the note says the
    model may still be resident, and the caller stops the run. Passing
    ``quiet`` false re-raises any other failure.
    """

    try:
        adapter.unload(model, "5090")
    except AdapterError as exc:
        if str(exc) == UNLOAD_TIMEOUT_NOTE:
            return UNLOAD_TIMEOUT_NOTE
        if quiet:
            return None
        raise
    except Exception:
        if quiet:
            return None
        raise
    return None


def _iso(epoch: float) -> str:
    import datetime

    return datetime.datetime.fromtimestamp(epoch, tz=datetime.timezone.utc).isoformat(timespec="seconds")


def _measure_model(
    spec: dict,
    entry: dict,
    adapter,
    seams: FillSeams,
    *,
    grant: str | None,
    spec_rel: str,
    run_id: str,
    date: str,
    notes: str,
    supervisor=None,
    repair: list | None = None,
) -> dict:
    model = entry["model"]
    version = _text(seams.ollama_version())
    driver = _text(seams.driver())
    cuda = _text(seams.cuda_backend())
    facts = facts_from_log(_optional_reader(seams, "server_log"))
    ggml, ggml_reason = ggml_cuda_sha256(_optional_reader(seams, "ggml_dll"))
    pid_value = _optional_reader(seams, "runner_pid")
    runner = pid_value if isinstance(pid_value, int) and not isinstance(pid_value, bool) else None

    def row(**kwargs):
        kwargs.setdefault("flash", str(facts["flash_attention"]))
        kwargs.setdefault("kv", str(facts["kv_cache_type"]))
        kwargs.setdefault("ggml", ggml)
        kwargs.setdefault("default_context", str(facts["default_context"]))
        kwargs.setdefault("backend_library", str(facts["backend_library"]))
        kwargs.setdefault("server_reason", facts["reason"])
        kwargs.setdefault("ggml_reason", ggml_reason)
        kwargs.setdefault("runner_pid", runner)
        return _row(spec, entry, **kwargs)

    baseline = _optional_int(seams.card_mib())

    def _kind(error: str) -> str | None:
        if supervisor is None:
            return None
        return failure_kind(error)

    def _remember(flash_value: str, kv_value: str) -> None:
        """The first good reading. A preset fingerprint is left alone."""

        if supervisor is None or supervisor.recorded is not None:
            return
        if not version or version == UNRECORDED:
            return
        flash = flash_value or UNRECORDED
        kv = kv_value or UNRECORDED
        sha = ""
        line = ""
        ensure = getattr(supervisor, "_ensure_start", None)
        start = ensure() if callable(ensure) else None
        if start is not None:
            if not flash or flash == UNRECORDED:
                flash = start.env.get("OLLAMA_FLASH_ATTENTION") or flash
            if not kv or kv == UNRECORDED:
                kv = start.env.get("OLLAMA_KV_CACHE_TYPE") or kv
        try:
            live = supervisor.seams.environment(url_for("5090"))
        except Exception:
            live = None
        if isinstance(live, ServerFingerprint):
            if live.ggml_cuda_sha256 and live.ggml_cuda_sha256 != UNRECORDED:
                sha = live.ggml_cuda_sha256
            if live.backend_line and live.backend_line != UNRECORDED:
                line = live.backend_line
        supervisor.remember(
            ServerFingerprint(
                version=version,
                flash_attention=flash,
                kv_cache_type=kv,
                ggml_cuda_sha256=sha,
                backend_line=line,
                baseline_mib=baseline,
                model=model,
            )
        )

    def _inconclusive(error: str, **extra) -> dict:
        return row(
            grant=grant,
            spec_rel=spec_rel,
            run_id=run_id,
            date=date,
            notes=_join(_join(INCONCLUSIVE, error), notes),
            aborted=True,
            version=version,
            driver=driver,
            cuda=cuda,
            baseline=baseline,
            **extra,
        )
    if baseline is not None and baseline >= spec["abort_mib"]:
        return row(
            grant=grant,
            spec_rel=spec_rel,
            run_id=run_id,
            date=date,
            notes=_join(f"idle baseline reached abort_mib ({baseline} MiB)", notes),
            aborted=True,
            version=version,
            driver=driver,
            cuda=cuda,
            baseline=baseline,
        )
    try:
        shown = adapter.show(model, "5090")
        default_ctx = context_length(shown)
        quant = quantization_level(shown)
        flash = str(facts["flash_attention"])
        kv = str(facts["kv_cache_type"])
    except AdapterError as exc:
        error = str(exc)
        kind = _kind(error)
        _remember(UNRECORDED, UNRECORDED)
        if kind and repair is not None:
            repair.append((kind, error))
            return _inconclusive(error)
        return row(
            grant=grant,
            spec_rel=spec_rel,
            run_id=run_id,
            date=date,
            notes=_join(error, notes),
            aborted=True,
            version=version,
            driver=driver,
            cuda=cuda,
            baseline=baseline,
        )
    _remember(flash, kv)

    num_predict = _refuse_confirm_num_predict(spec.get("confirm_num_predict", 1))
    fraction = float(spec["confirm_prompt_fraction"])
    sample_ms = spec.get("sample_ms", 200)
    if isinstance(sample_ms, bool) or not isinstance(sample_ms, int) or sample_ms < 1:
        sample_ms = 200
    seed = (spec.get("dataset") or {}).get("seed")
    if not isinstance(seed, str) or not seed.strip():
        return row(
            grant=grant,
            spec_rel=spec_rel,
            run_id=run_id,
            date=date,
            notes=_join("a prompt seed is required", notes),
            aborted=True,
            version=version,
            driver=driver,
            cuda=cuda,
            baseline=baseline,
            quant=quant,
            default_ctx=default_ctx,
            flash=flash,
            kv=kv,
            ggml=ggml,
        )

    try:
        weights_mib = _weights_mib(adapter, model)
    except (AdapterError, OSError, TypeError, ValueError, RuntimeError):
        weights_mib = None
    aborted: list[int] = []
    predicted_over: list[int] = []
    one_window_bound: list[tuple[int, int]] = []
    measured: list[tuple[int, int]] = []
    resident_at: dict[int, bool] = {}
    after_unload: list[tuple[int | None, int | None]] = []
    prompts: dict[int, tuple[int, str, int, int]] = {}
    sizes: dict[int, tuple[int, int, bool]] = {}
    dedicated_by_ctx: dict[int, int] = {}
    shift = spec.get("shift") is False
    truncate = spec.get("truncate") is False
    current: dict[str, int | None] = {"ctx": None}
    hung: dict[str, str | None] = {"note": None}

    def on_abort() -> None:
        try:
            cancel = getattr(adapter, "cancel", None)
            if callable(cancel):
                cancel()
        finally:
            try:
                note = _unload(adapter, model)
                if note:
                    hung["note"] = note
            finally:
                after_unload.append((current["ctx"], _settled_card(seams, spec["abort_mib"])))

    def stop_if_unload_timed_out() -> None:
        note = _unload(adapter, model, quiet=False)
        if note:
            hung["note"] = note
            raise _UnloadTimeout(note)

    sampler = CardSampler(seams, spec["abort_mib"], sample_ms, on_abort)

    count_timeout = roster_timeout(adapter, spec)

    def count(text: str, ctx: int) -> int:
        settings = {"num_ctx": ctx, "temperature": 0}
        if shift:
            settings["shift"] = False
        if truncate:
            settings["truncate"] = False
        try:
            counted = adapter.count_tokens(
                model, text, settings=settings, device="5090", timeout=count_timeout
            )
        except AdapterError as exc:
            if str(exc) == COUNT_TIMEOUT_NOTE:
                raise _CountRefused(COUNT_TIMEOUT_NOTE) from exc
            if sampler.over:
                raise _WindowOver()
            kind = _kind(str(exc))
            if kind:
                raise _Repairable(kind, str(exc))
            raise
        if sampler.over:
            raise _WindowOver()
        return counted

    def observe(ctx: int, kind: str) -> Observation:
        current["ctx"] = ctx
        prompt = seed
        if kind == "confirm":
            target = max(1, int(ctx * fraction))
            try:
                text, tokens, fit = fit_prompt_words(seed, target, lambda piece: count(piece, ctx))
            except _WindowOver:
                aborted.append(ctx)
                return _over(sampler.peak)
            except (SearchError, AdapterError) as exc:
                if sampler.over:
                    aborted.append(ctx)
                    return _over(sampler.peak)
                kind = _kind(str(exc))
                if kind:
                    raise _Repairable(kind, str(exc))
                return Observation(False, sampler.peak, error=str(exc))
            if _count_leaves_loaded(entry.get("keep_alive")):
                stop_if_unload_timed_out()
            try:
                cold_tokens = count(text, ctx)
            except _WindowOver:
                aborted.append(ctx)
                return _over(sampler.peak)
            except AdapterError as exc:
                if sampler.over:
                    aborted.append(ctx)
                    return _over(sampler.peak)
                kind = _kind(str(exc))
                if kind:
                    raise _Repairable(kind, str(exc))
                return Observation(False, sampler.peak, error=str(exc))
            if cold_tokens != tokens:
                if _count_leaves_loaded(entry.get("keep_alive")):
                    stop_if_unload_timed_out()
                raise _CountRefused(
                    f"the search counted {tokens} and a cold load counted {cold_tokens}"
                )
            prompts[ctx] = (cold_tokens, _prompt_method(fit), tokens, cold_tokens)
            if _count_leaves_loaded(entry.get("keep_alive")):
                stop_if_unload_timed_out()
            prompt = text
        settings = {
            "prompt": prompt,
            "num_predict": num_predict,
            "num_ctx": ctx,
            "temperature": 0,
        }
        if shift:
            settings["shift"] = False
        if truncate:
            settings["truncate"] = False
        request = StepRequest(f"{kind}-{ctx}", "generate", model, "5090", "ollama", settings)
        before = _window_readings(seams)
        try:
            result = adapter.run(request)
        except AdapterError as exc:
            if sampler.over:
                aborted.append(ctx)
                return _over(sampler.peak)
            kind = _kind(str(exc))
            if kind:
                raise _Repairable(kind, str(exc))
            return Observation(False, sampler.peak, error=str(exc))
        if sampler.over:
            aborted.append(ctx)
            return _over(sampler.peak)
        if not result.ok:
            error = result.error or "generate failed"
            kind = _kind(error)
            if kind:
                raise _Repairable(kind, error)
            return Observation(False, sampler.peak, error=error)
        resident, size, vram = _residency(adapter.loaded_models(), model)
        if size is not None and vram is not None:
            sizes[ctx] = (size, vram, resident)
        after = _window_readings(seams)
        shared_mib, baseline, dedicated_delta = window_delta(before, after)
        if after is not None and after[2] is not None:
            dedicated_by_ctx[ctx] = after[2]
        duration = result.prompt_eval_duration_ns
        ttft = None
        if isinstance(duration, int) and not isinstance(duration, bool) and duration >= 0:
            ttft = duration / 1_000_000
        return Observation(
            resident,
            sampler.peak,
            tok_s=result.throughput,
            ttft_ms=ttft,
            shared_mib=shared_mib,
            prompt_tok_s=result.prompt_tok_s,
            shared_baseline=baseline,
            baseline_taken=True,
            dedicated_delta=dedicated_delta,
        )

    def probe(ctx: int, kind: str) -> Observation:
        sampler.wait_for_abort()
        if hung["note"]:
            raise _UnloadTimeout(hung["note"])
        points = _measured_points(measured)
        predicted = _predict_mib(measured, ctx) if len(points) >= 2 else None
        # A larger window needs a bound. A smaller one is the downward search:
        # the trained window did not fit, and the next try is a halve.
        if len(points) == 1 and ctx > points[0][0]:
            ctx_1, reading_1 = points[0]
            if weights_mib is None or baseline is None or not resident_at.get(ctx_1, False):
                raise _NoBound("the second window had no bound")
            predicted = _one_window_mib(reading_1, ctx_1, baseline, weights_mib, ctx)
            if predicted is None:
                raise _NoBound("the second window had no bound")
            one_window_bound.append((ctx, predicted))
        if predicted is not None and predicted >= spec["abort_mib"]:
            predicted_over.append(ctx)
            return Observation(False, predicted, error=PREDICTED_OVER)
        sampler.reset()
        try:
            obs = sampler.cover(lambda: observe(ctx, kind))
        except _WindowOver:
            aborted.append(ctx)
            obs = _over(sampler.peak)
        if kind == "load" and isinstance(obs.peak_mib, int) and obs.error != ABORT_ERROR:
            measured.append((ctx, obs.peak_mib))
            resident_at[ctx] = bool(obs.resident)
        sampler.wait_for_abort()
        if hung["note"]:
            raise _UnloadTimeout(hung["note"])
        return obs

    spec_cap = spec.get("max_ctx", 1_048_576)
    if isinstance(spec_cap, bool) or not isinstance(spec_cap, int) or spec_cap < 1:
        spec_cap = default_ctx
    search_cap = min(spec_cap, default_ctx)
    try:
        found = search(
            default_ctx=search_cap,
            probe=probe,
            abort_mib=spec["abort_mib"],
            grain=spec["grain"],
            shared_growth_mib=spec["spill_shared_mib"],
            prompt_ratio=spec["spill_prompt_ratio"],
            max_ctx=search_cap,
        )
    except _Repairable as exc:
        if repair is not None:
            repair.append((exc.kind, exc.error))
        return _inconclusive(
            exc.error,
            quant=quant,
            default_ctx=default_ctx,
            flash=flash,
            kv=kv,
            ggml=ggml,
        )
    except SearchError as exc:
        return row(
            grant=grant,
            spec_rel=spec_rel,
            run_id=run_id,
            date=date,
            notes=_join(str(exc), notes),
            aborted=True,
            version=version,
            driver=driver,
            cuda=cuda,
            baseline=baseline,
            quant=quant,
            default_ctx=default_ctx,
            flash=flash,
            kv=kv,
            ggml=ggml,
            after_unload=after_unload,
            predicted_over=predicted_over,
            weights_mib=weights_mib,
            one_window_bound=one_window_bound,
        )
    except (_NoBound, _UnloadTimeout) as exc:
        return row(
            grant=grant,
            spec_rel=spec_rel,
            run_id=run_id,
            date=date,
            notes=_join(str(exc), notes),
            aborted=True,
            version=version,
            driver=driver,
            cuda=cuda,
            baseline=baseline,
            quant=quant,
            default_ctx=default_ctx,
            flash=flash,
            kv=kv,
            ggml=ggml,
            after_unload=after_unload,
            predicted_over=predicted_over,
            aborted_windows=aborted,
            sample_ms=sample_ms,
            weights_mib=weights_mib,
            one_window_bound=one_window_bound,
        )
    except _CountRefused as exc:
        return row(
            grant=grant,
            spec_rel=spec_rel,
            run_id=run_id,
            date=date,
            notes=_join(str(exc), notes),
            aborted=True,
            version=version,
            driver=driver,
            cuda=cuda,
            baseline=baseline,
            quant=quant,
            default_ctx=default_ctx,
            flash=flash,
            kv=kv,
            ggml=ggml,
            after_unload=after_unload,
            predicted_over=predicted_over,
            aborted_windows=aborted,
            sample_ms=sample_ms,
            weights_mib=weights_mib,
            one_window_bound=one_window_bound,
        )

    prompt_meta = prompts.get(found.max_ctx) if found.max_ctx is not None else None
    size_meta = sizes.get(found.max_ctx) if found.max_ctx is not None else None
    dedicated = dedicated_by_ctx.get(found.max_ctx) if found.max_ctx is not None else None
    stuck = [
        _stuck_note(ctx, reading)
        for ctx, reading in after_unload
        if reading is None or reading >= spec["abort_mib"]
    ]
    recorded = _join(found.reason, notes)
    if stuck:
        recorded = _join(recorded, "; ".join(stuck))
    return row(
        grant=grant,
        spec_rel=spec_rel,
        run_id=run_id,
        date=date,
        notes=recorded,
        aborted=False,
        version=version,
        driver=driver,
        cuda=cuda,
        baseline=baseline,
        quant=quant,
        default_ctx=default_ctx,
        flash=flash,
        kv=kv,
        ggml=ggml,
        found=found,
        prompt_meta=prompt_meta,
        size_meta=size_meta,
        dedicated=dedicated,
        aborted_windows=aborted,
        sample_ms=sample_ms,
        after_unload=after_unload,
        predicted_over=predicted_over,
        weights_mib=weights_mib,
        one_window_bound=one_window_bound,
    )


def _join(reason: str, extra: str) -> str:
    if extra:
        return f"{reason}. {extra}" if reason else extra
    return reason


def _row(
    spec: dict,
    entry: dict,
    *,
    grant: str | None,
    spec_rel: str,
    run_id: str,
    date: str,
    notes: str,
    aborted: bool,
    version: str,
    driver: str,
    cuda: str,
    baseline: int | None,
    quant: str | None = None,
    default_ctx: int | None = None,
    flash: str = UNRECORDED,
    kv: str = UNRECORDED,
    ggml: str = UNRECORDED,
    found=None,
    prompt_meta: tuple[int, str, int, int] | None = None,
    size_meta: tuple[int, int, bool] | None = None,
    dedicated: int | None = None,
    aborted_windows: list[int] | None = None,
    sample_ms: int | None = None,
    after_unload: list[tuple[int | None, int | None]] | None = None,
    predicted_over: list[int] | None = None,
    weights_mib: int | None = None,
    one_window_bound: list[tuple[int, int]] | None = None,
    default_context: str = UNRECORDED,
    backend_library: str = UNRECORDED,
    server_reason: str | None = None,
    ggml_reason: str | None = None,
    runner_pid: int | None = None,
) -> dict:
    num_predict = _refuse_confirm_num_predict(spec.get("confirm_num_predict", 1))
    settings = {
        "thinking_level": None,
        "num_predict": num_predict,
        "num_ctx": default_ctx,
        "batch": None,
        "static_shape": None,
        "precision": None,
        "grant": grant,
        "max_resident_ctx": None if found is None else found.max_ctx,
        "peak_mib": None if found is None else found.peak_mib,
        "ttft_ms": None if found is None else found.ttft_ms,
        "decode_tok_s": None if found is None else found.tok_s,
        "prompt_tok_s": None if found is None else found.prompt_tok_s,
        "shared_mib": None if found is None else found.shared_mib,
        "idle_baseline_mib": baseline,
        "ollama_version": version,
        "cuda_backend": cuda,
        "digest": entry["revision_or_digest"],
        "quantization": quant or entry.get("quant") or UNRECORDED,
        "flash_attention": flash,
        "kv_cache_type": kv,
        "nvidia_driver": driver,
        "ggml_cuda_sha": ggml,
        "default_context": default_context,
        "backend_library": backend_library,
        "shift": spec.get("shift"),
        "truncate": spec.get("truncate"),
    }
    if server_reason:
        settings["server_facts_reason"] = server_reason
    if ggml_reason:
        settings["ggml_cuda_reason"] = ggml_reason
    if runner_pid is not None:
        settings["runner_pid"] = runner_pid
    if sample_ms is not None:
        settings["sample_ms"] = sample_ms
    if found is not None:
        settings["limit"] = found.limit
        settings["search_calls"] = [list(call) for call in found.calls]
        if found.dedicated_delta is not None:
            settings["dedicated_delta_mib"] = found.dedicated_delta
    if aborted_windows:
        settings["aborted_windows"] = list(aborted_windows)
    if after_unload:
        settings["after_unload_mib"] = [{"ctx": ctx, "mib": mib} for ctx, mib in after_unload]
    if predicted_over:
        settings["predicted_over"] = list(predicted_over)
    if weights_mib is not None:
        settings["weights_mib"] = weights_mib
    if one_window_bound:
        settings["one_window_bound"] = [{"ctx": ctx, "mib": mib} for ctx, mib in one_window_bound]
    if prompt_meta is not None:
        settings["prompt_tokens"] = prompt_meta[0]
        settings["prompt_token_method"] = prompt_meta[1]
        settings["prompt_token_search"] = prompt_meta[2]
        settings["prompt_token_cold"] = prompt_meta[3]
    if dedicated is not None:
        settings["dedicated_mib"] = dedicated
    model = {
        "repo_or_tag": entry["model"],
        "revision_or_digest": entry["revision_or_digest"],
        "quant": quant or entry.get("quant") or UNRECORDED,
        "licence": entry.get("licence") or UNRECORDED,
    }
    if size_meta is not None:
        model["size"] = size_meta[0]
        model["size_vram"] = size_meta[1]
    throughput = None if found is None else found.tok_s
    row = {
        "id": f"{spec['id']}/{entry['model']}/{run_id}",
        "task": "generate",
        "model": model,
        "engine": {"name": "ollama", "version": version},
        "device": "5090",
        "settings": settings,
        "quality": [],
        "latency": _blank_latency(throughput),
        "date": date,
        "repeats": 1,
        "provisional": True,
        "provenance": {"experiment_spec": spec_rel, "run_id": run_id},
        "notes": notes,
    }
    if size_meta is not None:
        row["gpu_resident"] = size_meta[2]
    if aborted:
        row["aborted"] = True
    stamp_family(row)
    validate_row(row)
    return row
