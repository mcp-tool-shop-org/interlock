"""Live fill-card run. The search in :mod:`switchyard.harness.e5` decides.

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
re-read have finished. The spec ceiling is not a probe.
Nothing here rests between models. A missing shared-memory reading or a
missing prompt rate is a spill, which the pure search already treats as a
step back. A confirmation prefix is counted again from a cold load, and a
disagreement refuses the model.

Tests pass fakes. The real nvidia-smi and counter readers run only when a
caller builds :func:`real_fill_seams` and then calls them.
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from switchyard.adapters.base import AdapterError, StepRequest
from switchyard.adapters.ollama import context_length, quantization_level, url_for
from switchyard.cli import announce_run
from switchyard.families import stamp_family
from switchyard.guards import GuardError, require_card_empty, require_grant, require_heartbeat
from switchyard.harness.e5 import (
    SearchError,
    Observation,
    fit_prompt_words,
    search,
)
from switchyard.harness.receipts import write_receipt
from switchyard.receipt import validate_row

UNRECORDED = "unrecorded"
ABORT_ERROR = "card memory reached abort_mib"
PREDICTED_OVER = "predicted_over"
UNLOAD_SETTLE_S = 10.0
_SETTLE_STEP_S = 0.2
MEMORY_ARGV = ("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits")
DRIVER_ARGV = ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits")
CUDA_ARGV = ("nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader,nounits")
SHARED_COUNTER = r"\GPU Process Memory(*)\Shared Usage"
DEDICATED_COUNTER = r"\GPU Process Memory(*)\Dedicated Usage"
_MIB = 1024 * 1024


class _WindowOver(Exception):
    """The sampler reached abort_mib while this window was in flight."""


class _CountRefused(Exception):
    """The cold re-count of a confirmation prefix disagreed with the search."""


class _NoBound(Exception):
    """The next window has no share that can be trusted, so it is not loaded."""


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


def real_fill_seams(adapter, *, run=None, read_counter=None, pids=None) -> FillSeams:
    """The command's seams. ``run``, ``read_counter`` and ``pids`` are test fakes."""

    run = run or subprocess.run
    read_counter = read_counter or (lambda counter: _powershell_counter(counter, run))
    pids = pids or (lambda: _runner_pids(run))

    def card_mib() -> int | None:
        return parse_mib(_capture(run, MEMORY_ARGV))

    def shared_mib() -> int | None:
        return shared_mib_for_runners(parse_counter_text(read_counter(SHARED_COUNTER) or ""), pids())

    def dedicated_mib() -> int | None:
        return shared_mib_for_runners(parse_counter_text(read_counter(DEDICATED_COUNTER) or ""), pids())

    def driver() -> str | None:
        return first_line(_capture(run, DRIVER_ARGV))

    def cuda() -> str | None:
        return first_line(_capture(run, CUDA_ARGV))

    return FillSeams(
        card_mib=card_mib,
        shared_mib=shared_mib,
        dedicated_mib=dedicated_mib,
        ollama_version=lambda: _ollama_version(adapter),
        driver=driver,
        cuda_backend=cuda,
        sleep=time.sleep,
        clock=time.monotonic,
        threaded=True,
    )


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

        There is no timeout. The settle bound is about 10 seconds and the
        unload is on top of that, so a shorter wait would return early.
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


def mib_for_pid(samples: list[tuple[str, float]], pid: int) -> int | None:
    """The largest Shared or Dedicated sample whose instance names this pid."""

    needle = f"pid_{pid}_".lower()
    found = []
    for name, cooked in samples:
        if not isinstance(name, str) or needle not in name.lower():
            continue
        if isinstance(cooked, bool) or not isinstance(cooked, (int, float)) or cooked < 0:
            continue
        found.append(int(cooked / _MIB))
    if not found:
        return None
    return max(found)


def shared_mib_for_runners(samples: list[tuple[str, float]], pids: list[int]) -> int | None:
    """One runner pid. Several matches is a missing reading, not a guess."""

    if len(pids) != 1:
        return None
    return mib_for_pid(samples, pids[0])


def runner_pids(lines: list[str]) -> list[int]:
    """Ollama runner processes. The server process itself is not a match.

    A match names ollama and runner, or its executable is the bundled runner.
    """

    found: list[int] = []
    for line in lines:
        if not isinstance(line, str):
            continue
        parts = line.split(None, 2)
        if not parts or not parts[0].isdigit():
            continue
        blob = line.lower()
        name = parts[1].lower() if len(parts) > 1 else ""
        runner = "runner" in blob and "ollama" in blob
        named = name.startswith("llama-server") or "ollama_llama" in name
        if not runner and not named:
            continue
        pid = int(parts[0])
        if pid not in found:
            found.append(pid)
    return found


def facts_from_show(payload: dict) -> tuple[str, str, str]:
    """Flash attention, KV cache type, and a ggml cuda sha. Absent stays unrecorded."""

    flash = UNRECORDED
    kv = UNRECORDED
    ggml = UNRECORDED
    pairs: list[tuple[str, str]] = []
    parameters = payload.get("parameters")
    if isinstance(parameters, str):
        for line in parameters.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split(None, 1)
            if len(parts) == 2:
                pairs.append((parts[0].lower(), parts[1].strip()))
    info = payload.get("model_info")
    if isinstance(info, dict):
        for key, value in info.items():
            if isinstance(key, str) and isinstance(value, (str, int, float)) and not isinstance(value, bool):
                pairs.append((key.lower(), str(value)))
    for key, value in pairs:
        if flash == UNRECORDED and ("flash_attn" in key or "flash_attention" in key):
            flash = value
        if kv == UNRECORDED and "kv" in key and "cache" in key:
            kv = value
        if ggml == UNRECORDED and "ggml" in key and ("sha" in key or "cuda" in key):
            ggml = value
    return flash, kv, ggml


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


def _runner_pids(run) -> list[int]:
    script = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -match 'ollama' -or ($_.CommandLine -and $_.CommandLine -match 'ollama') } | "
        "ForEach-Object { $_.ProcessId.ToString() + ' ' + $_.Name + ' ' + $_.CommandLine }"
    )
    text = _capture(run, ["powershell", "-NoProfile", "-Command", script])
    return runner_pids([] if text is None else text.splitlines())


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
) -> list[dict]:
    """Measure each roster model. Unload at the end of the model and on any exit."""

    rows: list[dict] = []
    stop_note = None
    try:
        for entry in spec["models"]:
            require_grant(grant, uses_5090=True)
            require_heartbeat(heartbeat, wall())
            adapter = _ollama(adapters)
            require_card_empty(adapter.loaded_models())
            marker.entered = True
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
                )
            finally:
                _unload(adapter, entry["model"])
            rows.append(row)
            if row.get("aborted"):
                stop_note = row.get("notes") or "fill-card step failed"
                break
    finally:
        try:
            if rows:
                write_receipt(root, run_id, rows)
        finally:
            if marker.entered:
                marker.close(_iso(wall()))
    if stop_note:
        from switchyard.harness.runner import RunnerError

        raise RunnerError(stop_note)
    announce_run(rows, finished=True)
    return rows


def _unload(adapter, model: str) -> None:
    try:
        adapter.unload(model, "5090")
    except Exception:
        return


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
) -> dict:
    model = entry["model"]
    version = _text(seams.ollama_version())
    driver = _text(seams.driver())
    cuda = _text(seams.cuda_backend())
    baseline = _optional_int(seams.card_mib())
    if baseline is not None and baseline >= spec["abort_mib"]:
        return _row(
            spec,
            entry,
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
        flash, kv, ggml = facts_from_show(shown)
    except AdapterError as exc:
        return _row(
            spec,
            entry,
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
        )

    num_predict = spec.get("confirm_num_predict", 1)
    if isinstance(num_predict, bool) or not isinstance(num_predict, int) or num_predict < 0:
        num_predict = 1
    fraction = float(spec["confirm_prompt_fraction"])
    sample_ms = spec.get("sample_ms", 200)
    if isinstance(sample_ms, bool) or not isinstance(sample_ms, int) or sample_ms < 1:
        sample_ms = 200
    seed = (spec.get("dataset") or {}).get("seed")
    if not isinstance(seed, str) or not seed.strip():
        return _row(
            spec,
            entry,
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

    def on_abort() -> None:
        try:
            cancel = getattr(adapter, "cancel", None)
            if callable(cancel):
                cancel()
        finally:
            try:
                adapter.unload(model, "5090")
            finally:
                after_unload.append((current["ctx"], _settled_card(seams, spec["abort_mib"])))

    sampler = CardSampler(seams, spec["abort_mib"], sample_ms, on_abort)

    def count(text: str, ctx: int) -> int:
        settings = {"num_ctx": ctx, "temperature": 0}
        if shift:
            settings["shift"] = False
        if truncate:
            settings["truncate"] = False
        try:
            counted = adapter.count_tokens(model, text, settings=settings, device="5090")
        except AdapterError:
            if sampler.over:
                raise _WindowOver()
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
                return Observation(False, sampler.peak, error=str(exc))
            if _count_leaves_loaded(entry.get("keep_alive")):
                adapter.unload(model, "5090")
            try:
                cold_tokens = count(text, ctx)
            except _WindowOver:
                aborted.append(ctx)
                return _over(sampler.peak)
            except AdapterError as exc:
                if sampler.over:
                    aborted.append(ctx)
                    return _over(sampler.peak)
                return Observation(False, sampler.peak, error=str(exc))
            if cold_tokens != tokens:
                if _count_leaves_loaded(entry.get("keep_alive")):
                    adapter.unload(model, "5090")
                raise _CountRefused(
                    f"the search counted {tokens} and a cold load counted {cold_tokens}"
                )
            prompts[ctx] = (cold_tokens, _prompt_method(fit), tokens, cold_tokens)
            if _count_leaves_loaded(entry.get("keep_alive")):
                adapter.unload(model, "5090")
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
        try:
            result = adapter.run(request)
        except AdapterError as exc:
            if sampler.over:
                aborted.append(ctx)
                return _over(sampler.peak)
            return Observation(False, sampler.peak, error=str(exc))
        if sampler.over:
            aborted.append(ctx)
            return _over(sampler.peak)
        if not result.ok:
            return Observation(False, sampler.peak, error=result.error or "generate failed")
        resident, size, vram = _residency(adapter.loaded_models(), model)
        if size is not None and vram is not None:
            sizes[ctx] = (size, vram, resident)
        reading = _optional_int(seams.dedicated_mib())
        if reading is not None:
            dedicated_by_ctx[ctx] = reading
        duration = result.prompt_eval_duration_ns
        ttft = None
        if isinstance(duration, int) and not isinstance(duration, bool) and duration >= 0:
            ttft = duration / 1_000_000
        return Observation(
            resident,
            sampler.peak,
            tok_s=result.throughput,
            ttft_ms=ttft,
            shared_mib=_optional_int(seams.shared_mib()),
            prompt_tok_s=result.prompt_tok_s,
        )

    def probe(ctx: int, kind: str) -> Observation:
        sampler.wait_for_abort()
        points = _measured_points(measured)
        predicted = _predict_mib(measured, ctx) if len(points) >= 2 else None
        if len(points) == 1 and ctx != points[0][0]:
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
        return obs

    try:
        found = search(
            default_ctx=default_ctx,
            probe=probe,
            abort_mib=spec["abort_mib"],
            grain=spec["grain"],
            shared_growth_mib=spec["spill_shared_mib"],
            prompt_ratio=spec["spill_prompt_ratio"],
            max_ctx=spec.get("max_ctx", 1_048_576),
        )
    except SearchError as exc:
        return _row(
            spec,
            entry,
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
    except _NoBound as exc:
        return _row(
            spec,
            entry,
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
        return _row(
            spec,
            entry,
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
    return _row(
        spec,
        entry,
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
) -> dict:
    num_predict = spec.get("confirm_num_predict", 1)
    if isinstance(num_predict, bool) or not isinstance(num_predict, int):
        num_predict = 1
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
        "shift": spec.get("shift"),
        "truncate": spec.get("truncate"),
    }
    if sample_ms is not None:
        settings["sample_ms"] = sample_ms
    if found is not None:
        settings["limit"] = found.limit
        settings["search_calls"] = [[kind, ctx] for kind, ctx in found.calls]
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
