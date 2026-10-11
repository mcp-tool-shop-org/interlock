"""The run watcher: a thread that can end a run the main thread cannot.

Python cannot raise KeyboardInterrupt inside a native call, so a
``compile_model`` that runs for an hour cannot be stopped from the main
thread, and a hard kill writes no receipt. The watcher runs beside the run and
does two jobs, both from outside the native call:

- compile bound: a compile that has run longer than the ceiling is recorded,
  the receipt is flushed, and the process ends with ``EXIT_COMPILE_TIMEOUT``;
- RAM stop: system RAM at or above the stop percent flushes the receipt and
  ends the process with ``EXIT_RAM_STOP``;
- operator stop: a file ``receipts/<run_id>.stop`` flushes the receipt, closes
  the marker with the file's first line as the reason, and ends the process
  with ``EXIT_OPERATOR_STOP``.

The runner owns what to write (the callbacks); the watcher owns when. The
clock, the RAM and working-set samplers and the exit function are injectable,
so tests prove the receipt is written while a compile is still blocked and
never end the test process.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from switchyard.adapters.base import BuildRefused

try:  # missing psutil means the RAM guard records "unrecorded" and does nothing
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

COMPILE_CEILING_S = 900.0
RAM_STOP_PCT = 88.0
RAM_SAMPLE_S = 2.0
# Distinctive, non-zero, and not a signal number or a Python usage error.
EXIT_COMPILE_TIMEOUT = 75
EXIT_RAM_STOP = 76
EXIT_OPERATOR_STOP = 77
UNRECORDED = "unrecorded"
GIB = 2**30
# Blocked order (R&D ruling 2026-10-10): between blocks the used RAM must come
# back to within RELEASE_TOLERANCE_BYTES of the run's starting baseline, waited
# for up to RELEASE_WAIT_S; a build is refused when used RAM plus the largest
# build step seen for the model would cross BUILD_GUARD_PCT of total RAM.
RELEASE_TOLERANCE_BYTES = 3 * GIB
RELEASE_WAIT_S = 60.0
RELEASE_POLL_S = 2.0
BUILD_GUARD_PCT = 85
BUILD_STEP_8B_BYTES = 12 * GIB
BUILD_STEP_DEFAULT_BYTES = 6 * GIB


def _system_ram_percent() -> float:
    return float(psutil.virtual_memory().percent)


def _own_rss_bytes() -> int:
    return int(psutil.Process().memory_info().rss)


def _system_used_bytes() -> int:
    return int(psutil.virtual_memory().used)


def _system_total_bytes() -> int:
    return int(psutil.virtual_memory().total)


def read_start_state(uses_5090: bool) -> dict:
    """Device state at run start. Recorded, never gated: a value that cannot be
    read is ``unrecorded`` and nothing here raises."""

    state: dict = {}
    if uses_5090:
        keys = ("gpu_temp_c", "sm_clock_mhz", "power_w", "memory_used_mib")
        state.update({key: UNRECORDED for key in keys})
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu,clocks.sm,power.draw,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            parts = [item.strip() for item in out.stdout.splitlines()[0].split(",")] if out.returncode == 0 else []
            for key, raw in zip(keys, parts):
                try:
                    state[key] = float(raw)
                except ValueError:
                    pass
        except Exception:
            pass
    state.update(cpu_percent=UNRECORDED, cpu_freq_mhz=UNRECORDED, ram_pct=UNRECORDED)
    if psutil is not None:
        try:
            state["cpu_percent"] = float(psutil.cpu_percent(interval=0.2))
        except Exception:
            pass
        try:
            mhz = getattr(psutil.cpu_freq(), "current", None)
            if mhz is not None:
                state["cpu_freq_mhz"] = float(mhz)
        except Exception:
            pass
        try:
            state["ram_pct"] = float(psutil.virtual_memory().percent)
        except Exception:
            pass
    return state


def _exit_now(code: int) -> None:  # pragma: no cover - ends the process
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(code)


@dataclass
class WatchHooks:
    """The watcher's seams. The defaults are the real ones.

    ``ram_sampler`` None means no RAM reading is possible (psutil missing). The
    same holds for ``used_sampler`` and ``total_sampler`` (bytes), which blocked
    order reads for the release check and the build guard; ``sleep`` paces the
    release wait.
    ``start_thread`` False lets a test drive :meth:`RunWatch.tick` itself.
    """

    clock: Callable[[], float] = time.monotonic
    ram_sampler: Callable[[], float] | None = _system_ram_percent if psutil is not None else None
    rss_sampler: Callable[[], int] | None = _own_rss_bytes if psutil is not None else None
    used_sampler: Callable[[], int] | None = _system_used_bytes if psutil is not None else None
    total_sampler: Callable[[], int] | None = _system_total_bytes if psutil is not None else None
    sleep: Callable[[float], None] = time.sleep
    exit_fn: Callable[[int], None] = _exit_now
    state_sampler: Callable[[bool], dict] = read_start_state
    interval_s: float = RAM_SAMPLE_S
    lock_wait_s: float = 10.0  # how long the watcher waits for the run's state lock before flushing without it
    start_thread: bool = True


class RunWatch:
    def __init__(
        self,
        tracker,
        hooks: WatchHooks,
        *,
        ceiling_s: float,
        on_compile_timeout: Callable[[dict], None],
        on_ram_stop: Callable[[str], None],
        stop_pct: float = RAM_STOP_PCT,
        stop_file: Path | None = None,
        on_operator_stop: Callable[[str], None] | None = None,
    ) -> None:
        self.tracker = tracker
        self.hooks = hooks
        self.ceiling_s = float(ceiling_s)
        self.stop_pct = float(stop_pct)
        self._on_compile_timeout = on_compile_timeout
        self._on_ram_stop = on_ram_stop
        self.stop_file = stop_file
        self._on_operator_stop = on_operator_stop
        self._lock = threading.Lock()
        self._peak_pct: float | None = None
        self._samples = 0
        self._compile_seen: tuple[str, float] | None = None
        self._peak_rss = 0
        self._fired = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- what the rows record ---------------------------------------------

    def ram_guard(self):
        """``settings.ram_guard`` for a row: the stop, the peak seen and the samples taken."""

        if self.hooks.ram_sampler is None:
            return "unrecorded"
        with self._lock:
            return {
                "stop_pct": _plain(self.stop_pct),
                "sample_s": _plain(self.hooks.interval_s),
                "peak_pct": self._peak_pct,
                "samples": self._samples,
            }

    def sample_ram(self) -> float | None:
        """One RAM reading, folded into the peak. None when it cannot be read."""

        sampler = self.hooks.ram_sampler
        if sampler is None:
            return None
        try:
            percent = float(sampler())
        except Exception:
            return None
        with self._lock:
            self._samples += 1
            if self._peak_pct is None or percent > self._peak_pct:
                self._peak_pct = percent
        return percent

    # --- the thread --------------------------------------------------------

    def start(self) -> None:
        if not self.hooks.start_thread or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="switchyard-run-watch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.wait(self.hooks.interval_s):
            if self.tick():
                return

    def tick(self) -> bool:
        """One look. True when the watcher fired and asked the process to end."""

        if self._fired:
            return True
        reason = self._operator_stop_reason()
        if reason is not None:
            handler = self._on_operator_stop
            self._fire(lambda: handler(reason), EXIT_OPERATOR_STOP)
            return True
        percent = self.sample_ram()
        self._sample_working_set()
        if percent is not None and percent >= self.stop_pct:
            self._fire(
                lambda: self._on_ram_stop(f"system RAM reached {percent:.1f}% (stop at {_plain(self.stop_pct)}%)"),
                EXIT_RAM_STOP,
            )
            return True
        active = self.tracker.active()
        if active is not None:
            label, started = active
            elapsed = self.hooks.clock() - started
            if elapsed > self.ceiling_s:
                info = {
                    "site": label,
                    "elapsed_s": elapsed,
                    "ceiling_s": self.ceiling_s,
                    "peak_working_set_gb": self._peak_rss / GIB if self._peak_rss else None,
                }
                self._fire(lambda: self._on_compile_timeout(info), EXIT_COMPILE_TIMEOUT)
                return True
        return False

    def _operator_stop_reason(self) -> str | None:
        """The reason in ``<run_id>.stop`` (first line, trimmed), or None when there is no stop file.

        The file is renamed to ``.stop.done`` before the watcher acts, so it is never read twice."""

        path = self.stop_file
        if path is None or self._on_operator_stop is None:
            return None
        try:
            if not path.is_file():
                return None
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        lines = text.splitlines()
        reason = (lines[0].strip() if lines else "") or "no reason given"
        try:
            os.replace(path, path.with_name(path.name + ".done"))
        except OSError:
            try:
                path.unlink()
            except OSError:
                pass
        return reason

    def _sample_working_set(self) -> None:
        """The process working set, peaked over the running compile (reset when a new one begins)."""

        active = self.tracker.active()
        if active != self._compile_seen:
            self._compile_seen = active
            self._peak_rss = 0
        sampler = self.hooks.rss_sampler
        if sampler is None:
            return
        try:
            self._peak_rss = max(self._peak_rss, int(sampler()))
        except Exception:
            pass

    def _fire(self, record: Callable[[], None], code: int) -> None:
        self._fired = True
        try:
            record()
        except Exception as exc:  # the process must still end
            try:
                print(f"switchyard: the run watch could not write the receipt: {exc}", file=sys.stderr)
            except Exception:  # pragma: no cover
                pass
        finally:
            self.hooks.exit_fn(code)


def read_bytes(sampler: Callable[[], int] | None) -> int | None:
    """One byte reading, or None when there is no sampler or it fails."""

    if sampler is None:
        return None
    try:
        return int(sampler())
    except Exception:
        return None


class BuildGuard:
    """The pre-build memory check of blocked order.

    An adapter wraps each pipeline or model build in :meth:`building`. Before
    the build it refuses (``BuildRefused``) when used RAM plus the largest build
    step seen for that model would cross ``pct`` of total RAM; the first build of
    a model uses a default step (12 GiB for an 8B model, else 6 GiB). Around the
    build it measures the step: the growth of the process working set (used RAM
    when no working-set sampler exists). With no RAM reading the guard allows
    the build and records nothing.
    """

    def __init__(self, hooks: WatchHooks, *, pct: float = BUILD_GUARD_PCT) -> None:
        self.hooks = hooks
        self.pct = pct
        self.steps: dict[str, int] = {}
        self.refused: list[dict] = []

    def step_for(self, model: str) -> int:
        seen = self.steps.get(model)
        if seen is not None:
            return seen
        return BUILD_STEP_8B_BYTES if "8b" in model.lower() else BUILD_STEP_DEFAULT_BYTES

    def check(self, model: str) -> None:
        used = read_bytes(self.hooks.used_sampler)
        total = read_bytes(self.hooks.total_sampler)
        if used is None or total is None:
            return
        step = self.step_for(model)
        limit = total * self.pct / 100
        if used + step > limit:
            self.refused.append({"model": model, "used": used, "step": step, "total": total})
            raise BuildRefused(
                f"memory headroom: used {used / GIB:.1f} GiB plus a {step / GIB:.1f} GiB build step "
                f"for {model} would pass {_plain(self.pct)}% of {total / GIB:.1f} GiB"
            )

    @contextmanager
    def building(self, model: str):
        self.check(model)
        sampler = self.hooks.rss_sampler or self.hooks.used_sampler
        before = read_bytes(sampler)
        try:
            yield
        finally:
            after = read_bytes(sampler)
            if before is not None and after is not None and after > before:
                self.steps[model] = max(self.steps.get(model, 0), after - before)


def _plain(value: float):
    return int(value) if float(value).is_integer() else value
