"""The run watcher: a thread that can end a run the main thread cannot.

Python cannot raise KeyboardInterrupt inside a native call, so a
``compile_model`` that runs for an hour cannot be stopped from the main
thread, and a hard kill writes no receipt. The watcher runs beside the run and
does two jobs, both from outside the native call:

- compile bound: a compile that has run longer than the ceiling is recorded,
  the receipt is flushed, and the process ends with ``EXIT_COMPILE_TIMEOUT``;
- RAM stop: system RAM at or above the stop percent flushes the receipt and
  ends the process with ``EXIT_RAM_STOP``.

The runner owns what to write (the callbacks); the watcher owns when. The
clock, the RAM and working-set samplers and the exit function are injectable,
so tests prove the receipt is written while a compile is still blocked and
never end the test process.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable

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
GIB = 2**30


def _system_ram_percent() -> float:
    return float(psutil.virtual_memory().percent)


def _own_rss_bytes() -> int:
    return int(psutil.Process().memory_info().rss)


def _exit_now(code: int) -> None:  # pragma: no cover - ends the process
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(code)


@dataclass
class WatchHooks:
    """The watcher's seams. The defaults are the real ones.

    ``ram_sampler`` None means no RAM reading is possible (psutil missing).
    ``start_thread`` False lets a test drive :meth:`RunWatch.tick` itself.
    """

    clock: Callable[[], float] = time.monotonic
    ram_sampler: Callable[[], float] | None = _system_ram_percent if psutil is not None else None
    rss_sampler: Callable[[], int] | None = _own_rss_bytes if psutil is not None else None
    exit_fn: Callable[[int], None] = _exit_now
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
    ) -> None:
        self.tracker = tracker
        self.hooks = hooks
        self.ceiling_s = float(ceiling_s)
        self.stop_pct = float(stop_pct)
        self._on_compile_timeout = on_compile_timeout
        self._on_ram_stop = on_ram_stop
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


def _plain(value: float):
    return int(value) if float(value).is_integer() else value
