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

A slot probe is separate from that thread. The runner opens one around each
measured call: a reading before, one a second while the call runs, and one
after. Those readings are system-wide counters only (total CPU, RAM, the
disk counter). The process walk runs in ``close``, after the measured call
has returned, and only when that slot's CPU max is over 30%. The receipt
records it as ``top_after_call``. Tests pass a ``slot_reader`` and leave the
thread off.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from interlock.adapters.base import BuildRefused

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
            for key, raw in zip(keys, parts, strict=False):
                try:
                    state[key] = float(raw)
                except ValueError:
                    pass
        except Exception:  # noqa: S110  # a probe failure stays unrecorded
            pass
    state.update(cpu_percent=UNRECORDED, cpu_freq_mhz=UNRECORDED, ram_pct=UNRECORDED)
    if psutil is not None:
        try:
            state["cpu_percent"] = float(psutil.cpu_percent(interval=0.2))
        except Exception:  # noqa: S110  # a probe failure stays unrecorded
            pass
        try:
            mhz = getattr(psutil.cpu_freq(), "current", None)
            if mhz is not None:
                state["cpu_freq_mhz"] = float(mhz)
        except Exception:  # noqa: S110  # a probe failure stays unrecorded
            pass
        try:
            state["ram_pct"] = float(psutil.virtual_memory().percent)
        except Exception:  # noqa: S110  # a probe failure stays unrecorded
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
    # None plus start_thread False means a test takes no per-slot samples.
    # None plus start_thread True is the real reader (production).
    slot_reader: Callable[[str | None], dict] | None = None


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _span(readings: list[dict], key: str) -> dict | None:
    values = [row[key] for row in readings if row.get(key) is not None]
    if not values:
        return None
    return {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}


def summarize_slot(readings: list[dict]) -> dict:
    """Min, max and mean for one slot.

    A disk counter that cannot be mapped is ``unrecorded`` with a reason.
    It is not a quiet zero and it is not a bare null. Process names are not
    taken here. ``SlotProbe.close`` adds ``top_after_call`` after the call,
    and only when this slot's CPU max is over 30%.
    """

    cpu = _span(readings, "cpu_pct")
    drives = [row["drive"] for row in readings if row.get("drive")]
    disk = _span(readings, "disk_read_mb_s")
    reasons = [row["disk_reason"] for row in readings if row.get("disk_reason")]
    out = {
        "samples": len(readings),
        "cpu_pct": cpu,
        "ram_pct": _span(readings, "ram_pct"),
        "disk_read_mb_s": disk if disk is not None or not reasons else UNRECORDED,
        "load_drive": drives[-1] if drives else None,
    }
    if disk is None and reasons:
        out["disk_reason"] = reasons[-1]
    return out


def disk_number_script(letter: str) -> str:
    """PowerShell that prints the disk number for one drive letter.

    Windows names ``psutil`` counters ``PhysicalDrive<n>``. ``(Get-Partition |
    Get-Disk).Number`` is that ``n``. This does not call Get-PhysicalDisk.
    """

    if len(letter) != 1 or not letter.isalpha():
        raise OSError(f"not a drive letter: {letter!r}")
    safe = letter.upper()
    return (
        f"$n = (Get-Partition -DriveLetter {safe} -ErrorAction Stop | Get-Disk).Number; "
        "if ($null -eq $n) { throw 'no disk number' }; "
        "[string]$n"
    )


def windows_disk_number(letter: str, run=None) -> int:
    """The physical-disk number for a drive letter. Tests pass ``run`` and do not start PowerShell."""

    script = disk_number_script(letter)
    if run is None:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    else:
        completed = run(script)
    if getattr(completed, "returncode", 1) != 0:
        detail = (getattr(completed, "stderr", None) or getattr(completed, "stdout",
            None) or "disk number query failed")
        first = str(detail).strip().splitlines()
        raise OSError(first[0][:200] if first else "disk number query failed")
    text = str(getattr(completed, "stdout", "") or "").strip()
    if not text.isdigit():
        raise OSError("disk number query did not return a number")
    return int(text)


class _DiskRate:
    """Read MB/s from the physical disk behind a drive letter.

    The first look at a disk is a baseline, not a rate. A mapping or counter
    failure returns a reason so the receipt can say ``unrecorded``.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic, disk_number: Callable[[str],
        int] | None = None) -> None:
        self._clock = clock
        self._disk_number = disk_number if disk_number is not None else windows_disk_number
        self._last: dict[str, tuple[float, int]] = {}
        self._numbers: dict[str, int | None] = {}
        self._reasons: dict[str, str] = {}

    def mb_s(self, drive: str | None) -> tuple[float | None, str | None]:
        if not drive or drive == UNRECORDED:
            return None, None
        letter = drive[:1].upper()
        if not letter.isalpha():
            return None, f"not a drive letter: {drive}"
        if psutil is None:
            return None, "psutil is not installed"
        number, map_reason = self._number(letter)
        if number is None:
            return None, map_reason or "the drive letter could not be mapped to a physical disk"
        try:
            counters = psutil.disk_io_counters(perdisk=True) or {}
        except Exception as exc:
            return None, f"disk counters unreadable: {type(exc).__name__}"
        key = _disk_key(counters, number)
        if key is None:
            return None, f"no counter named PhysicalDrive{number}"
        try:
            read = int(counters[key].read_bytes)
        except Exception as exc:
            return None, f"disk counter unreadable: {type(exc).__name__}"
        now = self._clock()
        previous = self._last.get(key)
        self._last[key] = (now, read)
        if previous is None:
            return None, None
        elapsed = now - previous[0]
        if elapsed <= 0:
            return None, None
        delta = read - previous[1]
        if delta < 0:
            return None, None
        return delta / elapsed / (1024 * 1024), None

    def _number(self, letter: str) -> tuple[int | None, str | None]:
        if letter in self._numbers:
            return self._numbers[letter], self._reasons.get(letter)
        try:
            number = int(self._disk_number(letter))
        except Exception as exc:
            reason = str(exc).strip().splitlines()
            text = reason[0][:200] if reason else "the drive letter could not be mapped to a physical disk"
            self._numbers[letter] = None
            self._reasons[letter] = text
            return None, text
        self._numbers[letter] = number
        return number, None


def _disk_key(counters: dict, number: int) -> str | None:
    """The ``PhysicalDrive<n>`` counter. A letter key is not a Windows counter."""

    wanted = f"PHYSICALDRIVE{number}"
    for key in counters:
        label = str(key).upper().replace("\\", "").rstrip(":")
        if label == wanted:
            return key
    return None


_DISK = _DiskRate()


# How long the after-call rank waits between its two process snapshots.
TOP_AFTER_GAP_S = 1.0


def _own_pids() -> set[int] | None:
    """This process and its children. None when that tree cannot be read."""

    try:
        me = psutil.Process()
        excluded = {int(me.pid)}
        for child in me.children(recursive=True):
            excluded.add(int(child.pid))
    except Exception:
        return None
    return excluded


def _cpu_seconds(proc) -> float | None:
    """User plus system CPU time, in seconds. None when the counter is missing."""

    times = proc.cpu_times()
    user = _number(getattr(times, "user", None))
    system = _number(getattr(times, "system", None))
    if user is None or system is None:
        return None
    return user + system


def _snapshot_cpu() -> dict[int, tuple[float, str]]:
    """One look at every process: pid to (user + system seconds, name)."""

    found: dict[int, tuple[float, str]] = {}
    try:
        processes = psutil.process_iter()
    except Exception:
        return found
    for proc in processes:
        try:
            seconds = _cpu_seconds(proc)
            if seconds is None:
                continue
            found[int(proc.pid)] = (seconds, str(proc.name()))
        except Exception:  # noqa: S112  # a dead process is skipped
            continue
    return found


def _top_other(
    limit: int = 3,
    *,
    sleep: Callable[[float], None] | None = None,
    gap_s: float = TOP_AFTER_GAP_S,
) -> list[str]:
    """Other processes ranked by CPU time gained between two snapshots.

    The rank is the user-plus-system time gained across about a second.
    pid 0 is always left out. This process and its children are left out.
    pid 4 (System) stays, so it ranks only when the kernel was busy.
    ``SlotProbe.close`` calls this after the measured call, and only when
    that slot's CPU max is over 30%.
    """

    if psutil is None:
        return []
    own = _own_pids()
    if own is None:
        return []
    excluded = set(own)
    excluded.add(0)
    pause = time.sleep if sleep is None else sleep
    first = _snapshot_cpu()
    pause(gap_s)
    second = _snapshot_cpu()
    ranked: list[tuple[float, str]] = []
    for pid, (later, name) in second.items():
        if pid in excluded:
            continue
        earlier = first.get(pid)
        if earlier is None:
            continue
        delta = later - earlier[0]
        if delta <= 0:
            continue
        ranked.append((delta, name))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [name for _delta, name in ranked[:limit]]


def read_slot(drive: str | None) -> dict:
    """One sample of total CPU, used RAM and disk read MB/s.

    System-wide counters only. ``drive`` is the letter the weights are loaded
    from. Disk stays None until that letter is known and the counter can be
    read. The process walk is ``_top_other``, and it does not run from here.
    """

    cpu = None
    ram = None
    if psutil is not None:
        try:
            cpu = float(psutil.cpu_percent(interval=None))
        except Exception:
            cpu = None
        try:
            ram = float(psutil.virtual_memory().percent)
        except Exception:
            ram = None
    rate, reason = _DISK.mb_s(drive)
    return {
        "cpu_pct": cpu,
        "ram_pct": ram,
        "disk_read_mb_s": rate,
        "disk_reason": reason,
    }


class SlotProbe:
    """Readings around one measured call. The runner opens it and closes it."""

    def __init__(self, reader: Callable[[str | None], dict], *, start_thread: bool = True,
        interval_s: float = 1.0) -> None:
        self._reader = reader
        self._start_thread = start_thread
        self.interval_s = float(interval_s)
        self._drive: str | None = None
        self._readings: list[dict] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._summary: dict | None = None

    def set_drive(self, drive: str | None) -> None:
        with self._lock:
            self._drive = None if not drive or drive == UNRECORDED else str(drive)

    def sample(self) -> None:
        with self._lock:
            drive = self._drive
        try:
            raw = self._reader(drive) or {}
        except Exception:
            raw = {}
        reason = raw.get("disk_reason")
        reading = {
            "cpu_pct": _number(raw.get("cpu_pct")),
            "ram_pct": _number(raw.get("ram_pct")),
            "disk_read_mb_s": _number(raw.get("disk_read_mb_s")),
            "disk_reason": None if not reason else str(reason).strip().splitlines()[0][:200],
            "drive": drive,
        }
        with self._lock:
            self._readings.append(reading)

    def open(self) -> None:
        self.sample()
        if self._start_thread and self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="interlock-slot-sample", daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.sample()

    def close(self) -> dict:
        """The slot summary. A second call does not sample or rank again.

        When the slot's CPU max is over 30%, rank other processes from two
        snapshots taken after the measured call. An idle slot does not walk.
        """

        if self._closed:
            return self._summary or {}
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)
        self.sample()
        summary = self.summary()
        cpu = summary.get("cpu_pct")
        maximum = _number(cpu.get("max")) if isinstance(cpu, dict) else None
        if maximum is not None and maximum > 30:
            summary["top_after_call"] = _top_other(3)
        self._summary = summary
        self._closed = True
        return summary

    def summary(self) -> dict:
        with self._lock:
            readings = list(self._readings)
        return summarize_slot(readings)


def slot_probe(hooks: WatchHooks) -> SlotProbe | None:
    """A probe for this run, or None when a test asked for no slot samples."""

    if hooks.slot_reader is None and not hooks.start_thread:
        return None
    return SlotProbe(hooks.slot_reader or read_slot, start_thread=hooks.start_thread)


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
        self._thread = threading.Thread(target=self._loop, name="interlock-run-watch", daemon=True)
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
        except Exception:  # noqa: S110  # a sampler failure leaves the peak
            pass

    def _fire(self, record: Callable[[], None], code: int) -> None:
        self._fired = True
        try:
            record()
        except Exception as exc:  # the process must still end
            try:
                print(f"interlock: the run watch could not write the receipt: {exc}", file=sys.stderr)
            except Exception:  # noqa: S110  # the process must still end
                pass  # pragma: no cover
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
