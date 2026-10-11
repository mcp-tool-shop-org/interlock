"""The adapter interface.

An adapter runs one measured step and unloads the model it loaded. The harness
talks only to this shape. Kimi's OpenVINO adapter plugs in by adding
``switchyard.adapters.openvino`` with a function named ``adapter`` that returns
an object with ``run`` and ``unload``. The harness imports that module only
when a step names the OpenVINO engine. It does not fall back to another device.

``run`` returns a result. A failed step has ``ok`` false and does not raise,
so the harness can stop the sequence and still unload.

A paired arm names a companion. The harness calls ``start_session`` on that
companion's adapter, runs the measured step, then calls ``stop_session``.
The companion keeps running between those two calls. ``stop_session`` ends
that load. It is not ``unload``.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol


class AdapterError(Exception):
    """The adapter refused the step before measuring it."""


class NativeBinaryRejected(AdapterError):
    """The device refused the compiled native binary at execute time.

    Raised only at the execute site, after this run's compile succeeded, with
    the Level Zero signature ``ZE_RESULT_ERROR_INVALID_NATIVE_BINARY``
    (``0x7800000f``). ``compile_s`` carries that compile's time, so the
    harness's capability row can record that the compile finished. Nothing is
    retried.
    """

    def __init__(self, message: str, *, compile_s: float | None = None) -> None:
        super().__init__(message)
        self.compile_s = compile_s


COMPILER_MESSAGE_MARKER = "[NPU_VCL] Compiler returned msg"
COMPILE_REJECTED_CODE = "0x78000007"


def compile_rejected_error(error: str | None) -> bool:
    """The NPU compiler refused the graph (ruling on #36).

    Both markers must be present: the compiler's own message line and the Level
    Zero code ``0x78000007`` (``ZE_RESULT_ERROR_INVALID_NULL_POINTER``). Either
    alone is some other failure and stays an abort.
    """

    if not isinstance(error, str) or not error:
        return False
    return COMPILER_MESSAGE_MARKER in error and COMPILE_REJECTED_CODE in error.lower()


class CompileRejected(AdapterError):
    """The NPU compiler refused the graph, raised inside ``compile_model``.

    Raised only at the compile site, with the signature in
    :func:`compile_rejected_error`. ``failed_s`` is the time from the start of
    the build to the failure: for embed that is the compile; for generate it
    also includes the pipeline load. The harness reads the subtype to tell a compile
    refusal from an execute failure. Nothing is retried.
    """

    def __init__(self, message: str, *, failed_s: float | None = None) -> None:
        super().__init__(message)
        self.failed_s = failed_s


class BuildRefused(AdapterError):
    """The build guard refused a pipeline or model build for lack of memory headroom.

    Raised before the build starts, so nothing was compiled. The harness marks the
    block's unfinished arms not run (reason ``memory headroom``); it is not a
    capability outcome and never reaches the routing table.
    """


class CompileTracker:
    """Which native compile is running, and since when.

    A native ``compile_model`` cannot be interrupted from Python, so the
    harness watches it from another thread. An adapter brackets each compile
    with :meth:`compiling`; the watcher reads :meth:`active`. Thread-safe. A
    compile that never returns stays ``active`` until the process ends.
    """

    def __init__(self, clock=None) -> None:
        self.clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._active: tuple[str, float] | None = None

    @contextmanager
    def compiling(self, label: str):
        with self._lock:
            self._active = (label, self.clock())
        try:
            yield
        finally:
            with self._lock:
                self._active = None

    def active(self) -> tuple[str, float] | None:
        """``(label, started)`` of the running compile, or ``None``."""

        with self._lock:
            return self._active


@dataclass(frozen=True)
class StepRequest:
    step_id: str
    task: str
    model: str
    device: str
    engine: str
    settings: dict


@dataclass(frozen=True)
class StepResult:
    """One measured step.

    ``latency_ms`` is the step latency and does not include compile time.
    ``compile_ms`` and ``first_call_ms`` are reported on a cold step, separate
    from ``latency_ms``. ``throughput`` for a generate step is server-reported
    decode (``eval_count / eval_duration``), not wall clock. ``prompt_tok_s``
    is ``prompt_eval_count / prompt_eval_duration``. ``output`` is the measured
    vectors or the generation. ``resolved`` carries values learned at run time,
    such as ``num_ctx`` and ``quant``. ``sensors`` may include ``gpu_temp_c``,
    ``sm_clock_mhz``, ``power_w``, ``shared_mib`` and ``dedicated_mib``.
    """

    ok: bool
    latency_ms: float | None = None
    throughput: float | None = None
    throughput_unit: str | None = None
    error: str | None = None
    compile_ms: float | None = None
    first_call_ms: float | None = None
    prompt_tok_s: float | None = None
    resolved: dict | None = None
    output: object | None = None
    sensors: dict | None = None


class Adapter(Protocol):
    def run(self, request: StepRequest) -> StepResult:
        """Measure one step. A failed measurement returns ``ok`` false."""

    def unload(self, model: str, device: str | None = None) -> None:
        """Drop the model. A real Ollama unload uses keep_alive 0.

        ``device`` selects the endpoint. An adapter that serves more than one
        device must not unload through a stored default.
        """
