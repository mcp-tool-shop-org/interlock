"""The adapter interface.

An adapter runs one measured step and unloads the model it loaded. The harness
talks only to this shape. Kimi's OpenVINO adapter plugs in by adding
``switchyard.adapters.openvino`` with a function named ``adapter`` that returns
an object with ``run`` and ``unload``. The harness imports that module only
when a step names the OpenVINO engine. It does not fall back to another device.

``run`` returns a result. A failed step has ``ok`` false and does not raise,
so the harness can stop the sequence and still unload.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class AdapterError(Exception):
    """The adapter refused the step before measuring it."""


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
    ok: bool
    latency_ms: float | None = None
    throughput: float | None = None
    throughput_unit: str | None = None
    error: str | None = None


class Adapter(Protocol):
    def run(self, request: StepRequest) -> StepResult:
        """Measure one step. A failed measurement returns ``ok`` false."""

    def unload(self, model: str) -> None:
        """Drop the model. A real Ollama unload uses keep_alive 0."""
