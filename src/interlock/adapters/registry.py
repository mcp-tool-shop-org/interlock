"""Which adapter serves an engine name.

OpenVINO is imported only here, and only when a step asks for it. A missing
module is a refusal, not a fall back onto Ollama or the CPU.
"""

from __future__ import annotations

import importlib

from interlock.adapters.base import Adapter, AdapterError
from interlock.adapters.offrig_calibrate import OffrigCalibrateAdapter
from interlock.adapters.ollama import OllamaAdapter


def for_engine(engine: str, *, transport=None, executor=None) -> Adapter:
    if engine == "ollama":
        return OllamaAdapter(transport=transport)
    if engine == "offrig-calibrate":
        return OffrigCalibrateAdapter(executor=executor)
    if engine == "openvino":
        return _openvino()
    raise AdapterError(f"unknown engine {engine}")


def _openvino() -> Adapter:
    try:
        module = importlib.import_module("interlock.adapters.openvino")
    except ImportError as exc:
        raise AdapterError(
            "OpenVINO adapter is not in this tree. The Intel lane owns "
            "interlock.adapters.openvino. The harness does not fall back."
        ) from exc
    factory = getattr(module, "adapter", None)
    if not callable(factory):
        raise AdapterError("interlock.adapters.openvino must expose adapter()")
    return factory()
