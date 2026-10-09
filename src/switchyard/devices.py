"""OpenVINO device allow-list.

The 5090 is never selected here. OpenVINO lists it as ``GPU.1``. Work that
belongs on that card goes through Ollama on 127.0.0.1:11434, and only inside
a Publisher grant. The iGPU is the GPU whose full name contains "Intel".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

NPU = "NPU"
CPU = "CPU"
IGPU = "iGPU"
CARD_5090 = "5090"

_BARE_REFUSAL = frozenset({"AUTO", "HETERO", "MULTI"})
_GPU_ID = re.compile(r"^GPU(?:\.(\d+))?$", re.IGNORECASE)


class DeviceRefused(Exception):
    """The requested device is not an allowed, explicit device."""


@dataclass(frozen=True)
class ResolvedDevice:
    kind: str
    ov_id: str
    full_name: str


def _as_text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DeviceRefused("a device entry needs a string id and full name")
    return value.strip()


def resolve_device(requested: str, available: list[dict]) -> ResolvedDevice:
    """Resolve one explicit device against an OpenVINO enumeration.

    ``available`` entries are ``{"id": "GPU.0", "full_name": "Intel(R) Graphics"}``.
    ``requested`` is ``NPU``, ``CPU``, ``iGPU``, or an OpenVINO id such as ``GPU.0``.
    ``AUTO``, a bare ``GPU``, ``GPU.1`` and any non-Intel GPU are refused.
    Nothing here falls back to another device.
    """

    token = _as_text(requested)
    upper = token.upper()
    if upper in _BARE_REFUSAL or upper.startswith(("AUTO", "HETERO", "MULTI")):
        raise DeviceRefused(f"refused device selector {token!r}")
    if upper in {CARD_5090, "CUDA", "NVIDIA"}:
        raise DeviceRefused("the 5090 is not an OpenVINO device; use Ollama on 127.0.0.1:11434 inside a grant")

    catalog = _catalog(available)
    if upper in {NPU, CPU}:
        return _require_id(catalog, upper, upper)
    if token == IGPU or upper == "IGPU":
        return _intel_igpu(catalog)
    gpu_index = _gpu_index(token)
    if gpu_index is not None or upper == "GPU":
        return _resolve_gpu_id(token, gpu_index, catalog)
    raise DeviceRefused(f"unknown device {token!r}")


def _catalog(available: list[dict]) -> dict[str, str]:
    catalog: dict[str, str] = {}
    for entry in available:
        if not isinstance(entry, dict):
            raise DeviceRefused("a device entry must be a mapping with id and full_name")
        ov_id = _as_text(entry.get("id"))
        full_name = _as_text(entry.get("full_name"))
        key = ov_id.upper()
        if key in catalog:
            raise DeviceRefused(f"duplicate OpenVINO id {ov_id}")
        catalog[key] = full_name
    return catalog


def _require_id(catalog: dict[str, str], ov_id: str, kind: str) -> ResolvedDevice:
    full_name = catalog.get(ov_id.upper())
    if full_name is None:
        raise DeviceRefused(f"{ov_id} is not in the device list")
    return ResolvedDevice(kind=kind, ov_id=ov_id.upper(), full_name=full_name)


def _gpu_index(token: str) -> str | None:
    match = _GPU_ID.fullmatch(token.strip())
    if match is None:
        return None
    return match.group(1)


def _resolve_gpu_id(token: str, index: str | None, catalog: dict[str, str]) -> ResolvedDevice:
    if index is None:
        raise DeviceRefused("bare GPU is refused; name NPU, CPU, or the Intel iGPU")
    if index == "1":
        raise DeviceRefused("GPU.1 is refused; that OpenVINO id is the 5090")
    ov_id = f"GPU.{index}".upper()
    full_name = catalog.get(ov_id)
    if full_name is None:
        raise DeviceRefused(f"{ov_id} is not in the device list")
    if not _is_intel(full_name):
        raise DeviceRefused(f"{ov_id} is not an Intel GPU ({full_name})")
    return ResolvedDevice(kind=IGPU, ov_id=ov_id, full_name=full_name)


def _intel_igpu(catalog: dict[str, str]) -> ResolvedDevice:
    found: list[ResolvedDevice] = []
    for ov_id, full_name in catalog.items():
        index = _gpu_index(ov_id)
        if index is None or index == "1":
            continue
        if _is_intel(full_name):
            found.append(ResolvedDevice(kind=IGPU, ov_id=ov_id, full_name=full_name))
    if not found:
        raise DeviceRefused("no Intel iGPU in the device list")
    if len(found) > 1:
        names = ", ".join(item.ov_id for item in found)
        raise DeviceRefused(f"more than one Intel GPU ({names}); name the id")
    return found[0]


def _is_intel(full_name: str) -> bool:
    return "intel" in full_name.lower()
