"""OpenVINO device allow-list.

The 5090 is never selected here. On this rig OpenVINO lists it as ``GPU.1``.
That index is a topology pin: it is refused even when the name or the vendor
looks like an Intel GPU. A topology change leaves the iGPU unreachable. It
never makes the 5090 selectable. The vendor check is the second protection.

Work that belongs on the 5090 goes through Ollama on 127.0.0.1:11434, and
only inside a Publisher grant.

When the enumeration carries ``device_type`` and ``device_architecture``, the
iGPU is the GPU with vendor ``0x8086`` and ``DEVICE_TYPE`` ``INTEGRATED``.
``FULL_DEVICE_NAME`` is recorded on the result and is not the decision.
Names drift across driver releases. Vendor ids do not. An enumeration with
neither property (a recorded list, not a live ``Core``) still requires the
full name to contain "Intel".

The harness passes a live ``Core()`` enumeration at resolve time. It does not
pass a list deserialized from a receipt or a spec. Those go stale across
driver updates.
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
_VENDOR = re.compile(r"vendor=0x([0-9a-fA-F]+)", re.IGNORECASE)


class DeviceRefused(Exception):
    """The requested device is not an allowed, explicit device."""


@dataclass(frozen=True)
class ResolvedDevice:
    kind: str
    ov_id: str
    full_name: str


@dataclass(frozen=True)
class _Listed:
    full_name: str
    device_type: str | None
    architecture: str | None


def _as_text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DeviceRefused("a device entry needs a string id and full name")
    return value.strip()


def resolve_device(requested: str, available: list[dict]) -> ResolvedDevice:
    """Resolve one explicit device against an OpenVINO enumeration.

    ``available`` entries carry ``id`` and ``full_name``. A live ``Core()``
    enumeration also carries ``device_type`` and ``device_architecture``.
    Pass that live list. Do not pass a list saved in a receipt or a spec.

    ``requested`` is ``NPU``, ``CPU``, ``iGPU``, or an OpenVINO id such as
    ``GPU.0``. ``AUTO``, ``HETERO``, ``MULTI``, a bare ``GPU``, ``GPU.1`` and
    any GPU that is not vendor ``0x8086`` and ``INTEGRATED`` are refused.
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


def _optional(entry: dict, *keys: str) -> str | None:
    wanted = {key.lower() for key in keys}
    for key, raw in entry.items():
        if str(key).lower() not in wanted:
            continue
        if raw is None:
            return None
        if not isinstance(raw, str) or not raw.strip():
            raise DeviceRefused("device_type and device_architecture must be strings when set")
        return raw.strip()
    return None


def _catalog(available: list[dict]) -> dict[str, _Listed]:
    catalog: dict[str, _Listed] = {}
    for entry in available:
        if not isinstance(entry, dict):
            raise DeviceRefused("a device entry must be a mapping with id and full_name")
        ov_id = _as_text(entry.get("id"))
        full_name = _as_text(entry.get("full_name"))
        listed = _Listed(
            full_name=full_name,
            device_type=_optional(entry, "device_type", "DEVICE_TYPE"),
            architecture=_optional(entry, "device_architecture", "DEVICE_ARCHITECTURE"),
        )
        key = ov_id.upper()
        if key in catalog:
            raise DeviceRefused(f"duplicate OpenVINO id {ov_id}")
        catalog[key] = listed
    return catalog


def _require_id(catalog: dict[str, _Listed], ov_id: str, kind: str) -> ResolvedDevice:
    listed = catalog.get(ov_id.upper())
    if listed is None:
        raise DeviceRefused(f"{ov_id} is not in the device list")
    return ResolvedDevice(kind=kind, ov_id=ov_id.upper(), full_name=listed.full_name)


def _gpu_index(token: str) -> str | None:
    match = _GPU_ID.fullmatch(token.strip())
    if match is None:
        return None
    return match.group(1)


def _resolve_gpu_id(token: str, index: str | None, catalog: dict[str, _Listed]) -> ResolvedDevice:
    if index is None:
        raise DeviceRefused("bare GPU is refused; name NPU, CPU, or the Intel iGPU")
    if index == "1":
        raise DeviceRefused(
            "GPU.1 is refused; on this rig that OpenVINO id is the 5090. "
            "The index pin stays if the name or vendor changes"
        )
    ov_id = f"GPU.{index}".upper()
    listed = catalog.get(ov_id)
    if listed is None:
        raise DeviceRefused(f"{ov_id} is not in the device list")
    _require_igpu(ov_id, listed)
    return ResolvedDevice(kind=IGPU, ov_id=ov_id, full_name=listed.full_name)


def _intel_igpu(catalog: dict[str, _Listed]) -> ResolvedDevice:
    found: list[ResolvedDevice] = []
    for ov_id, listed in catalog.items():
        index = _gpu_index(ov_id)
        if index is None or index == "1":
            continue
        if _igpu_verdict(listed) == "no":
            continue
        _require_igpu(ov_id, listed)
        found.append(ResolvedDevice(kind=IGPU, ov_id=ov_id, full_name=listed.full_name))
    if not found:
        raise DeviceRefused("no Intel iGPU in the device list")
    if len(found) > 1:
        names = ", ".join(item.ov_id for item in found)
        raise DeviceRefused(f"more than one Intel GPU ({names}); name the id")
    return found[0]


def _require_igpu(ov_id: str, listed: _Listed) -> None:
    verdict = _igpu_verdict(listed)
    if verdict == "incomplete":
        raise DeviceRefused(f"{ov_id} enumeration needs both device_type and device_architecture")
    if verdict == "no":
        raise DeviceRefused(f"{ov_id} is not an Intel iGPU ({listed.full_name})")


def _igpu_verdict(listed: _Listed) -> str:
    """``yes`` when vendor and type say iGPU, ``name`` for a property-free Intel name, else ``no`` or ``incomplete``."""

    typed = listed.device_type
    architecture = listed.architecture
    if typed is None and architecture is None:
        return "name" if "intel" in listed.full_name.lower() else "no"
    if typed is None or architecture is None:
        return "incomplete"
    if typed.upper() != "INTEGRATED":
        return "no"
    if _vendor(architecture) != "8086":
        return "no"
    return "yes"


def _vendor(architecture: str) -> str | None:
    match = _VENDOR.search(architecture)
    if match is None:
        return None
    return match.group(1).lower()
