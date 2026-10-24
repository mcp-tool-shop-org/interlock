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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

NPU = "NPU"
CPU = "CPU"
IGPU = "iGPU"
CARD_5090 = "5090"

RIG_DEVICES = (NPU, IGPU, CPU, CARD_5090)
REMOTE_PROVIDERS = ("colab",)
COLAB_CARDS = ("A100-40", "A100-80", "L4", "T4", "G4")
REMOTE_DEVICES = tuple(f"colab:{card}" for card in COLAB_CARDS)

_BARE_REFUSAL = frozenset({"AUTO", "HETERO", "MULTI"})
_GPU_ID = re.compile(r"^GPU(?:\.(\d+))?$", re.IGNORECASE)
_VENDOR = re.compile(r"vendor=0x([0-9a-fA-F]+)", re.IGNORECASE)
_HASHED_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_ZONE_POOL = re.compile(
    r"^(?:(?:us|europe|asia|australia|southamerica|northamerica|me|africa)-[a-z0-9]+-[a-z0-9]+"
    r"|colab-(?:pool|zone|free|pro|payg)(?:-[a-z0-9]+)*)$"
)
_ACCOUNT_PREFIX = re.compile(
    r"^(?:account|acct|user|uid|owner|email|github|hf|huggingface)[:_-]",
    re.IGNORECASE,
)
_NUMERIC_VERSION = re.compile(r"^[0-9]+(?:\.[0-9]+)+$")



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


def reject_unhashed_account_text(value: object, field_name: str, *, allow_hashed: bool = False) -> str:
    """Refuse plaintext account identifiers, repo handles, emails, or home paths on a remote_device field."""

    if not isinstance(value, str) or not value.strip():
        raise DeviceRefused(f"{field_name} must be a non-empty string")
    token = value.strip()
    if allow_hashed and _HASHED_ID.fullmatch(token):
        return token
    if "@" in token or "/" in token or "\\" in token or _ACCOUNT_PREFIX.search(token):
        raise DeviceRefused(
            f"{field_name} {token!r} looks like an unhashed account id or path; "
            "hash with sha256:<64-hex> or omit"
        )
    return token


def validate_remote_pool(pool: object) -> str:
    """Validate that ``pool`` is a zone or pool identifier, never an unhashed account id."""

    if not isinstance(pool, str) or not pool.strip():
        raise DeviceRefused("remote_device.pool must be a non-empty zone string")
    token = reject_unhashed_account_text(pool, "remote_device.pool", allow_hashed=True)
    if _HASHED_ID.fullmatch(token):
        return token
    if not _ZONE_POOL.fullmatch(token):
        raise DeviceRefused(
            f"remote_device.pool {token!r} looks like an unhashed account id or path; "
            "pool must be a zone (e.g. us-central1-a) or sha256:<64-hex>"
        )
    return token


@dataclass(frozen=True)
class RemoteDeviceIdentity:
    provider: str
    card: str
    vram_mib: int
    driver_version: str
    cuda_version: str
    pool: str

    def __post_init__(self) -> None:
        reject_unhashed_account_text(self.provider, "remote_device.provider")
        reject_unhashed_account_text(self.card, "remote_device.card")
        if isinstance(self.vram_mib, str):
            reject_unhashed_account_text(self.vram_mib, "remote_device.vram_mib")
        reject_unhashed_account_text(self.driver_version, "remote_device.driver_version")
        reject_unhashed_account_text(self.cuda_version, "remote_device.cuda_version")
        validate_remote_pool(self.pool)
        if self.provider not in REMOTE_PROVIDERS:
            raise DeviceRefused(
                f"unsupported remote provider {self.provider!r}; expected one of {REMOTE_PROVIDERS}"
            )
        if self.card not in COLAB_CARDS:
            raise DeviceRefused(
                f"unsupported colab card {self.card!r}; expected one of {COLAB_CARDS}"
            )
        if isinstance(self.vram_mib, bool) or not isinstance(self.vram_mib, int) or self.vram_mib <= 0:
            raise DeviceRefused("remote_device.vram_mib must be a positive integer")
        if not _NUMERIC_VERSION.fullmatch(self.driver_version):
            raise DeviceRefused(
                f"remote_device.driver_version {self.driver_version!r} must be a numeric version string"
            )
        if not _NUMERIC_VERSION.fullmatch(self.cuda_version):
            raise DeviceRefused(
                f"remote_device.cuda_version {self.cuda_version!r} must be a numeric version string"
            )

    @property
    def logical_device(self) -> str:
        return f"{self.provider}:{self.card}"

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "card": self.card,
            "vram_mib": self.vram_mib,
            "driver_version": self.driver_version,
            "cuda_version": self.cuda_version,
            "pool": self.pool,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> RemoteDeviceIdentity:
        if not isinstance(raw, Mapping):
            raise DeviceRefused("remote_device must be a mapping")
        expected = {"provider", "card", "vram_mib", "driver_version", "cuda_version", "pool"}
        missing = sorted(expected - set(raw.keys()))
        if missing:
            raise DeviceRefused(f"remote_device missing required fields: {', '.join(missing)}")
        extra = sorted(set(raw.keys()) - expected)
        if extra:
            if any("account" in key.lower() or "user" in key.lower() for key in extra):
                raise DeviceRefused(f"remote_device refuses unhashed account id field: {', '.join(extra)}")
            raise DeviceRefused(f"remote_device has unexpected fields: {', '.join(extra)}")
        return cls(
            provider=raw["provider"],  # type: ignore[arg-type]
            card=raw["card"],  # type: ignore[arg-type]
            vram_mib=raw["vram_mib"],  # type: ignore[arg-type]
            driver_version=raw["driver_version"],  # type: ignore[arg-type]
            cuda_version=raw["cuda_version"],  # type: ignore[arg-type]
            pool=raw["pool"],  # type: ignore[arg-type]
        )


def parse_remote_device(device: str) -> tuple[str, str]:
    """Parse ``colab:<card>`` into ``(provider, card)``."""

    if not isinstance(device, str) or ":" not in device:
        raise DeviceRefused(f"not a remote device identifier: {device!r}")
    provider, card = device.split(":", 1)
    if provider not in REMOTE_PROVIDERS:
        raise DeviceRefused(f"unsupported remote provider {provider!r} in {device!r}")
    if card not in COLAB_CARDS:
        raise DeviceRefused(f"unsupported colab card {card!r} in {device!r}")
    return provider, card


def is_rig_device(device: object) -> bool:
    return isinstance(device, str) and device in RIG_DEVICES


def is_colab_device(device: object) -> bool:
    return isinstance(device, str) and device in REMOTE_DEVICES


def is_remote_device(device: object) -> bool:
    if not isinstance(device, str):
        return False
    return device in REMOTE_DEVICES or device.startswith("runpod:")


def device_pool_kind(device: object) -> str:
    if is_rig_device(device):
        return "rig"
    if is_remote_device(device):
        return "remote"
    raise DeviceRefused(f"unknown device pool for {device!r}")


def matches_device_filter(device: object, pattern: str | None) -> bool:
    """Match an exact device or a ``<provider>:*`` wildcard prefix."""

    if pattern is None or pattern == "":
        return True
    if not isinstance(device, str):
        return False
    if pattern.endswith(":*"):
        prefix = pattern[:-1]
        return device.startswith(prefix)
    return device == pattern


def partition_rows_by_pool(rows: Iterable[Mapping[str, object]]) -> dict[str, list[Mapping[str, object]]]:
    """Separate routing rows into disjoint ``rig`` and ``remote`` pools."""

    pools: dict[str, list[Mapping[str, object]]] = {"rig": [], "remote": []}
    for row in rows:
        kind = device_pool_kind(row.get("device"))
        pools[kind].append(row)
    return pools


def assert_single_pool(rows: Iterable[Mapping[str, object]]) -> str:
    """Ensure a sequence of rows belongs to a single pool; remote rows are never pooled with rig rows."""

    pools = partition_rows_by_pool(rows)
    if pools["rig"] and pools["remote"]:
        raise DeviceRefused("remote rows are never pooled with rig rows")
    if pools["remote"]:
        return "remote"
    return "rig"

