"""Versions a receipt pins, read without compiling a model or opening a device.

OpenVINO and the Intel stack come from installed distribution metadata.
The NPU value is the Intel AI Boost driver version from the OS driver store.
The Intel "Shared GPU/NPU Memory Override" is read, read-only, from the Intel
adapter's registry key. A missing fact stays ``unrecorded``. Power plan,
Ollama, the NVIDIA driver, and the CUDA backend are not read here.
"""

from __future__ import annotations

import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

UNRECORDED = "unrecorded"

# Distributions the OpenVINO adapter actually imports. Names are the ones
# importlib.metadata resolves; hyphens and underscores are the same package.
_STACK = ("openvino", "openvino-genai", "openvino-tokenizers")

_DRIVER_CACHE: dict[str, str] = {}


def dist_version(name: str) -> str:
    """The installed distribution version, or ``unrecorded`` when it is absent."""

    try:
        found = version(name)
    except PackageNotFoundError:
        return UNRECORDED
    except Exception:
        return UNRECORDED
    text = str(found).strip()
    return text or UNRECORDED


def package_versions(names: tuple[str, ...] = _STACK) -> str:
    """``name==version`` for each installed stack package, sorted by name.

    One fact when nothing from the stack is installed: ``unrecorded``.
    """

    found: dict[str, str] = {}
    for name in names:
        version = dist_version(name)
        if version == UNRECORDED:
            continue
        found.setdefault(name.lower().replace("_", "-"), f"{name.lower().replace('_', '-')}=={version}")
    if not found:
        return UNRECORDED
    return " ".join(found[key] for key in sorted(found))


def npu_driver_version() -> str:
    """The Intel AI Boost driver version, or ``unrecorded``.

    This reads the OS driver store. It does not construct an OpenVINO device
    and it does not compile a model. Two different versions is not a fact
    this function will guess between.
    """

    cached = _DRIVER_CACHE.get("npu")
    if cached is not None:
        return cached
    value = _read_npu_driver()
    _DRIVER_CACHE["npu"] = value
    return value


def clear_driver_cache() -> None:
    """Drop the driver-store cache. Tests use this around a patched reader."""

    _DRIVER_CACHE.clear()


def driver_version_from_lines(lines: list[str]) -> str:
    """One driver version from query lines. Zero or several versions do not count."""

    versions = {line.strip() for line in lines if line.strip()}
    if len(versions) != 1:
        return UNRECORDED
    version = versions.pop()
    if any(char.isspace() for char in version):
        return UNRECORDED
    return version


def _read_npu_driver() -> str:
    if sys.platform != "win32":
        return UNRECORDED
    script = (
        "Get-CimInstance Win32_PnPSignedDriver | "
        "Where-Object { $_.DeviceName -like '*AI Boost*' } | "
        "Select-Object -ExpandProperty DriverVersion"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return UNRECORDED
    if completed.returncode != 0:
        return UNRECORDED
    return driver_version_from_lines(completed.stdout.splitlines())


_DISPLAY_CLASS = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
# Where the Intel Graphics Software override lands: a percent of system RAM the
# iGPU and NPU may pin. Found on the Robot rig as 57 (36.1 GB of 63 GB).
_OVERRIDE_SUBKEY = "GMM"
_OVERRIDE_VALUE = "GpuSystemMemoryPinninglimit"


def igpu_shared_memory_override(reader=None) -> str:
    """The Intel "Shared GPU/NPU Memory Override", as ``"<n>%"``, or ``unrecorded``.

    Read-only. ``reader`` is a test seam: a zero-argument callable returning the
    percent as a number or ``None``. Any failure to read is ``unrecorded``,
    never an error.
    """

    try:
        percent = (reader or _read_override_percent)()
    except Exception:
        return UNRECORDED
    if isinstance(percent, bool) or not isinstance(percent, (int, float)) or not 0 < percent <= 100:
        return UNRECORDED
    return f"{percent:g}%"


def _read_override_percent():
    if sys.platform != "win32":
        return None
    import winreg  # pragma: no cover - Windows only

    found = set()  # pragma: no cover
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _DISPLAY_CLASS) as cls:  # pragma: no cover
        index = 0
        while True:
            try:
                name = winreg.EnumKey(cls, index)
            except OSError:
                break
            index += 1
            if not name.isdigit():
                continue
            try:
                with winreg.OpenKey(cls, name) as adapter:
                    provider = str(winreg.QueryValueEx(adapter, "ProviderName")[0])
                    if "intel" not in provider.lower():
                        continue
                    with winreg.OpenKey(adapter, _OVERRIDE_SUBKEY) as gmm:
                        found.add(int(winreg.QueryValueEx(gmm, _OVERRIDE_VALUE)[0]))
            except (OSError, ValueError):
                continue
    return found.pop() if len(found) == 1 else None  # pragma: no cover
