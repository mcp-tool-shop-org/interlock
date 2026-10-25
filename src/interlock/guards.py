"""Guards that stop a step before it touches a device.

The heartbeat check uses the file's modified time and does not open it.
Opening the watchdog heartbeat collides with the watchdog's own writes.
"""

from __future__ import annotations

from pathlib import Path

from interlock.devices import CARD_5090, CPU, IGPU, NPU

HEARTBEAT_MAX_AGE_S = 60
HEARTBEAT_ENV = "WATCHDOG_HEARTBEAT"


class WatchdogOptOut:
    """The explicit ``--no-watchdog`` choice: this run uses no VRAM watchdog, so there is no heartbeat to read.

    An unset heartbeat is ``None`` and refuses. Only this marker, which only the flag creates, skips the guard.
    """

    def __repr__(self) -> str:
        return "NO_WATCHDOG"


NO_WATCHDOG = WatchdogOptOut()
NO_WATCHDOG_NOTE = "no watchdog: run with --no-watchdog; the heartbeat guard was not applied"


class GuardError(Exception):
    """A step is refused. The caller does not start the next step."""


def require_grant(grant: str | None, *, uses_5090: bool) -> str | None:
    if not uses_5090:
        return grant.strip() if isinstance(grant, str) and grant.strip() else None
    if not isinstance(grant, str) or not grant.strip():
        raise GuardError("REFUSED: a 5090 step requires --grant")
    return grant.strip()


def require_npu_session(session: str | None, *, device: str) -> str | None:
    if device not in {NPU, IGPU}:
        return session.strip() if isinstance(session, str) and session.strip() else None
    if not isinstance(session, str) or not session.strip():
        raise GuardError(f"REFUSED: a {device} step requires --npu-session")
    return session.strip()


def require_cpu_window(window: str | None, *, required: bool) -> str | None:
    if not required:
        return window.strip() if isinstance(window, str) and window.strip() else None
    if not isinstance(window, str) or not window.strip():
        raise GuardError("REFUSED: a CPU-heavy step requires --cpu-window")
    return window.strip()


def heartbeat_age_s(path: Path, now: float) -> float:
    """Age of the heartbeat from its modified time. The file is not opened."""

    try:
        modified = path.stat().st_mtime
    except OSError:
        return float("inf")
    return now - modified


def require_heartbeat(
    path: Path | WatchdogOptOut | None, now: float, *, max_age_s: float = HEARTBEAT_MAX_AGE_S
) -> float:
    """Refuse a stale or missing heartbeat. With no heartbeat configured it refuses too, naming how to set one.

    Only the ``--no-watchdog`` marker skips the check; the run records that it did.
    """

    if path is NO_WATCHDOG:
        return 0.0
    if path is None:
        raise GuardError(
            f"REFUSED: no watchdog heartbeat is configured; set {HEARTBEAT_ENV} (or [paths] heartbeat in "
            "interlock.toml) to the watchdog's heartbeat file, or pass --no-watchdog if you run no watchdog"
        )
    age = heartbeat_age_s(path, now)
    if age > max_age_s:
        shown = "missing" if age == float("inf") else f"{age:.0f}s"
        raise GuardError(f"REFUSED: watchdog heartbeat is {shown} old")
    return age


def require_card_empty(models: list) -> None:
    count = len(models)
    if count != 0:
        raise GuardError(f"REFUSED: {count} model(s) still on the card")


def assess_step(
    *,
    device: str,
    grant: str | None = None,
    npu_session: str | None = None,
    cpu_window: str | None = None,
    cpu_heavy: bool = False,
) -> dict[str, str | None]:
    """Apply the flag guards for one step. Heartbeat and card are separate."""

    if device == CARD_5090:
        return {
            "grant": require_grant(grant, uses_5090=True),
            "npu_session": None,
            "cpu_window": None,
        }
    if device in {NPU, IGPU}:
        return {
            "grant": None,
            "npu_session": require_npu_session(npu_session, device=device),
            "cpu_window": None,
        }
    if device == CPU:
        return {
            "grant": None,
            "npu_session": None,
            "cpu_window": require_cpu_window(cpu_window, required=cpu_heavy),
        }
    raise GuardError(f"REFUSED: unknown device {device!r}")
