"""Shared test setup.

A run built without explicit ``watch=`` hooks would start the real watcher
thread, which samples this machine's RAM and calls ``os._exit`` at the stop
line. A test must never end the test process because the machine running it
happens to be busy, so the runner's default hooks are replaced here by ones
that start no thread and raise instead of exiting. Tests of the watcher pass
their own hooks and are unaffected.
"""

from __future__ import annotations

import pytest

import interlock.harness.runner as runner
from interlock.harness.watch import WatchHooks


def _no_device_state(uses_5090: bool) -> dict:
    return {"cpu_percent": "unrecorded", "cpu_freq_mhz": "unrecorded", "ram_pct": "unrecorded"}


def _refuse_exit(code: int) -> None:
    raise AssertionError(f"a test reached the watcher's process exit (code {code})")


@pytest.fixture(autouse=True)
def _no_live_watcher(monkeypatch):
    monkeypatch.setattr(
        runner,
        "WatchHooks",
        lambda *args, **kwargs: WatchHooks(*args, **{"start_thread": False, "exit_fn": _refuse_exit, "state_sampler": _no_device_state, **kwargs}),
    )
