import os
from pathlib import Path

import pytest

from interlock.guards import (
    HEARTBEAT_MAX_AGE_S,
    GuardError,
    assess_step,
    require_card_empty,
    require_cpu_window,
    require_grant,
    require_heartbeat,
    require_npu_session,
)


def test_grant_is_required_only_for_the_5090():
    assert require_grant(None, uses_5090=False) is None
    assert require_grant("  chain  ", uses_5090=True) == "chain"
    with pytest.raises(GuardError, match="--grant"):
        require_grant(None, uses_5090=True)
    with pytest.raises(GuardError, match="--grant"):
        require_grant("   ", uses_5090=True)


def test_npu_session_covers_npu_and_igpu_only():
    assert require_npu_session(" ledger ", device="NPU") == "ledger"
    assert require_npu_session("ledger", device="iGPU") == "ledger"
    assert require_npu_session(None, device="CPU") is None
    with pytest.raises(GuardError, match="iGPU"):
        require_npu_session("", device="iGPU")


def test_cpu_window_is_required_only_when_the_leg_is_heavy():
    assert require_cpu_window(None, required=False) is None
    assert require_cpu_window(" after the chain ", required=True) == "after the chain"
    with pytest.raises(GuardError, match="--cpu-window"):
        require_cpu_window(None, required=True)


def test_an_unset_heartbeat_reads_as_infinitely_stale_instead_of_crashing():
    from interlock.guards import heartbeat_age_s

    assert heartbeat_age_s(None, 1_000.0) == float("inf")


def test_an_unreadable_heartbeat_is_refused_with_how_to_set_one_and_never_as_an_infinite_age(tmp_path: Path):
    wanted = (
        "REFUSED: no watchdog heartbeat: set WATCHDOG_HEARTBEAT (or [paths] heartbeat) to the watchdog's "
        "heartbeat file, or pass --no-watchdog if you run no watchdog"
    )
    for unreadable in (None, tmp_path / "absent"):
        with pytest.raises(GuardError) as caught:
            require_heartbeat(unreadable, 1_000.0)
        assert str(caught.value) == wanted
        assert "inf" not in str(caught.value)
    path = tmp_path / "hb"
    path.write_text("pulse", encoding="utf-8")
    os.utime(path, (0, 0))
    with pytest.raises(GuardError, match=r"REFUSED: watchdog heartbeat is 1000s old"):
        require_heartbeat(path, 1_000.0)


def test_heartbeat_uses_mtime_and_refuses_a_stale_or_missing_file(tmp_path: Path):
    path = tmp_path / "_watchdog_HEARTBEAT"
    path.write_text("do not read this", encoding="utf-8")
    now = 1_000_000.0
    os.utime(path, (now - HEARTBEAT_MAX_AGE_S, now - HEARTBEAT_MAX_AGE_S))
    age = require_heartbeat(path, now)
    assert age == pytest.approx(HEARTBEAT_MAX_AGE_S)

    with pytest.raises(GuardError, match="61s"):
        require_heartbeat(path, now + 1)
    with pytest.raises(GuardError, match="no watchdog heartbeat"):
        require_heartbeat(tmp_path / "absent", now)


def test_card_must_be_empty():
    require_card_empty([])
    with pytest.raises(GuardError, match="1 model"):
        require_card_empty([{"name": "qwen3:8b"}])


def test_assess_step_refuses_the_wrong_flag_and_unknown_devices():
    granted = assess_step(device="5090", grant="the grant")
    assert granted["grant"] == "the grant"
    session = assess_step(device="NPU", npu_session="interlock E1")
    assert session["npu_session"] == "interlock E1"
    heavy = assess_step(device="CPU", cpu_heavy=True, cpu_window="quiet")
    assert heavy["cpu_window"] == "quiet"
    with pytest.raises(GuardError, match="--grant"):
        assess_step(device="5090")
    with pytest.raises(GuardError, match="--npu-session"):
        assess_step(device="iGPU", grant="not a session")
    with pytest.raises(GuardError, match="unknown device"):
        assess_step(device="GPU.1", grant="no")
