import os
from pathlib import Path

import pytest

from switchyard.guards import (
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


def test_heartbeat_uses_mtime_and_refuses_a_stale_or_missing_file(tmp_path: Path):
    path = tmp_path / "_watchdog_HEARTBEAT"
    path.write_text("do not read this", encoding="utf-8")
    now = 1_000_000.0
    os.utime(path, (now - HEARTBEAT_MAX_AGE_S, now - HEARTBEAT_MAX_AGE_S))
    age = require_heartbeat(path, now)
    assert age == pytest.approx(HEARTBEAT_MAX_AGE_S)

    with pytest.raises(GuardError, match="61s"):
        require_heartbeat(path, now + 1)
    with pytest.raises(GuardError, match="missing"):
        require_heartbeat(tmp_path / "absent", now)


def test_card_must_be_empty():
    require_card_empty([])
    with pytest.raises(GuardError, match="1 model"):
        require_card_empty([{"name": "qwen3:8b"}])


def test_assess_step_refuses_the_wrong_flag_and_unknown_devices():
    granted = assess_step(device="5090", grant="the grant")
    assert granted["grant"] == "the grant"
    session = assess_step(device="NPU", npu_session="switchyard E1")
    assert session["npu_session"] == "switchyard E1"
    heavy = assess_step(device="CPU", cpu_heavy=True, cpu_window="quiet")
    assert heavy["cpu_window"] == "quiet"
    with pytest.raises(GuardError, match="--grant"):
        assess_step(device="5090")
    with pytest.raises(GuardError, match="--npu-session"):
        assess_step(device="iGPU", grant="not a session")
    with pytest.raises(GuardError, match="unknown device"):
        assess_step(device="GPU.1", grant="no")
