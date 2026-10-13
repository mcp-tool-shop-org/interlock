"""E5 spill gates measure capacity, and the predictor learns from measured loads.

Fixtures are the committed e5-2026-10-13b receipt. Nothing here starts a device.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from interlock.adapters.ollama import OllamaAdapter
from interlock.cli import main
from interlock.harness.e5 import PREDICTED_OVER, SHARED_GATE_FROM_MIB, Observation, is_spill, search
from interlock.harness.fill import _predict_mib, wait_until_stable
from interlock.harness.runner import RunnerError, dry_run, parse_models, resolve_models, run_spec
from interlock.harness.spec import load_spec
from tests.test_fill_card_live import _Transport, _ctxs, _fresh, _model, _run, _seams, _spec, _bind
from tests.test_harness import E4, _resident

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "receipts" / "e5-2026-10-13b.jsonl"
RECEIPT_C = ROOT / "receipts" / "e5-2026-10-13c.jsonl"
# Above the recorded granite peak and below the shared gate, so a new load
# does not invert the measured slope.
FALLBACK_PEAK = 28000


def _rows() -> dict[str, dict]:
    found = {}
    for line in RECEIPT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        found[row["model"]["repo_or_tag"]] = row
    return found


def _resident_peaks(row: dict) -> dict[int, int]:
    peaks = {}
    for call in row["settings"]["search_calls"]:
        if call[0] == "load" and call[5] == "resident":
            peaks[call[1]] = call[2]
    return peaks


def _rejudge(calls: list) -> int:
    """The highest recorded confirm that still passes the new gates.

    Confirms are walked upward. Shared growth compares with the floor confirm
    and counts only at the capacity gate. The prompt rate compares with the
    previous confirm that passed.
    """

    confirms = [call for call in calls if call[0] == "confirm" and isinstance(call[1], int)]
    confirms.sort(key=lambda call: call[1])
    floor = confirms[0]

    def observation(call) -> Observation:
        return Observation(True, call[2], shared_mib=call[3], prompt_tok_s=call[4])

    floor_obs = observation(floor)
    best = floor[1]
    previous = floor_obs
    for call in confirms[1:]:
        current = observation(call)
        if is_spill(
            current,
            floor_obs,
            shared_growth_mib=256,
            prompt_ratio=0.5,
            prompt_reference=previous,
        ):
            break
        best = call[1]
        previous = current
    return best


def test_llama_recorded_calls_confirm_129024():
    row = _rows()["llama3.1:8b"]
    calls = row["settings"]["search_calls"]
    confirms = [call for call in calls if call[0] == "confirm"]
    assert max(call[2] for call in confirms) == 24229
    assert max(call[2] for call in confirms) < SHARED_GATE_FROM_MIB
    assert _rejudge(calls) == 129024


def test_gemma_slope_loads_16384(tmp_path: Path):
    recorded = _rows()["gemma4:31b"]
    settings = recorded["settings"]
    assert {"ctx": 16384, "mib": 38922} in settings["one_window_bound"]
    transport = _Transport(
        None,
        trained=16384,
        tag_bytes=settings["weights_mib"] * 1024 * 1024,
        idle_mib=settings["idle_baseline_mib"],
        peaks=_resident_peaks(recorded),
        fallback_peak=FALLBACK_PEAK,
    )
    seams = _seams(transport)
    _repo, rows = _run(tmp_path, transport, seams=seams, models=[_model("gemma4:31b")], max_ctx=16384)
    row = rows[0]
    calls = row["settings"]["search_calls"]
    assert ["load", 16384, FALLBACK_PEAK, 1000, 100, "resident"] in calls
    assert {"ctx": 16384, "mib": 38922} in row["settings"]["one_window_bound"]
    slopes = [item for item in row["settings"]["slope_predictions"] if item["ctx"] == 16384]
    assert slopes
    assert slopes[-1]["mib"] < 30500
    assert slopes[-1]["mib"] != 38922
    assert row["settings"]["max_resident_ctx"] == 16384
    assert all(item["shared_gate_applied"] is False for item in row["settings"]["window_gates"])
    floor = row["settings"]["window_gates"][0]
    assert floor["ctx"] == 2048
    assert floor["prompt_neighbour_ctx"] is None
    assert any(item["prompt_neighbour_ctx"] == 2048 for item in row["settings"]["window_gates"])


def test_granite_slope_loads_32768(tmp_path: Path):
    recorded = _rows()["granite4.1:30b"]
    settings = recorded["settings"]
    peaks = _resident_peaks(recorded)
    assert _predict_mib([(28672, peaks[28672]), (30720, peaks[30720])], 32768) == 27788
    transport = _Transport(
        None,
        trained=32768,
        tag_bytes=settings["weights_mib"] * 1024 * 1024,
        idle_mib=settings["idle_baseline_mib"],
        peaks=peaks,
        fallback_peak=FALLBACK_PEAK,
    )
    seams = _seams(transport)
    _repo, rows = _run(tmp_path, transport, seams=seams, models=[_model("granite4.1:30b")], max_ctx=32768)
    row = rows[0]
    calls = row["settings"]["search_calls"]
    assert 32768 in _ctxs(transport)
    assert ["load", 32768, FALLBACK_PEAK, 1000, 100, "resident"] in calls
    slope = next(item for item in row["settings"]["slope_predictions"] if item["ctx"] == 32768)
    assert slope["mib"] == 27788
    assert slope["left_ctx"] == 28672
    assert slope["right_ctx"] == 30720
    assert slope["rise_mib"] == peaks[30720] - peaks[28672]
    assert {"ctx": 32768, "mib": 38498} in row["settings"]["one_window_bound"]
    assert row["settings"]["max_resident_ctx"] == 32768


def test_shared_gate_applies_at_28452_and_spills_on_300():
    floor = _resident(1000, shared=1000, prompt=100)
    assert is_spill(
        _resident(SHARED_GATE_FROM_MIB, shared=1300, prompt=100),
        floor,
        shared_growth_mib=256,
        prompt_ratio=0.5,
    ) is True
    assert is_spill(
        _resident(SHARED_GATE_FROM_MIB - 1, shared=1300, prompt=100),
        floor,
        shared_growth_mib=256,
        prompt_ratio=0.5,
    ) is False
    assert is_spill(
        _resident(SHARED_GATE_FROM_MIB, shared=1256, prompt=100),
        floor,
        shared_growth_mib=256,
        prompt_ratio=0.5,
    ) is False
    missing = Observation(True, 1000, shared_mib=None, prompt_tok_s=100)
    assert is_spill(missing, floor, shared_growth_mib=256, prompt_ratio=0.5) is True

    def high(ctx, _kind):
        if ctx == 2048:
            return floor
        return _resident(SHARED_GATE_FROM_MIB, shared=1300, prompt=100)

    found = search(default_ctx=4096, probe=high, abort_mib=30500, max_ctx=4096)
    assert found.max_ctx == 2048
    assert any(call[0] == "confirm" and call[1] == 4096 and call[5] == "spill" for call in found.calls)
    assert found.gates[-1]["shared_gate_applied"] is True
    assert found.gates[0]["prompt_neighbour_ctx"] is None

    def low(ctx, _kind):
        if ctx == 2048:
            return floor
        return _resident(SHARED_GATE_FROM_MIB - 1, shared=1300, prompt=100)

    held = search(default_ctx=4096, probe=low, abort_mib=30500, max_ctx=4096)
    assert held.max_ctx == 4096
    assert held.gates[-1]["shared_gate_applied"] is False


def test_a_neighbour_prompt_cliff_spills():
    def cliff(ctx, _kind):
        prompt = 100 if ctx <= 4096 else 49
        return _resident(20000, prompt=prompt)

    found = search(default_ctx=8192, probe=cliff, abort_mib=30500, max_ctx=8192)
    assert found.max_ctx == 4096
    assert any(call[0] == "confirm" and call[1] == 6144 and call[5] == "spill" for call in found.calls)
    gate = next(item for item in found.gates if item["ctx"] == 6144)
    assert gate["prompt_neighbour_ctx"] == 4096
    assert gate["prompt_neighbour_tok_s"] == 100
    assert gate["shared_gate_applied"] is False

    def level(ctx, _kind):
        prompt = 100 if ctx <= 4096 else 50
        return _resident(20000, prompt=prompt)

    held = search(default_ctx=8192, probe=level, abort_mib=30500, max_ctx=8192)
    assert held.max_ctx == 8192


def test_predicted_over_records_limit_predicted(tmp_path: Path):
    weights = 1000 * 1024 * 1024
    transport = _Transport(None, busy_mib=28000, tag_bytes=weights, trained=16384, abort_at=16384)
    seams = _seams(transport)
    _repo, rows = _run(tmp_path, transport, seams=seams, max_ctx=16384)
    assert rows[0]["settings"]["limit"] == "predicted"
    assert rows[0]["settings"]["max_resident_ctx"] == 2048
    assert 4096 in rows[0]["settings"]["predicted_over"]
    assert 4096 not in _ctxs(transport)


def test_the_three_valid_models_calls_confirm_the_same_caps():
    rows = _rows()
    for name, cap in (("mistral-small:24b", 32768), ("qwen3:14b", 40960), ("qwen3:8b", 40960)):
        row = rows[name]
        calls = row["settings"]["search_calls"]
        assert row["settings"]["max_resident_ctx"] == cap
        assert row["settings"]["limit"] == "max_ctx"
        assert row["settings"].get("predicted_over") is None
        assert _rejudge(calls) == cap
        assert max(call[2] for call in calls if call[0] == "confirm") < SHARED_GATE_FROM_MIB


def test_models_subset_is_validated_and_recorded(tmp_path: Path):
    with pytest.raises(RunnerError, match="at least one"):
        parse_models("")
    with pytest.raises(RunnerError, match="at least one"):
        parse_models(",")
    repo = tmp_path / "repo"
    roster = [_model("gemma4:31b"), _model("granite4.1:30b"), _model("llama3.1:8b")]
    path = _bind(repo, _spec(roster, max_ctx=4096))
    loaded = load_spec(path)
    with pytest.raises(RunnerError, match="unknown model"):
        resolve_models(loaded, ["no-such"])
    with pytest.raises(RunnerError, match="more than once"):
        resolve_models(loaded, ["llama3.1:8b", "llama3.1:8b"])
    with pytest.raises(RunnerError, match="at least one"):
        resolve_models(loaded, [])
    _chosen, recorded = resolve_models(loaded, ["llama3.1:8b", "gemma4:31b"])
    assert recorded == ["gemma4:31b", "llama3.1:8b"]
    text = dry_run(path, repo, models=["llama3.1:8b", "gemma4:31b"])
    assert "models: 3" in text
    assert "models measured: gemma4:31b, llama3.1:8b" in text
    assert text.strip().endswith("devices not started (--dry-run)")
    with pytest.raises(RunnerError, match="fill-card"):
        dry_run(E4, ROOT, models=["qwen3:8b"])
    assert main(["run", str(path), "--dry-run", "--root", str(repo), "--models", "no-such"]) == 1
    assert main(["run", str(path), "--dry-run", "--root", str(repo), "--models", "llama3.1:8b,llama3.1:8b"]) == 1
    assert main(["run", str(path), "--dry-run", "--root", str(repo), "--models", "llama3.1:8b"]) == 0

    transport = _Transport(None, trained=4096)
    seams = _seams(transport)
    rows = run_spec(
        path,
        root=repo,
        grant="g",
        adapters={"ollama": OllamaAdapter(transport=transport)},
        heartbeat=_fresh(tmp_path),
        date="2026-10-13",
        run_id="subset-1",
        seams=seams,
        models=["llama3.1:8b", "gemma4:31b"],
    )
    assert [row["model"]["repo_or_tag"] for row in rows] == ["gemma4:31b", "llama3.1:8b"]
    assert rows[0]["settings"]["models"] == ["gemma4:31b", "llama3.1:8b"]
    assert rows[1]["settings"]["models"] == ["gemma4:31b", "llama3.1:8b"]

    whole = _Transport(None, trained=4096)
    _whole_repo, whole_rows = _run(tmp_path, whole, seams=_seams(whole), folder="whole", max_ctx=4096)
    assert "models" not in whole_rows[0]["settings"]


def _receipt_c() -> dict[str, dict]:
    found = {}
    for line in RECEIPT_C.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        found[row["model"]["repo_or_tag"]] = row
    return found


def _replay(row: dict):
    """Walk ``search`` through one recorded trace. The probe refuses a different call."""

    recorded = row["settings"]["search_calls"]
    cursor = {"i": 0}

    def probe(ctx, kind):
        call = recorded[cursor["i"]]
        cursor["i"] += 1
        assert (kind, ctx) == (call[0], call[1])
        reason = call[5]
        peak, shared, prompt = call[2], call[3], call[4]
        if reason == "predicted_over":
            return Observation(False, peak, error=PREDICTED_OVER)
        if reason == "card memory reached abort_mib":
            return Observation(False, peak, error=reason, shared_mib=shared, prompt_tok_s=prompt)
        return Observation(True, peak, shared_mib=shared, prompt_tok_s=prompt)

    num = row["settings"]["num_ctx"]
    found = search(default_ctx=num, probe=probe, abort_mib=30500, grain=2048, max_ctx=num)
    assert cursor["i"] == len(recorded)
    return found


def test_a_measured_confirm_abort_is_margin_and_a_predicted_stop_stays_predicted():
    rows = _receipt_c()
    gemma = _replay(rows["gemma4:31b"])
    assert gemma.limit == "margin"
    assert gemma.max_ctx == 86016
    granite = _replay(rows["granite4.1:30b"])
    assert granite.limit == "predicted"
    assert granite.max_ctx == 43008


def test_granite_4096_confirm_would_have_waited():
    calls = {
        (call[0], call[1]): call[2]
        for call in _receipt_c()["granite4.1:30b"]["settings"]["search_calls"]
    }
    assert calls[("confirm", 4096)] == 27329
    assert calls[("confirm", 2048)] == 20366
    assert calls[("confirm", 6144)] == 21655
    assert calls[("load", 43008)] == 30233
    assert calls[("confirm", 4096)] - calls[("confirm", 6144)] > 512
    # Within a model the search keeps the model resident, so the waiter is the resident floor.
    # 27329 is still moving; the settled reading is the later neighbour, not that peak.
    script = [26000, 24000, 22000, 21655]
    index = {"i": 0}

    def read():
        i = index["i"]
        index["i"] += 1
        return script[min(i, len(script) - 1)]

    clock = {"t": 0.0}
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        clock["t"] += float(seconds)

    found, reason = wait_until_stable(
        read,
        sleep,
        lambda: clock["t"],
        tolerance_mib=512,
        timeout_s=30,
        first=27329,
    )
    assert reason is None
    assert slept
    assert found == 21655
    assert found != 27329
