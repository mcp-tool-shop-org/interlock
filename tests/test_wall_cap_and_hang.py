"""Ladder 1.0.8: a per-cell wall cap, a deadline on every VRAM sample (the T4 r4 hang), phase markers, the host-RAM
wording, and the not-for-import identity of a rig check of the harness.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from interlock.devices import DeviceRefused, RemoteDeviceIdentity
from interlock.harness import linux_cuda
from interlock.harness.linux_cuda import (
    CALIBRATION_TIMEOUT_S,
    CELL_WALL_S,
    LinuxShapeFailure,
    measure_linux_ollama_generate,
    run_linux_generate_ladder,
)
from interlock.importers.common import ImportError_
from interlock.importers.remote import import_remote_window
from interlock.receipt import ReceiptError, validate_row
from interlock.router.policy import Decision, Refusal, Request, decide
from tests.fake_ollama import FakeOllama

ROOT = Path(__file__).resolve().parents[1]
IDENT = RemoteDeviceIdentity(
    provider="colab", card="T4", vram_mib=15360, driver_version="550.54.15", cuda_version="12.4", pool="us-central1-a"
)
RIG = RemoteDeviceIdentity.rig("RTX 5090", 32607, "581.29", "13.0")


def _measure(server: FakeOllama, target: int, concurrency: int = 1, repeats: int = 2, sample=None) -> dict:
    return measure_linux_ollama_generate(
        server,
        sample or (lambda: (1000, 15360)),
        model="m",
        prompt_tokens=target,
        concurrency=concurrency,
        repeats=repeats,
        clock=server.clock,
    )


def _ladder(server: FakeOllama, lengths, concurrencies=(1,), identity=IDENT, sample=None) -> list[dict]:
    return run_linux_generate_ladder(
        spec_id="C2-GENERATE", spec_rel="experiments/c2-generate/spec.yaml", run_id="t4-r5", date="2026-10-26",
        remote_device=identity, model="phi3.5:3.8b", revision_or_digest="d" * 64,
        measure_shape=lambda length, conc: _measure(server, length, conc, sample=sample),
        prompt_lengths=lengths, concurrencies=concurrencies, quant="Q4_K_M", licence="mit",
    )


# -- 1. the per-cell wall cap ------------------------------------------------------------------------------------------


def test_the_cap_and_the_calibration_timeout_are_named_with_their_defaults():
    assert CELL_WALL_S == 1800
    assert CALIBRATION_TIMEOUT_S == 300  # was 900; bounded from r3's slowest verify (25.1 s at p16384) and the rig's
    assert linux_cuda.CELL_DEADLINE_S == 2 * CELL_WALL_S


def test_a_cell_whose_projection_is_over_the_cap_is_refused_before_any_timed_pass():
    # c4, 250 s verify (inside the calibration timeout): 3 x (ceil(4 x 250.148) + 60) = 3183 s > 1800 s
    server = FakeOllama(overhead_s=250.0)
    with pytest.raises(LinuxShapeFailure) as caught:
        _measure(server, 512, concurrency=4, repeats=2)
    assert caught.value.failure_class == "timeout"
    assert caught.value.reason.startswith("projected 3183 s > cell wall 1800 s; not run")
    assert all(call["num_predict"] == 1 for call in server.all_calls)  # calibration calls only: no timed pass was sent


def test_the_projection_is_an_expected_wall_time_without_a_safety_factor():
    # A 295 s verify at c4: 3 passes x (ceil(4 x 295.148) + 60) = 3723 s. The request timeout keeps its 3x (clamped to
    # 900 s); the projection is what the cell is expected to take, and CELL_DEADLINE_S is the real bound.
    server = FakeOllama(overhead_s=295.0)
    with pytest.raises(LinuxShapeFailure) as caught:
        _measure(server, 512, concurrency=4, repeats=2)
    assert linux_cuda.request_timeout_s(4, 295.148) == 900
    assert caught.value.reason.startswith("projected 3723 s")


def test_an_a100_80_shaped_c8_cell_with_a_30_second_verify_measures():
    # A resident 30B-class c8 x 32k cell on an A100-80 or G4: a 30 s verify, passes of about 5 minutes. v4's 3x factor
    # projected 3 x (ceil(3 x 8 x 30) + 60) = 2340 s and refused it; the expected time is 3 x (240 + 60) = 900 s.
    server = FakeOllama(overhead_s=29.85)
    result = _measure(server, 512, concurrency=8, repeats=2)
    assert result["cell_wall_s"] == CELL_WALL_S
    assert result["request_timeout_s"] == linux_cuda.request_timeout_s(8, 30.0)  # the per-request 3x is unchanged
    assert result["request_timeout_s"] > 700


def test_a_t4_shaped_slow_cell_is_still_refused():
    # r3's T4 evidence: a c8 cell whose single verify call takes 100 s (Go-side overhead on 2 vCPUs) cannot finish four
    # passes in 30 minutes: 4 x (ceil(8 x 100.148) + 60) = 3448 s.
    with pytest.raises(LinuxShapeFailure) as caught:
        _measure(FakeOllama(overhead_s=100.0), 512, concurrency=8, repeats=3)
    assert caught.value.failure_class == "timeout" and caught.value.reason.startswith("projected 3448 s")


def test_a_cell_under_the_cap_measures_and_the_record_carries_the_cap():
    server = FakeOllama(overhead_s=100.0)  # 3 x (100 + 60) = 480 s <= 1800 s
    result = _measure(server, 512, repeats=2)
    assert result["cell_wall_s"] == CELL_WALL_S
    (row,) = _ladder(FakeOllama(overhead_s=100.0), (512,))
    assert row["settings"]["cell_wall_s"] == 1800
    validate_row(row)


def test_a_refused_cell_is_a_timeout_row_that_also_records_the_cap_and_its_budget():
    (row,) = _ladder(FakeOllama(overhead_s=250.0), (512,), (4,))
    assert row["outcome"] == "timeout" and row["notes"].startswith("timeout: projected")
    assert row["settings"]["cell_wall_s"] == 1800 and row["settings"]["request_timeout_s"] == 900


def test_calibration_calls_carry_the_lower_timeout():
    server = FakeOllama()
    _measure(server, 512)
    calibration = {call["timeout"] for call in server.all_calls if call["num_predict"] == 1}
    assert calibration == {300}


# -- 5. the VRAM sampler has a deadline; the cap is an outer deadline over the whole cell ----------------------


@pytest.fixture
def quick_sampler(monkeypatch):
    monkeypatch.setattr(linux_cuda, "SAMPLER_TIMEOUT_S", 0.3)


def test_a_sampler_that_never_answers_makes_the_cell_unmeasured_within_the_deadline(quick_sampler):
    started = time.monotonic()
    with pytest.raises(LinuxShapeFailure) as caught:
        _measure(FakeOllama(), 512, sample=lambda: threading.Event().wait())
    assert caught.value.failure_class == "unmeasured"
    assert "VRAM sampler did not answer within 0.3 s" in caught.value.reason
    assert time.monotonic() - started < 5


def test_a_sampler_that_answers_late_but_inside_the_deadline_still_measures(monkeypatch):
    monkeypatch.setattr(linux_cuda, "SAMPLER_TIMEOUT_S", 3.0)

    def late() -> tuple[int, int]:
        time.sleep(0.2)
        return (1000, 15360)

    assert _measure(FakeOllama(), 512, sample=late)["size"] > 0


def test_a_sampler_that_hangs_mid_cell_is_also_unmeasured(quick_sampler):
    calls = {"n": 0}

    def sample() -> tuple[int, int]:
        calls["n"] += 1
        if calls["n"] == 2:  # the first call (the cell's start) answers; the one after the first pass does not
            threading.Event().wait()
        return (1000, 15360)

    with pytest.raises(LinuxShapeFailure) as caught:
        _measure(FakeOllama(), 512, sample=sample)
    assert caught.value.failure_class == "unmeasured" and "did not answer" in caught.value.reason


def test_a_failing_sampler_is_still_a_sampler_error_not_a_hang(quick_sampler):
    def broken() -> tuple[int, int]:
        raise RuntimeError("nvidia-smi went away")

    with pytest.raises(LinuxShapeFailure) as caught:
        _measure(FakeOllama(), 512, sample=broken)
    assert caught.value.failure_class == "sampler_error" and "went away" in caught.value.reason


def test_a_call_that_blocks_forever_ends_as_a_row_at_the_outer_deadline(monkeypatch):
    monkeypatch.setattr(linux_cuda, "CELL_DEADLINE_S", 0.3)
    hung = threading.Event()

    def blocked(length: int, concurrency: int) -> dict:
        hung.wait()
        return {}

    rows = run_linux_generate_ladder(
        spec_id="C2-GENERATE", spec_rel="experiments/c2-generate/spec.yaml", run_id="t4-r5", date="2026-10-26",
        remote_device=IDENT, model="phi3.5:3.8b", revision_or_digest="d" * 64, measure_shape=blocked,
        prompt_lengths=(512, 1024), concurrencies=(1,), quant="Q4_K_M", licence="mit",
    )
    assert [row["outcome"] for row in rows] == ["unmeasured", "unmeasured"]  # both cells ended; the ladder did not hang
    assert "outer deadline" in rows[0]["notes"]


def test_the_harness_names_each_phase_on_stderr(capsys):
    _measure(FakeOllama(), 512)
    err = capsys.readouterr().err
    for phase in ("cell p512 c1: start, sampling VRAM", "probe 9001 sent", "probe 9001 answered; checking residency",
                  "probe 9002 sent", "verify 1 sent", "calibrated in", "pass 1 of 3 (warmup) sent",
                  "pass 3 answered in", "end, measured"):
        assert phase in err, phase


def _spec_repo(tmp_path: Path) -> Path:
    spec_rel = "experiments/c2-generate/spec.yaml"
    dest = tmp_path / spec_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / spec_rel, dest)
    for cmd in (
        ["git", "init"], ["git", "config", "user.email", "dev@example.com"], ["git", "config", "user.name", "dev"],
        ["git", "add", "."], ["git", "commit", "-m", "spec"],
    ):
        subprocess.run(cmd, cwd=tmp_path, check=True, capture_output=True)  # noqa: S603, S607
    return dest


def test_the_t4_r4_hang_after_a_host_ram_kill_ends_as_a_row_the_next_cell_starts_and_the_run_closes(
    tmp_path, quick_sampler, monkeypatch
):
    monkeypatch.setattr(linux_cuda, "_nvidia_smi_query", lambda timeout_s: "1000, 15360" + chr(10))  # fallback
    from interlock.harness.runner import run_spec

    spec = _spec_repo(tmp_path)
    server = FakeOllama(vram_mib=6000.0)  # c4 at 16k and 32k need more host RAM than the VM has: SIGKILL on load
    seen = {"kills": 0, "blocked": 0}

    def sampler() -> tuple[int, int]:
        if server.kills > seen["kills"]:  # the first sample after a kill never returns (NVML after the OOM killer)
            seen["kills"] = server.kills
            seen["blocked"] += 1
            threading.Event().wait()
        return (1000, 15360)

    outcome: dict[str, Any] = {}

    def run() -> None:
        outcome["rows"] = run_spec(
            spec, root=tmp_path, run_id="c2-hang", date="2026-10-26",
            python_path=str(tmp_path.parent / "external-python"),
            models=["qwen3:8b"],
            seams={"server": server, "remote_device": IDENT, "sample_vram_mib": sampler},
        )

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(120)
    assert not worker.is_alive(), "the run hung: a blocked VRAM sample was not given a deadline"
    rows = outcome["rows"]
    assert server.kills >= 1 and seen["blocked"] >= 1
    hung = [r for r in rows if "VRAM sampler did not answer" in r.get("notes", "")]
    assert hung and all(r["outcome"] == "unmeasured" for r in hung)
    first = rows.index(hung[0])
    assert any(r.get("outcome") is None for r in rows[first + 1 :]), "no cell after the hang was measured"
    assert any(r.get("outcome") == "oom" and r["notes"].startswith("oom: host RAM") for r in rows)
    record = json.loads((tmp_path / "receipts" / "c2-hang.run.json").read_text(encoding="utf-8"))
    assert record["closed"] is True


# -- 3. the host-RAM wording ----------------------------------------------------------------------------------


def test_a_total_given_in_mib_has_no_rounding_margin_clause():
    log = (
        'time=2026-10-26T14:00:00.000Z level=INFO source=sched.go:658 msg="system memory" total="12975 MiB"\n'
        "load_tensors: loading model tensors\nload_tensors: offloaded 15/33 layers to GPU\n"
        "load_tensors:   CPU_Mapped model buffer size =   70.00 MiB\n"
        "llama_kv_cache:        CPU KV buffer size = 20000.00 MiB\n"
    )

    class Killed(FakeOllama):
        def post(self, path, payload, timeout=None):
            self.log_text = log
            raise RuntimeError("llama-server process has terminated: signal: killed")

    with pytest.raises(LinuxShapeFailure) as caught:
        _measure(Killed(), 2048)
    reason = caught.value.reason
    assert caught.value.failure_class == "oom"
    assert "the VM has 12975 MiB (Ollama system memory)" in reason
    assert "0.0 MiB" not in reason and "margin" not in reason and "0.1 GiB" not in reason


# -- 4. a rig-check identity: not for import -------------------------------------------------------------------


def test_a_rig_identity_has_a_free_text_card_and_a_rig_device_name():
    assert RIG.logical_device == "rig:RTX5090" and RIG.not_for_import is True
    assert IDENT.logical_device == "colab:T4" and IDENT.not_for_import is False
    with pytest.raises(DeviceRefused):
        RemoteDeviceIdentity(provider="rig", card="RTX 5090", vram_mib=32607, driver_version="581.29",
                             cuda_version="13.0", pool="us-central1-a")  # a rig check's pool is rig-local
    with pytest.raises(DeviceRefused):
        RemoteDeviceIdentity.rig("not a card!!", 32607, "581.29", "13.0")


def test_every_row_built_with_a_rig_identity_is_marked_not_for_import_and_validates():
    rows = _ladder(FakeOllama(vram_mib=8000.0), (512, 8192), (1, 4), identity=RIG)
    assert {row["outcome"] for row in rows if "outcome" in row} >= {"oom"} and any("outcome" not in row for row in rows)
    for row in rows:
        assert row["device"] == "rig:RTX5090" and row["not_for_import"] is True
        assert row["remote_device"]["provider"] == "rig"
        validate_row(row)


def test_a_rig_row_without_the_mark_is_refused_by_the_schema():
    (row,) = _ladder(FakeOllama(), (512,), identity=RIG)
    del row["not_for_import"]
    with pytest.raises(ReceiptError):
        validate_row(row)


def test_colab_rows_are_unchanged_and_carry_no_mark():
    (row,) = _ladder(FakeOllama(), (512,))
    assert "not_for_import" not in row and row["device"] == "colab:T4"
    validate_row(row)


def test_interlock_import_refuses_a_rig_row_naming_the_field():
    (row,) = _ladder(FakeOllama(), (512,), identity=RIG)
    with pytest.raises(ImportError_) as caught:
        import_remote_window((json.dumps(row) + "\n").encode("utf-8"), source_path="receipts/rig.jsonl")
    assert "not_for_import" in str(caught.value) and "rig check" in str(caught.value)
    envelope = {"run_id": "r", "date": "2026-10-26", "remote_device": RIG.to_dict(), "records": [{"task": "generate"}]}
    with pytest.raises(ImportError_) as caught:
        import_remote_window(json.dumps(envelope).encode("utf-8"), source_path="receipts/rig.json")
    assert "not_for_import" in str(caught.value)


def test_the_router_excludes_a_rig_row_with_its_own_code_and_never_ranks_it():
    (rig_row,) = _ladder(FakeOllama(), (512,), identity=RIG)
    (colab_row,) = _ladder(FakeOllama(), (512,))
    colab_row["id"] += "-colab"
    request = Request(task="generate", as_of=date(2026, 10, 26), model="phi3.5:3.8b", allow_remote=True)
    result = decide([rig_row, colab_row], request)
    assert isinstance(result, Decision) and result.document["receipts"] == [colab_row["id"]]
    assert {"id": rig_row["id"], "code": "rig_check"} in result.document["excluded"]
    only_rig = decide([rig_row], request)
    assert isinstance(only_rig, Refusal) and {"id": rig_row["id"], "code": "rig_check"} in only_rig.document["excluded"]
