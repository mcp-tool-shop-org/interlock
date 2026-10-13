"""E5's idle-card pin and the preflight that refuses before a load.

The counter fixtures are the harness's InstanceName=CookedValue rendering of
the idle sample from 2026-10-12. The 5090 LUID is the elided token from the
brief. Nothing here opens a socket, runs nvidia-smi, or reads a GPU counter.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from interlock.harness.e5 import (
    Observation,
    SearchResult,
    baseline_is_spilling,
    is_spill,
    floor_confirm_shared,
    search,
    shared_deltas,
)
from interlock.harness.fill import (
    ADAPTER_DEDICATED,
    ADAPTER_SHARED,
    INCONCLUSIVE,
    FillPreflightError,
    _apply_spill_baseline,
    _runner_pid,
    config_facts,
    driver_cuda,
    judge_preflight,
    listener_exe,
    listener_log,
    live_run,
    load_line_facts,
    log_redirect,
    match_adapter_instance,
    parse_counter_text,
    parse_mib,
    pin_failure,
    real_fill_seams,
    resolve_fill_paths,
    runner_pid,
)
from interlock.harness.spec import SpecError, load_spec, validate_spec

ROOT = Path(__file__).resolve().parents[1]
E5 = ROOT / "experiments" / "e5" / "spec.yaml"
E5_SMOKE = ROOT / "experiments" / "e5-smoke" / "spec.yaml"
FIXTURES = ROOT / "tests" / "fixtures" / "e5"
_GPU = "luid_0x00000000_0x000140e1_phys_0"
_SHA = "a" * 40
_GGML = "ab" * 32
_LOG = (
    'msg="server config" env="map[OLLAMA_FLASH_ATTENTION:true OLLAMA_KV_CACHE_TYPE:q8_0 '
    'OLLAMA_HOST:http://127.0.0.1:11434]"\n'
)


def _sections(name: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current = None
    for line in (FIXTURES / name).read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections[current] = []
            continue
        if current is not None:
            sections[current].append(line)
    return {key: "\n".join(lines) + "\n" for key, lines in sections.items()}


def _idle():
    parts = _sections("idle-card-pin.txt")
    nvidia = parse_mib(parts["nvidia-smi"])
    dedicated = parse_counter_text(parts["dedicated"])
    shared = parse_counter_text(parts["shared"])
    return nvidia, dedicated, shared


def _clear(monkeypatch) -> None:
    for name in (
        "INTERLOCK_OLLAMA_LOG",
        "INTERLOCK_OLLAMA_EXE",
        "SWITCHYARD_OLLAMA_LOG",
        "SWITCHYARD_OLLAMA_EXE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_exact_match_fails_on_the_idle_card_and_the_512_pin_picks_the_5090():
    nvidia, dedicated, shared = _idle()
    assert nvidia == 2490
    assert len(dedicated) == 4
    assert len(shared) == 4
    assert match_adapter_instance(dedicated, nvidia) is None
    assert match_adapter_instance(dedicated, nvidia, 512) == _GPU
    assert pin_failure(dedicated, nvidia, 512) is None
    header = (FIXTURES / "nvidia-smi-617.42.txt").read_text(encoding="utf-8")
    assert "CUDA UMD Version: 13.4" in header
    assert "KMD Version: 617.42" in header


def test_two_adapters_within_tolerance_refuse_and_never_pick():
    parts = _sections("two-adapters-within-tolerance.txt")
    nvidia = parse_mib(parts["nvidia-smi"])
    dedicated = parse_counter_text(parts["dedicated"])
    assert match_adapter_instance(dedicated, nvidia, 512) is None
    assert pin_failure(dedicated, nvidia, 512) == "several instances within adapter_match_tolerance_mib"


def test_zero_adapters_within_tolerance_refuse():
    _nvidia, dedicated, _shared = _idle()
    zeros = [(name, 0.0) for name, _cooked in dedicated]
    assert match_adapter_instance(zeros, 2490, 512) is None
    assert pin_failure(zeros, 2490, 512) == "zero instances within adapter_match_tolerance_mib"


def test_later_reads_follow_the_pinned_instance():
    parts = _sections("idle-card-pin.txt")
    state = {"dedicated": parts["dedicated"], "shared": parts["shared"]}

    def read_counter(counter):
        if counter == ADAPTER_DEDICATED:
            return state["dedicated"]
        if counter == ADAPTER_SHARED:
            return state["shared"]
        raise AssertionError(counter)

    def run(argv, **kwargs):
        raise AssertionError(argv)

    seams = real_fill_seams(object(), run=run, read_counter=read_counter)
    assert seams.shared_mib() is None
    assert seams.dedicated_mib() is None
    seams.bind_pin(_GPU)
    assert seams.shared_mib() == 74
    assert seams.dedicated_mib() == 2508
    mib = 1024 * 1024
    state["dedicated"] = (
        f"{_GPU}={4000 * mib}\n"
        f"luid_0x00000000_0x00011111_phys_0={2490 * mib}\n"
    )
    state["shared"] = (
        f"{_GPU}={80 * mib}\n"
        f"luid_0x00000000_0x00011111_phys_0={10 * mib}\n"
    )
    later = parse_counter_text(state["dedicated"])
    assert match_adapter_instance(later, 2490) == "luid_0x00000000_0x00011111_phys_0"
    assert seams.dedicated_mib() == 4000
    assert seams.shared_mib() == 80
    assert seams.adapter_reading() == (_GPU, 80, 4000)
    state["shared"] = "luid_0x00000000_0x00011111_phys_0=0\n"
    assert seams.adapter_reading() is None


def test_driver_cuda_accepts_both_headers_and_the_check_refuses_a_miss():
    umd = (FIXTURES / "nvidia-smi-617.42.txt").read_text(encoding="utf-8")
    assert "CUDA UMD Version: 13.4" in umd
    assert "KMD Version: 617.42" in umd
    assert driver_cuda(umd) == "driver 13.4"
    older = "NVIDIA-SMI 617.14    Driver Version: 617.14    CUDA Version: 13.2\n"
    assert driver_cuda(older) == "driver 13.2"
    assert driver_cuda("KMD Version: 617.42\n") is None
    nvidia, dedicated, shared = _idle()
    base = dict(
        nvidia_mib=nvidia,
        dedicated=dedicated,
        shared=shared,
        tolerance_mib=512,
        log_text=_LOG,
        ggml_sha=_GGML,
        ggml_reason=None,
        git={"git_sha": _SHA, "clean_tree": True},
        log_source="listener-parent",
        exe_source="listener",
    )
    record = judge_preflight(**base, driver_cuda_value=driver_cuda(umd))
    assert record["driver_cuda"] == "driver 13.4"
    assert record["instance"] == _GPU
    assert record["dedicated_mib"] == 2508
    assert record["shared_mib"] == 74
    assert record["log_source"] == "listener-parent"
    assert "\\" not in record["log_source"]
    with pytest.raises(FillPreflightError, match="driver CUDA did not parse"):
        judge_preflight(**base, driver_cuda_value=None)
    with pytest.raises(FillPreflightError, match="driver CUDA did not parse"):
        judge_preflight(**base, driver_cuda_value=driver_cuda("KMD Version: 617.42\n"))


def test_a_pin_with_no_number_and_a_listener_miss_are_refusals():
    assert pin_failure([], None, 512) == "nvidia-smi memory.used returned no number"
    assert pin_failure([("luid_a", -1.0)], 2490, 512) == "zero instances within adapter_match_tolerance_mib"
    assert match_adapter_instance([("luid_a", 1.0)], True) is None

    class Completed:
        def __init__(self, stdout, returncode=0):
            self.stdout = stdout
            self.returncode = returncode

    assert listener_exe(lambda *args, **kwargs: Completed(b'{"exe": "D:/ollama/ollama.exe"}')).endswith("ollama.exe")
    assert listener_exe(lambda *args, **kwargs: Completed('{"exe": "  "}')) is None
    assert listener_exe(lambda *args, **kwargs: Completed('{"exe": "D:/ollama/ollama.exe"}', 1)) is None

    def missing(*args, **kwargs):
        raise OSError("lookup failed")

    assert listener_exe(missing) is None


def test_the_log_path_comes_from_the_listener_parent_and_the_tray_is_never_used(tmp_path: Path, monkeypatch):
    _clear(monkeypatch)
    tray = tmp_path / "appdata" / "Ollama" / "server.log"
    tray.parent.mkdir(parents=True)
    tray.write_text("tray log\n", encoding="utf-8")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    served = tmp_path / "system-11434.log"
    served.write_text("server\n", encoding="utf-8")
    exe = tmp_path / "dist" / "ollama.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"exe")
    parent = f'cmd.exe /c ollama serve >> "{served}" 2>&1'
    calls = []

    def run(argv, **kwargs):
        calls.append(argv[-1] if argv else "")
        assert argv[0] == "powershell"
        script = argv[-1]
        if "CommandLine" in script:
            return parent
        if "ConvertTo-Json" in script:
            return json.dumps({"exe": str(exe)})
        raise AssertionError(script)

    paths = resolve_fill_paths(run)
    assert paths["log_source"] == "listener-parent"
    assert paths["exe_source"] == "listener"
    assert Path(paths["log_path"]) == served
    assert Path(paths["log_path"]) != tray
    assert paths["dll"].endswith("ggml-cuda.dll")
    assert "\\" not in paths["log_source"]
    assert "/" not in paths["log_source"]
    assert any("CommandLine" in script for script in calls)
    assert not any("LOCALAPPDATA" in script or "server.log" in script for script in calls)
    assert listener_exe(lambda *args, **kwargs: "not-json") is None

    monkeypatch.setenv("INTERLOCK_OLLAMA_LOG", str(tmp_path / "override.log"))

    def log_refused(argv, **kwargs):
        script = argv[-1] if argv else ""
        if "CommandLine" in script:
            raise AssertionError("parent lookup ran under the override")
        if "ConvertTo-Json" in script:
            return json.dumps({"exe": str(exe)})
        raise AssertionError(script)

    overridden_log = resolve_fill_paths(log_refused)
    assert overridden_log["log_source"] == "INTERLOCK_OLLAMA_LOG"
    assert Path(overridden_log["log_path"]) == tmp_path / "override.log"
    assert overridden_log["exe_source"] == "listener"

    monkeypatch.setenv("INTERLOCK_OLLAMA_EXE", str(tmp_path / "override.exe"))

    def refuse(*args, **kwargs):
        raise AssertionError("listener ran")

    overridden = resolve_fill_paths(refuse)
    assert overridden["log_source"] == "INTERLOCK_OLLAMA_LOG"
    assert overridden["exe_source"] == "INTERLOCK_OLLAMA_EXE"
    assert Path(overridden["exe"]) == tmp_path / "override.exe"
    assert "/" not in overridden["exe_source"]

    def unresolved(argv, **kwargs):
        script = argv[-1] if argv else ""
        if "CommandLine" in script:
            return "cmd.exe /c ollama serve"
        if "ConvertTo-Json" in script:
            return json.dumps({"exe": str(exe)})
        raise AssertionError(script)

    monkeypatch.delenv("INTERLOCK_OLLAMA_LOG", raising=False)
    monkeypatch.delenv("INTERLOCK_OLLAMA_EXE", raising=False)
    missing = resolve_fill_paths(unresolved)
    assert missing["log_path"] is None
    assert missing["log_source"] is None
    assert str(tray) != missing["log_path"]


def test_the_e5_spec_loads_and_rejects_a_wrong_tolerance():
    loaded = load_spec(E5)
    assert loaded["adapter_match_tolerance_mib"] == 512
    assert loaded["spill_shared_mib"] == 256
    assert loaded["adapter_match_tolerance_mib"] != loaded["spill_shared_mib"]
    text = E5.read_text(encoding="utf-8")
    assert text.count("Amendment, 2026-10-12") == 2
    assert "CUDA UMD Version" in text
    assert "adapter_match_tolerance_mib is 512" in text
    assert "INTERLOCK_OLLAMA_LOG" in text
    assert "never a fallback" in text
    assert "OLLAMA_HOST 127.0.0.1:11434" in text
    assert "first-load lines" in text
    assert "llama-server.exe" in text
    assert "%LOCALAPPDATA%" in text
    amendment = next(line for line in text.splitlines() if "after e5-2026-10-12c" in line)
    assert "never a fallback" in amendment
    assert "unset" in amendment
    with pytest.raises(SpecError, match="adapter_match_tolerance_mib"):
        validate_spec({**loaded, "adapter_match_tolerance_mib": 256})


def test_the_config_line_is_recorded_as_it_is_and_empty_is_unset():
    text = (FIXTURES / "server-config-11434.txt").read_text(encoding="utf-8")
    assert "OLLAMA_KV_CACHE_TYPE:" in text
    assert "OLLAMA_FLASH_ATTENTION:false" in text
    assert "OLLAMA_HOST:http://127.0.0.1:11434" in text
    assert "Users" not in text
    facts = config_facts(text)
    assert facts["reason"] is None
    assert facts["flash_attention"] == "false"
    assert facts["kv_cache_type"] == "unset"
    assert facts["host"] == "http://127.0.0.1:11434"
    nvidia, dedicated, shared = _idle()
    record = judge_preflight(
        nvidia_mib=nvidia,
        dedicated=dedicated,
        shared=shared,
        tolerance_mib=512,
        log_text=text,
        ggml_sha=_GGML,
        ggml_reason=None,
        git={"git_sha": _SHA, "clean_tree": True},
        driver_cuda_value="driver 13.4",
        log_source="listener-parent",
        exe_source="listener",
    )
    assert record["flash_attention"] == "false"
    assert record["kv_cache_type"] == "unset"
    bare = text.replace("OLLAMA_HOST:http://127.0.0.1:11434", "OLLAMA_HOST:127.0.0.1:11434")
    assert config_facts(bare)["host"] == "127.0.0.1:11434"
    with pytest.raises(FillPreflightError, match="OLLAMA_HOST"):
        judge_preflight(
            nvidia_mib=nvidia,
            dedicated=dedicated,
            shared=shared,
            tolerance_mib=512,
            log_text=text.replace("OLLAMA_HOST:http://127.0.0.1:11434", "OLLAMA_HOST:http://127.0.0.1:11490"),
            ggml_sha=_GGML,
            ggml_reason=None,
            git={"git_sha": _SHA, "clean_tree": True},
            driver_cuda_value="driver 13.4",
            log_source="listener-parent",
            exe_source="listener",
        )
    with pytest.raises(FillPreflightError, match="no server-config line"):
        judge_preflight(
            nvidia_mib=nvidia,
            dedicated=dedicated,
            shared=shared,
            tolerance_mib=512,
            log_text="ollama started\n",
            ggml_sha=_GGML,
            ggml_reason=None,
            git={"git_sha": _SHA, "clean_tree": True},
            driver_cuda_value="driver 13.4",
            log_source="listener-parent",
            exe_source="listener",
        )


def test_load_lines_parse_the_recorded_segment_and_a_missing_line_refuses():
    segment = (FIXTURES / "load-lines-11434.txt").read_text(encoding="utf-8")
    assert "Users" not in segment
    assert "--flash-attn auto" in segment
    assert "Flash Attention enabled" in segment
    assert "K (f16)" in segment and "V (f16)" in segment
    parsed = load_line_facts(segment)
    assert parsed["reason"] is None
    assert parsed["flash_attn"] == "auto"
    assert parsed["flash_attention"] == "enabled"
    assert parsed["kv_cache_type"] == "f16"
    assert parsed["cuda_host_model_mib"] == 333.84
    assert parsed["cuda_host_compute_mib"] == 18.01
    assert parsed["cuda_host_output_mib"] == 0.58
    assert parsed["cuda_host_sum_mib"] == 352.43
    without_host = "\n".join(line for line in segment.splitlines() if "CUDA_Host" not in line)
    refused = load_line_facts(without_host)
    assert refused["reason"] is not None
    assert "CUDA_Host model" in refused["reason"]
    assert "CUDA_Host compute" in refused["reason"]
    assert "CUDA_Host output" in refused["reason"]
    disabled = segment.replace("Flash Attention enabled", "Flash Attention disabled")
    assert load_line_facts(disabled)["flash_attention"] == "disabled"
    split = segment.replace("V (f16)", "V (q8_0)")
    assert load_line_facts(split)["kv_cache_type"] == "K f16, V q8_0"
    without_kv = "\n".join(line for line in segment.splitlines() if "llama_kv_cache" not in line)
    missing = load_line_facts(without_kv)
    assert missing["reason"] is not None
    assert "kv cache" in missing["reason"]
    assert load_line_facts("")["reason"] is not None
    assert "flash-attn" in load_line_facts("")["reason"]


def test_a_parent_redirect_parses_and_the_runner_is_llama_server():
    recorded = r"cmd.exe /c ollama serve >> E:\AI\ollama-cu134\system-11434.log 2>&1"
    assert log_redirect(recorded) == r"E:\AI\ollama-cu134\system-11434.log"
    assert log_redirect('cmd.exe /c ollama serve >> "E:\\logs\\server.log" 2>&1') == r"E:\logs\server.log"
    assert log_redirect("cmd.exe /c ollama serve") is None
    assert log_redirect(None) is None
    assert log_redirect(">> &1") is None

    def down(*_args, **_kwargs):
        raise OSError("lookup failed")

    assert listener_log(down) is None

    class Failed:
        returncode = 1
        stdout = recorded

    assert listener_log(lambda *_args, **_kwargs: Failed()) is None
    assert runner_pid(["42 7 llama-server.exe"]) == 42
    assert runner_pid(["42 7 LLAMA-SERVER.EXE"]) == 42
    assert runner_pid(["42 7 ollama.exe"]) is None
    assert runner_pid(["42 7 llama-server.exe", "43 7 llama-server.exe"]) is None
    assert runner_pid([]) is None


def test_the_live_default_runner_returns_a_childs_stdout(monkeypatch):
    # A string-returning fake hides this. The live default used to be a bare
    # subprocess.run, so stdout was None and both lookups resolved nothing.
    completed = live_run([sys.executable, "-c", "print('captured')"])
    assert completed.stdout.strip() == "captured"
    seams = real_fill_seams(object())
    assert inspect.getclosurevars(seams.runner_pid).nonlocals["run"] is live_run

    def captured(args, **kwargs):
        assert kwargs.get("capture_output") is True
        assert kwargs.get("text") is True
        script = args[-1]
        if "CommandLine" in script:
            text = r"cmd.exe /c ollama serve >> E:\AI\ollama-cu134\system-11434.log 2>&1"
        elif "ConvertTo-Json" in script:
            text = '{"exe": "D:\\\\ollama\\\\ollama.exe"}'
        else:
            text = "42 5 llama-server.exe"
        return subprocess.CompletedProcess(args, 0, stdout=text, stderr="")

    monkeypatch.setattr("interlock.harness.e4b.subprocess.run", captured)
    monkeypatch.setattr("interlock.harness.fill.subprocess.run", captured)
    assert listener_log(live_run) == r"E:\AI\ollama-cu134\system-11434.log"
    assert listener_exe(live_run) == r"D:\ollama\ollama.exe"
    assert _runner_pid(live_run) == 42


# Recorded on the e5 smoke, qwen3:8b, windows 2048 through 8192.
_SMOKE_SHARED = {2048: 505, 4096: 507, 6144: 531, 8192: 534}
_SMOKE_IDLE = 72
_SMOKE_PROMPT = 40.0


def _smoke_host() -> dict:
    parsed = load_line_facts((FIXTURES / "load-lines-11434.txt").read_text(encoding="utf-8"))
    return {
        "model": parsed["cuda_host_model_mib"],
        "compute": parsed["cuda_host_compute_mib"],
        "output": parsed["cuda_host_output_mib"],
        "sum": parsed["cuda_host_sum_mib"],
    }


def _window(shared: int, *, prompt: float = _SMOKE_PROMPT, peak: int = 1000) -> Observation:
    return Observation(
        True,
        peak,
        shared_mib=shared,
        prompt_tok_s=prompt,
        tok_s=10,
        ttft_ms=1,
        shared_baseline=_SMOKE_IDLE,
        baseline_taken=True,
    )


def _smoke_calls() -> list[tuple]:
    """The recorded e5-smoke calls: one load at 8192, then confirms down to 2048."""

    recorded = (
        ("load", 8192, "resident"),
        ("confirm", 8192, "confirmed"),
        ("confirm", 6144, "confirmed"),
        ("confirm", 4096, "confirmed"),
        ("confirm", 2048, "confirmed"),
    )
    return [
        (kind, ctx, 1000, _SMOKE_SHARED[ctx], _SMOKE_PROMPT, reason)
        for kind, ctx, reason in recorded
    ]


def test_the_recorded_smoke_deltas_confirm_against_the_floor():
    host = _smoke_host()
    assert host == {"model": 333.84, "compute": 18.01, "output": 0.58, "sum": 352.43}
    over = (_SMOKE_SHARED[2048] - _SMOKE_IDLE) - host["sum"]
    assert round(over, 1) == 80.6
    assert baseline_is_spilling(_SMOKE_IDLE, _SMOKE_SHARED[2048], host["sum"]) is False
    calls = _smoke_calls()
    assert floor_confirm_shared(calls) == (2048, 505, _SMOKE_PROMPT)
    assert [item["delta_mib"] for item in shared_deltas(calls, 505)] == [29, 29, 26, 2, 0]
    reference = _window(_SMOKE_SHARED[2048])
    for shared in _SMOKE_SHARED.values():
        assert is_spill(_window(shared), reference, shared_growth_mib=256, prompt_ratio=0.5) is False

    found = SearchResult(
        8192,
        "confirmed",
        calls,
        peak_mib=1000,
        tok_s=10,
        ttft_ms=1,
        prompt_tok_s=_SMOKE_PROMPT,
        shared_mib=_SMOKE_SHARED[8192],
    )
    fields = _apply_spill_baseline(found, host, _SMOKE_IDLE)
    assert fields["baseline_check"] == "ok"
    assert fields["idle_shared_mib"] == _SMOKE_IDLE
    assert fields["baseline_ctx"] == 2048
    assert fields["baseline_shared_mib"] == 505
    assert fields["baseline_prompt_tok_s"] == _SMOKE_PROMPT
    assert fields["cuda_host_model_mib"] == 333.84
    assert fields["cuda_host_compute_mib"] == 18.01
    assert fields["cuda_host_output_mib"] == 0.58
    assert fields["cuda_host_sum_mib"] == 352.43
    assert [item["delta_mib"] for item in fields["shared_delta_mib"]] == [29, 29, 26, 2, 0]
    assert found.max_ctx == 8192
    assert found.reason == "confirmed"


def test_a_model_resident_at_its_first_load_is_judged_against_the_floor():
    """The search starts at the cap. The cap's own load is not the reference.

    Load and confirm at 8192 read the same shared value, so a comparison with
    that load is a delta of 0. Three hundred MiB over the floor confirm still
    spills. A cap within 256 of the floor confirms, and baseline_ctx stays 2048.
    """

    def spilling(ctx, kind):
        shared = 805 if ctx == 8192 else 505
        peak = 28452 if kind == "confirm" and ctx == 8192 else 1000
        return _window(shared, peak=peak)

    found = search(default_ctx=8192, probe=spilling, abort_mib=30500, grain=2048, max_ctx=8192)
    assert found.calls[0] == ("load", 2048, 1000, 505, _SMOKE_PROMPT, "resident")
    assert found.calls[1] == ("confirm", 2048, 1000, 505, _SMOKE_PROMPT, "confirmed")
    assert ("load", 8192, 1000, 805, _SMOKE_PROMPT, "resident") in found.calls
    assert ("confirm", 8192, 28452, 805, _SMOKE_PROMPT, "spill") in found.calls
    assert found.max_ctx == 6144
    assert found.reason == "confirmed"
    host = _smoke_host()
    fields = _apply_spill_baseline(found, host, _SMOKE_IDLE)
    assert fields["baseline_ctx"] == 2048
    assert fields["baseline_shared_mib"] == 505
    cap = next(
        item for item in fields["shared_delta_mib"] if item["kind"] == "confirm" and item["ctx"] == 8192
    )
    assert cap["delta_mib"] == 300

    def within(ctx, _kind):
        return _window(505 + 256 if ctx == 8192 else 505)

    held = search(default_ctx=8192, probe=within, abort_mib=30500, grain=2048, max_ctx=8192)
    assert held.calls[0][1] == 2048
    assert held.max_ctx == 8192
    assert held.reason == "confirmed"
    assert floor_confirm_shared(held.calls) == (2048, 505, _SMOKE_PROMPT)
    held_fields = _apply_spill_baseline(held, host, _SMOKE_IDLE)
    assert held_fields["baseline_check"] == "ok"
    assert held_fields["baseline_ctx"] == 2048


def test_a_window_300_mib_over_the_baseline_spills():
    reference = _window(505)
    assert is_spill(_window(505 + 256, peak=28452), reference, shared_growth_mib=256, prompt_ratio=0.5) is False
    assert is_spill(_window(505 + 300, peak=28452), reference, shared_growth_mib=256, prompt_ratio=0.5) is True
    assert is_spill(_window(505 + 300, peak=28451), reference, shared_growth_mib=256, prompt_ratio=0.5) is False

    def probe(ctx, kind):
        shared = _SMOKE_SHARED[ctx]
        peak = 1000
        if kind == "confirm" and ctx == 8192:
            shared = 505 + 300
            peak = 28452
        return _window(shared, peak=peak)

    found = search(default_ctx=2048, probe=probe, abort_mib=30500, grain=2048, max_ctx=8192)
    spilled = [call for call in found.calls if call[0] == "confirm" and call[1] == 8192]
    assert spilled == [("confirm", 8192, 28452, 805, _SMOKE_PROMPT, "spill")]
    assert found.max_ctx == 6144
    assert found.reason == "confirmed"
    assert any(call[0] == "confirm" and call[1] == 6144 and call[5] == "confirmed" for call in found.calls)


def test_a_baseline_past_the_cuda_host_sum_is_inconclusive():
    host = _smoke_host()
    assert baseline_is_spilling(0, 356, 100.0) is False
    assert baseline_is_spilling(0, 357, 100.0) is True
    # 681 is the first integer shared reading on the smoke buffers that clears the allowance.
    assert baseline_is_spilling(_SMOKE_IDLE, 680, host["sum"]) is False
    assert baseline_is_spilling(_SMOKE_IDLE, 681, host["sum"]) is True

    calls = [
        ("confirm", 2048, 1000, 681, _SMOKE_PROMPT, "confirmed") if call[0] == "confirm" and call[1] == 2048 else call
        for call in _smoke_calls()
    ]
    found = SearchResult(8192, "confirmed", calls, peak_mib=1000, tok_s=10, prompt_tok_s=_SMOKE_PROMPT, shared_mib=534)
    fields = _apply_spill_baseline(found, host, _SMOKE_IDLE)
    assert fields["baseline_check"] == "spilling"
    assert fields["baseline_ctx"] == 2048
    assert fields["baseline_shared_mib"] == 681
    assert fields["idle_shared_mib"] == _SMOKE_IDLE
    floor_row = next(
        item for item in fields["shared_delta_mib"] if item["kind"] == "confirm" and item["ctx"] == 2048
    )
    assert floor_row["delta_mib"] == 0
    assert found.max_ctx is None
    assert found.reason == f"{INCONCLUSIVE}: the baseline may be spilling"
    assert found.peak_mib is None
    assert found.shared_mib is None
    assert all(call[5] != "confirmed" for call in found.calls)
    assert found.calls[-1][5] == "baseline spilling"

    unchecked = SearchResult(2048, "confirmed", [("load", 2048, 1000, 505, _SMOKE_PROMPT, "resident")], peak_mib=9)
    missing_idle = _apply_spill_baseline(unchecked, host, None)
    assert missing_idle["baseline_check"] == "spilling"
    assert unchecked.max_ctx is None
    idle_bool = SearchResult(2048, "confirmed", [("load", 2048, 1000, 505, _SMOKE_PROMPT, "resident")])
    assert _apply_spill_baseline(idle_bool, host, True)["baseline_check"] == "spilling"
    assert idle_bool.max_ctx is None
    no_sum = SearchResult(2048, "confirmed", [("load", 2048, 1000, 505, _SMOKE_PROMPT, "resident")])
    assert _apply_spill_baseline(no_sum, {**host, "sum": None}, _SMOKE_IDLE)["baseline_check"] == "spilling"
    assert no_sum.max_ctx is None
    empty = SearchResult(None, "no resident window", [])
    assert _apply_spill_baseline(empty, host, _SMOKE_IDLE)["baseline_check"] == "spilling"
    assert empty.max_ctx is None


def test_the_baseline_helpers_skip_a_reading_that_is_not_the_floor_confirm():
    calls = [
        ("load",),
        ("load", 1024, 1, 100, _SMOKE_PROMPT, "resident"),
        ("confirm", 8192, 1, 534, _SMOKE_PROMPT, "confirmed"),
        ("confirm", True, 1, 505, _SMOKE_PROMPT, "confirmed"),
        ("confirm", 4096, 1, True, _SMOKE_PROMPT, "confirmed"),
        ("confirm", "3000", 1, 505, _SMOKE_PROMPT, "confirmed"),
        ("confirm", 2048, 1, 505, _SMOKE_PROMPT, "confirmed"),
        ("confirm", 2048, 1, 400, True, "confirmed"),
        ("confirm", 1024, 1, 10, "nope", "confirmed"),
    ]
    assert floor_confirm_shared(calls) == (1024, 10, None)
    assert floor_confirm_shared([]) is None
    assert floor_confirm_shared([("load", 2048, 1, 505, 1.0, "resident")]) is None
    numbered = [
        ("confirm", 2048, 1, 505, 40, "confirmed"),
        ("confirm", 6144, 1, 531, 40.0, "confirmed"),
    ]
    assert floor_confirm_shared(numbered) == (2048, 505, 40.0)
    deltas = shared_deltas([("confirm", 4096), ("load", 2048, 1, True, 1, "resident"), ("load", 2048, 1, 507, 1, "resident")], 505)
    assert deltas[0]["delta_mib"] is None
    assert deltas[1]["delta_mib"] is None
    assert deltas[2]["delta_mib"] == 2


def test_a_missing_shared_reading_is_still_a_spill():
    reference = _window(505)
    missing = Observation(True, 1000, shared_mib=None, prompt_tok_s=_SMOKE_PROMPT)
    assert is_spill(missing, reference, shared_growth_mib=256, prompt_ratio=0.5) is True

    def probe(ctx, kind):
        if kind == "confirm":
            return missing
        return _window(_SMOKE_SHARED[ctx])

    found = search(default_ctx=8192, probe=probe, abort_mib=30500, grain=2048, max_ctx=8192)
    assert found.max_ctx is None
    assert found.reason == "no confirmed window"
    assert any(call[0] == "confirm" and call[3] is None and call[5] == "spill" for call in found.calls)


def test_both_specs_load_and_carry_the_spill_baseline_amendment():
    for path in (E5, E5_SMOKE):
        loaded = load_spec(path)
        validate_spec(loaded)
        text = path.read_text(encoding="utf-8")
        assert text.count("Amendment, 2026-10-13") == 4
        amendments = [line for line in text.splitlines() if "Amendment, 2026-10-13" in line]
        earlier = amendments[0]
        assert "lowest resident window" in earlier
        assert "spill_shared_mib (256)" in earlier
        assert "CUDA_Host" in earlier
        assert "inconclusive, never confirmed" in earlier
        later = amendments[1]
        assert "superseded" in later
        assert "floor window" in later
        assert "loaded and confirmed first" in later
        latest = amendments[2]
        assert "capacity pressure" in latest
        assert "shared_gate_from_mib (28452)" in latest
        assert "next lower confirmed window" in latest
        assert "measured peak-per-token slope" in latest
        assert "receipt limit is predicted" in latest
        settled = amendments[3]
        assert "after e5-2026-10-13c" in settled
        assert "settle_tolerance_mib (512)" in settled
        assert "settle_timeout_s (30)" in settled
        assert "resident floor" in settled
        assert "idle baseline" in settled
        assert loaded["settle_tolerance_mib"] == 512
        assert loaded["settle_timeout_s"] == 30
        assert loaded["spill_shared_mib"] == 256
        assert loaded["spill_prompt_ratio"] == 0.5
        assert loaded["shared_gate_from_mib"] == 28452
        assert loaded["abort_mib"] == 30500
        assert loaded["ceiling_mib"] == 31200
    assert load_spec(E5)["id"] == "E5"
    assert load_spec(E5_SMOKE)["id"] == "E5-smoke"
    assert E5.read_text(encoding="utf-8").count("Amendment, 2026-10-12") == 2
