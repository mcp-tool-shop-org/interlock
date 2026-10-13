"""E5's idle-card pin and the preflight that refuses before a load.

The counter fixtures are the harness's InstanceName=CookedValue rendering of
the idle sample from 2026-10-12. The 5090 LUID is the elided token from the
brief. Nothing here opens a socket, runs nvidia-smi, or reads a GPU counter.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from interlock.harness.fill import (
    ADAPTER_DEDICATED,
    ADAPTER_SHARED,
    FillPreflightError,
    driver_cuda,
    judge_preflight,
    listener_exe,
    match_adapter_instance,
    parse_counter_text,
    parse_mib,
    pin_failure,
    real_fill_seams,
    resolve_fill_paths,
)
from interlock.harness.spec import SpecError, load_spec, validate_spec

ROOT = Path(__file__).resolve().parents[1]
E5 = ROOT / "experiments" / "e5" / "spec.yaml"
FIXTURES = ROOT / "tests" / "fixtures" / "e5"
_GPU = "luid_0x00000000_0x000140e1_phys_0"
_SHA = "a" * 40
_GGML = "ab" * 32
_LOG = (
    'msg="server config" env="map[OLLAMA_FLASH_ATTENTION:true OLLAMA_KV_CACHE_TYPE:q8_0]"\n'
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
        log_source="LOCALAPPDATA",
        exe_source="listener",
    )
    record = judge_preflight(**base, driver_cuda_value=driver_cuda(umd))
    assert record["driver_cuda"] == "driver 13.4"
    assert record["instance"] == _GPU
    assert record["dedicated_mib"] == 2508
    assert record["shared_mib"] == 74
    assert record["log_source"] == "LOCALAPPDATA"
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


def test_paths_use_the_listener_and_localappdata_and_overrides_win(tmp_path: Path, monkeypatch):
    _clear(monkeypatch)
    app = tmp_path / "appdata"
    log = app / "Ollama" / "server.log"
    log.parent.mkdir(parents=True)
    log.write_text("log\n", encoding="utf-8")
    exe = tmp_path / "dist" / "ollama.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"exe")
    calls = []

    def run(argv, **kwargs):
        calls.append(tuple(argv))
        assert argv[0] == "powershell"
        assert "11434" in argv[-1]
        assert "RemotePort" in argv[-1]
        return json.dumps({"exe": str(exe)})

    paths = resolve_fill_paths(run, localappdata=str(app))
    assert paths["log_source"] == "LOCALAPPDATA"
    assert paths["exe_source"] == "listener"
    assert Path(paths["log_path"]) == log
    assert paths["dll"].endswith("ggml-cuda.dll")
    assert calls
    assert listener_exe(lambda *args, **kwargs: "not-json") is None

    monkeypatch.setenv("INTERLOCK_OLLAMA_LOG", str(tmp_path / "override.log"))
    monkeypatch.setenv("INTERLOCK_OLLAMA_EXE", str(tmp_path / "override.exe"))

    def refuse(*args, **kwargs):
        raise AssertionError("listener ran")

    overridden = resolve_fill_paths(refuse, localappdata=str(tmp_path / "unused"))
    assert overridden["log_source"] == "INTERLOCK_OLLAMA_LOG"
    assert overridden["exe_source"] == "INTERLOCK_OLLAMA_EXE"
    assert Path(overridden["log_path"]) == tmp_path / "override.log"
    assert Path(overridden["exe"]) == tmp_path / "override.exe"
    assert "\\" not in overridden["log_source"]
    assert "/" not in overridden["exe_source"]


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
    with pytest.raises(SpecError, match="adapter_match_tolerance_mib"):
        validate_spec({**loaded, "adapter_match_tolerance_mib": 256})
