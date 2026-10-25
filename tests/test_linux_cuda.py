"""Tests for the Linux CUDA measuring path and committed C2 ladder specs.

All tests run on recorded nvidia-smi / NVML fixtures and a mocked Ollama.
No live GPU, no Windows performance counters, and no Colab session.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from interlock.harness.spec import SpecError, load_spec, validate_spec
from interlock.receipt import validate_row

ROOT = Path(__file__).resolve().parents[1]

NVIDIA_SMI_QUERY_L4 = "NVIDIA L4, 1420, 23034, 550.54.15\n"
NVIDIA_SMI_HEADER_CUDA_12_4 = (
    "==============NVSMI LOG==============\n"
    "Driver Version                            : 550.54.15\n"
    "CUDA Version                              : 12.4\n"
)


def _remote_ident(card: str = "L4", vram_mib: int = 23034) -> Any:
    from interlock.devices import RemoteDeviceIdentity

    return RemoteDeviceIdentity(
        provider="colab",
        card=card,
        vram_mib=vram_mib,
        driver_version="550.54.15",
        cuda_version="12.4",
        pool="us-central1-a",
    )


def test_linux_cuda_probe_parses_nvidia_smi_and_nvml_without_windows_counters():
    from interlock.harness.linux_cuda import (
        LinuxCudaError,
        compute_abort_mib,
        map_card_name,
        probe_nvidia_smi,
        probe_nvml,
    )

    reading = probe_nvidia_smi(
        runner=lambda argv: (
            NVIDIA_SMI_QUERY_L4 if "--query-gpu" in " ".join(argv) else NVIDIA_SMI_HEADER_CUDA_12_4
        ),
        pool="us-central1-a",
    )
    assert reading.card == "L4"
    assert reading.used_mib == 1420
    assert reading.total_mib == 23034
    assert reading.driver_version == "550.54.15"
    assert reading.cuda_version == "12.4"
    assert reading.to_remote_identity().logical_device == "colab:L4"
    assert compute_abort_mib(23034, 0.90) == 20730

    assert map_card_name("NVIDIA A100-SXM4-40GB", 40960) == "A100-40"
    assert map_card_name("NVIDIA A100-SXM4-80GB", 81920) == "A100-80"
    assert map_card_name("Tesla T4", 15360) == "T4"
    assert map_card_name("NVIDIA RTX PRO 6000 Blackwell", 98304) == "G4"

    class FakeMem:
        used = 2048 * 1024 * 1024
        total = 40960 * 1024 * 1024

    class FakeNvml:
        def nvmlInit(self):
            return None

        def nvmlShutdown(self):
            return None

        def nvmlDeviceGetHandleByIndex(self, index):
            assert index == 0
            return "handle-0"

        def nvmlDeviceGetName(self, _handle):
            return b"NVIDIA A100-SXM4-40GB"

        def nvmlDeviceGetMemoryInfo(self, _handle):
            return FakeMem()

        def nvmlSystemGetDriverVersion(self):
            return "550.54.15"

        def nvmlSystemGetCudaDriverVersion_v2(self):
            return 12040

    nvml_reading = probe_nvml(nvml=FakeNvml(), pool="europe-west4-a")
    assert nvml_reading.card == "A100-40"
    assert nvml_reading.used_mib == 2048
    assert nvml_reading.total_mib == 40960
    assert nvml_reading.cuda_version == "12.4"
    assert nvml_reading.to_remote_identity().pool == "europe-west4-a"

    with pytest.raises(LinuxCudaError, match="abort_fraction"):
        compute_abort_mib(23034, 1.0)
    with pytest.raises(LinuxCudaError, match="total_vram_mib"):
        compute_abort_mib(0, 0.90)
    with pytest.raises(LinuxCudaError, match="unrecognized"):
        map_card_name("Unknown GPU", 8192)
    assert map_card_name("L4", 23034) == "L4"
    with pytest.raises(LinuxCudaError, match="expected 1 GPU row"):
        probe_nvidia_smi(runner=lambda _argv: "")
    with pytest.raises(LinuxCudaError, match="malformed"):
        probe_nvidia_smi(runner=lambda _argv: "only,two\n")
    with pytest.raises(LinuxCudaError, match="non-numeric"):
        probe_nvidia_smi(runner=lambda _argv: "NVIDIA L4, bad, 23034, 550.54.15\n")
    with pytest.raises(LinuxCudaError, match="CUDA Version"):
        probe_nvidia_smi(
            runner=lambda argv: NVIDIA_SMI_QUERY_L4 if "--query-gpu" in " ".join(argv) else "no version\n"
        )


def test_linux_ollama_server_lifecycle_starts_and_stops_run_owned_instance(tmp_path: Path):
    from interlock.harness.linux_cuda import LinuxCudaError, LinuxOllamaServer

    with pytest.raises(LinuxCudaError, match="dedicated port"):
        LinuxOllamaServer(host="127.0.0.1:11434", log_path=tmp_path / "ollama.log")
    with pytest.raises(LinuxCudaError, match="dedicated port"):
        LinuxOllamaServer(host="127.0.0.1:11490", log_path=tmp_path / "ollama.log")
    with pytest.raises(LinuxCudaError, match="include port"):
        LinuxOllamaServer(host="127.0.0.1", log_path=tmp_path / "ollama.log")

    events: list[str] = []

    class FakeProc:
        pid = 4242

        def poll(self):
            return None

        def terminate(self):
            events.append("terminate")

        def wait(self, timeout=None):
            events.append(f"wait:{timeout}")
            return 0

    server = LinuxOllamaServer(
        host="127.0.0.1:11441",
        log_path=tmp_path / "ollama.log",
        popen=lambda argv, **kwargs: (events.append(f"spawn:{argv[0]}"), FakeProc())[1],
        http_get=lambda url: {"version": "0.35.1"} if url.endswith("/api/version") else {"models": []},
        http_post=lambda url, payload: (events.append(f"unload:{payload['model']}"), {"done": True})[1],
    )
    with server as running:
        assert running.version == "0.35.1"
        assert running.pid == 4242
        running.unload("qwen3:8b")
    assert events == ["spawn:ollama", "unload:qwen3:8b", "terminate", "wait:5.0"]


def test_linux_fill_card_ladder_records_resident_ceiling_and_abort_fraction():
    from interlock.harness.linux_cuda import run_linux_fill_card

    ident = _remote_ident("L4", 23034)
    # Abort line at 0.90 * 23034 = 20730 MiB.
    # Model trained_ctx is 32768. Contexts <= 16384 stay resident under 20730 MiB;
    # 32768 peaks at 21500 MiB (>= 20730 MiB abort_mib) so search steps down to 16384.
    def fake_window(num_ctx: int):
        if num_ctx > 16384:
            return {
                "size": 9000,
                "size_vram": 9000,
                "peak_mib": 21500,
                "ttft_ms": 420.0,
                "decode_tok_s": 38.0,
                "prompt_tok_s": 620.0,
            }
        return {
            "size": 9000,
            "size_vram": 9000,
            "peak_mib": 8000 + (num_ctx // 2048) * 1200,
            "ttft_ms": 180.0,
            "decode_tok_s": 44.5,
            "prompt_tok_s": 780.0,
        }

    row = run_linux_fill_card(
        spec_id="C2-FILL-CARD",
        spec_rel="experiments/c2-fill-card/spec.yaml",
        run_id="c2-l4-fill-01",
        date="2026-10-24",
        remote_device=ident,
        model="qwen3:14b",
        revision_or_digest="bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8",
        trained_ctx=32768,
        abort_fraction=0.90,
        grain=2048,
        measure_window=fake_window,
    )
    validate_row(row)
    assert row["device"] == "colab:L4"
    assert row["remote_device"]["card"] == "L4"
    assert row["gpu_resident"] is True
    assert row["settings"]["max_resident_ctx"] == 16384
    assert row["settings"]["abort_mib"] == 20730
    assert row["settings"]["abort_fraction"] == 0.90
    assert row["settings"]["peak_mib"] == 8000 + 8 * 1200
    assert "shared_mib" not in row["settings"] or row["settings"]["shared_mib"] is None

    # Non-power-of-two trained_ctx, partial offload, and prompt-rate cliff in fill-card:
    offload_row = run_linux_fill_card(
        spec_id="C2-FILL-CARD",
        spec_rel="experiments/c2-fill-card/spec.yaml",
        run_id="c2-l4-offload",
        date="2026-10-24",
        remote_device=ident,
        model="qwen3:14b",
        revision_or_digest="bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8",
        trained_ctx=10240,
        abort_fraction=0.90,
        grain=2048,
        measure_window=lambda ctx: {
            "size": 9000,
            "size_vram": 9000 if ctx <= 2048 else 6000,
            "peak_mib": 10000,
            "ttft_ms": 120.0,
            "decode_tok_s": 40.0,
            "prompt_tok_s": 800.0,
        },
    )
    assert offload_row["settings"]["max_resident_ctx"] == 2048

    cliff_row = run_linux_fill_card(
        spec_id="C2-FILL-CARD",
        spec_rel="experiments/c2-fill-card/spec.yaml",
        run_id="c2-l4-cliff",
        date="2026-10-24",
        remote_device=ident,
        model="qwen3:14b",
        revision_or_digest="bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8",
        trained_ctx=8192,
        abort_fraction=0.90,
        grain=2048,
        measure_window=lambda ctx: {
            "size": 9000,
            "size_vram": 9000,
            "peak_mib": 10000,
            "ttft_ms": 120.0,
            "decode_tok_s": 40.0,
            "prompt_tok_s": 800.0 if ctx <= 2048 else 100.0,
        },
    )
    assert cliff_row["settings"]["max_resident_ctx"] == 2048

    below_grain = run_linux_fill_card(
        spec_id="C2-FILL-CARD",
        spec_rel="experiments/c2-fill-card/spec.yaml",
        run_id="c2-l4-small",
        date="2026-10-24",
        remote_device=ident,
        model="qwen3:14b",
        revision_or_digest="bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8",
        trained_ctx=1024,
        abort_fraction=0.90,
        grain=2048,
        measure_window=fake_window,
    )
    assert below_grain["outcome"] == "ctx_exceeded"


def test_linux_cuda_records_all_four_failure_classes_as_rows_never_skips():
    from interlock.harness.linux_cuda import LinuxShapeFailure, build_linux_failure_row, run_linux_fill_card

    t4 = _remote_ident("T4", 15360)
    for failure_class, detail in (
        ("oom", "CUDA out of memory at num_ctx 16384 (peak 15100 MiB >= abort 13824 MiB)"),
        ("load_refused", "model requires 22400 MiB VRAM, colab:T4 has 15360 MiB"),
        ("timeout", "generate timed out after 120.0s at concurrency 8"),
        ("ctx_exceeded", "requested num_ctx 65536 exceeds trained context 32768"),
    ):
        row = build_linux_failure_row(
            spec_id="C2-GENERATE",
            spec_rel="experiments/c2-generate/spec.yaml",
            run_id=f"c2-t4-{failure_class}",
            date="2026-10-24",
            task="generate",
            model="gemma4:31b",
            revision_or_digest="6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7",
            remote_device=t4,
            failure_class=failure_class,
            reason=detail,
            num_ctx=16384,
            prompt_tokens=4096,
            concurrency=4,
        )
        validate_row(row)
        assert row["outcome"] == failure_class
        assert row["settings"]["failure_class"] == failure_class
        assert detail in row["settings"]["failure_reason"]
        assert detail in row["notes"]
        assert row["latency"]["throughput"] is None

    def _raise_refused(_ctx: int):
        raise LinuxShapeFailure("load_refused", "model requires 22400 MiB, colab:T4 has 15360 MiB")

    # Also verify run_linux_fill_card records a failure row when floor grain refuses to load.
    refused_row = run_linux_fill_card(
        spec_id="C2-FILL-CARD",
        spec_rel="experiments/c2-fill-card/spec.yaml",
        run_id="c2-t4-refused",
        date="2026-10-24",
        remote_device=t4,
        model="gemma4:31b",
        revision_or_digest="6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7",
        trained_ctx=32768,
        abort_fraction=0.90,
        grain=2048,
        measure_window=_raise_refused,
    )
    validate_row(refused_row)
    assert refused_row["outcome"] == "load_refused"
    assert "22400 MiB" in refused_row["notes"]


def test_linux_generate_ladder_sweeps_512_to_32k_and_concurrency_1_4_8():
    from interlock.harness.linux_cuda import (
        GENERATE_CONCURRENCIES,
        GENERATE_PROMPT_LENGTHS,
        LinuxShapeFailure,
        run_linux_generate_ladder,
    )

    assert GENERATE_PROMPT_LENGTHS == (512, 1024, 2048, 4096, 8192, 16384, 32768)
    assert GENERATE_CONCURRENCIES == (1, 4, 8)

    a100 = _remote_ident("A100-40", 40960)

    def fake_generate(prompt_tokens: int, concurrency: int):
        if prompt_tokens == 32768 and concurrency == 8:
            raise LinuxShapeFailure("oom", "peak 38200 MiB exceeded abort_mib 36864 MiB at 32k x8", peak_mib=38200)
        if prompt_tokens == 32768 and concurrency == 4:
            return {
                "ttft_ms": 900.0,
                "decode_tok_s": 40.0,
                "latency_ms": 1200.0,
                "peak_mib": 37000,
                "size": 9000,
                "size_vram": 9000,
            }
        return {
            "ttft_ms": 40.0 + prompt_tokens * 0.02 * concurrency,
            "decode_tok_s": 85.0 / (1.0 + 0.15 * (concurrency - 1)),
            "latency_ms": 250.0 + prompt_tokens * 0.05,
            "peak_mib": 14000 + (prompt_tokens // 512) * 20 * concurrency,
            "size": 9000,
            "size_vram": 9000,
        }

    rows = run_linux_generate_ladder(
        spec_id="C2-GENERATE",
        spec_rel="experiments/c2-generate/spec.yaml",
        run_id="c2-a100-gen-01",
        date="2026-10-24",
        remote_device=a100,
        model="qwen3:14b",
        revision_or_digest="bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8",
        measure_shape=fake_generate,
    )
    assert len(rows) == 21
    for r in rows:
        validate_row(r)
        assert r["device"] == "colab:A100-40"
    ok_rows = [r for r in rows if r.get("outcome") is None]
    oom_rows = [r for r in rows if r.get("outcome") == "oom"]
    assert len(ok_rows) == 19
    assert len(oom_rows) == 2
    assert oom_rows[-1]["settings"]["prompt_tokens"] == 32768
    assert oom_rows[-1]["settings"]["concurrency"] == 8
    assert "36864 MiB" in oom_rows[-1]["notes"]


def test_linux_embed_rerank_nli_ladder_sweeps_batches_1_8_32():
    from interlock.harness.linux_cuda import (
        THROUGHPUT_BATCH_SIZES,
        THROUGHPUT_TASKS,
        LinuxShapeFailure,
        run_linux_throughput_ladder,
    )

    assert THROUGHPUT_TASKS == ("embed", "rerank", "nli")
    assert THROUGHPUT_BATCH_SIZES == (1, 8, 32)

    l4 = _remote_ident("L4", 23034)
    models = {
        "embed": ("BAAI/bge-base-en-v1.5", "digest-embed"),
        "rerank": ("BAAI/bge-reranker-v2-m3", "digest-rerank"),
        "nli": ("cross-encoder/nli-deberta-v3-base", "6c749ce3425cd33b46d187e45b92bbf96ee12ec7"),
    }

    def fake_batch(task: str, batch: int):
        if task == "nli" and batch == 32:
            raise LinuxShapeFailure("timeout", "nli batch 32 exceeded 60s timeout")
        return {
            "throughput": 30.0 * batch,
            "warm_p50_ms": 12.0 * batch,
            "warm_p95_ms": 15.0 * batch,
            "peak_mib": 2200 + batch * 50,
            "size": 1800,
            "size_vram": 1800,
        }

    rows = run_linux_throughput_ladder(
        spec_id="C2-THROUGHPUT",
        spec_rel="experiments/c2-throughput/spec.yaml",
        run_id="c2-l4-tput-01",
        date="2026-10-24",
        remote_device=l4,
        models_by_task=models,
        measure_batch=fake_batch,
    )
    assert len(rows) == 9
    for r in rows:
        validate_row(r)
    timed_out = [r for r in rows if r.get("outcome") == "timeout"]
    assert len(timed_out) == 1
    assert timed_out[0]["task"] == "nli"
    assert timed_out[0]["settings"]["batch"] == 32


def test_committed_c2_specs_validate_and_carry_standards_headers():
    specs = (
        ROOT / "experiments" / "c2-fill-card" / "spec.yaml",
        ROOT / "experiments" / "c2-generate" / "spec.yaml",
        ROOT / "experiments" / "c2-throughput" / "spec.yaml",
    )
    for path in specs:
        assert path.is_file(), f"missing committed spec {path}"
        text = path.read_text(encoding="utf-8")
        for header in (
            "PIN_PER_STEP:",
            "ANDON_AUTHORITY:",
            "NAMED_COMPENSATORS:",
            "DECOMPOSE_BY_SECRETS:",
            "UNCERTAINTY_GATED_HUMANS:",
            "EXTERNAL_VERIFIER:",
        ):
            assert header in text, f"{path} missing {header}"
        loaded = load_spec(path)
        assert loaded["platform"] == "linux-cuda"

    with pytest.raises(SpecError):
        validate_spec({"id": "bad", "question": "q", "decision": "d", "procedure": "generate-ladder",
                       "needs": {"grant": False, "npu_session": False, "cpu_window": False},
                       "stop_rules": ["stop"], "dataset": {"name": "d"}, "metrics": ["m"]})
    base_linux = yaml.safe_load((ROOT / "experiments" / "c2-fill-card" / "spec.yaml").read_text(encoding="utf-8"))
    with pytest.raises(SpecError, match="Windows shared-memory"):
        validate_spec({**base_linux, "spill_shared_mib": 256})
    with pytest.raises(SpecError, match="abort_fraction"):
        validate_spec({**base_linux, "abort_fraction": 0.1})
    with pytest.raises(SpecError, match="2048-token grid"):
        validate_spec({**base_linux, "grain": 1024})
    with pytest.raises(SpecError, match="confirm_prompt_fraction"):
        validate_spec({**base_linux, "confirm_prompt_fraction": 1.5})
    with pytest.raises(SpecError, match="spill_prompt_ratio"):
        validate_spec({**base_linux, "spill_prompt_ratio": 0.9})
    with pytest.raises(SpecError, match="model roster"):
        validate_spec({**base_linux, "models": []})
    with pytest.raises(SpecError, match="missing device"):
        validate_spec({**base_linux, "models": [{"model": "m", "revision_or_digest": "d", "engine": "ollama"}]})
    with pytest.raises(SpecError, match="remote colab"):
        validate_spec(
            {
                **base_linux,
                "models": [{"model": "m", "revision_or_digest": "d", "engine": "ollama", "device": "5090"}],
            }
        )
    gen_linux = yaml.safe_load((ROOT / "experiments" / "c2-generate" / "spec.yaml").read_text(encoding="utf-8"))
    with pytest.raises(SpecError, match="prompt_lengths"):
        validate_spec({**gen_linux, "prompt_lengths": [512]})
    with pytest.raises(SpecError, match="concurrencies"):
        validate_spec({**gen_linux, "concurrencies": [1]})
    batch_linux = yaml.safe_load((ROOT / "experiments" / "c2-throughput" / "spec.yaml").read_text(encoding="utf-8"))
    with pytest.raises(SpecError, match="tasks"):
        validate_spec({**batch_linux, "tasks": ["embed"]})
    with pytest.raises(SpecError, match="batch_sizes"):
        validate_spec({**batch_linux, "batch_sizes": [1]})


def test_map_card_name_checks_vram_ranges_and_engine_version_is_never_hardcoded():
    from interlock.harness.linux_cuda import (
        UNRECORDED,
        LinuxCudaError,
        build_linux_failure_row,
        map_card_name,
        run_linux_fill_card,
    )
    from interlock.importers.remote import import_remote_window

    assert map_card_name("NVIDIA RTX PRO 6000 Blackwell", 98304) == "G4"
    with pytest.raises(LinuxCudaError, match="VRAM"):
        map_card_name("NVIDIA RTX PRO 6000 Blackwell", 24576)
    with pytest.raises(LinuxCudaError, match="VRAM"):
        map_card_name("A100-80", 40960)
    with pytest.raises(LinuxCudaError, match="VRAM"):
        map_card_name("T4", 40960)

    ident = _remote_ident("T4", 15360)
    failure_row = build_linux_failure_row(
        spec_id="C2-FILL-CARD",
        spec_rel="experiments/c2-fill-card/spec.yaml",
        run_id="c2-t4-default-ver",
        date="2026-10-24",
        task="generate",
        model="qwen3:8b",
        revision_or_digest="500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41",
        remote_device=ident,
        failure_class="oom",
        reason="CUDA out of memory",
        num_ctx=4096,
    )
    assert failure_row["engine"]["version"] == UNRECORDED

    fill_row = run_linux_fill_card(
        spec_id="C2-FILL-CARD",
        spec_rel="experiments/c2-fill-card/spec.yaml",
        run_id="c2-t4-fill-ver",
        date="2026-10-24",
        remote_device=ident,
        model="qwen3:8b",
        revision_or_digest="500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41",
        trained_ctx=4096,
        abort_fraction=0.90,
        grain=2048,
        measure_window=lambda _ctx: {
            "size": 5000,
            "size_vram": 5000,
            "peak_mib": 8000,
            "ttft_ms": 90.0,
            "decode_tok_s": 42.0,
            "prompt_tok_s": 700.0,
        },
    )
    assert fill_row["engine"]["version"] == UNRECORDED

    imported = import_remote_window(
        b'{"run_id":"r1","date":"2026-10-24","remote_device":{"provider":"colab","card":"T4","vram_mib":15360,'
        b'"driver_version":"550.54.15","cuda_version":"12.4","pool":"us-central1-a"},'
        b'"records":[{"task":"generate","model":"qwen3:8b","revision_or_digest":"500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41",'
        b'"num_ctx":2048,"failure_class":"oom","failure_reason":"CUDA OOM"}]}',
        source_path="receipts/r1.json",
    )
    assert imported[0]["engine"]["version"] == UNRECORDED


def test_linux_fill_card_bisects_2048_grid_and_throughput_ladder_records_oom_on_offload_or_abort():
    from interlock.harness.linux_cuda import run_linux_fill_card, run_linux_throughput_ladder

    ident = _remote_ident("A100-40", 40960)
    # Windows <= 24576 stay under abort_mib (36864 MiB); windows >= 26624 reach abort_mib.
    # A doubling search hits 16384 (pass) then 32768 (fail) and must bisect the 2048 grid to find 24576.
    visited: list[int] = []

    def measure_window_bisect(num_ctx: int):
        visited.append(num_ctx)
        return {
            "size": 18000,
            "size_vram": 18000,
            "peak_mib": 22000 + (num_ctx // 2048) * 1150 if num_ctx <= 24576 else 38000,
            "ttft_ms": 150.0,
            "decode_tok_s": 62.0,
            "prompt_tok_s": 900.0,
        }

    row = run_linux_fill_card(
        spec_id="C2-FILL-CARD",
        spec_rel="experiments/c2-fill-card/spec.yaml",
        run_id="c2-a100-bisect",
        date="2026-10-24",
        remote_device=ident,
        model="qwen3:14b",
        revision_or_digest="bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8",
        trained_ctx=32768,
        abort_fraction=0.90,
        grain=2048,
        measure_window=measure_window_bisect,
    )
    validate_row(row)
    assert row["settings"]["max_resident_ctx"] == 24576
    assert 24576 in visited

    # Throughput ladder must record oom failure rows when size_vram < size or peak_mib >= abort_mib.
    l4 = _remote_ident("L4", 23034)
    models = {
        "embed": ("BAAI/bge-base-en-v1.5", "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"),
        "rerank": ("BAAI/bge-reranker-v2-m3", "5617a9f61b028005a4858fdac845db406aefb181"),
        "nli": ("cross-encoder/nli-deberta-v3-base", "6c749ce3425cd33b46d187e45b92bbf96ee12ec7"),
    }

    def measure_batch_spill(task: str, batch: int):
        if task == "embed" and batch == 32:
            return {
                "throughput": 120.0,
                "warm_p50_ms": 80.0,
                "warm_p95_ms": 90.0,
                "peak_mib": 12000,
                "size": 2000,
                "size_vram": 1000,  # partial offload!
            }
        if task == "rerank" and batch == 32:
            return {
                "throughput": 80.0,
                "warm_p50_ms": 110.0,
                "warm_p95_ms": 120.0,
                "peak_mib": 21500,  # >= abort_mib (20730 MiB)!
                "size": 2000,
                "size_vram": 2000,
            }
        return {
            "throughput": 40.0 * batch,
            "warm_p50_ms": 10.0 * batch,
            "warm_p95_ms": 12.0 * batch,
            "peak_mib": 2500 + batch * 40,
            "size": 2000,
            "size_vram": 2000,
        }

    tput_rows = run_linux_throughput_ladder(
        spec_id="C2-THROUGHPUT",
        spec_rel="experiments/c2-throughput/spec.yaml",
        run_id="c2-l4-tput-oom",
        date="2026-10-24",
        remote_device=l4,
        models_by_task=models,
        measure_batch=measure_batch_spill,
    )
    oom_rows = [r for r in tput_rows if r.get("outcome") == "oom"]
    assert len(oom_rows) == 2
    assert {r["task"] for r in oom_rows} == {"embed", "rerank"}


def test_interlock_run_dry_run_and_offline_execution_produce_validated_linux_cuda_receipt(tmp_path: Path, capsys):
    import json
    import shutil
    import subprocess

    from interlock.cli import main
    from interlock.harness.runner import dry_run, run_spec

    spec_rels = (
        "experiments/c2-fill-card/spec.yaml",
        "experiments/c2-generate/spec.yaml",
        "experiments/c2-throughput/spec.yaml",
    )
    for spec_rel in spec_rels:
        dest = tmp_path / spec_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / spec_rel, dest)

    for cmd in (
        ["git", "init"],
        ["git", "config", "user.email", "dev@example.com"],
        ["git", "config", "user.name", "dev"],
        ["git", "add", "."],
        ["git", "commit", "-m", "spec"],
    ):
        subprocess.run(cmd, cwd=tmp_path, check=True, capture_output=True)  # noqa: S603, S607

    for spec_rel in spec_rels:
        plan_text = dry_run(tmp_path / spec_rel, tmp_path)
        assert "platform: linux-cuda" in plan_text
        assert "abort_fraction: 0.9" in plan_text
        assert main(["run", str(tmp_path / spec_rel), "--root", str(tmp_path), "--dry-run"]) == 0
        cli_out = capsys.readouterr().out
        assert "platform: linux-cuda" in cli_out

    spec_copy = tmp_path / "experiments" / "c2-fill-card" / "spec.yaml"

    class FakeMem:
        used = 10240 * 1024 * 1024
        total = 23034 * 1024 * 1024

    class FakeNvml:
        def nvmlInit(self):
            return None

        def nvmlShutdown(self):
            return None

        def nvmlDeviceGetHandleByIndex(self, _index):
            return "h0"

        def nvmlDeviceGetName(self, _handle):
            return "NVIDIA L4"

        def nvmlDeviceGetMemoryInfo(self, _handle):
            return FakeMem()

        def nvmlSystemGetDriverVersion(self):
            return "550.54.15"

        def nvmlSystemGetCudaDriverVersion_v2(self):
            return 12040

    class FakeProc:
        pid = 9191

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

    def fake_http_get(url: str) -> dict[str, Any]:
        if url.endswith("/api/version"):
            return {"version": "0.6.2"}
        if url.endswith("/api/ps"):
            return {"models": [{"name": "qwen3:8b", "size": 5200000000, "size_vram": 5200000000}]}
        return {}

    def fake_http_post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("keep_alive") == 0:
            return {"done": True}
        opts = payload.get("options") or {}
        ctx = int(opts.get("num_ctx") or 2048)
        if ctx > 8192:
            return {"error": f"request ({ctx} tokens) exceeds the available context size (8192 tokens)"}
        return {
            "load_duration": 20_000_000,
            "prompt_eval_count": int(ctx * 0.9),
            "prompt_eval_duration": 100_000_000,
            "eval_count": 16,
            "eval_duration": 250_000_000,
        }

    rows = run_spec(
        spec_copy,
        root=tmp_path,
        run_id="c2-l4-e2e",
        date="2026-10-24",
        python_path=str( tmp_path.parent / "external-python" ),
        models=["qwen3:8b"],
        seams={
            "nvml": FakeNvml(),
            "pool": "us-central1-a",
            "popen": lambda *_a, **_kw: FakeProc(),
            "http_get": fake_http_get,
            "http_post": fake_http_post,
        },
    )
    assert len(rows) >= 1
    receipt_file = tmp_path / "receipts" / "c2-l4-e2e.jsonl"
    assert receipt_file.is_file()
    written_rows = [json.loads(line) for line in receipt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(written_rows) == len(rows)
    for r in written_rows:
        validate_row(r)
        assert r["device"] == "colab:L4"
        assert r["engine"]["version"] == "0.6.2"

    # Also exercise generate-ladder and batch-ladder offline execution via run_spec + nvidia-smi fallback:
    def fake_smi_runner(argv: list[str]) -> str:
        if "--query-gpu=name,memory.used,memory.total,driver_version" in argv:
            return "NVIDIA L4, 8192, 23034, 550.54.15\n"
        return (
            "+-----------------------------------------------------------------------------------------+\n"
            "| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |\n"
        )

    gen_rows = run_spec(
        tmp_path / "experiments" / "c2-generate" / "spec.yaml",
        root=tmp_path,
        run_id="c2-l4-gen-e2e",
        date="2026-10-24",
        python_path=str(tmp_path.parent / "external-python"),
        models=["qwen3:8b"],
        seams={
            "runner": fake_smi_runner,
            "pool": "us-central1-a",
            "popen": lambda *_a, **_kw: FakeProc(),
            "http_get": fake_http_get,
            "http_post": fake_http_post,
        },
    )
    assert len(gen_rows) >= 21
    assert len({r["id"] for r in gen_rows}) == len(gen_rows)

    tput_rows = run_spec(
        tmp_path / "experiments" / "c2-throughput" / "spec.yaml",
        root=tmp_path,
        run_id="c2-l4-tput-e2e",
        date="2026-10-24",
        python_path=str(tmp_path.parent / "external-python"),
        models=["BAAI/bge-base-en-v1.5", "BAAI/bge-reranker-v2-m3", "cross-encoder/nli-deberta-v3-base"],
        seams={
            "nvml": FakeNvml(),
            "pool": "us-central1-a",
            "popen": lambda *_a, **_kw: FakeProc(),
            "http_get": fake_http_get,
            "http_post": lambda _url, _payload: {"total_duration": 25_000_000},
        },
    )
    assert len(tput_rows) == 18


def test_c2_specs_cover_sweep_plan_matrix_quants_and_hashed_gemma_repo():
    fill_spec = load_spec(ROOT / "experiments" / "c2-fill-card" / "spec.yaml")
    gen_spec = load_spec(ROOT / "experiments" / "c2-generate" / "spec.yaml")
    tput_spec = load_spec(ROOT / "experiments" / "c2-throughput" / "spec.yaml")

    expected_cards = {"colab:A100-40", "colab:A100-80", "colab:G4", "colab:L4", "colab:T4"}
    for spec in (fill_spec, gen_spec, tput_spec):
        cards = {entry["device"] for entry in spec["models"]}
        assert cards == expected_cards
        assert spec["replication"]["vms_per_card"] == 3

    for spec in (fill_spec, gen_spec):
        models = {entry["model"] for entry in spec["models"]}
        # Plaintext gemma4:31b is replaced by its sha256:<64-hex> hashed repo id per C2-SWEEP-PLAN.md.
        assert "gemma4:31b" not in models
        assert any(m.startswith("sha256:") and len(m) == len("sha256:") + 64 for m in models)
        assert "qwen3:8b" in models
        assert "phi3.5:3.8b" in models
        assert "muse-glimmer:30b" in models
        quants = {entry["quant"] for entry in spec["models"]}
        assert {"Q4_K_M", "Q6_K", "Q8_0", "BF16"}.issubset(quants)
        # Pre-registered refusal cells (e.g. 31B Q8_0 on T4) stay in the roster as refusal cells, never dropped.
        assert any(
            entry["device"] == "colab:T4" and entry.get("expected_outcome") == "load_refused"
            for entry in spec["models"]
        )

    tput_models = {entry["model"] for entry in tput_spec["models"]}
    assert {
        "nomic-ai/nomic-embed-text-v1.5",
        "BAAI/bge-base-en-v1.5",
        "BAAI/bge-reranker-v2-m3",
        "Qwen/Qwen3-Embedding-0.6B",
        "cross-encoder/nli-deberta-v3-base",
    }.issubset(tput_models)
    for entry in tput_spec["models"]:
        assert entry["revision_or_digest"] != "unrecorded"


def test_c2_multi_quant_row_ids_never_collide_and_load_refused_cells_are_honoured():
    from interlock.harness.linux_cuda import (
        LinuxOllamaServer,
        build_linux_failure_row,
        run_linux_fill_card,
        run_linux_generate_ladder,
        run_linux_spec,
    )

    t4 = _remote_ident("T4", 15360)
    fill_q4 = run_linux_fill_card(
        spec_id="C2-FILL-CARD",
        spec_rel="experiments/c2-fill-card/spec.yaml",
        run_id="r1",
        date="2026-10-24",
        remote_device=t4,
        model="qwen3:8b",
        revision_or_digest="500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41",
        trained_ctx=4096,
        abort_fraction=0.90,
        grain=256,
        quant="Q4_K_M",
        licence="apache-2.0",
        measure_window=lambda _ctx: {
            "size": 5000,
            "size_vram": 5000,
            "peak_mib": 6000,
            "prompt_tok_s": 500.0,
            "decode_tok_s": 60.0,
            "ttft_ms": 80.0,
        },
    )
    fill_q8 = run_linux_fill_card(
        spec_id="C2-FILL-CARD",
        spec_rel="experiments/c2-fill-card/spec.yaml",
        run_id="r1",
        date="2026-10-24",
        remote_device=t4,
        model="qwen3:8b",
        revision_or_digest="500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41",
        trained_ctx=4096,
        abort_fraction=0.90,
        grain=256,
        quant="Q8_0",
        licence="apache-2.0",
        measure_window=lambda _ctx: {
            "size": 8800,
            "size_vram": 8800,
            "peak_mib": 10000,
            "prompt_tok_s": 400.0,
            "decode_tok_s": 45.0,
            "ttft_ms": 95.0,
        },
    )
    assert fill_q4["id"] != fill_q8["id"]
    assert "Q4_K_M" in fill_q4["id"]
    assert "Q8_0" in fill_q8["id"]

    gen_q4 = run_linux_generate_ladder(
        spec_id="C2-GENERATE",
        spec_rel="experiments/c2-generate/spec.yaml",
        run_id="r1",
        date="2026-10-24",
        remote_device=t4,
        model="qwen3:8b",
        revision_or_digest="500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41",
        prompt_lengths=(512,),
        concurrencies=(1,),
        quant="Q4_K_M",
        licence="apache-2.0",
        measure_shape=lambda _p, _c: {
            "ttft_ms": 50.0,
            "decode_tok_s": 70.0,
            "latency_ms": 120.0,
            "peak_mib": 6000,
            "size": 5000,
            "size_vram": 5000,
        },
    )[0]
    gen_q8 = run_linux_generate_ladder(
        spec_id="C2-GENERATE",
        spec_rel="experiments/c2-generate/spec.yaml",
        run_id="r1",
        date="2026-10-24",
        remote_device=t4,
        model="qwen3:8b",
        revision_or_digest="500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41",
        prompt_lengths=(512,),
        concurrencies=(1,),
        quant="Q8_0",
        licence="apache-2.0",
        measure_shape=lambda _p, _c: {
            "ttft_ms": 60.0,
            "decode_tok_s": 55.0,
            "latency_ms": 140.0,
            "peak_mib": 9500,
            "size": 8800,
            "size_vram": 8800,
        },
    )[0]
    assert gen_q4["id"] != gen_q8["id"]
    assert "Q4_K_M" in gen_q4["id"]
    assert "Q8_0" in gen_q8["id"]

    fail_q4 = build_linux_failure_row(
        spec_id="C2-FILL-CARD",
        spec_rel="experiments/c2-fill-card/spec.yaml",
        run_id="r1",
        date="2026-10-24",
        task="generate",
        model="muse-glimmer:30b",
        revision_or_digest="9a44c2e1b8f07d55e1c3420a8812d0f7e6a91c34b208f71e5d3a2c490b118ef2",
        remote_device=t4,
        failure_class="load_refused",
        reason="needs 18000 MiB",
        quant="Q4_K_M",
        licence="apache-2.0",
    )
    fail_q8 = build_linux_failure_row(
        spec_id="C2-FILL-CARD",
        spec_rel="experiments/c2-fill-card/spec.yaml",
        run_id="r1",
        date="2026-10-24",
        task="generate",
        model="muse-glimmer:30b",
        revision_or_digest="9a44c2e1b8f07d55e1c3420a8812d0f7e6a91c34b208f71e5d3a2c490b118ef2",
        remote_device=t4,
        failure_class="load_refused",
        reason="needs 32000 MiB",
        quant="Q8_0",
        licence="apache-2.0",
    )
    assert fail_q4["id"] != fail_q8["id"]
    assert "Q4_K_M" in fail_q4["id"]
    assert "Q8_0" in fail_q8["id"]

    # Verify c2-generate covers multiple quants per (card, model) cell per C2-SWEEP-PLAN.md §1:
    gen_spec = load_spec(ROOT / "experiments" / "c2-generate" / "spec.yaml")
    by_card_model: dict[tuple[str, str], set[str]] = {}
    for entry in gen_spec["models"]:
        by_card_model.setdefault((entry["device"], entry["model"]), set()).add(entry["quant"])
    for (card, model_name), cell_quants in by_card_model.items():
        assert len(cell_quants) >= 3, f"({card}, {model_name}) in c2-generate only has {cell_quants}"

    # Verify run_linux_spec honours expected_outcome: load_refused without calling Ollama on refused cells,
    # and passes distinct quant-tagged model names to Ollama for runnable quants:
    queried_models: list[str] = []

    def fake_get(url: str) -> dict[str, Any]:
        if url.endswith("/api/version"):
            return {"version": "0.6.2"}
        if url.endswith("/api/ps"):
            return {"models": [{"name": queried_models[-1], "size": 5000, "size_vram": 5000}]}
        return {}

    def fake_post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("keep_alive") == 0:
            return {"done": True}
        queried_models.append(str(payload.get("model")))
        return {
            "prompt_eval_count": 512,
            "prompt_eval_duration": 10_000_000,
            "eval_count": 16,
            "eval_duration": 50_000_000,
        }

    class DummyProc:
        pid = 1234

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

    srv = LinuxOllamaServer(
        popen=lambda *_a, **_kw: DummyProc(),
        http_get=fake_get,
        http_post=fake_post,
    )
    srv.start()
    fill_spec = load_spec(ROOT / "experiments" / "c2-fill-card" / "spec.yaml")
    t4_rows = run_linux_spec(
        fill_spec,
        spec_path=ROOT / "experiments" / "c2-fill-card" / "spec.yaml",
        spec_rel="experiments/c2-fill-card/spec.yaml",
        run_id="t4-matrix",
        date="2026-10-24",
        remote_device=t4,
        sample_vram_mib=lambda: (6000, 15360),
        server=srv,
    )
    assert len({r["id"] for r in t4_rows}) == len(t4_rows)
    refused_rows = [r for r in t4_rows if r.get("outcome") == "load_refused"]
    assert len(refused_rows) >= 2
    assert not any(m.startswith("sha256:") for m in queried_models)
    assert "qwen3:8b-q4_K_M" in queried_models
    assert "qwen3:8b-q8_0" in queried_models


def test_linux_ollama_server_polls_until_ready_on_startup(tmp_path: Path, monkeypatch):
    import time

    from interlock.harness.linux_cuda import LinuxCudaError, LinuxOllamaServer

    class DummyProc:
        pid = 5555

        def __init__(self, exit_code: int | None = None) -> None:
            self._exit_code = exit_code

        def poll(self):
            return self._exit_code

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

    attempts = 0

    def flaky_http_get(url: str) -> dict[str, Any]:
        nonlocal attempts
        if url.endswith("/api/version"):
            attempts += 1
            if attempts < 3:
                raise ConnectionRefusedError("server still starting")
            return {"version": "0.6.2"}
        return {}

    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    srv = LinuxOllamaServer(
        log_path=tmp_path / "ollama.log",
        popen=lambda *_a, **_kw: DummyProc(),
        http_get=flaky_http_get,
        http_post=lambda _u, _p: {},
    )
    with srv:
        assert srv.version == "0.6.2"
    assert attempts == 3
    assert len(sleeps) == 2

    # If the child process exits before /api/version responds, startup raises LinuxCudaError immediately:
    dead_srv = LinuxOllamaServer(
        log_path=tmp_path / "dead.log",
        popen=lambda *_a, **_kw: DummyProc(exit_code=1),
        http_get=lambda _u: (_ for _ in ()).throw(ConnectionRefusedError("down")),
        http_post=lambda _u, _p: {},
    )
    with pytest.raises(LinuxCudaError, match="exited"):
        dead_srv.start()


def test_measure_linux_ollama_generate_runs_concurrent_requests_full_prompt_and_repeat_slots():
    from interlock.harness.linux_cuda import (
        LinuxOllamaServer,
        measure_linux_ollama_generate,
        run_linux_generate_ladder,
    )

    class DummyProc:
        pid = 7777

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

    prompts_seen: list[str] = []

    def fake_get(url: str) -> dict[str, Any]:
        if url.endswith("/api/version"):
            return {"version": "0.6.2"}
        if url.endswith("/api/ps"):
            return {"models": [{"name": "qwen3:8b", "size": 5200, "size_vram": 5200}]}
        return {}

    def fake_post(_url: str, payload: dict[str, Any]) -> dict[str, Any]:
        prompts_seen.append(str(payload.get("prompt") or ""))
        return {
            "prompt_eval_duration": 20_000_000,
            "eval_count": 128,
            "eval_duration": 1_000_000_000,
        }

    srv = LinuxOllamaServer(
        popen=lambda *_a, **_kw: DummyProc(),
        http_get=fake_get,
        http_post=fake_post,
    )
    srv.start()
    measured = measure_linux_ollama_generate(
        srv,
        lambda: (8000, 23034),
        model="qwen3:8b",
        prompt_tokens=2048,
        concurrency=4,
        num_predict=128,
    )
    # 1 warmup pass + 3 repeat passes, each dispatching 4 requests = 16 requests total
    assert len(prompts_seen) == 16
    assert len(prompts_seen[0].split()) >= 2048
    slots = measured.get("slots")
    assert isinstance(slots, list) and len(slots) == 4
    repeat_slots = [s for s in slots if s["phase"] == "repeat"]
    assert len(repeat_slots) == 3

    l4 = _remote_ident("L4", 23034)
    rows_with_slots = run_linux_generate_ladder(
        spec_id="C2-GENERATE",
        spec_rel="experiments/c2-generate/spec.yaml",
        run_id="r-slots",
        date="2026-10-24",
        remote_device=l4,
        model="qwen3:8b",
        revision_or_digest="500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41",
        prompt_lengths=(2048,),
        concurrencies=(4,),
        quant="Q4_K_M",
        licence="apache-2.0",
        measure_shape=lambda _p, _c: measured,
    )
    assert rows_with_slots[0]["repeats"] == 3
    assert len([s for s in rows_with_slots[0]["artifacts"]["slots"] if s["phase"] == "repeat"]) == 3

    # When a callback omits "slots", only 1 repeat slot is synthesized so repeats must be 1, never 3:
    rows_without_slots = run_linux_generate_ladder(
        spec_id="C2-GENERATE",
        spec_rel="experiments/c2-generate/spec.yaml",
        run_id="r-noslots",
        date="2026-10-24",
        remote_device=l4,
        model="qwen3:8b",
        revision_or_digest="500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41",
        prompt_lengths=(2048,),
        concurrencies=(1,),
        quant="Q4_K_M",
        licence="apache-2.0",
        measure_shape=lambda _p, _c: {
            "ttft_ms": 20.0,
            "decode_tok_s": 128.0,
            "latency_ms": 1020.0,
            "peak_mib": 8000,
            "size": 5200,
            "size_vram": 5200,
        },
    )
    assert rows_without_slots[0]["repeats"] == 1


def test_measure_linux_ollama_batch_dispatches_per_task_and_throughput_ladder_preserves_quant_licence_and_repeats():
    from interlock.harness.linux_cuda import (
        LinuxOllamaServer,
        measure_linux_ollama_batch,
        run_linux_throughput_ladder,
    )

    class DummyProc:
        pid = 8888

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

    called_paths: list[str] = []

    def fake_get(url: str) -> dict[str, Any]:
        if url.endswith("/api/version"):
            return {"version": "0.6.2"}
        if url.endswith("/api/ps"):
            return {"models": [{"name": "m", "size": 1200, "size_vram": 1200}]}
        return {}

    def fake_post(url: str, _payload: dict[str, Any]) -> dict[str, Any]:
        called_paths.append(url.split("11441", 1)[-1])
        return {"total_duration": 20_000_000}

    srv = LinuxOllamaServer(
        popen=lambda *_a, **_kw: DummyProc(),
        http_get=fake_get,
        http_post=fake_post,
    )
    srv.start()

    called_paths.clear()
    res = measure_linux_ollama_batch(
        srv,
        lambda: (2000, 23034),
        model="m",
        task="embed",
        batch=8,
    )
    assert called_paths[0] == "/api/embed"
    assert called_paths == ["/api/embed"] * 4
    assert len([s for s in res["slots"] if s["phase"] == "repeat"]) == 3

    l4 = _remote_ident("L4", 23034)
    tput_rows = run_linux_throughput_ladder(
        spec_id="C2-THROUGHPUT",
        spec_rel="experiments/c2-throughput/spec.yaml",
        run_id="r-tput",
        date="2026-10-24",
        remote_device=l4,
        models_by_task={"embed": ("BAAI/bge-base-en-v1.5", "a8f91b24e31d8c721094e3f6b5a08c19d74e2310")},
        tasks=("embed",),
        batch_sizes=(8,),
        quant="FP16",
        licence="mit",
        measure_batch=lambda _t, _b: {
            "throughput": 400.0,
            "warm_p50_ms": 20.0,
            "warm_p95_ms": 22.0,
            "peak_mib": 2000,
            "size": 1200,
            "size_vram": 1200,
        },
    )
    assert tput_rows[0]["model"]["quant"] == "FP16"
    assert tput_rows[0]["model"]["licence"] == "mit"
    assert tput_rows[0]["repeats"] == 1


def test_part_a_c2_spec_entries_each_have_one_source_verified_digests_and_unpublished_quant_refusals():
    from interlock.harness.linux_cuda import _ollama_model_tag

    fill_spec = load_spec(ROOT / "experiments" / "c2-fill-card" / "spec.yaml")
    gen_spec = load_spec(ROOT / "experiments" / "c2-generate" / "spec.yaml")
    tput_spec = load_spec(ROOT / "experiments" / "c2-throughput" / "spec.yaml")

    confirmed_ollama_digests = {
        "qwen3:8b-q4_K_M": "500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41",
        "qwen3:8b-q8_0": "e56358ca25dd14db6853a9f68a92d717aaa6f0a94250a72d1a0f3d86a9f30130",
        "qwen3:8b-fp16": "0b358e6e9d8cb39eaa4aa6b78136d09b91acb4be08c01bb707367811caed69ae",
        "phi3.5:3.8b-mini-instruct-q4_K_M": "57096159698481b47ea17995d1a71f28eea1e26fc801df39a841b29badb4aa63",
        "phi3.5:3.8b-mini-instruct-q6_K": "64777e5c68030bbf6eba5761cc6efe22a69b2f9a46fdce942d2b50181dff787a",
        "phi3.5:3.8b-mini-instruct-q8_0": "8b50e8e1e21653ad0085ed8209f4f0cf7f9af78e0352ebd2d303f7b604d313be",
        "phi3.5:3.8b-mini-instruct-fp16": "4611ed13d4961f1ae0dab1b2e1c50f70777e21bdf97be507cc27312aa412d15a",
        "gemma4:31b-it-q4_K_M": "17ba34c06c801ba22f6047bab21ced9429c6d43e0e42ecf339b0f3e5a61e8d97",
        "gemma4:31b-it-q8_0": "ac97b596ee1f1eca8cc5e35b7aab3226051c0204cd8324f3de4e0bb46b00c533",
        "gemma4:31b-it-bf16": "df1661a2a6aaa94faf24bbaf23a691877a27d0e6052598f105a225618a079e76",
        "muse-glimmer:30b-q4_K_M": "de878ce33ad81d060001db1469a02eebe4d86f0ad58cfe52dc062fdcbe4464c1",
        "muse-glimmer:30b-q8_0": "6ffafcdbd728b7d7ce94ad7db12c2b8a5ba4a1fdaad20c0d4b45a1257458485a",
        "muse-glimmer:30b-bf16": "318fb1b9b81696d14f4f2784b3bda2e4e633046a52fcc9fe0b94e74f889f5efe",
        "nomic-embed-text:137m-v1.5-fp16": "0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f",
        "qwen3-embedding:0.6b-fp16": "67a7592a8852b7b36f6328cdec193cfb7639769f5edf344ba709b088b2a51042",
    }

    for spec in (fill_spec, gen_spec, tput_spec):
        for entry in spec["models"]:
            has_tag = bool(entry.get("ollama_tag"))
            has_gguf = isinstance(entry.get("gguf"), dict)
            is_declared_refusal = entry.get("expected_outcome") == "load_refused" and (
                str(entry.get("failure_reason") or "").startswith("no published ")
                or "out of scope" in str(entry.get("failure_reason") or "").lower()
            )
            if has_tag or has_gguf:
                assert has_tag != has_gguf, f"entry {entry} must name at most one of ollama_tag or gguf"
                if has_tag:
                    tag = entry["ollama_tag"]
                    assert tag in confirmed_ollama_digests, f"unverified ollama_tag {tag!r}"
                    assert entry["revision_or_digest"] == confirmed_ollama_digests[tag]
                if has_gguf:
                    gguf = entry["gguf"]
                    assert set(gguf) >= {"repo", "revision", "file", "sha256"}
                    assert len(str(gguf["revision"])) == 40
                    assert len(str(gguf["sha256"])) == 64
            else:
                assert is_declared_refusal, (
                    f"entry {entry['model']} ({entry.get('quant')}) on {entry['device']} lacks a source"
                )

    gemma_hash = "sha256:6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7"
    for spec in (fill_spec, gen_spec):
        for entry in spec["models"]:
            if entry["model"] == gemma_hash and entry["quant"] == "Q4_K_M":
                assert entry["ollama_tag"] == "gemma4:31b-it-q4_K_M"
                derived_tag = _ollama_model_tag(entry["model"], entry["quant"], entry.get("ollama_tag"))
                assert derived_tag == "gemma4:31b-it-q4_K_M"
            if entry["model"] == gemma_hash and entry["quant"] == "Q6_K":
                assert entry.get("expected_outcome") == "load_refused"
                assert entry.get("failure_reason") == "no published Q6_K build"
            if entry["model"] == "muse-glimmer:30b" and entry["quant"] == "Q6_K":
                assert entry.get("expected_outcome") == "load_refused"
                assert entry.get("failure_reason") == "no published Q6_K build"

    base_linux = yaml.safe_load((ROOT / "experiments" / "c2-fill-card" / "spec.yaml").read_text(encoding="utf-8"))
    with pytest.raises(SpecError, match="cannot name both"):
        validate_spec(
            {
                **base_linux,
                "models": [
                    {
                        **base_linux["models"][0],
                        "ollama_tag": "qwen3:8b-q4_K_M",
                        "gguf": {
                            "repo": "Qwen/Qwen3-8B-GGUF",
                            "revision": "7c41481f57cb95916b40956ab2f0b139b296d974",
                            "file": "Qwen3-8B-Q6_K.gguf",
                            "sha256": "cb042ccd76795a8830d6be6bd4165245847cc68e41797b13bd61aed4c2cfbce6",
                        },
                    }
                ],
            }
        )
    with pytest.raises(SpecError, match="must name exactly one source"):
        validate_spec(
            {
                **base_linux,
                "models": [
                    {
                        "model": "qwen3:8b",
                        "revision_or_digest": "500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41",
                        "quant": "Q4_K_M",
                        "licence": "apache-2.0",
                        "engine": "ollama",
                        "device": "colab:T4",
                    }
                ],
            }
        )
    with pytest.raises(SpecError, match="40-hex"):
        validate_spec(
            {
                **base_linux,
                "models": [
                    {
                        "model": "qwen3:8b",
                        "revision_or_digest": "cb042ccd76795a8830d6be6bd4165245847cc68e41797b13bd61aed4c2cfbce6",
                        "quant": "Q6_K",
                        "licence": "apache-2.0",
                        "engine": "ollama",
                        "device": "colab:T4",
                        "gguf": {
                            "repo": "Qwen/Qwen3-8B-GGUF",
                            "revision": "main",
                            "file": "Qwen3-8B-Q6_K.gguf",
                            "sha256": "cb042ccd76795a8830d6be6bd4165245847cc68e41797b13bd61aed4c2cfbce6",
                        },
                    }
                ],
            }
        )
    with pytest.raises(SpecError, match="64-hex"):
        validate_spec(
            {
                **base_linux,
                "models": [
                    {
                        "model": "qwen3:8b",
                        "revision_or_digest": "cb042ccd76795a8830d6be6bd4165245847cc68e41797b13bd61aed4c2cfbce6",
                        "quant": "Q6_K",
                        "licence": "apache-2.0",
                        "engine": "ollama",
                        "device": "colab:T4",
                        "gguf": {
                            "repo": "Qwen/Qwen3-8B-GGUF",
                            "revision": "7c41481f57cb95916b40956ab2f0b139b296d974",
                            "file": "Qwen3-8B-Q6_K.gguf",
                            "sha256": "short",
                        },
                    }
                ],
            }
        )
    with pytest.raises(SpecError, match="invalid ollama_tag"):
        validate_spec(
            {
                **base_linux,
                "models": [
                    {
                        "model": "qwen3:8b",
                        "revision_or_digest": "500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41",
                        "quant": "Q4_K_M",
                        "licence": "apache-2.0",
                        "engine": "ollama",
                        "device": "colab:T4",
                        "ollama_tag": "   ",
                    }
                ],
            }
        )
    with pytest.raises(SpecError, match="gguf must be a mapping"):
        validate_spec(
            {
                **base_linux,
                "models": [
                    {
                        "model": "qwen3:8b",
                        "revision_or_digest": "cb042ccd76795a8830d6be6bd4165245847cc68e41797b13bd61aed4c2cfbce6",
                        "quant": "Q6_K",
                        "licence": "apache-2.0",
                        "engine": "ollama",
                        "device": "colab:T4",
                        "gguf": "not-a-dict",
                    }
                ],
            }
        )
    with pytest.raises(SpecError, match="gguf is missing file"):
        validate_spec(
            {
                **base_linux,
                "models": [
                    {
                        "model": "qwen3:8b",
                        "revision_or_digest": "cb042ccd76795a8830d6be6bd4165245847cc68e41797b13bd61aed4c2cfbce6",
                        "quant": "Q6_K",
                        "licence": "apache-2.0",
                        "engine": "ollama",
                        "device": "colab:T4",
                        "gguf": {
                            "repo": "Qwen/Qwen3-8B-GGUF",
                            "revision": "7c41481f57cb95916b40956ab2f0b139b296d974",
                            "sha256": "cb042ccd76795a8830d6be6bd4165245847cc68e41797b13bd61aed4c2cfbce6",
                        },
                    }
                ],
            }
        )


def test_part_b_c2_rerank_and_nli_are_declared_out_of_scope_rows_and_ollama_batch_refuses_non_embed():
    from interlock.harness.linux_cuda import (
        LinuxCudaError,
        LinuxOllamaServer,
        _batch_endpoint_and_payload,
        run_linux_spec,
    )

    tput_spec = load_spec(ROOT / "experiments" / "c2-throughput" / "spec.yaml")
    for entry in tput_spec["models"]:
        if entry["task"] in ("rerank", "nli"):
            assert entry.get("expected_outcome") == "load_refused"
            reason = str(entry.get("failure_reason") or "")
            assert "out of scope" in reason.lower()

    for bad_task in ("rerank", "nli"):
        with pytest.raises(LinuxCudaError, match="does not serve"):
            _batch_endpoint_and_payload("m", bad_task, 8)

    class DummyProc:
        pid = 9999

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

    called_urls: list[str] = []

    def fake_get(url: str) -> dict[str, Any]:
        if url.endswith("/api/version"):
            return {"version": "0.6.2"}
        if url.endswith("/api/ps"):
            return {"models": [{"name": "m", "size": 500, "size_vram": 500}]}
        return {}

    def fake_post(url: str, _payload: dict[str, Any]) -> dict[str, Any]:
        called_urls.append(url)
        return {"total_duration": 10_000_000}

    srv = LinuxOllamaServer(
        popen=lambda *_a, **_kw: DummyProc(),
        http_get=fake_get,
        http_post=fake_post,
    )
    srv.start()
    l4 = _remote_ident("L4", 23034)
    rows = run_linux_spec(
        tput_spec,
        spec_path=ROOT / "experiments" / "c2-throughput" / "spec.yaml",
        spec_rel="experiments/c2-throughput/spec.yaml",
        run_id="r-part-b",
        date="2026-10-24",
        remote_device=l4,
        sample_vram_mib=lambda: (2000, 23034),
        server=srv,
    )
    assert not any("/api/rerank" in u or "/api/nli" in u for u in called_urls)
    rerank_nli_rows = [r for r in rows if r["task"] in ("rerank", "nli")]
    assert len(rerank_nli_rows) >= 6
    for r in rerank_nli_rows:
        validate_row(r)
        assert r["outcome"] == "load_refused"
        assert "out of scope" in r["settings"]["failure_reason"].lower()


def test_part_c_throughput_inputs_use_fixed_token_lengths_128_and_512_with_distinct_row_ids():
    from interlock.harness.linux_cuda import (
        THROUGHPUT_TOKEN_LENGTHS,
        LinuxCudaError,
        LinuxOllamaServer,
        _batch_endpoint_and_payload,
        _build_fixed_token_item,
        _ollama_model_tag,
        measure_linux_ollama_batch,
        run_linux_throughput_ladder,
    )

    assert THROUGHPUT_TOKEN_LENGTHS == (128, 512)
    tput_spec = load_spec(ROOT / "experiments" / "c2-throughput" / "spec.yaml")
    assert tuple(tput_spec.get("token_lengths") or ()) == (128, 512)
    with pytest.raises(SpecError, match="token_lengths"):
        validate_spec({**tput_spec, "token_lengths": [512]})
    with pytest.raises(LinuxCudaError, match="positive integer"):
        _build_fixed_token_item(0)
    with pytest.raises(LinuxCudaError, match="unsupported batch task"):
        _batch_endpoint_and_payload("m", "generate", 1)
    assert _ollama_model_tag("BAAI/bge-base-en-v1.5", "unrecorded") == "baai-bge-base-en-v1.5"

    ep_128, payload_128 = _batch_endpoint_and_payload("m", "embed", 4, token_length=128)
    ep_512, payload_512 = _batch_endpoint_and_payload("m", "embed", 4, token_length=512)
    assert ep_128 == "/api/embed" and ep_512 == "/api/embed"
    assert len(payload_128["input"]) == 4 and len(payload_512["input"]) == 4
    assert len(payload_128["input"][0].split()) == 128
    assert len(payload_512["input"][0].split()) == 512
    assert payload_128["input"][0] != "item-0"

    l4 = _remote_ident("L4", 23034)
    nomic_sha = "0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f"
    rows = run_linux_throughput_ladder(
        spec_id="C2-THROUGHPUT",
        spec_rel="experiments/c2-throughput/spec.yaml",
        run_id="r-len",
        date="2026-10-24",
        remote_device=l4,
        models_by_task={"embed": ("nomic-ai/nomic-embed-text-v1.5", nomic_sha)},
        tasks=("embed",),
        batch_sizes=(8,),
        token_lengths=(128, 512),
        quant="fp16",
        licence="apache-2.0",
        measure_batch=lambda _t, _b, **_kw: {
            "throughput": 250.0,
            "warm_p50_ms": 32.0,
            "warm_p95_ms": 35.0,
            "peak_mib": 1800,
            "size": 550,
            "size_vram": 550,
        },
    )
    assert len(rows) == 2
    row_128, row_512 = rows[0], rows[1]
    assert row_128["id"] != row_512["id"]
    assert "/l128-b8/" in row_128["id"]
    assert "/l512-b8/" in row_512["id"]
    assert row_128["settings"]["prompt_tokens"] == 128
    assert row_512["settings"]["prompt_tokens"] == 512
    assert row_128["settings"]["static_shape"] == "8x128"
    assert row_512["settings"]["static_shape"] == "8x512"

    # Also verify measure_linux_ollama_batch forwards token_length to payload generation:
    class DummyProc:
        pid = 4321

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

    seen_inputs: list[list[str]] = []
    srv = LinuxOllamaServer(
        popen=lambda *_a, **_kw: DummyProc(),
        http_get=lambda url: {"version": "0.6.2"}
        if url.endswith("/api/version")
        else {"models": [{"name": "m", "size": 500, "size_vram": 500}]},
        http_post=lambda _url, payload: (seen_inputs.append(list(payload["input"])), {"total_duration": 10_000_000})[1],
    )
    srv.start()
    measure_linux_ollama_batch(srv, lambda: (1500, 23034), model="m", task="embed", batch=2, token_length=128)
    assert len(seen_inputs[-1][0].split()) == 128


def test_part_d_c2_sources_cli_resolves_every_c2_entry_to_one_source_and_command(capsys, tmp_path):
    import json

    from interlock.cli import main as cli_main
    from interlock.harness.linux_cuda import LinuxCudaError, format_c2_sources, resolve_c2_sources

    for spec_rel in (
        "experiments/c2-fill-card/spec.yaml",
        "experiments/c2-generate/spec.yaml",
        "experiments/c2-throughput/spec.yaml",
    ):
        spec_path = ROOT / spec_rel
        spec = load_spec(spec_path)
        resolved = resolve_c2_sources(spec)
        assert len(resolved) == len(spec["models"])
        for entry, item in zip(spec["models"], resolved, strict=True):
            assert item["device"] == entry["device"]
            assert item["model"] == entry["model"]
            assert item["source_kind"] in ("ollama_tag", "gguf", "load_refused")
            if item["source_kind"] == "ollama_tag":
                assert item["expected_digest"] == entry["revision_or_digest"]
                assert item["served_tag"] == entry["ollama_tag"]
                assert item["command"] == f"ollama pull {entry['ollama_tag']}"
            elif item["source_kind"] == "gguf":
                gguf = entry["gguf"]
                assert item["expected_digest"] == gguf["sha256"]
                served = item["served_tag"]
                assert served and not served.startswith("sha256:") and "/" not in served
                assert f"hf download {gguf['repo']} {gguf['file']} --revision {gguf['revision']}" in item["command"]
                assert f"ollama create {served} -f Modelfile" in item["command"]
            else:
                assert item["expected_digest"] == entry["revision_or_digest"]
                assert item["served_tag"] is None
                assert item["command"].startswith("skip-pull (")

        text_out = format_c2_sources(spec)
        assert len(text_out.strip().splitlines()) == len(spec["models"])

        rc = cli_main(["c2-sources", str(spec_path)])
        assert rc == 0
        captured = capsys.readouterr()
        assert len(captured.out.strip().splitlines()) == len(spec["models"])

        rc_json = cli_main(["c2-sources", str(spec_path), "--json"])
        assert rc_json == 0
        parsed_json = json.loads(capsys.readouterr().out)
        assert len(parsed_json) == len(spec["models"])

    # Invalid / non-linux-cuda spec or malformed entry refuses cleanly:
    with pytest.raises(LinuxCudaError, match="linux-cuda"):
        resolve_c2_sources({"platform": "windows", "models": []})
    with pytest.raises(LinuxCudaError, match="non-empty"):
        resolve_c2_sources({"platform": "linux-cuda", "models": []})
    with pytest.raises(LinuxCudaError, match="cannot name both"):
        resolve_c2_sources(
            {
                "platform": "linux-cuda",
                "models": [
                    {
                        "device": "colab:T4",
                        "model": "qwen3:8b",
                        "quant": "Q4_K_M",
                        "ollama_tag": "qwen3:8b-q4_K_M",
                        "gguf": {"repo": "r", "revision": "a" * 40, "file": "f", "sha256": "b" * 64},
                        "revision_or_digest": "500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41",
                    }
                ],
            }
        )
    with pytest.raises(LinuxCudaError, match="exactly one source"):
        resolve_c2_sources(
            {
                "platform": "linux-cuda",
                "models": [
                    {
                        "device": "colab:T4",
                        "model": "qwen3:8b",
                        "quant": "Q4_K_M",
                        "revision_or_digest": "500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41",
                    }
                ],
            }
        )
    bad_spec = tmp_path / "bad.yaml"
    bad_spec.write_text("not: a: valid: spec\n", encoding="utf-8")
    assert cli_main(["c2-sources", str(bad_spec)]) == 1
    capsys.readouterr()


def test_fix1_gemma4_31b_real_ollama_sources_and_measured_refusal_reasons():
    fill_spec = load_spec(ROOT / "experiments" / "c2-fill-card" / "spec.yaml")
    gen_spec = load_spec(ROOT / "experiments" / "c2-generate" / "spec.yaml")
    gemma_hash = "sha256:6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7"

    for spec in (fill_spec, gen_spec):
        gemma_entries = [e for e in spec["models"] if e["model"] == gemma_hash]
        assert len(gemma_entries) == 20
        for entry in gemma_entries:
            q = entry["quant"]
            card = entry["device"]
            assert "gguf" not in entry, f"Gemma 4 31B {q} on {card} still uses gguf={entry.get('gguf')}"
            if q == "Q4_K_M":
                assert entry.get("ollama_tag") == "gemma4:31b-it-q4_K_M"
                assert (
                    entry["revision_or_digest"]
                    == "17ba34c06c801ba22f6047bab21ced9429c6d43e0e42ecf339b0f3e5a61e8d97"
                )
                if card == "colab:T4":
                    assert entry.get("expected_outcome") == "load_refused"
                    assert "17805 MiB" in str(entry.get("failure_reason"))
                else:
                    assert entry.get("expected_outcome") is None
            elif q == "Q6_K":
                assert entry.get("expected_outcome") == "load_refused"
                assert entry.get("failure_reason") == "no published Q6_K build"
                assert (
                    entry["revision_or_digest"]
                    == "17ba34c06c801ba22f6047bab21ced9429c6d43e0e42ecf339b0f3e5a61e8d97"
                )
            elif q == "Q8_0":
                assert entry.get("ollama_tag") == "gemma4:31b-it-q8_0"
                assert (
                    entry["revision_or_digest"]
                    == "ac97b596ee1f1eca8cc5e35b7aab3226051c0204cd8324f3de4e0bb46b00c533"
                )
                if card in ("colab:T4", "colab:L4"):
                    assert entry.get("expected_outcome") == "load_refused"
                    assert "31124 MiB" in str(entry.get("failure_reason"))
                else:
                    assert entry.get("expected_outcome") is None
                    assert entry.get("failure_reason") is None
            elif q == "BF16":
                assert entry.get("ollama_tag") == "gemma4:31b-it-bf16"
                assert (
                    entry["revision_or_digest"]
                    == "df1661a2a6aaa94faf24bbaf23a691877a27d0e6052598f105a225618a079e76"
                )
                if card in ("colab:T4", "colab:L4", "colab:A100-40"):
                    assert entry.get("expected_outcome") == "load_refused"
                    assert "58568 MiB" in str(entry.get("failure_reason"))
                else:
                    assert entry.get("expected_outcome") is None
                    assert entry.get("failure_reason") is None

        refused_by_card: dict[str, list[dict[str, Any]]] = {
            "colab:A100-40": [],
            "colab:A100-80": [],
            "colab:G4": [],
            "colab:L4": [],
            "colab:T4": [],
        }
        for entry in spec["models"]:
            if isinstance(entry.get("gguf"), dict):
                assert not str(entry["gguf"]["repo"]).startswith("sha256:")
                assert "/" in str(entry["gguf"]["repo"])
            if entry.get("expected_outcome") == "load_refused":
                reason = str(entry.get("failure_reason") or "").strip()
                assert reason, f"missing failure_reason on {entry['device']} {entry['model']} {entry['quant']}"
                assert "no published " in reason or "MiB" in reason or "out of scope" in reason.lower()
                refused_by_card[entry["device"]].append(entry)

        assert len(refused_by_card["colab:T4"]) == 9
        assert len(refused_by_card["colab:L4"]) == 6
        assert len(refused_by_card["colab:A100-40"]) == 4
        assert len(refused_by_card["colab:A100-80"]) == 2
        assert len(refused_by_card["colab:G4"]) == 2

    base_linux = yaml.safe_load((ROOT / "experiments" / "c2-fill-card" / "spec.yaml").read_text(encoding="utf-8"))
    with pytest.raises(SpecError, match="gguf.repo"):
        validate_spec(
            {
                **base_linux,
                "models": [
                    {
                        "model": gemma_hash,
                        "revision_or_digest": "280af6832eca23cb322c4dcc65edfea98a21b8f8ab07dc7553bd6f7e6e7a3313",
                        "quant": "Q4_K_M",
                        "licence": "apache-2.0",
                        "engine": "ollama",
                        "device": "colab:L4",
                        "gguf": {
                            "repo": gemma_hash,
                            "revision": "c9f3b814e2076d51a48c920b1e73f60d5a4219e8",
                            "file": "gemma-4-31B-it-Q4_K_M.gguf",
                            "sha256": "280af6832eca23cb322c4dcc65edfea98a21b8f8ab07dc7553bd6f7e6e7a3313",
                        },
                    }
                ],
            }
        )
    with pytest.raises(SpecError, match="failure_reason"):
        validate_spec(
            {
                **base_linux,
                "models": [
                    {
                        "model": "qwen3:8b",
                        "ollama_tag": "qwen3:8b-fp16",
                        "revision_or_digest": "0b358e6e9d8cb39eaa4aa6b78136d09b91acb4be08c01bb707367811caed69ae",
                        "quant": "FP16",
                        "licence": "apache-2.0",
                        "engine": "ollama",
                        "device": "colab:T4",
                        "expected_outcome": "load_refused",
                    }
                ],
            }
        )











