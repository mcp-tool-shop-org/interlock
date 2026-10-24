"""Tests for the Linux CUDA measuring path and committed C2 ladder specs.

All tests run on recorded nvidia-smi / NVML fixtures and a mocked Ollama.
No live GPU, no Windows performance counters, and no Colab session.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from interlock.devices import RemoteDeviceIdentity
from interlock.harness.spec import SpecError, load_spec, validate_spec
from interlock.receipt import validate_row

ROOT = Path(__file__).resolve().parents[1]

NVIDIA_SMI_QUERY_L4 = "NVIDIA L4, 1420, 23034, 550.54.15\n"
NVIDIA_SMI_HEADER_CUDA_12_4 = (
    "==============NVSMI LOG==============\n"
    "Driver Version                            : 550.54.15\n"
    "CUDA Version                              : 12.4\n"
)


def _remote_ident(card: str = "L4", vram_mib: int = 23034) -> RemoteDeviceIdentity:
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
    assert map_card_name("NVIDIA RTX PRO 6000 Blackwell", 24576) == "G4"

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

