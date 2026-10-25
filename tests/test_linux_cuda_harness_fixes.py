"""Tests for the 1.0.2 Linux CUDA harness fixes found reviewing the C2 launcher.

Four defects: an nvidia-smi failure ended `interlock run` in a raw traceback; a model stayed loaded
while the next quant took its VRAM baseline; the receipt was written once at the end, so a lost VM
lost every finished cell; and `run` could not name a quant. No GPU and no Colab session: fakes only.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from interlock.cli import main
from interlock.harness.runner import run_spec

ROOT = Path(__file__).resolve().parents[1]
SPEC_RELS = (
    "experiments/c2-fill-card/spec.yaml",
    "experiments/c2-generate/spec.yaml",
    "experiments/c2-throughput/spec.yaml",
)
FILL = "experiments/c2-fill-card/spec.yaml"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    for spec_rel in SPEC_RELS:
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
    return tmp_path


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


class Ollama:
    """A fake Ollama: it records every generate and every unload."""

    def __init__(self) -> None:
        self.generated: list[str] = []
        self.unloaded: list[str] = []

    def get(self, url: str) -> dict[str, Any]:
        if url.endswith("/api/version"):
            return {"version": "0.6.2"}
        resident = self.generated[-1] if self.generated else "x"
        return {"models": [{"name": resident, "size": 5200000000, "size_vram": 5200000000}]}

    def post(self, _url: str, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("keep_alive") == 0:
            self.unloaded.append(payload["model"])
            return {"done": True}
        self.generated.append(payload["model"])
        ctx = int((payload.get("options") or {}).get("num_ctx") or 2048)
        if ctx > 8192:
            return {"error": f"request ({ctx} tokens) exceeds the available context size (8192 tokens)"}
        return {
            "load_duration": 20_000_000,
            "prompt_eval_count": int(ctx * 0.9),
            "prompt_eval_duration": 100_000_000,
            "eval_count": 16,
            "eval_duration": 250_000_000,
        }


def _seams(ollama: Ollama, **extra: Any) -> dict[str, Any]:
    return {
        "nvml": FakeNvml(),
        "pool": "us-central1-a",
        "popen": lambda *_a, **_kw: FakeProc(),
        "http_get": ollama.get,
        "http_post": ollama.post,
        **extra,
    }


def _run(repo: Path, run_id: str, ollama: Ollama, **kw: Any) -> list[dict]:
    seams = kw.pop("seams", None) or _seams(ollama)
    return run_spec(
        repo / FILL,
        root=repo,
        run_id=run_id,
        date="2026-10-25",
        python_path=str(repo.parent / "external-python"),
        seams=seams,
        **kw,
    )


# ── 1. nvidia-smi failure is a structured refusal ────────────────────────────────────────────────


class _NoNvml:
    """NVML that is installed but finds no GPU."""

    def nvmlInit(self):
        raise RuntimeError("NVML Shared Library Not Found")


@pytest.mark.parametrize(
    "error",
    [
        subprocess.CalledProcessError(9, ["nvidia-smi"], stderr="NVIDIA-SMI has failed"),
        FileNotFoundError(2, "No such file or directory: 'nvidia-smi'"),
    ],
    ids=["CalledProcessError", "FileNotFoundError"],
)
def test_run_without_an_nvidia_gpu_is_a_structured_refusal(repo, capsys, monkeypatch, error):
    import interlock.harness.linux_cuda as lc

    def broken_smi(_argv):
        raise error

    monkeypatch.setitem(sys.modules, "pynvml", _NoNvml())
    monkeypatch.setattr(lc, "_default_nvidia_smi_runner", broken_smi)
    code = main(["run", str(repo / FILL), "--root", str(repo), "--run-id", "refused"])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.startswith("REFUSED: no NVIDIA GPU")
    assert "Traceback" not in captured.err + captured.out


# ── 2. each quant is unloaded; ps names tolerate :latest; a stuck server is killed ───────────────


def test_each_prepared_quant_is_unloaded_after_its_cells(repo):
    ollama = Ollama()
    _run(repo, "unload", ollama, models=["qwen3:8b"])
    served = list(dict.fromkeys(ollama.generated))
    assert len(served) >= 2
    assert ollama.unloaded == served


def test_a_quant_is_unloaded_even_when_a_cell_raises():
    from interlock.harness.linux_cuda import run_linux_spec

    class Server:
        version = "0.6.2"

        def __init__(self) -> None:
            self.unloaded: list[str] = []

        def get(self, _path):
            return {}

        def post(self, _path, _payload):
            return {}

        def unload(self, model: str) -> None:
            self.unloaded.append(model)

    spec = {
        "id": "T",
        "procedure": "fill-card",
        "platform": "linux-cuda",
        "models": [{"model": "m", "repo_or_tag": "m", "quant": "Q4_K_M", "ollama_tag": "m:q4", "device": "colab:L4"}],
    }

    def sampler():
        raise RuntimeError("sampler lost")

    server = Server()
    with pytest.raises(RuntimeError, match="sampler lost"):
        run_linux_spec(
            spec,
            spec_path=Path("experiments/t/spec.yaml"),
            run_id="r",
            date="2026-10-25",
            remote_device=_remote_ident(),
            sample_vram_mib=sampler,
            server=server,  # type: ignore[arg-type]
        )
    assert server.unloaded == ["m:q4"]


def _remote_ident():
    from interlock.devices import RemoteDeviceIdentity

    return RemoteDeviceIdentity(
        provider="colab",
        card="L4",
        vram_mib=23034,
        driver_version="550.54.15",
        cuda_version="12.4",
        pool="us-central1-a",
    )


def test_ps_query_matches_a_latest_tag_with_two_models_resident():
    from interlock.harness.linux_cuda import _query_ps_entry

    class Server:
        def get(self, _path):
            return {
                "models": [
                    {"name": "other:q4", "model": "other:q4", "size": 1, "size_vram": 1},
                    {"name": "foo:latest", "model": "foo:latest", "size": 700, "size_vram": 600},
                ]
            }

    assert _query_ps_entry(Server(), "foo") == (700, 600)  # type: ignore[arg-type]


def test_stopping_a_server_that_ignores_terminate_kills_it():
    from interlock.harness.linux_cuda import LinuxOllamaServer

    class Stuck:
        killed = False
        waits = 0

        def poll(self):
            return None

        def terminate(self):
            return None

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            self.waits += 1
            if timeout is not None and not self.killed:
                raise subprocess.TimeoutExpired("ollama", timeout)
            return -9

    server = LinuxOllamaServer(popen=lambda *_a, **_kw: Stuck())
    server._proc = Stuck()
    server.stop()
    assert server._proc.killed is True
    assert server._proc.waits == 2


# ── 3. the receipt holds every finished cell ─────────────────────────────────────────────────────


def test_a_lost_run_keeps_the_cells_it_finished(repo):
    ollama = Ollama()

    def sampler():
        if len(set(ollama.generated)) >= 2:
            raise RuntimeError("VM lost")
        return (10240, 23034)

    with pytest.raises(RuntimeError, match="VM lost"):
        _run(
            repo,
            "lost",
            ollama,
            models=["qwen3:8b"],
            seams=_seams(ollama, remote_device=_remote_ident(), sample_vram_mib=sampler),
        )
    lines = (repo / "receipts" / "lost.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    assert len(rows) >= 1
    assert all(row["device"] == "colab:L4" for row in rows)


def test_a_completed_run_writes_the_same_bytes_as_one_final_write(repo):
    from interlock.harness.receipts import _dump

    ollama = Ollama()
    rows = _run(repo, "whole", ollama, models=["qwen3:8b"])
    expected = repo / "expected.jsonl"
    _dump(expected, rows)
    assert (repo / "receipts" / "whole.jsonl").read_bytes() == expected.read_bytes()


# ── 4. --quants ──────────────────────────────────────────────────────────────────────────────────


def test_run_measures_only_the_named_quants(repo):
    ollama = Ollama()
    rows = _run(repo, "q", ollama, models=["qwen3:8b"], quants=["Q4_K_M", "Q8_0"])
    assert {row["model"]["quant"] for row in rows} == {"Q4_K_M", "Q8_0"}


def test_run_without_quants_measures_every_quant(repo):
    ollama = Ollama()
    rows = _run(repo, "all", ollama, models=["qwen3:8b"])
    assert {row["model"]["quant"] for row in rows} >= {"Q4_K_M", "Q6_K", "Q8_0"}


def test_an_unknown_quant_is_refused_with_the_spec_quants(repo, capsys):
    code = main(
        ["run", str(repo / FILL), "--root", str(repo), "--models", "qwen3:8b", "--quants", "Q9_9", "--run-id", "bad"]
    )
    err = capsys.readouterr().err
    assert code == 1
    assert "Q9_9" in err
    assert "Q4_K_M" in err
    assert not (repo / "receipts" / "bad.jsonl").exists()


def test_quants_parse_and_dry_run_lists_them(repo, capsys):
    argv = ["run", str(repo / FILL), "--root", str(repo), "--dry-run", "--models", "qwen3:8b", "--quants", "Q4_K_M"]
    code = main(argv)
    assert code == 0
    assert "quants measured: Q4_K_M" in capsys.readouterr().out
