import sys
import types

import pytest

from interlock.harness.e4b import E4bError, capture_run, read_git
from interlock.harness.ollama_serve import ServeError, pin_install


def test_the_live_runner_returns_the_child_stdout():
    # test_fake_hides_platform: the fakes returned stdout, the live default (bare subprocess.run) returned None.
    completed = capture_run([sys.executable, "-c", "print('pin-report')"])
    assert completed.stdout.strip() == "pin-report"

SHA = "0123456789abcdef0123456789abcdef01234567"


def _done(stdout, code=0):
    return types.SimpleNamespace(stdout=stdout, returncode=code)


def test_read_git_refuses_a_runner_that_captured_nothing(tmp_path):
    with pytest.raises(E4bError, match="no output"):
        read_git(tmp_path, lambda args: _done(None))


def test_read_git_refuses_a_non_zero_git_call(tmp_path):
    with pytest.raises(E4bError, match="exit code 128"):
        read_git(tmp_path, lambda args: _done(SHA, 0) if "rev-parse" in args else _done("", 128))
    with pytest.raises(E4bError, match="exit code 1"):
        read_git(tmp_path, lambda args: _done("", 1))


def test_read_git_refuses_a_short_sha(tmp_path):
    with pytest.raises(E4bError, match="40-hex"):
        read_git(tmp_path, lambda args: _done("abc\n") if "rev-parse" in args else _done(""))


def test_read_git_is_clean_only_on_a_successful_empty_porcelain(tmp_path):
    clean = read_git(tmp_path, lambda args: _done(SHA + "\n") if "rev-parse" in args else _done(""))
    assert clean == {"git_sha": SHA, "clean_tree": True, "porcelain": []}
    dirty = read_git(tmp_path, lambda args: _done(SHA) if "rev-parse" in args else _done(" M x\n"))
    assert dirty["clean_tree"] is False


def test_pin_install_refuses_an_empty_cuda_sha(tmp_path):
    exe = tmp_path / "ollama.exe"
    exe.write_text("")
    models = tmp_path / "models"
    models.mkdir()
    lib = tmp_path / "lib" / "ollama" / "cuda_v13"
    lib.mkdir(parents=True)
    (lib / "ggml-cuda.dll").write_text("")
    with pytest.raises(ServeError, match="64-hex"):
        pin_install(exe, models, read_sha=lambda path: "")
    assert pin_install(exe, models, read_sha=lambda path: "f" * 64).cuda_sha256 == "f" * 64
