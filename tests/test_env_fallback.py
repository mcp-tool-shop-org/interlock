"""INTERLOCK_* is the name. SWITCHYARD_* still works once, and warns."""

from __future__ import annotations

import json

from interlock.harness.fill import _configured_dll, _read_configured_log
from interlock.ollama_supervisor import _OLD_ENV_WARNED, load_serve_start, read_project_env


def _clear(monkeypatch):
    _OLD_ENV_WARNED.clear()
    for name in (
        "INTERLOCK_OLLAMA_LOG",
        "INTERLOCK_OLLAMA_EXE",
        "INTERLOCK_OLLAMA_SERVE",
        "SWITCHYARD_OLLAMA_LOG",
        "SWITCHYARD_OLLAMA_EXE",
        "SWITCHYARD_OLLAMA_SERVE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_an_old_environment_name_warns_and_is_read(monkeypatch, capsys):
    _clear(monkeypatch)
    monkeypatch.setenv("SWITCHYARD_OLLAMA_LOG", "kept")
    assert read_project_env("INTERLOCK_OLLAMA_LOG") == "kept"
    assert "SWITCHYARD_OLLAMA_LOG is deprecated; set INTERLOCK_OLLAMA_LOG" in capsys.readouterr().err
    assert read_project_env("INTERLOCK_OLLAMA_LOG") == "kept"
    assert capsys.readouterr().err == ""


def test_the_new_environment_name_wins_without_a_warning(monkeypatch, capsys):
    _clear(monkeypatch)
    monkeypatch.setenv("INTERLOCK_OLLAMA_EXE", "new")
    monkeypatch.setenv("SWITCHYARD_OLLAMA_EXE", "old")
    assert read_project_env("INTERLOCK_OLLAMA_EXE") == "new"
    assert capsys.readouterr().err == ""


def test_the_old_serve_record_name_still_loads(tmp_path, monkeypatch, capsys):
    record = tmp_path / "serve.json"
    record.write_text(
        json.dumps({"argv": ["ollama", "serve"], "env": {}, "log_path": "", "library_path": ""}),
        encoding="utf-8",
    )
    _clear(monkeypatch)
    monkeypatch.setenv("SWITCHYARD_OLLAMA_SERVE", str(record))
    found = load_serve_start()
    assert found is not None
    assert found.argv == ("ollama", "serve")
    assert "deprecated" in capsys.readouterr().err


def test_the_old_log_and_exe_names_still_resolve(tmp_path, monkeypatch, capsys):
    log = tmp_path / "serve.log"
    log.write_text("load_backend: loaded\n", encoding="utf-8")
    exe = tmp_path / "ollama.exe"
    exe.write_bytes(b"")
    _clear(monkeypatch)
    monkeypatch.setenv("SWITCHYARD_OLLAMA_LOG", str(log))
    monkeypatch.setenv("SWITCHYARD_OLLAMA_EXE", str(exe))
    assert _read_configured_log() == "load_backend: loaded\n"
    dll = _configured_dll()
    assert dll is not None
    assert dll.replace("\\", "/").endswith("lib/ollama/cuda_v13/ggml-cuda.dll")
    err = capsys.readouterr().err
    assert "SWITCHYARD_OLLAMA_LOG is deprecated" in err
    assert "SWITCHYARD_OLLAMA_EXE is deprecated" in err
