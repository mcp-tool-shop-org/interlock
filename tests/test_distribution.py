"""PyPI refused the name interlock. The distribution name is the one it allows."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

KEYWORDS = [
    "local-ai",
    "gpu",
    "npu",
    "ollama",
    "openvino",
    "device-router",
    "scheduling",
]


def test_the_distribution_name_is_mcptoolshop_interlock():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["name"] == "mcptoolshop-interlock"
    assert project["scripts"]["interlock"] == "interlock.cli:main"
    assert project["description"].startswith("Local AI device router:")
    assert project["keywords"] == KEYWORDS


def test_the_help_header_names_the_router(capsys):
    from interlock.cli import main

    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    text = capsys.readouterr().out
    header, _, _rest = text.partition("positional arguments:")
    assert "interlock: local AI device router" in header
    assert "Route a job from measured receipts." in header
