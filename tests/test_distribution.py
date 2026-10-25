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


def test_the_classifiers_match_what_ci_tests_and_the_license_is_an_spdx_expression():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    classifiers = project["classifiers"]
    assert "Development Status :: 5 - Production/Stable" in classifiers
    assert "Programming Language :: Python :: 3.13" in classifiers
    assert "Operating System :: POSIX :: Linux" in classifiers
    # The license is the SPDX expression (PEP 639). A License classifier next to it is
    # deprecated and PyPI can refuse the pair, so none is listed.
    assert project["license"] == "MIT"
    assert not [c for c in classifiers if c.startswith("License ::")]


def test_current_python_is_the_target_and_the_lockfile_agrees():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["requires-python"] == ">=3.13"
    classifiers = project["classifiers"]
    for version in ("3.13", "3.14", "3.15"):
        assert f"Programming Language :: Python :: {version}" in classifiers
    assert "Programming Language :: Python :: 3.12" not in classifiers
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    assert lock["requires-python"] == ">=3.13"
    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.15"


def test_the_intel_extra_resolves_only_where_openvino_publishes_wheels():
    # OpenVINO 2026.4.1 has cp313 and cp314 wheels and none for 3.15 (PyPI, read 2026-10-25).
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    intel = project["optional-dependencies"]["intel"]
    for requirement in intel:
        if requirement.startswith("openvino"):
            assert requirement.endswith('; python_version < "3.15"'), requirement


def test_the_ci_tests_current_python_with_the_required_check_name_kept():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    # The ruleset requires the status check "test". Its job keeps that id and runs on 3.15.
    nl = chr(10)
    assert nl + "  test:" + nl in ci
    assert nl + "  test-py313:" + nl in ci
    assert nl + "  test-py314:" + nl in ci
    test_job = ci.split(nl + "  test:" + nl, 1)[1].split(nl + "  test-py313:", 1)[0]
    assert 'python-version: "3.15"' in test_job
    assert "atlas" in test_job
    for leg, version in (("test-py313", "3.13"), ("test-py314", "3.14")):
        body = ci.split(nl + "  " + leg + ":" + nl, 1)[1].split(nl + "  test-py3", 1)[0]
        assert f'python-version: "{version}"' in body
        assert "atlas" not in body.replace("# Atlas", "")


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
