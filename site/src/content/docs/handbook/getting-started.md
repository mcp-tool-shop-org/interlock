---
title: Getting started
description: Install interlock from PyPI or from a clone.
sidebar:
  order: 1
---

Python 3.12 only: `requires-python` is `>=3.12,<3.13`.

From PyPI:

```bash
pip install mcptoolshop-interlock
```

or

```bash
uv tool install mcptoolshop-interlock
```

From a clone, for development and live measurement runs:

```bash
git clone https://github.com/mcp-tool-shop-org/interlock
cd interlock
uv sync
uv run interlock --help
```

The distribution name is `mcptoolshop-interlock`. The import and the command are `interlock`. CI runs the tests on Ubuntu. A live measurement run is written for the Windows rig this table was measured on. Since 1.0.0, a `platform: linux-cuda` spec runs on a remote Colab VM instead.

`uv sync` does not install the `intel` extra. Tests do not need OpenVINO. A live NPU or iGPU run does, and that extra is pinned in `pyproject.toml`.

## A first command that starts nothing

`interlock check` reads the committed receipts and `table/routing.jsonl`. An empty tree is fine. A home-directory path in a receipt is not.

```bash
uv run interlock check
```

Ask the table for a route. This does not start a device, read a ledger, or call offrig.

```bash
uv run interlock route --task generate --model Phi --thinking off --ctx 1024 --busy 5090
```

`--model` matches a substring of the repo or tag. Hyphens and colons are ignored, so `Qwen3-8B` matches `qwen3:8b`. A short substring can match an unintended repo. The matched model is printed.

## One command for the maintainer

`verify.sh` syncs the lockfile, audits dependencies, runs the tests, builds a wheel and an sdist, and smoke-checks `interlock check` and `interlock --help`. It does not start a device. The build lands in `dist/`, which is gitignored, and it is not published.

## What a real run needs

A measurement run is a separate decision from installing. `interlock run` on a real spec needs the flag its devices call for: `--grant` for the 5090, `--npu-session` for the NPU or the iGPU, and `--cpu-window` for a CPU-heavy leg. `--dry-run` prints the plan and those guards, and starts nothing. See [Usage](./usage/).
