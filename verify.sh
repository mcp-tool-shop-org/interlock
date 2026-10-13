#!/usr/bin/env bash
# Install, audit, test, build, and smoke-check. A live device is not started.
set -euo pipefail
cd "$(dirname "$0")"
uv sync --frozen
uv run --with pip-audit==2.10.1 pip-audit
uv run pytest
uv build
uv run interlock check
uv run interlock --help >/dev/null
