"""Where the few filesystem locations interlock uses are configured.

No path of the studio's rig is written in code. Each location is read from an environment variable, then from the
``[paths]`` table of ``interlock.toml`` in the working directory, and otherwise falls back to a portable default (a
temporary or user-cache location) or is off. A value is returned as it was given, never expanded to a home path.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

CONFIG_NAME = "interlock.toml"


class PathsConfigError(Exception):
    """``interlock.toml`` could not be read as a ``[paths]`` table."""


def configured(env_name: str, key: str) -> str | None:
    """The setting from the environment, else ``[paths] <key>`` in ``./interlock.toml``, else None."""

    value = os.environ.get(env_name)
    if value is not None and value.strip():
        return value.strip()
    path = Path.cwd() / CONFIG_NAME
    if not path.is_file():
        return None
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise PathsConfigError(f"{CONFIG_NAME}: {exc}") from exc
    table = document.get("paths", {})
    if not isinstance(table, dict):
        raise PathsConfigError(f"{CONFIG_NAME}: [paths] must be a table")
    found = table.get(key)
    if found is None:
        return None
    if not isinstance(found, str) or not found.strip():
        raise PathsConfigError(f"{CONFIG_NAME}: [paths] {key} must be a path string")
    return found.strip()
