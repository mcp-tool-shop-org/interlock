"""Where the full ``interlock import`` rebuild finds its three source repos.

The commit each source must be at is pinned (``pins.py``). Where the clone lives is configuration: the
``--source name=path`` flag (repeatable), or ``[import.sources]`` in ``<root>/interlock.toml``. The flag wins.
There is no default path.
"""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from pathlib import Path

from interlock.importers.common import ImportError_
from interlock.importers.pins import ASPIRE_COMMIT, NPU_BUILDOUT_COMMIT, RND_COMMIT

CONFIG_NAME = "interlock.toml"
SOURCE_PINS = {"rnd": RND_COMMIT, "npu": NPU_BUILDOUT_COMMIT, "aspire": ASPIRE_COMMIT}


def parse_source_flags(flags: Sequence[str] | None) -> dict[str, str]:
    """``name=path`` strings to ``{name: path}``, the path exactly as given."""

    out: dict[str, str] = {}
    for flag in flags or ():
        name, sep, path = str(flag).partition("=")
        if not sep or not name.strip() or not path.strip():
            raise ImportError_(f"--source {flag!r} must be name=path, with name one of {', '.join(SOURCE_PINS)}")
        name = name.strip()
        if name not in SOURCE_PINS:
            raise ImportError_(f"unknown source {name!r} in --source; the sources are {', '.join(SOURCE_PINS)}")
        out[name] = path.strip()
    return out


def load_config_sources(root: Path) -> dict[str, str]:
    """``[import.sources]`` from ``<root>/interlock.toml``, or nothing when there is no such file."""

    path = root / CONFIG_NAME
    if not path.is_file():
        return {}
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ImportError_(f"{CONFIG_NAME}: {exc}") from exc
    table = document.get("import", {}).get("sources", {})
    if not isinstance(table, dict):
        raise ImportError_(f"{CONFIG_NAME}: [import.sources] must be a table of name = \"path\"")
    out: dict[str, str] = {}
    for name, value in table.items():
        if name not in SOURCE_PINS:
            raise ImportError_(f"{CONFIG_NAME}: unknown source {name!r}; the sources are {', '.join(SOURCE_PINS)}")
        if not isinstance(value, str) or not value.strip():
            raise ImportError_(f"{CONFIG_NAME}: source {name!r} must be a path string")
        out[name] = value.strip()
    return out


def resolve_sources(flags: Sequence[str] | None, root: Path) -> dict[str, str]:
    """The rebuild's sources: the config file, overridden name by name by the flags."""

    return {**load_config_sources(root), **parse_source_flags(flags)}


def require_all_sources(sources: dict[str, str]) -> None:
    """Refuse a rebuild that has a source missing, naming each one, its pinned commit and how to set it."""

    missing = [name for name in SOURCE_PINS if name not in sources]
    if not missing:
        return
    lines = [f"REFUSED: the full rebuild needs {len(missing)} source repo(s) that are not configured:"]
    for name in missing:
        lines.append(
            f"  {name}: a clone with commit {SOURCE_PINS[name]}; set it with --source {name}=<path> "
            f'or [import.sources] {name} = "<path>" in {CONFIG_NAME}'
        )
    lines.append("To add a remote window to the table without a rebuild, use --remote-window alone.")
    raise ImportError_("\n".join(lines))
