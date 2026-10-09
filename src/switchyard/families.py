"""Lineage of a model tag, from one committed table.

The lookup is the exact tag. A prefix, a substring, or a differently spelled
tag is a miss, and a miss has no family. Adding a family means adding a row
to ``table/model-families.json``.
"""

from __future__ import annotations

import json
from pathlib import Path


class FamilyError(Exception):
    """The committed family table cannot be read."""


def families_path() -> Path:
    return Path(__file__).resolve().parents[2] / "table" / "model-families.json"


def load_families() -> dict[str, str]:
    path = families_path()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FamilyError(f"model family table: {exc}") from exc
    if not isinstance(document, dict) or not document:
        raise FamilyError("model family table must map exact tags to a lineage")
    table: dict[str, str] = {}
    for tag, family in document.items():
        if not isinstance(tag, str) or not tag or not isinstance(family, str) or not family:
            raise FamilyError("model family table keys and values must be non-empty strings")
        table[tag] = family
    return table


def family_of(tag: str) -> str | None:
    """The lineage for this exact tag, or None when the table does not list it."""

    if not isinstance(tag, str):
        raise FamilyError("a model tag must be a string")
    return load_families().get(tag)


def stamp_family(row: dict) -> None:
    """Write the table's family. An unlisted tag does not get a family field."""

    family = family_of(row["model"]["repo_or_tag"])
    if family is None:
        row.pop("model_family", None)
    else:
        row["model_family"] = family
