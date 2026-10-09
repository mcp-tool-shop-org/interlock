"""Routing-row schema and the repository check.

``switchyard check`` is green when the repository has no receipts yet.
Every receipt that is present has to validate, and none may carry a
home-directory path.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

_HOME = re.compile(r"(?i)(?:[A-Za-z]:[\\/]+Users[\\/]|[\\/]Users[\\/]|[\\/]home[\\/])")
_VALIDATOR: Draft202012Validator | None = None


class ReceiptError(Exception):
    """A receipt, or a set of them, does not validate."""


def schema() -> dict:
    text = (
        Path(__file__).resolve().parent.joinpath("schema", "routing-row.schema.json").read_text(encoding="utf-8")
    )
    return json.loads(text)


def validator() -> Draft202012Validator:
    global _VALIDATOR
    if _VALIDATOR is None:
        document = schema()
        Draft202012Validator.check_schema(document)
        _VALIDATOR = Draft202012Validator(document)
    return _VALIDATOR


def validate_row(row: dict) -> None:
    errors = sorted(validator().iter_errors(row), key=lambda item: list(item.path))
    if errors:
        raise ReceiptError(_format_error(errors[0]))
    _reject_home_paths(row)
    for metric in row["quality"]:
        interval = metric["interval"]
        if interval["low"] > interval["high"]:
            raise ReceiptError(f"{row['id']}: interval low is above high for {metric['name']}")
    provenance = row["provenance"]
    if "source_receipt_path" in provenance and not provenance.get("commit_sha"):
        if not provenance.get("content_sha256"):
            raise ReceiptError(f"{row['id']}: a receipt with no commit sha needs content_sha256")


def check_tree(root: Path) -> list[str]:
    """Return human-readable errors. An empty tree returns an empty list."""

    problems: list[str] = []
    for path in receipt_paths(root):
        try:
            rows = load_jsonl(path) if path.suffix == ".jsonl" else _rows_in(
                json.loads(path.read_text(encoding="utf-8")), path
            )
        except (json.JSONDecodeError, ReceiptError) as exc:
            problems.append(f"{path}: {exc}")
            continue
        for index, row in enumerate(rows):
            try:
                validate_row(row)
            except ReceiptError as exc:
                label = row.get("id") if isinstance(row, dict) else None
                where = f"{path}#{label or index}"
                problems.append(f"{where}: {exc}")
    return problems


def receipt_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    receipts = root / "receipts"
    if receipts.is_dir():
        paths.extend(sorted(receipts.rglob("*.json")))
    table = root / "table" / "routing.jsonl"
    if table.is_file():
        paths.append(table)
    return paths


def _rows_in(document: object, path: Path) -> list:
    if isinstance(document, dict):
        return [document]
    if isinstance(document, list):
        return document
    raise ReceiptError(f"{path}: a receipt file must be an object or a list")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ReceiptError(f"{path}:{line_number}: {exc}") from exc
    return rows


def _reject_home_paths(value: object, where: str = "$") -> None:
    if isinstance(value, str):
        if _HOME.search(value):
            raise ReceiptError(f"home-directory path in {where}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_home_paths(item, f"{where}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_home_paths(item, f"{where}[{index}]")


def _format_error(error: ValidationError) -> str:
    location = "/".join(str(part) for part in error.path) or "$"
    return f"{location}: {error.message}"
