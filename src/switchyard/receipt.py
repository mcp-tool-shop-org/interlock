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

from switchyard.families import FamilyError, family_of

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
        _check_interval(row, metric)
    for metric in row.get("per_type", {}).values():
        _check_interval(row, metric)
    _check_family(row)
    _check_residency(row)
    _check_device_lost(row)
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


_TIMING = ("cold_ms", "warm_p50_ms", "warm_p95_ms", "throughput", "compile_ms", "first_call_ms")


def _check_device_lost(row: dict) -> None:
    """A device_lost row says the device cannot do the setting. It is not a speed."""

    if row.get("outcome") != "device_lost":
        return
    latency = row.get("latency") or {}
    if any(latency.get(key) is not None for key in _TIMING) or latency.get("throughput_unit"):
        raise ReceiptError(f"{row['id']}: a device_lost row has no timing")
    if row.get("artifacts"):
        raise ReceiptError(f"{row['id']}: a device_lost row has no timing")


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


def _check_interval(row: dict, metric: dict) -> None:
    interval = metric["interval"]
    if interval["low"] > interval["high"]:
        raise ReceiptError(f"{row['id']}: interval low is above high for {metric['name']}")


def _check_family(row: dict) -> None:
    tag = row["model"]["repo_or_tag"]
    try:
        expected = family_of(tag)
    except FamilyError as exc:
        raise ReceiptError(f"{row['id']}: {exc}") from exc
    actual = row["model_family"] if "model_family" in row else None
    if expected is None:
        if actual is not None:
            raise ReceiptError(f"{row['id']}: {tag} is not in the family table, so model_family must be absent")
        return
    if actual != expected:
        raise ReceiptError(f"{row['id']}: model_family for {tag} is {expected}, from the family table")


def _check_residency(row: dict) -> None:
    model = row["model"]
    size = model.get("size")
    vram = model.get("size_vram")
    stated = row["gpu_resident"] if "gpu_resident" in row else None
    measured = size is not None and vram is not None
    if not measured:
        if stated is not None:
            raise ReceiptError(f"{row['id']}: gpu_resident needs model.size and model.size_vram from the load")
        return
    if stated is None:
        raise ReceiptError(f"{row['id']}: gpu_resident must record whether size_vram equals size")
    if stated != (size == vram):
        raise ReceiptError(f"{row['id']}: gpu_resident is true only when size_vram equals size")


def _format_error(error: ValidationError) -> str:
    location = "/".join(str(part) for part in error.path) or "$"
    return f"{location}: {error.message}"
