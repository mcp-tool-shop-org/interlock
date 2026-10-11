"""Routing-row schema and the repository check.

``switchyard check`` is green when the repository has no receipts yet.
Every receipt and every table row has to validate. None may carry a
home-directory path, a hostname other than ``127.0.0.1`` or ``localhost``,
or a token-shaped value.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from switchyard.families import FamilyError, family_of

RUN_MARKER_SUFFIX = ".run.json"
_HOME = re.compile(r"(?i)(?:[A-Za-z]:[\\/]+Users[\\/]|[\\/]Users[\\/]|[\\/]home[\\/])")
_LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})
_HOST_SUFFIXES = frozenset(
    {
        "com",
        "net",
        "org",
        "io",
        "ai",
        "co",
        "dev",
        "app",
        "cloud",
        "sh",
        "local",
        "lan",
        "internal",
        "home",
        "arpa",
        "localdomain",
        "invalid",
        "test",
        "example",
        "edu",
        "gov",
    }
)
_IPV4 = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
_URL_HOST = re.compile(r"(?i)\b(?:https?|wss?)://(\[[^\]]+\]|[^/\s:]+)")
_FQDN = re.compile(r"(?<![A-Za-z0-9_\\/])(?:[A-Za-z0-9-]+\.)+([A-Za-z]{2,})(?![A-Za-z0-9_])")
_TOKEN_PREFIX = re.compile(r"(?<![A-Za-z0-9])(?:hf_|sk-|ghp_)[A-Za-z0-9]")
_SECRET_FIELD = re.compile(r"(?i)(?:^|_)(?:key|token|secret|password|passwd)(?:$|_)")
_LONG_HEX = re.compile(r"(?<![A-Za-z0-9])[0-9a-fA-F]{32,}(?![A-Za-z0-9])")
_LONG_B64 = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{32,}={0,2}(?![A-Za-z0-9+/])")
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
        # <run_id>.run.json is the run-start marker, not a receipt.
        paths.extend(sorted(path for path in receipts.rglob("*.json") if not path.name.endswith(RUN_MARKER_SUFFIX)))
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


_TIMING = (
    "cold_ms",
    "warm_p50_ms",
    "warm_p95_ms",
    "throughput",
    "compile_ms",
    "first_call_ms",
    "compile_warm_reload_ms",
)
_UNTIMED = frozenset({"device_lost", "hang", "compile_rejected", "compile_timeout"})
# The compile finished and the device refused the native binary at execute.
# Such a row keeps the compile time and nothing else.
_COMPILE_ONLY = frozenset({"native_binary_rejected"})
CAPABILITY_OUTCOMES = frozenset(_UNTIMED | _COMPILE_ONLY)


def is_speed_row(row: dict) -> bool:
    """A capability row, a failed parity guard or a descriptive row is recorded, and it is not a speed.

    A failed guard keeps its timing. It is not an untimed outcome. A descriptive
    row (a drift-block repeat) is evidence about the run and never routes.
    """

    return (
        row.get("outcome") not in CAPABILITY_OUTCOMES
        and row.get("parity_valid") is not False
        and row.get("descriptive") is not True
    )


def _check_device_lost(row: dict) -> None:
    """A capability row says the device cannot do the setting. It is not a speed.

    ``device_lost``, ``hang``, ``compile_rejected`` and ``compile_timeout`` carry no timing. ``native_binary_rejected``
    keeps the compile time only: the compile finished, the execute did not.
    """

    outcome = row.get("outcome")
    if outcome in _UNTIMED:
        allowed: frozenset[str] = frozenset()
    elif outcome in _COMPILE_ONLY:
        allowed = frozenset({"compile_ms"})
    else:
        return
    latency = row.get("latency") or {}
    if any(latency.get(key) is not None for key in _TIMING if key not in allowed) or latency.get(
        "throughput_unit"
    ):
        raise ReceiptError(f"{row['id']}: a {outcome} row has no timing")
    if row.get("artifacts"):
        raise ReceiptError(f"{row['id']}: a {outcome} row has no timing")


def _reject_home_paths(value: object, where: str = "$", field: str | None = None) -> None:
    """Refuse a home path, a non-loopback hostname, or a token-shaped value."""

    if isinstance(value, str):
        if _HOME.search(value):
            raise ReceiptError(f"home-directory path in {where}")
        host = _forbidden_host(value)
        if host is not None:
            raise ReceiptError(f"hostname {host} in {where}")
        if _token_shaped(value, field):
            raise ReceiptError(f"token-shaped value in {where}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_home_paths(item, f"{where}.{key}", str(key))
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_home_paths(item, f"{where}[{index}]", field)


def _forbidden_host(text: str) -> str | None:
    for match in _IPV4.finditer(text):
        ip = match.group(0)
        parts = ip.split(".")
        if any(not part.isdigit() or int(part) > 255 for part in parts):
            continue
        if ip not in _LOOPBACK:
            return ip
    for match in _URL_HOST.finditer(text):
        raw = match.group(1)
        host = raw[1:-1] if raw.startswith("[") and raw.endswith("]") else raw
        if host.casefold() not in _LOOPBACK:
            return host
    for match in _FQDN.finditer(text):
        name = match.group(0)
        if match.group(1).lower() not in _HOST_SUFFIXES:
            continue
        if name.casefold() not in _LOOPBACK:
            return name
    return None


def _token_shaped(value: str, field: str | None) -> bool:
    if _TOKEN_PREFIX.search(value):
        return True
    if field is None or _SECRET_FIELD.search(field) is None:
        return False
    return _LONG_HEX.search(value) is not None or _LONG_B64.search(value) is not None


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
