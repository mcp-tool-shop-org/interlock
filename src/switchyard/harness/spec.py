"""Load a pre-registered experiment spec.

The spec is the contract for the run. Required fields are the ones in hard
rule 5, plus the device and grant needs, plus the procedure the harness knows
how to dry-run.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

PROCEDURES = frozenset({"interleaved", "fill-card", "offrig-calibrate"})
ABORT_MIB = 30500
CEILING_MIB = 31200
REQUIRED = ("id", "question", "decision", "procedure", "needs", "stop_rules", "dataset", "metrics")


class SpecError(Exception):
    """The spec is missing a pre-registration field or asks for a forbidden setup."""


def load_spec(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    return validate_spec(data)


def validate_spec(data: object) -> dict:
    if not isinstance(data, dict):
        raise SpecError("a spec must be a mapping")
    for key in REQUIRED:
        if key not in data or data[key] in (None, "", []):
            raise SpecError(f"spec is missing {key}")
    if data["procedure"] not in PROCEDURES:
        raise SpecError(f"unknown procedure {data['procedure']!r}")
    needs = data["needs"]
    if not isinstance(needs, dict):
        raise SpecError("needs must be a mapping")
    for flag in ("grant", "npu_session", "cpu_window"):
        if not isinstance(needs.get(flag), bool):
            raise SpecError(f"needs.{flag} must be true or false")
    if not isinstance(data["stop_rules"], list) or not data["stop_rules"]:
        raise SpecError("stop_rules must be a non-empty list")
    procedure = data["procedure"]
    if procedure == "interleaved":
        _interleaved(data)
    elif procedure == "fill-card":
        _fill_card(data)
    else:
        _offrig(data)
    return data


def _interleaved(data: dict) -> None:
    if not isinstance(data.get("warmups"), int) or data["warmups"] < 0:
        raise SpecError("warmups must be an integer >= 0")
    if not isinstance(data.get("repeats"), int) or data["repeats"] < 1:
        raise SpecError("repeats must be an integer >= 1")
    band = data.get("band")
    if not isinstance(band, (int, float)) or isinstance(band, bool) or not 0 < float(band) <= 1:
        raise SpecError("band must be a fraction between 0 and 1")
    arms = data.get("arms")
    if not isinstance(arms, list) or not arms:
        raise SpecError("interleaved specs need arms")
    seen = set()
    for arm in arms:
        if not isinstance(arm, dict) or not arm.get("id"):
            raise SpecError("every arm needs an id")
        if arm["id"] in seen:
            raise SpecError(f"duplicate arm {arm['id']}")
        seen.add(arm["id"])
        for key in ("task", "device", "engine", "model", "revision_or_digest", "licence"):
            if not arm.get(key):
                raise SpecError(f"arm {arm['id']} is missing {key}")
        if arm["device"] not in {"NPU", "CPU", "iGPU", "5090"}:
            raise SpecError(f"arm {arm['id']} names a device the resolver does not allow")
        batch = arm.get("batch")
        if arm["task"] == "nli" and batch != 1:
            raise SpecError(f"arm {arm['id']} NLI batch must be 1")


def _fill_card(data: dict) -> None:
    if data.get("abort_mib") != ABORT_MIB or data.get("ceiling_mib") != CEILING_MIB:
        raise SpecError(f"fill-card stops at {ABORT_MIB} MiB and never tests the {CEILING_MIB} MiB ceiling")
    if data.get("grain") != 2048:
        raise SpecError("fill-card bisects on a 2048-token grid")
    fraction = data.get("confirm_prompt_fraction")
    if not isinstance(fraction, (int, float)) or isinstance(fraction, bool) or not 0 < float(fraction) < 1:
        raise SpecError("confirm_prompt_fraction must be between 0 and 1")
    for key in ("spill_tok_ratio", "spill_ttft_ratio"):
        if not isinstance(data.get(key), (int, float)) or isinstance(data.get(key), bool):
            raise SpecError(f"{key} must be a number")
    if not 0 < float(data["spill_tok_ratio"]) <= 1:
        raise SpecError("spill_tok_ratio must be in (0, 1]")
    if float(data["spill_ttft_ratio"]) < 1:
        raise SpecError("spill_ttft_ratio must be >= 1")
    models = data.get("models")
    if not isinstance(models, list) or not models:
        raise SpecError("fill-card names the model roster")
    for model in models:
        for key in ("model", "revision_or_digest", "engine", "device"):
            if not model.get(key):
                raise SpecError(f"a roster entry is missing {key}")
        if model["device"] != "5090" or model["engine"] != "ollama":
            raise SpecError("fill-card measures local 5090 Ollama models")
    if data["needs"]["grant"] is not True:
        raise SpecError("fill-card requires a grant")


def _offrig(data: dict) -> None:
    for key in ("model", "url", "gold"):
        if not data.get(key):
            raise SpecError(f"offrig-calibrate is missing {key}")
    if data["needs"]["grant"] is not True:
        raise SpecError("a 5090 calibrate requires a grant")
