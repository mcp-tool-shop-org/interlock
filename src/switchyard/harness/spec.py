"""Load a pre-registered experiment spec.

The spec is the contract for the run. Required fields are the ones in hard
rule 5, plus the device and grant needs, plus the procedure the harness knows
how to dry-run.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from switchyard.harness.e5 import PROMPT_RATIO, SHARED_GROWTH_MIB
from switchyard.tasks import TASKS

PROCEDURES = frozenset({"interleaved", "fill-card", "offrig-calibrate"})
ABORT_MIB = 30500
CEILING_MIB = 31200
DEVICES = frozenset({"NPU", "CPU", "iGPU", "5090"})
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
    ceiling = data.get("call_ceiling_s")
    if ceiling is not None and (
        isinstance(ceiling, bool) or not isinstance(ceiling, (int, float)) or ceiling < 60
    ):
        raise SpecError("call_ceiling_s must be a number of at least 60")
    compile_ceiling = data.get("compile_ceiling_s")
    if compile_ceiling is not None and (
        isinstance(compile_ceiling, bool) or not isinstance(compile_ceiling, (int, float)) or compile_ceiling < 60
    ):
        raise SpecError("compile_ceiling_s must be a number of at least 60")
    _decision_metric(data)
    _choice_rule(data)
    return data


_LATENCY_KEYS = (
    "cold_ms",
    "warm_p50_ms",
    "warm_p95_ms",
    "throughput",
    "throughput_unit",
    "compile_ms",
    "first_call_ms",
    "compile_warm_reload_ms",
)
PREFIX_CACHING = ("on", "off", "default")


def _decision_metric(data: dict) -> None:
    """``decision_metric``: a row field path ("latency.throughput") that every finished arm must carry."""

    metric = data.get("decision_metric")
    if metric is None:
        return
    head, _, tail = metric.partition(".") if isinstance(metric, str) else ("", "", "")
    if head != "latency" or tail not in _LATENCY_KEYS:
        raise SpecError(f"decision_metric must be one of: " + ", ".join(f"latency.{key}" for key in _LATENCY_KEYS))


def _prefix_caching(holder: dict, where: str) -> None:
    value = holder.get("prefix_caching")
    if value is not None and value not in PREFIX_CACHING:
        raise SpecError(f"{where} prefix_caching must be on, off or default")


def _interleaved(data: dict) -> None:
    _prefix_caching(data, "spec")
    for arm in data.get("arms") or []:
        if isinstance(arm, dict):
            _prefix_caching(arm, f"arm {arm.get('id')}")
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
        if arm["device"] not in DEVICES:
            raise SpecError(f"arm {arm['id']} names a device the resolver does not allow")
        if arm["task"] not in TASKS:
            raise SpecError(f"arm {arm['id']} task {arm['task']!r} is not an exact routing task")
        batch = arm.get("batch")
        if arm["task"] == "nli" and batch != 1:
            raise SpecError(f"arm {arm['id']} NLI batch must be 1")
        companion_device = arm.get("companion_device")
        if companion_device is not None and companion_device not in DEVICES:
            raise SpecError(f"arm {arm['id']} names a companion device the resolver does not allow")
        not_run = arm.get("not_run")
        if not_run is not None and (not isinstance(not_run, str) or not not_run.strip()):
            raise SpecError(f"arm {arm['id']} not_run needs a reason")
        if arm["task"] == "generate" and not str(arm.get("prompt") or "").strip() and not _builds_prompts(data):
            # e3-ref's first run sent 14 empty prompts and measured nothing: a generate arm
            # needs its own prompt, or a dataset seed and target the harness builds one from.
            raise SpecError(f"generate arm {arm['id']} has no prompt")
    for arm in arms:
        companion = arm.get("companion")
        if companion is not None and companion not in seen:
            raise SpecError(f"arm {arm['id']} companion is not an arm")
    cache = data.get("compile_cache")
    if cache not in {"fresh", "reuse"}:
        raise SpecError("compile_cache must be fresh or reuse")
    order = data.get("order", "interleaved")
    if order not in {"interleaved", "aba", "blocked"}:
        raise SpecError(f"unknown order {order!r}")
    if order == "aba":
        _pairs(data, seen)
    if order == "blocked":
        _blocks(data, seen)
    elif "blocks" in data or "drift_block" in data:
        raise SpecError("blocks and drift_block belong to order: blocked")


def _builds_prompts(data: dict) -> bool:
    """True when the dataset gives the harness a seed: built to a token target, or passed through as the prompt."""

    dataset = data.get("dataset")
    return isinstance(dataset, dict) and bool(str(dataset.get("seed") or "").strip())


def _blocks(data: dict, arm_ids: set) -> None:
    """``order: blocked``: the arms are split into blocks run one after another.

    Every arm sits in exactly one block. An optional drift block repeats some
    arms of one block at the end, as descriptive rows that never reach the table.
    """

    if "pairs" in data:
        raise SpecError("blocked order has no pairs; pairs belong to order: aba")
    blocks = data.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise SpecError("blocked order needs at least one block")
    block_ids: dict[str, list] = {}
    placed: dict[str, str] = {}
    for block in blocks:
        if not isinstance(block, dict) or not isinstance(block.get("id"), str) or not block["id"]:
            raise SpecError("every block needs an id")
        if block["id"] in block_ids:
            raise SpecError(f"duplicate block {block['id']}")
        members = block.get("arms")
        if not isinstance(members, list) or not members:
            raise SpecError(f"block {block['id']} needs arms")
        for arm_id in members:
            if arm_id not in arm_ids:
                raise SpecError(f"block {block['id']} names an arm that is not in the spec: {arm_id}")
            if arm_id in placed:
                raise SpecError(f"arm {arm_id} is in blocks {placed[arm_id]} and {block['id']}; an arm is in exactly one block")
            placed[arm_id] = block["id"]
        block_ids[block["id"]] = members
    missing = sorted(arm_ids - set(placed))
    if missing:
        raise SpecError(f"arms in no block: {', '.join(missing)}")
    drift = data.get("drift_block")
    if drift is None:
        return
    if not isinstance(drift, dict) or not isinstance(drift.get("id"), str) or not drift["id"]:
        raise SpecError("drift_block needs an id")
    if drift["id"] in block_ids:
        raise SpecError(f"drift_block id {drift['id']} repeats a block id")
    source = drift.get("repeat_of")
    if source not in block_ids:
        raise SpecError("drift_block repeat_of must name a block")
    members = drift.get("arms")
    if not isinstance(members, list) or not members:
        raise SpecError("drift_block needs arms")
    for arm_id in members:
        if arm_id not in block_ids[source]:
            raise SpecError(f"drift_block arm {arm_id} is not in block {source}")
    if len(set(members)) != len(members):
        raise SpecError("drift_block names an arm twice")
    repeats = drift.get("repeats")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise SpecError("drift_block repeats must be an integer >= 1")


def _fill_card(data: dict) -> None:
    if data.get("abort_mib") != ABORT_MIB or data.get("ceiling_mib") != CEILING_MIB:
        raise SpecError(f"fill-card stops at {ABORT_MIB} MiB and never tests the {CEILING_MIB} MiB ceiling")
    if data.get("grain") != 2048:
        raise SpecError("fill-card bisects on a 2048-token grid")
    fraction = data.get("confirm_prompt_fraction")
    if not isinstance(fraction, (int, float)) or isinstance(fraction, bool) or not 0 < float(fraction) < 1:
        raise SpecError("confirm_prompt_fraction must be between 0 and 1")
    if data.get("spill_shared_mib") != SHARED_GROWTH_MIB:
        raise SpecError(f"spill_shared_mib is {SHARED_GROWTH_MIB} MiB over the post-load shared usage")
    if data.get("spill_prompt_ratio") != PROMPT_RATIO:
        raise SpecError(f"spill_prompt_ratio is {PROMPT_RATIO}")
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


def _pairs(data: dict, arm_ids: set) -> None:
    pairs = data.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise SpecError("aba order needs pairs")
    arms = {arm["id"]: arm for arm in data["arms"]}
    seen = set()
    for pair in pairs:
        if not isinstance(pair, dict) or not pair.get("id"):
            raise SpecError("every pair needs an id")
        if pair["id"] in seen:
            raise SpecError(f"duplicate pair {pair['id']}")
        seen.add(pair["id"])
        alone = pair.get("alone")
        paired_id = pair.get("paired")
        if alone not in arm_ids or paired_id not in arm_ids:
            raise SpecError("a pair names an arm that is not in the spec")
        paired = arms[paired_id]
        companion = paired.get("companion")
        if companion not in arm_ids:
            raise SpecError(f"paired arm {paired_id} needs a companion arm")
        expected = paired.get("companion_device")
        if expected is None:
            raise SpecError(f"paired arm {paired_id} needs companion_device")
        if arms[companion]["device"] != expected:
            raise SpecError(f"arm {paired_id} companion_device does not match the companion arm")


def _choice_rule(data: dict) -> None:
    if "choice_rule" not in data:
        if "selection_sha" in data:
            raise SpecError("selection_sha belongs with a choice_rule")
        return
    rule = data["choice_rule"]
    if not isinstance(rule, dict):
        raise SpecError("choice_rule must be a mapping")
    allowed = {"eligibility_floor", "ranking_key", "tie_rule", "fallback"}
    extra = set(rule) - allowed
    if extra or set(rule) != allowed:
        raise SpecError("choice_rule holds the eligibility floor, ranking key, tie rule, and fallback")
    floor = rule["eligibility_floor"]
    if not isinstance(floor, dict) or set(floor) != {"metric", "minimum"}:
        raise SpecError("choice_rule.eligibility_floor holds a metric and a minimum")
    if not isinstance(floor["metric"], str) or not floor["metric"]:
        raise SpecError("choice_rule.eligibility_floor.metric must name a quality metric")
    minimum = floor["minimum"]
    if isinstance(minimum, bool) or not isinstance(minimum, (int, float)):
        raise SpecError("choice_rule.eligibility_floor.minimum must be a number")
    if not isinstance(rule["ranking_key"], str) or not rule["ranking_key"]:
        raise SpecError("choice_rule.ranking_key must name a quality metric")
    tie = rule["tie_rule"]
    if not isinstance(tie, dict) or set(tie) != {"toward"}:
        raise SpecError("choice_rule.tie_rule holds toward, the committed choice")
    if not isinstance(tie["toward"], str) or not tie["toward"]:
        raise SpecError("choice_rule.tie_rule.toward must name the committed choice")
    if not isinstance(rule["fallback"], str) or not rule["fallback"]:
        raise SpecError("choice_rule.fallback must name a row")
    sha = data.get("selection_sha")
    if not isinstance(sha, str) or not sha:
        raise SpecError("a choice_rule needs the frozen selection_sha beside it")


def _offrig(data: dict) -> None:
    for key in ("model", "url", "gold"):
        if not data.get(key):
            raise SpecError(f"offrig-calibrate is missing {key}")
    if data["needs"]["grant"] is not True:
        raise SpecError("a 5090 calibrate requires a grant")
