"""Choose a row from the routing table. No I/O, no clock, no device probe.

The request carries ``as_of``. The command line defaults that to today's date.
Nothing here reads the Publisher's ledger, the watchdog, ``/api/ps``, or offrig.

Ranking, one metric per task. The bound is never a median.

| Task | Rank by | Conservative bound |
|---|---|---|
| ``generate`` | highest decode tok/s | minimum ``phase=repeat`` slot throughput |
| every other task | lowest warm latency | maximum ``phase=repeat`` slot ``latency_ms`` |

Or ``claim_wall_s.max`` in milliseconds when the row has no repeat slots.

A higher-is-better floor (``--min-<metric>``) uses that same minimum for
``tok-s``, and the quality interval's low end for any other metric name.
``--max-latency-ms`` uses the latency maximum. A row with no such spread is
not a candidate. There is no outlier trimming: one slow repeat sets the bound.

``--ctx N`` is an exact measured-length cell when the row id carries ``lN``
(decode rate is not portable across prompt lengths). A row with ``num_ctx``
and no ``lN`` tag meets the request when its window is at least N.

Ties break toward a device other than the 5090, then toward the lower row id.

A row dated more than 30 days before ``as_of`` is stale. A row with
``provisional`` set, or ``repeats`` below 2, is provisional. Both are flags.
They stay eligible.

Capability rows (``native_binary_rejected``, ``compile_rejected``,
``compile_timeout``, ``hang``, ``device_lost``, ``oom``, ``load_refused``,
``timeout``, ``ctx_exceeded``) and unmeasured cells (``sampler_error``,
``generation_aborted``, ``unmeasured``) are excluded, with those codes, and never ranked. A row
with ``parity_valid`` false is excluded the same way, code ``parity_invalid``.
When a row is both, the capability code is the one shown.
Missing ``parity_valid`` stays eligible.

``thinking`` matches a family. ``off`` matches only ``off``. ``on`` matches
any recorded thinking level. ``low``, ``medium`` and ``high`` match exactly
when the request names one. The decision's settings keep the row's exact
level.

TTFT (``--max-ttft-ms``) is refused unless every row that would be compared
records the same known ``settings.prefix_caching`` (``on``, ``off``, or
``default``). Unset means unknown, not "all the same".

A remote row is a ``runpod:`` or ``colab:`` device. Remote ``colab:`` rows stay
out of the pool unless the request opts in (``--allow-remote`` or
``--device colab:<card>``). A ``runpod:`` row's ``settings.offrig_plan`` must
include ``profile`` and ``max_hours``. ``--budget-left-usd`` drops a remote
row whose ``settings.worst_case_usd`` is missing or greater than the flag,
and that row is a runner-up with the reason. With no flag the suggestion is
kept and marked budget unchecked. This function does not read a balance.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta

from interlock.devices import is_colab_device, matches_device_filter

EXCLUDED_OUTCOMES = (
    "native_binary_rejected",
    "compile_rejected",
    "compile_timeout",
    "hang",
    "device_lost",
    "oom",
    "load_refused",
    "timeout",
    "ctx_exceeded",
    "sampler_error",
    "generation_aborted",
    "unmeasured",
)

_KNOWN_PREFIX = frozenset({"on", "off", "default"})
_DECODE_FLOORS = frozenset({"tok_s", "decode_tok_s"})
_LENGTH = re.compile(r"(?:^|[-_/])l(\d+)(?:[-_/]|$)")
_STALE_AFTER = timedelta(days=30)


@dataclass(frozen=True)
class Request:
    """One routing question. ``as_of`` is supplied by the caller."""

    task: str
    as_of: date
    model: str | None = None
    min_metrics: Mapping[str, float] = field(default_factory=dict)
    max_latency_ms: float | None = None
    max_ttft_ms: float | None = None
    ctx: int | None = None
    thinking: str | None = None
    busy: frozenset[str] = frozenset()
    quote_rule: int | None = None
    num_predict: int | None = None
    budget_left_usd: float | None = None
    allow_remote: bool = False
    device: str | None = None


@dataclass(frozen=True)
class Decision:
    """A choice. ``document`` is the route-decision.v1 object."""

    document: dict


@dataclass(frozen=True)
class Refusal:
    """No row met the constraints, or TTFT could not be compared."""

    document: dict


def decide(rows: Sequence[Mapping], request: Request) -> Decision | Refusal:
    """Return the choice, or a refusal that names the constraint and the gap."""

    visible_rows = [row for row in rows if _remote_eligible(row, request)]
    excluded: list[dict] = []
    failure_reasons: list[str] = []
    pool: list[Mapping] = []
    for row in visible_rows:
        outcome = row.get("outcome")
        if outcome in EXCLUDED_OUTCOMES and _task_model(row, request):
            excluded.append({"id": str(row["id"]), "code": str(outcome)})
            unmeasured = outcome in {"sampler_error", "generation_aborted", "unmeasured"}
            measured = None if unmeasured else _measured_failure_reason(row)
            if measured is not None:
                failure_reasons.append(f"{row.get('device')}: {measured}")
        elif outcome in EXCLUDED_OUTCOMES:
            continue
        elif row.get("parity_valid") is False and _task_model(row, request):
            excluded.append({"id": str(row["id"]), "code": "parity_invalid"})
        elif row.get("parity_valid") is False:
            continue
        else:
            pool.append(row)
    excluded.sort(key=lambda item: item["id"])
    version_notes: list[str] = []

    steps: list[tuple[str, object]] = [("task", None), ("model", None)]
    if request.device is not None:
        steps.append(("device", request.device))
    if request.thinking is not None:
        steps.append(("thinking", request.thinking))
    if request.ctx is not None:
        steps.append(("ctx", request.ctx))
    if request.quote_rule is not None:
        steps.append(("quote_rule", request.quote_rule))
    if request.num_predict is not None:
        steps.append(("num_predict", request.num_predict))
    if request.busy:
        steps.append(("busy", None))
    later: list[tuple[str, object]] = [("remote_plan", None)]
    if request.budget_left_usd is not None:
        later.append(("budget", request.budget_left_usd))
    for name in sorted(request.min_metrics):
        later.append((f"min:{name}", request.min_metrics[name]))
    if request.max_latency_ms is not None:
        later.append(("max_latency_ms", request.max_latency_ms))
    if request.max_ttft_ms is not None:
        later.append(("max_ttft_ms", request.max_ttft_ms))

    current = pool
    identity = [name for name, _ in steps if name in {"task", "model", "device"}][-1]
    for constraint, limit in steps:
        kept, dropped = _split(current, request, constraint, limit)
        if not kept:
            return _empty(constraint, limit, dropped, excluded, request, failure_reasons=failure_reasons)
        current = kept
        if constraint == identity:
            # Before any floor: a v1 row must not win, or be listed, because the v2 row failed a floor.
            before = current
            current, version_notes = _newest_ladder_per_device_model(before)
            current, across = _newest_ladder_version(current)  # and across devices: v1 is never ranked against v2
            version_notes = [*version_notes, *across]
            dropped_ids = {id(row) for row in before} - {id(row) for row in current}
            pool = [row for row in pool if id(row) not in dropped_ids]

    if request.max_ttft_ms is not None and current:
        refusal = _ttft_refusal(current, request, excluded)
        if refusal is not None:
            return refusal

    for constraint, limit in later:
        kept, dropped = _split(current, request, constraint, limit)
        if not kept:
            return _empty(constraint, limit, dropped, excluded, request, failure_reasons=failure_reasons)
        current = kept

    direction = "higher" if request.task == "generate" else "lower"
    metric = "decode_tok_s" if request.task == "generate" else "warm_latency_ms"
    ranked = [row for row in current if _rank_bound(row, request) is not None]
    if not ranked:
        return _empty(metric, None, current, excluded, request, failure_reasons=failure_reasons)

    ranked.sort(key=lambda row: _rank_key(row, request))
    winner = ranked[0]
    runners = _runners(pool, winner, ranked, request)
    return Decision(
        _document(winner, ranked, runners, excluded, request, direction, metric, pool, version_notes)
    )


def ladder_version(row: Mapping) -> int | None:
    """The generate ladder that measured a row. A Colab ladder cell with no field is version 1.

    Other rows (a fill-card row, the rig's own harnesses) did not come from the ladder and have no version.
    """

    value = (row.get("settings") or {}).get("ladder_version")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    settings = row.get("settings") or {}
    # Ladder cells carry a concurrency; a fill-card row (max_resident_ctx) is not a ladder row.
    if row.get("task") == "generate" and is_colab_device(row.get("device")) and "concurrency" in settings:
        return 1
    return None


def _newest_ladder_per_device_model(pool: list[Mapping]) -> tuple[list[Mapping], list[str]]:
    """Within one device and model, only the newest ladder version is a candidate."""

    newest: dict[tuple[str, str], int] = {}
    for row in pool:
        version = ladder_version(row)
        if version is not None:
            key = (str(row.get("device")), _model_label(row))
            newest[key] = max(newest.get(key, version), version)
    kept = [
        row
        for row in pool
        if (version := ladder_version(row)) is None or version == newest[(str(row.get("device")), _model_label(row))]
    ]
    dropped = len(pool) - len(kept)
    notes = [f"{dropped} older-ladder row(s) not compared"] if dropped else []
    return kept, notes


def _newest_ladder_version(ranked: list[Mapping]) -> tuple[list[Mapping], list[str]]:
    """Speeds from different ladder versions are never ranked against each other."""

    versions = {v for row in ranked if (v := ladder_version(row)) is not None}
    if len(versions) < 2:
        return ranked, []
    newest = max(versions)
    kept = [row for row in ranked if ladder_version(row) in (None, newest)]
    older = sorted(versions - {newest})
    return kept, [f"ranked on ladder version {newest} only; version(s) {', '.join(map(str, older))} not compared"]


def _document(
    winner: Mapping,
    ranked: list[Mapping],
    runners: list[dict],
    excluded: list[dict],
    request: Request,
    direction: str,
    metric: str,
    pool: Sequence[Mapping],
    version_notes: Sequence[str] = (),
) -> dict:
    bound = _rank_bound(winner, request)
    assert bound is not None  # noqa: S101  # rank only sees a row that has a bound
    if len(ranked) > 1:
        nxt = ranked[1]
        next_bound = _rank_bound(nxt, request)
        assert next_bound is not None  # noqa: S101  # rank only sees a row that has a bound
        delta = (bound - next_bound) if direction == "higher" else (next_bound - bound)
        next_id: str | None = str(nxt["id"])
    else:
        next_bound = None
        delta = None
        next_id = None
    notes: list[str] = list(version_notes)
    if "5090" not in request.busy and not _any_5090(pool, request):
        notes.append("no 5090 row")
    remote = _is_remote(winner)
    plan = _plan_args(winner) if _is_runpod(winner) else None
    if remote and request.budget_left_usd is None:
        notes.append("budget unchecked")
    budget = "not_applicable"
    if remote and request.budget_left_usd is None:
        budget = "unchecked"
    elif remote:
        budget = "within"
    document = {
        "version": 1,
        "task": request.task,
        "model": _model_label(winner),
        "engine": {
            "name": winner["engine"]["name"],
            "version": winner["engine"]["version"],
        },
        "device": winner["device"],
        "settings": dict(winner.get("settings") or {}),
        "receipts": [str(winner["id"])],
        "margin": {
            "metric": metric,
            "direction": direction,
            "bound": bound,
            "next_bound": next_bound,
            "delta": delta,
            "next_id": next_id,
        },
        "runners_up": runners,
        "excluded": excluded,
        "flags": _flags(winner, request),
        "notes": notes,
        "budget": budget,
    }
    if remote:
        document["remote"] = True
        remote_ident = winner.get("remote_device")
        if isinstance(remote_ident, Mapping):
            document["remote_device"] = dict(remote_ident)
    if plan is not None:
        document["offrig_plan"] = plan
        worst = _worst_case(winner)
        if worst is not None:
            document["worst_case_usd"] = worst
    return document


def _any_5090(pool: Sequence[Mapping], request: Request) -> bool:
    """True when a 5090 row matches task, model, thinking, context and the pins.

    Busy and metric floors are not part of this. A 5090 row that lost on the
    bound is still a row. The "no 5090 row" note is only for a request that
    never had one.
    """

    for row in pool:
        if row.get("device") != "5090":
            continue
        if not _task_model(row, request):
            continue
        if request.thinking is not None and not _thinking_matches(_thinking(row), request.thinking):
            continue
        if request.ctx is not None and not _meets_ctx(row, request.ctx):
            continue
        if request.quote_rule is not None and _quote_rule(row) != request.quote_rule:
            continue
        if request.num_predict is not None and _num_predict(row) != request.num_predict:
            continue
        return True
    return False


def _runners(pool: Sequence[Mapping], winner: Mapping, ranked: list[Mapping], request: Request) -> list[dict]:
    ranked_ids = {str(row["id"]) for row in ranked}
    winner_id = str(winner["id"])
    ordered: list[Mapping] = [row for row in ranked if str(row["id"]) != winner_id]
    rest = [
        row
        for row in pool
        if _task_model(row, request) and str(row["id"]) not in ranked_ids and str(row["id"]) != winner_id
    ]
    rest.sort(key=lambda row: str(row["id"]))
    found: list[dict] = []
    for row in ordered + rest:
        constraint, reason = _why(row, winner, request)
        found.append({"id": str(row["id"]), "constraint": constraint, "reason": reason})
    return found


def _why(row: Mapping, winner: Mapping, request: Request) -> tuple[str, str]:
    if request.device is not None and not matches_device_filter(row.get("device"), request.device):
        return "device", f"device {row.get('device')} does not match {request.device}"
    if request.thinking is not None and not _thinking_matches(_thinking(row), request.thinking):
        return "thinking", f"thinking {_thinking(row)} does not match {request.thinking}"
    if request.ctx is not None and not _meets_ctx(row, request.ctx):
        recorded = _length_tag(row)
        shown = "none" if recorded is None else str(recorded)
        window = (row.get("settings") or {}).get("num_ctx")
        if recorded is None and isinstance(window, int):
            return "ctx", f"num_ctx {window} does not cover {request.ctx}"
        return "ctx", f"measured length {shown}, requested {request.ctx}"
    if request.quote_rule is not None and _quote_rule(row) != request.quote_rule:
        return "quote_rule", f"quote_rule {_quote_rule(row)} does not match {request.quote_rule}"
    if request.num_predict is not None and _num_predict(row) != request.num_predict:
        return "num_predict", f"num_predict {_num_predict(row)} does not match {request.num_predict}"
    if row.get("device") in request.busy:
        return "busy", f"{row.get('device')} is busy"
    if _is_runpod(row) and _plan_args(row) is None:
        return "remote_plan", "remote row has no offrig_plan profile and max_hours"
    if request.budget_left_usd is not None and _is_remote(row) and not _within_budget(row, request.budget_left_usd):
        return "budget", _budget_reason(row, request.budget_left_usd)
    for name in sorted(request.min_metrics):
        if not _meets_min(row, name, request.min_metrics[name]):
            bound = _floor_bound(row, name)
            return _public_constraint(f"min:{name}"), _short_reason(name, bound, request.min_metrics[name])
    if request.max_latency_ms is not None and not _meets_max_latency(row, request.max_latency_ms):
        bound = latency_bound_ms(row)
        return "max_latency_ms", _latency_reason(bound, request.max_latency_ms)
    if request.max_ttft_ms is not None and not _meets_ttft(row, request.max_ttft_ms):
        return "max_ttft_ms", _latency_reason(ttft_bound_ms(row), request.max_ttft_ms).replace("latency", "ttft")
    mine = _rank_bound(row, request)
    theirs = _rank_bound(winner, request)
    metric = "decode_tok_s" if request.task == "generate" else "warm_latency_ms"
    if mine is None:
        return metric, f"no conservative {metric} bound (no repeat spread)"
    if theirs is not None and mine == theirs and row.get("device") == "5090" and winner.get("device") != "5090":
        return "tie_break", "tie on the conservative bound; the choice leaves the 5090 free"
    if theirs is not None and mine == theirs:
        return "tie_break", "tie on the conservative bound; the lower row id won"
    return metric, f"conservative bound {mine:g} lost to {theirs:g}"


def _empty(
    constraint: str,
    limit: object,
    dropped: Sequence[Mapping],
    excluded: list[dict],
    request: Request,
    *,
    failure_reasons: Sequence[str] = (),
) -> Refusal:
    closest_row, closest = _closest(constraint, limit, dropped)
    experiment = experiment_of(closest_row) if closest_row is not None else "unregistered"
    gap = ""
    if closest and closest.get("gap") is not None:
        gap = f" Closest miss {closest['id']} is short by {closest['gap']:g}."
    elif closest:
        gap = f" Closest miss is {closest['id']}."
    else:
        gap = " Nothing was close."
    hint = (
        f"The constraint that emptied the set is {constraint}.{gap} Experiment {experiment} is the cell that would "
        f"fill it."
    )
    if constraint.startswith("min:") or constraint in {"decode_tok_s", "warm_latency_ms", "max_latency_ms",
        "max_ttft_ms"}:
        hint += " The bound is the conservative end of the measured spread, not the median."
    if failure_reasons:
        hint += " Measured failure reasons: " + "; ".join(failure_reasons) + "."
    return Refusal(
        {
            "code": "no_candidate",
            "message": "no row meets the constraints",
            "hint": hint,
            "constraint": _public_constraint(constraint),
            "closest": closest,
            "experiment": experiment,
            "excluded": excluded,
        }
    )


def _ttft_refusal(rows: Sequence[Mapping], request: Request, excluded: list[dict]) -> Refusal | None:
    modes = [prefix_mode(row) for row in rows]
    known = [mode for mode in modes if mode is not None]
    missing = len(known) != len(modes)
    mixed = len(set(known)) > 1
    if not missing and not mixed:
        return None
    if missing:
        code = "prefix_caching_unknown"
        hint = (
            "settings.prefix_caching is unset, so TTFT cannot be routed. "
            "Unset means unknown, not a shared cache mode."
        )
    else:
        code = "prefix_caching_mixed"
        listed = ", ".join(sorted(set(known)))
        hint = f"settings.prefix_caching differs across the compared rows ({listed}), so TTFT cannot be routed."
    sample = rows[0]
    return Refusal(
        {
            "code": code,
            "message": "TTFT is not comparable on these rows",
            "hint": hint,
            "constraint": "max_ttft_ms",
            "closest": {"id": str(sample["id"]), "metric": "ttft_ms", "bound": None, "limit": request.max_ttft_ms,
                "gap": None},
            "experiment": experiment_of(sample),
            "excluded": excluded,
        }
    )


def _closest(constraint: str, limit: object, dropped: Sequence[Mapping]) -> tuple[Mapping | None, dict | None]:
    if not dropped:
        return None, None
    if constraint.startswith("min:") and isinstance(limit, (int, float)):
        name = constraint.split(":", 1)[1]
        return _closest_short(dropped, name, float(limit), higher=True, metric=constraint)
    if constraint == "max_latency_ms" and isinstance(limit, (int, float)):
        return _closest_latency(dropped, float(limit), latency_bound_ms, "warm_latency_ms", constraint)
    if constraint == "max_ttft_ms" and isinstance(limit, (int, float)):
        return _closest_latency(dropped, float(limit), ttft_bound_ms, "ttft_ms", constraint)
    if constraint in {"decode_tok_s", "warm_latency_ms"}:
        return _closest_missing_bound(dropped, constraint)
    row = min(dropped, key=lambda item: str(item["id"]))
    bound = _identity_bound(row, constraint)
    return row, {"id": str(row["id"]), "metric": _public_constraint(constraint), "bound": bound,
        "limit": limit if isinstance(limit, (int, float)) else None, "gap": None}


def _closest_short(
    dropped: Sequence[Mapping], name: str, limit: float, *, higher: bool, metric: str
) -> tuple[Mapping | None, dict | None]:
    scored: list[tuple[float, str, Mapping, float]] = []
    for row in dropped:
        bound = _floor_bound(row, name)
        if bound is None:
            continue
        gap = (limit - bound) if higher else (bound - limit)
        scored.append((gap, str(row["id"]), row, bound))
    if not scored:
        row = min(dropped, key=lambda item: str(item["id"]))
        return row, {"id": str(row["id"]), "metric": metric, "bound": None, "limit": limit, "gap": None}
    gap, _, row, bound = min(scored)
    return row, {"id": str(row["id"]), "metric": metric, "bound": bound, "limit": limit, "gap": gap}


def _closest_latency(dropped, limit, bound_of, metric, constraint):
    scored = []
    for row in dropped:
        bound = bound_of(row)
        if bound is None:
            continue
        scored.append((bound - limit, str(row["id"]), row, bound))
    if not scored:
        row = min(dropped, key=lambda item: str(item["id"]))
        return row, {"id": str(row["id"]), "metric": metric, "bound": None, "limit": limit, "gap": None}
    gap, _, row, bound = min(scored)
    return row, {"id": str(row["id"]), "metric": constraint, "bound": bound, "limit": limit, "gap": gap}


def _closest_missing_bound(dropped: Sequence[Mapping], metric: str) -> tuple[Mapping | None, dict | None]:
    row = min(dropped, key=lambda item: str(item["id"]))
    return row, {"id": str(row["id"]), "metric": metric, "bound": None, "limit": None, "gap": None}


def _identity_bound(row: Mapping, constraint: str) -> float | None:
    if constraint == "quote_rule":
        value = _quote_rule(row)
        return float(value) if isinstance(value, int) else None
    if constraint == "num_predict":
        value = _num_predict(row)
        return float(value) if isinstance(value, int) else None
    if constraint == "ctx":
        length = _length_tag(row)
        return float(length) if length is not None else None
    return None


def _split(rows: Sequence[Mapping], request: Request, constraint: str, limit: object) -> tuple[list[Mapping],
    list[Mapping]]:
    kept: list[Mapping] = []
    dropped: list[Mapping] = []
    for row in rows:
        if _keeps(row, request, constraint, limit):
            kept.append(row)
        else:
            dropped.append(row)
    return kept, dropped


def _keeps(row: Mapping, request: Request, constraint: str, limit: object) -> bool:
    if constraint == "task":
        return row.get("task") == request.task
    if constraint == "model":
        return _model_matches(row, request.model)
    if constraint == "device":
        return matches_device_filter(row.get("device"), str(limit))
    if constraint == "thinking":
        return _thinking_matches(_thinking(row), str(limit))
    if constraint == "ctx":
        return _meets_ctx(row, int(limit))  # type: ignore[arg-type]
    if constraint == "quote_rule":
        return _quote_rule(row) == limit
    if constraint == "num_predict":
        return _num_predict(row) == limit
    if constraint == "busy":
        return row.get("device") not in request.busy
    if constraint == "remote_plan":
        return not _is_runpod(row) or _plan_args(row) is not None
    if constraint == "budget":
        if not _is_remote(row):
            return True
        return _within_budget(row, float(limit))  # type: ignore[arg-type]
    if constraint.startswith("min:"):
        return _meets_min(row, constraint.split(":", 1)[1], float(limit))  # type: ignore[arg-type]
    if constraint == "max_latency_ms":
        return _meets_max_latency(row, float(limit))  # type: ignore[arg-type]
    if constraint == "max_ttft_ms":
        return _meets_ttft(row, float(limit))  # type: ignore[arg-type]
    return True


def _public_constraint(constraint: str) -> str:
    if constraint.startswith("min:"):
        return "min_" + _norm(constraint.split(":", 1)[1])
    return constraint


def _short_reason(name: str, bound: float | None, limit: float) -> str:
    if bound is None:
        return f"no conservative bound for {name}"
    return f"conservative bound {bound:g} is under the floor {limit:g}"


def _latency_reason(bound: float | None, limit: float) -> str:
    if bound is None:
        return "no conservative latency bound"
    return f"conservative bound {bound:g} ms exceeds {limit:g} ms"


def _budget_reason(row: Mapping, budget: float) -> str:
    worst = _worst_case(row)
    if worst is None:
        return "no worst_case_usd on the row, so the budget flag cannot be checked"
    return f"worst case {worst:g} USD exceeds budget left {budget:g} USD"


def _within_budget(row: Mapping, budget: float) -> bool:
    worst = _worst_case(row)
    if worst is None:
        return False
    return worst <= budget


def _meets_min(row: Mapping, name: str, floor: float) -> bool:
    bound = _floor_bound(row, name)
    return bound is not None and bound >= floor


def _meets_max_latency(row: Mapping, cap: float) -> bool:
    bound = latency_bound_ms(row)
    return bound is not None and bound <= cap


def _meets_ttft(row: Mapping, cap: float) -> bool:
    bound = ttft_bound_ms(row)
    return bound is not None and bound <= cap


def _floor_bound(row: Mapping, name: str) -> float | None:
    if _norm(name) in _DECODE_FLOORS:
        return decode_bound(row)
    return quality_bound(row, name)


def _rank_bound(row: Mapping, request: Request) -> float | None:
    if request.task == "generate":
        return decode_bound(row)
    return latency_bound_ms(row)


def _rank_key(row: Mapping, request: Request) -> tuple:
    bound = _rank_bound(row, request)
    assert bound is not None  # noqa: S101  # rank only sees a row that has a bound
    better = -bound if request.task == "generate" else bound
    leaves_card_free = 0 if row.get("device") != "5090" else 1
    return (better, leaves_card_free, str(row.get("id")))


def decode_bound(row: Mapping) -> float | None:
    """Minimum repeat-slot decode throughput. Cold and warmup do not count."""

    samples = [
        float(slot["throughput"])
        for slot in _repeat_slots(row)
        if isinstance(slot.get("throughput"), (int, float))
    ]
    if not samples:
        return None
    return min(samples)


def latency_bound_ms(row: Mapping) -> float | None:
    """Maximum repeat-slot latency. A claim-wall spread's max is the fallback."""

    samples = [
        float(slot["latency_ms"])
        for slot in _repeat_slots(row)
        if isinstance(slot.get("latency_ms"), (int, float))
    ]
    if samples:
        return max(samples)
    spread = (row.get("latency") or {}).get("claim_wall_s")
    if isinstance(spread, dict) and isinstance(spread.get("max"), (int, float)):
        return float(spread["max"]) * 1000.0
    return None


def ttft_bound_ms(row: Mapping) -> float | None:
    """Maximum repeat-slot TTFT. A median on the row is not a bound."""

    samples = [
        float(slot["ttft_ms"])
        for slot in _repeat_slots(row)
        if isinstance(slot.get("ttft_ms"), (int, float))
    ]
    if not samples:
        return None
    return max(samples)


def quality_bound(row: Mapping, name: str) -> float | None:
    """Low end of the named quality interval."""

    want = _norm(name)
    for metric in row.get("quality") or []:
        if not isinstance(metric, dict) or _norm(str(metric.get("name", ""))) != want:
            continue
        low = (metric.get("interval") or {}).get("low")
        if isinstance(low, (int, float)):
            return float(low)
    return None


def prefix_mode(row: Mapping) -> str | None:
    """``on``, ``off``, or ``default``. Anything else, including unset, is unknown."""

    value = (row.get("settings") or {}).get("prefix_caching")
    if not isinstance(value, str):
        return None
    mode = value.casefold()
    if mode not in _KNOWN_PREFIX:
        return None
    return mode


def experiment_of(row: Mapping | None) -> str:
    """Spec id from ``experiments/<id>/``, or ``unregistered`` when the row has none."""

    if row is None:
        return "unregistered"
    spec = (row.get("provenance") or {}).get("experiment_spec")
    if not isinstance(spec, str) or not spec:
        return "unregistered"
    parts = spec.replace("\\", "/").split("/")
    if "experiments" not in parts:
        return "unregistered"
    index = parts.index("experiments")
    if index + 1 >= len(parts):
        return "unregistered"
    name = parts[index + 1]
    if not name:
        return "unregistered"
    return name.upper()


def _flags(row: Mapping, request: Request) -> list[str]:
    flags: list[str] = []
    if _is_stale(row, request.as_of):
        flags.append("stale")
    if _is_provisional(row):
        flags.append("provisional")
    return flags


def _is_stale(row: Mapping, as_of: date) -> bool:
    text = row.get("date")
    try:
        measured = date.fromisoformat(str(text))
    except (TypeError, ValueError):
        return True
    return (as_of - measured) > _STALE_AFTER


def _is_provisional(row: Mapping) -> bool:
    if row.get("provisional") is True:
        return True
    repeats = row.get("repeats")
    return not isinstance(repeats, int) or repeats < 2


def _repeat_slots(row: Mapping) -> list[Mapping]:
    slots = (row.get("artifacts") or {}).get("slots") or []
    if not isinstance(slots, list):
        return []
    return [slot for slot in slots if isinstance(slot, dict) and slot.get("phase") == "repeat"]


def _task_model(row: Mapping, request: Request) -> bool:
    return row.get("task") == request.task and _model_matches(row, request.model)


def _model_matches(row: Mapping, query: str | None) -> bool:
    if query is None or query == "":
        return True
    repo = str((row.get("model") or {}).get("repo_or_tag", ""))
    if query.casefold() in repo.casefold():
        return True
    want = _alnum(query)
    return bool(want) and want in _alnum(repo)


def _alnum(text: str) -> str:
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def _norm(name: str) -> str:
    return name.casefold().replace("-", "_").replace("/", "_")


def _thinking(row: Mapping) -> str | None:
    value = (row.get("settings") or {}).get("thinking_level")
    return str(value) if isinstance(value, str) else None


def _thinking_matches(recorded: str | None, requested: str) -> bool:
    """``off`` matches only ``off``. ``on`` matches any recorded thinking level.

    ``low``, ``medium`` and ``high`` match exactly when the request names one.
    A missing level matches nothing. The decision keeps the row's exact level.
    """

    if recorded is None:
        return False
    want, have = requested.casefold(), recorded.casefold()
    if want == "on":
        return have != "off"
    return have == want


def _quote_rule(row: Mapping) -> int | None:
    value = (row.get("settings") or {}).get("quote_rule")
    return value if isinstance(value, int) else None


def _num_predict(row: Mapping) -> int | None:
    value = (row.get("settings") or {}).get("num_predict")
    return value if isinstance(value, int) else None


def _length_tag(row: Mapping) -> int | None:
    match = _LENGTH.search(str(row.get("id", "")))
    if match is None:
        return None
    return int(match.group(1))


def _meets_ctx(row: Mapping, needed: int) -> bool:
    tagged = _length_tag(row)
    if tagged is not None:
        return tagged == needed
    window = (row.get("settings") or {}).get("num_ctx")
    if isinstance(window, int):
        return window >= needed
    return False


def _is_runpod(row: Mapping) -> bool:
    return str(row.get("device", "")).startswith("runpod:")


def _is_remote(row: Mapping) -> bool:
    return _is_runpod(row) or is_colab_device(row.get("device"))


def _remote_eligible(row: Mapping, request: Request) -> bool:
    if not is_colab_device(row.get("device")):
        return True
    if request.allow_remote:
        return True
    return request.device is not None and request.device.startswith("colab:")


def _measured_failure_reason(row: Mapping) -> str | None:
    settings = row.get("settings")
    if isinstance(settings, Mapping):
        reason = settings.get("failure_reason")
        if isinstance(reason, str) and reason.strip():
            return reason.strip()
    notes = row.get("notes")
    if isinstance(notes, str) and notes.strip():
        return notes.strip()
    return None


def _worst_case(row: Mapping) -> float | None:
    value = (row.get("settings") or {}).get("worst_case_usd")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _plan_args(row: Mapping) -> dict | None:
    raw = (row.get("settings") or {}).get("offrig_plan")
    if not isinstance(raw, dict):
        return None
    profile = raw.get("profile")
    hours = raw.get("max_hours")
    if not isinstance(profile, str) or not profile or not isinstance(hours, (int, float)):
        return None
    plan: dict = {"profile": profile, "max_hours": float(hours)}
    for key in ("note", "max_price_hr", "no_fallback", "wait_minutes", "container_disk_gb"):
        if key in raw and raw[key] is not None:
            plan[key] = raw[key]
    return plan


def _model_label(row: Mapping) -> str:
    repo = str(row["model"]["repo_or_tag"])
    revision = str(row["model"].get("revision_or_digest") or "")
    if "@" in repo or not revision or revision in repo:
        return repo
    return f"{repo}@{revision}"
