"""Warm-ups, repeats, and cold kept apart.

Arms are interleaved inside each round: every arm takes its nth repeat before
any arm takes repeat n+1. Warm-up samples are not part of the percentiles.
Cold is one sample per arm and is stored on its own.
"""

from __future__ import annotations

from dataclasses import dataclass


class TimingError(Exception):
    """The schedule or the summary was asked for something it cannot compute."""


@dataclass(frozen=True)
class Slot:
    phase: str
    step_id: str
    round_index: int
    role: str = "measure"
    pair: str | None = None


def schedule(step_ids: list[str], *, warmups: int, repeats: int) -> list[Slot]:
    if not step_ids:
        raise TimingError("a schedule needs at least one arm")
    if warmups < 0 or repeats < 1:
        raise TimingError("warmups must be >= 0 and repeats must be >= 1")
    slots: list[Slot] = []
    for step_id in step_ids:
        slots.append(Slot("cold", step_id, 0, "cold"))
    for round_index in range(warmups):
        for step_id in step_ids:
            slots.append(Slot("warmup", step_id, round_index))
    for round_index in range(repeats):
        for step_id in step_ids:
            slots.append(Slot("repeat", step_id, round_index))
    return slots


def percentile(values: list[float], pct: float) -> float:
    """Linear percentile. ``pct`` is 0–100. One sample returns that sample."""

    if not values:
        raise TimingError("a percentile needs at least one sample")
    if pct < 0 or pct > 100:
        raise TimingError("a percentile is between 0 and 100")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def within_band(alone: float, paired: float, *, band: float) -> bool:
    """True when ``paired`` is within ``band`` (a fraction) of ``alone``."""

    if band < 0:
        raise TimingError("a band cannot be negative")
    scale = abs(alone)
    if scale == 0:
        return paired == 0
    return abs(paired - alone) / scale <= band


def schedule_aba(pairs: list[dict], *, warmups: int, repeats: int) -> list[Slot]:
    """Alone, paired, alone, once per pair per round.

    Cold is still one slot per arm, before the rounds. The two alone slots in
    a round are the neighbours of that pair's paired slot.
    """

    if not pairs:
        raise TimingError("aba needs at least one pair")
    if warmups < 0 or repeats < 1:
        raise TimingError("warmups must be >= 0 and repeats must be >= 1")
    cold_ids: list[str] = []
    for pair in pairs:
        for key in ("alone", "paired"):
            step_id = pair[key]
            if step_id not in cold_ids:
                cold_ids.append(step_id)
    slots = [Slot("cold", step_id, 0, "cold") for step_id in cold_ids]
    for phase, count in (("warmup", warmups), ("repeat", repeats)):
        for round_index in range(count):
            for pair in pairs:
                slots.append(Slot(phase, pair["alone"], round_index, "alone-before", pair["id"]))
                slots.append(Slot(phase, pair["paired"], round_index, "paired", pair["id"]))
                slots.append(Slot(phase, pair["alone"], round_index, "alone-after", pair["id"]))
    return slots


def relative_spread(values: list[float]) -> float:
    """``(max - min) / |median|``. A zero median is 0 only when the samples are 0."""

    if not values:
        raise TimingError("a spread needs samples")
    med = percentile(values, 50)
    width = max(values) - min(values)
    if med == 0:
        return 0.0 if width == 0 else float("inf")
    return width / abs(med)


def aba_call(before: list[float], after: list[float], paired: list[float], *, band: float) -> dict:
    """Compare a paired median with the mean of its neighbouring alone medians.

    The call is inconclusive when the alone samples' own spread is greater
    than half the band. That half is the noise rule, not a second knob.
    """

    empty = {"call": "inconclusive", "alone_spread": None, "reference": None, "paired_median": None}
    if not before or not after or not paired:
        return empty
    alone = list(before) + list(after)
    spread = relative_spread(alone)
    reference = (percentile(before, 50) + percentile(after, 50)) / 2
    paired_median = percentile(paired, 50)
    if spread > (band / 2):
        call = "inconclusive"
    elif within_band(reference, paired_median, band=band):
        call = "within"
    else:
        call = "outside"
    return {
        "call": call,
        "alone_spread": None if spread == float("inf") else spread,
        "reference": reference,
        "paired_median": paired_median,
    }


@dataclass(frozen=True)
class Summary:
    cold_ms: float | None
    warm_p50_ms: float | None
    warm_p95_ms: float | None
    throughput: float | None
    throughput_unit: str | None
    compile_ms: float | None = None
    first_call_ms: float | None = None


def summarize(
    cold_ms: float | None,
    warm_ms: list[float],
    warm_throughput: list[float],
    *,
    throughput_unit: str | None = None,
    compile_ms: float | None = None,
    first_call_ms: float | None = None,
) -> Summary:
    """Percentiles use the repeat samples only. An empty warm list stays null.

    The throughput unit is the one the samples reported. This summary does not
    invent ``tok/s`` for an embed or NLI row.
    """

    latency_p50 = percentile(warm_ms, 50) if warm_ms else None
    latency_p95 = percentile(warm_ms, 95) if warm_ms else None
    throughput = percentile(warm_throughput, 50) if warm_throughput else None
    return Summary(
        cold_ms=cold_ms,
        warm_p50_ms=latency_p50,
        warm_p95_ms=latency_p95,
        throughput=throughput,
        throughput_unit=throughput_unit if throughput is not None else None,
        compile_ms=compile_ms,
        first_call_ms=first_call_ms,
    )
