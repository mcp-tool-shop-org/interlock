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


def schedule(step_ids: list[str], *, warmups: int, repeats: int) -> list[Slot]:
    if not step_ids:
        raise TimingError("a schedule needs at least one arm")
    if warmups < 0 or repeats < 1:
        raise TimingError("warmups must be >= 0 and repeats must be >= 1")
    slots: list[Slot] = []
    for step_id in step_ids:
        slots.append(Slot("cold", step_id, 0))
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


@dataclass(frozen=True)
class Summary:
    cold_ms: float | None
    warm_p50_ms: float | None
    warm_p95_ms: float | None
    throughput: float | None
    throughput_unit: str | None


def summarize(cold_ms: float | None, warm_ms: list[float], warm_throughput: list[float]) -> Summary:
    """Percentiles use the repeat samples only. An empty warm list stays null."""

    latency_p50 = percentile(warm_ms, 50) if warm_ms else None
    latency_p95 = percentile(warm_ms, 95) if warm_ms else None
    throughput = percentile(warm_throughput, 50) if warm_throughput else None
    return Summary(
        cold_ms=cold_ms,
        warm_p50_ms=latency_p50,
        warm_p95_ms=latency_p95,
        throughput=throughput,
        throughput_unit="tok/s" if throughput is not None else None,
    )
