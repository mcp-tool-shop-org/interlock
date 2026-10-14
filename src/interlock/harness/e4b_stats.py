"""E4b's estimate: Hodges-Lehmann, an exact Wilcoxon interval, and TOST.

The interval needs no seed. A percentile bootstrap of the median is a footnote
and never decides. Zeros stay in the Hodges-Lehmann averages. The signed-rank
probabilities are the usual no-tie null distribution on every observation.
"""

from __future__ import annotations

import math
import random


class StatsError(Exception):
    """The estimate was asked for something it cannot compute."""


def effect(before: float, after: float, paired: float) -> float:
    """``ln(paired) - ln(mean(before, after))``. Both sides must be positive.

    Decode tokens per second make a slower pair negative. Latency makes a
    slower pair positive. The formula is the same; the metric supplies the sign.
    """

    if before <= 0 or after <= 0 or paired <= 0:
        raise StatsError("a log ratio needs positive measurements")
    reference = (before + after) / 2
    return math.log(paired) - math.log(reference)


def walsh_averages(values: list[float]) -> list[float]:
    """Every ``(d_i + d_j) / 2`` for ``i <= j``, sorted."""

    if not values:
        raise StatsError("Hodges-Lehmann needs at least one effect")
    found = []
    for i, left in enumerate(values):
        for right in values[i:]:
            found.append((left + right) / 2)
    found.sort()
    return found


def median(values: list[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        raise StatsError("a median needs at least one value")
    mid = count // 2
    if count % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def hodges_lehmann(values: list[float]) -> float:
    """The median of the Walsh averages."""

    return median(walsh_averages(values))


def _signed_rank_counts(n: int) -> list[int]:
    """How many sign patterns give each positive-rank sum. Length is ``n(n+1)/2 + 1``."""

    if n < 1:
        raise StatsError("the signed-rank distribution needs at least one observation")
    total = n * (n + 1) // 2
    ways = [0] * (total + 1)
    ways[0] = 1
    for rank in range(1, n + 1):
        for score in range(total, rank - 1, -1):
            ways[score] += ways[score - rank]
    return ways


def signed_rank_cdf(n: int, score: int) -> float:
    """``P(T+ <= score)`` under the no-tie null. A negative score has probability 0."""

    ways = _signed_rank_counts(n)
    if score < 0:
        return 0.0
    capped = min(score, len(ways) - 1)
    return sum(ways[: capped + 1]) / (2**n)


def exact_interval(values: list[float], *, level: float = 0.90) -> dict:
    """The narrowest Wilcoxon interval whose exact coverage is at least ``level``.

    Coverage of the interval from Walsh order statistic ``k`` (1-based) through
    ``N-k+1`` is ``1 - 2 P(T+ <= k-1)``. The largest such ``k`` is the narrowest
    interval. When even ``k = 1`` falls short, the interval is the full set of
    Walsh averages and the achieved level is 1.
    """

    if not 0 < level < 1:
        raise StatsError("the confidence level must be between 0 and 1")
    ordered = walsh_averages(values)
    count = len(values)
    best_k = 0
    for k in range(1, len(ordered) // 2 + 1):
        achieved = 1 - 2 * signed_rank_cdf(count, k - 1)
        if achieved + 1e-12 >= level:
            best_k = k
        else:
            break
    if best_k == 0:
        return {
            "low": ordered[0],
            "high": ordered[-1],
            "achieved": 1.0,
            "k": 0,
        }
    achieved = 1 - 2 * signed_rank_cdf(count, best_k - 1)
    return {
        "low": ordered[best_k - 1],
        "high": ordered[len(ordered) - best_k],
        "achieved": achieved,
        "k": best_k,
    }


def tost(low: float, high: float, *, band: float) -> dict:
    """Equivalence against ``+/- band`` on the log scale.

    Free when the whole interval lies inside the band. Not free when the whole
    interval lies beyond one side. Anything else is inconclusive. An interval
    straddles a bound when that bound sits strictly inside it; that is the
    extension trigger.
    """

    if band <= 0:
        raise StatsError("the band must be positive")
    if low > high:
        raise StatsError("the interval is backwards")
    inside = low >= -band and high <= band
    above = low > band
    below = high < -band
    straddles_upper = low < band < high
    straddles_lower = low < -band < high
    if inside:
        verdict = "free"
    elif above or below:
        verdict = "not_free"
    else:
        verdict = "inconclusive"
    return {
        "verdict": verdict,
        "side": "above" if above else "below" if below else None,
        "straddles": straddles_upper or straddles_lower,
    }


def bootstrap_median(values: list[float], *, resamples: int = 50000, seed: int = 20261012, level: float = 0.90) -> dict:
    """Percentile bootstrap of the sample median. A footnote, never a verdict."""

    if resamples < 1:
        raise StatsError("the bootstrap needs at least one resample")
    if not values:
        raise StatsError("the bootstrap needs at least one effect")
    rng = random.Random(seed)  # noqa: S311  # seeded bootstrap footnote, not a key
    count = len(values)
    found = []
    for _ in range(resamples):
        sample = [values[rng.randrange(count)] for _ in range(count)]
        found.append(median(sample))
    found.sort()
    tail = (1 - level) / 2
    return {
        "median": median(found),
        "low": _order_stat(found, tail),
        "high": _order_stat(found, 1 - tail),
        "resamples": resamples,
        "seed": seed,
        "level": level,
        "decides": False,
    }


def _order_stat(ordered: list[float], quantile: float) -> float:
    """Linear order statistic. ``quantile`` is 0–1."""

    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * quantile
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def decide(values: list[float], *, band: float = 0.10, level: float = 0.90, allow_extension: bool = True) -> dict:
    """One pair's verdict from its ``d_i`` list.

    Ten triplets whose interval straddles a bound ask for the one extension
    to fifteen. The caller runs those triplets and calls this again with
    ``allow_extension`` false.
    """

    estimate = hodges_lehmann(values)
    interval = exact_interval(values, level=level)
    call = tost(interval["low"], interval["high"], band=band)
    extend = allow_extension and len(values) == 10 and call["straddles"]
    verdict = "extend" if extend else call["verdict"]
    return {
        "d_i": list(values),
        "hodges_lehmann": estimate,
        "low": interval["low"],
        "high": interval["high"],
        "achieved": interval["achieved"],
        "verdict": verdict,
        "side": None if extend else call["side"],
        "straddles": call["straddles"],
        "extension_used": False,
    }
