"""Fill-the-card search.

Double from the model's default context, then bisect on a 2048-token grid
between the last pass and the first fail. Confirmation is a separate probe
with the long prompt. A window that exceeds ``abort_mib``, fails residency,
or looks slow against the default-context reference steps back one grid notch.

The watchdog is not this search. ``abort_mib`` is the sampler's own stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class SearchError(Exception):
    """The search was given a context it cannot start from."""


@dataclass(frozen=True)
class Observation:
    resident: bool
    peak_mib: int | None
    tok_s: float | None = None
    ttft_ms: float | None = None
    error: str | None = None


@dataclass
class SearchResult:
    max_ctx: int | None
    reason: str
    calls: list[tuple[str, int]] = field(default_factory=list)
    peak_mib: int | None = None
    tok_s: float | None = None
    ttft_ms: float | None = None


def over_margin(obs: Observation, abort_mib: int) -> bool:
    if obs.error or not obs.resident or obs.peak_mib is None:
        return True
    return obs.peak_mib > abort_mib


def is_slow(obs: Observation, reference: Observation, *, tok_ratio: float, ttft_ratio: float) -> bool:
    if reference.tok_s is None or reference.ttft_ms is None:
        return True
    if obs.tok_s is None or obs.ttft_ms is None:
        return True
    if obs.tok_s < reference.tok_s * tok_ratio:
        return True
    if obs.ttft_ms > reference.ttft_ms * ttft_ratio:
        return True
    return False


def fit_prompt(seed: str, target_tokens: int, count_tokens) -> str:
    """Repeat ``seed`` up to ``target_tokens``, then keep the shortest prefix."""

    if not isinstance(seed, str) or not seed:
        raise SearchError("a prompt seed is required")
    if target_tokens < 1:
        raise SearchError("a prompt needs at least one token")
    block = seed
    while count_tokens(block) < target_tokens:
        block += seed
        if len(block) > 1_000_000:
            raise SearchError("the seed never reaches the token target")
    low = 1
    high = len(block)
    while low < high:
        mid = (low + high) // 2
        if count_tokens(block[:mid]) >= target_tokens:
            high = mid
        else:
            low = mid + 1
    return block[:low]


def search(
    *,
    default_ctx: int,
    probe,
    abort_mib: int,
    grain: int = 2048,
    tok_ratio: float = 0.5,
    ttft_ratio: float = 2.0,
    max_ctx: int = 1_048_576,
) -> SearchResult:
    if default_ctx < 1 or grain < 1:
        raise SearchError("default context and grain must be positive")
    if abort_mib < 1:
        raise SearchError("abort_mib must be positive")
    calls: list[tuple[str, int]] = []
    loaded: dict[int, Observation] = {}

    def measure(ctx: int, kind: str) -> Observation:
        if ctx > max_ctx:
            raise SearchError(f"context {ctx} is above the search cap")
        obs = probe(ctx, kind)
        calls.append((kind, ctx))
        if kind == "load":
            loaded[ctx] = obs
        return obs

    ctx = default_ctx
    last_pass: int | None = None
    first_fail: int | None = None
    seen: set[int] = set()
    while ctx not in seen and ctx <= max_ctx:
        seen.add(ctx)
        obs = measure(ctx, "load")
        if over_margin(obs, abort_mib):
            first_fail = ctx
            break
        last_pass = ctx
        doubled = ctx * 2
        if doubled <= ctx or doubled > max_ctx:
            break
        ctx = doubled

    if last_pass is None:
        return SearchResult(None, "default context is not resident", calls)

    if first_fail is not None:
        low = last_pass
        high = first_fail
        while high - low > grain:
            mid = low + (high - low) // 2
            snapped = low + ((mid - low) // grain) * grain
            if snapped <= low:
                snapped = low + grain
            # The gap is wider than one grain, so this point stays below high.
            obs = measure(snapped, "load")
            if over_margin(obs, abort_mib):
                high = snapped
            else:
                low = snapped
        last_pass = low

    reference = loaded.get(default_ctx)
    if reference is None:
        return SearchResult(None, "the default context has no load sample", calls)

    candidate = last_pass
    while candidate >= default_ctx:
        obs = measure(candidate, "confirm")
        if over_margin(obs, abort_mib) or is_slow(obs, reference, tok_ratio=tok_ratio, ttft_ratio=ttft_ratio):
            candidate -= grain
            continue
        return SearchResult(
            candidate,
            "confirmed",
            calls,
            peak_mib=obs.peak_mib,
            tok_s=obs.tok_s,
            ttft_ms=obs.ttft_ms,
        )
    return SearchResult(None, "no confirmed window", calls)
