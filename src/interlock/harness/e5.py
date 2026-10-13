"""Fill-the-card search.

The first window is the search cap: the trained context length, or the spec
cap when that is lower. Nothing above that cap is loaded. When the first
window fits, the search may still climb toward the cap and bisect on the
grain. When it does not fit, the search halves until a window does (floor:
one grain), then bisects upward between that window and the one that failed.

Confirmation is a separate probe. A window that exceeds ``abort_mib`` or
fails residency steps back one grid notch. So does a spill.

The primary spill signal is growth in the 5090 adapter's shared GPU memory
over the model's own lowest resident window, which is also the prompt-rate
reference. A pre-window reading is the idle sample. It is not this
comparison. The secondary signal is prompt-processing rate
(``prompt_eval_count / prompt_eval_duration``). Either one trips the spill.
A missing shared reading or a missing prompt rate is a spill. Decode tokens
per second and raw time-to-first-token are recorded and are not gates.

``limit`` is ``margin`` when residency or ``abort_mib`` stopped the search,
and ``max_ctx`` when the cap stopped it.

The watchdog is not this search. ``abort_mib`` is the sampler's own stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class SearchError(Exception):
    """The search was given a context it cannot start from."""


SHARED_GROWTH_MIB = 256
PROMPT_RATIO = 0.5
# A matching allowance for the idle-card pin. Not the spill threshold.
ADAPTER_MATCH_TOLERANCE_MIB = 512


@dataclass(frozen=True)
class Observation:
    resident: bool
    peak_mib: int | None
    tok_s: float | None = None
    ttft_ms: float | None = None
    error: str | None = None
    shared_mib: int | None = None
    prompt_tok_s: float | None = None
    shared_baseline: int | None = None
    baseline_taken: bool = False
    dedicated_delta: int | None = None


@dataclass
class SearchResult:
    max_ctx: int | None
    reason: str
    calls: list[tuple] = field(default_factory=list)
    peak_mib: int | None = None
    tok_s: float | None = None
    ttft_ms: float | None = None
    limit: str | None = None
    prompt_tok_s: float | None = None
    shared_mib: int | None = None
    dedicated_delta: int | None = None


def over_margin(obs: Observation, abort_mib: int) -> bool:
    if obs.error or not obs.resident or obs.peak_mib is None:
        return True
    return obs.peak_mib > abort_mib


def is_spill(
    obs: Observation,
    reference: Observation,
    *,
    shared_growth_mib: int,
    prompt_ratio: float,
) -> bool:
    """True when shared memory grew past the threshold, or the prompt rate fell.

    The baseline is ``reference``, the lowest resident window. A pre-window
    reading stored on the observation is the idle sample and is not this
    comparison. Missing shared usage or ``prompt_tok_s`` on either side is a
    spill. Decode ``tok_s`` and ``ttft_ms`` are ignored here.
    """

    baseline = reference.shared_mib
    if baseline is None or obs.shared_mib is None:
        return True
    if obs.shared_mib - baseline > shared_growth_mib:
        return True
    if reference.prompt_tok_s is None or obs.prompt_tok_s is None:
        return True
    if obs.prompt_tok_s < reference.prompt_tok_s * prompt_ratio:
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


def _word_cuts(text: str) -> list[int]:
    """Ends of each word, and of the whitespace that follows it.

    A cut is an index into ``text``. The prefix ``text[:cut]`` ends on a word
    boundary, so the trim never splits a word.
    """

    cuts: list[int] = []
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        while index < length and not text[index].isspace():
            index += 1
        cuts.append(index)
        space = index
        while space < length and text[space].isspace():
            space += 1
        if space != index:
            cuts.append(space)
        index = space
    return cuts


def _as_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SearchError("the token count must be a non-negative integer")
    return value


def fit_prompt_words(seed: str, target_tokens: int, count_tokens) -> tuple[str, int, str]:
    """Repeat the whole seed, then trim on word boundaries.

    The prefix counts as exactly ``target_tokens`` when a word boundary lands
    on that count. Otherwise it is the shortest word-prefix whose count is
    still at least the target. The result is ``(prefix, count, fit)`` where
    ``fit`` is ``exact`` or ``shortest_at_or_above``.

    The number of repeats is estimated from the seed's own count, so a long
    target takes a few counts, not one count per repeat. Adding one seed at a
    time cost thousands of counts at a 118k-token target, each re-reading the
    whole prompt (E5's first live run, 2026-10-12). The trim then searches only
    the words past the longest prefix already known to be short.

    A longer prefix is assumed to count as at least as many tokens, the same
    assumption :func:`fit_prompt` makes for characters. This trim is the Ollama
    dataset rule. Engines with a local tokenizer keep :func:`fit_prompt`.
    """

    if not isinstance(seed, str) or not seed.strip():
        raise SearchError("a prompt seed is required")
    if isinstance(target_tokens, bool) or not isinstance(target_tokens, int) or target_tokens < 1:
        raise SearchError("a prompt needs at least one token")
    per_seed = _as_count(count_tokens(seed))
    repeats = max(1, -(-target_tokens // per_seed)) if per_seed > 0 else 1
    if len(seed) * repeats > 1_000_000:
        raise SearchError("the seed never reaches the token target")
    short_len = 0  # the longest prefix length known to count below the target
    if repeats > 1:
        below = seed * (repeats - 1)
        if _as_count(count_tokens(below)) < target_tokens:
            short_len = len(below)
        else:
            repeats = repeats - 1
    block = seed * repeats
    counted = _as_count(count_tokens(block))
    while counted < target_tokens:
        short_len = len(block)
        per = max(1, counted // repeats)
        more = max(1, -(-(target_tokens - counted) // per))
        block += seed * more
        repeats += more
        if len(block) > 1_000_000:
            raise SearchError("the seed never reaches the token target")
        counted = _as_count(count_tokens(block))
    cuts = _word_cuts(block)
    low = 0
    while low < len(cuts) - 1 and cuts[low] <= short_len:
        low += 1
    high = len(cuts) - 1
    best_text, best_count = block, counted
    while low < high:
        mid = (low + high) // 2
        text = block[: cuts[mid]]
        number = _as_count(count_tokens(text))
        if number >= target_tokens:
            high = mid
            best_text, best_count = text, number
        else:
            low = mid + 1
    fit = "exact" if best_count == target_tokens else "shortest_at_or_above"
    return best_text, best_count, fit


def _reading(kind: str, ctx: int, obs: Observation, reason: str) -> tuple:
    """One search call, in the order the receipt stores it."""

    return (kind, ctx, obs.peak_mib, obs.shared_mib, obs.prompt_tok_s, reason)


def _load_reason(obs: Observation, abort_mib: int) -> str:
    if obs.error:
        return obs.error
    if not obs.resident:
        return "not resident"
    if obs.peak_mib is None or obs.peak_mib > abort_mib:
        return "over abort_mib"
    return "resident"


def _halve(ctx: int, grain: int) -> int | None:
    """The next smaller grain multiple, or None at the floor."""

    if ctx <= grain:
        return None
    nxt = (ctx // 2 // grain) * grain
    if nxt < grain:
        nxt = grain
    if nxt >= ctx:
        return None
    return nxt


def _bisect(low: int, high: int, grain: int, measure, abort_mib: int) -> int:
    """The largest grain point in ``(low, high)`` that fits. ``low`` already fits."""

    while high - low > grain:
        mid = low + (high - low) // 2
        snapped = low + ((mid - low) // grain) * grain
        if snapped <= low:
            snapped = low + grain
        if snapped >= high:
            break
        obs = measure(snapped, "load")
        if over_margin(obs, abort_mib):
            high = snapped
        else:
            low = snapped
    return low


def baseline_is_spilling(idle_mib: int, baseline_mib: int, host_sum_mib: float, *, allowance_mib: int = SHARED_GROWTH_MIB) -> bool:
    """The lowest window, minus idle, is more than the CUDA_Host sum plus the allowance.

    Equality at the allowance is not spilling. The caller has the three readings.
    """

    return (baseline_mib - idle_mib) - host_sum_mib > allowance_mib


def lowest_resident_shared(calls: list[tuple]) -> tuple[int, int] | None:
    """``(ctx, shared_mib)`` of the lowest resident load. That window is the baseline."""

    chosen: tuple[int, int] | None = None
    for call in calls:
        if len(call) < 6 or call[0] != "load" or call[5] != "resident":
            continue
        ctx, shared = call[1], call[3]
        if isinstance(ctx, bool) or isinstance(shared, bool):
            continue
        if not isinstance(ctx, int) or not isinstance(shared, int):
            continue
        if chosen is None or ctx < chosen[0]:
            chosen = (ctx, shared)
    return chosen


def shared_deltas(calls: list[tuple], baseline_mib: int) -> list[dict]:
    """Each window's shared reading minus the baseline. A missing reading stays None."""

    rows = []
    for call in calls:
        shared = call[3] if len(call) > 3 else None
        delta = None
        if isinstance(shared, int) and not isinstance(shared, bool):
            delta = shared - baseline_mib
        rows.append({"kind": call[0], "ctx": call[1], "delta_mib": delta})
    return rows


def _lowest_resident(loaded: dict[int, Observation], abort_mib: int) -> Observation | None:
    fitting = [ctx for ctx, obs in loaded.items() if not over_margin(obs, abort_mib)]
    if not fitting:
        return None
    return loaded[min(fitting)]


def search(
    *,
    default_ctx: int,
    probe,
    abort_mib: int,
    grain: int = 2048,
    shared_growth_mib: int = SHARED_GROWTH_MIB,
    prompt_ratio: float = PROMPT_RATIO,
    max_ctx: int = 1_048_576,
) -> SearchResult:
    if default_ctx < 1 or grain < 1:
        raise SearchError("default context and grain must be positive")
    if abort_mib < 1:
        raise SearchError("abort_mib must be positive")
    if shared_growth_mib < 0:
        raise SearchError("shared_growth_mib cannot be negative")
    if not 0 < prompt_ratio <= 1:
        raise SearchError("prompt_ratio must be in (0, 1]")
    calls: list[tuple] = []
    loaded: dict[int, Observation] = {}
    limit: str | None = None

    def measure(ctx: int, kind: str) -> Observation:
        if ctx > max_ctx:
            raise SearchError(f"context {ctx} is above the search cap")
        obs = probe(ctx, kind)
        reason = _load_reason(obs, abort_mib) if kind == "load" else "confirm"
        calls.append(_reading(kind, ctx, obs, reason))
        if kind == "load":
            loaded[ctx] = obs
        return obs

    first = measure(default_ctx, "load")
    last_pass: int | None = None
    if over_margin(first, abort_mib):
        # The cap does not fit. Halve to a resident window, then bisect upward.
        high = default_ctx
        low: int | None = None
        ctx = default_ctx
        while True:
            nxt = _halve(ctx, grain)
            if nxt is None:
                break
            ctx = nxt
            obs = measure(ctx, "load")
            if not over_margin(obs, abort_mib):
                low = ctx
                break
            high = ctx
        if low is None:
            return SearchResult(None, "no resident window", calls, limit="margin")
        last_pass = _bisect(low, high, grain, measure, abort_mib)
        limit = "margin"
    else:
        ctx = default_ctx
        last_pass = default_ctx
        first_fail: int | None = None
        seen = {default_ctx}
        while True:
            doubled = ctx * 2
            if doubled <= ctx or doubled > max_ctx:
                limit = "max_ctx"
                break
            ctx = doubled
            if ctx in seen:
                break
            seen.add(ctx)
            obs = measure(ctx, "load")
            if over_margin(obs, abort_mib):
                first_fail = ctx
                limit = "margin"
                break
            last_pass = ctx
        if first_fail is not None:
            last_pass = _bisect(last_pass, first_fail, grain, measure, abort_mib)

    reference = _lowest_resident(loaded, abort_mib)
    if reference is None or last_pass is None:
        return SearchResult(None, "no resident window", calls, limit=limit or "margin")

    candidate = last_pass
    while candidate >= grain:
        obs = measure(candidate, "confirm")
        spilled = is_spill(
            obs,
            reference,
            shared_growth_mib=shared_growth_mib,
            prompt_ratio=prompt_ratio,
        )
        if over_margin(obs, abort_mib) or spilled:
            reason = "spill" if spilled and not over_margin(obs, abort_mib) else _load_reason(obs, abort_mib)
            calls[-1] = _reading("confirm", candidate, obs, reason)
            candidate -= grain
            continue
        calls[-1] = _reading("confirm", candidate, obs, "confirmed")
        return SearchResult(
            candidate,
            "confirmed",
            calls,
            peak_mib=obs.peak_mib,
            tok_s=obs.tok_s,
            ttft_ms=obs.ttft_ms,
            limit=limit,
            prompt_tok_s=obs.prompt_tok_s,
            shared_mib=obs.shared_mib,
            dedicated_delta=obs.dedicated_delta,
        )
    return SearchResult(None, "no confirmed window", calls, limit=limit)
