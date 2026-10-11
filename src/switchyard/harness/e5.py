"""Fill-the-card search.

Double from the model's default context, then bisect on a 2048-token grid
between the last pass and the first fail. Confirmation is a separate probe
with the long prompt. A window that exceeds ``abort_mib`` or fails residency
steps back one grid notch. So does a spill.

The primary spill signal is growth in the Ollama runner's shared GPU memory
(``\\GPU Process Memory(*)\\Shared Usage``) over the post-load value. The
secondary signal is prompt-processing rate
(``prompt_eval_count / prompt_eval_duration``), which does not grow with
prompt length the way time-to-first-token does. Decode tokens per second and
raw time-to-first-token are recorded and are not gates.

The default-context load must yield ``shared_mib`` (the post-load shared
usage) and ``prompt_tok_s``. A confirm that is missing either one is a spill,
not a pass. ``tok_s`` and ``ttft_ms`` may be absent. ``limit`` is ``margin``
when residency or ``abort_mib`` stopped the climb, and ``max_ctx`` when the
search cap stopped it.

The watchdog is not this search. ``abort_mib`` is the sampler's own stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class SearchError(Exception):
    """The search was given a context it cannot start from."""


SHARED_GROWTH_MIB = 256
PROMPT_RATIO = 0.5


@dataclass(frozen=True)
class Observation:
    resident: bool
    peak_mib: int | None
    tok_s: float | None = None
    ttft_ms: float | None = None
    error: str | None = None
    shared_mib: int | None = None
    prompt_tok_s: float | None = None


@dataclass
class SearchResult:
    max_ctx: int | None
    reason: str
    calls: list[tuple[str, int]] = field(default_factory=list)
    peak_mib: int | None = None
    tok_s: float | None = None
    ttft_ms: float | None = None
    limit: str | None = None
    prompt_tok_s: float | None = None
    shared_mib: int | None = None


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

    Missing ``shared_mib`` or ``prompt_tok_s`` on either side is a spill.
    Decode ``tok_s`` and ``ttft_ms`` are ignored here.
    """

    if reference.shared_mib is None or obs.shared_mib is None:
        return True
    if obs.shared_mib - reference.shared_mib > shared_growth_mib:
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

    A longer prefix is assumed to count as at least as many tokens, the same
    assumption :func:`fit_prompt` makes for characters. This trim is the Ollama
    dataset rule. Engines with a local tokenizer keep :func:`fit_prompt`.
    """

    if not isinstance(seed, str) or not seed.strip():
        raise SearchError("a prompt seed is required")
    if isinstance(target_tokens, bool) or not isinstance(target_tokens, int) or target_tokens < 1:
        raise SearchError("a prompt needs at least one token")
    block = seed
    counted = _as_count(count_tokens(block))
    while counted < target_tokens:
        block += seed
        if len(block) > 1_000_000:
            raise SearchError("the seed never reaches the token target")
        counted = _as_count(count_tokens(block))
    cuts = _word_cuts(block)
    low = 0
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
    calls: list[tuple[str, int]] = []
    loaded: dict[int, Observation] = {}
    limit: str | None = None

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
            limit = "margin"
            break
        last_pass = ctx
        doubled = ctx * 2
        if doubled <= ctx or doubled > max_ctx:
            limit = "max_ctx"
            break
        ctx = doubled

    if last_pass is None:
        return SearchResult(None, "default context is not resident", calls, limit="margin")

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

    reference = loaded[default_ctx]

    candidate = last_pass
    while candidate >= default_ctx:
        obs = measure(candidate, "confirm")
        spilled = is_spill(
            obs,
            reference,
            shared_growth_mib=shared_growth_mib,
            prompt_ratio=prompt_ratio,
        )
        if over_margin(obs, abort_mib) or spilled:
            candidate -= grain
            continue
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
        )
    return SearchResult(None, "no confirmed window", calls, limit=limit)
