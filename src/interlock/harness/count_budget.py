"""How long one Ollama token count may take.

The allowance is twice a cold load of the largest model in the run's roster,
plus a prefill term for the longest count the spec can ask for. Sizes come
from ``/api/tags``. A missing or unusable size uses ``MISSING_TAG_BYTES``.
``COLD_LOAD_S_PER_GIB`` and ``PREFILL_TOKENS_PER_S`` are config allowances,
not times measured on this rig. ``PREFILL_TOKENS_PER_S`` is the Publisher's
122_000 tokens in 19.6 seconds.
"""

from __future__ import annotations

GIB = 1024**3
COLD_LOAD_S_PER_GIB = 20.0
MISSING_TAG_BYTES = 16 * GIB
PREFILL_TOKENS_PER_S = 6000.0


def _bytes(size: object) -> int:
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        return MISSING_TAG_BYTES
    return size


def prompt_tokens(spec: dict) -> int:
    """The longest count this spec can ask for, in tokens.

    An interleaved run uses ``dataset.target_tokens``. A fill-card run also
    counts a confirmation prompt, a fraction of ``max_ctx``.
    """

    tokens = 0
    dataset = spec.get("dataset") if isinstance(spec.get("dataset"), dict) else None
    if isinstance(dataset, dict):
        target = dataset.get("target_tokens")
        if isinstance(target, int) and not isinstance(target, bool) and target > 0:
            tokens = target
    if spec.get("procedure") == "fill-card":
        fraction = spec.get("confirm_prompt_fraction")
        window = spec.get("max_ctx")
        if (
            isinstance(fraction, (int, float))
            and not isinstance(fraction, bool)
            and isinstance(window, int)
            and not isinstance(window, bool)
            and window > 0
            and 0 < float(fraction) <= 1
        ):
            tokens = max(tokens, int(float(window) * float(fraction)))
    return tokens


def count_timeout_s(sizes: list, *, prompt_tokens: int = 0) -> float:
    """Seconds: twice the cold-load allowance, plus the prefill term."""

    largest = max((_bytes(size) for size in sizes), default=MISSING_TAG_BYTES)
    load = 2.0 * (largest / GIB) * COLD_LOAD_S_PER_GIB
    if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
        return load
    return load + (prompt_tokens / PREFILL_TOKENS_PER_S)


def roster_pairs(spec: dict) -> list[tuple[str, str]]:
    """Ollama ``(model, device)`` pairs the run can load."""

    pairs: list[tuple[str, str]] = []
    if spec.get("procedure") == "fill-card":
        entries = spec.get("models") or []
        for entry in entries:
            if isinstance(entry, dict) and isinstance(entry.get("model"), str):
                device = entry.get("device") if isinstance(entry.get("device"), str) else "5090"
                pairs.append((entry["model"], device))
        return pairs
    for entry in spec.get("arms") or []:
        if not isinstance(entry, dict) or entry.get("engine") != "ollama":
            continue
        if isinstance(entry.get("model"), str):
            device = entry.get("device") if isinstance(entry.get("device"), str) else "5090"
            pairs.append((entry["model"], device))
    return pairs


def roster_timeout(adapter, spec: dict) -> float:
    """The count budget for this spec, from ``/api/tags`` sizes."""

    reader = getattr(adapter, "tag_size", None)
    sizes: list = []
    seen: set[tuple[str, str]] = set()
    for model, device in roster_pairs(spec):
        if (model, device) in seen:
            continue
        seen.add((model, device))
        if not callable(reader):
            sizes.append(None)
            continue
        try:
            sizes.append(reader(model, device))
        except Exception:
            sizes.append(None)
    return count_timeout_s(sizes, prompt_tokens=prompt_tokens(spec))
