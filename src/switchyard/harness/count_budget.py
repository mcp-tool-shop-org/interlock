"""How long one Ollama token count may take.

The allowance is twice a cold load of the largest model in the run's roster.
Sizes come from ``/api/tags``. A missing or unusable size uses
``MISSING_TAG_BYTES``. ``COLD_LOAD_S_PER_GIB`` is a config allowance, not a
time measured on this rig.
"""

from __future__ import annotations

GIB = 1024**3
COLD_LOAD_S_PER_GIB = 20.0
MISSING_TAG_BYTES = 16 * GIB


def _bytes(size: object) -> int:
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        return MISSING_TAG_BYTES
    return size


def count_timeout_s(sizes: list) -> float:
    """Seconds: twice the cold-load allowance of the largest size."""

    largest = max((_bytes(size) for size in sizes), default=MISSING_TAG_BYTES)
    return 2.0 * (largest / GIB) * COLD_LOAD_S_PER_GIB


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
    return count_timeout_s(sizes)
