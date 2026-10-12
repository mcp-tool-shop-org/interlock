"""Exact task names. A bare ``verify`` is not a task.

Results do not move between these names. The family of a tag is not decided
here: that lookup is the committed table, one exact key at a time.
"""

from __future__ import annotations

TASKS = frozenset(
    {
        "embed",
        "rerank",
        "nli",
        "generate",
        "engine",
        "offer",
        "verify:grounded",
        "verify:reasoning",
        "verify:math-ladder",
        "judge:planted-error",
        "judge:same-meaning-rewrite",
    }
)

# Calibration check_type values that this import knows. Anything else stops.
CHECK_TO_TASK = {
    "grounded": "verify:grounded",
    "reasoning": "verify:reasoning",
}

MATH_LADDER_TASK = "verify:math-ladder"

# Rate names a judge row may carry. Each one is its own metric, with its own
# n and interval. Imported chain rows keep false_accept and abstain; those
# numbers are not rewritten into these names.
JUDGE_RATES = (
    "false_reject_rate",
    "false_accept_rate",
    "abstain_rate",
    "consensus_agreement",
)