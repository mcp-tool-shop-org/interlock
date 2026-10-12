"""Warm up until two consecutive calls agree, then measure.

An integer ``warmups`` stays a fixed schedule built by :func:`interlock.harness.timing.schedule`.
``warmups: {until_steady, max, tolerance}`` issues the next warm-up round only
while an arm has not yet agreed, and never more than ``max`` warm calls.
Repeats start after every arm in the block is steady, dropped, or out of
warm-ups. Warm-up samples stay out of the percentiles because they are a
``warmup`` phase, and the summary reads repeats only.
"""

from __future__ import annotations

from collections import deque

from interlock.harness.timing import Slot


def agrees(earlier: float, later: float, tolerance: float) -> bool:
    """True when ``later`` is within ``tolerance`` (a fraction) of ``earlier``.

    A zero earlier sample agrees only with another zero. The scale is the
    earlier call, the same way a band uses its reference.
    """

    scale = abs(earlier)
    if scale == 0:
        return later == 0
    return abs(later - earlier) / scale <= tolerance


def call_metric(result, metric: str | None) -> float | None:
    """The decision metric on one call. A summary-only name uses that call's latency."""

    if not isinstance(metric, str) or "." not in metric:
        return None
    tail = metric.split(".", 1)[1]
    if tail == "throughput":
        value = result.throughput
    elif tail in {"cold_ms", "warm_p50_ms", "warm_p95_ms", "first_call_ms"}:
        value = result.latency_ms
    elif tail == "compile_ms":
        value = result.compile_ms
    elif tail == "compile_warm_reload_ms":
        resolved = result.resolved if isinstance(getattr(result, "resolved", None), dict) else {}
        value = resolved.get("compile_warm_reload_ms")
    else:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def warmup_policy(spec: dict) -> dict | None:
    """The until-steady policy, or None when ``warmups`` is a fixed integer."""

    value = spec.get("warmups")
    if isinstance(value, int):
        return None
    return {"max": int(value["max"]), "tolerance": float(value["tolerance"])}


def warmup_trace(records: list[dict], arm_id: str) -> list[dict]:
    """Every warm-up call's timing, in order. Repeats are not included."""

    trace = []
    for row in records:
        if row["step_id"] != arm_id or row["phase"] != "warmup":
            continue
        trace.append(
            {
                "round": row["round"],
                "latency_ms": row["latency_ms"],
                "throughput": row["throughput"],
            }
        )
    return trace


class BlockPlan:
    """One block's arms, whether it skips the cold slot, and how many repeats it owes."""

    def __init__(self, block_id: str | None, arms: list[str], drift: bool, repeats: int) -> None:
        self.id = block_id
        self.arms = list(arms)
        self.drift = drift
        self.repeats = repeats


class SlotStream:
    """Slots in run order. A dynamic policy may append the next round after a result."""

    def __init__(self, items: list[tuple[Slot, str | None]], dynamic: UntilSteady | None = None) -> None:
        self.queue: deque[tuple[Slot, str | None]] = deque(items)
        self.needed: dict[str, int] = {}
        for slot, _block_id in items:
            self._count(slot)
        self.dynamic = dynamic
        if dynamic is not None:
            dynamic.attach(self)

    def _count(self, slot: Slot) -> None:
        self.needed[slot.step_id] = self.needed.get(slot.step_id, 0) + 1

    def add(self, slot: Slot, block_id: str | None) -> None:
        self._count(slot)
        self.queue.append((slot, block_id))

    def pop(self) -> tuple[Slot, str | None] | None:
        if not self.queue and self.dynamic is not None:
            if not self.dynamic.on_empty():
                return None
        if not self.queue:
            return None
        return self.queue.popleft()

    def note(self, slot: Slot, value: float | None) -> None:
        if self.dynamic is not None:
            self.dynamic.note(slot, value)

    def accepts_more(self, arm_id: str) -> bool:
        """True while a later slot for this arm may still be appended.

        A fixed schedule never appends. Until-steady appends the next warm
        round, and the repeats, only after the current round is noted. An arm
        whose count already matches must not finish during that wait.
        """

        del arm_id
        if self.dynamic is None:
            return False
        return not self.dynamic.finished_warm

    def was_dropped(self, arm_id: str) -> bool:
        if self.dynamic is None:
            return False
        return arm_id in self.dynamic.dropped

    def is_steady(self, arm_id: str) -> bool:
        if self.dynamic is None:
            return False
        return bool(self.dynamic.settled.get(arm_id))


class UntilSteady:
    """Issue warm-ups until two consecutive calls agree, up to ``max_n``."""

    def __init__(self, blocks: list[BlockPlan], max_n: int, tolerance: float) -> None:
        self.blocks = blocks
        self.max_n = max_n
        self.tolerance = tolerance
        self.index = -1
        self.stream: SlotStream | None = None
        self.settled: dict[str, bool] = {}
        self.samples: dict[str, list[float]] = {}
        self.steady: set[str] = set()
        self.dropped: set[str] = set()
        self.waiting: dict[str, bool] = {}
        self.round = 0
        self.accepting = False
        self.finished_warm = False

    def attach(self, stream: SlotStream) -> None:
        self.stream = stream
        self.on_empty()

    def on_empty(self) -> bool:
        self.index += 1
        if self.index >= len(self.blocks):
            return False
        block = self.blocks[self.index]
        self.samples = {}
        self.steady = set()
        self.dropped = set()
        self.waiting = {}
        self.round = 0
        self.accepting = False
        self.finished_warm = False
        if not block.drift:
            for arm in block.arms:
                self.stream.add(Slot("cold", arm, 0, "cold"), block.id)
        self._start_round()
        return True

    def note(self, slot: Slot, value: float | None) -> None:
        if not self.accepting or slot.phase != "warmup":
            return
        if slot.step_id not in self.waiting or self.waiting[slot.step_id]:
            return
        if value is None:
            self.dropped.add(slot.step_id)
        else:
            series = self.samples.setdefault(slot.step_id, [])
            series.append(value)
            if len(series) >= 2 and agrees(series[-2], series[-1], self.tolerance):
                self.steady.add(slot.step_id)
        self.waiting[slot.step_id] = True
        if all(self.waiting.values()):
            self.round += 1
            self._start_round()

    def _start_round(self) -> None:
        block = self.blocks[self.index]
        if self.round >= self.max_n:
            self._finish_warm()
            return
        arms = [arm for arm in block.arms if arm not in self.steady and arm not in self.dropped]
        if not arms:
            self._finish_warm()
            return
        self.accepting = True
        self.waiting = {arm: False for arm in arms}
        for arm in arms:
            self.stream.add(Slot("warmup", arm, self.round), block.id)

    def _finish_warm(self) -> None:
        if self.finished_warm:
            return
        self.finished_warm = True
        self.accepting = False
        self.waiting = {}
        block = self.blocks[self.index]
        for arm in block.arms:
            if arm in self.dropped:
                continue
            self.settled[arm] = arm in self.steady
        for round_index in range(block.repeats):
            for arm in block.arms:
                if arm in self.dropped:
                    continue
                self.stream.add(Slot("repeat", arm, round_index), block.id)
