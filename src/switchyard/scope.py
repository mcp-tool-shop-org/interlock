"""Which rows may answer a task.

A direct answer uses only rows of that exact task, and only that answer can
satisfy a hard constraint. Another task's rows are proxy evidence: the answer
names those source tasks, and the proxy cannot satisfy a hard constraint.
"""

from __future__ import annotations

from typing import NamedTuple

from switchyard.tasks import TASKS


class ScopeError(Exception):
    """The requested task is not one switchyard knows."""


class Answer(NamedTuple):
    task: str
    evidence: str
    rows: tuple
    source_tasks: tuple
    satisfies_hard_constraint: bool


def select(rows: list[dict], task: str, *, allow_proxy: bool = False, exclude_families: tuple | list = ()) -> Answer:
    """Rows for ``task``. Proxy is off unless the caller asks, and it never fills a hard constraint.

    ``exclude_families`` drops rows whose ``model_family`` is named. A row with
    no family is kept: the table did not assign one, so the exclusion does not
    invent one either.
    """

    if task not in TASKS:
        raise ScopeError(f"unknown task {task!r}")
    if isinstance(exclude_families, str):
        raise ScopeError("exclude_families names lineages, it is not itself a family")
    excluded = frozenset(exclude_families)
    kept = [row for row in rows if row.get("model_family") not in excluded]
    direct = tuple(row for row in kept if row.get("task") == task)
    if direct:
        return Answer(task, "direct", direct, (task,), True)
    if not allow_proxy:
        return Answer(task, "none", (), (), False)
    proxy = tuple(row for row in kept if row.get("task") != task)
    if not proxy:
        return Answer(task, "none", (), (), False)
    sources = tuple(sorted({row["task"] for row in proxy}))
    return Answer(task, "proxy", proxy, sources, False)
