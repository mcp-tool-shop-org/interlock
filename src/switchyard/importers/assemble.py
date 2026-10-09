"""Build the routing table from pinned receipts.

Reads rnd and npu-buildout with `git show <pin>:<path>`. The pin must be a
commit in that repo. The worktree is not read. Ladder runs are in the rnd
pin. Offer rows are optional and come only from a version-1 snapshot file.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from switchyard.importers.calibration import calibration_rows
from switchyard.importers.common import ImportError_, sha256_bytes
from switchyard.importers.ladder import ladder_rows
from switchyard.importers.nli import nli_rows, prereg_names_intel_igpu
from switchyard.importers.offers import offer_rows
from switchyard.importers.pins import (
    LADDER_RUNS,
    NPU_BUILDOUT_COMMIT,
    NPU_REPO,
    RND_COMMIT,
    RND_REPO,
)
from switchyard.importers.probe import probe_rows

_CHAIN = "experiments/verifier-gold/calibration/results/2026-10-09-chain"
_TIERS = "experiments/verifier-gold/calibration/tiers.json"
_GOLD = "experiments/verifier-gold/math-ladder/ladder.jsonl"
_LADDER = "experiments/verifier-gold/math-ladder/results/2026-10-09-ladder"
_PROBES = (
    "experiments/npu-probe/results/2026-10-09-probe.json",
    "experiments/npu-probe/results/2026-10-09-mac.json",
    "experiments/npu-probe/results/2026-10-09-mac-chain.json",
)
_NLI = "experiments/verifier-gold/calibration/results/2026-10-09-nli-floor.json"
_PREREG = "experiments/verifier-gold/calibration/nli-floor.md"


def import_receipts(
    *,
    rnd_repo: Path | None = None,
    npu_repo: Path | None = None,
    offers_path: Path | None = None,
    rnd_pin: str = RND_COMMIT,
    npu_pin: str = NPU_BUILDOUT_COMMIT,
) -> tuple[list[dict], list[str]]:
    rnd = Path(rnd_repo or RND_REPO)
    npu = Path(npu_repo or NPU_REPO)
    require_pin(rnd, rnd_pin)
    require_pin(npu, npu_pin)
    rows: list[dict] = []
    skipped: list[str] = []
    rows.extend(_calibration(rnd, rnd_pin))
    rows.extend(_ladder(rnd, rnd_pin))
    for path in _PROBES:
        document = _json(show(rnd, rnd_pin, path))
        probe, skipped_here = probe_rows(
            document,
            source_path=path,
            commit_sha=rnd_pin,
            file_date="2026-10-09",
        )
        rows.extend(probe)
        skipped.extend(skipped_here)
    rows.extend(_nli(npu, npu_pin))
    if offers_path is not None:
        content = offers_path.read_bytes()
        rows.extend(offer_rows(_json(content), source_path=_offer_source(offers_path), content=content))
    _unique(rows)
    rows.sort(key=lambda row: row["id"])
    return rows, skipped


def _offer_source(path: Path) -> str:
    """A receipt path that can be committed. A home directory is never one."""

    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def require_pin(repo: Path, pin: str) -> None:
    kind = _git(repo, "cat-file", "-t", pin).decode("utf-8").strip()
    if kind != "commit":
        raise ImportError_(f"{pin} in {repo.as_posix()} is {kind}, not a commit")


def show(repo: Path, pin: str, path: str) -> bytes:
    try:
        return _git(repo, "show", f"{pin}:{path}")
    except subprocess.CalledProcessError as exc:
        raise ImportError_(f"missing {pin[:12]}:{path}") from exc


def _calibration(rnd: Path, pin: str) -> list[dict]:
    tiers = _json(show(rnd, pin, _TIERS))
    listing = _git(rnd, "ls-tree", "--name-only", pin, f"{_CHAIN}/").decode("utf-8")
    runs = sorted(
        line.rstrip("/")
        for line in listing.splitlines()
        if line.rstrip("/").rsplit("/", 1)[-1].startswith("cal-")
    )
    if len(runs) != 12:
        raise ImportError_(f"expected 12 calibration runs at the pin, found {len(runs)}")
    rows: list[dict] = []
    for run in runs:
        manifest = _json(show(rnd, pin, f"{run}/manifest.json"))
        metrics = _json(show(rnd, pin, f"{run}/metrics.json"))
        verdicts = _jsonl(show(rnd, pin, f"{run}/verdicts.jsonl"))
        rows.extend(
            calibration_rows(
                manifest,
                metrics,
                verdicts,
                tiers,
                source_dir=run,
                commit_sha=pin,
            )
        )
    return rows


def _ladder(rnd: Path, pin: str) -> list[dict]:
    gold_bytes = show(rnd, pin, _GOLD)
    gold_lines = _jsonl(gold_bytes)
    rows: list[dict] = []
    for name in LADDER_RUNS:
        base = f"{_LADDER}/{name}"
        try:
            manifest_bytes = show(rnd, pin, f"{base}/manifest.json")
            verdict_bytes = show(rnd, pin, f"{base}/verdicts.jsonl")
        except ImportError_ as exc:
            raise ImportError_(f"missing ladder run {base}") from exc
        manifest = _json(manifest_bytes)
        _check_gold(manifest, gold_bytes, name)
        rows.extend(
            ladder_rows(
                gold_lines,
                manifest,
                _jsonl(verdict_bytes),
                source_dir=base,
                gold_bytes=gold_bytes,
                verdict_bytes=verdict_bytes,
                commit_sha=pin,
            )
        )
    return rows


def _check_gold(manifest: dict, gold_bytes: bytes, name: str) -> None:
    wanted = [item["sha256"] for item in manifest.get("gold", []) if item.get("file") == "ladder.jsonl"]
    digest = sha256_bytes(gold_bytes)
    if wanted != [digest]:
        raise ImportError_(f"{name} gold sha does not match ladder.jsonl at the rnd pin")


def _nli(npu: Path, pin: str) -> list[dict]:
    prereg = show(npu, pin, _PREREG).decode("utf-8")
    document = _json(show(npu, pin, _NLI))
    return nli_rows(
        document,
        source_path=_NLI,
        commit_sha=pin,
        prereg_says_intel_igpu=prereg_names_intel_igpu(prereg),
    )


def _unique(rows: list[dict]) -> None:
    seen: set[str] = set()
    for row in rows:
        if row["id"] in seen:
            raise ImportError_(f"duplicate row id {row['id']}")
        seen.add(row["id"])


def _git(repo: Path, *args: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise ImportError_(detail or f"git {' '.join(args)} failed in {repo.as_posix()}") from exc


def _json(data: bytes) -> dict:
    return json.loads(data.decode("utf-8"))


def _jsonl(data: bytes) -> list[dict]:
    rows = []
    for line in data.decode("utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
