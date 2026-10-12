"""Build the routing table from pinned receipts.

Reads rnd, npu-buildout and aspire-si with `git show <pin>:<path>`. The pin
must be a commit in that repo. The worktree is not read. Ladder runs are in
the rnd pin. Offer rows are optional and come only from a version-1 snapshot
file. aspire-si is read at the B pin only: the 3b judge-control receipts are
a later commit, and this assembler does not import them.

When ``local_root`` is set, the #25 E1 rerun in that tree is copied in.
The receipt files stay as measured. The table copy drops embedding artifacts.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from interlock.importers.aspire_judges import b_rows
from interlock.importers.calibration import calibration_rows
from interlock.importers.common import ImportError_, sha256_bytes
from interlock.importers.ladder import ladder_rows
from interlock.importers.nli import nli_rows, prereg_names_intel_igpu
from interlock.importers.offers import offer_rows
from interlock.importers.pins import (
    ASPIRE_COMMIT,
    ASPIRE_REPO,
    LADDER_RUNS,
    NPU_BUILDOUT_COMMIT,
    NPU_REPO,
    RND_COMMIT,
    RND_REPO,
)
from interlock.importers.probe import probe_rows
from interlock.receipt import ReceiptError, validate_row

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
# B's four files at aspire-si #78. judge-control-selection.json is in the same
# tree and is not a receipt. Do not call judge_control_rows from this pin.
_AUDITOR = "docs/runs/receipts/2026-10-08-auditor"
_B_READOUT = f"{_AUDITOR}/B-readout.json"
_B_MANIFEST = f"{_AUDITOR}/B-manifest.json"
_B_VERDICTS = f"{_AUDITOR}/verdicts-error-B-mistral-small_24b.json"
_B_SELECTION = f"{_AUDITOR}/B-selection.json"
# #25's re-run only. The earlier rerun files measured a 13-token seed.
_E1_RERUN2 = (
    "receipts/e1-intel-2026-10-09-rerun2.jsonl",
    "receipts/e1-intel-2026-10-09-rerun2-b8tail.jsonl",
    "receipts/e1-intel-2026-10-09-rerun2-nomic-diag.jsonl",
)


def import_receipts(
    *,
    rnd_repo: Path | None = None,
    npu_repo: Path | None = None,
    aspire_repo: Path | None = None,
    offers_path: Path | None = None,
    rnd_pin: str = RND_COMMIT,
    npu_pin: str = NPU_BUILDOUT_COMMIT,
    aspire_pin: str = ASPIRE_COMMIT,
    local_root: Path | None = None,
) -> tuple[list[dict], list[str]]:
    rnd = Path(rnd_repo or RND_REPO)
    npu = Path(npu_repo or NPU_REPO)
    aspire = Path(aspire_repo or ASPIRE_REPO)
    require_pin(rnd, rnd_pin)
    require_pin(npu, npu_pin)
    require_pin(aspire, aspire_pin)
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
    rows.extend(_aspire_b(aspire, aspire_pin))
    if offers_path is not None:
        content = offers_path.read_bytes()
        rows.extend(offer_rows(_json(content), source_path=_offer_source(offers_path), content=content))
    if local_root is not None:
        rows.extend(e1_rerun2_rows(Path(local_root)))
    _unique(rows)
    rows.sort(key=lambda row: row["id"])
    return rows, skipped


def e1_rerun2_rows(root: Path) -> list[dict]:
    """Table copies of the #25 rerun.

    The receipt files are not rewritten and parity is not recomputed.
    A pass or a miss keeps its timing, without artifacts. ``device_lost``
    and ``hang`` land as untimed capability rows. Anything else is refused.
    """

    rows: list[dict] = []
    for rel in _E1_RERUN2:
        path = root / rel
        if not path.is_file():
            raise ImportError_(f"missing local receipt {rel}")
        for row in _jsonl(path.read_bytes()):
            rows.append(_e1_table_copy(row))
    return rows


def _e1_table_copy(row: dict) -> dict:
    from interlock.harness.receipts import for_table

    outcome = row.get("outcome")
    try:
        if outcome in ("device_lost", "hang"):
            return for_table(row)
        if row.get("parity_valid") not in (True, False):
            raise ImportError_(
                f"{row.get('id', '?')}: an E1 row must be parity-stamped or an untimed capability row"
            )
        copied = dict(row)
        copied.pop("artifacts", None)
        validate_row(copied)
    except ReceiptError as exc:
        raise ImportError_(str(exc)) from exc
    return copied


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


def _aspire_b(repo: Path, pin: str) -> list[dict]:
    readout = show(repo, pin, _B_READOUT)
    manifest = show(repo, pin, _B_MANIFEST)
    verdicts = show(repo, pin, _B_VERDICTS)
    selection = show(repo, pin, _B_SELECTION)
    return b_rows(
        readout=readout,
        manifest=manifest,
        verdicts=verdicts,
        selection=selection,
        source_dir=_AUDITOR,
        date=_written_date(selection),
        commit_sha=pin,
    )


def _written_date(selection: bytes) -> str:
    written = _json(selection).get("written")
    if not isinstance(written, str) or len(written) < 10:
        raise ImportError_("B's selection has no written date")
    parts = written[:10].split("-")
    if len(parts) != 3:
        raise ImportError_("B's selection has no written date")
    year, month, day = parts
    dated = (
        len(year) == 4
        and len(month) == 2
        and len(day) == 2
        and year.isdigit()
        and month.isdigit()
        and day.isdigit()
    )
    if not dated:
        raise ImportError_("B's selection has no written date")
    return written[:10]


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
