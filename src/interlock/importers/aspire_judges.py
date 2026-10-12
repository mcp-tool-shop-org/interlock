"""ASPIRE's judge controls, one row per judge.

Two receipts from the aspire-si Auditor run (its plan, addendum 5, amendments 1 and 2):

- B: mistral-small:24b as a third judge on planted errors (``judge:planted-error``). There is no gold
  here. The control measured agreement with a two-judge consensus, so the row carries
  ``consensus_agreement`` (pooled, and per consensus side in ``per_type``) and ``abstain_rate``. It
  carries no false-reject or false-accept rate, because consensus is not truth.
- The stage 3b judge control: each candidate second rewrite judge on a set built by construction
  (``judge:same-meaning-rewrite``). There the labels are true by rule, so the row carries
  ``false_reject_rate`` (a harmless edit called "changed"), ``false_accept_rate`` (a harmful edit called
  "same"), ``abstain_rate`` ("unsure") and per-type rates, plus residency and cost where the run
  measured them.

The counts are recomputed from the verdict files, not copied from the readout, and the readout is
checked against them. A field the run did not record is left out or marked unrecorded, never filled.
``assemble`` reads the files with ``git show`` of the aspire-si pin and passes that sha. A caller
that has only the bytes leaves ``commit_sha`` null. Either way the readout's content sha256 is recorded.
"""

from __future__ import annotations

import json
import statistics

from interlock.families import stamp_family
from interlock.importers.common import (
    UNRECORDED,
    ImportError_,
    blank_latency,
    blank_settings,
    counted_metric,
    device_for_ollama,
    sha256_bytes,
    source_provenance,
)
from interlock.receipt import validate_row

# B's runner at aspire-si f549858 read matched_set.OLLAMA, which is this URL; its manifest predates
# recording the URL.
B_OLLAMA_URL = "http://127.0.0.1:11434"
_USABLE_PLANTED = {"wrong", "not wrong", "unsure"}


class AspireImportError(ImportError_):
    """An ASPIRE judge receipt that doesn't agree with itself; the import refuses rather than guesses."""

_USABLE_REWRITE = {"same", "changed", "unsure"}


def _admits(verdict: dict) -> bool:
    return verdict.get("outcome") == "wrong" and verdict.get("grammatical") is True


def _mean_seconds(verdicts: list[dict]) -> float | None:
    seconds = [v["seconds"] for v in verdicts if isinstance(v.get("seconds"), (int, float))]
    return statistics.fmean(seconds) if seconds else None


def _thinking(think) -> str:
    return "on" if think else "off"


def b_rows(
    *,
    readout: bytes,
    manifest: bytes,
    verdicts: bytes,
    selection: bytes,
    source_dir: str,
    date: str,
    commit_sha: str | None = None,
) -> list[dict]:
    """One judge:planted-error row for B's third judge."""

    report, man, judged, sel = (json.loads(x.decode("utf-8")) for x in (readout, manifest, verdicts, selection))
    dataset = sha256_bytes(selection)
    both_wrong, both_not = sel["control_both_wrong"], sel["control_both_not_wrong"]
    if sorted(man["control"]) != sorted(both_wrong + both_not) or man["splits"] != sel["splits"]:
        raise AspireImportError("B's manifest and its frozen selection name different items")
    missing = [cid for cid in sel["order"] if cid not in judged]
    if missing:
        raise AspireImportError(f"B's verdicts lack {len(missing)} of the selected items")

    agree_wrong = sum(_admits(judged[c]) for c in both_wrong)
    agree_not = sum(judged[c].get("outcome") == "not wrong" for c in both_not)
    for side, k in (("both wrong", agree_wrong), ("both not wrong", agree_not)):
        if report["control"][side]["agrees"] != k:
            raise AspireImportError(f"B's readout says {report['control'][side]['agrees']} for {side}; the verdicts give {k}")
    every = [judged[c] for c in sel["order"]]
    unsure = sum(v.get("outcome") == "unsure" for v in every)
    unusable = sum(v.get("outcome") not in _USABLE_PLANTED for v in every)
    passed = bool(report["reading"]["control_passed"])

    row = {
        "id": f"aspire/B/{man['judge']}/judge:planted-error",
        "task": "judge:planted-error",
        "model": {
            "repo_or_tag": man["judge"],
            "revision_or_digest": man["digest"],
            "quant": UNRECORDED,
            "licence": UNRECORDED,
        },
        "engine": {"name": "ollama", "version": UNRECORDED},
        "device": device_for_ollama(B_OLLAMA_URL),
        "settings": blank_settings(
            thinking_level=_thinking(man["think"]),
            num_predict=man["options"]["num_predict"],
            num_ctx=man["options"]["num_ctx"],
            prompt_sha256=man["prompt_sha256"],
            schema_sha256=man["schema_sha256"],
            selection_sha256=dataset,
            control_rule="consensus agreement >= 25/28 on each side, strict scoring",
        ),
        "quality": [
            counted_metric("consensus_agreement", agree_wrong + agree_not, len(both_wrong) + len(both_not), dataset),
            counted_metric("abstain_rate", unsure, len(every), dataset),
        ],
        "per_type": {
            "consensus:both-wrong": counted_metric("consensus_agreement", agree_wrong, len(both_wrong), dataset),
            "consensus:both-not-wrong": counted_metric("consensus_agreement", agree_not, len(both_not), dataset),
        },
        "cost": {"seconds": _mean_seconds(every), "tokens": None},
        "latency": blank_latency(),
        "date": date,
        "repeats": 1,
        "provisional": True,
        "provenance": source_provenance(f"{source_dir}/B-readout.json", commit_sha, readout),
        "notes": (
            "Single run, provisional. Agreement is with a two-judge consensus (gemma4:31b and muse-glimmer), "
            "not with gold, so no false-reject or false-accept rate is recorded. "
            f"Control {'passed' if passed else 'failed'} under the pre-registered 25/28-per-side line; "
            "the pooled consensus_agreement can hide a failed side, so read per_type. "
            f"Unusable replies: {unusable}. cost.seconds is the mean per call. "
            "Ollama version and quantisation were not recorded by this run."
        ),
    }
    stamp_family(row)
    validate_row(row)
    return [row]


def judge_control_rows(
    *,
    readout: bytes,
    manifest: bytes,
    verdicts: dict[str, bytes],
    selection: bytes,
    source_dir: str,
    date: str,
) -> list[dict]:
    """One judge:same-meaning-rewrite row per candidate judge of the 3b control."""

    report, man, sel = (json.loads(x.decode("utf-8")) for x in (readout, manifest, selection))
    dataset = sha256_bytes(selection)
    if man["selection_sha256"] != dataset:
        raise AspireImportError("the control's manifest names a different selection")
    items = {r["id"]: r for r in sel["items"]}
    device = device_for_ollama(man["ollama_url"])
    rows = []
    for judge in man["judges"]:
        tag = judge["model"]
        key = tag.replace(":", "_")
        if key not in verdicts:
            raise AspireImportError(f"no verdict file for {tag}")
        judged = json.loads(verdicts[key].decode("utf-8"))
        missing = [cid for cid in sel["order"] if cid not in judged]
        if missing:
            raise AspireImportError(f"{tag}'s verdicts lack {len(missing)} items")
        by_side = {side: [c for c in sel["order"] if items[c]["side"] == side] for side in ("harmless", "harmful", "identity")}
        harmless = [judged[c] for c in by_side["harmless"]]
        harmful = [judged[c] for c in by_side["harmful"]]
        identity = [judged[c] for c in by_side["identity"]]
        same_strict = sum(v.get("outcome") == "same" and v.get("claim_changed") is False for v in harmless)
        caught = sum(v.get("outcome") == "changed" for v in harmful)
        scores = report["scores"][tag]
        if (scores["harmless_agree"], scores["harmful_agree"]) != (same_strict, caught):
            raise AspireImportError(f"{tag}: the readout's agreement counts differ from its verdicts")
        decided = harmless + harmful
        per_type = {}
        for kind in sorted({items[c]["kind"] for c in by_side["harmless"]}):
            group = [judged[c] for c in by_side["harmless"] if items[c]["kind"] == kind]
            per_type[f"harmless:{kind}"] = counted_metric(
                "false_reject_rate", sum(v.get("outcome") == "changed" for v in group), len(group), dataset
            )
        for kind in sorted({items[c]["kind"] for c in by_side["harmful"]}):
            group = [judged[c] for c in by_side["harmful"] if items[c]["kind"] == kind]
            per_type[f"harmful:{kind}"] = counted_metric(
                "false_accept_rate", sum(v.get("outcome") == "same" for v in group), len(group), dataset
            )
        identity_changed = sum(v.get("outcome") == "changed" for v in identity)
        per_type["identity"] = counted_metric("false_reject_rate", identity_changed, len(identity), dataset)
        resident = (report.get("residency") or {}).get(tag) or {}
        model = {
            "repo_or_tag": tag,
            "revision_or_digest": judge["digest"],
            "quant": judge.get("quant") or UNRECORDED,
            "licence": UNRECORDED,
        }
        row = {
            "id": f"aspire/3b-judge-control/{tag}/judge:same-meaning-rewrite",
            "task": "judge:same-meaning-rewrite",
            "model": model,
            "engine": {"name": "ollama", "version": man.get("ollama_version") or UNRECORDED},
            "device": device,
            "settings": blank_settings(
                thinking_level=_thinking(judge["think"]),
                num_predict=judge["options"]["num_predict"],
                num_ctx=judge["options"]["num_ctx"],
                prompt_sha256=man["prompt_sha256"],
                schema_sha256=man["schema_sha256"],
                selection_sha256=dataset,
            ),
            "quality": [
                counted_metric("false_reject_rate", sum(v.get("outcome") == "changed" for v in harmless), len(harmless), dataset),
                counted_metric("false_accept_rate", sum(v.get("outcome") == "same" for v in harmful), len(harmful), dataset),
                counted_metric("abstain_rate", sum(v.get("outcome") == "unsure" for v in decided), len(decided), dataset),
            ],
            "per_type": per_type,
            "cost": {"seconds": _mean_seconds(harmless + harmful + identity), "tokens": None},
            "latency": blank_latency(),
            "date": date,
            "repeats": 1,
            "provisional": True,
            "provenance": source_provenance(f"{source_dir}/judge-control-readout.json", None, readout),
            "notes": (
                "Single run, provisional. Labels are true by construction (rule-made edits). "
                f"Strict agreement: harmless 'same' {same_strict}/{len(harmless)}, harmful 'changed' {caught}/{len(harmful)}; "
                f"identity copies called changed {identity_changed}/{len(identity)}. "
                f"Unusable replies: {sum(v.get('outcome') not in _USABLE_REWRITE for v in harmless + harmful + identity)}. "
                f"Chosen by the pre-registered rule: {report['result']['choice'] == tag}. "
                "Constructed edits are easier than real rewrites: this ranks judges, it doesn't validate them. "
                "cost.seconds is the mean per call."
            ),
        }
        if resident.get("size") is not None and resident.get("size_vram") is not None:
            model["size"], model["size_vram"] = resident["size"], resident["size_vram"]
            row["gpu_resident"] = resident["size"] == resident["size_vram"]
        stamp_family(row)
        validate_row(row)
        rows.append(row)
    return rows
