"""interlock command line."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

from interlock.receipt import check_tree, receipt_paths
from interlock.tasks import TASKS

_ROUTE_EPILOG = """
constraints:
  --min-<metric> X   floor on a conservative bound. --min-tok-s is the minimum
                     repeat-slot decode tok/s. Any other name is a quality
                     metric, judged on the low end of its interval.
  --max-latency-ms   maximum repeat-slot latency (or claim-wall max), in ms.
  --max-ttft-ms      maximum repeat-slot time to first token. Refuses unless
                     every compared row records the same settings.prefix_caching
                     (on, off, or default). Unset means unknown.
  --concurrency N    generate only, default 1. Ranks and floors rows measured at
                     N concurrent requests (decode_tok_s is the pass aggregate).
                     A row with no concurrency counts as 1. No match: a refusal
                     naming the concurrencies that exist. The margin is between
                     distinct candidates (device, model, quant), never between
                     two shapes of one model. Fill-card rows are load probes and
                     never rank in generate (excluded: fill_card).
  --ctx N            exact measured length when the row id has lN. A row with
                     num_ctx and no lN tag qualifies when num_ctx is at least N.
  --thinking         family of settings.thinking_level. off matches only off.
                     on matches any recorded thinking level. low, medium and
                     high match exactly when named. The decision keeps the
                     row's exact level.
  --busy DEVICE      repeatable. An input. interlock does not read a ledger,
                     the watchdog, or /api/ps.
  --budget-left-usd  An input, like --busy. A remote row whose worst case
                     exceeds it is dropped and shown in the runners-up. With
                     no flag, a remote suggestion is marked budget unchecked.
                     interlock does not read offrig.
  --allow-remote     allow remote colab:<card> rows into the candidate pool.
                     Default off.
  --device FILTER    exact device (colab:L4, 5090) or provider wildcard
                     (colab:*). A colab: filter implies remote eligibility.

A row with no repeat spread is not ranked on that metric. The margin is the
gap between conservative bounds, never a median. Capability rows
(native_binary_rejected, compile_rejected, compile_timeout, hang,
device_lost, oom, load_refused, timeout, ctx_exceeded, sampler_error, generation_aborted, unmeasured) and a row with
parity_valid false (parity_invalid) are listed under excluded and are not
candidates. No row meets the constraints: exit 1 and a structured error. It
does not fall back.
"""


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    min_metrics: dict[str, float] = {}
    if raw and raw[0] == "route":
        try:
            raw, min_metrics = _take_min_metrics(raw)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    parser = argparse.ArgumentParser(
        prog="interlock",
        description="interlock: local AI device router. Route a job from measured receipts.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser(
        "check",
        help="validate every receipt and table row, including the privacy check",
    )
    check.add_argument("--root", type=Path, default=Path("."))
    check.add_argument(
        "--host-port",
        action="store_true",
        help="also refuse a single-label host and port, such as myrig:11434. A model tag is not a host",
    )
    check.set_defaults(func=_cmd_check)

    table = sub.add_parser("table", help="print routing rows, filtered by task or device")
    table.add_argument("--root", type=Path, default=Path("."))
    table.add_argument("--task")
    table.add_argument("--device")
    table.set_defaults(func=_cmd_table)

    importer = sub.add_parser(
        "import",
        help="rebuild the routing table from pinned receipts and the local E1 rerun",
    )
    importer.add_argument("--root", type=Path, default=Path("."))
    importer.add_argument("--offers", type=Path, default=None)
    importer.add_argument(
        "--remote-window",
        action="append",
        type=Path,
        default=None,
        help="remote window JSON file to import (repeatable)",
    )
    importer.add_argument(
        "--source",
        action="append",
        default=None,
        metavar="NAME=PATH",
        help="where a full rebuild finds a source repo: rnd, npu or aspire (repeatable). "
        "Also [import.sources] in <root>/interlock.toml; the flag wins. A merge needs none",
    )
    importer.add_argument(
        "--allow-shrink",
        action="store_true",
        help="let a full rebuild drop the rows it owns but no longer produces; every dropped id is printed",
    )
    importer.set_defaults(func=_cmd_import)

    run = sub.add_parser("run", help="run a committed experiment spec (the plan only with --dry-run)")
    run.add_argument("spec", type=Path)
    run.add_argument("--root", type=Path, default=Path("."))
    run.add_argument("--grant")
    run.add_argument("--npu-session")
    run.add_argument("--cpu-window")
    run.add_argument("--run-id")
    run.add_argument("--notes", default="")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--pilot", action="store_true")
    run.add_argument(
        "--no-watchdog",
        action="store_true",
        help="run a 5090 step with no VRAM watchdog heartbeat; without it an unset WATCHDOG_HEARTBEAT refuses. "
        "The choice is written into the run's receipt.",
    )
    run.add_argument(
        "--trace",
        choices=("off", "events", "relay"),
        default="off",
        help=(
            "log every model call in order under logs/: events (stdlib) or relay "
            "(NeMo Relay ATOF, needs interlock[relay])"
        ),
    )
    run.add_argument(
        "--pairs",
        default=None,
        help="comma-separated E4b pair ids (a subset of the spec). The receipt records them",
    )
    run.add_argument(
        "--models",
        default=None,
        help="comma-separated fill-card model names (a subset of the spec). The receipt records them",
    )
    run.add_argument(
        "--quants",
        default=None,
        help=(
            "comma-separated quants of a linux-cuda spec to measure (with --models, or on its own); "
            "an unknown quant is refused with the spec's list"
        ),
    )
    run.add_argument(
        "--quarantine",
        type=Path,
        default=None,
        help="where a stopped run is set aside (default: aborted/ under --root)",
    )
    run.set_defaults(func=_cmd_run)

    runs = sub.add_parser("runs", help="list measurement runs, or set a stray aside")
    runs_sub = runs.add_subparsers(dest="runs_command", required=True)
    runs_list = runs_sub.add_parser("list", help="show each run id and its state")
    runs_list.add_argument("--root", type=Path, default=Path("."))
    runs_list.add_argument("--quarantine", type=Path, default=None)
    runs_list.set_defaults(func=_cmd_runs_list)
    runs_quarantine = runs_sub.add_parser("quarantine", help="move a stray series aside; does not start a device")
    runs_quarantine.add_argument("run_id")
    runs_quarantine.add_argument("--root", type=Path, default=Path("."))
    runs_quarantine.add_argument("--quarantine", type=Path, default=None)
    runs_quarantine.set_defaults(func=_cmd_runs_quarantine)

    health_cmd = sub.add_parser(
        "ollama-health",
        help="print health for the local Ollama ports, 11434 and 11490",
    )
    health_cmd.set_defaults(func=_cmd_ollama_health)

    c2_sources = sub.add_parser(
        "c2-sources",
        help="print the resolved source, pull or create command, and expected digest for each entry in a C2 spec",
    )
    c2_sources.add_argument("spec", type=Path)
    c2_sources.add_argument("--json", action="store_true", help="print resolved entries as JSON")
    c2_sources.add_argument(
        "--verify-live",
        action="store_true",
        help="query registry.ollama.ai and huggingface.co to verify every source, digest, and quant label",
    )
    c2_sources.set_defaults(func=_cmd_c2_sources)

    route = sub.add_parser(
        "route",
        help="choose a model, engine and device from the routing table",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_ROUTE_EPILOG,
    )
    route.add_argument("--task", required=True, choices=sorted(TASKS))
    route.add_argument(
        "--model",
        help="substring of the model repo or tag; hyphens and colons are ignored. A short substring can match an "
            "unintended repo. The matched model is printed",
    )
    route.add_argument("--max-latency-ms", type=float)
    route.add_argument("--max-ttft-ms", type=float)
    route.add_argument("--ctx", type=int)
    route.add_argument(
        "--thinking",
        help="thinking family: off matches only off; on matches any recorded level; low, medium and high match exactly "
            "when named",
    )
    route.add_argument("--busy", action="append", default=None, metavar="DEVICE")
    route.add_argument("--quote-rule", type=int)
    route.add_argument("--num-predict", type=int)
    route.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="generate only: rank and floor rows measured at this concurrency (default 1). A row is the pass "
            "aggregate across that many requests; a row with no concurrency counts as 1",
    )
    route.add_argument("--budget-left-usd", type=float)
    route.add_argument("--device", help="exact device or provider wildcard such as colab:L4 or colab:*")
    route.add_argument(
        "--allow-remote",
        action="store_true",
        help="allow remote colab:<card> rows to compete (default off)",
    )
    route.add_argument("--as-of", help="YYYY-MM-DD. Defaults to today. The policy does not read the clock.")
    route.add_argument("--json", action="store_true", help="print a route-decision.v1 document, or a structured error")
    route.add_argument("--root", type=Path, default=Path("."))
    route.set_defaults(func=_cmd_route)

    args = parser.parse_args(raw)
    if getattr(args, "command", None) == "route":
        args.min_metrics = min_metrics
    return args.func(args)


def _take_min_metrics(argv: list[str]) -> tuple[list[str], dict[str, float]]:
    kept: list[str] = []
    metrics: dict[str, float] = {}
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg.startswith("--min-") and len(arg) > len("--min-"):
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise ValueError(f"{arg} needs a number")
            name = arg[len("--min-") :]
            try:
                metrics[name] = float(argv[index + 1])
            except ValueError as exc:
                raise ValueError(f"{arg} needs a number") from exc
            index += 2
            continue
        kept.append(arg)
        index += 1
    return kept, metrics


def _cmd_check(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    problems = check_tree(root, host_port=args.host_port)
    count = _count_rows(root)
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        print(f"interlock check: {len(problems)} problem(s) in {count} receipt(s)", file=sys.stderr)
        return 1
    print(f"interlock check: {count} receipt(s), ok")
    return 0


def _cmd_table(args: argparse.Namespace) -> int:
    from interlock.devices import matches_device_filter
    from interlock.receipt import load_jsonl

    path = args.root.resolve() / "table" / "routing.jsonl"
    rows = load_jsonl(path) if path.is_file() else []
    if args.task:
        rows = [row for row in rows if row["task"] == args.task]
    if args.device:
        rows = [row for row in rows if matches_device_filter(row.get("device"), args.device)]
    from interlock.router.policy import ladder_version, throughput_measured, ttft_measured

    # Generate-ladder rows are grouped by device, model and ladder version, and show the version: speeds from
    # different versions are different measurements.
    plain = [row for row in rows if ladder_version(row) is None]
    ladder = sorted(
        (row for row in rows if ladder_version(row) is not None),
        key=lambda row: (str(row["device"]), str(row["model"]["repo_or_tag"]), ladder_version(row)),
    )
    for row in [*plain, *ladder]:
        model = row["model"]["repo_or_tag"]
        version = ladder_version(row)
        marks = []
        if version is not None:
            marks.append(f"ladder v{version}")
        if not ttft_measured(row):
            marks.append("ttft: not measured")
        if not throughput_measured(row):
            marks.append("throughput: not measured")
        suffix = "\t" + "\t".join(marks) if marks else ""
        print(f"{row['id']}\t{row['task']}\t{row['device']}\t{model}{suffix}")
    print(f"{len(rows)} row(s)")
    return 0


def _rebuild_owns(row: dict) -> bool:
    """A full rebuild owns the rows its own sources produce: the pinned imports and the E1 rerun.

    Every other row (E2-E4, remote windows, any other branch-2 row) is the table's, not the rebuild's, and a
    rebuild leaves it as it is.
    """

    if str(row.get("device", "")).startswith("colab:"):
        return False
    provenance = row.get("provenance") or {}
    return "source_receipt_path" in provenance or str(row.get("id", "")).startswith("E1/")


def _refuse_duplicate_ids(existing: list[dict], path: Path) -> None:
    from interlock.importers.common import ImportError_

    seen: set[str] = set()
    repeated: list[str] = []
    for row in existing:
        if row["id"] in seen and row["id"] not in repeated:
            repeated.append(row["id"])
        seen.add(row["id"])
    if repeated:
        raise ImportError_(
            f"REFUSED: {path.name} holds duplicate row id(s): {', '.join(repeated[:5])}"
            + (f" and {len(repeated) - 5} more" if len(repeated) > 5 else "")
            + ". Fix the table before importing"
        )


def _merge_rebuild(path: Path, rebuilt: list[dict], *, allow_shrink: bool) -> tuple[list[dict], list[str]]:
    """The table a full rebuild writes: the rebuilt rows plus every existing row the rebuild does not own.

    Returns ``(rows, dropped_ids)``. A row the rebuild owns and no longer produces is dropped only with
    ``allow_shrink``; without it the rebuild is refused and the table stays as it was.
    """

    from interlock.importers.assemble import _unique
    from interlock.importers.common import ImportError_
    from interlock.receipt import load_jsonl

    existing = load_jsonl(path)
    _refuse_duplicate_ids(existing, path)
    produced = {row["id"] for row in rebuilt}
    dropped = [row["id"] for row in existing if _rebuild_owns(row) and row["id"] not in produced]
    if dropped and not allow_shrink:
        raise ImportError_(
            f"REFUSED: the rebuild no longer produces {len(dropped)} of the {sum(map(_rebuild_owns, existing))} "
            f"row(s) it owns in {path.name}; first missing: {', '.join(dropped[:5])}. "
            "Pass --allow-shrink to drop them (every dropped id is printed), "
            "or import remote windows alone with --remote-window to merge them into the table"
        )
    kept = [row for row in existing if not _rebuild_owns(row)]
    merged = [*rebuilt, *kept]
    _unique(merged)
    merged.sort(key=lambda row: row["id"])
    return merged, dropped


def _import_remote_windows(root: Path, windows: list[Path]) -> int:
    """Merge remote window receipts into the existing table. Nothing is rebuilt, and nothing is dropped."""

    from interlock.importers.assemble import _unique
    from interlock.importers.remote import import_remote_window
    from interlock.receipt import load_jsonl

    table = root / "table"
    path = table / "routing.jsonl"
    existing = load_jsonl(path)
    added: list[dict] = []
    for window in windows:
        added.extend(import_remote_window(Path(window).read_bytes(), source_path=Path(window).as_posix()))
    rows = [*existing, *added]
    _refuse_duplicate_ids(existing, path)
    _unique(rows)
    rows.sort(key=lambda row: row["id"])
    table.mkdir(exist_ok=True)
    scratch = path.with_name(path.name + ".tmp")
    scratch.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    os.replace(scratch, path)
    print(f"interlock import: {len(added)} remote row(s) added, {len(rows)} in the table")
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    from interlock.harness.receipts import write_atomic
    from interlock.importers.assemble import import_receipts
    from interlock.importers.common import ImportError_
    from interlock.importers.gaps import load_candidates, render_gaps
    from interlock.importers.pins import ASPIRE_COMMIT
    from interlock.importers.sources import SOURCE_PINS, require_all_sources, resolve_sources
    from interlock.receipt import ReceiptError

    root = args.root.resolve()
    table = root / "table"
    try:
        if args.remote_window and args.offers is None:
            return _import_remote_windows(root, args.remote_window)
        sources = resolve_sources(args.source, root)
        require_all_sources(sources)
        rebuilt, skipped = import_receipts(
            rnd_repo=Path(sources["rnd"]),
            npu_repo=Path(sources["npu"]),
            aspire_repo=Path(sources["aspire"]),
            offers_path=args.offers,
            local_root=root,
            remote_windows=args.remote_window,
        )
        rows, dropped = _merge_rebuild(table / "routing.jsonl", rebuilt, allow_shrink=args.allow_shrink)
        # Everything that can fail is done before anything is written: the candidates are read and the gaps page is
        # rendered first, then each file is replaced atomically, so a failed import leaves the table as it was.
        candidates = load_candidates(table / "candidates.json")
        note = (
            "Ladder rows are imported from rnd at 08ca419f647a3ed69ee645610eaa1c7f35f40f91. "
            "Difficulty length quartiles are not imported. Tier rows are. "
            "Chain and ladder rows with no pins.quote_rule are quote rule 1 by the pre-#47 inference. "
            f"B's planted-error judge row is imported from aspire-si at {ASPIRE_COMMIT}. "
            "That commit has B's receipts only. The 3b judge-control receipts are a later pin."
        )
        # Each source's name and pinned commit, with the path as the user gave it (never resolved), go in
        # gaps.md and never in a row's provenance.
        used = "\n".join(f"- {name} @ {pin}: {sources[name]}" for name, pin in SOURCE_PINS.items())
        gaps_text = (
            render_gaps(rebuilt, candidates, ladder_note=note, skipped=skipped) + f"\n## Import sources\n\n{used}\n"
        )
        lines = [json.dumps(row, sort_keys=True) for row in rows]
        table.mkdir(exist_ok=True)
        for row_id in dropped:
            print(f"dropped {row_id}")
        if dropped:
            write_atomic(table / "import-dropped.txt", "\n".join(dropped) + "\n")
        write_atomic(table / "routing.jsonl", "\n".join(lines) + ("\n" if lines else ""))
        write_atomic(table / "gaps.md", gaps_text)
    except (ImportError_, ReceiptError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"interlock import: {len(rows)} row(s), {len(skipped)} skipped, {len(dropped)} dropped")
    return 0


def run_report(rows: list[dict], *, finished: bool) -> str:
    """What a run prints when it ends.

    Each hang, device_lost, native_binary_rejected, compile_rejected or compile_timeout arm is one line, before
    ``done``. A run that returns cleanly still shows the loss.
    """

    lines = []
    for row in rows:
        outcome = row.get("outcome")
        if outcome not in ("device_lost", "hang", "native_binary_rejected", "compile_rejected", "compile_timeout"):
            continue
        lines.append(f"{_arm_id(row)} {row['device']} {outcome}")
    if finished:
        lines.append("done")
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def announce_run(rows: list[dict], *, finished: bool) -> None:
    """Print :func:`run_report`. The run command's end is this text."""

    text = run_report(rows, finished=finished)
    if text:
        print(text, end="")


def _arm_id(row: dict) -> str:
    parts = str(row.get("id", "")).split("/")
    if len(parts) >= 3:
        return parts[-2]
    return parts[-1]


def _cmd_runs_list(args: argparse.Namespace) -> int:
    from interlock.harness.lifecycle import format_runs, list_runs

    print(format_runs(list_runs(args.root.resolve(), args.quarantine)), end="")
    return 0


def _cmd_runs_quarantine(args: argparse.Namespace) -> int:
    from interlock.harness.lifecycle import LifecycleError, quarantine_stray

    root = args.root.resolve()
    sha = None
    try:
        from interlock.harness.e4b import capture_run, read_git

        found = read_git(root, capture_run)
        if isinstance(found, dict):
            sha = found.get("git_sha")
    except Exception:
        sha = None
    try:
        quarantine_stray(root, args.run_id, sha=sha, setting=args.quarantine)
    except LifecycleError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from interlock.adapters.base import AdapterError
    from interlock.guards import GuardError
    from interlock.harness.e4b import E4bError, parse_pairs
    from interlock.harness.lifecycle import LifecycleError
    from interlock.harness.linux_cuda import LinuxCudaError
    from interlock.harness.ollama_serve import ServeError
    from interlock.harness.runner import RunnerError, dry_run, heartbeat_path, parse_models, parse_quants, run_spec
    from interlock.harness.spec import SpecError, load_spec
    from interlock.paths import PathsConfigError

    try:
        pairs = parse_pairs(getattr(args, "pairs", None))
        models = parse_models(getattr(args, "models", None))
        quants = parse_quants(getattr(args, "quants", None))
        if args.dry_run:
            plan = dry_run(args.spec, args.root.resolve(), pilot=args.pilot, pairs=pairs, models=models, quants=quants)
            print(plan, end="")
            return 0
        spec = load_spec(args.spec)
        if spec.get("id") == "E4b":
            from interlock.harness.lifecycle import resolve_run_id

            chosen = resolve_run_id(
                args.root.resolve(),
                base="e4b-pilot" if args.pilot else "e4b-block",
                explicit=args.run_id,
                setting=args.quarantine,
            )
            if not args.run_id:
                print(f"run id: {chosen}", flush=True)
            args.run_id = chosen
        from interlock.guards import NO_WATCHDOG
        from interlock.ollama_supervisor import OllamaSupervisor, real_seams

        heartbeat = heartbeat_path(NO_WATCHDOG if getattr(args, "no_watchdog", False) else None)
        run_spec(
            args.spec,
            root=args.root.resolve(),
            grant=args.grant,
            npu_session=args.npu_session,
            cpu_window=args.cpu_window,
            run_id=args.run_id,
            notes=args.notes,
            supervisor=OllamaSupervisor(
                real_seams(),
                grant=args.grant,
                heartbeat=heartbeat,
            ),
            heartbeat=heartbeat,
            pilot=args.pilot,
            trace=args.trace,
            quarantine=getattr(args, "quarantine", None),
            pairs=pairs,
            models=models,
            quants=quants,
        )
    except (
        AdapterError,
        GuardError,
        RunnerError,
        SpecError,
        E4bError,
        LifecycleError,
        LinuxCudaError,
        PathsConfigError,
        ServeError,
        OSError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


def _cmd_ollama_health(_args: argparse.Namespace) -> int:
    from interlock.ollama_supervisor import health_report

    text, ok = health_report()
    print(text, end="")
    return 0 if ok else 1


def _cmd_c2_sources(args: argparse.Namespace) -> int:
    from interlock.harness.linux_cuda import (
        LinuxCudaError,
        format_c2_sources,
        format_c2_sources_live,
        resolve_c2_sources,
        verify_c2_sources_live,
    )
    from interlock.harness.spec import SpecError, load_spec

    try:
        spec = load_spec(args.spec)
        if getattr(args, "verify_live", False):
            if args.json:
                print(json.dumps(verify_c2_sources_live(spec), indent=2))
            else:
                print(format_c2_sources_live(spec), end="")
        elif args.json:
            print(json.dumps(resolve_c2_sources(spec), indent=2))
        else:
            print(format_c2_sources(spec), end="")
    except (SpecError, LinuxCudaError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


def _cmd_route(args: argparse.Namespace) -> int:
    from interlock.router.policy import Request, decide

    path = args.root.resolve() / "table" / "routing.jsonl"
    if not path.is_file():
        return _route_fail(
            args,
            {
                "code": "no_table",
                "message": f"no routing table at {path}",
                "hint": "interlock import builds table/routing.jsonl",
                "constraint": "table",
                "closest": None,
                "experiment": "unregistered",
                "excluded": [],
            },
        )
    try:
        as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()
    except ValueError:
        print("--as-of must be YYYY-MM-DD", file=sys.stderr)
        return 1
    rows = _load_jsonl(path)
    request = Request(
        task=args.task,
        as_of=as_of,
        model=args.model,
        min_metrics=dict(args.min_metrics),
        max_latency_ms=args.max_latency_ms,
        max_ttft_ms=args.max_ttft_ms,
        ctx=args.ctx,
        thinking=args.thinking,
        busy=frozenset(args.busy or []),
        quote_rule=args.quote_rule,
        num_predict=args.num_predict,
        budget_left_usd=args.budget_left_usd,
        allow_remote=bool(args.allow_remote),
        device=args.device,
        concurrency=args.concurrency,
    )
    outcome = decide(rows, request)
    document = outcome.document
    if document.get("code"):
        return _route_fail(args, document)
    if args.json:
        print(json.dumps(document))
    else:
        _print_decision(document)
    return 0


def _route_fail(args: argparse.Namespace, document: dict) -> int:
    if args.json:
        print(json.dumps(document))
    else:
        print(document["message"], file=sys.stderr)
        print(document["hint"], file=sys.stderr)
        print(f"constraint: {document['constraint']}", file=sys.stderr)
        print(f"experiment: {document['experiment']}", file=sys.stderr)
        if document.get("closest"):
            print(f"closest: {json.dumps(document['closest'])}", file=sys.stderr)
    return 1


def _print_decision(document: dict) -> None:
    engine = document["engine"]
    margin = document["margin"]
    print(f"{document['device']}\t{document['model']}\t{engine['name']} {engine['version']}")
    delta = margin["delta"]
    gap = "no runner-up" if delta is None else f"delta {delta:g} over {margin['next_bound']:g}"
    print(f"margin {margin['metric']} bound {margin['bound']:g} ({gap})")
    print("receipts: " + ", ".join(document["receipts"]))
    if document["flags"]:
        print("flags: " + ", ".join(document["flags"]))
    for note in document["notes"]:
        print(f"note: {note}")
    if document.get("offrig_plan"):
        print("offrig_plan: " + json.dumps(document["offrig_plan"]))
        print(f"budget: {document['budget']}")
    for item in document["runners_up"]:
        print(f"runner-up: {item['id']}\t{item['constraint']}\t{item['reason']}")
    for item in document["excluded"]:
        print(f"excluded: {item['id']}\t{item['code']}")


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _count_rows(root: Path) -> int:
    from interlock.receipt import load_jsonl

    total = 0
    for path in receipt_paths(root):
        if path.suffix == ".jsonl":
            total += len(load_jsonl(path))
        else:
            total += 1
    return total
