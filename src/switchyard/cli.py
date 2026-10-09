"""switchyard command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from switchyard.receipt import check_tree, receipt_paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="switchyard", description="Route a job from measured receipts.")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="validate every receipt in the repository")
    check.add_argument("--root", type=Path, default=Path("."))
    check.set_defaults(func=_cmd_check)

    table = sub.add_parser("table", help="print routing rows, filtered by task or device")
    table.add_argument("--root", type=Path, default=Path("."))
    table.add_argument("--task")
    table.add_argument("--device")
    table.set_defaults(func=_cmd_table)

    importer = sub.add_parser("import", help="rebuild the routing table from pinned receipts")
    importer.add_argument("--root", type=Path, default=Path("."))
    importer.add_argument("--offers", type=Path, default=None)
    importer.set_defaults(func=_cmd_import)

    args = parser.parse_args(argv)
    return args.func(args)


def _cmd_check(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    problems = check_tree(root)
    count = _count_rows(root)
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        print(f"switchyard check: {len(problems)} problem(s) in {count} receipt(s)", file=sys.stderr)
        return 1
    print(f"switchyard check: {count} receipt(s), ok")
    return 0


def _cmd_table(args: argparse.Namespace) -> int:
    from switchyard.receipt import load_jsonl

    path = args.root.resolve() / "table" / "routing.jsonl"
    rows = load_jsonl(path) if path.is_file() else []
    if args.task:
        rows = [row for row in rows if row["task"] == args.task]
    if args.device:
        rows = [row for row in rows if row["device"] == args.device]
    for row in rows:
        model = row["model"]["repo_or_tag"]
        print(f"{row['id']}\t{row['task']}\t{row['device']}\t{model}")
    print(f"{len(rows)} row(s)")
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    from switchyard.importers.assemble import import_receipts
    from switchyard.importers.gaps import load_candidates, render_gaps

    root = args.root.resolve()
    rows, skipped = import_receipts(offers_path=args.offers)
    table = root / "table"
    table.mkdir(exist_ok=True)
    lines = [json.dumps(row, sort_keys=True) for row in rows]
    (table / "routing.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    candidates = load_candidates(table / "candidates.json")
    note = (
        "Ladder rows are imported from rnd at 08ca419f647a3ed69ee645610eaa1c7f35f40f91. "
        "Difficulty length quartiles are not imported. Tier rows are. "
        "Chain and ladder rows with no pins.quote_rule are quote rule 1 by the pre-#47 inference."
    )
    (table / "gaps.md").write_text(
        render_gaps(rows, candidates, ladder_note=note, skipped=skipped),
        encoding="utf-8",
    )
    print(f"switchyard import: {len(rows)} row(s), {len(skipped)} skipped")
    return 0


def _count_rows(root: Path) -> int:
    from switchyard.receipt import load_jsonl

    total = 0
    for path in receipt_paths(root):
        if path.suffix == ".jsonl":
            total += len(load_jsonl(path))
        else:
            total += 1
    return total
