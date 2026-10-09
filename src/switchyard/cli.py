"""switchyard command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from switchyard.receipt import check_tree, receipt_paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="switchyard", description="Route a job from measured receipts.")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="validate every receipt in the repository")
    check.add_argument("--root", type=Path, default=Path("."))
    check.set_defaults(func=_cmd_check)

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


def _count_rows(root: Path) -> int:
    from switchyard.receipt import load_jsonl

    total = 0
    for path in receipt_paths(root):
        if path.suffix == ".jsonl":
            total += len(load_jsonl(path))
        else:
            total += 1
    return total
