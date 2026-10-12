"""Record Ollama request and response fixtures.

Run this on the rig, from the switchyard repo, against the local server.
It requires a 5090 grant, a fresh watchdog heartbeat, and an idle card.
It does not pick a cloud model.

    uv run python scripts/record_ollama_fixtures.py --model qwen3:8b --host http://127.0.0.1:11434 --timeout 15 --out tests/fixtures/ollama --grant GRANT

One case, at a window large enough that Ollama does not raise num_ctx to a floor:

    uv run python scripts/record_ollama_fixtures.py --model qwen3:8b --host http://127.0.0.1:11434 --timeout 15 --out tests/fixtures/ollama --grant GRANT --only truncate_over_num_ctx --num-ctx 2048 --prompt-repeats 2500
"""

from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path

from switchyard.adapters.ollama_cases import case_by_name
from switchyard.adapters.record_fixtures import (
    IDLE_CEILING_MIB,
    RECORD_TIMEOUT_S,
    RecordError,
    http_exchange,
    read_card_mib,
    record_fixtures,
)
from switchyard.harness.runner import heartbeat_path


def main(
    argv: list[str] | None = None,
    *,
    transport=None,
    card_mib=None,
    today: str | None = None,
    heartbeat: Path | None = None,
    now: float | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Record Ollama exchanges for the fakes.")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=RECORD_TIMEOUT_S)
    parser.add_argument("--idle-mib", type=int, default=IDLE_CEILING_MIB)
    parser.add_argument("--out", default="tests/fixtures/ollama")
    parser.add_argument("--grant", required=True)
    parser.add_argument("--only", action="append", default=None)
    parser.add_argument("--num-ctx", type=int, default=None)
    parser.add_argument("--prompt-repeats", type=int, default=None)
    args = parser.parse_args(argv)
    selected = None
    if args.only:
        try:
            selected = tuple(case_by_name(name) for name in args.only)
        except KeyError as exc:
            print(f"unknown case {exc.args[0]}", file=sys.stderr)
            return 2
    if today is None:
        today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    if heartbeat is None:
        heartbeat = heartbeat_path()
    if now is None:
        now = time.time()
    try:
        written = record_fixtures(
            model=args.model,
            host=args.host,
            out_dir=Path(args.out),
            timeout=args.timeout,
            ceiling=args.idle_mib,
            today=today,
            transport=http_exchange if transport is None else transport,
            card_mib=read_card_mib if card_mib is None else card_mib,
            grant=args.grant,
            heartbeat=heartbeat,
            now=now,
            cases=selected,
            num_ctx=args.num_ctx,
            prompt_repeats=args.prompt_repeats,
        )
    except RecordError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    for path in written:
        print(path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
