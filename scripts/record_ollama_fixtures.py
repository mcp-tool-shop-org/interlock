"""Record Ollama request and response fixtures.

Run this on the rig, from the switchyard repo, against the local server.
It does not pick a cloud model. It stops when the card is not idle.

    uv run python scripts/record_ollama_fixtures.py --model qwen3:8b --host http://127.0.0.1:11434 --timeout 15 --out tests/fixtures/ollama
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

from switchyard.adapters.record_fixtures import (
    IDLE_CEILING_MIB,
    RECORD_TIMEOUT_S,
    RecordError,
    http_exchange,
    read_card_mib,
    record_fixtures,
)


def main(argv: list[str] | None = None, *, transport=None, card_mib=None, today: str | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record Ollama exchanges for the fakes.")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=RECORD_TIMEOUT_S)
    parser.add_argument("--idle-mib", type=int, default=IDLE_CEILING_MIB)
    parser.add_argument("--out", default="tests/fixtures/ollama")
    args = parser.parse_args(argv)
    if today is None:
        today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
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
        )
    except RecordError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    for path in written:
        print(path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
