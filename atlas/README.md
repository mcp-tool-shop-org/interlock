# switchyard: how it works

Mapped at 2026-10-12 from commit ca8b5f9 by Atlas 1.24.0.

## What this is

switchyard routes each job to a model, engine and device from a receipt-backed table. (written by a person)

7 parts, mostly Python (71 files). Work enters through 2 doors; the busiest is CI, which reaches 2 parts. People run switchyard.

## What changed since 2026-10-11 (3de07c6)

- experiments/e5/spec.yaml is now read by tests/test_fill_card_live.py.
- scripts/record_ollama_fixtures.py is new and belongs to no part, so atlas check fails on it against the previous map.
- 14 files added and 30 changed content, across 4 parts.

## What comes in

1. **CI.** On a pull request to main; on a push to main touching 11 paths; or by hand. Runs tests/.
2. **switchyard** (a command people run). Runs src/switchyard/cli.py.

## What happens through CI

1. The workflow runs tests/ in tests.
2. That reaches src (42 files).
3. It uploads coverage to Codecov.

## Who reads the results

CI writes nothing this map can see.

## The other doors

**switchyard** (a command people run) runs src/switchyard/cli.py.

## What breaks what

- **src** is imported only from tests, by 1 part (tests), and sits on the path of 2 doors.

## What tends to change together

- **src/switchyard/adapters/openvino.py** and **tests/test_openvino_adapter.py** changed together in 15 of 19 commits, and the tests part imports the src part.
- **src/switchyard/harness/receipts.py** and **src/switchyard/receipt.py** changed together in 7 of 10 commits, inside the src part.
- **src/switchyard/harness/runner.py** and **tests/test_harness.py** changed together in 12 of 21 commits, and the tests part imports the src part.
- **src/switchyard/adapters/base.py** and **src/switchyard/harness/receipts.py** changed together in 5 of 10 commits, inside the src part.
- **src/switchyard/importers/gaps.py** and **src/switchyard/receipt.py** changed together in 5 of 10 commits, inside the src part.

Confidence is low: fewer than 25 source files reach 10 revisions in the window.

Window: 180 days; a pair counts from 3 shared commits, since 6 source files reach 10 revisions; the floor rises to 10 when 25 do.

## What no test touches

Every code part is imported by at least one test.

## Written but never read

No place this map can see is written, so none goes unread.

## Helpers that look duplicated

No two parts export a helper that looks alike.

## Generated, never hand-edited

Nothing in this repository writes to a tracked place this map can see.

## Hand-authored

People write .github/, assets/, docs/, experiments/ and the repository root; 3 writes with paths built at run time may land here.

## Where to start

src/switchyard/cli.py → src/switchyard/receipt.py → src/switchyard/families.py

Read those in order to follow one run of switchyard end to end. This path follows switchyard (a command people run) from its entry, since CI runs only tests.

## What this map cannot see

- 1 import site names a declared dependency that shares its name with a local module (openvino); it is read as the dependency, which is not in this repository.
- 3 writes and 3 reads use paths built at run time and are not named here.
- 5 writes and 21 reads go to a path their caller passes, not to this repository.
- 3 reads go to the directory the command is run in (receipts/), not to this repository.
- 18 files belong to no part: receipts/e1-intel-2026-10-09-rerun-b8tail.jsonl, receipts/e1-intel-2026-10-09-rerun.jsonl, receipts/e1-intel-2026-10-09-rerun2-b8tail.jsonl and 15 more.
- Statistics confidence is low: fewer than 25 source files reach 10 revisions in the window.

Regenerate with `npx --yes @dogfood-lab/atlas map`.
