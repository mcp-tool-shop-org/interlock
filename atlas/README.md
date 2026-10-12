# interlock: how it works

Mapped at 2026-10-12 from commit 426f106 by Atlas 1.24.0.

## What this is

interlock routes each job to a model, engine and device from a receipt-backed table. (written by a person)

7 parts, mostly Python (84 files). Work enters through 3 doors; the busiest is CI, which reaches 2 parts. It publishes to PyPI. People run interlock.

## What changed since 2026-10-12 (64ff924)

Nothing structural changed since 2026-10-12; 6 files changed content.

## What comes in

1. **CI.** On a pull request to main; on a push to main touching 11 paths; or by hand. Runs tests/.
2. **Release.** When a release is published; or by hand. Checks src/interlock/.
3. **interlock** (a command people run). Runs src/interlock/cli.py.

## What happens through CI

1. The workflow runs tests/ in tests.
2. That reaches src (47 files).
3. It uploads coverage to Codecov.

## Who reads the results

CI writes nothing this map can see.

## The other doors

**Release** checks src/interlock/ and publishes to PyPI.

**interlock** (a command people run) runs src/interlock/cli.py.

## What breaks what

- **src** is imported only from tests, by 1 part (tests), and sits on the path of 3 doors.

## What tends to change together

- **src/switchyard/adapters/openvino.py** and **tests/test_openvino_adapter.py** changed together in 17 of 22 commits.
- **src/switchyard/harness/fill.py** and **tests/test_fill_card_live.py** changed together in 5 of 7 commits.
- **src/switchyard/harness/receipts.py** and **src/switchyard/receipt.py** changed together in 7 of 13 commits.
- **src/switchyard/harness/runner.py** and **tests/test_harness.py** changed together in 14 of 27 commits.
- **src/switchyard/adapters/ollama.py** and **tests/test_fill_card_live.py** changed together in 4 of 8 commits.

1 file changed together with its own test, as expected.

Confidence is low: fewer than 25 source files reach 10 revisions in the window.

Window: 180 days; a pair counts from 3 shared commits, since 8 source files reach 10 revisions; the floor rises to 10 when 25 do.

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

src/interlock/cli.py → src/interlock/receipt.py → src/interlock/families.py

Read those in order to follow one run of interlock end to end. This path follows interlock (a command people run) from its entry, since CI runs only tests.

## What this map cannot see

- 1 import site names a declared dependency that shares its name with a local module (openvino); it is read as the dependency, which is not in this repository.
- 3 writes and 4 reads use paths built at run time and are not named here.
- 12 writes and 37 reads go to a path their caller passes, not to this repository.
- 3 reads go to the directory the command is run in (receipts/), not to this repository.
- 83 files belong to no part: receipts/e1-intel-2026-10-09-rerun-b8tail.jsonl, receipts/e1-intel-2026-10-09-rerun.jsonl, receipts/e1-intel-2026-10-09-rerun2-b8tail.jsonl and 80 more.
- Statistics confidence is low: fewer than 25 source files reach 10 revisions in the window.

Regenerate with `npx --yes @dogfood-lab/atlas map`.
