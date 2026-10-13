# interlock: how it works

Mapped at 2026-10-13 from commit cc91aed by Atlas 1.24.0.

## What this is

interlock routes each job to a model, engine and device from a receipt-backed table. (written by a person)

10 parts, mostly Python (87 files). Work enters through 3 doors; the busiest is CI, which reaches 3 parts. It publishes to PyPI. People run interlock.

## What changed since 2026-10-13 (0db35c8)

- scripts now imports src.
- tests now imports scripts.
- receipts is a new part, drawn from `receipts/**`.
- scripts is a new part, drawn from `scripts/**`.
- table is a new part, drawn from `table/**`.
- 86 files moved, across 3 parts.

## What comes in

1. **CI.** On a pull request to main; on a push to main touching 11 paths; or by hand. Runs tests/.
2. **Release.** When a release is published; or by hand. Checks src/interlock/.
3. **interlock** (a command people run). Runs src/interlock/cli.py.

## What happens through CI

1. The workflow runs tests/ in tests.
2. That reaches scripts (1 file) and src (47 files).
3. It uploads coverage to Codecov.

## Who reads the results

CI writes nothing this map can see.

## The other doors

**Release** checks src/interlock/ and publishes to PyPI.

**interlock** (a command people run) runs src/interlock/cli.py.

## What breaks what

- **src** is imported by 1 part (scripts), and by 1 more only from tests; it sits on the path of 3 doors.
- **scripts** is imported only from tests, by 1 part (tests), and sits on the path of 1 door.

## What tends to change together

- **src/switchyard/adapters/openvino.py** and **tests/test_openvino_adapter.py** changed together in 17 of 22 commits.
- **src/switchyard/harness/receipts.py** and **src/switchyard/receipt.py** changed together in 7 of 13 commits.

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

People write .github/, assets/, docs/, experiments/, receipts/, the repository root and table/; 3 writes with paths built at run time may land here.

## Where to start

src/interlock/cli.py → src/interlock/receipt.py → src/interlock/families.py

Read those in order to follow one run of interlock end to end. This path follows interlock (a command people run) from its entry, since CI runs only tests.

## What this map cannot see

- 1 import site names a declared dependency that shares its name with a local module (openvino); it is read as the dependency, which is not in this repository.
- 3 writes and 4 reads use paths built at run time and are not named here.
- 12 writes and 39 reads go to a path their caller passes, not to this repository.
- 3 reads go to the directory the command is run in (receipts/), not to this repository.
- Statistics confidence is low: fewer than 25 source files reach 10 revisions in the window.

Regenerate with `npx --yes @dogfood-lab/atlas map`.
