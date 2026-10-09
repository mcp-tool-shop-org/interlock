# switchyard: how it works

Mapped at 2026-10-09 from commit a4c84a3 by Atlas 1.24.0.

## What this is

switchyard routes each job to a model, engine and device from a receipt-backed table. (written by a person)

6 parts, mostly Python (22 files). Work enters through 2 doors; the busiest is CI, which reaches 2 parts. People run switchyard.

## What changed since 2026-10-09 (f4f2b4b)

- CI's pull request trigger now also names `main`.
- CI's push trigger now also names `codecov.yml` and `main`.
- table/candidates.json is new and belongs to no part, so atlas check fails on it against the previous map.
- table/gaps.md is new and belongs to no part, so atlas check fails on it against the previous map.
- table/routing.jsonl is new and belongs to no part, so atlas check fails on it against the previous map.
- 15 files added and 10 changed content, across 5 parts.

## What comes in

1. **CI.** On a pull request to main; on a push to main touching 11 paths; or by hand. Runs tests/.
2. **switchyard** (a command people run). Runs src/switchyard/cli.py.

## What happens through CI

1. The workflow runs tests/ in tests.
2. That reaches src (13 files).
3. It uploads coverage to Codecov.

## Who reads the results

CI writes nothing this map can see.

## The other doors

**switchyard** (a command people run) runs src/switchyard/cli.py.

## What breaks what

- **src** is imported only from tests, by 1 part (tests), and sits on the path of 2 doors.

## What tends to change together

No two source files changed together often enough to name.

Window: 180 days; a pair counts from 3 shared commits, since the window holds fewer than 30 qualifying commits.

## What no test touches

Every code part is imported by at least one test.

## Written but never read

No place this map can see is written, so none goes unread.

## Helpers that look duplicated

No two parts export a helper that looks alike.

## Generated, never hand-edited

Nothing in this repository writes to a tracked place this map can see.

## Hand-authored

People write .github/, assets/, docs/ and the repository root. Nothing in this repository writes to them.

## Where to start

src/switchyard/cli.py → src/switchyard/importers/assemble.py → src/switchyard/importers/calibration.py → src/switchyard/importers/common.py

Read those in order to follow one run of switchyard end to end. This path follows switchyard (a command people run) from its entry, since CI runs only tests.

## What this map cannot see

- 1 read uses a path built at run time and is not named here.
- 2 writes and 5 reads go to a path their caller passes, not to this repository.
- 3 files belong to no part: table/candidates.json, table/gaps.md and table/routing.jsonl.
- Statistics confidence is low: fewer than 30 qualifying commits in the window, and fewer than 25 source files reach 10 revisions.

Regenerate with `npx --yes @dogfood-lab/atlas map`.
