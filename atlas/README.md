# switchyard: how it works

Mapped at 2026-10-09 from commit f4f2b4b by Atlas 1.24.0.

## What this is

switchyard routes each job to a model, engine and device from a receipt-backed table. (written by a person)

6 parts, mostly Python (11 files). Work enters through 2 doors; the busiest is CI, which reaches 2 parts. People run switchyard.

## What changed since 2026-10-09 (a65b12f)

- tests now imports src.
- CI (.github/workflows/ci.yml) is a new door. It starts on a pull request; on a push touching 10 paths; or by hand. It runs tests/.
- switchyard (pyproject.toml) is a new command. It runs src/switchyard/cli.py.
- README.md is now read by pyproject.toml.
- src/switchyard/schema/routing-row.schema.json is now read by src/switchyard/receipt.py.
- 32 files added, across 6 parts.

## What comes in

1. **CI.** On a pull request; on a push touching 10 paths; or by hand. Runs tests/.
2. **switchyard** (a command people run). Runs src/switchyard/cli.py.

## What happens through CI

1. The workflow runs tests/ in tests.
2. That reaches src (4 files).

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

src/switchyard/cli.py → src/switchyard/receipt.py

Read those in order to follow one run of switchyard end to end. This path follows switchyard (a command people run) from its entry, since CI runs only tests.

## What this map cannot see

- 1 read uses a path built at run time and is not named here.
- 2 reads go to a path their caller passes, not to this repository.
- Statistics confidence is low: fewer than 30 qualifying commits in the window, and fewer than 25 source files reach 10 revisions.

Regenerate with `npx --yes @dogfood-lab/atlas map`.
