# interlock: how it works

Mapped at 2026-10-13 from commit e8628ca by Atlas 1.24.0.

## What this is

interlock routes each job to a model, engine and device from a receipt-backed table. (written by a person)

10 parts, mostly Python (87 files), CSS (2), TypeScript (2), Astro (1), JavaScript (1) and shell (1). Work enters through 4 doors; the busiest is CI, which reaches 3 parts. It publishes to PyPI. It deploys a site to GitHub Pages. People run interlock.

## What changed since 2026-10-13 (cc91aed)

- Deploy site to GitHub Pages (.github/workflows/pages.yml) is a new door. It starts on a push to main touching 2 paths; or by hand. It runs site/astro.config.mjs and site/src/.
- site/src/content/docs/ is now read by site/astro.config.mjs.
- site/src/content/docs/handbook/ is now read by site/astro.config.mjs.
- site/astro.config.mjs is new and belongs to no part, so atlas check fails on it against the previous map.
- site/package-lock.json is new and belongs to no part, so atlas check fails on it against the previous map.
- site/package.json is new and belongs to no part, so atlas check fails on it against the previous map.
- And 15 more new files that belong to no part.
- 29 files added, 7 removed and 6 changed content, across 4 parts.

## What comes in

1. **CI.** On a pull request to main; on a push to main touching 11 paths; or by hand. Runs tests/.
2. **Release.** When a release is published; or by hand. Checks src/interlock/.
3. **Deploy site to GitHub Pages.** On a push to main touching 2 paths; or by hand. Runs site/astro.config.mjs and site/src/.
4. **interlock** (a command people run). Runs src/interlock/cli.py.

## What happens through CI

1. The workflow runs tests/ in tests.
2. That reaches scripts (1 file) and src (47 files).
3. It uploads coverage to Codecov.

## Who reads the results

CI writes nothing this map can see.

## The other doors

**Release** checks src/interlock/ and publishes to PyPI.

**Deploy site to GitHub Pages** runs site/astro.config.mjs and site/src/, and deploys the site.

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

verify.sh runs in no workflow.

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
- 18 files belong to no part: site/astro.config.mjs, site/package-lock.json, site/package.json and 15 more.
- Statistics confidence is low: fewer than 25 source files reach 10 revisions in the window.

Regenerate with `npx --yes @dogfood-lab/atlas map`.
