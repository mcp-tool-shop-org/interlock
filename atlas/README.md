# interlock: how it works

Mapped at 2026-10-27 from commit ad14035 by Atlas 1.24.0.

## What this is

interlock routes each job to a model, engine and device from a receipt-backed table. (written by a person)

10 parts, mostly Python (116 files), CSS (2), TypeScript (2), Astro (1), JavaScript (1) and shell (1). Work enters through 5 doors; the busiest is CI, which reaches 3 parts. It publishes to PyPI. It deploys a site to GitHub Pages. People run interlock.

## What changed since 2026-10-26 (2d1f6c0)

Nothing structural changed since 2026-10-26; 17 files changed content.

## What comes in

1. **CI.** On a pull request to main; on a push to main touching 11 paths; or by hand. Runs src/interlock/cli.py, tests/fuzz_receipts.py and tests/.
2. **Release.** When a release is published; or by hand. Checks src/interlock/.
3. **Deploy site to GitHub Pages.** On a push to main touching 2 paths; or by hand. Runs site/astro.config.mjs and site/src/.
4. **OpenSSF Scorecard.** On a `branch_protection_rule` event; on a schedule (`43 15 * * 1`), Monday at 15:43 UTC; or by hand. Runs no file this map can see.
5. **interlock** (a command people run). Runs src/interlock/cli.py.

## What happens through CI

1. The workflow runs src/interlock/cli.py in src and tests/fuzz_receipts.py and tests/ in tests.
2. That reaches scripts (2 files).
3. It uploads coverage to Codecov.

## Who reads the results

CI writes nothing this map can see.

## The other doors

**Release** checks src/interlock/, publishes to PyPI, and uploads dist/*.tar.gz and dist/*.whl to the release.

**Deploy site to GitHub Pages** runs site/astro.config.mjs and site/src/, and deploys the site.

**OpenSSF Scorecard** runs no file this map can see and scans code with CodeQL.

**interlock** (a command people run) runs src/interlock/cli.py.

## What breaks what

- **src** is imported by 1 part (scripts), and by 1 more only from tests; it sits on the path of 3 doors.
- **scripts** is imported only from tests, by 1 part (tests), and sits on the path of 1 door.

## What tends to change together

- **src/switchyard/adapters/openvino.py** and **tests/test_openvino_adapter.py** changed together in 17 of 23 commits.
- **site/src/site-config.ts** and **src/interlock/__init__.py** changed together in 11 of 15 commits.
- **src/interlock/importers/remote.py** and **src/interlock/receipt.py** changed together in 4 of 7 commits, inside the src part.
- **src/interlock/harness/fill.py** and **tests/test_e5_preflight.py** changed together in 5 of 9 commits, and the tests part imports the src part.
- **src/switchyard/harness/receipts.py** and **src/switchyard/receipt.py** changed together in 7 of 13 commits.

2 files changed together with their own tests, as expected.

Confidence is low: fewer than 25 source files reach 10 revisions in the window.

Window: 180 days; a pair counts from 3 shared commits, since 16 source files reach 10 revisions; the floor rises to 10 when 25 do.

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

People write .github/, assets/, docs/, experiments/, receipts/, the repository root and table/; 5 writes with paths built at run time may land here.

## Where to start

.github/workflows/ci.yml → src/interlock/cli.py → src/interlock/receipt.py → src/interlock/devices.py

Read those in order to follow one pull request end to end.

## What this map cannot see

- 1 import site names a declared dependency that shares its name with a local module (openvino); it is read as the dependency, which is not in this repository.
- 5 writes and 6 reads use paths built at run time and are not named here.
- 17 writes and 46 reads go to a path their caller passes, not to this repository.
- 5 reads go to the directory the command is run in (experiments/, interlock.toml and receipts/), not to this repository.
- 18 files belong to no part: site/astro.config.mjs, site/package-lock.json, site/package.json and 15 more.
- Statistics confidence is low: fewer than 25 source files reach 10 revisions in the window.

Regenerate with `npx --yes @dogfood-lab/atlas map`.
