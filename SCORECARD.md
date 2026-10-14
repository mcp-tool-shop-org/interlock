# Scorecard

**Repo:** mcp-tool-shop-org/interlock
**Date:** 2026-10-12 assessment, brought up to date 2026-10-14. The scores below are that assessment. They are not re-scored.
**Type tags:** [all] [pypi] [cli]

## Pre-Remediation Assessment

| Category | Score | Notes |
|----------|-------|-------|
| A. Security | 7/10 | SECURITY.md and a threat model already existed. The advisory link still named the previous repository. Telemetry was not stated. |
| B. Error Handling | 4/10 | `route --json` is a structured refusal. Other commands print a line. Exits are 0, 1, 75, 76, 77. |
| C. Operator Docs | 7/10 | README, CHANGELOG, and LICENSE were current. No handbook. The README wordmark still said the previous name. |
| D. Shipping Hygiene | 5/10 | `uv.lock` and `requires-python` were set. No verify script, no dependency scanner in CI, no release tag. |
| E. Identity (soft) | 2/10 | A local logo only. No landing page, no homepage, no translations. |
| **Overall** | **25/50** | |

## Key gaps before the 2026-10-12 treatment

1. No landing page or handbook, and the README logo was a local file whose wordmark still said the previous name.
2. CI had no dependency scanner.
3. Error reporting is not the org's structured shape, and there is no logging-level switch. This treatment does not change that contract.
4. Translations are not in the tree yet. The Publisher runs them.
5. The site lockfile carries 10 moderate advisories in Starlight's expressive-code chain. None are high or critical.

## Remediation Priority

| Priority | Item | Estimated effort |
|----------|------|-----------------|
| 1 | Landing page, handbook, brand logo, README header | this treatment |
| 2 | `verify.sh` and `pip-audit` in CI | this treatment |
| 3 | Translations | Publisher, after this PR |
| 4 | Structured errors and logging levels | a later product brief, not 0.1.0 |

## Post-Remediation

| Category | Before | After |
|----------|--------|-------|
| A. Security | 7/10 | 8/10 |
| B. Error Handling | 4/10 | 4/10 |
| C. Operator Docs | 7/10 | 8/10 |
| D. Shipping Hygiene | 5/10 | 8/10 |
| E. Identity (soft) | 2/10 | 7/10 |
| **Overall** | 25/50 | 35/50 |

After: the advisory link names interlock, the threat model says there is no telemetry, and the identity scan of the git-tracked tree was CLEAN. B is unchanged on purpose. The handbook is seven pages. `verify.sh` and pip-audit are in place. The Translations box stayed open for the Publisher.

## As of 2026-10-14

No new scores. This is what has landed since the table above.

- OpenSSF Best Practices is passing for project 15467. The README shows that badge next to the Scorecard badge.
- Dependabot opens a grouped pull request monthly for the uv lock, the docs-site npm lock, and GitHub Actions, at most three open per ecosystem.
- CI runs an Atheris fuzz target against the receipt-line reader and the spec loader. Atheris has no Windows wheel, so that step runs on the Ubuntu job.
- A published GitHub release attaches the wheel, the sdist, and a SLSA provenance file (`*.intoto.jsonl`). This pull request does not publish a release.
- pip-audit remains the CI dependency scanner. The site lockfile's moderate advisories in Starlight's expressive-code and postcss chain are still open. This pull request does not bump them.
- The translated READMEs are in the tree from the 2026-10-12 treatment. The Translations box stays open because the Publisher runs them again after this README.
