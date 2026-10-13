# Scorecard

**Repo:** mcp-tool-shop-org/interlock
**Date:** 2026-10-12
**Type tags:** [all] [pypi] [cli]

## Pre-Remediation Assessment

| Category | Score | Notes |
|----------|-------|-------|
| A. Security | 7/10 | SECURITY.md and a threat model already existed. The advisory link still named switchyard. Telemetry was not stated. |
| B. Error Handling | 4/10 | `route --json` is a structured refusal. Other commands print a line. Exits are 0, 1, 75, 76, 77. |
| C. Operator Docs | 7/10 | README, CHANGELOG, and LICENSE were current. No handbook. The README wordmark still said switchyard. |
| D. Shipping Hygiene | 5/10 | `uv.lock` and `requires-python` were set. No verify script, no dependency scanner in CI, no release tag. |
| E. Identity (soft) | 2/10 | A local logo only. No landing page, no homepage, no translations. |
| **Overall** | **25/50** | |

## Key Gaps

1. No landing page or handbook, and the README logo was a local file whose wordmark still said switchyard.
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

After: the advisory link names interlock, the threat model says there is no telemetry, and the identity scan of the git-tracked tree was CLEAN. B is unchanged on purpose. The handbook is seven pages. `verify.sh` and pip-audit are in place. E is short of translations, which are not checked off.
