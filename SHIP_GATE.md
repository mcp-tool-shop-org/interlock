# Ship Gate

> No repo is "done" until every applicable line is checked.
> Copy this into your repo root. Check items off per-release.

**Tags:** `[all]` every repo · `[npm]` `[pypi]` `[vsix]` `[desktop]` `[container]` published artifacts · `[mcp]` MCP servers · `[cli]` CLI tools

---

## A. Security Baseline

- [x] `[all]` SECURITY.md exists (reporting contact, supported versions, response timeline) — `shipcheck security-docs` passed 2026-10-12. Contact is a GitHub private advisory on this repo, not a mailbox. Supported versions are 0.1.x. This prototype does not publish a response-time promise. (A1)
- [x] `[all]` README includes threat model paragraph (data touched, data NOT touched, permissions required) — `shipcheck security-docs` passed 2026-10-12. (A2)
- [x] `[all]` No secrets, tokens, or credentials in source or diagnostics output — `shipcheck secrets` skipped (nothing is published, so there is no tarball yet). The identity scan of the git-tracked tree was RESULT CLEAN on 2026-10-12. (A3)
- [x] `[all]` No telemetry by default — the threat model says interlock does not send telemetry (2026-10-12)

### Default safety posture

- [x] `[cli|mcp|desktop]` Dangerous actions require an explicit consent flag (2026-10-12). A 5090 step needs `--grant`, an NPU or iGPU step needs `--npu-session`, and a CPU-heavy leg needs `--cpu-window`. An Ollama restart re-checks that grant. The flags are not named `--allow-*`.
- [x] `[cli|mcp|desktop]` File operations stay under the run root named by `--root` (receipts, aborted, compile-cache, logs, pilot-e4b, table) (2026-10-12)
- [ ] `[mcp]` SKIP: not an MCP server
- [ ] `[mcp]` SKIP: not an MCP server

## B. Error Handling

- [ ] `[all]` SKIP: only `interlock route --json` returns a document with code, message, hint, constraint, and experiment. The other commands print a line and an exit code. This treatment does not change that contract.
- [ ] `[cli]` SKIP: exits are 0, 1, and the harness stops 75 (compile timeout), 76 (RAM), and 77 (stop file). They are not 0, 1, 2, and 3. This treatment does not renumber them.
- [ ] `[cli]` SKIP: there is no `--debug` flag. An uncaught error can still print a traceback. This treatment does not add a debug switch.
- [ ] `[mcp]` SKIP: not an MCP server
- [ ] `[mcp]` SKIP: not an MCP server
- [ ] `[desktop]` SKIP: not a desktop app
- [ ] `[vscode]` SKIP: not a VS Code extension

## C. Operator Docs

- [x] `[all]` README is current: what it does, install from a clone, usage, Python 3.12, Ubuntu CI, Windows for a live run (2026-10-12)
- [x] `[all]` CHANGELOG.md (Keep a Changelog format) — Unreleased and 0.1.0 (2026-10-12)
- [x] `[all]` LICENSE file present and repo states support status — MIT, supported 0.1.x, prototype (2026-10-12)
- [x] `[cli]` `--help` output accurate for all commands and flags — checked against `interlock --help`, `route --help`, `run --help`, and `runs --help` on 2026-10-12
- [ ] `[cli|mcp|desktop]` SKIP: no silent, normal, verbose, or debug logging levels. Receipt privacy is `interlock check`, not a log redactor. This treatment does not add a logging switch.
- [ ] `[mcp]` SKIP: not an MCP server
- [x] `[complex]` Operator handbook is the Starlight site under `site/src/content/docs/handbook/` (welcome, getting started, usage, experiments, configuration, architecture, security), not a root HANDBOOK.md (2026-10-12)

## D. Shipping Hygiene

- [x] `[all]` `verify` script exists (test + build + smoke in one command) — `verify.sh` (2026-10-12)
- [x] `[all]` Version in manifest matches git tag — `shipcheck manifest` passed 2026-10-12: no semver tag exists, and 0.1.0 stays. Unfinished work stays 0.x (2026-09-09). This treatment does not tag. (D2)
- [x] `[all]` Dependency scanning runs in CI — `uv run --with pip-audit==2.10.1 pip-audit` in ci.yml (D3)
- [x] `[all]` No known high/critical vulnerabilities — `pip-audit` 2.10.1 on the project environment found none on 2026-10-12 (the local distribution is not on PyPI, so it is skipped). The site tree has 10 moderate advisories in Starlight's expressive-code and postcss chain, and no high or critical. (D4)
- [ ] `[all]` SKIP: the org rule says not to add dependabot.yml unless a brief asks. The update bot is optional. Alerts stay the security outcome.
- [ ] `[npm]` SKIP: not an npm package. PyPI trusted publishing is already in release.yml, and this treatment does not publish.
- [ ] `[npm]` SKIP: not an npm package
- [x] `[pypi]` `requires-python` is `>=3.12,<3.13` — `shipcheck manifest` engines passed 2026-10-12 (D6)
- [x] `[pypi]` Lockfile committed (`uv.lock`) — `shipcheck manifest` lockfile passed 2026-10-12. Wheel and sdist are built by `verify.sh` (`uv build`) and are not published. (D7)
- [ ] `[vsix]` SKIP: not a VS Code extension
- [ ] `[desktop]` SKIP: not a desktop app

## E. Identity (soft gate — does not block ship)

- [x] `[all]` Logo in README header — centred brand URL, width 400 (2026-10-12). The image resolves when the brand PR merges.
- [ ] `[all]` Translations (polyglot-mcp, 8 languages)
- [x] `[org]` Landing page (@mcptoolshop/site-theme) and Starlight handbook, base `/interlock` (2026-10-12)
- [x] `[all]` GitHub repo metadata: description and topics were already set. Homepage set to https://mcp-tool-shop-org.github.io/interlock/ (2026-10-12)

---

## Gate Rules

**Hard gate (A–D):** Must pass before any version is tagged or published.
If a section doesn't apply, mark `SKIP:` with justification — don't leave it unchecked.

**Soft gate (E):** Should be done. Product ships without it, but isn't "whole."

**Executed vs attested.** `shipcheck audit` only *counts these checkboxes* — it does not read your repo, so a box can be green while the fact is false. The lines that say **"executed by `npx @mcptoolshop/shipcheck <gate>`"** are backed by a command that reads the real artifact and exits 1 on the real defect. Run those gates (they are wired into shipcheck's own `verify`); don't just tick their boxes. Executed today: **A1/A2** (`security-docs`), **A3** (`secrets`), **D2/D6/D7** (`manifest`), **D3-config + OIDC/provenance** (`ci`), **real vulnerabilities + alerting** (`deps`), **D5** (`pack`), plus front-door (`front-door`) and dogfood freshness (`dogfood`). Every other line is still an attestation you are vouching for. Note the two dependency layers: `ci` proves a scanner is *configured*; `deps` proves there are *no known vulnerabilities* — a repo can pass the first while failing the second.

**Checking off:**
```
- [x] `[all]` SECURITY.md exists (2026-02-27)
```

**Skipping:**
```
- [ ] `[pypi]` SKIP: not a Python project
```
