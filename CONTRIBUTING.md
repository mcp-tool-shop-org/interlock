# Contributing

## Reporting a bug

Open a [GitHub issue](https://github.com/mcp-tool-shop-org/interlock/issues). Include:

- the version you ran (`interlock --version`, or the git revision)
- the command
- a receipt, or a log excerpt that shows the failure

Take home paths and user names out of that excerpt before you paste it.

## Security

A security issue follows [SECURITY.md](SECURITY.md). Do not file it as a public issue.

## Changes

Changes come in as pull requests against `main`.

Every pull request:

- ships tests with the change
- keeps coverage at 90 or above (the CI floor)
- leaves CI green
- adds an entry under Unreleased in `CHANGELOG.md`

Run this before you open the pull request:

```
uv sync --frozen
uv run ruff check
uv run interlock check
uv run pytest
```
