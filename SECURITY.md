# Security

interlock 0.x is a prototype. What it does depends on the command.

## interlock route

Advisory only. It reads committed table rows, touches no device, reads no ledger, and calls nothing over the network. A route is a suggestion. It does not start a device.

## interlock run

Drives local devices: CPU, the Intel iGPU, the NPU via OpenVINO, and the RTX 5090 via loopback Ollama. The 5090 needs an explicit `--grant`; the NPU and the iGPU need `--npu-session`; a CPU-heavy leg needs `--cpu-window`. It calls Ollama on 127.0.0.1 only, and downloads only spec-named models at pinned revisions from Hugging Face.

It writes receipts under `receipts/`, a run marker, and a compile cache on local disk.

## offrig

`interlock route` can emit an `offrig_plan` argument object. interlock never launches, rents, or spends anything.

## interlock check, table, and import

`check` and `table` read local files; `import` rebuilds the files under `table/` from committed receipts. None of them touches the network.

## What receipts contain

A receipt may record a model repo and a revision or digest, an engine and version, device names and driver versions, settings, a dataset hash, timings, the run's start state (CPU %, RAM %), and grant text.

A receipt must not carry a home path, a hostname beyond loopback (`127.0.0.1`, `localhost`, `::1`), a token, or a user identity. `interlock check` enforces the path, host and token rules below; it does not detect a person's name, so the maintainers run a separate identity scan before every push. `interlock check` refuses a home-directory path, including one an expanded profile variable would produce, a UNC path (`\\server\share\...`), an `ssh://` or `sftp://` host, a `git@host:` or `user@host.domain` form, and a token-shaped value. Token prefixes refused anywhere include `hf_`, `sk-`, `ghp_`, `github_pat_`, `gho_`, `ghs_`, `ghu_`, `xox[abpr]-`, an `AKIA` access-key id, and `Bearer` followed by a token. Hex or base64 values of 32 or more characters are refused only in fields named like key, token, secret or password.

A bare single-label host and port, such as `myrig:11434`, is a known limit of that check. `interlock check --host-port` turns the extra check on. It does not flag a model tag such as `gemma4:31b`.

## Supported versions

Only the latest 0.x release receives security fixes.

## Reporting a vulnerability

Use a GitHub private advisory on [mcp-tool-shop-org/interlock](https://github.com/mcp-tool-shop-org/interlock/security/advisories/new). Do not open a public issue for a vulnerability. The maintainers read those advisories. This 0.x prototype does not publish a response-time promise.

Release assets carry a SLSA provenance file (`*.intoto.jsonl`) on the same GitHub release; verify with `slsa-verifier verify-artifact --provenance-path <name>.intoto.jsonl --source-uri github.com/mcp-tool-shop-org/interlock --source-tag <the-tag> <the-asset>`.
