---
title: Security
description: What interlock touches, what a receipt must not carry, and how to report a vulnerability.
sidebar:
  order: 6
---

interlock 0.1.x is a prototype. What it touches depends on the command. There is no telemetry.

## By command

`interlock route` is advisory only. It reads committed table rows, touches no device, reads no ledger, and calls nothing over the network. A route is a suggestion. It does not start a device.

`interlock run` drives local devices: CPU, the Intel iGPU, the NPU via OpenVINO, and the RTX 5090 via loopback Ollama. The 5090 needs an explicit `--grant`. The NPU and the iGPU need `--npu-session`. A CPU-heavy leg needs `--cpu-window`. It calls Ollama on loopback only, and downloads only spec-named models at pinned revisions. It writes receipts under `receipts/`, a run marker, and a compile cache on local disk.

`interlock check` and `interlock table` read local files. `interlock import` rebuilds `table/` from committed receipts. None of them touches the network. `interlock runs` does not start a device.

`interlock route` can emit an `offrig_plan` argument object. interlock never launches, rents, or spends anything.

## What a receipt may hold

A receipt may record a model repo and a revision or digest, an engine and version, device names and driver versions, settings, a dataset hash, timings, the run's start state, and grant text.

A receipt must not carry a home path, a hostname beyond loopback, a token, or a user identity. `interlock check` enforces the path, host, and token rules. It does not detect a person's name. The maintainers run a separate identity scan before a push. That scan is not this program.

`interlock check` refuses a home-directory path, including one an expanded profile variable would produce, a UNC path, an `ssh://` or `sftp://` host, a `git@host:` form, a `user@host.domain` form, and a token-shaped value. Hex or base64 values of 32 or more characters are refused only in fields named like key, token, secret, or password.

A bare single-label host and port is a known limit of that check. `interlock check --host-port` turns the extra check on. It does not flag a model tag.

## Network

Ollama calls, when a run is actually started, stay on loopback. Cloud model tags are refused. Nothing in this repository listens on a port. The 5090 path checks that the watchdog heartbeat is at most 60 seconds old and that no model is loaded. The heartbeat file is not opened. Only its modified time is read.

A remote price enters only as a dated snapshot file. interlock does not open offrig's database.

## Supported versions and reporting

Supported versions are 0.1.x. Report a vulnerability through a GitHub private advisory on [mcp-tool-shop-org/interlock](https://github.com/mcp-tool-shop-org/interlock/security/advisories/new). Do not open a public issue for a vulnerability. The maintainers read those advisories. This prototype does not publish a response-time promise.

Release assets carry a SLSA provenance file (`*.intoto.jsonl`) on the same GitHub release; verify with `slsa-verifier verify-artifact --provenance-path <name>.intoto.jsonl --source-uri github.com/mcp-tool-shop-org/interlock --source-tag <the-tag> <the-asset>`.

`SECURITY.md` at the repository root is the reporting file. This page is the short form of it, not the same text.
