---
title: Usage
description: interlock route, run, runs, table, and check.
sidebar:
  order: 2
---

Six commands. `route`, `table`, `check`, `import`, and `runs` do not start a device. `run` does, unless you pass `--dry-run`.

## interlock route

`interlock route` answers from `table/routing.jsonl`. It does not start a device, read a ledger, or call offrig.

```bash
interlock route --task generate --model Phi --thinking off --ctx 1024 --busy 5090
```

`--task` is required. The tasks are `embed`, `engine`, `generate`, `judge:planted-error`, `judge:same-meaning-rewrite`, `nli`, `offer`, `rerank`, `verify:grounded`, `verify:math-ladder`, and `verify:reasoning`.

| Flag | What it does |
| --- | --- |
| `--model` | Substring of the repo or tag. Hyphens and colons are ignored. |
| `--min-<metric>` | Floor on a conservative bound. `--min-tok-s` is the minimum repeat-slot decode tok/s. Any other name is a quality metric, judged on the low end of its interval. |
| `--max-latency-ms` | Maximum repeat-slot latency, or claim-wall max, in milliseconds. |
| `--max-ttft-ms` | Maximum repeat-slot time to first token. Refuses unless every compared row records the same `settings.prefix_caching` (`on`, `off`, or `default`). Unset means unknown. |
| `--ctx` | Exact measured length when the row id has `lN`. A row with `num_ctx` and no `lN` tag qualifies when `num_ctx` is at least N. |
| `--thinking` | `off` matches only off. `on` matches any recorded level, and the decision keeps that level. `low`, `medium`, and `high` match exactly when named. |
| `--busy` | Repeatable. An input. interlock does not read a ledger, the watchdog, or a process list. |
| `--allow-remote` | Allow remote `colab:<card>` rows to compete (default off). |
| `--device` | Exact device or provider wildcard such as `colab:L4` or `colab:*`. |
| `--budget-left-usd` | An input, like `--busy`. A remote row whose worst case exceeds it is a runner-up. With no flag, a remote suggestion is marked budget unchecked. |
| `--as-of` | `YYYY-MM-DD`. Defaults to today. The policy does not read the clock. |
| `--json` | Print a document valid against `src/interlock/schema/route-decision.v1.json`, or a structured error. |
| `--quote-rule`, `--num-predict` | Extra constraints for a verify route. A quote rule the table does not have is a refusal. |
| `--root` | Project root. The default is the current directory. |

Generate ranks by the lowest repeat-slot decode tok/s. Every other task ranks by the highest repeat-slot latency, or by the maximum claim-wall when the row has no repeat slots. The margin is the gap between those conservative bounds, never a median. Ties leave the 5090 free.

A row older than 30 days is stale. A row marked provisional, or with fewer than two repeats, is provisional. Both stay eligible and are flagged.

`native_binary_rejected`, `compile_rejected`, `compile_timeout`, `hang`, `device_lost`, `oom`, `load_refused`, `timeout`, `ctx_exceeded`, and a row with `parity_valid` false are listed under excluded. They are not candidates. When no row meets the constraints the command exits 1 and names the constraint and any measured failure reason. It does not fall back.

Remote `colab:<card>` cards are opt-in: a default `interlock route` request never selects a remote card unless `--allow-remote` or `--device colab:<card>` is passed.

A remote suggestion includes an `offrig_plan` object when present on the row. interlock does not call `offrig_plan` or `offrig_launch`, and it does not spend. See `docs/offrig-integration.md`.

## interlock table

Prints routing rows, filtered by `--task` or `--device` (including exact remote cards such as `colab:L4` or wildcards such as `colab:*`).

```bash
interlock table --task generate --device NPU
interlock table --device colab:L4
interlock table --device colab:*
```

## interlock check

Validates every receipt under `receipts/` and `table/routing.jsonl`, including the privacy check.

```bash
interlock check
interlock check --host-port
```

`--host-port` also refuses a single-label host and port, such as a machine name followed by a port. A model tag such as `gemma4:31b` is not a host. Without the flag, that single-label form is a known limit of the check.

## interlock import

Rebuilds `table/` from the committed receipts.

```bash
interlock import
interlock import --remote-window receipts/c2-l4-window-01.json
```

`--offers` points at an offers file, and `--remote-window <path>` (repeatable) imports a remote Colab window receipt JSON into `table/routing.jsonl` while preserving every repeat slot in `artifacts.slots`. The default rebuild does not need either flag. Import does not read `aborted/`.

## interlock run

Runs a committed spec. The spec path is the one argument.

```bash
interlock run experiments/e4/spec.yaml --dry-run
interlock run experiments/e4/spec.yaml --grant "<grant text>" --npu-session "<session>"
```

| Flag | What it does |
| --- | --- |
| `--dry-run` | Print the plan and the guards a real run would need. Start nothing. |
| `--grant` | Required before a 5090 step starts. The runner records the grant text. |
| `--npu-session` | Required before an NPU or iGPU step starts. |
| `--cpu-window` | Required before a CPU-heavy leg starts. |
| `--run-id` | Use this id. An id already in use is refused. |
| `--pilot` | For E4b, write the clock pilot and do not write receipts or the routing table. |
| `--pairs` | Comma-separated E4b pair ids, a subset of the spec. The receipt records them. |
| `--notes` | A note stored with the run. The default is empty. |
| `--quarantine` | Where a stopped run is set aside. The default is `aborted/` under `--root`. |
| `--root` | Project root. The default is the current directory. |

With no `--run-id`, E4b uses `e4b-block` or `e4b-pilot`, then `<base>-2`, `<base>-3`, and so on. The chosen id is the first line of the run and a field on the receipt.

## interlock runs

Neither subcommand starts Ollama or a device.

```bash
interlock runs list
interlock runs quarantine <run-id>
```

`list` prints each run id and a state: `complete`, `aborted`, or `stray`. `quarantine` moves a stray series into `aborted/` with a note that it was set aside by hand. `--quarantine` chooses that directory. `--root` is the project root.

`receipts/` must not hold a `*-series/` directory that has no receipt. The next run refuses to start while one is there, and names `interlock runs quarantine`.

## interlock ollama-health

Prints one line per local Ollama port the supervisor knows, and exits 1 if either is down. A port is healthy when `GET /api/version` answers within 5 seconds. `--dry-run` on `interlock run` prints the restart policy and does not contact either port.

## Exit codes

| Code | When |
| --- | --- |
| 0 | The command finished. |
| 1 | A user error, a route refusal, or a check failure. |
| 75 | A native compile ran past `compile_ceiling_s` (default 900 s). The receipt is flushed first. |
| 76 | System RAM at 88% aborted the run. The receipt is flushed first. |
| 77 | A stop file `receipts/<run-id>.stop` asked the watcher to end the run. |

These are not 0, 1, 2, and 3. A route refusal on `--json` is a document with `code`, `message`, `hint`, `constraint`, and `experiment`. The other commands print a line.
