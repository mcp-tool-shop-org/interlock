---
title: Architecture
description: How a job becomes a route, and what each part of the tree owns.
sidebar:
  order: 5
---

`interlock route` is one pure function of the rows and the request. It does not read the clock, a ledger, the watchdog, or a device. The harness is the other door. It runs a committed spec and writes receipts. Import turns those receipts into the table the route reads.

```text
spec.yaml  ->  interlock run  ->  receipts/*.jsonl
                                      |
                                      v
                               interlock import
                                      |
                                      v
                              table/routing.jsonl
                                      |
                                      v
                               interlock route
```

## The parts

`devices` is the allow-list. The local rig names are `NPU`, `iGPU`, `CPU`, and `5090`. Remote Colab cards are named `colab:A100-40`, `colab:A100-80`, `colab:L4`, `colab:T4`, and `colab:G4`, and each remote row carries `remote_device` identity (`provider`, `card`, `vram_mib`, `driver_version`, `cuda_version`, and a zone `pool` string—never an account id). Remote rows are kept apart from rig rows and are never pooled with them. `GPU.1` is the 5090 as OpenVINO sees it on this rig. The resolver refuses that index even if the name or the vendor changes, refuses a bare `GPU`, refuses `AUTO`, `HETERO`, and `MULTI`, and refuses a GPU that is not vendor `0x8086` and `INTEGRATED`. A list with neither device property still requires the full name to contain "Intel". A missing device is an error.

`guards` is the grant, the session, the heartbeat, and the empty card. A 5090 step without `--grant` does not start. An NPU or iGPU step without `--npu-session` does not start. A CPU-heavy leg waits for `--cpu-window`.

`receipt` is the schema and `interlock check`. `importers` rebuilds `table/routing.jsonl` from pinned receipts. `interlock table` prints that table. `table/gaps.md` is the experiment backlog.

The harness runs a committed spec. A rejected compile, a hang, and a lost device are recorded, and they are not speed. A watcher thread guards every run from outside the main thread.

`router/policy.py` is the choice. When the set of rows is empty, the command exits 1 and names the constraint. It does not guess.

## Two adapters

The 5090 is reached through Ollama on the local machine. Cloud model tags are refused. E4b runs on its own Ollama, pinned to the install that is already listening for the 5090, and stops that process tree at the end of the block and on abort. The everyday servers are not opened by that harness.

OpenVINO runs, when a run is actually started, use `NPU`, `CPU`, or the Intel iGPU. They never select `GPU.1`. An embed, NLI, or rerank step on the iGPU refuses a missing static shape before compile, unless the spec sets `dynamic_shape: true`, and that opt-in is recorded on the row.

## What a route is not

A route is not a server, a proxy, or a launch. A remote `colab:<card>` row is eligible only when the request opts in (`--allow-remote` or `--device colab:<card>`), and failed remote shapes (`oom`, `load_refused`, `timeout`, `ctx_exceeded`) are recorded as capability rows so refusals name the measured reason. A remote row can also carry an `offrig_plan` object so a later tool could price a pod. interlock does not call that tool and does not spend.

The schema for a `--json` decision is `src/interlock/schema/route-decision.v1.json`.
