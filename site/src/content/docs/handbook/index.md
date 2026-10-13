---
title: Welcome
description: What interlock is, what it will not do, and where to read next.
sidebar:
  order: 0
---

interlock is a local AI device router. You name a task and the constraints that matter. It answers with a model, an engine, a device, and the receipt behind every number.

Railway interlocking stops a train being routed onto a conflicting path. This tool does that for devices. OpenVINO's AUTO and HETERO plugins can pick a device once you have already picked the model. Nothing in that stack picks the model. That gap is the product.

This is a prototype, version 0.1.x. The CLI, the receipt format, and the experiment specs change without notice. Nothing is on PyPI. interlock is advisory only: it suggests a route and it never serves traffic. Every number in the table was measured on one rig, an Intel Core Ultra 9 285K with its NPU and iGPU, plus an RTX 5090. Treat a route as that rig's evidence, not a general benchmark.

## Read next

- [Getting started](./getting-started/) installs from a clone.
- [Usage](./usage/) covers `route`, `run`, `runs`, `table`, and `check`.
- [Experiments and receipts](./experiments/) is the spec model, the run lifecycle, and the E4b result.
- [Configuration](./configuration/) is the `INTERLOCK_*` variables.
- [Architecture](./architecture/) is how a route is chosen.
- [Security](./security/) is the threat model.

## What it will not do

interlock does not serve traffic and it does not proxy requests. It does not train models. It does not edit offrig, npu-serve, or rnd. Those stay with their owners. A serving layer needs its own brief.

It does not fall back. When no measured row meets the constraints, `interlock route` exits 1 and names the constraint.
