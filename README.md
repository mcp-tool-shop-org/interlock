# switchyard

<img src="assets/logo/switchyard_512.png" width="256" alt="switchyard">

A rail switchyard sends each car onto the right track. This one sends each job to a model, an engine and a device.

You give it a task (embed, rerank, NLI, verify, generate) and the constraints that matter (quality, latency, context, thinking level). It answers with a model, an engine, a device and the settings, and it shows the receipt behind every number. OpenVINO's AUTO and HETERO plugins can pick a device once you have already picked the model. Nothing in that stack picks the model. That gap is the product.

What this program is allowed to touch is in the [threat model](#threat-model).

The router itself comes last. The work before it is a receipt schema, a routing table built from receipts the studio already has, and a harness that pre-registers an experiment before any device time is spent.

## What it will not do

switchyard does not serve traffic and it does not proxy requests. It does not train models. It does not edit offrig, npu-serve or rnd. Those stay with their owners. A serving layer needs its own brief.

## Devices

Every device is named.

| Name | What it is |
|---|---|
| `NPU` | Intel AI Boost on the Core Ultra 9 285K |
| `iGPU` | The Intel GPU. A live enumeration must say vendor `0x8086` and device type `INTEGRATED`. The full name is recorded, not the decision |
| `CPU` | The same processor's CPU |
| `5090` | The RTX 5090, reached only through Ollama at `127.0.0.1:11434` |

`GPU.1` is the 5090 as OpenVINO sees it on this rig. The resolver refuses that index even if the name or the vendor changes, refuses a bare `GPU`, refuses `AUTO`, `HETERO` and `MULTI`, and refuses a GPU that is not vendor `0x8086` and `INTEGRATED`. A list with neither device property still requires the full name to contain "Intel". A missing device is an error. switchyard does not fall off onto a weaker one.

The NPU is exclusive. A 5090 step runs only inside a Publisher grant. The runner records that grant. CPU-heavy legs wait for a window the Publisher announces.

## Receipts

A routing row names the task, the model (repo or tag, revision or digest, quantisation, licence), the engine and its version, the device, the settings, and the quality and latency it measured. Quality metrics carry their own `n`, interval and dataset hash. Latency keeps cold and warm apart. A single run is marked provisional. The row points at the source receipt and the commit it was imported from, or at an experiment spec and a run id.

```
switchyard check
```

That validates every receipt under `receipts/` and `table/routing.jsonl`. An empty tree is fine. A home-directory path in a receipt is not.

## Threat model

switchyard touches local devices and local HTTP, and the network only to download a model that an experiment spec already named, from that model's publisher.

- Receipts and the routing table are local files.
- Ollama calls, when a run is actually started, go to `127.0.0.1:11434` (the 5090) or `127.0.0.1:11490` (CPU). Cloud model tags are refused.
- OpenVINO runs, when a run is actually started, use `NPU`, `CPU`, or the Intel iGPU (vendor `0x8086` and `INTEGRATED`). They never select `GPU.1`.
- A 5090 step without `--grant` does not start. An NPU or iGPU step without `--npu-session` does not start.
- The 5090 path checks that the VRAM watchdog heartbeat is at most 60 seconds old, that no model is loaded, and that the card has rested 15 minutes since the previous model. The heartbeat file is not opened. Only its modified time is read.
- Nothing here listens on a port.

## Layout

`devices` is the allow-list. `guards` is the grant, the session, the heartbeat, the empty card and the rest. `receipt` is the schema and `switchyard check`. Importers, the experiment harness and the advisory router land in later changes, each on its own branch.

The build brief and the four feedback notes it names are snapshotted in `docs/handoff/`. The copy in the studio memory store stays authoritative.

## Develop

The environment is its own, at `E:/AI/envs/switchyard`, Python 3.12. It does not share the NPU environment. OpenVINO 2026.4.1 is the `intel` extra, pinned to the same versions as that environment. Tests do not need it.

```
uv sync --frozen
uv run switchyard check
uv run pytest
```
