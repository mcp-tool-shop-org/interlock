---
title: Configuration
description: The INTERLOCK_* variables, WATCHDOG_HEARTBEAT, and the interlock.toml file.
sidebar:
  order: 3
---

A measurement run reads a few environment variables. `route`, `table`, `check`, and `runs` do not need them. Two things can also be set in an optional `interlock.toml` in the working directory: `[paths]` (below) and `[import.sources]` (where `interlock import` finds its source repositories; see the README). An unknown key in either table is refused, naming the valid ones. An environment variable beats the file, and there is no default path on anyone's rig.

`SWITCHYARD_*` still works for this release. When a `SWITCHYARD_` name answers and the `INTERLOCK_` name does not, interlock warns once on stderr and tells you the new name.

## INTERLOCK_OLLAMA_SERVE

Path of a JSON file that records how Ollama was started for this run. A restart replays that record. A missing record refuses the restart before anything is stopped.

The fields are `argv`, `env`, `log_path`, `library_path`, `ggml_cuda_sha256`, and `backend_line`. interlock does not sign in. There is no `OLLAMA_CUDA_BACKEND` variable. If a backend has to be forced, the variable is `OLLAMA_LLM_LIBRARY`, and it belongs in the recorded start line.

At the start of a run the record's `OLLAMA_*` and `LLAMA_ARG_*` set is compared with the live process environment. A mismatch disables restarts for that run. The reason is written on the run marker as `ollama_restart_disabled`. Nothing is stopped.

`SWITCHYARD_OLLAMA_SERVE` is the old name.

## INTERLOCK_OLLAMA_LOG

Path of the Ollama log that the fill-card reader opens, when the run was given one. An unreadable log is unrecorded, with a reason. The fill-card search does not invent flash attention, the KV cache type, or the backend library.

`SWITCHYARD_OLLAMA_LOG` is the old name.

## INTERLOCK_OLLAMA_EXE

Path of the `ollama` executable. The CUDA library path is derived beside it, under `lib/ollama/cuda_v13/`. The run records the library it was given. A server that comes back on a different library stops the run.

`SWITCHYARD_OLLAMA_EXE` is the old name.

## WATCHDOG_HEARTBEAT

Path of the VRAM watchdog heartbeat file, or `heartbeat` under `[paths]` in `interlock.toml`. The 5090 path checks that the file's modified time is at most 60 seconds old, and that no model is loaded, before a step starts. The file is not opened. Only its modified time is read. There is no default path: with neither set, the run refuses before any device is touched, and the message names `WATCHDOG_HEARTBEAT`. If you run no watchdog, pass `--no-watchdog` to `interlock run`. That is the only way to skip the check; it is never implied by an unset variable, and the receipt records it (in its `notes`, and in the `watchdog` field of an E4b header or pilot document). The empty-card check still runs. This name is not an `INTERLOCK_` name.

## INTERLOCK_SCRATCH_ROOT

Path of the scratch directory offrig opens as its project for `offrig verify calibrate`, or `scratch_root` under `[paths]`. The default is `interlock-scratch` under the system temp directory.

## INTERLOCK_WEIGHTS_HOT

Path of the hot weights cache an arm with `hf_cache: hot` reads, or `weights_hot` under `[paths]`. A missing copy is a refusal. The default is `hot/` under `HF_HOME`, or under the user's Hugging Face cache.

## interlock.toml

```toml
[paths]
scratch_root = "scratch"
weights_hot = "hf-hot"
heartbeat = "watchdog/HEARTBEAT"
```

A value is used as you gave it and is never expanded to a home path in anything interlock publishes.

## What is not configuration

`interlock route --budget-left-usd` and `--busy` are inputs on the command. interlock does not read offrig's balance, a ledger, or a process list to fill them in.

A model download, when a run is actually started, is limited to a model the committed spec already named. That is the spec, not an environment variable.
