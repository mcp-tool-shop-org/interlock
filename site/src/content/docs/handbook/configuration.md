---
title: Configuration
description: The INTERLOCK_* variables, and the one watchdog path.
sidebar:
  order: 3
---

interlock has no config file. A measurement run reads a few environment variables. `route`, `table`, `check`, and `runs` do not need them.

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

Path of the VRAM watchdog heartbeat file. The 5090 path checks that the file's modified time is at most 60 seconds old, and that no model is loaded, before a step starts. The file is not opened. Only its modified time is read. The default is a path on the measurement rig. Set the variable to point somewhere else. This name is not an `INTERLOCK_` name.

## What is not configuration

`interlock route --budget-left-usd` and `--busy` are inputs on the command. interlock does not read offrig's balance, a ledger, or a process list to fill them in.

A model download, when a run is actually started, is limited to a model the committed spec already named. That is the spec, not an environment variable.
