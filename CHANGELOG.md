# Changelog

## Unreleased

- Fixed: E4b's live runner was a bare `subprocess.run`, which captures no output, so the 11434 pin read "" and the pilot refused ("the 11434 listener report is not JSON"); the receipt's git sha and clean flag read the same way. The default is now `capture_run` (captured, text). Another `test_fake_hides_platform` instance: the fakes returned stdout. Found live by the Publisher, 2026-10-12.
- Fixed: `read_git` fails closed (R&D on #84). Both git calls must exit 0 and HEAD must be a 40-hex sha, and the tree is clean only on a successful, empty porcelain read; a runner that captured nothing used to record `git_sha ""` and `clean_tree true`. The pilot and the block read git before any device starts. `pin_install` refuses a CUDA sha that isn't a 64-hex digest.

- Fixed: the 11434 listener lookups (E4b's pin, E5's runner, the supervisor) no longer filter on `-State Listen`. On Windows build 26340 a listening socket reports state 0, so the E4b pilot refused with "the 11434 listener report is not JSON". They now take the socket on the port that has no remote port. Found live by the Publisher, 2026-10-12.

- If the Ollama log's size cannot be read, a restart stops the run instead of checking the whole log. A missing log is still an empty log.
- After an Ollama restart, the `load_backend` line is taken from the log bytes written after the spawn. A line from the previous server is not a match. When the new server has not written one yet, the check runs after the first model load and before that sample is kept. At the start of a run, the recorded `OLLAMA_*` and `LLAMA_ARG_*` set is compared with the live server. A mismatch disables restarts for that run and is written on the marker. After a restart the same full set is compared. An unreadable `/api/ps` during a hang, while health still answers, refuses the restart.
- E4b runs on its own Ollama at 127.0.0.1:11441, pinned to the install that is listening on 11434, and stops that process tree at block end and on abort. 11434 and 11490 are not opened. Probes are GET /api/version and GET /api/ps only. Ending a companion session unloads with keep_alive 0; an empty generate is no longer a stop. Each slot has its own warm-up, a 5090 slot waits until the clock is steady, and a refused slot is re-run once. The pair verdict is the Hodges-Lehmann estimate with an exact Wilcoxon interval. `switchyard run experiments/e4b/spec.yaml --pilot` writes the clock pilot under pilot-e4b/ and does not write receipts or the routing table. E4's noise gate is unchanged.
- E4b's own Ollama inherits the process environment with every OLLAMA_* and LLAMA_ARG_* key removed, then sets OLLAMA_HOST, OLLAMA_MODELS and OLLAMA_DEBUG. The receipt keeps those three. The run refuses unless this start's log shows cuda_v13 on its load_backend line, the first load reports size_vram equal to size, and that model's digest matches the spec pin. The watchdog heartbeat and an empty card are required before the server starts, for the block and for --pilot. A 5090 companion stops its generate loop and stays loaded until block end; NPU and iGPU companions stop and unload (spec amendment 2026-10-12).
- E5 fill-card reads the 5090 adapter's shared usage, matched by dedicated usage, and gates on the delta from the baseline taken before the window. The search stops at the model's trained context length, and a window that is not resident is halved to one grain and then bisected upward. Flash attention, the KV cache type, the default context and the backend library come from the server log. A missing log is unrecorded with a reason. Each search call on the row keeps its peak, shared usage, prompt rate and reason.
- The Ollama catalogue follows its recorded fixtures. `truncate_over_num_ctx` is a reply at `num_ctx` 8, which is what 0.35.1 returned. The no-limit cases note that `num_ctx` 64 did not bound them (11 prompt tokens and 144 eval tokens, stopped at end of turn); the 15 second timeout and the unload were the guard. The recorder accepts `--only`, `--num-ctx` and `--prompt-repeats` so one case can be recorded again.
- An OpenVINO embed, nli, or rerank step on the iGPU refuses a missing `static_shape` before compile, as the NPU already does. A dynamic iGPU shape is allowed only when the spec sets `dynamic_shape: true`, and that opt-in is recorded on the row. Generate is unchanged: it builds an LLMPipeline, not the graph that raises `rank().is_static()`. The fakes return the recorded output shape per task (position rows for embed, one flat label row for nli and rerank). Pooling `none` refuses a multi-row matrix instead of taking row 0. The real export drops its batch axis before pooling, and a vector with no batch axis is refused.
- A generate that is not the unload shape, including a companion stop, is a load. The recorder sends the unload and waits for the card before it checks that the card is idle again, including when that later check refuses. Each request is written to stderr as it is sent.
- After a recorded load, the recorder sends the unload and then polls the card and an empty `/api/ps` for about 10 seconds before it refuses. A refusal or error after a load still sends that unload, so the model is not left resident.
- The three no-limit recordings (`num_predict` 0, null, and missing) send `num_ctx` 64 and `shift` false, so the window ends the generate. The fixture keeps `behaviour` `no_reply`, so a fake still refuses, and it stores `eval_count` above 1. The recorder requires `--grant` and checks the heartbeat and an empty card before any request. The `num_predict` 1 fixture records `cold_load_s`. The count budget adds a prefill term: the longest count the spec can ask for, at 6,000 tokens per second.
- Ollama fakes check each request against a recorded exchange. `scripts/record_ollama_fixtures.py` writes those exchanges; it is run on the rig, and it refuses a cloud model or a card that is not idle. `num_predict` 0, null, and missing stay `no_reply` for the fakes. A token count stops after twice the cold-load allowance of the largest roster model (`/api/tags` size, or 16 GiB when the size is missing), plus that prefill term, and the arm is refused. `confirm_num_predict` below 1 is refused when the spec is validated, and again before a fill-card model is contacted.
- A hung local Ollama on `127.0.0.1:11434` is restarted between steps. Port `11490` is reported by `switchyard ollama-health` and is never stopped. Before the stop, the run re-checks the grant, the watchdog heartbeat, and that `/api/ps` lists no other session's model. `ollama serve` is replayed from the start line recorded for the run, and the log is appended. The CUDA library sha256, the server's `load_backend` line, flash attention, and the KV cache type must match. The first kept sample after a restart stays out of the warm percentiles. A slow request whose server still answers is left alone. At most two restarts a run. The step in flight is inconclusive, and a measurement taken across a restart is not kept.
- OpenVINO NLI and rerank steps take the model's flat logit row as their output. Pooling `none` used to expect one row per position, so a real cross-encoder failed with "'float' object is not iterable" (E4's sixth live run, iGPU, 2026-10-12). The fakes had returned per-position rows.
- The Ollama prompt fit estimates the seed repeats from the seed's own count, then trims only past the longest prefix known to be short. It used to add one seed per count, which took thousands of counts at a 118k-token target: E5's first live run spent over 30 minutes on one model without a row.
- The Ollama token count asks for one completion token, never zero. Ollama 0.35.1 reads `num_predict` 0 as no limit and generates until the context fills. E4's first live run (2026-10-11) sat at full card power for 12 minutes without writing a row. The zero-then-retry path is gone.

## 0.1.0 — prototype

Advisory only. `switchyard route` refuses when no measured row meets the constraints. It does not fall back, serve traffic, read a ledger, or call offrig. The bound is the conservative end of the measured spread, never a median. A parity miss is excluded. `thinking off` matches only off. `thinking on` matches any recorded thinking level, and the decision keeps that level.

### Phase 0

Scaffold: the device allow-list, the grant and session guards, receipt validation, and CI with a coverage floor of 90. The iGPU is the Intel integrated GPU.

### Phase 1

The routing table is built from pinned receipts.

### Phase 2

The harness runs a committed spec and writes receipts. A rejected compile, a hang, and a lost device are recorded, and they are not speed.

### Phase 3

E1, E2, and E3 measure embed and small-model generation on CPU, the Intel iGPU, and the NPU. A 5090 reference anchors Qwen3-8B.

### Phase 4

`switchyard route` chooses from that table. The choice is one pure function of the rows and the request. When the set is empty, the command exits 1 and names the constraint. It does not guess.
