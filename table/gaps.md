# Gap report

This is the experiment backlog. A cell is open when the table has no row that fills it.
Imported rows: 456.

The inventory below was rendered at 456 rows and was not regenerated. `switchyard route` does not rewrite this file, and this change does not re-run `switchyard import`. The E1–E4 cells in that inventory are stale against the table on disk. The router section at the bottom is the one that describes what `switchyard route` can and cannot answer.

Ladder rows are imported from rnd at 08ca419f647a3ed69ee645610eaa1c7f35f40f91. Difficulty length quartiles are not imported. Tier rows are. Chain and ladder rows with no pins.quote_rule are quote rule 1 by the pre-#47 inference. B's planted-error judge row is imported from aspire-si at 3b05eab928894311993ffa3af2075fef39cccce6. That commit has B's receipts only. The 3b judge-control receipts are a later pin.

## Why E1–E4 are open

### E1

No open cell. This experiment's candidates are filled.

### E2

- `E2-qwen3-embed-NPU`. No speed or CPU-parity row for Qwen3-Embedding-0.6B on the NPU. This is not a retrieval-quality claim.
- `E2-qwen3-embed-iGPU`. No speed or CPU-parity row for Qwen3-Embedding-0.6B on the iGPU.
- `E2-qwen3-embed-CPU`. No CPU reference row for Qwen3-Embedding-0.6B. Parity is measured against this device.

### E3

- `E3-qwen3-8b-ov-NPU`. No OpenVINO generation receipt for Qwen3-8B-int4-cw-ov on the NPU. Verify rows do not fill a generate cell.
- `E3-qwen3-8b-ov-iGPU`. No OpenVINO generation receipt for Qwen3-8B-int4-cw-ov on the iGPU.
- `E3-qwen3-8b-ov-CPU`. No OpenVINO generation receipt for Qwen3-8B-int4-cw-ov on the CPU.
- `E3-phi-NPU`. The smaller channel-wise model chosen for E3 is Phi-3.5-mini-instruct-int4-cw-ov (MIT, text). No generation receipt on the NPU.
- `E3-phi-iGPU`. No generation receipt for Phi-3.5-mini-instruct-int4-cw-ov on the iGPU.
- `E3-phi-CPU`. No generation receipt for Phi-3.5-mini-instruct-int4-cw-ov on the CPU.
- `E3-qwen3-8b-5090`. qwen3:8b has verify rows on the 5090. Those are not a generation speed reference. E3's 5090 leg is speed only, inside a grant.

### E4

- `E4-5090-alone`. No contention measurement: 5090 decode alone versus with an NPU embed load and versus with an iGPU NLI load.
- `E4-npu-embed`. No measurement of NPU embed latency with and without the 5090 busy. E4 uses bge-base so it does not depend on E2.
  - Partial `E1/bge-base-npu-l128-b1/e1-intel-2026-10-09-rerun2` is missing contention.
  - Partial `E1/bge-base-npu-l256-b1/e1-intel-2026-10-09-rerun2` is missing contention.
  - Partial `E1/bge-base-npu-l512-b1/e1-intel-2026-10-09-rerun2` is missing contention.
  - Partial `probe/2026-10-09-probe/BAAI/bge-base-en-v1.5/NPU/b1/embed` is missing contention.
  - Partial `probe/2026-10-09-probe/BAAI/bge-base-en-v1.5/NPU/b8/embed` is missing contention.
- `E4-igpu-nli`. No measurement of iGPU NLI latency with and without the 5090 busy. Batch 8 on the NPU is not this cell.
  - Partial `nli/cross-encoder/nli-deberta-v3-base/grounded/heldout` is missing contention.
  - Partial `nli/cross-encoder/nli-deberta-v3-base/grounded/tune` is missing contention.
  - Partial `nli/cross-encoder/nli-deberta-v3-base/reasoning/heldout` is missing contention.
  - Partial `nli/cross-encoder/nli-deberta-v3-base/reasoning/tune` is missing contention.

## Other open cells

- `quote-rule-2-grounded`. No verify:grounded row is pinned to quote rule 2. Chain rows whose verdicts omit pins.quote_rule are quote rule 1 by the pre-#47 inference, and that source is on the row. A math-ladder row does not fill this cell.
- `quote-rule-2-reasoning`. No verify:reasoning row is pinned to quote rule 2. Chain rows whose verdicts omit pins.quote_rule are quote rule 1 by the pre-#47 inference, and that source is on the row. A grounded row does not fill this cell.
- `quote-rule-2-math-ladder`. No verify:math-ladder row is pinned to quote rule 2. Ladder rows whose verdicts omit pins.quote_rule are quote rule 1 by the pre-#47 inference, and that source is on the row. A chain row does not fill this cell.
- `runpod-snapshot`. No dated RunPod offer or budget snapshot is in the table. offrig gpus and offrig budget --show print text today. switchyard does not parse that text and does not open offrig's database. A version-1 snapshot file can be imported when one is recorded.
- `rerank-any`. The table has no rerank row.
- `qwen3-8b-heldout-grounded`. qwen3:8b verify:grounded rows in this import are the tune split. Held-out is not in the table.
- `qwen3-8b-heldout-reasoning`. qwen3:8b verify:reasoning rows in this import are the tune split. Held-out is not in the table.

## What the table does hold

- embed / CPU: 22
- embed / NPU: 15
- embed / iGPU: 18
- engine / CPU: 8
- engine / NPU: 11
- engine / iGPU: 8
- judge:planted-error / 5090: 1
- nli / CPU: 2
- nli / NPU: 1
- nli / iGPU: 4
- verify:grounded / 5090: 24
- verify:math-ladder / 5090: 312
- verify:reasoning / 5090: 30

## Skipped source rows

6 source row(s) were not imported.
- experiments/npu-probe/results/2026-10-09-probe.json#11: error row, not a routing row
- experiments/npu-probe/results/2026-10-09-mac-chain.json#1: GPU.0 has no Intel full name in this file
- experiments/npu-probe/results/2026-10-09-mac-chain.json#3: GPU.0 has no Intel full name in this file
- experiments/npu-probe/results/2026-10-09-mac-chain.json#4: error row, not a routing row
- experiments/npu-probe/results/2026-10-09-mac-chain.json#5: GPU.0 has no Intel full name in this file
- experiments/npu-probe/results/2026-10-09-mac-chain.json#7: GPU.0 has no Intel full name in this file

## Aborted runs

- Aborted run `e2-2026-10-10b` stopped at `qwen3emb-cpu-l2048-b8 warmup 0`: hang: no result within 60 s; arm inconclusive, device pair poisoned for the run
- Aborted run `e2-b8tail-2026-10-10c` stopped at `qwen3emb-npu-l512-b8 cold`: Compilation failed. Level0 pfnCreate2 result: ZE_RESULT_ERROR_INVALID_NULL_POINTER, code 0x78000007 - pointer argument may not be nullptr . [NPU_VCL] Compiler returned msg: Target buffer size '2048' … (re-recorded as compile_rejected under ruling #36)

- Aborted run `e2-b8tail-2026-10-10d` (no receipt; hard kill): `qwen3emb-npu-l2048-b8` compile_model ran about 70 min, 0 bytes, 35.3 GB working set; inconclusive, had its try (see experiments/e3/run-notes-2026-10-10-11.md)
- Aborted run `e3-2026-10-10b` stopped at `qwen3-8b-npu-l1024 cold`: pipeline sized for l256 reused (MAX_PROMPT_LEN 310 < 958 tokens); fixed in #45
- Aborted run `e3-2026-10-10c` (no receipt; hard kill at 89.2% RAM): every pipeline resident under the interleaved order; replaced by order: blocked (#48, #49)
- Aborted run `e3-2026-10-11` (hard kill after the Phi blocks): genai counters lost as MeanStdPair (fixed in #50); its 9 Phi rows carry no throughput and are evidence only

## Router v0 (2026-10-11)

What `switchyard route` refuses for lack of data, and which experiment would fill it.

- TTFT routing needs rows that record `settings.prefix_caching`. The harness records it from #53 onward. E3 predates that, and none of the 502 committed rows carry the field. Unset means unknown, not a shared cache mode. The importer was not changed in this PR.
- Quote rule 2 has no committed receipt. A verify route that asks for it refuses, and the experiment is `unregistered`.
- No `runpod:` row is in the table. A remote suggestion is a shape the policy can return. It is not a measured receipt.
- Qwen3-8B generate with the 5090 busy and a tok/s floor of 15 refuses. The best Intel conservative bound is about 12.01 tok/s (E3, iGPU, l256, thinking low). The experiment that fills the floor is E3. The 5090 reference (E3-REF) meets 15 and is removed by `--busy` before the floor is applied, so the closest miss is an E3 Intel row.
- Qwen3-Embedding on the NPU at 512 is excluded, not ranked. `E2/qwen3emb-npu-l512-b1` is `native_binary_rejected`. `E2/qwen3emb-npu-l512-b8` is `compile_rejected`. The l128 row that passed (about 598 ms warm) is a different cell and is not one of those codes.
- CPU Phi at l1024, thinking off, has a worst repeat near 7.04 tok/s and a median near 20.4. CPU Phi needs a quiet-window rerun before its bound is trusted. v0 does not trim that outlier. For that cell the iGPU bound is the choice, and the NPU (about 15.13) is the runner-up on the bound, because the CPU bound collapses.
- The Phi re-run harness can keep a hot cache, pass ENABLE_MMAP, cold-copy the weights on the snapshot's own drive, warm until two calls agree, and sample total CPU, the top other processes, used RAM and the disk read rate on the load drive during a slot. That is harness support. It is not a measurement. The E3 Phi cells stay open until a generation receipt lands, and the quiet-window note above stays open.
- There is no 5090 Phi row. The same Phi request without `--busy` still chooses the iGPU and says there is no 5090 row.
- The 8×2048 compile hang has no receipt row. It stays an aborted run above, not a routing candidate.
