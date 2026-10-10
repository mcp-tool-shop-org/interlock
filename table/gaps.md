# Gap report

This is the experiment backlog. A cell is open when the table has no row that fills it.
Imported rows: 456.

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

