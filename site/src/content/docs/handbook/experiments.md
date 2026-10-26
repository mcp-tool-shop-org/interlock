---
title: Experiments and receipts
description: The pre-registered spec, the run lifecycle, and what E1 through E5 decided, including E4b.
sidebar:
  order: 4
---

Every experiment is pre-registered. Its spec under `experiments/<id>/spec.yaml` is committed, with the question, the decision it feeds, the arms, the dataset, the metrics, and the stop rules, before any device time. `interlock run` refuses a spec that was edited after its commit. The spec's blob is checked against that commit before the run (`require_unmodified`) and printed by `--dry-run`. It is not written into the receipt. An E4b receipt records the harness `git_sha` and `clean_tree`. A fill-card receipt records `experiment_spec` and `run_id`, and when preflight ran it also records `git_sha` and `clean_tree`. Any other shared-runner receipt records `experiment_spec` and `run_id` only. By practice, a rule is not changed after results are seen. A dated amendment comes before the run whose results count. A run set aside as void is recorded as such.

## What a receipt holds

A routing row names the task, the model (repo or tag, revision or digest, quantisation, licence), the engine and its version, the device, the settings, and the quality and latency it measured. Quality metrics carry their own `n`, interval, and dataset hash. Latency keeps cold and warm apart. A single run is marked provisional. The row points at the source receipt and the commit it was imported from, or at an experiment spec and a run id.

A run can record a capability instead of a speed. `device_lost` and `hang` carry no timing. `native_binary_rejected` keeps only the compile time, and only arms with that same model, device, and static shape are marked not run. `compile_timeout` is a compile that ran past the ceiling. All of these are provisional and never count as speeds. The routing table also refuses aborted rows. `table/gaps.md` lists the aborted run.

## Run lifecycle

A run ends in one of two states.

A finished run leaves `receipts/<run-id>.jsonl` and, for E4b, its series beside it. A run that stops partway writes that receipt first, with the calls that finished and `aborted: true`, then moves the receipt, the series, the log, and the run marker into `aborted/<run-id>-<sha>/` with a `NOTE.json`. `aborted/` is gitignored. It is not a result: `interlock import` does not read it, and those runs never reach the routing table.

Every run also writes `receipts/<run-id>.run.json` before it touches a device. That marker is gitignored. A run killed from outside still leaves a record that it started. To stop a run cleanly from outside, create `receipts/<run-id>.stop`. The watcher flushes the receipt and exits 77.

`interlock runs list` shows `complete`, `aborted`, and `stray`. `interlock runs quarantine <run-id>` sets a stray series aside by hand. See [Usage](./usage/).

## E1

E1 asks, for each embed model on each Intel device, what a cold load costs in units of that arm's own warm median call. The decision is a break-even count: compile of a warm reload, plus any extra in the first call, divided by the warm median. The keep-resident threshold is 100. The question is whether the router may treat a device, the NPU in particular, as warm-only. The spec is `experiments/e1/spec.yaml`. The receipts are the `e1-intel-2026-10-09*` files.

## E2

E2 asks whether Qwen3-Embedding-0.6B, as the pinned OpenVINO fp16 export, is a candidate embedder worth a retrieval-benchmark slot. It makes no retrieval-quality claim. Two gates have to pass: cosine parity of at least 0.999 against the CPU batch-1 arm at the same length, and a warm median on the NPU or the iGPU at batch 1 and length 512 that is at most 0.9 times the CPU warm median. The spec is `experiments/e2/spec.yaml`.

## E3 and the 5090 reference

E3 asks whether the Intel side can host an always-on small generator, and at what latency. Two channel-wise int4 models, Phi-3.5-mini and Qwen3-8B, run through OpenVINO on the NPU, the iGPU, and the CPU. The viability gate, chosen before any run, is 15 tokens per second or more at a 1024-token prompt with thinking off, on at least one model and one Intel device. The spec is `experiments/e3/spec.yaml`. The run notes are `experiments/e3/run-notes-2026-10-10-11.md`.

The README's summary of those receipts: Phi-3.5-mini decodes at about 20 to 22 tokens/s on the CPU and the iGPU, and about 15 tokens/s on the NPU, at a 1024-token prompt with thinking off. Qwen3-8B at the same prompt decodes at about 10 to 12 tokens/s on the CPU, the iGPU, and the NPU.

E3-REF is the speed reference for that ratio, not a gate of its own. It measures `qwen3:8b` on the 5090 through Ollama at the 1024-token prompt, thinking off and on. The spec is `experiments/e3-ref/spec.yaml`. The README's summary of that receipt: about 208 tokens/s with thinking off and 211 with it on. `interlock route` ranks by the worst repeat, so its choices are more cautious than these summaries.

Some shapes do not run. The NPU rejected Qwen3-Embedding at 512-token inputs, one at a time and in batches of 8. Those rows are capability findings, never ranked as slow candidates.

## E4

E4 asks the contention question that E4b later repeats: while `qwen3:8b` decodes on the 5090, does a sustained NPU embedding load or a sustained iGPU NLI load change that decode rate, and does the 5090 change the NPU or iGPU latency? E4's rule compares paired medians with a band of 0.10, and it is valid only when the alone samples' own spread is at most half the band. The spec is `experiments/e4/spec.yaml`. E4b's rates are not comparable to E4's. The prompt changed.

## E5

E5 asks, for each named local 5090 model, the largest `num_ctx` that stays fully resident and at or under 30500 MiB, including the peak while a long prompt is processed. A window that fails residency, spills, or exceeds that line steps back one 2048-token notch. Time to first token and decode tokens per second are recorded and are not gates. The spec is `experiments/e5/spec.yaml`.

## E4b

E4b asks E4's question again, with a counting prompt shared by every arm and with the triplet as the unit of analysis. The spec is `experiments/e4b/spec.yaml`.

For triplet *i* the effect is the log ratio of the paired slot against the mean of that triplet's own alone-before and alone-after. It is not a chain shared with the next triplet. The estimate is the Hodges-Lehmann estimate of those log ratios, with an exact Wilcoxon signed-rank interval. The interval used is the narrowest exact interval whose confidence is at least 90 percent. The achieved level is recorded on the receipt. A percentile bootstrap is a footnote and never decides.

The pair is free when the whole interval lies inside ±0.10 on the log scale, about 0.905 times to 1.105 times. A consistent speed-up is coupling too, so an interval that lies entirely outside the band on either side is not free. Anything else is inconclusive. An inconclusive pair is not a pass.

The sign depends on the device. For the 5090 the measurement is decode tokens per second, and a slower pair is negative. For the NPU and the iGPU the measurement is warm latency in milliseconds, and a slower pair is positive.

Two committed receipts hold the result. Percent below is 100 times the log ratio, to two decimals.

`receipts/e4b-block-2.jsonl` decided the two 5090-side pairs and left the Intel-side pairs inconclusive. Those two had refusal events and no interval. They are not hidden.

| Pair | Estimate | Interval | Verdict |
| --- | --- | --- | --- |
| 5090-npu | −0.83% | [−1.20, −0.46] | free |
| 5090-igpu | +0.57% | [+0.24, +1.38] | free |
| npu-5090 | none | none | inconclusive |
| igpu-5090 | none | none | inconclusive |

`receipts/e4b-block-3.jsonl` re-ran the two inconclusive pairs. Block 2's Intel-side pairs were refused because the harness started the Intel calls before the 5090 companion was steadily decoding, which the spec already required. The dated amendment, written after block 2 and before block 3, made the harness wait for one steady warm decode. The band, the 22.5 MHz tolerance and the estimator did not change. Both intervals sit inside the band.

| Pair | Estimate | Interval | Verdict |
| --- | --- | --- | --- |
| npu-5090 | −0.61% | [−1.31, +0.06] | free |
| igpu-5090 | +2.48% | [−0.99, +4.82] | free |

On each decided pair the achieved coverage is 0.916015625. The method on the receipt is `exact-wilcoxon-signed-rank`. The band is 0.10.

In plain words: the four decided pairs are free under the rule that was written down before the runs. The router may schedule those pairs as independent lanes on that evidence. The two pairs that block 2 could not decide were run again, and the second run is the one that counts for them. Block 2's inconclusive rows stay in the receipt.

The first live attempt did not leave a committed receipt. The committed blocks are block 2 and block 3.

rnd drew the charts from these receipts. They are up:

- [Verdict](https://mcp-tool-shop-org.github.io/rnd/charts/e4b-verdict.svg)
- [Triplet effects](https://mcp-tool-shop-org.github.io/rnd/charts/e4b-triplet-effects.svg)
- [Pilot clock](https://mcp-tool-shop-org.github.io/rnd/charts/e4b-pilot-clock.svg)
- [iGPU latency](https://mcp-tool-shop-org.github.io/rnd/charts/e4b-igpu-latency.svg)

## C2: remote Colab cards

C2 measures the same questions on rented Colab cards (`colab:A100-40`, `colab:A100-80`, `colab:L4`, `colab:T4`, `colab:G4`) through the Linux CUDA path, `platform: linux-cuda`. The specs are pre-registered like the rig's:

- `experiments/c2-fill-card/`: the largest resident context per model, with 2048-token grid bisection.
- `experiments/c2-generate/`: time to first token and decode rate at prompt lengths 512 to 32768 and concurrencies 1, 4 and 8.
- `experiments/c2-throughput/`: `embed` at batches 1, 8 and 32, with fixed 128-token and 510-token inputs built per model with its own tokenizer at a pinned revision (`experiments/c2-throughput/inputs.json`). `rerank` and `nli` are out of scope: stock Ollama serves neither, so those cells are pre-registered as `load_refused` with that reason.

Every entry names a checked source: an Ollama tag with its registry manifest digest, or a GGUF file with its Hugging Face repo, 40-hex commit and sha256. A cell with no published build, such as Gemma 4 31B at Q6_K, is pre-registered as `load_refused` with the reason `no published Q6_K build`. `interlock c2-sources <spec>` prints each entry's source, pull command and expected digest. `--verify-live` checks them against `registry.ollama.ai` and `huggingface.co`, including each quant label and each throughput tokenizer's digest.

A shape that runs out of memory, is refused at load, times out or exceeds the context is kept as a capability row with its measured reason (`oom`, `load_refused`, `timeout`, `ctx_exceeded`). `load_refused` is only for a recognised load or VRAM refusal. A generation that Ollama aborts (for example on its repeat limit) is `generation_aborted`, and any other unrecognised error is `unmeasured`: both mean the cell was not measured and say nothing about the card.

The generate ladder is at version 2 (`settings.ladder_version`). Its prompt is varied, deterministic text ending in an instruction, so a model does not loop on it. The context window is the prompt plus `num_predict` plus 64 tokens, and the word count is calibrated per cell so the prompt the model evaluates is within 2% of the target. The row records that evaluated length (`settings.prompt_tokens`) and the target (`settings.prompt_tokens_target`). A prompt that cannot be brought to its length is `unmeasured`, never a row with a length it did not time. A Colab ladder row without a version is version 1, and the router never ranks it against version 2. The rig's own harness rows carry no version and route as before. The harness reads residency from `/api/ps`, the card from NVML or `nvidia-smi`, and the engine version from `/api/version` (or records `unrecorded` when the server failed before answering).

A window's receipt reaches the table through `interlock import --remote-window`. Remote rows stay in their own pool and are routed only on request. No remote window has been imported yet.
