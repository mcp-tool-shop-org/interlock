---
name: switchyard-router-buildout-handoff
description: Handoff to Grok (2026-10-09, the Director's ask) to build switchyard, the device-aware model router, experiment-first. A receipt schema and routing table, importers for the studio's existing receipts, a pre-registered experiment harness with device and grant guards, four first experiments (cold/warm start, Qwen3-Embedding-0.6B on the Intel devices, small LLMs on the NPU/iGPU, NPU/iGPU/5090 contention), then an advisory router. Repo mcp-tool-shop-org/switchyard (placeholder name). R&D reviews every PR.
metadata:
  type: project
---

# switchyard: build the device-aware model router, experiment-first

**Name:** `switchyard` (the Director, 2026-10-09). It was first created as the placeholder `np-you`, then `np-u`;
GitHub redirects both. The old placeholder appears nowhere else in this brief, the code or the docs.

**Ordered by the Director, 2026-10-09:** "Let's start building this out, so that we can start running
experiments to get the data we need instead of waiting for it to accumulate." Written by Claude (R&D), who
reviews every PR. The builder is **Grok**. Mike relays messages, and the Publisher holds the devices.

## Before anything else

- **Cast:**
  - **the Director (Mike)** ordered the build and relays every message;
  - **R&D** (Claude, the rnd seat) wrote this brief and reviews every PR;
  - **the Publisher** (Claude, another session) holds the devices (NPU ledger, 5090 grants) and merges PRs;
  - **Kimi** (K3) owns the NPU build-out lane (npu-serve, `npu-buildout`).
- **Precedence:**
  1. Mike's live word;
  2. this brief's hard rules;
  3. the org rules in `E:/AI/.claude/rules/`;
  4. the feedback files below.

  A conflict between this brief and what you observe on the rig (a path, a version, a sha): **stop and ask,
  don't improvise.** The org coverage rule and this brief agree (a floor at the measured value, target 90);
  they aren't in conflict.
- **Stop rule:** if any file this brief names is missing or differs from its description, stop and report it
  through Mike before writing code.
- **Acknowledge before Phase 0,** via Mike, with:
  - the cloned sha;
  - the venv path and Python version;
  - the commit shas you pinned for each import source (see "Pinned sources");
  - your Phase 0 plan, in a few lines.
- **Device requests:** name them "switchyard E1" … "switchyard E4" in ledger and grant messages.
- **Branches:** `grok/phase-<n>-<slug>` (e.g. `grok/phase-0-scaffold`), with one PR per phase against `main`.

## Read first

All paths below are relative to this file's directory (`the studio memory store`). Its
`MEMORY.md` is the index.

1. `device-aware-model-router-direction.md`: what the router is, and the rules it inherits.
2. `intel-npu-side-of-the-rig.md` and the intel-npu skill: the
   device rules, and what's measured so far.
3. `kimi-intel-npu-buildout-handoff.md`: Kimi's lane (npu-serve, the retrieval benchmark, the NLI floor, the
   SigLIP/Whisper probes). **switchyard consumes Kimi's receipts and never redoes Kimi's work.**
4. `Feedback/feedback_thinking_off_or_low_by_default.md`: thinking level is a routed variable, using only the levels a
   model really has.
5. Feedback:
   - `Feedback/feedback_git_add_explicit_paths.md`;
   - `Feedback/feedback_check_ci_after_every_push.md`;
   - `Feedback/feedback_read_exit_before_next_gpu_step.md`;
   - `Feedback/feedback_gpu_use_through_the_publisher.md`.
6. The org rules in `E:/AI/.claude/rules/`: `github-actions.md`, `atlas-map.md`, `coverage-bar.md`,
   `workflow-standards.md`.
   - These four are the ones that bind switchyard. The other files in that directory are for the marketing site,
     shipcheck and publishing, and don't apply yet.

## What switchyard is (and isn't, yet)

A **dispatcher**. Given a task (embed, rerank, NLI, verify, generate) and its constraints (minimum quality,
maximum latency, context length, thinking level), it picks **model, engine, device and settings** from a
routing table **where every row cites a receipt**, and it explains the choice.
- OpenVINO's AUTO/HETERO pick a device for a model you've already chosen; nothing picks the model. That gap
  is the product.
- **Experiment-first:** most of the work is the harness that *produces* routing rows on demand. The router
  is a thin policy over the table and comes last.
- **Not in this handoff:** serving live traffic or proxying requests, training, and changes to offrig,
  npu-serve or rnd. A serving layer needs its own handoff and the Director's go.

## Where things are (verified 2026-10-09)

- **Pinned sources (verified 2026-10-09 ~13:40; re-check before importing):**
  - **rnd main at `961f635`** (rnd 1.1.4.3.26). It holds the calibration chain results
    (`experiments/verifier-gold/calibration/results/2026-10-09-chain/`) **and** the math-ladder results
    (`experiments/verifier-gold/math-ladder/results/2026-10-09-ladder/`: six runs, both report views, the logs).
    Import from these committed copies, not from `E:/AI/rnd-ladder`.
  - **Kimi's `npu-buildout` at `275e511`** (unmerged). Pin that sha, and re-import from the merge commit when
    the PR merges.
- **Repo:** `mcp-tool-shop-org/switchyard`. It's internal, and `main` holds one commit (`a65b12f`, a LICENSE). Clone it to `E:/AI/switchyard`. That directory already exists, empty and not a git repo; that's expected,
  and the clone fills it. The remote is `https://github.com/mcp-tool-shop-org/switchyard.git`. Repo-first is satisfied: the repo exists,
  `main` is the default branch, and origin is correct.
- **Devices:**
  - Intel Core Ultra 9 285K: NPU "Intel AI Boost", the Intel iGPU, and the CPU.
  - RTX 5090 (32 GB), CUDA 13.4.
  - **OpenVINO also lists the 5090 as `GPU.1`. Never use it:** find the iGPU by `FULL_DEVICE_NAME`
    containing "Intel".
- **Engines on the rig:**
  - Ollama on the 5090 at `127.0.0.1:11434`, the CUDA 13.4 system install;
  - CPU-only Ollama at `127.0.0.1:11490`;
  - OpenVINO 2026.4.1;
  - npu-serve, port 11491. It's on Kimi's unmerged branch: **don't start it or depend on it** until Kimi's
    PR merges.
- **Existing receipts to import** (read-only, pinned by commit sha):
  - **rnd** `E:/AI/Research and Development` (public, `mcp-tool-shop-org/rnd`):
    - `experiments/verifier-gold/calibration/results/2026-10-09-chain/`: six models × grounded/reasoning on
      tune, per-claim verdicts with timing; the difficulty tiers are in `tiers.json` / `difficulty.py`;
    - the math ladder: `experiments/verifier-gold/math-ladder/` (`ladder.jsonl`, `report.py`). Runs are in
      `E:/AI/rnd-ladder/ladder-*`, and the results are being committed to rnd today;
    - `experiments/npu-probe/results/*.json`: the NPU/iGPU/CPU engine and embedding receipts.
  - **Kimi's branch** `npu-buildout` (worktree `E:/AI/rnd-npu-buildout`): the NLI floor and phase receipts.
    Import from the committed sha; re-import when it merges.
- **Existing tools to reuse, not rebuild:** the math ladder generator and scorer, offrig's
  `verify calibrate` (the verdict contract; switchyard never re-implements verification), and Kimi's
  `mac_bench.py` / `probe.py` patterns for device timing.

## Hard rules (break one and the PR is sent back)

1. **Devices.**
   - Name every device explicitly (`NPU`, `CPU`, and the iGPU by name). No `AUTO`, no `GPU` without an
     index check. Any 5090 work goes through Ollama on 11434 only.
   - A unit test must prove the device resolver refuses `GPU.1` and any non-Intel GPU.
2. **The NPU is exclusive** and sits on the Publisher's ledger. Before every NPU or iGPU session, send the
   Publisher (via Mike) one line: what, how long, which device. Start when it's granted, and say when you're
   done. Kimi shares the NPU, so a session never overlaps one of Kimi's.
3. **The 5090 runs only inside a Publisher grant.**
   - The runner refuses a 5090 step unless `--grant <text>` is passed, and records the grant in the receipt.
   - Before each model, as `chain_step.sh` in rnd does:
     - check the watchdog heartbeat `E:/AI/training/_watchdog_HEARTBEAT` is ≤ 60 s old;
     - check `/api/ps` is empty;
     - **rest the card 15 minutes between models** (the Director's rule);
     - unload after.
   - **Read each step's exit before starting the next.** Send "card free" when done.
   - CUDA 13.4 only.
4. **CPU-heavy legs** (CPU timing baselines, anything that loads all cores for minutes) run only in a window
   the Publisher announces. Never during someone's timed run.
5. **Pre-register before you run.**
   - Every experiment gets a spec file committed **before** it runs, holding:
     - the question;
     - the decision the result feeds;
     - candidates and settings;
     - the dataset;
     - metrics, repeats and the pass/decision rule;
     - stop rules.
   - Never change a rule after seeing results. A dated amendment is allowed **only before** the run, and
     it's committed. A rule that proves unworkable: stop and ask.
6. **Receipts, not claims.**
   - Every number in a README or the routing table comes from a committed receipt JSON that names the model
     repo/tag, revision or digest, quantization, engine and its version, device, settings, dataset and its
     hash, date and repeat count.
   - Report spread (median, min–max), and cold vs warm separately.
   - Mark single runs "provisional".
7. **Local and free only.** No Ollama Cloud (`*-cloud` / `:cloud`), no OpenRouter, nothing paid.
8. **Downloads.**
   - Only models named in an experiment spec, from their publishers. Pin the revision and record the
     licence.
   - No `trust_remote_code` without pinning and reading the code first.
   - Models go in the HF cache at `E:/AI-Models/hf-cache`.
   - switchyard gets **its own venv**: uv, Python 3.12, `E:/AI/envs/switchyard`, with versions pinned and matching the
     NPU venv (OpenVINO 2026.4.1, …). **Don't touch Kimi's `E:/AI/envs/npu-openvino`.**
9. **Hands off what isn't switchyard:** offrig is the Publisher's, npu-serve and `npu-buildout` are Kimi's, and rnd
   is R&D's. Read their receipts, never edit them. A finding about one of them goes to its owner, through
   Mike.
10. **Git.**
    - Branch and PR, never push to main. Stage **exact paths**: never `git add .` or `git add <dir>`.
    - CI must be green before you call a PR done.
    - **R&D reviews; the Publisher merges.**
    - Even though the repo is internal, run the identity scan before every push:
      `the studio identity-scan script, on the whole tree`
    - No home-directory paths in committed receipts (`E:/AI/...` is fine), and no secrets.
11. **Org standards from the first commit:**
    - `ci.yml`: ubuntu-latest, `push.paths` filter, `pull_request` with no paths filter if it's a required
      check, `concurrency`, `workflow_dispatch`, at most 2 push-triggered workflows;
    - the Atlas map, plus `npx --yes @dogfood-lab/atlas@1.24.0 check` in CI (re-map from a clean worktree when
      it drifts);
    - a coverage floor set to the measured value (the target is 90);
    - every workflow file (the runner, specs) carries a **Standards compliance** section per
      `workflow-standards.md`.

## Phases (each one a PR, reviewed before the next starts)

### Phase 0: scaffold and lockdown (no devices)

- **Package and module `switchyard`,** `pyproject.toml` (uv, pinned), and a CLI `switchyard`.
- **`devices.py`:** the device resolver and allow-list, with tests for refusing `GPU.1`, a non-Intel GPU and
  AUTO.
- **`guards.py`:**
  - the grant guard (`--grant` required for any 5090 step);
  - the NPU-session guard (`--npu-session <text>` required for NPU/iGPU steps);
  - the heartbeat, empty-card and rest checks.

  All of them are tested with fakes.
- **`receipt.py`:** the routing-row schema, with a JSON Schema file plus a validator; `switchyard check` validates
  every receipt in the repo. The fields:
  - task;
  - model (repo or tag, revision or digest, quant, licence);
  - engine and version;
  - device;
  - settings (thinking level, num_predict, num_ctx, batch, static shape, precision);
  - quality metrics, each with its own n, interval and source dataset hash;
  - latency (cold, warm p50/p95), throughput and compile time;
  - date, repeats, provisional flag;
  - provenance (source receipt path + commit sha, or experiment spec + run id).
- **CI**, Atlas, a coverage floor, and a README with the threat model (what switchyard touches: local devices and
  local HTTP; never the network beyond the HF download).
- **The logo (the Director's, 2026-10-09):** copy it into the repo in this PR. The Downloads folder isn't
  durable.
  - Master: the Downloads copy `Qwen-Image-2512_00001_ (7).png` (1328×1328, on a white margin) →
    `assets/logo/switchyard-master.png`.
  - The icon set, the Downloads copies `switchyard_{16,32,64,128,256,512}.png` (cropped to the dark tile) →
    `assets/logo/switchyard_<n>.png`.
  - The README header shows `assets/logo/switchyard_512.png` at width 256.
  - **Not** the public brand repo yet: switchyard is internal, and `mcp-tool-shop-org/brand` is public.
    Moving the logo to `brand/logos/switchyard/readme.png` (the org convention) happens when the Director
    makes the repo public.
- **A snapshot of this brief:** add `docs/handoff/` with this brief and the four feedback files from "Read
  first". Rewrite absolute home-directory paths as plain references, so no home-directory path is
  committed. Note the date and the memory path they came from. The brief stays authoritative in memory, and
  the snapshot makes the repo self-contained. Run the identity scan before pushing.
- **Accept:** the tests pass in CI, `switchyard check` is green on an empty table, and the guards refuse in tests.

### Phase 1: the routing table from existing receipts (no devices)

- **Importers,** one per source, each pinned to a commit sha:
  - **offrig calibration runs:** one row per model × check type × split. Quality = the default-rule metrics
    with their Wilson intervals, plus difficulty-tier rows. Latency comes from the per-claim `wall_seconds`.
  - **math-ladder runs:** one row per model × cell, on both views (final verdict, and model verdict labelled
    post-hoc).
  - **npu-probe results:** device speed and parity rows.
  - **Kimi's NLI floor:** on the committed sha.
- **Output:** `table/routing.jsonl` plus `switchyard table` (a readable view filtered by task or device), and a gap
  report: which task × device × candidate cells have no row.
  - **The gap report is the experiment backlog.** It should show why E1–E4 below exist.
- **Accept:** every imported number traces to its source file and sha; a test re-derives five rows from the
  raw files.

### Phase 2: the experiment harness (no devices until Phase 3)

- **Spec format,** YAML or JSON in `experiments/<id>/spec.*`: the pre-registration fields from rule 5, plus
  the device and grant needs.
- **Runner:**
  - `switchyard run <spec> [--grant …] [--npu-session …] [--dry-run]`. It validates the spec, checks the guards,
    and refuses to start if the spec's committed hash doesn't match.
  - **Timing method:** warm-ups, then N repeats, interleaving devices, with cold start measured separately.
  - It writes receipts and imports them into the table.
- **Adapters:**
  - Ollama HTTP (11434, 11490);
  - OpenVINO direct (embeddings via `openvino` + tokenizer, and LLM generation via `openvino-genai`
    `LLMPipeline` on NPU/GPU(Intel)/CPU).
  - An npu-serve adapter only after Kimi's PR merges.
- **Environment capture** in every receipt: driver and NPU firmware versions, OpenVINO and package versions,
  power plan, and what else was loaded (Ollama `/api/ps`).
- **Write and commit the pre-registered specs for E1–E4 in this phase.** Experiment design is part of Phase 2,
  not Phase 3: the specs are the pre-registrations, and R&D reviews them in the Phase 2 PR before any device
  time is asked for.
- **Accept:**
  - a `--dry-run` of each Phase 3 spec passes;
  - fake-adapter tests cover the timing math, repeat interleaving and the receipt shape;
  - a test proves a spec edited after commit is refused.

### Phase 3: the first experiments (one PR each; pre-register, then run in the right window)

Each one feeds a router decision. None makes a quality claim it didn't measure.

- **E1, cold vs warm, per device.** For the embedders already measured (nomic-embed-text v1.5 ONNX, bge-base),
  on NPU, iGPU and CPU: model load and compile time, first-call latency, then warm p50/p95 at batch 1 and 8
  (batch 8 is **not** run on the NPU for cross-encoders: it hung the NPU), at 3 input lengths.
  - **Decision:** whether the router may treat the NPU as warm-only (that is, keep a model resident).
  - **Windows:** an NPU session, plus a short CPU slot.
- **E2, Qwen3-Embedding-0.6B on the Intel devices.** OpenVINO 2026.0 added NPU support for it.
  - Pin the revision and check the licence first.
  - Speed on NPU, iGPU and CPU, plus parity against its own CPU reference (minimum cosine, by length).
  - **No retrieval-quality claim yet:** quality comes from Kimi's pre-registered retrieval benchmark, once
    merged. Spec E2 so it can feed that benchmark later.
  - **Decision:** whether it's a candidate embedder worth a benchmark slot.
- **E3, small LLM generation on the Intel side.**
  - **Models:** `OpenVINO/Qwen3-8B-int4-cw-ov` and one smaller model from Intel's "LLMs optimized for NPU"
    collection (`gemma-3-4b-it-int4-cw-ov` or `Phi-3.5-mini-instruct-int4-cw-ov`; choose in the spec, licence
    checked). Use channel-wise builds only: group-quant IRs reportedly crash the NPU compiler.
  - **Measure on NPU, iGPU and CPU** through `openvino-genai`: compile time, time to first token, and decode
    tokens/s at 3 prompt lengths (about 256, 1k and 4k tokens), with thinking off and on where the model has
    it.
  - **Reference:** the same family on the 5090 via Ollama (`qwen3:8b`) for speed only, **inside a grant**.
  - Capture outputs. No quality claim.
  - **Expectation to test:** a community run on this CPU saw NPU decoding slower than the iGPU and CPU for a
    3B model.
  - **Decision:** whether the Intel side can host an always-on small generator, and at what latency.
- **E4, contention (the router's co-scheduling question).**
  - 5090 decode tokens/s for `qwen3:8b` via Ollama **alone**, versus with a sustained NPU embedding load,
    versus with an iGPU NLI load.
  - The reverse too: NPU/iGPU latency with and without the 5090 busy. Record CPU and package power if
    readable.
  - **Short:** one grant, one NPU session, booked together with the Publisher.
  - **Decision:** whether the router can schedule the Intel devices freely while the 5090 works.

### Phase 4: router v0, advisory only

- `switchyard route --task <t> [--min-<metric> x] [--max-latency-ms y] [--ctx n] [--thinking off|on|…]` returns the
  chosen model, engine, device and settings, with:
  - **the receipts it relied on** (row ids and their sources);
  - **the margin:** quality is judged on the interval's conservative bound, never the point estimate;
  - **the runners-up,** and why they lost.
- **The policy** is explicit and tested:
  - meet every hard constraint on the conservative bound;
  - then the lowest warm latency;
  - ties go to a device that leaves the 5090 free;
  - stale or provisional rows are flagged.
  - **No row meets the constraints → it refuses loudly, naming the gap and the experiment that would fill
    it.** It never falls back silently to a weaker option.
- Device availability is an input (e.g. `--busy 5090`); switchyard doesn't probe the ledger itself.
- **Accept:**
  - property tests of the policy (monotonicity: tightening a constraint never yields a strictly better
    choice);
  - golden tests on the Phase 1 table.

## Out of scope (ask first)

Serving or proxying live requests, any fine-tuning or distillation (the distillation exercise is R&D's and
ASPIRE's, with its own pre-registration), changes to offrig, npu-serve, rnd or Kimi's branch, registering
services or scheduled tasks, driver or firmware updates, and anything paid.

## Hand-back (per PR, in the PR description, which Mike brings to R&D)

1. What was done; every number with its receipt path.
2. Every deviation from this brief or from a spec, and why.
3. The device windows used (NPU sessions, grants), for the Publisher's ledger.
4. What's still unmeasured: the updated gap report.

## Verification by R&D

R&D reviews every PR:
- the guards' tests;
- the importers re-derived against the raw files;
- each experiment's spec committed **before** its receipts (checked with git history);
- receipts checked against the spec;
- the router policy checked against its tests.

## Standards compliance (this handoff)

| Standard | Score | Evidence |
|---|---|---|
| PIN_PER_STEP | 2 | Every receipt pins model revision or digest, engine version, device, settings and dataset hash. The runner refuses a spec whose committed hash changed. |
| ANDON_AUTHORITY | 2 | The guards refuse a 5090 step without a grant, an NPU step without a session, a stale heartbeat or a loaded card. A non-zero step exit stops the sequence. Unworkable rules stop and ask. |
| NAMED_COMPENSATORS | 2 | Table below; nothing irreversible is in scope. |
| DECOMPOSE_BY_SECRETS | 2 | devices / guards / receipt schema / importers / adapters / harness / router are separate modules. switchyard reads other repos' receipts, never their code paths. |
| UNCERTAINTY_GATED_HUMANS | 2 | Device windows go through the Publisher. Each experiment's decision rule is pre-registered. The router refuses rather than guess. Serving is gated on the Director. |
| EXTERNAL_VERIFIER | 2 | R&D (Claude, a different family from Grok) reviews every PR against receipts. |

### Compensators

| Action | Undo | State after | Owner |
|---|---|---|---|
| Push a branch / open a PR | close the PR, delete the branch | main unchanged | Grok (R&D reviews) |
| Model downloads to the HF cache | delete the `models--…` folders | cache as before | Grok |
| Create the `E:/AI/envs/switchyard` venv | delete the directory | as before | Grok |
| A model loaded on the 5090 | `keep_alive: 0` unload, then confirm `/api/ps` is empty, then "card free" | card free | Grok, inside the grant |
| An NPU/iGPU session | end the process, confirm the device is idle, tell the Publisher | NPU free | Grok |

## Log

- 2026-10-09: written by R&D at the Director's ask; Mike relays to Grok. E2 and E3 take over the
  OpenVINO-model trials that were queued for Kimi's next handoff (see `kimi-intel-npu-buildout-handoff.md`).
- 2026-10-09 ~13:30, the Publisher's slotting terms for Phase 3:
  - E1–E3 use NPU ledger slots whenever the NPU is free.
  - E1's CPU slot and E3's qwen3:8b reference share the CPU window or a free card gap.
  - **E4 gets its own booked block:** a grant plus an NPU session together, never next to a timed calibration
    or ladder run.
  - PR terms as rule 10/11: Atlas plus `atlas check` from the first workflow, a coverage floor at the measured
    value, and the identity scan before every push. The Publisher merges after R&D's review.
- 2026-10-09 ~13:45: revised after Kimi's consult review (D1–D9). The reading list now uses real file paths
  (no `[[…]]` slugs). Added: cast, precedence, the stop rule, the acknowledgement, branch naming, pinned
  sources with a ladder pre-flight, a note that the empty `E:/AI/switchyard` is expected, a snapshot of the brief
  into `docs/handoff/` in Phase 0, and the E1–E4 specs moved explicitly into Phase 2.
