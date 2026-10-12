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
  don't improvise.** On coverage, the Director's word (90% before the prototype release) is the bar. The org
  rule's "ratchet from the measured value" applies to existing repos; switchyard is new and is built to 90 from
  the start.
- **Stop rule:** if any file this brief names is missing or differs from its description, stop and report it
  through Mike before writing code.
- **Acknowledge before Phase 0,** via Mike, with:
  - the cloned sha;
  - the venv path and Python version;
  - the commit shas you pinned for each import source (see "Pinned sources");
  - your Phase 0 plan, in a few lines.
- **Device requests:** name them "switchyard E1" … "switchyard E5" in ledger and grant messages.
- **Kimi owns the Intel lane** (`kimi-switchyard-intel-lane-handoff.md`, same directory):
  - `src/switchyard/adapters/openvino.py`, the E1–E3 specs and running E1–E3 are Kimi's; the rest is yours;
  - Kimi second-reviews `devices.py` and your Phase 2 adapter interface;
  - one owner per file, so neither of you edits the other's;
  - define the adapter interface in Phase 2 so an OpenVINO adapter can plug in without touching the harness
    core. Kimi's build starts after that merges.
  - E4 and E5 stay yours, as does the 5090 `qwen3:8b` reference for E3.
- **Branches:** `grok/phase-<n>-<slug>` (e.g. `grok/phase-0-scaffold`), with one PR per phase against `main`.

## Read first

All paths below are relative to this file's directory (the studio memory store). Its
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
  - **rnd main at `08ca419`** (R&D's answer to Grok, 2026-10-09; the chain and ladder data trees are byte-identical to `961f635`, and the two commits in between only add gemma-rerun pre-registration text). It holds the calibration chain results
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
  - RTX 5090 (32 GB), CUDA 13.4. (OpenVINO names, verified live by Kimi: `Intel(R) AI Boost` (NPU, vendor 3720), `Intel(R) Graphics (iGPU)` GPU.0 (vendor 0x8086), `NVIDIA GeForce RTX 5090 (dGPU)` GPU.1 (vendor 0x10de).)
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
      `python ~/.grok/bin/identity-scan.py .`
    - No home-directory paths in committed receipts (`E:/AI/...` is fine), and no secrets.
11. **Org standards from the first commit:**
    - `ci.yml`: ubuntu-latest, `push.paths` filter, `pull_request` with no paths filter if it's a required
      check, `concurrency`, `workflow_dispatch`, at most 2 push-triggered workflows;
    - the Atlas map, plus `npx --yes @dogfood-lab/atlas@1.24.0 check` in CI (re-map from a clean worktree when
      it drifts);
    - **coverage at 90% (the Director, 2026-10-09):** CI runs `coverage report --fail-under=90`, and
      `codecov.yml` sets `project` and `patch` targets of 90% with a 1% project threshold, as rnd does.
      - Upload to Codecov over OIDC with no stored token, with `fail_ci_if_error: false`, so Codecov's
        handling of a non-public repo can't break CI. The `--fail-under=90` floor is the gate.
      - Phase 0 starts with the floor at 90. Code arrives with its tests, so it never drops below.
    - every workflow file (the runner, specs) carries a **Standards compliance** section per
      `workflow-standards.md`.

## The tie-in with offrig (the Director, 2026-10-09: "make sure that switchyard ties into offrig")

offrig is the Publisher's: rented RunPod GPUs, the verifier and `verify calibrate`, the indexer, project
memory, and the spending caps. switchyard and offrig meet at **two contracts**, and neither reaches into the
other's internals. Pin offrig at **main `751fb1d`**, the build the Publisher installs today.

**1. offrig → switchyard (what switchyard reads).**
- **Calibration runs:** the `verify calibrate` run directories (`manifest.json`, `metrics.json`,
  `verdicts.jsonl`) are a first-class importer (Phase 1).
  - Key each row by offrig version, engine, model digest, think, `num_predict`, `quote_rule` and split.
  - A verify row from quote rule 1 and one from rule 2 are **different rows**, never merged.
- **RunPod offers and budgets:** these make "remote" a device class in the table. A row like
  `runpod:<gpu_type>` holds $/hr, VRAM, boot and pull time, and measured per-claim time from pod runs.
  - **offrig's CLI has no JSON output today** (`offrig gpus`, `offrig budget --show` print text; only the MCP
    side-car's `offrig_status` returns JSON). The Publisher is adding it (2026-10-09):
    - **`offrig budget --show --json`** comes in the account-budget PR in progress. It's a versioned object,
      `"version": 1`, with `null` for unknowns:
      - per provider: `cap`, `cap_source`, `committed`, `spent`, `remaining`, `balance`, `warning`;
      - the overall ceiling;
      - the account-wide totals across projects.
    - **`offrig gpus --json`** comes in a small follow-up PR.
    - **Pin the importers to `"version": 1`,** and refuse any other version loudly.
  - Until those land, switchyard reads **recorded snapshots** (a fixture file with its date), and the rows
    are marked `price_snapshot: <date>`.
  - **Never parse offrig's text output in production code, and never open offrig's SQLite store:** its
    schema is private and migrates.
- **The embed endpoint:** offrig's indexer embeds through `embed_url` / `OFFRIG_EMBED_URL` (default CPU Ollama
  11490). Which endpoint (11490, or npu-serve on the NPU) is a routing decision switchyard makes from Kimi's
  benchmark receipts.

**2. switchyard → offrig (what offrig can consume).**
- **A versioned decision contract:** `switchyard route --json` emits a document valid against
  `schemas/route-decision.v1.json`. It holds:
  - task, model, engine, device or endpoint, and settings (think, `num_predict`, `quote_rule`, batch);
  - the receipts relied on, and the runners-up;
  - for a **remote** route, an **offrig plan suggestion** with exactly the fields `offrig_plan` takes
    (profile, gpu count and types, max hours, max price/hr) and the worst-case dollars.
    - Verify those field names against offrig's MCP tool schema at `751fb1d`, and cite the source lines.
- **switchyard never spends.** It never calls `offrig_plan`, `offrig_launch` or any other offrig tool that
  rents, and never sets a cap. A remote suggestion is handed to a person or agent, who plans it through offrig,
  whose caps (per provider, offrig#51) stay the only spend guard. A remote option whose worst case exceeds that
  provider's remaining budget (read-only) is dropped, and the reason is shown.
- **Calibrate what is served:** a `verify` route must cite an offrig calibration receipt for the **same**
  model digest, engine, think, `num_predict` and quote rule. With no such receipt, the router refuses and names
  the calibration that would fill the gap (the rule in offrig#50).
- **A design-note draft for the Publisher,** `docs/offrig-integration.md` in switchyard. It covers how offrig
  could consume the contract:
  - (a) `verify`'s default per engine;
  - (b) the indexer's `embed_url`;
  - (c) local vs pod vs serverless for large models.

  It's a draft. Wiring offrig to call switchyard is the Publisher's decision and PR, never switchyard's.

**3. offrig as an experiment adapter.** The harness may run `offrig verify calibrate` as an adapter (Phase 2),
under the same grant and rest guards.
- Use **scratch projects only** (`--project E:/AI/switchyard-scratch/<exp>`), never R&D's or any real project.
  The new binary migrates a store on first open.
- Local Ollama only. Pod-backed calibration waits for offrig's OpenAI-compatible backend.

## Design draft: the self-repairing runner (the Director's idea, 2026-10-09; not a build order yet)

**The idea (Mike):** a runner that manages runs as needed, so errors become prompts to the runner to repair,
without affecting the tests or training. The shape below is the Publisher's proposal as refined by R&D. It's
**for review by Kimi** (relayed by Mike) and is built only after Phases 0–2 and only on the Director's go.

**The principle:** a repair may change *how* a run gets done, never *what* is measured. Code enforces that
with an allowlist; it is never the runner's judgement. An LLM agent may *propose* in an escalation, but it
can't run anything off the allowlist. Its proposal is untrusted text, like any model output.

**1. The identity fingerprint, checked before and after every repair.** It holds:
- the offrig binary sha256;
- the Ollama version;
- the model digest;
- the run's full settings (think, num_predict, resolved num_ctx, quote rule, temperature, seed, structured);
- the gold or dataset hash;
- the grant id.

- **The 5090 Ollama's backend:** the system Ollama on 11434 runs a hand-installed CUDA 13.4 backend, started with a
  specific environment (`OLLAMA_HOST`, `OLLAMA_MODELS` on E:, logging to
  `E:/AI/ollama-cu134/system-11434.log`). The fingerprint records:
  - that the loaded backend is `lib/ollama/cuda_v13`;
  - the 13.4 ggml-cuda library's sha (`6d3128e9…`; take the full value from the Publisher's record, not from
    this note);
  - the start line.

If any of them differs after a repair, the run parks and escalates. That catches the subtle cases: an Ollama
that auto-updated when it restarted, or a reloaded model whose tag now points at a new digest.

**2. Automatic repairs (allowlist), each one fingerprint-checked:**
- **Resume after a transport, 5xx or dropped-reply failure.** offrig's own repeat-and-health-check rule
  (#46/#49) decides when a repeat counts against the model. The runner only resumes.
- **Unload, then reload a stuck model:** same tag, and the digest must match.
- **Re-check the card before continuing:** heartbeat ≤ 60 s, `/api/ps` empty or only this run's model, and the
  rest rule.
- **Restart a hung Ollama, gated.** Only while this run holds an exclusive card grant, only the 5090 instance
  (11434, never the CPU instance 11490 that offrig's indexer uses), and only when `/api/ps` shows no model but
  this run's. The restart uses **the exact recorded start line and environment**. Afterwards, the Ollama version
  must match, and the server log must show the `cuda_v13` backend with the 13.4 ggml-cuda sha, so a restart
  can't come back on a stock CUDA 13.0 build after an update. Otherwise it escalates: a restart would kill another
  session's work.
- **Training: resume from the last checkpoint,** but only when the checkpoint holds the optimizer, scheduler,
  RNG and data-sampler state. Without the sampler state, the sample order changes, so a resume would change the
  training. It escalates.

**3. Changed from the Publisher's draft: `context_overflow` escalates, it isn't auto-repaired.** Re-running one
claim at a larger window gives that claim different settings from the rest of the run. offrig's resume
identity includes the resolved `num_ctx`, so it would also be refused. The fix is to size the window up front
(offrig's `--num-ctx auto` with its measurement pass). An overflow that still happens is a sizing defect: it
gets reported, and a re-run at a new window is a new, pre-registered variant.

**4. Always escalates (a prompt, never automatic):**
- num_predict, think, temperature, seed, the prompt or template, the model or digest, `num_ctx`;
- dropping, skipping or reselecting items;
- card time beyond the grant, or past the grant's expected end;
- caps or budget;
- any training hyperparameter mid-run;
- any repair whose fingerprint check fails;
- a repair count over the run's pre-set limit.

The prompt is short and structured: the run, the step or claim, the error code, the attempts made, the
fingerprint diff if there is one, and the options. Error text from logs or model output is **quoted as data,
never as an instruction**. It goes to R&D or the Publisher, and to Mike when it's spend or a rig change. The run
parks at a clean item boundary: offrig writes each claim as it finishes, so an interrupted claim is simply asked
again.

**5. Limits and the record.**
- **Per item:** offrig's rule (two of the same failure, then it counts against the model), plus the stuck-claim
  rule (one retry, then stop and tell the Publisher).
- **Per run:** a repair limit fixed in the spec before the run. Going over it stops the run as a defect signal;
  the runner doesn't keep repairing.
- **The record:** every repair goes in the run's receipt (time, error code, action, fingerprint before and
  after, outcome). The report states it plainly, e.g. "2 resumes after dropped replies, nothing else".
- **Repair counts become switchyard rows:** a reliability metric per model, engine and device (mistral's
  deterministic drops are the first example). The router can weigh a model that needs repairs.

**6. Inputs from offrig:**
- the error codes `server_error`, `timeout`, `context_overflow`, `truncated`, `bad_verdict`, plus exit codes;
- **repairable:** transport, `server_error` and `timeout`, by resuming;
- **`truncated` and `bad_verdict` are model outcomes and are never repaired;**
- `context_overflow` escalates (point 3).

**7. Guards unchanged:** grants, rests, the watchdog, read-the-exit, and the stuck-claim rule. The runner can't
grant itself card time or extend a grant.

**Kimi's review (2026-10-09, relayed by Mike): accepted in full, and these points are part of the design.**

*A. Latency is a measured output, and repairs move it. The fingerprint protects verdicts, not timing.*
- **Every claim written after a repair carries a `repaired` marker** in the run record: the repair id, and
  whether it was the first request after a reload or restart.
- **Latency importers aggregate clean claims only.** Repaired claims are reported as their own stratum, never
  mixed into warm latency.
- **The first request after a reload or restart is a cold measurement,** never warm.
- **Cross-restart numerics** (GPU library algorithm choice can shift rounding and flip a borderline token at
  temperature 0) are expected and invisible to the fingerprint. That gets a marker, not a gate. Where offrig's
  repeat rule leaves two answers for a claim that spans a repair, a post-hoc pass reports whether they agree.

*B. The fingerprint is checked on a cadence,* every N claims or M minutes (fixed in the spec), not only around
repairs, so drift with no repair (an auto-updating Ollama) is caught. A difference parks and escalates as
before.

*C. Restart gates on a shared rig, added to point 2's "Restart a hung Ollama":*
- **`/api/ps` sees only this server.** So, immediately before a restart:
  - re-read the **grant ledger itself** (not the runner's memory), with lease expiry and a fencing token;
  - check the device: `nvidia-smi --query-compute-apps` listing any PID on the 5090 that isn't this run's
    Ollama means refuse and escalate.
- **TOCTOU:** re-check `/api/ps` immediately before stopping. The fencing token bounds and exposes a lost race.
- **Freshness:** the `cuda_v13` and ggml-cuda sha log line must **postdate** the restart, by timestamp or file
  write time. The new process's start time must be after the stop: an answering port isn't proof, since a
  failed stop would leave the old process up.
- **A failed stop or start is never retried automatically.** The cap on restarts per run is 1, then escalate.
  A half-failed restart takes 11434 down for everyone.
- **Dependency:** the grant ledger must be machine-readable (grant id, holder, device, lease expiry, fencing
  token). Today grants are messages. **The Publisher is building it** as a `publisher grant` command that
  writes one JSON ledger for the 5090 and NPU lines (id, holder, device, lease expiry, fencing token, granted
  and freed times), after its account-budget and adaptive-ctx PRs land. Until it exists, an automatic restart
  isn't allowed; restarts escalate. switchyard's grant guard (`--grant`) reads that ledger once it exists.

*D. Training resume is exact only with the full state. Otherwise it escalates:*
- **All four RNG streams:** python `random`, numpy, torch CPU, and torch CUDA (every device). Augmentation often
  uses numpy.
- **Position counters:** global step, epoch, samples consumed, and the gradient-accumulation micro-step offset.
  Check per trainer how the sampler interacts with epoch rebuilds.
- **The AMP GradScaler state** (scale and growth tracker) when mixed precision is on.
- **Non-optimizer model state:** EMA weights and any tracked auxiliary tensors.
- **Data-pipeline determinism:** worker seeds and persistent-worker state, or a stated rule that the loader is
  single-process and deterministic given the sampler state and epoch.
- **For training runs, the fingerprint adds** the trainer's code commit sha and the determinism flags
  (deterministic algorithms, cuDNN benchmark), which are settings, the way think and seed are for inference.

**8. Testing:**
- fault injection with fakes for every allowlisted repair path;
- a test that each non-allowlisted change escalates;
- a fingerprint-drift test (an Ollama version or model digest that changes across a restart parks the run);
- a test that the repair limit stops the run;
- a test that error text containing an instruction is quoted in the escalation, not acted on.
- tests for the `repaired` and cold markers, and for latency aggregation excluding repaired claims;
- a cadence fingerprint test (drift between claims with no repair parks the run);
- restart refusal on a foreign PID in `nvidia-smi`, on an expired or stale lease, on a stale backend log line,
  and on an old process still alive; no automatic retry of a failed restart;
- a training-resume refusal for each missing piece of state (each RNG stream, counters, scaler, EMA, code sha).

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
- **CI**, Atlas, the 90% coverage floor and Codecov config, and a README with the threat model (what switchyard touches: local devices and
  local HTTP; never the network beyond the HF download).
- **The logo (the Director's, 2026-10-09):** copy it into the repo in this PR. **Superseded by a regeneration:**
  Mike is having it redone (a transparent background, a separate wordmark version, "Cloud" replacing the
  duplicate GPU chip). When the new files land, replace `assets/logo/`, and open the brand-repo PR **with the
  new version**, not the first draft. The Downloads folder isn't
  durable.
  - Master: the Downloads master PNG (1328×1328, on a white margin) →
    `assets/logo/interlock-master.png`.
  - The icon set `switchyard_{16,32,64,128,256,512}.png` (cropped to the dark tile) →
    `assets/logo/interlock_<n>.png`.
  - **Also the brand repo (org convention), approved by the Director:** switchyard goes public soon. Open a
    PR to `mcp-tool-shop-org/brand` adding `logos/switchyard/readme.png` (the 512 icon). Its pre-commit hook
    regenerates `manifest.json`; commit both together, per brand's README "Adding a Logo Manually". The
    Publisher merges.
  - The switchyard README header uses
    `https://raw.githubusercontent.com/mcp-tool-shop-org/brand/main/logos/switchyard/readme.png` at width 256,
    as rnd does. Until the brand PR merges, it uses the in-repo `assets/logo/interlock_512.png`.
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
- **offrig (see "The tie-in with offrig"):** calibration-run rows keyed by quote rule and settings, plus dated
  RunPod offer and budget snapshots as `runpod:<gpu>` rows.
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
  - An `offrig verify calibrate` adapter (scratch projects, guards apply).
- **Environment capture** in every receipt: driver and NPU firmware versions, OpenVINO and package versions,
  power plan, and what else was loaded (Ollama `/api/ps`).
- **Write and commit the pre-registered specs for E4 and E5 in this phase.** Experiment design is part of Phase
  2, not Phase 3: the specs are the pre-registrations, and R&D reviews them in the Phase 2 PR before any device
  time is asked for. **E1–E3 specs are Kimi's** (`kimi-switchyard-intel-lane-handoff.md`), written after this
  phase's adapter interface merges.
- **The adapter interface** is defined here, so Kimi's OpenVINO adapter plugs in without touching the harness
  core.
- **Accept:**
  - a `--dry-run` of the E4 and E5 specs passes (E1–E3 follow in Kimi's PRs);
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

- **E5, "fill the card" context sizing (the Director, 2026-10-09).** For each local model on the 5090: the largest
  `num_ctx` that keeps it **fully resident** on the GPU and **under the watchdog ceiling** (31.2 GB card total).
  - **Who uses it:** offrig's `--num-ctx auto` takes it as its ceiling ("fill the card, then contain").
    switchyard emits the row, and offrig reads it later, via the Publisher's PR.
  - **The row:** `max_resident_ctx` and the VRAM at that size, keyed by:
    - model digest and quantization;
    - the Ollama version and CUDA backend (`cuda_v13`, the ggml-cuda sha);
    - the server environment that changes memory (flash attention, KV cache type);
    - the **idle baseline VRAM** measured before the load.

    The baseline matters: the desktop alone held about 2.5 GB at the 14:25 card check, and the result only
    holds with that baseline.
  - **Pre-register these** in the spec:
    - **The "fully resident" test:** `/api/ps` `size_vram == size`, **and** the server log for that load shows
      every layer offloaded to the GPU (no CPU layers).
    - **The search:** to keep loads few (each `num_ctx` change reloads the model), double from the model's
      default context until a step fails residency or the margin, then bisect at 2048-token granularity between
      the last pass and the first fail. Stepping linearly by 2048 from the start would cost dozens of loads per
      model.
    - **The stop rule, with a margin under the ceiling, never testing the ceiling itself:**
      - stop at the first step where card total VRAM (from `nvidia-smi`) exceeds **31.2 GB minus a fixed
        margin** (e.g. 0.7 GB, fixed in the spec);
      - or where residency fails, or the load errors.

      **The watchdog does not protect E5** (the Publisher, from `E:/AI/training/_watchdog.ps1`):
      - it samples every 2 s and kills only after 3 breaches in a row (VRAM ≥ 31200 MiB, ≥ 87 °C, RAM ≥ 90%);
      - it kills only *guarded* processes (listed venv interpreters, `E:/AI/llama.cpp/llama-server.exe`, a WSL
        pattern). **Ollama's bundled runner is not guarded**, so an Ollama breach kills nothing, writes
        `_watchdog_NO_TARGET`, and spills into WDDM paging.

      So **E5's own sampler is the guard:** abort the step and unload the model (`keep_alive: 0`) the moment
      card VRAM reaches the margin, a peak of **≤ 30500 MiB** (0.7 GB under the ceiling). It must never rely on
      the watchdog. Adding Ollama's runner to the watchdog is a rig-wide change, and the Publisher is raising it
      with Mike separately.
    - **The peak is measured, not assumed.** Load-only, with `num_predict` 1, allocates the KV cache, but the
      compute buffers peak while a long prompt is being processed. So the candidate max gets a **confirmation
      step**:
      - a prompt filling about 90% of the window, with `num_predict` 1;
      - `nvidia-smi` sampled every ~200 ms throughout;
      - the reported VRAM is that peak, and it must stay under the margin.

      If it doesn't, step back one 2048 notch and confirm again.
    - **Spill detector (Kimi):** 200 ms sampling can miss short spikes, and WDDM paging can keep the reported
      VRAM under the margin while the model quietly pages, which shows up as latency, not VRAM. So the
      confirmation step also records time to first token and decode tokens/s, against the same model at a
      clearly resident smaller window. A "resident but slow" window fails, and the ceiling steps back.
    - **The margin is an assumption to validate, not a measured value yet.** The first E5 run logs the
      desktop/WDDM VRAM variance during loads. The 0.7 GB margin is re-derived from that, and the spec says so.
  - **Card:** a short grant per model (loads, plus one confirmation prompt each), booked like E1–E4, with the
    rest rule between models.
  - **Staleness:** a row is invalid when the Ollama version, the backend sha, the digest, the server
    environment, the idle baseline **or the NVIDIA driver version** changes (Kimi: drivers move allocations). The router flags it stale, and E5 is rerun for that model.
  - **Context the brief records (from the Publisher):**
    - Ollama 0.35.1 accepts request-level `shift:false` and `truncate:false`. offrig's adaptive-ctx PR sends
      both, so a full window is a clean error, never a silent shift.
    - The runner's "continue at a bigger window" (reload, then resume the same reply from the exact tokens,
      which needs raw mode for thinking models) stays **escalate-only** until a pre-registered equivalence test
      shows resumed replies match uninterrupted ones **token for token** at temperature 0 and a fixed seed.
      "Same verdict" is too loose, since a judge can agree by luck. The test covers thinking models in raw mode
      explicitly, because template reconstruction is where implementations diverge (Kimi).
      The Publisher's caution: llama.cpp evaluates a resumed prefix in batches, which can differ numerically
      from token-by-token decoding of the same tokens, so the test may fail with nothing wrong. A failure keeps
      the feature escalate-only, which is the safe default. Never loosen the test to make it pass.
    - Paged-KV engines (vLLM, SGLang) grow context natively. That's a routing-table column (`kv_paging: true`),
      not something E5 measures.
    - 8-bit KV quantization roughly doubles the fit but changes outputs, so it's a calibrated variant (its own
      rows and its own calibration), never a silent setting.

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

- **The offrig contract:** `schemas/route-decision.v1.json`, `switchyard route --json`, and the
  `docs/offrig-integration.md` draft.
  - **Accept:** every golden decision validates against the schema; a remote decision's plan suggestion
    validates against `offrig_plan`'s input schema at the pinned offrig sha; and a verify route with no
    matching calibration receipt refuses.

### Phase 5: the prototype release gate (the Director, 2026-10-09)

The Director's plan: build it, hold coverage at 90%, release it as a **prototype**, then test in the open.
switchyard goes public at that point. The gate, all checked in one PR before anything is released:
- **Phases 0–4 merged,** CI green on main, coverage ≥ 90% enforced (floor and Codecov), and the Atlas check
  passing.
- **README (a public surface, written by the Publisher, not Grok;** Grok's README is fine while the repo is
  internal):
  - a **Prototype** status line saying it's advisory only, never serves traffic, and has a single-rig scope;
  - what it does, and the routing table's provenance rules;
  - the threat model;
  - how to run the experiments;
  - the brand logo.
- **SECURITY.md, CHANGELOG.md, LICENSE** present, and the identity scan clean over the whole tracked tree.
- **Nothing rig-private in the committed receipts:** no home paths, no tokens, no hostnames beyond
  `127.0.0.1`. Device names and driver versions are fine.
- **The version is `0.1.0`,** marked prototype. Shipcheck's v1.0.0 minimum applies to the later full
  treatment, not to this.
- **Then stop.** Flipping the repo public, the tag and the GitHub release are the Director's go; the Publisher
  executes them. A public flip is an outward action, with "set it back to internal" as its compensator.

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
| Brand-repo logo PR (public) | revert the merge commit in `brand`; `brand manifest` regenerates | logo gone from brand | Grok opens, the Publisher merges |
| Flip switchyard public (Phase 5) | `gh repo edit mcp-tool-shop-org/switchyard --visibility internal --accept-visibility-change-consequences` | internal again; anything already cloned or cached stays out | the Publisher, on the Director's go |

## Log

- 2026-10-09: written by R&D at the Director's ask; Mike relays to Grok. E2 and E3 take over the
  OpenVINO-model trials that were queued for Kimi's next handoff (see `kimi-intel-npu-buildout-handoff.md`).
- 2026-10-09 (afternoon), the Publisher's slotting terms for Phase 3:
  - E1–E3 use NPU ledger slots whenever the NPU is free.
  - E1's CPU slot and E3's qwen3:8b reference share the CPU window or a free card gap.
  - **E4 gets its own booked block:** a grant plus an NPU session together, never next to a timed calibration
    or ladder run.
  - PR terms as rule 10/11: Atlas plus `atlas check` from the first workflow, a coverage floor at the measured
    value, and the identity scan before every push. The Publisher merges after R&D's review.
- 2026-10-09 (afternoon): revised after Kimi's consult review (D1–D9). The reading list now uses real file paths
  (no `[[…]]` slugs). Added: cast, precedence, the stop rule, the acknowledgement, branch naming, pinned
  sources with a ladder pre-flight, a note that the empty `E:/AI/switchyard` is expected, a snapshot of the brief
  into `docs/handoff/` in Phase 0, and the E1–E4 specs moved explicitly into Phase 2.
- 2026-10-09 (afternoon), the Director: switchyard goes public soon as a prototype, after it's built with
  coverage at 90%, then it's tested in the open. Added: the 90% bar from Phase 0, the logo in the brand repo
  now, and Phase 5 (the prototype release gate). The public flip, tag and release are the Director's go.
- 2026-10-09 (afternoon), the Director: "make sure that switchyard ties into offrig." Added the tie-in: two
  contracts (offrig's calibrate runs, offers and budget in; a versioned route-decision JSON with offrig plan
  suggestions out), switchyard never spends, "calibrate what is served" enforced, offrig as an experiment
  adapter in scratch projects, and a design-note draft for the Publisher. Asked the Publisher for `--json` on
  `offrig gpus` and `offrig budget --show`.
- 2026-10-09 (afternoon), the Publisher: `offrig budget --show --json` (version 1) lands in the account-budget PR, and
  `offrig gpus --json` follows. The importers pin `"version": 1`. Wiring route decisions into offrig is the
  Publisher's call, after the integration draft.
- 2026-10-09 (afternoon), the Director (via the Publisher): a self-repairing runner. Added as a **design draft** (not a
  build order), after Phases 0–2. R&D's refinements to the Publisher's shape: an identity fingerprint checked
  around every repair; Ollama restarts gated to an exclusive grant on 11434 with a version check;
  `context_overflow` escalates instead of being re-run at a larger window; a training resume needs the sampler and
  RNG state; repair counts become reliability rows; error text is quoted as data. For Kimi's review, relayed by
  Mike.
- 2026-10-09 (afternoon), the Publisher accepted R&D's six refinements and added the CUDA 13.4 backend to the
  fingerprint (the recorded start line, `cuda_v13` in the log, the ggml-cuda sha).
- 2026-10-09 (afternoon), Kimi's review of the self-repairing runner was folded in. Added: latency markers, with
  clean-only latency aggregation and cold-after-restart; a fingerprint checked on a cadence; restart gates
  (re-read the ledger with a fencing token, an nvidia-smi foreign-PID check, a TOCTOU re-check, log and process
  freshness, no retry on a failed restart); and the full training-resume state (four RNG streams, counters,
  GradScaler, EMA, loader determinism, trainer sha and determinism flags). New dependency: a machine-readable
  grant ledger from the Publisher.
- 2026-10-09 (afternoon), Grok's Phase 0 is PR #1 (CI green). R&D's review asks for Codecov (90/90, OIDC) and CI
  hygiene (SHA-pinned actions, timeout, permissions, push only on main). Answers to Grok:
  - **pin rnd at `08ca419`**;
  - **the chain and ladder rows are quote rule 1, by a mechanical inference, not a default.** Every offrig verdict
    written by a binary at or after #47 (`8114f07`) pins `quote_rule`, and 0 of the 7,636 verdict lines in both
    result trees carry it, so all were judged by a pre-#47 binary, which had only rule 1. Record it as
    `quote_rule: 1` with `quote_rule_source: "pins lack quote_rule; offrig#47 (8114f07) pins it on every
    verdict"`.
- 2026-10-09 (afternoon), the Director (via the Publisher): E5, "fill the card" context sizing, the ceiling for
  offrig's `--num-ctx auto`. R&D's refinements:
  - a doubling-then-bisect search, not a linear 2048 walk;
  - a stop margin below the watchdog ceiling, never testing the ceiling, with the watchdog's behaviour
    confirmed first;
  - full residency checked by both `/api/ps` and the log's layer count;
  - a confirmation step measuring peak VRAM with a ~90% prompt (compute buffers peak there, not at load);
  - rows keyed by the idle baseline and the server environment.
- 2026-10-09 (afternoon), Mike asked for a Kimi handoff on switchyard. Kimi owns the Intel lane: the OpenVINO
  adapter, the E1–E3 specs and runs, and second review of `devices.py` and the adapter interface. Grok keeps the
  core, E4, E5 and the router. One owner per file.
- 2026-10-09 (afternoon), Kimi's consult: PR #1's `devices.py` should decide Intel by vendor 0x8086 plus
  INTEGRATED (requested change, posted on the PR); the GPU.1 pin is documented; three more test params; a live
  `Core()` at resolve time. Folded into E5: a spill detector via latency, the margin as an assumption to
  validate, the driver version as a staleness key, and token-exact equivalence for "continue at a bigger
  window". Throughput rows claim measured values, never spec (the NPU's advertised 13.1 int8 TOPS vs 2.38
  measured).
- 2026-10-09 (afternoon), Grok: PR #1 (Phase 0) is fixed and R&D-approved. PR #2 (Phase 1, routing table, 408
  rows, rnd pinned at 08ca419, quote rule 1 by inference) is R&D-approved with notes. Grok flagged the brief's
  Phase 2 contradiction (E1–E3 specs listed under Grok's phase but given to Kimi), fixed: Phase 2 is the
  harness, the adapter interface, and the E4 and E5 specs; E1–E3 specs are Kimi's. The brand-repo logo PR is
  still open as an item.
- 2026-10-09 (afternoon), the Publisher merged switchyard #1 (860dbd9, a merge commit) and #2 (d86e63b, a squash).
  CI was green and both identity scans clean. Notes: the logo is being regenerated (replace the files, and send
  the brand PR with the new version), and the public README is the Publisher's to write before going public.
