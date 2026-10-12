# Changelog

## Unreleased

- Ollama fakes check each request against a recorded exchange. `scripts/record_ollama_fixtures.py` writes those exchanges; it is run on the rig, and it refuses a cloud model or a card that is not idle. `num_predict` 0, null, and missing are recorded as no reply within the timeout. A token count stops after twice the cold-load allowance of the largest roster model (`/api/tags` size, or 16 GiB when the size is missing) and the arm is refused. `confirm_num_predict` below 1 is refused when the spec is validated, and again before a fill-card model is contacted.
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
