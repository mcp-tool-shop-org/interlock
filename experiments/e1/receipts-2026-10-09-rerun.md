# Label for receipts/e1-intel-2026-10-09-rerun*.jsonl — INVALID as routing rows

These receipts answer operations questions only (the new device_lost and
not-run machinery works; the tail-block rule was exercised). They must not
be imported as routing rows. Two bugs invalidate the embedding semantics and
the parity verdict, found by the CPU-only parity investigation the same day:

1. token_type_ids omitted (adapter, fixed in PR #22). Every arm ran with an
   uninitialized token-type buffer. Latency/compile timings are shape-valid;
   embedding values and parity cosines are not.
2. Prompt expansion missing (runner, Grok's fix). The runner passed
   dataset.seed verbatim (~13 tokens); the spec's 2048-token prefix never
   ran. Warm medians are costs of a 13-token call, so B_fresh/B_reuse units
   are wrong until the re-run.

The 45 measured rows and the one device_lost row stay on the record as
evidence. The full E1 re-run is pre-registered to happen after both fixes
and the warm-reload receipt field land, and after the NPU health check.
