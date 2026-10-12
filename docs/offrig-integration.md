# offrig integration (draft)

This is a note for the Publisher. Wiring offrig to call interlock is the Publisher's decision and the Publisher's PR. interlock does not call `offrig_plan` or `offrig_launch`, does not open offrig's database, and does not spend.

The 2026-10-09 handoff described a schema at `schemas/route-decision.v1.json`, an offrig pin of `751fb1d`, and a read of the remaining budget. The 2026-10-11 brief moved the schema to `src/interlock/schema/route-decision.v1.json`, pinned the plan input at `8f2a94af0c354a2952ec099beac8fc90d8f11ffb`, and made the budget an input flag. This draft follows that brief. `interlock route --json` emits a document that validates against the schema in the tree. A remote document's `offrig_plan` validates against `tests/fixtures/offrig-plan-input.8f2a94a.json`, which is the `offrig_plan` inputSchema captured from offrig-mcp `tools/list` at that pin. The required fields are `profile` and `max_hours`. Optional fields are `note`, `max_price_hr`, `no_fallback`, `wait_minutes` and `container_disk_gb`. `worst_case_usd` sits beside the plan, on the decision. It is not a plan field.

## (a) verify, default per engine

A `verify:grounded`, `verify:reasoning` or `verify:math-ladder` route cites the calibration row it ranked. The decision's model (repo or tag, and the revision when the repo does not already carry it), engine name and version, and settings (`thinking_level`, `num_predict`, `quote_rule`) are the default offrig can use for `verify` on that engine. The same digest, engine, think, `num_predict` and quote rule have to be on the row. A request that names a quote rule the table does not have refuses. Quote rule 2 has no committed receipt and no spec, so the refusal's experiment is `unregistered`.

## (b) the indexer embed_url

An `embed` route names a device, an engine and a model. That is the local endpoint offrig's indexer could point at with `embed_url` / `OFFRIG_EMBED_URL`. interlock does not set the variable and does not start the server. A capability row for that model and device is listed under `excluded` and is not the suggestion.

## (c) local, pod, serverless

`NPU`, `iGPU`, `CPU` and `5090` are local. A `runpod:` device is a remote suggestion. The decision then carries `offrig_plan`. The `profile` string is offrig's own (`small`, `medium`, `frontier`, or `job`). Pod and serverless are that profile, not a second interlock device. `--budget-left-usd` drops a remote row whose `settings.worst_case_usd` is missing or greater than the flag, and the row is a runner-up with the reason. With no flag the suggestion is kept and `budget` is `unchecked`. The flag is an input. interlock does not read offrig's balance, the Publisher's ledger, the watchdog, or `/api/ps`.

There is no `runpod:` row in the committed table. The remote shape is what the policy returns when a row of that shape is in the table. It is not a measured receipt.
