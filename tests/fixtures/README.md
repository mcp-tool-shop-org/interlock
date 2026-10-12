`ollama/<version>/*.json` is written by `scripts/record_ollama_fixtures.py` on the rig. Those files are not in the tree until that command has been run. The three no-limit cases send `num_ctx` 64 and `shift` false. Their `behaviour` is `no_reply`, and `eval_count` is above 1.

offrig-plan-input.8f2a94a.json is the inputSchema of tool offrig_plan, captured 2026-10-11.

Command: offrig-mcp
Request: {"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}

That tools/list response is PlanArgs at offrig 8f2a94af0c354a2952ec099beac8fc90d8f11ffb. PlanArgs in crates/offrig-mcp/src/main.rs is identical at that commit and at 47917bdddde57983e0d74164485e4491dec44f95.
