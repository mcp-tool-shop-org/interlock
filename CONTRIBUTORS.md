# Contributors

interlock is maintained by [MCP Tool Shop](https://mcp-tool-shop.github.io/). Much of its code was written by AI
coding assistants working in the studio's seats, and each one is credited here. Pull requests are counted by branch
prefix, as merged on `main` on 2026-10-16.

## Written with Grok

Parts of this repository were **Written with Grok** (xAI), through Grok's coding seat: 40 merged pull requests on
`grok/*` branches. They include:

- the Phase 0 scaffold and lockdown, the Phase 1 routing table, and the Phase 2 harness with its faked adapters;
- the run supervisor: hang and device-loss handling, Ollama restarts, unload and idle waits;
- the E1–E5 experiment work on the importer, the fill-card search, E4b and the E5 spill gates;
- the advisory router v0, the rename to interlock, and the 0.1.0 release preparation.

## Kimi

Kimi (Moonshot AI), run through Ollama Cloud: 21 merged pull requests on `kimi/*` branches. They include:

- the Intel lane's OpenVINO adapter;
- the E1, E2 and E3 pre-registrations, amendments and receipts;
- last-token pooling, `token_type_ids` feeding, and the per-arm call bounds.

## Claude

Claude (Anthropic), through Claude Code, in the Publisher and R&D sessions: the remaining pull requests. These are
the harness and genai fixes, the receipts, the README and the release.

## How a contribution is credited

New commits from an AI seat carry a `Co-authored-by:` trailer and a `Seat:` trailer naming the seat. The history is
not rewritten: the counts above come from branch names, and this file is the record.
