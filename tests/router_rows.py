"""Small routing rows for the policy tests. Not a receipt, and not the committed table."""

from datetime import date


AS_OF = date(2026, 10, 11)


def measured(
    row_id,
    device,
    *,
    task="generate",
    throughput=10.0,
    latency=100.0,
    spread=0.0,
    repo="acme/Phi-test",
    revision="abc123",
    thinking="off",
    quote_rule=None,
    num_predict=None,
    num_ctx=None,
    prefix=None,
    when="2026-10-10",
    repeats=3,
    provisional=False,
    outcome=None,
    spec="experiments/e3/spec.yaml",
    quality=None,
    claim_wall=None,
    ttft=None,
    worst=None,
    plan=None,
    slots="repeat",
    parity=None,
):
    settings = {}
    if thinking is not None:
        settings["thinking_level"] = thinking
    if quote_rule is not None:
        settings["quote_rule"] = quote_rule
    if num_predict is not None:
        settings["num_predict"] = num_predict
    if num_ctx is not None:
        settings["num_ctx"] = num_ctx
    if prefix is not None:
        settings["prefix_caching"] = prefix
    if worst is not None:
        settings["worst_case_usd"] = worst
    if plan is not None:
        settings["offrig_plan"] = plan
    if slots == "repeat":
        repeat = {"phase": "repeat", "throughput": throughput, "latency_ms": latency}
        other = {"phase": "repeat", "throughput": throughput + spread, "latency_ms": latency}
        if ttft is not None:
            repeat["ttft_ms"] = ttft
            other["ttft_ms"] = ttft
        artifacts = {
            "slots": [
                {"phase": "cold", "throughput": 1000.0, "latency_ms": 1.0, "ttft_ms": 1.0},
                {"phase": "warmup", "throughput": 500.0, "latency_ms": 1.0},
                repeat,
                other,
            ]
        }
    elif slots == "broken":
        artifacts = {"slots": "not-a-list"}
    else:
        artifacts = {"slots": []}
    latency_obj = {"throughput": throughput + max(spread, 1.0)}
    if claim_wall is not None:
        latency_obj["claim_wall_s"] = {"min": 0.1, "max": claim_wall}
    if spec:
        provenance = {"experiment_spec": spec, "run_id": "t"}
    else:
        provenance = {"source_receipt_path": "receipts/x.json", "commit_sha": "abc"}
    engine = "ollama" if device == "5090" or str(device).startswith("runpod:") else "openvino"
    row = {
        "id": row_id,
        "task": task,
        "model": {"repo_or_tag": repo, "revision_or_digest": revision},
        "engine": {"name": engine, "version": "2026.4.1"},
        "device": device,
        "settings": settings,
        "date": when,
        "repeats": repeats,
        "provisional": provisional,
        "provenance": provenance,
        "artifacts": artifacts,
        "latency": latency_obj,
        "quality": quality or [],
    }
    if outcome:
        row["outcome"] = outcome
    if parity is not None:
        row["parity_valid"] = parity
    return row
