"""Stage 1 of the interlock-relay design, on fakes: the off-on-off triples on interlock's own ABA order.

Arm A and arm B are the same model and settings. B is traced (``trace: true``), A isn't. The tracer switches the
sink's events per slot and the counter stays on. After every slot both arms take the same barrier. No device is used:
the Ollama adapter talks to a fake transport.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from interlock.adapters.ollama import OllamaAdapter
from interlock.calltrace import Tracer
from interlock.harness.runner import RunnerError, run_spec
from interlock.harness.spec import SpecError, validate_spec
from interlock.harness.watch import WatchHooks
from interlock.receipt import check_tree, load_jsonl
from tests.test_run_path import _bind

MODES = ["events", pytest.param("relay", marks=pytest.mark.relay)]


def _need(mode):
    if mode == "relay":
        pytest.importorskip("nemo_relay")


class Server:
    """A fake Ollama: answers show, generate and unload, and logs every request (the H1 ground truth)."""

    def __init__(self):
        self.requests = []

    def __call__(self, method, url, body=None, **kwargs):
        self.requests.append((url.rsplit("/api/", 1)[-1], dict(body or {})))
        if url.endswith("/api/show"):
            return {"model_info": {"qwen3.context_length": 40960}, "details": {"quantization_level": "Q4_K_M"}}
        if body and body.get("keep_alive") == 0 and body.get("prompt") == "":
            return {"done": True}
        return {"response": "ok", "total_duration": 450_000_000, "eval_count": 64, "eval_duration": 320_000_000,
                "prompt_eval_count": 256, "prompt_eval_duration": 30_000_000, "load_duration": 1_000_000}


def _arm(arm_id, *, trace, salt=False):
    arm = {
        "id": arm_id, "task": "generate", "device": "CPU", "engine": "ollama", "model": "qwen3:8b",
        "revision_or_digest": "abc", "quant": "unrecorded", "licence": "unrecorded", "engine_version": "0.35.1",
        "thinking_level": "off", "num_predict": 64, "num_ctx": 4096, "batch": None,
        "prompt": "The switchyard counts the same cars on every pass.", "measure": "decode_tok_s", "trace": trace,
    }
    if salt:
        arm["prompt_salt"] = "per_call"
    return arm


def _spec(*, repeats=3, barrier=250, salt=False):
    return {
        "id": "RELAY-S1-T", "procedure": "interleaved", "order": "aba",
        "question": "Does tracing a call change what interlock measures?",
        "decision": "A test verdict, computed after the run.",
        "needs": {"grant": False, "npu_session": False, "cpu_window": False},
        "stop_rules": ["a failed step stops the sequence"], "dataset": {"name": "fixture"},
        "metrics": ["decode_tok_s", "latency_ms"], "warmups": 1, "repeats": repeats, "band": 0.1,
        "compile_cache": "reuse", "rest_s": 0, "trace_barrier_ms": barrier,
        "arms": [_arm("A", trace=False, salt=salt), _arm("B", trace=True, salt=salt)],
        "pairs": [{"id": "relay", "alone": "A", "paired": "B", "contrast": "trace"}],
    }


def _run(tmp_path, mode, *, slept=None, salt=False, run_id="s1"):
    server = Server()
    repo = tmp_path / "repo"
    spec_path = _bind(repo, _spec(salt=salt))
    hooks = WatchHooks(ram_sampler=None, start_thread=False,
                       sleep=(slept.append if slept is not None else (lambda s: None)))
    rows = run_spec(spec_path, root=repo, adapters={"ollama": OllamaAdapter(transport=server)},
                    date="2026-10-14", run_id=run_id, trace=mode, watch=hooks)
    return repo, rows, server


def _slots(repo, run_id="s1"):
    rows = load_jsonl(repo / "receipts" / f"{run_id}.jsonl")
    return {row["id"].split("/")[1]: row["artifacts"]["slots"] for row in rows}


@pytest.mark.parametrize("mode", MODES)
def test_only_the_traced_arms_calls_reach_the_trace_and_every_call_is_counted(tmp_path: Path, mode):
    _need(mode)
    repo, rows, server = _run(tmp_path, mode)
    assert check_tree(repo) == []
    slots = _slots(repo)
    a_ids = {i for s in slots["A"] for i in s["call_ids"]}
    b_ids = {i for s in slots["B"] for i in s["call_ids"]}
    assert all(s["traced"] is False for s in slots["A"]) and all(s["traced"] is True for s in slots["B"])
    text = next((repo / "logs").glob("s1.*.jsonl")).read_text(encoding="utf-8")
    lines = [json.loads(x) for x in text.splitlines()]
    if mode == "events":
        logged = {x["call_id"] for x in lines if "call_id" in x}
    else:
        llm = [x for x in lines if x.get("category") == "llm"]
        starts = {x["metadata"]["call_id"] for x in llm if x["scope_category"] == "start"}
        ends = {x["metadata"]["call_id"] for x in llm if x["scope_category"] == "end"}
        assert starts == ends  # no start without its end
        logged = starts
    assert b_ids and logged == b_ids  # exactly the traced slots' calls: nothing from A, nothing between slots
    assert not (a_ids & logged)
    # H1's table: every request the server saw is counted, run-level calls included, in logs/<run_id>.calls.json.
    table = json.loads((repo / "logs" / "s1.calls.json").read_text(encoding="utf-8"))["counts"]
    assert sum(v["n"] for kinds in table.values() for v in kinds.values()) == len(server.requests)
    assert sum(v["n"] for row in rows for v in row["artifacts"]["call_counts"].values()) == sum(
        v["n"] for arm, kinds in table.items() if arm != "_run" for v in kinds.values())


@pytest.mark.parametrize("mode", MODES)
def test_every_slot_takes_the_barrier_and_records_its_hook_time(tmp_path: Path, mode):
    _need(mode)
    slept = []
    repo, _rows, _server = _run(tmp_path, mode, slept=slept)
    slots = _slots(repo)
    measured = len(slots["A"]) + len(slots["B"])
    assert slept.count(0.25) == measured  # one barrier after every slot, in both arms
    for arm_slots in slots.values():
        assert all(isinstance(s["hook_us"], float) and s["hook_us"] >= 0 for s in arm_slots)


def test_the_aba_order_gives_off_on_off_triples(tmp_path: Path):
    repo, _rows, _server = _run(tmp_path, "events")
    slots = _slots(repo)
    rounds = {}
    for arm_id, arm_slots in slots.items():
        for s in arm_slots:
            if s["phase"] == "repeat":
                rounds.setdefault(s["round"], {})[s["role"]] = (arm_id, s["traced"])
    assert len(rounds) == 3
    for roles in rounds.values():
        assert roles == {"alone-before": ("A", False), "paired": ("B", True), "alone-after": ("A", False)}


def test_a_salted_arm_sends_a_distinct_prompt_start_on_every_call(tmp_path: Path):
    repo, _rows, server = _run(tmp_path, "events", salt=True)
    prompts = [body["prompt"] for path, body in server.requests if path == "generate" and body.get("prompt")]
    assert len(prompts) == len(set(prompts)) > 4
    assert all(p.startswith("[") for p in prompts)
    salts = [s["prompt_salt"] for arm_slots in _slots(repo).values() for s in arm_slots]
    assert len(salts) == len(set(salts))  # one number per call, across both arms


def test_a_spec_with_traced_arms_refuses_to_run_untraced(tmp_path: Path):
    repo = tmp_path / "repo"
    spec_path = _bind(repo, _spec())
    with pytest.raises(RunnerError, match="traces some arms"):
        run_spec(spec_path, root=repo, adapters={"ollama": OllamaAdapter(transport=Server())}, date="2026-10-14",
                 trace="off")
    assert not (repo / "receipts").exists()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda s: s["arms"][0].update(trace="yes"), "trace must be true or false"),
        (lambda s: s["arms"][0].update(prompt_salt="sometimes"), "prompt_salt must be per_call"),
        (lambda s: s.update(trace_barrier_ms=-1), "trace_barrier_ms"),
        (lambda s: s.update(trace_barrier_ms=True), "trace_barrier_ms"),
        (lambda s: s["arms"][1].update(num_predict=128), "differ in num_predict"),
        (lambda s: s["arms"][1].update(trace=False), "paired arm trace true"),
        (lambda s: s["pairs"][0].update(contrast="companion"), "contrast must be trace"),
    ],
)
def test_the_stage_1_spec_keys_are_checked(change, message):
    spec = _spec()
    change(spec)
    with pytest.raises(SpecError, match=message):
        validate_spec(spec)


def test_a_call_keeps_the_emission_it_started_with(tmp_path: Path):
    tracer = Tracer(root=tmp_path, run_id="sw", mode="events")
    with tracer:
        tracer.set_emitting(True)
        traced = tracer.start("generate", "/api/generate", model="m", device="5090")
        tracer.set_emitting(False)
        untraced = tracer.start("generate", "/api/generate", model="m", device="5090")
        tracer.end(traced, "ok")  # ends after the switch went off, and is still logged
        tracer.end(untraced, "ok")
        assert tracer.emitting is False
    lines = [json.loads(x) for x in (tmp_path / "logs" / "sw.events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [x["call_id"] for x in lines] == [traced.call_id]
    assert tracer.counts_for("_run")["generate"]["n"] == 2
