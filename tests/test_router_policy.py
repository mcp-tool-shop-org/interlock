"""The route policy is a pure function. These tests never read the committed table."""

from datetime import date
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from switchyard.router.policy import (
    EXCLUDED_OUTCOMES,
    Decision,
    Refusal,
    Request,
    decode_bound,
    decide,
    experiment_of,
    latency_bound_ms,
    prefix_mode,
    ttft_bound_ms,
)
from tests.router_rows import AS_OF, measured

ROOT = Path(__file__).resolve().parents[1]
_QUIET = settings(max_examples=25, deadline=None, database=None)


def ask(**kwargs):
    if "busy" in kwargs and not isinstance(kwargs["busy"], frozenset):
        kwargs["busy"] = frozenset(kwargs["busy"])
    task = kwargs.pop("task", "generate")
    return Request(task, kwargs.pop("as_of", AS_OF), **kwargs)


def _not_strictly_better(before, after, *, higher: bool):
    if isinstance(before, Refusal):
        assert isinstance(after, Refusal)
        return
    if isinstance(after, Refusal):
        return
    left = before.document["margin"]["bound"]
    right = after.document["margin"]["bound"]
    if higher:
        assert right <= left
    else:
        assert right >= left


@st.composite
def _generate_rows(draw):
    count = draw(st.integers(min_value=1, max_value=4))
    devices = draw(st.lists(st.sampled_from(["CPU", "iGPU", "NPU", "5090"]), min_size=count, max_size=count))
    rows = []
    for index, device in enumerate(devices):
        rows.append(
            measured(
                f"g/{device}/{index}",
                device,
                throughput=float(draw(st.integers(min_value=1, max_value=80))),
                latency=float(draw(st.integers(min_value=10, max_value=400))),
            )
        )
    return rows


@_QUIET
@given(
    rows=_generate_rows(),
    floor=st.integers(min_value=0, max_value=100),
    cap=st.integers(min_value=1, max_value=500),
)
def test_tightening_a_constraint_never_yields_a_strictly_better_choice(rows, floor, cap):
    loose = decide(rows, ask(task="generate"))
    by_floor = decide(rows, ask(task="generate", min_metrics={"tok-s": float(floor)}))
    by_latency = decide(rows, ask(task="generate", max_latency_ms=float(cap)))
    _not_strictly_better(loose, by_floor, higher=True)
    _not_strictly_better(loose, by_latency, higher=True)
    if isinstance(loose, Decision):
        device = loose.document["device"]
        busied = decide(rows, ask(task="generate", busy={device}))
        _not_strictly_better(loose, busied, higher=True)

    slow = measured("e/slow", "CPU", task="embed", latency=100)
    fast = measured("e/fast", "iGPU", task="embed", latency=10)
    open_latency = decide([slow, fast], ask(task="embed"))
    capped = decide([slow, fast], ask(task="embed", max_latency_ms=50))
    assert isinstance(open_latency, Decision)
    assert open_latency.document["device"] == "iGPU"
    _not_strictly_better(open_latency, capped, higher=False)
    tighter = decide([slow, fast], ask(task="embed", max_latency_ms=5))
    assert isinstance(tighter, Refusal)


@_QUIET
@given(rows=_generate_rows(), code=st.sampled_from(EXCLUDED_OUTCOMES))
def test_a_capability_rejected_row_never_changes_the_decision(rows, code):
    request = ask(task="generate")
    before = decide(rows, request)
    extra = measured("rejected/extra", "NPU", outcome=code, throughput=10_000)
    after = decide([*rows, extra], request)
    assert {key: value for key, value in after.document.items() if key != "excluded"} == {
        key: value for key, value in before.document.items() if key != "excluded"
    }
    assert extra["id"] in {item["id"] for item in after.document["excluded"]}
    assert all(item["id"] != extra["id"] for item in after.document.get("runners_up", []))
    other_task = measured("rejected/embed", "CPU", task="embed", outcome=code, throughput=10_000)
    untouched = decide([*rows, other_task], request)
    assert untouched.document == before.document


@_QUIET
@given(rows=_generate_rows())
def test_a_parity_invalid_row_never_changes_the_decision(rows):
    request = ask(task="generate")
    before = decide(rows, request)
    extra = measured("parity/extra", "NPU", parity=False, throughput=10_000)
    after = decide([*rows, extra], request)
    assert {key: value for key, value in after.document.items() if key != "excluded"} == {
        key: value for key, value in before.document.items() if key != "excluded"
    }
    assert {"id": extra["id"], "code": "parity_invalid"} in after.document["excluded"]
    assert all(item["id"] != extra["id"] for item in after.document.get("runners_up", []))
    other_task = measured("parity/embed", "CPU", task="embed", parity=False, throughput=10_000)
    untouched = decide([*rows, other_task], request)
    assert untouched.document == before.document


@_QUIET
@given(rows=_generate_rows())
def test_busy_on_the_chosen_device_never_returns_that_device(rows):
    request = ask(task="generate")
    chosen = decide(rows, request)
    if isinstance(chosen, Refusal):
        return
    device = chosen.document["device"]
    again = decide(rows, ask(task="generate", busy={device}))
    if isinstance(again, Decision):
        assert again.document["device"] != device
        assert device not in again.document["receipts"][0]


def test_bounds_ignore_cold_slots_medians_and_a_missing_spread():
    row = measured("b", "CPU", throughput=7.0, spread=4.0, latency=80)
    assert decode_bound(row) == 7.0
    assert row["latency"]["throughput"] == 11.0
    assert latency_bound_ms(row) == 80
    assert decode_bound({"artifacts": {"slots": "not-a-list"}}) is None
    assert latency_bound_ms({"artifacts": {"slots": []}, "latency": {}}) is None
    claim = measured("c", "CPU", slots="empty", claim_wall=1.25, task="verify:grounded")
    assert latency_bound_ms(claim) == 1250
    assert ttft_bound_ms({"artifacts": {"slots": [{"phase": "repeat"}]}, "genai": {"ttft_ms_p50": 3}}) is None
    chosen = decide([row], ask())
    assert isinstance(chosen, Decision)
    assert chosen.document["margin"]["bound"] == 7.0
    assert chosen.document["margin"]["bound"] != row["latency"]["throughput"]
    assert chosen.document["margin"]["next_bound"] is None
    assert chosen.document["margin"]["delta"] is None
    assert chosen.document["margin"]["next_id"] is None


def test_ties_leave_the_5090_free_and_then_the_lower_id():
    card = measured("b", "5090", throughput=10, repo="phi:mini")
    igpu = measured("a", "iGPU", throughput=10, repo="phi:mini")
    tied = decide([card, igpu], ask(model="Phi"))
    assert isinstance(tied, Decision)
    assert tied.document["device"] == "iGPU"
    assert tied.document["runners_up"][0]["constraint"] == "tie_break"
    assert "5090" in tied.document["runners_up"][0]["reason"]
    assert "no 5090 row" not in tied.document["notes"]

    later = measured("m/b", "CPU", throughput=10)
    earlier = measured("m/a", "NPU", throughput=10)
    by_id = decide([later, earlier], ask())
    assert isinstance(by_id, Decision)
    assert by_id.document["receipts"] == ["m/a"]
    assert "lower row id" in by_id.document["runners_up"][0]["reason"]


def test_a_synthetic_5090_phi_row_wins_when_one_exists():
    """Not a golden. The committed table has no 5090 Phi row, and this one is invented here."""

    igpu = measured("phi/igpu", "iGPU", throughput=21, repo="OpenVINO/Phi-3.5-mini-instruct-int4-cw-ov@abc")
    card = measured("phi/5090", "5090", throughput=100, repo="phi:mini")
    won = decide([igpu, card], ask(model="Phi"))
    assert isinstance(won, Decision)
    assert won.document["device"] == "5090"
    assert won.document["model"] == "phi:mini@abc123"
    assert "no 5090 row" not in won.document["notes"]

    busy = decide([igpu, card], ask(model="Phi", busy={"5090"}))
    assert isinstance(busy, Decision)
    assert busy.document["device"] == "iGPU"
    assert "no 5090 row" not in busy.document["notes"]

    missing = decide([igpu], ask(model="Phi"))
    assert isinstance(missing, Decision)
    assert missing.document["notes"] == ["no 5090 row"]

    weak = measured("phi/5090", "5090", throughput=3, repo="phi:mini")
    lost = decide([igpu, weak], ask(model="Phi"))
    assert isinstance(lost, Decision)
    assert lost.document["device"] == "iGPU"
    assert "no 5090 row" not in lost.document["notes"]


def test_model_labels_and_the_alnum_match():
    tagged = measured("tag", "5090", throughput=100, repo="qwen3:8b", revision="deadbeef")
    ov = measured("ov", "iGPU", throughput=12, repo="OpenVINO/Qwen3-8B-int4-cw-ov@abc", revision="abc")
    bigger = measured("big", "CPU", throughput=200, repo="qwen3:14b")
    chosen = decide([ov, tagged, bigger], ask(model="Qwen3-8B"))
    assert isinstance(chosen, Decision)
    assert chosen.document["device"] == "5090"
    assert chosen.document["model"] == "qwen3:8b@deadbeef"
    assert all(item["id"] != "big" for item in chosen.document["runners_up"])

    already = measured("at", "CPU", repo="qwen3:8b@deadbeef", revision="deadbeef")
    assert decide([already], ask()).document["model"] == "qwen3:8b@deadbeef"
    embedded = measured("in", "CPU", repo="qwen3:8b", revision="8b")
    assert decide([embedded], ask()).document["model"] == "qwen3:8b"
    blank = measured("blank", "CPU", repo="qwen3:8b", revision="")
    assert decide([blank], ask()).document["model"] == "qwen3:8b"


def test_stale_and_provisional_are_flags_not_filters():
    fresh = measured("f", "iGPU", throughput=5, when="2026-09-11", repeats=2, provisional=False)
    old = measured("o", "CPU", throughput=50, when="2026-09-10", repeats=1, provisional=False)
    flagged = decide([fresh, old], ask())
    assert isinstance(flagged, Decision)
    assert flagged.document["device"] == "CPU"
    assert flagged.document["flags"] == ["stale", "provisional"]

    same = decide([fresh], ask())
    assert isinstance(same, Decision)
    assert same.document["flags"] == []
    marked = measured("p", "CPU", repeats=5, provisional=True)
    assert "provisional" in decide([marked], ask()).document["flags"]
    broken = measured("d", "CPU", when="not-a-date")
    assert "stale" in decide([broken], ask()).document["flags"]

    early = decide([old], ask(as_of=date(2026, 9, 20)))
    late = decide([old], ask(as_of=date(2026, 10, 11)))
    assert isinstance(early, Decision) and isinstance(late, Decision)
    assert early.document["receipts"] == late.document["receipts"]
    assert "stale" not in early.document["flags"]
    assert "stale" in late.document["flags"]


def test_ctx_is_an_exact_length_and_num_ctx_is_a_window():
    short = measured("m-l256/a", "iGPU", throughput=30)
    long = measured("m-l1024/a", "CPU", throughput=10)
    opened = decide([short, long], ask())
    assert isinstance(opened, Decision)
    assert opened.document["device"] == "iGPU"
    scoped = decide([short, long], ask(ctx=1024))
    assert isinstance(scoped, Decision)
    assert scoped.document["receipts"] == ["m-l1024/a"]
    assert scoped.document["runners_up"][0]["constraint"] == "ctx"
    missed = decide([short, long], ask(ctx=2048))
    assert isinstance(missed, Refusal)
    assert missed.document["constraint"] == "ctx"

    window = measured(
        "verify/gemma/q-1/np-64",
        "CPU",
        task="verify:grounded",
        num_ctx=2048,
        slots="empty",
        claim_wall=2.0,
    )
    text = measured(
        "verify/gemma/q-1/np-32",
        "iGPU",
        task="verify:grounded",
        num_ctx="2048",
        slots="empty",
        claim_wall=0.5,
    )
    covered = decide([window, text], ask(task="verify:grounded", ctx=1024))
    assert isinstance(covered, Decision)
    assert covered.document["receipts"] == ["verify/gemma/q-1/np-64"]
    too_wide = decide([window], ask(task="verify:grounded", ctx=4096))
    assert isinstance(too_wide, Refusal)
    assert too_wide.document["constraint"] == "ctx"


def test_identity_constraints_name_the_miss():
    row = measured("E3/arm/a", "CPU", thinking="on", quote_rule=1, num_predict=32, spec="experiments/e3/spec.yaml")
    thinking = decide([row], ask(thinking="off"))
    assert isinstance(thinking, Refusal)
    assert thinking.document["constraint"] == "thinking"
    assert thinking.document["experiment"] == "E3"
    quoted = decide([row], ask(quote_rule=2))
    assert isinstance(quoted, Refusal)
    assert quoted.document["constraint"] == "quote_rule"
    assert quoted.document["closest"]["bound"] == 1
    predicted = decide([row], ask(num_predict=64))
    assert isinstance(predicted, Refusal)
    assert predicted.document["constraint"] == "num_predict"
    assert predicted.document["closest"]["metric"] == "num_predict"
    assert predicted.document["closest"]["bound"] == 32

    kept = measured("keep", "iGPU", throughput=4, thinking="off")
    other = measured("other", "CPU", throughput=40, thinking="low")
    chosen = decide([kept, other], ask(thinking="off"))
    assert isinstance(chosen, Decision)
    assert chosen.document["device"] == "iGPU"
    assert chosen.document["runners_up"][0]["constraint"] == "thinking"


def test_thinking_on_matches_any_level_and_keeps_the_recorded_one():
    low = measured("low-arm", "iGPU", throughput=4, thinking="low", repo="phi:mini")
    turned = measured("on-arm", "CPU", throughput=3, thinking="on", repo="phi:mini")
    off = measured("off-arm", "NPU", throughput=100, thinking="off", repo="phi:mini")
    blank = measured("unset", "CPU", throughput=100, thinking=None, repo="phi:mini")
    card = measured("card", "5090", throughput=50, thinking="low", repo="phi:mini")
    chosen = decide([low, turned, off, blank, card], ask(model="Phi", thinking="On"))
    assert isinstance(chosen, Decision)
    assert chosen.document["receipts"] == ["card"]
    assert chosen.document["settings"]["thinking_level"] == "low"
    assert "no 5090 row" not in chosen.document["notes"]
    reasons = {item["id"]: item["constraint"] for item in chosen.document["runners_up"]}
    assert reasons["low-arm"] != "thinking"
    assert reasons["on-arm"] != "thinking"
    assert reasons["off-arm"] == "thinking"
    assert reasons["unset"] == "thinking"
    exact = decide([low, turned], ask(thinking="low"))
    assert isinstance(exact, Decision)
    assert exact.document["receipts"] == ["low-arm"]
    assert exact.document["settings"]["thinking_level"] == "low"
    assert exact.document["runners_up"][0]["constraint"] == "thinking"
    named = decide([low, measured("high-arm", "CPU", throughput=80, thinking="high")], ask(thinking="high"))
    assert isinstance(named, Decision)
    assert named.document["receipts"] == ["high-arm"]
    assert named.document["settings"]["thinking_level"] == "high"
    intel = measured("intel", "iGPU", throughput=4, thinking="off", repo="phi:mini")
    quiet = decide([intel, card], ask(model="Phi", thinking="off"))
    assert isinstance(quiet, Decision)
    assert quiet.document["device"] == "iGPU"
    assert quiet.document["notes"] == ["no 5090 row"]


def test_parity_invalid_is_excluded_and_a_capability_code_wins():
    good = measured("keep", "CPU", throughput=8, parity=True)
    missing = measured("open", "iGPU", throughput=6)
    bad = measured("bad", "NPU", throughput=1000, parity=False)
    both = measured("both", "CPU", throughput=1000, outcome="hang", parity=False)
    stranger = measured("stranger", "NPU", throughput=1000, parity=False, repo="other/model")
    other = measured("other-task", "NPU", task="embed", throughput=1000, parity=False)
    chosen = decide([good, missing, bad, both, stranger, other], ask(model="Phi"))
    assert isinstance(chosen, Decision)
    assert chosen.document["receipts"] == ["keep"]
    codes = {item["id"]: item["code"] for item in chosen.document["excluded"]}
    assert codes == {"bad": "parity_invalid", "both": "hang"}
    assert all(item["id"] not in codes for item in chosen.document["runners_up"])


def test_quality_and_latency_floors_name_the_gap():
    slow = measured(
        "slow",
        "CPU",
        task="rerank",
        latency=400,
        quality=[{"name": "min_cosine_vs_cpu", "interval": {"low": 0.9, "high": 1.0}}],
    )
    fast = measured(
        "fast",
        "iGPU",
        task="rerank",
        latency=40,
        quality=[{"name": "min-cosine-vs-cpu", "interval": {"low": 0.2, "high": 0.4}}],
    )
    chosen = decide([fast, slow], ask(task="rerank", min_metrics={"min-cosine-vs-cpu": 0.5}))
    assert isinstance(chosen, Decision)
    assert chosen.document["device"] == "CPU"
    assert chosen.document["runners_up"][0]["constraint"] == "min_min_cosine_vs_cpu"
    assert chosen.document["margin"]["metric"] == "warm_latency_ms"
    assert chosen.document["margin"]["direction"] == "lower"

    short = decide([fast], ask(task="rerank", min_metrics={"min-cosine-vs-cpu": 0.5}))
    assert isinstance(short, Refusal)
    assert short.document["constraint"] == "min_min_cosine_vs_cpu"
    assert short.document["closest"]["bound"] == 0.2
    assert short.document["closest"]["gap"] == pytest.approx(0.3)

    over = decide([measured("L", "CPU", latency=500)], ask(max_latency_ms=100))
    assert isinstance(over, Refusal)
    assert over.document["constraint"] == "max_latency_ms"
    assert over.document["closest"]["bound"] == 500
    assert over.document["closest"]["gap"] == 400

    floor = decide(
        [measured("a", "CPU", throughput=20), measured("b", "iGPU", throughput=10)],
        ask(min_metrics={"tok-s": 15}),
    )
    assert isinstance(floor, Decision)
    assert floor.document["runners_up"][0]["constraint"] == "min_tok_s"

    missing = decide([measured("n", "CPU", slots="empty")], ask())
    assert isinstance(missing, Refusal)
    assert missing.document["constraint"] == "decode_tok_s"
    assert missing.document["closest"]["bound"] is None
    empty = decide([], ask())
    assert isinstance(empty, Refusal)
    assert empty.document["closest"] is None
    assert "Nothing was close" in empty.document["hint"]
    assert empty.document["experiment"] == "unregistered"


def test_ttft_with_one_known_mode_uses_the_repeat_slot_max():
    assert prefix_mode({"settings": {"prefix_caching": "ON"}}) == "on"
    assert prefix_mode({"settings": {"prefix_caching": "library default (unknown)"}}) is None
    assert prefix_mode({"settings": {"prefix_caching": 1}}) is None
    assert prefix_mode({}) is None

    slow = measured("t-slow", "CPU", prefix="off", ttft=400, throughput=30)
    fast = measured("t-fast", "iGPU", prefix="off", ttft=50, throughput=10)
    kept = decide([slow, fast], ask(max_ttft_ms=100))
    assert isinstance(kept, Decision)
    assert kept.document["device"] == "iGPU"
    assert kept.document["runners_up"][0]["constraint"] == "max_ttft_ms"
    assert ttft_bound_ms(slow) == 400

    mixed = decide(
        [slow, measured("t-on", "NPU", prefix="on", ttft=10, throughput=90)],
        ask(max_ttft_ms=10_000),
    )
    assert isinstance(mixed, Refusal)
    assert mixed.document["code"] == "prefix_caching_mixed"
    assert "settings.prefix_caching" in mixed.document["hint"]

    unknown = decide([measured("t-unset", "CPU")], ask(max_ttft_ms=100))
    assert isinstance(unknown, Refusal)
    assert unknown.document["code"] == "prefix_caching_unknown"
    assert "settings.prefix_caching" in unknown.document["hint"]

    median_only = measured("t-p50", "CPU", prefix="default")
    refused = decide([median_only], ask(max_ttft_ms=100))
    assert isinstance(refused, Refusal)
    assert refused.document["constraint"] == "max_ttft_ms"
    assert refused.document["code"] == "no_candidate"


def test_ttft_mixed_prefix_caching_refuses():
    rows = [
        measured("on", "CPU", prefix="on", ttft=10),
        measured("off", "iGPU", prefix="off", ttft=10),
    ]
    refused = decide(rows, ask(max_ttft_ms=1000))
    assert isinstance(refused, Refusal)
    assert refused.document["code"] == "prefix_caching_mixed"
    assert "on" in refused.document["hint"] and "off" in refused.document["hint"]


def test_excluded_codes_are_listed_and_a_different_task_is_not():
    good = measured("keep", "CPU", throughput=10)
    rows = [
        good,
        measured("lost", "NPU", task="embed", outcome="device_lost", throughput=999),
        measured("hung", "iGPU", outcome="hang", throughput=999),
        measured("timed", "CPU", outcome="compile_timeout", throughput=999),
        measured("nat", "NPU", outcome="native_binary_rejected", throughput=999),
        measured("comp", "NPU", outcome="compile_rejected", throughput=999),
        measured("other", "NPU", outcome="hang", repo="other/model", throughput=999),
    ]
    chosen = decide(rows, ask(model="Phi"))
    assert isinstance(chosen, Decision)
    assert chosen.document["receipts"] == ["keep"]
    codes = {item["id"]: item["code"] for item in chosen.document["excluded"]}
    assert "lost" not in codes
    assert "other" not in codes
    assert codes == {
        "hung": "hang",
        "timed": "compile_timeout",
        "nat": "native_binary_rejected",
        "comp": "compile_rejected",
    }
    assert all(item["id"] not in codes for item in chosen.document["runners_up"])


def test_remote_without_budget_is_unchecked_and_validates_offrig_plan():
    from jsonschema import Draft202012Validator

    plan = {
        "profile": "small",
        "max_hours": 2,
        "note": None,
        "max_price_hr": 1.5,
        "gpu_count": 4,
        "no_fallback": False,
        "wait_minutes": 0,
        "container_disk_gb": 20,
    }
    remote = measured("remote/a", "runpod:A40", throughput=40, worst=3, plan=plan)
    local = measured("local/a", "CPU", throughput=5)
    chosen = decide([local, remote], ask())
    assert isinstance(chosen, Decision)
    assert chosen.document["device"] == "runpod:A40"
    assert chosen.document["budget"] == "unchecked"
    assert "budget unchecked" in chosen.document["notes"]
    assert chosen.document["worst_case_usd"] == 3
    assert "gpu_count" not in chosen.document["offrig_plan"]
    assert "note" not in chosen.document["offrig_plan"]
    assert chosen.document["offrig_plan"]["no_fallback"] is False
    schema = Draft202012Validator(
        __import__("json").loads((ROOT / "tests" / "fixtures" / "offrig-plan-input.8f2a94a.json").read_text(encoding="utf-8"))
    )
    schema.validate(chosen.document["offrig_plan"])


def test_remote_over_budget_is_a_runner_up():
    remote = measured(
        "remote/a",
        "runpod:A40",
        throughput=40,
        worst=9,
        plan={"profile": "job", "max_hours": 1},
    )
    local = measured("local/a", "CPU", throughput=5)
    chosen = decide([local, remote], ask(budget_left_usd=2))
    assert isinstance(chosen, Decision)
    assert chosen.document["device"] == "CPU"
    assert chosen.document["budget"] == "not_applicable"
    match = next(item for item in chosen.document["runners_up"] if item["id"] == "remote/a")
    assert match["constraint"] == "budget"
    assert "exceeds" in match["reason"]

    alone = decide([remote], ask(budget_left_usd=2))
    assert isinstance(alone, Refusal)
    assert alone.document["constraint"] == "budget"
    bare = measured("remote/bare", "runpod:A40", throughput=40, plan={"profile": "job", "max_hours": 1})
    dropped = decide([local, bare], ask(budget_left_usd=10))
    assert isinstance(dropped, Decision)
    assert dropped.document["device"] == "CPU"
    assert "worst_case_usd" in next(item["reason"] for item in dropped.document["runners_up"] if item["id"] == "remote/bare")


def test_remote_within_budget_is_marked_within():
    remote = measured(
        "remote/a",
        "runpod:A40",
        throughput=40,
        worst=2,
        plan={"profile": "job", "max_hours": 1},
    )
    local = measured("local/a", "CPU", throughput=5)
    chosen = decide([local, remote], ask(budget_left_usd=2))
    assert isinstance(chosen, Decision)
    assert chosen.document["device"] == "runpod:A40"
    assert chosen.document["budget"] == "within"
    assert "budget unchecked" not in chosen.document["notes"]


def test_a_remote_without_a_plan_cannot_win():
    naked = measured("remote/naked", "runpod:A40", throughput=80, worst=1)
    local = measured("local/a", "CPU", throughput=5)
    refused = decide([naked], ask())
    assert isinstance(refused, Refusal)
    assert refused.document["constraint"] == "remote_plan"
    chosen = decide([local, naked], ask())
    assert isinstance(chosen, Decision)
    assert chosen.document["device"] == "CPU"
    assert any(item["constraint"] == "remote_plan" for item in chosen.document["runners_up"])


def test_experiment_of_reads_the_spec_directory_only():
    assert experiment_of(None) == "unregistered"
    assert experiment_of({"provenance": {}}) == "unregistered"
    assert experiment_of({"provenance": {"experiment_spec": ""}}) == "unregistered"
    assert experiment_of({"provenance": {"experiment_spec": "experiments/e3-ref/spec.yaml"}}) == "E3-REF"
    assert experiment_of({"provenance": {"experiment_spec": "experiments\\e4\\spec.yaml"}}) == "E4"
    assert experiment_of({"provenance": {"experiment_spec": "experiments/"}}) == "unregistered"
    assert experiment_of({"provenance": {"experiment_spec": "receipts/x.json"}}) == "unregistered"
    assert experiment_of(measured("x", "CPU", spec="")) == "unregistered"


def test_mismatched_5090_rows_and_thin_runners_are_named():
    from switchyard.router.policy import _keeps

    igpu = measured(
        "ig",
        "iGPU",
        repo="phi:mini",
        thinking="off",
        quote_rule=2,
        num_predict=64,
        num_ctx=2048,
        throughput=10,
    )
    cards = [
        measured("c-think", "5090", repo="phi:mini", thinking="on", quote_rule=2, num_predict=64, num_ctx=2048, throughput=100),
        measured("c-l256/x", "5090", repo="phi:mini", thinking="off", quote_rule=2, num_predict=64, throughput=100),
        measured("c-quote", "5090", repo="phi:mini", thinking="off", quote_rule=1, num_predict=64, num_ctx=2048, throughput=100),
        measured("c-np", "5090", repo="phi:mini", thinking="off", quote_rule=2, num_predict=32, num_ctx=2048, throughput=100),
    ]
    missed = decide(cards + [igpu], ask(model="Phi", thinking="off", ctx=1024, quote_rule=2, num_predict=64))
    assert isinstance(missed, Decision)
    assert missed.document["device"] == "iGPU"
    assert missed.document["notes"] == ["no 5090 row"]

    wide = measured("wide", "iGPU", task="embed", num_ctx=4096, slots="empty", claim_wall=2)
    narrow = measured("narrow", "CPU", task="embed", num_ctx=128, slots="empty", claim_wall=1)
    bare = measured("bare-l256/x", "NPU", task="embed", slots="empty", claim_wall=1)
    lengths = decide([wide, narrow, bare], ask(task="embed", ctx=1024))
    assert isinstance(lengths, Decision)
    reasons = {item["id"]: item["reason"] for item in lengths.document["runners_up"]}
    assert reasons["narrow"] == "num_ctx 128 does not cover 1024"
    assert reasons["bare-l256/x"] == "measured length 256, requested 1024"

    good = measured("ok", "iGPU", quote_rule=2, num_predict=64, throughput=5)
    bad_quote = measured("bad-quote", "CPU", quote_rule=1, num_predict=64, throughput=50)
    bad_predict = measured("bad-predict", "NPU", quote_rule=2, num_predict=32, throughput=40)
    pinned = decide([good, bad_quote, bad_predict], ask(quote_rule=2, num_predict=64))
    assert isinstance(pinned, Decision)
    named = {item["id"]: item["constraint"] for item in pinned.document["runners_up"]}
    assert named["bad-quote"] == "quote_rule"
    assert named["bad-predict"] == "num_predict"

    ranked = measured("ranked", "CPU", throughput=10)
    unranked = measured("unranked", "iGPU", slots="empty")
    spread = decide([ranked, unranked], ask())
    assert isinstance(spread, Decision)
    assert "no conservative decode_tok_s bound" in next(
        item["reason"] for item in spread.document["runners_up"] if item["id"] == "unranked"
    )

    scored = measured(
        "scored",
        "CPU",
        task="rerank",
        slots="empty",
        claim_wall=1,
        quality=[{"name": "cosine", "interval": {"low": 0.9, "high": 1}}],
    )
    unscored = measured(
        "unscored",
        "iGPU",
        task="rerank",
        slots="empty",
        claim_wall=0.2,
        quality=[{"name": "cosine", "interval": {}}, "not-a-dict"],
    )
    kept = decide([scored, unscored], ask(task="rerank", min_metrics={"cosine": 0.5}))
    assert isinstance(kept, Decision)
    assert "no conservative bound for cosine" in next(
        item["reason"] for item in kept.document["runners_up"] if item["id"] == "unscored"
    )
    closest = decide([unscored], ask(task="rerank", min_metrics={"cosine": 0.5}))
    assert isinstance(closest, Refusal)
    assert closest.document["closest"]["bound"] is None

    timed = measured("timed", "CPU", latency=50)
    untimed = measured("untimed", "iGPU", slots="empty")
    latency = decide([timed, untimed], ask(max_latency_ms=1000))
    assert isinstance(latency, Decision)
    assert "no conservative latency bound" in next(
        item["reason"] for item in latency.document["runners_up"] if item["id"] == "untimed"
    )

    assert experiment_of({"provenance": {"experiment_spec": "experiments"}}) == "unregistered"
    shapeless = measured(
        "remote/shapeless",
        "runpod:A40",
        throughput=80,
        plan={"profile": "job", "max_hours": "1"},
    )
    refused = decide([shapeless], ask())
    assert isinstance(refused, Refusal)
    assert refused.document["constraint"] == "remote_plan"
    assert _keeps(ranked, ask(), "not-a-constraint", None) is True


def test_the_policy_does_not_read_the_clock_or_a_file():
    text = (ROOT / "src" / "switchyard" / "router" / "policy.py").read_text(encoding="utf-8")
    assert "date.today" not in text
    assert "datetime.now" not in text
    assert "open(" not in text
