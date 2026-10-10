"""The OpenVINO adapter, exercised through fakes. CI has no intel extra; these
tests must pass without it, and skip-guard what genuinely needs the hardware."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest

from switchyard.adapters import registry
from switchyard.adapters.base import AdapterError, NativeBinaryRejected, StepRequest
from switchyard.adapters import openvino as ovm


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        self.t += 0.01
        return self.t


class FakeTokenizer:
    pad_token_id = 0
    model_max_length = 512
    eos_token_id = 102

    def __call__(self, text: str, return_attention_mask: bool = True):
        ids = [101] + [abs(hash(w)) % 30000 + 1 for w in str(text).split()] + [102]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class FakeCompiled:
    def __init__(self, fail=False, msg="device exploded"):
        self.calls = 0
        self.fail = fail
        self.msg = msg

    def infer_lists(self, inputs, batch_rows: int):
        self.calls += 1
        if self.fail:
            raise OSError(self.msg)
        mask = inputs["attention_mask"]
        n = sum(mask)
        # one row per position, like a real last_hidden_state — last-token pooling needs it
        return [[float(n), 1.0, 2.0]] * max(len(mask), 1)


class FakeGraph:
    def __init__(self):
        self.inputs = [_FakePort(), _FakePort()]
        self.reshape_args = None

    def reshape(self, shapes):
        self.reshape_args = {str(k): v for k, v in shapes.items()}


class _FakePort:
    def get_partial_shape(self):
        return ["?", "?"]


class FakeCore:
    def __init__(self, fail_compile=False):
        self.available_devices = ["CPU", "GPU.0", "GPU.1", "NPU"]
        self.compiled: list[tuple] = []
        self.graphs: list[FakeGraph] = []
        self.fail_compile = fail_compile

    def get_property(self, dev, prop):
        return {
            "CPU": "Intel Core Ultra",
            "GPU.0": "Intel(R) Graphics (iGPU)",
            "GPU.1": "NVIDIA GeForce RTX 5090",
            "NPU": "Intel AI Boost",
        }[dev]

    def read_model(self, path):
        graph = FakeGraph()
        self.graphs.append(graph)
        return graph

    def compile_model(self, graph, device, config):
        if self.fail_compile:
            raise OSError("compile blew up")
        self.compiled.append((device, dict(config)))
        return FakeCompiled()


class FakePerf:
    def get_ttft(self):
        return 12.5

    def get_tpot(self):
        return 3.0

    def get_throughput(self):
        return 33.0

    def get_num_input_tokens(self):
        return 7

    def get_num_generated_tokens(self):
        return 8


class FakePipeline:
    generated: list[str] = []

    def __init__(self, path, device, config):
        self.path = path
        self.device = device
        self.config = dict(config)

    def generate(self, prompt, config=None):
        # The real LLMPipeline returns a plain str for a str prompt and
        # DecodedResults (texts, perf_metrics) only for a list of prompts.
        self.generated.append(prompt)
        if isinstance(prompt, str):
            return f"text:{prompt}"
        return type("Decoded", (), {"texts": [f"text:{p}" for p in prompt], "perf_metrics": FakePerf()})()


class FakeGenai:
    class GenerationConfig:
        def __init__(self):
            self.max_new_tokens = 0
            self.do_sample = False
            self.temperature = 1.0

    LLMPipeline = FakePipeline


def make_adapter(**over) -> ovm.OpenvinoAdapter:
    kwargs = dict(core=FakeCore(), tokenizer_loader=lambda m, r: FakeTokenizer(),
                  genai_module=FakeGenai, clock=FakeClock())
    kwargs.update(over)
    return ovm.OpenvinoAdapter(**kwargs)


def req(task="embed", model="acme/embed@deadbeef", device="CPU", engine="openvino", **settings):
    return StepRequest(step_id="a1", task=task, model=model, device=device, engine=engine, settings=settings)


# --- pure helpers ---------------------------------------------------------


def test_parse_model_ref():
    assert ovm.parse_model_ref("acme/embed") == ("acme/embed", None)
    assert ovm.parse_model_ref("acme/embed@abc123") == ("acme/embed", "abc123")


def test_parse_model_ref_existing_path(tmp_path):
    assert ovm.parse_model_ref(str(tmp_path)) == (str(tmp_path), None)


def test_parse_static_shape():
    assert ovm.parse_static_shape(None) is None
    assert ovm.parse_static_shape("1x128") == (1, 128)
    assert ovm.parse_static_shape([8, 256]) == (8, 256)
    with pytest.raises(AdapterError):
        ovm.parse_static_shape("128")
    with pytest.raises(AdapterError):
        ovm.parse_static_shape("0x128")


def test_quant_of():
    assert ovm.quant_of("m-int4-ov", None) == "int4"
    assert ovm.quant_of("m", "FP16") == "fp16"
    assert ovm.quant_of("m", None) == "unrecorded"


def test_pad_truncate():
    ids, mask, dropped = ovm.pad_truncate([1, 2], 4, 0)
    assert ids == [1, 2, 0, 0] and mask == [1, 1, 0, 0] and dropped == 0
    ids, mask, dropped = ovm.pad_truncate([1, 2, 3, 4, 5], 3, 0)
    assert ids == [1, 2, 3] and mask == [1, 1, 1] and dropped == 2
    with pytest.raises(AdapterError):
        ovm.pad_truncate([1, 2, 3], 2, 0, allow_truncate=False)


def test_pool_and_normalize():
    matrix = [[1.0, 2.0], [3.0, 4.0], [9.0, 9.0]]
    vec = ovm.pool(matrix, [1, 1, 0], "mean")
    assert vec[0] == pytest.approx(2.0) and vec[1] == pytest.approx(3.0)
    assert ovm.pool(matrix, [1, 1, 0], "cls")[0] == pytest.approx(1.0)
    assert ovm.pool([[5.0]], [1], "none")[0] == pytest.approx(5.0)
    with pytest.raises(AdapterError):
        ovm.pool(matrix, [0, 0, 0], "mean")
    out = ovm.normalize([3.0, 4.0])
    assert (out[0] ** 2 + out[1] ** 2) == pytest.approx(1.0)
    assert ovm.normalize([0.0, 0.0]) == [0.0, 0.0]


# --- embed steps ----------------------------------------------------------


def local_model(tmp_path):
    (tmp_path / "openvino_model.xml").write_text("<xml/>", encoding="utf-8")
    return str(tmp_path)


def test_embed_step_ok_and_compile_once(tmp_path, monkeypatch):
    a = make_adapter()
    core = a._injected_core
    m = local_model(tmp_path)
    monkeypatch.setitem(ovm._POOLING, m, "cls")
    r1 = a.run(req(model=m, device="NPU", static_shape="1x128"))
    assert r1.ok and r1.latency_ms is not None
    assert r1.throughput_unit == "items/s" and r1.throughput and r1.throughput > 0
    assert r1.compile_ms is not None and r1.first_call_ms is not None
    assert r1.resolved["pooling"] == "cls" and r1.resolved["num_ctx"] == 512
    assert len(r1.output) == 3
    assert core.graphs[-1].reshape_args  # static reshape happened
    r2 = a.run(req(model=m, device="NPU", static_shape="1x128"))
    assert r2.compile_ms is None and r2.first_call_ms is None  # cached entry


def test_embed_revision_pinned(tmp_path, monkeypatch):
    a = make_adapter()
    monkeypatch.setitem(ovm._POOLING, "acme/missing-embed", "mean")
    # hub ids without network are refused before measuring
    with pytest.raises(AdapterError):
        a.run(req(model="acme/missing-embed@abc", device="CPU"))
    m = local_model(tmp_path)
    monkeypatch.setitem(ovm._POOLING, m, "mean")
    r = a.run(req(model=m, device="CPU"))
    assert r.resolved["revision"] == "unrecorded"


def test_embed_refusals(tmp_path):
    a = make_adapter()
    with pytest.raises(AdapterError):
        a.run(req(model=local_model(tmp_path), device="NPU"))  # NPU needs static_shape
    with pytest.raises(AdapterError):
        a.run(req(model=local_model(tmp_path), device="GPU.1"))  # the 5090
    with pytest.raises(AdapterError):
        a.run(req(model=local_model(tmp_path), device="wristwatch"))
    with pytest.raises(AdapterError):
        a.run(req(task="sculpt", model=local_model(tmp_path)))
    with pytest.raises(AdapterError):
        a.run(req(model=local_model(tmp_path), engine="ollama"))


def test_count_tokens_is_the_encode_and_a_short_prompt_records_zero_dropped(tmp_path, monkeypatch):
    a = make_adapter()
    m = local_model(tmp_path)
    monkeypatch.setitem(ovm._POOLING, m, "cls")
    assert a.count_tokens(m, "one two") == 4
    short = a.run(req(model=m, device="CPU", static_shape="1x8", prompt="one two"))
    assert short.resolved["prompt_tokens"] == 4
    assert short.resolved["truncated_tokens"] == 0
    long_prompt = " ".join(f"w{i}" for i in range(20))
    long = a.run(req(model=m, device="NPU", static_shape="1x8", prompt=long_prompt))
    assert long.resolved["prompt_tokens"] == a.count_tokens(m, long_prompt)
    assert long.resolved["truncated_tokens"] == long.resolved["prompt_tokens"] - 8


def test_count_tokens_reads_the_three_value_encode(tmp_path):
    """count_tokens uses _encode's (ids, mask, token_type_ids) and counts the ids."""

    class TTTokenizer(FakeTokenizer):
        def __call__(self, text, return_attention_mask=True):
            enc = dict(super().__call__(text, return_attention_mask))
            enc["token_type_ids"] = [7] * len(enc["input_ids"])
            return enc

    a = make_adapter(tokenizer_loader=lambda _model, _ref: TTTokenizer())
    m = local_model(tmp_path)
    repo, ref = ovm.parse_model_ref(m)
    ids, mask, token_types = a._encode(a._load_tokenizer(repo, ref), "one two")
    assert len(ids) == len(mask) == len(token_types) == 4
    assert token_types == [7, 7, 7, 7]
    assert a.count_tokens(m, "one two") == len(ids)


def test_embed_truncate_false_refuses(tmp_path, monkeypatch):
    a = make_adapter()
    m = local_model(tmp_path)
    monkeypatch.setitem(ovm._POOLING, m, "cls")
    long_prompt = " ".join(f"w{i}" for i in range(600))
    with pytest.raises(AdapterError):
        a.run(req(model=local_model(tmp_path), device="NPU", static_shape="1x8",
                  prompt=long_prompt, truncate=False))
    r = a.run(req(model=m, device="NPU", static_shape="1x8", prompt=long_prompt))
    assert r.ok and r.resolved["truncated_tokens"] > 0


def test_compile_cache_and_precision_config(tmp_path, monkeypatch):
    a = make_adapter()
    m = local_model(tmp_path)
    monkeypatch.setitem(ovm._POOLING, m, "mean")
    a.run(req(model=m, device="CPU", static_shape="1x16",
              precision="fp16", compile_cache="fresh", compile_cache_dir="D:/cache/x"))
    _dev, config = a._injected_core.compiled[-1]
    assert config["CACHE_DIR"] == "D:/cache/x"
    assert config["INFERENCE_PRECISION_HINT"] == "fp16"


def test_nli_pooling_none(tmp_path):
    a = make_adapter()
    r = a.run(req(task="nli", model=local_model(tmp_path), device="CPU", static_shape="1x16", batch=1))
    assert r.ok and r.resolved["pooling"] == "none"


def test_measurement_failure_is_ok_false(tmp_path, monkeypatch):
    m = local_model(tmp_path)
    monkeypatch.setitem(ovm._POOLING, m, "cls")
    a = make_adapter(core=FakeCore(fail_compile=True))
    r = a.run(req(model=m, device="CPU"))
    assert r.ok is False and "compile blew up" in r.error


# --- generate steps -------------------------------------------------------


def test_generate_step(tmp_path):
    a = make_adapter()
    r = a.run(req(task="generate", model=str(tmp_path), device="GPU.0", num_predict=16, prompt="hi"))
    assert r.ok and r.throughput_unit == "tok/s" and r.throughput == pytest.approx(33.0)
    assert r.output == "text:hi"
    assert r.compile_ms is not None and r.first_call_ms is not None
    genai = r.resolved["genai"]
    assert genai["ttft_ms"] == pytest.approx(12.5) and genai["generated_tokens"] == 8
    # A str prompt would get a bare str back, with no perf_metrics (E3's first run).
    assert FakePipeline.generated[-1] == ["hi"]
    r2 = a.run(req(task="generate", model=str(tmp_path), device="GPU.0", prompt="hi"))
    assert r2.compile_ms is None  # pipeline cached


def test_generate_records_thinking_and_temperature(tmp_path):
    a = make_adapter()
    r = a.run(req(task="generate", model=str(tmp_path), thinking_level="low", temperature=0.2))
    assert r.ok and r.resolved["thinking_level"] == "low"


def test_generate_refusals_and_missing_genai(tmp_path):
    a = make_adapter(genai_module=None)
    if importlib.util.find_spec("openvino_genai") is None:
        with pytest.raises(AdapterError):
            a.run(req(task="generate", model=str(tmp_path)))
    with pytest.raises(AdapterError):
        a.run(req(task="generate", model=str(tmp_path), device="GPU.1"))


# --- sessions and unload --------------------------------------------------


def test_session_lifecycle_embed(tmp_path):
    import time as _time

    a = make_adapter()
    m = local_model(tmp_path)
    r = req(model=m, device="NPU", static_shape="1x64")
    a.start_session(r)
    _time.sleep(0.05)
    with pytest.raises(AdapterError):
        a.unload(m)  # session still running
    a.stop_session(r)
    a.unload(m)  # now fine
    assert a._session is None


def test_session_double_start_and_wrong_stop(tmp_path):
    a = make_adapter()
    m1, m2 = local_model(tmp_path / "a" if False else tmp_path), None
    r1 = req(model=m1, device="CPU", static_shape="1x16")
    a.start_session(r1)
    try:
        with pytest.raises(AdapterError):
            a.start_session(r1)  # already running
        with pytest.raises(AdapterError):
            a.stop_session(req(model="someone-else"))
    finally:
        a.stop_session(r1)


def test_session_start_fault_raises():
    a = make_adapter(core=FakeCore(fail_compile=True))
    with pytest.raises(AdapterError):
        a.start_session(req(model="/nonexistent/dir/xyz", device="CPU"))


def test_unload_drops_compiled_and_tokenizer(tmp_path, monkeypatch):
    a = make_adapter()
    m = local_model(tmp_path)
    monkeypatch.setitem(ovm._POOLING, m, "cls")
    a.run(req(model=m, device="CPU", static_shape="1x16"))
    assert a._compiled and a._tokenizers
    a.unload(m)
    assert not a._compiled and not a._tokenizers
    r = a.run(req(model=m, device="CPU", static_shape="1x16"))
    assert r.compile_ms is not None  # recompiled after unload


# --- registry / import contract -------------------------------------------


def test_registry_finds_the_adapter():
    a = registry.for_engine("openvino")
    assert isinstance(a, ovm.OpenvinoAdapter)
    assert callable(ovm.adapter)
    assert isinstance(ovm.adapter(), ovm.OpenvinoAdapter)


def test_enumerate_devices_shape():
    class _C:
        available_devices = ["CPU"]

        def get_property(self, d, p):
            return "a cpu"

    assert ovm.OpenvinoAdapter.enumerate_devices(_C()) == [{"id": "CPU", "full_name": "a cpu"}]


@pytest.mark.skipif(importlib.util.find_spec("openvino") is None, reason="no intel extra in this venv")
def test_real_core_enumerates_this_rig():
    a = ovm.adapter()
    ids = [d["id"] for d in ovm.OpenvinoAdapter.enumerate_devices(a._core())]
    assert "CPU" in ids and "NPU" in ids


def test_embed_unknown_repo_refused(tmp_path, monkeypatch):
    a = make_adapter()
    monkeypatch.setitem(ovm._POOLING, "acme/known", "cls")
    m = local_model(tmp_path)  # a local dir with no recipe on record
    with pytest.raises(AdapterError, match="pooling recipe"):
        a.run(req(model=m, device="CPU", static_shape="1x16"))


def test_pooling_table_matches_the_measured_recipes():
    assert ovm._POOLING["BAAI/bge-base-en-v1.5"] == "cls"
    assert ovm._POOLING["BAAI/bge-small-en-v1.5"] == "cls"
    assert ovm._POOLING["nomic-ai/nomic-embed-text-v1.5"] == "mean"
    assert ovm._pooling_for("nli", "cross-encoder/nli-deberta-v3-base") == "none"


# --- per-call timeout and warm-reload colds (E1 review) --------------------


class SlowCompiled(FakeCompiled):
    def infer_lists(self, inputs, batch_rows: int):
        import time as _t

        _t.sleep(1.0)
        return super().infer_lists(inputs, batch_rows)


class SlowCore(FakeCore):
    def compile_model(self, graph, device, config):
        self.compiled.append((device, dict(config)))
        return SlowCompiled()


def test_hang_marks_inconclusive_and_poisons(tmp_path, monkeypatch):
    m = local_model(tmp_path)
    monkeypatch.setitem(ovm._POOLING, m, "cls")
    a = make_adapter(core=SlowCore(), call_timeout_s=0.05)
    r = a.run(req(model=m, device="NPU", static_shape="1x16"))
    assert r.ok is False and r.error.startswith("hang:")
    assert "inconclusive" in r.error
    with pytest.raises(AdapterError, match="poisoned"):
        a.run(req(model=m, device="NPU", static_shape="1x16"))


def test_hang_drops_nothing_for_other_devices(tmp_path, monkeypatch):
    m = local_model(tmp_path)
    monkeypatch.setitem(ovm._POOLING, m, "cls")
    core = FakeCore()
    a = make_adapter(core=core)
    ok = a.run(req(model=m, device="CPU", static_shape="1x16"))
    assert ok.ok
    a._poisoned.add((m, "NPU"))  # an NPU hang must not evict the CPU entry
    again = a.run(req(model=m, device="CPU", static_shape="1x16"))
    assert again.ok and again.compile_ms is None


def test_warm_reload_compile_recorded_on_fresh_cache(tmp_path, monkeypatch):
    m = local_model(tmp_path)
    monkeypatch.setitem(ovm._POOLING, m, "cls")
    a = make_adapter()
    r = a.run(req(model=m, device="NPU", static_shape="1x16",
                  compile_cache="fresh", compile_cache_dir="D:/cache/e1-x"))
    assert r.ok and r.compile_ms is not None
    assert r.resolved["compile_warm_reload_ms"] is not None
    assert a._injected_core.compiled  # first compile
    assert len([c for c in a._injected_core.compiled if c[1].get("CACHE_DIR")]) == 2  # fresh + reload
    r2 = a.run(req(model=m, device="NPU", static_shape="1x16",
                   compile_cache="fresh", compile_cache_dir="D:/cache/e1-x"))
    assert "compile_warm_reload_ms" not in (r2.resolved or {})  # only the cold row carries it


def test_no_warm_reload_without_cache_dir(tmp_path, monkeypatch):
    m = local_model(tmp_path)
    monkeypatch.setitem(ovm._POOLING, m, "cls")
    a = make_adapter()
    r = a.run(req(model=m, device="CPU", static_shape="1x16"))
    assert r.ok and "compile_warm_reload_ms" not in r.resolved


def test_generate_hang_poisons_pipeline(tmp_path):
    class SlowGenai(FakeGenai):
        class LLMPipeline(FakePipeline):
            def generate(self, prompt, config=None):
                import time as _t

                _t.sleep(1.0)
                return super().generate(prompt, config)

    a = make_adapter(genai_module=SlowGenai, call_timeout_s=0.05)
    r = a.run(req(task="generate", model=str(tmp_path), device="CPU"))
    assert r.ok is False and r.error.startswith("hang:")


def test_session_companion_hang_surfaces_at_stop(tmp_path, monkeypatch):
    import time as _time

    m = local_model(tmp_path)
    monkeypatch.setitem(ovm._POOLING, m, "cls")
    a = make_adapter(core=SlowCore(), call_timeout_s=0.05)
    r = req(model=m, device="CPU", static_shape="1x16")
    # the hang surfaces at start on a fast machine (fault seen during the
    # settle window), at stop on a slower one — both are correct
    with pytest.raises(AdapterError, match="hang"):
        a.start_session(r)
        _time.sleep(0.2)
        if a._session is not None:
            a.stop_session(r)
    assert a._session is None
def test_pool_last_token():
    matrix = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    assert ovm.pool(matrix, [1, 1, 0], "last") == [2.0, 2.0]
    assert ovm.pool(matrix, [1, 1, 1], "last") == [3.0, 3.0]
    with pytest.raises(AdapterError):
        ovm.pool(matrix, [0, 0, 0], "last")
    assert ovm._POOLING["OpenVINO/Qwen3-Embedding-0.6B-fp16-ov"] == "last"


def test_last_pooling_records_eos_state(tmp_path, monkeypatch):
    m = local_model(tmp_path)
    monkeypatch.setitem(ovm._POOLING, m, "last")
    a = make_adapter()
    r = a.run(req(model=m, device="CPU", static_shape="1x64", prompt="two words here"))
    assert r.ok and r.resolved["pooling"] == "last"
    assert r.resolved["eos_appended"] is True and r.resolved["last_row_is_eos"] is True
    long_prompt = " ".join(f"w{i}" for i in range(600))
    r2 = a.run(req(model=m, device="CPU", static_shape="1x8", prompt=long_prompt))
    assert r2.resolved["eos_appended"] is True      # the tokenizer appended it
    assert r2.resolved["last_row_is_eos"] is False  # truncation cut it off


# --- NPU genai pipeline sizing (E3 review) --------------------------------


def test_generate_npu_records_genai_config(tmp_path):
    a = make_adapter()
    r = a.run(req(task="generate", model=str(tmp_path), device="NPU", num_predict=64, prompt="hi there"))
    assert r.ok
    # FakeTokenizer: "hi there" -> [101, w1, w2, 102] = 4 tokens + 64 margin
    assert r.resolved["genai_config"] == {"MAX_PROMPT_LEN": 68, "MIN_RESPONSE_LEN": 96}
    (entry,) = [v for k, v in a._pipelines.items() if k[:2] == (str(tmp_path), "NPU")]
    assert entry["pipe"].config["MAX_PROMPT_LEN"] == 68


def test_generate_npu_builds_a_pipeline_per_prompt_size(tmp_path):
    # E3's second run: a pipeline sized for a short prompt was reused for a long one,
    # and the NPU refused the long prompt (MAX_PROMPT_LEN too small).
    a = make_adapter()
    short = a.run(req(task="generate", model=str(tmp_path), device="NPU", num_predict=64, prompt="hi there"))
    long = a.run(req(task="generate", model=str(tmp_path), device="NPU", num_predict=64,
                     prompt="hi there " * 20))
    assert short.ok and long.ok
    assert long.compile_ms is not None  # a new pipeline, not the short one reused
    sizes = sorted(v["pipe"].config["MAX_PROMPT_LEN"] for k, v in a._pipelines.items() if k[:2] == (str(tmp_path), "NPU"))
    assert sizes == [68, long.resolved["genai_config"]["MAX_PROMPT_LEN"]] and sizes[1] > 68
    again = a.run(req(task="generate", model=str(tmp_path), device="NPU", num_predict=64, prompt="hi there"))
    assert again.compile_ms is None  # the short pipeline is still cached


def test_generate_npu_unsupported_length_is_recorded_not_fatal(tmp_path):
    class NpuFailGenai(FakeGenai):
        class LLMPipeline(FakePipeline):
            def __init__(self, path, device, config):
                if device == "NPU":
                    raise RuntimeError("programmatic crash at 4096")
                super().__init__(path, device, config)

    a = make_adapter(genai_module=NpuFailGenai)
    r = a.run(req(task="generate", model=str(tmp_path), device="NPU", prompt="hi"))
    assert r.ok is True
    assert r.latency_ms is None and r.throughput is None
    assert r.resolved["outcome"] == "unsupported length"
    # the same model builds fine elsewhere — the marker is NPU-specific
    r2 = a.run(req(task="generate", model=str(tmp_path), device="CPU", prompt="hi"))
    assert r2.ok and r2.latency_ms is not None


def test_generate_non_npu_has_no_genai_config(tmp_path):
    a = make_adapter()
    r = a.run(req(task="generate", model=str(tmp_path), device="CPU"))
    assert r.ok and "genai_config" not in r.resolved
# --- device loss (Level Zero DEVICE_LOST, seen live 2026-10-09) ------------

LOST_MSG = "infer request failed: ZE_RESULT_ERROR_DEVICE_LOST: device hung or reset"


class LostCore(FakeCore):
    def compile_model(self, graph, device, config):
        if device == "NPU":
            raise RuntimeError(LOST_MSG)
        return super().compile_model(graph, device, config)


class LostInferCore(FakeCore):
    def compile_model(self, graph, device, config):
        self.compiled.append((device, dict(config)))
        if device == "NPU":
            return FakeCompiled(fail=True, msg=LOST_MSG)
        return FakeCompiled()


def test_device_loss_at_compile_is_a_recorded_result(tmp_path, monkeypatch):
    m = local_model(tmp_path)
    monkeypatch.setitem(ovm._POOLING, m, "cls")
    a = make_adapter(core=LostCore())
    r = a.run(req(model=m, device="NPU", static_shape="1x16"))
    assert r.ok is True
    assert r.resolved["outcome"] == "device_lost"
    assert r.latency_ms is None and r.throughput is None
    assert r.compile_ms is None and r.first_call_ms is None
    assert not a._poisoned  # a device loss is not a poison: the pair is recorded, not refused


def test_device_loss_blocks_only_that_device(tmp_path, monkeypatch):
    m = local_model(tmp_path)
    monkeypatch.setitem(ovm._POOLING, m, "cls")
    a = make_adapter(core=LostCore())
    first = a.run(req(model=m, device="NPU", static_shape="1x16"))
    assert first.resolved["outcome"] == "device_lost"
    compiles_after_loss = len(a._injected_core.compiled)
    rest = a.run(req(model=m, device="NPU", static_shape="1x32"))
    assert rest.ok and rest.resolved["outcome"] == "not run"
    assert rest.latency_ms is None and rest.throughput is None
    assert len(a._injected_core.compiled) == compiles_after_loss  # nothing touched the NPU again
    cpu = a.run(req(model=m, device="CPU", static_shape="1x16"))
    assert cpu.ok and cpu.latency_ms is not None  # other devices run on


def test_device_loss_mid_infer_records_and_purges(tmp_path, monkeypatch):
    m = local_model(tmp_path)
    monkeypatch.setitem(ovm._POOLING, m, "cls")
    a = make_adapter(core=LostInferCore())
    r = a.run(req(model=m, device="NPU", static_shape="1x16"))
    assert r.ok and r.resolved["outcome"] == "device_lost"
    assert r.latency_ms is None and r.throughput is None
    # the dead compiled entry is dropped, not kept as if still loaded
    assert not [key for key in a._compiled if key[2] == "NPU"]


def test_non_loss_failures_keep_their_old_shape(tmp_path, monkeypatch):
    m = local_model(tmp_path)
    monkeypatch.setitem(ovm._POOLING, m, "cls")
    a = make_adapter(core=FakeCore(fail_compile=True))
    r = a.run(req(model=m, device="CPU", static_shape="1x16"))
    assert r.ok is False and "compile blew up" in r.error
    assert a._lost_devices == set()


def test_device_loss_generate_build_is_a_result_not_unsupported_length(tmp_path):
    class LostGenai(FakeGenai):
        class LLMPipeline(FakePipeline):
            def __init__(self, path, device, config):
                if device == "NPU":
                    raise RuntimeError(LOST_MSG)
                super().__init__(path, device, config)

    a = make_adapter(genai_module=LostGenai)
    r = a.run(req(task="generate", model=str(tmp_path), device="NPU", prompt="hi"))
    assert r.ok is True and r.resolved["outcome"] == "device_lost"
    assert r.latency_ms is None and r.throughput is None
    assert (str(tmp_path), "NPU") not in a._pipelines
    cpu = a.run(req(task="generate", model=str(tmp_path), device="CPU", prompt="hi"))
    assert cpu.ok and cpu.latency_ms is not None


def test_device_loss_generate_call_marks_and_skips(tmp_path):
    class LostCallGenai(FakeGenai):
        class LLMPipeline(FakePipeline):
            def generate(self, prompt, config=None):
                if self.device == "NPU":
                    raise RuntimeError(LOST_MSG)
                return super().generate(prompt, config)

    a = make_adapter(genai_module=LostCallGenai)
    r = a.run(req(task="generate", model=str(tmp_path), device="NPU", prompt="hi"))
    assert r.ok is True and r.resolved["outcome"] == "device_lost"
    assert (str(tmp_path), "NPU") not in a._pipelines
    rest = a.run(req(task="generate", model=str(tmp_path), device="NPU", prompt="hi there"))
    assert rest.ok and rest.resolved["outcome"] == "not run"


REJECTED_MSG = (
    "L0 pfnAppendGraphExecute: ZE_RESULT_ERROR_INVALID_NATIVE_BINARY, "
    "code 0x7800000f - native binary is not supported by the device"
)


class RejectedInferCore(FakeCore):
    """The E2 failure: compile writes a blob, execute refuses the binary."""

    def compile_model(self, graph, device, config):
        self.compiled.append((device, dict(config)))
        if device == "NPU":
            return FakeCompiled(fail=True, msg=REJECTED_MSG)
        return FakeCompiled()


def test_native_binary_rejected_raises_with_the_compile_time(tmp_path, monkeypatch):
    m = local_model(tmp_path)
    monkeypatch.setitem(ovm._POOLING, m, "cls")
    a = make_adapter(core=RejectedInferCore())
    with pytest.raises(NativeBinaryRejected, match="0x7800000f") as caught:
        a.run(req(model=m, device="NPU", static_shape="1x16"))
    assert caught.value.compile_s is not None
    assert a._lost_devices == set()  # a rejection is not a device loss
    cpu = a.run(req(model=m, device="CPU", static_shape="1x16"))
    assert cpu.ok and cpu.latency_ms is not None  # the other devices run on


def test_native_binary_rejected_generate_raises_with_the_compile_time(tmp_path):
    class RejectedCallGenai(FakeGenai):
        class LLMPipeline(FakePipeline):
            def generate(self, prompt, config=None):
                if self.device == "NPU":
                    raise RuntimeError(REJECTED_MSG)
                return super().generate(prompt, config)

    a = make_adapter(genai_module=RejectedCallGenai)
    with pytest.raises(NativeBinaryRejected, match="0x7800000f") as caught:
        a.run(req(task="generate", model=str(tmp_path), device="NPU", prompt="hi"))
    assert caught.value.compile_s is not None
    cpu = a.run(req(task="generate", model=str(tmp_path), device="CPU", prompt="hi"))
    assert cpu.ok and cpu.latency_ms is not None


def test_device_loss_in_companion_start_marks_the_device(tmp_path, monkeypatch):
    m = local_model(tmp_path)
    monkeypatch.setitem(ovm._POOLING, m, "cls")
    a = make_adapter(core=LostCore())
    # the fault still surfaces: concurrent measurements are invalid without the load
    with pytest.raises(AdapterError, match="failed to start"):
        a.start_session(req(model=m, device="NPU", static_shape="1x16"))
    # but later one-shot steps on that device are recorded, not attempted
    r = a.run(req(model=m, device="NPU", static_shape="1x16"))
    assert r.ok and r.resolved["outcome"] == "not run"
# --- through the harness: the adapter's device-loss outcomes become rows/notes ---------------
# The runner side of this contract is Grok's (#19/#20, tests in test_harness.py). These two prove
# the ADAPTER's outcomes land in it: a resolved outcome of device_lost becomes one capability row
# and marks the device off for the runner, and a resolved outcome of "not run" writes no row.

def _repo(repo_path):
    import yaml
    repo_path.mkdir(parents=True, exist_ok=True)
    for argv in (["init"], ["config", "user.email", "dev@example.com"], ["config", "user.name", "dev"],
                 ["config", "core.autocrlf", "false"]):
        subprocess.run(["git", *argv], cwd=repo_path, check=True, capture_output=True)
    return repo_path


def _commit_spec(repo, spec: dict):
    import yaml
    path = repo / "experiments" / "t" / "spec.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8", newline="\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=repo, check=True, capture_output=True)
    return path


def _harness_arm(arm_id, device, model, shape):
    return {"id": arm_id, "task": "embed", "device": device, "engine": "openvino",
            "model": model, "revision_or_digest": "unrecorded", "quant": "unrecorded",
            "licence": "unrecorded", "engine_version": "2026.4.1", "thinking_level": "off",
            "num_predict": None, "num_ctx": None, "batch": 1, "static_shape": shape,
            "precision": None}


def _harness_spec(arms):
    return {"id": "T", "procedure": "interleaved",
            "question": "Do adapter-reported outcomes become rows and notes?",
            "decision": "A device loss is one capability row; a not run step is only a note.",
            "needs": {"grant": False, "npu_session": True, "cpu_window": True},
            "stop_rules": ["a failed step stops the sequence"],
            "dataset": {"name": "fixture"}, "metrics": ["latency_ms"],
            "warmups": 0, "repeats": 1, "band": 0.1, "compile_cache": "reuse", "rest_s": 900,
            "arms": arms}


def _run_harness(tmp_path, adapter, arms, notes=""):
    from switchyard.harness.runner import run_interleaved
    from switchyard.receipt import load_jsonl

    repo = _repo(tmp_path / "repo")
    spec_path = _commit_spec(repo, _harness_spec(arms))
    (repo / "table").mkdir()
    rows = run_interleaved({"openvino": adapter}, grant=None, npu_session="t", cpu_window="t",
                           root=repo, repo=repo, spec_path=spec_path, run_id="run-1",
                           date="2026-10-09", notes=notes)
    table = load_jsonl(repo / "table" / "routing.jsonl")
    return {row["id"]: row for row in rows}, {row["id"]: row for row in table}


def _row_model() -> Path:
    """A model directory whose path is allowed on a routing row.

    pytest's tmp_path on Windows sits under the user profile, and a receipt
    refuses a home-directory path. The system temp directory does not.
    """

    import os
    import tempfile

    base = Path(tempfile.gettempdir())
    home = Path.home()
    if base == home or home in base.parents:
        windir = os.environ.get("WINDIR")
        if windir:
            base = Path(windir) / "Temp"
    directory = Path(tempfile.mkdtemp(prefix="sy-model-", dir=base))
    (directory / "openvino_model.xml").write_text("<xml/>", encoding="utf-8")
    return directory


def test_runner_makes_a_capability_row_of_the_adapters_device_loss(tmp_path, monkeypatch):
    model_dir = _row_model()
    m = str(model_dir)
    monkeypatch.setitem(ovm._POOLING, m, "cls")
    adapter = make_adapter(core=LostInferCore())
    arms = [_harness_arm("A", "NPU", m, "1x16"),
            _harness_arm("B", "NPU", m, "1x32"),
            _harness_arm("C", "CPU", m, "1x16")]
    rows, table = _run_harness(tmp_path, adapter, arms)
    assert set(rows) == {"T/A/run-1", "T/C/run-1"}  # B was never measured and has no row
    lost = rows["T/A/run-1"]
    assert lost["outcome"] == "device_lost"
    assert all(value is None for value in lost["latency"].values())
    assert "artifacts" not in lost
    assert "can't do this setting" in lost["notes"]
    assert rows["T/C/run-1"]["latency"]["cold_ms"] is not None  # the other device ran on
    notes = rows["T/C/run-1"]["notes"]
    assert "not run:" in notes and "B cold" in notes  # the skipped arm is listed in the run notes
    assert set(table) == set(rows)
    import shutil
    shutil.rmtree(model_dir, ignore_errors=True)


def test_runner_writes_no_row_for_the_adapters_not_run(tmp_path, monkeypatch):
    model_dir = _row_model()
    m = str(model_dir)
    monkeypatch.setitem(ovm._POOLING, m, "cls")
    adapter = make_adapter()
    adapter._lost_devices.add("NPU")  # the device was already lost earlier in the run
    arms = [_harness_arm("B", "NPU", m, "1x16"),
            _harness_arm("A", "CPU", m, "1x16")]
    rows, table = _run_harness(tmp_path, adapter, arms, notes="kept")
    assert set(rows) == {"T/A/run-1"}  # B's step wrote no measured row
    assert rows["T/A/run-1"]["latency"]["cold_ms"] is not None
    notes = rows["T/A/run-1"]["notes"]
    assert notes.startswith("kept. not run: B cold") and "B repeat 0" in notes
    assert set(table) == set(rows)
    import shutil
    shutil.rmtree(model_dir, ignore_errors=True)


# --- token_type_ids: an omitted declared input infers garbage (E1 rerun, 2026-10-09) ---


_NEEDS_NUMPY = pytest.mark.skipif(importlib.util.find_spec("numpy") is None,
                                  reason="the real infer path needs numpy (intel extra)")


class _NamedPort:
    def __init__(self, name):
        self._name = name

    def get_any_name(self):
        return self._name


class TTCompiled:
    """CompiledModel-shaped stand-in for the real (numpy) infer path."""

    def __init__(self, declare_tt=True):
        names = ["input_ids", "attention_mask"] + (["token_type_ids"] if declare_tt else [])
        self.inputs = [_NamedPort(n) for n in names]
        self.outputs = ["last_hidden_state"]
        self.arrays = None

    def __call__(self, arrays):
        import numpy as np

        self.arrays = {k: np.asarray(v) for k, v in arrays.items()}
        length = self.arrays["input_ids"].shape[1]
        return {"last_hidden_state": np.ones((1, length, 4), dtype=np.float32)}


@_NEEDS_NUMPY
def test_infer_embed_feeds_token_type_ids_when_declared():
    a = make_adapter()
    model = TTCompiled(declare_tt=True)
    out = a._infer_embed({"model": model}, [1, 2, 0], [1, 1, 0], [7, 7, 0], 1)
    assert model.arrays["token_type_ids"].tolist() == [[7, 7, 0]]
    assert len(out) == 3 and len(out[0]) == 4


@_NEEDS_NUMPY
def test_infer_embed_omits_undeclared_token_type_ids():
    a = make_adapter()
    model = TTCompiled(declare_tt=False)
    a._infer_embed({"model": model}, [1, 2, 0], [1, 1, 0], [0, 0, 0], 1)
    assert "token_type_ids" not in model.arrays


def test_encode_supplies_token_type_ids_or_zeros():
    a = make_adapter()
    ids, mask, tt = a._encode(FakeTokenizer(), "one two")
    assert len(ids) == len(mask) == len(tt) and all(t == 0 for t in tt)

    class TTTokenizer(FakeTokenizer):
        def __call__(self, text, return_attention_mask=True):
            enc = dict(super().__call__(text, return_attention_mask))
            enc["token_type_ids"] = [7] * len(enc["input_ids"])
            return enc

    ids, mask, tt = a._encode(TTTokenizer(), "one two")
    assert tt == [7] * len(ids)


def test_pad_like_lockstep():
    assert ovm.pad_like([7, 7], 4) == [7, 7, 0, 0]
    assert ovm.pad_like([7, 7, 7, 7], 2) == [7, 7]
    assert ovm.pad_like([], 3) == [0, 0, 0]


@_NEEDS_NUMPY
def test_run_embed_pads_token_type_ids_in_lockstep(tmp_path, monkeypatch):
    m = local_model(tmp_path)
    monkeypatch.setitem(ovm._POOLING, m, "cls")
    compiled = TTCompiled(declare_tt=True)

    class TTCore(FakeCore):
        def compile_model(self, graph, device, config):
            self.compiled.append((device, dict(config)))
            return compiled

    a = make_adapter(core=TTCore())
    r = a.run(req(model=m, device="CPU", static_shape="1x8", batch=1))
    assert r.ok, r.error
    ids = compiled.arrays["input_ids"][0].tolist()
    assert compiled.arrays["token_type_ids"][0].tolist() == [0] * len(ids) and len(ids) == 8


@pytest.mark.skipif(importlib.util.find_spec("openvino") is None, reason="no intel extra in this venv")
@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="no transformers in this venv")
def test_cpu_parity_is_exact_across_shapes_and_batches():
    """E1 rerun regression pin: omitting token_type_ids gave cosine plateaus of
    0.988 / 0.93 / 0.57 between shapes and batches on CPU. With the feed, the same
    tokens must give the same vector at every static shape and batch size."""
    import math

    a = ovm.adapter()

    def vec(model, shape, batch):
        r = a.run(StepRequest(step_id="p", task="embed", model=model, device="CPU", engine="openvino",
                              settings={"prompt": "The switchyard parity probe stays short.",
                                        "static_shape": shape, "batch": batch}))
        assert r.ok, r.error
        return r.output

    def cos(x, y):
        d = sum(i * j for i, j in zip(x, y))
        return d / (math.sqrt(sum(i * i for i in x)) * math.sqrt(sum(j * j for j in y)))

    cases = [
        ("BAAI/bge-small-en-v1.5@5c38ec7c405ec4b44b94cc5a9bb96e735b38267a", [16, 128, 512]),
        ("nomic-ai/nomic-embed-text-v1.5@e9b6763023c676ca8431644204f50c2b100d9aab", [16, 512, 1024]),
    ]
    for model, lengths in cases:
        ref = vec(model, f"1x{lengths[0]}", 1)
        for length in lengths[1:]:
            for batch in (1, 8):
                got = vec(model, f"{batch}x{length}", batch)
                assert cos(ref, got) >= 1 - 1e-6, f"{model} {batch}x{length} cosine {cos(ref, got)}"


def test_resolved_marks_token_type_ids_fed(tmp_path, monkeypatch):
    a = make_adapter()
    m = local_model(tmp_path)
    monkeypatch.setitem(ovm._POOLING, m, "cls")
    r = a.run(req(model=m, device="CPU", static_shape="1x8", batch=1))
    assert r.ok and r.resolved["token_type_ids_fed"] is True  # fake seam always receives it


@_NEEDS_NUMPY
def test_resolved_marks_token_type_ids_absent_when_undeclared(tmp_path, monkeypatch):
    compiled = TTCompiled(declare_tt=False)

    class NoTTCore(FakeCore):
        def compile_model(self, graph, device, config):
            self.compiled.append((device, dict(config)))
            return compiled

    a = make_adapter(core=NoTTCore())
    m = local_model(tmp_path)
    monkeypatch.setitem(ovm._POOLING, m, "cls")
    r = a.run(req(model=m, device="CPU", static_shape="1x8", batch=1))
    assert r.ok and r.resolved["token_type_ids_fed"] is False
def test_ir_xml_hub_download_fetches_the_weights_sidecar(tmp_path, monkeypatch):
    calls = []

    def fake_hub_download(repo, filename, revision=None):
        calls.append((repo, filename, revision))
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
        return str(path)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = fake_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    path = make_adapter()._model_path("acme/ir-embed", "abc123")
    assert path.name == "openvino_model.xml"
    assert calls[:2] == [
        ("acme/ir-embed", "openvino_model.xml", "abc123"),
        ("acme/ir-embed", "openvino_model.bin", "abc123"),
    ]


def test_ir_xml_hub_download_refuses_when_weights_are_missing(tmp_path, monkeypatch):
    def fake_hub_download(repo, filename, revision=None):
        if filename.endswith(".bin"):
            raise RuntimeError("404")
        path = tmp_path / filename
        path.write_text("x", encoding="utf-8")
        return str(path)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = fake_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    with pytest.raises(AdapterError, match="weights"):
        make_adapter()._model_path("acme/ir-embed", "abc123")


def test_onnx_hub_download_stays_single_file(tmp_path, monkeypatch):
    calls = []

    def fake_hub_download(repo, filename, revision=None):
        calls.append(filename)
        if filename.endswith(".xml"):
            raise RuntimeError("no IR export")
        path = tmp_path / filename
        path.write_text("x", encoding="utf-8")
        return str(path)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = fake_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    path = make_adapter()._model_path("acme/onnx-embed", "abc123")
    assert path.name == "model.onnx"
    # the failed .xml probe fetched no weights; the .onnx artifact stands alone
    assert calls == ["openvino_model.xml", "model.onnx"]
