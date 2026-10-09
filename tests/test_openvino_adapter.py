"""The OpenVINO adapter, exercised through fakes. CI has no intel extra; these
tests must pass without it, and skip-guard what genuinely needs the hardware."""

from __future__ import annotations

import importlib.util

import pytest

from switchyard.adapters import registry
from switchyard.adapters.base import AdapterError, StepRequest
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
        self.generated.append(prompt)
        return type("Decoded", (), {"texts": [f"text:{prompt}"], "perf_metrics": FakePerf()})()


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
    assert a._pipelines[(str(tmp_path), "NPU")]["pipe"].config["MAX_PROMPT_LEN"] == 68


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
