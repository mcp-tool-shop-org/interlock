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

    def __call__(self, text: str, return_attention_mask: bool = True):
        ids = [101] + [abs(hash(w)) % 30000 + 1 for w in str(text).split()] + [102]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class FakeCompiled:
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def infer_lists(self, inputs, batch_rows: int):
        self.calls += 1
        if self.fail:
            raise OSError("device exploded")
        mask = inputs["attention_mask"]
        n = sum(mask)
        return [[float(n), 1.0, 2.0]] * batch_rows


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
