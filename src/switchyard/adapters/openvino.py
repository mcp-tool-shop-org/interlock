"""The OpenVINO adapter: the Intel lane (CPU, iGPU, NPU).

The harness talks to ``run``, ``unload``, ``start_session`` and
``stop_session``. The registry finds this module by name and calls
``adapter()``. The module imports cleanly without the intel extra; the
missing package surfaces as an AdapterError only when a step or session is
actually requested.

Measurement policy, fixed by the adapter and recorded in ``resolved``:

``embed`` steps pool per model from the committed ``_POOLING`` table (cls
for the bge family, mean for nomic — the recipes the npu-probe rows were
measured with), then L2-normalise. An embed arm on a repo with no recipe is
refused rather than pooled silently. ``nli`` and ``rerank`` steps return the
single-row logits un-pooled. Batch rows above the first repeat the same
input (``batch_fill: repeat``) because a step carries one prompt; the NLI
batch is pinned to 1 by the spec validator. Inputs longer than the static
length are right-truncated and the dropped count is recorded as
``truncated_tokens``; an arm that sets ``truncate: false`` is refused
instead of silently measuring something else.

``generate`` steps run openvino-genai's LLMPipeline, greedy unless the arm
sets a temperature. Decode throughput and TTFT come from genai's own
performance counters, echoed raw under ``resolved["genai"]``.

The NPU infers static shapes only, so a NPU step without ``static_shape``
(``"BxL"``) is refused before any compile. AUTO, HETERO, Multi and GPU.1 are
refused by devices.resolve_device. A ``compile_cache: fresh`` spec gets a
new driver cache directory from the harness, which keeps cold compile times
honest. ``keep_alive`` is an Ollama concept and is ignored here; ``shift``
belongs to the adaptive-window lane and is likewise ignored.
"""

from __future__ import annotations

import gc
import threading
import time
import warnings
from pathlib import Path

from switchyard.adapters.base import AdapterError, StepRequest, StepResult
from switchyard.devices import DeviceRefused, resolve_device

try:  # CI has no intel extra; importing this module must still succeed.
    import openvino as ov
except ImportError:  # pragma: no cover
    ov = None


EMBED_UNIT = "items/s"
GENERATE_UNIT = "tok/s"
_SESSION_NEW_TOKENS = 8
_QUANTS = ("int4", "int8", "fp16", "fp32")


def _need_openvino() -> None:
    if ov is None:
        raise AdapterError(
            "the 'openvino' package is not installed; install the intel extra to run Intel experiments"
        )


def parse_model_ref(model: str) -> tuple[str, str | None]:
    """Split "repo@revision" (hub style). Local paths never carry a pin."""
    if "@" in model and not Path(model).exists():
        repo, _, ref = model.rpartition("@")
        if repo and ref:
            return repo, ref
    return model, None


def parse_static_shape(text: object) -> tuple[int, int] | None:
    if text in (None, ""):
        return None
    if isinstance(text, (list, tuple)) and len(text) == 2:
        return int(text[0]), int(text[1])
    parts = str(text).lower().split("x")
    if len(parts) != 2:
        raise AdapterError(f"static_shape {text!r} is not BxL")
    try:
        batch, length = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise AdapterError(f"static_shape {text!r} is not BxL") from exc
    if batch < 1 or length < 1:
        raise AdapterError(f"static_shape {text!r} needs positive dims")
    return batch, length


def quant_of(model: str, setting: object) -> str:
    for source in (str(setting or ""), model):
        low = source.lower()
        for q in _QUANTS:
            if q in low:
                return q
    return "unrecorded"


def pad_truncate(ids: list[int], length: int, pad_id: int, *, allow_truncate: bool = True) -> tuple[list[int], list[int], int]:
    """Right-pad/truncate to ``length``. Returns (ids, mask, dropped)."""
    dropped = max(0, len(ids) - length)
    if dropped and not allow_truncate:
        raise AdapterError(
            f"the input is {len(ids)} tokens over a static length {length} and the arm sets truncate: false"
        )
    out = list(ids[:length])
    mask = [1] * len(out)
    while len(out) < length:
        out.append(pad_id)
        mask.append(0)
    return out, mask, dropped


def pool(matrix, mask, how: str) -> list[float]:
    """how cls|mean over unmasked rows; 'none' enforces one logit row.

    Pure Python on purpose: the harness CI has no numpy, and the row count here
    is the sequence length, never large.
    """
    rows = [[float(x) for x in row] for row in matrix]
    if not rows:
        raise AdapterError("the model returned no rows")
    if how in ("cls", "none"):
        return rows[0]
    total = float(sum(mask))
    if total <= 0:
        raise AdapterError("the attention mask is all zeros")
    acc = [0.0] * len(rows[0])
    for row, keep in zip(rows, mask):
        if keep:
            for j, x in enumerate(row):
                acc[j] += x
    return [x / total for x in acc]


def normalize(vec) -> list[float]:
    import math

    v = [float(x) for x in vec]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 0 else v


class OpenvinoAdapter:
    """The Adapter Protocol, implemented over openvino / openvino-genai.

    Test seams, all injectable: ``core`` (an ov.Core stand-in),
    ``tokenizer_loader``, ``genai_module`` and ``clock``.
    """

    def __init__(self, *, core=None, tokenizer_loader=None, genai_module=None, clock=None, default_device: str | None = None) -> None:
        self._injected_core = core
        self._tokenizer_loader = tokenizer_loader
        self._genai_module = genai_module
        self.clock = clock or time.perf_counter
        self.default_device = default_device
        self._tokenizers: dict = {}
        self._compiled: dict = {}
        self._pipelines: dict = {}
        self._session: dict | None = None

    @staticmethod
    def enumerate_devices(core) -> list[dict]:
        """[{"id", "full_name"}] from a Core. This is the resolver's only input."""
        devices = []
        for dev in core.available_devices:
            try:
                name = str(core.get_property(dev, "FULL_DEVICE_NAME"))
            except Exception:
                name = ""
            devices.append({"id": str(dev), "full_name": name})
        return devices

    def _core(self):
        if self._injected_core is not None:
            return self._injected_core
        _need_openvino()
        if getattr(self, "_core_obj", None) is None:
            self._core_obj = ov.Core()
        return self._core_obj

    def _resolve_ov_id(self, device: str) -> str:
        try:
            return resolve_device(device, self.enumerate_devices(self._core())).ov_id
        except DeviceRefused as exc:
            raise AdapterError(str(exc)) from exc

    def _load_tokenizer(self, model: str, revision: str | None):
        key = (model, revision)
        if key not in self._tokenizers:
            loader = self._tokenizer_loader or _default_tokenizer_loader
            self._tokenizers[key] = loader(model, revision)
        return self._tokenizers[key]

    def _model_path(self, model: str, revision: str | None) -> Path:
        """The embedder artifact: a local dir's openvino_model.xml/model.onnx, or a hub download."""
        path = Path(model)
        if path.is_dir():
            for name in ("openvino_model.xml", "model.onnx"):
                cand = path / name
                if cand.exists():
                    return cand
            raise AdapterError(f"{model} has no openvino_model.xml or model.onnx")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise AdapterError("huggingface_hub is needed for hub model ids") from exc
        last: Exception | None = None
        for name in ("openvino_model.xml", "model.onnx", "onnx/model.onnx"):
            try:
                return Path(hf_hub_download(model, name, revision=revision))
            except Exception as exc:
                last = exc
        raise AdapterError(f"no embedder artifact found for {model}: {last}")

    def _load_compiled(self, file: Path, ov_id: str, static: tuple[int, int] | None, config: dict) -> tuple[object, float]:
        core = self._core()
        graph = core.read_model(str(file))
        if static is not None:
            batch, length = static
            try:
                shapes = {port: [batch, length] for port in graph.inputs if len(port.get_partial_shape()) == 2}
                if shapes:
                    graph.reshape(shapes)
            except Exception as exc:
                raise AdapterError(f"the static shape {batch}x{length} does not fit {file.name}: {exc}") from exc
        t0 = self.clock()
        compiled = core.compile_model(graph, ov_id, config or {})
        return compiled, self.clock() - t0

    def _embedder(self, model: str, revision: str | None, ov_id: str, static: tuple[int, int] | None, compile_config: dict) -> dict:
        key = (model, revision, ov_id, static, tuple(sorted(compile_config.items())))
        entry = self._compiled.get(key)
        if entry is None:
            file = self._model_path(model, revision)
            compiled, compile_s = self._load_compiled(file, ov_id, static, compile_config)
            entry = {"model": compiled, "compile_s": compile_s, "warmed": False, "path": str(file)}
            self._compiled[key] = entry
        return entry

    def _encode(self, tokenizer, text: str) -> tuple[list[int], list[int]]:
        out = tokenizer(text, return_attention_mask=True)
        ids = out["input_ids"]
        mask = out.get("attention_mask") or [1] * len(ids)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if hasattr(mask, "tolist"):
            mask = mask.tolist()
        while ids and isinstance(ids[0], list):
            ids = ids[0]
        while mask and isinstance(mask[0], list):
            mask = mask[0]
        return [int(i) for i in ids], [int(m) for m in mask]

    def _infer_embed(self, entry: dict, ids: list[int], mask: list[int], batch_rows: int) -> list:
        model = entry["model"]
        if hasattr(model, "infer_lists"):  # fake seam; not part of OpenVINO's API
            return model.infer_lists({"input_ids": ids, "attention_mask": mask}, batch_rows)
        import numpy as np  # real inference implies the intel extra, which ships numpy

        arrays = {
            "input_ids": np.asarray([ids] * batch_rows, dtype=np.int64),
            "attention_mask": np.asarray([mask] * batch_rows, dtype=np.int64),
        }
        result = model(arrays)
        return np.asarray(result[model.outputs[0]])[0].tolist()


def _default_tokenizer_loader(model: str, revision: str | None):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise AdapterError("the 'transformers' package is needed for tokenizers; install the intel extra") from exc
    kwargs = {"revision": revision} if revision else {}
    return AutoTokenizer.from_pretrained(model, **kwargs)

class _Settings:
    """request.settings with the harness-added keys split out and Nones gone."""

    def __init__(self, request: StepRequest) -> None:
        raw = dict(request.settings or {})
        self.spec_path = raw.pop("spec", None)
        self.compile_cache = raw.pop("compile_cache", None)
        v = raw.pop("compile_cache_dir", None)
        self.compile_cache_dir = None if v is None else str(v)
        self.values = {k: v for k, v in raw.items() if v is not None}

    def get(self, key, default=None):
        return self.values.get(key, default)


def _compile_config(s: _Settings) -> dict:
    config: dict = {}
    if s.compile_cache == "fresh" and s.compile_cache_dir:
        config["CACHE_DIR"] = s.compile_cache_dir
    precision = s.get("precision")
    if precision:
        config["INFERENCE_PRECISION_HINT"] = str(precision)
    return config


# Pooling belongs to the model, not the task. The recipes below are the ones
# the npu-probe/npu-serve rows on rnd's npu-buildout were measured with
# (bge-base a5beb1e, bge-small 5c38ec7, nomic e9b6763 — every arm still pins
# its own revision). An embed arm on a repo not listed here is refused.
_POOLING = {
    "BAAI/bge-base-en-v1.5": "cls",
    "BAAI/bge-small-en-v1.5": "cls",
    "nomic-ai/nomic-embed-text-v1.5": "mean",
}


def _pooling_for(task: str, repo: str) -> str:
    if task != "embed":
        return "none"
    recipe = _POOLING.get(repo)
    if recipe is None:
        raise AdapterError(
            f"no pooling recipe on record for {repo!r}; pooling is per model, "
            "so the recipe (cls|mean) belongs in _POOLING before that arm runs"
        )
    return recipe


def _run_embed(self, request: StepRequest, s: _Settings, resolved: dict) -> StepResult:
    ov_id = self._resolve_ov_id(request.device)
    static = parse_static_shape(s.get("static_shape"))
    if ov_id == "NPU" and static is None:
        raise AdapterError("the NPU needs a static_shape BxL; dynamic shapes are not supported there")
    repo, ref = parse_model_ref(request.model)
    pooling = _pooling_for(request.task, repo)  # refused before anything loads
    tokenizer = self._load_tokenizer(repo, ref)
    ids, mask = self._encode(tokenizer, str(s.get("prompt") or "hello"))
    dropped = 0
    if static is not None:
        ids, mask, dropped = pad_truncate(ids, static[1], int(getattr(tokenizer, "pad_token_id", 0) or 0),
                                          allow_truncate=bool(s.get("truncate", True)))
    batch_rows = int(s.get("batch", 1))
    entry = self._embedder(repo, ref, ov_id, static, _compile_config(s))
    compiled_now = not entry["warmed"]
    t0 = self.clock()
    matrix = self._infer_embed(entry, ids, mask, batch_rows if static is None else static[0])
    call_s = self.clock() - t0
    vec = normalize(pool(matrix, mask, pooling))
    resolved.update(
        revision=ref or "unrecorded",
        pooling=pooling,
        num_ctx=getattr(tokenizer, "model_max_length", None) or "unrecorded",
    )
    if dropped:
        resolved["truncated_tokens"] = dropped
    if (static[0] if static else batch_rows) > 1:
        resolved["batch_fill"] = "repeat"
    kwargs = dict(
        ok=True,
        latency_ms=call_s * 1000,
        throughput=(static[0] if static else batch_rows) / call_s if call_s > 0 else None,
        throughput_unit=EMBED_UNIT,
        output=vec,
        resolved=resolved,
        sensors=None,
    )
    if compiled_now:
        kwargs["compile_ms"] = entry["compile_s"] * 1000
        kwargs["first_call_ms"] = call_s * 1000
        entry["warmed"] = True
    return StepResult(**kwargs)


def _run_generate(self, request: StepRequest, s: _Settings, resolved: dict) -> StepResult:
    ov_id = self._resolve_ov_id(request.device)
    pipe, compile_s = self._pipeline(request.model, ov_id, _compile_config(s))
    key = (request.model, ov_id)
    genai = self._genai_module or _real_genai_module()
    config = genai.GenerationConfig()
    config.max_new_tokens = int(s.get("num_predict", 128))
    if s.get("temperature") is not None:
        config.do_sample = True
        config.temperature = float(s.get("temperature"))
    t0 = self.clock()
    out = pipe.generate(str(s.get("prompt") or "hello"), config)
    call_s = self.clock() - t0
    perf = out.perf_metrics
    text = out.texts[0] if getattr(out, "texts", None) else str(out)
    resolved.update(revision=parse_model_ref(request.model)[1] or "unrecorded")
    try:
        resolved["genai"] = {
            "ttft_ms": float(perf.get_ttft()),
            "tpot_ms": float(perf.get_tpot()),
            "input_tokens": int(perf.get_num_input_tokens()),
            "generated_tokens": int(perf.get_num_generated_tokens()),
        }
    except Exception:
        pass
    if s.get("thinking_level") is not None:
        resolved["thinking_level"] = s.get("thinking_level")
    first = not self._pipelines[key].get("warmed")
    throughput = None
    try:
        throughput = float(perf.get_throughput())
    except Exception:
        pass
    kwargs = dict(
        ok=True,
        latency_ms=call_s * 1000,
        throughput=throughput,
        throughput_unit=GENERATE_UNIT,
        output=text,
        resolved=resolved,
        sensors=None,
    )
    if first:
        kwargs["compile_ms"] = compile_s * 1000
        kwargs["first_call_ms"] = call_s * 1000
        self._pipelines[key]["warmed"] = True
    return StepResult(**kwargs)

_GENAI_REAL = None


def _real_genai_module():
    global _GENAI_REAL
    if _GENAI_REAL is None:
        try:
            import openvino_genai as module
        except ImportError as exc:
            raise AdapterError(
                "the 'openvino-genai' package is needed for generate steps; install the intel extra"
            ) from exc
        _GENAI_REAL = module
    return _GENAI_REAL


def _model_dir(model: str) -> Path:
    """openvino-genai loads a whole export directory, not a single file."""
    path = Path(parse_model_ref(model)[0])
    if path.is_dir():
        return path
    repo, ref = parse_model_ref(model)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise AdapterError("huggingface_hub is needed for hub model ids") from exc
    try:
        return Path(snapshot_download(repo, revision=ref))
    except Exception as exc:
        raise AdapterError(f"the model {model} could not be fetched: {exc}") from exc


def _pipeline(self, model: str, ov_id: str, compile_config: dict):
    key = (model, ov_id)
    entry = self._pipelines.get(key)
    if entry is None:
        genai = self._genai_module or _real_genai_module()
        directory = _model_dir(model)
        t0 = self.clock()
        pipe = genai.LLMPipeline(str(directory), ov_id, compile_config or {})
        compile_s = self.clock() - t0
        entry = {"pipe": pipe, "warmed": False, "path": str(directory)}
        self._pipelines[key] = entry
        return pipe, compile_s
    return entry["pipe"], None


def _run(self, request: StepRequest) -> StepResult:
    if request.engine != "openvino":
        raise AdapterError(f"the OpenVINO adapter was handed engine {request.engine!r}")
    s = _Settings(request)
    resolved: dict = {"quant": quant_of(request.model, s.get("quant"))}
    if request.task in ("embed", "nli", "rerank"):
        step = _run_embed
    elif request.task == "generate":
        step = _run_generate
    else:
        raise AdapterError(f"the OpenVINO adapter does not run task {request.task!r}")
    try:
        return step(self, request, s, resolved)
    except AdapterError:
        raise  # a refusal, raised before measuring
    except Exception as exc:
        return StepResult(ok=False, error=str(exc), resolved=resolved or None, sensors=None)


def _unload(self, model: str, device: str | None = None) -> None:
    """Drop the compiled embedders and pipelines for this model; the process keeps nothing."""
    self._compiled = {k: v for k, v in self._compiled.items() if k[0] != parse_model_ref(model)[0] and k[0] != model}
    self._pipelines = {k: v for k, v in self._pipelines.items() if k[0] not in (model, parse_model_ref(model)[0])}
    for key in list(self._tokenizers):
        if key[0] in (model, parse_model_ref(model)[0]):
            del self._tokenizers[key]
    gc.collect()


OpenvinoAdapter.run = _run
OpenvinoAdapter.unload = _unload
OpenvinoAdapter._pipeline = _pipeline

def _start_session(self, request: StepRequest) -> None:
    """Begin a companion load; returns while it runs. State lives on the adapter,
    keyed by the arm's model+device, so stop_session can end exactly this load."""
    if self._session is not None:
        raise AdapterError("a companion session is already running; pair one arm at a time")
    s = _Settings(request)
    ov_id = self._resolve_ov_id(request.device)
    stop = threading.Event()
    box: dict = {"fault": None, "call_s": None}

    def loop() -> None:
        try:
            if request.task == "generate":
                genai = self._genai_module or _real_genai_module()
                pipe, _ = _pipeline(self, request.model, ov_id, _compile_config(s))
                config = genai.GenerationConfig()
                config.max_new_tokens = _SESSION_NEW_TOKENS
                while not stop.is_set():
                    t0 = self.clock()
                    pipe.generate(str(s.get("prompt") or "hello"), config)
                    box["call_s"] = self.clock() - t0
            else:
                repo, ref = parse_model_ref(request.model)
                static = parse_static_shape(s.get("static_shape"))
                if ov_id == "NPU" and static is None:
                    raise AdapterError("the NPU needs a static_shape BxL; dynamic shapes are not supported there")
                tokenizer = self._load_tokenizer(repo, ref)
                ids, mask = self._encode(tokenizer, str(s.get("prompt") or "hello"))
                if static is not None:
                    ids, mask, _ = pad_truncate(ids, static[1], int(getattr(tokenizer, "pad_token_id", 0) or 0))
                entry = self._embedder(repo, ref, ov_id, static, _compile_config(s))
                rows = static[0] if static else int(s.get("batch", 1))
                while not stop.is_set():
                    t0 = self.clock()
                    self._infer_embed(entry, ids, mask, rows)
                    box["call_s"] = self.clock() - t0
        except Exception as exc:  # surfaced as a refusal on start, or on stop mid-run
            box["fault"] = exc

    worker = threading.Thread(target=loop, name=f"switchyard-ov-{request.step_id}", daemon=True)
    self._session = {"stop": stop, "worker": worker, "box": box, "model": request.model, "device": ov_id}
    worker.start()
    time.sleep(0.05)
    if box["fault"] is not None:
        stop.set()
        worker.join(timeout=5.0)
        self._session = None
        fault = box["fault"]
        if isinstance(fault, AdapterError):
            raise fault
        raise AdapterError(f"the companion session failed to start: {fault}")


def _stop_session(self, request: StepRequest) -> None:
    """End the companion load. This is not unload; the model stays compiled."""
    sess = self._session
    if sess is None:
        return
    if request.model != sess["model"]:
        raise AdapterError(
            f"stop_session for {request.model!r} while a session for {sess['model']!r} is running"
        )
    sess["stop"].set()
    sess["worker"].join(timeout=10.0)
    if sess["worker"].is_alive():
        warnings.warn("an OpenVINO session thread did not stop within 10 s; it exits with the process", RuntimeWarning)
    self._session = None
    fault = sess["box"]["fault"]
    if fault is not None and not isinstance(fault, AdapterError):
        raise AdapterError(f"the companion session failed mid-run: {fault}")


def _guarded_unload(self, model: str, device: str | None = None) -> None:
    if self._session is not None:
        raise AdapterError("a companion session is still running; stop_session before unload")
    _unload(self, model, device)


OpenvinoAdapter.start_session = _start_session
OpenvinoAdapter.stop_session = _stop_session
OpenvinoAdapter.unload = _guarded_unload


def adapter(**kwargs) -> OpenvinoAdapter:
    """The registry entry point."""
    return OpenvinoAdapter(**kwargs)
