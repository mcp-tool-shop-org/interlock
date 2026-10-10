"""The OpenVINO adapter: the Intel lane (CPU, iGPU, NPU).

The harness talks to ``run``, ``unload``, ``start_session`` and
``stop_session``. The registry finds this module by name and calls
``adapter()``. The module imports cleanly without the intel extra; the
missing package surfaces as an AdapterError only when a step or session is
actually requested.

Measurement policy, fixed by the adapter and recorded in ``resolved``:

``embed`` steps pool per model from the committed ``_POOLING`` table (cls
for the bge family, mean for nomic — the recipes the npu-probe rows were
measured with — and last-token for Qwen3-Embedding, its documented
recipe), then L2-normalise. An embed arm on a repo with no recipe is
refused rather than pooled silently. ``nli`` and ``rerank`` steps return the
single-row logits un-pooled. Batch rows above the first repeat the same
input (``batch_fill: repeat``) because a step carries one prompt; the NLI
batch is pinned to 1 by the spec validator. Inputs longer than the static
length are right-truncated. ``prompt_tokens`` is the encoded length and
``truncated_tokens`` is the dropped count, including zero when the prefix
fits. An arm that sets ``truncate: false`` is refused instead of silently
measuring something else.

``generate`` steps run openvino-genai's LLMPipeline, greedy unless the arm
sets a temperature. Decode throughput and TTFT come from genai's own
performance counters, echoed raw under ``resolved["genai"]``. On the NPU the
KV cache is sized explicitly (MAX_PROMPT_LEN = prompt + 64, MIN_RESPONSE_LEN
= num_predict + 32; genai 2026.4.1 NPU pipeline keys, default 1024), the
values are recorded under ``resolved["genai_config"]``, and a pipeline that
still cannot cover the prompt ends the arm with ``ok`` true and
``resolved["outcome"] = "unsupported length"`` — recorded, not retried, and
the sequence continues. A Level Zero device loss is a capability row too,
``resolved["outcome"] = "device_lost"`` with no timing fields, after which
that device is off for the rest of the run and its remaining steps report
``"not run"`` without touching the hardware. A companion session that loses
its device still surfaces the fault — the concurrent measurements are invalid
without the load — and the device is marked the same way, so later one-shot
steps on it are recorded, not attempted. A compiled binary the device refuses
at execute (``ZE_RESULT_ERROR_INVALID_NATIVE_BINARY``, ``0x7800000f``) is
raised as :class:`NativeBinaryRejected` with the compile time attached; the
runner scopes it to model x device x static shape and the other shapes still
run. An NPU compile that the compiler refuses (``[NPU_VCL] Compiler returned
msg`` with ``0x78000007``) is raised as :class:`CompileRejected` with the time
to failure; any other compile error is left as it was.

The NPU infers static shapes only, so a NPU step without ``static_shape``
(``"BxL"``) is refused before any compile. AUTO, HETERO, Multi and GPU.1 are
refused by devices.resolve_device. A ``compile_cache: fresh`` spec gets a
new driver cache directory from the harness; the first compile into it is
B_fresh's numerator, and an immediate second compile into the same dir is
recorded as ``resolved["compile_warm_reload_ms"]`` — the realistic reload
with the driver cache warm, B_reuse's numerator. Every measured call carries
a hard per-call bound (``call_timeout_s``, 60 s by default): a hang returns
``ok=False`` with a "hang:" error, drops the entry, and poisons the
model+device pair for the rest of the run. ``keep_alive`` is an Ollama
concept and is ignored here; ``shift`` belongs to the adaptive-window lane
and is likewise ignored.
"""

from __future__ import annotations

import gc
import threading
import time
import warnings
from pathlib import Path

from switchyard.adapters.base import (
    AdapterError,
    CompileRejected,
    NativeBinaryRejected,
    StepRequest,
    StepResult,
    compile_rejected_error,
)
from switchyard.devices import DeviceRefused, resolve_device

try:  # CI has no intel extra; importing this module must still succeed.
    import openvino as ov
except ImportError:  # pragma: no cover
    ov = None


EMBED_UNIT = "items/s"
GENERATE_UNIT = "tok/s"
_SESSION_NEW_TOKENS = 8
_QUANTS = ("int4", "int8", "fp16", "fp32")
# Level Zero device-loss signature, seen live: nomic NPU 8x1024 reset the
# device mid-run (driver 32.0.100.5540; no Windows event-log entry — the L0
# result code was the only witness). The list is exactly what the runner
# matches (#19/#20): the L0 code and its outcome-string forms, lowering the
# text first. The driver's own sentence says "device hung", but that phrase
# was never seen without the code, so it is not a match on its own. Distinct
# from a 60 s timeout hang: a hang ends the run, a device loss becomes a
# result row and the device is then off for the rest of the run.
_LOST_MARKERS = ("ze_result_error_device_lost", "device_lost", "device lost")

# Level Zero native-binary rejection, seen live: E2's Qwen3-Embedding l512
# compile wrote a complete blob and the NPU refused the binary at execute
# (driver 32.0.100.5540). The compile succeeded, so this is not a compile
# error, a device loss, or a hang: the runner turns it into a capability row
# scoped to model x device x static shape (ruling #33).
_REJECTED_MARKERS = ("ze_result_error_invalid_native_binary", "0x7800000f")


def _is_native_binary_rejected(exc: BaseException) -> bool:
    return any(marker in str(exc).lower() for marker in _REJECTED_MARKERS)


def _is_device_lost(exc: BaseException) -> bool:
    text = str(exc).lower()
    if text.startswith("hang:"):  # the same precedence the runner applies
        return False
    return any(marker in text for marker in _LOST_MARKERS)


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


def pad_like(values: list[int], length: int, fill: int = 0) -> list[int]:
    """Pad/truncate a parallel field in lockstep with :func:`pad_truncate` (token_type_ids)."""
    return list(values[:length]) + [fill] * max(0, length - len(values))


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
    if how == "last":  # Qwen3-Embedding's recipe: the last unmasked row
        keep = [i for i, m in enumerate(mask) if m]
        if not keep:
            raise AdapterError("the attention mask is all zeros")
        return rows[keep[-1]]
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

    def __init__(self, *, core=None, tokenizer_loader=None, genai_module=None, clock=None, default_device: str | None = None, call_timeout_s: float = 60.0) -> None:
        self._injected_core = core
        self._tokenizer_loader = tokenizer_loader
        self._genai_module = genai_module
        self.clock = clock or time.perf_counter
        self.default_device = default_device
        self.call_timeout_s = float(call_timeout_s)
        self._tokenizers: dict = {}
        self._compiled: dict = {}
        self._pipelines: dict = {}
        self._session: dict | None = None
        self._poisoned: set = set()  # (model, device) keys that hung once
        self._lost_devices: set = set()  # ov ids that reset this run

    def count_tokens(self, model: str, text: str) -> int:
        """Tokens of ``text`` on this model's tokenizer, special tokens included.

        This is the encode the embed path uses. It does not compile a device.
        """

        repo, ref = parse_model_ref(model)
        ids, _mask, _token_types = self._encode(self._load_tokenizer(repo, ref), text)
        return len(ids)

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
                artifact = Path(hf_hub_download(model, name, revision=revision))
            except Exception as exc:
                last = exc
                continue
            if name.endswith(".xml"):
                # OpenVINO IR: the .xml reads its weights from the sibling .bin.
                weights = name[:-4] + ".bin"
                try:
                    hf_hub_download(model, weights, revision=revision)
                except Exception as exc:
                    raise AdapterError(f"{model} artifact {name} has no weights {weights}: {exc}") from exc
            return artifact
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
            warm_reload_s = None
            if compile_config.get("CACHE_DIR"):
                # The realistic reload: same graph, same dir, driver cache already
                # populated by the first compile. B_reuse is counted from this.
                _reload, warm_reload_s = self._load_compiled(file, ov_id, static, compile_config)
            entry = {"model": compiled, "compile_s": compile_s, "warm_reload_s": warm_reload_s,
                     "warmed": False, "path": str(file)}
            self._compiled[key] = entry
        return entry

    def _encode(self, tokenizer, text: str) -> tuple[list[int], list[int], list[int]]:
        out = tokenizer(text, return_attention_mask=True)
        ids = out["input_ids"]
        mask = out.get("attention_mask") or [1] * len(ids)
        tt = out.get("token_type_ids")
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if hasattr(mask, "tolist"):
            mask = mask.tolist()
        if hasattr(tt, "tolist"):
            tt = tt.tolist()
        while ids and isinstance(ids[0], list):
            ids = ids[0]
        while mask and isinstance(mask[0], list):
            mask = mask[0]
        while tt and isinstance(tt[0], list):
            tt = tt[0]
        ids_i = [int(i) for i in ids]
        # token_type_ids default to zeros; an OMITTED input gets garbage, not zeros.
        tt_i = [int(t) for t in tt] if tt else [0] * len(ids_i)
        return ids_i, [int(m) for m in mask], tt_i

    def _infer_embed(self, entry: dict, ids: list[int], mask: list[int], tt: list[int], batch_rows: int) -> list:
        model = entry["model"]
        if hasattr(model, "infer_lists"):  # fake seam; not part of OpenVINO's API
            return model.infer_lists({"input_ids": ids, "attention_mask": mask, "token_type_ids": list(tt)}, batch_rows)
        import numpy as np  # real inference implies the intel extra, which ships numpy

        arrays = {
            "input_ids": np.asarray([ids] * batch_rows, dtype=np.int64),
            "attention_mask": np.asarray([mask] * batch_rows, dtype=np.int64),
        }
        # Feed token_type_ids only when the compiled graph declares it. Never omit a
        # declared input: OpenVINO infers it from an uninitialized buffer, not zeros
        # (E1 rerun, 2026-10-09: cosine plateaus 0.988/0.93/0.57 on CPU). Models
        # without the input (Qwen3-Embedding) must not receive it.
        if any(port.get_any_name() == "token_type_ids" for port in model.inputs):
            arrays["token_type_ids"] = np.asarray([tt] * batch_rows, dtype=np.int64)
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
        v = raw.pop("call_timeout_s", None)
        self.call_timeout_s = None if v is None else float(v)
        self.values = {k: v for k, v in raw.items() if v is not None}

    def get(self, key, default=None):
        return self.values.get(key, default)


def _bound(self, s: _Settings) -> float:
    """The per-call bound for this request: the run loop's per-arm value
    (ruling #36) when present, else the adapter default."""

    return s.call_timeout_s if s.call_timeout_s is not None else self.call_timeout_s


def _timed_call(fn, timeout_s: float):
    """Run one measured call with a hard bound. A hang returns ("hang", None);
    the worker is a daemon and cannot outlive the process.

    A hang on the NPU is a real failure mode (seen with a batch-8 cross
    encoder on npu-buildout), so the bound is measurement, not courtesy.
    """
    box: dict = {}

    def work() -> None:
        try:
            box["value"] = fn()
        except Exception as exc:  # re-raised by the caller
            box["fault"] = exc

    worker = threading.Thread(target=work, name="switchyard-ov-call", daemon=True)
    worker.start()
    worker.join(timeout=timeout_s)
    if worker.is_alive():
        return "hang", None
    if "fault" in box:
        raise box["fault"]
    return "ok", box.get("value")


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
    "OpenVINO/Qwen3-Embedding-0.6B-fp16-ov": "last",
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
    if ov_id in self._lost_devices:
        resolved["outcome"] = "not run"
        return StepResult(ok=True, resolved=resolved, sensors=None)
    if (parse_model_ref(request.model)[0], ov_id) in self._poisoned:
        raise AdapterError(
            f"{request.model} on {ov_id} hung earlier in this run; the device pair is poisoned for the run"
        )
    static = parse_static_shape(s.get("static_shape"))
    if ov_id == "NPU" and static is None:
        raise AdapterError("the NPU needs a static_shape BxL; dynamic shapes are not supported there")
    repo, ref = parse_model_ref(request.model)
    pooling = _pooling_for(request.task, repo)  # refused before anything loads
    tokenizer = self._load_tokenizer(repo, ref)
    ids, mask, tt = self._encode(tokenizer, str(s.get("prompt") or "hello"))
    encoded_len = len(ids)
    eos_appended = None
    if pooling == "last":
        eos_id = getattr(tokenizer, "eos_token_id", None)
        eos_appended = bool(eos_id is not None and ids and ids[-1] == eos_id)
    dropped = 0
    if static is not None:
        ids, mask, dropped = pad_truncate(ids, static[1], int(getattr(tokenizer, "pad_token_id", 0) or 0),
                                          allow_truncate=bool(s.get("truncate", True)))
        tt = pad_like(tt, static[1])
    batch_rows = int(s.get("batch", 1))
    build_t0 = self.clock()
    try:
        entry = self._embedder(repo, ref, ov_id, static, _compile_config(s))
    except AdapterError:
        raise
    except Exception as exc:
        if _is_device_lost(exc):
            self._lost_devices.add(ov_id)
            resolved["outcome"] = "device_lost"
            return StepResult(ok=True, resolved=resolved, sensors=None)
        if ov_id == "NPU" and compile_rejected_error(str(exc)):
            raise CompileRejected(str(exc), failed_s=self.clock() - build_t0) from exc
        raise
    compiled_now = not entry["warmed"]
    t0 = self.clock()
    try:
        bound = _bound(self, s)
        status, matrix = _timed_call(
            lambda: self._infer_embed(entry, ids, mask, tt, batch_rows if static is None else static[0]),
            bound,
        )
    except Exception as exc:
        if _is_device_lost(exc):
            self._lost_devices.add(ov_id)
            self._compiled = {k: v for k, v in self._compiled.items() if not (k[0] == repo and k[2] == ov_id)}
            resolved["outcome"] = "device_lost"
            return StepResult(ok=True, resolved=resolved, sensors=None)
        if _is_native_binary_rejected(exc):
            raise NativeBinaryRejected(
                str(exc), compile_s=entry["compile_s"] if compiled_now else None
            ) from exc
        raise
    if status == "hang":
        self._compiled = {k: v for k, v in self._compiled.items() if not (k[0] == repo and k[2] == ov_id)}
        self._poisoned.add((repo, ov_id))
        return StepResult(
            ok=False,
            error=f"hang: no result within {bound:g} s; arm inconclusive, device pair poisoned for the run",
            resolved=resolved or None,
            sensors=None,
        )
    call_s = self.clock() - t0
    vec = normalize(pool(matrix, mask, pooling))
    resolved.update(
        revision=ref or "unrecorded",
        pooling=pooling,
        num_ctx=getattr(tokenizer, "model_max_length", None) or "unrecorded",
    )
    resolved["prompt_tokens"] = encoded_len
    resolved["truncated_tokens"] = dropped
    # Receipt marker: the embed input was fed iff the graph declares it
    # (E1, 2026-10-09: an omitted declared input infers a stale buffer).
    # False means the model has no such input (e.g. Qwen3-Embedding).
    model = entry["model"]
    resolved["token_type_ids_fed"] = bool(
        hasattr(model, "infer_lists")
        or any(port.get_any_name() == "token_type_ids" for port in getattr(model, "inputs", []))
    )
    if eos_appended is not None:
        # Last-token pooling reads the last unmasked row: correct only when
        # that row is the tokenizer's EOS. Truncation cuts EOS off the kept
        # region, so a truncated "last" arm records last_row_is_eos false.
        resolved["eos_appended"] = eos_appended
        resolved["last_row_is_eos"] = eos_appended and dropped == 0
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
        if entry.get("warm_reload_s") is not None:
            resolved["compile_warm_reload_ms"] = entry["warm_reload_s"] * 1000
        entry["warmed"] = True
    return StepResult(**kwargs)


def _run_generate(self, request: StepRequest, s: _Settings, resolved: dict) -> StepResult:
    ov_id = self._resolve_ov_id(request.device)
    if ov_id in self._lost_devices:
        resolved["outcome"] = "not run"
        return StepResult(ok=True, resolved=resolved, sensors=None)
    if (request.model, ov_id) in self._poisoned:
        raise AdapterError(
            f"{request.model} on {ov_id} hung earlier in this run; the device pair is poisoned for the run"
        )
    compile_config = _compile_config(s)
    num_predict = int(s.get("num_predict", 128))
    if ov_id == "NPU":
        repo, ref = parse_model_ref(request.model)
        prompt_tokens = len(self._encode(self._load_tokenizer(repo, ref), str(s.get("prompt") or "hello"))[0])
        compile_config.update(_npu_genai_config(prompt_tokens, num_predict))
        resolved["genai_config"] = {
            "MAX_PROMPT_LEN": compile_config["MAX_PROMPT_LEN"],
            "MIN_RESPONSE_LEN": compile_config["MIN_RESPONSE_LEN"],
        }
    build_t0 = self.clock()
    try:
        pipe, compile_s = self._pipeline(request.model, ov_id, compile_config)
    except AdapterError:
        raise
    except Exception as exc:
        if _is_device_lost(exc):
            self._lost_devices.add(ov_id)
            resolved["outcome"] = "device_lost"
            return StepResult(ok=True, resolved=resolved or None, sensors=None)
        if ov_id == "NPU" and compile_rejected_error(str(exc)):
            raise CompileRejected(str(exc), failed_s=self.clock() - build_t0) from exc
        if ov_id == "NPU":
            # recorded, not retried: the KV cache cannot cover this prompt;
            # ok stays true so the sequence continues past the arm
            resolved["outcome"] = "unsupported length"
            return StepResult(ok=True, resolved=resolved or None, sensors=None)
        raise AdapterError(f"the pipeline for {request.model} on {ov_id} did not build: {exc}") from exc
    key = _pipeline_key(request.model, ov_id, compile_config)
    genai = self._genai_module or _real_genai_module()
    config = genai.GenerationConfig()
    config.max_new_tokens = num_predict
    if s.get("temperature") is not None:
        config.do_sample = True
        config.temperature = float(s.get("temperature"))
    t0 = self.clock()
    try:
        bound = _bound(self, s)
        status, out = _timed_call(lambda: pipe.generate([str(s.get("prompt") or "hello")], config), bound)
    except Exception as exc:
        if _is_device_lost(exc):
            self._lost_devices.add(ov_id)
            self._pipelines.pop(key, None)
            resolved["outcome"] = "device_lost"
            return StepResult(ok=True, resolved=resolved, sensors=None)
        if _is_native_binary_rejected(exc):
            raise NativeBinaryRejected(str(exc), compile_s=compile_s) from exc
        raise
    if status == "hang":
        self._pipelines.pop(key, None)
        self._poisoned.add((request.model, ov_id))
        return StepResult(
            ok=False,
            error=f"hang: no result within {bound:g} s; arm inconclusive, device pair poisoned for the run",
            resolved=resolved or None,
            sensors=None,
        )
    call_s = self.clock() - t0
    # A list prompt makes generate return DecodedResults, which carries
    # perf_metrics; a str prompt would return a bare str with no counters.
    perf = out.perf_metrics
    text = out.texts[0]
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


def _npu_genai_config(prompt_tokens: int, num_predict: int) -> dict:
    """NPU LLM pipeline sizing (openvino-genai 2026.4.1: the NPU pipeline
    allocates the KV cache statically, MAX_PROMPT_LEN defaulting to 1024).
    The prompt gets a 64-token margin; the response floor covers
    num_predict plus 32."""
    return {
        "MAX_PROMPT_LEN": int(prompt_tokens) + 64,
        "MIN_RESPONSE_LEN": int(num_predict) + 32,
    }


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


def _pipeline_key(model: str, ov_id: str, compile_config: dict | None) -> tuple:
    """One pipeline per model, device and compile config. On the NPU the config
    sizes the static KV cache (MAX_PROMPT_LEN), so a pipeline built for a short
    prompt must not be reused for a longer one."""

    return (model, ov_id, tuple(sorted((compile_config or {}).items())))


def _pipeline(self, model: str, ov_id: str, compile_config: dict):
    key = _pipeline_key(model, ov_id, compile_config)
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
                config_map = _compile_config(s)
                if ov_id == "NPU":
                    repo, ref = parse_model_ref(request.model)
                    n_tok = len(self._encode(self._load_tokenizer(repo, ref), str(s.get("prompt") or "hello"))[0])
                    config_map.update(_npu_genai_config(n_tok, int(s.get("num_predict", 128))))
                pipe, _ = _pipeline(self, request.model, ov_id, config_map)
                config = genai.GenerationConfig()
                config.max_new_tokens = _SESSION_NEW_TOKENS
                while not stop.is_set():
                    t0 = self.clock()
                    bound = _bound(self, s)
                    status, _ = _timed_call(lambda: pipe.generate(str(s.get("prompt") or "hello"), config), bound)
                    if status == "hang":
                        box["fault"] = AdapterError(f"hang: companion call exceeded {bound:g} s")
                        return
                    box["call_s"] = self.clock() - t0
            else:
                repo, ref = parse_model_ref(request.model)
                static = parse_static_shape(s.get("static_shape"))
                if ov_id == "NPU" and static is None:
                    raise AdapterError("the NPU needs a static_shape BxL; dynamic shapes are not supported there")
                tokenizer = self._load_tokenizer(repo, ref)
                ids, mask, tt = self._encode(tokenizer, str(s.get("prompt") or "hello"))
                if static is not None:
                    ids, mask, _ = pad_truncate(ids, static[1], int(getattr(tokenizer, "pad_token_id", 0) or 0))
                    tt = pad_like(tt, static[1])
                entry = self._embedder(repo, ref, ov_id, static, _compile_config(s))
                rows = static[0] if static else int(s.get("batch", 1))
                while not stop.is_set():
                    t0 = self.clock()
                    bound = _bound(self, s)
                    status, _ = _timed_call(lambda: self._infer_embed(entry, ids, mask, tt, rows), bound)
                    if status == "hang":
                        box["fault"] = AdapterError(f"hang: companion call exceeded {bound:g} s")
                        return
                    box["call_s"] = self.clock() - t0
        except Exception as exc:  # surfaced as a refusal on start, or on stop mid-run
            if _is_device_lost(exc):
                self._lost_devices.add(ov_id)
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
    if isinstance(fault, AdapterError):
        raise fault
    if fault is not None:
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
