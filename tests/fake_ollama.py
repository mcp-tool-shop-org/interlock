"""A fake Ollama that behaves the way the real one does where the generate ladder measures it.

Read from Ollama v0.40.9 and v0.40.11 (llm/llama_server.go, envconfig/config.go) and from T4 r2's llama-server log:

- ``prompt_eval_count`` is the full length of the prompt, but llama-server reuses the longest common token prefix of
  previous prompt (``cache_prompt`` is always on), so a repeated prompt prefills about one token and its
  ``prompt_eval_duration`` is near zero.
- A prompt of ``num_ctx - 1`` tokens or more is cut to ``num_ctx - max((num_ctx - num_keep) // 2, 1)`` tokens
  (``num_keep`` is 4), and that is the count reported.
- ``OLLAMA_NUM_PARALLEL`` defaults to 1: with fewer slots than requests, the requests queue.
- A load whose weights and KV (``num_ctx`` times the slots) do not fit ``vram_mib`` is a partial offload: ``/api/ps``
  shows ``size_vram < size`` and the log says ``offloaded N/M layers`` and ``CPU KV buffer size`` (T4 r3). When the host
  buffers exceed ``system_total_mib`` the process is killed on its first call: ``signal: killed``.
- A request that takes longer than the timeout it was sent with raises ``TimeoutError``.

Time is virtual. Each request takes its prefill plus its decode, on the earliest free slot; ``clock()`` returns the
finish time of the latest request, so a pass's wall clock is exact whatever the threads do.
"""

from __future__ import annotations

import math
import threading
from typing import Any

from interlock.harness.linux_cuda import parse_load_report

NUM_KEEP = 4


class FakeOllama:
    def __init__(
        self,
        *,
        tokens_per_word: float = 1.0,
        prefill_tok_s: float = 4000.0,
        decode_tok_s: float = 50.0,
        cache: bool = True,
        honour_parallel: bool = True,
        report_slots: bool = True,
        long_prompt_bias: float = 0.0,
        overhead_s: float = 0.0,
        overhead_s_per_ktok: float = 0.0,
        vram_mib: float | None = None,
        weights_mib: float = 2400.0,
        kv_mib_per_token: float = 0.375,
        layers_total: int = 33,
        mapped_mib: float = 70.0,
        system_total_mib: float = 12975.0,
        log_lines: bool = True,
        kill_early: bool = False,
    ) -> None:
        self.tokens_per_word = tokens_per_word
        self.prefill_tok_s = prefill_tok_s
        self.decode_tok_s = decode_tok_s
        self.cache = cache
        self.honour_parallel = honour_parallel
        self.report_slots = report_slots
        self.overhead_s = overhead_s  # time that is neither prefill nor decode (Ollama's Go side)
        self.overhead_s_per_ktok = overhead_s_per_ktok
        self.long_prompt_bias = long_prompt_bias  # T4 r2: long prompts ran ~1.2% over the probes
        self.vram_mib = vram_mib  # None: everything fits
        self.weights_mib = weights_mib
        self.kv_mib_per_token = kv_mib_per_token
        self.layers_total = layers_total
        self.mapped_mib = mapped_mib
        self.system_total_mib = system_total_mib
        self.log_lines = log_lines
        self.log_text = ""
        self._log_start = 0
        self.kill_early = kill_early  # killed after its system-memory line, before any load_tensors block
        self.all_calls: list[dict[str, Any]] = []  # every request, probes included
        self.timeouts: list[float | None] = []
        self._load = {"on_gpu": layers_total, "cpu_kv": 0.0, "kv": 0.0, "killed": False}
        self.version = "0.40.11"
        self.kills = 0  # loads that ended in a SIGKILL
        self.num_parallel = 1
        self.restarts = 0
        self.prompts: list[str] = []  # the timed prompts: every call that asked for more than one token
        self.contexts: list[int] = []
        self.evaluated_counts: list[int] = []
        self._lock = threading.Lock()
        self._slot_free = [0.0]
        self._now = 0.0
        self._arrival = 0.0
        self._last_words: list[str] = []
        self._loaded_ctx: int | None = None

    # -- the server surface the harness uses -------------------------------------------------------------------------
    def get(self, _path: str) -> dict[str, Any]:
        size = self.weights_mib + self._load["kv"]
        return {"models": [{"name": "m", "size": size, "size_vram": size - self._load["cpu_kv"]}]}

    def begin_cell(self) -> None:
        self._log_start = len(self.log_text)

    def load_report(self) -> dict[str, Any]:
        return parse_load_report(self.log_text[self._log_start :])

    def _log_load(self, num_ctx: int) -> None:
        """What llama-server logs when the model loads at this window and slot count (formats read from T4 r3)."""

        kv = num_ctx * self.num_parallel * self.kv_mib_per_token
        free = (self.vram_mib - self.weights_mib) if self.vram_mib is not None else kv
        fraction = 1.0 if free >= kv else max(0.0, free) / kv
        on_gpu = self.layers_total if fraction >= 1.0 else int(self.layers_total * fraction)
        cpu_kv = 0.0 if fraction >= 1.0 else kv * (1.0 - fraction)
        killed = self.kill_early or cpu_kv + self.mapped_mib > self.system_total_mib
        self._load = {"on_gpu": on_gpu, "cpu_kv": cpu_kv, "kv": kv, "killed": killed}
        system_line = (
            f'time=2026-10-26T14:00:00.000Z level=INFO source=sched.go:658 msg="system memory" '
            f'total="{self.system_total_mib / 1024:.1f} GiB" free="10.8 GiB" free_swap="0 B"'
        )
        if self.kill_early:
            if self.log_lines:
                self.log_text += system_line + "\n"
            return
        lines = [
            system_line,
            "load_tensors: loading model tensors, this can take a while... (load_mode = mmap)",
            f"load_tensors: offloaded {on_gpu}/{self.layers_total} layers to GPU",
            f"load_tensors:   CPU_Mapped model buffer size =   {self.mapped_mib:.2f} MiB",
        ]
        if cpu_kv > 0:
            lines.append(f"llama_kv_cache:        CPU KV buffer size = {cpu_kv:.2f} MiB")
        if fraction > 0:
            lines.append(f"llama_kv_cache:      CUDA0 KV buffer size = {kv * fraction:.2f} MiB")
        if killed:
            lines.append(
                'time=2026-10-26T14:00:11.000Z level=INFO source=sched.go:686 msg="Load failed" '
                'error="llama-server process has terminated: signal: killed"'
            )
        if self.log_lines:
            self.log_text += "\n".join(lines) + "\n"

    def ensure_parallel(self, slots: int) -> None:
        if slots == self.num_parallel:
            return
        self.restarts += 1
        self.num_parallel = slots if self.honour_parallel else 1
        self._slot_free = [self._now] * self.num_parallel
        self._last_words = []  # a new process has an empty cache
        self._loaded_ctx = None

    def effective_slots(self) -> int | None:
        return self.num_parallel if self.report_slots else None

    def clock(self) -> float:
        with self._lock:
            self._arrival = self._now
            return self._now

    # -- the model ---------------------------------------------------------------------------------------------------
    def tokens_of(self, prompt: str) -> int:
        words = len(prompt.split())
        bias = 1.0 + self.long_prompt_bias if words > 300 else 1.0
        return math.ceil(words * self.tokens_per_word * bias)

    def evaluated(self, payload: dict[str, Any]) -> int:
        """What Ollama reports as ``prompt_eval_count``: the prompt's tokens, or the truncated count."""

        tokens = self.tokens_of(payload["prompt"])
        window = payload["options"]["num_ctx"]
        if tokens <= window - 1:
            return tokens
        return window - max((window - NUM_KEEP) // 2, 1)

    def start(self) -> FakeOllama:
        return self

    def stop(self) -> None:
        return None

    def unload(self, _model: str) -> None:
        return None

    def post(self, path: str, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        if path != "/api/generate":
            return {}
        if "options" not in payload:  # an unload or a pull is not a generation
            return {}
        options = payload["options"]
        num_predict = int(options.get("num_predict") or 1)
        words = payload["prompt"].split()
        with self._lock:
            if num_predict == 1:
                self._arrival = self._now  # the calibration's calls are sequential: each starts when the last ended
            self.all_calls.append({"prompt": payload["prompt"], "num_predict": num_predict, "timeout": timeout})
            self.timeouts.append(timeout)
            if self._loaded_ctx != options["num_ctx"]:
                self._loaded_ctx = options["num_ctx"]
                self._last_words = []  # a reload drops the cache
                self._log_load(options["num_ctx"])
            if self._load["killed"]:
                self.kills += 1
                raise RuntimeError(
                    "HTTP 500: llama-server process has terminated: signal: killed (the load needed more host memory)"
                )
            evaluated = self.evaluated(payload)
            common = 0
            if self.cache:
                for old, new in zip(self._last_words, words, strict=False):
                    if old != new:
                        break
                    common += 1
            cached = min(math.ceil(common * self.tokens_per_word), evaluated - 1)
            self._last_words = words
            uncached = max(1, evaluated - cached)
            prefill_s = uncached / self.prefill_tok_s
            decode_s = num_predict / self.decode_tok_s
            index = min(range(len(self._slot_free)), key=self._slot_free.__getitem__)
            start = max(self._slot_free[index], self._arrival)
            overhead = self.overhead_s + self.overhead_s_per_ktok * evaluated / 1000.0
            finish = start + overhead + prefill_s + decode_s
            self._slot_free[index] = finish
            self._now = max(self._now, finish)
            total_s = finish - self._arrival  # Ollama's total_duration counts the wait for a free slot
            limit = 120.0 if timeout is None else timeout  # the fixed urlopen timeout a caller gets without one
            if total_s > limit:
                raise TimeoutError(f"timed out after {limit:.0f} s")
            if num_predict > 1:
                self.prompts.append(payload["prompt"])
                self.contexts.append(options["num_ctx"])
                self.evaluated_counts.append(evaluated)
        return {
            "prompt_eval_count": evaluated,
            "prompt_eval_duration": int(prefill_s * 1e9),
            "eval_count": num_predict,
            "eval_duration": int(decode_s * 1e9),
            "load_duration": 0,
            "total_duration": int(total_s * 1e9),
        }
